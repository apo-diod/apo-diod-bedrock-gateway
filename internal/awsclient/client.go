package awsclient

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/retry"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go-v2/credentials/endpointcreds"
	"github.com/aws/aws-sdk-go-v2/credentials/stscreds"
	"github.com/aws/aws-sdk-go-v2/feature/ec2/imds"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/sts"
)

const (
	defaultFallbackRegion = "us-east-1"
	defaultECSBaseURI     = "http://169.254.170.2"

	imdsHopLimitHint = "hint: if running inside Docker bridge networking on EC2, EC2 IMDSv2 defaults to hop-limit=1 which drops container token requests; run 'aws ec2 modify-instance-metadata-options --instance-id <id> --http-tokens required --http-put-response-hop-limit 2' or use 'network_mode: host'"
)

// BedrockRuntimeAPI abstracts Amazon Bedrock Runtime data-plane operations.
type BedrockRuntimeAPI interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(ctx context.Context, params *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error)
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

// BedrockControlAPI abstracts Amazon Bedrock control-plane model discovery operations.
type BedrockControlAPI interface {
	ListFoundationModels(ctx context.Context, params *bedrock.ListFoundationModelsInput, optFns ...func(*bedrock.Options)) (*bedrock.ListFoundationModelsOutput, error)
	ListInferenceProfiles(ctx context.Context, params *bedrock.ListInferenceProfilesInput, optFns ...func(*bedrock.Options)) (*bedrock.ListInferenceProfilesOutput, error)
}

// Bundle holds initialized AWS SDK configuration and Bedrock clients.
type Bundle struct {
	AWSConfig        aws.Config
	Region           string
	CredentialSource string
	Runtime          BedrockRuntimeAPI
	Control          BedrockControlAPI
}

// CheckReady verifies that AWS credentials can be resolved (cached or refreshed).
func (b *Bundle) CheckReady(ctx context.Context) error {
	if b == nil || b.AWSConfig.Credentials == nil {
		return errors.New("aws credentials provider not initialized")
	}
	checkCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	creds, err := b.AWSConfig.Credentials.Retrieve(checkCtx)
	if err != nil {
		return wrapIMDSHintIfApplicable(b.CredentialSource, err)
	}
	if !creds.HasKeys() {
		return errors.New("aws credentials provider returned empty key material")
	}
	return nil
}

// BuildOptions allows injecting custom transports and local mock endpoints in tests.
type BuildOptions struct {
	HTTPClient             *http.Client
	IMDSEndpoint           string
	ECSBaseEndpoint        string
	SharedConfigFiles      []string
	SharedCredentialsFiles []string
}

// Option mutates BuildOptions.
type Option func(*BuildOptions)

// WithHTTPClient overrides the HTTP client used by AWS SDK calls and credential providers.
func WithHTTPClient(hc *http.Client) Option {
	return func(o *BuildOptions) { o.HTTPClient = hc }
}

// WithIMDSEndpoint overrides the EC2 IMDSv2 endpoint (used for testing).
func WithIMDSEndpoint(ep string) Option {
	return func(o *BuildOptions) { o.IMDSEndpoint = ep }
}

// WithECSBaseEndpoint overrides the default 169.254.170.2 ECS credentials host (used for testing).
func WithECSBaseEndpoint(ep string) Option {
	return func(o *BuildOptions) { o.ECSBaseEndpoint = ep }
}

// WithSharedFiles overrides the shared config and credentials file paths (used for testing).
func WithSharedFiles(configFiles, credFiles []string) Option {
	return func(o *BuildOptions) {
		o.SharedConfigFiles = configFiles
		o.SharedCredentialsFiles = credFiles
	}
}

// Build constructs the AWS configuration and Bedrock clients according to config.AWSConfig.
func Build(ctx context.Context, cfg config.AWSConfig, opts ...Option) (*Bundle, error) {
	var bo BuildOptions
	for _, fn := range opts {
		fn(&bo)
	}

	maxAttempts := cfg.MaxAttempts
	if maxAttempts < 1 {
		maxAttempts = 3
	}
	retryerFn := func() aws.Retryer {
		return retry.NewAdaptiveMode(func(ao *retry.AdaptiveModeOptions) {
			ao.StandardOptions = append(ao.StandardOptions, func(so *retry.StandardOptions) {
				so.MaxAttempts = maxAttempts
			})
		})
	}

	var (
		awsCfg aws.Config
		err    error
	)

	source := strings.ToLower(strings.TrimSpace(cfg.CredentialSource))
	if source == "" {
		source = config.CredentialSourceAuto
	}

	switch source {
	case config.CredentialSourceAuto:
		awsCfg, err = buildAutoConfig(ctx, cfg, bo, retryerFn)
	case config.CredentialSourceIMDS:
		awsCfg, err = buildIMDSConfig(ctx, cfg, bo, retryerFn)
	case config.CredentialSourceECS:
		awsCfg, err = buildECSConfig(cfg, bo, retryerFn)
	case config.CredentialSourceWebIdentity:
		awsCfg, err = buildWebIdentityConfig(cfg, bo, retryerFn)
	case config.CredentialSourceEnv:
		awsCfg, err = buildEnvConfig(cfg, bo, retryerFn)
	case config.CredentialSourceProfile:
		awsCfg, err = buildProfileConfig(ctx, cfg, bo, retryerFn)
	default:
		return nil, fmt.Errorf("unsupported aws.credential_source %q", cfg.CredentialSource)
	}
	if err != nil {
		return nil, wrapIMDSHintIfApplicable(source, err)
	}

	if awsCfg.Region == "" {
		awsCfg.Region = defaultFallbackRegion
	}

	// Optional STS AssumeRole wrapper across any base credential source.
	if cfg.AssumeRoleARN != "" {
		stsClient := sts.NewFromConfig(awsCfg, func(o *sts.Options) {
			if cfg.STSEndpoint != "" {
				o.BaseEndpoint = aws.String(cfg.STSEndpoint)
			}
		})
		externalID := cfg.AssumeRoleExternalID
		assumeProvider := stscreds.NewAssumeRoleProvider(stsClient, cfg.AssumeRoleARN, func(o *stscreds.AssumeRoleOptions) {
			if externalID != "" {
				o.ExternalID = aws.String(externalID)
			}
		})
		awsCfg.Credentials = aws.NewCredentialsCache(assumeProvider)
	}

	runtimeClient := bedrockruntime.NewFromConfig(awsCfg, func(o *bedrockruntime.Options) {
		if cfg.BedrockRuntimeEndpoint != "" {
			o.BaseEndpoint = aws.String(cfg.BedrockRuntimeEndpoint)
		}
	})
	controlClient := bedrock.NewFromConfig(awsCfg, func(o *bedrock.Options) {
		if cfg.BedrockEndpoint != "" {
			o.BaseEndpoint = aws.String(cfg.BedrockEndpoint)
		}
	})

	bundle := &Bundle{
		AWSConfig:        awsCfg,
		Region:           awsCfg.Region,
		CredentialSource: source,
		Runtime:          runtimeClient,
		Control:          controlClient,
	}

	if cfg.VerifyCredentialsOnStartup {
		verifyCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		if err := bundle.CheckReady(verifyCtx); err != nil {
			return nil, fmt.Errorf("aws startup credential verification failed (source=%s): %w", source, err)
		}
	}

	return bundle, nil
}

func buildAutoConfig(ctx context.Context, cfg config.AWSConfig, bo BuildOptions, retryerFn func() aws.Retryer) (aws.Config, error) {
	loadOpts := []func(*awsconfig.LoadOptions) error{
		awsconfig.WithRetryer(retryerFn),
	}
	if cfg.Region != "" {
		loadOpts = append(loadOpts, awsconfig.WithRegion(cfg.Region))
	} else {
		loadOpts = append(loadOpts, awsconfig.WithEC2IMDSRegion())
	}
	if cfg.Profile != "" {
		loadOpts = append(loadOpts, awsconfig.WithSharedConfigProfile(cfg.Profile))
	}
	if bo.HTTPClient != nil {
		loadOpts = append(loadOpts, awsconfig.WithHTTPClient(bo.HTTPClient))
	}
	if bo.IMDSEndpoint != "" {
		loadOpts = append(loadOpts, awsconfig.WithEC2IMDSEndpoint(bo.IMDSEndpoint))
	}
	if len(bo.SharedConfigFiles) > 0 {
		loadOpts = append(loadOpts, awsconfig.WithSharedConfigFiles(bo.SharedConfigFiles))
	}
	if len(bo.SharedCredentialsFiles) > 0 {
		loadOpts = append(loadOpts, awsconfig.WithSharedCredentialsFiles(bo.SharedCredentialsFiles))
	}
	return awsconfig.LoadDefaultConfig(ctx, loadOpts...)
}

func buildIMDSConfig(ctx context.Context, cfg config.AWSConfig, bo BuildOptions, retryerFn func() aws.Retryer) (aws.Config, error) {
	imdsOpts := imds.Options{}
	if bo.HTTPClient != nil {
		imdsOpts.HTTPClient = bo.HTTPClient
	}
	endpoint := strings.TrimSpace(bo.IMDSEndpoint)
	if endpoint == "" {
		endpoint = strings.TrimSpace(os.Getenv("AWS_EC2_METADATA_SERVICE_ENDPOINT"))
	}
	if endpoint != "" {
		imdsOpts.Endpoint = endpoint
	}
	imdsClient := imds.New(imdsOpts)

	region := cfg.Region
	if region == "" {
		regCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		out, err := imdsClient.GetRegion(regCtx, &imds.GetRegionInput{})
		if err != nil {
			return aws.Config{}, fmt.Errorf("discover region from EC2 IMDSv2: %w (%s)", err, imdsHopLimitHint)
		}
		region = strings.TrimSpace(out.Region)
	}

	rawProvider := ec2rolecreds.New(func(o *ec2rolecreds.Options) {
		o.Client = imdsClient
	})
	wrappedProvider := imdsHintProvider{inner: rawProvider}

	awsCfg := aws.Config{
		Region:      region,
		Credentials: aws.NewCredentialsCache(wrappedProvider),
		Retryer:     retryerFn,
	}
	if bo.HTTPClient != nil {
		awsCfg.HTTPClient = bo.HTTPClient
	}
	return awsCfg, nil
}

func buildECSConfig(cfg config.AWSConfig, bo BuildOptions, retryerFn func() aws.Retryer) (aws.Config, error) {
	relURI := strings.TrimSpace(os.Getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"))
	fullURI := strings.TrimSpace(os.Getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI"))
	if relURI == "" && fullURI == "" {
		return aws.Config{}, errors.New("credential_source=ecs requires AWS_CONTAINER_CREDENTIALS_RELATIVE_URI or AWS_CONTAINER_CREDENTIALS_FULL_URI")
	}

	targetEndpoint := fullURI
	if relURI != "" {
		base := strings.TrimRight(bo.ECSBaseEndpoint, "/")
		if base == "" {
			base = defaultECSBaseURI
		}
		if !strings.HasPrefix(relURI, "/") {
			relURI = "/" + relURI
		}
		targetEndpoint = base + relURI
	}

	authToken := os.Getenv("AWS_CONTAINER_AUTHORIZATION_TOKEN")
	authTokenFile := strings.TrimSpace(os.Getenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"))

	provider := endpointcreds.New(targetEndpoint, func(o *endpointcreds.Options) {
		if bo.HTTPClient != nil {
			o.HTTPClient = bo.HTTPClient
		}
		if authTokenFile != "" {
			o.AuthorizationTokenProvider = endpointcreds.TokenProviderFunc(func() (string, error) {
				b, err := os.ReadFile(authTokenFile)
				if err != nil {
					return "", fmt.Errorf("read AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE: %w", err)
				}
				return strings.TrimSpace(string(b)), nil
			})
		} else if authToken != "" {
			o.AuthorizationToken = authToken
		}
	})

	awsCfg := aws.Config{
		Region:      nonEmptyOrDefault(cfg.Region, defaultFallbackRegion),
		Credentials: aws.NewCredentialsCache(provider),
		Retryer:     retryerFn,
	}
	if bo.HTTPClient != nil {
		awsCfg.HTTPClient = bo.HTTPClient
	}
	return awsCfg, nil
}

func buildWebIdentityConfig(cfg config.AWSConfig, bo BuildOptions, retryerFn func() aws.Retryer) (aws.Config, error) {
	roleARN := strings.TrimSpace(os.Getenv("AWS_ROLE_ARN"))
	tokenFile := strings.TrimSpace(os.Getenv("AWS_WEB_IDENTITY_TOKEN_FILE"))
	sessionName := strings.TrimSpace(os.Getenv("AWS_ROLE_SESSION_NAME"))
	if roleARN == "" || tokenFile == "" {
		return aws.Config{}, errors.New("credential_source=web_identity requires AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE")
	}

	region := nonEmptyOrDefault(cfg.Region, defaultFallbackRegion)
	baseCfg := aws.Config{
		Region:      region,
		Credentials: aws.AnonymousCredentials{},
		Retryer:     retryerFn,
	}
	if bo.HTTPClient != nil {
		baseCfg.HTTPClient = bo.HTTPClient
	}
	stsClient := sts.NewFromConfig(baseCfg, func(o *sts.Options) {
		if cfg.STSEndpoint != "" {
			o.BaseEndpoint = aws.String(cfg.STSEndpoint)
		}
	})

	provider := stscreds.NewWebIdentityRoleProvider(
		stsClient,
		roleARN,
		stscreds.IdentityTokenFile(tokenFile),
		func(o *stscreds.WebIdentityRoleOptions) {
			if sessionName != "" {
				o.RoleSessionName = sessionName
			}
		},
	)

	awsCfg := aws.Config{
		Region:      region,
		Credentials: aws.NewCredentialsCache(provider),
		Retryer:     retryerFn,
	}
	if bo.HTTPClient != nil {
		awsCfg.HTTPClient = bo.HTTPClient
	}
	return awsCfg, nil
}

func buildEnvConfig(cfg config.AWSConfig, bo BuildOptions, retryerFn func() aws.Retryer) (aws.Config, error) {
	accessKeyID := strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID"))
	secretAccessKey := strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY"))
	sessionToken := strings.TrimSpace(os.Getenv("AWS_SESSION_TOKEN"))
	if accessKeyID == "" || secretAccessKey == "" {
		return aws.Config{}, errors.New("credential_source=env requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
	}

	provider := credentials.NewStaticCredentialsProvider(accessKeyID, secretAccessKey, sessionToken)
	awsCfg := aws.Config{
		Region:      nonEmptyOrDefault(cfg.Region, defaultFallbackRegion),
		Credentials: provider,
		Retryer:     retryerFn,
	}
	if bo.HTTPClient != nil {
		awsCfg.HTTPClient = bo.HTTPClient
	}
	return awsCfg, nil
}

func buildProfileConfig(ctx context.Context, cfg config.AWSConfig, bo BuildOptions, retryerFn func() aws.Retryer) (aws.Config, error) {
	profile := cfg.Profile
	if profile == "" {
		profile = strings.TrimSpace(os.Getenv("AWS_PROFILE"))
	}
	if profile == "" {
		profile = awsconfig.DefaultSharedConfigProfile
	}

	sharedOpts := []func(*awsconfig.LoadSharedConfigOptions){}
	if len(bo.SharedConfigFiles) > 0 {
		sharedOpts = append(sharedOpts, func(o *awsconfig.LoadSharedConfigOptions) {
			o.ConfigFiles = bo.SharedConfigFiles
		})
	}
	if len(bo.SharedCredentialsFiles) > 0 {
		sharedOpts = append(sharedOpts, func(o *awsconfig.LoadSharedConfigOptions) {
			o.CredentialsFiles = bo.SharedCredentialsFiles
		})
	}

	shared, err := awsconfig.LoadSharedConfigProfile(ctx, profile, sharedOpts...)
	if err != nil {
		return aws.Config{}, fmt.Errorf("load aws shared profile %q: %w", profile, err)
	}

	region := cfg.Region
	if region == "" {
		region = strings.TrimSpace(shared.Region)
	}
	region = nonEmptyOrDefault(region, defaultFallbackRegion)

	// Static credentials in shared files: pin directly so ambient AWS_ACCESS_KEY_ID env cannot shadow the profile.
	if shared.Credentials.HasKeys() && shared.RoleARN == "" {
		provider := credentials.NewStaticCredentialsProvider(
			shared.Credentials.AccessKeyID,
			shared.Credentials.SecretAccessKey,
			shared.Credentials.SessionToken,
		)
		awsCfg := aws.Config{
			Region:      region,
			Credentials: provider,
			Retryer:     retryerFn,
		}
		if bo.HTTPClient != nil {
			awsCfg.HTTPClient = bo.HTTPClient
		}
		return awsCfg, nil
	}

	// Dynamic profile (SSO, credential_process, role_arn + source_profile): load with IMDS disabled.
	loadOpts := []func(*awsconfig.LoadOptions) error{
		awsconfig.WithSharedConfigProfile(profile),
		awsconfig.WithRegion(region),
		awsconfig.WithRetryer(retryerFn),
		awsconfig.WithEC2IMDSClientEnableState(imds.ClientDisabled),
	}
	if bo.HTTPClient != nil {
		loadOpts = append(loadOpts, awsconfig.WithHTTPClient(bo.HTTPClient))
	}
	if len(bo.SharedConfigFiles) > 0 {
		loadOpts = append(loadOpts, awsconfig.WithSharedConfigFiles(bo.SharedConfigFiles))
	}
	if len(bo.SharedCredentialsFiles) > 0 {
		loadOpts = append(loadOpts, awsconfig.WithSharedCredentialsFiles(bo.SharedCredentialsFiles))
	}
	return awsconfig.LoadDefaultConfig(ctx, loadOpts...)
}

type imdsHintProvider struct {
	inner aws.CredentialsProvider
}

func (p imdsHintProvider) Retrieve(ctx context.Context) (aws.Credentials, error) {
	creds, err := p.inner.Retrieve(ctx)
	if err != nil {
		return aws.Credentials{}, wrapIMDSHintIfApplicable(config.CredentialSourceIMDS, err)
	}
	return creds, nil
}

func wrapIMDSHintIfApplicable(source string, err error) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	if strings.Contains(msg, imdsHopLimitHint) {
		return err
	}
	if source == config.CredentialSourceIMDS ||
		strings.Contains(msg, "169.254.169.254") ||
		strings.Contains(msg, "ec2imds") ||
		strings.Contains(msg, "EC2RoleRequestError") {
		return fmt.Errorf("%w (%s)", err, imdsHopLimitHint)
	}
	return err
}

func nonEmptyOrDefault(val, fallback string) string {
	if strings.TrimSpace(val) != "" {
		return strings.TrimSpace(val)
	}
	return fallback
}
