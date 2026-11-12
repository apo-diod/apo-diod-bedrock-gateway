package awsclient

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
)

func clearAWSEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"AWS_ACCESS_KEY_ID",
		"AWS_SECRET_ACCESS_KEY",
		"AWS_SESSION_TOKEN",
		"AWS_REGION",
		"AWS_DEFAULT_REGION",
		"AWS_PROFILE",
		"AWS_ROLE_ARN",
		"AWS_WEB_IDENTITY_TOKEN_FILE",
		"AWS_ROLE_SESSION_NAME",
		"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
		"AWS_CONTAINER_CREDENTIALS_FULL_URI",
		"AWS_CONTAINER_AUTHORIZATION_TOKEN",
		"AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
		"AWS_EC2_METADATA_SERVICE_ENDPOINT",
	} {
		t.Setenv(k, "")
	}
}

func TestBuild_EnvCredentials(t *testing.T) {
	clearAWSEnv(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIAENVTEST123456")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret-env-key")
	t.Setenv("AWS_SESSION_TOKEN", "session-tok")

	bundle, err := Build(context.Background(), config.AWSConfig{
		Region:                     "us-west-2",
		CredentialSource:           config.CredentialSourceEnv,
		MaxAttempts:                3,
		VerifyCredentialsOnStartup: true,
	})
	if err != nil {
		t.Fatalf("Build(env) failed: %v", err)
	}
	if bundle.Region != "us-west-2" {
		t.Errorf("expected region us-west-2, got %q", bundle.Region)
	}
	creds, err := bundle.AWSConfig.Credentials.Retrieve(context.Background())
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}
	if creds.AccessKeyID != "AKIAENVTEST123456" || creds.SessionToken != "session-tok" {
		t.Errorf("unexpected credentials: %+v", creds)
	}

	// Missing env credentials must fail fast without falling back to IMDS.
	clearAWSEnv(t)
	_, err = Build(context.Background(), config.AWSConfig{
		Region:                     "us-west-2",
		CredentialSource:           config.CredentialSourceEnv,
		MaxAttempts:                2,
		VerifyCredentialsOnStartup: true,
	})
	if err == nil || !strings.Contains(err.Error(), "AWS_ACCESS_KEY_ID") {
		t.Fatalf("expected fail-fast env error, got: %v", err)
	}
}

func TestBuild_IMDSv2_RegionAutoDiscoveryAndHopLimitHint(t *testing.T) {
	clearAWSEnv(t)

	exp := time.Now().UTC().Add(1 * time.Hour).Format(time.RFC3339)
	imdsSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && r.URL.Path == "/latest/api/token":
			ttl := r.Header.Get("X-aws-ec2-metadata-token-ttl-seconds")
			if ttl == "" {
				http.Error(w, "missing ttl", http.StatusBadRequest)
				return
			}
			w.Header().Set("X-aws-ec2-metadata-token-ttl-seconds", ttl)
			_, _ = io.WriteString(w, "imds-v2-session-token")
		case r.Method == http.MethodGet && r.URL.Path == "/latest/dynamic/instance-identity/document":
			if r.Header.Get("X-aws-ec2-metadata-token") != "imds-v2-session-token" {
				http.Error(w, "missing imdsv2 token", http.StatusUnauthorized)
				return
			}
			_, _ = io.WriteString(w, `{"region":"eu-west-1"}`)
		case r.Method == http.MethodGet && r.URL.Path == "/latest/meta-data/placement/region":
			if r.Header.Get("X-aws-ec2-metadata-token") != "imds-v2-session-token" {
				http.Error(w, "missing imdsv2 token", http.StatusUnauthorized)
				return
			}
			_, _ = io.WriteString(w, "eu-west-1")
		case r.Method == http.MethodGet && r.URL.Path == "/latest/meta-data/iam/security-credentials/":
			if r.Header.Get("X-aws-ec2-metadata-token") != "imds-v2-session-token" {
				http.Error(w, "missing imdsv2 token", http.StatusUnauthorized)
				return
			}
			_, _ = io.WriteString(w, "ec2-gateway-role")
		case r.Method == http.MethodGet && r.URL.Path == "/latest/meta-data/iam/security-credentials/ec2-gateway-role":
			if r.Header.Get("X-aws-ec2-metadata-token") != "imds-v2-session-token" {
				http.Error(w, "missing imdsv2 token", http.StatusUnauthorized)
				return
			}
			_, _ = fmt.Fprintf(w, `{
				"Code": "Success",
				"Type": "AWS-HMAC",
				"AccessKeyId": "ASIAIMDSTEST999",
				"SecretAccessKey": "imds-secret",
				"Token": "imds-token",
				"Expiration": %q
			}`, exp)
		default:
			http.NotFound(w, r)
		}
	}))
	defer imdsSrv.Close()

	bundle, err := Build(
		context.Background(),
		config.AWSConfig{
			Region:                     "", // auto-discover from IMDS
			CredentialSource:           config.CredentialSourceIMDS,
			MaxAttempts:                2,
			VerifyCredentialsOnStartup: true,
		},
		WithIMDSEndpoint(imdsSrv.URL),
		WithHTTPClient(imdsSrv.Client()),
	)
	if err != nil {
		t.Fatalf("Build(imds) failed: %v", err)
	}
	if bundle.Region != "eu-west-1" {
		t.Errorf("expected auto-discovered region eu-west-1, got %q", bundle.Region)
	}
	creds, err := bundle.AWSConfig.Credentials.Retrieve(context.Background())
	if err != nil {
		t.Fatalf("Retrieve(imds) failed: %v", err)
	}
	if creds.AccessKeyID != "ASIAIMDSTEST999" {
		t.Errorf("expected ASIAIMDSTEST999, got %q", creds.AccessKeyID)
	}

	// Simulate dropped IMDSv2 token packet (hop-limit=1 failure) and verify remediation hint.
	brokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
	}))
	defer brokenSrv.Close()

	_, err = Build(
		context.Background(),
		config.AWSConfig{
			Region:                     "us-east-1",
			CredentialSource:           config.CredentialSourceIMDS,
			MaxAttempts:                1,
			VerifyCredentialsOnStartup: true,
		},
		WithIMDSEndpoint(brokenSrv.URL),
		WithHTTPClient(brokenSrv.Client()),
	)
	if err == nil || !strings.Contains(err.Error(), "--http-put-response-hop-limit 2") {
		t.Fatalf("expected IMDSv2 hop-limit remediation hint, got: %v", err)
	}
}

func TestBuild_ECSCredentials(t *testing.T) {
	clearAWSEnv(t)
	dir := t.TempDir()
	tokFile := filepath.Join(dir, "ecs-token")
	if err := os.WriteFile(tokFile, []byte("ecs-bearer-secret\n"), 0o600); err != nil {
		t.Fatalf("write token file: %v", err)
	}

	exp := time.Now().UTC().Add(1 * time.Hour).Format(time.RFC3339)
	ecsSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/credentials/task-uuid" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "ecs-bearer-secret" {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		_, _ = fmt.Fprintf(w, `{
			"AccessKeyId": "ASIAECSTEST111",
			"SecretAccessKey": "ecs-secret",
			"Token": "ecs-session",
			"Expiration": %q
		}`, exp)
	}))
	defer ecsSrv.Close()

	t.Setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/task-uuid")
	t.Setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", tokFile)

	bundle, err := Build(
		context.Background(),
		config.AWSConfig{
			Region:                     "us-east-2",
			CredentialSource:           config.CredentialSourceECS,
			MaxAttempts:                2,
			VerifyCredentialsOnStartup: true,
		},
		WithECSBaseEndpoint(ecsSrv.URL),
		WithHTTPClient(ecsSrv.Client()),
	)
	if err != nil {
		t.Fatalf("Build(ecs) failed: %v", err)
	}
	creds, err := bundle.AWSConfig.Credentials.Retrieve(context.Background())
	if err != nil {
		t.Fatalf("Retrieve(ecs) failed: %v", err)
	}
	if creds.AccessKeyID != "ASIAECSTEST111" {
		t.Errorf("expected ASIAECSTEST111, got %q", creds.AccessKeyID)
	}
}

func TestBuild_WebIdentityAndAssumeRoleAndPrivateLinkEndpoints(t *testing.T) {
	clearAWSEnv(t)
	dir := t.TempDir()
	jwtFile := filepath.Join(dir, "eks-jwt")
	if err := os.WriteFile(jwtFile, []byte("eyJhbGciOiJSUzI1NiJ9.fake.jwt"), 0o600); err != nil {
		t.Fatalf("write jwt file: %v", err)
	}

	var stsWebIdentityCalls atomic.Int32
	var stsAssumeRoleCalls atomic.Int32
	exp := time.Now().UTC().Add(1 * time.Hour).Format(time.RFC3339)

	stsSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		form := string(bodyBytes)
		w.Header().Set("Content-Type", "text/xml")
		switch {
		case strings.Contains(form, "Action=AssumeRoleWithWebIdentity"):
			stsWebIdentityCalls.Add(1)
			_, _ = fmt.Fprintf(w, `<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleWithWebIdentityResult>
    <Credentials>
      <AccessKeyId>ASIAWEBID111</AccessKeyId>
      <SecretAccessKey>webid-secret</SecretAccessKey>
      <SessionToken>webid-tok</SessionToken>
      <Expiration>%s</Expiration>
    </Credentials>
  </AssumeRoleWithWebIdentityResult>
</AssumeRoleWithWebIdentityResponse>`, exp)
		case strings.Contains(form, "Action=AssumeRole"):
			if !strings.Contains(form, "ExternalId=ext-secret-42") {
				http.Error(w, "missing external id", http.StatusBadRequest)
				return
			}
			stsAssumeRoleCalls.Add(1)
			_, _ = fmt.Fprintf(w, `<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleResult>
    <Credentials>
      <AccessKeyId>ASIAASSUMED222</AccessKeyId>
      <SecretAccessKey>assumed-secret</SecretAccessKey>
      <SessionToken>assumed-tok</SessionToken>
      <Expiration>%s</Expiration>
    </Credentials>
  </AssumeRoleResult>
</AssumeRoleResponse>`, exp)
		default:
			http.Error(w, "unexpected sts action", http.StatusBadRequest)
		}
	}))
	defer stsSrv.Close()

	var bedrockVPCHits atomic.Int32
	bedrockVPCSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bedrockVPCHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"modelSummaries":[]}`)
	}))
	defer bedrockVPCSrv.Close()

	t.Setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/eks-pod-role")
	t.Setenv("AWS_WEB_IDENTITY_TOKEN_FILE", jwtFile)

	bundle, err := Build(
		context.Background(),
		config.AWSConfig{
			Region:                     "us-east-1",
			CredentialSource:           config.CredentialSourceWebIdentity,
			AssumeRoleARN:              "arn:aws:iam::999999999999:role/cross-account-bedrock",
			AssumeRoleExternalID:       "ext-secret-42",
			MaxAttempts:                2,
			VerifyCredentialsOnStartup: true,
			STSEndpoint:                stsSrv.URL,
			BedrockEndpoint:            bedrockVPCSrv.URL,
			BedrockRuntimeEndpoint:     bedrockVPCSrv.URL,
		},
		WithHTTPClient(stsSrv.Client()),
	)
	if err != nil {
		t.Fatalf("Build(web_identity + assume_role) failed: %v", err)
	}
	if stsWebIdentityCalls.Load() == 0 || stsAssumeRoleCalls.Load() == 0 {
		t.Fatalf("expected both web identity (%d) and assume role (%d) STS calls", stsWebIdentityCalls.Load(), stsAssumeRoleCalls.Load())
	}

	// Verify Bedrock PrivateLink endpoint override receives control-plane request.
	_, err = bundle.Control.ListFoundationModels(context.Background(), &bedrock.ListFoundationModelsInput{})
	if err != nil {
		t.Fatalf("ListFoundationModels against mock PrivateLink endpoint failed: %v", err)
	}
	if bedrockVPCHits.Load() != 1 {
		t.Errorf("expected 1 hit on PrivateLink Bedrock endpoint, got %d", bedrockVPCHits.Load())
	}
}

func TestBuild_ProfilePinnedCredentials(t *testing.T) {
	clearAWSEnv(t)
	// Set ambient env credentials to verify profile mode pins to the profile file rather than env.
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIAWRONGFROMENV")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "wrong-secret")

	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config")
	credFile := filepath.Join(dir, "credentials")

	if err := os.WriteFile(cfgFile, []byte("[profile bedrock-prod]\nregion = ca-central-1\n"), 0o600); err != nil {
		t.Fatalf("write shared config: %v", err)
	}
	if err := os.WriteFile(credFile, []byte("[bedrock-prod]\naws_access_key_id = AKIAPROFILEGOOD\naws_secret_access_key = profile-secret\n"), 0o600); err != nil {
		t.Fatalf("write shared creds: %v", err)
	}

	bundle, err := Build(
		context.Background(),
		config.AWSConfig{
			CredentialSource:           config.CredentialSourceProfile,
			Profile:                    "bedrock-prod",
			MaxAttempts:                2,
			VerifyCredentialsOnStartup: true,
		},
		WithSharedFiles([]string{cfgFile}, []string{credFile}),
	)
	if err != nil {
		t.Fatalf("Build(profile) failed: %v", err)
	}
	if bundle.Region != "ca-central-1" {
		t.Errorf("expected region ca-central-1 from profile, got %q", bundle.Region)
	}
	creds, err := bundle.AWSConfig.Credentials.Retrieve(context.Background())
	if err != nil {
		t.Fatalf("Retrieve(profile) failed: %v", err)
	}
	if creds.AccessKeyID != "AKIAPROFILEGOOD" {
		t.Errorf("expected AKIAPROFILEGOOD, got %q", creds.AccessKeyID)
	}
}

func TestBuild_AutoChain(t *testing.T) {
	clearAWSEnv(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIAAUTOCHAIN777")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "auto-secret")

	bundle, err := Build(context.Background(), config.AWSConfig{
		Region:                     "ap-southeast-1",
		CredentialSource:           config.CredentialSourceAuto,
		MaxAttempts:                3,
		VerifyCredentialsOnStartup: true,
	})
	if err != nil {
		t.Fatalf("Build(auto) failed: %v", err)
	}
	if err := bundle.CheckReady(context.Background()); err != nil {
		t.Fatalf("CheckReady(auto) failed: %v", err)
	}
	creds, _ := bundle.AWSConfig.Credentials.Retrieve(context.Background())
	if creds.AccessKeyID != "AKIAAUTOCHAIN777" {
		t.Errorf("expected AKIAAUTOCHAIN777, got %q", creds.AccessKeyID)
	}
}

