package main

import (
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/awsclient"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/server"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/store"
)

var version = "dev"

func main() {
	var (
		configFlag      string
		healthcheckFlag bool
		versionFlag     bool
	)
	flag.StringVar(&configFlag, "config", "", "Path to config.toml (overrides GATEWAY_CONFIG; default: config.toml)")
	flag.BoolVar(&healthcheckFlag, "healthcheck", false, "Probe local /healthz endpoint and exit 0 if healthy (for distroless HEALTHCHECK)")
	flag.BoolVar(&versionFlag, "version", false, "Print version and exit")
	flag.Parse()

	if versionFlag {
		fmt.Println(version)
		return
	}

	cfgPath := config.ResolvePath(configFlag)
	if healthcheckFlag {
		targetURL, insecureTLS := resolveHealthcheckTarget(cfgPath)
		if err := runHealthcheck(targetURL, insecureTLS); err != nil {
			fmt.Fprintf(os.Stderr, "healthcheck failed: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if err := run(cfgPath); err != nil {
		logger := server.NewLogger("error", os.Stderr)
		logger.Error("bedrock-gateway exited with error", "error", err.Error())
		os.Exit(1)
	}
}

func run(cfgPath string) error {
	ctx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	cfg, err := config.Load(cfgPath)
	if err != nil {
		return err
	}

	logger := server.NewLogger(cfg.Server.LogLevel, os.Stdout)
	logger.Info("starting bedrock-gateway",
		"version", version,
		"listen_addr", cfg.Server.ListenAddr,
		"metrics_listen_addr", cfg.Server.MetricsListenAddr,
		"aws_credential_source", cfg.AWS.CredentialSource,
		"aws_region_configured", cfg.AWS.Region,
		"sqlite_path", cfg.Storage.SQLitePath,
		"tenants_count", len(cfg.Tenants),
	)

	st, err := store.Open(ctx, cfg.Storage.SQLitePath, cfg.Storage.ResponsesTTL.Duration)
	if err != nil {
		return fmt.Errorf("initialize sqlite store: %w", err)
	}
	defer func() { _ = st.Close() }()
	st.StartCleanupLoop(ctx, cfg.Storage.CleanupInterval.Duration, logger)

	awsBundle, err := awsclient.Build(ctx, cfg.AWS)
	if err != nil {
		return fmt.Errorf("initialize aws bedrock client: %w", err)
	}
	logger.Info("aws bedrock client initialized",
		"region", awsBundle.Region,
		"credential_source", awsBundle.CredentialSource,
		"assume_role_enabled", cfg.AWS.AssumeRoleARN != "",
		"custom_runtime_endpoint", cfg.AWS.BedrockRuntimeEndpoint != "",
	)

	mc := metrics.New()
	gw := server.BuildGateway(server.WireInput{
		Config:        cfg,
		RawRuntime:    awsBundle.Runtime,
		ControlClient: awsBundle.Control,
		AWSReadyCheck: awsBundle.CheckReady,
		Store:         st,
		Logger:        logger,
		Metrics:       mc,
	})

	errCh := make(chan error, 2)

	if gw.MetricsServer != nil {
		go func() {
			logger.Info("metrics listener starting", "addr", gw.MetricsServer.Addr)
			if sErr := gw.MetricsServer.ListenAndServe(); sErr != nil && !errors.Is(sErr, http.ErrServerClosed) {
				errCh <- fmt.Errorf("metrics http server: %w", sErr)
			}
		}()
	}

	go func() {
		var sErr error
		if cfg.Server.TLSCertFile != "" && cfg.Server.TLSKeyFile != "" {
			logger.Info("https listener starting", "addr", gw.MainServer.Addr)
			sErr = gw.MainServer.ListenAndServeTLS(cfg.Server.TLSCertFile, cfg.Server.TLSKeyFile)
		} else {
			logger.Info("http listener starting", "addr", gw.MainServer.Addr)
			sErr = gw.MainServer.ListenAndServe()
		}
		if sErr != nil && !errors.Is(sErr, http.ErrServerClosed) {
			errCh <- fmt.Errorf("main http server: %w", sErr)
		}
	}()

	select {
	case <-ctx.Done():
		logger.Info("shutdown signal received; draining active requests")
	case serveErr := <-errCh:
		return serveErr
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	var shutdownErr error
	if err := gw.MainServer.Shutdown(shutdownCtx); err != nil {
		shutdownErr = errors.Join(shutdownErr, fmt.Errorf("shutdown main server: %w", err))
	}
	if gw.MetricsServer != nil {
		if err := gw.MetricsServer.Shutdown(shutdownCtx); err != nil {
			shutdownErr = errors.Join(shutdownErr, fmt.Errorf("shutdown metrics server: %w", err))
		}
	}
	logger.Info("bedrock-gateway stopped cleanly")
	return shutdownErr
}

func resolveHealthcheckTarget(cfgPath string) (string, bool) {
	if envURL := strings.TrimSpace(os.Getenv("GATEWAY_HEALTHCHECK_URL")); envURL != "" {
		return envURL, strings.HasPrefix(strings.ToLower(envURL), "https://")
	}
	scheme := "http"
	hostPort := "127.0.0.1:8080"
	if cfg, err := config.Load(cfgPath); err == nil {
		if cfg.Server.TLSCertFile != "" {
			scheme = "https"
		}
		h, p, splitErr := net.SplitHostPort(cfg.Server.ListenAddr)
		if splitErr == nil && p != "" {
			switch h {
			case "", "0.0.0.0", "::", "[::]":
				h = "127.0.0.1"
			}
			hostPort = net.JoinHostPort(h, p)
		}
	}
	return fmt.Sprintf("%s://%s/healthz", scheme, hostPort), scheme == "https"
}

func runHealthcheck(targetURL string, insecureTLS bool) error {
	transport := &http.Transport{}
	if insecureTLS {
		transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // Local loopback liveness probe
	}
	client := &http.Client{
		Timeout:   3 * time.Second,
		Transport: transport,
	}
	resp, err := client.Get(targetURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code %d from %s", resp.StatusCode, targetURL)
	}
	return nil
}
