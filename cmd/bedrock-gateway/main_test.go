package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestHealthcheckAndTargetResolution(t *testing.T) {
	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer okSrv.Close()

	if err := runHealthcheck(okSrv.URL+"/healthz", false); err != nil {
		t.Fatalf("expected healthy probe, got error: %v", err)
	}
	if err := runHealthcheck(okSrv.URL+"/missing", false); err == nil {
		t.Fatal("expected error on 404 healthcheck")
	}

	t.Setenv("GATEWAY_HEALTHCHECK_URL", "")
	cfgFile := filepath.Join(t.TempDir(), "hc.toml")
	_ = os.WriteFile(cfgFile, []byte("access_key = \"sk-1\"\n[server]\nlisten_addr = \"0.0.0.0:9123\"\n"), 0o600)

	u, isTLS := resolveHealthcheckTarget(cfgFile)
	if u != "http://127.0.0.1:9123/healthz" || isTLS {
		t.Errorf("expected http://127.0.0.1:9123/healthz (tls=false), got %q (tls=%v)", u, isTLS)
	}
}
