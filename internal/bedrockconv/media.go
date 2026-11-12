package bedrockconv

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const maxRemoteMediaBytes = 20 * 1024 * 1024 // 20 MiB cap for remote media downloads

var blockedPrefixes = mustParsePrefixes([]string{
	"0.0.0.0/8",
	"10.0.0.0/8",
	"100.64.0.0/10", // CGNAT / shared VPC space
	"127.0.0.0/8",
	"169.254.0.0/16", // EC2 IMDS (169.254.169.254) & ECS task metadata (169.254.170.2)
	"172.16.0.0/12",
	"192.0.0.0/24",
	"192.168.0.0/16",
	"198.18.0.0/15",
	"224.0.0.0/4",
	"240.0.0.0/4",
	"::/128",
	"::1/128",
	"fc00::/7",  // IPv6 ULA including EKS Pod Identity fd00:ec2::23
	"fe80::/10", // IPv6 link-local
	"ff00::/8",  // IPv6 multicast
})

func mustParsePrefixes(cidrs []string) []netip.Prefix {
	out := make([]netip.Prefix, 0, len(cidrs))
	for _, c := range cidrs {
		p, err := netip.ParsePrefix(c)
		if err != nil {
			panic(err)
		}
		out = append(out, p)
	}
	return out
}

// IsBlockedMediaIP reports whether addr falls in loopback, link-local/IMDS, RFC1918, CGNAT, ULA, or multicast space.
func IsBlockedMediaIP(addr netip.Addr) bool {
	if !addr.IsValid() {
		return true
	}
	u := addr.Unmap()
	if u.IsLoopback() || u.IsPrivate() || u.IsLinkLocalUnicast() || u.IsLinkLocalMulticast() || u.IsMulticast() || u.IsUnspecified() {
		return true
	}
	for _, p := range blockedPrefixes {
		if p.Contains(u) {
			return true
		}
	}
	return false
}

// NewSSRFSafeHTTPClient builds an HTTP client that validates resolved socket peer addresses at connect time.
func NewSSRFSafeHTTPClient(timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	dialer := &net.Dialer{
		Timeout: 5 * time.Second,
		Control: func(network, address string, _ syscall.RawConn) error {
			host, _, err := net.SplitHostPort(address)
			if err != nil {
				return fmt.Errorf("invalid dial address %q: %w", address, err)
			}
			ip, err := netip.ParseAddr(host)
			if err != nil {
				return fmt.Errorf("non-IP dial target %q: %w", host, err)
			}
			if IsBlockedMediaIP(ip) {
				return fmt.Errorf("blocked non-public media target IP %s (SSRF protection)", ip.String())
			}
			return nil
		},
	}
	transport := &http.Transport{
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: timeout,
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return errors.New("too many media redirects")
			}
			if req.URL.Scheme != "https" && req.URL.Scheme != "http" {
				return fmt.Errorf("unsupported redirect scheme %q", req.URL.Scheme)
			}
			return nil
		},
	}
}

// ParseDataURI decodes a "data:[<mediatype>][;base64],<data>" URI.
func ParseDataURI(uri string) (mediaType string, raw []byte, err error) {
	if !strings.HasPrefix(uri, "data:") {
		return "", nil, errors.New("not a data: URI")
	}
	rest := uri[len("data:"):]
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", nil, errors.New("malformed data: URI (missing comma)")
	}
	header := rest[:comma]
	payload := rest[comma+1:]

	parts := strings.Split(header, ";")
	isBase64 := false
	if len(parts) > 0 && strings.Contains(parts[0], "/") {
		mediaType = strings.ToLower(strings.TrimSpace(parts[0]))
		parts = parts[1:]
	}
	for _, p := range parts {
		if strings.EqualFold(strings.TrimSpace(p), "base64") {
			isBase64 = true
		}
	}
	if !isBase64 {
		decodedStr, err := url.QueryUnescape(payload)
		if err != nil {
			return "", nil, fmt.Errorf("decode data URI text payload: %w", err)
		}
		if mediaType == "" {
			mediaType = "text/plain"
		}
		return mediaType, []byte(decodedStr), nil
	}

	cleaned := strings.Map(func(r rune) rune {
		if r == '\n' || r == '\r' || r == ' ' || r == '\t' {
			return -1
		}
		return r
	}, payload)
	decoded, err := base64.StdEncoding.DecodeString(cleaned)
	if err != nil {
		decoded, err = base64.RawStdEncoding.DecodeString(cleaned)
		if err != nil {
			return "", nil, fmt.Errorf("decode base64 data URI: %w", err)
		}
	}
	return mediaType, decoded, nil
}

func resolveMediaBytes(ctx context.Context, part ContentPart, allowRemote bool, mediaClient *http.Client) (string, []byte, *openai.APIError) {
	if len(part.Data) > 0 {
		return strings.ToLower(strings.TrimSpace(part.MediaType)), part.Data, nil
	}
	u := strings.TrimSpace(part.RemoteURL)
	if u == "" {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_media", "media content part has neither inline data nor URL")
	}
	if strings.HasPrefix(u, "data:") {
		mt, data, err := ParseDataURI(u)
		if err != nil {
			return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_data_uri", err.Error())
		}
		if len(data) == 0 {
			return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "empty_media", "decoded base64 media payload is empty")
		}
		return mt, data, nil
	}

	if !allowRemote {
		return "", nil, openai.NewError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"remote_media_disabled",
			"Remote http(s) media URLs are disabled by gateway policy (allow_remote_media_urls=false); provide inline base64 data: URIs instead.",
		)
	}

	parsedURL, err := url.Parse(u)
	if err != nil || (parsedURL.Scheme != "https" && parsedURL.Scheme != "http") {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_media_url", "media URL must use http or https scheme or be a base64 data: URI")
	}

	if mediaClient == nil {
		mediaClient = NewSSRFSafeHTTPClient(10 * time.Second)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_media_url", err.Error())
	}
	resp, err := mediaClient.Do(req)
	if err != nil {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "media_fetch_failed", fmt.Sprintf("failed to fetch remote media URL: %v", err))
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "media_fetch_failed", fmt.Sprintf("remote media URL returned HTTP %d", resp.StatusCode))
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxRemoteMediaBytes+1))
	if err != nil {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "media_fetch_failed", fmt.Sprintf("read remote media body: %v", err))
	}
	if len(body) > maxRemoteMediaBytes {
		return "", nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "media_too_large", "remote media exceeds 20 MiB maximum size")
	}
	mt := strings.ToLower(strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0]))
	if mt == "" || mt == "application/octet-stream" {
		mt = strings.ToLower(strings.TrimSpace(part.MediaType))
	}
	if mt == "" {
		mt = strings.ToLower(strings.TrimSpace(strings.Split(http.DetectContentType(body), ";")[0]))
	}
	return mt, body, nil
}

func mapImageFormat(mediaType string, data []byte) (types.ImageFormat, *openai.APIError) {
	mt := strings.ToLower(strings.TrimSpace(mediaType))
	if mt == "" && len(data) > 0 {
		mt = strings.ToLower(strings.Split(http.DetectContentType(data), ";")[0])
	}
	switch mt {
	case "image/png", "png":
		return types.ImageFormatPng, nil
	case "image/jpeg", "image/jpg", "jpeg", "jpg":
		return types.ImageFormatJpeg, nil
	case "image/gif", "gif":
		return types.ImageFormatGif, nil
	case "image/webp", "webp":
		return types.ImageFormatWebp, nil
	default:
		return "", openai.NewError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"unsupported_image_format",
			fmt.Sprintf("Unsupported image media type %q; Bedrock Converse supports png, jpeg, gif, and webp.", mediaType),
		)
	}
}

func mapDocumentFormat(mediaType, filename string) (types.DocumentFormat, *openai.APIError) {
	mt := strings.ToLower(strings.TrimSpace(mediaType))
	lowerName := strings.ToLower(strings.TrimSpace(filename))
	switch {
	case mt == "application/pdf" || mt == "pdf" || strings.HasSuffix(lowerName, ".pdf"):
		return types.DocumentFormatPdf, nil
	case mt == "text/csv" || mt == "csv" || strings.HasSuffix(lowerName, ".csv"):
		return types.DocumentFormatCsv, nil
	case mt == "text/html" || mt == "html" || strings.HasSuffix(lowerName, ".html"):
		return types.DocumentFormatHtml, nil
	case mt == "text/markdown" || mt == "md" || strings.HasSuffix(lowerName, ".md"):
		return types.DocumentFormatMd, nil
	case mt == "text/plain" || mt == "txt" || strings.HasSuffix(lowerName, ".txt"):
		return types.DocumentFormatTxt, nil
	case mt == "application/msword" || strings.HasSuffix(lowerName, ".doc"):
		return types.DocumentFormatDoc, nil
	case mt == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || strings.HasSuffix(lowerName, ".docx"):
		return types.DocumentFormatDocx, nil
	case mt == "application/vnd.ms-excel" || strings.HasSuffix(lowerName, ".xls"):
		return types.DocumentFormatXls, nil
	case mt == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" || strings.HasSuffix(lowerName, ".xlsx"):
		return types.DocumentFormatXlsx, nil
	default:
		return "", openai.NewError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"unsupported_document_format",
			fmt.Sprintf("Unsupported document media type %q.", mediaType),
		)
	}
}

// sanitizeBedrockDocName strips extensions and disallowed punctuation per Bedrock DocumentBlock.Name rules.
func sanitizeBedrockDocName(raw string, idx int) string {
	raw = strings.TrimSpace(raw)
	if dot := strings.LastIndexByte(raw, '.'); dot > 0 {
		raw = raw[:dot]
	}
	var b strings.Builder
	prevSpace := false
	for _, r := range raw {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
			r == '-' || r == '(' || r == ')' || r == '[' || r == ']':
			b.WriteRune(r)
			prevSpace = false
		case r == ' ' || r == '_' || r == '.':
			if !prevSpace && b.Len() > 0 {
				b.WriteByte(' ')
				prevSpace = true
			}
		}
	}
	out := strings.TrimSpace(b.String())
	if out == "" {
		return fmt.Sprintf("document-%d", idx+1)
	}
	if len(out) > 200 {
		out = strings.TrimSpace(out[:200])
	}
	return out
}
