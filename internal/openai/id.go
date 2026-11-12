package openai

import (
	"crypto/rand"
	"encoding/hex"
)

// NewID generates an OpenAI-style prefixed random hex identifier (e.g. "chatcmpl-...", "resp_...").
func NewID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}
