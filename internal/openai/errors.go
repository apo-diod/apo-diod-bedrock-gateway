package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/aws/smithy-go"
)

// Error types and codes aligned with the OpenAI REST API specification.
const (
	TypeInvalidRequest = "invalid_request_error"
	TypeAuthentication = "authentication_error"
	TypePermission     = "permission_error"
	TypeNotFound       = "not_found_error"
	TypeRateLimit      = "rate_limit_error"
	TypeServer         = "server_error"
)

// ErrorEnvelope is the top-level JSON structure returned on errors.
type ErrorEnvelope struct {
	Error ErrorBody `json:"error"`
}

// ErrorBody matches OpenAI's error object schema.
type ErrorBody struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// APIError represents an HTTP-mappable OpenAI error.
type APIError struct {
	Status     int
	Type       string
	Code       string
	Param      *string
	Message    string
	RetryAfter string
}

func (e *APIError) Error() string {
	return e.Message
}

// NewError creates a new APIError.
func NewError(status int, errType, code, message string) *APIError {
	return &APIError{
		Status:  status,
		Type:    errType,
		Code:    code,
		Message: message,
	}
}

// NewParamError creates a new APIError referencing a specific request parameter.
func NewParamError(status int, errType, code, param, message string) *APIError {
	p := param
	return &APIError{
		Status:  status,
		Type:    errType,
		Code:    code,
		Param:   &p,
		Message: message,
	}
}

// WriteAPIError serializes an APIError as an OpenAI JSON error response.
func WriteAPIError(w http.ResponseWriter, apiErr *APIError) {
	if apiErr == nil {
		apiErr = NewError(http.StatusInternalServerError, TypeServer, "internal_error", "internal server error")
	}
	if apiErr.RetryAfter != "" {
		w.Header().Set("Retry-After", apiErr.RetryAfter)
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	status := apiErr.Status
	if status <= 0 {
		status = http.StatusInternalServerError
	}
	w.WriteHeader(status)

	var codePtr *string
	if apiErr.Code != "" {
		c := apiErr.Code
		codePtr = &c
	}
	env := ErrorEnvelope{
		Error: ErrorBody{
			Message: apiErr.Message,
			Type:    apiErr.Type,
			Param:   apiErr.Param,
			Code:    codePtr,
		},
	}
	_ = json.NewEncoder(w).Encode(env)
}

// WriteError writes an OpenAI JSON error response directly.
func WriteError(w http.ResponseWriter, status int, errType, code, message string) {
	WriteAPIError(w, NewError(status, errType, code, message))
}

// CheckMaxBytesError checks whether err came from http.MaxBytesReader and converts it to 413.
func CheckMaxBytesError(err error) (*APIError, bool) {
	var maxErr *http.MaxBytesError
	if errors.As(err, &maxErr) {
		return NewError(
			http.StatusRequestEntityTooLarge,
			TypeInvalidRequest,
			"request_too_large",
			"request body exceeds configured max_request_body_bytes limit",
		), true
	}
	return nil, false
}

// BedrockErrorCode extracts a bounded-cardinality error code string from a Bedrock/Smithy error.
func BedrockErrorCode(err error) string {
	if err == nil {
		return "none"
	}
	var smithyErr smithy.APIError
	if errors.As(err, &smithyErr) {
		switch smithyErr.ErrorCode() {
		case "ValidationException",
			"AccessDeniedException",
			"ResourceNotFoundException",
			"ThrottlingException",
			"ServiceQuotaExceededException",
			"ModelTimeoutException",
			"ModelNotReadyException",
			"ServiceUnavailableException",
			"ModelStreamErrorException",
			"ModelErrorException",
			"InternalServerException":
			return smithyErr.ErrorCode()
		default:
			return "OtherAWSException"
		}
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "DeadlineExceeded"
	}
	if errors.Is(err, context.Canceled) {
		return "Canceled"
	}
	return "TransportError"
}

// FromBedrockError maps AWS Smithy/Bedrock errors and context errors to OpenAI APIError responses.
func FromBedrockError(err error) *APIError {
	if err == nil {
		return nil
	}
	var existing *APIError
	if errors.As(err, &existing) {
		return existing
	}

	var smithyErr smithy.APIError
	if errors.As(err, &smithyErr) {
		msg := smithyErr.ErrorMessage()
		if msg == "" {
			msg = smithyErr.Error()
		}
		switch smithyErr.ErrorCode() {
		case "ValidationException":
			return NewError(http.StatusBadRequest, TypeInvalidRequest, "invalid_request", msg)
		case "AccessDeniedException":
			return NewError(http.StatusForbidden, TypePermission, "access_denied", msg)
		case "ResourceNotFoundException":
			return NewError(http.StatusNotFound, TypeNotFound, "model_not_found", msg)
		case "ThrottlingException", "ServiceQuotaExceededException":
			ae := NewError(http.StatusTooManyRequests, TypeRateLimit, "rate_limit_exceeded", msg)
			ae.RetryAfter = "1"
			return ae
		case "ModelTimeoutException":
			return NewError(http.StatusGatewayTimeout, TypeServer, "model_timeout", msg)
		case "ModelNotReadyException", "ServiceUnavailableException":
			return NewError(http.StatusServiceUnavailable, TypeServer, "service_unavailable", msg)
		case "ModelStreamErrorException", "ModelErrorException", "InternalServerException":
			return NewError(http.StatusBadGateway, TypeServer, "upstream_model_error", msg)
		default:
			return NewError(http.StatusBadGateway, TypeServer, "upstream_error", msg)
		}
	}

	if errors.Is(err, context.DeadlineExceeded) {
		return NewError(http.StatusGatewayTimeout, TypeServer, "timeout", "upstream Bedrock request timed out")
	}
	if errors.Is(err, context.Canceled) {
		return NewError(http.StatusBadRequest, TypeInvalidRequest, "request_canceled", "client canceled request")
	}

	return NewError(http.StatusBadGateway, TypeServer, "upstream_error", err.Error())
}
