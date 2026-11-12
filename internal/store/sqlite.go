package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

// ErrNotFound indicates a response ID does not exist, has expired, or belongs to a different tenant.
var ErrNotFound = errors.New("response not found")

// StoredResponse represents a persisted /v1/responses record in SQLite.
type StoredResponse struct {
	ID                   string
	TenantID             string
	PreviousResponseID   string
	Model                string
	Status               string
	InputItemsJSON       []byte
	OutputItemsJSON      []byte
	CanonicalHistoryJSON []byte
	ResponseJSON         []byte
	CreatedAt            int64
	ExpiresAt            *int64
}

// Store manages pure-Go SQLite persistence for /v1/responses state and TTL expiration.
type Store struct {
	db  *sql.DB
	ttl time.Duration
}

const schemaSQL = `
CREATE TABLE IF NOT EXISTS responses (
	id TEXT PRIMARY KEY,
	tenant_id TEXT NOT NULL,
	previous_response_id TEXT,
	model TEXT NOT NULL,
	status TEXT NOT NULL,
	input_items_json BLOB NOT NULL,
	output_items_json BLOB NOT NULL,
	canonical_history_json BLOB NOT NULL,
	response_json BLOB NOT NULL,
	created_at INTEGER NOT NULL,
	expires_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_responses_tenant_created ON responses (tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_responses_expires_at ON responses (expires_at);
`

// Open opens (and migrates) the pure-Go SQLite database at sqlitePath in WAL mode.
func Open(ctx context.Context, sqlitePath string, responsesTTL time.Duration) (*Store, error) {
	sqlitePath = strings.TrimSpace(sqlitePath)
	if sqlitePath == "" {
		return nil, errors.New("sqlite_path must not be empty")
	}

	dsn := sqlitePath
	if sqlitePath != ":memory:" && !strings.HasPrefix(sqlitePath, "file:") {
		dir := filepath.Dir(sqlitePath)
		if dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o750); err != nil {
				return nil, fmt.Errorf("create sqlite directory %q: %w", dir, err)
			}
		}
		dsn = "file:" + sqlitePath + "?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=synchronous(NORMAL)"
	}

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %q: %w", sqlitePath, err)
	}

	if sqlitePath == ":memory:" {
		// In-memory SQLite databases are per-connection unless pinned to 1 connection.
		db.SetMaxOpenConns(1)
	} else {
		db.SetMaxOpenConns(4)
		db.SetMaxIdleConns(4)
	}

	for _, pragma := range []string{
		"PRAGMA busy_timeout = 5000;",
		"PRAGMA journal_mode = WAL;",
		"PRAGMA foreign_keys = ON;",
	} {
		if _, err := db.ExecContext(ctx, pragma); err != nil {
			_ = db.Close()
			return nil, fmt.Errorf("apply sqlite %q: %w", pragma, err)
		}
	}

	if _, err := db.ExecContext(ctx, schemaSQL); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("migrate sqlite schema: %w", err)
	}

	return &Store{
		db:  db,
		ttl: responsesTTL,
	}, nil
}

// TTL returns the configured responses TTL duration (0 means no automatic expiry).
func (s *Store) TTL() time.Duration {
	if s == nil {
		return 0
	}
	return s.ttl
}

// Ping verifies SQLite database liveness for /readyz checks.
func (s *Store) Ping(ctx context.Context) error {
	if s == nil || s.db == nil {
		return errors.New("sqlite store not initialized")
	}
	var one int
	return s.db.QueryRowContext(ctx, "SELECT 1").Scan(&one)
}

// Close closes the underlying SQLite connection pool.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// SaveResponse inserts or replaces a tenant-scoped response record.
func (s *Store) SaveResponse(ctx context.Context, rec StoredResponse) error {
	var prev any
	if rec.PreviousResponseID != "" {
		prev = rec.PreviousResponseID
	}
	var exp any
	if rec.ExpiresAt != nil {
		exp = *rec.ExpiresAt
	}

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO responses (
			id, tenant_id, previous_response_id, model, status,
			input_items_json, output_items_json, canonical_history_json, response_json,
			created_at, expires_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`,
		rec.ID,
		rec.TenantID,
		prev,
		rec.Model,
		rec.Status,
		rec.InputItemsJSON,
		rec.OutputItemsJSON,
		rec.CanonicalHistoryJSON,
		rec.ResponseJSON,
		rec.CreatedAt,
		exp,
	)
	if err != nil {
		return fmt.Errorf("insert response %q: %w", rec.ID, err)
	}
	return nil
}

// GetResponse loads a non-expired response strictly scoped to (id, tenantID).
func (s *Store) GetResponse(ctx context.Context, tenantID, id string) (*StoredResponse, error) {
	nowUnix := time.Now().Unix()
	row := s.db.QueryRowContext(ctx, `
		SELECT
			id, tenant_id, COALESCE(previous_response_id, ''), model, status,
			input_items_json, output_items_json, canonical_history_json, response_json,
			created_at, expires_at
		FROM responses
		WHERE id = ? AND tenant_id = ? AND (expires_at IS NULL OR expires_at > ?)
	`, id, tenantID, nowUnix)

	var rec StoredResponse
	var exp sql.NullInt64
	err := row.Scan(
		&rec.ID,
		&rec.TenantID,
		&rec.PreviousResponseID,
		&rec.Model,
		&rec.Status,
		&rec.InputItemsJSON,
		&rec.OutputItemsJSON,
		&rec.CanonicalHistoryJSON,
		&rec.ResponseJSON,
		&rec.CreatedAt,
		&exp,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("query response %q: %w", id, err)
	}
	if exp.Valid {
		v := exp.Int64
		rec.ExpiresAt = &v
	}
	return &rec, nil
}

// DeleteResponse deletes a response strictly scoped to (id, tenantID), returning true if a row existed.
func (s *Store) DeleteResponse(ctx context.Context, tenantID, id string) (bool, error) {
	res, err := s.db.ExecContext(ctx, `
		DELETE FROM responses WHERE id = ? AND tenant_id = ?
	`, id, tenantID)
	if err != nil {
		return false, fmt.Errorf("delete response %q: %w", id, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, err
	}
	return n > 0, nil
}

// CleanupExpired deletes expired response rows and returns the deleted row count.
func (s *Store) CleanupExpired(ctx context.Context, nowUnix int64) (int64, error) {
	res, err := s.db.ExecContext(ctx, `
		DELETE FROM responses WHERE expires_at IS NOT NULL AND expires_at <= ?
	`, nowUnix)
	if err != nil {
		return 0, fmt.Errorf("cleanup expired responses: %w", err)
	}
	return res.RowsAffected()
}

// StartCleanupLoop launches a background TTL reaper goroutine that exits when ctx is canceled.
func (s *Store) StartCleanupLoop(ctx context.Context, interval time.Duration, logger *slog.Logger) {
	if s == nil || s.ttl <= 0 || interval <= 0 {
		return
	}
	if logger == nil {
		logger = slog.Default()
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				cleanCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
				deleted, err := s.CleanupExpired(cleanCtx, time.Now().Unix())
				cancel()
				if err != nil {
					if ctx.Err() == nil {
						logger.Warn("sqlite responses TTL cleanup failed", "error", err.Error())
					}
				} else if deleted > 0 {
					logger.Debug("reaped expired responses from sqlite", "deleted_count", deleted)
				}
			}
		}
	}()
}
