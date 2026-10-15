-- spacr-first-responder D1 schema (database spacr-support).
-- Apply with: npx wrangler d1 execute spacr-support --remote --file=schema.sql
-- Safe to re-run: every statement is idempotent.

-- Daily counters, one row per UTC day, route and scope.
-- route: report, report_dry_run, diagnose, ai
-- scope: 'global', 'install:<uuid>' or 'ip:<truncated salted SHA-256>'; raw IPs are never stored.
-- Rows from earlier days are pruned opportunistically by the Worker.
CREATE TABLE IF NOT EXISTS daily_counters (
  day   TEXT    NOT NULL,
  route TEXT    NOT NULL,
  scope TEXT    NOT NULL,
  n     INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, route, scope)
);

-- Issue de-duplication: error signature -> GitHub issue.
-- issue_number 0 means another request is creating the issue right now.
-- last_comment_day caps "+1" comments at one per signature per UTC day.
-- has_summary is 1 once the issue or one of its comments carries an AI summary.
CREATE TABLE IF NOT EXISTS issue_signatures (
  signature        TEXT    PRIMARY KEY,
  issue_number     INTEGER NOT NULL DEFAULT 0,
  count            INTEGER NOT NULL DEFAULT 1,
  first_seen       TEXT    NOT NULL,
  last_seen        TEXT    NOT NULL,
  last_comment_day TEXT,
  has_summary      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS issue_signatures_issue ON issue_signatures (issue_number);

-- Saved AI answers, so a repeated error or question costs no model call.
-- key: an error signature (src/report.js signatureOf; a user error's includes
-- its normalised problem list), "summary:<signature>" for a relay-written issue
-- summary, or "q:<SHA-256>" of a normalised question + screen|module + spaCR
-- major.minor. Read by primary key only.
-- kind: 'error' or 'question'. spacr_minor: the asker's spaCR major.minor;
-- another one asks again. created / last_used: Unix seconds. bytes: stored size.
-- Expired after AI_CACHE_TTL_DAYS; past AI_CACHE_MAX_ROWS or AI_CACHE_MAX_BYTES
-- about 10% go, fewest hits and least recently used first. An error's entries
-- are dropped when its issue is found closed.
CREATE TABLE IF NOT EXISTS ai_cache (
  key         TEXT    PRIMARY KEY,
  kind        TEXT    NOT NULL CHECK (kind IN ('error', 'question')),
  text        TEXT    NOT NULL,
  model       TEXT    NOT NULL,
  tier        TEXT    NOT NULL,
  cause_slug  TEXT,
  spacr_minor TEXT    NOT NULL,
  created     INTEGER NOT NULL,
  last_used   INTEGER NOT NULL,
  hits        INTEGER NOT NULL DEFAULT 0,
  bytes       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ai_cache_eviction ON ai_cache (hits, last_used);

-- One row (id 1) with the cache's running row and byte counts and its insert
-- count, so no request counts the table; corrected at every cleanup.
CREATE TABLE IF NOT EXISTS ai_cache_meta (
  id      INTEGER PRIMARY KEY CHECK (id = 1),
  rows    INTEGER NOT NULL DEFAULT 0,
  bytes   INTEGER NOT NULL DEFAULT 0,
  inserts INTEGER NOT NULL DEFAULT 0
);

-- The 0.2.4 cache table, replaced by ai_cache.
DROP TABLE IF EXISTS ai_explanations;

-- Likely causes the AI named (CAUSE: <slug>), and the hub issue for each:
-- the oldest open issue with that cause, which gets a comment for every new
-- issue with the same cause. Different signatures stay separate issues.
CREATE TABLE IF NOT EXISTS ai_causes (
  slug       TEXT    PRIMARY KEY,
  hub_issue  INTEGER NOT NULL,
  count      INTEGER NOT NULL DEFAULT 1,
  first_seen TEXT    NOT NULL,
  last_seen  TEXT    NOT NULL
);
