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

-- Saved AI explanations, so a repeated error is answered without a model call.
-- signature: the error signature (src/report.js signatureOf); a relay-written
-- issue summary is stored under "summary:<signature>".
-- spacr_minor: the reporter's spaCR major.minor; another version asks again.
-- created: ISO time the text was written; older than 30 days asks again.
-- hits: how many requests reused it. Dropped when its issue is found closed.
CREATE TABLE IF NOT EXISTS ai_explanations (
  signature   TEXT    PRIMARY KEY,
  text        TEXT    NOT NULL,
  model       TEXT    NOT NULL,
  tier        TEXT    NOT NULL,
  spacr_minor TEXT    NOT NULL,
  created     TEXT    NOT NULL,
  hits        INTEGER NOT NULL DEFAULT 0,
  cause       TEXT
);

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
