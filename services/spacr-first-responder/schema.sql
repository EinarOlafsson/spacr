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
CREATE TABLE IF NOT EXISTS issue_signatures (
  signature        TEXT    PRIMARY KEY,
  issue_number     INTEGER NOT NULL DEFAULT 0,
  count            INTEGER NOT NULL DEFAULT 1,
  first_seen       TEXT    NOT NULL,
  last_seen        TEXT    NOT NULL,
  last_comment_day TEXT
);

CREATE INDEX IF NOT EXISTS issue_signatures_issue ON issue_signatures (issue_number);
