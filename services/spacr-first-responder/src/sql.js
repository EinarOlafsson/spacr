export const SQL = Object.freeze({
  reserve:
    "INSERT INTO daily_counters (day, route, scope, n) VALUES (?1, ?2, ?3, ?4) " +
    "ON CONFLICT (day, route, scope) DO UPDATE SET n = n + ?4 WHERE daily_counters.n + ?4 <= ?5 " +
    "RETURNING n",
  adjust:
    "UPDATE daily_counters SET n = MAX(0, n + ?4) WHERE day = ?1 AND route = ?2 AND scope = ?3",
  raiseTo:
    "INSERT INTO daily_counters (day, route, scope, n) VALUES (?1, ?2, ?3, ?4) " +
    "ON CONFLICT (day, route, scope) DO UPDATE SET n = MAX(daily_counters.n, excluded.n)",
  read: "SELECT n FROM daily_counters WHERE day = ?1 AND route = ?2 AND scope = ?3",
  countRoute: "SELECT COUNT(*) AS c FROM daily_counters WHERE day = ?1 AND route = ?2",
  prune: "DELETE FROM daily_counters WHERE day < ?1",
  bumpSignature:
    "UPDATE issue_signatures SET count = count + 1, last_seen = ?2 WHERE signature = ?1 " +
    "RETURNING issue_number, count, last_comment_day, has_summary",
  claimSignature:
    "INSERT INTO issue_signatures (signature, issue_number, count, first_seen, last_seen, last_comment_day) " +
    "VALUES (?1, 0, 1, ?2, ?2, NULL) ON CONFLICT (signature) DO NOTHING RETURNING signature",
  setIssueNumber: "UPDATE issue_signatures SET issue_number = ?2, has_summary = ?3 WHERE signature = ?1",
  resetSignature:
    "UPDATE issue_signatures SET issue_number = ?2, count = 1, last_comment_day = NULL, has_summary = ?3 WHERE signature = ?1",
  claimSummary:
    "UPDATE issue_signatures SET has_summary = 1 WHERE signature = ?1 AND has_summary = 0 RETURNING signature",
  unclaimSummary: "UPDATE issue_signatures SET has_summary = 0 WHERE signature = ?1",
  dropSignature: "DELETE FROM issue_signatures WHERE signature = ?1",
  cacheGet:
    "SELECT kind, text, model, tier, cause_slug, spacr_minor, created FROM ai_cache WHERE key = ?1",
  cachePut:
    "INSERT INTO ai_cache (key, kind, text, model, tier, cause_slug, spacr_minor, created, last_used, hits, bytes) " +
    "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, 0, ?9) ON CONFLICT (key) DO UPDATE SET kind = excluded.kind, " +
    "text = excluded.text, model = excluded.model, tier = excluded.tier, cause_slug = excluded.cause_slug, " +
    "spacr_minor = excluded.spacr_minor, created = excluded.created, last_used = excluded.last_used, hits = 0, " +
    "bytes = excluded.bytes",
  cacheHit: "UPDATE ai_cache SET hits = hits + 1, last_used = ?2 WHERE key = ?1",
  cacheDrop: "DELETE FROM ai_cache WHERE key = ?1 OR key = ?2",
  cacheExpire: "DELETE FROM ai_cache WHERE created < ?1",
  cacheEvict:
    "DELETE FROM ai_cache WHERE key IN (SELECT key FROM ai_cache ORDER BY hits ASC, last_used ASC LIMIT ?1)",
  cacheCount: "SELECT COUNT(*) AS rows, COALESCE(SUM(bytes), 0) AS bytes FROM ai_cache",
  cacheMetaBump:
    "INSERT INTO ai_cache_meta (id, rows, bytes, inserts) VALUES (1, 1, ?1, 1) ON CONFLICT (id) DO UPDATE SET " +
    "rows = ai_cache_meta.rows + 1, bytes = ai_cache_meta.bytes + ?1, inserts = ai_cache_meta.inserts + 1 " +
    "RETURNING rows, bytes, inserts",
  cacheMetaSet:
    "INSERT INTO ai_cache_meta (id, rows, bytes, inserts) VALUES (1, ?1, ?2, 0) ON CONFLICT (id) DO UPDATE SET " +
    "rows = ?1, bytes = ?2",
  cacheMetaRead: "SELECT rows, bytes, inserts FROM ai_cache_meta WHERE id = 1",
  getCause: "SELECT slug, hub_issue, count FROM ai_causes WHERE slug = ?1",
  insertCause:
    "INSERT INTO ai_causes (slug, hub_issue, count, first_seen, last_seen) VALUES (?1, ?2, 1, ?3, ?3) " +
    "ON CONFLICT (slug) DO NOTHING",
  bumpCause: "UPDATE ai_causes SET count = count + 1, last_seen = ?2 WHERE slug = ?1",
  setCauseHub: "UPDATE ai_causes SET hub_issue = ?2 WHERE slug = ?1",
  claimDailyComment:
    "UPDATE issue_signatures SET last_comment_day = ?2 WHERE signature = ?1 " +
    "AND issue_number > 0 AND (last_comment_day IS NULL OR last_comment_day <> ?2) " +
    "RETURNING issue_number, count, has_summary",
});
