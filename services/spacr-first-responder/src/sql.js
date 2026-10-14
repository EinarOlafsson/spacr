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
  prune: "DELETE FROM daily_counters WHERE day < ?1",
  bumpSignature:
    "UPDATE issue_signatures SET count = count + 1, last_seen = ?2 WHERE signature = ?1 " +
    "RETURNING issue_number, count, last_comment_day, has_summary",
  claimSignature:
    "INSERT INTO issue_signatures (signature, issue_number, count, first_seen, last_seen, last_comment_day) " +
    "VALUES (?1, 0, 1, ?2, ?2, NULL) ON CONFLICT (signature) DO NOTHING RETURNING signature",
  setIssueNumber: "UPDATE issue_signatures SET issue_number = ?2, has_summary = ?3 WHERE signature = ?1",
  claimSummary:
    "UPDATE issue_signatures SET has_summary = 1 WHERE signature = ?1 AND has_summary = 0 RETURNING signature",
  unclaimSummary: "UPDATE issue_signatures SET has_summary = 0 WHERE signature = ?1",
  dropSignature: "DELETE FROM issue_signatures WHERE signature = ?1",
  claimDailyComment:
    "UPDATE issue_signatures SET last_comment_day = ?2 WHERE signature = ?1 " +
    "AND issue_number > 0 AND (last_comment_day IS NULL OR last_comment_day <> ?2) " +
    "RETURNING issue_number, count, has_summary",
});
