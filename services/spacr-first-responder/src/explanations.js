import { SQL } from "./sql.js";

export const EXPLANATION_MAX_AGE_DAYS = 30;
export const SUMMARY_PREFIX = "summary:";

export function spacrMinor(version) {
  const m = /^(\d+)\.(\d+)/.exec(String(version || ""));
  return m ? `${m[1]}.${m[2]}` : String(version || "");
}

export function isFresh(row, report, now, maxAgeDays = EXPLANATION_MAX_AGE_DAYS) {
  if (!row || !row.text) return false;
  if (row.spacr_minor !== spacrMinor(report.spacr_version)) return false;
  const created = Date.parse(row.created);
  if (!Number.isFinite(created)) return false;
  return now.getTime() - created < maxAgeDays * 86400000;
}

export async function savedExplanation(db, key, report, now) {
  try {
    const row = await db.prepare(SQL.getExplanation).bind(key).first();
    if (!isFresh(row, report, now)) return null;
    await db.prepare(SQL.hitExplanation).bind(key).run();
    return { text: row.text, model: row.model, tier: row.tier, cause: row.cause || null };
  } catch (_) {
    return null;
  }
}

export async function causeOnFile(db, signature) {
  try {
    for (const key of [signature, SUMMARY_PREFIX + signature]) {
      const row = await db.prepare(SQL.getExplanation).bind(key).first();
      if (row && row.cause) return row.cause;
    }
  } catch (_) {
    /* no cause known */
  }
  return null;
}

export async function saveExplanation(db, key, report, now, { text, model, tier, cause = null }) {
  try {
    await db
      .prepare(SQL.putExplanation)
      .bind(key, text, model, tier, spacrMinor(report.spacr_version), now.toISOString(), cause)
      .run();
  } catch (_) {
    /* a lost cache write only costs a later model call */
  }
}

export async function dropExplanations(db, signature) {
  try {
    await db.prepare(SQL.dropExplanations).bind(signature, SUMMARY_PREFIX + signature).run();
  } catch (_) {
    /* the next fresh answer replaces it anyway */
  }
}
