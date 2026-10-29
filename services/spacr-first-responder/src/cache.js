import { SQL } from "./sql.js";
import { MODEL_RATES, limits } from "./config.js";
import { SYSTEM_PROMPT } from "./prompts.js";
import { GLOBAL, ROUTE } from "./quota.js";
import { sha256Hex, utcDay } from "./util.js";

export const SUMMARY_PREFIX = "summary:";
export const QUESTION_PREFIX = "q:";
export const QUESTION_CONTEXT_MAX = 500;
export const CLEANUP_EVERY = 50;
export const EVICT_FRACTION = 0.1;
export const CACHE_DEFAULTS = Object.freeze({ maxRows: 20000, maxBytes: 60000000, ttlDays: 30 });

const COUNTER_LIMIT = 1_000_000_000;

function intVar(env, name, fallback) {
  const n = Number.parseInt(env && env[name], 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export function cacheLimits(env) {
  return {
    maxRows: intVar(env, "AI_CACHE_MAX_ROWS", CACHE_DEFAULTS.maxRows),
    maxBytes: intVar(env, "AI_CACHE_MAX_BYTES", CACHE_DEFAULTS.maxBytes),
    ttlDays: intVar(env, "AI_CACHE_TTL_DAYS", CACHE_DEFAULTS.ttlDays),
  };
}

export function spacrMinor(version) {
  const m = /^(\d+)\.(\d+)/.exec(String(version || ""));
  return m ? `${m[1]}.${m[2]}` : String(version || "");
}

export function normaliseQuestion(question) {
  return String(question || "")
    .toLowerCase()
    .replace(/<[^<>]*>/g, " ")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function questionContextChars(report) {
  const frames = (report.frames || []).reduce(
    (n, f) => n + String(f.file || "").length + String(f.function || "").length + String(f.code || "").length,
    0,
  );
  return String(report.question || "").length + String(report.message || "").length + frames;
}

export async function questionKey(report) {
  if (questionContextChars(report) > QUESTION_CONTEXT_MAX) return null;
  const norm = normaliseQuestion(report.question);
  if (!norm) return null;
  const place = `${report.screen || ""}|${report.module || ""}`;
  return QUESTION_PREFIX + (await sha256Hex(`${norm}\n${place}\n${spacrMinor(report.spacr_version)}\n${SYSTEM_PROMPT}`));
}

export function isFresh(row, report, now, ttlDays = CACHE_DEFAULTS.ttlDays) {
  if (!row || !row.text) return false;
  if (row.spacr_minor !== spacrMinor(report.spacr_version)) return false;
  const created = Number(row.created);
  if (!Number.isFinite(created)) return false;
  return Math.floor(now.getTime() / 1000) - created < ttlDays * 86400;
}

async function countToday(db, now, route) {
  try {
    await db.prepare(SQL.reserve).bind(utcDay(now), route, GLOBAL, 1, COUNTER_LIMIT).first();
  } catch (_) {
    /* the hit rate is a statistic only */
  }
}

export async function cached(env, key, report, now) {
  const db = env.DB;
  try {
    const row = await db.prepare(SQL.cacheGet).bind(key).first();
    const cfg = limits(env);
    const modelAllowed = row && MODEL_RATES[row.model] && [cfg.bestModel, cfg.lightModel].includes(row.model);
    if (!isFresh(row, report, now, cacheLimits(env).ttlDays) || !modelAllowed) {
      await countToday(db, now, ROUTE.cacheMiss);
      return null;
    }
    await db.prepare(SQL.cacheHit).bind(key, Math.floor(now.getTime() / 1000)).run();
    await countToday(db, now, ROUTE.cacheHit);
    return { text: row.text, model: row.model, tier: row.tier, cause: row.cause_slug || null };
  } catch (_) {
    return null;
  }
}

export async function cleanup(env, now) {
  const db = env.DB;
  const lim = cacheLimits(env);
  const nowS = Math.floor(now.getTime() / 1000);
  await db.prepare(SQL.cacheExpire).bind(nowS - lim.ttlDays * 86400).run();
  let count = await db.prepare(SQL.cacheCount).bind().first();
  let rows = Number((count && count.rows) || 0);
  let bytes = Number((count && count.bytes) || 0);
  if (rows > lim.maxRows || bytes > lim.maxBytes) {
    const evict = Math.max(1, Math.ceil(rows * EVICT_FRACTION), rows - lim.maxRows);
    await db.prepare(SQL.cacheEvict).bind(evict).run();
    count = await db.prepare(SQL.cacheCount).bind().first();
    rows = Number((count && count.rows) || 0);
    bytes = Number((count && count.bytes) || 0);
  }
  await db.prepare(SQL.cacheMetaSet).bind(rows, bytes).run();
  return { rows, bytes };
}

export async function store(env, key, kind, report, now, { text, model, tier, cause = null }) {
  const db = env.DB;
  try {
    const nowS = Math.floor(now.getTime() / 1000);
    const bytes = new TextEncoder().encode(text).length + key.length;
    await db
      .prepare(SQL.cachePut)
      .bind(key, kind, text, model || "", tier || "", cause, spacrMinor(report.spacr_version), nowS, bytes)
      .run();
    const meta = await db.prepare(SQL.cacheMetaBump).bind(bytes).first();
    const lim = cacheLimits(env);
    if (!meta || meta.inserts % CLEANUP_EVERY === 0 || meta.rows > lim.maxRows || meta.bytes > lim.maxBytes) {
      await cleanup(env, now);
    }
  } catch (_) {
    /* a lost cache write only costs a later model call */
  }
}

export async function causeOnFile(db, signature) {
  try {
    for (const key of [signature, SUMMARY_PREFIX + signature]) {
      const row = await db.prepare(SQL.cacheGet).bind(key).first();
      if (row && row.cause_slug) return row.cause_slug;
    }
  } catch (_) {
    /* no cause known */
  }
  return null;
}

export async function dropError(db, signature) {
  try {
    await db.prepare(SQL.cacheDrop).bind(signature, SUMMARY_PREFIX + signature).run();
  } catch (_) {
    /* the next fresh answer replaces it anyway */
  }
}

export async function cacheStats(db, now) {
  try {
    const meta = await db.prepare(SQL.cacheMetaRead).bind().first();
    const day = utcDay(now);
    const hit = await db.prepare(SQL.read).bind(day, ROUTE.cacheHit, GLOBAL).first();
    const miss = await db.prepare(SQL.read).bind(day, ROUTE.cacheMiss, GLOBAL).first();
    const hits = Number((hit && hit.n) || 0);
    const misses = Number((miss && miss.n) || 0);
    return {
      rows: Number((meta && meta.rows) || 0),
      bytes: Number((meta && meta.bytes) || 0),
      today: { hits, misses, hit_rate: hits + misses ? Math.round((hits / (hits + misses)) * 1000) / 1000 : null },
    };
  } catch (_) {
    return null;
  }
}
