import { SQL } from "./sql.js";
import { sha256Hex } from "./util.js";

export const ROUTE = Object.freeze({
  report: "report",
  reportDryRun: "report_dry_run",
  diagnose: "diagnose",
  aiBest: "ai_best",
  aiLight: "ai_light",
  aiNeurons: "ai_neurons",
  aiInstallNeurons: "ai_install_neurons",
  cacheHit: "cache_hit",
  cacheMiss: "cache_miss",
});

export const GLOBAL = "global";

export function installScope(installId) {
  return `install:${installId}`;
}

export async function hashIp(ip, day, salt) {
  if (!ip) return "ip:unknown";
  const digest = await sha256Hex(`${salt || ""}|${day}|${ip}`);
  return `ip:${digest.slice(0, 32)}`;
}

export async function reserve(db, day, route, scope, amount, limit) {
  if (!(amount > 0) || !(limit > 0) || amount > limit) return null;
  const row = await db.prepare(SQL.reserve).bind(day, route, scope, amount, limit).first();
  if (!row || typeof row.n !== "number" || row.n > limit) return null;
  return row.n;
}

export async function adjust(db, day, route, scope, delta) {
  if (!delta) return;
  await db.prepare(SQL.adjust).bind(day, route, scope, delta).run();
}

export async function release(db, day, route, scope, amount = 1) {
  try {
    await adjust(db, day, route, scope, -amount);
  } catch (_) {
    /* best effort */
  }
}

export async function raiseTo(db, day, route, scope, value) {
  await db.prepare(SQL.raiseTo).bind(day, route, scope, value).run();
}

export async function readCount(db, day, route, scope) {
  const row = await db.prepare(SQL.read).bind(day, route, scope).first();
  return row && typeof row.n === "number" ? row.n : 0;
}

export async function readRow(db, day, route, scope) {
  return (await db.prepare(SQL.read).bind(day, route, scope).first()) || null;
}

export async function countRoute(db, day, route) {
  const row = await db.prepare(SQL.countRoute).bind(day, route).first();
  return row && typeof row.c === "number" ? row.c : 0;
}

export async function reserveAll(db, day, buckets) {
  const taken = [];
  for (const b of buckets) {
    const n = await reserve(db, day, b.route, b.scope, 1, b.limit);
    if (n === null) {
      for (const t of taken.reverse()) await release(db, day, t.route, t.scope);
      return { ok: false, refused: b.name };
    }
    taken.push({ ...b, n });
  }
  return { ok: true, taken };
}

export async function pruneOldCounters(db, day) {
  try {
    await db.prepare(SQL.prune).bind(day).run();
  } catch (_) {
    /* best effort */
  }
}
