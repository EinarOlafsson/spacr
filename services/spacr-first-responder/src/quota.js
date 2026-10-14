import { SQL } from "./sql.js";
import { sha256Hex } from "./util.js";

export async function hashIp(ip, day, salt) {
  if (!ip) return "ip:unknown";
  const digest = await sha256Hex(`${salt || ""}|${day}|${ip}`);
  return `ip:${digest.slice(0, 32)}`;
}

async function reserveOne(db, day, route, scope, limit) {
  if (!(limit > 0)) return false;
  const row = await db.prepare(SQL.reserve).bind(day, route, scope, limit).first();
  return Boolean(row && typeof row.n === "number" && row.n <= limit);
}

async function releaseOne(db, day, route, scope) {
  await db.prepare(SQL.release).bind(day, route, scope).run();
}

export async function reserveAll(db, day, buckets) {
  const taken = [];
  for (const b of buckets) {
    const ok = await reserveOne(db, day, b.route, b.scope, b.limit);
    if (!ok) {
      for (const t of taken.reverse()) {
        try {
          await releaseOne(db, day, t.route, t.scope);
        } catch (_) {
          /* best effort */
        }
      }
      return { ok: false, refused: b.name };
    }
    taken.push(b);
  }
  return { ok: true };
}

export async function pruneOldCounters(db, day) {
  try {
    await db.prepare(SQL.prune).bind(day).run();
  } catch (_) {
    /* best effort */
  }
}
