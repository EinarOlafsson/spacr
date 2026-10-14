import { runDiagnosis } from "./diagnose.js";
import { hashIp, pruneOldCounters, reserveAll } from "./quota.js";
import { fileReport, issueBody, issueTitle, signatureOf } from "./report.js";
import { HttpError, VERSION, intVar, json, readJsonBody, refusedQuota, utcDay } from "./util.js";
import { validatePayload } from "./validate.js";

export const ROUTES = ["GET /health", "POST /report", "POST /diagnose"];

export const DEFAULT_LIMITS = Object.freeze({
  DAILY_REPORT_LIMIT: 200,
  DAILY_AI_LIMIT: 30,
  PER_INSTALL_DAILY_LIMIT: 10,
  PER_IP_DAILY_LIMIT: 20,
});

function limit(env, name) {
  return intVar(env, name, DEFAULT_LIMITS[name]);
}

async function buckets(env, request, route, globalRoute, globalLimitName, install, day) {
  const ipScope = await hashIp(request.headers.get("cf-connecting-ip"), day, env.IP_HASH_SALT);
  return [
    { name: "install", route, scope: `install:${install}`, limit: limit(env, "PER_INSTALL_DAILY_LIMIT") },
    { name: "ip", route, scope: ipScope, limit: limit(env, "PER_IP_DAILY_LIMIT") },
    { name: "global", route: globalRoute, scope: "global", limit: limit(env, globalLimitName) },
  ];
}

async function handleReport(request, env, ctx, now) {
  const body = await readJsonBody(request);
  const report = validatePayload(body, "report");
  const day = utcDay(now);
  const route = report.dry_run ? "report_dry_run" : "report";
  const plan = await buckets(env, request, route, route, "DAILY_REPORT_LIMIT", report.install_id, day);
  const quota = await reserveAll(env.DB, day, plan);
  if (!quota.ok) return refusedQuota(quota.refused);
  if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(pruneOldCounters(env.DB, day));
  if (report.dry_run) {
    const signature = await signatureOf(report);
    return json(200, {
      dry_run: true,
      title: issueTitle(report),
      body: issueBody(report, signature),
      signature,
    });
  }
  const result = await fileReport(env, report, { day, now: now.toISOString() });
  return json(200, result);
}

async function handleDiagnose(request, env, ctx, now) {
  const body = await readJsonBody(request);
  const report = validatePayload(body, "diagnose");
  const day = utcDay(now);
  const plan = await buckets(env, request, "diagnose", "ai", "DAILY_AI_LIMIT", report.install_id, day);
  const quota = await reserveAll(env.DB, day, plan);
  if (!quota.ok) return refusedQuota(quota.refused);
  if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(pruneOldCounters(env.DB, day));
  const result = await runDiagnosis(env, report);
  if (result.refused) return refusedQuota("ai");
  return json(200, result);
}

export async function handle(request, env, ctx, now = new Date()) {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  try {
    if (path === "/health" || path === "/") {
      if (request.method !== "GET" && request.method !== "HEAD") {
        throw new HttpError(405, "use GET");
      }
      return json(200, { ok: true, version: VERSION, routes: ROUTES });
    }
    if (path === "/report" || path === "/diagnose") {
      if (request.method !== "POST") throw new HttpError(405, "use POST");
      if (!env.DB) throw new HttpError(503, "relay is not configured");
      if (path === "/report") return await handleReport(request, env, ctx, now);
      if (!env.AI) throw new HttpError(503, "diagnosis is not available");
      return await handleDiagnose(request, env, ctx, now);
    }
    throw new HttpError(404, "not found");
  } catch (err) {
    if (err instanceof HttpError) return json(err.status, { error: err.message });
    console.log(`relay: unexpected ${err && err.name ? err.name : "error"} on ${path}`);
    return json(500, { error: "internal error" });
  }
}

export default {
  fetch(request, env, ctx) {
    return handle(request, env, ctx);
  },
};
