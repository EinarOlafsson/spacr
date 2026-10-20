import { cacheStats } from "./cache.js";
import { limits } from "./config.js";
import { buildAllowance } from "./allowance.js";
import { handleDiagnose, summarizeWithRelay } from "./diagnose.js";
import { GLOBAL, ROUTE, hashIp, installScope, pruneOldCounters, reserveAll } from "./quota.js";
import { causeFor, fileReport, issueBody, issueTitle, obtainSummary, signatureOf, summaryFields } from "./report.js";
import { HttpError, VERSION, json, readJsonBody, refusedQuota, utcDay } from "./util.js";
import { UUID_RE, validatePayload } from "./validate.js";

export const ROUTES = ["GET /health", "GET /allowance", "POST /report", "POST /diagnose"];

async function handleReport(request, env, ctx, now) {
  const body = await readJsonBody(request);
  const report = validatePayload(body, "report");
  const cfg = limits(env);
  const day = utcDay(now);
  const route = report.dry_run ? ROUTE.reportDryRun : ROUTE.report;
  const ipScope = await hashIp(request.headers.get("cf-connecting-ip"), day, env.IP_HASH_SALT);
  const quota = await reserveAll(env.DB, day, [
    { name: "install", route, scope: installScope(report.install_id), limit: cfg.reportsPerInstall },
    { name: "ip", route, scope: ipScope, limit: cfg.perIp },
    { name: "global", route, scope: GLOBAL, limit: cfg.dailyReports },
  ]);
  if (!quota.ok) return refusedQuota(quota.refused);
  if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(pruneOldCounters(env.DB, day));
  const getSummary = () => summarizeWithRelay(env, report, now);
  const allowance = () => buildAllowance(env.DB, day, cfg, report.install_id, now, Boolean(env.AI));
  if (report.dry_run) {
    const signature = await signatureOf(report);
    const summary = await obtainSummary(report, getSummary);
    return json(200, {
      dry_run: true,
      title: issueTitle(report),
      body: issueBody(report, signature, summary, { cause: await causeFor(env, report, signature, summary), related: [] }),
      signature,
      ...summaryFields(summary),
      allowance: await allowance(),
    });
  }
  const result = await fileReport(env, report, { day, now: now.toISOString(), getSummary });
  return json(200, { ...result, allowance: await allowance() });
}

async function handleAllowance(url, env, now) {
  const installId = url.searchParams.get("install_id") || "";
  if (!UUID_RE.test(installId)) throw new HttpError(400, "query parameter install_id must be a UUID");
  const id = installId.toLowerCase();
  const cfg = limits(env);
  const day = utcDay(now);
  return json(200, await buildAllowance(env.DB, day, cfg, id, now, Boolean(env.AI)));
}

export async function handle(request, env, ctx, now = new Date()) {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  try {
    if (path === "/health" || path === "/") {
      if (request.method !== "GET" && request.method !== "HEAD") throw new HttpError(405, "use GET");
      const health = { ok: true, version: VERSION, routes: ROUTES };
      if (env && env.DB) {
        const cache = await cacheStats(env.DB, now);
        if (cache) health.cache = cache;
      }
      return json(200, health);
    }
    if (path === "/allowance") {
      if (request.method !== "GET") throw new HttpError(405, "use GET");
      if (!env.DB) throw new HttpError(503, "relay is not configured");
      return await handleAllowance(url, env, now);
    }
    if (path === "/report" || path === "/diagnose") {
      if (request.method !== "POST") throw new HttpError(405, "use POST");
      if (!env.DB) throw new HttpError(503, "relay is not configured");
      if (path === "/report") return await handleReport(request, env, ctx, now);
      if (!env.AI) throw new HttpError(503, "diagnosis is not available");
      const report = validatePayload(await readJsonBody(request), "diagnose");
      const res = await handleDiagnose(report, request, env, now);
      if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(pruneOldCounters(env.DB, utcDay(now)));
      return res;
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
