import { nextUtcMidnight } from "./config.js";
import { TYPICAL_PROMPT_TOKENS, neuronCost, poolState, typicalCost } from "./neurons.js";
import { GLOBAL, ROUTE, countRoute, installScope, readCount, readRow, release, reserve, adjust } from "./quota.js";

export const ACTIVE_USERS_DEFINITION =
  "installs that used hosted AI (diagnosis or report summary) today, UTC, counting you; allowance queries do not count";

export function shareFor(cfg, activeUsers) {
  const fair = Math.floor(cfg.neuronCap / (activeUsers + 1));
  return Math.min(cfg.neuronCap, Math.max(cfg.minShare, fair));
}

function pct(part, whole) {
  return whole > 0 ? Math.round((part / whole) * 1000) / 10 : 0;
}

export async function aiState(db, day, cfg, installId, aiAvailable = true) {
  const scope = installScope(installId);
  const mine = await readRow(db, day, ROUTE.aiInstallNeurons, scope);
  const others = await countRoute(db, day, ROUTE.aiInstallNeurons);
  const activeUsers = others + (mine ? 0 : 1);
  const share = shareFor(cfg, activeUsers);
  const used = mine && typeof mine.n === "number" ? mine.n : 0;
  const poolUsed = await readCount(db, day, ROUTE.aiNeurons, GLOBAL);
  return {
    scope,
    activeUsers,
    share,
    used,
    remaining: Math.max(0, share - used),
    poolUsed,
    pool: aiAvailable ? poolState(cfg, poolUsed) : "exhausted",
    bestUsed: cfg.bestPerInstall > 0 ? await readCount(db, day, ROUTE.aiBest, scope) : 0,
    lightUsed: cfg.lightPerInstall > 0 ? await readCount(db, day, ROUTE.aiLight, scope) : 0,
  };
}

function capLeft(cap, used) {
  return cap > 0 ? Math.max(0, cap - used) : Infinity;
}

export function tierFor(cfg, st, bestCost, lightCost) {
  if (st.pool === "exhausted") return "none";
  if (st.pool === "best" && capLeft(cfg.bestPerInstall, st.bestUsed) > 0 && st.remaining >= bestCost) return "best";
  if (capLeft(cfg.lightPerInstall, st.lightUsed) > 0 && st.remaining >= lightCost) return "light";
  return "none";
}

export function aiView(cfg, st) {
  const bestLeft =
    st.pool === "best"
      ? Math.min(Math.floor(st.remaining / typicalCost(cfg.bestModel)), capLeft(cfg.bestPerInstall, st.bestUsed))
      : 0;
  const lightLeft =
    st.pool !== "exhausted"
      ? Math.min(Math.floor(st.remaining / typicalCost(cfg.lightModel)), capLeft(cfg.lightPerInstall, st.lightUsed))
      : 0;
  const tierNow = tierFor(
    cfg,
    st,
    neuronCost(cfg.bestModel, TYPICAL_PROMPT_TOKENS, 600),
    neuronCost(cfg.lightModel, TYPICAL_PROMPT_TOKENS, 600),
  );
  return {
    neurons_left: st.remaining,
    share_neurons: st.share,
    share_pct: pct(st.share, cfg.neuronCap),
    active_users: st.activeUsers,
    active_users_definition: ACTIVE_USERS_DEFINITION,
    best_left: bestLeft,
    light_left: lightLeft,
    tier_now: tierNow,
  };
}

export async function reportsLeft(db, day, cfg, installId) {
  const used = await readCount(db, day, ROUTE.report, installScope(installId));
  return Math.max(0, cfg.reportsPerInstall - used);
}

export async function buildAllowance(db, day, cfg, installId, now, aiAvailable = true) {
  const st = await aiState(db, day, cfg, installId, aiAvailable);
  const ai = aiView(cfg, st);
  return {
    ai,
    best_left: ai.best_left,
    light_left: ai.light_left,
    reports_left: await reportsLeft(db, day, cfg, installId),
    pool: st.pool,
    pool_pct_left: pct(Math.max(0, cfg.neuronCap - st.poolUsed), cfg.neuronCap),
    resets_at: nextUtcMidnight(now),
  };
}

function tierSpec(cfg, tier) {
  return tier === "best"
    ? { model: cfg.bestModel, capRoute: ROUTE.aiBest, cap: cfg.bestPerInstall }
    : { model: cfg.lightModel, capRoute: ROUTE.aiLight, cap: cfg.lightPerInstall };
}

export async function grantAi(db, day, cfg, st, inputTokens, maxOutput, { allowBest = true } = {}) {
  if (st.pool === "exhausted") return { ok: false, scope: "global" };
  const order = allowBest && st.pool === "best" ? ["best", "light"] : ["light"];
  let reason = "install";
  for (const tier of order) {
    const spec = tierSpec(cfg, tier);
    const cost = neuronCost(spec.model, inputTokens, maxOutput);
    if (spec.cap > 0 && (await reserve(db, day, spec.capRoute, st.scope, 1, spec.cap)) === null) continue;
    const mine = await reserve(db, day, ROUTE.aiInstallNeurons, st.scope, cost, st.share);
    if (mine === null) {
      if (spec.cap > 0) await release(db, day, spec.capRoute, st.scope);
      continue;
    }
    const pool = await reserve(db, day, ROUTE.aiNeurons, GLOBAL, cost, cfg.neuronCap);
    if (pool === null) {
      await release(db, day, ROUTE.aiInstallNeurons, st.scope, cost);
      if (spec.cap > 0) await release(db, day, spec.capRoute, st.scope);
      reason = "global";
      continue;
    }
    return { ok: true, tier, model: spec.model, reserved: cost, scope: st.scope, spec };
  }
  return { ok: false, scope: reason };
}

export async function refundUser(db, day, grant) {
  await release(db, day, ROUTE.aiInstallNeurons, grant.scope, grant.reserved);
  if (grant.spec.cap > 0) await release(db, day, grant.spec.capRoute, grant.scope);
}

export async function settle(db, day, grant, actual) {
  const delta = actual - grant.reserved;
  if (!delta) return;
  try {
    await adjust(db, day, ROUTE.aiNeurons, GLOBAL, delta);
    await adjust(db, day, ROUTE.aiInstallNeurons, grant.scope, delta);
  } catch (_) {
    /* keep the conservative reservation */
  }
}
