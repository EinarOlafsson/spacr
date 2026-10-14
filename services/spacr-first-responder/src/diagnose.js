import { MODEL_RATES, limits, nextUtcMidnight } from "./config.js";
import {
  GLOBAL,
  ROUTE,
  hashIp,
  installScope,
  raiseTo,
  readCount,
  release,
  reserve,
  adjust,
} from "./quota.js";
import { environmentLine, tracebackText } from "./report.js";
import { HttpError, MAX_BODY_BYTES, json, utcDay } from "./util.js";

export const MAX_OUTPUT_TOKENS = 600;
export const MAX_ADVICE_CHARS = 6000;
export const CHARS_PER_TOKEN_FLOOR = 2.5;

export const SYSTEM_PROMPT = [
  "You are the spaCR assistant. spaCR is an open-source Python desktop application for",
  "microscopy image analysis and pooled CRISPR screens: it segments cells, nuclei and",
  "pathogens (Cellpose models), measures phenotypes, trains classifiers, and maps",
  "sequencing reads to guide RNAs. It runs on Windows, macOS and Linux with Python, Qt",
  "and PyTorch.",
  "",
  "A user sends an anonymised error report: exception type, a short message, the spaCR",
  "module, and a traceback with file names only. Help them recover:",
  "1. Give the most likely causes, most likely first, each in one or two sentences.",
  "2. Give concrete next steps the user can try inside spaCR or their environment",
  "   (settings to check, inputs to verify, package versions, GPU/CPU fallbacks).",
  "3. Say when the report looks like a spaCR bug that the maintainers should fix.",
  "Never claim certainty; say what you are unsure about. Do not invent spaCR settings,",
  "functions or files you cannot see. Do not ask for personal data, paths or",
  "credentials. Keep the answer under 300 words and use short Markdown lists.",
].join("\n");

export const MAX_PROMPT_TOKENS = Math.ceil((MAX_BODY_BYTES + SYSTEM_PROMPT.length + 400) / CHARS_PER_TOKEN_FLOOR);

export function neuronCost(model, inputTokens, outputTokens) {
  const r = MODEL_RATES[model];
  return Math.ceil((inputTokens * r.input + outputTokens * r.output) / 1e6);
}

export function estimateTokens(text) {
  return Math.ceil(text.length / CHARS_PER_TOKEN_FLOOR) + 8;
}

export function worstCaseCost(model, inputTokens = MAX_PROMPT_TOKENS) {
  return neuronCost(model, inputTokens, MAX_OUTPUT_TOKENS);
}

export function userPrompt(r) {
  const parts = [
    `Environment: ${environmentLine(r)}`,
    `Module: ${r.module}`,
    `Exception: ${r.exception_type}: ${r.message || ""}`,
    "Traceback (most recent call last):",
    tracebackText(r.frames),
  ];
  if (r.user_note) parts.push("", `User note: ${r.user_note}`);
  if (r.question) parts.push("", `User question: ${r.question}`);
  return parts.join("\n");
}

export function isAllocationError(err) {
  return /\b4006\b|neuron|daily free allocation/i.test(String((err && (err.message || err)) || ""));
}

export function isRateError(err) {
  return /rate limit|too many requests|capacity|\b3040\b/i.test(String((err && (err.message || err)) || ""));
}

function checkModels(cfg) {
  if (!MODEL_RATES[cfg.bestModel] || !MODEL_RATES[cfg.lightModel]) {
    console.log("diagnose: a configured model is not on the free allowlist");
    throw new HttpError(503, "diagnosis is not available");
  }
}

export function poolState(cfg, used) {
  const remaining = cfg.neuronCap - used;
  if (remaining >= cfg.lightThreshold * cfg.neuronCap && remaining >= worstCaseCost(cfg.bestModel)) {
    return "best";
  }
  if (remaining >= worstCaseCost(cfg.lightModel)) return "light";
  return "exhausted";
}

export async function aiAllowance(db, day, cfg, installId) {
  const scope = installScope(installId);
  const best = await readCount(db, day, ROUTE.aiBest, scope);
  const light = await readCount(db, day, ROUTE.aiLight, scope);
  return {
    best_left: Math.max(0, cfg.bestPerInstall - best),
    light_left: Math.max(0, cfg.lightPerInstall - light),
  };
}

function extractAdvice(result) {
  if (!result) return "";
  if (typeof result.response === "string") return result.response;
  const choice = Array.isArray(result.choices) ? result.choices[0] : null;
  if (choice && choice.message && typeof choice.message.content === "string") return choice.message.content;
  return "";
}

function refused(scope) {
  return json(429, { refused: "quota", scope });
}

async function tryTier(db, day, cfg, scope, tier, inputTokens) {
  const route = tier === "best" ? ROUTE.aiBest : ROUTE.aiLight;
  const perInstall = tier === "best" ? cfg.bestPerInstall : cfg.lightPerInstall;
  const model = tier === "best" ? cfg.bestModel : cfg.lightModel;
  const slot = await reserve(db, day, route, scope, 1, perInstall);
  if (slot === null) return { ok: false, reason: "install" };
  const cost = worstCaseCost(model, inputTokens);
  const pool = await reserve(db, day, ROUTE.aiNeurons, GLOBAL, cost, cfg.neuronCap);
  if (pool === null) {
    await release(db, day, route, scope);
    return { ok: false, reason: "global" };
  }
  return { ok: true, tier, route, model, reserved: cost };
}

export async function handleDiagnose(report, request, env, now) {
  const cfg = limits(env);
  checkModels(cfg);
  const db = env.DB;
  const day = utcDay(now);
  const scope = installScope(report.install_id);
  const ipScope = await hashIp(request.headers.get("cf-connecting-ip"), day, env.IP_HASH_SALT);

  if ((await reserve(db, day, ROUTE.diagnose, ipScope, 1, cfg.perIp)) === null) return refused("ip");

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userPrompt(report) },
  ];
  const inputTokens = estimateTokens(messages.map((m) => m.content).join("\n"));
  const used = await readCount(db, day, ROUTE.aiNeurons, GLOBAL);
  const pool = poolState(cfg, used);

  if (pool === "exhausted") {
    await release(db, day, ROUTE.diagnose, ipScope);
    return refused("global");
  }

  let grant = null;
  let lastReason = "global";
  if (pool === "best") {
    const best = await tryTier(db, day, cfg, scope, "best", inputTokens);
    if (best.ok) grant = best;
  }
  if (!grant) {
    const light = await tryTier(db, day, cfg, scope, "light", inputTokens);
    if (light.ok) grant = light;
    else lastReason = light.reason;
  }
  if (!grant) {
    await release(db, day, ROUTE.diagnose, ipScope);
    if (lastReason === "install") {
      const left = await aiAllowance(db, day, cfg, report.install_id);
      return refused(left.best_left > 0 ? "global" : "install");
    }
    return refused("global");
  }

  let result;
  try {
    result = await env.AI.run(grant.model, {
      messages,
      max_tokens: MAX_OUTPUT_TOKENS,
      temperature: 0.3,
    });
  } catch (err) {
    await release(db, day, grant.route, scope);
    if (isAllocationError(err)) {
      await raiseTo(db, day, ROUTE.aiNeurons, GLOBAL, cfg.neuronCap);
      return refused("global");
    }
    if (isRateError(err)) return refused("global");
    console.log("diagnose: model call failed");
    throw new HttpError(502, "diagnosis is unavailable; try again later");
  }

  let advice = extractAdvice(result)
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .trim();
  const usage = (result && result.usage) || {};
  const inTok = Number.isFinite(usage.prompt_tokens) ? usage.prompt_tokens : inputTokens;
  const outTok = Number.isFinite(usage.completion_tokens) ? usage.completion_tokens : estimateTokens(advice);
  const actual = neuronCost(grant.model, inTok, outTok);
  try {
    await adjust(db, day, ROUTE.aiNeurons, GLOBAL, actual - grant.reserved);
  } catch (_) {
    /* keep the conservative reservation */
  }

  if (!advice) {
    await release(db, day, grant.route, scope);
    throw new HttpError(502, "diagnosis returned no answer; try again later");
  }
  if (advice.length > MAX_ADVICE_CHARS) advice = `${advice.slice(0, MAX_ADVICE_CHARS)}…`;
  const allowance = await aiAllowance(db, day, cfg, report.install_id);
  return json(200, {
    advice,
    model: grant.model,
    tier: grant.tier,
    allowance: { ...allowance, resets_at: nextUtcMidnight(now) },
  });
}

export const SUMMARY_MAX_CHARS = 1200;
export const SUMMARY_MAX_TOKENS = 350;
export const SUMMARY_TIMEOUT_MS = 20000;

export const SUMMARY_PROMPT = [
  "You summarise anonymised spaCR error reports for the maintainers' GitHub issue.",
  "spaCR is an open-source Python desktop application for microscopy image analysis",
  "and pooled CRISPR screens. From the exception, message, module and traceback, write",
  "a short plain-text summary with two parts: the likely cause, and where in spaCR it",
  "happens (module, file and function from the traceback). Base it only on the report.",
  "Do not speculate about the user, their data, their lab or their intentions. Do not",
  "include greetings or advice to the user. Say when the cause is uncertain. At most",
  "150 words.",
].join("\n");

function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error("summary timeout")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

export async function summarizeWithRelay(env, report, now, timeoutMs = SUMMARY_TIMEOUT_MS) {
  try {
    if (!env.AI) return null;
    const cfg = limits(env);
    if (!MODEL_RATES[cfg.lightModel]) return null;
    const db = env.DB;
    const day = utcDay(now);
    const scope = installScope(report.install_id);
    const used = await readCount(db, day, ROUTE.aiNeurons, GLOBAL);
    if (poolState(cfg, used) === "exhausted") return null;
    const messages = [
      { role: "system", content: SUMMARY_PROMPT },
      { role: "user", content: userPrompt({ ...report, question: null }) },
    ];
    const inputTokens = estimateTokens(messages.map((m) => m.content).join("\n"));
    const slot = await reserve(db, day, ROUTE.aiLight, scope, 1, cfg.lightPerInstall);
    if (slot === null) return null;
    const reserved = neuronCost(cfg.lightModel, inputTokens, SUMMARY_MAX_TOKENS);
    if ((await reserve(db, day, ROUTE.aiNeurons, GLOBAL, reserved, cfg.neuronCap)) === null) {
      await release(db, day, ROUTE.aiLight, scope);
      return null;
    }
    let result;
    try {
      result = await withTimeout(
        env.AI.run(cfg.lightModel, { messages, max_tokens: SUMMARY_MAX_TOKENS, temperature: 0.2 }),
        timeoutMs,
      );
    } catch (err) {
      await release(db, day, ROUTE.aiLight, scope);
      if (isAllocationError(err)) await raiseTo(db, day, ROUTE.aiNeurons, GLOBAL, cfg.neuronCap);
      console.log("summary: model call failed");
      return null;
    }
    let text = extractAdvice(result)
      .replace(/<think>[\s\S]*?<\/think>/g, "")
      .trim();
    const usage = (result && result.usage) || {};
    const inTok = Number.isFinite(usage.prompt_tokens) ? usage.prompt_tokens : inputTokens;
    const outTok = Number.isFinite(usage.completion_tokens) ? usage.completion_tokens : estimateTokens(text);
    try {
      await adjust(db, day, ROUTE.aiNeurons, GLOBAL, neuronCost(cfg.lightModel, inTok, outTok) - reserved);
    } catch (_) {
      /* keep the conservative reservation */
    }
    if (!text) {
      await release(db, day, ROUTE.aiLight, scope);
      return null;
    }
    if (text.length > SUMMARY_MAX_CHARS) text = `${text.slice(0, SUMMARY_MAX_CHARS - 1)}…`;
    return { text, source: "relay", model: cfg.lightModel };
  } catch (_) {
    console.log("summary: skipped after an internal error");
    return null;
  }
}
