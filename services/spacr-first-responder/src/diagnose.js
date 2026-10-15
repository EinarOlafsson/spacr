import { aiState, buildAllowance, grantAi, refundUser, settle } from "./allowance.js";
import { MODEL_RATES, limits } from "./config.js";
import { MAX_OUTPUT_TOKENS, estimateTokens, neuronCost } from "./neurons.js";
import { splitCause } from "./causes.js";
import { SUMMARY_PREFIX, saveExplanation, savedExplanation } from "./explanations.js";
import { SUMMARY_PROMPT, SYSTEM_PROMPT, isExplanationRequest, userPrompt } from "./prompts.js";
import { signatureOf } from "./report.js";
import { GLOBAL, ROUTE, hashIp, raiseTo, release, reserve } from "./quota.js";
import { HttpError, json, utcDay } from "./util.js";

export { MAX_OUTPUT_TOKENS, MAX_PROMPT_TOKENS, neuronCost, poolState, worstCaseCost } from "./neurons.js";
export { EXPLAIN_QUESTION, SUMMARY_PROMPT, SYSTEM_PROMPT, userPrompt } from "./prompts.js";

export const MAX_ADVICE_CHARS = 6000;
export const SUMMARY_MAX_CHARS = 1200;
export const SUMMARY_MAX_TOKENS = 350;
export const SUMMARY_TIMEOUT_MS = 20000;

export function isAllocationError(err) {
  return /\b4006\b|neuron|daily free allocation/i.test(String((err && (err.message || err)) || ""));
}

export function isRateError(err) {
  return /rate limit|too many requests|capacity|\b3040\b/i.test(String((err && (err.message || err)) || ""));
}

function modelsAllowed(cfg) {
  return Boolean(MODEL_RATES[cfg.bestModel] && MODEL_RATES[cfg.lightModel]);
}

function extractText(result) {
  if (!result) return "";
  let text = "";
  if (typeof result.response === "string") text = result.response;
  else {
    const choice = Array.isArray(result.choices) ? result.choices[0] : null;
    if (choice && choice.message && typeof choice.message.content === "string") text = choice.message.content;
  }
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function actualCost(model, result, inputTokens, text) {
  const usage = (result && result.usage) || {};
  const inTok = Number.isFinite(usage.prompt_tokens) ? usage.prompt_tokens : inputTokens;
  const outTok = Number.isFinite(usage.completion_tokens) ? usage.completion_tokens : estimateTokens(text);
  return neuronCost(model, inTok, outTok);
}

function refused(scope) {
  return json(429, { refused: "quota", scope });
}

function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error("model timeout")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

export async function handleDiagnose(report, request, env, now) {
  const cfg = limits(env);
  if (!modelsAllowed(cfg)) {
    console.log("diagnose: a configured model is not on the free allowlist");
    throw new HttpError(503, "diagnosis is not available");
  }
  const db = env.DB;
  const day = utcDay(now);
  const ipScope = await hashIp(request.headers.get("cf-connecting-ip"), day, env.IP_HASH_SALT);
  if ((await reserve(db, day, ROUTE.diagnose, ipScope, 1, cfg.perIp)) === null) return refused("ip");

  const cacheKey = isExplanationRequest(report) ? await signatureOf(report) : null;
  if (cacheKey) {
    const saved = await savedExplanation(db, cacheKey, report, now);
    if (saved) {
      return json(200, {
        advice: saved.text,
        model: saved.model,
        tier: saved.tier,
        cause: saved.cause,
        cached: true,
        allowance: await buildAllowance(db, day, cfg, report.install_id, now),
      });
    }
  }

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userPrompt(report) },
  ];
  const inputTokens = estimateTokens(messages.map((m) => m.content).join("\n"));
  const st = await aiState(db, day, cfg, report.install_id);
  const grant = await grantAi(db, day, cfg, st, inputTokens, MAX_OUTPUT_TOKENS);
  if (!grant.ok) {
    await release(db, day, ROUTE.diagnose, ipScope);
    return refused(grant.scope);
  }

  let result;
  try {
    result = await env.AI.run(grant.model, { messages, max_tokens: MAX_OUTPUT_TOKENS, temperature: 0.3 });
  } catch (err) {
    await refundUser(db, day, grant);
    if (isAllocationError(err)) {
      await raiseTo(db, day, ROUTE.aiNeurons, GLOBAL, cfg.neuronCap);
      return refused("global");
    }
    if (isRateError(err)) return refused("global");
    console.log("diagnose: model call failed");
    throw new HttpError(502, "diagnosis is unavailable; try again later");
  }

  const raw = extractText(result);
  await settle(db, day, grant, actualCost(grant.model, result, inputTokens, raw));
  const parsed = splitCause(raw);
  let advice = parsed.text;
  const cause = isExplanationRequest(report) ? parsed.cause : null;
  if (!advice) {
    await refundUser(db, day, grant);
    throw new HttpError(502, "diagnosis returned no answer; try again later");
  }
  if (advice.length > MAX_ADVICE_CHARS) advice = `${advice.slice(0, MAX_ADVICE_CHARS)}…`;
  if (cacheKey) await saveExplanation(db, cacheKey, report, now, { text: advice, model: grant.model, tier: grant.tier, cause });
  return json(200, {
    advice,
    model: grant.model,
    tier: grant.tier,
    cause,
    cached: false,
    allowance: await buildAllowance(db, day, cfg, report.install_id, now),
  });
}

export async function summarizeWithRelay(env, report, now, timeoutMs = SUMMARY_TIMEOUT_MS) {
  try {
    if (!env.AI) return null;
    const cfg = limits(env);
    if (!modelsAllowed(cfg)) return null;
    const db = env.DB;
    const day = utcDay(now);
    const cacheKey = SUMMARY_PREFIX + (await signatureOf(report));
    const saved = await savedExplanation(db, cacheKey, report, now);
    if (saved) return { text: saved.text, source: "relay", model: saved.model, cached: true, cause: saved.cause };
    const messages = [
      { role: "system", content: SUMMARY_PROMPT },
      { role: "user", content: userPrompt({ ...report, question: null }) },
    ];
    const inputTokens = estimateTokens(messages.map((m) => m.content).join("\n"));
    const st = await aiState(db, day, cfg, report.install_id);
    const grant = await grantAi(db, day, cfg, st, inputTokens, SUMMARY_MAX_TOKENS, { allowBest: false });
    if (!grant.ok) return null;
    let result;
    try {
      result = await withTimeout(
        env.AI.run(grant.model, { messages, max_tokens: SUMMARY_MAX_TOKENS, temperature: 0.2 }),
        timeoutMs,
      );
    } catch (err) {
      await refundUser(db, day, grant);
      if (isAllocationError(err)) await raiseTo(db, day, ROUTE.aiNeurons, GLOBAL, cfg.neuronCap);
      console.log("summary: model call failed");
      return null;
    }
    const raw = extractText(result);
    await settle(db, day, grant, actualCost(grant.model, result, inputTokens, raw));
    const parsed = splitCause(raw);
    let text = parsed.text;
    if (!text) {
      await refundUser(db, day, grant);
      return null;
    }
    if (text.length > SUMMARY_MAX_CHARS) text = `${text.slice(0, SUMMARY_MAX_CHARS - 1)}…`;
    await saveExplanation(db, cacheKey, report, now, { text, model: grant.model, tier: grant.tier, cause: parsed.cause });
    return { text, source: "relay", model: grant.model, cached: false, cause: parsed.cause };
  } catch (_) {
    console.log("summary: skipped after an internal error");
    return null;
  }
}
