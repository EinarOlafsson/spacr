import { MODEL_RATES } from "./config.js";
import { SYSTEM_PROMPT } from "./prompts.js";
import { MAX_BODY_BYTES } from "./util.js";

export const MAX_OUTPUT_TOKENS = 1200;
export const CHARS_PER_TOKEN_FLOOR = 2.5;
export const TYPICAL_PROMPT_TOKENS = 1500;
export const TYPICAL_INPUT_TOKENS = 1000;
export const TYPICAL_OUTPUT_TOKENS = 400;

export const MAX_PROMPT_TOKENS = Math.ceil((MAX_BODY_BYTES + SYSTEM_PROMPT.length + 400) / CHARS_PER_TOKEN_FLOOR);

export function neuronCost(model, inputTokens, outputTokens) {
  const r = MODEL_RATES[model];
  return Math.ceil((inputTokens * r.input + outputTokens * r.output) / 1e6);
}

export function estimateTokens(text) {
  return Math.ceil(text.length / CHARS_PER_TOKEN_FLOOR) + 8;
}

export function worstCaseCost(model, inputTokens = MAX_PROMPT_TOKENS, outputTokens = MAX_OUTPUT_TOKENS) {
  return neuronCost(model, inputTokens, outputTokens);
}

export function typicalCost(model) {
  return neuronCost(model, TYPICAL_INPUT_TOKENS, TYPICAL_OUTPUT_TOKENS);
}

export function poolState(cfg, used) {
  const remaining = cfg.neuronCap - used;
  if (remaining >= cfg.lightThreshold * cfg.neuronCap && remaining >= worstCaseCost(cfg.bestModel)) {
    return "best";
  }
  if (remaining >= worstCaseCost(cfg.lightModel)) return "light";
  return "exhausted";
}
