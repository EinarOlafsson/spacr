export const FREE_NEURONS_PER_DAY = 10000;

export const MODEL_RATES = Object.freeze({
  "@cf/google/gemma-4-26b-a4b-it": { input: 9091, output: 27273 },
  "@cf/qwen/qwen2.5-coder-32b-instruct": { input: 60000, output: 90909 },
  "@cf/meta/llama-3.3-70b-instruct-fp8-fast": { input: 26668, output: 204805 },
  "@cf/meta/llama-3.1-8b-instruct-fp8": { input: 13778, output: 26128 },
  "@cf/meta/llama-3.2-3b-instruct": { input: 4625, output: 30475 },
});

export const DEFAULTS = Object.freeze({
  AI_MODEL_BEST: "@cf/google/gemma-4-26b-a4b-it",
  AI_MODEL_LIGHT: "@cf/meta/llama-3.1-8b-instruct-fp8",
  AI_BEST_PER_INSTALL: 0,
  AI_LIGHT_PER_INSTALL: 0,
  AI_MIN_SHARE_NEURONS: 300,
  AI_DAILY_NEURON_CAP: 9000,
  AI_LIGHT_THRESHOLD: 0.4,
  REPORTS_PER_INSTALL: 5,
  DAILY_REPORT_LIMIT: 200,
  PER_IP_DAILY_LIMIT: 60,
});

function intOf(env, name) {
  const n = Number.parseInt(env && env[name], 10);
  return Number.isFinite(n) && n >= 0 ? n : DEFAULTS[name];
}

function fractionOf(env, name) {
  const n = Number.parseFloat(env && env[name]);
  return Number.isFinite(n) && n >= 0 && n <= 1 ? n : DEFAULTS[name];
}

export function limits(env) {
  return {
    bestModel: (env && env.AI_MODEL_BEST) || DEFAULTS.AI_MODEL_BEST,
    lightModel: (env && env.AI_MODEL_LIGHT) || DEFAULTS.AI_MODEL_LIGHT,
    bestPerInstall: intOf(env, "AI_BEST_PER_INSTALL"),
    lightPerInstall: intOf(env, "AI_LIGHT_PER_INSTALL"),
    minShare: intOf(env, "AI_MIN_SHARE_NEURONS"),
    neuronCap: Math.min(intOf(env, "AI_DAILY_NEURON_CAP"), FREE_NEURONS_PER_DAY),
    lightThreshold: fractionOf(env, "AI_LIGHT_THRESHOLD"),
    reportsPerInstall: intOf(env, "REPORTS_PER_INSTALL"),
    dailyReports: intOf(env, "DAILY_REPORT_LIMIT"),
    perIp: intOf(env, "PER_IP_DAILY_LIMIT"),
  };
}

export function nextUtcMidnight(now) {
  const d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  return d.toISOString();
}
