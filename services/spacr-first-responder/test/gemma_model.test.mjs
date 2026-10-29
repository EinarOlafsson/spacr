import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { DEFAULTS, MODEL_RATES, FREE_NEURONS_PER_DAY, limits } from "../src/config.js";
import { questionKey, store } from "../src/cache.js";
import { handle } from "../src/index.js";
import { MAX_OUTPUT_TOKENS, neuronCost } from "../src/neurons.js";
import { validatePayload } from "../src/validate.js";
import { FakeD1, makeRequest, sampleReport } from "./helpers.mjs";

const GEMMA = "@cf/google/gemma-4-26b-a4b-it";
const LIGHT = "@cf/meta/llama-3.1-8b-instruct-fp8";
const NOW = new Date("2026-10-29T12:00:00Z");
const report = () => validatePayload(sampleReport({ question: "How do I generate object masks?", frames: [], message: "" }), "diagnose");

function setup() {
  const calls = [];
  const env = { DB: new FakeD1(), AI: { async run(model, input) {
    calls.push({ model, input });
    return { choices: [{ message: { content: "<think>private</think>1. Open Mask Generation.", reasoning_content: "hidden reasoning" } }],
      usage: { prompt_tokens: 400, completion_tokens: 60 } };
  } } };
  return { env, calls };
}

async function ask(env) {
  const response = await handle(makeRequest("/diagnose", { body: report(), headers: { "cf-connecting-ip": "203.0.113.7" } }), env, null, NOW);
  return { status: response.status, body: await response.json() };
}

test("Gemma primary and official neuron rates agree with deployment while free limits remain capped", () => {
  assert.equal(DEFAULTS.AI_MODEL_BEST, GEMMA);
  assert.equal(DEFAULTS.AI_MODEL_LIGHT, LIGHT);
  assert.deepEqual(MODEL_RATES[GEMMA], { input: 9091, output: 27273 });
  assert.ok(readFileSync(new URL("../wrangler.toml", import.meta.url), "utf8").includes(`AI_MODEL_BEST = "${GEMMA}"`));
  assert.equal(limits({ AI_DAILY_NEURON_CAP: "999999" }).neuronCap, FREE_NEURONS_PER_DAY);
});

test("Gemma completion content is returned without reasoning under the reserved 600-token cap", async () => {
  const { env, calls } = setup();
  const result = await ask(env);
  assert.equal(result.status, 200);
  assert.equal(result.body.model, GEMMA);
  assert.equal(result.body.advice, "1. Open Mask Generation.");
  assert.equal(calls[0].input.max_completion_tokens, MAX_OUTPUT_TOKENS);
  assert.equal(calls[0].input.max_tokens, undefined);
  assert.deepEqual(calls[0].input.chat_template_kwargs, { enable_thinking: false });
  assert.equal(env.DB.counters.get("2026-10-29|ai_neurons|global"), neuronCost(GEMMA, 400, 60));
});

test("the existing low-pool Llama fallback retains its response and token parameter", async () => {
  const { env, calls } = setup();
  env.DB.counters.set("2026-10-29|ai_neurons|global", 9000 - 3599);
  env.AI.run = async (model, input) => { calls.push({ model, input }); return { response: "Use the compatible input.", usage: { prompt_tokens: 400, completion_tokens: 60 } }; };
  const result = await ask(env);
  assert.equal(result.status, 200);
  assert.equal(result.body.model, LIGHT);
  assert.equal(result.body.advice, "Use the compatible input.");
  assert.equal(calls[0].input.max_tokens, MAX_OUTPUT_TOKENS);
  assert.equal(calls[0].input.max_completion_tokens, undefined);
  assert.equal(calls[0].input.chat_template_kwargs, undefined);
});

test("fresh answers from a no-longer-configured Qwen model do not mask the Gemma switch", async () => {
  const { env, calls } = setup();
  const key = await questionKey(report());
  await store(env, key, "question", report(), NOW,
    { text: "Old Qwen advice", model: "@cf/qwen/qwen2.5-coder-32b-instruct", tier: "best" });
  const result = await ask(env);
  assert.equal(result.status, 200);
  assert.equal(result.body.cached, false);
  assert.equal(calls.length, 1);
  assert.equal(result.body.model, GEMMA);
  const again = await ask(env);
  assert.equal(again.body.cached, true);
  assert.equal(calls.length, 1);
});
