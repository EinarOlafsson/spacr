import assert from "node:assert/strict";
import { test } from "node:test";

import { MODEL_RATES, limits } from "../src/config.js";
import {
  MAX_OUTPUT_TOKENS,
  SYSTEM_PROMPT,
  neuronCost,
  poolState,
  worstCaseCost,
} from "../src/diagnose.js";
import { handle } from "../src/index.js";
import { FakeD1, makeRequest, sampleReport } from "./helpers.mjs";

const DAY1 = new Date("2026-10-14T12:00:00Z");
const DAY2 = new Date("2026-10-15T12:00:00Z");
const BEST = "@cf/qwen/qwen2.5-coder-32b-instruct";
const LIGHT = "@cf/meta/llama-3.1-8b-instruct-fp8";
const POOL = "2026-10-14|ai_neurons|global";
const USAGE = { prompt_tokens: 500, completion_tokens: 200 };

function makeEnv(overrides = {}) {
  const calls = [];
  const env = {
    DB: new FakeD1(),
    AI: {
      async run(model, input) {
        calls.push({ model, input });
        if (env.aiError) throw new Error(env.aiError);
        const out = { response: "1. Likely cause: mask and image sizes differ.\n2. Next step: check the resize setting." };
        if (!env.noUsage) out.usage = { ...USAGE, total_tokens: 700 };
        return out;
      },
    },
    AI_MODEL_BEST: BEST,
    AI_MODEL_LIGHT: LIGHT,
    AI_BEST_PER_INSTALL: "5",
    AI_LIGHT_PER_INSTALL: "10",
    AI_DAILY_NEURON_CAP: "9000",
    AI_LIGHT_THRESHOLD: "0.4",
    PER_IP_DAILY_LIMIT: "60",
    ...overrides,
  };
  return { env, calls };
}

async function post(env, body, { now = DAY1, ip = "203.0.113.7" } = {}) {
  const res = await handle(makeRequest("/diagnose", { body, headers: { "cf-connecting-ip": ip } }), env, null, now);
  return { status: res.status, body: await res.json() };
}

async function allowance(env, id = sampleReport().install_id, now = DAY1) {
  const res = await handle(makeRequest(`/allowance?install_id=${id}`, { method: "GET" }), env, null, now);
  return res.json();
}

function installN(i) {
  return `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`;
}

test("model rates cover both defaults and worst cases fit the free allocation", () => {
  const cfg = limits({});
  assert.equal(cfg.bestModel, BEST);
  assert.equal(cfg.lightModel, LIGHT);
  assert.ok(MODEL_RATES[BEST] && MODEL_RATES[LIGHT]);
  assert.ok(worstCaseCost(LIGHT) < worstCaseCost(BEST));
  assert.ok(worstCaseCost(BEST) < 9000 * 0.4);
  assert.equal(limits({ AI_DAILY_NEURON_CAP: "50000" }).neuronCap, 10000);
});

test("first question uses the best model with a bounded, fixed prompt", async () => {
  const { env, calls } = makeEnv();
  const r = await post(env, sampleReport({ question: "How do I fix this?" }));
  assert.equal(r.status, 200);
  assert.equal(r.body.model, BEST);
  assert.equal(r.body.tier, "best");
  assert.deepEqual(r.body.allowance, { best_left: 4, light_left: 10, resets_at: "2026-10-15T00:00:00.000Z" });
  assert.match(r.body.advice, /Likely cause/);
  const { input } = calls[0];
  assert.equal(input.max_tokens, MAX_OUTPUT_TOKENS);
  assert.equal(input.messages[0].content, SYSTEM_PROMPT);
  assert.match(SYSTEM_PROMPT, /Never claim certainty/);
  assert.match(input.messages[1].content, /User question: How do I fix this\?/);
});

test("neurons are settled from the usage tokens the model returns", async () => {
  const { env } = makeEnv();
  await post(env, sampleReport());
  assert.equal(env.DB.counters.get(POOL), neuronCost(BEST, 500, 200));
  await post(env, sampleReport({ install_id: installN(1) }));
  assert.equal(env.DB.counters.get(POOL), 2 * neuronCost(BEST, 500, 200));
});

test("without usage the cost is estimated and stays below the reservation", async () => {
  const { env } = makeEnv({ noUsage: true });
  await post(env, sampleReport());
  const charged = env.DB.counters.get(POOL);
  assert.ok(charged > 0);
  assert.ok(charged < worstCaseCost(BEST));
});

test("a user moves from best to light after five, and is refused after fifteen", async () => {
  const { env, calls } = makeEnv();
  const tiers = [];
  for (let i = 0; i < 15; i++) {
    const r = await post(env, sampleReport());
    assert.equal(r.status, 200, `call ${i}`);
    tiers.push(r.body.tier);
    if (i === 4) assert.deepEqual([r.body.allowance.best_left, r.body.allowance.light_left], [0, 10]);
    if (i === 5) assert.equal(r.body.model, LIGHT);
  }
  assert.deepEqual(tiers, [...Array(5).fill("best"), ...Array(10).fill("light")]);
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "install" });
  assert.equal(calls.length, 15);
  const other = await post(env, sampleReport({ install_id: installN(2) }));
  assert.equal(other.body.tier, "best");
  const tomorrow = await post(env, sampleReport(), { now: DAY2 });
  assert.equal(tomorrow.body.tier, "best");
});

test("a low pool serves everyone light without spending their best allowance", async () => {
  const { env, calls } = makeEnv();
  env.DB.counters.set(POOL, 9000 - 3599);
  assert.equal(poolState(limits(env), 9000 - 3599), "light");
  const r = await post(env, sampleReport());
  assert.equal(r.status, 200);
  assert.equal(r.body.tier, "light");
  assert.equal(calls[0].model, LIGHT);
  assert.deepEqual([r.body.allowance.best_left, r.body.allowance.light_left], [5, 9]);
  assert.equal((await allowance(env)).pool, "light");
});

test("a high pool still serves best", async () => {
  const { env } = makeEnv();
  env.DB.counters.set(POOL, 9000 - 3700);
  assert.equal((await post(env, sampleReport())).body.tier, "best");
});

test("light exhausted while the pool is low refuses with scope global", async () => {
  const { env } = makeEnv({ AI_LIGHT_PER_INSTALL: "1" });
  env.DB.counters.set(POOL, 6000);
  assert.equal((await post(env, sampleReport())).body.tier, "light");
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "global" });
});

test("too little budget for a worst-case light call refuses everyone and refunds", async () => {
  const { env, calls } = makeEnv();
  env.DB.counters.set(POOL, 9000 - worstCaseCost(LIGHT) + 1);
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "global" });
  assert.equal(calls.length, 0);
  for (const [key, n] of env.DB.counters) {
    if (key !== POOL) assert.equal(n, 0, key);
  }
  const a = await allowance(env);
  assert.equal(a.pool, "exhausted");
  assert.deepEqual(a.ai, { best_left: 5, light_left: 10 });
});

test("the neuron cap is never exceeded under a burst", async () => {
  const { env } = makeEnv({ AI_DAILY_NEURON_CAP: "1000", PER_IP_DAILY_LIMIT: "1000" });
  let served = 0;
  for (let i = 0; i < 200; i++) {
    const r = await post(env, sampleReport({ install_id: installN(i) }));
    if (r.status === 200) served++;
    assert.ok((env.DB.counters.get(POOL) ?? 0) <= 1000);
  }
  assert.ok(served > 0);
  assert.equal((await post(env, sampleReport({ install_id: installN(999) }))).status, 429);
});

test("per-IP guard refuses with scope ip and keeps the user's allowance", async () => {
  const { env } = makeEnv({ PER_IP_DAILY_LIMIT: "2" });
  await post(env, sampleReport({ install_id: installN(1) }));
  await post(env, sampleReport({ install_id: installN(2) }));
  const r = await post(env, sampleReport({ install_id: installN(3) }));
  assert.deepEqual(r.body, { refused: "quota", scope: "ip" });
  assert.deepEqual((await allowance(env, installN(3))).ai, { best_left: 5, light_left: 10 });
});

test("Workers AI 4006 maps to 429 global, closes the pool and refunds the question", async () => {
  const { env } = makeEnv();
  env.aiError = "AiError: 4006: you have used up your daily free allocation of 10,000 neurons";
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "global" });
  assert.equal(env.DB.counters.get(POOL), 9000);
  const a = await allowance(env);
  assert.deepEqual(a.ai, { best_left: 5, light_left: 10 });
  assert.equal(a.pool, "exhausted");
});

test("other model errors are 502 JSON, refund the question and keep the reservation", async () => {
  const { env } = makeEnv();
  env.aiError = "InferenceUpstreamError: internal";
  const r = await post(env, sampleReport());
  assert.equal(r.status, 502);
  assert.ok(!JSON.stringify(r.body).includes("InferenceUpstream"));
  assert.deepEqual((await allowance(env)).ai, { best_left: 5, light_left: 10 });
  assert.ok(env.DB.counters.get(POOL) > 0);
});

test("diagnose validates like report", async () => {
  const { env, calls } = makeEnv();
  const r = await post(env, sampleReport({ question: "my key is ghp_abcdefghijklmnopqrstuvwxyz0123456789" }));
  assert.equal(r.status, 400);
  assert.match(r.body.error, /question.*token/);
  assert.equal(calls.length, 0);
});

test("a model outside the free allowlist is refused", async () => {
  const { env, calls } = makeEnv({ AI_MODEL_BEST: "@cf/moonshotai/kimi-k2.6" });
  const r = await post(env, sampleReport());
  assert.equal(r.status, 503);
  assert.equal(calls.length, 0);
});

test("allowance reflects use without consuming", async () => {
  const { env } = makeEnv();
  assert.deepEqual(await allowance(env), {
    ai: { best_left: 5, light_left: 10 },
    reports_left: 5,
    pool: "best",
    resets_at: "2026-10-15T00:00:00.000Z",
  });
  await post(env, sampleReport());
  const snapshot = new Map(env.DB.counters);
  const a = await allowance(env);
  assert.deepEqual(a.ai, { best_left: 4, light_left: 10 });
  assert.deepEqual(env.DB.counters, snapshot);
});
