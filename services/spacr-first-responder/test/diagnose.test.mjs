import assert from "node:assert/strict";
import { test } from "node:test";

import { ACTIVE_USERS_DEFINITION, shareFor } from "../src/allowance.js";
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
const ME = sampleReport().install_id;

function mineKey(id = ME, day = "2026-10-14") {
  return `${day}|ai_install_neurons|install:${id}`;
}

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
    AI_DAILY_NEURON_CAP: "9000",
    AI_MIN_SHARE_NEURONS: "300",
    AI_LIGHT_THRESHOLD: "0.4",
    PER_IP_DAILY_LIMIT: "1000",
    ...overrides,
  };
  return { env, calls };
}

let distinct = 0;

async function post(env, body, { now = DAY1, ip = "203.0.113.7", same = false } = {}) {
  if (!same && body && typeof body.exception_type === "string") {
    distinct += 1;
    body = { ...body, exception_type: `${body.exception_type}${distinct}` };
  }
  const res = await handle(makeRequest("/diagnose", { body, headers: { "cf-connecting-ip": ip } }), env, null, now);
  return { status: res.status, body: await res.json() };
}

async function allowance(env, id = ME, now = DAY1) {
  const res = await handle(makeRequest(`/allowance?install_id=${id}`, { method: "GET" }), env, null, now);
  return res.json();
}

function installN(i) {
  return `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`;
}

function activate(env, count, used = 10) {
  for (let i = 0; i < count; i++) env.DB.counters.set(mineKey(installN(1000 + i)), used);
}

test("share is cap / (N + 1) with a floor and the pool cap as ceiling", () => {
  const cfg = limits({});
  assert.equal(cfg.neuronCap, 9000);
  assert.equal(cfg.minShare, 300);
  assert.equal(shareFor(cfg, 1), 4500);
  assert.equal(shareFor(cfg, 2), 3000);
  assert.equal(shareFor(cfg, 3), 2250);
  assert.equal(shareFor(cfg, 10), 818);
  assert.equal(shareFor(cfg, 40), 300, "floor");
  assert.equal(shareFor(limits({ AI_MIN_SHARE_NEURONS: "20000" }), 5), 9000, "never above the pool cap");
  assert.equal(limits({ AI_DAILY_NEURON_CAP: "50000" }).neuronCap, 10000);
  assert.equal(cfg.bestPerInstall, 0);
  assert.equal(cfg.lightPerInstall, 0);
});

test("model rates cover both defaults and worst cases fit the free allocation", () => {
  assert.ok(MODEL_RATES[BEST] && MODEL_RATES[LIGHT]);
  assert.ok(worstCaseCost(LIGHT) < worstCaseCost(BEST));
  assert.ok(worstCaseCost(BEST) < 9000 * 0.4);
});

test("one user: best model, 50% share, full allowance object", async () => {
  const { env, calls } = makeEnv();
  const r = await post(env, sampleReport({ question: "How do I fix this?" }));
  assert.equal(r.status, 200);
  assert.equal(r.body.model, BEST);
  assert.equal(r.body.tier, "best");
  const spent = neuronCost(BEST, 500, 200);
  const a = r.body.allowance;
  assert.deepEqual(a.ai, {
    neurons_left: 4500 - spent,
    share_neurons: 4500,
    share_pct: 50,
    active_users: 1,
    active_users_definition: ACTIVE_USERS_DEFINITION,
    best_left: Math.floor((4500 - spent) / neuronCost(BEST, 1000, 400)),
    light_left: Math.floor((4500 - spent) / neuronCost(LIGHT, 1000, 400)),
    tier_now: "best",
  });
  assert.equal(a.best_left, a.ai.best_left, "old top-level field kept");
  assert.equal(a.light_left, a.ai.light_left, "old top-level field kept");
  assert.equal(a.pool, "best");
  assert.equal(a.pool_pct_left, Math.round(((9000 - spent) / 9000) * 1000) / 10);
  assert.equal(a.reports_left, 5);
  assert.equal(a.resets_at, "2026-10-15T00:00:00.000Z");
  assert.equal(env.DB.counters.get(mineKey()), spent);
  assert.equal(env.DB.counters.get(POOL), spent);
  const { input } = calls[0];
  assert.equal(input.max_tokens, MAX_OUTPUT_TOKENS);
  assert.equal(input.messages[0].content, SYSTEM_PROMPT);
  assert.match(input.messages[1].content, /User question: How do I fix this\?/);
});

test("shares for 2, 3 and 10 active users", async () => {
  for (const [others, share, pctValue] of [
    [1, 3000, 33.3],
    [2, 2250, 25],
    [9, 818, 9.1],
  ]) {
    const { env } = makeEnv();
    activate(env, others);
    const before = await allowance(env);
    assert.equal(before.ai.active_users, others + 1);
    assert.equal(before.ai.share_neurons, share);
    assert.equal(before.ai.share_pct, pctValue);
    const r = await post(env, sampleReport());
    assert.equal(r.status, 200);
    assert.equal(r.body.allowance.ai.share_neurons, share);
    assert.equal(r.body.allowance.ai.active_users, others + 1);
  }
});

test("real users joining shrink the share step by step", async () => {
  const { env } = makeEnv();
  const shares = [];
  for (let i = 0; i < 10; i++) {
    const r = await post(env, sampleReport({ install_id: installN(i) }));
    assert.equal(r.status, 200);
    shares.push(r.body.allowance.ai.share_neurons);
  }
  assert.deepEqual(shares, [4500, 3000, 2250, 1800, 1500, 1285, 1125, 1000, 900, 818]);
});

test("a share that shrinks below what was used locks that user out until reset", async () => {
  const { env, calls } = makeEnv();
  env.DB.counters.set(mineKey(), 2000);
  assert.equal((await allowance(env)).ai.neurons_left, 2500);
  activate(env, 3);
  const a = await allowance(env);
  assert.equal(a.ai.share_neurons, 1800);
  assert.equal(a.ai.neurons_left, 0);
  assert.equal(a.ai.tier_now, "none");
  assert.equal(a.ai.best_left, 0);
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "install" });
  assert.equal(calls.length, 0);
  assert.equal(env.DB.counters.get(mineKey()), 2000);
  const tomorrow = await post(env, sampleReport(), { now: DAY2 });
  assert.equal(tomorrow.status, 200);
  assert.equal(tomorrow.body.allowance.ai.share_neurons, 4500);
});

test("the floor keeps a minimum share when many users are active", async () => {
  const { env } = makeEnv();
  activate(env, 49);
  const a = await allowance(env);
  assert.equal(a.ai.active_users, 50);
  assert.equal(a.ai.share_neurons, 300);
  const r = await post(env, sampleReport());
  assert.equal(r.status, 200);
  assert.ok(r.body.allowance.ai.neurons_left < 300);
});

test("a nearly used share falls from best to light, then is refused", async () => {
  const { env, calls } = makeEnv();
  env.DB.counters.set(mineKey(), 4500 - 60);
  const first = await post(env, sampleReport());
  assert.equal(first.status, 200);
  assert.equal(first.body.tier, "light");
  assert.equal(calls[0].model, LIGHT);
  env.DB.counters.set(mineKey(), 4500 - 5);
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "install" });
});

test("a low pool serves everyone light even with share left", async () => {
  const { env, calls } = makeEnv();
  env.DB.counters.set(POOL, 9000 - 3599);
  assert.equal(poolState(limits(env), 9000 - 3599), "light");
  const r = await post(env, sampleReport());
  assert.equal(r.status, 200);
  assert.equal(r.body.tier, "light");
  assert.equal(calls[0].model, LIGHT);
  const a = r.body.allowance;
  assert.equal(a.pool, "light");
  assert.equal(a.ai.tier_now, "light");
  assert.equal(a.ai.best_left, 0);
  assert.ok(a.ai.light_left > 0);
  assert.equal(a.pool_pct_left, Math.round(((3599 - neuronCost(LIGHT, 500, 200)) / 9000) * 1000) / 10);
});

test("a high pool still serves best", async () => {
  const { env } = makeEnv();
  env.DB.counters.set(POOL, 9000 - 3700);
  assert.equal((await post(env, sampleReport())).body.tier, "best");
});

test("pool exhaustion refuses everyone with scope global and refunds", async () => {
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
  assert.equal(a.ai.tier_now, "none");
  assert.equal(a.ai.neurons_left, 4500);
});

test("the neuron cap is never exceeded under a burst of new users", async () => {
  const { env } = makeEnv({ AI_DAILY_NEURON_CAP: "1000" });
  let served = 0;
  for (let i = 0; i < 200; i++) {
    const r = await post(env, sampleReport({ install_id: installN(i) }));
    if (r.status === 200) served++;
    assert.ok((env.DB.counters.get(POOL) ?? 0) <= 1000);
  }
  assert.ok(served > 0);
  assert.equal((await post(env, sampleReport({ install_id: installN(999) }))).status, 429);
});

test("optional per-install caps still limit best and light counts", async () => {
  const { env } = makeEnv({ AI_BEST_PER_INSTALL: "1", AI_LIGHT_PER_INSTALL: "1" });
  assert.equal((await post(env, sampleReport())).body.tier, "best");
  const second = await post(env, sampleReport());
  assert.equal(second.body.tier, "light");
  assert.equal(second.body.allowance.ai.best_left, 0);
  assert.equal(second.body.allowance.ai.light_left, 0);
  const third = await post(env, sampleReport());
  assert.deepEqual(third.body, { refused: "quota", scope: "install" });
});

test("per-IP guard refuses with scope ip and consumes no share", async () => {
  const { env } = makeEnv({ PER_IP_DAILY_LIMIT: "2" });
  await post(env, sampleReport({ install_id: installN(1) }));
  await post(env, sampleReport({ install_id: installN(2) }));
  const r = await post(env, sampleReport({ install_id: installN(3) }));
  assert.deepEqual(r.body, { refused: "quota", scope: "ip" });
  assert.equal(env.DB.counters.has(mineKey(installN(3))), false);
});

test("allowance is non-consuming and does not count the caller as active", async () => {
  const { env } = makeEnv();
  activate(env, 1);
  const snapshot = new Map(env.DB.counters);
  for (let i = 0; i < 5; i++) {
    const a = await allowance(env, installN(i));
    assert.equal(a.ai.active_users, 2);
  }
  assert.deepEqual(env.DB.counters, snapshot);
  const a = await allowance(env);
  assert.equal(a.ai.active_users_definition, ACTIVE_USERS_DEFINITION);
  assert.equal(a.reports_left, 5);
  assert.equal(a.pool, "best");
  assert.equal(a.ai.tier_now, "best");
  const bad = await handle(makeRequest("/allowance?install_id=x", { method: "GET" }), env, null, DAY1);
  assert.equal(bad.status, 400);
});

test("without usage the cost is estimated and stays below the reservation", async () => {
  const { env } = makeEnv({ noUsage: true });
  await post(env, sampleReport());
  const charged = env.DB.counters.get(POOL);
  assert.ok(charged > 0);
  assert.ok(charged < worstCaseCost(BEST));
  assert.equal(env.DB.counters.get(mineKey()), charged);
});

test("Workers AI 4006 maps to 429 global, closes the pool and refunds the share", async () => {
  const { env } = makeEnv();
  env.aiError = "AiError: 4006: you have used up your daily free allocation of 10,000 neurons";
  const r = await post(env, sampleReport());
  assert.deepEqual(r.body, { refused: "quota", scope: "global" });
  assert.equal(env.DB.counters.get(POOL), 9000);
  assert.equal(env.DB.counters.get(mineKey()), 0);
  assert.equal((await allowance(env)).pool, "exhausted");
});

test("other model errors are 502, refund the share and keep the pool reservation", async () => {
  const { env } = makeEnv();
  env.aiError = "InferenceUpstreamError: internal";
  const r = await post(env, sampleReport());
  assert.equal(r.status, 502);
  assert.ok(!JSON.stringify(r.body).includes("InferenceUpstream"));
  assert.equal(env.DB.counters.get(mineKey()), 0);
  assert.ok(env.DB.counters.get(POOL) > 0);
});

test("diagnose validates like report, including frames[].code", async () => {
  const { env, calls } = makeEnv();
  const r = await post(env, sampleReport({ question: "my key is ghp_abcdefghijklmnopqrstuvwxyz0123456789" }));
  assert.equal(r.status, 400);
  assert.match(r.body.error, /question.*token/);
  const withCode = sampleReport({ kind: "user_error" });
  withCode.frames[2].code = "crop = image[y0:y1, x0:x1]";
  const ok = await post(env, withCode);
  assert.equal(ok.status, 200);
  assert.match(calls[0].input.messages[1].content, /measure\.py:412 in measure_crop\n {6}crop = image\[y0:y1, x0:x1\]/);
  assert.match(calls[0].input.messages[1].content, /Kind: reported as a settings or usage error/);
});

test("a model outside the free allowlist is refused", async () => {
  const { env, calls } = makeEnv({ AI_MODEL_BEST: "@cf/moonshotai/kimi-k2.6" });
  const r = await post(env, sampleReport());
  assert.equal(r.status, 503);
  assert.equal(calls.length, 0);
});
