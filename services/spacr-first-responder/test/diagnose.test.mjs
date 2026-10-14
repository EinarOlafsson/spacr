import assert from "node:assert/strict";
import { test } from "node:test";

import { MAX_OUTPUT_TOKENS, SYSTEM_PROMPT } from "../src/diagnose.js";
import { handle } from "../src/index.js";
import { FakeD1, makeRequest, sampleReport } from "./helpers.mjs";

const DAY1 = new Date("2026-10-14T12:00:00Z");
const DAY2 = new Date("2026-10-15T12:00:00Z");

function makeEnv(overrides = {}) {
  const calls = [];
  const env = {
    DB: new FakeD1(),
    AI: {
      async run(model, input) {
        calls.push({ model, input });
        if (env.aiError) throw new Error(env.aiError);
        return { response: "1. Likely cause: mask and image sizes differ.\n2. Next step: check the resize setting." };
      },
    },
    AI_MODEL: "@cf/qwen/qwen2.5-coder-32b-instruct",
    DAILY_AI_LIMIT: "3",
    PER_INSTALL_DAILY_LIMIT: "10",
    PER_IP_DAILY_LIMIT: "20",
    ...overrides,
  };
  return { env, calls };
}

async function post(env, body, { now = DAY1, ip = "203.0.113.7" } = {}) {
  const res = await handle(makeRequest("/diagnose", { body, headers: { "cf-connecting-ip": ip } }), env, null, now);
  return { status: res.status, body: await res.json() };
}

function installN(i) {
  return `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`;
}

test("diagnose returns advice and model with a bounded, fixed prompt", async () => {
  const { env, calls } = makeEnv();
  const r = await post(env, sampleReport({ question: "How do I fix this?" }));
  assert.equal(r.status, 200);
  assert.equal(r.body.model, "@cf/qwen/qwen2.5-coder-32b-instruct");
  assert.match(r.body.advice, /Likely cause/);
  assert.equal(calls.length, 1);
  const { input } = calls[0];
  assert.equal(input.max_tokens, MAX_OUTPUT_TOKENS);
  assert.equal(input.messages[0].role, "system");
  assert.equal(input.messages[0].content, SYSTEM_PROMPT);
  assert.match(SYSTEM_PROMPT, /spaCR/);
  assert.match(SYSTEM_PROMPT, /Never claim certainty/);
  assert.match(input.messages[1].content, /User question: How do I fix this\?/);
  assert.match(input.messages[1].content, /measure\.py:412 in measure_crop/);
});

test("diagnose validates like report", async () => {
  const { env, calls } = makeEnv();
  const r = await post(env, sampleReport({ question: "my key is ghp_abcdefghijklmnopqrstuvwxyz0123456789" }));
  assert.equal(r.status, 400);
  assert.match(r.body.error, /question.*token/);
  assert.equal(calls.length, 0);
});

test("global AI budget is enforced and resets the next UTC day", async () => {
  const { env, calls } = makeEnv({ DAILY_AI_LIMIT: "3" });
  for (let i = 0; i < 3; i++) {
    const r = await post(env, sampleReport({ install_id: installN(i) }), { ip: `198.51.100.${i + 1}` });
    assert.equal(r.status, 200);
  }
  const r = await post(env, sampleReport({ install_id: installN(9) }), { ip: "198.51.100.99" });
  assert.equal(r.status, 429);
  assert.equal(r.body.refused, "quota");
  assert.equal(calls.length, 3);
  const tomorrow = await post(env, sampleReport({ install_id: installN(9) }), { now: DAY2, ip: "198.51.100.99" });
  assert.equal(tomorrow.status, 200);
});

test("budget of zero disables diagnosis without calling the model", async () => {
  const { env, calls } = makeEnv({ DAILY_AI_LIMIT: "0" });
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.equal(calls.length, 0);
});

test("Workers AI free-allocation errors become refused quota", async () => {
  const { env } = makeEnv();
  env.aiError = "AiError: 4006: you have used up your daily free allocation of 10,000 neurons";
  const r = await post(env, sampleReport());
  assert.equal(r.status, 429);
  assert.equal(r.body.refused, "quota");
});

test("other model errors are 502 JSON without details", async () => {
  const { env } = makeEnv();
  env.aiError = "InferenceUpstreamError: internal";
  const r = await post(env, sampleReport());
  assert.equal(r.status, 502);
  assert.ok(!JSON.stringify(r.body).includes("InferenceUpstream"));
});

test("a model outside the free allowlist is refused", async () => {
  const { env, calls } = makeEnv({ AI_MODEL: "@cf/moonshotai/kimi-k2.6" });
  const r = await post(env, sampleReport());
  assert.equal(r.status, 503);
  assert.equal(calls.length, 0);
});
