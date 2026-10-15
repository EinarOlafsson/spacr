import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { EXPLAIN_QUESTION } from "../src/diagnose.js";
import { SUMMARY_PREFIX } from "../src/explanations.js";
import { resetTokenCache } from "../src/github.js";
import { handle } from "../src/index.js";
import { signatureOf } from "../src/report.js";
import { validatePayload } from "../src/validate.js";
import { FakeD1, FakeGitHub, generateTestKeyPem, makeRequest, sampleReport } from "./helpers.mjs";

const { pem } = await generateTestKeyPem();
const DAY1 = new Date("2026-10-14T12:00:00Z");
const BEST = "@cf/qwen/qwen2.5-coder-32b-instruct";
const LIGHT = "@cf/meta/llama-3.1-8b-instruct-fp8";
const ME = sampleReport().install_id;
const MINE = `2026-10-14|ai_install_neurons|install:${ME}`;
const POOL = "2026-10-14|ai_neurons|global";

let gh;
let env;
let aiCalls;

beforeEach(() => {
  resetTokenCache();
  gh = new FakeGitHub().install();
  aiCalls = [];
  env = {
    DB: new FakeD1(),
    AI: {
      async run(model, input) {
        aiCalls.push({ model, input });
        return { response: `**What happened** answer ${aiCalls.length}`, usage: { prompt_tokens: 400, completion_tokens: 60 } };
      },
    },
    AI_MODEL_BEST: BEST,
    AI_MODEL_LIGHT: LIGHT,
    AI_DAILY_NEURON_CAP: "9000",
    AI_MIN_SHARE_NEURONS: "300",
    PER_IP_DAILY_LIMIT: "1000",
    APP_ID: "5315672",
    INSTALLATION_ID: "171361114",
    REPO: "EinarOlafsson/spacr",
    GITHUB_APP_PRIVATE_KEY: pem,
  };
});

afterEach(() => gh.restore());

async function call(path, body, now = DAY1) {
  const res = await handle(makeRequest(path, { body, headers: { "cf-connecting-ip": "203.0.113.7" } }), env, null, now);
  return { status: res.status, body: await res.json() };
}

const days = (n) => new Date(DAY1.getTime() + n * 86400000);

test("a repeated explanation is answered from the cache without a share charge", async () => {
  const first = await call("/diagnose", sampleReport({ question: EXPLAIN_QUESTION }));
  assert.equal(first.status, 200);
  assert.equal(first.body.cached, false);
  assert.equal(aiCalls.length, 1);
  const spent = env.DB.counters.get(MINE);
  const pool = env.DB.counters.get(POOL);
  const other = "5f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b";
  const again = await call("/diagnose", sampleReport({ install_id: other }));
  assert.equal(again.status, 200);
  assert.equal(again.body.cached, true);
  assert.equal(again.body.advice, first.body.advice);
  assert.equal(again.body.model, first.body.model);
  assert.equal(again.body.tier, first.body.tier);
  assert.equal(aiCalls.length, 1, "no model call");
  assert.equal(env.DB.counters.get(POOL), pool, "no pool neurons");
  assert.equal(env.DB.counters.get(MINE), spent);
  assert.equal(env.DB.counters.has(`2026-10-14|ai_install_neurons|install:${other}`), false, "not marked active");
  assert.equal(again.body.allowance.ai.active_users, 2, "the allowance is read, not charged");
  const sig = await signatureOf(validatePayload(sampleReport(), "diagnose"));
  assert.equal(env.DB.explanations.get(sig).hits, 1);
});

test("a miss calls the model and stores the answer", async () => {
  assert.equal(env.DB.explanations.size, 0);
  const r = await call("/diagnose", sampleReport());
  assert.equal(r.body.cached, false);
  const sig = await signatureOf(validatePayload(sampleReport(), "diagnose"));
  const row = env.DB.explanations.get(sig);
  assert.equal(row.text, r.body.advice);
  assert.equal(row.spacr_minor, "1.5");
  assert.equal(row.created, DAY1.toISOString());
});

test("another spaCR minor version or an answer 30 days old asks the model again", async () => {
  await call("/diagnose", sampleReport());
  const newer = await call("/diagnose", sampleReport({ spacr_version: "1.6.0" }));
  assert.equal(newer.body.cached, false);
  assert.equal(aiCalls.length, 2);
  const patch = await call("/diagnose", sampleReport({ spacr_version: "1.6.3" }), days(29));
  assert.equal(patch.body.cached, true);
  const stale = await call("/diagnose", sampleReport({ spacr_version: "1.6.3" }), days(31));
  assert.equal(stale.body.cached, false);
  assert.equal(aiCalls.length, 3);
});

test("a user error's signature follows its problem list, numbers and paths aside", async () => {
  const base = { kind: "user_error", exception_type: "ConfigurationError", module: "mask" };
  const one = validatePayload(sampleReport({ ...base, message: "The run did not start.\n- cell_channel: no channel is set" }), "report");
  const two = validatePayload(sampleReport({ ...base, message: "The run did not start.\n- src does not exist\n- cell_channel: no channel is set" }), "report");
  const twoAgain = validatePayload(
    sampleReport({ ...base, message: "The run did not start: 2 problems.\n- cell_channel: no channel is set\n- src <PATH> does not exist 12" }),
    "report",
  );
  assert.notEqual(await signatureOf(one), await signatureOf(two));
  assert.equal(await signatureOf(two), await signatureOf(twoAgain));
  const fault = validatePayload(sampleReport({ message: "different text" }), "report");
  assert.equal(await signatureOf(fault), await signatureOf(validatePayload(sampleReport(), "report")));
});

test("a free-form chat question is never cached", async () => {
  await call("/diagnose", sampleReport({ question: "How do I pick a diameter?" }));
  const again = await call("/diagnose", sampleReport({ question: "How do I pick a diameter?" }));
  assert.equal(again.body.cached, false);
  assert.equal(aiCalls.length, 2);
  assert.equal(env.DB.explanations.size, 0);
});

test("relay summaries are cached too, and dropped when the issue is found closed", async () => {
  const first = await call("/report", sampleReport({ summarize: true }));
  assert.equal(first.body.ai_summary_source, "relay");
  assert.equal(first.body.ai_summary_cached, false);
  const sig = await signatureOf(validatePayload(sampleReport(), "report"));
  assert.ok(env.DB.explanations.has(SUMMARY_PREFIX + sig));
  await call("/diagnose", sampleReport());
  assert.ok(env.DB.explanations.has(sig));
  const dry = await call("/report", sampleReport({ summarize: true, dry_run: true }));
  assert.equal(dry.body.ai_summary_cached, true);
  assert.equal(aiCalls.length, 2);
  gh.issueState = "closed";
  const recurred = await call("/report", sampleReport());
  assert.equal(recurred.body.recurred_from, 101);
  assert.equal(env.DB.explanations.has(sig), false);
  assert.equal(env.DB.explanations.has(SUMMARY_PREFIX + sig), false);
  const fresh = await call("/diagnose", sampleReport());
  assert.equal(fresh.body.cached, false);
});
