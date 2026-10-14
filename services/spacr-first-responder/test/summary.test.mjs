import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { SUMMARY_MAX_CHARS, SUMMARY_PROMPT, worstCaseCost } from "../src/diagnose.js";
import { resetTokenCache } from "../src/github.js";
import { handle } from "../src/index.js";
import { FakeD1, FakeGitHub, generateTestKeyPem, makeRequest, sampleReport } from "./helpers.mjs";

const { pem } = await generateTestKeyPem();
const DAY1 = new Date("2026-10-14T12:00:00Z");
const DAY2 = new Date("2026-10-15T12:00:00Z");
const LIGHT = "@cf/meta/llama-3.1-8b-instruct-fp8";
const POOL = "2026-10-14|ai_neurons|global";
const LIGHT_KEY = `2026-10-14|ai_light|install:${sampleReport().install_id}`;
const HEADING = "### AI summary (generated, may be wrong)";

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
        if (env.aiError) throw new Error(env.aiError);
        if (env.aiHang) return new Promise(() => {});
        return {
          response: env.aiText ?? "Likely cause: mask and image shapes differ. Happens in spacr.measure, measure.py, _measure_intensity.",
          usage: { prompt_tokens: 400, completion_tokens: 60 },
        };
      },
    },
    APP_ID: "5315672",
    INSTALLATION_ID: "171361114",
    REPO: "EinarOlafsson/spacr",
    GITHUB_APP_PRIVATE_KEY: pem,
  };
});

afterEach(() => gh.restore());

async function post(body, now = DAY1) {
  const res = await handle(makeRequest("/report", { body, headers: { "cf-connecting-ip": "203.0.113.7" } }), env, null, now);
  return { status: res.status, body: await res.json() };
}

function issueCall() {
  return gh.calls.find((c) => /\/issues$/.test(c.url));
}

function commentCalls() {
  return gh.calls.filter((c) => /\/comments$/.test(c.url));
}

test("client summary goes into the issue with its source and costs no AI", async () => {
  const r = await post(sampleReport({ ai_summary: "Shape mismatch between mask and image in measure_crop." }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "client");
  const body = issueCall().body.body;
  assert.ok(body.includes(HEADING));
  assert.match(body, /Shape mismatch between mask and image/);
  assert.match(body, /_summary written by the reporter's own AI_/);
  assert.equal(aiCalls.length, 0);
  assert.equal(r.body.allowance.light_left, 10);
});

test("client summary is checked for personal data", async () => {
  const r = await post(sampleReport({ ai_summary: "Fails reading /home/alice/plate1/img.tif" }));
  assert.equal(r.status, 400);
  assert.match(r.body.error, /ai_summary.*absolute path/);
  const long = await post(sampleReport({ ai_summary: "x".repeat(2001) }));
  assert.equal(long.status, 400);
  const bad = await post(sampleReport({ summarize: "yes" }));
  assert.equal(bad.status, 400);
  assert.equal(gh.calls.length, 0);
});

test("relay summary uses the light model, one light question and the shared pool", async () => {
  const r = await post(sampleReport({ summarize: true }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "relay");
  assert.equal(aiCalls.length, 1);
  assert.equal(aiCalls[0].model, LIGHT);
  assert.equal(aiCalls[0].input.messages[0].content, SUMMARY_PROMPT);
  assert.match(SUMMARY_PROMPT, /Do not speculate about the user/);
  const body = issueCall().body.body;
  assert.ok(body.includes(HEADING));
  assert.match(body, new RegExp(`_summary written by spaCR AI \\(${LIGHT.replace(/[.\/]/g, "\\$&")}\\)_`));
  assert.deepEqual(r.body.allowance, { reports_left: 4, best_left: 5, light_left: 9, resets_at: "2026-10-15T00:00:00.000Z" });
  assert.ok(env.DB.counters.get(POOL) > 0);
});

test("client summary wins over summarize", async () => {
  const r = await post(sampleReport({ ai_summary: "Own summary.", summarize: true }));
  assert.equal(r.body.ai_summary_source, "client");
  assert.equal(aiCalls.length, 0);
});

test("relay summary is capped at 1200 characters and neutralises markup", async () => {
  env.aiText = `<!-- hidden --> @someone ${"word ".repeat(400)}`;
  const r = await post(sampleReport({ summarize: true }));
  assert.equal(r.body.ai_summary_source, "relay");
  const body = issueCall().body.body;
  const section = body.slice(body.indexOf(HEADING));
  assert.ok(!section.includes("<!-- hidden"));
  assert.ok(!section.includes("@someone"));
  const quoted = section.split("\n").filter((l) => l.startsWith("> ")).join("").length;
  assert.ok(quoted <= SUMMARY_MAX_CHARS + 120);
});

test("no light allowance left: the report is filed without a summary", async () => {
  env.AI_LIGHT_PER_INSTALL = "0";
  const r = await post(sampleReport({ summarize: true }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "none");
  assert.equal(aiCalls.length, 0);
  assert.ok(!issueCall().body.body.includes(HEADING));
});

test("exhausted pool: filed without a summary, nothing consumed", async () => {
  env.DB.counters.set(POOL, 9000 - worstCaseCost(LIGHT) + 1);
  const r = await post(sampleReport({ summarize: true }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "none");
  assert.equal(aiCalls.length, 0);
  assert.equal(env.DB.counters.get(LIGHT_KEY) ?? 0, 0);
});

test("AI error: filed without a summary and the light question is refunded", async () => {
  env.aiError = "InferenceUpstreamError";
  const r = await post(sampleReport({ summarize: true }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "none");
  assert.equal(r.body.allowance.light_left, 10);
});

test("AI 4006: filed without a summary and the pool is closed", async () => {
  env.aiError = "4006: daily free allocation exceeded";
  const r = await post(sampleReport({ summarize: true }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "none");
  assert.equal(env.DB.counters.get(POOL), 9000);
});

test("AI timeout: filed without a summary", async () => {
  const { summarizeWithRelay } = await import("../src/diagnose.js");
  env.aiHang = true;
  const s = await summarizeWithRelay(env, { ...sampleReport(), question: null }, DAY1, 20);
  assert.equal(s, null);
  assert.equal(env.DB.counters.get(LIGHT_KEY), 0);
});

test("dry run includes the relay summary without calling GitHub", async () => {
  const r = await post(sampleReport({ dry_run: true, summarize: true }));
  assert.equal(r.status, 200);
  assert.equal(r.body.ai_summary_source, "relay");
  assert.ok(r.body.body.includes(HEADING));
  assert.equal(r.body.allowance.light_left, 9);
  assert.equal(gh.calls.length, 0);
});

test("dedupe: +1 carries a summary only while the issue has none", async () => {
  await post(sampleReport());
  assert.ok(!issueCall().body.body.includes(HEADING));
  const second = await post(sampleReport({ summarize: true }));
  assert.equal(second.body.deduplicated, true);
  assert.equal(second.body.ai_summary_source, "relay");
  assert.ok(commentCalls()[0].body.body.includes(HEADING));
  const third = await post(sampleReport({ ai_summary: "Another view." }), DAY2);
  assert.equal(third.body.commented, true);
  assert.equal(third.body.ai_summary_source, "none");
  assert.ok(!commentCalls()[1].body.body.includes(HEADING));
  assert.equal(aiCalls.length, 1);
});

test("dedupe: an issue opened with a summary gets none in its +1", async () => {
  await post(sampleReport({ ai_summary: "First summary." }));
  const second = await post(sampleReport({ summarize: true }));
  assert.equal(second.body.ai_summary_source, "none");
  assert.equal(aiCalls.length, 0);
  assert.ok(!commentCalls()[0].body.body.includes(HEADING));
});

test("dedupe: no comment slot today means no summary and no AI spent", async () => {
  await post(sampleReport());
  await post(sampleReport());
  const third = await post(sampleReport({ summarize: true }));
  assert.equal(third.body.commented, false);
  assert.equal(third.body.ai_summary_source, "none");
  assert.equal(aiCalls.length, 0);
});

test("dedupe: a failed relay summary leaves the slot open for a later report", async () => {
  await post(sampleReport());
  env.aiError = "InferenceUpstreamError";
  const second = await post(sampleReport({ summarize: true }));
  assert.equal(second.body.ai_summary_source, "none");
  delete env.aiError;
  const third = await post(sampleReport({ summarize: true }), DAY2);
  assert.equal(third.body.ai_summary_source, "relay");
});
