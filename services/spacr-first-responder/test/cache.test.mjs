import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { EXPLAIN_QUESTION } from "../src/diagnose.js";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

import { SUMMARY_PREFIX, cleanup, normaliseQuestion, questionKey, store } from "../src/cache.js";
import { SQL } from "../src/sql.js";
import { resetTokenCache } from "../src/github.js";
import { handle } from "../src/index.js";
import { signatureOf } from "../src/report.js";
import { validatePayload } from "../src/validate.js";
import { sha256Hex } from "../src/util.js";
import { SYSTEM_PROMPT } from "../src/prompts.js";
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
  assert.equal(env.DB.cache.get(sig).hits, 1);
});

test("a miss calls the model and stores the answer", async () => {
  assert.equal(env.DB.cache.size, 0);
  const r = await call("/diagnose", sampleReport());
  assert.equal(r.body.cached, false);
  const sig = await signatureOf(validatePayload(sampleReport(), "diagnose"));
  const row = env.DB.cache.get(sig);
  assert.equal(row.text, r.body.advice);
  assert.equal(row.spacr_minor, "1.5");
  assert.equal(row.kind, "error");
  assert.equal(row.created, DAY1.getTime() / 1000);
  assert.equal(row.last_used, row.created);
  assert.ok(row.bytes >= row.text.length);
  assert.deepEqual(env.DB.cacheMeta, { rows: 1, bytes: row.bytes, inserts: 1 });
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

test("a chat question is cached by its normalised text, screen, module and spaCR minor", async () => {
  const ask = (question, extra = {}) =>
    sampleReport({ question, screen: "Mask", message: "", frames: [], ...extra });
  const first = await call("/diagnose", ask("How do I pick a diameter?"));
  assert.equal(first.body.cached, false);
  const same = await call("/diagnose", ask("  how do I pick   a DIAMETER "));
  assert.equal(same.body.cached, true);
  assert.equal(same.body.advice, first.body.advice);
  assert.equal(aiCalls.length, 1);
  await call("/diagnose", ask("How do I pick a diameter?", { screen: "Measure" }));
  await call("/diagnose", ask("How do I pick a diameter?", { spacr_version: "1.6.0" }));
  assert.equal(aiCalls.length, 3, "another screen or minor asks again");
  const rows = [...env.DB.cache.values()];
  assert.ok(rows.every((r) => r.kind === "question" && r.key.startsWith("q:")));
});

test("question cache identity includes the current prompt and ignores a fresh legacy answer", async () => {
  const question = "How do I generate object masks?";
  const report = validatePayload(sampleReport({ question, screen: "Mask", frames: [], message: "", spacr_version: "1.5.1.7" }), "diagnose");
  const legacyInput = `${normaliseQuestion(question)}\nMask|${report.module}\n1.5`;
  const legacyKey = "q:" + await sha256Hex(legacyInput);
  await store(env, legacyKey, "question", report, DAY1,
    { text: "Old generic advice without module links", model: BEST, tier: "best" });
  const currentKey = await questionKey(report);
  assert.equal(currentKey, "q:" + await sha256Hex(`${legacyInput}\n${SYSTEM_PROMPT}`));
  assert.notEqual(currentKey, legacyKey);
  const first = await call("/diagnose", report);
  assert.equal(first.status, 200);
  assert.equal(first.body.cached, false);
  assert.equal(aiCalls.length, 1);
  assert.notEqual(first.body.advice, "Old generic advice without module links");
  const second = await call("/diagnose", { ...report, question: "HOW do I generate OBJECT masks!" });
  assert.equal(second.body.cached, true);
  assert.equal(second.body.advice, first.body.advice);
  assert.equal(aiCalls.length, 1);
  assert.ok(env.DB.cache.has(legacyKey), "legacy entry is left for ordinary TTL cleanup");
});

test("question normalisation folds case, whitespace and punctuation", async () => {
  assert.equal(normaliseQuestion("  How do I -- pick a DIAMETER?!\n"), "how do i pick a diameter");
  assert.equal(normaliseQuestion("Hvað er <PATH> þvermál?"), "hvað er þvermál");
  const a = await questionKey(validatePayload(sampleReport({ question: "Pick, a diameter?", frames: [], message: "" }), "diagnose"));
  const b = await questionKey(validatePayload(sampleReport({ question: "pick a diameter", frames: [], message: "" }), "diagnose"));
  assert.equal(a, b);
  assert.match(a, /^q:[0-9a-f]{64}$/);
});

test("a question with more than 500 characters of pasted context is never cached", async () => {
  const long = `Why does this fail? ${"x ".repeat(300)}`;
  await call("/diagnose", sampleReport({ question: long, frames: [], message: "" }));
  const again = await call("/diagnose", sampleReport({ question: long, frames: [], message: "" }));
  assert.equal(again.body.cached, false);
  const context = "shape mismatch ".repeat(19);
  const pasted = `Why? ${"line of pasted log ".repeat(12)}`;
  await call("/diagnose", sampleReport({ question: pasted, message: context }));
  const twice = await call("/diagnose", sampleReport({ question: pasted, message: context }));
  assert.equal(twice.status, 200);
  assert.equal(twice.body.cached, false);
  assert.equal(aiCalls.length, 4);
  assert.equal(env.DB.cache.size, 0);
});

const NOW = new Date("2026-10-14T12:00:00Z");
const report = validatePayload(sampleReport(), "diagnose");
const put = (e, key, now = NOW) => store(e, key, "error", report, now, { text: `answer ${key}`, model: LIGHT, tier: "light" });

test("cleanup runs on every 50th insert, not on each request", async () => {
  const e = { DB: new FakeD1() };
  for (let i = 0; i < 49; i++) await put(e, `k${i}`);
  assert.equal(e.DB.log.filter((q) => q === SQL.cacheCount).length, 0, "no COUNT(*) before the 50th insert");
  await put(e, "k49");
  assert.equal(e.DB.log.filter((q) => q === SQL.cacheCount).length, 1);
  assert.equal(e.DB.cacheMeta.rows, 50);
});

test("expired rows go first, then the fewest-hit, least recently used 10%", async () => {
  const e = { DB: new FakeD1(), AI_CACHE_MAX_ROWS: "10", AI_CACHE_TTL_DAYS: "30" };
  await put(e, "old", new Date(NOW.getTime() - 31 * 86400000));
  for (let i = 0; i < 10; i++) {
    await put(e, `k${i}`, new Date(NOW.getTime() + i * 1000));
  }
  for (const k of ["k0", "k1", "k2"]) e.DB.cache.get(k).hits = 5;
  e.DB.cache.get("k3").last_used = NOW.getTime() / 1000 + 100;
  await put(e, "k10", new Date(NOW.getTime() + 20000));
  assert.equal(e.DB.cache.has("old"), false, "expired");
  assert.equal(e.DB.cache.has("k4"), false, "fewest hits, oldest use");
  assert.equal(e.DB.cache.has("k5"), false, "about 10% go");
  for (const k of ["k0", "k1", "k2", "k3", "k6", "k10"]) assert.ok(e.DB.cache.has(k), k);
  assert.equal(e.DB.cacheMeta.rows, e.DB.cache.size);
  assert.ok(e.DB.cache.size <= 10);
});

test("the byte cap evicts too, and the meta row is corrected", async () => {
  const e = { DB: new FakeD1(), AI_CACHE_MAX_BYTES: "200" };
  for (let i = 0; i < 20; i++) await put(e, `key-${i}`, new Date(NOW.getTime() + i * 1000));
  const bytes = [...e.DB.cache.values()].reduce((n, r) => n + r.bytes, 0);
  assert.ok(bytes <= 200 + 40, `bytes ${bytes}`);
  assert.ok(e.DB.cache.has("key-19"), "the newest stays");
  assert.equal(e.DB.cache.has("key-0"), false, "the oldest went");
  const out = await cleanup(e, NOW);
  assert.deepEqual(out, { rows: e.DB.cache.size, bytes });
});

test("/health shows cache rows, bytes and today's hit rate, no keys or text", async () => {
  await call("/diagnose", sampleReport());
  await call("/diagnose", sampleReport());
  await call("/diagnose", sampleReport());
  const res = await handle(new Request("https://relay.example/health"), env, null, DAY1);
  const body = await res.json();
  assert.equal(body.version, "0.2.5");
  assert.equal(body.cache.rows, 1);
  assert.ok(body.cache.bytes > 0);
  assert.deepEqual(body.cache.today, { hits: 2, misses: 1, hit_rate: 0.667 });
  const text = JSON.stringify(body);
  assert.equal(text.includes("answer"), false);
  assert.equal(/[0-9a-f]{64}/.test(text), false);
});

test("every ai_cache lookup is by primary key (sqlite plan)", (t) => {
  const reads = [SQL.cacheGet, SQL.cacheHit, SQL.cacheDrop];
  for (const q of reads) assert.match(q, /WHERE key = \?1/);
  let sqlite;
  try {
    execFileSync("sqlite3", ["-version"]);
    sqlite = "sqlite3";
  } catch (_) {
    t.skip("sqlite3 is not installed");
    return;
  }
  const schema = readFileSync(new URL("../schema.sql", import.meta.url), "utf8");
  const plan = (q) =>
    execFileSync(sqlite, [":memory:"], { input: `${schema}\nEXPLAIN QUERY PLAN ${q.replace(/\?\d/g, "'x'")};\n` }).toString();
  for (const q of reads) assert.match(plan(q), /USING (INDEX sqlite_autoindex_ai_cache_1|PRIMARY KEY)|MULTI-INDEX OR/, q);
  assert.match(plan(SQL.cacheEvict), /ai_cache_eviction/);
});

test("relay summaries are cached too, and dropped when the issue is found closed", async () => {
  const first = await call("/report", sampleReport({ summarize: true }));
  assert.equal(first.body.ai_summary_source, "relay");
  assert.equal(first.body.ai_summary_cached, false);
  const sig = await signatureOf(validatePayload(sampleReport(), "report"));
  assert.ok(env.DB.cache.has(SUMMARY_PREFIX + sig));
  await call("/diagnose", sampleReport());
  assert.ok(env.DB.cache.has(sig));
  const dry = await call("/report", sampleReport({ summarize: true, dry_run: true }));
  assert.equal(dry.body.ai_summary_cached, true);
  assert.equal(aiCalls.length, 2);
  gh.issueState = "closed";
  const recurred = await call("/report", sampleReport());
  assert.equal(recurred.body.recurred_from, 101);
  assert.equal(env.DB.cache.has(sig), false);
  assert.equal(env.DB.cache.has(SUMMARY_PREFIX + sig), false);
  const fresh = await call("/diagnose", sampleReport());
  assert.equal(fresh.body.cached, false);
});
