import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { resetTokenCache } from "../src/github.js";
import { handle } from "../src/index.js";
import { signatureOf } from "../src/report.js";
import { FakeD1, FakeGitHub, generateTestKeyPem, makeRequest, sampleReport } from "./helpers.mjs";

const { pem } = await generateTestKeyPem();

let gh;
let env;

beforeEach(() => {
  resetTokenCache();
  gh = new FakeGitHub().install();
  env = {
    DB: new FakeD1(),
    APP_ID: "5315672",
    INSTALLATION_ID: "171361114",
    REPO: "EinarOlafsson/spacr",
    GITHUB_APP_PRIVATE_KEY: pem,
    DAILY_REPORT_LIMIT: "200",
    PER_INSTALL_DAILY_LIMIT: "10",
    PER_IP_DAILY_LIMIT: "20",
  };
});

afterEach(() => gh.restore());

const DAY1 = new Date("2026-10-14T12:00:00Z");
const DAY2 = new Date("2026-10-15T12:00:00Z");

async function post(path, body, now = DAY1, ip = "203.0.113.7") {
  const res = await handle(makeRequest(path, { body, headers: { "cf-connecting-ip": ip } }), env, null, now);
  return { status: res.status, body: await res.json(), headers: res.headers };
}

test("health lists routes", async () => {
  const res = await handle(makeRequest("/health", { method: "GET" }), env, null, DAY1);
  const body = await res.json();
  assert.equal(res.status, 200);
  assert.equal(body.ok, true);
  assert.deepEqual(body.routes, ["GET /health", "POST /report", "POST /diagnose"]);
});

test("unknown routes and wrong methods return JSON errors", async () => {
  let r = await handle(makeRequest("/nope", { method: "GET" }), env, null, DAY1);
  assert.equal(r.status, 404);
  assert.ok((await r.json()).error);
  r = await handle(makeRequest("/report", { method: "GET" }), env, null, DAY1);
  assert.equal(r.status, 405);
});

test("invalid JSON and oversized bodies are rejected", async () => {
  let r = await post("/report", "{not json");
  assert.equal(r.status, 400);
  assert.match(r.body.error, /JSON/);
  r = await post("/report", sampleReport({ user_note: "a ".repeat(9000) }));
  assert.equal(r.status, 413);
  assert.equal(gh.calls.length, 0);
});

test("validation errors are 400 with a clear message and no GitHub call", async () => {
  const r = await post("/report", sampleReport({ message: "failed for /home/alice/x.tif" }));
  assert.equal(r.status, 400);
  assert.match(r.body.error, /absolute path/);
  assert.equal(gh.calls.length, 0);
});

test("dry run returns title and body without GitHub", async () => {
  const r = await post("/report", sampleReport({ dry_run: true, user_note: "ping @maintainer" }));
  assert.equal(r.status, 200);
  assert.equal(r.body.dry_run, true);
  assert.match(r.body.title, /^\[user-report\] ValueError in spacr\.measure: operands/);
  assert.match(r.body.body, /measure\.py:233 in _measure_intensity/);
  assert.match(r.body.body, /spacr-signature: [0-9a-f]{64}/);
  assert.ok(!r.body.body.includes("@maintainer"), "mentions are neutralised");
  assert.equal(gh.calls.length, 0);
  assert.equal(env.DB.signatures.size, 0);
});

test("first report opens an issue with the user-report label", async () => {
  const r = await post("/report", sampleReport());
  assert.equal(r.status, 200);
  assert.deepEqual(r.body, {
    issue_number: 101,
    issue_url: "https://github.com/EinarOlafsson/spacr/issues/101",
    deduplicated: false,
  });
  const create = gh.calls.find((c) => c.url.endsWith("/repos/EinarOlafsson/spacr/issues"));
  assert.deepEqual(create.body.labels, ["user-report"]);
  assert.equal(create.headers["user-agent"], "spacr-first-responder");
  assert.equal(create.headers.authorization, "Bearer ghs_testtoken");
  const tokenCall = gh.calls.find((c) => c.url.includes("/access_tokens"));
  assert.deepEqual(tokenCall.body, { repositories: ["spacr"], permissions: { issues: "write" } });
  assert.ok(!JSON.stringify(r.body).includes("ghs_"), "token never returned");
});

test("duplicate adds one +1 comment per day, then only counts", async () => {
  await post("/report", sampleReport());
  const second = await post("/report", sampleReport({ spacr_version: "1.5.1.6", os: "windows" }));
  assert.equal(second.body.deduplicated, true);
  assert.equal(second.body.issue_number, 101);
  assert.equal(second.body.commented, true);
  const comment = gh.calls.find((c) => /\/issues\/101\/comments$/.test(c.url));
  assert.match(comment.body.body, /^\+1 from spaCR 1\.5\.1\.6 · windows/);
  assert.match(comment.body.body, /Reports so far: 2/);

  const third = await post("/report", sampleReport());
  assert.equal(third.body.deduplicated, true);
  assert.equal(third.body.commented, false);
  assert.equal(gh.count(/\/comments$/), 1);
  assert.equal(gh.count(/\/issues$/), 1);

  const nextDay = await post("/report", sampleReport(), DAY2);
  assert.equal(nextDay.body.commented, true);
  assert.equal(gh.count(/\/comments$/), 2);
  const sig = await signatureOf(sampleReport());
  assert.equal(env.DB.signatures.get(sig).count, 4);
});

test("signature ignores line numbers but not the top frames", async () => {
  const a = await signatureOf(sampleReport());
  const moved = sampleReport();
  moved.frames = moved.frames.map((f) => ({ ...f, line: f.line + 3 }));
  assert.equal(await signatureOf(moved), a);
  const other = sampleReport();
  other.frames[4] = { file: "_methods.py", function: "_sum", line: 1 };
  assert.notEqual(await signatureOf(other), a);
});

test("a deleted issue is replaced by a new one", async () => {
  await post("/report", sampleReport());
  gh.commentStatus = 410;
  const r = await post("/report", sampleReport());
  assert.equal(r.body.deduplicated, false);
  assert.equal(r.body.issue_number, 102);
});

test("GitHub failure releases the signature claim and returns 502", async () => {
  gh.issueStatus = 500;
  const r = await post("/report", sampleReport());
  assert.equal(r.status, 502);
  assert.equal(env.DB.signatures.size, 0);
  assert.ok(!JSON.stringify(r.body).includes("ghs_"));
});

test("per-install quota returns 429 refused quota", async () => {
  env.PER_INSTALL_DAILY_LIMIT = "2";
  assert.equal((await post("/report", sampleReport({ dry_run: true }))).status, 200);
  assert.equal((await post("/report", sampleReport({ dry_run: true }))).status, 200);
  const r = await post("/report", sampleReport({ dry_run: true }));
  assert.equal(r.status, 429);
  assert.equal(r.body.refused, "quota");
  const other = sampleReport({ dry_run: true, install_id: "11111111-2222-4333-8444-555555555555" });
  assert.equal((await post("/report", other)).status, 200);
});

test("per-IP quota is keyed on a hash, never the raw IP", async () => {
  env.PER_IP_DAILY_LIMIT = "1";
  assert.equal((await post("/report", sampleReport({ dry_run: true }))).status, 200);
  const other = sampleReport({ dry_run: true, install_id: "11111111-2222-4333-8444-555555555555" });
  const r = await post("/report", other);
  assert.equal(r.status, 429);
  assert.equal(r.body.refused, "quota");
  for (const key of env.DB.counters.keys()) assert.ok(!key.includes("203.0.113.7"));
  assert.equal((await post("/report", other, DAY1, "198.51.100.9")).status, 200);
});

test("global quota refuses and refunds the per-install reservation", async () => {
  env.DAILY_REPORT_LIMIT = "1";
  assert.equal((await post("/report", sampleReport())).status, 200);
  const r = await post("/report", sampleReport({ install_id: "11111111-2222-4333-8444-555555555555" }), DAY1, "198.51.100.9");
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "global" });
  assert.equal(env.DB.counters.get("2026-10-14|report|install:11111111-2222-4333-8444-555555555555"), 0);
  assert.equal(gh.count(/\/issues$/), 1);
});

test("missing private key gives 503 without leaking configuration", async () => {
  delete env.GITHUB_APP_PRIVATE_KEY;
  const r = await post("/report", sampleReport());
  assert.equal(r.status, 503);
  assert.deepEqual(r.body, { error: "relay is not configured" });
});

test("installation token is cached between reports", async () => {
  await post("/report", sampleReport());
  await post("/report", sampleReport({ exception_type: "KeyError" }));
  assert.equal(gh.count(/access_tokens$/), 1);
});
