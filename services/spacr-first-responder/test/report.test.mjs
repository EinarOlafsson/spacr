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
    REPORTS_PER_INSTALL: "5",
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
  assert.deepEqual(body.routes, ["GET /health", "GET /allowance", "POST /report", "POST /diagnose"]);
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
  r = await post("/report", sampleReport({ user_note: "a ".repeat(17000) }));
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
  const { allowance, ...rest } = r.body;
  assert.deepEqual(rest, {
    issue_number: 101,
    issue_url: "https://github.com/EinarOlafsson/spacr/issues/101",
    deduplicated: false,
    ai_summary_source: "none",
  });
  assert.equal(allowance.reports_left, 4);
  assert.equal(allowance.resets_at, "2026-10-15T00:00:00.000Z");
  assert.equal(allowance.pool, "exhausted", "no AI binding in this env");
  assert.equal(allowance.ai.tier_now, "none");
  assert.equal(allowance.best_left, 0);
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
  gh.stateStatus = 404;
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
  env.REPORTS_PER_INSTALL = "2";
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

test("report allowance counts down to 429 install at five a day and resets", async () => {
  const lefts = [];
  for (let i = 0; i < 5; i++) {
    const r = await post("/report", sampleReport({ exception_type: `Error${i}` }));
    assert.equal(r.status, 200);
    lefts.push(r.body.allowance.reports_left);
  }
  assert.deepEqual(lefts, [4, 3, 2, 1, 0]);
  const r = await post("/report", sampleReport({ exception_type: "Error9" }));
  assert.equal(r.status, 429);
  assert.deepEqual(r.body, { refused: "quota", scope: "install" });
  const tomorrow = await post("/report", sampleReport({ exception_type: "Error9" }), DAY2);
  assert.equal(tomorrow.status, 200);
  assert.equal(tomorrow.body.allowance.reports_left, 4);
  assert.equal(tomorrow.body.allowance.resets_at, "2026-10-16T00:00:00.000Z");
});

test("a dry run does not use the real report allowance", async () => {
  const r = await post("/report", sampleReport({ dry_run: true }));
  assert.equal(r.body.allowance.reports_left, 5);
});

test("allowance endpoint validates the uuid and consumes nothing", async () => {
  let res = await handle(makeRequest("/allowance?install_id=nope", { method: "GET" }), env, null, DAY1);
  assert.equal(res.status, 400);
  assert.match((await res.json()).error, /install_id/);
  await post("/report", sampleReport());
  const before = new Map(env.DB.counters);
  const url = `/allowance?install_id=${sampleReport().install_id}`;
  for (let i = 0; i < 3; i++) {
    res = await handle(makeRequest(url, { method: "GET" }), env, null, DAY1);
    assert.equal(res.status, 200);
  }
  const a = await res.json();
  assert.equal(a.reports_left, 4);
  assert.equal(a.pool, "exhausted");
  assert.equal(a.resets_at, "2026-10-15T00:00:00.000Z");
  assert.deepEqual(env.DB.counters, before);
});

test("kind user_error gets the settings prefix and the user-error label; dedupe as before", async () => {
  const r = await post("/report", sampleReport({ kind: "user_error", exception_type: "SettingsError" }));
  assert.equal(r.status, 200);
  const create = gh.calls.find((c) => /\/issues$/.test(c.url));
  assert.match(create.body.title, /^\[user-report\]\[settings\] SettingsError in spacr\.measure/);
  assert.deepEqual(create.body.labels, ["user-report", "user-error"]);
  assert.match(create.body.body, /Kind: \*\*user error\*\*/);
  const again = await post("/report", sampleReport({ kind: "user_error", exception_type: "SettingsError" }));
  assert.equal(again.body.deduplicated, true);
  const fault = await post("/report", sampleReport({ kind: "fault", exception_type: "OtherError" }));
  const second = gh.calls.filter((c) => /\/issues$/.test(c.url))[1];
  assert.equal(fault.body.deduplicated, false);
  assert.match(second.body.title, /^\[user-report\] OtherError/);
  assert.deepEqual(second.body.labels, ["user-report"]);
  const bad = await post("/report", sampleReport({ kind: "crash" }));
  assert.equal(bad.status, 400);
  assert.match(bad.body.error, /kind/);
});

test("frames[].code is rendered under its frame line in the traceback", async () => {
  const report = sampleReport();
  report.frames[3].code = "    mean = np.mean(image[mask > 0])";
  const r = await post("/report", { ...report, dry_run: true });
  assert.equal(r.status, 200);
  assert.match(r.body.body, /measure\.py:233 in _measure_intensity\n {6}mean = np\.mean\(image\[mask > 0\]\)/);
});

test("a closed issue is not +1'd: a [recurred] issue is opened and the signature re-pointed", async () => {
  await post("/report", sampleReport());
  gh.issueState = "closed";
  const r = await post("/report", sampleReport());
  assert.equal(r.body.deduplicated, false);
  assert.equal(r.body.recurred_from, 101);
  assert.equal(r.body.issue_number, 102);
  assert.equal(gh.count(/\/comments$/), 0);
  const create = gh.calls.filter((c) => c.method === "POST" && /\/issues$/.test(c.url))[1];
  assert.match(create.body.title, /^\[recurred\] \[user-report\]/);
  assert.match(create.body.body, /Seen again after #101 was closed\./);
  const sig = await signatureOf(sampleReport());
  const row = env.DB.signatures.get(sig);
  assert.equal(row.issue_number, 102);
  assert.equal(row.count, 1);
  assert.equal(row.last_comment_day, null);
  assert.equal(row.has_summary, 0);
});

test("an open issue gets the comment after one state check", async () => {
  await post("/report", sampleReport());
  const r = await post("/report", sampleReport());
  assert.equal(r.body.commented, true);
  assert.equal(r.body.recurred_from, undefined);
  assert.equal(gh.count(/\/issues\/101$/), 1);
  assert.equal(gh.count(/\/comments$/), 1);
});

test("the state is not read when the daily comment cap blocks the comment", async () => {
  await post("/report", sampleReport());
  await post("/report", sampleReport());
  const checks = gh.count(/\/issues\/101$/);
  gh.issueState = "closed";
  const third = await post("/report", sampleReport());
  assert.equal(third.body.commented, false);
  assert.equal(third.body.deduplicated, true);
  assert.equal(gh.count(/\/issues\/101$/), checks);
  assert.equal(gh.count(/\/issues$/), 1);
});
