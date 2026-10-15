import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { normaliseCause, splitCause } from "../src/causes.js";
import { resetTokenCache } from "../src/github.js";
import { handle } from "../src/index.js";
import { SYSTEM_PROMPT, SUMMARY_PROMPT } from "../src/prompts.js";
import { issueBody } from "../src/report.js";
import { validatePayload } from "../src/validate.js";
import { FakeD1, FakeGitHub, generateTestKeyPem, makeRequest, sampleReport } from "./helpers.mjs";

const { pem } = await generateTestKeyPem();
const DAY1 = new Date("2026-10-14T12:00:00Z");

const DIAG = {
  chain: [{ type: "OSError", message: "disk error on <PATH>" }],
  settings: { src: "<path: exists=no>", cell_channel: null, cellpose_model: "cyto3", cell_diameter: 30.5, save: [true, false], notes: "<text>" },
  inputs: { images: 12, extensions: ["tif"], shape: [3, 512, 512], dtype: "uint16", channels: 3 },
  environment: {
    os_version: "7.0.0-38-generic",
    arch: "x86_64",
    cores: 32,
    ram_gb: 126,
    gpu_model: "NVIDIA GeForce RTX 4090",
    vram_gb: 24,
    cuda: true,
    mps: false,
    install: "editable",
    versions: { torch: "2.6.0", cellpose: "4.2.1", numpy: "1.26.4", scikit_image: "0.23.2", pyside6: "6.10.0", qt: "6.10.0" },
  },
  run: { stage: "preprocess_generate_masks", elapsed_s: 3.2, dry_run: false },
  log_tail: "[INFO] reading <PATH>\n[ERROR] failed",
};

let gh;
let env;
let aiText;

beforeEach(() => {
  resetTokenCache();
  gh = new FakeGitHub().install();
  aiText = "**What happened** The channel is not set.\nCAUSE: No object channel set";
  env = {
    DB: new FakeD1(),
    AI: { async run() { return { response: aiText, usage: { prompt_tokens: 300, completion_tokens: 50 } }; } },
    AI_MODEL_BEST: "@cf/qwen/qwen2.5-coder-32b-instruct",
    AI_MODEL_LIGHT: "@cf/meta/llama-3.1-8b-instruct-fp8",
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

async function call(path, body) {
  const res = await handle(makeRequest(path, { body, headers: { "cf-connecting-ip": "203.0.113.7" } }), env, null, DAY1);
  return { status: res.status, body: await res.json() };
}

const created = () => gh.calls.filter((c) => c.method === "POST" && /\/issues$/.test(c.url));
const comments = () => gh.calls.filter((c) => /\/comments$/.test(c.url));

test("every diagnostic field is validated and kept", () => {
  const r = validatePayload(sampleReport(DIAG), "report");
  for (const key of Object.keys(DIAG)) assert.deepEqual(r[key], DIAG[key], key);
  const bad = [
    { chain: [{ type: "Bad Type", message: "" }] },
    { chain: new Array(6).fill({ type: "E", message: "" }) },
    { settings: { "bad name": 1 } },
    { settings: { a: { nested: 1 } } },
    { settings: { a: "x".repeat(61) } },
    { inputs: { images: -1 } },
    { inputs: { extensions: ["TIF!"] } },
    { inputs: { owner: "alice" } },
    { environment: { install: "brew" } },
    { environment: { versions: { torch: "2.6 beta" } } },
    { environment: { hostname: "lab" } },
    { run: { stage: "has spaces" } },
    { run: { elapsed_s: -1 } },
    { log_tail: "x".repeat(3001) },
  ];
  for (const extra of bad) assert.throws(() => validatePayload(sampleReport(extra), "report"), /field|chain/, JSON.stringify(extra).slice(0, 60));
});

test("personal data in diagnostic text is redacted or dropped, never kept", () => {
  const r = validatePayload(
    sampleReport({
      chain: [{ type: "OSError", message: "at /home/alice/x" }],
      settings: { owner: "alice@example.org", host: "10.1.2.3" },
      environment: { gpu_model: "ghp_abcdefghijklmnopqrstuvwxyz0123456789" },
      log_tail: "reading /home/alice/plate1",
    }),
    "report",
  );
  assert.equal(r.chain[0].message, "<redacted>");
  assert.deepEqual(r.settings, { owner: "<redacted>", host: "<redacted>" });
  assert.equal(r.environment.gpu_model, "<redacted>");
  assert.equal(r.log_tail, null);
  assert.ok(!JSON.stringify(r).includes("alice"));
});

test("the issue has run, input, environment, settings and log sections, the AI summary last", () => {
  const r = validatePayload(sampleReport({ ...DIAG, screen: "Mask", module: "mask" }), "report");
  const body = issueBody(r, "a".repeat(64), { text: "Summary.", source: "client", model: null }, { cause: "no-object-channel-set", related: [7] });
  for (const heading of [
    "### Exception chain (oldest first)",
    "### Run",
    "### Input (no names)",
    "### Environment",
    "### Settings in effect",
    "### Console before the failure (scrubbed)",
    "### Same likely cause",
  ]) {
    assert.ok(body.includes(heading), heading);
    assert.ok(body.indexOf(heading) < body.indexOf("### AI summary"), heading);
  }
  assert.match(body, /\| Stage \| `preprocess_generate_masks` \|/);
  assert.match(body, /\| GPU model \| NVIDIA GeForce RTX 4090 \|/);
  assert.match(body, /\| cellpose \| 4\.2\.1 \|/);
  assert.match(body, /- First image shape: 3 × 512 × 512/);
  assert.match(body, /<details>\n<summary>6 settings<\/summary>/);
  assert.match(body, /src\s+= "<path: exists=no>"/);
  assert.match(body, /<details>\n<summary>last 2 lines<\/summary>/);
  assert.match(body, /- #7/);
});

test("a cause slug is parsed, normalised and kept out of the text users see", () => {
  assert.equal(normaliseCause("No Object-Channel SET!"), "no-object-channel-set");
  assert.equal(normaliseCause("one two three four five six seven"), "one-two-three-four-five");
  assert.equal(normaliseCause("x".repeat(60)), "x".repeat(40));
  assert.equal(normaliseCause("!!!"), null);
  assert.deepEqual(splitCause("Fix it.\n\n**CAUSE:** `input-folder-not-found`"), { text: "Fix it.", cause: "input-folder-not-found" });
  assert.deepEqual(splitCause("No key here."), { text: "No key here.", cause: null });
  assert.match(SYSTEM_PROMPT, /CAUSE: <kebab-case-slug>/);
  assert.match(SUMMARY_PROMPT, /CAUSE: <kebab-case-slug>/);
});

test("diagnose strips the CAUSE line and stores the slug with the cached explanation", async () => {
  const r = await call("/diagnose", sampleReport());
  assert.equal(r.body.advice, "**What happened** The channel is not set.");
  assert.equal(r.body.cause, "no-object-channel-set");
  const again = await call("/diagnose", sampleReport());
  assert.equal(again.body.cached, true);
  assert.equal(again.body.cause, "no-object-channel-set");
  assert.ok(!again.body.advice.includes("CAUSE"));
});

test("an issue gets the cause label, and the second signature with that cause comments on the hub", async () => {
  const first = await call("/report", sampleReport({ summarize: true }));
  assert.equal(first.status, 200);
  assert.ok(created()[0].body.labels.includes("cause:no-object-channel-set"));
  assert.ok(!created()[0].body.body.includes("CAUSE:"));
  assert.match(created()[0].body.body, /### Same likely cause/);
  assert.equal(env.DB.causes.get("no-object-channel-set").hub_issue, 101);
  assert.equal(comments().length, 0);

  const second = await call("/report", sampleReport({ summarize: true, exception_type: "KeyError" }));
  assert.equal(second.body.issue_number, 102, "a different signature stays a separate issue");
  assert.equal(second.body.deduplicated, false);
  assert.match(created()[1].body.body, /- #101/);
  const hub = comments();
  assert.equal(hub.length, 1);
  assert.match(hub[0].url, /\/issues\/101\/comments$/);
  assert.equal(hub[0].body.body, "Related report with the same likely cause: #102");
  assert.equal(env.DB.causes.get("no-object-channel-set").count, 2);
});

test("a client summary's CAUSE line labels the issue and is stripped from the body", async () => {
  await call("/report", sampleReport({ ai_summary: "Own words.\nCAUSE: cuda-out-of-memory" }));
  const body = created()[0].body;
  assert.ok(body.labels.includes("cause:cuda-out-of-memory"));
  assert.ok(body.body.includes("> Own words."));
  assert.ok(!body.body.includes("CAUSE: cuda"));
});

test("without permission to create labels the cause goes into the title", async () => {
  gh.labelStatus = 403;
  await call("/report", sampleReport({ cause: "input folder not found" }));
  const issue = created()[0].body;
  assert.match(issue.title, / · Cause: input-folder-not-found$/);
  assert.ok(!issue.labels.some((l) => l.startsWith("cause:")));
});

test("a closed hub hands over to the new issue", async () => {
  await call("/report", sampleReport({ cause: "no-object-channel-set" }));
  gh.issueStates.set(101, "closed");
  await call("/report", sampleReport({ cause: "no-object-channel-set", exception_type: "KeyError" }));
  assert.equal(comments().length, 0);
  assert.equal(env.DB.causes.get("no-object-channel-set").hub_issue, 102);
});
