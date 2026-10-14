import assert from "node:assert/strict";
import { test } from "node:test";

import { HttpError } from "../src/util.js";
import { findSensitive, validatePayload } from "../src/validate.js";
import { sampleReport } from "./helpers.mjs";

function rejects(body, kind, pattern) {
  assert.throws(
    () => validatePayload(body, kind),
    (err) => err instanceof HttpError && err.status === 400 && pattern.test(err.message),
  );
}

test("accepts a well-formed report", () => {
  const out = validatePayload(sampleReport({ user_note: "Happened after loading plate 3 twice.", dry_run: true }), "report");
  assert.equal(out.exception_type, "ValueError");
  assert.equal(out.frames.length, 5);
  assert.equal(out.dry_run, true);
});

test("accepts nullable optional environment fields", () => {
  const out = validatePayload(sampleReport({ qt: null, torch: null, gpu: null }), "report");
  assert.equal(out.torch, null);
});

test("accepts a diagnose payload with a question", () => {
  const out = validatePayload(sampleReport({ question: "Why do my masks not match the image size?" }), "diagnose");
  assert.match(out.question, /masks/);
});

test("rejects unknown fields", () => {
  rejects(sampleReport({ hostname: "lab-pc" }), "report", /unknown field: hostname/);
  rejects(sampleReport({ question: "x" }), "report", /unknown field: question/);
  rejects(sampleReport({ dry_run: true }), "diagnose", /unknown field: dry_run/);
});

test("rejects bad schema version, uuid and os", () => {
  rejects(sampleReport({ schema_version: 2 }), "report", /schema_version/);
  rejects(sampleReport({ install_id: "not-a-uuid" }), "report", /install_id/);
  rejects(sampleReport({ os: "solaris" }), "report", /os must be/);
  rejects(sampleReport({ install_id: undefined }), "report", /missing required field: install_id/);
});

test("rejects frames with paths, extra keys or too many entries", () => {
  rejects(sampleReport({ frames: [{ file: "/home/alice/spacr/core.py", function: "f", line: 1 }] }), "report", /without a path/);
  rejects(sampleReport({ frames: [{ file: "C:\\Users\\bob\\core.py", function: "f", line: 1 }] }), "report", /without a path/);
  rejects(sampleReport({ frames: [{ file: "core.py", function: "f", line: 1, locals: {} }] }), "report", /unknown field: locals/);
  rejects(sampleReport({ frames: [{ file: "core.py", function: "f", line: -1 }] }), "report", /line/);
  const many = Array.from({ length: 41 }, (_, i) => ({ file: "a.py", function: "f", line: i }));
  rejects(sampleReport({ frames: many }), "report", /more than 40/);
});

test("rejects over-long text fields", () => {
  rejects(sampleReport({ message: "x".repeat(301) }), "report", /exceeds 300/);
  rejects(sampleReport({ user_note: "x".repeat(1001) }), "report", /exceeds 1000/);
  rejects(sampleReport({ question: "x".repeat(2001) }), "diagnose", /exceeds 2000/);
});

test("rejects personal data and secrets in free text", () => {
  const cases = [
    ["FileNotFoundError: /home/alice/plates/p1.tif", /absolute path/],
    ["cannot open /mnt/data/screen", /absolute path/],
    ["cannot open C:\\Users\\Bob\\x.tif", /absolute path/],
    ["see ~/spacr/settings.csv", /absolute path/],
    ["share \\\\labnas\\screens\\x", /absolute path/],
    ["contact alice@example.org", /e-mail/],
    ["connection to 192.168.1.20 refused", /IP address/],
    ["bind fe80:0:0:0:1ff:fe23:4567:890a failed", /IP address/],
    ["token ghp_abcdefghijklmnopqrstuvwxyz0123456789", /token or key/],
    ["key sk-proj-abcdefghijklmnop1234", /token or key/],
    ["AKIAABCDEFGHIJKLMNOP leaked", /token or key/],
    ["hf_abcdefghijklmnopqrstuvwxyz1234 used", /token or key/],
    ["auth eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiIxMjMifQ.sig", /token or key/],
    ["password=hunter2", /token or key/],
    ["value Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MGFiY2RlZg== here", /token or key/],
    ["-----BEGIN RSA PRIVATE KEY-----", /token or key/],
  ];
  for (const [text, re] of cases) {
    rejects(sampleReport({ message: text.slice(0, 300) }), "report", re);
    rejects(sampleReport({ user_note: text }), "report", re);
  }
});

test("does not flag ordinary error text", () => {
  const ok = [
    "operands could not be broadcast together with shapes (512,512) (256,256)",
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "upgraded to spaCR 1.5.1.5 and version 1.5.1.4 worked",
    "see https://github.com/EinarOlafsson/spacr/issues/12",
    "ratio 3/4 and/or channel 1",
    "KeyError: 'cell_mask_dim'",
  ];
  for (const text of ok) assert.equal(findSensitive(text), null, text);
});
