import assert from "node:assert/strict";
import { test } from "node:test";

import { userPrompt, SYSTEM_PROMPT } from "../src/prompts.js";
import { issueBody, issueTitle, plusOneBody, problemLines } from "../src/report.js";
import { VERSION } from "../src/util.js";
import { MAX_ERROR_TEXT, MAX_USER_ERROR_MESSAGE, validatePayload } from "../src/validate.js";
import { sampleReport } from "./helpers.mjs";

const MESSAGE = [
  "The run did not start: 2 settings problems.",
  "- src does not exist (its parent folder does not exist either; is the drive or share mounted?)",
  "- cell_channel: no segmentation channel is set: every registered object channel is None.",
].join("\n");
const CONSOLE = "✗ The run did not start:\n  1. src: src does not exist: <PATH>\n  2. cell_channel: no segmentation channel is set";
const SUMMARY = { text: "**What happened** Two settings are wrong.", source: "relay", model: "m" };

function refused(overrides = {}) {
  return validatePayload(
    sampleReport({
      kind: "user_error",
      exception_type: "ConfigurationError",
      module: "mask",
      message: MESSAGE,
      screen: "Mask",
      error_text: CONSOLE,
      frames: [
        { file: "core.py", function: "preprocess_generate_masks", line: 424 },
        { file: "errors.py", function: "raise_if_strict", line: 820 },
      ],
      ...overrides,
    }),
    "report",
  );
}

test("the relay is version 0.2.4", () => {
  assert.equal(VERSION, "0.2.4");
});

test("screen is optional, validated and at most 60 characters", () => {
  assert.equal(refused().screen, "Mask");
  assert.equal(refused({ screen: undefined }).screen, null);
  assert.equal(refused({ screen: "Align & Stitch" }).screen, "Align & Stitch");
  assert.throws(() => refused({ screen: "x".repeat(61) }), /screen/);
  assert.throws(() => refused({ screen: "/home/a" }), /screen/);
  assert.throws(() => refused({ screen: 3 }), /screen/);
});

test("a user error may carry an 800-character problem list, a fault only 300", () => {
  const long = `The run did not start.\n- ${"a ".repeat(380)}`;
  assert.ok(long.length > 300 && long.length <= MAX_USER_ERROR_MESSAGE);
  assert.equal(refused({ message: long }).message, long);
  assert.throws(() => refused({ message: long, kind: "fault" }), /message exceeds 300/);
  assert.throws(() => refused({ message: "x".repeat(801) }), /message exceeds 800/);
});

test("error_text is optional, capped at 1500 and dropped when it holds personal data", () => {
  assert.equal(refused().error_text, CONSOLE);
  assert.equal(refused({ error_text: undefined }).error_text, null);
  assert.throws(() => refused({ error_text: "x".repeat(MAX_ERROR_TEXT + 1) }), /error_text exceeds 1500/);
  assert.equal(refused({ error_text: "failed at /home/alice/data" }).error_text, null);
});

test("the issue carries the problems, console error, call site and screen before the AI summary", () => {
  const r = refused();
  const body = issueBody(r, "f".repeat(64), SUMMARY);
  assert.match(body, /\*\*ConfigurationError\*\* in `mask` on the \*\*Mask\*\* screen/);
  assert.match(body, /### Problems \(2\)\n\n1\. src does not exist/);
  assert.match(body, /2\. cell_channel: no segmentation channel is set/);
  assert.match(body, /### Console error \(scrubbed\)\n\n```text\n✗ The run did not start:/);
  assert.match(body, /### Where the run was refused \(call site/);
  assert.match(body, /core\.py:424 in preprocess_generate_masks/);
  const ai = body.indexOf("### AI summary (generated, may be wrong)");
  assert.ok(ai > body.indexOf("### Problems"));
  assert.ok(ai > body.indexOf("### Console error"));
  assert.ok(ai > body.indexOf("### Environment"));
  assert.equal(issueTitle(r), "[user-report][settings] ConfigurationError in mask: The run did not start: 2 settings problems.");
  assert.deepEqual(problemLines(r.message).length, 2);
});

test("a +1 comment carries this occurrence's error and problems in a details block", () => {
  const comment = plusOneBody(refused(), 3, null);
  assert.match(comment, /^\+1 from spaCR /);
  assert.match(comment, /<details>\n<summary>This occurrence: The run did not start: 2 settings problems\.<\/summary>/);
  assert.match(comment, /### Problems \(2\)/);
  assert.match(comment, /### Console error \(scrubbed\)/);
  assert.match(comment, /<\/details>/);
});

test("the prompt names the screen and tells Mask from Make Masks", () => {
  assert.match(userPrompt(refused()), /^Screen: Mask$/m);
  assert.match(SYSTEM_PROMPT, /"Make Masks" is a DIFFERENT screen: the manual mask editor/);
  assert.match(SYSTEM_PROMPT, /When a Screen is given, call it by exactly that name/);
});
