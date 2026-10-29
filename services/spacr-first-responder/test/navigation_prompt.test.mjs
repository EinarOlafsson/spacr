import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import catalog from "../src/navigation_catalog.json" with { type: "json" };
import { SYSTEM_PROMPT } from "../src/prompts.js";
import { MAX_PROMPT_TOKENS, CHARS_PER_TOKEN_FLOOR } from "../src/neurons.js";
import { MAX_BODY_BYTES } from "../src/util.js";

const resources = new URL("../../../spacr/resources/", import.meta.url);
const workflows = JSON.parse(readFileSync(new URL("module_workflows.json", resources))).modules;
const tutorials = JSON.parse(readFileSync(new URL("tutorial_index.json", resources))).catalog;
const apiBase = "https://einarolafsson.github.io/spacr/api/";
const tutorialBase = "https://einarolafsson.github.io/spacr/tutorials/#lesson=";

test("relay navigation contains real unique routes and only published help destinations", () => {
  assert.equal(catalog.schema, 1);
  assert.ok(catalog.modules.length >= 50);
  assert.equal(new Set(catalog.modules.map(m => m.key)).size, catalog.modules.length);
  for (const module of catalog.modules) {
    assert.match(module.key, /^[a-z0-9_]+$/);
    assert.ok(module.name);
    if (module.api) {
      assert.ok(module.api.startsWith(apiBase));
      const suffix = module.api.slice(apiBase.length);
      assert.match(suffix, /^spacr\/[a-z0-9_/]+\/index\.html$/);
      const path = suffix.replace(/\/index\.html$/, "").replaceAll("/", ".");
      const file = new URL(`../../../${path.replaceAll(".", "/")}.py`, import.meta.url);
      const packageFile = new URL(`../../../${path.replaceAll(".", "/")}/__init__.py`, import.meta.url);
      assert.ok(readable(file) || readable(packageFile), `${module.key} API module exists`);
    }
    if (module.tutorial) {
      assert.ok(module.tutorial.startsWith(tutorialBase));
      const lesson = module.tutorial.slice(tutorialBase.length);
      assert.ok(tutorials.some(row => row.lesson === lesson && row.app_key === module.key), module.key);
    }
    assert.deepEqual(module.inputs, workflows[module.key]?.inputs || []);
    assert.deepEqual(module.outputs, workflows[module.key]?.outputs || []);
    assert.ok(SYSTEM_PROMPT.includes(`${module.key} | ${module.name} |`));
  }
  assert.ok(catalog.modules.some(m => m.key === "mask" && m.preview === "Live preview"));
  assert.ok(catalog.modules.some(m => m.key === "measure" && m.preview === "Crop preview"));
  assert.equal(catalog.modules.find(m => m.key === "3d").tutorial, "");
});

function readable(url) {
  try { readFileSync(url); return true; } catch { return false; }
}

test("question guidance gives grounded module navigation, preview and the correct batch handoff", () => {
  assert.match(SYSTEM_PROMPT, /reply in the user's language/);
  assert.match(SYSTEM_PROMPT, /numbered GUI steps/);
  assert.match(SYSTEM_PROMPT, /Navigator keyword: type it and press Enter/);
  assert.match(SYSTEM_PROMPT, /\[Mask\]\(spacr:\/\/module\/mask\)/);
  assert.match(SYSTEM_PROMPT, /Keep all links\n.*out of the main answer text/);
  assert.match(SYSTEM_PROMPT, /once in first-mention order/);
  assert.match(SYSTEM_PROMPT, /Tools footer is required/);
  assert.match(SYSTEM_PROMPT, /\[API\]\(exact API URL\) \| \[Tutorial\]/);
  assert.match(SYSTEM_PROMPT, /same catalog row supplies them; omit missing links/);
  assert.match(SYSTEM_PROMPT, /Propagate settings ON, then use Actions > Run/);
  assert.match(SYSTEM_PROMPT, /Run preview does not process the plate/);
  assert.match(SYSTEM_PROMPT, /Measure on the generated merged project/);
  assert.match(SYSTEM_PROMPT, /do not send every workflow to Measure/);
  assert.match(SYSTEM_PROMPT, /never runs an analysis or changes settings/);
  assert.doesNotMatch(SYSTEM_PROMPT, /the only link you|never invent other page URLs/);
});

test("catalog grounding preserves error/privacy scope and fits the actual quota reservation", () => {
  for (const heading of ["What happened", "Likely cause", "How to fix it", "Is this a spaCR bug?"]) {
    assert.ok(SYSTEM_PROMPT.includes(`**${heading}**`));
  }
  assert.match(SYSTEM_PROMPT, /CAUSE: <kebab-case-slug>/);
  assert.match(SYSTEM_PROMPT, /Never ask for personal data, paths or credentials/);
  assert.match(SYSTEM_PROMPT, /Under 250 words/);
  assert.ok(SYSTEM_PROMPT.length < 16000, "bounded compact reference context");
  assert.equal(MAX_PROMPT_TOKENS,
    Math.ceil((MAX_BODY_BYTES + SYSTEM_PROMPT.length + 400) / CHARS_PER_TOKEN_FLOOR));
});
