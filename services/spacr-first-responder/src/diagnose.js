import { environmentLine, tracebackText } from "./report.js";
import { HttpError } from "./util.js";

export const MAX_OUTPUT_TOKENS = 600;
export const MAX_ADVICE_CHARS = 6000;

export const FREE_MODELS = Object.freeze([
  "@cf/qwen/qwen2.5-coder-32b-instruct",
  "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "@cf/qwen/qwen3-30b-a3b-fp8",
  "@cf/meta/llama-3.1-8b-instruct",
  "@cf/meta/llama-3.1-8b-instruct-fp8",
]);

export const DEFAULT_MODEL = "@cf/qwen/qwen2.5-coder-32b-instruct";

export const SYSTEM_PROMPT = [
  "You are the spaCR assistant. spaCR is an open-source Python desktop application for",
  "microscopy image analysis and pooled CRISPR screens: it segments cells, nuclei and",
  "pathogens (Cellpose models), measures phenotypes, trains classifiers, and maps",
  "sequencing reads to guide RNAs. It runs on Windows, macOS and Linux with Python, Qt",
  "and PyTorch.",
  "",
  "A user sends an anonymised error report: exception type, a short message, the spaCR",
  "module, and a traceback with file names only. Help them recover:",
  "1. Give the most likely causes, most likely first, each in one or two sentences.",
  "2. Give concrete next steps the user can try inside spaCR or their environment",
  "   (settings to check, inputs to verify, package versions, GPU/CPU fallbacks).",
  "3. Say when the report looks like a spaCR bug that the maintainers should fix.",
  "Never claim certainty; say what you are unsure about. Do not invent spaCR settings,",
  "functions or files you cannot see. Do not ask for personal data, paths or",
  "credentials. Keep the answer under 300 words and use short Markdown lists.",
].join("\n");

export function resolveModel(env) {
  const model = env.AI_MODEL || DEFAULT_MODEL;
  if (!FREE_MODELS.includes(model)) {
    console.log("diagnose: configured model is not on the free allowlist");
    throw new HttpError(503, "diagnosis is not available");
  }
  return model;
}

export function userPrompt(r) {
  const parts = [
    `Environment: ${environmentLine(r)}`,
    `Module: ${r.module}`,
    `Exception: ${r.exception_type}: ${r.message || ""}`,
    "Traceback (most recent call last):",
    tracebackText(r.frames),
  ];
  if (r.user_note) parts.push("", `User note: ${r.user_note}`);
  if (r.question) parts.push("", `User question: ${r.question}`);
  return parts.join("\n");
}

export function isQuotaError(err) {
  const text = String((err && (err.message || err)) || "");
  return /\b4006\b|neuron|allocation|quota|rate limit|exceeded/i.test(text);
}

export async function runDiagnosis(env, r) {
  const model = resolveModel(env);
  let result;
  try {
    result = await env.AI.run(model, {
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt(r) },
      ],
      max_tokens: MAX_OUTPUT_TOKENS,
      temperature: 0.3,
    });
  } catch (err) {
    if (isQuotaError(err)) return { refused: true };
    console.log("diagnose: model call failed");
    throw new HttpError(502, "diagnosis is unavailable; try again later");
  }
  let advice = result && typeof result.response === "string" ? result.response : "";
  advice = advice.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  if (!advice) throw new HttpError(502, "diagnosis returned no answer; try again later");
  if (advice.length > MAX_ADVICE_CHARS) advice = `${advice.slice(0, MAX_ADVICE_CHARS)}…`;
  return { advice, model };
}
