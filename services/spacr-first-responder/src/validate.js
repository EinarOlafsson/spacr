import { HttpError } from "./util.js";

export const SCHEMA_VERSION = 1;
export const MAX_FRAMES = 40;
export const MAX_MESSAGE = 300;
export const MAX_NOTE = 1000;
export const MAX_QUESTION = 2000;
export const MAX_AI_SUMMARY = 2000;
export const MAX_CODE = 160;
export const KINDS = Object.freeze(["fault", "user_error"]);

const OS_VALUES = new Set(["linux", "windows", "macos"]);

export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const VERSION_RE = /^[0-9A-Za-z][0-9A-Za-z.+_-]{0,39}$/;
const GPU_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const MODULE_RE = /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,79}$/;
const EXC_RE = /^[A-Za-z_][A-Za-z0-9_.]{0,119}$/;
const FILE_RE = /^(?:<[A-Za-z0-9_ -]{1,40}>|[A-Za-z0-9_][A-Za-z0-9_.-]{0,99})$/;
const FUNC_RE = /^(?:<[A-Za-z0-9_ -]{1,40}>|[A-Za-z_][A-Za-z0-9_.]{0,119})$/;

const SENSITIVE_PATTERNS = [
  ["an absolute path", /(?:^|[^A-Za-z0-9_.~:\/-])\/[^\s/'"]+\/[^\s'"]*/],
  ["an absolute path", /(?:^|[^A-Za-z0-9_.:\/-])\/(?:home|Users|root|mnt|media|Volumes|tmp|var|opt|srv|data|scratch|private|net|nas[A-Za-z0-9_]*)\b/i],
  ["an absolute path", /(?:^|[^A-Za-z0-9_])~[\\/]/],
  ["an absolute path", /(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]/],
  ["an absolute path", /\\\\[^\s\\]+\\/],
  ["an absolute path", /\bfile:\/\//i],
  ["an e-mail address", /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}/],
  ["a URL with credentials", /:\/\/[^\s/@]+@/],
  ["an IP address", /(?<![\w.])(?<![vV])(?<!version )(?<!spacr )(?<!spaCR )(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])/],
  ["an IP address", /(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}(?![\w:])/],
  ["an IP address", /(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4})?(?![\w:])/],
  ["a token or key", /-----BEGIN [A-Z ]*(?:PRIVATE KEY|CERTIFICATE)/],
  ["a token or key", /\b(?:gh[opsur]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})/],
  ["a token or key", /\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}/],
  ["a token or key", /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/],
  ["a token or key", /\bxox[abeoprs]-[A-Za-z0-9-]{10,}/],
  ["a token or key", /\bhf_[A-Za-z0-9]{20,}/],
  ["a token or key", /\bAIza[0-9A-Za-z_-]{30,}/],
  ["a token or key", /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/],
  ["a token or key", /\b(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|auth)\s*[:=]\s*\S+/i],
  ["a token or key", /\bBearer\s+[A-Za-z0-9._~+/-]{12,}/i],
];

const LONG_RANDOM_RE = /[A-Za-z0-9_+=-]{32,}/g;

function looksRandom(run) {
  const digits = (run.match(/[0-9]/g) || []).length;
  return /[A-Za-z]/.test(run) && digits >= 4;
}

export function findSensitive(text) {
  if (typeof text !== "string" || text.length === 0) return null;
  for (const [label, re] of SENSITIVE_PATTERNS) {
    if (re.test(text)) return label;
  }
  const runs = text.match(LONG_RANDOM_RE) || [];
  for (const run of runs) {
    if (looksRandom(run)) return "a token or key";
  }
  return null;
}

function bad(message) {
  return new HttpError(400, message);
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function checkString(body, field, re, { optional = false, nullable = false } = {}) {
  const v = body[field];
  if (v === undefined) {
    if (optional) return null;
    throw bad(`missing required field: ${field}`);
  }
  if (v === null) {
    if (nullable) return null;
    throw bad(`field ${field} must not be null`);
  }
  if (typeof v !== "string") throw bad(`field ${field} must be a string`);
  if (!re.test(v)) throw bad(`field ${field} has an invalid format`);
  return v;
}

function checkText(body, field, maxLen, { optional = true } = {}) {
  const v = body[field];
  if (v === undefined || v === null) {
    if (optional) return null;
    throw bad(`missing required field: ${field}`);
  }
  if (typeof v !== "string") throw bad(`field ${field} must be a string`);
  if (v.length > maxLen) throw bad(`field ${field} exceeds ${maxLen} characters`);
  if (/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/.test(v)) {
    throw bad(`field ${field} contains control characters`);
  }
  const hit = findSensitive(v);
  if (hit) throw bad(`field ${field} looks like it contains ${hit}; remove it and retry`);
  return v;
}

function checkFrames(body) {
  const frames = body.frames;
  if (frames === undefined) throw bad("missing required field: frames");
  if (!Array.isArray(frames)) throw bad("field frames must be an array");
  if (frames.length > MAX_FRAMES) throw bad(`field frames has more than ${MAX_FRAMES} entries`);
  return frames.map((f, i) => {
    if (!isPlainObject(f)) throw bad(`frames[${i}] must be an object`);
    for (const key of Object.keys(f)) {
      if (!["file", "function", "line", "code"].includes(key)) {
        throw bad(`frames[${i}] has unknown field: ${key}`);
      }
    }
    if (typeof f.file !== "string") throw bad(`frames[${i}].file must be a string`);
    if (/[\\/:]/.test(f.file)) {
      throw bad(`frames[${i}].file must be a file name without a path`);
    }
    if (!FILE_RE.test(f.file)) throw bad(`frames[${i}].file has an invalid format`);
    if (typeof f.function !== "string" || !FUNC_RE.test(f.function)) {
      throw bad(`frames[${i}].function has an invalid format`);
    }
    if (!Number.isInteger(f.line) || f.line < 0 || f.line > 10_000_000) {
      throw bad(`frames[${i}].line must be a non-negative integer`);
    }
    const frame = { file: f.file, function: f.function, line: f.line };
    if (f.code !== undefined && f.code !== null) {
      if (typeof f.code !== "string") throw bad(`frames[${i}].code must be a string`);
      if (f.code.length > MAX_CODE) throw bad(`frames[${i}].code exceeds ${MAX_CODE} characters`);
      if (/[\u0000-\u0008\u000A-\u001F\u007F]/.test(f.code)) {
        throw bad(`frames[${i}].code must be a single line without control characters`);
      }
      const hit = findSensitive(f.code);
      if (hit) throw bad(`frames[${i}].code looks like it contains ${hit}; remove it and retry`);
      if (f.code.trim()) frame.code = f.code;
    }
    return frame;
  });
}

const COMMON_FIELDS = [
  "schema_version",
  "install_id",
  "spacr_version",
  "os",
  "python",
  "qt",
  "torch",
  "gpu",
  "module",
  "exception_type",
  "message",
  "frames",
  "user_note",
  "kind",
];

const REPORT_FIELDS = new Set([...COMMON_FIELDS, "dry_run", "ai_summary", "summarize"]);
const DIAGNOSE_FIELDS = new Set([...COMMON_FIELDS, "question"]);

export function validatePayload(body, kind) {
  if (!isPlainObject(body)) throw bad("request body must be a JSON object");
  const allowed = kind === "diagnose" ? DIAGNOSE_FIELDS : REPORT_FIELDS;
  for (const key of Object.keys(body)) {
    if (!allowed.has(key)) throw bad(`unknown field: ${key.slice(0, 40)}`);
  }
  if (body.schema_version !== SCHEMA_VERSION) {
    throw bad(`field schema_version must be ${SCHEMA_VERSION}`);
  }
  const out = { schema_version: SCHEMA_VERSION };
  out.install_id = checkString(body, "install_id", UUID_RE).toLowerCase();
  out.spacr_version = checkString(body, "spacr_version", VERSION_RE);
  if (typeof body.os !== "string" || !OS_VALUES.has(body.os)) {
    throw bad("field os must be one of linux, windows, macos");
  }
  out.os = body.os;
  out.python = checkString(body, "python", VERSION_RE);
  out.qt = checkString(body, "qt", VERSION_RE, { optional: true, nullable: true });
  out.torch = checkString(body, "torch", VERSION_RE, { optional: true, nullable: true });
  out.gpu = checkString(body, "gpu", GPU_RE, { optional: true, nullable: true });
  out.module = checkString(body, "module", MODULE_RE);
  out.exception_type = checkString(body, "exception_type", EXC_RE);
  out.message = checkText(body, "message", MAX_MESSAGE, { optional: false });
  out.frames = checkFrames(body);
  out.user_note = checkText(body, "user_note", MAX_NOTE);
  if (body.kind !== undefined && body.kind !== null && !KINDS.includes(body.kind)) {
    throw bad("field kind must be fault or user_error");
  }
  out.kind = body.kind === "user_error" ? "user_error" : "fault";
  if (kind === "diagnose") {
    out.question = checkText(body, "question", MAX_QUESTION);
  } else {
    if (body.dry_run !== undefined && typeof body.dry_run !== "boolean") {
      throw bad("field dry_run must be a boolean");
    }
    out.dry_run = body.dry_run === true;
    if (body.summarize !== undefined && typeof body.summarize !== "boolean") {
      throw bad("field summarize must be a boolean");
    }
    out.summarize = body.summarize === true;
    out.ai_summary = checkText(body, "ai_summary", MAX_AI_SUMMARY);
    if (out.ai_summary !== null && out.ai_summary.trim() === "") out.ai_summary = null;
  }
  return out;
}
