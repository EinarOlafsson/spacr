import { normaliseCause } from "./causes.js";
import { HttpError } from "./util.js";

export const SCHEMA_VERSION = 1;
export const MAX_FRAMES = 40;
export const MAX_MESSAGE = 300;
export const MAX_USER_ERROR_MESSAGE = 800;
export const MAX_ERROR_TEXT = 1500;
export const MAX_SCREEN = 60;
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
const SCREEN_RE = /^[A-Za-z0-9][A-Za-z0-9 &()+.,'-]{0,59}$/;
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

function checkErrorText(body) {
  const v = body.error_text;
  if (v === undefined || v === null) return null;
  if (typeof v !== "string") throw bad("field error_text must be a string");
  if (v.length > MAX_ERROR_TEXT) throw bad(`field error_text exceeds ${MAX_ERROR_TEXT} characters`);
  if (/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/.test(v)) {
    throw bad("field error_text contains control characters");
  }
  if (!v.trim() || findSensitive(v)) return null;
  return v;
}

export const MAX_CHAIN = 5;
export const MAX_SETTINGS = 150;
export const MAX_SETTING_TEXT = 60;
export const MAX_LOG_TAIL = 3000;
export const REDACTED = "<redacted>";
const SETTING_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
const FACT_RE = /^[A-Za-z0-9 ._()+<>-]*$/;
const EXT_RE = /^[a-z0-9]{1,8}$/;
const DTYPE_RE = /^[A-Za-z0-9_.<>-]{1,16}$/;
const STAGE_RE = /^[A-Za-z0-9_.<>-]{0,60}$/;
const INSTALL_TYPES = new Set(["pip", "conda", "installer", "editable", "unknown"]);
const VERSION_KEYS = new Set(["torch", "cellpose", "numpy", "scikit_image", "pyside6", "qt"]);
const CONTROL_RE = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/;

function onlyKeys(obj, field, keys) {
  if (!isPlainObject(obj)) throw bad(`field ${field} must be an object`);
  for (const key of Object.keys(obj)) {
    if (!keys.includes(key)) throw bad(`field ${field} has unknown key: ${key.slice(0, 40)}`);
  }
}

function count(v, field, max = 10_000_000) {
  if (!Number.isInteger(v) || v < 0 || v > max) throw bad(`field ${field} must be a non-negative integer`);
  return v;
}

function fact(v, field, max) {
  if (typeof v !== "string" || v.length > max || !FACT_RE.test(v)) throw bad(`field ${field} has an invalid format`);
  return findSensitive(v) ? REDACTED : v;
}

function settingValue(v, field, depth = 0) {
  if (v === null || typeof v === "boolean") return v;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) throw bad(`field ${field} must be finite`);
    return v;
  }
  if (typeof v === "string") {
    if (v.length > MAX_SETTING_TEXT || CONTROL_RE.test(v)) throw bad(`field ${field} is too long or has control characters`);
    return findSensitive(v) ? REDACTED : v;
  }
  if (Array.isArray(v) && depth === 0) {
    if (v.length > 16) throw bad(`field ${field} has more than 16 items`);
    return v.map((x, i) => settingValue(x, `${field}[${i}]`, 1));
  }
  throw bad(`field ${field} must be a number, boolean, null, short text or a short list`);
}

function checkDiagnostics(body) {
  const out = { chain: null, settings: null, inputs: null, environment: null, run: null, log_tail: null, cause: null };
  if (body.chain !== undefined && body.chain !== null) {
    if (!Array.isArray(body.chain) || body.chain.length > MAX_CHAIN) throw bad(`field chain must be an array of at most ${MAX_CHAIN}`);
    out.chain = body.chain.map((c, i) => {
      onlyKeys(c, `chain[${i}]`, ["type", "message"]);
      if (typeof c.type !== "string" || !EXC_RE.test(c.type)) throw bad(`chain[${i}].type has an invalid format`);
      const message = typeof c.message === "string" ? c.message : "";
      if (message.length > MAX_MESSAGE || CONTROL_RE.test(message)) throw bad(`chain[${i}].message is invalid`);
      return { type: c.type, message: findSensitive(message) ? REDACTED : message };
    });
  }
  if (body.settings !== undefined && body.settings !== null) {
    if (!isPlainObject(body.settings)) throw bad("field settings must be an object");
    const keys = Object.keys(body.settings);
    if (keys.length > MAX_SETTINGS) throw bad(`field settings has more than ${MAX_SETTINGS} entries`);
    out.settings = {};
    for (const key of keys) {
      if (!SETTING_NAME_RE.test(key)) throw bad("field settings has an invalid setting name");
      out.settings[key] = settingValue(body.settings[key], `settings.${key}`);
    }
  }
  if (body.inputs !== undefined && body.inputs !== null) {
    const v = body.inputs;
    onlyKeys(v, "inputs", ["images", "channels", "extensions", "shape", "dtype"]);
    out.inputs = {};
    if (v.images !== undefined) out.inputs.images = count(v.images, "inputs.images");
    if (v.channels !== undefined) out.inputs.channels = count(v.channels, "inputs.channels", 100000);
    if (v.extensions !== undefined) {
      if (!Array.isArray(v.extensions) || v.extensions.length > 10 || !v.extensions.every((e) => typeof e === "string" && EXT_RE.test(e))) {
        throw bad("field inputs.extensions must be up to 10 short lower-case extensions");
      }
      out.inputs.extensions = v.extensions;
    }
    if (v.shape !== undefined) {
      if (!Array.isArray(v.shape) || v.shape.length > 6) throw bad("field inputs.shape must be up to 6 integers");
      out.inputs.shape = v.shape.map((n, i) => count(n, `inputs.shape[${i}]`, 10_000_000));
    }
    if (v.dtype !== undefined) {
      if (typeof v.dtype !== "string" || !DTYPE_RE.test(v.dtype)) throw bad("field inputs.dtype has an invalid format");
      out.inputs.dtype = v.dtype;
    }
  }
  if (body.environment !== undefined && body.environment !== null) {
    const v = body.environment;
    onlyKeys(v, "environment", ["os_version", "arch", "cores", "ram_gb", "gpu_model", "vram_gb", "cuda", "mps", "install", "versions"]);
    out.environment = {};
    if (v.os_version !== undefined) out.environment.os_version = fact(v.os_version, "environment.os_version", 40);
    if (v.arch !== undefined) out.environment.arch = fact(v.arch, "environment.arch", 20);
    if (v.gpu_model !== undefined) out.environment.gpu_model = fact(v.gpu_model, "environment.gpu_model", 60);
    for (const key of ["cores", "ram_gb", "vram_gb"]) {
      if (v[key] !== undefined) out.environment[key] = count(v[key], `environment.${key}`, 100000);
    }
    for (const key of ["cuda", "mps"]) {
      if (v[key] !== undefined) {
        if (typeof v[key] !== "boolean") throw bad(`field environment.${key} must be a boolean`);
        out.environment[key] = v[key];
      }
    }
    if (v.install !== undefined) {
      if (!INSTALL_TYPES.has(v.install)) throw bad("field environment.install must be pip, conda, installer, editable or unknown");
      out.environment.install = v.install;
    }
    if (v.versions !== undefined) {
      onlyKeys(v.versions, "environment.versions", [...VERSION_KEYS]);
      out.environment.versions = {};
      for (const [key, ver] of Object.entries(v.versions)) {
        if (typeof ver !== "string" || !VERSION_RE.test(ver)) throw bad(`field environment.versions.${key} has an invalid format`);
        out.environment.versions[key] = ver;
      }
    }
  }
  if (body.run !== undefined && body.run !== null) {
    const v = body.run;
    onlyKeys(v, "run", ["stage", "elapsed_s", "dry_run"]);
    out.run = {};
    if (v.stage !== undefined) {
      if (typeof v.stage !== "string" || !STAGE_RE.test(v.stage)) throw bad("field run.stage has an invalid format");
      out.run.stage = v.stage;
    }
    if (v.elapsed_s !== undefined) {
      if (typeof v.elapsed_s !== "number" || !Number.isFinite(v.elapsed_s) || v.elapsed_s < 0 || v.elapsed_s > 1e8) {
        throw bad("field run.elapsed_s must be a non-negative number");
      }
      out.run.elapsed_s = v.elapsed_s;
    }
    if (v.dry_run !== undefined) {
      if (typeof v.dry_run !== "boolean") throw bad("field run.dry_run must be a boolean");
      out.run.dry_run = v.dry_run;
    }
  }
  if (body.log_tail !== undefined && body.log_tail !== null) {
    const v = body.log_tail;
    if (typeof v !== "string") throw bad("field log_tail must be a string");
    if (v.length > MAX_LOG_TAIL) throw bad(`field log_tail exceeds ${MAX_LOG_TAIL} characters`);
    if (CONTROL_RE.test(v)) throw bad("field log_tail contains control characters");
    out.log_tail = v.trim() && !findSensitive(v) ? v : null;
  }
  if (body.cause !== undefined && body.cause !== null) {
    if (typeof body.cause !== "string" || body.cause.length > 80) throw bad("field cause must be a short slug");
    out.cause = normaliseCause(body.cause);
  }
  return out;
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
  "screen",
  "error_text",
];

const DIAGNOSTIC_FIELDS = ["chain", "settings", "inputs", "environment", "run", "log_tail", "cause"];
const REPORT_FIELDS = new Set([...COMMON_FIELDS, ...DIAGNOSTIC_FIELDS, "dry_run", "ai_summary", "summarize"]);
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
  if (body.kind !== undefined && body.kind !== null && !KINDS.includes(body.kind)) {
    throw bad("field kind must be fault or user_error");
  }
  out.kind = body.kind === "user_error" ? "user_error" : "fault";
  const maxMessage = out.kind === "user_error" ? MAX_USER_ERROR_MESSAGE : MAX_MESSAGE;
  out.message = checkText(body, "message", maxMessage, { optional: false });
  out.frames = checkFrames(body);
  out.user_note = checkText(body, "user_note", MAX_NOTE);
  out.screen = checkString(body, "screen", SCREEN_RE, { optional: true, nullable: true });
  if (out.screen !== null && out.screen.length > MAX_SCREEN) throw bad(`field screen exceeds ${MAX_SCREEN} characters`);
  out.error_text = checkErrorText(body);
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
    Object.assign(out, checkDiagnostics(body));
  }
  return out;
}
