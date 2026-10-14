export const VERSION = "0.2.1";

export const MAX_BODY_BYTES = 16 * 1024;

export class HttpError extends Error {
  constructor(status, message, extra) {
    super(message);
    this.status = status;
    this.extra = extra || null;
  }
}

export function json(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
  });
}

export function refusedQuota(scope) {
  return json(429, { refused: "quota", scope });
}

export async function readJsonBody(request, maxBytes = MAX_BODY_BYTES) {
  const declared = request.headers.get("content-length");
  if (declared !== null && Number(declared) > maxBytes) {
    throw new HttpError(413, `request body exceeds ${maxBytes} bytes`);
  }
  if (!request.body) {
    throw new HttpError(400, "request body is required");
  }
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      try {
        await reader.cancel();
      } catch (_) {
        /* ignore */
      }
      throw new HttpError(413, `request body exceeds ${maxBytes} bytes`);
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (_) {
    throw new HttpError(400, "request body is not valid UTF-8");
  }
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new HttpError(400, "request body is not valid JSON");
  }
}

export async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function utcDay(now = new Date()) {
  return now.toISOString().slice(0, 10);
}

export function intVar(env, name, fallback) {
  const raw = env && env[name];
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}
