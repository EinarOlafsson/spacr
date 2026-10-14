import { HttpError } from "./util.js";

export const GITHUB_API = "https://api.github.com";
export const USER_AGENT = "spacr-first-responder";
export const DEFAULT_API_VERSION = "2022-11-28";

let tokenCache = null;

export function resetTokenCache() {
  tokenCache = null;
}

function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlJson(obj) {
  return b64url(new TextEncoder().encode(JSON.stringify(obj)));
}

function pemBody(pem) {
  const b64 = pem
    .replace(/-----BEGIN [A-Z ]+-----/g, "")
    .replace(/-----END [A-Z ]+-----/g, "")
    .replace(/\\n/g, "")
    .replace(/\s+/g, "");
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function derLength(n) {
  if (n < 0x80) return [n];
  const bytes = [];
  while (n > 0) {
    bytes.unshift(n & 0xff);
    n >>= 8;
  }
  return [0x80 | bytes.length, ...bytes];
}

function derTag(tag, content) {
  return Uint8Array.from([tag, ...derLength(content.length), ...content]);
}

export function pkcs1ToPkcs8(pkcs1) {
  const version = Uint8Array.from([0x02, 0x01, 0x00]);
  const algorithm = Uint8Array.from([
    0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05, 0x00,
  ]);
  const key = derTag(0x04, pkcs1);
  return derTag(0x30, Uint8Array.from([...version, ...algorithm, ...key]));
}

export async function importPrivateKey(pem) {
  if (typeof pem !== "string" || !pem.includes("PRIVATE KEY")) {
    throw new HttpError(503, "relay is not configured");
  }
  let der = pemBody(pem);
  if (/BEGIN RSA PRIVATE KEY/.test(pem)) der = pkcs1ToPkcs8(der);
  return crypto.subtle.importKey(
    "pkcs8",
    der,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

export async function signAppJwt(pem, appId, nowSeconds = Math.floor(Date.now() / 1000)) {
  const key = await importPrivateKey(pem);
  const header = b64urlJson({ alg: "RS256", typ: "JWT" });
  const payload = b64urlJson({ iat: nowSeconds - 60, exp: nowSeconds + 540, iss: String(appId) });
  const input = `${header}.${payload}`;
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(input));
  return `${input}.${b64url(new Uint8Array(sig))}`;
}

function headers(env, auth) {
  return {
    accept: "application/vnd.github+json",
    authorization: `Bearer ${auth}`,
    "user-agent": USER_AGENT,
    "x-github-api-version": env.GITHUB_API_VERSION || DEFAULT_API_VERSION,
    "content-type": "application/json",
  };
}

export async function installationToken(env, nowMs = Date.now()) {
  if (tokenCache && tokenCache.expiresAt - 5 * 60 * 1000 > nowMs) return tokenCache.token;
  if (!env.APP_ID || !env.INSTALLATION_ID || !env.GITHUB_APP_PRIVATE_KEY) {
    throw new HttpError(503, "relay is not configured");
  }
  let jwt;
  try {
    jwt = await signAppJwt(env.GITHUB_APP_PRIVATE_KEY, env.APP_ID, Math.floor(nowMs / 1000));
  } catch (err) {
    if (err instanceof HttpError) throw err;
    console.log("github: private key could not be imported");
    throw new HttpError(503, "relay is not configured");
  }
  const repoName = String(env.REPO || "").split("/")[1];
  const res = await fetch(`${GITHUB_API}/app/installations/${env.INSTALLATION_ID}/access_tokens`, {
    method: "POST",
    headers: headers(env, jwt),
    body: JSON.stringify({ repositories: [repoName], permissions: { issues: "write" } }),
  });
  if (!res.ok) {
    console.log(`github: installation token request failed with status ${res.status}`);
    throw new HttpError(502, "GitHub is unavailable; try again later");
  }
  const data = await res.json();
  const expiresAt = Date.parse(data.expires_at) || nowMs + 50 * 60 * 1000;
  tokenCache = { token: data.token, expiresAt };
  return data.token;
}

export async function githubRequest(env, method, path, body) {
  const token = await installationToken(env);
  const res = await fetch(`${GITHUB_API}/repos/${env.REPO}${path}`, {
    method,
    headers: headers(env, token),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (res.status === 401) resetTokenCache();
  return res;
}

export async function createIssue(env, title, body, labels) {
  const res = await githubRequest(env, "POST", "/issues", { title, body, labels });
  if (!res.ok) {
    console.log(`github: create issue failed with status ${res.status}`);
    throw new HttpError(502, "GitHub is unavailable; try again later");
  }
  const data = await res.json();
  return { number: data.number, url: data.html_url };
}

export async function createComment(env, number, body) {
  const res = await githubRequest(env, "POST", `/issues/${number}/comments`, { body });
  if (res.status === 404 || res.status === 410) return { gone: true };
  if (!res.ok) {
    console.log(`github: comment failed with status ${res.status}`);
    throw new HttpError(502, "GitHub is unavailable; try again later");
  }
  return { gone: false };
}
