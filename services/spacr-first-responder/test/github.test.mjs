import assert from "node:assert/strict";
import { test } from "node:test";

import { importPrivateKey, pkcs1ToPkcs8, signAppJwt } from "../src/github.js";
import { generateTestKeyPem } from "./helpers.mjs";

function b64urlDecode(s) {
  return Buffer.from(s.replace(/-/g, "+").replace(/_/g, "/"), "base64");
}

test("RS256 app JWT verifies with the matching public key", async () => {
  const { pem, publicKey } = await generateTestKeyPem();
  const now = 1_790_000_000;
  const jwt = await signAppJwt(pem, "5315672", now);
  const [h, p, s] = jwt.split(".");
  assert.deepEqual(JSON.parse(b64urlDecode(h)), { alg: "RS256", typ: "JWT" });
  const payload = JSON.parse(b64urlDecode(p));
  assert.equal(payload.iss, "5315672");
  assert.equal(payload.iat, now - 60);
  assert.ok(payload.exp - now <= 600 && payload.exp > now);
  const ok = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    publicKey,
    b64urlDecode(s),
    new TextEncoder().encode(`${h}.${p}`),
  );
  assert.equal(ok, true);
  const bad = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    publicKey,
    b64urlDecode(s),
    new TextEncoder().encode(`${h}.${p}x`),
  );
  assert.equal(bad, false);
});

test("PKCS#1 PEM from GitHub is wrapped to PKCS#8 and signs identically", async () => {
  const { pem, der, publicKey } = await generateTestKeyPem();
  const pkcs1 = der.slice(26);
  assert.deepEqual([...pkcs1ToPkcs8(pkcs1)], [...der]);
  const b64 = Buffer.from(pkcs1).toString("base64").match(/.{1,64}/g).join("\n");
  const rsaPem = `-----BEGIN RSA PRIVATE KEY-----\n${b64}\n-----END RSA PRIVATE KEY-----\n`;
  await importPrivateKey(rsaPem);
  const jwt = await signAppJwt(rsaPem, "1", 1_790_000_000);
  const [h, p, s] = jwt.split(".");
  const ok = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", publicKey, b64urlDecode(s), new TextEncoder().encode(`${h}.${p}`));
  assert.equal(ok, true);
  assert.equal(jwt, await signAppJwt(pem, "1", 1_790_000_000));
});

test("a non-key secret is refused as not configured", async () => {
  await assert.rejects(importPrivateKey("hello"), /not configured/);
});
