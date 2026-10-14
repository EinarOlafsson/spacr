import { webcrypto } from "node:crypto";

import { SQL } from "../src/sql.js";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

export class FakeD1 {
  constructor() {
    this.counters = new Map();
    this.signatures = new Map();
    this.log = [];
  }

  prepare(sql) {
    const db = this;
    return {
      bind(...args) {
        return {
          first: async () => db.exec(sql, args, "first"),
          run: async () => db.exec(sql, args, "run"),
          all: async () => db.exec(sql, args, "all"),
        };
      },
    };
  }

  exec(sql, args, mode) {
    this.log.push(sql);
    const rows = this.apply(sql, args);
    if (mode === "first") return rows[0] ?? null;
    if (mode === "all") return { results: rows, success: true };
    return { success: true, meta: { changes: rows.length } };
  }

  apply(sql, a) {
    switch (sql) {
      case SQL.reserve: {
        const key = `${a[0]}|${a[1]}|${a[2]}`;
        const cur = this.counters.get(key);
        if (cur === undefined) {
          this.counters.set(key, a[3]);
          return [{ n: a[3] }];
        }
        if (cur + a[3] <= a[4]) {
          this.counters.set(key, cur + a[3]);
          return [{ n: cur + a[3] }];
        }
        return [];
      }
      case SQL.adjust: {
        const key = `${a[0]}|${a[1]}|${a[2]}`;
        if (this.counters.has(key)) this.counters.set(key, Math.max(0, this.counters.get(key) + a[3]));
        return [];
      }
      case SQL.raiseTo: {
        const key = `${a[0]}|${a[1]}|${a[2]}`;
        this.counters.set(key, Math.max(this.counters.get(key) ?? a[3], a[3]));
        return [];
      }
      case SQL.read: {
        const key = `${a[0]}|${a[1]}|${a[2]}`;
        return this.counters.has(key) ? [{ n: this.counters.get(key) }] : [];
      }
      case SQL.countRoute: {
        let c = 0;
        for (const key of this.counters.keys()) {
          const [day, route] = key.split("|");
          if (day === a[0] && route === a[1]) c++;
        }
        return [{ c }];
      }
      case SQL.prune: {
        for (const key of [...this.counters.keys()]) {
          if (key.split("|")[0] < a[0]) this.counters.delete(key);
        }
        return [];
      }
      case SQL.bumpSignature: {
        const row = this.signatures.get(a[0]);
        if (!row) return [];
        row.count += 1;
        row.last_seen = a[1];
        return [{ issue_number: row.issue_number, count: row.count, last_comment_day: row.last_comment_day, has_summary: row.has_summary }];
      }
      case SQL.claimSignature: {
        if (this.signatures.has(a[0])) return [];
        this.signatures.set(a[0], {
          signature: a[0],
          issue_number: 0,
          count: 1,
          first_seen: a[1],
          last_seen: a[1],
          last_comment_day: null,
          has_summary: 0,
        });
        return [{ signature: a[0] }];
      }
      case SQL.setIssueNumber: {
        const row = this.signatures.get(a[0]);
        if (row) {
          row.issue_number = a[1];
          row.has_summary = a[2];
        }
        return [];
      }
      case SQL.claimSummary: {
        const row = this.signatures.get(a[0]);
        if (!row || row.has_summary) return [];
        row.has_summary = 1;
        return [{ signature: a[0] }];
      }
      case SQL.unclaimSummary: {
        const row = this.signatures.get(a[0]);
        if (row) row.has_summary = 0;
        return [];
      }
      case SQL.dropSignature: {
        this.signatures.delete(a[0]);
        return [];
      }
      case SQL.claimDailyComment: {
        const row = this.signatures.get(a[0]);
        if (!row || !(row.issue_number > 0) || row.last_comment_day === a[1]) return [];
        row.last_comment_day = a[1];
        return [{ issue_number: row.issue_number, count: row.count, has_summary: row.has_summary }];
      }
      default:
        throw new Error(`FakeD1: unexpected SQL ${sql}`);
    }
  }
}

export function sampleReport(overrides = {}) {
  return {
    schema_version: 1,
    install_id: "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b",
    spacr_version: "1.5.1.5",
    os: "linux",
    python: "3.11.9",
    qt: "6.7.2",
    torch: "2.4.1",
    gpu: "cuda",
    module: "spacr.measure",
    exception_type: "ValueError",
    message: "operands could not be broadcast together with shapes (512,512) (256,256)",
    frames: [
      { file: "gui.py", function: "run", line: 120 },
      { file: "core.py", function: "preprocess_generate_masks", line: 88 },
      { file: "measure.py", function: "measure_crop", line: 412 },
      { file: "measure.py", function: "_measure_intensity", line: 233 },
      { file: "_methods.py", function: "_mean", line: 118 },
    ],
    ...overrides,
  };
}

export function makeRequest(path, { method = "POST", body, headers = {} } = {}) {
  const init = { method, headers: { "content-type": "application/json", ...headers } };
  if (body !== undefined) init.body = typeof body === "string" ? body : JSON.stringify(body);
  return new Request(`https://spacr-first-responder.example.workers.dev${path}`, init);
}

export async function generateTestKeyPem() {
  const pair = await crypto.subtle.generateKey(
    {
      name: "RSASSA-PKCS1-v1_5",
      modulusLength: 2048,
      publicExponent: new Uint8Array([1, 0, 1]),
      hash: "SHA-256",
    },
    true,
    ["sign", "verify"],
  );
  const der = new Uint8Array(await crypto.subtle.exportKey("pkcs8", pair.privateKey));
  const b64 = Buffer.from(der).toString("base64").match(/.{1,64}/g).join("\n");
  return {
    pem: `-----BEGIN PRIVATE KEY-----\n${b64}\n-----END PRIVATE KEY-----\n`,
    der,
    publicKey: pair.publicKey,
  };
}

export class FakeGitHub {
  constructor() {
    this.calls = [];
    this.nextIssue = 101;
    this.commentStatus = 201;
    this.issueStatus = 201;
  }

  install() {
    this.original = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => this.handle(String(url), init);
    return this;
  }

  restore() {
    globalThis.fetch = this.original;
  }

  async handle(url, init) {
    const body = init.body ? JSON.parse(init.body) : null;
    this.calls.push({ url, method: init.method, headers: init.headers, body });
    const reply = (status, payload) =>
      new Response(JSON.stringify(payload), { status, headers: { "content-type": "application/json" } });
    if (/\/app\/installations\/\d+\/access_tokens$/.test(url)) {
      return reply(201, { token: "ghs_testtoken", expires_at: new Date(Date.now() + 3600e3).toISOString() });
    }
    if (/\/repos\/[^/]+\/[^/]+\/issues$/.test(url)) {
      if (this.issueStatus >= 300) return reply(this.issueStatus, { message: "fail" });
      const number = this.nextIssue++;
      return reply(201, { number, html_url: `https://github.com/EinarOlafsson/spacr/issues/${number}` });
    }
    if (/\/issues\/\d+\/comments$/.test(url)) {
      return reply(this.commentStatus, { id: 1 });
    }
    return reply(404, { message: "Not Found" });
  }

  count(re) {
    return this.calls.filter((c) => re.test(c.url)).length;
  }
}
