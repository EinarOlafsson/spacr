# spacr-first-responder

A small Cloudflare Worker that lets spaCR users without a GitHub or AI account

- send an **anonymised error report**, which the relay files as a GitHub issue on
  `EinarOlafsson/spacr` through the `spacr-first-responder` GitHub App, and
- ask for a **hosted AI diagnosis** of an error from a free Workers AI model.

Plain JavaScript ES modules, no build step, no runtime dependencies. The GitHub App
private key lives only in the Worker's secret store; nothing secret ships in spaCR.

## Routes

| Route | Request | Success | Errors |
| --- | --- | --- | --- |
| `GET /health` | none | `{ok, version, routes}` | |
| `POST /report` | report JSON (below), optional `dry_run: true` | `{issue_number, issue_url, deduplicated}`; dry run: `{dry_run: true, title, body, signature}` | `400 {error}`, `413 {error}`, `429 {refused: "quota"}`, `502/503 {error}` |
| `POST /diagnose` | report JSON plus optional `question` | `{advice, model}` | same as `/report` |

`GET /` answers like `/health`. Every response is JSON.

### Report schema (version 1)

Unknown fields are rejected. Body at most 16 KB.

| Field | Rule |
| --- | --- |
| `schema_version` | `1` |
| `install_id` | random UUID (a fairness hint, not authentication) |
| `spacr_version`, `python` | version string, max 40 chars |
| `qt`, `torch` | version string or `null` |
| `os` | `linux`, `windows` or `macos` |
| `gpu` | short lower-case class such as `cuda`, `mps`, `cpu`, or `null` |
| `module` | spaCR module/app key, e.g. `spacr.measure` |
| `exception_type` | e.g. `ValueError` |
| `message` | scrubbed message, max 300 chars |
| `frames` | max 40 `{file, function, line}`; `file` is a base name, no path |
| `user_note` | optional, max 1000 chars |
| `dry_run` | `/report` only, optional boolean |
| `question` | `/diagnose` only, optional, max 2000 chars |

Free text (`message`, `user_note`, `question`) is refused with a clear 400 if it looks
like it contains an absolute path (Unix, `~/`, Windows drive, UNC, `file://`), an
e-mail address, an IPv4/IPv6 address, a URL with credentials, or a token or key
(GitHub, OpenAI-style, AWS, Slack, Hugging Face, Google, JWT, PEM, `password=`-style
pairs, long random strings). The spaCR client must scrub before sending; the relay
only double-checks.

### De-duplication

Signature = SHA-256 of `exception_type`, `module` and the five innermost frames
(`file:function`; line numbers are left out so one bug keeps one issue across
releases). A new signature opens an issue labelled `user-report`. A known signature
adds a `+1` comment with the version/OS line, at most one comment per signature per
UTC day; further repeats only increase the stored count. If the issue was deleted,
a new one is opened. Counters and signatures live in D1 (`schema.sql`); raw reports
are never stored, and IP addresses are only used as a daily salted hash.

## Zero cost

The relay is built for **Workers Free, D1 Free and free Workers AI models only**.
When a free allowance runs out, requests fail or are refused (`429 {"refused":"quota"}`);
nothing is billed. Do not add a payment method, Workers Paid, AI Gateway credits or
a paid model. Free is not unlimited, and Cloudflare can change its terms.

| Limit | Free allowance (Cloudflare) | Relay setting (`wrangler.toml`) |
| --- | --- | --- |
| Worker requests | 100,000/day | |
| D1 | 5 M rows read, 100,000 rows written/day, 5 GB | about 5 writes per report |
| Workers AI | 10,000 neurons/day per account, reset 00:00 UTC | `DAILY_AI_LIMIT = 30` diagnoses/day |
| Reports | | `DAILY_REPORT_LIMIT = 200` issues+comments/day |
| Per install, per route | | `PER_INSTALL_DAILY_LIMIT = 10` |
| Per IP (hashed), per route | | `PER_IP_DAILY_LIMIT = 20` |

### AI model

`AI_MODEL = "@cf/qwen/qwen2.5-coder-32b-instruct"` (32k context; tuned for code and
tracebacks). It is not on Cloudflare's list of models that need Workers Paid, which on
2026-10-14 names kimi-k2.6, kimi-k2.7-code, glm-5.2, glm-5.3, glm-5.3-flash,
deepseek-v4-flash-0731 and deepseek-v4-pro-0813. Its price is 60,000 neurons per
million input tokens and 90,909 per million output tokens. With `max_tokens = 600`
and the 16 KB input cap, one diagnosis costs at most about 310 neurons (typically about
90), so 30 a day stay under 10,000 even in the worst case.

The Worker only calls models on a built-in allowlist (`src/diagnose.js`). The other
allowed models are `@cf/meta/llama-3.3-70b-instruct-fp8-fast` (output costs 2.25 times
as much, so lower `DAILY_AI_LIMIT` to 15), `@cf/qwen/qwen3-30b-a3b-fp8` and
`@cf/meta/llama-3.1-8b-instruct`. Sources:
<https://developers.cloudflare.com/workers-ai/platform/pricing/>,
<https://developers.cloudflare.com/changelog/post/2026-07-28-models-require-workers-paid/>,
<https://developers.cloudflare.com/workers-ai/models/qwen2.5-coder-32b-instruct/>.

Before the first deployment, check on the dashboard that **Workers & Pages -> Plans**
still shows **Free** and that the model card is not marked as needing Workers Paid.

## Owner setup (exact steps)

You need: the Cloudflare account that owns the `spacr-first-responder` Worker,
the GitHub App private key file you downloaded from GitHub, and a terminal with
Node.js 22 or newer (required by Wrangler 4) for the terminal steps.

### 1. Deploy the code

Pick **one** route.

**A. Dashboard (Git build).** Cloudflare dashboard -> **Workers & Pages** ->
**spacr-first-responder** -> **Settings** -> **Build** -> **Connect** a Git repository:

- Repository: `EinarOlafsson/spacr`
- Production branch: `main`
- Root directory: `services/spacr-first-responder`
- Build command: leave empty
- Deploy command: `npx wrangler deploy`
- Build variables and secrets: leave empty (the private key is a runtime secret, step 3)

Save. Each push to `main` that reaches this folder redeploys the Worker
(Workers Builds Free includes a monthly allowance of build minutes).

**B. Terminal.**

```sh
cd services/spacr-first-responder
npx wrangler login
npx wrangler deploy
```

`wrangler deploy` applies the bindings and variables in `wrangler.toml` (AI, DB,
APP_ID, INSTALLATION_ID, REPO, AI_MODEL and the limits). Secrets are kept.

### 2. Create the D1 tables

```sh
cd services/spacr-first-responder
npx wrangler login
npx wrangler d1 execute spacr-support --remote --file=schema.sql
```

Running it again is harmless.

### 3. Store the GitHub App private key as a secret

GitHub downloads the key as PKCS#1 (`-----BEGIN RSA PRIVATE KEY-----`). WebCrypto
needs PKCS#8 (`-----BEGIN PRIVATE KEY-----`). Convert it locally; never use an
online converter:

```sh
openssl pkcs8 -topk8 -nocrypt \
  -in spacr-first-responder.2026-10-14.private-key.pem \
  -out spacr-first-responder.pkcs8.pem
head -1 spacr-first-responder.pkcs8.pem   # must print -----BEGIN PRIVATE KEY-----
```

Upload it from the file, so it never appears in your shell history or a chat:

```sh
npx wrangler secret put GITHUB_APP_PRIVATE_KEY < spacr-first-responder.pkcs8.pem
```

Or on the dashboard: **Workers & Pages** -> **spacr-first-responder** -> **Settings** ->
**Variables and Secrets** -> **Add** -> Type **Secret**, name `GITHUB_APP_PRIVATE_KEY`,
paste the whole PKCS#8 file including the BEGIN/END lines -> **Deploy**.

(The relay also accepts the original PKCS#1 file and converts it in memory, but PKCS#8
is the documented format.) Delete the converted copy afterwards, or keep it with the
original in a password manager.

Optional, recommended: a salt for the hashed IP counters.

```sh
openssl rand -hex 32 | npx wrangler secret put IP_HASH_SALT
```

### 4. Smoke test

```sh
URL=https://spacr-first-responder.einar-olafsson.workers.dev

curl -s $URL/health

curl -s -X POST $URL/report -H 'content-type: application/json' -d '{
  "schema_version": 1,
  "install_id": "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b",
  "spacr_version": "1.5.1.5", "os": "linux", "python": "3.11.9",
  "qt": "6.7.2", "torch": "2.4.1", "gpu": "cuda",
  "module": "spacr.measure", "exception_type": "ValueError",
  "message": "operands could not be broadcast together with shapes (512,512) (256,256)",
  "frames": [{"file": "measure.py", "function": "measure_crop", "line": 412}],
  "dry_run": true
}'

curl -s -X POST $URL/diagnose -H 'content-type: application/json' -d '{
  "schema_version": 1,
  "install_id": "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b",
  "spacr_version": "1.5.1.5", "os": "linux", "python": "3.11.9",
  "qt": "6.7.2", "torch": "2.4.1", "gpu": "cuda",
  "module": "spacr.measure", "exception_type": "ValueError",
  "message": "operands could not be broadcast together with shapes (512,512) (256,256)",
  "frames": [{"file": "measure.py", "function": "measure_crop", "line": 412}],
  "question": "What should I check first?"
}'
```

Expected: `/health` returns `{"ok":true,...}`; the dry run returns the issue title
and body it *would* file, without touching GitHub; `/diagnose` returns
`{"advice": "...", "model": "@cf/qwen/qwen2.5-coder-32b-instruct"}`.

Dropping `"dry_run": true` files a real public issue. Do that once, deliberately, to
confirm the App can create issues and apply the `user-report` label (create the
label in the repository first if you want its colour), then send the same report
again to see the `+1` comment.

## Development

```sh
cd services/spacr-first-responder
node --test test/
```

Tests use an in-memory D1 fake, a mocked `fetch` for GitHub and a mocked `env.AI`;
they need no network or account. `schema.sql` is the source of truth for the tables;
`src/sql.js` holds every statement the Worker runs.

Configuration names: variables `APP_ID`, `INSTALLATION_ID`, `REPO`, `AI_MODEL`,
`DAILY_AI_LIMIT`, `DAILY_REPORT_LIMIT`, `PER_INSTALL_DAILY_LIMIT`, `PER_IP_DAILY_LIMIT`,
optional `GITHUB_API_VERSION` (default `2022-11-28`); secrets `GITHUB_APP_PRIVATE_KEY`
and optional `IP_HASH_SALT`.
