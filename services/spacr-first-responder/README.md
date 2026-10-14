# spacr-first-responder

A small Cloudflare Worker that lets spaCR users without a GitHub or AI account

- send an **anonymised error report**, which the relay files as a GitHub issue on
  `EinarOlafsson/spacr` through the `spacr-first-responder` GitHub App, optionally
  with an AI summary, and
- ask for a **hosted AI diagnosis** of an error from a free Workers AI model.

Plain JavaScript ES modules, no build step, no runtime dependencies. The GitHub App
private key lives only in the Worker's secret store; nothing secret ships in spaCR.

## Routes

Every response is JSON. `GET /` answers like `/health`.

| Route | Request | 200 response |
| --- | --- | --- |
| `GET /health` | none | `{ok, version, routes}` |
| `GET /allowance?install_id=<uuid>` | none; consumes nothing | the allowance object below |
| `POST /report` | report JSON, optional `dry_run`, `ai_summary`, `summarize`, `kind` | `{issue_number, issue_url, deduplicated, commented?, recurred_from?, ai_summary_source, allowance}` |
| `POST /report` with `dry_run: true` | same | `{dry_run: true, title, body, signature, ai_summary_source, allowance}` |
| `POST /diagnose` | report JSON plus optional `question` | `{advice, model, tier: "best"\|"light", allowance}` |

The allowance object is the same on all three routes:

```json
{
  "ai": {
    "neurons_left": 4451,
    "share_neurons": 4500,
    "share_pct": 50,
    "active_users": 1,
    "active_users_definition": "installs that used hosted AI (diagnosis or report summary) today, UTC, counting you; allowance queries do not count",
    "best_left": 45,
    "light_left": 178,
    "tier_now": "best"
  },
  "best_left": 45,
  "light_left": 178,
  "reports_left": 5,
  "pool": "best",
  "pool_pct_left": 99.5,
  "resets_at": "2026-10-15T00:00:00.000Z"
}
```

- `tier_now` is `"best"`, `"light"` or `"none"`. `pool` is `"best"`, `"light"` or
  `"exhausted"`.
- `best_left` and `light_left` are estimates: remaining share divided by a typical
  answer's cost (1,000 tokens in, 400 out). They are repeated at the top level for
  clients built against 3b50b593d.
- `ai_summary_source` is `"client"`, `"relay"` or `"none"`.
- `resets_at` is the next 00:00 UTC, in ISO 8601.

Errors: `400 {error}` (validation), `413 {error}` (body over 16 KB), `404/405 {error}`,
`429 {refused: "quota", scope}` with scope `install`, `ip` or `global`,
`502 {error}` (GitHub or model failure), `503 {error}` (not configured).

### Report schema (version 1)

Unknown fields are rejected. Body at most 16 KB. Every field added since 3b50b593d
is optional, so older clients keep working.

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
| `frames` | max 40 `{file, function, line, code?}`; `file` is a base name, no path; `code` is the optional source line, single line, max 160 chars |
| `kind` | optional `"fault"` (default) or `"user_error"` (settings or usage error) |
| `user_note` | optional, max 1000 chars |
| `dry_run` | `/report` only, optional boolean |
| `ai_summary` | `/report` only, optional, max 2000 chars, written by the user's own AI |
| `summarize` | `/report` only, optional boolean: ask the relay to write the summary |
| `question` | `/diagnose` only, optional, max 2000 chars |

Free text (`message`, `user_note`, `ai_summary`, `question`, `frames[].code`) is
refused with a clear 400 if it looks like it contains an absolute path (Unix, `~/`,
Windows drive, UNC, `file://`), an e-mail address, an IPv4/IPv6 address, a URL with
credentials, or a token or key (GitHub, OpenAI-style, AWS, Slack, Hugging Face,
Google, JWT, PEM, `password=`-style pairs, long random strings). The spaCR client
must scrub before sending; the relay only double-checks.

### Issues

- **Title.** `[user-report] <exception> in <module>: <message>`. A
  `kind: "user_error"` report gets `[user-report][settings]` and the extra label
  `user-error`.
- **Traceback.** File names only. A frame's `code` is shown indented under its
  `file:line in function` line, and is also sent to the model for diagnoses and
  summaries.
- **De-duplication.**
  - Signature = SHA-256 of `exception_type`, `module` and the five innermost frames
    (`file:function`). Line numbers, `code` and `kind` are left out, so one bug keeps
    one issue across releases.
  - A new signature opens an issue labelled `user-report`.
  - A known signature adds a `+1` comment with the version/OS line, at most one
    comment per signature per UTC day; further repeats only increase the stored
    count.
  - Before a comment would be posted (so at most once per signature per UTC day),
    the relay reads the issue's state. If it is open, the `+1` comment is added as
    above. If it is closed, no comment is added: a new issue titled
    `[recurred] <title>` is opened with the normal body plus
    `Seen again after #N was closed.`, the signature is re-pointed to it (count,
    last comment day and summary flag reset) and the response has
    `deduplicated: false` and `recurred_from: N`.
  - If the issue was deleted (404/410), a new one is opened.
- **Storage.** Counters and signatures live in D1 (`schema.sql`); raw reports are
  never stored, and IP addresses are only used as a daily salted hash.

### AI summary on issues

An issue body (or a `+1` comment) can carry a section
`### AI summary (generated, may be wrong)`, followed by its source:
"summary written by the reporter's own AI" (`ai_summary` from the client) or
"summary written by spaCR AI (<model>)" (written by the relay when `summarize: true`
and there is no `ai_summary`).

The relay summary:

- uses the LIGHT model with a fixed prompt: likely cause and where in spaCR it
  happens, no speculation about the user;
- is limited to 1,200 characters, 350 output tokens and a 20 s timeout;
- is paid from the install's AI share, like a LIGHT diagnosis, and counted in the
  shared neuron pool.

The report is **never** refused because of the summary. If the share is used up,
the pool is exhausted, the model fails or it times out, the issue is filed without
a summary (`ai_summary_source: "none"`) and any reservation is refunded to the share.

A `+1` comment carries a summary only while the issue has none yet (`has_summary`
in `issue_signatures`), and only when that day's comment slot is free, so no AI is
spent on a summary that would not be posted.

## Allowances

All limits are `[vars]` in `wrangler.toml`; counters reset at 00:00 UTC.

| What | Limit | Setting |
| --- | --- | --- |
| AI neuron pool, shared by all installs | 9,000 of the free 10,000 neurons a day | `AI_DAILY_NEURON_CAP` (clamped to 10,000) |
| AI share per install | `floor(cap / (N + 1))`, at least 300, at most the cap | `AI_MIN_SHARE_NEURONS` |
| BEST model `@cf/qwen/qwen2.5-coder-32b-instruct` | while at least 40% of the pool is left and the share covers this call's worst case | `AI_MODEL_BEST`, `AI_LIGHT_THRESHOLD` |
| LIGHT model `@cf/meta/llama-3.1-8b-instruct-fp8` (diagnoses and summaries) | otherwise, while the share and the pool cover this call's worst case | `AI_MODEL_LIGHT` |
| Optional hard caps on BEST / LIGHT calls per install | 0 = off (default) | `AI_BEST_PER_INSTALL`, `AI_LIGHT_PER_INSTALL` |
| Error reports | 5 per install, 200 in total (issues and comments) | `REPORTS_PER_INSTALL`, `DAILY_REPORT_LIMIT` |
| Per hashed IP, per route (second guard) | 60 | `PER_IP_DAILY_LIMIT` |

**Active users.** N counts the installs that used hosted AI today (a diagnosis or a
relay summary), including the one asking. A `/allowance` query does not make an
install active, so idle app launches do not shrink anyone's share. It reports N as
if the caller were to use AI now.

| Active users N | Share | % of the pool | Typical BEST answers (97 neurons) | Typical LIGHT answers (25 neurons) |
| --- | --- | --- | --- | --- |
| 1 | 4,500 | 50% | 46 | 180 |
| 2 | 3,000 | 33.3% | 30 | 120 |
| 3 | 2,250 | 25% | 23 | 90 |
| 10 | 818 | 9.1% | 8 | 32 |
| 29 or more | 300 (floor) | 3.3% | 3 | 12 |

Each install's neurons used today count against its **current** share. When new
users arrive the share shrinks; an install that has already used more than the new
share gets `429 scope "install"` until the reset. Shares are a fairness rule, not a
reservation: when many installs are active, the floors can add up to more than the
pool, and the pool cap still wins (`429 scope "global"`).

The ladder for `/diagnose` and relay summaries:

1. Diagnoses only: the per-IP guard (60 a day) must pass; otherwise `429 scope "ip"`.
2. If the pool cannot fit a worst-case LIGHT call: `429 scope "global"`.
3. If at least 40% of the pool is left (3,600 neurons), the relay tries BEST. In one
   atomic statement each, it reserves this call's worst-case cost (estimated prompt
   tokens plus the full output budget) against the install's share, then against
   the pool. Summaries skip this step.
4. Otherwise, or if that fails, it tries LIGHT the same way. Below 40% everyone is
   served LIGHT.
5. If neither fits: `429 scope "install"` when the share is short, `"global"` when the
   pool is short. Every earlier reservation is refunded.
6. After the call, both reservations are settled to the real cost from the `usage`
   tokens the model returns (or an estimate from the text).
   - A failed call refunds the share and keeps the conservative pool reservation.
   - Workers AI error 4006 (free allocation used up) fills the pool to the cap and
     answers `429 scope "global"`.

Reservations are single atomic D1 statements (`INSERT ... ON CONFLICT DO UPDATE
... WHERE n + amount <= limit RETURNING n`), so concurrent requests can never push
a share or the pool over its limit.

### Neuron math

Prices in neurons per million tokens, from Cloudflare's pricing page (2026-10-14):

| Model | Input | Output | Worst case per diagnosis* | Typical (1,000 in, 400 out) |
| --- | --- | --- | --- | --- |
| `@cf/qwen/qwen2.5-coder-32b-instruct` (BEST) | 60,000 | 90,909 | 483 | 97 |
| `@cf/meta/llama-3.1-8b-instruct-fp8` (LIGHT) | 13,778 | 26,128 | 114 | 25 |

\* 16 KB body plus the system prompt counted at 2.5 characters per token (about
7,100 tokens) and the full 600 output tokens. A relay summary is at most about 108
neurons (350 output tokens). The reservation for a real call uses that call's own
prompt, so a typical report reserves about 95 (BEST) or 25 (LIGHT) neurons, and the
300-neuron floor always covers at least one BEST answer.

With the defaults, BEST is served while the pool spends its first 5,400 neurons
(about 55 typical BEST answers), then LIGHT until fewer than 114 neurons remain
(about 140 typical LIGHT answers). The 1,000-neuron margin under the free 10,000
covers estimation error; the relay never reserves past `AI_DAILY_NEURON_CAP`.

### Zero cost

The relay is built for **Workers Free, D1 Free and free Workers AI models only**.
When a free allowance runs out, requests are refused (`429 {"refused":"quota"}`) or,
for summaries, skipped; nothing is billed. Do not add a payment method, Workers Paid,
AI Gateway credits or a paid model. Free is not unlimited, and Cloudflare can change
its terms. Other free allowances: Workers 100,000 requests/day; D1 5 M rows read,
100,000 rows written per day and 5 GB.

Model choice:

- BEST `@cf/qwen/qwen2.5-coder-32b-instruct`: 32k context, tuned for code and
  tracebacks.
- LIGHT `@cf/meta/llama-3.1-8b-instruct-fp8`: the cheapest capable plain instruct
  model in the catalog (4.4 times cheaper input and 3.5 times cheaper output than
  BEST). `@cf/meta/llama-3.1-8b-instruct` no longer has a catalog page (404 on
  2026-10-14), and there is no `-fast` 8B page. `@cf/google/gemma-4-26b-a4b-it`
  (9,091 / 27,273) is cheaper still and named free by Cloudflare, but it is a
  reasoning model with an OpenAI-style response; evaluate it on real reports before
  switching.
- Neither model is on Cloudflare's list of models that need Workers Paid, which on
  2026-10-14 names kimi-k2.6, kimi-k2.7-code, glm-5.2, glm-5.3, glm-5.3-flash,
  deepseek-v4-flash-0731 and deepseek-v4-pro-0813.

The Worker only calls models on a built-in allowlist with known prices
(`MODEL_RATES` in `src/config.js`): the two defaults,
`@cf/meta/llama-3.3-70b-instruct-fp8-fast` and `@cf/meta/llama-3.2-3b-instruct`.
Any other `AI_MODEL_BEST`/`AI_MODEL_LIGHT` answers 503. Sources:

- <https://developers.cloudflare.com/workers-ai/platform/pricing/>
- <https://developers.cloudflare.com/changelog/post/2026-07-28-models-require-workers-paid/>
- <https://developers.cloudflare.com/workers-ai/models/qwen2.5-coder-32b-instruct/>
- <https://developers.cloudflare.com/workers-ai/models/llama-3.1-8b-instruct-fp8/>

Before the first deployment, check on the dashboard that **Workers & Pages -> Plans**
still shows **Free** and that neither model card is marked as needing Workers Paid.

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
APP_ID, INSTALLATION_ID, REPO, the two models and the limits). Plain variables set
only on the dashboard are replaced; secrets are kept.

### 2. Create the D1 tables

```sh
cd services/spacr-first-responder
npx wrangler login
npx wrangler d1 execute spacr-support --remote --file=schema.sql
```

Running it again is harmless. The per-install AI shares added after 3b50b593d need
no schema change; they use the existing `daily_counters` table. If you applied an
earlier copy of `schema.sql` that had no `has_summary` column, add it once:

```sh
npx wrangler d1 execute spacr-support --remote \
  --command="ALTER TABLE issue_signatures ADD COLUMN has_summary INTEGER NOT NULL DEFAULT 0"
```

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

curl -s "$URL/allowance?install_id=3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b"

curl -s -X POST $URL/report -H 'content-type: application/json' -d '{
  "schema_version": 1,
  "install_id": "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b",
  "spacr_version": "1.5.1.5", "os": "linux", "python": "3.11.9",
  "qt": "6.7.2", "torch": "2.4.1", "gpu": "cuda",
  "module": "spacr.measure", "exception_type": "ValueError",
  "message": "operands could not be broadcast together with shapes (512,512) (256,256)",
  "frames": [{"file": "measure.py", "function": "measure_crop", "line": 412}],
  "dry_run": true,
  "summarize": true
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

Expected: `/health` returns `{"ok":true,...}`; `/allowance` returns the allowance
object with `tier_now: "best"`, 5 reports left and `pool: "best"`; the dry run returns the issue title and body
it *would* file, with an "AI summary" section and `ai_summary_source: "relay"`,
without touching GitHub (it does spend a few neurons of the share and one dry-run
report);
`/diagnose` returns `{"advice": "...", "model": "@cf/qwen/qwen2.5-coder-32b-instruct",
"tier": "best", "allowance": {...}}`.

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

Configuration names: variables `APP_ID`, `INSTALLATION_ID`, `REPO`, `AI_MODEL_BEST`,
`AI_MODEL_LIGHT`, `AI_BEST_PER_INSTALL`, `AI_LIGHT_PER_INSTALL`, `AI_DAILY_NEURON_CAP`,
`AI_MIN_SHARE_NEURONS`,
`AI_LIGHT_THRESHOLD`, `REPORTS_PER_INSTALL`, `DAILY_REPORT_LIMIT`, `PER_IP_DAILY_LIMIT`,
optional `GITHUB_API_VERSION` (default `2022-11-28`); secrets `GITHUB_APP_PRIVATE_KEY`
and optional `IP_HASH_SALT`.
