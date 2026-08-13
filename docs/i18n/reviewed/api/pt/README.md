# Portuguese API review evidence

These schema-1 files preserve Codex-assisted reviewed candidates for the
Portuguese API hard tail discovered on 2026-08-12. They are evidence inputs,
not shipping catalogs, and no generator reads them automatically. Promotion
must remain accepted-only and bind each record to its exact canonical source
and reviewed contextual source.

- Source revision during the original review: `74a5b6ae8e47ccd1eba65ba00ac323bad3433d3c`.
- Revalidated without record drift against the 6,357-document extraction at
  `20b5ed52` after the unrelated merge-cardinality source work landed.
- Originating forensic artifact: `/tmp/spacr-pt-v2-forensic.json`, SHA-256
  `8557f72b13217c917cddf4fa19a4861b9bd7b7c6f7774f76faee81db5a112946`.
- Intended cache contract after explicit admission: `api-block-v7`.
- Review method: Codex-assisted translation followed by exact label/source/
  context alignment, source-hash verification, `_syntax_preserved`, canonical
  `_api_block_valid`, and contextual `_api_block_valid`.
- Result: the original 60/60 records passed every listed check. They are now
  admitted only through the exact source/context-bound reviewed path; a stale
  source, context, hash, or target fails closed.

| Records | File | SHA-256 |
|---:|---|---|
| 0–19 | `2026-08-12-tail-000-019.json` | `3331b081f0dca9be03b4725943582c060c37dadfb999f5ed29edcc3069a6133e` |
| 20–39 | `2026-08-12-tail-020-039.json` | `7bb1ada2a938362b5d48465fbc005482b59f91ee85c9db00d3c4d77837f05e5f` |
| 40–59 | `2026-08-12-tail-040-059.json` | `5b918fa17499a83eb53d88216f9b3e70de91b3ab5ca1f2750f76aed9da202068` |

The next fresh strict run recovered all 60 original records and reduced the
Portuguese hard tail to 46 unique blocks. All 46 current-tail blocks were then
reviewed against the 6,357-record corpus and preserved separately:

| Fresh-tail records | File | SHA-256 |
|---:|---|---|
| 0–15 | `2026-08-12-tail-current-000-015.json` | `a09ae523ee90de832c8bf10cfb5640f80a3efe1766ab1cf493fc09e88208431b` |
| 16–30 | `2026-08-12-tail-current-016-030.json` | `7a3d474b45eca47703cd10294ecf8715f027955bfa482b409dd8543aa95db8e7` |
| 31–45 | `2026-08-12-tail-current-031-045.json` | `d53a6e4bf530b945b79a56f59cd75d91c5fdc952ee32e63737880c2982a88e52` |

All 46 pass the same source-hash, syntax, canonical, and contextual gates.
The subsequent strict generator run admitted all 46 with zero model decoding
and verified the complete 6,357-symbol Portuguese API catalog.

The old-v2 indices 60–122 are historical only. The fresh strict audit is
authoritative and the Portuguese API tail is now closed; runtime localization
remains a separate release gate.
