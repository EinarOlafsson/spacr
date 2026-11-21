# Item spec: unknown keys in a settings file are reported, and optionally refused

- **Lifecycle:** future (alpha), behind "Show alpha features" for the GUI notice (rule 41). The CLI and Python behaviour is a warning only, so default runs are unchanged; the refusal is opt-in.
- **Number:** 1002 (1000 and 1001 are taken on the lane-1000 and codex-tr-1000-1001 branches; 996-999 on the spec branches). Check `features/feature_list/` and the claims files before landing.
- **Rule:** a settings-file rerun must stay IDENTICAL: the resolved dict handed to the pipeline is byte-for-byte what it is today in default mode. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly 04ec46d370.
- **Paper-critical:** no, but it protects every paper run: a pinned settings file whose key the tree does not read looks valid and is not.
- **Readiness:** no pipeline step is added or changed, so no row in `proposals/READINESS.md`.
- **Schema:** neutral. No new row in `spacr.settings` defaults, no `SETTINGS_SCHEMA_VERSION` change, no entry in `DEFAULT_CHANGES`. The strict switch is a control key consumed by the resolver and removed before the pipeline sees the dict.
- **Size estimate:** S, about 0.5 day. About 90 lines of code (a shared finder of about 40, the resolver wiring about 25, the CLI flag and GUI notice about 25) and about 140 lines of tests. Small enough to ride in train 7 if no other item touches `cli.resolve_settings`.

## Why
`spacr.cli.resolve_settings` layers defaults, the settings file and `--set`. Only `--set` is checked (`apply_overrides` raises `SettingsError` on a key spaCR does not know, with a difflib suggestion). A key that arrives from the FILE is merged by `resolved.update(...)` and never compared with anything.

Concrete case, found 2026-11-20 on nightly 04ec46d370: a train 6/7 settings file carried `ops_ph_tile_batch`, `ops_ph_image_format` and `ops_crop_image_format`. That tree does not read them. The run started, wrote its outputs under the default values and printed no warning. Nothing in the output showed that three requested settings had been dropped.

A partial check already exists but does not reach this path: `spacr.validate._check_unknown_keys` warns on unknown keys, with a typo suggestion at a 0.85 cutoff, but only inside `validate_settings`, which runs in the pre-flight. It is skipped by `--no-preflight`, absent from `resolve_settings` itself, and never called by `spacr.cli_remote`, the distributed-jobs screen, `spacr.batch.resolve_job_settings` or the GUI settings load. It also stays silent for a blank value, and for a key beginning with `_`.

The Funk et al. 1120 run scripts (`final_funk_runs_1120/scripts/run_step.py`, `validate_settings.py`) already refuse unknown keys themselves, with a private list (`spacr_settings_version`, `ops_phases`). This item moves that behaviour into spaCR so the scripts, the CLI and the GUI share one rule and one allowlist.

## Behaviour
1. **Shared finder.** One function, `spacr.settings.unknown_setting_keys(settings, app='') -> list[tuple[str, str | None]]`, returns `(key, nearest_known_key_or_None)` for every key of the file that is not known. "Known" is exactly the set `spacr.validate._check_unknown_keys` uses today (`_known_setting_keys()`, the app's extra keys, `_module_default_keys(app)`, `model_metadata`) plus `expected_types`, minus nothing else. A key that is retired, renamed (`surviving_setting_name`) or withdrawn is not unknown; the existing migration notice speaks for it. A key on the allowlist (below) is not unknown. The nearest key is `difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)`, the same call and cutoff as `apply_overrides`. `validate._check_unknown_keys` becomes a thin wrapper over this finder (its 0.85 typo cutoff and its blank-value rule are kept there so existing validate tests pass), so there is one definition of "known".
2. **Default: warn.** `resolve_settings` calls the finder on the keys that came from the file (before `--set`, which keeps its own refusal). Each unknown key produces one line through `tr()`, for example `Unknown setting '{key}' ignored; nearest known setting: '{nearest}'.` and, without a near match, `Unknown setting '{key}' ignored.` One line per key, naming the key and the nearest key, never one merged line. The value is not changed or dropped from the dict: the resolved dict is identical to today's.
3. **Where the line goes.** When the caller passes the existing `moved` list, the notices are appended to it as `spacr.validate.Problem` WARNING objects, so `--dry-run` and `validate` print them once in their own report. Without `moved` they go to `LOG.warning`, as `_under_todays_names` does, so a plain `resolve_settings` call in a script is no longer silent. `cmd_run` with `--no-preflight` therefore still warns (it already logs the moved notices in that case).
4. **Strict: refuse.** When strict is on, the same finder raises one `SettingsError` that lists every unknown key with its nearest key, before the run starts. Exit code is the usage code, as for any `SettingsError`. Nothing is run and nothing is written.
5. **Strict applies to the file only.** `--set` already refuses unknown keys in every mode.
6. **No change to what the pipeline receives.** The control key `settings_strict` is removed from the resolved dict after use, so it is never written to the run's saved settings, never compared by the schema ratchets, and never appears in `--describe`.

## Settings and flags
| Name | Where | Default | Meaning |
|---|---|---|---|
| `--strict-settings` | `spacr-run <module>` and `spacr-run validate` | off | refuse unknown keys instead of warning |
| `--no-strict-settings` | same | - | force the warning mode even if the file says `settings_strict: true` |
| `settings_strict` | settings file (control key, not a schema row) | false | same as the flag; on the allowlist as a control key |

Precedence, highest first: `--no-strict-settings` / `--strict-settings`, `--set settings_strict=true`, the file's `settings_strict`, default off. Python callers pass `strict: Optional[bool] = None` to `resolve_settings`, with the same precedence (an explicit argument wins). The key is parsed with `coerce_value` as a bool, so `true`, `True` and `1` behave as in every other boolean setting.

## Allowlist (record-only keys)
A record-only key documents or pins a run (provenance, a run note, a tool's private option) and is read by no pipeline. It must not warn, and must not be refused in strict mode.

- **Single source of truth:** `spacr.settings.RECORD_ONLY_SETTINGS` (exact names, a frozenset) and `spacr.settings.RECORD_ONLY_PREFIXES` (a tuple of prefixes). Nothing else lists them; `validate`, the CLI, the GUI notice and the run scripts import these two names.
- **Initial exact names:** `spacr_settings_version` (already handled by `_settings_file_version`), `ops_phases` (from `run_step.py`), `spatial_tile_batch`, `settings_strict`, and the 965 keys `seg_cells_from_nucleus_seeds`, `seg_merge_small_nuclei_frac`, `seg_tile_overlap_px`, `seg_tile_px`, `seg_diameter_sites`.
- **Initial prefixes:** `_comment`, `_meta` (so `_comment`, `_comment_ops`, `_meta`, `_meta_run` are all accepted).
- **Retirement rule:** a name leaves the list when it becomes a real setting. The 965 `seg_*` keys leave when item 965 lands them in `expected_types`; a ratchet test (T9) fails if a listed name is also a known setting, so the list cannot go stale.
- A comment or meta value that is a nested object is left alone. The finder never looks inside values.
- The scripts `run_step.py` and `validate_settings.py` (final_funk_runs_1120) replace their private list by an import of these two names when they next change. That edit is advisory and is not part of this item's tree.

## GUI behaviour (alpha)
- Loading a settings file (settings screens, distributed-jobs screen) never refuses. With "Show alpha features" on, a non-modal notice in the existing message strip lists the unknown keys, one row per key with the nearest key beside it (`'{key}' -> '{nearest}'`, or just `'{key}'`). The keys stay in the file and are not applied to any control.
- With the alpha switch off, the GUI keeps today's silent behaviour; the CLI and Python warnings are not alpha.
- No new control, no new Preferences row, no schema row. A "Remove unknown keys" button is explicitly out of scope.
- All new strings go through `tr()` and ship with the 9 locale records.

## Callers to wire
- `spacr.cli.resolve_settings` (core change); `cmd_run`, `cmd_validate`: add the flag, pass `strict=` and the existing `moved` list.
- `spacr.cli_remote` (`resolve_settings(module, args.settings, args.set or [])`): inherits the warning; passes the flag through.
- `spacr.qt.screens.distributed_jobs` (job settings resolution): inherits the warning; no strict.
- `spacr.batch.resolve_job_settings`: inherits the warning (it does not pass `moved`, so it logs).
- `spacr.cli.cmd_validate` without `--module`: uses `load_settings_file` and `apply_overrides` only; it calls the finder with `app=''` (only the generic known set, so only the typo-like unknown keys are reported).
- `spacr.ops_decode.resolve_settings` is a different function (preset resolver) and is untouched. `run_ops` and `ops_engine` receive a dict, not a file; they are not changed. A Python caller that wants the check calls `unknown_setting_keys` directly.

## Validation and acceptance
- (a) Default mode: for every settings file in the existing CLI tests, the resolved dict equals today's byte for byte.
- (b) The 1120 case: a file with `ops_ph_tile_batch`, `ops_ph_image_format`, `ops_crop_image_format` on a tree that does not read them warns three times, once per key, in default mode, and refuses with one error naming all three in strict mode.
- (c) Strict off by default; no schema ratchet moves.

## Tests (CPU, synthetic, under 1 s each)
- **T1** default mode: a file with one unknown key resolves to a dict equal to the same file without it plus that key; one warning is logged naming the key.
- **T2** nearest key: a typo `cell_diameter_x` for a real key reports that real key; a key with no match (`zzz_not_a_setting`) reports no nearest key and no `Did you mean`.
- **T3** one line per key: three unknown keys give three notices, none merged.
- **T4** strict via `strict=True`, via `--strict-settings`, via `--set settings_strict=true` and via `settings_strict: true` in the file: each raises `SettingsError` listing every unknown key; the exit code is the usage code.
- **T5** precedence: the file says strict, `--no-strict-settings` warns instead; `strict=False` beats the file.
- **T6** allowlist: `spatial_tile_batch`, each 965 `seg_*` key, `_comment`, `_comment_ops`, `_meta`, `_meta_run`, `ops_phases`, `spacr_settings_version` give no warning and no strict refusal; a near miss such as `seg_tile_pxx` or `meta_run` does.
- **T7** `settings_strict` is absent from the resolved dict and from the settings the run saves.
- **T8** retired and renamed keys are not reported as unknown; the existing migration notice is unchanged; with a `moved` list the unknown notices land in it as WARNING problems and nothing is logged twice (`validate` prints each key once).
- **T9** ratchet: no name in `RECORD_ONLY_SETTINGS` is in `expected_types` or the module defaults, and `validate._check_unknown_keys` and the finder agree on the known set for every module.
- **T10** the GUI notice (alpha on) lists the rows with nearest keys, does not change the loaded controls, and is absent with alpha off; every new `tr()` string has its 9 locale records.
- **T11** `--set` still refuses an unknown key with its existing message (no regression).

## Arrival contract
New documented callables: `spacr.settings.unknown_setting_keys`; the constants `RECORD_ONLY_SETTINGS` and `RECORD_ONLY_PREFIXES`; the `strict` argument of `spacr.cli.resolve_settings`. Each lands in the same commit with its arrival-contract record and its 9-locale records (the notice strings, the two flag help texts, the GUI row text). Catalogue symbols, help-index entry and consumer-map row ship with it; no catch-up agent. Run the hash-comment, raw-reader, nested-function and documented-callable ratchets before landing.

## Known limits (reported, not blockers)
- "Unknown" means unknown to the module being run plus the generic set. A key valid for another module of the same app family is reported, which is the intended point (it is ignored by this run).
- A key that is real but unread (`DEAD_SETTINGS`) is a separate existing check and is not duplicated here.
- The nearest-key hint is lexical; it can name a wrong key for a short unrelated name. It is a hint, not a rename.
