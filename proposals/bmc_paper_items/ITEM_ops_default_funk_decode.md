# Item 946: the default OPS decode becomes the proven Funk decode

- **Number:** 946 (assigned by master 2026-11-14).
- **Lifecycle:** feature. It takes items 873 (rule (a) + S2 preset) and 939 (per-well auto threshold) out of alpha.
- **Maintainer decision:** 2026-11-14, "make rule_a_s2 the default".
- **Found by:** the stage-2 readiness check of 2026-11-14 (read-only, nightly 83d2cab88b). Stage 2's criteria (a)–(f) pass with the production script decode. spaCR's shipped defaults (`ops_decode_preset 'none'`, `ops_decode_threshold_mode 'manual'`, single-read quality 0.03, pixel quality 0.10) are the pre-873 decode, which is not a proven setting.
- **Order:** build after item 942 lands. Auto mode uses 942's shared precision + recall tuner.

## Spec
1. **New defaults:**
   - `ops_decode_preset 'rule_a_s2'`;
   - `ops_decode_threshold_mode 'auto'` (per-well tuning on decoys only, the maintainer's 2026-11-13 stage-2 rule).
2. **Schema:**
   - bump `SETTINGS_SCHEMA_VERSION` to the next free number at merge time (read max(DEFAULT_CHANGES) on nightly; never hard-code it);
   - add one DEFAULT_CHANGES row per changed default, so older settings files rerun with 'none' / 'manual'.
3. **Alpha:**
   - remove 873 and 939 from the alpha registry and from the alpha deck registry (`tools/refresh_alpha_deck.py`) in the implementation commit;
   - update the alpha-count checks.
4. **Translations:** update the tooltips and help text of the changed settings, with 9-locale records in the same commit.

## Proof check (pre-set)
- **Reproduction:**
  - decode plates A–D from the raw tiles through `run_ops` (spaCR only, shipped defaults);
  - the per-cell calls equal the production script calls (rule (a) + S2, per-well decoy thresholds): identical integer and categorical outputs;
  - per-well FDR ≤ 0.1 % as in S2 (f).
- **Identity:** the 939 per-well thresholds equal the script's on all 23 wells of A–D (the 21 not yet checked plus A A1 and D A1).
- **Back-compatibility:** a pre-946 settings file reruns byte-identically to its original output on 2 FOVs.
- **Speed:** time per FOV is not slower than the 932 result (same machine, same FOVs).
- **On pass:** stage 2 can be proposed as Ready in proposals/READINESS.md.
