# ops-fields-store English source handoff

Source branch: `ws-offload-ws-ops-fields-store` (items 0aq, 0ar, s4b: OPS phenotype fields in fov.ome.zarr through the shared image writer; the OPS crop index, and the Measure crop index, in the crop_index table of the one store of the project, `measurements/spacr.duckdb`; value evidence for the dtypes of the OPS phenotype tables).
Manifest: `tr_ops-fields-store_sources.json` (task identity `ops-fields-store`).

Translations are DROPPED until the end (maintainer, 2026-11-23 15:10): the lander records this file in trains/TRANSLATION_DEBT.md and lands without a translation pin.

## Translation assignment (when translations resume)

Prepare source-bound reviewed records for sv, de, es, zh_CN, pt, hi, ko, is and fr, through the normal scoped API and runtime loaders, from a fresh spacr-translations main. Keep code, RST roles, placeholders (`{store}`, `{problems}`), file and table names (`fov.ome.zarr`, `crop_index`, `measurements/`, `spacr.duckdb`, `rows.jsonl`) exact. Reviewed-record task only: no catalog builds, source changes or pins.

## Public API scope

Revised docstrings (2): the `spacr.crop_store` module and `spacr.image_store.verify_store` (new keyword `names`). No public name added or removed. Bodies are in the manifest.

## Runtime scope

One new runtime source in `spacr/ops_engine.py` (`_ph_store_fields`): "{store}: the phenotype fields written do not match their checksums: {problems}".

## Settings

No setting added, removed or reworded.
