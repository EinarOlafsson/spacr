# ops-store-only English source handoff

Source branch: `ws-offload-ws-ops-store-only` (item 0an: every OPS table in the run's spaCR store; ops_storage store-only; ph_plate profile).
Manifest: `tr_ops-store-only_sources.json` (task identity `ops-store-only`).

Translations are DROPPED until the end (maintainer, 2026-11-23 15:10): the lander records this file in trains/TRANSLATION_DEBT.md and lands without a translation pin.

## Translation assignment (when translations resume)

Prepare source-bound reviewed records for sv, de, es, zh_CN, pt, hi, ko, is and fr, through the normal scoped API and runtime loaders, from a fresh spacr-translations main. Keep code, RST roles, placeholders, table names (`ops_geometry`, `measurements/`, `spacr.duckdb`) and setting values (`spacr_store`, `current`) exact. Reviewed-record task only: no catalog builds, source changes or pins.

## Public API scope

Revised docstrings (5): the `spacr.ops_store` module, `spacr.ops_store.write_table`, `read_table`, `row_count`, `objects_ready`. No public name added or removed. Bodies are in the manifest.

## Runtime scope

One runtime source replaced in `spacr/ops_engine.py` (`run_ops`): "the phenotype phases need dst_root, the folder holding the run's spaCR store (measurements/)" (was "... the folder holding measurements.db").

## Setting tooltip revised (1)

`ops_storage`: the store is the only target; 'current' is a retired value that runs as 'spacr_store'; default 'spacr_store'. The combo now offers only 'spacr_store'.
