# phtables-store English source handoff

Source branch: `ws-offload-claude-home-phtables-store` (P0: OPS ph_tables streamed into the spaCR store).
Manifest: `tr_phtables-store_sources.json` (task identity `phtables-store`).

## Translation assignment

Prepare source-bound reviewed records for sv, de, es, zh_CN, pt, hi, ko, is and fr, through the normal scoped API and runtime loaders, from a fresh spacr-translations main. Keep code, RST roles, placeholders (`{path}`, `{table}`), dtype names and method names exact. This is a reviewed-record task only: no catalog builds, source changes or pins.

## Public API scope

Added owners (11): `spacr.store.PartWriter` with `write`, `close`, `abort`; `spacr.store.MeasurementWriter` with `stored_types`, `open_part`, `finish_part`, `write`, `close`, `abort`. Revised owner (1): `spacr.store.write_part` (its docstring gained the sentence on `PartWriter`). The exact English bodies are in the manifest.

## Runtime scope

One new runtime source in `spacr/ops_engine.py` (`_ph_cell_chunks`): "{path} does not hold the cells the well table {table} holds for its field; run ph_measure again".

## Retired

`ops_tables_parquet_threshold` was removed (listed in `spacr.validate.RETIRED_SETTINGS`); its label, section label and tooltip rows were deleted from every locale catalog in the source commit. Retire the matching reviewed records.
