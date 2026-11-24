# phtables-store English source handoff

Source branch: `ws-offload-ws-phtables-store` (P0: OPS ph_tables streamed into the spaCR store).
Manifest: `tr_phtables-store_sources.json` (task identity `phtables-store`).

Translations are DROPPED until the end (maintainer, 2026-11-23 15:10): the lander records this file in trains/TRANSLATION_DEBT.md and lands without a translation pin.

## Translation assignment (when translations resume)

Prepare source-bound reviewed records for sv, de, es, zh_CN, pt, hi, ko, is and fr, through the normal scoped API and runtime loaders, from a fresh spacr-translations main. Keep code, RST roles, placeholders (`{path}`, `{table}`), dtype names and method names exact. This is a reviewed-record task only: no catalog builds, source changes or pins.

## Public API scope

Added owners (6): `spacr.store.writer`, `spacr.store.write_table`, `spacr.store.WellWriter` with `append`, `finish`, `abort`. `spacr.store.write_part` is unchanged. The exact English bodies are in the manifest.

## Runtime scope

One new runtime source in `spacr/ops_engine.py` (`_ph_cell_chunks`): "{path} does not hold the cells the well table {table} holds for its field; run ph_measure again".

## Setting tooltips revised (2)

`ops_storage` (ph_tables now always writes the store in measurements/; spacr_store adds reads, barcodes and crops there) and `ops_tables_integer_keys` (no longer "indexed": the store has no SQLite indexes). English bodies are in the manifest.

## Retired

`ops_tables_parquet_threshold` was removed (listed in `spacr.validate.RETIRED_SETTINGS`); its label, section label and tooltip rows were deleted from every locale catalog in the source commit. Retire the matching reviewed records. The nested API helper `spacr.ops_engine._ph_insert_rows.declared` was retired with the SQLite writer.
