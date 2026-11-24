# store-s4 English source handoff

Source branch: `ws-offload-ws-store-s4` (unified store, step S4: Measure and Make Masks write the spaCR store natively; measurements.db is no longer written).
Manifest: `tr_store-s4_sources.json` (task identity `store-s4`).

Translations are DROPPED until the end (maintainer, 2026-11-23 15:10): the lander records this file in trains/TRANSLATION_DEBT.md and lands without a translation pin.

## Translation assignment (when translations resume)

Prepare source-bound reviewed records for sv, de, es, zh_CN, pt, hi, ko, is and fr from a fresh spacr-translations main. Keep code, placeholders (`{what}`, `{key}`), setting names and file names (`measurements.db`, `spacr.duckdb`, `spacr_store`) exact.

## Public API scope

None. `spacr.measurement_store` declares `__all__ = []` (internal vocabulary over `spacr.store`).

## Runtime scope

One new runtime source in `spacr/measure.py` (`_report_steps_not_on_the_store`), see the manifest.

## Setting tooltips revised (2)

`measurement_backend` (spacr_store only; retired values still load) and `measurement_backend_target` (now only the export target of a retired value). The `measurement_backend` combo offers `spacr_store` only; the values sqlite, duckdb, parquet and postgres are retired (spacr.settings._RETIRED_MEASUREMENT_BACKENDS; spacr.validate warns).
