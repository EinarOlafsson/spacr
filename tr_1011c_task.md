# Item1011c L3 source handover

The combined handover is [tr_1011_task.md](tr_1011_task.md); all 13 exact admitted runtime blocks, source hashes and the private/test inventory are in [tr_1011c_sources.json](tr_1011c_sources.json). Home owns locale records and catalog rebuilds for sv, de, es, zh_CN, pt, hi, ko, is and fr.

WORK_TASKS v88 admits this existing user-visible ValueError through the exact `_ph_segmentation_inputs` scanner owner in `tools/build_i18n_catalogs.py`. It is already wrapped in `tr()`; the runtime file is unchanged.

- Table: `UI`
- Source/ID: `ops_ph_channels {channels} must name {nuclear} and the cell channel {cell}`
- Location: `spacr/ops_engine.py:4671`
- SHA256: `15dfbc983a90d0f5f9d02f3f73f6e51a79df0ee9bf1b79920d4151508de6c7fe`
- Named fields: `channels`, `nuclear`, `cell`

The `_mask_batches._calibration_line` summary remains English console/log text through `log=print`, as in987b. L3 does not route it to a dialog or status label; no `tr()` or locale record is required. This resolves the earlier owner follow-up retained historically in the unchanged L2 JSON.

No locale records or catalogs were written here. The raw three catalog/review failures and strict published-alpha-page deferral are recorded in the combined handover; none is claimed passed.
