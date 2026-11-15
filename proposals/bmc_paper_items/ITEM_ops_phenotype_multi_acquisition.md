# 947 OPS phenotype folders that hold two acquisitions sharing a channel

Lifecycle: new (bug / usability). Filed 2026-11-14 from the stage-1 Ready check
(`BMCspacr/reanalysis/proof_stage1_stage2/REPORT_stage1_ready_check.md`).

## Problem

Funk plate B's phenotype folder holds two acquisitions that both carry the
A594 channel. Nightly 35ca0c47f0 refuses the whole folder; older code accepted
it (and silently picked one). The proof check worked around it by pointing
placement at the DAPI-GFP-A594 acquisition folder. Users with the same layout
get a refusal that does not say which acquisitions clash or how to choose.

## Build

1. The refusal names the clashing acquisitions (folder or file-name group),
   the shared channel, and the setting that resolves it.
2. A setting to pick the phenotype acquisition when several are present
   (e.g. `ops_ph_acquisition`, default empty = refuse with the message above,
   as now). No silent choice.
3. Settings key, label, tooltip in 9 locales; API records for any new or
   changed public callable; schema row only if a default changes (it does not).

## Acceptance

- Plate B A1 placement with the folder as delivered: default refuses with the
  new message; with the acquisition set to the DAPI-GFP-A594 group the
  placement is byte-identical to the proof check's run.
- Tests: refusal text (offscreen, no data), selection by setting, single
  acquisition folders unchanged.
