# Bug item: diagnostic figures differ between spacr-run and a direct run_ops call

- **Found:** 2026-11-19, item 977 identity check (master lane claude-977-1118).
- **Rule broken:** the maintainer's standing rule that rerunning a settings file in spaCR gives identical output.
- **Status:** spec only. Not fixed in 977, by the master's decision.

## What happens
- Well A1 of the 977 profiles archive was run twice on the same inputs and settings:
  - once through `spacr.ops_engine.run_ops(settings, wells=["A1"], phases=...)` from a driver script (spaCR 03ed36da72);
  - once through `spacr-run ops --settings settings/ops_profiles_A1.csv` (spaCR 17c737bc2d).
- 138 of 139 outputs are identical: every table, the calls, masks, per-cell features, merge, guides and genes.
- The one difference is `results/ops_decode/decode_tuning_curve_A1.png`:
  - same size (750 x 570) and same plotted data;
  - but 91,033 pixels differ, because the title is drawn much larger in the driver run;
  - the PNG metadata is identical (Matplotlib 3.11.1, 150 dpi).
- Evidence:
  - `/mnt/wd4tb/scratch_977/run_A1/` vs `/mnt/wd4tb/scratch_977/run_A1_spacrrun/`;
  - `/mnt/wd4tb/scratch_977/work/identity_A1_wholewell.json`.

## Likely cause (to confirm)
- `spacr.ops_decode.write_tuning_curve` draws inside `figures.style.figure_style("print")` and sets `fontsize=9` on the title.
- The larger title in the driver run suggests one of two things:
  - a process-wide matplotlib state that one entry point sets and the other does not (rcParams, style sheet or backend set up by `spacr.cli`);
  - or `plot.save_figure` behaving differently depending on what was imported first.
- The code difference between the two SHAs does not touch this function.

## Spec
1. Find which entry point changes the figure state, and make every OPS figure independent of process-wide matplotlib state. Each figure carries its own complete style, so `spacr-run`, a direct `run_ops` call, the GUI and a notebook give byte-identical PNGs for the same data.
2. Regression test: draw `write_tuning_curve` in a fresh subprocess twice, once after importing `spacr.cli` (the `spacr-run` path) and once after importing only `spacr.ops_engine`. Assert the PNG bytes are equal.
3. Extend the OPS identity checks so that PNG outputs are compared by bytes, not only the tables.
