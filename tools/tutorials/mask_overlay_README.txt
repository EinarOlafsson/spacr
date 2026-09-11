Mask overlay API workaround — recorded four-channel example only
==============================================================

This is an explicit plotting workaround, NOT a repair of the Mask GUI or its
native overlay loop. The recorded native run generated two merged arrays but
only one overlay: its directory scan attempted to plot .spacr_plane_layout.json.
Do not delete that metadata file; other readers need its channel/label layout.
Do not enable pickle loading to treat metadata as an image.

In the same environment where spaCR is installed, unpack this ZIP and run:

  python export_mask_overlays.py --source /path/to/plate1/test/merged --destination /path/to/new_overlays

Use the actual merged output of YOUR completed mask run, not a stale folder.
The output folder must be new and outside the merged input folder. Keep the
original run, console, settings, QC reports and its partial-overlay warning.

This helper accepts the exact recorded layout: four intensity channels
[0, 1, 2, 3], then cell, nucleus and pathogen masks at planes 4, 5 and 6.
Cell intensity is channel 1; nucleus intensity is 0; pathogen intensity is 2.
It refuses a different layout instead of silently mislabelling planes. It
selects regular .npy inputs only, never JSON or symbolic links.

The helper calls spacr.plot.plot_image_mask_overlay separately for each array.
Display settings are outlines, 1st/99th percentiles, thickness 3, default
outline colours and four image channels. PNG export has an explicitly opaque
black background, so the real white panel titles remain readable rather than
disappearing on an image viewer's transparency checkerboard. This is a figure
export setting, not retouching the saved image. Display normalization does NOT
rewrite source intensities. It writes new PNGs and overlay_checks.json;
the latter records source hashes, the exact plotting implementation and checks.

All four displayed channel/contour arrays are independently checked against
the input planes. The combined panel is checked for correct foreground coverage,
not for a preferred colour per object or a validated biological interpretation.
Every PNG is reopened and decoded, and every original array and layout hash must
remain unchanged. The original native plotting loop stays unfixed.

In the recording, the new two-field run used explicit flow thresholds 0.4,
not the example archive's 100 (which disables flow filtering). These are
demonstration settings, not data-derived optimal settings. The fields have
25/19/8 and 52/50/58 final cell/nucleus/pathogen labels respectively. The first
field's pathogen QC still warns near_empty_field: eight objects do not support
the per-field size checks. Cell QC precedes cell-mask adjustment; its counts
are not interchangeable with final merged-mask counts.

These checks establish plotting/input correspondence, not segmentation
accuracy. Inspect the masks yourself before measurement or downstream analysis.
The earlier live-preview count demonstration uses a different field and is
not a benchmark against these two batch fields. No new model training, AI
provider request or scientific approval is part of this workaround.
