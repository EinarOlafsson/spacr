# Item 1000: OPS Live works like Mask Generation's Live view (alpha)

- **Number:** 1000, assigned by the master session on 2026-11-20.
- **Lifecycle:** future (alpha), like the rest of OPS Live (item 971, `OpsLiveSwitch`). With "Show alpha features" off nothing changes: the OPS screen, its settings and every run output stay byte-identical to nightly.
- **Requested by the maintainer (2026-11-20)** after reviewing the OPS Live preview: it must work exactly like the Live view in Mask Generation and reuse its parts.
- **Code:** `spacr/ops_live.py` (the work) and `spacr/qt/widgets/ops_live_preview.py` (the tabs); the grid-search inputs of `spacr/qt/widgets/ops_grid_search_panel.py` move into the Spots settings panel. No new module.
- **Reused from Mask Generation's Live view:** `_ZoomView` (wheel zoom, pan, right-click save, ruler) from `spacr/qt/widgets/live_preview.py`; the panel-scale slider `install_preview_scale`; the settings-dialog pattern of `LiveSettingsDialog` (Close + checkable "Propagate settings" that copies now and while on); the shared `LivePreviewContract` run/cancel guard; `ActivitySpinner` for the running indicator. Graphs use pyqtgraph, already a hard dependency (`setup.py`, `spacr/qt/widgets/fast_plots.py`).

## Bugs (first)

**B1. A tab never raises.** Merge & aggregation and Clustering showed `ValueError: merge and aggregation needs a finished run in dst_root (measurements.db and the phenotype cell table)`.
- Cause: the preview used the test data's stored nuclei, calls and cells only when `genotype_source` was empty. With `genotype_source` pointing into the downloaded test data (`ops_testdata_funk2022_A1/...`), the preview took the folder for a user's own data and asked for a finished run in `dst_root`.
- Fix: a `genotype_source` inside an OPS test-data folder is recognised as the test data, so merge and clustering use its stored `reference/` tables (objects, barcodes, phenotype cells, masks); the test data already holds them.
- Any missing input is a notice in the tab, not an error: what is missing and what to run first (for a user's data: the OPS run's stitch → ph_measure phases into `dst_root`; for clustering: a merge with at least 5 genes above `ops_profile_min_cells`).

**B2. A running tab says so.** Opening a tab started its step while the tab still said "Not run yet". Every tab now shows, in the tab, a spinner and "Running <step>…" (and the tab title carries "running") from the moment its pass is scheduled until it ends.

## Features

**F1. Every tab** (Align & stitch, Spots & genotyping, Objects & phenotype, Merge & aggregation, Clustering):
- an **Image quality** slider, 10–100 % (default 100 %): the resolution the pictures are rendered at;
- pictures in Mask Generation's zoomable view;
- a **Settings…** button opening that tab's settings panel (the step's main settings) with **Close** and **Propagate settings**. A change in the panel re-runs that tab's step on the preview data and redraws it; Propagate copies the panel's values into the OPS form (while on, every later change too);
- **laptop mode** (Preferences → performance level Laptop, `preferences.get_laptop_mode() == "on"`): switching Live on runs the tab on view once; after that nothing refreshes by itself. A **Refresh view** button (in the tab and in the panel) applies the changed settings and quality and redraws (maintainer 2026-11-20).
- a Settings-panel change reruns its tab even while Live is off (maintainer 2026-11-20: keep).
- graphs have no background of their own and take the page's text colour from the theme, so they read like the page in the light and the dark theme; data colours are the theme's cool accents.

**F1b. Mask Generation's Live view gets the same** (maintainer 2026-11-20): the shared 10–100 % image-quality control (`_ImageQualityControl` in `live_preview.py`, also used by OPS Live) and, in laptop mode, a Refresh view button; a settings change that would re-filter the masks waits for it. Quality is drawn by `_ZoomView.set_render_percent`: the picture is resampled and stretched back over the same scene, so zoom, pan, the ruler and the hovered pixel stay in image pixels, and saving keeps the whole picture.

**F2. Align & stitch shows what the step does**, from the real stitch/phenotype/align run on the preview data:
- the stitched block with every tile, every placed phenotype field and the sampled crop;
- the **stitched seam** between two neighbouring tiles, before (raster grid from `ops_raster_overlap`) and after stitching, with the stitch offset;
- the **phenotype field on the sequencing tiles** at its placement (scale, rotation, centre);
- one crop's cycles **before and after cycle alignment**, with every cycle's offset in the table.

**F3. Spots & genotyping:** the grid-search inputs move from the tab into its Settings panel; the read-quality histogram and the calls-per-nucleus bars become Qt (pyqtgraph) graphs, redrawn with the image after each settings change.

**F4. Objects & phenotype:** Settings panel, zoomable image, Qt graphs of two measured features, live update.

**F5. Merge & aggregation and Clustering:** the same buttons and behaviour; matched pairs, matches per field, cells per gene, the embedding and cluster sizes are Qt graphs; plus B1.

## Acceptance

- Offscreen Qt tests: B1 (a test-data `genotype_source` uses the stored tables; a missing run is a notice, never an exception text), B2 (spinner and tab title while running), F1 (slider re-renders at the chosen resolution, zoom view per picture, panel Close/Propagate, laptop mode waits for Refresh), F3/F4 (pyqtgraph graphs on the page), the existing item 971 tests updated to the new page.
- Off-state: with alpha off the OPS Live switch stays hidden; no setting default changes (no schema row).

## Follow-ups

- Tutorials showing OPS Live must be re-recorded when tutorial work resumes (lane owner spacr-d7).
- **Later follow-up (maintainer 2026-11-20, not in this item):** an Align & stitch "before placement" picture of the phenotype field at the raster seed's own prediction, beside the placed field. This item shows the placed field only.
