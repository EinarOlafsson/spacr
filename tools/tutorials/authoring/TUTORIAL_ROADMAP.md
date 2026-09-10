# spaCR Tutorial Roadmap

This file is the canonical production order for the tutorial series.

## Authoritative source

- Repository: `/mnt/firecuda2/codex/repo/spacr`
- Branch: `nightly`
- Planning snapshot: `cc9d486`
- The local branch was four commits ahead of `origin/nightly` when this
  roadmap was reconciled.
- Module authority: `spacr/qt/app.py`, especially the `APPS` registry.
- Tutorial framework: `spacr/qt/tutorial/`.

## Series 1 — Installation and core analysis

1. PyPI, GitHub, and conda-forge overview
2. Installation with Conda
   - Install the official package directly with
     `conda install conda-forge::spacr`.
3. Installation with pip
4. Platform installers
5. Home screen and navigation
6. Python API and headless workflows
7. Mask
8. Measure
9. Annotate
10. Classify — computer vision
11. Classify — machine learning
12. Map Barcodes
13. Regression
14. Make Masks
15. Image UMAP
16. Activation maps

## Series 2 — Time-resolved analysis and segmentation models

1. Timelapse
2. Motility assay
3. Train Cellpose
4. Cellpose Masks
5. Model Compare
6. Model Zoo

## Series 3 — Annotation QC and biological assays

1. Annotation agreement
2. Plaque assay
3. Recruitment
4. Invasion assay
5. Replication assay

## Series 4 — Operations, reporting, and data utilities

1. Training Runs
2. Report
3. Plate Queue
4. External Masks
5. Align and Stitch
6. Plate Viewer
7. Database Browser
8. Format Converter

`Model Compare` is already tutorial 5 in Series 2. The nightly app registry
contains one Model Compare module, so it is not repeated here.

## Series 5 — Additional registered modules

1. Import Project
2. Batch Runner
3. Distributed Jobs
4. Classifier Evaluation
5. Run History
6. Classify — unified image/feature workflow
7. Curate
8. Illumination
9. Data Manager
10. Project Browser
11. Napari Bridge
12. Barcode QC
13. Hit List
14. Methods & Results
15. Run Compare
16. Control Charts
17. Pipeline Graph
18. Prediction Profiler
19. QC Dashboard
20. Image Scatter
21. Lineage
22. Layer Viewer
23. Graph Builder
24. AnnData Export
25. PCA
26. Tabulate
27. Feature Dictionary
28. Small Multiples
29. Gate Editor
30. Feature Explorer
31. Outliers
32. Experiment Design
33. Power / Design
34. Dose–Response

The roadmap covers all 63 modules exposed after spaCR loads both the built-in
and self-registering nightly app registries. Gate Editor is its current
standalone Explore module, and Annotate also links to that same editor for
population-driven annotation. The unified Classify lesson covers the current
entry point; the CV and ML lessons remain as focused workflow references.

## Optional supporting workflow tutorials

1. Updating and uninstalling spacr
2. GPU, CPU, and memory configuration
3. Settings, presets, and reproducible configurations
4. Input naming conventions and metadata detection
5. Output folders, databases, and exported files
6. Plotting and figure export
7. Command-line and unattended batch execution
8. Sequencing quality control
9. Troubleshooting, logs, and bug reports
10. Importing and exporting trained models

## Voice and localization design

- Offer the 24 release-approved English voices for English narration. The
  pause-heavy `af_nicole`, `af_alloy`, `af_kore`, and `af_nova` voices remain
  in the audition archive but are not exposed by the player.
- Group the other 26 voices by their supported language.
- Select narration language before selecting a voice.
- Store one silent master video and separate narration/subtitle tracks.
- Maintain a technical pronunciation dictionary for each language.
- Render narration in short sentence-level segments, then assemble and master it.

The existing renderer currently accepts one Piper `.onnx` voice. Production
should replace that hard-wired narrator with a provider-neutral narration
interface so the same silent master can be rendered with every supported
Kokoro language and voice without recapturing the UI.

## Video rendering design

- Capture a clean app screenshot only when a scripted action changes the UI.
- Treat those screenshots as visual keyframes rather than repeatedly grabbing
  an unchanged window.
- Show a small, solid magenta point only for a scripted click. Animate it
  smoothly to the click target; passive explanation and overview steps have
  no pointer.
- Preserve the selected widget at full brightness, draw a blue focus ring,
  one pixel wide, and dim the rest of the interface.
- Allow overview steps to retain full-screen brightness.
- Use live frame capture only when spaCR itself is displaying meaningful
  animation, progress, or playback.
- Render the assembled keyframes and overlays as a 3840×2160, 30 fps H.264
  video with Open Sans Regular typography, a separate narration track, and
  SRT subtitles.

## Production status

- Voice catalogue: complete for all 24 release-approved English voices.
- Portal narration catalogue: complete for 50 Kokoro voices across eight
  languages; selectors filter voices by language and retain the user's choice.
- All 50 configured voices have encoded narration and per-scene timing
  manifests. The browser maps those timings onto one shared silent master.
- Experiment images: received.
- Tutorial order: recorded in this roadmap.
- Nightly route coverage: 44 Home modules, 23 workflows reached through a
  current host module, and six non-application lessons, for 73 lessons total.
  All eight full-language catalogs preserve the same route structure.
- Existing automated scripts: Home, Mask, Measure, Crop workflow, Classify
  (CV), and Timelapse.
- Tutorials 1–5 have 4K review renders, silent masters, `af_heart`
  narration, and English subtitles:
  PyPI/GitHub, Conda, pip, platform installers, and Home/navigation.
- Tutorial 1 is pointer-free and sends PyPI to every TTS backend as the exact
  continuous lowercase token `pypie`. Pronunciation is controlled before
  synthesis; rendered vowels are never shortened or spliced.
- Tutorials 2–4 are pointer-free command/platform walkthroughs. Tutorial 2
  installs the published conda-forge package directly, without invoking pip.
- Tutorial 5 was captured directly from the nightly application and uses the
  magenta point only for the click that opens Mask.
- The current tutorial workspace is
  `/mnt/firecuda2/Claude/toxoplasma_projects/tutorials`.
  Its `orig/` folder contains 3,136 CQ1 TIFF files: 784 complete four-channel
  fields. The associated database, results, merged arrays, and module settings
  remain in their respective project folders.
- Live Mask mapping: C1/Nuclei → index 0, C2/ER and cells → index 1,
  C3/lipid droplets and organelles → index 2, C4/parasites → index 3;
  downstream channel of interest is ER/index 1.
- Mask validation uses the source CSV plus local-path, five-image test-mode,
  deterministic-sampling, and retained-intermediate overrides. The walkthrough
  uses the C2 ER field for a cell-only Live Preview, teaches viewers to judge
  preprocessing from their own image, and demonstrates object filtering
  without prescribing dataset-specific normalization or area values.
- `tools/capture_mask_experiment.py` captures the current app at 4K, derives
  highlight geometry from the live widgets, and uses the real test-run masks.
- The Mask capture contains 31 reusable UI states. The published walkthrough
  selects only the 11 states needed for a surface-level workflow, with focus
  geometry derived from the live widgets and no dataset-specific values stated
  as general recommendations.
- The experimental Measure remake is complete: 25 captured scenes, a 4K
  silent master, captions in 14 languages, and all 50 supported narrated
  tracks. It tours all eight current settings categories and uses the real
  five-field measurement outputs and crop gallery. The next dataset-backed
  production target is Annotate.
- Conda-forge status: the official package is published; the installation
  tutorial uses `conda install conda-forge::spacr` directly.
- Delivery uses GitHub Pages for the player, posters, and 1440p silent videos,
  plus the `einarolafsson/spacr-tutorials` Hugging Face dataset for narration,
  timing sidecars, and 4K silent masters. No third-party video-channel copy is
  part of the current publishing path.
