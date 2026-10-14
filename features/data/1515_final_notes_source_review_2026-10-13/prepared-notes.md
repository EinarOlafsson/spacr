## spaCR 1.5.1.5

### New

- Make Masks opens 3D images in the same view as 2D ones. A slice slider and the Z wheel move through the stack, and the view can be Default, Orthogonal or True 3D. Draw and edit masks on any slice and delete objects; edited volumes are saved as whole sections.
- Make Masks supports transmission electron microscopy (TEM) images.
- A FIB-SEM example volume with 3D mitochondria masks from HeLa cells ([EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/), CC0). Fetch it with `spacr-download fibsem`.
- Make Masks has a Hide judged fields switch to skip fields already marked Keep or Discard. The choice is remembered for each folder. The terminal queue offers the same filter with `spacr-make-masks --exclude-verdicts`.
- Export settings... on every settings form, and Save settings CSV... in Distributed Jobs (#140, #141, #142).
- Preferences > Appearance > Accessibility: High contrast, Text size, Reduce motion, Stronger focus outlines and Keyboard shortcuts.
- When startup animation is enabled, one borderless window opens at the available screen size. The spaCR logo, built from microscopy tiles, holds for at least 2 s, longer while the application loads, and then breaks apart over 4 s. Help has a replay entry. The other startup dialogs (drafts, installer consent, tour, What's new) wait until the sequence ends or is skipped.
- Gate Editor: the mode, plane and spin keys can be rebound, and a picker lets you choose between overlapping elements.
- Model Zoo has a model card viewer with three pages (scores, training facts, about). Cards download from Hugging Face, are cached, and fall back to the bundled card when offline.
- `spacr-tutorial` opens the online tutorials.

### Improved

- The README and Model Zoo highlight well detector v3. Figure-panel detection retains v2, which was evaluated on cropped panels in published figures. The plaque segmentation default is toxoplasma_plaque_v4.
- The nine PV and plaque segmentation cards use the ground-truth re-score of 2026-10-11 (object F1, boundary F1, precision, recall, AJI and Dice), comparing each model with its matching stock Cellpose-SAM base on that model's recorded held-out subset. The README shows score cards for the default models, with links to older models and alternatives.
- Sixteen settings that accept a fixed list are now dropdowns, for example `spatial_platform`. Stored values are unchanged.
- Regression: `dependent_variable` is a dropdown of the numeric columns in the score file, with a note when `pred` is missing (#144).
- Measure reads a pasted list of plate folders as plates (#138, #139).
- Classify: relative database paths in `src` work, and a missing database is reported with the full path searched (#143).
- Mixed model: Stop is reported as a stop, the nested design is checked before the fit with the setting to change named, and optimiser fallbacks are handled (#145).
- Field backgrounds on Mask screens and in Spaceout do less work per frame, and colour waves use less CPU. Ripples from a closed window run outward only, and the ripple slider scales waves up to 200 %.
- With Dynamic animation enabled, active backdrops are capped at no more than 30 fps during CPU-heavy jobs, while Spaceout fractals pause.
- Preferences reuses its dialog, and appearance updates run in smaller steps. Opacity previews update Home in place; cold first-Apply responsiveness remains under review.
- Screen backgrounds paint without repeatedly reading Preferences.
- Saved segmentation masks keep their field's dimensions, and masks of a different size are regenerated before reuse.
- The README is rebuilt from images, and the slide deck is rebuilt to match. README and interface text are translated into nine additional languages.
- Release automation can build and publish Apptainer (SIF) images after successful Docker publication.

### Fixed

- GPU usage polling reads through NVML instead of starting `nvidia-smi`, and usage bars paint with the next frame.
- Command palette: a key typed while the list is rebuilding is kept instead of crashing the app.
- Regression panel: fixed a thread error.
- Make Masks: outside Box mode, a linked image with no box ledger opens and saves without the "Cannot load box annotations" popup.
- Closing Model Zoo or a backend installer during a download safely retires its worker thread. Animated graphics resources are released before the app exits.
- Starting a run, importing Templates and updating the figure gallery avoid heavy imports or image decoding on the interface thread.
- A source checkout or editable install reports its own version.
- Wells named F01, F02 are read as wells, and class counts ignore the `.spacr_crop_format.json` file.

### Alpha

Experimental features are marked alpha. Show alpha features and Show alpha species reveal the corresponding experimental controls and organism pages.

- MitoNet as a Make Masks backend for TEM images.
- Spatial Mask Generation and Spatial Map Barcodes submodules, opened from the Spatial switches.
- The separate alpha application is now named ParaScape. spaCR installs, updates and opens it using its PyPI package.
- Other organisms stay behind Show alpha species; Toxoplasma is always shown.
