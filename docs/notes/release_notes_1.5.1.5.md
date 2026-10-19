## spaCR 1.5.1.5

### Upgrading

On macOS, how to reach 1.5.1.5 depends on the version you have:

- 1.5.0.8, 1.5.0.9 and 1.5.1.0: install the 1.5.1.5 `.pkg` from the [releases page](https://github.com/EinarOlafsson/spacr/releases). Do not use Help > Check for updates: the updater in these versions removes the user runtime and then stops before installing. The `.pkg` restores a working installation.
- 1.5.0.1: install the `.pkg`. Its in-app update cannot run.
- 1.5.0.4 to 1.5.0.7 and 1.5.1.2 to 1.5.1.4: Help > Check for updates works. On an Intel Mac, an update that lands on 1.5.1.4 reports "Numpy is not available"; updating to 1.5.1.5 repairs it.

On every platform, a new install of 1.5.1.3 or 1.5.1.4, or an update to either, now installs PySide6 6.12.0, which can crash spaCR. 1.5.1.5 excludes 6.12.0, so updating to 1.5.1.5 replaces it.

### New

- Make Masks opens 3D images in the same view as 2D ones. A slice slider and the Z wheel move through the stack, and the view can be Default, Orthogonal or True 3D. Draw and edit masks on any slice and delete objects; edited volumes are saved as whole sections.
- Make Masks supports transmission electron microscopy (TEM) images.
- A FIB-SEM example volume with 3D mitochondria masks (EMPIAR-10982, HeLa, CC0). Fetch it with `spacr-download fibsem`.
- Export settings... on every settings form, and Save settings CSV... in Distributed Jobs (#140, #141, #142).
- Preferences > Appearance > Accessibility: High contrast, Text size, Reduce motion, Stronger focus outlines and Keyboard shortcuts.
- Startup uses one borderless window that opens at screen size. The spaCR logo, built from microscopy tiles, holds for 2 s and then breaks apart over 4 s. Help has a replay entry. The other startup dialogs (drafts, installer consent, tour, What's new) wait until the sequence ends or is skipped.
- Gate Editor: the mode, plane and spin keys can be rebound, and a picker lets you choose between overlapping elements.
- Model Zoo has a model card viewer with three pages (scores, training facts, about). Cards download from Hugging Face, are cached, and fall back to the bundled card when offline.
- `spacr-tutorial` opens the online tutorials.
- Object detection has a 3D switch. When on, 3D thresholds are used, and Cellpose runs in 3D or stitches 2D slices into 3D objects.
- Make Masks measures the current 3D image.
- Make Masks has an Edit Label action and a Time button. +Cell and +Daughter record division lineage, which is stored in 32-bit TIFF masks.
- Anonymous error reports can be sent without a GitHub account. They are on by default and can be turned off in Preferences. spaCR AI is available to help with errors.
- Updates download and stage in the background and ask once to Restart. Preferences > Performance > Uninstall... removes the current version or all versions.
- Submit to cluster reads your `~/.ssh` config and handles logins that need a second factor.
- Plaque model toxoplasma_plaque_v5.

### Improved

- Updates stage replacement runtimes, check downloaded installers and verify the requested installed version before relaunch. User data and configured model/cache folders are protected; optional cleanup is off by default and offers a graphical folder preview. For the first upgrade from an older frozen Windows installation, use the new desktop installer before relying on the new in-app update path.
- Saved workspaces retain selected databases, montage and graph settings, regression views and queued figures. Copied references are verified before restoration.
- Installer and mask-editing messages are refreshed across the nine interface languages.
- The well detector v3 is the default in the app, the README and the model cards. The plaque default is toxoplasma_plaque_v5; v3 is recommended for Gel Doc wells.
- Model cards use the ground-truth re-score of 2026-10-11 (object F1, precision, recall, AJI, Dice, with stock Cellpose-SAM for comparison). The README shows one score card per model.
- Sixteen settings that accept a fixed list are now dropdowns, for example `spatial_platform`. Stored values are unchanged.
- Regression: `dependent_variable` is a dropdown of the numeric columns in the score file, with a note when `pred` is missing (#144).
- Measure reads a pasted list of plate folders as plates (#138, #139).
- Classify: relative database paths in `src` work, and a missing database is reported with the full path searched (#143).
- Mixed model: Stop is reported as a stop, the nested design is checked before the fit with the setting to change named, and optimiser fallbacks are handled (#145).
- Magnification sets the default Cellpose diameter. A diameter of 0 and "auto" are the same.
- Make Masks: each disconnected brush stroke is a new object. Hold Shift to continue an object across gaps; hold Alt to pan.
- Accessibility names now cover lists, the main window and Preferences.
- Wording in the app, docs and API is factual and reviewed in nine languages, and settings section titles are translated.
- The GUI no longer freezes during the Regression example download and batch run, and Regression results open faster on first use.
- Pages with folds open faster the first time.
- Preference reads are kept open instead of being reopened.
- Invasion and other long analyses no longer starve the GUI, because garbage collection is deferred to idle time.
- Plate View applies only the newest recompute.
- Mask screens render their field backgrounds more efficiently.
- Preferences opens, applies and closes faster, and Apply, Keep and Revert no longer freeze the app.
- Make Masks costs less per frame.
- Every saved mask records its field's size, and a mask of a different size is never reused.
- The README is rebuilt from images, and the slide deck is rebuilt to match. README and interface text are translated into nine languages.
- Apptainer (SIF) images are published with each release, after the Docker images.

### Fixed

- Closing the main window saves pending Annotate labels and drains its page loader and database writer.
- Corrected the restart command and launch environment for frozen desktop applications, while preserving saved workspace state.
- Prevented environment capture from relaunching frozen desktop applications.
- Installations avoid PySide6 6.12.0, which can crash rendering on Python 3.10.
- Failed cache moves retain data and location settings, restore an existing empty destination after successful rollback, and do not broaden its permissions.
- Own-DINO classifier loading rejects teacher checkpoints that change during loading.
- Embeddings reports unreadable or empty label tables without discarding existing labels, vectors or scorecards.
- Resizable panes keep their remembered sizes and close safely with their screen.
- Figures restored from a saved workspace retain thumbnail detail on high-DPI screens.
- Module screens no longer stall every 2 s. The GPU is read through NVML instead of starting `nvidia-smi`, and usage bars paint with the next frame.
- Command palette: a key typed while the list is rebuilding is kept instead of crashing the app.
- Regression panel: fixed a thread error.
- Legacy PNG masks load in Make Masks.
- The workspace is saved on the GUI thread.
- Text is re-measured after a font change.
- Make Masks: a linked image with no box ledger opens and saves without the "Cannot load box annotations" popup.
- A source checkout or editable install reports its own version.
- Wells named F01, F02 are read as wells, and class counts ignore the `.spacr_crop_format.json` file.

### Alpha

These appear only with Show alpha features on.

- Own-DINO classifiers can be trained from a supplied teacher checkpoint and applied from their saved artifacts, using explicit channel order and saved input scaling. Predictions and features can be exported as tables.
- MitoNet as a Make Masks backend for TEM images.
- Spatial Mask Generation and Spatial Map Barcodes submodules, opened from the Spatial switches.
- Embeddings offer DINOv2 and can use an official Cell-DINO checkpoint with explicit channel mapping.
- Other organisms stay behind Show alpha species; Toxoplasma is always shown.
