## spaCR 1.5.1.7

In this release I have added guided Protocols to Navigator, a layer view and multichannel images to Make Masks, a drug-screen mode in Regression and more sequencing input formats in Map Barcodes, made Measure much faster to store its results, and fixed several problems that could lose or merge mask data.

### Upgrading

Upgrade an existing Python environment with:

```bash
python -m pip install --upgrade "spacr==1.5.1.7"
```

Desktop installers are listed under **Assets** below.

- Some keyboard shortcuts changed to follow the usual platform conventions: Preferences is on Ctrl+, (the platform's own Preferences key where there is one), Zoom is on Ctrl+0 and Ctrl+plus/minus, Home is on Alt+Home, and the background toggles use Ctrl+Shift. Restart the background moved to Help > Window and no longer has a key. Shortcuts you saved before this change are carried over to the new defaults. If a key you chose yourself is already used, the action whose default changed is left without a key rather than taking yours.
- On Python 3.15, spaCR now selects a compatible numcodecs build.

### New

- **Navigator Protocols.** A Protocols control beside the AI icon in the Navigator composer guides you through your experiment one step at a time: input type, format, metadata and experiment. Each step explains the selected choice, with Previous and Next, and then opens the matching protocol and flowchart. There are 71 software protocols reached through 120 input paths, with shared diagrams, guidance for each choice and links that open the right module with your source folder already filled in. Each protocol step states the assumptions it makes about your experiment. Protocols can be exported as PDF checklists, and the reader has three backgrounds. Navigator search, answers and questions keep working while a protocol is unfinished.
- Navigator links to the published tutorials, and its learning action opens the module Dock.
- **Make Masks layers.** Make Masks shows a field as a stack of layers, similar to napari: one image layer per raw channel, plus label layers for the mask you are editing and any loaded masks. A new Layers section lists them, so you can show, hide, rename and reorder layers and choose the active one. For the active layer you can set opacity, blending, colormap, contrast limits, and filled or outlined labels.
- **Multichannel images.** Make Masks opens multichannel files (RGB, channel-last, ImageJ and OME-TIFF) with every channel as a layer. Mask generation now splits multichannel files into one raw plane per channel in file order, the same way it reads separate channel files, so both give the same masks. RGB PNG/JPEG files and local OME-Zarr images convert too.
- **Time mode in Make Masks.**
  - Edit Label with Time on suggests the next free label in the movie. If another cell already has the label you type, spaCR asks whether to give that cell a new label in every frame; the traced cell then takes the label in every frame. Undo and redo restore both.
  - The Time button's menu has Paint label while scrolling: each mouse-wheel step gives the object under the pointer the current label.
  - The Daughter cells dialog offers budding, mitosis or fission. Remove division folds the first daughter back into the mother as one undo step.
  - While a movie's lineage loads, Make Masks shows how many frames are done out of the total, with a Cancel button. Time labelling steps through the movie in time order and reaches finished frames when you resume.
- **Drug-screen Regression.** Regression has a new `screen_type` setting. With `screen_type='drug_screen'`, each drug at each concentration is compared with the vehicle-control wells in one ordinary least-squares model, with a plate effect when the wells span several plates. Doses are never pooled. Each level reports its effect, standard error, 95% interval, p value, Benjamini-Hochberg q value and percent of vehicle. Results go to `results/drug_screen/<response>/` as `well_data.csv`, `drug_effects.csv` and a figure. Wells with two treatments, treated wells without a positive dose, plates without vehicle wells and per-cell analysis are refused with the reason. The pooled CRISPR path is unchanged and stays the default. In Protocols, the Drug screen branch offers Regression beside Dose-Response.
- **Map Barcodes input formats.** Map Barcodes reads `.fastq`, `.fq`, `.fastq.gz` and `.fq.gz` files, detecting compression from the file contents, and leaves I1/I2 index reads out. BAM, CRAM, SRA, BCL, FASTA and bzip2/zstd/xz/ORA files are now named, with the conversion they need, instead of being silently ignored. A new `sequencer` setting (auto, illumina, element, mgi, ultima, ont, pacbio, other) is checked against the read headers. `mode='mixed'` decodes paired samples as pairs and unpaired samples from their single read. Each sample's files, compression, header type, sequencer and pairing are recorded in `read_inputs.json`. An empty FASTQ is reported instead of failing the run.
- **Gate Editor and Graph Builder annotations.** Both modules annotate selected points the same way, through the plot's right-click menu: select points in Graph Builder, or use a gate in Gate Editor, then give them a label or category. Both have a slider that limits how many points are drawn, choosing them at random, so large tables stay usable on slower computers. The limit only affects drawing; no rows or annotations are removed.
- spaCR AI answers end with links that take you to the relevant module or documentation page. The hosted assistant now runs on Gemma.
- An Overview guide in the documentation covers plate navigation and exports, in all nine documentation languages.
- Model Zoo has a new candidate plaque model, `toxoplasma_plaque_v7`, with its score card. On the held-out test images it reaches an object F1 of 0.668 (95% CI 0.576–0.756), against 0.200 for stock Cellpose-SAM. It is a candidate only: `toxoplasma_plaque_v6` stays the default.
- Model Zoo: models downloaded from the community catalogue are now found by the previews, by a new spaCR process and by runs with downloads off or the catalogue host down.

### Improved

- **Measure is faster to store results.** Each measurement frame is now written to `measurements.db` in one database transaction instead of committing every row separately. On a spinning disk, one 900-cell field of one table took 18 s before and 0.13 s now. Per-object intensity-distance, child-to-parent and component measurements now work only inside each object's bounding box; the tables are unchanged.
- Make Masks:
  - Keep is the one save action: it saves the mask, moves to the next field and shows a Saved mark. Save mask and stay is in Keep's menu.
  - Large views are composited off the main thread, so restyling or zooming a large multichannel field does not freeze the window.
  - Opening a calibrated TIFF and saving its first mask no longer wait for PyTorch to load.
  - The title, toolbar and panels fold before the image gets too small, so the image's bottom bar stays visible at 720p and at large text sizes.
- Annotate: the toolbar is arranged in labelled groups that wrap as a whole, setup commands are on a Setup menu (including Generate annotation database), and the delete actions are kept apart from navigation.
- Settings screens: Run is the only filled (primary) button; Submit remote is a secondary button. Import settings and Export settings are icon buttons next to Templates. Clear console, Copy console and the Preferences button are in the console's header.
- Gate Editor stays responsive: Save graph draws in a separate process and always writes its PNG, the 3D projection and clustering run in the background, and the graph is redrawn only when needed.
- Tooltips, the settings box, dialogs, buttons, table headers and help text are sized for their text at every Zoom from 50 % to 200 % and in all languages, so text is no longer cut off. This covers Make Masks, Annotate, Regression, Queue, Database browser, Convert, Batch, Profiler, Run history, Storage, Power, Experiment Design, Align, Distributed jobs and Preferences. Long field placeholders show a short caption, with the full description in the tooltip.
- Preferences > Appearance > Show tooltips now turns off every tooltip, including the settings box and the setting help in the bottom strip. The two tooltip surface switches are greyed out while it is off.
- Field and panel borders have at least 3:1 contrast against every page colour in light and dark themes. Red is used only for destructive actions, so Cancel, Close, Reset and Clear buttons are neutral. Disabled Run and Stop buttons lose their action colour. Drop-down arrows are drawn as chevrons.
- Keyboard use: Dock section headings take focus and fold with Space or Enter, Navigator cards follow the composer in Tab order, settings forms are reached first with Tab, and dialogs use the platform's button order with Cancel as the safe default.
- Organism pages hide alpha-stage module tiles unless Show alpha features is on, as Home does, and label modules that are not built yet as Planned.
- Cellpose Apply preview uses the same normalization and resampling as the batch run.
- Plaque preview: tuned preview values are kept when you apply them to the settings form, and the preview controls, missing-model consent and settings handoff were corrected.
- UniProt annotation retries a busy server for up to 20 s. If it stays busy, the note says that UniProt is busy or unreachable instead of reporting no results. Neospora caninum now uses taxon 29176, because UniProt no longer files entries under 572307.
- Overview exports a field to PDF at quality 1600 and shows Back navigation when wells and images are open.
- Graph Builder keeps each colour level's hue when a filter or sample removes other levels.
- Closed dialogs and windows are now freed from memory. Before, opening and closing all Preferences dialogs could keep about 24 MB for the rest of the session.
- Screens that start folded are prepared while the app is idle, and the Model Zoo picker fills its table in steps, so both open with fewer pauses.
- Anonymous error reports now record which shutdown phase was running and which kinds of background work were still active, and, when the annotation database cannot open at startup, safe facts about the file such as its type and the SQLite result code. They never include paths, file names, user names, hosts or data. This is follow-up for #175, #176, #177 and #178.
- ParaScape upgrades are repaired in place, and ParaScape uses white artwork on dark themes.
- User guides, in-app text and the API documentation were reviewed for factual wording and translated into nine languages.

### Fixed

- Object masks with more than 65,535 objects are saved as 32-bit images. Before, object 65,536 became background and higher labels overwrote other objects. Measure now stops with an error naming the plane and object instead of merging such objects into one.
- Mask generation deletes the raw `orig/` images only when every field's merged array is complete and readable. Before, one merged field was enough for `orig/` to be deleted. When fields are missing, `orig/` is kept and the missing fields are listed.
- `seg_qc='stop'` now stops the Mask run, in classic and V2 Mask runs, instead of printing "Segmentation QC skipped" and carrying on.
- Generate Masks works out each role's Cellpose channel from the current raw channel choices, so a channel position recorded by an earlier run can no longer send a role to the wrong channel.
- Image import never writes its output over the acquired source images, even when the output path is another spelling of the same folder.
- Opening a multichannel field keeps its paired masks.
- A preview refuses a cached model file whose checksum does not match its catalogue entry.
- Shutting down spaCR waits for unsafe background work instead of closing underneath it, and a closed spaCR process now ends instead of hanging at exit.
- A corrupt TIFF is reported as unreadable instead of raising an unexpected error.
- A newer pending QC move is no longer overwritten when an older one finishes, and pending image loads are discarded before their source files are re-sorted.
- A new figure in the figure queue no longer reselects an older figure that had the same id.
- Regression says fits are queued only after the queue has accepted them.
- A pivot with the same column on both axes shows a notice instead of an error.

### Alpha and remaining work

Experimental features are still labelled as alpha and hidden behind Preferences > Show alpha features, which is off by default. This release adds no new alpha features. For example, the OPS switch in Mask is alpha and appears only with Show alpha features on.

The anonymous diagnostics for #175 to #178 help identify the cause of those reports; they are not a confirmed fix for it. I have not yet confirmed the cause of a reported cluster run in which later masks appeared to use the nucleus model; the channel safeguard listed under Fixed is in place, and the investigation continues. Swapping two cell IDs in Time mode is planned for a later release.
