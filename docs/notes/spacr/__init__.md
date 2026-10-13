# Notes from `spacr/__init__.py`

Prose lifted out of `spacr/__init__.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 9-25

```python
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

APPLE METAL OPERATOR FALLBACK, AND IT HAS TO BE SET HERE.

Metal implements most but not all of torch's operators. A missing one raises NotImplementedError mid-run rather than degrading, which on the reporting iMac took cellpose down at `aten::upsample_linear1d` -- after the model had loaded and the image was already on the card. This flag turns those into a quiet CPU detour for the op that is missing.

TORCH READS IT WHEN THE MPS BACKEND REGISTERS, WHICH IS AT `import torch`. Setting it later has no effect at all -- measured: identical code fails when the variable is set after the import and succeeds when set before it. That is why this sits at the top of the package rather than in `spacr.accelerator`, which is imported far too late to matter.

setdefault, not assignment: a user who set it to 0 deliberately wants the hard failure, and that is a legitimate way to find out which operator is costing them a round trip. See instruction 319.

### lines 28-32

```python
from ._version import __version__
```

``spacr.version`` answers detailed environment/version queries and therefore imports ``importlib.metadata``.  That machinery was more than 60% of a clean installed ``import spacr`` even though the wheel already knows its version. The release helper keeps this literal synchronized with setup.py; callers that explicitly import ``spacr.version`` retain the metadata-backed API.

### lines 35-40

```python
_warnings.filterwarnings(
```

Third-party FutureWarnings that fire at import — noise the user can't act on from inside spaCR. Silenced before the modules that trigger them import. The Statsmodels warning formerly listed here was fixed at its source by switching from the deprecated ``logit`` alias to ``Logit``. (Users can re-enable with `warnings.filterwarnings("default")` in their own code.)

### lines 58-69

```python
_warnings.filterwarnings(
```

Cellpose 4 builds a sparse COO tensor in `dynamics.py` and torch notes that invariant checking is off. It fires on the first mask of every run, names a torch internal, and there is nothing a spaCR user can do about it.

Both patterns are written the way `spacr.qt._LIBRARY_NOISE` explains, and for the same two reasons. The message is not anchored: `filterwarnings` matches it with `re.match`, so the anchored version this replaced would have missed the same notice from any build of torch that prefixes it. The module IS given, and it is the raising frame's dotted `__name__` -- NOT the `cellpose/dynamics.py` path a traceback shows, which is the natural thing to write here and matches nothing. Scoping it to cellpose means the sentence is only ignored where it is noise.

### lines 77-79

```python
_DOCUMENTED_SUBMODULES: tuple[str, ...] = (
```

The submodules this package documents, in the order that groups them by what they are for. This is the frozen-bundle floor, not the whole list see `_SUBMODULES` below, which adds whatever else is on disk.

### lines 87-90

```python
"tabular",
```

One reader and one writer for every table spaCR opens: CSV, SQLite, Parquet, Feather, Excel. Normalises column names through `spacr.schema` on read, so the CSV picker and the run agree about what a column is called. pandas + sqlite3 only, so a picker can import it.

### lines 96-98

```python
"settings_spec",
```

Which widget a setting gets, decided without importing a GUI: the Tk and Qt front ends read the same spec rather than each keeping their own opinion about what a given key looks like.

### lines 103-105

```python
"measure_hooks",
```

Opt-in preprocessing / region-filter extension points for the measure path. Separate from `measure` so registering a hook does not import matplotlib, skimage and cv2.

### lines 107-109

```python
"roi",
```

A drawn region of interest, honoured by Measure. Pure geometry plus the mask it resolves to, so a headless run can apply an ROI drawn in the GUI without importing one.

### lines 111-113

```python
"illumination",
```

Illumination / flat-field correction. Estimates the microscope's uneven illumination from the plate's own fields and applies it through the preprocessing hook above, so measure.py needs no second path.

### lines 117-121

```python
"sequencing_qc",
```

QC over what `sequencing` produced — reads per well, starved wells, barcode collisions, unmapped reads, library coverage — plus the gRNAs-per-well target that derives the abundance threshold. Separate from `sequencing` so the multiprocessing read workers do not import the plotting and statistics only the post-run analysis needs.

### lines 124-125  _(unsure)_

```python
"lineage",
```

cell → nucleus → pathogen, read as the tree the `cell_id` links in measurements.db already describe. Query-only; it adds no column.

### lines 148-149  _(unsure)_

```python
"intensity_rescale",
```

Plate-wide intensity rescaling provenance and the desktop installer's hardware/consent hand-off are both public, dependency-light modules.

### lines 165-166  _(unsure)_

```python
"curation",
```

Correcting a mask and a track by hand, on the record: every edit is journalled so a curated result still says where it came from.

### lines 173-174

```python
"image_stitch",
```

The other half of reading a folder of images: a field that arrived as tiles is put back together, so spaCR's filename needs no tile slot.

### lines 181-183

```python
"confusion",
```

The confusion matrix as a set of live queries rather than a picture — "which objects are in this cell" is answerable, so a misclassified object can be opened instead of counted.

### lines 194-196

```python
"object_roles",
```

The one registry of what object kinds exist. Eleven modules used to spell the vocabulary out independently and now derive from this, so it is imported by nearly everything that touches a mask.

### lines 199-205

```python
"organelle_types",
```

The organelle presets: one cell-biology choice that fills in the fifty-three organelle settings a user would otherwise have to reason about. `settings`, `settings_spec` and `measure` all import it, so it is part of the surface whether or not it is listed here — and `test_smoke.py::test_lazy_loader_matches_files` is what says so. It was added without this line, which turned every cell of `compat-matrix` red on the same assertion.

### lines 207-210

```python
"ome_zarr",
```

Image I/O against the two standards a lab is most likely to already be keeping plates in. Both sit behind optional extras, so importing either without its dependency names the `pip install "spacr[...]"` that fixes it rather than raising six frames deep.

### line 216  _(unsure)_

```python
"doctor",
```

Whole-installation diagnosis behind `spacr-doctor`.

### lines 218-220

```python
"crashreport",
```

`spacr-crashreport`: everything a maintainer needs about a failed run in one attachable file, so a bug report is a file rather than a remembered traceback.

### lines 223-227

```python
"cli_download",
```

`spacr-download`: the published example data as a command instead of a button, so a cluster login node can stage what a batch job will read. `example_archives` is the Qt-free half of the GUI's own downloader which repositories publish what, and how an archive is fetched, unpacked and repaired -- shared by both.

### lines 237-240

```python
"layers",
```

A napari-style layer model — images, labels, points and shapes in one world — and manual counting on top of a points layer. Both are plain data models with no Qt in them, so the viewer is a renderer of a state that tests (and notebooks) can build directly.

### lines 243-246

```python
"napari_bridge",
```

The napari bridge: a field's image and mask out to napari, the corrected labels back, written the way spaCR writes masks and recorded in the same append-only curation ledger the brush uses. napari is an optional extra and is never imported at module scope.

### lines 248-249  _(unsure)_

```python
"selection",
```

The shared filter/selection model the linked views are built on. Pure pandas, no Qt, so it is usable headless and from a notebook too.

### line 251

```python
"regression_qc",
```

Diagnostic figures for a fitted regression.

### lines 254-256

```python
"regression_summary",
```

spaCR's OWN summary of a run, for the regression types statsmodels writes none for and for the permutation path, which has no fitted model at all. Every field is a number or a stated reason.

### lines 258-259  _(unsure)_

```python
"localisation",
```

Where each gene's protein lives, for colouring ONE compartment against grey. Pure pandas: the join belongs to the screen, not to the picture.

### lines 261-264

```python
"figure_sink",
```

Saved and visible are the SAME event. A module hands a figure here and it is written AND announced -- through no pyplot registry, so a figure built as a bare matplotlib.figure.Figure reaches the GUI too. That was the whole ~19-panel regression QC report, on disk and invisible.

### lines 266-267  _(unsure)_

```python
"control_names",
```

Canonical control identifiers and the optional example-data downloader are dependency-light public helpers used by both the GUI and notebooks.

### lines 271-273

```python
"columns",
```

A column that is not there offers the columns that are. Reads only the header row, so it can populate a GUI dropdown on the GUI thread from a score CSV that is hundreds of megabytes.

### lines 275-279

```python
"annotation",
```

Everything spaCR knows about a Toxoplasma gene, joined onto an export by gene NUMBER. Next to `localisation` because it is the same join widened from one compartment to the whole annotation, and separate from `toxo` because that module draws figures and this one only reads the five bundled CSVs.

### lines 281-284

```python
"regression_backends",
```

WHO fits the model, as opposed to WHICH model is fitted. `mixed_gpu` is the profiled REML objective in torch, for the fit that dominates a screen's runtime; `regression_backends` is the inventory the panel offers and greys.

### lines 287-291

```python
"rra",
```

The two backends that answer "which genes are involved" WITHOUT ever forming `gene_fraction` -- the sum of a gene's guide fractions, which makes a guide-and-gene design singular by construction. `rra` ranks guides and aggregates by rank (MAGeCK alpha-RRA); `group_lasso` penalises a gene's guides as a block. Pure numpy/scipy, no Qt, no ml.

### lines 294-297

```python
"baseline",
```

What an effect size is measured FROM, and the sentence that says so. Separate from `figures` because the answer belongs to the fit, not to the picture: the console summary and the exported stats table state the same baseline the panel does.

### lines 299-301

```python
"cell_montage",
```

The cells behind a coefficient: which objects a dot on the volcano is most consistent with. Pure pandas -- the montage is a Qt tab, but WHICH objects to show is a question about the screen, not about a widget.

### lines 303-305

```python
"plate_measurements",
```

The headless half of the per-plate measurements merge: {plate: db} plus the chosen tables in, one merged frame out, by CALLING multi_database and merge_tables rather than aggregating anything itself.

### lines 307-309

```python
"thresholds",
```

How wide a coefficient has to be before it counts as a hit. Seven ways of measuring the control spread, in one place so the run and the plot's right-click menu cannot offer different ones.

### lines 311-312

```python
"regression_spec",
```

Which regression backends exist and which settings each one reads. Pure data; imports NOTHING, which is why it is not part of ml.

### lines 314-316

```python
"refit",
```

Building the settings for a second run of the same screen through a different model. No Qt: the GUI offers the gesture, but what a re-fit is allowed to change is a question about the fit, not about a menu.

### lines 318-322

```python
"power_simulate",
```

The spaCRPower port: `power_simulate` generates a synthetic pooled screen, `power_model` fits the horseshoe-Poisson hit model to it. They are separate modules because the simulator is cheap and dependency-free while the model pulls in torch, and a parameter sweep re-runs the first far more often than the second.

### lines 325-328

```python
"hits",
```

The ranked, annotated, filterable deliverable of a screen, and the interrogation of the model that ranked it: move one input, watch the prediction move. `hits` is what a collaborator receives; `profiler` is how you decide whether to believe it.

### lines 334-336

```python
"runctx",
```

One id, one seed, one error policy, for a whole run. Everything that records anything about a run reads it from here rather than growing a second opinion about which run it is in.

### lines 339-340  _(unsure)_

```python
"run_compare",
```

Two runs of the same project side by side: what changed in the settings, how many fewer objects, which hits moved.

### lines 342-346

```python
"macro",
```

The macro recorder: every run also emits the Python script that repeats it — real imports, a real settings dict, a real call — with the run id, the settings hash and a machine-readable record of what was chosen versus defaulted. `run_journal` hooks it; nothing else needs to know it exists.

### lines 349-350

```python
"methods_export",
```

Methods and Results sections written from a run digest, so the prose cannot drift from the settings that produced the numbers.

### lines 356-359

```python
"ports",
```

The pipeline contract. `ports` declares what each module consumes and produces and answers "can this module run here?" before a run starts; `artifacts` records what produced every file, so "is this result still current?" has an answer. Both are dependency-light on purpose.

### lines 362-365

```python
"pipeline_graph",
```

Built directly on that record: `pipeline_graph` is the DAG of what produced what with staleness marked, and `chaining` is the same graph read forwards — a module's inputs default to where the last run *actually* wrote rather than to a path retyped by hand.

### lines 368-370

```python
"data_manager",
```

Disk accounting built on those two: what a project costs per artifact kind, what of it is regenerable and may therefore be pruned, and archiving that leaves the registry knowing where the data went.

### lines 372-375

```python
"projects",
```

Every project on disk in one list — stage reached, size, last run and what is stale — assembled from `ports`, `artifacts`, `data_manager` and `chaining` rather than re-derived. A project the registry has never seen is listed too, and is reported as unexamined rather than clean.

### lines 380-386

```python
"classify",
```

The Classify overhaul and the Gate Editor added these and the lazy loader was not told. `test_lazy_loader_matches_files` caught it on every CI platform: a module that exists as a file but is not listed cannot be reached as `spacr.<name>`, so `import spacr; spacr.filters` raised AttributeError while `from spacr import filters` worked which is the kind of split that gets diagnosed as "sometimes the import fails".

### lines 403-407

```python
"multiple_testing",       # every FDR / FWER correction, in one place
```

The regression surface, added over 2026-08-15/16. A module missing from this tuple is not reachable as `spacr.<name>` at all -- the lazy loader is the only path -- so leaving one out ships a module nobody outside the package can import, and `test_lazy_loader_matches_files` exists to catch exactly that.

### line 408, trailing  _(unsure)_

```python
"multiple_testing",
```

every FDR / FWER correction, in one place

### line 409, trailing  _(unsure)_

```python
"volcano_style",
```

the volcano's thresholds and their rules

### line 410, trailing  _(unsure)_

```python
"guide_concordance",
```

do a gene's own guides agree in direction

### line 411, trailing  _(unsure)_

```python
"regression_diagnostics",
```

design, residual and inference panels

### line 412, trailing  _(unsure)_

```python
"regression_search",
```

the dependent-variable search

### line 413, trailing  _(unsure)_

```python
"metadata_resolution",
```

which metadata column is which

### line 414, trailing  _(unsure)_

```python
"multi_database",
```

read and merge several measurement databases

### line 415, trailing  _(unsure)_

```python
"measurement_scan",
```

which measurement has genes with an effect

### lines 416-419

```python
"gene_facts",
```

Everything known about ONE gene, gathered from the bundled annotation and the screen's own table. `gene_facts` answers the question and `gene_tile` renders it; both are pure pandas, so the Gene tab can be built headless and tested without Qt.

### line 421, trailing  _(unsure)_

```python
"gene_tile",
```

everything spaCR knows about one gene

### line 426, trailing  _(unsure)_

```python
"parameter_sweep",
```

the settings sweep and its containment

### line 427, trailing  _(unsure)_

```python
"sweep_child",
```

one contained trial, exec'd in its own cgroup

### line 428, trailing  _(unsure)_

```python
"trial_metrics",
```

what makes a sweep row judgeable

### line 429, trailing  _(unsure)_

```python
"workspace",
```

saved GUI context around a recorded run

### line 430, trailing  _(unsure)_

```python
"figure_style",
```

the older per-figure style store

### line 431, trailing  _(unsure)_

```python
"style_base",
```

shared values for current figure renderers

### lines 432-433

```python
"dependent_join",
```

Dependency-light helpers shared by the regression, classification, and streamed-image interfaces.
