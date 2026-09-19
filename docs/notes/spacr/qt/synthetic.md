# Notes from `spacr/qt/synthetic.py`

Prose lifted out of `spacr/qt/synthetic.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [cellvoyager_filename](#cellvoyager_filename) (1 entry)
- [Module level](#module-level) (4 entries)
- [_synth_blob_image](#_synth_blob_image) (1 entry)
- [_synth_field](#_synth_field) (7 entries)
- [_mask_roles](#_mask_roles) (1 entry)
- [_emit_images](#_emit_images) (1 entry)
- [generate_classify_demo](#generate_classify_demo) (3 entries)
- [generate_timelapse_demo](#generate_timelapse_demo) (1 entry)
- [_channel_settings](#_channel_settings) (1 entry)
- [demo_settings](#demo_settings) (21 entries)
- [_phred_run](#_phred_run) (1 entry)
- [_fastq_header](#_fastq_header) (1 entry)

## cellvoyager_filename

### lines 98-100  _(unsure)_

```python
def cellvoyager_filename(
```

Filename builder — matches spacr's cellvoyager regex

## Module level

### lines 144-145  _(unsure)_

```python
CHANNEL_LAYOUT = {
```

Channel layout every mask default expects. Keys are the settings names in spacr.settings that pipeline functions read.

### lines 1141-1143  _(unsure)_

```python
FASTQ_READ_LENGTH = 150
```

Synthetic FASTQ generator — matches EO1_R1_001.fastq.gz structure

### lines 1145-1149

```python
FASTQ_READ_LENGTH = 150
```

NovaSeq X read layout observed in EO1_R1_001.fastq.gz:

header: @<instr>:<run>:<flowcell>:<lane>:<tile>:<x>:<y> 1:N:0:<i7> seq   : 150 bp qual  : 150 bp of Illumina 1.8+ Phred+33 scores Every read of the real fastq carried i7 index GCTTGCGC.

### lines 1172-1192

```python
SEQ_TARGET = "TGCTGTTTCCAGCATAGCTCTTAAAC"
```

the read frame the shipped barcode-mapping defaults expect

spacr.settings.set_default_generate_barecode_mapping anchors on `target_sequence`, slices `window_length` bases starting `offset_start` from the anchor, and splits that window with DEFAULT_BARCODE_REGEX:

^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<rowID>.{8}).*

So the 89-base window has to be laid out exactly like this, and it is:

[ 0: 8]  column barcode        8

[ 8:34]  SEQ_TARGET           26   <- the anchor; supplies TGCTG…TAAAC [34:55]  gRNA barcode         21 [55:60]  SEQ_GRNA_SUFFIX       5   <- the AACTT the regex demands [60:68]  SEQ_FILL              8 [68:73]  SEQ_ROW_PREFIX        5   <- the AGAAG the regex demands [73:81]  row barcode           8 [81:89]  SEQ_TAIL              8

and the anchor sits `-offset_start` = 8 bases into the window, which is why the column barcode is exactly 8 long.

## _synth_blob_image

### line 290  _(unsure)_

```python
r = blob_radius * (0.7 + 0.6 * rng.random())
```

Slight per-blob intensity + radius jitter

## _synth_field

### line 335, trailing  _(unsure)_

```python
cells: List[Tuple[float, float, float]] = []
```

(cy, cx, size_scale)

### lines 344-347

```python
cells.append((cy, cx, 0.82 + 0.36 * rng.random()))
```

Per-cell size jitter. Without it every object has the identical area, the median absolute deviation of the size distribution is zero, and seg_qc's robust range collapses to a point so that every object reads as a size outlier.

### line 357  _(unsure)_

```python
for role, sigma, radius in (
```

cell + nucleus: concentric, one per lattice site

### line 372  _(unsure)_

```python
path_mask = np.zeros(shape, dtype=np.uint16)
```

pathogens: inside the cell and clear of its nucleus

### lines 385-387

```python
angle = base_angle + 2 * np.pi * k / n
```

Evenly spaced around the cell centre, so several pathogens in one cell never land on top of each other and the mask holds n objects rather than one peanut.

### line 398  _(unsure)_

```python
org_mask = np.zeros(shape, dtype=np.uint16)
```

organelles: a rosette of puncta per cell

### lines 415-416

```python
for chan in sorted(wanted - set(images)):
```

Any channel the caller asked for that is not one of the four roles gets plain background rather than a KeyError further down.

## _mask_roles

### lines 423-425  _(unsure)_

```python
def _mask_roles(channels: Sequence[int]) -> List[str]:
```

Generators — one per app family

## _emit_images

### lines 460-461  _(unsure)_

```python
seed = _stable_seed(plate, well, f)
```

One seed per (well, field): every timepoint of that field is the same cells, drifting.

## generate_classify_demo

### lines 770-771  _(unsure)_

```python
cls = 1 if k % 2 == 0 else 2
```

Alternate blob patterns to give the classifier something to discriminate (label 1 = dense, label 2 = sparse).

### line 778

```python
arr8 = (arr / 256).astype(np.uint8)
```

Save as an 8-bit RGB PNG (what spacr.io stores).

### lines 781-783

```python
field = k // _CROPS_PER_FIELD + 1
```

Fields of _CROPS_PER_FIELD objects each, so the crops carry the same plate/well/field/time/label name spaCR parses metadata back out of.

## generate_timelapse_demo

### lines 882-884

```python
settings["timelapse_frame_limits"] = [0, times]
```

[start, end] is a *slice* of frame indices — spacr.object does `stack[limits[0]:limits[1]]`. [1, times] therefore silently threw away the first frame of every field; [0, times] keeps all of them.

## _channel_settings

### lines 897-899  _(unsure)_

```python
def _channel_settings(channels: Sequence[int]) -> Dict[str, Any]:
```

Settings — reverse-engineered per app so the demo actually runs

## demo_settings

### lines 949-953

```python
acquisition: Dict[str, Any] = {
```

Only the apps that ingest raw acquisition files parse filenames or care about the objective. Measure reads merged/*.npy, whose field names and plane layout are already fixed, so shipping these to it would be more keys the Measure screen has no widget for and measure_crop never reads — accepted, dropped, and impossible to notice.

### lines 965-968

```python
"cell_diameter": _RADIUS_CELL * 2,
```

The demo draws cells at _RADIUS_CELL and nuclei at _RADIUS_NUCLEUS; Cellpose 4 rescales by 30/diameter, so telling it the truth is what puts the objects near the size cpsam was trained on.

### lines 972-977

```python
"cell_background": _BACKGROUND,
```

The camera offset the images are actually drawn on. All three object channels, not just the cell: `*_background` is multiplied by `*_signal_to_noise` to set the normalisation ceiling, and a demo that declares the right offset for one channel and the 100 default for the other two normalises them differently for no reason.

### 2026-09-19, `cell_flow_threshold`

```python
"cell_flow_threshold": 0.4,
```

It was 1.0 from the first commit of this generator (4307d6299, 2026-07-21), and 1.0 was not a choice made for the synthetic data. It was the shipped default of that day. The same dict copied `cell_background` 100, signal-to-noise 10 and `cell_CP_prob` 0, which were all defaults then too. So the demo follows the shipped default, which is 0.4 since 428 (GitHub #123). Nucleus and pathogen are not set here and take the same default.

WHAT THAT COSTS THE DEMO: NOTHING, measured 2026-09-19 through spaCR's own Mask pipeline. `generate_mask_demo` (default `fields=2`, four fields), its settings CSV loaded the way the Demos menu loads it, then `preprocess_generate_masks`, on the GPU; the `gen_mask_settings.csv` each run wrote confirms the thresholds it used. Labels counted in the merged stacks, per field, as cell / nucleus / pathogen / organelle:

    flow thresholds 0.4 / 0.4 / 0.4 (since 428)      16/16/17/64  16/16/20/64  16/16/18/64  16/16/16/64
    flow thresholds 1.0 / 100 / 100 (before 428)     16/16/17/64  16/16/20/64  16/16/18/64  16/16/16/64

Every drawn cell and nucleus is kept in every field at 0.4, and the counts are identical to the old settings field by field. `tests/test_demo_pipelines.py::test_mask_demo_segments_every_object_it_drew` asserts the 16/16 and passes at 0.4.

A NUMBER THAT DOES NOT DESCRIBE THE DEMO, recorded so it is not taken for the demo's cost again. Calling stock `cpsam` directly on the demo's raw channel images (diameters 40 / 16 / 10, `cellprob_threshold` 0) does lose objects at 0.4: 4 of 16 cells in one field and a pathogen in three, where 1.0 and 100 keep them. That call skips spaCR's preprocessing and normalisation, so it is not what the demo runs, and it was briefly mistaken for the demo's cost while 428 was being built. On the Mask pipeline there is no reason to set 1.0 here to keep the drawn objects.

### lines 981-983

```python
"cell_signal_to_noise": 10,
```

Real key is capital-S `cell_signal_to_noise`. The demo shipped `cell_signal_to_noise` for a year: not a spaCR setting, so it was accepted, ignored, and the default used instead.

### lines 989-993

```python
"cell_model_name": "cpsam",
```

'cyto' / 'nuclei' until Cellpose 4 removed them. The demo settings are what a new user copies, so they name the model that exists; a legacy value in a real settings file is still accepted and mapped forward by settings.normalize_cellpose_model_name.

### lines 999-1001

```python
return {
```

No `*_channel` here: measure_crop indexes merged/*.npy with

`*_mask_dim` and never reads the raw acquisition channel keys, and the Qt Measure screen has no widget for them either.

### lines 1013-1015

```python
"png_size": [64, 64],
```

png_size is a [height, width] pair, not a scalar — a bare int is a hard pre-flight error ("png_size=64 is a int, but list is expected").

### lines 1017-1022

```python
"png_channel_mapping": {"r": 2, "g": 1, "b": 0},
```

The declared mapping, not the retired `png_dims` list. This value is identical to what `png_dims=[0, 1, 2]` always meant on screen -- the 405 plane in blue -- but it says so, and it is a key the Measure screen actually renders. Written as the old key it had no widget, so the demo's own setting was dropped on the floor when the pack was applied.

### lines 1024-1038

```python
}
```

No `normalize` / `normalize_by` here, deliberately. measure_crop reads `normalize` as a [low, high] percentile PAIR, but spacr.settings declares it ``bool`` and the Qt Measure screen therefore renders it as a Toggle: importing a demo that shipped `normalize=[1, 99]` put **False** in the form (`_apply_value` does `str(val).lower() in ("true","1","yes")`), so the CSV on disk and the form the user is looking at disagreed about how every crop is scaled. `normalize_by` alone is inert — measure.py only consults it when `normalize` is a list — so shipping it would be decoration. Omitting both leaves the measure defaults (normalize=False, normalize_by='png'), which is what the run does anyway, and nothing is silently rewritten. Making [1, 99] loadable needs a real widget for the pair in spacr/qt/screens/settings_model.py + a `(bool, list)` type in spacr/settings.py; neither is in this module.

### lines 1043-1051

```python
"plot": False,
```

Every other demo carries plot=False through `base`; classify does not spread `base` (its keys are a different set) and so shipped nothing, which meant `deep_spacr_defaults` supplied its own default of True. That is not cosmetic: with plotting on, `_plot_training_curves` runs at the end of every epoch, and outside the Qt GUI (where the bridge replaces `plt.show`) the interactive backend's blocking show parks the run in the Qt main loop forever — the classify demo never finished. The blocking show is fixed at source, and the demo says what it means here.

### lines 1053-1056

```python
"dataset_mode": "annotation",
```

The crops are labelled in png_list, not by well metadata, so the dataset has to be built in 'annotation' mode. The shipped default is 'metadata', which selects on the class column ('columnID') and would build two classes out of one well.

### lines 1058-1062

```python
"annotation_column": "annotate",
```

Only the singular key: generate_training_dataset falls back to `[settings['annotation_column']]` when `annotation_columns` is unset, and the plural spelling — which io.py reads — is not declared in spacr.settings, so shipping it makes pre-flight warn "did you mean 'annotation_column'?" on every demo load.

### lines 1064-1066

```python
"png_type": "cell_png",
```

`annotated_classes` is gone: nothing ever read it, so shipping it in a demo taught the shape of a setting that did nothing. The classes come from dataset_mode instead.

### lines 1073-1075

```python
"model_type": "resnet50",
```

'cnn' is not a model: model_type is fed to torchvision, and the GUI offers only names from that list. resnet50 is the smallest of them that trains sensibly on 64 px crops.

### lines 1077-1084

```python
"train_channels": ["r", "g", "b"],
```

`train_channels`, not `channels`. Classify runs spacr.deep_spacr.deep_spacr, which selects the crop's colour planes with `settings['train_channels']` (r/g/b letters) and never reads `channels` at all — `channels` is not even in deep_spacr_defaults, so the Classify screen has no widget for it and `_apply_demo_to_screen` dropped it on the floor. The demo's crops are a greyscale plane replicated into RGB, so all three planes carry signal.

### lines 1086-1090

```python
}
```

No `channel_of_interest`: it is a spacr.ml recruitment/regression setting (it picks `pathogen_channel_<n>_mean_intensity` columns), deep_spacr never reads it, and the Classify screen renders it as a QSpinBox — so the `None` this used to ship came back from the form as 3.

### lines 1098-1100

```python
"timelapse_frame_limits": [0, 8],
```

A slice, not an inclusive 1-based range: see generate_timelapse_demo, which overwrites this with the real frame count.

### lines 1103-1107

```python
"timelapse_mode": "iou",
```

'trackastra' is the shipped default and the better tracker, but it is an optional dependency: a machine without it cannot run the demo at all. 'iou' ships with spaCR, needs no tuning, and is exactly right for objects that drift a couple of pixels per frame — which is what this dataset is.

### lines 1116-1120

```python
"grna_csv": os.path.join(barcodes, "grna.csv"),
```

The three CSVs spacr.sequencing.map_sequences_to_names reads; each needs 'name' and 'sequence' columns. Leaving them unset is a hard pre-flight error, and the demo used to ship `barcode_length` / `barcode_offset` / `processes` — none of which is a spaCR setting.

### lines 1128-1130

```python
"window_length": SEQ_WINDOW_LENGTH,
```

`window_length` since 364's rename; the constant beside it already had the right word, which is the argument for the setting having it too.

## _phred_run

### lines 1229-1230  _(unsure)_

```python
tail = int(length * 0.33)
```

Fade quality toward the tail — real reads drop below Q20 near the end. Roughly halve the base quality over the last third.

## _fastq_header

### line 1353, trailing

```python
x = 1000 + (index % 9000)
```

1000..9999
