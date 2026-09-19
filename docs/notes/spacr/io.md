# Notes from `spacr/io.py`

Prose lifted out of `spacr/io.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (9 entries)
- [display](#display) (1 entry)
- [migrate_unescaped_plate_names](#migrate_unescaped_plate_names) (4 entries)
- [process_non_tif_non_2D_images.save_grayscale_images](#process_non_tif_non_2d_imagessave_grayscale_images) (2 entries)
- [process_non_tif_non_2D_images.split_channels](#process_non_tif_non_2d_imagessplit_channels) (5 entries)
- [process_non_tif_non_2D_images.load_image](#process_non_tif_non_2d_imagesload_image) (1 entry)
- [process_non_tif_non_2D_images.convert_grayscale_to_tiff](#process_non_tif_non_2d_imagesconvert_grayscale_to_tiff) (1 entry)
- [process_non_tif_non_2D_images](#process_non_tif_non_2d_images) (3 entries)
- [_load_images_and_labels](#_load_images_and_labels) (2 entries)
- [_load_normalized_images_and_labels](#_load_normalized_images_and_labels) (8 entries)
- [CombineLoaders.__init__](#combineloaders__init__) (1 entry)
- [CombineLoaders.__next__](#combineloaders__next__) (2 entries)
- [NoClassDataset.__init__](#noclassdataset__init__) (1 entry)
- [spacrDataset.__init__](#spacrdataset__init__) (4 entries)
- [spacrDataset.load_image](#spacrdatasetload_image) (1 entry)
- [spacrDataLoader.__init__](#spacrdataloader__init__) (1 entry)
- [spacrDataLoader._preload_next_batches](#spacrdataloader_preload_next_batches) (2 entries)
- [spacrDataLoader.__iter__](#spacrdataloader__iter__) (1 entry)
- [spacrDataLoader.cleanup](#spacrdataloadercleanup) (1 entry)
- [TarImageDataset.__init__](#tarimagedataset__init__) (1 entry)
- [_rename_and_organize_image_files](#_rename_and_organize_image_files) (11 entries)
- [_merge_file](#_merge_file) (4 entries)
- [_generate_time_lists](#_generate_time_lists) (4 entries)
- [_move_to_chan_folder](#_move_to_chan_folder) (5 entries)
- [_merge_channels](#_merge_channels) (2 entries)
- [_concatenate_channel](#_concatenate_channel) (8 entries)
- [_normalize_img_batch](#_normalize_img_batch) (8 entries)
- [_concatenate_and_normalize_impl](#_concatenate_and_normalize_impl) (11 entries)
- [concatenate_and_normalize](#concatenate_and_normalize) (1 entry)
- [_get_lists_for_normalization](#_get_lists_for_normalization) (2 entries)
- [_normalize_stack](#_normalize_stack) (3 entries)
- [_create_movies_from_npy_per_channel](#_create_movies_from_npy_per_channel) (6 entries)
- [delete_empty_subdirectories](#delete_empty_subdirectories) (4 entries)
- [select_fields](#select_fields) (1 entry)
- [preprocess_img_data](#preprocess_img_data) (12 entries)
- [_check_masks](#_check_masks) (1 entry)
- [_save_figure](#_save_figure) (1 entry)
- [_read_and_join_tables](#_read_and_join_tables) (7 entries)
- [_save_settings_to_db](#_save_settings_to_db) (7 entries)
- [_mask_movie_frame_geometry](#_mask_movie_frame_geometry) (1 entry)
- [_save_mask_timelapse_as_gif](#_save_mask_timelapse_as_gif) (5 entries)
- [_save_mask_timelapse_as_gif._update](#_save_mask_timelapse_as_gif_update) (9 entries)
- [_save_object_counts_to_database._count_objects](#_save_object_counts_to_database_count_objects) (1 entry)
- [_save_object_counts_to_database](#_save_object_counts_to_database) (1 entry)
- [_create_database](#_create_database) (1 entry)
- [_save_array_atomic](#_save_array_atomic) (2 entries)
- [_load_and_concatenate_arrays](#_load_and_concatenate_arrays) (14 entries)
- [read_plot_model_stats._plot_and_save](#read_plot_model_stats_plot_and_save) (3 entries)
- [read_plot_model_stats](#read_plot_model_stats) (4 entries)
- [_save_model](#_save_model) (2 entries)
- [_save_progress._save_df_to_csv](#_save_progress_save_df_to_csv) (1 entry)
- [_save_progress](#_save_progress) (4 entries)
- [_read_db](#_read_db) (8 entries)
- [_read_and_merge_data](#_read_and_merge_data) (2 entries)
- [convert_numpy_to_tiff](#convert_numpy_to_tiff) (4 entries)
- [generate_cellpose_train_test](#generate_cellpose_train_test) (1 entry)
- [parse_gz_files](#parse_gz_files) (2 entries)
- [open_crop_source](#open_crop_source) (1 entry)
- [LazyCropPNG._stream](#lazycroppng_stream) (1 entry)
- [mark_crop_output_folder](#mark_crop_output_folder) (1 entry)
- [generate_dataset](#generate_dataset) (10 entries)
- [_write_crop_tar](#_write_crop_tar) (1 entry)
- [expected_sampled_fractions](#expected_sampled_fractions) (1 entry)
- [format_class_balance_report](#format_class_balance_report) (1 entry)
- [make_cv_folds](#make_cv_folds) (5 entries)
- [make_validation_holdout](#make_validation_holdout) (1 entry)
- [_classification_data_dir](#_classification_data_dir) (2 entries)
- [_classification_transform](#_classification_transform) (2 entries)
- [generate_cv_loaders](#generate_cv_loaders) (3 entries)
- [generate_loaders](#generate_loaders) (9 entries)
- [generate_training_dataset](#generate_training_dataset) (21 entries)
- [generate_training_dataset._ensure_unique_dir](#generate_training_dataset_ensure_unique_dir) (1 entry)
- [generate_training_dataset._load_png_table](#generate_training_dataset_load_png_table) (1 entry)
- [generate_training_dataset._fix_path_under_src](#generate_training_dataset_fix_path_under_src) (1 entry)
- [generate_training_dataset._annotation_classes_from_columns](#generate_training_dataset_annotation_classes_from_columns) (12 entries)
- [training_dataset_from_annotation](#training_dataset_from_annotation) (7 entries)
- [training_dataset_from_annotation_metadata](#training_dataset_from_annotation_metadata) (7 entries)
- [generate_dataset_from_lists](#generate_dataset_from_lists) (9 entries)
- [convert_separate_files_to_yokogawa](#convert_separate_files_to_yokogawa) (10 entries)
- [convert_to_yokogawa](#convert_to_yokogawa) (22 entries)
- [prepare_cellpose_dataset](#prepare_cellpose_dataset) (3 entries)
- [_listdir_visible](#_listdir_visible) (1 entry)

## Module level

### line 3, trailing  _(unsure)_

```python
import readlif.reader
```

`import readlif` alone does not bind the submodule

### lines 47-48  _(unsure)_

```python
pyczi = None
```

Backward-compatible injection point used by tests and advanced callers. ``None`` means "load the optional reader on first CZI conversion".

### lines 51-53

```python
from .errors import RunLedger
```

Fail-loud accounting. Every per-file skip below is recorded on a RunLedger so a batch that lost 40 of 384 files says so at the end and stamps the artifact it produced, instead of writing a silently-short result.

### lines 60-64

```python
from . import convert as _cv
```

One definition of what a well is called. spacr.convert imports nothing heavier than spacr.schema, so this costs nothing here, and it is the reason the two Yokogawa converters below can name a well on a 1536-well plate: they used to carry three hand-written copies of "ABCDEFGHIJKLMNOP" and range(1, 25), which stop at P24.

### lines 68-70

```python
from .png_list import (PNG_LIST_ID_COLUMNS, _merged_field_paths,
```

RE-EXPORTED, NOT DEFINED HERE ANY MORE. These need pandas and sqlite3 and nothing else, and every caller that imported them from this module paid torch + torchvision + cv2 for the privilege. See spacr.png_list.

### lines 76-78

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 3643-3654

```python
MASK_MOVIE_DPI = 100
```

A tracked-mask movie is sized from the mask, not from a constant.

It used to open ``plt.subplots(figsize=(50, 50))`` and write the animation at ``dpi=80``, so every frame was 50 x 80 = 4000 px on a side whatever the field was: 64 MB of RGBA per frame for a mask that is usually a few hundred pixels across. Measured on a 128 x 128 mask, five frames took 8 s and came out 4000 x 4000. The movie is written whenever ``save`` is true on any of the three tracking backends and a real timelapse is tens to hundreds of frames, so the cost was paid on every tracking run and grew with the run's length rather than with the field's size. It is also why the three ``if plot or save:`` call sites in spacr.timelapse had no test that let them run: writing one real movie cost seconds and hundreds of megabytes.

### lines 5242-5270

```python
CROP_OBJECT_TYPES = (
```

On-demand crops

:mod:`spacr.crops` cuts a single object straight out of ``merged/*.npy`` the array already holds both the intensity planes and the integer label-mask planes, so the crop the PNG folder holds can be reproduced on demand, pixel for pixel, without the folder existing. Until now only the Qt Annotate screen was wired to it; the image UMAP and the Classify dataset builders still required a pre-generated folder, which costs disk, has to be regenerated whenever a crop setting changes, and goes stale silently.

Everything below is the seam those consumers use. It is deliberately additive**: ``crop_source='png'`` (and ``'auto'`` on any project that has a crop folder) behaves exactly as before, byte for byte.

Two rules hold everywhere in this section:

1. A crop is only ever produced by :mod:`spacr.crops`. On-demand crops go through ``CropSource.get`` -> ``crops.png_view``; crops read back off disk go through ``crops.read_crop_png``. Neither path re-implements the channel handling, and neither goes around the format versioning added in 341f446. 2. Any folder of crop PNGs spaCR *writes* is stamped with the crop-format sidecar before it is filled -- with the current (RGB) format when the crops were cut here, and with the SOURCE folder's format when they were byte-copied out of one. An unmarked folder means legacy, so leaving a freshly written folder unmarked is the one mistake that silently reverses everything downstream.

### lines 6143-6158

```python
CLASS_BALANCE_MODES = ('none', 'weighted_sampler', 'sqrt_weighted_sampler', 'weighted_loss')
```

Class-imbalance handling and cross-validation splitting

Both live here because both change *how the training data is split and weighted* — the one place that decision is made is generate_loaders.

Two rules are load-bearing and are enforced by tests:

1. A WeightedRandomSampler is only ever attached to the TRAIN loader. Resampling validation or test data changes the class prior the metrics are measured against, so a "balanced" accuracy would no longer describe the real screen. 2. Cross-validation folds are group-aware by default. Crops taken from the same well (or field, or plate) share illumination, focus, seeding density and edge effects; splitting them across folds lets the model recognise the well rather than the phenotype and inflates every score.

## display

### lines 15-18

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## migrate_unescaped_plate_names

### lines 190-197

```python
if len(parts) <= 4:
```

Three fixed tail tokens -- well, field, time -- so anything beyond four components is a plate holding a raw separator. Testing THAT rather than "does escaping change the name" is what makes a second run a no-op: escaping is not idempotent, because a literal percent is escaped first, and `exp%5F1_A01_1_1` would become `exp%255F1_A01_1_1`. A migration that corrupts on its second run is worse than one that never ran.

### lines 201-202  _(unsure)_

```python
safe = escape_field_stem_plate(stem, timelapse=True)
```

timelapse=True is the writer's grammar, not the run's setting: three fixed tail tokens either way.

### lines 205-207

```python
continue
```

Not a field stem. A folder can hold a sidecar or a hand-dropped array, and renaming one on a guess is how a migration loses a file.

### lines 209-220

```python
planned.append((os.path.join(base, name),
```

NO `safe == stem` GUARD. Reaching it needs a stem with more than four components that escaping leaves unchanged, and there is no such stem: more than four components means the plate holds a separator, and escaping a separator always changes the string. Checked against 20,000 random plate names as well as argued every one changed.

The idempotency the comment above is about comes from the `len(parts) <= 4` test, not from this one: a stem that has already been migrated has four components and never reaches here at all.

## process_non_tif_non_2D_images.save_grayscale_images

### line 270  _(unsure)_

```python
def save_grayscale_images(image, base_name, folder, dtype, channel=None, z=None, t=None):
```

Helper function to save grayscale images

### lines 290-291  _(unsure)_

```python
suffix = f"_C{channel}"
```

Every splitter call supplies its 1-based channel index; keeping a channel-less arm here only made two planes able to share a name.

## process_non_tif_non_2D_images.split_channels

### line 301  _(unsure)_

```python
def split_channels(image, folder, base_name, dtype):
```

Function to handle splitting of multi-dimensional images into grayscale channels

### line 318  _(unsure)_

```python
return
```

Grayscale image, already processed separately

### line 322  _(unsure)_

```python
for c in range(image.shape[2]):
```

3D image: (height, width, channels)

### line 327  _(unsure)_

```python
for z in range(image.shape[3]):
```

4D image: (height, width, channels, Z-dimension)

### line 333  _(unsure)_

```python
for t in range(image.shape[4]):
```

5D image: (height, width, channels, Z-dimension, Time)

## process_non_tif_non_2D_images.load_image

### lines 356-360

```python
image = np.array(Image.open(file_path))
```

Return a numpy dtype like every sibling branch. Returning PIL's mode string here fed image.astype('RGB') -> TypeError (swallowed, so multi-channel PNG/JPEG were silently dropped), and astype('L') -> uint64, inflating 8-bit greyscale 8x despite the "bit depth is preserved" contract.

## process_non_tif_non_2D_images.convert_grayscale_to_tiff

### line 377  _(unsure)_

```python
def convert_grayscale_to_tiff(image, filename, folder, dtype):
```

Function to check if an image is grayscale and save it as a TIFF if it isn't already

## process_non_tif_non_2D_images

### line 399  _(unsure)_

```python
ledger = RunLedger('process_non_tif_non_2D_images')
```

Loop through all files in the folder

### line 420

```python
base_name = os.path.splitext(filename)[0]
```

Otherwise, split channels and save images

### line 424

```python
ledger.finalize()
```

Last thing on screen, so a partial conversion cannot scroll past.

## _load_images_and_labels

### lines 429-431

```python
"""Load a Cellpose training set, keeping each name beside its pixels.
```

Cellpose 4 no longer exposes submodules as attributes of the package root. Import the IO boundary explicitly, and only when this Cellpose dataset helper is used.

### lines 455-467

```python
images = []
```

THE NAMES ARE BUILT BESIDE THE PIXELS, one append each, never separately. They used to be `sorted(basename(f) for f in image_files)` while `images` was filled in the CALLER's order, and the caller shuffles: `spacr_cellpose.py:169` and `:296` do `random.shuffle(all_image_files)` before calling here. `identify_masks_finetune` then writes each mask as `os.path.join(dst, image_names[file_index])` over `enumerate(images)`, so every mask landed under a DIFFERENT image's filename -- a whole plate of segmentations silently attributed to the wrong wells.

Sorting was only half of it. Each loop below `continue`s past a file that will not read, which shortened `images` while the precomputed name list kept every entry, so one unreadable file misnamed every mask after it even when the input was already in order.

## _load_normalized_images_and_labels

### line 561  _(unsure)_

```python
if isinstance(percentiles, list) and len(percentiles) == 2:
```

Ensure percentiles are valid

### line 587  _(unsure)_

```python
for i, img_file in enumerate(image_files):
```

Load, normalize, and resize images

### line 595  _(unsure)_

```python
if channels is not None and image.ndim == 3:
```

Select specific channels if needed

### line 605  _(unsure)_

```python
if percentiles is None:
```

Calculate percentiles if not provided

### line 611  _(unsure)_

```python
for percentile in [98, 99, 99.9, 99.99, 99.999]:
```

Ensure `signal_thresholds` and `p` are floats for comparison

### line 618  _(unsure)_

```python
if target_height and target_width:
```

Resize image if required

### line 625  _(unsure)_

```python
if percentiles is None:
```

Calculate average percentiles if needed

### line 646  _(unsure)_

```python
if label_files is not None:
```

Load and resize labels if provided

## CombineLoaders.__init__

### lines 675-677

```python
self.loader_iters = [(i, iter(loader))
```

Carry the ORIGINAL loader index alongside each iterator: the list is shuffled and pruned as loaders empty, so a positional index would not identify which loader a batch came from.

## CombineLoaders.__next__

### lines 693-697

```python
continue
```

Exhausted: it sits at a position before `pos` and is dropped below. Do NOT pop here — mutating the list while enumerating it skips the entry that shifts into the freed slot, which silently discarded still-live loaders and truncated the combined stream.

### line 702  _(unsure)_

```python
self.loader_iters = []
```

Every remaining loader raised StopIteration on this pass.

## NoClassDataset.__init__

### lines 756-759

```python
self.filenames = [
```

Hidden files are not images. A crop folder carries a

`.spacr_crop_format.json` sidecar (spacr.crops), and a folder that has been near a Mac or Windows carries .DS_Store / Thumbs.db; every one of them used to be handed to Image.open as a sample.

## spacrDataset.__init__

### lines 840-844

```python
if not os.path.isdir(class_path):
```

A class folder that was never created is a real condition, not a crash: generate_training_dataset only makes a folder for a class it actually selected rows for. Skip it so the empty-dataset guard below can report every class at once, rather than dying on os.listdir of the first missing one.

### lines 847-850

```python
class_files = [os.path.join(class_path, f) for f in os.listdir(class_path)
```

Hidden files are not samples: a class folder written by generate_dataset_from_lists carries the crop-format sidecar `.spacr_crop_format.json`, and any folder that has been near a Mac carries .DS_Store. Both used to reach Image.open.

### lines 857-864

```python
if not self.filenames:
```

An empty dataset must say so HERE, where the directory and the class names are still in hand. Left to run on, shuffle_dataset() does `zip(*[])` and dies with "not enough values to unpack (expected 2, got 0)" -- which tells the user nothing about which folder was empty or which classes were looked for. This is reachable whenever generate_training_dataset selects no rows: a class_metadata value that matches nothing, an annotation column with no positives, or a filter that removed everything.

### lines 889-891

```python
workers = min(len(self.filenames), 32)
```

PIL decoding is I/O-heavy and releases the GIL. Threads avoid forking a live Qt/PyTorch process (unsafe and warned against by Python 3.13) while still overlapping disk reads and decoding.

## spacrDataset.load_image

### lines 913-915

```python
with Image.open(img_path) as source:
```

Force decoding while the file is open, then detach the returned image.  Leaving PIL's lazy decoder/file handle alive in a background prefetch thread can race interpreter/loader cleanup.

## spacrDataLoader.__init__

### lines 966-973

```python
self.batch_queue = queue.Queue(maxsize=max(1, preload_batches))
```

NOTE: the preloader used to run in a multiprocessing.Process writing into a multiprocessing.Queue, and __next__ stopped as soon as `not process.is_alive() and queue.empty()`. Both are unreliable: an mp.Queue can still have data in flight after the child exits, and the child only ever advanced its OWN copy of the iterator. The loader therefore silently yielded a truncated stream (0 of 4 batches in testing). A daemon THREAD shares the iterator, needs no pickling, and a sentinel gives an unambiguous end-of-stream signal.

## spacrDataLoader._preload_next_batches

### lines 1006-1008

```python
self._error = e
```

Hand the failure to the consumer instead of swallowing it: a collate/decode error used to look identical to "dataset is empty", which silently trains a model on no data.

### lines 1012-1014

```python
while not stop_signal.is_set():
```

On normal exhaustion the consumer needs a sentinel. During cleanup it is deliberately omitted: a full bounded queue must never strand the producer while the owner is joining it.

## spacrDataLoader.__iter__

### lines 1041-1045

```python
if self._iteration_active:
```

``list(iter(loader))`` calls ``iter`` twice: once explicitly and once inside ``list``. Iterators must return themselves without restarting while active, otherwise the abandoned first producer can decode/pin batches that are never yielded and the stream does twice the work. A fresh pass still starts after exhaustion or cleanup.

## spacrDataLoader.cleanup

### lines 1093-1094  _(unsure)_

```python
deadline = time.monotonic() + 5
```

Drain repeatedly while joining: the producer may finish a decode after the first drain and briefly refill the bounded queue.

## TarImageDataset.__init__

### line 1138  _(unsure)_

```python
from . import crops
```

Open the tar file just to build the list of members

## _rename_and_organize_image_files

### line 1193  _(unsure)_

```python
def _rename_and_organize_image_files(src, regex, batch_size=100, metadata_type='', img_format='.t...
```

@log_function_call

### line 1233, trailing  _(unsure)_

```python
all_filenames = [f for f in all_filenames if not f.startswith('.')]
```

Exclude hidden files

### line 1236  _(unsure)_

```python
batching_keys = list(image_paths_by_key.keys())
```

Convert dictionary keys to a list for batching

### lines 1240-1242

```python
fov_channels = {}
```

fov_channels[output_filename][channel] = MIP array. We collect every channel's MIP into this dict and only write the concatenated stack once a FOV has been fully assembled (below).

### line 1247  _(unsure)_

```python
batch_keys = batching_keys[idx:idx+batch_size]
```

Select batch keys and create a subset of the dictionary for this batch

### line 1252  _(unsure)_

```python
for i, (key, images) in enumerate(images_by_key.items()):
```

Process each batch of images

### lines 1257-1260

```python
if not images:
```

load_images_from_paths deliberately skips unreadable files, so this list can be empty. np.stack([]) below used to raise and abort the whole ingest before stack/ was written, discarding every healthy FOV in the plate because of one corrupt raw.

### lines 1266-1297

```python
output_filename = _escaped_field_stem(
```

One FOV file per TIMEPOINT, timelapse or not. The timelapse branch used to drop the timeID and name every frame of a field `<plate>_<well>_<field>.tif`, which had two effects, both silent:

the `np.maximum` combine below then folded all N frames into one max projection, so the movie was destroyed before anything downstream saw it, and `_generate_time_lists` — which both `_concatenate_channel` and `concatenate_and_normalize` group on — skips any name with fewer than four underscore-separated parts, so it returned [] for the whole plate. No `*_norm_timelapse.npz` was written, no masks were generated, and `preprocess_generate_masks` died much later in `_pivot_counts_table` on "no such table: object_counts".

The non-timelapse spelling is exactly what

`_generate_time_lists` parses (plate_well_field_time), so there is nothing for the timelapse branch to spell differently.

The plate is escaped because it is FREE TEXT: it comes from a regex group or, more often, from `os.path.basename(src)` — a folder name, which very often holds an underscore. A plate folder called `exp_1` used to produce `exp_1_A01_1_1.npy`, five separator-delimited components for a four-component grammar, and `utils._map_wells` returned the string 'error' in all five slots: the plate could not be measured at all. `escape_field_stem_plate` writes `exp%5F1_A01_1_1`, which `schema.parse_field_stem` reads back as plate `exp_1` character for character.

### lines 1303-1310

```python
_chans = fov_channels.setdefault(output_filename, {})
```

Combine, do not overwrite. The grouping key built in utils._extract_filename_metadata includes sliceID, so with cellvoyager/cq1 metadata every z-plane arrives as its OWN key and this assignment let each plane replace the last: a 21-plane stack silently became one arbitrarily chosen plane, decided by os.listdir order, with no warning and no log line. (Under metadata_type='auto' the regex has no sliceID group, so every plane is already in `images` and this is a no-op.)

### lines 1324-1325  _(unsure)_

```python
os.makedirs(stack_path, exist_ok=True)
```

Assemble each FOV's channels into a single stacked .npy, using the same sorted-channel order the old folder-based _merge_channels used.

### lines 1347-1348

```python
if save_original_images:
```

Handle the raw input images: keep a backup copy under orig/ only when requested, otherwise delete them (the pixels now live in stack/).

## _merge_file

### line 1381  _(unsure)_

```python
file_root, file_ext = os.path.splitext(file_name)
```

Construct new file path

### line 1385  _(unsure)_

```python
if not os.path.exists(new_file):
```

Check if the new file exists and create the stack directory if it doesn't

### line 1397, trailing  _(unsure)_

```python
del img
```

Explicitly delete the reference to the image to free up memory

### line 1398, trailing  _(unsure)_

```python
if i % 10 == 0:
```

Periodically suggest garbage collection

## _generate_time_lists

### line 1433, trailing  _(unsure)_

```python
continue
```

Skip file on conversion error

### line 1437, trailing  _(unsure)_

```python
continue
```

Skip file if not correctly formatted

### line 1439  _(unsure)_

```python
sorted_grouped_filenames = [sorted(files, key=lambda x: x[0]) for files in file_dict.values()]
```

Sort each list by timepoint, but keep them grouped

### line 1441  _(unsure)_

```python
sorted_file_lists = [[filename for _, filename in group] for group in sorted_grouped_filenames]
```

Extract just the filenames from each group

## _move_to_chan_folder

### lines 1492-1494

```python
if wellID[0].isdigit():
```

Undo zero padding, but keep a token that holds no integer rather than turning it into '0' — see utils._int_or_token.

### line 1507, trailing  _(unsure)_

```python
print(f'Converted Well ID: {orig_wellID} to {wellID}')
```

, end='\r', flush=True)

### lines 1509-1513

```python
newname = _escaped_field_stem(
```

Same escape as _rename_and_organize_image_files, and for the same reason: plateID falls back to the source FOLDER NAME when the regex has no plateID group, and a folder called `exp_1` puts a fifth component into a four-component name that nothing downstream can split.

### line 1525  _(unsure)_

```python
valid_exts = ['.tif', '.png']
```

Move original images to a new directory

### lines 1536-1539

```python
ledger.finalize()
```

Files whose metadata could not be parsed never reach a channel folder; without this the plate silently continues with fewer fields. Returns None, as it always has — callers treat this as a side-effecting sorter and one existing caller asserts on the None.

## _merge_channels

### lines 1568-1577

```python
print(f'No single-channel folders in {src}; stack/ will be built '
```

No per-channel sub-folders, which is the NORMAL layout now:

_rename_and_organize_image_files builds stack/ straight from an in-memory {fov: {channel: mip}} dict and never creates them. This function is the older two-step path, kept for folders that still have them.

Returning lets preprocess_img_data fall through to the branch that builds stack/ directly. Indexing chan_dirs[0] instead raised IndexError from inside a stage whose message named the plate, so a perfectly ordinary folder read as a corrupt one.

### line 1585  _(unsure)_

```python
if not os.path.exists(stack_dir):
```

Create the 'stack' directory if it doesn't exist

## _concatenate_channel

### lines 1633-1637

```python
start = time.time()
```

`start` used to be bound only in the non-timelapse branch, so this branch raised UnboundLocalError on its first group and the except below reported it as a filename-metadata problem while silently writing nothing. Time per group, to match the group-based files_processed/files_to_process below.

### lines 1655-1656  _(unsure)_

```python
files_to_process = len(time_stack_path_lists)
```

A count, not the list-of-lists: print_progress normalises a list via len(set(...)), which raises on unhashable lists.

### line 1675, trailing  _(unsure)_

```python
batch_index = 0
```

Added this to name the output files

### line 1683, trailing  _(unsure)_

```python
filenames_batch.append(os.path.basename(path))
```

store the filename

### line 1706, trailing  _(unsure)_

```python
batch_index += 1
```

increment this after each batch is saved

### line 1707, trailing  _(unsure)_

```python
del stack
```

delete to free memory

### line 1708, trailing  _(unsure)_

```python
stack_ls = []
```

empty the list for the next batch

### line 1709, trailing  _(unsure)_

```python
filenames_batch = []
```

empty the filenames list for the next batch

## _normalize_img_batch

### lines 1729-1730  _(unsure)_

```python
channels = [int(c) for c in channels]
```

Channel indices may arrive as strings (e.g. from a settings CSV); coerce so ``stack[:, :, :, channel]`` indexing works.

### line 1735  _(unsure)_

```python
time_ls = []
```

for channel in range(stack.shape[-1]):

### lines 1739-1743

```python
background = settings.get('background', 100)
```

Default normalisation params for any channel that isn't one of the recognised object channels (e.g. an organelle channel, or an intensity-only channel measured but not segmented). Without these defaults a channel matching NONE of the object types below raised UnboundLocalError: 'background'.

### lines 1763-1764

```python
if settings.get('organelle_channel') is not None and channel == settings['organelle_channel']:
```

Organelle channel — use organelle-specific settings when present, otherwise the generic defaults above.

### line 1777  _(unsure)_

```python
if remove_background:
```

Step 3: Remove background if required

### line 1781  _(unsure)_

```python
non_zero_single_channel = single_channel[single_channel != 0]
```

Step 4: Calculate global lower percentile for the channel

### line 1785  _(unsure)_

```python
global_upper = None
```

Step 5: Calculate global upper percentile for the channel

### line 1794, trailing  _(unsure)_

```python
global_upper = np.percentile(non_zero_single_channel, 99.5)
```

Fallback in case no upper percentile met the threshold

## _concatenate_and_normalize_impl

### lines 2005-2008

```python
if settings is None:
```

`settings = {}` used to be substituted here, but the very next reads are settings['timelapse'] / ['randomize'] / ['batch_size'], so the empty dict could only ever produce a cryptic KeyError from deep inside the function (after masks/ had already been created). Say what is actually wrong.

### lines 2018-2022

```python
channels = [int(c) for c in channels if c is not None]
```

Coerce channel indices to int up-front so both the per-batch normalisation and the ``normalized_stack[..., channels]`` slice work even when channels came through as strings ('0', '1', ...). Drop Nones first: an unused object channel is passed as None, and coercing before the (later) None filter made int(None) raise TypeError.

### lines 2054-2055

```python
ledger = RunLedger('concatenate_and_normalize')
```

Every FOV that fails to load is dropped from the normalised stacks. Nothing downstream can tell, so account for it here.

### lines 2116-2118

```python
if i == 0 and settings.get('plot'):
```

Only plot when the user asked for it: an interactive matplotlib backend makes plt.show() block, which would hang the whole pipeline in a script/terminal run.

### lines 2128-2130

```python
raise
```

A partially corrected timelapse cannot be presented as a successful raw/off run. Let the run policy record the real failure and leave provenance incomplete for resume.

### lines 2150-2154

```python
with ledger.item(path, stage='load_npy',
```

An unreadable file must skip only its own accumulation. The old `continue` also jumped past the batch-flush check below, so a bad file in the final position discarded every good image already collected in that batch (and elsewhere merged two batches into one, silently changing the per-batch normalisation grouping).

### lines 2167-2168

```python
if stack_ls and ((i + 1) % settings['batch_size'] == 0 or i + 1 == nr_files):
```

`stack_ls and` guards the case where every file in a batch failed: np.stack([]) would raise.

### lines 2197-2199

```python
arrays = dict(data=normalized_stack,
```

Lossless-compressed so the on-disk normalised batch is much smaller (np.load reads it transparently); it's deleted with masks/ after merged/ is built unless keep_intermediate is set.

### lines 2206-2207  _(unsure)_

```python
if batch_index == 0 and settings.get('plot'):
```

Gated on settings['plot'] — see the timelapse branch above:

an interactive backend blocks the pipeline on plt.show().

### lines 2228-2230

```python
_invalidate_v1_segmentation_outputs(os.path.dirname(src))
```

Invalidate downstream products BEFORE provenance becomes complete. A crash after finish must never leave a complete correction record beside masks/merged fields drawn from the superseded pixels.

### lines 2237-2239

```python
ledger.finalize()
```

Emitted last so a partially-loaded stack cannot scroll off the top of a 400-line progress log. No stamp: output_fldr is masks/, which the segmentation step globs, and a stray sidecar there is not worth the risk.

## concatenate_and_normalize

### lines 2264-2266

```python
with tempfile.TemporaryDirectory(
```

Cellpose enumerates every top-level NPZ in masks/. Build elsewhere so a failure cannot expose a mixed old/new set, and let TemporaryDirectory's context guarantee cleanup on every exception path.

## _get_lists_for_normalization

### line 2287  _(unsure)_

```python
backgrounds = []
```

Initialize the lists

### lines 2293-2294  _(unsure)_

```python
for ch in [settings['nucleus_channel'], settings['cell_channel'], settings['pathogen_channel']]:
```

Iterate through the channels and append the corresponding values if the channel is not None for ch in settings['channels']:

## _normalize_stack

### lines 2381-2383

```python
non_zero_single_channel = single_channel[single_channel != 0]
```

Choose an upper percentile whose global signal clears the requested threshold. An all-zero channel has no percentile; leave it untouched and let every frame take the zero-SNR path.

### line 2393  _(unsure)_

```python
arr_2d_normalized = np.zeros_like(single_channel, dtype=single_channel.dtype)
```

Normalize the pixels in each image to the global percentiles and then dtype.

### lines 2397-2401

```python
lower = upper = 0.0
```

Seeded because the per-frame progress print below formats these unconditionally while they are only assigned for frames that have non-zero pixels: a blank FIRST frame used to abort the whole run with UnboundLocalError, and a later blank frame reported the previous frame's percentiles.

## _create_movies_from_npy_per_channel

### line 2514  _(unsure)_

```python
files = [f for f in os.listdir(src) if f.endswith('.npy')]
```

Organize files by plate, well, field

### lines 2528-2530

```python
_times, paths = zip(*file_list)
```

Every group is created by appending one file, so it is non-empty. Unpacking that invariant directly avoids an impossible zero-iteration arm in a loop whose only purpose was to build these two collections.

### lines 2534-2542

```python
for channel in range(arrays.shape[-1]):
```

`paths` follows the time-sorted file_list above. np.stack retains the former leading time dimension. Names remain basenames for the movie overlay. Loading failures still propagate exactly as they did in the loop. A group can therefore never reach np.stack with no arrays. NOTE: this loop must stay INSIDE the per-(plate, well, field) loop. When it was dedented, `arrays` was unbound if no filename matched the regex (UnboundLocalError) and only the LAST field ever got a movie — every other field was silently dropped.

### line 2544  _(unsure)_

```python
channel_arrays = arrays[..., channel]
```

Extract the current channel for all time points

### line 2546  _(unsure)_

```python
channel_data_flat = channel_arrays.reshape(-1)
```

Flatten the channel data to compute global percentiles

### line 2551  _(unsure)_

```python
normalized_channel_arrays_3d = [arr[..., np.newaxis] for arr in normalized_channel_arrays]
```

Convert the list of 2D arrays into a list of 3D arrays with a single channel

## delete_empty_subdirectories

### line 2563  _(unsure)_

```python
for dirpath, dirnames, filenames in os.walk(folder_path, topdown=False):
```

Check each item in the specified folder

### line 2565  _(unsure)_

```python
for dirname in dirnames:
```

os.walk is used with topdown=False to start from the innermost directories and work upwards.

### line 2567  _(unsure)_

```python
full_dir_path = os.path.join(dirpath, dirname)
```

Construct the full path to the subdirectory

### line 2569  _(unsure)_

```python
try:
```

Try to remove the directory and catch any error (like if the directory is not empty)

## select_fields

### lines 2575-2576

```python
def select_fields(names, fields):
```

An error occurred, likely because the directory is not empty print(f"Skipping non-empty directory: {full_dir_path}")

## preprocess_img_data

### line 2696  _(unsure)_

```python
valid_extensions = [ext for ext in extensions if ext in valid_ext]
```

Filter only valid extensions

### line 2698  _(unsure)_

```python
img_format = None
```

Determine most common valid extension

### lines 2744-2747

```python
seen = {}
```

Deduplicate while tracking positions. Coerce to int: channel indices loaded from a settings CSV (or passed from the GUI) can arrive as strings like '0', which then blow up array indexing downstream (stack[:, :, :, '0'] -> IndexError).

### line 2763, trailing  _(unsure)_

```python
from .plot import plot_arrays
```

used below; import here so the plot step

### line 2764  _(unsure)_

```python
settings = set_default_settings_preprocess_img_data(settings)
```

doesn't raise NameError under try/except

### lines 2791-2798

```python
img_format = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.nd2', '.czi', '.lif']
```

No `img_format is not None` guard. _merge_channels is the older path and only produces stack/ when the folder has per-channel sub-directories; the modern ingest has none, so with img_format=None nothing built stack/ at all and the run died in concatenate_and_normalize on a missing directory, two functions away from the cause. _rename_and_organize_image_files is the only thing that can create it here, so it runs whenever stack/ is still absent.

### lines 2800-2801  _(unsure)_

```python
nr_channel_folders = _rename_and_organize_image_files(
```

Builds the stack/ arrays directly from an in-memory channel dict (no per-channel sub-folders) and returns the channel count.

### lines 2807-2812

```python
all_imgs = len([f for f in os.listdir(stack_path) if f.endswith('.npy')]) if os.path.isdir(stack_...
```

Make sure no batches will be of only one image

This counted len(stack_path) — the number of CHARACTERS in the path string, which always ends in 'stack' — so the check fired (or stayed silent) purely because of how long src happened to be. Count the .npy stacks that concatenate_and_normalize will actually batch over instead.

### lines 2818-2821

```python
if last_batch_size == 1:
```

Report, don't raise: the stack is already written by this point so aborting cannot fix the batching, it only skipped the channel-count fix-up, the movies, the plot and the MIP below — silently corrupting the output of an otherwise fine run.

### lines 2850-2861

```python
entries = []
```

Emptiness, not absence: _rename_and_organize_image_files creates stack/ before it has anything to put in it, so a folder with no matching images leaves a real but empty directory. Checking only that the path exists let the run continue to completion and write an empty measurement set, which is worse than stopping — it looks like a result.

Say what is wrong here, where src is known, rather than letting os.listdir fail on a path the user never named. The usual cause is src pointing at a folder of PLATES rather than at a plate: the images are one level down, so nothing matched and nothing was organised.

### lines 2903-2906

```python
try:
```

`seen` is keyed on int(ch) (see the dedup loop above), so looking the raw value up meant a string channel index ('0') never matched and no cellpose_* key was written — leaving the objects to be segmented on the wrong plane, silently. Uncoercible values are dropped as before.

### line 2911  _(unsure)_

```python
settings[f"cellpose_{key}"] = seen[ch]
```

The same keys and coercion populated `seen` above, so this key exists.

## _check_masks

### line 2959  _(unsure)_

```python
existing_files_mask = [
```

True means this field must be generated.

## _save_figure

### lines 3034-3036

```python
from .plot import save_figure
```

Imported here, not at module scope: `spacr.plot` pulls in torch, cv2, seaborn, statsmodels and pingouin, and this module is on the cold measure-worker spawn path. See tests/test_measure_spawn.py.

## _read_and_join_tables

### lines 3331-3334

```python
raw = png_list_df[id_column]
```

Two different reasons, reported separately because they call for two different actions: NULL means "this row is another crop mode's" and is expected in a multi-mode database, while a token that is not a number means the crop's own name could not be read.

### lines 3360-3363

```python
png_list_df = png_list_df.rename(columns={png_time: cell_time})
```

The two tables spell one concept two ways. Align the copy of png_list, never the object table: the object table's column survives into the result and renaming it there would change the schema the caller gets back.

### lines 3387-3393

```python
_lost_png = _before_png - len(merged)
```

THE INNER JOIN'S LOSS IS SAID OUT LOUD. png_list joins inner a cell with no attributable crop cannot be classified, annotated or displayed -- but "your population just shrank" is not something a reader should have to infer from a row count. Crops whose id is 'omulti'/'onone'/'error' are dropped during the id migration above, which reports itself; the CELLS they would have matched disappear here, which did not.

### lines 3404-3407

```python
for entity in CHILD_ROLES:
```

From the registry, not a literal: ORGANELLE was missing from both of these loops, so asking for it returned a frame with no organelle columns and no message. Naming the child roles once is what lets a second organelle reach every reader (instruction 76).

### lines 3411-3416

```python
print(f"{entity} was measured without a cell mask, so its rows "
```

A child table measured with cell_mask_dim=None has no parent link at all -- _merge_and_save_to_database drops 'cell_id' from its key columns in exactly that case, deliberately. The roll-up onto the cell is then not merely empty, it is undefined, and this used to be a bare KeyError('cell_id') naming neither the table nor the setting behind it.

### lines 3449-3452

```python
for entity in CHILD_ROLES:
```

From the registry, not a literal: ORGANELLE was missing from both of these loops, so asking for it returned a frame with no organelle columns and no message. Naming the child roles once is what lets a second organelle reach every reader (instruction 76).

### lines 3470-3478

```python
from . import schema as _schema
```

EVERY PLATE ID COMES BACK IN ONE SPELLING. A screen written by an older run stamps its plate `pplate1` and everything computed since stamps it `plate1`, so the two never join: an ML run over 60,816 real cells scored every one of them and wrote none back, reporting that its own database "probably comes from a different experiment".

`schema.normalise_plate_columns` is the one rule, and it is applied on READ -- nothing on disk is rewritten, so an old database keeps working and a re-read of it produces the same keys as a fresh run.

## _save_settings_to_db

### line 3593  _(unsure)_

```python
settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
```

Convert the settings dictionary into a DataFrame

### line 3595  _(unsure)_

```python
settings_df['setting_value'] = settings_df['setting_value'].apply(str)
```

Convert all values in the 'setting_value' column to strings

### lines 3597-3599

```python
src = os.path.dirname(settings['src'])
```

(No display here — save_settings already renders the settings table via pretty_print_settings; displaying again produced the double print.) Determine the directory path

### line 3602  _(unsure)_

```python
os.makedirs(directory, exist_ok=True)
```

Create the directory if it doesn't exist

### line 3604  _(unsure)_

```python
conn = sqlite3.connect(f'{directory}/measurements.db', timeout=5)
```

Database connection and saving the settings DataFrame

### lines 3618-3621

```python
if not _settings_history_rows(conn):
```

Migrate what is already on disk before it is replaced. A database written before the history table existed carries exactly one snapshot, in `settings`; without this it would be the one run that still gets forgotten.

### lines 3639-3640  _(unsure)_

```python
conn.close()
```

Closed on every path: an open connection holds the lock, and this runs immediately before measure_crop's workers start writing.

## _mask_movie_frame_geometry

### lines 3688-3690

```python
shapes = [np.asarray(mask).shape[:2] for mask in masks]
```

`shape[:2]` is not enough on its own: a corrupt or one-dimensional array gives a one-tuple, and unpacking it would raise a ValueError about iterables rather than about the movie.

## _save_mask_timelapse_as_gif

### lines 3758-3762

```python
with figure_style(theme_target()):
```

Set the face color for the figure to black

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 3765, trailing  _(unsure)_

```python
ax.set_facecolor('black')
```

Set the axes background color to black

### line 3766, trailing  _(unsure)_

```python
ax.axis('off')
```

Turn off the axis

### lines 3767-3768

```python
plt.subplots_adjust(left=0, right=1, top=1 - band, bottom=band,
```

Leave the two bands the captions are drawn into; at top=1 the frame counter was rendered above the canvas and never reached the file.

### line 3772, trailing  _(unsure)_

```python
filename_text_obj = None
```

Initialize a variable to keep track of the text object

## _save_mask_timelapse_as_gif._update

### line 3785, trailing  _(unsure)_

```python
nonlocal filename_text_obj, frame_text_obj
```

Reference the nonlocal variables to update them

### line 3787, trailing  _(unsure)_

```python
filename_text_obj.remove()
```

Remove the previous text object if it exists

### line 3791, trailing  _(unsure)_

```python
ax.clear()
```

Clear the axis to draw the new frame

### line 3792, trailing  _(unsure)_

```python
ax.axis('off')
```

Ensure axis is still off after clearing

### line 3800, trailing  _(unsure)_

```python
filename_text = filenames[frame]
```

Get the filename corresponding to the current frame

### line 3801, trailing  _(unsure)_

```python
filename_text_obj = fig.text(0.5, band / 2, filename_text, ha='center', va='center', fontsize=geo...
```

Adjust text position, size, and color as needed

### line 3803  _(unsure)_

```python
for label_value in np.unique(current_mask):
```

Annotate each object with its label number from the mask

### line 3805, trailing  _(unsure)_

```python
if label_value == 0: continue
```

Skip background

### line 3809  _(unsure)_

```python
if tracks_df is not None:
```

Overlay tracks

## _save_object_counts_to_database._count_objects

### line 3843  _(unsure)_

```python
if unique[0] == 0:
```

Assuming 0 is the background label, remove it from the count

## _save_object_counts_to_database

### line 3875  _(unsure)_

```python
cursor.executemany('''
```

Batch insert or update the object counts

## _create_database

### lines 3898-3900

```python
print(error)
```

Preserve the historical helper contract: an unusable destination is reported to the caller's console without taking down a processing run. Schema compatibility errors below remain explicit.

## _save_array_atomic

### lines 3968-3969

```python
fd, tmp_path = tempfile.mkstemp(prefix='.spacr_tmp_', suffix='.npy',
```

Same directory as the destination: os.replace is only atomic within one filesystem, and /tmp is routinely a different one.

### lines 3974-3975

```python
with open(tmp_path, 'wb') as handle:
```

allow_pickle stays at numpy's default (False) — these are plain numeric stacks and a pickled payload here would be a bug.

## _load_and_concatenate_arrays

### lines 4037-4042

```python
try:
```

THE MASK FOLDERS THAT EXIST, LISTED ONCE. The check below runs per role, and 326 took `ORGANELLE_ROLES` from four to 702 -- so the `os.path.exists` it used to do became 700-odd stat calls against one directory on every merge, to answer a question one listing answers for all of them. Missing directory is the ordinary case for a run that segmented nothing, and is an empty set rather than an error.

### lines 4083-4086

```python
reference_npy = next(
```

A resume may skip every array, so validate the existing manifest before deciding what is complete. Reusing arrays under a different role layout is worse than redoing work: all planes still exist, but their biological names have changed and measurement would return plausible wrong values.

### lines 4123-4126

```python
already_done = set()
```

Opt-in resume: skip fields whose merged stack is already there AND verified complete. Reported before any work starts, so a resume that rejects three truncated leftovers says so rather than quietly re-merging them.

### line 4138  _(unsure)_

```python
for idx, filename in enumerate(reference_files):
```

Iterate through each file in the reference folder

### lines 4142-4143

```python
if filename.endswith('.npy') and os.path.splitext(filename)[0] not in already_done:
```

`and not already done` rather than a `continue`, so a skipped field still advances the progress bar instead of the counter jumping.

### lines 4147-4148

```python
exists_in_all_folders = all(
```

Check if this file exists in all the other specified folders. Masks may be .tif (new, compressed) or legacy .npy — resolve both.

### line 4154  _(unsure)_

```python
ref_array_path = os.path.join(reference_folder, filename)
```

Load and potentially modify the array from the reference folder

### lines 4159-4165

```python
concatenated_array = np.take(concatenated_array, channels, axis=-1)
```

axis=-1, not axis=2. The channel axis is the LAST one, and it happens to be axis 2 only for a 2-D (Y, X, C) field. On a z-stack -- (Z, Y, X, C) -- axis 2 is X, so asking for channels [0, 1] returned a two-pixel-wide image with every channel still attached, which then merged, measured and produced numbers. The two spellings are identical for 2-D, so the ordinary path is unchanged.

### line 4202  _(unsure)_

```python
stack_ls.append(concatenated_array)
```

Add the array from the reference folder to 'stack_ls'

### line 4205  _(unsure)_

```python
for folder in folder_paths[1:]:
```

For each of the other folders, load the mask (tif or npy).

### lines 4209-4214

```python
if array.ndim in (2, concatenated_array.ndim - 1):
```

A mask carries the image's spatial axes and no channel axis, so it needs one appended -- whether that is (Y, X) -> (Y, X, 1) or (Z, Y, X) -> (Z, Y, X, 1). Testing `ndim == 2` covered only the first, and a 3-D mask reached np.concatenate one axis short of the image it belongs to.

### line 4226  _(unsure)_

```python
padded_shapes = [shape + (0,) * (max_tuple_length - len(shape)) for shape in unique_shapes]
```

Pad shorter tuples with zeros to make them all the same length

### line 4228  _(unsure)_

```python
max_dims = np.max(np.array(padded_shapes), axis=0)
```

Now create a NumPy array and find the maximum dimensions

### line 4238  _(unsure)_

```python
stack = np.concatenate(padded_stack_ls, axis=-1)
```

Concatenate the padded arrays along the channel dimension (last dimension)

## read_plot_model_stats._plot_and_save

### lines 4307-4311

```python
with figure_style(theme_target()):
```

Create subplots

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 4315  _(unsure)_

```python
sns.lineplot(ax=axes[0], x='epoch', y=column, data=train_df, marker='o', color='red')
```

Plotting

### line 4319  _(unsure)_

```python
axes[0].set_title(f'Train {column} vs. Epoch', fontsize=20)
```

Set titles and labels

## read_plot_model_stats

### line 4337  _(unsure)_

```python
train_df = pd.read_csv(train_file_path, index_col=0)
```

Read the CSVs into DataFrames

### line 4341  _(unsure)_

```python
fldr_1 = os.path.dirname(train_file_path)
```

Get the folder path for saving plots

### lines 4344-4346

```python
with plt.rc_context():
```

`sns.set` writes a whole seaborn theme into matplotlib's process-wide rcParams. Scoped, so reading the model stats does not restyle every figure the session draws afterwards.

### line 4352  _(unsure)_

```python
_plot_and_save(train_df, val_df, column='accuracy', save=save, path=fldr_1)
```

Plot and save the results

## _save_model

### lines 4410-4411  _(unsure)_

```python
if intermedeate_save is True or intermedeate_save is None:
```

``True`` preserves the historical default thresholds; ``False`` disables archival snapshots while best/last checkpoints remain available.

### lines 4420-4422

```python
saved_archive = None
```

Archive only improving epochs. The old implementation wrote another file every epoch above a threshold and encoded the threshold (95.0) instead of the measured accuracy (for example 97.63).

## _save_progress._save_df_to_csv

### line 4468, trailing  _(unsure)_

```python
f.flush()
```

Ensure data is written to the file system

## _save_progress

### line 4474  _(unsure)_

```python
os.makedirs(dst, exist_ok=True)
```

Save accuracy, loss, PRAUC

### line 4479  _(unsure)_

```python
_save_df_to_csv(results_path_train, train_df)
```

Save training data

### line 4482  _(unsure)_

```python
if validation_df is not None:
```

Save validation data if available

### line 4486  _(unsure)_

```python
read_plot_model_stats(results_path_train, results_path_validation, save=True)
```

Call read_plot_model_stats after ensuring the files are saved

## _read_db

### lines 4536-4550

```python
if isinstance(db_loc, (str, os.PathLike)):
```

A `~` PATH IS EXPANDED HERE, once, for every reader.

GitHub issue #108 (auto-filed 2026-08-17, macOS): a `src` beginning with `~` produced `~/.../measurements/measurements.db`, which `ensure_database_schema` -> `migrate_database` resolved against the WORKING DIRECTORY and then refused with `FileNotFoundError: ~<DB>`.

It is fixed here rather than in `migrate_database`, whose docstring states the non-expansion as a deliberate contract ("made absolute but not tilde-expanded"), and rather than at each of the ~99 sites that build a measurements path by string concatenation. This is the funnel they all pass through.

expandvars too: a settings CSV carried between machines routinely holds $HOME or %USERPROFILE%, and the failure is identical.

### lines 4562-4564

```python
for table in tables:
```

Validate the caller's identifiers before a schema migration walks every table in the database. This keeps the public read API's error stable even when a malformed table happens to exist in SQLite.

### lines 4568-4588

```python
directory = os.path.dirname(os.path.abspath(db_loc)) or "."
```

A DATABASE NOBODY CAN WRITE TO IS STILL A DATABASE THAT CAN BE READ.

GitHub issue #115 (SMB mounts): `ensure_database_schema` renames legacy columns and stamps the schema version, so calling it before every read meant that reading a PRE-MIGRATION database from a read-only source an SMB share mounted read-only, a colleague's archived plate, a dataset on a read-only volume -- died with `OperationalError: attempt to write a readonly database`. The error named the write, not the reason, and the read was never the thing that needed writing.

Migration is not required in order to READ. `correct_metadata` below canonicalises the frame -- `plate`/`row`/`col` to `plateID`/`rowID`/ `columnID` and the rest -- so a legacy table is readable exactly as a current one is; the migration exists to make that permanent, not to make it possible. So when the database cannot be written, it is skipped and the file is opened read-only, which also stops SQLite trying to place a journal beside it.

BOTH the file and its DIRECTORY are checked: SQLite writes its journal into the directory, so a writable file in a read-only directory is not a writable database.

### line 4595, trailing  _(unsure)_

```python
chunksize = 100_000
```

internal safety setting; adjust if needed

### line 4605

```python
existing_tables = {
```

Optional but useful: fail early if a table name is wrong

### line 4623  _(unsure)_

```python
chunks = []
```

Read in chunks to reduce peak memory during SQL -> pandas conversion

### line 4629  _(unsure)_

```python
df = pd.read_sql_query(f"SELECT * FROM {quoted_table} LIMIT 0", conn)
```

Empty table: preserve columns

### line 4642  _(unsure)_

```python
del df
```

Drop local reference before next loop iteration

## _read_and_merge_data

### lines 4825-4853

```python
if 'prcf' in frame.columns and is_object_dtype(frame['prcf']):
```

THE KEY HALF THAT COMES FROM THE DATABASE IS MADE TEXT HERE.

Every object role below spells its per-object key the same way, `prcf + '_' + <parent id>`, and the parent id has just been rewritten with `.astype(str)` one line above the concatenation. `prcf` is whatever dtype the read inferred, and that is only the same dtype when the table had rows to infer from. A table with none comes back with every column at `object`, because there is nothing in it to look at, while the parent id beside it is a genuine string column -- and under pandas' string dtype the two cannot be added at all: `operation 'radd' not supported for dtype 'str' with dtype 'object'`, raised out of pyarrow from a line whose job is to spell a key. Older pandas concatenated the same pair silently, which is why this went unseen.

An object table with no rows is not a contrived case: a plate on which a role segmented nothing anywhere produces one, and so does any caller reading a table that was created but never written. Such a read is supposed to fail further down, on the location columns it really is missing, rather than here on a dtype.

Only an `object` column is touched. A `prcf` that is genuinely numeric is left alone so that it still fails loudly at the concatenation, as it does today, instead of being turned into digits that key nothing. Coercing the text case is the same rule `utils._split_data` already applies when it rebuilds `prcf` from the four location columns, so the key spelled on the way in matches the one spelled on the way out.

### lines 5028-5043

```python
metadata = metadata.assign(
```

`prcfo` -- the per-OBJECT key -- has to exist before the well count, because the count is of objects and `metadata_key` alone does not identify one. `object_label` is assigned by the segmenter per FIELD and restarts at 1 in each, so `nunique()` over a well counted distinct LABEL VALUES: a 9-field well holding 360 cells reported roughly 40, the size of its largest field.

That number is not cosmetic. `cells_per_well` is documented as the minimum a well must contribute and is used to drop under-populated wells, so a threshold of 100 discarded every well on a plate that averaged 360 cells -- and the wells it kept were the ones with the most crowded single field, which is the opposite of the intent. _split_data always rebuilds `prcf` from the four location components and returns it in metadata, even when an input carried a numeric prcf column. Every object role also consumes prcf before this point, so a fallback without it was both unreachable and (for timelapse) missing the time key.

## convert_numpy_to_tiff

### line 5094  _(unsure)_

```python
tiff_subdir = os.path.join(folder_path, 'tiff')
```

Create the subdirectory 'tiff' within the specified folder if it doesn't already exist

### line 5100  _(unsure)_

```python
for i, filename in enumerate(files):
```

Iterate over all files in the folder

### line 5107  _(unsure)_

```python
file_path = os.path.join(folder_path, filename)
```

Construct the full file path

### line 5112  _(unsure)_

```python
tiff_filename = os.path.splitext(filename)[0] + '.tif'
```

Construct the output TIFF file path

## generate_cellpose_train_test

### line 5173, trailing  _(unsure)_

```python
print(f'Copied {idx+1}/{len(ls)} images to {_type} set')
```

, end='\r', flush=True)

## parse_gz_files

### lines 5212-5213  _(unsure)_

```python
samples_dict.setdefault(stem, {})['R1'] = os.path.join(
```

No separator, so there is no mate to read off the name. A single-ended file still deserves to be seen.

### lines 5221-5223

```python
for position, token in enumerate(parts):
```

Illumina's full form is `<sample>_S1_L001_R1_001.fastq.gz`, so the mate is not always last. Look for it anywhere in the name before giving up on the file.

## open_crop_source

### lines 5430-5435

```python
print(f"crop_source={choice!r}: {exc}")
```

LOUD, NOT SILENT. This printed only under `verbose`, and every shipped caller passes verbose=False, so an unusable crop_source returned None without a word -- and the caller then fell back to the pre-generated PNGs. A user who asked for on-demand crops got a classifier trained on different data than they requested, with nothing in the log to say so.

## LazyCropPNG._stream

### lines 5505-5508

```python
raw = self._raw_bytes()
```

A crop PNG that spacr.crops cannot decode still has bytes on disk. Hand those over rather than losing the thumbnail: this is a display path, and a file that is merely unusual should not take the whole figure down.

## mark_crop_output_folder

### lines 5782-5784

```python
print(f"Warning: could not stamp the crop format on {folder}: {exc}")
```

Loud, never silent: the consequence of a missing marker is that the crops read back reversed. But failing a whole training run over a 300-byte sidecar helps nobody.

## generate_dataset

### lines 5881-5883

```python
if not paths:
```

generate_path_list_from_db returns None when the query fails

(a database with no png_list, say). correct_paths then died on an unbound local three frames away instead of saying so.

### line 5887, trailing  _(unsure)_

```python
paths = correct_paths(paths, src)
```

<- capture corrected paths

### line 5890  _(unsure)_

```python
if isinstance(settings['sample'], int) and settings['sample']:
```

sampling (guard against k > N)

### lines 5911-5913

```python
os.makedirs(dst, exist_ok=True)
```

ensure destination exists

A non-empty src list sets dst on its first iteration; an empty list is refused by the no-images check above before destination creation.

### line 5916  _(unsure)_

```python
date_name = datetime.date.today().strftime('%y%m%d')
```

Combine the temporary tar files into a final tar

### lines 5930-5934

```python
written, skipped = _write_crop_tar(selected_paths, tar_name, settings)
```

On-demand crops are cut here, in this process: a CropSource holds memory-mapped merged arrays and a per-field label index, and neither survives a fork usefully -- every worker would re-open and re-index every field it touched. The reads are the cost either way, so the pool buys nothing and the bookkeeping is simpler without it.

### line 5947  _(unsure)_

```python
temp_dir = os.path.join(dst, "temp_tars")
```

Create a temp folder in dst

### lines 5951-5952

```python
num_procs = max(1, min(max(2, cpu_count() - 2), total_images))
```

Chunking the data cap workers by total images so we don't spawn useless pools

### line 5991  _(unsure)_

```python
shutil.rmtree(temp_dir)
```

Delete the temp folder

### lines 5993-5996

```python
if written == 0:
```

`written`, not `total_images`: add_images_to_tar swallows a missing file with a print, so a tar built against a crop folder that has been deleted or moved used to be announced as "Saved 48 images" while holding none, and the run only failed later, inside inference, on an empty dataset.

## _write_crop_tar

### lines 6126-6128

```python
if name in used:
```

Two crops sharing a basename would overwrite each other inside the archive, which is exactly the collision spacr.predictions documents. Make the second one distinct instead of losing it.

## expected_sampled_fractions

### lines 6329-6330  _(unsure)_

```python
mass = [n * w for n, w in zip(counts, per_class)]
```

every sample of class c carries weight per_class[c]; class c therefore attracts n_c * per_class[c] of the total probability mass.

## format_class_balance_report

### lines 6383-6384

```python
effective = class_balance if split_name == 'train' else 'none'
```

Only the train split is ever resampled, so only it can show moved frequencies; showing them for validation/test would be a lie.

## make_cv_folds

### lines 6508-6510

```python
for c in range(n_classes):
```

Plain stratified k-fold: deal each class round-robin into the folds, starting at a per-class offset so fold 0 does not collect the remainder of every class.

### line 6526  _(unsure)_

```python
hist = {g: np.zeros(n_classes, dtype=float) for g in uniq}
```

Per-group class histogram, then greedy assignment largest-first.

### lines 6532-6536

```python
order = sorted(rng.permutation(uniq), key=lambda g: -hist[g].sum())
```

Largest group first — the big blocks have to land while the folds are still empty enough to take them — but shuffle before the (stable) sort so equally large groups arrive in a seed-dependent order. Sorting on the group name instead, as this used to, made the grouped branch ignore ``seed`` entirely and hand back one fixed partition.

### lines 6544-6545

```python
fold_rank = rng.permutation(k)
```

Folds the cost cannot separate are genuinely interchangeable, so let the seed choose between them rather than always fold 0.

### lines 6550-6553

```python
cost = round(
```

Spread of each class across folds, as a fraction of that class's total; lower is a more even stratification. Rounded so that folds differing only by float summation order tie honestly and the seed, not the noise, separates them.

## make_validation_holdout

### lines 6656-6657

```python
tie_break = np.random.default_rng(seed).permutation(len(candidates))
```

Folds the score cannot separate are interchangeable holdouts, so the seed picks between them instead of the first one always winning.

## _classification_data_dir

### lines 6794-6796

```python
if not os.path.isdir(data_dir):
```

Clear, actionable error when the train/test split hasn't been generated yet — the most common Train-CV mistake is pointing src at the plate folder before the annotated crops were split into train/ and test/.

### lines 6807-6810

```python
missing = [c for c in classes if not os.path.isdir(os.path.join(data_dir, c))]
```

FIX: raise an error instead of just printing when class folders are missing WHY: the original printed a warning but continued execution, silently training on a broken/incomplete dataset — this masks data problems that look like model performance problems

## _classification_transform

### lines 6827-6832

```python
n_ch = len(channel_idx)
```

FIX: match normalization mean/std tuple length to the actual number of selected channels, not a hardcoded 3 WHY: if you select only 1 channel (e.g. channels=['g']), the original Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) will crash or silently produce wrong values because the tensor has 1 channel but normalize expects 3

### line 6842, trailing  _(unsure)_

```python
*norm_transforms,
```

FIX: uses channel-count-aware normalization

## generate_cv_loaders

### lines 6933-6935

```python
num_workers = max(0, int(n_jobs)) if n_jobs is not None else 0
```

``0`` is PyTorch's documented in-process mode and is important for GUI stability: forcing a minimum of four workers ignored the setting and created multiprocessing queues/sockets even when callers disabled them.

### lines 6954-6955

```python
val_loader = DataLoader(
```

The validation loader is never sampled and never shuffled: its job is to measure the model against the real class prior.

### lines 6971-6972  _(unsure)_

```python
'dataset': data,
```

In-memory only: nested CV reuses the decoded dataset definition and creates Subsets without walking the image tree a second time.

## generate_loaders

### lines 7040-7042

```python
num_workers = max(0, int(n_jobs)) if n_jobs is not None else 0
```

Honour an explicit zero so callers can keep dataset reads in-process. A forced four-worker minimum made worker teardown unavoidable and could deadlock applications that already own GUI or database threads.

### lines 7070-7071

```python
report_class_balance(dataset_labels(train_dataset), classes=classes,
```

Skew is measured on the labels the model will actually see, after the split and after augmentation, and reported on every run.

### lines 7077-7078  _(unsure)_

```python
sampler, _ = make_class_balance_sampler(
```

A sampler and shuffle=True are mutually exclusive in DataLoader; the sampler already draws in random order.

### line 7087, trailing  _(unsure)_

```python
num_workers=num_workers,
```

FIX: was hardcoded to 1

### lines 7091-7096

```python
val_loaders = DataLoader(val_dataset, batch_size=batch_size,
```

FIX: don't shuffle the validation DataLoader

WHY: shuffling validation data wastes time and has zero benefit — evaluation metrics are computed over the entire set regardless of order The validation loader also never receives the sampler: resampling it would change the class prior the reported metrics are measured against, so a "balanced" accuracy would stop describing the screen.

### line 7098, trailing  _(unsure)_

```python
shuffle=False,
```

FIX: was True

### line 7099, trailing  _(unsure)_

```python
num_workers=num_workers,
```

FIX: was hardcoded to 1

### line 7107

```python
effective_balance = class_balance if split_name == 'train' else 'none'
```

Held-out test data is reported but never resampled.

### line 7119, trailing  _(unsure)_

```python
num_workers=num_workers,
```

FIX: was hardcoded to 1

## generate_training_dataset

### lines 7196-7200

```python
png_type = settings.get('path_string') or settings.get('png_type', 'cell_png')
```

`path_string` is the current name -- a substring that has to appear in the crop's path, which is what this always was. `png_type` is still accepted because it is in every settings CSV written before the rename, and because this function is called directly as well as through classify(), which is where the alias would otherwise be applied.

### line 7206  _(unsure)_

```python
if 'nucleus' not in tables:
```

Limits for merge helper

### line 7214  _(unsure)_

```python
if isinstance(settings['src'], str):
```

Normalize src to list

### lines 7479-7482

```python
first_src = settings['src'][0]
```

A multi-plate run writes one combined dataset beside the first plate. Previously ``dst_final`` was overwritten on every iteration, so the advertised ``training_all`` destination was abandoned and the combined files unexpectedly landed below the final plate.

### line 7487, trailing  _(unsure)_

```python
crop_db_path = None
```

last measurements.db, for the crop-format lookup

### line 7500  _(unsure)_

```python
if png_type:
```

Filter by image type if requested

### lines 7504-7507

```python
source = open_crop_source(settings, src, object_type=object_type)
```

Where the pixels come from. 'png' (and 'auto' with a crop folder present) leaves every list below holding plain paths, which generate_dataset_from_lists copies exactly as it always has. Only the merged source replaces them with on-demand handles.

### lines 7517-7520

```python
from .training_basis import resolve_basis
```

THROUGH `resolve_basis`, so a settings file naming the retired 'measurement' basis is MIGRATED here rather than raising -- which is the promise `RETIRED_BASES` makes, and it is only kept if every reader goes through the resolver instead of reading the key.

### lines 7548-7553

```python
import ast as _ast
```

A settings CSV stores the repr, and a caller that hands the string straight through used to be iterated one CHARACTER at a time -- "[['c1'], ['c2']]" became seventeen classes named '[', '[', "'", 'c', ... The Qt panel now collects a real list; this covers the CSV and CLI paths that do not go through it.

### lines 7567-7581

```python
meta_col = _class_column(settings)
```

The column the class_metadata values are matched against is the one the user named in 'metadata_type_by'. It used to be hard-coded to 'condition' -- a column no spaCR writer puts in png_list unless annotate_conditions has been run -- so a run configured with metadata_type_by='columnID' selected on a column it was never pointed at, and the guard below printed "got 0 classes" and then indexed the missing column anyway, turning a diagnosable misconfiguration into a bare KeyError several frames down. NOW READ OFF `classes`, which already names the column each class is defined by -- `metadata_type_by` was a second place to say the same thing, and two places to say it is two places to say it differently. A settings file that still carries the old key is honoured, so an old CSV runs unchanged.

### lines 7592-7594

```python
meta_values = png_df[meta_col].astype(str)
```

Compare as text: png_list holds 'c1'/'r1' strings but a fallback to the object table can hand back a numeric column, and class_metadata is whatever the settings CSV parsed to.

### lines 7601-7606

```python
if isinstance(cm, (list, tuple, set)):
```

One class per entry. An entry may be a single value

('c1') or a group of values (['c1','c2']) that share one label -- the list-of-lists form the GUI defaults to and every settings CSV on disk already carries. It used to be str()'d whole, so ['c1'] was matched as the literal text "['c1']" and selected nothing.

### lines 7617-7624

```python
ann_cols = settings.get('annotation_columns')
```

resolve_basis has already reduced the vocabulary to metadata or annotation and raises TrainingBasisError for everything else. The retired measurement spelling is migrated to annotation. Consequently this arm is exhaustive, not a fallback guess. Keeping another unknown-mode exception here duplicated a rule. Worse, that exception could never name an input that reached it. The resolver's tested error remains the single refusal surface. Old settings files therefore still migrate before dispatch.

### line 7627  _(unsure)_

```python
ann_cols = [settings.get('annotation_column')]
```

backward compatibility

### line 7634, trailing  _(unsure)_

```python
ann_vals = settings.get('annotation_values')
```

optional dict {col:[values]}

### line 7640  _(unsure)_

```python
if class_path_list is None:
```

Initialize global collectors (keep class order of first source)

### lines 7663-7666

```python
empty_classes = [
```

Never balance a populated class down to zero merely because another requested label does not occur. Besides discarding valid crops, the old behavior wrote a plausible-looking but empty train/test tree and failed much later in the DataLoader.

### line 7685  _(unsure)_

```python
class_path_list = _balance_lists(class_path_list)
```

Balance to smallest (optional)

### line 7688  _(unsure)_

```python
from .io import generate_dataset_from_lists
```

Write out

### lines 7703-7707

```python
from .classify_classes import _record_generated_folder_names
```

Expose the actual disk classes for downstream training. This is `class_folder_names`, NOT `classes`: what went to disk is a set of FOLDER names, while `classes` is the definition of what each class MEANS (name -> {column, value}). Overwriting the definitions with the folder listing discarded the columns and values the user had set.

### lines 7715-7717

```python
LOG.warning("the cv_dataset settings snapshot was not written (%s); "
```

The dataset is already on disk, so this must not undo it — but the snapshot beside it is how the split gets reproduced, and losing it in silence leaves a training set nobody can rebuild.

## generate_training_dataset._ensure_unique_dir

### line 7235

```python
dst = f"{base}_{j}"
```

Search is intentionally unbounded: every occupied suffix is real.

## generate_training_dataset._load_png_table

### lines 7263-7266

```python
print(f"No 'png_list' rows in {db_path}; falling back to the "
```

No png_list means no PNG folder was ever written. The objects are still in the measurement table, with the same well metadata the metadata rules select on, so fall back to those rather than returning "0 classes" for a project that has everything it needs.

## generate_training_dataset._fix_path_under_src

### lines 7303-7304

```python
if not os.path.isabs(p):
```

A relative path has no recorded root to rebuild from; it is already written relative to the screen, so join it and let the copy report.

## generate_training_dataset._annotation_classes_from_columns

### lines 7384-7385  _(unsure)_

```python
df = png_df.copy()
```

The annotation dispatcher rejects an empty column list before this helper is called, keeping the user-facing error at that boundary.

### line 7387  _(unsure)_

```python
df = png_df.copy()
```

Work with numeric-ish annotations 1/2; accept strings that can be cast to int.

### line 7400  _(unsure)_

```python
col_series = df[col].dropna()
```

Identify annotated values present (castable to int)

### line 7405  _(unsure)_

```python
vals = sorted(set(col_series.tolist()))
```

Non-numeric labels -> keep as-is

### line 7408  _(unsure)_

```python
if ann_vals_filter and col in ann_vals_filter:
```

Optional filter: {col: [allowed_values]}

### line 7413  _(unsure)_

```python
distinct_vals = []
```

Collect classes for each observed value

### line 7422  _(unsure)_

```python
if len(distinct_vals) == 1:
```

If only one annotated value (typical 1-only column), create <col>_random

### line 7427  _(unsure)_

```python
unann_paths = _class_items(df[df[col].isna()])
```

Unannotated = rows where column is NULL/NaN

### line 7437  _(unsure)_

```python
if len(unann_paths) >= pos_n:
```

Sample negatives

### line 7441

```python
rand_paths = unann_paths
```

Not enough; sample all unannotated (and we’ll balance later anyway)

### line 7447  _(unsure)_

```python
if write_rand_col and db_path:
```

Optionally persist a new column in DB and mark sampled as 1

### lines 7459-7462

```python
for p in rand_paths:
```

write 1 for sampled paths; NULL elsewhere (default). An on-demand handle carries the png_path its row named, so the column is written the same way whichever source produced the pixels.

## training_dataset_from_annotation

### line 7766  _(unsure)_

```python
print(f'Reading DataBase: {db_path}')
```

Connect to the database and retrieve the image paths and annotations

### line 7770  _(unsure)_

```python
query = f"SELECT png_path, {annotation_column} FROM png_list"
```

Retrieve all paths and annotations from the database

### line 7783  _(unsure)_

```python
class_paths = []
```

Filter paths based on annotated_classes

### line 7790  _(unsure)_

```python
if len(annotated_classes) == 1:
```

If only one class is provided, create an alternative list by sampling paths from all_paths that are not in the annotated class

### line 7796  _(unsure)_

```python
alt_class_paths = [path for path, annotation in all_paths if annotation != target_class]
```

Filter all_paths to exclude paths that belong to the target class

### line 7804  _(unsure)_

```python
sampled_target_class_paths = random.sample(class_paths[0], balanced_count)
```

Resample target class to match the smaller size

### line 7808  _(unsure)_

```python
class_paths[0] = sampled_target_class_paths
```

Update class paths

## training_dataset_from_annotation_metadata

### line 7869  _(unsure)_

```python
print(f'Reading DataBase: {db_path}')
```

Connect to the database and retrieve the image paths and annotations

### line 7873  _(unsure)_

```python
query = f"SELECT png_path, {annotation_column}, rowID, columnID FROM png_list"
```

Retrieve all paths and annotations from the database

### line 7900  _(unsure)_

```python
class_paths = []
```

Filter paths based on annotated_classes

### line 7907  _(unsure)_

```python
if len(annotated_classes) == 1:
```

If only one class is provided, create an alternative list by sampling paths from all_paths that are not in the annotated class

### line 7913  _(unsure)_

```python
alt_class_paths = [path for path, annotation in all_paths if annotation != target_class]
```

Filter all_paths to exclude paths that belong to the target class

### line 7921  _(unsure)_

```python
sampled_target_class_paths = random.sample(class_paths[0], balanced_count)
```

Resample target class to match the smaller size

### line 7925  _(unsure)_

```python
class_paths[0] = sampled_target_class_paths
```

Update class paths

## generate_dataset_from_lists

### lines 8017-8023

```python
every_item = [item for data in class_data for item in data]
```

Stamp BEFORE the first crop lands, for the reason spacr.crops.stamp_crop_folder gives: a run killed part-way through leaves a marked tree holding fewer crops, never an unmarked tree of corrected ones. The marker goes at the dataset root and describes the whole split -- not inside train/<class>/, because the class folders are enumerated as "the classes" and as "the samples", and a sidecar there would be counted as one of each.

### lines 8072-8082

```python
print(split_report.summary())
```

grouped_split accepts only candidates whose train and test sides both contain every supplied class, or raises before returning. Rechecking each class here duplicated that invariant after the split had already been accepted and could never reject a result. The grouped-split contract is pinned by its own negative test. That test deliberately supplies classes confined to one group. It observes the upstream, actionable refusal rather than this former second copy of the same rule. The accepted split can therefore be persisted directly. Every non-empty class has members on both sides by construction. Empty requested classes are handled below as explicit folders.

### line 8090  _(unsure)_

```python
train_class_dir = os.path.join(dst, f'train/{cls}')
```

Create directories

### lines 8099-8103

```python
print(f"Class {cls!r} selected no crops; its folders are empty.")
```

sklearn answers an empty class with "With n_samples=0, test_size=0.25 ... the resulting train set will be empty", which names the splitter's parameters rather than the rule that selected nothing. Say which class, keep the folder so the class list still matches the tree, and let the summary below flag it.

### lines 8106-8107  _(unsure)_

```python
train_data, test_data = grouped_splits[class_index]
```

Any non-empty class contributed to flat_items, which constructed grouped_splits above; the empty-class continue is the only bypass.

### line 8110  _(unsure)_

```python
for item in train_data:
```

Write train files

### line 8124  _(unsure)_

```python
for item in test_data:
```

Write test files

### line 8138  _(unsure)_

```python
empty = []
```

Print summary. The sidecar is not a crop, so it is not counted.

### lines 8150-8154

```python
print(f"Warning: {failed} of {total_files} crops could not be written "
```

A crop that cannot be written used to take the whole run down with a bare FileNotFoundError from shutil.copy, naming one file and not the scale of the problem. Say how many, and finish the split -- unless nothing landed at all, which is not a partial result but a broken input, and training on it would just be training on nothing.

## convert_separate_files_to_yokogawa

### line 8231  _(unsure)_

```python
for file in sorted(os.listdir(folder)):
```

Group files by (plateID, wellID, fieldID, timeID, chanID)

### line 8246  _(unsure)_

```python
plateID = meta.get('plateID', '1') or '1'
```

Optional metadata with defaults

### lines 8258-8261

```python
source_wells = sorted({region[:2] for region in files_by_region},
```

well assignment, before a single file is written

A well is a well, not a field: keyed on (plateID, wellID) so the two fields of one well do not become two wells, which is what keying on (plateID, wellID, fieldID) did.

### lines 8269-8270

```python
canonical_wells = {}
```

Pass 1: every source well that is a real address keeps it. Sized to the plate the addresses actually need — a folder holding AA13 is a 1536.

### line 8281, trailing  _(unsure)_

```python
continue
```

two source names for one address; pass 2 splits them

### line 8285  _(unsure)_

```python
for key in source_wells:
```

Pass 2: the rest, deterministically, skipping everything pass 1 claimed.

### line 8293  _(unsure)_

```python
for region, file_list in files_by_region.items():
```

Process files per region

### line 8298  _(unsure)_

```python
slice_ids = [sid for _, sid in file_list if sid is not None]
```

Check if multiple slices exist and are meaningful

### line 8307  _(unsure)_

```python
if len(unique_slices) > 1:
```

Perform MIP only if multiple unique slices are present

### line 8319  _(unsure)_

```python
original_files = ";".join(f[0] for f in file_list)
```

Log original filenames involved in MIP or single file rename

## convert_to_yokogawa

### lines 8370-8377

```python
well = _get_next_well(used_wells)
```

os.listdir contributes each filename once, so this file receives one well and every channel/time extracted inside this iteration reuses it. The former filename dictionary was queried before its sole write, so the lookup was always absent and its reuse arm was unreachable. Reuse happens inside this iteration through the local `well` value. Sorted traversal keeps assignments stable between identical runs. Each distinct source file still receives one distinct synthetic well. All planes extracted from that source retain that same assignment.

### lines 8396-8400

```python
mip_image = np.maximum.reduce([
```

np.max is a dispatcher, not a ufunc, so np.max.reduce raised AttributeError before a single frame was read: every ND2 silently produced no TIFF (and the IndexError handler below was dead code). np.maximum is the ufunc.

### lines 8420-8422

```python
ledger.record_failure(
```

A dropped frame silently shrinks the FOV set — record it as its own item so the summary shows how much of the ND2 never made it to disk.

### line 8431  _(unsure)_

```python
czi_reader = pyczi or _load_pylibczi()
```

Open the CZI in streaming mode

### line 8435  _(unsure)_

```python
bbox    = czidoc.total_bounding_box
```

1) Global dimension ranges

### line 8441  _(unsure)_

```python
scenes_bb = czidoc.scenes_bounding_rectangle
```

2) Scene → list of scene indices

### line 8445  _(unsure)_

```python
folder = os.path.dirname(path)
```

3) Output folder (same as .czi)

### line 8448  _(unsure)_

```python
for scene in scenes:
```

4) Loop scene × time × channel × Z

### line 8450  _(unsure)_

```python
scene_well = _get_next_well(used_wells)
```

assign a unique well for this scene

### line 8453  _(unsure)_

```python
F_idx = scene + 1 if scene is not None else 1
```

Field index = scene+1 (or 1 if no scene)

### line 8455  _(unsure)_

```python
A_idx = scene + 1 if scene is not None else 1
```

Scene index for “A”

### line 8461  _(unsure)_

```python
arr = czidoc.read(
```

Read exactly one 2D plane

### line 8468  _(unsure)_

```python
fn = (
```

Build Yokogawa‐style filename:

### line 8480  _(unsure)_

```python
write_tiff(
```

Write with lossless compression

### lines 8504-8509

```python
lif_file = readlif.reader.LifFile(path)
```

readlif's ACTUAL surface, checked against the installed 0.6.5. This block used to call `readlif.Reader`, `getIterImage` and `getFrame` -- an older camelCase API that no longer exists, so every LIF import died with AttributeError on the first line and the whole format was unusable.

### lines 8515-8518

```python
channels = range(getattr(image, 'channels', 1) or 1)
```

CHANNELS ARE NOT IN `dims`. Dims is namedtuple("Dims", "x y z t m"), so `dims.c` never existed and the old getattr default silently pinned every LIF to a single channel.

### line 8551  _(unsure)_

```python
t_dim = c_dim = 1
```

Defaults

### line 8554  _(unsure)_

```python
if ndim == 2:
```

Determine dimensions more explicitly

### line 8563, trailing  _(unsure)_

```python
if images.shape[0] <= 4:
```

Likely channels

### line 8571, trailing  _(unsure)_

```python
else:
```

Z-stack

### lines 8579-8587

```python
try:
```

The two leading axes are t and z, in an order the

SHAPE cannot reveal. This used to assume TZYX unconditionally, so a genuine (Z, T, Y, X) file had every z-plane written out as a "timepoint" and every projection taken over TIME rather than over z - wrong data under a confident filename, with nothing saying so. tifffile records the real order; ask it, and when the file does not declare one, say which way it was read and what that means if it is wrong.

### lines 8616-8617

```python
ledger.finalize(artifact=csv_path)
```

Stamp the artifact, then say so last. rename_log.csv on its own cannot tell you that three of the ten inputs never converted; the sidecar can.

## prepare_cellpose_dataset

### lines 8750-8769

```python
needed = target_size - dataset_len
```

A folder is shorter than target_size only when augmentation is enabled: without it target_size is the minimum folder size. EXACTLY `needed` augmented pairs keep every folder balanced. The branch therefore already proves augmentation was requested. With augmentation off, target_size is min(len(folder)), so every folder takes the sampled branch above and this arm is impossible. Tests exercise unequal folders with augmentation both off and on. Off samples every folder down to the smallest observed count. On grows every shorter folder to the largest observed count. Removing the duplicate inner flag leaves those outputs unchanged. It also makes the invariant visible at the target-size decision. No synthetic augmentation is performed unless the outer sizing rule selected the maximum, which only augment_data=True can do. The number added remains exactly target_size - dataset_len. Original pairs retain their explicit no-augmentation tag below. Generated pairs cycle distinct transform combinations first. Only after exhausting those combinations may one repeat. Sampling order remains randomized with the same random module. Train/test shuffling and indexing are untouched after this block. Thus this simplification removes only an unreachable false arm.

### lines 8773-8774  _(unsure)_

```python
combos = [(img_path, msk_path, aug)
```

Every distinct (pair, augmentation) combination, so a pair is re-augmented differently before any combination repeats.

### line 8785  _(unsure)_

```python
augmented_sampled = [
```

Add "no augmentation" tag to original files

## _listdir_visible

### 2026-09-19, GitHub #121 and #117

```python
return [name for name in os.listdir(folder) if not name.startswith('.')]
```

A Mask run on an Apple M4, plate on `/Volumes/jk-ummi`, died in `generate_cellpose_masks_sam` with numpy's "This file contains pickled (object) data". Nothing spaCR writes into `masks/` is pickled: the normalised archive holds `data` (float32) and `filenames` (a `<U` array), and it loads with `allow_pickle=False`. The #117 log names the file that failed one stage earlier, `stack/._test_N06_5_1.npy`. It is an AppleDouble sidecar. macOS writes `._<name>` beside a file that carries extended attributes when the volume cannot store them natively (exFAT, FAT, many SMB shares). The sidecar keeps the ending of the file it shadows, so every `os.listdir(...) if name.endswith('.npz')` took it for data. `np.load` reads any file that is neither `.npy` magic nor a zip as a pickle, which is why the message says "pickled".

Not `allow_pickle=True`: measured, it turns the ValueError into `UnpicklingError: Failed to interpret file ... as a pickle`, and it would let a file dropped into the folder run code at load. Not convert-on-read either: the files spaCR wrote in 1.5.0.8 were never object arrays and load unchanged. The fix is at the listing.

It surfaced there because the platform is the one that makes sidecars and the Mask path re-lists its own output folders four times (`stack/`, `masks/`, each `*_mask_stack/`, `merged/`). The raw-image listing in `_rename_and_organize_image_files` already skipped dot-files; nothing after it did. `concatenate_and_normalize` met the sidecar first, but its per-item ledger only logged it ("RUN INCOMPLETE - 1 of 2 items failed"). `generate_cellpose_masks_sam` has no ledger around its load, so it raised. The fix touched 37 listing sites in 20 functions: every listing the Mask run reaches in `io.py`, `object.py`, `core.py`, `utils.py` and `plot.py` goes through this helper, and `tests/test_a_stack_file_spacr_wrote_is_one_it_can_read.py` fails if one of them calls `os.listdir` directly again. `seg_qc._iter_masks` and `illumination._merged_files` (reached when segmentation illumination correction is on) filter inline, because `seg_qc` is tested to import no torch and this module imports it at load.

Every dot-file is left out, not just `._`: the atomic writers here name their temporaries `.spacr_tmp_*.npy` and `.spacr_npz_*.npz`, and a run killed mid-write leaves one behind with a data ending. Order is `os.listdir` order, unchanged, so batching and the seeded shuffle see the same sequence they did before.
