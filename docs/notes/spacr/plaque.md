# Notes from `spacr/plaque.py`

Prose lifted out of `spacr/plaque.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## detect_wells

### lines 218-220

```python
LOG.warning(
```

Reported, not silently dropped: a rejected well is a condition missing from the results, and a user who is not told will read that as "no plaques grew".

## `_to_detector_channel_order`, and why `detect_wells` takes RGB

Added for instruction 445, and written here by hand rather than lifted out of
a comment.

Ultralytics decodes a file path with OpenCV, so it trains and infers in BGR.
Handed an `H x W x 3` numpy array it converts nothing: it assumes the caller
has already ordered the channels BGR. `spacr/submodules.py` fed it
`cellpose.io.imread(path)`, which is RGB, so every plate photograph spaCR
split into wells was shown to the detector with red and blue exchanged.

**What it cost.** Scored on 2026-09-20 over the published v4 test split -- 129
images, 297 boxes, `einarolafsson/toxoplasma-plaque-well-detector-dataset`,
CPU, conf floor 0.001, imgsz 640, `min_axis_ratio` 0, IoU 0.5, P/R quoted at
the shipped confidence of 0.25:

| model | route | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| v1 (`yolo_welldetect_v3`) | file path | 0.8888 | 0.7696 | 0.6366 | 0.9024 |
| v1 | BGR array | 0.8888 | 0.7696 | 0.6366 | 0.9024 |
| v1 | RGB array | 0.5877 | 0.5199 | 0.5815 | 0.6128 |
| v2 (`yolo_welldetect_v4`) | file path | 0.9480 | 0.8443 | 0.8367 | 0.9663 |
| v2 | BGR array | 0.9480 | 0.8443 | 0.8367 | 0.9663 |
| v2 | RGB array | 0.6972 | 0.6260 | 0.6984 | 0.2963 |

The file-path route and the BGR array agree to four decimal places on every
measure; RGB does not. The file path is how every training image reached
these models, and these file-path numbers reproduce the repository's own
published `qc/test_metrics.json` (v2 0.9457 / 0.8341, v1 0.8838 / 0.7630),
which is what ties the reference route to the training route rather than
assuming it.

**Why the conversion is at `detect_wells`' own door** rather than in each
caller. There is exactly one call into ultralytics in the whole package --
`model.predict` in `detect_wells` -- so one door makes every caller right
instead of each caller remembering. It also matches the rest of spaCR, where
`cellpose.io.imread` is the house reader and RGB is what every other entry
point passes around. The cost is that a caller which was ALREADY converting
now converts twice, which is the original bug wearing a different hat and is
invisible in the array's shape and dtype;
`tools/measure_plaque_detector_transfer.py` was that caller, and
`tests/test_the_plaque_transfer_spike_counts_what_it_saw.py` now fails if a
swap reappears there.

Only an `H x W x 3` array is converted. A two-dimensional greyscale array has
no order to get wrong, reversing the last axis of an `H x W x 4` array would
move alpha to the front rather than exchange red and blue, and a file path is
ultralytics' own canonical input -- all three are passed through as they are.
