MODEL COMPARE: AGREEMENT BETWEEN TWO GENUINE SAVED MASKS
=====================================================

This is a pure Python API example, not a repaired GUI or an accuracy benchmark.
In the spaCR environment, unzip into a new folder and run from that folder:

  python model_compare_example.py --source inputs --output ../my_new_mask_comparison

The output folder must not exist and must be outside inputs. The recording
used spaCR 1.5.0.7 nightly. No GPU, model download or inference is needed here.
ops_geometry_example.py is included only because it supplies the streaming
checksum helper; importing it does not run an OPS workflow or access OPS data.

WHAT THESE FILES ARE
--------------------
inputs/image.tif is the unchanged real cell_pair_02 field from the existing
Apply Cellpose tutorial. inputs/batch.tif is that tutorial's actual saved
batch mask. inputs/preview.npy is its actual unfiltered live-preview mask.
All three are 512 by 512 pixels. Their exact hashes are pinned in the script
and source_manifest.json, so exchanging a similarly named field is rejected.
The original field is itself a documented crop from the downloadable tutorial
images; the original Apply lesson records its acquisition and crop recipe.

Both masks used the SAME stock cpsam model with different preprocessing.
The batch path used spaCR's explicit percentile normalization followed by
Cellpose normalize=False; the live preview used raw-image Cellpose defaults.
The Apply tutorial's independent reference checked these actual arrays.
They have NOT been independently annotated as correct. Neither is ground
truth, and their agreement does not decide which preprocessing is better.
No competing model weights, new training, or held-out accuracy are tested.

The current Model Compare GUI passes an unsupported invert argument to
Cellpose 4.2.1.1. Loading real fields in that screen works, but this example
does not call its inference backend, insert results into it, or fix it.
The working API demonstrated here is spacr.model_compare.compare_masks.

READING THE RESULTS
-------------------
The actual example has 94 objects in A, 95 in B, and 91 matched pairs at the
API's default IoU threshold. Every matched pair's overlap is independently
recounted from its real pixels. The identical-mask control must report full
agreement. The CSV and run.json retain the comparison and checks.

Mean matched IoU describes only matched pairs, not every object or pixel.
The matched fraction is 2 * matched_pairs / (objects_A + objects_B).
The background-excluded adjusted Rand index is also an agreement measure.
Neither metric is a precision, recall, or biological-accuracy measurement.

comparison.png shows both saved-mask outlines on the same original image.
Its third panel marks foreground/background disagreement only; it does NOT
show every instance split, merge or label disagreement. Inspect the masks
and directional split/merge columns as well as the aggregate scores.

Original inputs are read only and rechecked after the run. Preserve their
geometry, identity, preprocessing and checksums when comparing real models.
Use independently reviewed annotations and an appropriate held-out design
before drawing accuracy or model-selection conclusions.
