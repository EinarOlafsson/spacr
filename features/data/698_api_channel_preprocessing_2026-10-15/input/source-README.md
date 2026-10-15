---
license: cc-by-4.0
tags: [microscopy, segmentation, cellpose, toxoplasma, spacr]
---
# Cross-channel nuclei-from-cellmask dataset

3029 paired fields. `images/` is the
cell-mask input channel, `masks/` the matching instance-labelled nuclei ground truth (same stem =
same field). `split_by_well.csv` gives the train/test assignment; the split is **by well**, so no
well contributes to both sides - field-level splitting would leak.

Prepared with [spaCR](https://github.com/EinarOlafsson/spacr)
([PyPI](https://pypi.org/project/spacr/) | [conda-forge](https://anaconda.org/conda-forge/spacr)).

Used to train [einarolafsson/cross-channel-nuclei-from-cellmask-cpsam](https://huggingface.co/einarolafsson/cross-channel-nuclei-from-cellmask-cpsam).
