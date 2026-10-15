---
license: cc0-1.0
pretty_name: spaCR example data, Make Masks on FIB-SEM
task_categories:
- image-segmentation
tags:
- electron-microscopy
- fib-sem
- volume-em
- mitochondria
- 3d
- spacr
size_categories:
- n<1K
---

# spaCR example data: Make Masks on a FIB-SEM volume

One focused-ion-beam scanning electron microscopy (FIB-SEM) volume of a HeLa
cell, 128 x 256 x 256 voxels (Z, Y, X; 8-bit), with a 3-D instance label map
of every mitochondrion (uint16), for the 3-D mask editor on spaCR's Make Masks
"Load test data" menu. One menu entry opens the volume unsegmented; the other
opens it with these labels loaded for correction.

## Files

- `hela_fibsem.tif`: the volume, an ImageJ hyperstack (axes ZYX, unit nm,
  15 nm spacing).
- `ground_truth_masks/hela_fibsem.tif`: the labels, the same shape and
  calibration.
- `manifest.csv`: what a complete copy holds, with source and crop.

## Source

EMPIAR-10982, *Seven benchmark datasets of instance segmentation of
mitochondria: 6 diverse volume EM + 1 TEM (100 images) datasets*, folder
`data/mito_benchmarks/hela_cell` (Narayan K, Conrad RW; NCI/FNLCR).
https://www.ebi.ac.uk/empiar/EMPIAR-10982/ - DOI 10.6019/EMPIAR-10982

The crop is the central 128 sections (Z 64-191) of the 256^3 benchmark
volume at its full field. The image is unchanged; the labels are renumbered
1..n and stored as uint16. EMPIAR lists a 150 angstrom (15 nm) pixel and no
separate section thickness, so 15 nm is written on all three axes. The Z
value is an assumption: check it before measuring physical volumes.

Built by `tools/build_fibsem_example_dataset.py` in the spaCR repository.

## Licence

CC0 1.0, as EMPIAR releases its entries.

## Citation

Conrad R, Narayan K. Instance segmentation of mitochondria in electron
microscopy images with a generalist deep learning model trained on a diverse
dataset. Cell Systems 14(1):58-71.e5 (2023). doi:10.1016/j.cels.2022.12.006

Narayan K, Conrad RW. EMPIAR-10982 (2022). doi:10.6019/EMPIAR-10982
