# F548 selected-channel native normalization, 2026-10-06

Source/test commit `b3bb8cfb682b323563029f69f68fc1995427083a`.
Baseline `spacr/io.py` blob `c783869f041b1bac75dd402d621c05129191731d`;
candidate blob `aaa64503318ef9118017555cdf2b4b3ad1bb4e03`.

The private native TZYXC ingest path now allocates float32 normalized output
for selected channels only, writing each source channel into its requested
destination position. It no longer creates a full-source-channel normalized
array or a later advanced-index selection copy. The ordinary scalar
`_normalize_img_batch` keeps its full-shape output and existing duplicate
channel behavior. Native source hashing, cancellation, raw stack output and
receipt rules are unchanged.

Scratch parity compared 36 source-dtype, save-dtype, lower-percentile and
role-setting combinations with reordered, nonconsecutive and zero-only
channels. Complete array bytes and compressed NPZ bytes matched. The real
3T × 3Z × 4C native integration case selecting source channels `[3, 1]`
matches the scalar full-stack oracle in float32 bytes and compressed NPZ
bytes; the existing all-channel case still matches as well. A six-file
native/normalizer/watch/raw-volume cohort passed 191 tests under 4 GiB. A
focused 89-test branch run covered every new `spacr/io.py` statement and
arc. Fatal Ruff and diff checks passed.

## Paired full-ingest receipt

Both capped 4 GiB, CUDA-hidden processes used copied Convert output from
the same 2T × 2Z × 4C, 1536 × 1536 uint16 TIFF fixture, selecting `[3, 1]`.
There were 16 TIFFs, 75,501,568 total bytes, with identical ordered-TIFF
SHA256 aggregate `d4751eac370e041d2d356ace74764947b4f0c79e0f9829be96a150fc9cf67c7c`.
The fixture and both outputs occupied 290 MiB of scratch. The scripts and
exact process output are archived beside this note; large TIFFs and output
arrays are intentionally omitted.

| Source | Peak process RSS | Elapsed |
| --- | ---: | ---: |
| Baseline | 1534.62 MiB | 1.384 s |
| Candidate | 1462.74 MiB | 1.207 s |

Peak RSS was 71.88 MiB lower in this one paired local observation; this is
not a timing or memory bound. Both produced byte-identical raw
`plate1_A01_1_1.npy` (`4cef859ccd0415f9a121bb1a77abbfbddc5b1c00c56dc87d42a92448aed4ff30`),
`plate1_A01_1_2.npy` (`0d2ac3f1fa7e67b685083e6ddef2df8aa56d9fffb06f29bb0d6c60152f7f8c52`),
and compressed `plate1_A01_1_norm_timelapse.npz`
(`76b271a7d6992b2af7c75176e686fb57356ee764f44bb50a611386b6ab9b7a3e`).
Path-dependent source receipts differ between the two copied directories.

The selected-channel float32 normalized field still resides in memory.
Default paths retaining every source channel gain no material allocation
reduction here. Original vendor-scale data and uninterrupted serial Qt
acceptance remain open requirements.
