# F548 native channel-wise normalization, 2026-10-06

Source/test change: `66a1dcc9267b69a1d4288394ebf40bfd611fd191`.
Baseline `spacr/io.py` blob: `f4986140678c47062ccf4f367642b418072dc30a`.
Candidate `spacr/io.py` blob: `c783869f041b1bac75dd402d621c05129191731d`.

The native mapped TZYXC Mask ingest saves each acquired timepoint first, then
normalizes one channel at a time from its private staged NPYs. The existing
scalar percentile, background, float32, channel-order and NPZ publication
behavior is shared with the ordinary normalization path. Staged file hashes
are checked before and after reading; cancellation or damage leaves no
published stack or archive. The full float32 normalized field still resides
in memory. This is a bounded reduction, not arbitrary-series scalability.

## Paired full-ingest receipt

Both capped 4 GiB, CUDA-hidden processes used the same copied converted
2T × 2Z × 2C, 2048 × 2048 uint16 TIFF fixture and the same `measure.py`.
There were eight TIFFs, 67,110,912 total bytes, with identical ordered-TIFF
SHA256 aggregate `e1437298586ad0dbe43d02c9f7cb92eafe2c5ea317a50ea5d9eeed788cf06fd6`.
The source directory and derived receipt paths necessarily differ between
copies. This is one paired local observation, not a timing or memory bound.

| Source | Peak process RSS | Elapsed |
| --- | ---: | ---: |
| Baseline | 1668.24 MiB | 2.116 s |
| Candidate | 1635.77 MiB | 1.991 s |

Peak RSS was 32.47 MiB lower in the candidate. Both produced byte-identical
raw `plate1_A01_1_1.npy` (`2f778ebf93b5b3e624112c650992a09d9fa3738bc0bf6e8d15774232acab3d7d`),
`plate1_A01_1_2.npy` (`be03ccd5409ecef6569eb8be3da811536d71510baff3d96b8db198804aa93d5e`),
and compressed `plate1_A01_1_norm_timelapse.npz`
(`ae16d1ed6ed7cc5341e43daf467814f0379f70b6109d190527baf10f2c6c932c`).
Exact process output is in `full-baseline.log` and `full-candidate.log`.

## Focused validation

Six neighboring normalization/native/watch/raw-volume test files passed
188 cases under 4 GiB. The final two-file branch run passed 88 cases and
covered every new `spacr/io.py` statement and branch. The three public
docstring guard nodes passed, as did fatal Ruff and `git diff --check`.
The 3T × 3Z × 3C regression compares complete float32 bytes, channel order
and compressed NPZ bytes against the old scalar stack path. Stage mutation
before normalization, during mmap loading, and after normalization, plus
cancellation between channel reads, all refuse publication and remove the
private stage.

The protected uninterrupted Qt serial acceptance and real vendor-field
scalability remain open; neither is established by this local fixture.
