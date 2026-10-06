# F548 mapped native normalization workspace, 2026-10-06

Source/test commits: `15543bccb7bea5a0fcc46776f196e68196542565`,
`855419584077cf485f3c9a87e258b82090d401d9`, and
`e48a5ed122846716bef50a875220009fe386b787`.
The baseline `spacr/io.py` is blob `aaa64503318ef9118017555cdf2b4b3ad1bb4e03`
from `1248fbb3473a0ef8163406c7b157681a5cf4ec6d`, SHA-256
`32c291b23b819b2b2ce1ca853e31026411d343f1dd41e7805a3cdda08ea4e3f4`.
The measured candidate `spacr/io.py` SHA-256 is
`6ef38e31d93111a805626da5d31516d1544967ab045269f85a15b0aafe560a76`.

The native T-by-Z path still saves exact raw stacks first. It then holds a
selected-channel float32 output map, one mutable source-channel map, and a
compact scalar-order nonzero quantile map instead of materializing a whole
field selected array plus boolean/quantile copies. It uses the existing
background, percentile, per-plane rescale, staged-hash and atomic publication
rules. Before writing any private mapped data, a disk preflight budgets these
three mapped payloads and a staged NPZ bound derived from the actual filename
array. On Linux each map reserves its file blocks with `posix_fallocate`;
filesystems without that operation use bounded ordinary writes with a
cancellation checkpoint every MiB. ENOSPC and cancellation close all private
maps and discard the stage. Concurrent outside disk consumption remains
outside this preflight's guarantee.

## Same-fixture measurement

`benchmark.py` is the exact bounded script used for both processes. It
deterministically generates 2T × 2Z × 4C, 1280 × 1280 uint16 TIFF data,
converts it, and ingests selected channels `[3, 1]`. Separate baseline and
candidate processes each ran under a hard 4 GiB cgroup, with CUDA hidden.
The script retains this workstation's scratch path for an exact local replay;
`git show 1248fbb3473a0ef8163406c7b157681a5cf4ec6d:spacr/io.py` recovers
its baseline source. Full process output is preserved in `final-baseline.log`
and `final-candidate.log`.

| Source | Peak VmRSS | Peak RssAnon | Peak RssFile | Ingest time |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1,342,432 KiB | 878,900 KiB | 479,156 KiB | 2.758 s |
| Candidate | 1,265,096 KiB | 726,508 KiB | 540,548 KiB | 2.855 s |

The candidate had 75.5 MiB lower peak RSS and 148.8 MiB lower peak anonymous
RSS in this one paired sample; file-backed pages increased. This is not a
timing bound. Both archives were 31,779,821 bytes and byte-identical with
SHA-256 `37be3fd9789580363ea0116a638b259cdf1f5d847d4341f53f46a9631a14b0fa`.
Their arrays were float32 with shape `(2, 2, 1280, 1280, 2)`.

## Focused behavior

The final four-file native/watch cohort passed 139 tests under the 4 GiB
cap. It includes complete-array and compressed-NPZ parity against the old
scalar normalizer across source dtypes, channel order, role backgrounds,
lower percentiles and blanks. It also covers staged-source mutation,
cancellation during channel and quantile work, low disk preflight, ENOSPC at
each of three reservations, late-Z cancellation and interrupted fallback
reservation. Fatal Ruff and `git diff --check` passed.

This reduces one ingest phase's measured peak. The TIFF staging and mapped
output still consume storage, mapped pages count toward RSS, and downstream
Mask loading decompresses the whole NPZ. Native vendor-data scalability and
the protected uninterrupted Qt serial criterion remain open.
