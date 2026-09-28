# Live GitHub gutter measurement

The equal-gutter criterion **fails** on nightly commit
`90fb0c63f2393b8ab5301e8483b877d5c82ff6e8`, unchanged between the start and
end of measurement. The live README hash matched the local source.

| Viewport width | Horizontal median | Vertical median | Spread |
|---|---:|---:|---:|
| 1440 CSS px | 8.3798828125 | 14.3798828125 | 6 CSS px |
| 1024 CSS px | 5.8193359375 | 11.8193359375 | 6 CSS px |

The measurement combines live DOM image geometry with alpha bounds of the
exact loaded PNG bytes. It checks horizontal neighbors and vertical neighbors
within the same section; category headings are intentional separators. The
allowed spread was 1 CSS pixel. No page markup, CSS or threshold was changed.

The 1024px screenshot visibly covers all 21 tiles. The 1440px screenshot is
offset upward and cuts the lower grid; it is not full-grid visual coverage.
Both numerical viewport measurements completed before the acceptance failure.
Screenshot hashes and image-source hashes are preserved in `receipt.json`;
the screenshot files here are byte-identical to the original scratch copies.

`measure.py` preserves the executed measurement script. Its scratch output
directory was `/mnt/wd4tb/scratch/spacr-completion/366-live-gutters`. It ran
through `run.sh` there under a 4G cap, private HOME/TMPDIR, hidden CUDA and
Chromium `--disable-gpu`. The first attempt failed only because a screenshot
clip lay outside the default viewport; its receipt, log and original script
remain as `attempt1-*` in that scratch directory. The retry used full-page
screenshot coordinates and saved numbers before taking screenshots. It exited
1 on the actual unequal-gutter assertion.
