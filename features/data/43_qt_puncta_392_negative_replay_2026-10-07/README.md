# Exact 392 Qt puncta crash and bounded local negative replay

Hosted tests run 37550980965, job 112567504138, at source
`392ca4d6a2319fd88c2d352b1a0b021682123d6b` exited 139 in
`tests/qt/test_make_masks_center_pixel_puncta.py::test_choose_detect_save_and_undo_keeps_parents_and_exports_exact_values[True]`
at `qtbot.waitUntil` line 55. The complete job log is archived here.
The hosted Python version was 3.12.14 and Qt/shiboken were 6.11.2;
the local interpreter was Python 3.12.13 with the same Qt version.

The original workflow file-order planner places this node in Qt shard 1,
batch 59 of 241. The exact 16-file batch 59 passed locally (144 tests)
under a 4 GiB cap, CUDA hidden, offscreen Qt, two pytest workers and
fresh XDG caches. Immediately preceding batch 58 passed 252 tests;
batch 59 then passed 144 again with the same caches. The full replay
logs, file list and source/environment receipt are retained here.
The puncta test, Make Masks screen and `tests/conftest.py` have identical
Git blobs at 392 and 649, respectively:

- `12709bee6bc4123bce6d0ff28a47570734a78927`
- `46d8b8f3c95fae227abc2e7416f4e9fe1bc1464f`
- `df78195257599cb5b24baf4b94a7c6a1dd7c8930`

This is a negative local reproduction, not evidence that the hosted
crash was spurious or fixed. It establishes neither native cause nor a
full Qt suite verdict. `archive_puncta.py` regenerates this archive
from the complete scratch logs; `manifest.json` records their hashes.
