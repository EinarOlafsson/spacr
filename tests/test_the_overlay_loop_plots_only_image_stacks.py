"""The batch overlay loop draws merged image stacks, never their sidecars.

`preprocess_generate_masks` with ``plot=True`` used to list ``merged/``
unfiltered, shuffle it and plot the first ``examples_to_plot`` entries, so the
``.spacr_plane_layout.json`` that the merge step writes beside the stacks
could be handed to the overlay plot as an image. The test-mode example count
already counted ``.npy`` files only; the list now agrees with it.
"""
from __future__ import annotations

import numpy as np

from spacr.core import _overlay_candidates


def test_a_sidecar_json_is_not_offered_as_an_image(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate1_A01_1.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    np.save(merged / "plate1_A01_2.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    (merged / ".spacr_plane_layout.json").write_text("{}", encoding="utf-8")
    (merged / "notes.txt").write_text("x", encoding="utf-8")

    assert sorted(_overlay_candidates(str(merged))) == [
        "plate1_A01_1.npy", "plate1_A01_2.npy"]


def test_the_list_and_the_test_mode_count_agree(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    for index in range(3):
        np.save(merged / f"f{index}.npy", np.zeros((2, 2), dtype=np.uint8))
    (merged / ".spacr_plane_layout.json").write_text("{}", encoding="utf-8")

    counted = len([f for f in __import__("os").listdir(merged)
                   if f.endswith(".npy")])
    assert len(_overlay_candidates(str(merged))) == counted == 3
