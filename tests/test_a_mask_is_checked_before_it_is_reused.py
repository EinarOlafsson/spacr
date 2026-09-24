"""A mask a killed run cut short is never reused, with or without resume.

Item 430 (GitHub #118, #124) made every stack and normalised archive a Mask
re-run reuses atomic and checked. It left two things, both in the Codex lane
at the time:

* ``adjust_cells`` rewrites ``masks/cell_mask_stack/*.npy`` in place with
  ``np.save``. A kill during that write left a truncated cell mask under its
  final name.
* ``masks/<object>_mask_stack/*.npy`` are checked before reuse only when
  ``resume`` is on. Without it, :func:`spacr.utils.check_mask_folder` counted
  every file and :func:`spacr.io._check_masks` skipped every file that
  existed, so a truncated mask was taken as done and the merge died on it or
  the plate was reported finished without it.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (  # noqa: E402,F401
    _RUN, _settings, _stack_names, fake_cellpose, plate,
)


def _mask(value=1, shape=(32, 32)):
    mask = np.zeros(shape, np.uint16)
    mask[4:12, 4:12] = value
    mask[18:28, 18:28] = value + 1
    return mask


def _cut_short(path: Path) -> Path:
    """Leave ``path`` as an in-place ``np.save`` killed half-way leaves it."""
    data = Path(path).read_bytes()
    Path(path).write_bytes(data[: len(data) // 2])
    return path


def test_a_kill_while_adjusting_cells_leaves_the_mask_whole(
        tmp_path, monkeypatch):
    """The adjusted cell mask is written beside the old one and renamed over it.

    ``np.save`` is made to die half-way, as a SIGKILL or a full disk does.
    Before the fix it was writing onto the cell mask itself, which was left
    truncated under its final name.
    """
    from spacr.resume import validate_merged_field
    from spacr.utils import process_mask_file_adjust_cell

    folders = {}
    for role in ("pathogen", "cell", "nucleus"):
        folders[role] = tmp_path / "masks" / f"{role}_mask_stack"
        folders[role].mkdir(parents=True)
        np.save(folders[role] / "f1.npy", _mask())
    original = (folders["cell"] / "f1.npy").read_bytes()

    real_save = np.save

    def dies_half_way(file, arr, *args, **kwargs):
        buffer = io.BytesIO()
        real_save(buffer, arr, *args, **kwargs)
        data = buffer.getvalue()[: len(buffer.getvalue()) // 2]
        if hasattr(file, "write"):
            file.write(data)
        else:
            with open(file, "wb") as handle:
                handle.write(data)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np, "save", dies_half_way)
    with pytest.raises(OSError):
        process_mask_file_adjust_cell(
            "f1.npy", str(folders["pathogen"]), str(folders["cell"]),
            str(folders["nucleus"]))
    monkeypatch.setattr(np, "save", real_save)

    cell = folders["cell"] / "f1.npy"
    assert validate_merged_field(str(cell)) == (True, "done")
    assert cell.read_bytes() == original
    assert sorted(os.listdir(folders["cell"])) == ["f1.npy"]


def _mask_folder(root: Path, fields=("f1", "f2")):
    (root / "stack").mkdir(parents=True)
    masks = root / "masks" / "cell_mask_stack"
    masks.mkdir(parents=True)
    for field in fields:
        np.save(root / "stack" / f"{field}.npy", np.zeros((8, 8, 2), np.uint16))
        np.save(masks / f"{field}.npy", _mask(shape=(8, 8)))
    return masks


def test_the_mask_count_checks_each_mask_with_resume_off(tmp_path, capsys):
    """Two stacks and two masks, one of them cut short, are not all masks."""
    from spacr.utils import check_mask_folder

    masks = _mask_folder(tmp_path)
    assert check_mask_folder(str(tmp_path), "cell_mask_stack") is False
    capsys.readouterr()

    _cut_short(masks / "f2.npy")
    assert check_mask_folder(str(tmp_path), "cell_mask_stack",
                             resume=False) is True
    assert "All masks have been generated" not in capsys.readouterr().out


@pytest.mark.parametrize("resume", [False, True])
def test_an_unrelated_mask_cannot_stand_in_for_a_missing_field(tmp_path, resume):
    from spacr.utils import check_mask_folder

    masks = _mask_folder(tmp_path)
    (masks / "f2.npy").rename(masks / "other_plate.npy")
    assert check_mask_folder(str(tmp_path), "cell_mask_stack", resume=resume)


def test_extra_masks_do_not_force_complete_fields_to_run_again(tmp_path):
    from spacr.utils import check_mask_folder

    masks = _mask_folder(tmp_path)
    np.save(masks / "other_plate.npy", _mask(shape=(8, 8)))
    assert not check_mask_folder(str(tmp_path), "cell_mask_stack")


@pytest.mark.parametrize("resume", [False, True])
def test_a_damaged_mask_is_generated_again_and_named(tmp_path, capsys, resume):
    """The per-batch filter returns a damaged mask for segmenting, and says so."""
    from spacr.io import _check_masks

    masks = _mask_folder(tmp_path)
    _cut_short(masks / "f2.npy")

    batch, names = _check_masks(
        [np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))],
        ["f1.npy", "f2.npy", "f3.npy"], str(masks), resume=resume)

    assert names == ["f2.npy", "f3.npy"]
    assert len(batch) == 2
    out = capsys.readouterr().out
    assert f"{masks / 'f2.npy'} is damaged (truncated)" in out
    assert "f1.npy is damaged" not in out


def test_a_mask_run_without_resume_segments_a_truncated_mask_again(
        plate, fake_cellpose, capsys):
    """The user's path: Mask ran, a cell mask was cut short, Mask runs again.

    Before the fix, with resume off, the second run said "All masks have
    been generated for cell_mask_stack" and merged the truncated mask.
    """
    from spacr.core import preprocess_generate_masks
    from spacr.resume import validate_merged_field

    preprocess_generate_masks(_settings(plate, preprocess=True, **_RUN))
    fields = _stack_names(plate / "stack")
    assert _stack_names(plate / "merged") == fields
    cell_masks = plate / "masks" / "cell_mask_stack"
    damaged = _cut_short(cell_masks / fields[1])
    capsys.readouterr()

    try:
        preprocess_generate_masks(_settings(plate, preprocess=False,
                                            resume=False, **_RUN))
    except Exception as exc:                                 # noqa: BLE001
        pytest.fail(f"the re-run died on the truncated mask: "
                    f"{type(exc).__name__}: {exc}")

    out = capsys.readouterr().out
    assert "All masks have been generated for cell_mask_stack" not in out
    assert f"{damaged} is damaged" in out
    assert validate_merged_field(str(damaged)) == (True, "done")
    assert _stack_names(plate / "merged") == fields
    merged = np.load(plate / "merged" / fields[1])
    assert merged.shape[-1] == 2 + 2
