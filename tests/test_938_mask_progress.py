"""Item 938: Make Masks prints a Progress line inside the segmentation, per batch of fields."""
from __future__ import annotations

import re
import types

import numpy as np
import pytest

import spacr.object as objects
from tests.test_cov_object_masks_sam import _settings, force_cpu  # noqa: F401
from tests.test_masks_come_back_at_the_field_size import _WorkingScaleModel, FIELD


@pytest.fixture
def two_archives(tmp_path, monkeypatch, force_cpu):  # noqa: F811
    src = tmp_path / "masks"
    src.mkdir()
    pixels = np.full((2, *FIELD, 3), .5, dtype=np.float32)
    for index in range(2):
        np.savez(src / f"stack_{index}_norm.npz", data=pixels,
                 filenames=[f"plate1_E01_{index}_1.npy", f"plate1_E01_{index}_2.npy"])
    monkeypatch.setattr(objects, "cp_models",
                        types.SimpleNamespace(CellposeModel=_WorkingScaleModel))
    return src


def _run(src, **kwargs):
    settings = _settings(src, cell_channel=0, nucleus_channel=1, pathogen_channel=None,
                         batch_size=1, cell_diameter=30, nucleus_diameter=30, seg_qc="off")
    objects.generate_cellpose_masks_sam(str(src), settings, "nucleus", run_qc=False, **kwargs)


def _lines(text):
    return [(int(a), int(b)) for a, b in re.findall(
        r"Progress: (\d+)/(\d+), operation_type: nucleus_mask_gen", text)]


def test_a_progress_line_follows_every_batch_and_counts_all_the_roles(two_archives, capsys):
    _run(two_archives, progress_share=(1, 2))
    # 4 fields, one role already done of two: 4 + 1..4 over 8
    assert _lines(capsys.readouterr().out) == [(5, 8), (6, 8), (7, 8), (8, 8)]


def test_no_progress_line_without_a_share(two_archives, capsys):
    _run(two_archives)
    assert _lines(capsys.readouterr().out) == []
