"""The v2 pipeline's masks go through the same scorecard as v1's.

`seg_qc` defaults to 'report', so a v1 run scores every mask through
`object._run_seg_qc`, which globs a `<object_type>_mask_stack` folder. v2
has no such folder -- its mask is a CHANNEL of `merged/stack_<field>.npy` --
so the setting was accepted and silently scored nothing. These tests hold
the two layouts to the same meaning of `seg_qc`.
"""

import json
import os

import numpy as np
import pytest

from spacr._v1_v2_bridge import v2_mask_source
from spacr.core import _score_v2_masks


def _plate(tmp_path, fields=3, image_channels=("cell", "nucleus"),
           mask_channels=("cell",), objects=2, shape=(40, 40)):
    """A v2 plate: masks as trailing channels of the merged stacks."""
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    depth = len(image_channels) + len(mask_channels)
    for i in range(fields):
        stack = np.zeros((*shape, depth), np.uint16)
        if mask_channels:
            plane = stack[:, :, len(image_channels)]
            for k in range(objects):
                top = 4 + k * 14
                plane[top:top + 10, top:top + 10] = k + 1
        np.save(merged / f"stack_A01f{i + 1}.npy", stack)
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": list(image_channels),
        "mask_channels": list(mask_channels)}))
    return merged


# --- reading the v2 layout -------------------------------------------------


def test_it_finds_one_reader_per_field(tmp_path):
    _plate(tmp_path)
    assert sorted(v2_mask_source(tmp_path / "merged", "cell")) == [
        "A01f1", "A01f2", "A01f3"]


def test_the_readers_are_lazy(tmp_path):
    """A 1536-field plate must not be loaded to be scored, which is the
    promise the folder path already makes."""
    _plate(tmp_path, fields=4)
    source = v2_mask_source(tmp_path / "merged", "cell")
    assert all(callable(v) for v in source.values())


def test_it_reads_the_mask_plane_not_an_image_plane(tmp_path):
    """The image channels are empty here, so a wrong plane scores zero."""
    _plate(tmp_path, fields=1, objects=2)
    mask = list(v2_mask_source(tmp_path / "merged", "cell").values())[0]()
    assert sorted(np.unique(mask)) == [0, 1, 2]


def test_it_uses_channel_order_rather_than_assuming_the_last_plane(tmp_path):
    """Two masks, and the wanted one is not last."""
    _plate(tmp_path, fields=1, image_channels=("cell",),
           mask_channels=("cell", "nucleus"))
    merged = tmp_path / "merged"
    stack = np.load(merged / "stack_A01f1.npy")
    stack[:, :, 2] = 7          # the nucleus plane, a value cell never has
    np.save(merged / "stack_A01f1.npy", stack)
    mask = list(v2_mask_source(merged, "cell").values())[0]()
    assert 7 not in np.unique(mask)


def test_a_missing_sidecar_scores_nothing_rather_than_guessing(tmp_path):
    (tmp_path / "merged").mkdir()
    assert v2_mask_source(tmp_path / "merged", "cell") == {}


def test_a_sidecar_naming_no_mask_scores_nothing(tmp_path):
    _plate(tmp_path, mask_channels=())
    assert v2_mask_source(tmp_path / "merged", "cell") == {}


def test_one_unnamed_mask_is_scored_anyway(tmp_path):
    """Refusing over a naming difference would report "no masks" about a
    plate that has them."""
    _plate(tmp_path, fields=1, mask_channels=("object",))
    assert len(v2_mask_source(tmp_path / "merged", "cell")) == 1


def test_several_masks_and_no_name_match_scores_nothing(tmp_path):
    """With a choice to make, guessing would score the wrong object."""
    _plate(tmp_path, fields=1, mask_channels=("nucleus", "pathogen"))
    assert v2_mask_source(tmp_path / "merged", "cell") == {}


# --- scoring ---------------------------------------------------------------


def test_a_v2_run_writes_the_same_scorecard_v1_does(tmp_path):
    _plate(tmp_path)
    result = _score_v2_masks(tmp_path, {"seg_qc": "report", "verbose": False})
    assert result["mode"] == "report"
    assert len(result["field_qcs"]) == 3
    assert (tmp_path / "qc" / "segmentation_qc_cell.csv").exists()


def test_the_objects_it_counts_are_the_ones_in_the_mask(tmp_path):
    _plate(tmp_path, fields=2, objects=2)
    result = _score_v2_masks(tmp_path, {"seg_qc": "report", "verbose": False})
    assert [q.n_objects for q in result["field_qcs"]] == [2, 2]


def test_off_scores_nothing_and_writes_nothing(tmp_path):
    _plate(tmp_path)
    assert _score_v2_masks(tmp_path, {"seg_qc": "off"}) is None
    assert not (tmp_path / "qc").exists()


def test_flag_mode_records_flags_into_settings(tmp_path):
    """What a downstream step consumes, same as the v1 path."""
    settings = {"seg_qc": "flag", "verbose": False}
    _plate(tmp_path)
    _score_v2_masks(tmp_path, settings)
    assert "seg_qc_flags" in settings


def test_a_plate_with_no_masks_says_so_and_returns(tmp_path, capsys):
    (tmp_path / "merged").mkdir()
    assert _score_v2_masks(tmp_path, {"seg_qc": "report"}) is None
    assert "no cell masks to score" in capsys.readouterr().out


def test_an_undescribable_stack_is_flagged_rather_than_skipped(tmp_path):
    """A sidecar describing a plane the stack does not have.

    The field comes back flagged `unreadable_mask` and the plate FAILS,
    which is better than the outer skip this test first expected: a mask
    nobody can read is a segmentation problem the user has to see, not a
    scorecard bug to swallow. Hours of segmentation are still not lost --
    nothing raises.
    """
    _plate(tmp_path, fields=1)
    merged = tmp_path / "merged"
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": ["a", "b", "c", "d"], "mask_channels": ["cell"]}))
    result = _score_v2_masks(tmp_path, {"seg_qc": "report", "verbose": False})
    assert result is not None
    assert ["unreadable_mask"] == list(result["field_qcs"][0].flags)


def test_the_v2_branch_calls_it():
    """Wired, not merely defined -- the original bug was exactly a check
    that nothing invoked."""
    import inspect

    from spacr import core

    assert "_score_v2_masks(src, settings" in inspect.getsource(
        core.preprocess_generate_masks)


# --- the seg_qc change that made it possible -------------------------------


def test_seg_qc_accepts_a_thunk_without_materialising_the_plate():
    from spacr.seg_qc import score_masks

    mask = np.zeros((32, 32), int)
    mask[4:12, 4:12] = 1
    calls = []

    def thunk():
        calls.append(1)
        return mask

    scored = score_masks({"f1": thunk}, object_type="cell")
    assert len(scored) == 1 and calls == [1]


def test_a_plain_mask_in_a_mapping_still_works():
    """The change must not break the mapping form that already existed."""
    from spacr.seg_qc import score_masks

    mask = np.zeros((32, 32), int)
    mask[4:12, 4:12] = 1
    assert len(score_masks({"f1": mask}, object_type="cell")) == 1
