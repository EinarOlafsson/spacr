"""Mask layers in Make Masks: the Qt-free half.

The click rules of the Pick tool, the outlines drawn per layer, the
decisions log, the default save folder, and ``spacr-make-masks --layers``.
The screen half is ``tests/qt/test_make_masks_mask_layers.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import cli_make_masks
from spacr.curation_queue import STATUS_FILENAME
from spacr.qt import mask_engine as engine


def _labels(*boxes, shape=(24, 24)):
    out = np.zeros(shape, dtype=np.uint16)
    for label, (r0, r1, c0, c1) in boxes:
        out[r0:r1, c0:c1] = label
    return out


def test_a_click_on_a_curated_object_removes_it():
    curated = _labels((1, (2, 6, 2, 6)), (2, (10, 14, 10, 14)))
    out, action = engine._layer_pick(curated, curated.copy(), [], 3, 3, 9)
    assert action == {"kind": "remove", "id": 1}
    assert 1 not in out and 2 in out
    assert 1 in curated, "the overlay passed in was changed in place"


def test_a_second_click_restores_a_removed_base_object():
    base = _labels((1, (2, 6, 2, 6)), (2, (10, 14, 10, 14)))
    removed, _ = engine._layer_pick(base.copy(), base, [], 3, 3, 9)
    restored, action = engine._layer_pick(removed, base, [], 3, 3, 9)
    assert action == {"kind": "restore", "id": 1}
    assert np.array_equal(restored, base)


def test_a_hidden_base_cannot_be_restored_so_another_layer_is_picked():
    base = _labels((1, (2, 6, 2, 6)))
    model = _labels((5, (1, 7, 1, 7)))
    curated = np.zeros_like(base)
    out, action = engine._layer_pick(curated, None, [(1, model)], 3, 3, 9)
    assert action == {"kind": "add", "id": 9, "layer": 1, "source_id": 5}
    assert np.array_equal(out == 9, model == 5)


def test_an_added_object_fills_only_background():
    curated = _labels((1, (2, 6, 2, 6)))
    model = _labels((7, (4, 10, 4, 10)))
    out, action = engine._layer_pick(curated, curated.copy(), [(2, model)], 8, 8, 3)
    assert action["kind"] == "add" and action["id"] == 3
    assert np.array_equal(out == 1, curated == 1), "a curated object lost pixels"
    assert np.array_equal(out == 3, (model == 7) & (curated == 0))


def test_the_first_visible_layer_wins_and_background_changes_nothing():
    a = _labels((4, (10, 14, 10, 14)))
    b = _labels((8, (10, 14, 10, 14)))
    curated = np.zeros_like(a)
    _, action = engine._layer_pick(curated, None, [(1, a), (2, b)], 11, 11, 20)
    assert action["layer"] == 1 and action["source_id"] == 4
    assert engine._layer_pick(curated, None, [(1, a)], 0, 0, 20) == (None, None)


def test_an_id_too_large_for_the_overlay_widens_it():
    curated = np.zeros((8, 8), dtype=np.uint8)
    model = _labels((1, (0, 4, 0, 4)), shape=(8, 8))
    out, action = engine._layer_pick(curated, None, [(1, model)], 1, 1, 300)
    assert out.dtype == np.uint16 and action["id"] == 300 and 300 in out


def test_removed_base_objects_are_dashed_and_kept_ones_solid():
    base = _labels((1, (2, 8, 2, 8)), (2, (12, 18, 12, 18)))
    curated = base.copy()
    curated[curated == 1] = 0
    rgba = engine._layer_overlay([(base, "#ff0000", True)], curated, 0)
    solid = rgba[12:18, 12:18, 3] > 0
    dashed = rgba[2:8, 2:8, 3] > 0
    ring2 = np.zeros((6, 6), bool)
    ring2[[0, -1], :] = ring2[:, [0, -1]] = True
    assert np.array_equal(solid, ring2), "a kept object's outline is not solid"
    assert 0 < dashed.sum() < ring2.sum(), "a removed object's outline is not dashed"
    assert tuple(rgba[12, 12]) == (255, 0, 0, 255)


def test_each_visible_layer_has_its_own_colour_and_hidden_ones_none():
    a = _labels((1, (2, 8, 2, 8)))
    b = _labels((1, (12, 18, 12, 18)))
    curated = a.copy()
    rgba = engine._layer_overlay([(a, "#00ff00", True), (b, "#0000ff", True)],
                                 curated, 0)
    assert tuple(rgba[2, 2]) == (0, 255, 0, 255)
    assert tuple(rgba[12, 12]) == (0, 0, 255, 255)
    hidden = engine._layer_overlay([(a, "#00ff00", True), (b, "#0000ff", False)],
                                   curated, 0)
    assert hidden[12, 12, 3] == 0
    assert engine._layer_overlay([(a, "#00ff00", False)], curated, 0) is None


def test_the_decisions_log_names_removals_and_additions_with_their_layer():
    base = _labels((1, (2, 6, 2, 6)), (2, (10, 14, 10, 14)))
    curated = base.copy()
    curated[curated == 2] = 0
    curated[18:22, 18:22] = 3
    added = {3: {"layer": "/x/model", "source_id": 11},
             4: {"layer": "/x/model", "source_id": 12}}
    log = engine._layer_decisions("f.tif", "/x/masks", base, curated, added,
                                  saved="2026-10-25T12:00:00")
    assert log["removed"] == [2]
    assert log["added"] == [{"id": 3, "layer": "/x/model", "source_id": 11}]
    assert log["base_layer"] == "/x/masks" and log["image"] == "f.tif"
    carried = engine._layer_decisions("f.tif", "/x/masks", curated, curated, {},
                                      previous={"removed": [2]})
    assert carried["removed"] == [2], "a removal recorded earlier was dropped"


def test_the_default_save_folder_is_new_and_dated_or_the_newest_there(tmp_path):
    assert engine._default_curated_folder(str(tmp_path), "20261025") == str(
        tmp_path / "masks_curated_20261025")
    (tmp_path / "masks_curated_20261001").mkdir()
    (tmp_path / "masks_curated_20261020").mkdir()
    assert engine._default_curated_folder(str(tmp_path), "20261025") == str(
        tmp_path / "masks_curated_20261020")


def test_a_layer_is_matched_by_stem_and_a_wrong_shape_refused(tmp_path):
    imageio.imwrite(tmp_path / "f_00.tif", _labels((3, (0, 4, 0, 4))))
    assert engine._read_layer_mask(str(tmp_path), "f_00.png", (24, 24)).max() == 3
    assert engine._read_layer_mask(str(tmp_path), "f_01.tif", (24, 24)) is None
    with pytest.raises(ValueError):
        engine._read_layer_mask(str(tmp_path), "f_00.tif", (30, 30))


# ---------------------------------------------------------------------------
# spacr-make-masks --layers
# ---------------------------------------------------------------------------

@pytest.fixture
def three_layers(tmp_path: Path) -> Path:
    folder = tmp_path / "plaques"
    for name in ("masks", "model_a", "model_b"):
        (folder / name).mkdir(parents=True)
    rng = np.random.default_rng(1)
    for index in range(3):
        imageio.imwrite(folder / f"f_{index}.tif",
                        rng.integers(0, 4000, (24, 24), dtype=np.uint16))
        imageio.imwrite(folder / "masks" / f"f_{index}.tif", _labels((1, (2, 6, 2, 6))))
        imageio.imwrite(folder / "model_a" / f"f_{index}.tif", _labels((1, (10, 14, 10, 14))))
        imageio.imwrite(folder / "model_b" / f"f_{index}.tif", _labels((1, (16, 20, 2, 6))))
    return folder


@pytest.fixture
def handed(monkeypatch):
    caught = []

    def _fake(queue):
        caught.append((queue, cli_make_masks._take_layers()))
        return 0

    monkeypatch.setattr(cli_make_masks, "open_editor", _fake)
    return caught


def test_the_parser_takes_layers_and_a_save_folder():
    args = cli_make_masks.build_parser().parse_args(
        ["--folder", "/d", "--layers", "a", "b", "c", "--save-to", "/out"])
    assert args.layers == ["a", "b", "c"] and args.save_to == "/out"
    plain = cli_make_masks.build_parser().parse_args(["--folder", "/d"])
    assert plain.layers is None and plain.save_to is None


def test_bare_layer_names_are_found_in_the_folder_and_handed_over(
        three_layers, handed, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    code = cli_make_masks.main(["--folder", str(three_layers), "--order", "name",
                                "--layers", "masks", "model_a", "model_b"])
    assert code == 0
    queue, layers = handed[0]
    assert [Path(p).name for p in layers["layers"]] == ["masks", "model_a", "model_b"]
    assert Path(layers["save_to"]).parent == three_layers
    assert Path(layers["save_to"]).name.startswith("masks_curated_")
    assert [item.stem for item in queue.items] == ["f_0", "f_1", "f_2"]


def test_a_layer_that_is_not_a_folder_is_refused(three_layers, capsys):
    code = cli_make_masks.main(["--folder", str(three_layers),
                                "--layers", "masks", "nope"])
    assert code == cli_make_masks.EXIT_USAGE
    assert "nope is not a folder" in capsys.readouterr().err


def test_a_review_resumes_from_the_save_folder_not_the_folder(
        three_layers, handed, tmp_path):
    (three_layers / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\nf_0,done,1,2026-10-01T00:00:00\n"
        "f_1,done,1,2026-10-01T00:00:00\n", encoding="utf-8")
    target = tmp_path / "curated"
    target.mkdir()
    (target / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\nf_2,done,1,2026-10-25T00:00:00\n",
        encoding="utf-8")
    code = cli_make_masks.main(["--folder", str(three_layers), "--order", "name",
                                "--layers", "masks", "model_a",
                                "--save-to", str(target)])
    assert code == 0
    queue, layers = handed[0]
    assert [item.stem for item in queue.items] == ["f_0", "f_1"]
    assert layers["save_to"] == str(target.resolve())


def test_a_dry_run_names_the_layers_and_the_save_folder(three_layers, capsys):
    code = cli_make_masks.main(["--folder", str(three_layers), "--dry-run",
                                "--layers", "masks", "model_a", "model_b"])
    out = capsys.readouterr().out
    assert code == 0
    assert "mask layers (3, the first is the base)" in out
    assert "curated masks are saved to" in out and "masks_curated_" in out


def test_the_stored_save_folder_is_reused(three_layers, handed):
    kept = three_layers / "my_review"
    (three_layers / "curate_layers.json").write_text(
        json.dumps({"target": str(kept)}), encoding="utf-8")
    cli_make_masks.main(["--folder", str(three_layers), "--layers", "masks"])
    assert handed[0][1]["save_to"] == str(kept)
