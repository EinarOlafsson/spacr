"""Item 957: Make Masks point mode, the session and the points file.

``spacr-make-masks --folder <dir> --points`` opens a folder of images as a
resumable point queue. Each image's points are saved at once to
``<dir>/../annotations/<stem>_points.csv`` with the columns ``X``, ``Y``
(image pixels, column and row), ``unsure`` and ``note``. A scorer that does
``np.round(P[["Y", "X"]])`` and indexes a label image must land on exactly
the clicked pixel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr import cli_make_masks
from spacr.curation_queue import CurationQueueError, mark_state, read_status
from spacr.point_annotation import (
    LAYOUT_IMAGES,
    POINT_COLUMNS,
    PointSession,
    all_done,
    annotations_folder,
    build_point_queue,
    default_overlays,
    overlays_for,
    points_path,
    read_points,
    write_points,
)


def _crops(tmp_path, n=3, size=64):
    root = tmp_path / "handcheck_crops"
    clean = root / "annotate_clean"
    clean.mkdir(parents=True)
    for i in range(n):
        pixels = np.full((size, size), 20 * i, np.uint8)
        Image.fromarray(pixels).save(clean / f"crop{i:02d}_site3_y1_x2_clean.png")
    return clean


def test_points_file_is_named_after_the_stem_without_clean(tmp_path):
    assert points_path("a/crop01_site3_y1234_x567_clean.png", tmp_path).name == \
        "crop01_site3_y1234_x567_points.csv"
    assert points_path("field_7.tif", tmp_path).name == "field_7_points.csv"
    assert points_path("_clean.png", tmp_path).name == "_clean_points.csv"
    assert annotations_folder(tmp_path / "x" / "annotate_clean") == \
        (tmp_path / "x" / "annotations")


def test_a_bare_image_folder_opens_as_a_point_queue(tmp_path):
    clean = _crops(tmp_path)
    queue = build_point_queue(clean)
    assert queue.layout.kind == LAYOUT_IMAGES
    assert [item.stem for item in queue.items] == [
        f"crop{i:02d}_site3_y1_x2_clean" for i in range(3)]
    assert queue.summary.remaining == 3
    with pytest.raises(CurationQueueError):
        build_point_queue(tmp_path / "empty_missing")


def test_two_images_sharing_a_points_file_are_refused(tmp_path):
    clean = _crops(tmp_path, n=1)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(clean / "crop00_site3_y1_x2.png")
    with pytest.raises(CurationQueueError, match="both save their points"):
        build_point_queue(clean)


def test_a_seg_folder_is_refused(tmp_path):
    folder = tmp_path / "seg"
    folder.mkdir()
    np.save(folder / "a_seg.npy", {"masks": np.zeros((4, 4))}, allow_pickle=True)
    with pytest.raises(CurationQueueError, match="_seg.npy"):
        build_point_queue(folder)


def test_clicks_round_trip_through_the_scorer_convention(tmp_path):
    clean = _crops(tmp_path, n=1, size=512)
    image = next(clean.iterdir())
    session = PointSession(image, annotations_folder(clean), shape=(512, 512))
    clicks = [(10, 20), (511, 0), (0, 511), (255, 300)]
    for x, y in clicks:
        assert session.add(x, y) is not None
    assert session.add(512, 3) is None and session.add(-1, 3) is None
    frame = pd.read_csv(session.path)
    assert list(frame.columns) == list(POINT_COLUMNS)
    rows_cols = np.round(frame[["Y", "X"]].to_numpy()).astype(int).clip(0, 511)
    assert rows_cols.tolist() == [[y, x] for x, y in clicks]
    assert frame["unsure"].tolist() == [0, 0, 0, 0]
    assert frame["X"].dtype == float and frame["Y"].dtype == float


def test_remove_unsure_and_undo(tmp_path):
    clean = _crops(tmp_path, n=1)
    image = next(clean.iterdir())
    session = PointSession(image, annotations_folder(clean), shape=(64, 64))
    session.add(10, 10)
    session.add(30, 30)
    assert session.remove_nearest(40, 40, radius=3) is None
    assert session.remove_nearest(31, 29, radius=3).x == 30
    assert session.count == 1
    assert session.toggle_unsure().unsure
    assert session.toggle_unsure(50, 50, radius=2).unsure is False
    session.add(20, 20)
    assert session.toggle_unsure(21, 20, radius=3) == session.points[1]
    assert session.points[1].unsure and session.unsure_count == 1
    assert session.undo() and not session.points[1].unsure
    assert session.undo() and session.count == 1
    assert session.undo() and session.undo() and session.count == 1
    assert session.undo() and session.count == 2
    assert session.undo() and session.undo() and session.count == 0
    assert not session.undo()
    assert read_points(session.path) == []


def test_saving_is_atomic_and_overwrites(tmp_path):
    path = tmp_path / "annotations" / "a_points.csv"
    from spacr.point_annotation import Point

    write_points(path, [Point(1, 2), Point(3, 4, True, "two?")])
    write_points(path, [Point(5, 6)])
    assert read_points(path) == [Point(5.0, 6.0)]
    assert not list(path.parent.glob("*.tmp"))


def test_a_fiji_measure_file_reads(tmp_path):
    path = tmp_path / "f_points.csv"
    path.write_text(" ,Area,Mean,X,Y\n1,0,12,10.5,20\n2,0,13,3,4\n", encoding="utf-8")
    points = read_points(path)
    assert [(p.x, p.y, p.unsure) for p in points] == [(10.5, 20.0, False), (3.0, 4.0, False)]
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(CurationQueueError, match="no X and Y"):
        read_points(path)


def test_resume_keeps_points_and_done_flags(tmp_path):
    clean = _crops(tmp_path)
    queue = build_point_queue(clean)
    first = queue.items[0]
    session = PointSession(first.image, annotations_folder(clean))
    session.add(5, 6)
    session.toggle_unsure()
    mark_state(clean, first.stem, "done", n_objects=session.count)
    again = build_point_queue(clean)
    assert again.summary.done == 1 and again.summary.remaining == 2
    reopened = PointSession(first.image, annotations_folder(clean))
    assert [(p.x, p.y, p.unsure) for p in reopened.points] == [(5.0, 6.0, True)]
    assert not all_done(again)
    for item in again.items[1:]:
        mark_state(clean, item.stem, "done", n_objects=0)
    assert all_done(build_point_queue(clean))
    assert read_status(clean)[first.stem].n_objects == 1


def test_overlays_are_matched_by_points_stem(tmp_path):
    clean = _crops(tmp_path, n=2)
    overlays = clean.parent / "overlays"
    overlays.mkdir()
    for name in ("crop00_site3_y1_x2_spaCR_outlines.png",
                 "crop00_site3_y1_x2_PERISCOPE_outlines.png",
                 "crop01_site3_y1_x2_spaCR_outlines.png",
                 "crop00_site3_y1_x20_spaCR_outlines.png"):
        Image.fromarray(np.zeros((4, 4), np.uint8)).save(overlays / name)
    assert default_overlays(clean) == [overlays]
    found = overlays_for(clean / "crop00_site3_y1_x2_clean.png", [overlays])
    assert [p.name for p in found] == ["crop00_site3_y1_x2_PERISCOPE_outlines.png",
                                       "crop00_site3_y1_x2_spaCR_outlines.png"]


def test_cli_dry_run_lists_the_point_queue(tmp_path, capsys):
    clean = _crops(tmp_path)
    mark_state(clean, "crop01_site3_y1_x2_clean", "done", n_objects=0)
    assert cli_make_masks.main(["--folder", str(clean), "--points", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "point mode: images layout" in out
    assert str(annotations_folder(clean)) in out
    lines = [line for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert len(lines) == 3
    assert lines[1].endswith("done") and not lines[0].endswith("done")


def test_cli_refuses_mask_only_flags_with_points(tmp_path, capsys):
    clean = _crops(tmp_path)
    assert cli_make_masks.main(["--folder", str(clean), "--points", "--limit", "2"]) == 2
    assert "--limit" in capsys.readouterr().err
    assert cli_make_masks.main(["--folder", str(clean), "--overlays", str(clean)]) == 2
    assert cli_make_masks.main(["--folder", str(clean), "--points", "--overlays",
                                str(tmp_path / "nope"), "--dry-run"]) == 2


def test_cli_hands_the_point_session_to_the_screen(tmp_path, monkeypatch):
    clean = _crops(tmp_path)
    seen = {}

    def fake_run(argv):
        seen["argv"] = argv
        seen["points"] = cli_make_masks._take_points()
        return 0

    import spacr.qt as qt

    monkeypatch.setattr(qt, "run", fake_run)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert cli_make_masks.main(["--folder", str(clean), "--points",
                                "--no-hide-overlays"]) == 0
    assert seen["argv"] == ["make_masks"]
    assert seen["points"]["hide_overlays"] is False
    assert seen["points"]["queue"].folder == clean
    assert cli_make_masks._take_points() is None


def test_point_mode_is_an_alpha_widget():
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[957] == {"widgets": ("MakeMasksPointModeButton",)}
