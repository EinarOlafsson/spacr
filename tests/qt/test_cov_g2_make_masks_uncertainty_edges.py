"""Make Masks uncertainty and blind-mode edges."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from tests.qt.test_568_make_masks_uncertainty import (_blind_corner,  # noqa: E402,F401
                                                      screen)


def test_nothing_open_means_nothing_to_map_or_rank(screen):  # noqa: F811
    screen._canvas.image = None
    assert screen._on_map_uncertainty(segment=_blind_corner, threaded=False) is False
    assert "Open a folder" in screen._status_label.text()
    files = screen._image_files
    screen._image_files = []
    assert screen._on_rank_uncertainty(segment=_blind_corner, threaded=False) is False
    screen._image_files = files


def test_a_running_request_refuses_another(screen):  # noqa: F811
    screen._uncertainty_request = {"kind": "field"}
    assert screen._on_rank_uncertainty(segment=_blind_corner, threaded=False) is False
    screen._load_current()
    assert screen._on_map_uncertainty(segment=_blind_corner, threaded=False) is False
    screen._uncertainty_request = None


def test_stale_failed_and_superseded_results_are_dropped(screen, monkeypatch):  # noqa: F811
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append(text))
    screen._uncertainty_request = {"kind": "field"}
    screen._take_uncertainty(({"kind": "field"}, None, None))
    request = {"kind": "field", "token": -1, "image_reference": None}
    screen._uncertainty_request = request
    screen._take_uncertainty((request, None, RuntimeError("no model")))
    assert warned == ["no model"]
    screen._uncertainty_request = request
    screen._take_uncertainty((request, {"field": 0.1}, None))
    assert "discarded" in screen._status_label.text()


def test_saving_a_map_can_be_cancelled_or_fail(screen, monkeypatch, tmp_path):  # noqa: F811
    screen._load_current()
    assert screen._on_map_uncertainty(segment=_blind_corner, threaded=False)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert screen._on_save_uncertainty() is False
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append(text))
    import spacr.segmentation_uncertainty as su

    def refuse(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(su, "save_uncertainty_map", refuse)
    assert screen._on_save_uncertainty(str(tmp_path / "u.tif")) is False
    assert warned == ["read-only"]


def test_ranking_results_respect_blind_order_and_changed_fields(screen,  # noqa: F811
                                                                monkeypatch):
    import spacr.curation_queue as cq

    def refuse(target, rows):
        raise OSError("disk full")

    monkeypatch.setattr(cq, "_write_uncertainty", refuse)
    pairs = screen._field_pairs()
    scores = {tuple(pair): {"field": 0.5, "n_objects": 1, "n_passes": 2}
              for pair in pairs}
    first = (pairs[0][0], pairs[0][1] + cq.SEG_SUFFIX)
    scores[first] = scores.pop(tuple(pairs[0]))
    screen._blind = {"codes": {}}
    screen._apply_uncertainty_ranking(pairs, scores)
    assert "blind order was preserved" in screen._status_label.text()
    screen._blind = None
    screen._apply_uncertainty_ranking(list(reversed(pairs)), scores)
    assert "open fields changed" in screen._status_label.text()


def test_one_unreadable_field_does_not_discard_the_other_scores(
        screen, monkeypatch, caplog):  # noqa: F811
    import csv
    from pathlib import Path

    from spacr.qt.screens import make_masks as mm

    load = mm.engine.load_image_and_mask

    def read_field(folder, name, **layout):
        if name == "doubtful.tif":
            raise OSError("field unreadable")
        return load(folder, name, **layout)

    monkeypatch.setattr(mm.engine, "load_image_and_mask", read_field)
    assert screen._on_rank_uncertainty(segment=_blind_corner, threaded=False)
    assert screen._image_files == ["calm.tif", "doubtful.tif"]
    with (Path(screen._folder) / "curate_uncertainty.csv").open(
            newline="", encoding="utf-8") as handle:
        assert [row["stem"] for row in csv.DictReader(handle)] == ["calm"]
    assert "uncertainty of doubtful.tif could not be scored" in caplog.text


def test_all_unreadable_fields_leave_the_prior_ranking_file_untouched(
        screen, monkeypatch, caplog):  # noqa: F811
    """A failed batch cannot replace valid scores with an empty export."""
    from pathlib import Path

    from spacr.qt.screens import make_masks as mm

    ranking = Path(screen._folder) / "curate_uncertainty.csv"
    ranking.write_text("stem,uncertainty\nprior,0.8\n", encoding="utf-8")
    before = ranking.read_bytes()
    pairs = screen._field_pairs()

    def unreadable(*args, **kwargs):
        raise OSError("field unreadable")

    monkeypatch.setattr(mm.engine, "load_image_and_mask", unreadable)
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(mm, "is_headless", lambda: False)
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda parent, title, text: warnings.append((title, text))
        or QMessageBox.Ok)
    assert screen._on_rank_uncertainty(segment=_blind_corner, threaded=False)
    assert ranking.read_bytes() == before
    assert screen._field_pairs() == pairs
    assert "uncertainty of calm.tif could not be scored" in caplog.text
    assert "uncertainty of doubtful.tif could not be scored" in caplog.text
    assert warnings == [("Load failed", "field unreadable")]


def test_blind_helpers_without_their_buttons(screen, monkeypatch):  # noqa: F811
    screen._btn_blind = None
    screen._set_blind_checked(True)
    screen._btn_rois = None
    screen._blind_lock_rois(True)
    import os

    monkeypatch.setattr(os.path, "commonpath",
                        lambda paths: (_ for _ in ()).throw(ValueError("drives")))
    assert screen._start_blind()
    pairs = screen._field_pairs()
    screen._blind["original"] = screen._blind["original"][1:]
    screen._restore_blind_order()
    assert sorted(screen._field_pairs()) == sorted(pairs)
