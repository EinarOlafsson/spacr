"""Cellpose workbench: which knobs cross between tabs, and the trained checkpoint.

``carried_setting_keys`` reads the Apply module's propagation map; a missing
map or an unknown key carries nothing. The workbench is built for real (both
module pages, both settings models): seams that will not install, a carry
that raises and a masthead without an API link must each cost only
themselves. ``trained_checkpoint`` is walked over a table of training forms and
checkpoint folders on disk and must name the exact file Apply would load.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import train_cellpose as tc  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def workbench(qtbot):
    screen = tc.CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    return screen


def test_carried_setting_keys_come_from_the_propagation_map(monkeypatch,
                                                           subtests):
    """The map's values minus model_name and src; nothing when it is gone."""
    from spacr.qt import preview_registry

    with subtests.test(msg="the map names the carried knobs"):
        assert tc.carried_setting_keys() == (
            "diameter", "flow_threshold", "CP_prob", "custom_model",
            "normalize")

    with subtests.test(msg="model_name and src never cross"):
        spec = preview_registry.PREVIEWS[tc.APPLY_KEY]
        monkeypatch.setitem(
            preview_registry.PREVIEWS, tc.APPLY_KEY,
            dataclasses.replace(spec, propagation={
                "a": "model_name", "b": "src", "c": "diameter"}))
        assert tc.carried_setting_keys() == ("diameter",)
        monkeypatch.undo()

    with subtests.test(msg="an apply key the map does not know"):
        monkeypatch.setattr(preview_registry, "PREVIEWS", {})
        assert tc.carried_setting_keys() == ()
        monkeypatch.undo()

    with subtests.test(msg="an unreadable propagation map"):
        monkeypatch.setitem(sys.modules, "spacr.qt.preview_registry", None)
        assert tc.carried_setting_keys() == ()


def test_the_workbench_survives_each_seam_that_fails(qtbot, workbench,
                                                     monkeypatch, subtests):
    """Missing seams, a raising carry and no API link cost only themselves."""
    with subtests.test(msg="search, recipes and preview cannot install"):
        with monkeypatch.context() as patch:
            for name in ("spacr.qt.settings_search", "spacr.qt.recipes",
                         "spacr.qt.preview_registry"):
                patch.setitem(sys.modules, name, None)
            bare = tc.CellposeWorkbenchScreen()
            qtbot.addWidget(bare)
        assert [page.app_key for page in bare._screens] == \
            [tc.TRAIN_KEY, tc.APPLY_KEY]
        assert bare._tabs.count() == 2
        assert bare.active_app_key() == tc.TRAIN_KEY

    with subtests.test(msg="a carry that raises still switches the tab"):
        tried = []

        def explode(source, target):
            tried.append((source, target))
            raise RuntimeError("settings model is gone")

        monkeypatch.setattr(workbench, "carry", explode)
        workbench._tabs.setCurrentIndex(1)
        assert tried == [(workbench.train_screen, workbench.apply_screen)]
        assert workbench._tabs.currentIndex() == 1
        assert workbench.active_app_key() == tc.APPLY_KEY
        assert workbench._header.instruction_label.text() == tc.TABS[1][2]

    with subtests.test(msg="a masthead with no API link keeps its sentence"):
        header = workbench._header
        header.api_help = None
        workbench._tabs.setCurrentIndex(0)
        assert header.instruction_label.text() == tc.TABS[0][2]
        assert header.instruction_label.isHidden() is False
        assert workbench.active_app_key() == tc.TRAIN_KEY


def test_trained_checkpoint_names_the_file_apply_would_load(
        workbench, monkeypatch, tmp_path, subtests):
    """No form, no source/name or no folder is ""; finished beats snapshots."""
    folder = tmp_path.joinpath(*tc.CHECKPOINT_DIR)
    form = workbench.train_screen._settings_model

    def collect(values):
        monkeypatch.setattr(form, "collect", lambda: dict(values))

    def unreadable():
        raise RuntimeError("cannot collect")

    with subtests.test(msg="a form that cannot be read"):
        monkeypatch.setattr(form, "collect", unreadable)
        assert workbench.trained_checkpoint() == ""

    blanks = [("no source", {"src": "", "model_name": "m"}),
              ("no model name", {"src": str(tmp_path), "model_name": " "}),
              ("no checkpoint folder", {"src": str(tmp_path),
                                        "model_name": "my_model"})]
    for case_id, values in blanks:
        with subtests.test(msg=case_id):
            collect(values)
            assert workbench.trained_checkpoint() == ""

    folder.mkdir(parents=True)
    collect({"src": str(tmp_path), "model_name": "my_model"})
    (folder / "other_model_final").write_bytes(b"weights")
    (folder / "my_model_dir").mkdir()
    with subtests.test(msg="a folder with only other models' files"):
        assert workbench.trained_checkpoint() == ""

    snapshot = folder / f"my_model_{tc.EPOCH_INFIX}5"
    snapshot.write_bytes(b"weights")
    with subtests.test(msg="only an epoch snapshot is still offered"):
        assert workbench.trained_checkpoint() == str(snapshot)

    finished = folder / "my_model_final"
    finished.write_bytes(b"weights")
    os.utime(snapshot, (2_000_000_000, 2_000_000_000))
    os.utime(finished, (1_000_000_000, 1_000_000_000))
    with subtests.test(msg="the finished model wins over newer snapshots"):
        assert workbench.trained_checkpoint() == str(finished)

    later = folder / "my_model_b"
    later.write_bytes(b"weights")
    os.utime(later, (1_500_000_000, 1_500_000_000))
    with subtests.test(msg="among finished files the newest wins"):
        assert workbench.trained_checkpoint() == str(later)

    with subtests.test(msg="unreadable times fall back to the last name"):
        def no_time(_path):
            raise OSError("stat is unavailable")

        monkeypatch.setattr(os.path, "getmtime", no_time)
        assert workbench.trained_checkpoint() == str(finished)
