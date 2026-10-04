"""Make Masks folder jobs, guards and small helpers at their edges."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt.screens import make_masks as mm  # noqa: E402
from tests.qt.test_568_make_masks_uncertainty import screen  # noqa: E402,F401


def _busy(screen):  # noqa: F811
    screen._folder_job = SimpleNamespace(isRunning=lambda: True)


def test_folder_jobs_refuse_while_another_runs(screen, monkeypatch):  # noqa: F811
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append(title))
    _busy(screen)
    assert screen._start_consolidation("/x") is False
    assert screen._start_channel_sort(object()) is False
    screen._folder_job = None
    assert warned == ["Busy", "Busy"]


def test_failed_folder_jobs_are_reported(screen, monkeypatch):  # noqa: F811
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append(title))
    failed = SimpleNamespace(error=RuntimeError("disk"), result=None)
    screen._on_consolidated(failed)
    screen._on_channels_sorted(failed)
    assert warned == ["Consolidation failed", "Sorting failed"]


def test_a_consolidation_with_failed_copies_says_so(screen, monkeypatch, tmp_path):  # noqa: F811
    posted = []
    monkeypatch.setattr(screen._masks_console, "post",
                        lambda text, *a: posted.append(text))
    monkeypatch.setattr(screen, "_open_folder", lambda folder: None)
    result = SimpleNamespace(copied=1, output=tmp_path, manifest=tmp_path / "m",
                             failed=["a.tif"])
    screen._on_consolidated(SimpleNamespace(error=None, result=result))
    assert any("could not be copied" in line for line in posted)


def test_an_unreadable_folder_is_not_offered_consolidation(screen, monkeypatch):  # noqa: F811
    import spacr.folder_consolidation as fc

    def broken(*a, **k):
        raise OSError("permission")

    monkeypatch.setattr(fc, "nested_file_count", broken)
    assert screen._offer_consolidation("/x") is False


def test_a_picked_folder_that_is_consolidated_opens_later(screen, monkeypatch):  # noqa: F811
    opened = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: "/data"))
    monkeypatch.setattr(screen, "_offer_consolidation", lambda folder: True)
    monkeypatch.setattr(screen, "_open_folder", opened.append)
    screen._on_pick_folder()
    assert opened == []


def test_the_organize_dialog_is_refused_headless(screen, monkeypatch):  # noqa: F811
    monkeypatch.setattr(mm, "is_headless", lambda: True)
    assert screen._run_organize(object()) is False


def test_small_helpers_without_their_widgets(screen, monkeypatch):  # noqa: F811
    screen._secondary_relations = None
    assert screen._refresh_secondary_report() is None
    screen._canvas.preserve_ids = False
    assert screen._secondary_detail() == {}
    screen._api_tooltip_filter = None
    assert screen._on_filter_row_added(None) is None
    screen._btn_otsu = None
    assert screen._sync_detect_button("otsu") is None
    screen._method_widgets = None
    assert screen._method_params() == mm.organelle_modes.DEFAULT_PARAMS
    screen._secondary_widgets = None
    screen._propagate_widgets = None
    assert screen._cpu_params() == mm.cpu_modes.DEFAULT_PARAMS
    del screen._enh_background
    assert screen._enhancement_chain() == mm.detect_chain.NO_CHAIN
    screen._btn_discard = None
    screen._show_curation_verdict(True)
