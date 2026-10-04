"""The barcode search panel's buttons, statuses and wiring edges."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import map_barcodes as mb  # noqa: E402
from tests.qt.test_barcode_search_failure_recovery import panel  # noqa: E402,F401


def test_buttons_restart_cancel_and_apply(panel, monkeypatch):  # noqa: F811
    calls = []
    monkeypatch.setattr(panel, "cancel_search", lambda: calls.append("cancel"))
    monkeypatch.setattr(panel, "start_search", lambda: calls.append("start"))
    monkeypatch.setattr(panel, "apply_proposal", lambda: calls.append("apply"))
    panel._running = True
    panel.on_search_clicked()
    panel.on_cancel_clicked()
    panel.on_apply_clicked()
    assert calls == ["cancel", "start", "cancel", "apply"]


def test_a_running_search_is_not_started_twice(panel):  # noqa: F811
    panel._running = True
    assert panel.start_search() is False
    panel._running = False


def test_a_live_search_restarts_a_running_one(panel, monkeypatch):  # noqa: F811
    started = []
    monkeypatch.setattr(panel, "current_settings", lambda: {"src": "/x"})
    monkeypatch.setattr(panel, "start_search", lambda: started.append(True))
    panel._searched_inputs = ()
    panel._running = True
    panel._run_live_search()
    assert started == [True] and panel._running is False


def test_no_screen_means_no_settings(panel):  # noqa: F811
    panel._screen = None
    assert panel.current_settings() == {}


def test_a_chunk_without_a_report_finishes_the_search(panel, monkeypatch):  # noqa: F811
    finished = []
    monkeypatch.setattr(panel, "_finish", lambda: finished.append(True))
    panel._running = True
    panel._iterator = object()
    panel._on_chunk({"report": None})
    assert finished == [True]


def test_no_plan_means_no_barcode_kinds(panel):  # noqa: F811
    panel._plan = None
    assert panel._barcode_kinds() == ()


def _plan(others):
    return mb.BarcodeSearchPlan({"R1": "/r1.fastq.gz"}, (("g", "/g.csv", "grna"),),
                                "ANCHOR", "s1", others, "")


@pytest.mark.parametrize("others, phrase", [(1, "One"), (3, "3")])
def test_the_running_status_names_the_other_samples(panel, others, phrase):  # noqa: F811
    panel._plan = _plan(others)
    panel._set_running_status(SimpleNamespace(reads=1200))
    assert "s1" in panel.status.text() and phrase in panel.status.text()


class _Report:
    def __init__(self, found):
        self.reads = 500
        self._found = found

    def roles(self):
        from spacr.barcode_search import ANCHOR_ROLE
        return (ANCHOR_ROLE, "grna")

    def best_for_role(self, role):
        from spacr.barcode_search import PRESENT
        if not self._found:
            return None
        return SimpleNamespace(verdict=PRESENT, name="g", table="g")


@pytest.mark.parametrize("found, phrase", [(True, "Every barcode"),
                                           (False, "No barcode table")])
def test_the_finished_status_says_what_was_established(panel, found, phrase,  # noqa: F811
                                                       monkeypatch):
    panel._plan = _plan(0)
    panel._report = _Report(found)
    panel._set_finished_status()
    assert phrase in panel.status.text()
    panel._report = None
    before = panel.status.text()
    panel._set_finished_status()
    assert panel.status.text() == before


def test_a_failed_proposal_leaves_nothing_to_apply(panel, monkeypatch):  # noqa: F811
    def broken(*a, **k):
        raise ValueError("cannot propose")

    monkeypatch.setattr(mb, "describe_proposed_changes", broken, raising=False)
    monkeypatch.setattr(mb, "propose_settings", broken, raising=False)
    panel._report = _Report(True)
    panel._finish()
    assert panel.proposal() is None and panel.proposed_changes() == ()


def test_the_panel_installs_only_where_there_is_an_actions_row(qtbot):
    from PySide6.QtWidgets import QWidget

    screen = QWidget()
    qtbot.addWidget(screen)
    assert mb._insert_above_actions(screen, QWidget()) is False
    assert mb.install_barcode_search(screen) is None


def test_a_failing_card_build_installs_nothing(qtbot, monkeypatch):
    from PySide6.QtWidgets import QWidget

    def broken(screen, **kwargs):
        raise RuntimeError("no form")

    monkeypatch.setattr(mb, "build_barcode_search_card", broken)
    screen = QWidget()
    qtbot.addWidget(screen)
    assert mb.install_barcode_search(screen) is None


def test_search_inputs_flatten_lists_and_values_compare_as_text():
    key = mb._LIVE_SEARCH_KEYS[0]
    assert mb._search_inputs({key: ["/a", "/b"]})[0] == "/a\n/b"

    class _Odd:
        def __eq__(self, other):
            raise TypeError("incomparable")

        def __str__(self):
            return "same"

    assert mb._same_setting_value(_Odd(), "same") is True


def test_planning_without_sources_says_where_to_start():
    plan = mb.plan_barcode_search({"src": ["", "  "]})
    assert plan.problem
