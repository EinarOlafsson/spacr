"""Item 288: the Compare panel's gRNA and well choosers, answer by answer.

`MeasurementComparePanel` opens on the regression's top hits and lets the
user re-pick the gRNAs, then the wells those gRNAs are in. Each chooser can
end four ways -- nothing to offer, cancelled, the same selection given back,
a real change -- and only the last may redraw anything. These pin all four
for both choosers, plus the two places the panel must not lose a choice
already made: a new montage arriving (the Graph tab re-feeds the panel
rather than rebuilding it) and a default graph type that cannot be read.

The opening selection itself is pinned in
``test_show_offers_the_wells_the_guides_are_in.py``.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog                         # noqa: E402

import spacr.qt.widgets.measurement_compare_dialog as mcd     # noqa: E402

RESULTS = pd.DataFrame({
    "feature": ["Intercept", "grna[T.g1]", "grna[T.q]"],
    "coefficient": [0.1, 0.9, -0.5],
    "p_value": [1e-9, 1e-6, 1e-3],
})


@pytest.fixture
def objects():
    """g1 in two wells, q in a third, and well-mates beside them."""
    return pd.DataFrame({
        "prc": ["p1_r1_c1"] * 3 + ["p1_r1_c2"] * 3 + ["p1_r1_c3"] * 2,
        "grna": ["g1", "x", "y", "g1", "z", "w", "q", "r"],
        "area": [float(v) for v in range(8)],
    })


@pytest.fixture
def panel(qapp, qtbot, objects):
    widget = mcd.MeasurementComparePanel(objects.copy(), {"a": [0]},
                                         results=RESULTS)
    qtbot.addWidget(widget)
    return widget


def _answer(monkeypatch, verdict, chosen=()):
    monkeypatch.setattr(mcd._WellChoice, "exec", lambda self: verdict)
    monkeypatch.setattr(mcd._WellChoice, "chosen",
                        lambda self: set(chosen))


def test_objects_that_name_no_grna_offer_nothing_to_choose(qapp, qtbot):
    frame = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                          "area": [1.0, 2.0]})
    widget = mcd.MeasurementComparePanel(frame, {"a": [0]})
    qtbot.addWidget(widget)
    assert widget._choose_guides() is False
    assert "name no gRNA" in widget.selection_note.text()


def test_cancelling_the_grna_chooser_keeps_the_top_hits(panel, monkeypatch):
    before = list(panel._selected_guides)
    _answer(monkeypatch, QDialog.Rejected, {"q"})
    assert panel._choose_guides() is False
    assert panel._selected_guides == before == ["g1", "q"]


def test_ticking_the_same_grnas_again_is_not_a_change(panel, monkeypatch):
    _answer(monkeypatch, QDialog.Accepted, {"g1", "q"})
    assert panel._choose_guides() is False
    assert panel._selected_guides == ["g1", "q"]


def test_no_well_carrying_the_grnas_offers_no_wells(panel):
    panel._selected_guides = ["not_in_any_well"]
    assert panel._choose_selected_wells() is False
    assert "No well carries the chosen gRNAs" in \
        panel.selection_note.text()


def test_cancelling_the_well_chooser_keeps_the_wells(panel, monkeypatch):
    before = panel.selected_wells()
    _answer(monkeypatch, QDialog.Rejected, {"p1_r1_c1"})
    assert panel._choose_selected_wells() is False
    assert panel.selected_wells() == before


def test_ticking_the_same_wells_again_is_not_a_change(panel, monkeypatch):
    before = panel.selected_wells()
    _answer(monkeypatch, QDialog.Accepted, set(before))
    assert panel._choose_selected_wells() is False
    assert panel.selected_wells() == before


def test_narrowing_the_wells_draws_only_those_wells(panel, monkeypatch):
    assert panel.selected_wells() == ["p1_r1_c1", "p1_r1_c2", "p1_r1_c3"]
    _answer(monkeypatch, QDialog.Accepted, {"p1_r1_c3"})
    assert panel._choose_selected_wells() is True
    assert panel.selected_wells() == ["p1_r1_c3"]
    scoped, _report = panel.scoped_objects()
    assert set(scoped["prc"]) == {"p1_r1_c3"}
    assert "selected wells (1): p1_r1_c3" in panel.selection_note.text()


def test_a_new_montage_keeps_the_grnas_the_user_chose(panel, objects,
                                                      monkeypatch):
    _answer(monkeypatch, QDialog.Accepted, {"q"})
    assert panel._choose_guides() is True
    panel.set_data(objects.copy(), {"a": [0]})
    assert panel._selected_guides == ["q"]


def test_the_note_names_the_selection_without_a_scope_report(panel):
    panel._scope_report = {}
    panel._show_the_selection()
    lines = panel.selection_note.text().splitlines()
    assert lines[0].startswith("gRNAs (2): g1, q")
    assert len(lines) == 2
    assert panel.scope_wells_button.isEnabled()


def test_an_unreadable_default_graph_type_opens_on_the_first_plot(
        panel, monkeypatch):
    import spacr.graph_types as graph_types

    def unreadable(_shape):
        raise OSError("settings locked")

    monkeypatch.setattr(graph_types, "chosen_for", unreadable)
    assert panel._plot_to_start_on() == 0
