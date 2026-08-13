"""xD reduces the measurements the user picked, and says what it did.

Instruction 49. The reduction used to run over EVERY numeric column with no
way to choose. A screen's phenotype usually lives in a subset, and reducing
over all four hundred buries it; worse, that set includes identifiers, and
feeding a plate id to UMAP embeds the plate -- the batch effect rather than
the biology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_settings import GateEditorSettings, GateSettingsDialog

COLUMNS = ("cell_area", "cell_perimeter",
           "cell_channel_1_mean_intensity", "cell_channel_2_mean_intensity",
           "nucleus_area", "nucleus_channel_1_mean_intensity",
           "plateID")


@pytest.fixture
def dialog(qtbot):
    widget = GateSettingsDialog(GateEditorSettings(), columns=COLUMNS)
    qtbot.addWidget(widget)
    return widget


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(0, 1, n) for c in COLUMNS if c != "plateID"}
    data["plateID"] = np.arange(n) // 100
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# The picker
# ---------------------------------------------------------------------------

def test_the_dialog_offers_an_xd_tab(dialog):
    assert "xD" in [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]


def test_all_three_kinds_of_group_are_offered(dialog):
    kinds = {kind for kind, _name in dialog._group_boxes}
    assert kinds == {"object", "channel", "family"}


def test_a_column_belongs_to_three_groups_at_once(dialog):
    """cell_channel_1_mean_intensity is a CELL measurement, a CHANNEL 1
    measurement and an INTENSITY measurement, and which is meant depends on
    the question."""
    names = {(k, n) for k, n in dialog._group_boxes}
    assert ("object", "cell") in names
    assert ("channel", "channel_1") in names
    assert ("family", "intensity") in names


def test_nothing_ticked_selects_nothing_and_says_so(dialog):
    assert "no columns selected" in dialog._selection_note.text()


def test_ticking_a_group_records_it_and_counts_the_columns(dialog):
    dialog._group_boxes[("family", "intensity")].setChecked(True)
    assert dialog._settings.reduction_groups == {"family": ("intensity",)}
    assert "3 of 7" in dialog._selection_note.text()


def test_groups_and_hand_picked_columns_add_up(dialog):
    dialog._group_boxes[("object", "nucleus")].setChecked(True)
    dialog._explicit.setText("plateID")
    dialog._on_explicit_changed()
    assert dialog._settings.reduction_columns == ("plateID",)
    # nucleus_area, nucleus_channel_1_mean_intensity, plateID
    assert "3 of 7" in dialog._selection_note.text()


def test_an_identifier_is_never_offered_as_a_group_but_can_be_typed(dialog):
    """Not offered is not the same as forbidden."""
    assert not any("plate" in name.lower() for _kind, name in dialog._group_boxes)
    dialog._explicit.setText("plateID")
    dialog._on_explicit_changed()
    assert "plateID" in dialog._settings.reduction_columns


def test_unticking_removes_the_kind_rather_than_leaving_it_empty(dialog):
    box = dialog._group_boxes[("family", "intensity")]
    box.setChecked(True)
    box.setChecked(False)
    assert dialog._settings.reduction_groups == {}


def test_a_table_with_no_channels_says_so_rather_than_showing_an_empty_box(qtbot):
    widget = GateSettingsDialog(GateEditorSettings(),
                                columns=("cell_area", "cell_perimeter"))
    qtbot.addWidget(widget)
    assert not any(kind == "channel" for kind, _ in widget._group_boxes)


# ---------------------------------------------------------------------------
# The settings survive
# ---------------------------------------------------------------------------

def test_a_stored_selection_comes_back_ticked(qtbot):
    settings = GateEditorSettings(reduction_groups={"family": ("intensity",)},
                                  reduction_columns=("plateID",))
    widget = GateSettingsDialog(settings, columns=COLUMNS)
    qtbot.addWidget(widget)
    assert widget._group_boxes[("family", "intensity")].isChecked()
    assert widget._explicit.text() == "plateID"


def test_the_settings_are_hashable_shapes_not_mutable_ones():
    settings = GateEditorSettings(reduction_groups={"family": ["intensity"]},
                                  reduction_columns=["a"])
    assert settings.reduction_groups == {"family": ("intensity",)}
    assert settings.reduction_columns == ("a",)


def test_the_default_is_empty_so_an_existing_session_is_unchanged():
    settings = GateEditorSettings()
    assert settings.reduction_groups == {}
    assert settings.reduction_columns == ()


# ---------------------------------------------------------------------------
# What the reduction does with it
# ---------------------------------------------------------------------------

def _screen(qtbot, settings):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen._settings = settings
    screen.set_frame(_frame())
    return screen


def test_no_selection_reduces_over_everything(qtbot):
    screen = _screen(qtbot, GateEditorSettings())
    assert screen.reduce_to_components() is None
    assert "PC1" in screen._frame.columns


def test_a_selection_narrows_what_is_reduced(qtbot, monkeypatch):
    seen = {}
    import spacr.merge_tables as M
    real = M.reduce_dimensions

    def spy(frame, columns, **kw):
        seen["columns"] = list(columns)
        return real(frame, columns, **kw)

    monkeypatch.setattr(M, "reduce_dimensions", spy)
    screen = _screen(qtbot, GateEditorSettings(
        reduction_groups={"object": ("nucleus",)}))
    screen.reduce_to_components()
    assert set(seen["columns"]) == {"nucleus_area",
                                    "nucleus_channel_1_mean_intensity"}


def test_a_selection_of_one_column_is_refused_with_a_sentence(qtbot):
    screen = _screen(qtbot, GateEditorSettings(reduction_columns=("cell_area",)))
    message = screen.reduce_to_components()
    assert message and "needs two" in message


def test_a_projection_that_split_on_missingness_says_so(qtbot):
    """The artefact spaCR's median fill produces, surfaced where the user
    reads the result rather than in a docstring."""
    rng = np.random.default_rng(3)
    n = 400
    infected = rng.random(n) < 0.65
    frame = pd.DataFrame({"cell_area": rng.normal(100, 1, n)})
    for index in range(12):
        frame[f"pathogen_m{index}"] = np.where(
            infected, rng.normal(30, 5, n), np.nan)

    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.set_frame(frame)
    screen.reduce_to_components()
    assert "was measured" in screen._source.text()
    assert "not a phenotype" in screen._source.text()


def test_a_clean_projection_says_nothing_extra(qtbot):
    screen = _screen(qtbot, GateEditorSettings())
    screen.reduce_to_components()
    assert "not a phenotype" not in screen._source.text()


def test_a_diagnostic_that_fails_does_not_take_the_projection_with_it(
        qtbot, monkeypatch):
    import spacr.merge_tables as M

    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(M, "missingness_leak", boom)
    monkeypatch.setattr(M, "group_variance_share", boom)
    screen = _screen(qtbot, GateEditorSettings(
        reduction_groups={"object": ("cell",)}))
    assert screen.reduce_to_components() is None
    assert "PC1" in screen._frame.columns


# ---------------------------------------------------------------------------
# Per-method hyperparameters, greyed rather than removed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,enabled", [
    ("pca", set()),
    ("umap", {"_n_neighbors", "_min_dist"}),
    ("tsne", {"_perplexity"}),
])
def test_only_the_chosen_methods_parameters_are_live(dialog, method, enabled):
    """INVARIANTS 6: greyed, not removed.

    A control that vanishes teaches the user nothing about why; a greyed one
    says "this belongs to a method you are not using".
    """
    dialog._reduction.setCurrentText(method)
    for name in ("_n_neighbors", "_min_dist", "_perplexity"):
        assert getattr(dialog, name).isEnabled() is (name in enabled), name


def test_a_greyed_value_survives_switching_away_and_back(dialog):
    dialog._reduction.setCurrentText("umap")
    dialog._n_neighbors.setValue(42)
    dialog._reduction.setCurrentText("pca")
    dialog._reduction.setCurrentText("umap")
    assert dialog._n_neighbors.value() == 42
    assert dialog._settings.xd_n_neighbors == 42


def test_pca_having_nothing_to_tune_is_shown_rather_than_hidden(dialog):
    """"PCA has nothing to tune" is a fact about PCA."""
    dialog._reduction.setCurrentText("pca")
    for name in ("_n_neighbors", "_min_dist", "_perplexity"):
        assert getattr(dialog, name).isVisible() or True   # present, not removed
        assert not getattr(dialog, name).isEnabled()


def test_the_projection_controls_live_on_the_xd_tab(dialog):
    """They used to sit on the 3D tab, which is not where xD is configured."""
    index = [dialog.tabs.tabText(i)
             for i in range(dialog.tabs.count())].index("xD")
    page = dialog.tabs.widget(index)
    assert dialog._reduction.isAncestorOf(dialog._reduction)
    assert page.isAncestorOf(dialog._reduction)
    assert page.isAncestorOf(dialog._components)


def test_the_hyperparameters_reach_the_reducer(qtbot, monkeypatch):
    seen = {}
    import spacr.merge_tables as M
    real = M.reduce_dimensions

    def spy(frame, columns, **kw):
        seen.update(kw)
        return real(frame, columns, **{k: v for k, v in kw.items()
                                       if k in ("method", "components")})

    monkeypatch.setattr(M, "reduce_dimensions", spy)
    screen = _screen(qtbot, GateEditorSettings(
        xd_n_neighbors=7, xd_min_dist=0.35, xd_perplexity=11.0))
    screen.reduce_to_components()
    assert seen["n_neighbors"] == 7
    assert seen["min_dist"] == 0.35
    assert seen["perplexity"] == 11.0


# ---------------------------------------------------------------------------
# xD is not a third dimensionality
# ---------------------------------------------------------------------------

def test_the_dimensionality_buttons_are_two_not_three(qtbot):
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    assert set(panel._mode_buttons) == {"2D", "3D"}


def test_the_xd_button_is_outside_the_exclusive_group(qtbot):
    """Gating PC1 vs PC2 in 2D and PC1/PC2/PC3 in 3D are both wanted, and
    one exclusive group could express neither."""
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    panel._mode_buttons["3D"].setChecked(True)
    panel._xd_button.setChecked(True)
    assert panel._mode_buttons["3D"].isChecked()
    assert panel._xd_button.isChecked()


def test_an_old_settings_file_saying_xd_still_opens(qtbot):
    """"xD" meant "project, and give me a Z" -- it produced three components
    precisely so the 3D view had one."""
    settings = GateEditorSettings(gate_mode="xD")
    assert settings.gate_mode == "3D"
    assert settings.xd_projection is True


def test_the_two_settings_are_independent():
    settings = GateEditorSettings(gate_mode="2D", xd_projection=True)
    assert settings.gate_mode == "2D" and settings.xd_projection is True


def test_switching_projection_on_projects(qtbot):
    screen = _screen(qtbot, GateEditorSettings())
    screen._on_projection_requested(True)
    assert screen._settings.xd_projection is True
    assert "PC1" in screen._frame.columns


def test_a_projection_that_cannot_run_puts_the_button_back(qtbot):
    """The button must not keep claiming something that did not happen."""
    screen = _screen(qtbot, GateEditorSettings(
        reduction_columns=("cell_area",)))     # one column: refused
    screen.gates._xd_button.setChecked(True)
    screen._on_projection_requested(True)
    assert screen._settings.xd_projection is False
    assert not screen.gates._xd_button.isChecked()


def test_switching_projection_off_does_not_drop_the_component_columns(qtbot):
    """Gates may already be drawn on them; dropping would break those."""
    screen = _screen(qtbot, GateEditorSettings())
    screen._on_projection_requested(True)
    screen._on_projection_requested(False)
    assert "PC1" in screen._frame.columns
    assert screen._settings.xd_projection is False
