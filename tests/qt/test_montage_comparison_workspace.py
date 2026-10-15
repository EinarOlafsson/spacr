"""Montage comparison state survives restore without loading or drawing."""
from copy import deepcopy

import pandas as pd
import pytest
from PySide6.QtWidgets import QWidget

from spacr.crops import STREAM_IMAGES
from spacr.qt.widgets import cell_montage_view as montage

pytestmark = pytest.mark.qt


@pytest.fixture
def view(qtbot):
    widget = montage.CellMontageView(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


class ComparisonPanel(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.get("parent"))
        self.state = {"measurement": "cell_area", "selected_wells": None}
        self.events = ["construct"]

    def workspace_state(self):
        return self.state

    def apply_workspace_state(self, state):
        self.events.append("apply")
        self.state = deepcopy(state)
        return True

    def set_data(self, *args, **kwargs):
        self.events.append("set_data")

    def refresh(self):
        self.events.append("refresh")


def test_collect_copies_existing_comparison_without_building(view, monkeypatch):
    panel = ComparisonPanel(parent=view)
    panel.state["selected_guides"] = ["guide1"]
    view._graph_panel = panel
    monkeypatch.setattr(view, "_ensure_graph_tab", lambda: pytest.fail("collect built a panel"))
    saved = view.workspace_state()
    assert saved["comparison"] == panel.state
    saved["comparison"]["selected_guides"].append("changed")
    assert panel.state["selected_guides"] == ["guide1"]
    assert panel.events == ["construct"]


def test_restore_before_graph_exists_is_lazy_and_snapshot_preserved(view, monkeypatch):
    incoming = {"measurement": "cell_area", "selected_wells": [], "selected_guides": ["guide1"]}
    monkeypatch.setattr(view, "_ensure_graph_tab", lambda: pytest.fail("restore built a panel"))
    assert view.apply_workspace_state({"comparison": incoming})
    assert view._graph_panel is None
    assert view.workspace_state()["comparison"] == incoming
    incoming["selected_guides"].append("changed")
    assert view.workspace_state()["comparison"]["selected_guides"] == ["guide1"]


def test_restore_existing_graph_applies_without_refresh(view):
    panel = ComparisonPanel(parent=view)
    view._graph_panel = panel
    state = {"measurement": "cell_area", "selected_wells": None, "chosen_wells": []}
    assert view.apply_workspace_state({"comparison": state})
    assert panel.state == state
    assert panel.events == ["construct", "apply"]


def test_restore_blocks_loaded_montage_control_autoload_and_redraw(view, monkeypatch):
    view._plans = (object(),)
    monkeypatch.setattr(view, "build", lambda: pytest.fail("restore started image loading"))
    monkeypatch.setattr(view, "_redraw_from_cache", lambda: pytest.fail("restore rendered crops"))
    monkeypatch.setattr(view, "_can_redraw_without_loading", lambda: True)
    assert view.apply_workspace_state({"picture_mode": STREAM_IMAGES, "widgets": {"channels": "r,g"}})
    assert view.picture_mode() == STREAM_IMAGES
    assert view._channels.text() == "r,g"


def test_restore_retires_pending_load_and_queued_coefficients(view):
    view._pending = object()
    token = view._load_cancel
    view._queue[:] = ["gene_fraction:gene[old]"]
    generation = view._jobs._generation
    assert view.apply_workspace_state({"widgets": {"channels": "r"}})
    assert token.is_set()
    assert view._pending is None
    assert view._queue == []
    assert view._jobs._generation > generation


def test_restore_preserves_preexisting_signal_block_state(view):
    view._channels.blockSignals(True)
    assert view.apply_workspace_state({"widgets": {"channels": "r"}})
    assert view._channels.signalsBlocked()
    assert not view._source.signalsBlocked()


def test_explicit_graph_creation_consumes_pending_state_once(view, monkeypatch):
    from spacr.qt.widgets import measurement_compare_dialog

    monkeypatch.setattr(measurement_compare_dialog, "MeasurementComparePanel", ComparisonPanel)
    monkeypatch.setattr(view, "rows_to_compare", lambda: pd.DataFrame({"cell_area": [1, 2]}))
    monkeypatch.setattr(view, "picked_groups", lambda: {"guide1": [0]})
    state = {"measurement": "cell_area", "selected_wells": [], "kind": "box"}
    assert view.apply_workspace_state({"comparison": state})
    view._ensure_graph_tab()
    panel = view._graph_panel
    assert panel.state == state
    assert panel.events == ["construct", "apply", "refresh"]
    panel.state["kind"] = "jitter"
    view._ensure_graph_tab()
    assert panel.state["kind"] == "jitter"
    assert panel.events == ["construct", "apply", "refresh", "set_data"]
    assert view.workspace_state()["comparison"]["kind"] == "jitter"


def test_unsupported_state_leaves_existing_work_untouched(view):
    pending = object()
    view._pending = pending
    assert not view.apply_workspace_state({"comparison": "bad"})
    assert view._pending is pending
    assert not view.apply_workspace_state(None)
    view._pending = None


def test_real_panel_receives_deferred_graph_controls_on_explicit_open(view, monkeypatch):
    from spacr.qt.widgets import measurement_compare_dialog as compare

    objects = pd.DataFrame({
        "prc": ["p1_r1_c1"] * 3 + ["p1_r1_c2"] * 3,
        "plateID": ["p1"] * 6,
        "rowID": ["r1"] * 6,
        "columnID": ["c1"] * 3 + ["c2"] * 3,
        "grna": ["g1", "x", "y"] * 2,
        "cell_area": [float(i + 2) for i in range(6)],
        "cell_perimeter": [float(i + 12) for i in range(6)],
    })
    monkeypatch.setattr(view, "rows_to_compare", lambda: objects)
    monkeypatch.setattr(view, "_all_objects", lambda: objects)
    monkeypatch.setattr(view, "picked_groups", lambda: {"picked": [0, 3]})
    state = {
        "measurement": "cell_perimeter", "operator": "-",
        "second": "cell_area", "level": "cell", "kind": "bar",
        "spread": "sd", "scope": "all", "contrast": "within_well",
        "only": "picked", "controls": "control_a",
        "selected_guides": [], "selected_wells": [], "chosen_wells": None,
        "join_png_list": False, "join_dependent": True,
    }
    with monkeypatch.context() as guard:
        for name in ("build", "with_statistics", "join_measurements"):
            guard.setattr(compare, name, lambda *a, **k: pytest.fail("restore computed comparison"))
        assert view.apply_workspace_state({"comparison": state})
        assert view._graph_panel is None
        assert view.workspace_state()["comparison"] == state
    panel = view.compare_a_measurement()
    assert panel is not None
    assert panel.workspace_state() == state
    assert panel.comparison().measurement == "cell_perimeter - cell_area"
    assert panel.comparison().level == "cell"
    assert panel._canvas.spec.kind == "bar"
    assert panel._canvas.spec.spread == "sd"
    assert set(panel._canvas.spec.frame["group"]) == {"picked"}
    previous = panel.comparison()
    with monkeypatch.context() as guard:
        for name in ("build", "with_statistics", "join_measurements"):
            guard.setattr(compare, name, lambda *a, **k: pytest.fail("existing restore computed comparison"))
        assert view.apply_workspace_state({"comparison": {"kind": "box", "selected_wells": None}})
    assert panel.comparison() is previous
    assert panel.workspace_state()["kind"] == "box"
    assert panel.workspace_state()["selected_wells"] is None


def _loaded_sources(view):
    frame = pd.DataFrame({"feature": ["gene_fraction:gene[GRA14]"], "coefficient": [0.2]})
    view._frame_provider = lambda: frame
    view._results_provider = lambda: "/not-probed/run/results.csv"
    view._database_provider = lambda: [{"plate": "p1", "database": "/not-probed/measurements.db"}]


def test_restore_loaded_run_does_not_stat_optional_run_name(view, monkeypatch):
    from spacr.qt import path_probe

    _loaded_sources(view)
    monkeypatch.setattr(path_probe, "exists", lambda path: True)
    monkeypatch.setattr(montage.os.path, "isfile", lambda path: pytest.fail("restore performed filesystem stat"))
    assert view.apply_workspace_state({"coefficient": "gene_fraction:gene[GRA14]"})
    assert view._name == "GRA14"
    assert view._show.isEnabled()
    assert view.status_text() == 'Ready: press “Show the cells” for GRA14.'
    assert not view.__dict__.get("_restoring_workspace", False)


def test_restore_keeps_nonblocking_database_presence_validation(view, monkeypatch):
    from spacr.qt import path_probe

    _loaded_sources(view)
    probes = []
    monkeypatch.setattr(path_probe, "exists", lambda path: (probes.append(path), False)[1])
    assert view.apply_workspace_state({"coefficient": "gene_fraction:gene[GRA14]"})
    assert probes and set(probes) == {"/not-probed/measurements.db"}
    assert not view._show.isEnabled()
    assert "No measurement database" in view.status_text()


def test_restore_guard_and_signal_blocks_are_restored_after_exception(view, monkeypatch):
    view._restoring_workspace = "outer"
    view._channels.blockSignals(True)

    def fail(_values):
        raise ValueError("controlled restore failure")

    monkeypatch.setattr(view, "_write_back", fail)
    with pytest.raises(ValueError, match="controlled restore failure"):
        view.apply_workspace_state({"widgets": {"channels": "r"}})
    assert view._restoring_workspace == "outer"
    assert view._channels.signalsBlocked()
    assert not view._source.signalsBlocked()
