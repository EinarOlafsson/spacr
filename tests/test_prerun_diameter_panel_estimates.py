"""The pre-run diameter panel: drawing :class:`spacr.diameter.DiameterEstimate` rows and applying them.

A measured diameter is only worth having if it can be taken in one press, and
an unusable one (NaN by construction) must never be written.  The writes go
through the settings model first and the form widget second; when both
refuse, the setting is left alone.  The panel's worker job is run on the
calling thread through ``_start_job``'s own signature.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from spacr.diameter import DiameterEstimate
from spacr.qt import prerun


class _Model:
    """The attributes ``prerun`` reads off ``screen._settings_model``."""

    def __init__(self, widgets=None, settings=None):
        self._widgets = dict(widgets or {})
        self._settings = dict(settings or {})
        self.written = {}

    def collect(self):
        return dict(self._settings)

    def set_value_for_key(self, key, value):
        self.written[key] = value
        return True


class _Screen(QWidget):
    """A screen shaped like ``AppScreen`` for the parts prerun touches."""

    def __init__(self, widgets=None, settings=None):
        super().__init__()
        self._settings_model = _Model(widgets, settings)
        self._applied = []
        self._runtime_wrap = QWidget(self)
        QVBoxLayout(self._runtime_wrap)
        self._actions_row = QWidget(self._runtime_wrap)
        QHBoxLayout(self._actions_row)
        self._runtime_wrap.layout().addWidget(self._actions_row)

    def _apply_value(self, widget, value):
        self._applied.append((widget, value))
        widget.setText(str(value))


def _field(text=""):
    field = QLineEdit()
    field.setText(text)
    return field


def _estimate(object_type="cell", diameter=24.0, usable=True):
    if usable:
        return DiameterEstimate(
            object_type=object_type, diameter=diameter, low=diameter - 4,
            high=diameter + 4, n_objects=120, n_fields=5,
            method="threshold_otsu", confidence="high",
            note="measured from five fields")
    return DiameterEstimate(
        object_type=object_type, diameter=float("nan"), low=float("nan"),
        high=float("nan"), n_objects=0, n_fields=0, method="none",
        confidence="low", note="nothing was found in these fields")


@pytest.fixture
def panel_for(qtbot, tmp_path):
    def build(settings=None, estimator=None):
        fields = {"src": _field(str(tmp_path))}
        for obj in ("cell", "nucleus", "pathogen"):
            fields[f"{obj}_diameter"] = _field("")
        screen = _Screen(widgets=fields, settings=settings)
        qtbot.addWidget(screen)
        panel = prerun.DiameterPanel(screen, estimator=estimator)
        qtbot.addWidget(panel)
        return panel
    return build


def _texts(panel):
    return [label.text() for label in panel.findChildren(QLabel)]


def _raise(message):
    def refuse(*_args):
        raise RuntimeError(message)
    return refuse


APPLY = [
    # case, estimate, model setter, widget applier, result, model writes, widget text, logged
    ("the_model_takes_a_whole_number", _estimate(diameter=23.6), None, None,
     True, {"cell_diameter": 24}, "", ""),
    ("a_model_that_raises_falls_back_to_the_widget", _estimate(diameter=24.0),
     _raise("that key is locked"), None, True, {}, "24", "could not write cell_diameter"),
    ("a_model_that_declines_falls_back_to_the_widget", _estimate(diameter=24.4),
     lambda key, value: False, None, True, {}, "24", ""),
    ("both_refusing_leaves_the_setting_alone", _estimate(diameter=24.0),
     lambda key, value: False, _raise("that widget is gone"), False, {}, "",
     "could not write cell_diameter"),
    ("an_unusable_estimate_is_never_written", _estimate(usable=False), None, None,
     False, {}, "", ""),
]


def test_apply_writes_a_whole_number_through_the_first_route_that_takes_it(
        panel_for, caplog, subtests):
    """The model is tried first, the widget second; the status names the written value."""
    for case_id, estimate, setter, applier, ok, written, text, logged in APPLY:
        with subtests.test(msg=case_id):
            panel = panel_for()
            panel._on_estimated({"estimates": {"cell": estimate}})
            screen = panel._screen
            if setter is not None:
                screen._settings_model.set_value_for_key = setter
            if applier is not None:
                screen._apply_value = applier
            status = panel._status.text()
            caplog.clear()
            with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
                assert panel.apply("cell") is ok
            assert screen._settings_model.written == written
            assert screen._settings_model._widgets["cell_diameter"].text() == text
            assert panel._status.text() == ("cell_diameter set to 24 px." if ok else status)
            assert [r.getMessage() for r in caplog.records] == ([logged] if logged else [])
            assert panel.apply("nucleus") is False


def test_estimates_are_redrawn_as_one_row_per_object(panel_for, subtests):
    """A new estimate replaces the old row; an unusable one says "no estimate" and offers no button."""
    with subtests.test(msg="redrawn_rather_than_stacked"):
        panel = panel_for()
        seen = []
        panel.estimated.connect(seen.append)
        panel._on_estimated({"estimates": {"cell": _estimate()}})
        panel._on_estimated({"estimates": {"cell": _estimate(diameter=30.0)}})
        assert panel._rows_layout.count() == 1
        texts = _texts(panel)
        assert "cell: 30.0 px" in texts and "cell: 24.0 px" not in texts
        assert ("10th-90th percentile 26.0-34.0 px · measured on 120 object(s) "
                "across 5 field(s) · high confidence · threshold_otsu") in texts
        assert seen == [["cell"], ["cell"]]
        assert panel._screen._diameter_estimates["cell"].diameter == 30.0

    with subtests.test(msg="an_unusable_estimate_offers_nothing"):
        panel = panel_for()
        panel._on_estimated({"estimates": {"cell": _estimate(usable=False)}})
        texts = _texts(panel)
        assert "cell: no estimate" in texts
        assert "nothing was found in these fields" in texts
        assert panel._btn_use_all.isVisible() is False


USE_ALL = [
    ("every_usable_estimate_is_written",
     {"cell": _estimate("cell", 24.0), "nucleus": _estimate("nucleus", 12.0),
      "pathogen": _estimate("pathogen", usable=False)},
     {"cell_diameter": 24, "nucleus_diameter": 12},
     "Set cell_diameter, nucleus_diameter."),
    ("nothing_usable_says_nothing_new", {"cell": _estimate(usable=False)}, {}, None),
]


def test_use_all_writes_every_usable_estimate_and_names_them(panel_for, subtests):
    """*Use all* writes each usable diameter once and lists the keys; with none usable the status is unchanged."""
    for case_id, estimates, written, status in USE_ALL:
        with subtests.test(msg=case_id):
            panel = panel_for()
            panel._on_estimated({"estimates": estimates})
            before = panel._status.text()
            panel._on_use_all_clicked()
            assert panel._screen._settings_model.written == written
            assert panel._status.text() == (before if status is None else status)


def test_the_injected_estimator_is_the_one_the_job_calls(panel_for, tmp_path):
    """The measure job calls the injected estimator with the source, channels and field count."""
    seen = {}

    def estimator(src, channels, *, n_fields, metadata_type, custom_regex):
        seen.update(src=src, channels=dict(channels), n_fields=n_fields,
                    metadata_type=metadata_type, custom_regex=custom_regex)
        return {"cell": _estimate()}

    panel = panel_for(settings={"cell_channel": 0}, estimator=estimator)
    panel._start_job = lambda fn, box, on_done, key: (fn(box), on_done(box), True)[-1]

    panel._on_measure_clicked()

    assert seen == {"src": str(tmp_path), "channels": {"cell": 0}, "n_fields": 5,
                    "metadata_type": "cellvoyager", "custom_regex": None}
    assert panel.estimates["cell"].diameter == 24.0
    assert panel._btn_measure.isEnabled() is True
