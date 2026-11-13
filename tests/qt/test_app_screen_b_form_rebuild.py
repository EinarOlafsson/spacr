"""When a settings change reshapes the form, and what a rebuild may not lose.

The organelle count is the one setting that decides which rows the form holds.
The screen follows it on the field's commit signal, rebuilds only when the
shape on screen actually differs, never re-enters a rebuild, and clears its
re-entry guard however the rebuild ends. A settings file follows the same
rule (``_bulk_apply_changes_form_shape``), an old control-well trio is
migrated only when that is safe, and a verified preview model that the panel
refuses changes nothing at all.
"""
from __future__ import annotations

import hashlib
import logging
import os
import types
from copy import deepcopy

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QWidget                            # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


@pytest.fixture
def mask_screen(qtbot):
    widget = AppScreen("mask")
    qtbot.addWidget(widget)
    return widget


class _RecordingSignal:
    """A signal that records its connections, or refuses them."""

    def __init__(self, refuse=False):
        self.connected = []
        self._refuse = refuse

    def connect(self, slot):
        if self._refuse:
            raise RuntimeError("this signal has no receiver list")
        self.connected.append(slot)


# -- following the count and rebuilding the form -----------------------------

def _watch(env, **signals):
    widget = types.SimpleNamespace(**signals)
    env.mp.setattr(env.screen, "_settings_model", types.SimpleNamespace(
        _widgets={"number_of_organelles": widget}))
    env.screen._watch_the_settings_that_decide_the_form()
    return widget


def _watch_commit_signal(env):
    widget = _watch(env, editingFinished=_RecordingSignal())
    assert widget.editingFinished.connected == [
        env.screen._follow_the_organelle_count]


def _watch_no_model(env):
    env.mp.setattr(env.screen, "_settings_model", None)
    env.screen._watch_the_settings_that_decide_the_form()
    assert env.screen._settings_model is None


def _watch_falls_through_to_value_signal(env):
    widget = _watch(env, editingFinished=_RecordingSignal(refuse=True),
                    valueChanged=_RecordingSignal())
    assert widget.valueChanged.connected == [
        env.screen._follow_the_organelle_count]


def _watch_every_signal_refuses(env):
    widget = _watch(env, editingFinished=_RecordingSignal(refuse=True),
                    valueChanged=_RecordingSignal(refuse=True),
                    currentIndexChanged=_RecordingSignal(refuse=True))
    assert widget.valueChanged.connected == []
    assert widget.currentIndexChanged.connected == []


def _window_that_rebuilds(env):
    asked = []
    env.mp.setattr(env.screen.window(), "rebuild_app_screen",
                   lambda key, keep: asked.append((key, dict(keep))),
                   raising=False)
    return asked


def _rebuild_only_when_shape_changed(env):
    asked = _window_that_rebuilds(env)
    env.mp.setattr(env.screen, "_form_shape_on_screen",
                   env.screen._form_shape(), raising=False)
    env.mp.setattr(env.screen, "_deferred_form_values", None, raising=False)
    env.screen._rebuild_the_form()
    assert asked == []
    env.screen._form_shape_on_screen = (("number_of_organelles", "9"),)
    env.screen._rebuild_the_form()
    assert [key for key, _keep in asked] == [env.screen.app_key]


def _rebuild_not_reentered(env):
    asked = _window_that_rebuilds(env)
    env.mp.setattr(env.screen, "_form_shape_on_screen",
                   (("number_of_organelles", "9"),), raising=False)
    env.mp.setattr(env.screen, "_rebuilding_the_form", True, raising=False)
    env.screen._rebuild_the_form()
    assert asked == []
    env.screen._rebuilding_the_form = False
    env.screen._rebuild_the_form()
    assert len(asked) == 1


def _rebuild_failing_window_clears_guard(env):
    def refusing(_key, _keep):
        raise RuntimeError("the module registry is not loaded")

    env.mp.setattr(env.screen.window(), "rebuild_app_screen", refusing,
                   raising=False)
    env.mp.setattr(env.screen, "_form_shape_on_screen",
                   (("number_of_organelles", "9"),), raising=False)
    env.screen._rebuild_the_form()
    assert "could not rebuild the settings form" in env.caplog.text
    assert env.screen._rebuilding_the_form is False


def test_the_form_follows_its_shaping_setting_and_rebuilds_only_on_change(
        subtests, monkeypatch, caplog, screen):
    """The count is followed on the first commit signal that accepts a receiver
    (nothing when none does or there is no model); a rebuild is asked of the
    window only when the shape differs and none is under way, and a window that
    throws is logged and leaves the re-entry guard cleared."""
    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    cases = {
        "watch: commit signal": _watch_commit_signal,
        "watch: no model": _watch_no_model,
        "watch: falls through to valueChanged": _watch_falls_through_to_value_signal,
        "watch: every signal refuses": _watch_every_signal_refuses,
        "rebuild: only on changed shape": _rebuild_only_when_shape_changed,
        "rebuild: not re-entered": _rebuild_not_reentered,
        "rebuild: failing window": _rebuild_failing_window_clears_guard,
    }
    completed = []
    for name, case in cases.items():
        caplog.clear()
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(types.SimpleNamespace(screen=screen, mp=mp, caplog=caplog))
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- a settings file: reshape or not, and an old control-well trio -----------

def test_a_settings_file_never_reshapes_the_form_in_place(subtests,
                                                          mask_screen):
    """Changed object channels, a changed or unchanged organelle count, a form
    without a count and an identical dict all answer False: the progressive
    form grows organelle controls in place."""
    from spacr.organelle_types import NUMBER_OF_ORGANELLES as N

    channels = {N: 2, "nucleus_channel": None, "organelle_channel": 1,
                "organelleb_channel": None}
    cases = {
        "nucleus channel": ({"nucleus_channel": 0}, channels),
        "organelle channel": ({"organelle_channel": 3}, channels),
        "counted slot channel": ({"organelleb_channel": 2}, channels),
        "new count": ({N: 2}, {N: 1, "organelle_channel": 1}),
        "same count": ({N: 1}, {N: 1, "organelle_channel": 1}),
        "form without a count": ({N: 2, "nucleus_channel": 0},
                                 {"nucleus_channel": None}),
        "identical dict": ({"cell_diameter": 30}, {"cell_diameter": 30}),
    }
    for name, (settings, current) in cases.items():
        with subtests.test(case=name):
            assert mask_screen._bulk_apply_changes_form_shape(
                dict(settings), current) is False


def test_old_control_wells_are_migrated_only_when_that_is_safe(
        subtests, monkeypatch, caplog, screen):
    """A file that already names classes is returned as is, a successful
    translation writes ``classes``, and a translation that raises or yields no
    classes leaves the settings exactly as written (logging the failure)."""
    from spacr import classify_classes

    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    old = {"location_column": "col", "positive_control_id": "A1",
           "negative_control_id": "B2"}

    def refusing(_settings):
        raise ValueError("a well name that is not a well")

    cases = {
        "already names classes": (
            {"location_column": "col", "positive_control_id": "A1",
             "classes": [["A1"], ["B2"]]}, None, "same"),
        "translation succeeds": (
            old, lambda s: dict(s, classes=[["A1"], ["B2"]]), "classes"),
        "translation fails": (old, refusing, "same+logged"),
        "translates to nothing": (
            {"location_column": "col", "positive_control_id": "",
             "negative_control_id": ""}, lambda s: {"classes": []}, "same"),
    }
    for name, (settings, normalize, expect) in cases.items():
        caplog.clear()
        with subtests.test(case=name), monkeypatch.context() as mp:
            mp.setattr(screen, "_settings_model", types.SimpleNamespace(
                _widgets={"classes": QWidget()}))
            if normalize is not None:
                mp.setattr(classify_classes, "normalize_settings", normalize)
            migrated = screen._migrate_control_wells(settings)
            if expect == "classes":
                assert migrated["classes"] == [["A1"], ["B2"]]
            else:
                assert migrated is settings
            if expect == "same+logged":
                assert "could not migrate the control wells" in caplog.text


# -- a verified preview model the panel refuses -------------------------------

def test_a_refused_verified_preview_changes_nothing(
        subtests, monkeypatch, tmp_path, qtbot):
    """Whether the model field is missing, refuses the value, or the metadata
    is refused, the user is warned and the panel's settings, the model field's
    text and the checkpoint file are all left exactly as they were."""
    from PySide6.QtWidgets import QApplication, QMessageBox
    from spacr import model_zoo

    path = tmp_path / "verified.pt"
    raw_bytes = b"local verified preview checkpoint"
    path.write_bytes(raw_bytes)
    record = dict(key="preview-control", name="verified.pt", kind="cellpose",
                  sha256=hashlib.sha256(raw_bytes).hexdigest(),
                  repo_id="local/fixture", repo_type="model", version="1")
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS", (record,))
    entry = model_zoo._entry_from_mapping(record)
    reference = model_zoo._verified_model_reference(entry, str(path))
    assert reference is not None

    def text_of(field):
        return (field.currentText() if hasattr(field, "currentText")
                else field.text())

    for refusal in ("missing_widget", "value", "metadata"):
        with subtests.test(case=refusal), monkeypatch.context() as mp:
            screen = AppScreen("mask")
            qtbot.addWidget(screen)
            QApplication.processEvents()
            model = screen._settings_model
            field = model._widgets["cell_model_name"]
            before = deepcopy(model.collect())
            text = text_of(field)
            warnings = []
            mp.setattr(QMessageBox, "warning",
                       lambda *args: warnings.append(args[2]))
            if refusal == "missing_widget":
                mp.delitem(model._widgets, "cell_model_name")
            elif refusal == "value":
                original = model.set_value_for_key
                mp.setattr(model, "set_value_for_key",
                           lambda key, value: False if key == "cell_model_name"
                           else original(key, value))
            else:
                original = model.set_hidden_value
                mp.setattr(model, "set_hidden_value",
                           lambda key, value: False if key == "model_metadata"
                           else original(key, value))
            screen._propagate_verified_live_settings(
                {"cell_model_name": str(path), "cell_diameter": 97}, reference)
            if refusal == "missing_widget":
                model._widgets["cell_model_name"] = field
            assert warnings and model.collect() == before
            assert text_of(field) == text
            assert path.read_bytes() == raw_bytes
