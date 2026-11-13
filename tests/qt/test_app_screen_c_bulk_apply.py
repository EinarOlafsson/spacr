"""A bulk apply of settings (an import, a preset) with nothing or someone else to read.

``apply_settings_dict`` compares what is on screen with what is coming,
decides whether the import changes the form's shape, and either applies the
values here or hands them to the replacement screen the window builds. Each
step is guarded; these cases drive each guard with the input that makes the
step happen too, so an untouched value is the failure and not a dead branch.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


class _Model:
    """The parts of ``SettingsWidgets`` a bulk apply touches."""

    def __init__(self, collect=None, widgets=None):
        self._widgets = dict(widgets or {})
        self._defaults = {}
        self._applying_settings = False
        self._collect = collect
        self.presets = []
        self.dependencies = 0

    def collect(self):
        if self._collect is None:
            return dict(self._widgets)
        return self._collect()

    def set_hidden_value(self, key, value):
        return False

    def _refresh_contextual_widgets(self):
        pass

    def apply_organelle_presets_from_mapping(self, mapping):
        self.presets.append(dict(mapping))

    def _refresh_setting_dependencies(self):
        self.dependencies += 1

    def _valid_committed_value(self, key):
        return self._widgets.get(key)


def _unreadable():
    raise RuntimeError("half-built")


def test_a_bulk_apply_runs_its_rules_with_or_without_a_model(
        screen, monkeypatch, subtests):
    """The import is applied whatever state the model is in.

    Invariant: the shape check is handed what is on screen (``{}`` when the
    model cannot be collected); with no model the values are still applied,
    their count returned and the folds reset; with a model the loop runs
    inside the applying flag, which is cleared afterwards, and the organelle
    presets and cross-setting rules run once.
    """
    for name, model, current in [
            ("model cannot be collected", _Model(collect=_unreadable), {}),
            ("model is read", _Model(collect=lambda: {"src": "/old"}),
             {"src": "/old"})]:
        with subtests.test(case=name), monkeypatch.context() as m:
            seen = []
            m.setattr(screen, "_settings_model", model)
            m.setattr(screen, "_bulk_apply_changes_form_shape",
                      lambda settings, now: seen.append(dict(now)) or False)
            assert screen.apply_settings_dict({"src": "/data"}) == 0
            assert seen == [current]

    with subtests.test(case="no settings model"), monkeypatch.context() as m:
        applied = []
        m.setattr(screen, "_apply_each_setting",
                  lambda settings: applied.append(dict(settings)) or 3)
        m.setattr(screen, "_bulk_apply_changes_form_shape",
                  lambda settings, now: False)
        m.setattr(screen, "_settings_model", None)
        m.setattr(screen, "_folds_last_switched_on", None, raising=False)
        assert screen.apply_settings_dict({"src": "/data"}) == 3
        assert applied == [{"src": "/data"}]
        assert screen._folds_last_switched_on == ()

    with subtests.test(case="a model brackets the loop"), \
            monkeypatch.context() as m:
        model = _Model()
        during = []
        m.setattr(screen, "_settings_model", model)
        m.setattr(screen, "_bulk_apply_changes_form_shape",
                  lambda settings, now: False)
        m.setattr(screen, "_apply_each_setting",
                  lambda settings: during.append(model._applying_settings)
                  or 3)
        assert screen.apply_settings_dict({"src": "/data"}) == 3
        assert during == [True]
        assert model._applying_settings is False
        assert model.presets == [{"src": "/data"}]
        assert model.dependencies == 1


def test_a_shape_changing_apply_lands_on_exactly_one_screen(
        screen, monkeypatch, subtests):
    """A rebuild hands the values on, or they are applied here -- never lost.

    Invariant: when the import changes the form's shape the window is asked
    to rebuild with the values; a replacement screen is handed them and its
    count returned; no replacement, or the window handing back this same
    screen, applies them here once without recursing. Progressive organelle
    settings (a slot with no count, a slot or count change with one, a slot
    beyond the count) extend the form and are not a shape change.
    """
    handed = []
    fresh = types.SimpleNamespace(apply_settings_dict=lambda settings: (
        handed.append(dict(settings)) or 7))
    # (case, window's screens, count returned, applied here, handed on)
    cases = [
        ("no replacement appears", {}, 1, [{"src": "/data"}], []),
        ("the same screen comes back", "self", 1, [{"src": "/data"}], []),
        ("a replacement takes the settings", {"regression": fresh}, 7, [],
         [{"src": "/data"}]),
    ]
    for name, screens, count, applied_here, handed_on in cases:
        with subtests.test(case=name), monkeypatch.context() as m:
            handed.clear()
            applied = []
            window = types.SimpleNamespace(
                _screens={"regression": screen} if screens == "self"
                else dict(screens), rebuilt=[])
            window.rebuild_app_screen = (
                lambda key, values: window.rebuilt.append((key,
                                                           dict(values))))
            m.setattr(screen, "window", lambda: window)
            m.setattr(screen, "_settings_model", _Model())
            m.setattr(screen, "_bulk_apply_changes_form_shape",
                      lambda settings, now: True)
            m.setattr(screen, "_thread", None)
            m.setattr(screen, "_apply_each_setting",
                      lambda settings: applied.append(dict(settings)) or 1)
            assert screen.apply_settings_dict({"src": "/data"}) == count
            assert applied == applied_here
            assert handed == handed_on
            if screens == {}:
                assert window.rebuilt == [("regression", {"src": "/data"})]

    from spacr.organelle_types import NUMBER_OF_ORGANELLES

    with_count = {NUMBER_OF_ORGANELLES: 1, "organelle_channel": None}
    cases = [
        ("slot setting, no count", {"organelle_channel": 2}, {}),
        ("slot setting with a count", {"organelle_channel": 2}, with_count),
        ("count grows", {NUMBER_OF_ORGANELLES: 2}, with_count),
        ("slot beyond the count", {"organelleb_channel": 2},
         {NUMBER_OF_ORGANELLES: 1, "organelleb_channel": None}),
    ]
    for name, incoming, current in cases:
        with subtests.test(case=name):
            assert screen._bulk_apply_changes_form_shape(
                incoming, current) is False
