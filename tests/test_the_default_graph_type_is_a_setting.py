"""The setting decides which graph is drawn first, everywhere."""
from __future__ import annotations

import pytest

from spacr import graph_types


@pytest.fixture
def no_saved_choice(monkeypatch):
    """Every test starts from "nothing has been chosen"."""
    saved = {}
    monkeypatch.setattr(
        "spacr.qt.preferences.get_default_graph_type",
        lambda shape: saved.get(str(shape), ""))
    return saved


def test_without_a_choice_the_table_decides(no_saved_choice):
    """Unchanged behaviour for a user who never expressed a preference."""
    for shape, expected in graph_types.DEFAULTS.items():
        assert graph_types.default_for(shape) == expected


def test_the_saved_choice_wins(no_saved_choice):
    """Asked for 2026-08-28: the setting dictates what is shown from the start."""
    no_saved_choice["categorical_continuous"] = "violin"
    assert graph_types.default_for("categorical_continuous") == "violin"

    no_saved_choice["categorical_continuous"] = "bar"
    assert graph_types.default_for("categorical_continuous") == "bar"


def test_a_choice_that_does_not_fit_the_data_is_ignored(no_saved_choice):
    """A bar of two continuous axes is a different graph of different data."""
    no_saved_choice["continuous_continuous"] = "bar"
    assert graph_types.fits("continuous_continuous", "bar") is False
    assert graph_types.default_for("continuous_continuous") == \
        graph_types.DEFAULTS["continuous_continuous"]
    # And the module can say why, which is what the greyed menu entry shows.
    assert graph_types.why_not("continuous_continuous", "bar")


def test_a_missing_preferences_module_still_draws(monkeypatch):
    """Headless spaCR has no Qt; a figure still has to be drawn."""
    def _explode(_shape):
        raise RuntimeError("no preference store here")

    monkeypatch.setattr(
        "spacr.qt.preferences.get_default_graph_type", _explode)
    assert graph_types.default_for("categorical_continuous") == \
        graph_types.DEFAULTS["categorical_continuous"]


def test_an_unknown_shape_still_raises(no_saved_choice):
    """The contract was KeyError; honouring a preference must not hide that."""
    with pytest.raises(KeyError):
        graph_types.default_for("not_a_shape")


def test_every_graph_reaches_its_start_through_one_function():
    """`default_for` is the chokepoint, which is why the setting reaches all."""
    from spacr.qt.widgets import grouped_plot
    import inspect

    source = inspect.getsource(grouped_plot.PlotSpec.default_kind)
    assert "default_for" in source


def test_the_round_trip_persists(qapp, monkeypatch, tmp_path):
    """Saving and reading back the choice."""
    from spacr.qt import preferences

    store = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return store.get(key, default)

        def setValue(self, key, value):
            store[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(preferences, "_settings", lambda: _Mem())

    assert preferences.get_default_graph_type("continuous_only") == ""
    preferences.set_default_graph_type("continuous_only", "violin")
    assert preferences.get_default_graph_type("continuous_only") == "violin"

    # Clearing goes back to "nothing chosen", not to a stored copy.
    preferences.set_default_graph_type("continuous_only", "")
    assert preferences.get_default_graph_type("continuous_only") == ""
