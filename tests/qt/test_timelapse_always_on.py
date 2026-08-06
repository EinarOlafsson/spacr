"""The Timelapse module does not ask whether it is a timelapse.

Turning it off would leave a screen whose every remaining control -- the
frame interval, the axis order, the tracking setup, the transient filter --
is about a time dimension it had just been told to ignore. Mask Generation
is the module for that, and it is one click away.

The setting still has to exist and still has to be True. A key that is
merely *absent* from the settings dict is not the same as one set to True:
the pipeline falls back to its own default for a missing key, and the two
can disagree without anything saying so.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def model():
    from spacr.qt.screens.settings_model import SettingsWidgets

    return SettingsWidgets("timelapse")


def _rendered_keys(sections):
    return {widget.property("settingKey")
            for _title, rows in sections for _label, widget in rows}


def test_the_timelapse_toggle_is_not_offered(model, qapp):
    assert "timelapse" not in _rendered_keys(model.build_sections())


def test_no_widget_is_built_for_it_at_all(model, qapp):
    """Which is what actually hides it.

    The trailing "Other" section is built from `self._widgets`, so a key
    left out of every category still renders for as long as a widget
    exists. Removing it from the layout moved it to "Additional
    Settings"; removing it from the categories moved it to "Other".
    """
    model.build_sections()
    assert "timelapse" not in model._widgets


def test_it_is_still_in_the_settings_and_still_true(qapp):
    from spacr.qt.screens.settings_model import resolve_default_settings

    settings = resolve_default_settings("timelapse")
    assert "timelapse" in settings, (
        "the key was dropped rather than hidden; the pipeline would fall "
        "back to its own default")
    assert settings["timelapse"] is True


def test_it_is_forced_rather_than_defaulted(qapp, monkeypatch):
    """A settings CSV saved by an older build cannot turn this module off.

    `timelapse: False` used to round-trip through this screen, which would
    quietly make the Timelapse module a slower Mask Generation.
    """
    import spacr.settings as settings_module
    from spacr.qt.screens import settings_model

    real = settings_module.get_timelapse_settings

    def _stale(settings=None, **kwargs):
        out = real(settings={} if settings is None else settings, **kwargs)
        out["timelapse"] = False        # what an older CSV carried
        return out

    # `resolve_default_settings` imports the getter from `spacr.settings`
    # at call time, so that is where the stale value has to come from.
    monkeypatch.setattr(settings_module, "get_timelapse_settings", _stale)
    assert settings_model.resolve_default_settings("timelapse")["timelapse"] \
        is True


def test_the_search_index_does_not_offer_it_either(qapp):
    """Hidden from the settings search too, or it is reachable anyway.

    `categories_for_app` is what the Ctrl+F index is built from, so a key
    hidden only at render time would still be findable -- and clicking the
    result would scroll to a control that is not there.
    """
    from spacr.qt.screens.settings_model import categories_for_app
    from spacr.settings import categories

    rendered = categories_for_app("timelapse", categories)
    placed = {key for keys in rendered.values() for key in keys}
    assert "timelapse" not in placed


# ---------------------------------------------------------------------------
# The blast radius: no other module loses the control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", ["measure", "mask"])
def test_other_modules_are_untouched(app_key, qapp):
    """Measure legitimately offers it -- it decides how crops are grouped.

    Mask drops the key in `resolve_default_settings` for its own reasons,
    so this asserts what each module actually does rather than assuming
    they agree.
    """
    from spacr.qt.screens.settings_model import (SettingsWidgets,
                                                 resolve_default_settings)

    settings = resolve_default_settings(app_key)
    model = SettingsWidgets(app_key)
    shown = "timelapse" in _rendered_keys(model.build_sections())
    assert shown is ("timelapse" in settings), (
        f"{app_key}: the key is {'in' if 'timelapse' in settings else 'not in'} "
        f"its settings but {'is' if shown else 'is not'} rendered")


def test_hiding_is_declared_in_one_place(qapp):
    """So the next module that needs it does not invent a second mechanism."""
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    assert _APP_HIDDEN_KEYS.get("timelapse") == {"timelapse"}
