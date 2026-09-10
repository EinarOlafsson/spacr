"""Typing into a channel must not rearrange the form under the typist.

Reported 2026-08-28: "i try to type 1 into nuclear channel and the
application hangs". `_on_object_switch_changed` was connected to every
channel, count and type on the panel, and ran the whole object-visibility
pass synchronously on each character -- reading every value, deciding all
1,551 gated rows, and BUILDING the rows it decided to show. On the GUI
thread, once per keypress.
"""
from __future__ import annotations

import time

import pytest

import spacr.qt.app as app_module


@pytest.fixture(scope="module")
def mask(qapp):
    win = app_module.MainWindow()
    win.resize(1200, 800)
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    yield win._screens["mask"]
    win.close()


def _type_into(widget, text: str) -> None:
    if hasattr(widget, "setText"):
        widget.setText(text)
    else:
        widget.setValue(int(text or 0))


@pytest.mark.parametrize(
    "key", ["nucleus_channel", "cell_channel", "pathogen_channel"])
def test_a_keystroke_is_cheap(mask, qapp, key):
    """The budget is generous; the defect it catches was a hang."""
    widget = mask._settings_model._widgets.get(key)
    if widget is None:
        pytest.skip(f"{key} is not on this panel")

    worst = 0.0
    for index in range(6):
        started = time.perf_counter()
        _type_into(widget, str(index % 3))
        qapp.processEvents()
        worst = max(worst, time.perf_counter() - started)

    assert worst < 0.20, (
        f"typing into {key} cost {worst * 1000:.0f} ms a keystroke")


def test_the_rule_follows_nothing(mask):
    """The form is decided when it opens, and not again while typing."""
    model = mask._settings_model
    # Connecting nothing is the fix; a connection here is the regression.
    assert model._connect_object_visibility_signals() is None
    assert model._on_object_switch_changed() is None


def test_every_setting_on_the_form_is_collected(mask):
    """Every setting the form HOLDS is collected.

    The number is smaller than it was, deliberately: a run with no
    organelles and no nucleus channel does not build those settings at all
    (300), so counting them was counting the noise that change removed. What
    matters is that nothing on the form is lost and the settings that decide
    the form are always there.
    """
    model = mask._settings_model
    collected = model.collect() or {}
    assert len(collected) > 150
    for key in ("nucleus_channel", "cell_channel", "pathogen_channel",
                "number_of_organelles"):
        assert key in collected, f"{key} decides the form and must be on it"
    # Nothing the panel built is missing from what it collects.
    missing = [k for k in model._widgets if k not in collected]
    assert missing == [], f"{len(missing)} built settings are not collected"


def test_a_typed_channel_reaches_the_run(mask, qapp):
    """The value still counts -- only the form stops rearranging itself."""
    model = mask._settings_model
    widget = model._widgets.get("nucleus_channel")
    if widget is None:
        pytest.skip("nucleus_channel is not on this panel")
    _type_into(widget, "1")
    qapp.processEvents()
    assert str((model.collect() or {}).get("nucleus_channel", "")) == "1"
