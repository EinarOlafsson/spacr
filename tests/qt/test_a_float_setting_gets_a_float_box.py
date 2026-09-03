"""A setting spaCR types as a float must not get a whole-numbers-only box.

The control for a number is chosen from the DEFAULT VALUE's Python type, so a
float setting that happens to ship a round default shipped an ``int`` and got
a ``QSpinBox``. The box then refused every value between the whole ones, and
nothing said so -- typing 0.4 left 0 behind.

The settings this hit are the ones where fractions are the whole point:

* ``cell_flow_threshold`` ships 100 and is documented "usable range about
  0-3", with Cellpose's own default at 0.4. The user could pick 0, 1, 2 or 3.
* ``*_perimeter_fraction`` is declared a plain float and is a FRACTION. It
  could be set to 0 or 1.
* ``*_signal_to_noise`` and ``*_background``, which are intensity ratios and
  intensity floors.

All the measured casualties were on Mask. Regression is here as a second
module the ratchet watches, not as a known one: its ``alpha`` is typed float
and its factory ships ``1``, but the screen already gave it a float box.

So the rule is that the DECLARED type wins over the default's Python type,
and this is a ratchet over the module rather than a test for those settings.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

#: Modules that ship a round default for a setting typed float.
MODULES = ("mask", "regression")


def _int_boxes_for_float_settings(screen):
    """Keys given a whole-numbers-only box despite admitting a float."""
    from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
    from spacr.settings import expected_types

    wrong = []
    for key, widget in screen._settings_model._widgets.items():
        if not isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
            continue
        declared = expected_types.get(key)
        if declared is None:
            continue
        types = declared if isinstance(declared, tuple) else (declared,)
        if float in types:
            wrong.append(key)
    return sorted(wrong)


@pytest.mark.parametrize("app_key", MODULES)
def test_no_float_setting_is_given_a_whole_number_box(qtbot, app_key):
    """The ratchet."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)

    wrong = _int_boxes_for_float_settings(screen)
    assert not wrong, (
        f"{app_key} offers a whole-numbers-only box for settings that admit "
        f"a fraction, so the values between are unreachable: {wrong}")


def test_the_flow_threshold_accepts_cellposes_own_default(qtbot):
    """0.4 is the number the tooltip names, and it has to survive."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    model = screen._settings_model

    assert model.set_value_for_key("cell_flow_threshold", 0.4)

    assert model.collect()["cell_flow_threshold"] == pytest.approx(0.4)


def test_the_shipped_default_is_not_clamped_by_its_new_box(qtbot):
    """A float box has a domain, and 100 has to still fit in it.

    The promotion is worthless if it silently rewrites the default on the way
    -- that would change what every untouched run does.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.settings import get_timelapse_settings

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    shipped = get_timelapse_settings()
    collected = screen._settings_model.collect()

    for key in ("cell_flow_threshold", "cell_cellprob_threshold",
                "cell_signal_to_noise", "cell_background"):
        assert float(collected[key]) == pytest.approx(float(shipped[key])), (
            f"{key} was changed by the widget that shows it")
