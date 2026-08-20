"""Instruction 181 — the first lasso run a user makes has to be able to work.

From a live run on the reference screen:

    ValueError: lasso shrank all 790 coefficients to exactly zero at alpha=1

The refusal is right and stays: a lasso that kept every coefficient at zero
would be written out as "0 significant gRNAs", which is a result-shaped way of
saying the fit told you nothing and is indistinguishable on the page from a
screen with no hits.

What was wrong is everything around it. `alpha` shipped as the integer 1, so
the panel inferred an integer and built a QSpinBox: the documented 'auto'
could not be typed and neither could any value below 1 -- and every value the
control could reach collapses a fraction-scale design. The penalised families
were unrunnable from the GUI, and the `setdefault('alpha', 'auto')` that was
meant to prevent exactly this never fired, because a panel posts every key it
shows.
"""
from __future__ import annotations

import os

import pytest

from spacr.settings import (PENALISED_REGRESSION_TYPES,
                            get_perform_regression_default_settings as defaults)


# -- the default reaches the run -------------------------------------------

@pytest.mark.parametrize("family", PENALISED_REGRESSION_TYPES)
def test_a_penalised_family_cross_validates_its_penalty_by_default(family):
    assert defaults({"regression_type": family})["alpha"] == "auto"


@pytest.mark.parametrize("family", PENALISED_REGRESSION_TYPES)
def test_the_panels_posted_default_counts_as_absent(family, capsys):
    """The GUI posts every key it shows, so `setdefault` never fired."""
    resolved = defaults({"regression_type": family, "alpha": 1})
    assert resolved["alpha"] == "auto"
    # ANNOUNCED. A penalty chosen for the user and never named is one they
    # cannot put in a methods section.
    said = capsys.readouterr().out
    assert "alpha='auto'" in said and family in said


@pytest.mark.parametrize("family", PENALISED_REGRESSION_TYPES)
def test_a_penalty_the_user_actually_chose_is_left_alone(family, capsys):
    assert defaults({"regression_type": family, "alpha": 0.01})["alpha"] == 0.01
    # A literal float 1.0 is a choice; the integer 1 is what the panel had
    # lying around. The distinction is the whole of the rule.
    assert defaults({"regression_type": family, "alpha": 1.0})["alpha"] == 1.0
    assert "alpha='auto'" not in capsys.readouterr().out


def test_an_unpenalised_family_keeps_the_historical_default():
    """`alpha` means nothing to ols; changing it there would be a new bug."""
    assert defaults({"regression_type": "ols"})["alpha"] == 1
    assert defaults({"regression_type": "quantile"})["alpha"] == 1


# -- the control can express it --------------------------------------------

@pytest.mark.qt
def test_the_alpha_control_can_say_auto_and_can_say_a_small_number(qtbot):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QDoubleSpinBox

    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import AUTO_TEXT, _set_auto_or_number

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    model = screen._settings_model
    alpha = model._widgets["alpha"]

    # A QSpinBox could hold neither of the two values that matter.
    assert isinstance(alpha, QDoubleSpinBox)

    _set_auto_or_number(alpha, "auto")
    assert model._read_widget(alpha) == AUTO_TEXT
    assert alpha.text() == AUTO_TEXT          # and it SAYS so on screen

    _set_auto_or_number(alpha, 0.003)
    assert model._read_widget(alpha) == pytest.approx(0.003)

    _set_auto_or_number(alpha, None)
    assert model._read_widget(alpha) == AUTO_TEXT


@pytest.mark.qt
def test_the_panel_hands_the_run_something_the_run_accepts(qtbot):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import _set_auto_or_number

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    model = screen._settings_model
    _set_auto_or_number(model._widgets["alpha"], "auto")

    collected = model.collect() or {}
    assert collected["alpha"] == "auto"
    # And through the resolver the run itself uses.
    collected["regression_type"] = "lasso"
    assert defaults(collected)["alpha"] == "auto"
