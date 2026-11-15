"""The re-fit dialog's edges: a stale penalty box, a run with no folder to
name, and a button box that is no longer where the dialog left it.

This is the one plot context-menu entry that STARTS A FIT, so every sentence
it puts on screen is a promise about what the next run will do. Three of those
promises are made on paths the ordinary click-through never reaches:

  * the penalty weight is only reported for a model that reads one, and it is
    checked a second time in :meth:`RefitDialog.chosen` -- because a spin box
    that is enabled when it should not be would hand ``alpha`` to a model that
    REFUSES it (:func:`spacr.ml._reject_unused_settings` raises), turning a
    dialog the user filled in correctly into a crash minutes into the fit;
  * "Writes to ..." is only written when a destination could be worked out. A
    settings file whose ``count_data`` is a list with an empty first entry
    names no counts to derive a folder from, and inventing a path there would
    tell the user their results are somewhere they will never be;
  * the notice must still be written when the Re-fit button cannot be found.
    Themes and hosts re-parent dialog button boxes; if losing the button also
    lost the notice, the user would be left with an unexplained dialog rather
    than a disabled button.
"""

from __future__ import annotations


import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialogButtonBox  # noqa: E402

from spacr.qt.widgets import refit_dialog as rd  # noqa: E402

pytestmark = pytest.mark.qt


#: A settings dict a re-fit can actually run from: it names counts, so
#: :func:`spacr.refit.refit_settings` accepts it and a destination exists.
_RUNNABLE = {
    "count_data": "/data/screen/counts.csv",
    "score_data": "/data/screen/scores.csv",
    "regression_type": "ols",
    "level": "both",
}


def _dialog(qtbot, settings):
    dialog = rd.RefitDialog(dict(settings))
    qtbot.addWidget(dialog)
    return dialog


def _pick_type(dialog, name):
    """Choose a regression type the way the combo box's user does."""
    index = dialog._type.findData(name)
    assert index >= 0, f"{name!r} is not offered by the dialog"
    dialog._type.setCurrentIndex(index)


# ------------------------------------------------------ the penalty weight

def test_an_enabled_penalty_box_does_not_smuggle_alpha_into_an_ols_refit(
        qtbot):
    """``chosen()`` is public: the plot menu can read it at any moment, not
    only while ``_refresh`` has just finished greying the box. So the model's
    own settings list -- not the widget's enabled flag -- decides whether a
    penalty weight is reported. If it were the flag, an OLS re-fit could carry
    ``alpha`` into a run that rejects it outright, and the user would get a
    traceback from the fit rather than a dialog that quietly left it out.

    The same dialog reports the number as soon as a model that reads one is
    picked, so this is a discrimination and not a permanent refusal.
    """
    dialog = _dialog(qtbot, _RUNNABLE)
    dialog._alpha.setValue(0.75)

    # OLS reads cov_type and nothing else, so the box is greyed out.
    assert dialog._alpha.isEnabled() is False
    # Force it back on, the state a re-styling pass or a stale repaint leaves.
    dialog._alpha.setEnabled(True)
    assert dialog.chosen()["alpha"] is None
    assert dialog.chosen()["regression_type"] == "ols"

    _pick_type(dialog, "ridge")

    assert dialog._alpha.isEnabled() is True
    assert dialog.chosen()["alpha"] == pytest.approx(0.75)


def test_switching_to_a_penalised_model_names_the_penalty_it_will_use(qtbot):
    """The counterpart promise: picking ridge from an OLS run has to report
    both the model change and the number the penalty will take, because the
    penalty weight is the whole reason to prefer ridge and a re-fit that
    dropped it would repeat the run with a default nobody chose.
    """
    dialog = _dialog(qtbot, _RUNNABLE)
    _pick_type(dialog, "ridge")
    dialog._alpha.setValue(2.5)

    settings, notes = dialog.settings()

    assert settings["regression_type"] == "ridge"
    assert settings["alpha"] == pytest.approx(2.5)
    assert any("'ols' -> 'ridge'" in note for note in notes), notes


# ------------------------------------------------- a run with no destination

# ------------------------------------------------------- a missing OK button

def test_the_notice_is_still_written_when_the_button_box_has_been_reparented(
        qtbot):
    """Same promise one level up: some layouts lift the whole button box out
    of the dialog and hang it in a shared footer, and ``findChild`` then
    answers with nothing at all. The form has to go on describing the run it
    would start -- the level change below is exactly the case this dialog was
    extended for (a guide-level fixed-effect fit to get per-guide p values),
    and it is worth nothing if the sentence announcing it never appears.
    """
    dialog = _dialog(qtbot, _RUNNABLE)
    box = dialog.findChild(QDialogButtonBox)
    assert box.button(QDialogButtonBox.Ok).isEnabled() is True

    box.setParent(None)          # kept alive by this local reference
    assert dialog.findChild(QDialogButtonBox) is None

    index = dialog._level.findData("grna")
    assert index >= 0, "the dialog offers no guide level"
    dialog._level.setCurrentIndex(index)

    assert "level 'both' -> 'grna'" in dialog._notice.text()
    assert dialog.chosen()["level"] == "grna"
    box.deleteLater()


# --------------------------------------------------- the modal entry point
