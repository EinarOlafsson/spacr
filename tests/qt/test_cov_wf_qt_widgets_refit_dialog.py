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

def test_counts_recorded_as_an_empty_first_entry_name_no_output_folder(qtbot):
    """A settings CSV can record ``count_data`` as a list, and a run that was
    started from a picker the user cleared leaves an empty first entry. The
    list is still truthy, so the dialog does not refuse the re-fit -- but
    there is no path to derive a results folder from, and the notice must then
    say what it does know instead of naming a folder built from an empty
    string. "Writes to /results/ols" for a run that will write nowhere near
    there is worse than saying nothing about the destination.

    A dialog on the same settings with a real counts path does name its
    folder, which is what makes the silence above informative.
    """
    blank = _dialog(qtbot, dict(_RUNNABLE, count_data=[""]))
    real = _dialog(qtbot, _RUNNABLE)

    assert blank._notice.text() == (
        "Nothing to change: re-fitting these settings would repeat the run "
        "you are looking at.")
    assert "Writes to" not in blank._notice.text()
    assert real._notice.text() == "Writes to /data/screen/results/ols."


def test_a_reset_setting_is_still_reported_when_no_folder_can_be_named(qtbot):
    """Losing the destination line must not lose the notes beside it. The
    reset sentence is the one this dialog exists to print -- an OLS re-fit
    silently drops the previous run's ``alpha`` -- and a user whose settings
    name no derivable folder is entitled to that warning just as much as one
    whose settings do.
    """
    dialog = _dialog(qtbot, dict(_RUNNABLE, count_data=["", "counts.csv"],
                                 alpha=0.3))

    text = dialog._notice.text()

    assert "does not read alpha=0.3" in text
    assert "reset to default" in text
    assert "Writes to" not in text

    named = _dialog(qtbot, dict(_RUNNABLE, alpha=0.3))
    assert "does not read alpha=0.3" in named._notice.text()
    assert "Writes to /data/screen/results/ols." in named._notice.text()


# ------------------------------------------------------- a missing OK button

def test_the_notice_is_still_written_when_the_ok_button_has_been_taken_away(
        qtbot):
    """A host that restyles dialogs can rebuild the button box's contents, and
    the dialog then holds a box with no OK button in it. Enabling "Re-fit" is
    the last thing ``_refresh`` does, so a crash there would cost the user the
    notice as well as the button -- the sentence that says what the re-fit is
    about to change would never reach the screen, and the dialog would look
    like it had simply stopped responding to the form.
    """
    dialog = _dialog(qtbot, _RUNNABLE)
    box = dialog.findChild(QDialogButtonBox)

    # The button is there to begin with, and the fit is offered.
    assert box.button(QDialogButtonBox.Ok).text() == "Re-fit"
    assert box.button(QDialogButtonBox.Ok).isEnabled() is True

    box.clear()
    assert box.button(QDialogButtonBox.Ok) is None

    # A real edit of the form: tighten the significance level.
    dialog._fdr_alpha.setValue(0.01)

    assert "significance level 0.05 -> 0.01" in dialog._notice.text()
    assert "Writes to /data/screen/results/ols." in dialog._notice.text()


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


def test_a_refusal_still_reaches_the_button_when_the_box_is_where_it_was(
        qtbot):
    """The tolerance above must not be an excuse for never touching the
    button. Settings that name no counts cannot be re-fitted, and the only
    thing standing between the user and a fit that dies at its entry point is
    that ``_ok(False)`` finds the button and greys it out.
    """
    dialog = _dialog(qtbot, {"regression_type": "ols"})
    box = dialog.findChild(QDialogButtonBox)

    assert box.button(QDialogButtonBox.Ok).isEnabled() is False
    assert box.button(QDialogButtonBox.Cancel).isEnabled() is True
    assert "count data" in dialog._notice.text()


# --------------------------------------------------- the modal entry point

def test_cancelling_the_dialog_starts_no_fit_and_accepting_returns_the_run(
        monkeypatch, qtbot):
    """:func:`ask_refit` is what the plot's context menu actually calls, and
    it is the only place the user's Cancel is honoured. A re-fit is minutes of
    compute writing a new results folder, so a cancelled dialog has to return
    ``None`` -- not the settings it happened to be showing -- or the menu
    would start the fit the user just declined and leave a folder they never
    asked for beside their real results.

    Accepting the same dialog on the same settings does return the new run,
    which is what makes the ``None`` above a decision rather than a dead path.
    """
    def _cancel(self):
        return rd.QDialog.Rejected

    def _accept(self):
        # Pick a model that differs from the run on screen, the way a user
        # would before pressing "Re-fit".
        _pick_type(self, "ridge")
        return rd.QDialog.Accepted

    monkeypatch.setattr(rd.RefitDialog, "exec", _cancel)
    assert rd.ask_refit(dict(_RUNNABLE)) is None

    monkeypatch.setattr(rd.RefitDialog, "exec", _accept)
    settings, notes = rd.ask_refit(dict(_RUNNABLE))

    assert settings["regression_type"] == "ridge"
    assert settings["count_data"] == "/data/screen/counts.csv"
    assert any("'ols' -> 'ridge'" in note for note in notes), notes
