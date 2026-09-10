"""The re-fit dialog never promises "nothing will happen" before starting a fit.

Instruction 310, entry A53. ``RefitDialog._refresh`` ended::

    where = destination(settings)
    lines = list(notes)
    if where:
        lines.append(f"Writes to {where}.")
    self._notice.setText(" ".join(lines) if lines else
                         "Nothing to change: re-fitting these settings "
                         "would repeat the run you are looking at.")
    self._ok(True)

READ CLOSELY, THAT FALLBACK WAS REACHABLE IN EXACTLY ONE SITUATION. A resolved
destination always appends a line of its own, so ``lines`` could only be empty
when ``destination()`` returned ``None``. The single case where the dialog did
not know where the output would go was therefore the single case where it told
the user nothing would happen -- and left the button enabled to start a real
fit anyway.

``destination()`` returns ``None`` when no count-data path resolves or the
results folder cannot be created, so there is nothing to describe and nothing
safe to start. It now refuses and says why.

The settings below are the ones the entry names: ``count_data`` as a list whose
first entry is falsy. ``refit_settings`` accepts it because the list itself is
truthy, and ``destination`` rejects it further down.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialogButtonBox

import spacr.qt.widgets.refit_dialog as rd

#: The entry named ``count_data=[""]`` as the way in. IT NO LONGER IS:
#: ``refit_settings`` now raises for it first, with a better message than the
#: fallback ever gave, and ``_refresh`` returns at its ``except ValueError``
#: branch. Re-checked before writing these tests, and the other two routes
#: into ``destination() is None`` were checked too -- ``_next_results_folder``
#: only computes a name, so it returns a path even when ``results`` is a file
#: or the source directory is unreadable, and never raises OSError there.
#:
#: So the branch is defensive rather than live, and is reached here by
#: replacing ``destination``. That is deliberate: leaving it unreachable would
#: leave an uncoverable line, which is the thing instruction 288 exists to
#: remove, and leaving the old message would leave one that says the opposite
#: of the truth.
UNRESOLVABLE = {"regression_type": "ols", "count_data": [""]}


def _ok_button(dialog):
    box = dialog.findChild(QDialogButtonBox)
    return box.button(QDialogButtonBox.Ok)


@pytest.fixture()
def dialog(qtbot, tmp_path, monkeypatch):
    """A dialog whose settings are runnable but whose destination is unknown."""
    counts = tmp_path / "counts.csv"
    counts.write_text("well,count\na1,1\n", encoding="utf-8")
    # `_refresh` imports destination inside the method, so the name to replace
    # lives on spacr.refit, not on the dialog module.
    import spacr.refit

    monkeypatch.setattr(spacr.refit, "destination", lambda _settings: None)
    widget = rd.RefitDialog({"regression_type": "ols",
                             "count_data": [str(counts)]})
    qtbot.addWidget(widget)
    return widget


def test_the_entrys_own_input_is_refused_earlier_and_better(qtbot):
    """``count_data=[""]`` is rejected by refit_settings, not by the fallback."""
    widget = rd.RefitDialog(dict(UNRESOLVABLE))
    qtbot.addWidget(widget)
    assert _ok_button(widget).isEnabled() is False
    assert "no usable count data path" in widget._notice.text()


def test_it_does_not_claim_the_refit_would_change_nothing(dialog):
    """The false statement the entry is about."""
    assert "Nothing to change" not in dialog._notice.text(), (
        "the dialog said a re-fit would merely repeat the run on screen, "
        "when it could not work out where the output would go"
    )


def test_it_says_the_output_folder_could_not_be_worked_out(dialog):
    text = dialog._notice.text()
    assert "output folder" in text and "count data" in text, text


def test_the_refit_button_is_refused(dialog):
    assert _ok_button(dialog).isEnabled() is False, (
        "a re-fit whose destination the dialog could not derive must not be "
        "offered; a control that silently does the wrong thing is worse than "
        "one that is greyed out"
    )


def test_a_resolvable_destination_is_still_offered_and_named(qtbot, tmp_path):
    """The refusal above is a refusal, not the dialog's new normal state."""
    counts = tmp_path / "counts.csv"
    counts.write_text("well,count\na1,1\n", encoding="utf-8")
    widget = rd.RefitDialog({"regression_type": "ols",
                             "count_data": [str(counts)]})
    qtbot.addWidget(widget)

    assert _ok_button(widget).isEnabled() is True
    assert "Writes to " in widget._notice.text()
    assert "cannot be worked out" not in widget._notice.text()
