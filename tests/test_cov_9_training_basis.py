"""The training-basis vocabulary refuses a basis it does not know.

``dataset_mode`` decides what defines a training class. A name outside the
known set cannot be honoured and must not be guessed at: silently falling back
to one of the real bases would train the model on a completely different
definition of its classes than the user asked for.
"""
from __future__ import annotations

import pytest

from spacr import training_basis as tb


def test_an_unknown_basis_has_no_settings_and_says_which_exist():
    """Asking for the settings of an unknown basis raises and lists the real ones.

    The caller is usually a settings panel deciding which controls to grey
    out. Returning an empty tuple would grey out every control on the screen
    and leave no hint about what went wrong, so the error has to name the
    bases that do exist.
    """
    with pytest.raises(tb.TrainingBasisError) as excinfo:
        tb.settings_for_basis("morphology")
    message = str(excinfo.value)
    assert "morphology" in message
    for basis in tb.TRAINING_BASES:
        assert basis in message


def test_a_known_basis_still_reports_the_settings_it_reads():
    """The refusal above must not have broken the lookup it guards."""
    for basis in tb.TRAINING_BASES:
        assert tb.settings_for_basis(basis) == tb.BASIS_SETTINGS[basis]
