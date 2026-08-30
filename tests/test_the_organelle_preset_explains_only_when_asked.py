"""Filling organelle settings from a preset, and the explanation that is opt-in.

``apply_preset`` fills only what the user has NOT set, which is the whole
contract: a recommended value must never overwrite a chosen one. The
explanation beside it is opt-in because this runs on every settings build, and
printing a paragraph each time is how a user learns to stop reading the console.
"""
from __future__ import annotations

import pytest


def _a_shape():
    """A preset name that exists, taken from the module rather than assumed."""
    from spacr.organelle_types import apply_preset

    for name in ("punctate", "vesicular", "spherical", "filamentous"):
        try:
            apply_preset({"organelle_type": name})
            return name
        except Exception:
            continue
    pytest.skip("no known organelle preset resolved")


def test_a_preset_fills_only_what_the_user_left_unset():
    """The kept/applied split, which is the function's whole contract.

    A recommended value overwriting a chosen one is the failure that matters:
    the user would set a diameter, run, and get a different one with no
    indication which was used.
    """
    from spacr.organelle_types import apply_preset

    shape = _a_shape()
    filled = apply_preset({"organelle_type": shape})
    assert filled, "the preset supplied at least one value"

    chosen_key = next(k for k in filled
                      if k != "organelle_type" and filled[k] is not None)
    sentinel = "a value the user chose"

    out = apply_preset({"organelle_type": shape, chosen_key: sentinel})

    assert out[chosen_key] == sentinel


def test_a_value_explicitly_set_to_none_is_treated_as_unset():
    """``out[key] is not None`` is the test, not mere presence.

    A settings CSV round trip writes None for a cleared field, and treating
    that as a choice would leave the setting empty rather than recommended.
    """
    from spacr.organelle_types import apply_preset

    shape = _a_shape()
    filled = apply_preset({"organelle_type": shape})
    chosen_key = next(k for k in filled
                      if k != "organelle_type" and filled[k] is not None)

    out = apply_preset({"organelle_type": shape, chosen_key: None})

    assert out[chosen_key] == filled[chosen_key]


def test_nothing_is_printed_unless_the_caller_asks(capsys):
    """The ``if explain and preset.label:`` branch NOT taken.

    This runs on every settings build. Explaining every time is how the
    console becomes noise, so the default is silence.
    """
    from spacr.organelle_types import apply_preset

    apply_preset({"organelle_type": _a_shape()})

    assert capsys.readouterr().out == ""


def test_asking_for_an_explanation_prints_the_preset(capsys):
    """The taken side, so the silence above is visibly a decision."""
    from spacr.organelle_types import apply_preset

    apply_preset({"organelle_type": _a_shape()}, explain=True)

    printed = capsys.readouterr().out
    assert "organelle_type" in printed


def test_an_unknown_organelle_type_is_refused_by_name():
    """The vocabulary raise, which lists the shapes that do exist."""
    from spacr.organelle_types import apply_preset

    with pytest.raises(ValueError) as excinfo:
        apply_preset({"organelle_type": "nucleus"})

    assert "nucleus" in str(excinfo.value)
    assert "punctate" in str(excinfo.value)
