"""``rescale=False`` divides by zero inside Cellpose 4.2.

Cellpose 4.0 DEPRECATED-AND-IGNORED ``rescale``, so spaCR passing ``False``
cost nothing and the wrong type went unnoticed. 4.2 reads it again::

    niter_scale = 1 if rescale is None or not resample else rescale
    niter = int(200/niter_scale) if niter is None or niter == 0 else niter

``rescale=False`` with ``resample=True`` therefore evaluates ``int(200/False)``
-- ZeroDivisionError, raised from inside Cellpose, on a settings combination
both GUIs offer: ``rescale`` ships False and a user turning ``resample`` on is
doing an ordinary thing.

These tests do not need the weights or a GPU. The arithmetic is the bug, and
it is reproduced from the installed library's own source so it cannot drift
into testing a copy of the formula rather than the formula.
"""
from __future__ import annotations

import inspect

import pytest

from spacr.spacr_cellpose import cellpose_rescale


def _installed_niter_lines():
    """The two lines of Cellpose's eval that consume ``rescale``."""
    from cellpose import models

    source = inspect.getsource(models.CellposeModel.eval)
    return [line.strip() for line in source.splitlines()
            if "niter_scale" in line]


def _niter_for(rescale, resample, niter=None):
    """Cellpose 4.2's own computation, transcribed from the lines above."""
    niter_scale = 1 if rescale is None or not resample else rescale
    return int(200 / niter_scale) if niter is None or niter == 0 else niter


# ---------------------------------------------------------------------------
# The normaliser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [False, 0, 0.0, None, ""])
def test_every_falsy_rescale_becomes_none(value):
    """None is Cellpose's own "not set", and takes the neutral branch."""
    assert cellpose_rescale(value) is None


@pytest.mark.parametrize("value", [0.5, 1, 2.0])
def test_a_real_rescale_is_passed_through(value):
    """A user who asks for rescaling must still get it."""
    assert cellpose_rescale(value) == value


# ---------------------------------------------------------------------------
# The arithmetic the normaliser exists for
# ---------------------------------------------------------------------------

def test_false_and_resample_is_the_division_by_zero():
    """The bug itself, so the fix is not taken on faith."""
    with pytest.raises(ZeroDivisionError):
        _niter_for(rescale=False, resample=True)


def test_the_normalised_value_does_not_divide_by_zero():
    assert _niter_for(rescale=cellpose_rescale(False), resample=True) == 200


def test_the_normalised_value_matches_what_false_was_meant_to_mean():
    """`False` was always intended as "no rescaling", which is niter_scale=1.

    Asserted against the branch Cellpose takes when rescale is genuinely
    unset, so the fix restores the intent rather than picking a number.
    """
    assert (_niter_for(rescale=cellpose_rescale(False), resample=True)
            == _niter_for(rescale=None, resample=False))


def test_resample_off_was_always_safe():
    """Which is why the shipped defaults never hit it, and nobody noticed."""
    assert _niter_for(rescale=False, resample=False) == 200


# ---------------------------------------------------------------------------
# The call sites
# ---------------------------------------------------------------------------

def test_no_eval_call_in_spacr_cellpose_passes_a_bare_false():
    """Both `model.eval` calls here must go through the normaliser or None.

    Read with the AST rather than by grepping the text: the docstring of
    ``cellpose_rescale`` quotes ``rescale=False`` while explaining the bug,
    and a substring search cannot tell an explanation from a call.
    """
    import ast

    from spacr import spacr_cellpose

    tree = ast.parse(inspect.getsource(spacr_cellpose))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "eval"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "rescale":
                continue
            value = keyword.value
            # None is the CORRECT value -- Cellpose's own "not set". Only a
            # falsy value that is not None is the bug.
            if (isinstance(value, ast.Constant)
                    and value.value is not None and not value.value):
                offenders.append(
                    f"line {node.lineno}: rescale={value.value!r}")

    assert not offenders, (
        "a falsy literal rescale reached a Cellpose eval call again: "
        + "; ".join(offenders))


def test_the_settings_default_is_still_falsy():
    """If this changes, the normaliser stops being load-bearing and this
    file should be re-read rather than assumed."""
    from spacr.settings import get_identify_masks_finetune_default_settings

    settings = get_identify_masks_finetune_default_settings({})
    assert not settings["rescale"]


# ---------------------------------------------------------------------------
# Against the installed library
# ---------------------------------------------------------------------------

def test_the_transcribed_formula_still_matches_the_installed_source():
    """Guards the transcription above from drifting away from Cellpose.

    Skips where the installed release does not compute niter this way --
    4.0 ignores rescale entirely -- because there is nothing to match.
    """
    lines = _installed_niter_lines()
    if not any("rescale" in line for line in lines):
        # 4.0.7 computes the same niter from `image_scaling` (derived from
        # diameter) and never reads `rescale` at all, which is exactly why
        # passing False was harmless there. Nothing to match.
        pytest.skip(f"this Cellpose does not derive niter from rescale: {lines}")

    assert any("rescale is None or not resample" in line for line in lines), (
        f"Cellpose changed how rescale reaches niter: {lines}")
