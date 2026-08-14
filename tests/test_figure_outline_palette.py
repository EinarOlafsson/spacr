"""The published overlay figure has to be legible to its readers too.

Instruction 89 ends with the exported figure. The viewers were fixed first,
but the artefact that leaves spaCR and goes into a paper had cell drawn RED
and pathogen drawn GREEN -- the one pair red-green deficiency removes -- and
nothing let a user change it.

Everything asserted here is measured against Brettel-style simulations, the
same ones that chose the display primaries, so a future palette can replace
this one by beating the numbers rather than by argument.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from spacr.plot import OUTLINE_PALETTES, outline_palette_colours

DEUTERANOPE = np.array([[0.625, 0.700, 0.000],
                        [0.375, 0.300, 0.000],
                        [0.000, 0.000, 1.000]], np.float32)
PROTANOPE = np.array([[0.567, 0.433, 0.000],
                      [0.558, 0.442, 0.000],
                      [0.000, 0.242, 0.758]], np.float32)
TRITANOPE = np.array([[0.950, 0.050, 0.000],
                      [0.000, 0.433, 0.567],
                      [0.000, 0.475, 0.525]], np.float32)

_NAMED = {"red": (255, 0, 0), "blue": (0, 0, 255),
          "green": (0, 255, 0), "yellow": (255, 255, 0)}


def _rgb(colour):
    if colour in _NAMED:
        return np.array(_NAMED[colour], np.float32)
    text = colour.lstrip("#")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


def _worst_pair(palette_name, simulation):
    """Separation of the two outlines a reader would find hardest to tell apart."""
    colours = [_rgb(c) for c in outline_palette_colours(palette_name).values()]
    return min(float(np.linalg.norm(simulation @ a - simulation @ b))
               for a, b in itertools.combinations(colours, 2))


# ---------------------------------------------------------------------------
# The defect, measured rather than asserted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("simulation,name", [(DEUTERANOPE, "deuteranope"),
                                             (PROTANOPE, "protanope"),
                                             (TRITANOPE, "tritanope")])
def test_the_colourblind_palette_beats_the_default_under_every_deficiency(
        simulation, name):
    default = _worst_pair("default", simulation)
    safe = _worst_pair("colourblind", simulation)
    assert safe > default * 1.5, (
        f"{name}: colourblind {safe:.0f} vs default {default:.0f}")


def test_the_default_is_invisible_to_a_deuteranope_and_this_is_why():
    """cell=red beside pathogen=green, in the figure that goes into a paper."""
    assert _worst_pair("default", DEUTERANOPE) < 40
    assert _worst_pair("colourblind", DEUTERANOPE) > 100


def test_the_safe_palette_stays_usable_under_normal_vision():
    """It buys deficiency separation with normal separation, not for free."""
    identity = np.eye(3, dtype=np.float32)
    assert _worst_pair("colourblind", identity) > 100


# ---------------------------------------------------------------------------
# It must not change a figure anybody has already made
# ---------------------------------------------------------------------------

def test_default_is_the_default_and_is_unchanged():
    assert outline_palette_colours("default") == {
        "cell": "red", "nucleus": "blue",
        "pathogen": "green", "organelle": "yellow"}
    assert outline_palette_colours(None) == outline_palette_colours("default")


def test_an_unknown_palette_draws_the_historic_colours_rather_than_raising():
    """A figure in the old colours beats a pipeline that stops at plotting."""
    assert outline_palette_colours("chartreuse") == OUTLINE_PALETTES["default"]
    assert outline_palette_colours(17) == OUTLINE_PALETTES["default"]


def test_the_caller_cannot_mutate_the_shipped_palette():
    got = outline_palette_colours("default")
    got["cell"] = "chartreuse"
    assert OUTLINE_PALETTES["default"]["cell"] == "red"


def test_case_and_whitespace_do_not_silently_fall_back():
    assert (outline_palette_colours("  Colourblind ")
            == OUTLINE_PALETTES["colourblind"])


def test_every_palette_names_every_object():
    for name, mapping in OUTLINE_PALETTES.items():
        assert set(mapping) == {"cell", "nucleus", "pathogen", "organelle"}, name


# ---------------------------------------------------------------------------
# The setting is reachable
# ---------------------------------------------------------------------------

def test_the_setting_is_registered_with_a_type_and_a_tooltip():
    from spacr.settings import expected_types, tooltips

    assert expected_types["outline_palette"] is str
    tip = tooltips["outline_palette"]
    # The tooltip carries the measurement, so the choice can be argued with.
    assert "27" in tip and "142" in tip


def test_the_overlay_function_accepts_it():
    import inspect

    from spacr.plot import plot_image_mask_overlay

    params = inspect.signature(plot_image_mask_overlay).parameters
    assert params["outline_palette"].default == "default"


def test_the_pipeline_call_sites_no_longer_pass_a_parameter_that_does_not_exist():
    """`normalize=True` raised TypeError on every call, swallowed by a bare
    except, so this branch printed a failure instead of drawing an overlay."""
    import inspect

    from spacr import submodules
    from spacr.plot import plot_image_mask_overlay

    accepted = set(inspect.signature(plot_image_mask_overlay).parameters)
    source = inspect.getsource(submodules)
    call = source[source.index("plot_image_mask_overlay(file_path"):]
    call = call[:call.index("))") + 2]
    for keyword in ("normalize=",):
        assert keyword not in call
    assert "normalize" not in accepted
