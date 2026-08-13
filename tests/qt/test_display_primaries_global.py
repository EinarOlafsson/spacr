"""The colour-vision preference reaches every image spaCR draws.

Instruction 89 asked for a colourblind mode that applies to images being
shown, and set three conditions on it: it is a GLOBAL preference rather than
a per-screen toggle, it never touches the data, and the viewer says which
channel is drawn in which colour.

The preference itself already existed -- ``get_color_blind_mode`` and an
Okabe-Ito palette in :mod:`spacr.qt.preferences` -- and nothing read it. What
is tested here is the wiring: that one setting reaches the mask overlay, the
crop grid and Annotate; that the outlines drawn ON TOP of a recoloured image
are NOT themselves recoloured; and that ``auto`` stops handing two
compartments a pair the user cannot tell apart.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.live_preview import (
    overlay_masks,
    random_outline_colour,
    safe_outline_palette,
)
from spacr.qt.widgets.preview_contract import PRIMARY_NOTES, LivePreviewContract


@pytest.fixture
def cb_mode(monkeypatch):
    """Set the stored colour-vision mode without touching the real QSettings."""
    def _set(mode: str):
        import spacr.qt.preferences as P
        monkeypatch.setattr(P, "get_color_blind_mode", lambda: mode)
    return _set


# ---------------------------------------------------------------------------
# The bridge between the two vocabularies
# ---------------------------------------------------------------------------

def test_a_condition_maps_onto_a_rendering(cb_mode):
    """The preference names a condition; crops names a rendering."""
    from spacr.crops import DISPLAY_PRIMARIES
    from spacr.qt.preferences import image_display_primaries

    for stored, expected in (("off", "rgb"),
                             ("deuteranopia", "deuteranope"),
                             ("protanopia", "protanope"),
                             ("tritanopia", "tritanope")):
        cb_mode(stored)
        assert image_display_primaries() == expected
        assert expected in DISPLAY_PRIMARIES


def test_cmy_is_never_reached_by_having_a_deficiency(cb_mode):
    """CMY measured WORSE than RGB under deuteranope simulation.

    It ships because it is the publishing convention, so it must be a
    choice somebody makes, never a consequence of their vision.
    """
    from spacr.qt.preferences import _CB_MODE_TO_PRIMARIES

    assert "cmy" not in set(_CB_MODE_TO_PRIMARIES.values())


def test_an_unknown_stored_mode_falls_back_to_rgb(cb_mode):
    from spacr.qt.preferences import image_display_primaries

    cb_mode("achromatopsia_from_a_future_release")
    assert image_display_primaries() == "rgb"


# ---------------------------------------------------------------------------
# The overlay: image recoloured, outlines not
# ---------------------------------------------------------------------------

def _one_object():
    image = np.zeros((16, 16, 3), np.uint8)
    image[..., 0] = 200          # a red stain
    mask = np.zeros((16, 16), np.int32)
    mask[4:12, 4:12] = 1
    return image, {"cell": mask}


def test_the_image_is_recoloured():
    image, masks = _one_object()
    # normalise=False: a uniform field percentile-stretches to zero, and
    # what is under test is the primaries matrix, not the stretch.
    plain = overlay_masks(image, {}, primaries="rgb", normalise=False)
    shifted = overlay_masks(image, {}, primaries="deuteranope",
                            normalise=False)
    # Red moves into yellow: the green plane gains, red is kept.
    assert plain[0, 0].tolist() == [200, 0, 0]
    assert shifted[0, 0].tolist() == [200, 200, 0]


def test_the_outline_keeps_the_colour_the_user_asked_for():
    """The reason `primaries` is on overlay_masks and not on the pixmap call.

    Primaries are a channel-to-colour mapping and an outline is not a
    channel. Putting it through the same matrix would answer a request for
    a red outline with a yellow one.
    """
    image, masks = _one_object()
    drawn = overlay_masks(image, masks, outline_rgb=(240, 60, 60),
                          primaries="deuteranope", normalise=False)
    boundary = drawn[4, 4]
    assert boundary.tolist() == [240, 60, 60]


def test_the_transform_does_not_touch_its_input():
    image, masks = _one_object()
    before = image.copy()
    overlay_masks(image, masks, primaries="cmy")
    assert np.array_equal(image, before)


def test_rgb_is_the_identity():
    image, masks = _one_object()
    assert np.array_equal(overlay_masks(image, masks, primaries="rgb"),
                          overlay_masks(image, masks))


# ---------------------------------------------------------------------------
# Safe outline colours
# ---------------------------------------------------------------------------

def test_no_palette_when_the_preference_is_off(cb_mode):
    cb_mode("off")
    assert safe_outline_palette() is None


def test_a_deficiency_gets_the_okabe_ito_set(cb_mode):
    cb_mode("deuteranopia")
    palette = safe_outline_palette()
    assert palette is not None and len(palette) >= 6
    assert (0x00, 0x72, 0xB2) in palette          # Okabe-Ito blue
    for triple in palette:
        assert len(triple) == 3
        assert all(0 <= channel <= 255 for channel in triple)


def test_random_outline_colour_draws_from_the_palette_when_given_one():
    import random

    palette = [(1, 2, 3), (4, 5, 6)]
    drawn = {random_outline_colour(random.Random(seed), palette)
             for seed in range(20)}
    assert drawn <= set(palette)


def test_auto_deals_the_palette_rather_than_drawing_from_it(cb_mode, qtbot):
    """Independent draws still collide; dealing cannot.

    Two compartments sharing one colour is the exact failure the safe
    palette exists to prevent, so the colours are dealt without replacement.
    """
    from spacr.qt.widgets.live_preview import COMPARTMENTS, LivePreviewPanel

    cb_mode("deuteranopia")
    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._roll_auto_outline_colours()
    colours = [panel._auto_outline_colours[c] for c in COMPARTMENTS]
    assert len(set(colours)) == len(colours)
    assert set(colours) <= set(safe_outline_palette())


def test_a_preferences_module_that_raises_leaves_auto_random(monkeypatch):
    """Never crash the renderer over a palette."""
    import spacr.qt.preferences as P

    def boom():
        raise RuntimeError("no QSettings here")

    monkeypatch.setattr(P, "get_color_blind_mode", boom)
    assert safe_outline_palette() is None


def test_a_malformed_palette_entry_is_skipped(monkeypatch):
    import spacr.qt.preferences as P

    monkeypatch.setattr(P, "get_color_blind_mode", lambda: "deuteranopia")
    monkeypatch.setattr(P, "color_blind_categorical_palette",
                        lambda: ["#0072B2", "not-a-colour", "#GGGGGG"])
    assert safe_outline_palette() == [(0x00, 0x72, 0xB2)]


def test_a_palette_of_nothing_usable_falls_back_to_random(monkeypatch):
    import spacr.qt.preferences as P

    monkeypatch.setattr(P, "get_color_blind_mode", lambda: "deuteranopia")
    monkeypatch.setattr(P, "color_blind_categorical_palette", lambda: ["nope"])
    assert safe_outline_palette() is None


# ---------------------------------------------------------------------------
# Saying so on screen
# ---------------------------------------------------------------------------

class _Label:
    def __init__(self):
        self._text = ""

    def setText(self, text):
        self._text = str(text)

    def text(self):
        return self._text


class _Panel(LivePreviewContract):
    def __init__(self):
        self._status = _Label()


def test_the_status_line_names_the_mapping(cb_mode):
    cb_mode("tritanopia")
    panel = _Panel()
    panel.set_preview_status("Segmented 41 cells.")
    assert "Segmented 41 cells." in panel.preview_status()
    assert PRIMARY_NOTES["tritanope"] in panel.preview_status()


def test_the_note_is_not_repeated(cb_mode):
    cb_mode("deuteranopia")
    panel = _Panel()
    panel.set_preview_status("Done.")
    once = panel.preview_status()
    panel.set_preview_status(once)
    assert panel.preview_status().count("drawn as yellow") == 1


def test_plain_rgb_changes_nothing_at_all(cb_mode):
    cb_mode("off")
    panel = _Panel()
    panel.set_preview_status("Done.")
    assert panel.preview_status() == "Done."
    assert panel.display_primaries_note() == ""


def test_every_non_rgb_mode_has_a_sentence():
    from spacr.crops import DISPLAY_PRIMARIES

    assert set(PRIMARY_NOTES) == set(DISPLAY_PRIMARIES) - {"rgb"}


def test_a_mode_with_no_sentence_still_says_something(cb_mode, monkeypatch):
    panel = _Panel()
    monkeypatch.setattr(type(panel), "display_primaries",
                        lambda self: "some_future_mode", raising=False)
    assert "some_future_mode" in panel.display_primaries_note()


def test_without_qt_preferences_the_view_draws_plain_rgb(monkeypatch):
    """The pipeline-safe answer: untransformed, never a failure to draw."""
    import spacr.qt.preferences as P

    def boom():
        raise RuntimeError("no QSettings")

    monkeypatch.setattr(P, "image_display_primaries", boom)
    assert _Panel().display_primaries() == "rgb"
