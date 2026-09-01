"""The live preview segments organelles the way the run does.

`organelle_method` has eight values and only one is `cellpose`. The preview ran
Cellpose unconditionally, so seven of the eight could not be previewed at all
-- which is most of the fifty-odd organelle settings having no effect on
anything the user could see.

Reported 2026-09-01: "presently there is no way to live preview the organell
settings except for the cellpose model".

The pipeline's own `_segment_single_image` is called rather than a
reimplementation, so a preview that disagrees with the run is a bug in one
place instead of a difference between two.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.live_preview import _classical_organelle_mask


@pytest.fixture
def spots():
    """Two bright square spots on a dark field."""
    image = np.zeros((64, 64), dtype=np.uint16)
    image[20:26, 20:26] = 3000
    image[40:44, 40:44] = 2500
    return image


def test_a_classical_method_actually_segments(spots):
    mask = _classical_organelle_mask(
        spots, "organelle",
        {"organelle_morphology": "spots", "organelle_method": "otsu"})
    assert mask.dtype == np.int32
    assert mask.max() == 2, "both spots should be found and labelled apart"


def test_an_empty_field_yields_no_objects():
    """The control. Without it, a routine that labelled everything -- or
    nothing -- would satisfy the test above just as well."""
    mask = _classical_organelle_mask(
        np.zeros((64, 64), dtype=np.uint16), "organelle",
        {"organelle_morphology": "spots", "organelle_method": "otsu"})
    assert mask.max() == 0


def test_the_method_changes_the_answer(spots):
    """Proof the setting is REACHING the segmentation rather than being
    accepted and ignored, which is the defect this whole change is about."""
    otsu = _classical_organelle_mask(
        spots, "organelle",
        {"organelle_morphology": "spots", "organelle_method": "otsu",
         "organelle_tophat_radius": 0})
    dog = _classical_organelle_mask(
        spots, "organelle",
        {"organelle_morphology": "spots", "organelle_method": "dog",
         "organelle_tophat_radius": 0,
         "organelle_dog_sigma_low": 1.0, "organelle_dog_sigma_high": 8.0})
    assert not np.array_equal(otsu, dog)


def test_the_second_slot_previews_with_its_own_settings(spots):
    """Slot 2's keys are `organelleb_*`; the pipeline function reads
    `organelle_*`. Without the remap slot 2 would preview with slot 1's
    settings and quietly show the wrong answer."""
    mask = _classical_organelle_mask(
        spots, "organelleb",
        {"organelleb_morphology": "spots", "organelleb_method": "otsu"})
    assert mask.max() == 2


def test_a_slot_key_beats_the_generic_one(spots):
    """Both prefixes present: the SLOT's value must win, or a second organelle
    silently inherits the first's method."""
    settings = {
        "organelle_morphology": "spots", "organelle_method": "otsu",
        "organelle_tophat_radius": 0,
        "organelleb_morphology": "spots", "organelleb_method": "dog",
        "organelleb_tophat_radius": 0,
        "organelleb_dog_sigma_low": 1.0, "organelleb_dog_sigma_high": 8.0,
    }
    as_slot_two = _classical_organelle_mask(spots, "organelleb", settings)
    as_slot_one = _classical_organelle_mask(spots, "organelle", settings)
    assert not np.array_equal(as_slot_one, as_slot_two)


def test_missing_settings_are_defaulted_rather_than_raising(spots):
    """The classical routines index their settings directly, so a missing key
    is a KeyError inside a worker thread -- a preview that dies with no
    message rather than one that shows something."""
    mask = _classical_organelle_mask(
        spots, "organelle", {"organelle_morphology": "spots"})
    assert mask.shape == spots.shape


def test_cellpose_is_left_to_cellpose():
    """A source check: the branch must exclude 'cellpose', or the model path
    -- and the flow view that comes with it -- becomes unreachable."""
    from pathlib import Path

    import spacr.qt.widgets.live_preview as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'method != "cellpose"' in source
    assert 'obj.startswith("organelle") and method != "cellpose"' in source
