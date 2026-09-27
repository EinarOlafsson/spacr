"""Item 288: organelle slot names and methods at the edges of the vocabulary.

* A detector no morphology lists is described as such, for the picker.
* ``organellea`` is never a slot -- slot 1 is the bare word -- and asking
  for its number is refused.
* A settings key whose slot prefix is not a slot belongs to no slot.
* A hand-edited file naming a slot past the last one (three letters) does
  not raise; the implied count is clamped to what can exist.
"""
from __future__ import annotations

import pytest

from spacr import organelle_types as ot


def test_a_method_no_morphology_lists_says_so():
    assert ot.method_guidance("tea_leaves") == \
        "No organelle morphology lists tea_leaves as a detector."


def test_organellea_is_not_a_slot():
    with pytest.raises(ValueError) as excinfo:
        ot.organelle_number("organellea")
    assert "is not an organelle role" in str(excinfo.value)


def test_a_key_under_a_prefix_that_is_not_a_slot_belongs_to_no_slot():
    assert ot.organelle_role_of("organellea_channel") is None
    assert ot.organelle_role_of("organelleb_channel") == "organelleb"


def test_a_slot_past_the_last_one_is_clamped_to_the_last():
    assert ot.organelle_count({"organelleaaa_channel": 1}) == \
        ot.MAX_ORGANELLES
