"""There is one copy of the organelle defaults, and this proves it.

WHAT THIS FILE USED TO BE. Forty organelle defaults were written twice --
`set_default_settings_preprocess_generate_masks` and
`_set_organelle_defaults` -- and they agreed exactly while NOTHING ENFORCED
THAT. A run takes whichever factory it went through and never compares the
two, so a value corrected in one copy and not the other would measure the
same plate two ways, with no error and nothing in the log. This file scanned
both blocks and failed on any divergence.

That was a holding pattern. The duplication is now deleted: the mask factory
calls `_set_organelle_defaults`, and these tests assert the single source
instead of policing two.

Proved behaviour-free before the deletion, not assumed. Every one of the 40
shared keys was compared between the two factories: none missing, none
differing.
"""

import inspect
import re

import pytest

from spacr.settings import (_set_organelle_defaults,
                            set_default_settings_preprocess_generate_masks)


def mask_settings():
    return set_default_settings_preprocess_generate_masks({"src": "/tmp"})


def organelle_keys(settings):
    return {k: v for k, v in settings.items() if k.startswith("organelle_")}


# ---------------------------------------------------------------------------
# one source
# ---------------------------------------------------------------------------

def test_the_mask_factory_calls_the_owner_rather_than_repeating_it():
    source = inspect.getsource(set_default_settings_preprocess_generate_masks)
    assert "_set_organelle_defaults(settings)" in source

    # Only the keys the OWNER declares. The mask factory has organelle keys
    # of its own -- the merge/split and filtering ones -- which were never
    # duplicated and are not this test's business.
    owned = set(organelle_keys(_set_organelle_defaults({})))
    written_out = {k for k in re.findall(
        r"setdefault\(\s*['\"](organelle_[A-Za-z_]+)['\"]", source)
        if k in owned}
    assert not written_out, (
        f"{sorted(written_out)} are hand-written in the mask factory again; "
        f"they belong to _set_organelle_defaults")


def test_every_default_the_owner_declares_reaches_the_mask_factory():
    owned = organelle_keys(_set_organelle_defaults({}))
    produced = organelle_keys(mask_settings())

    assert owned, "the owner declares no organelle defaults at all"
    missing = set(owned) - set(produced)
    assert not missing, f"the mask factory never received {sorted(missing)}"


def test_the_values_are_the_owners_values():
    owned = organelle_keys(_set_organelle_defaults({}))
    produced = organelle_keys(mask_settings())

    differing = {k: (produced[k], owned[k])
                 for k in owned if produced[k] != owned[k]}
    assert not differing, differing


def test_a_caller_supplied_value_still_wins():
    """setdefault semantics: the owner fills gaps, it does not overwrite."""
    chosen = _set_organelle_defaults({"organelle_diameter": 999})
    assert chosen["organelle_diameter"] == 999


# ---------------------------------------------------------------------------
# the keys the mask factory owns itself are untouched
# ---------------------------------------------------------------------------

def test_the_mask_factory_keeps_its_own_organelle_keys():
    """Not every `organelle_*` key belongs to the detection block.

    The merge/split and filtering keys are declared by the mask factory and
    were never duplicated, so collapsing the copies must not have taken them
    with it.
    """
    produced = organelle_keys(mask_settings())
    for key in ("organelle_min_area", "organelle_max_area",
                "organelle_perimeter_fraction"):
        assert key in produced


def test_summarize_organelles_by_survived_the_deletion():
    """It sat inside the deleted block without being an `organelle_*` key."""
    assert mask_settings()["summarize_organelles_by"] == "cell"


# ---------------------------------------------------------------------------
# the difference that is real, and must not be "fixed"
# ---------------------------------------------------------------------------

def test_the_crop_pipelines_zero_min_size_is_left_alone():
    """`get_measure_crop_settings` sets organelle_min_size = 0 deliberately.

    That is a real difference between two PIPELINES, not a drift between two
    COPIES. An unscoped scan for `setdefault('organelle_*')` catches it and
    would fail on correct code -- which is how a guard gets deleted instead
    of heeded.
    """
    from spacr.settings import get_measure_crop_settings

    crop = get_measure_crop_settings({"src": "/tmp"})
    if "organelle_min_size" not in crop:
        pytest.skip("this pipeline no longer declares organelle_min_size")
    assert crop["organelle_min_size"] == 0
    assert _set_organelle_defaults({})["organelle_min_size"] == 10
