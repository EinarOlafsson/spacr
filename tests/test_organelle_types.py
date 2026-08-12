"""One choice a biologist recognises, in front of fifty-three they do not.

`organelle` is the most over-configured object class in spaCR: 53 settings
reached the mask pipeline under a single "Organelle" heading, and a user who
knew they were imaging lysosomes had to scroll past `organelle_ridge_sigmas`
to find the diameter.

THE PART THAT HAD TO BE VERIFIED RATHER THAN ASSUMED. The request offered
nine cell-biology categories and asked, in the same breath, to check that
they map onto the four `organelle_morphology` values. They do not. "Vesicular"
is a cell-biology category -- a membrane-bound compartment that carries cargo
-- and spots/ring/network/irregular is an image-appearance category: what the
segmentation has to find. They do not nest, because the same biological family
looks different at different sizes. A 200 nm transport vesicle is a
diffraction-limited dot; a 2 um vacuole is a visible ring. Both are Vesicular,
and both are on the maintainer's own list.

So the mapping is (type, size) -> morphology, and a preset that hard-coded one
morphology per type would be wrong for half the entries in that list --
silently. The user picks Vesicular, gets a spot detector, and their lysosomes
come out as rings of holes.
"""

import pytest

from spacr.object import _validate_organelle_settings
from spacr.organelle_types import (BASIC_SETTINGS, DEFAULT_TYPE,
                                   LEGAL_METHODS, ORGANELLE_TYPES,
                                   RING_RESOLVABLE_PX, TYPE_ORDER,
                                   apply_preset, is_basic, preset_for,
                                   resolve_type)
from spacr.settings import _set_organelle_defaults, categories

NAMED = [t for t in TYPE_ORDER if t != "custom"]


# ---------------------------------------------------------------------------
# nothing a preset ships may be rejected at run time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", TYPE_ORDER)
@pytest.mark.parametrize("diameter", [1, 5, 14, 15, 16, 30, 100, 1000])
def test_every_preset_passes_the_real_validator(name, diameter):
    """`_validate_organelle_settings` raises before any image is loaded.

    Checked against the VALIDATOR ITSELF, not the copy of its table in
    `organelle_types`, because a preset that ships an illegal pair fails at
    the start of a run rather than here.
    """
    preset = preset_for(name, diameter)
    morphology = preset.get("organelle_morphology")
    if morphology is None:
        assert name == "custom", (name, preset)
        assert preset == {}, preset
        return
    method = preset["organelle_method"]
    # The validator returns None and raises on a bad pair, so the assertion
    # that carries weight is the one about WHAT was shipped -- a legal pair
    # for the morphology that this diameter selected.
    assert _validate_organelle_settings(morphology, method) is None
    assert morphology in LEGAL_METHODS
    assert method in LEGAL_METHODS[morphology], (name, diameter, morphology,
                                                 method)


def test_the_mirrored_table_still_matches_the_validator():
    """The duplication is deliberate; drift in it is not."""
    import inspect

    source = inspect.getsource(_validate_organelle_settings)
    for morphology, methods in LEGAL_METHODS.items():
        assert f"'{morphology}'" in source
        for method in methods:
            assert f"'{method}'" in source, (morphology, method)


# ---------------------------------------------------------------------------
# size decides, and the two rows where it does are the point
# ---------------------------------------------------------------------------

def test_vesicular_is_a_dot_when_small_and_a_ring_when_large():
    """The row that proves type alone cannot pick the detector."""
    small = preset_for("vesicular", RING_RESOLVABLE_PX - 1)
    large = preset_for("vesicular", RING_RESOLVABLE_PX)

    assert small["organelle_morphology"] == "spots"
    assert large["organelle_morphology"] == "ring"


def test_spherical_splits_too():
    assert preset_for("spherical", 5)["organelle_morphology"] == "spots"
    assert preset_for("spherical", 60)["organelle_morphology"] == "irregular"


@pytest.mark.parametrize("name", ["punctate", "filamentous", "tubular",
                                  "reticular", "cisternal", "toroidal",
                                  "crescent"])
def test_the_types_that_do_not_split_are_stable_across_size(name):
    sizes = {preset_for(name, d)["organelle_morphology"]
             for d in (2, 10, 15, 40, 400)}
    assert len(sizes) == 1, (name, sizes)


def test_the_method_stays_legal_when_size_flips_the_morphology():
    """A preset recommending 'ridge' for a type that can become a ring would
    raise the moment the user's diameter tipped it over."""
    for diameter in (5, RING_RESOLVABLE_PX, 200):
        preset = preset_for("vesicular", diameter)
        assert preset["organelle_method"] in LEGAL_METHODS[
            preset["organelle_morphology"]]


def test_an_unknown_diameter_picks_the_small_reading():
    """None means "not stated". The dot is the safer default: asking for a
    ring detector on a solid object finds holes that are not there."""
    assert preset_for("vesicular", None)["organelle_morphology"] == "spots"


# ---------------------------------------------------------------------------
# custom changes nothing, which is what keeps old files meaning what they meant
# ---------------------------------------------------------------------------

def test_the_default_type_recommends_nothing():
    assert DEFAULT_TYPE == "custom"
    assert preset_for("custom", 30) == {}


def test_a_settings_file_written_before_this_existed_is_unchanged():
    """The acceptance criterion: no existing settings CSV changes meaning."""
    before = _set_organelle_defaults({})
    assert before["organelle_type"] == "custom"
    assert before["organelle_morphology"] == "spots"
    assert before["organelle_method"] == "otsu"


def test_an_unknown_type_raises_and_names_the_known_ones():
    """Falling back to 'custom' would mean a typo silently segments with
    different settings than the user asked for."""
    with pytest.raises(ValueError) as raised:
        resolve_type("mitochondria")
    message = str(raised.value)
    assert "mitochondria" in message
    assert "punctate" in message


# ---------------------------------------------------------------------------
# preset, do not override
# ---------------------------------------------------------------------------

def test_naming_a_type_gives_you_its_recommendation():
    out = _set_organelle_defaults({"organelle_type": "punctate"})
    assert out["organelle_morphology"] == "spots"
    assert out["organelle_method"] == "log"


def test_a_value_you_set_yourself_is_never_overwritten():
    """"A user who then changes organelle_method keeps that change; the type
    does not silently reassert itself"."""
    out = _set_organelle_defaults({"organelle_type": "punctate",
                                   "organelle_method": "adaptive"})
    assert out["organelle_method"] == "adaptive"


def test_your_morphology_wins_over_the_types():
    out = _set_organelle_defaults({"organelle_type": "punctate",
                                   "organelle_morphology": "irregular",
                                   "organelle_method": "otsu"})
    assert out["organelle_morphology"] == "irregular"


def test_the_precedence_is_user_then_preset_then_default():
    """The ordering bug this had first: running the preset AFTER the bare
    defaults meant organelle_method was already 'otsu', the preset saw a set
    key, kept it, and naming a type did nothing at all."""
    default_only = _set_organelle_defaults({})
    preset_only = _set_organelle_defaults({"organelle_type": "filamentous"})
    user_wins = _set_organelle_defaults({"organelle_type": "filamentous",
                                         "organelle_method": "otsu"})

    assert default_only["organelle_method"] == "otsu"     # the bare default
    assert preset_only["organelle_method"] == "ridge"     # the preset
    assert user_wins["organelle_method"] == "otsu"        # the user


def test_apply_preset_does_not_mutate_the_caller():
    original = {"organelle_type": "punctate"}
    apply_preset(original)
    assert original == {"organelle_type": "punctate"}


def test_the_defaults_helper_still_fills_in_place():
    """Every caller does `_set_organelle_defaults(settings)` WITHOUT taking
    the return value.

    Rebinding the local name instead of updating in place silently threw
    forty defaults onto a copy: the mask panel's 53 organelle settings became
    13, and nothing raised.
    """
    settings = {}
    _set_organelle_defaults(settings)
    assert len(settings) > 40, len(settings)
    assert "organelle_morphology" in settings


def test_the_console_explains_what_the_preset_did(capsys):
    """A preset is only better than 53 knobs if the user can see what it set."""
    apply_preset({"organelle_type": "reticular", "organelle_diameter": 20},
                 explain=True)
    printed = capsys.readouterr().out
    assert "Reticular" in printed
    assert "organelle_morphology" in printed
    assert "hysteresis" in printed


def test_the_explanation_marks_what_it_did_not_touch(capsys):
    apply_preset({"organelle_type": "punctate",
                  "organelle_method": "adaptive"}, explain=True)
    printed = capsys.readouterr().out
    assert "KEPT" in printed and "adaptive" in printed


# ---------------------------------------------------------------------------
# the table is honest about what it cannot do
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["toroidal", "crescent", "cisternal"])
def test_the_types_with_no_dedicated_detector_say_so(name):
    """"Crescent" and "Toroidal" have no detector today and reduce to 'ring'
    plus a shape filter, which is an honest answer."""
    caveat = ORGANELLE_TYPES[name].caveat
    assert caveat, name
    assert "NO " in caveat, caveat


def test_vesicular_warns_that_lysosomes_are_the_exception():
    """Lysosomes are on the maintainer's Vesicular list and image as solid
    blobby -- the one member the preset genuinely cannot serve."""
    assert "LYSOSOME" in ORGANELLE_TYPES["vesicular"].caveat.upper()


@pytest.mark.parametrize("name", NAMED)
def test_every_named_type_lists_the_structures_it_covers(name):
    assert ORGANELLE_TYPES[name].members, name


def test_every_maintainer_category_is_present():
    for name in ("vesicular", "filamentous", "punctate", "tubular",
                 "cisternal", "reticular", "spherical", "toroidal",
                 "crescent"):
        assert name in ORGANELLE_TYPES


# ---------------------------------------------------------------------------
# the category split, as a number
# ---------------------------------------------------------------------------

def test_the_visible_count_went_down_and_this_is_the_number():
    """53 settings under one heading became 3 visible by default.

    Instruction 72 took it to 6; instruction 73 then pulled the shared
    families -- object filtration and intensity handling -- out to headings
    of their own, because `organelle_min_size` and `cell_min_size` are one
    decision applied to two objects rather than two unrelated knobs. What is
    left under Organelle is the channel-shaped choices only.
    """
    assert len(categories["Organelle"]) == 3
    assert len(categories["Organelle advanced"]) == 35
    # Still 53 + organelle_type, just spread across four headings now.
    total = sum(len(categories[c]) for c in
                ("Organelle", "Organelle advanced",
                 "Object filtration", "Intensity handling"))
    assert total >= 54


def test_everything_is_still_reachable():
    """MOVED, NOT HIDDEN -- a setting that leaves the panel while staying in
    the settings dict is how a run gets a value nobody can see."""
    offered = set()
    for heading in ("Organelle", "Organelle advanced",
                    "Object filtration", "Intensity handling"):
        offered |= set(categories.get(heading, ()))
    defaults = {k for k in _set_organelle_defaults({})
                if k.startswith("organelle_")}
    assert defaults - offered <= {"organelle_channel", "organelle_mask_dim"}


@pytest.mark.parametrize("key", BASIC_SETTINGS)
def test_the_basic_set_is_what_a_biologist_recognises(key):
    assert is_basic(key)
    assert not key.startswith("organelle_ridge")
    assert not key.startswith("organelle_hysteresis")


def test_no_detection_parameter_leaked_into_the_basic_heading():
    for key in categories["Organelle"]:
        for jargon in ("ridge", "hysteresis", "sigma", "adaptive", "tophat",
                       "unet", "clahe", "watershed", "morph_radius"):
            assert jargon not in key, key


# ---------------------------------------------------------------------------
# the measure-stage gate, and the regression instruction 72 caused there
# ---------------------------------------------------------------------------
# `_spatial_organelle_eligible` read `organelle_type` FIRST and tested the raw
# string for membership of {'network','reticular','cisternal'}. Adding
# organelle_type broke it two ways at once, silently:
#
#   * the new default 'custom' is not in that set, so a run that said
#     organelle_morphology='network' and meant it had its explicit choice
#     SHADOWED by a default it never set;
#   * 'filamentous' and 'tubular' are not in the set either, though the preset
#     maps both to `network`.
#
# Both turned the spatial block back on for a single connected network, whose
# neighbour statistics are not a measurement of anything. The type is now
# resolved to a morphology before the test.

from spacr.measure import (_morphology_of_organelle_type,
                           _spatial_organelle_eligible)


@pytest.mark.parametrize("settings", [
    {"organelle_morphology": "network"},
    {"organelle_type": "custom", "organelle_morphology": "network"},
    {"organelle_type": "filamentous"},
    {"organelle_type": "tubular"},
    {"organelle_type": "reticular"},
])
def test_a_connected_network_never_gets_neighbour_statistics(settings):
    assert _spatial_organelle_eligible(settings) is False, settings


@pytest.mark.parametrize("settings", [
    {"organelle_morphology": "spots"},
    {"organelle_type": "punctate"},
    {"organelle_type": "vesicular", "organelle_diameter": 8},
    {"organelle_type": "vesicular", "organelle_diameter": 40},
    {"organelle_type": "spherical", "organelle_diameter": 40},
    {"organelle_type": "toroidal"},
])
def test_separable_objects_still_get_them(settings):
    assert _spatial_organelle_eligible(settings) is True, settings


def test_the_default_type_does_not_shadow_an_explicit_morphology():
    """The regression, stated as the thing that must hold.

    Every run now carries organelle_type='custom' whether the user chose it
    or not, so 'custom' must have NO opinion here.
    """
    from spacr.settings import _set_organelle_defaults

    settings = _set_organelle_defaults({"organelle_morphology": "network",
                                        "organelle_method": "ridge"})
    assert settings["organelle_type"] == "custom"
    assert _spatial_organelle_eligible(settings) is False


def test_custom_and_unknown_types_defer_to_the_morphology():
    assert _morphology_of_organelle_type({"organelle_type": "custom"}) is None
    assert _morphology_of_organelle_type({}) is None
    assert _morphology_of_organelle_type({"organelle_type": "nonsense"}) is None


def test_the_type_is_resolved_not_string_matched():
    """'filamentous' is not the word 'network', and that was the bug."""
    assert _morphology_of_organelle_type(
        {"organelle_type": "filamentous"}) == "network"
    assert _morphology_of_organelle_type(
        {"organelle_type": "tubular"}) == "network"


def test_a_measure_only_run_still_assumes_the_shipped_default(capsys):
    """Neither key reaches a measure-only run; the assumption is printed."""
    import spacr.measure as M

    M._SPATIAL_ORGANELLE_ASSUMED = False
    assert _spatial_organelle_eligible({}) is True
    assert "assumes the shipped default" in capsys.readouterr().out
