"""A settings panel must show the default the MODULE declares.

Instruction 77, finding 20. ``convert_settings_dict_for_gui`` returned
``special_cases[key]`` verbatim for ~35 keys, throwing away the value the
module's own ``set_default_*`` factory had just produced. That table is one
row per key for the whole application, so a single shared default reached
every module that declares the key.

The Qt panel repaired this downstream in
``settings_model._widget_for``. The Tk path did not, and on Tk the value is
not merely displayed -- ``check_settings`` builds the run dict out of the
widgets, so the table's value is what RUNS. Pressing Run on a freshly opened
Tk module without touching anything therefore submitted settings that module
never asked for.

The assertions below go through the real defaults factories rather than a
hand-written dict, because the defect is precisely a disagreement between
the factory and the table: a fixture stating what the factory "should"
return could not catch it.
"""
from __future__ import annotations

import pytest

from spacr.gui_utils import convert_settings_dict_for_gui
from spacr.settings import (
    deep_spacr_defaults,
    get_default_generate_activation_map_settings,
    get_identify_masks_finetune_default_settings,
    set_default_settings_preprocess_generate_masks,
)


#: (module label, factory, key) for every case the finding measured.
MEASURED = [
    ("Cellpose Masks", get_identify_masks_finetune_default_settings,
     "channels"),
    ("Classify", deep_spacr_defaults, "model_type"),
    ("Activation Maps", get_default_generate_activation_map_settings,
     "model_type"),
    ("Activation Maps", get_default_generate_activation_map_settings,
     "channels"),
]


def _normalise(value):
    """The two spellings of one list -- "[0, 0]" and "[0,0]" -- are equal."""
    return str(value).replace(" ", "")


@pytest.mark.parametrize("label,factory,key", MEASURED,
                         ids=[f"{lab}:{k}" for lab, _f, k in MEASURED])
def test_the_panel_preselects_the_factorys_own_value(label, factory, key):
    declared = factory({})[key]
    var_type, options, shown = convert_settings_dict_for_gui(
        factory({}))[key]

    assert var_type == "combo"
    assert _normalise(shown) == _normalise(declared), (
        f"Tk {label} would run {shown!r} where {key} is declared "
        f"{declared!r}")


@pytest.mark.parametrize("label,factory,key", MEASURED,
                         ids=[f"{lab}:{k}" for lab, _f, k in MEASURED])
def test_the_declared_value_is_selectable_at_all(label, factory, key):
    """It has to be IN the list, or the widget cannot hold it.

    ``get_default_generate_activation_map_settings`` declares
    ``channels=[1, 2, 3]`` and the curated channel list never offered it, so
    even a panel that tried to preselect the real default had nowhere to put
    it. Offering it is the same answer the Qt side reached.
    """
    declared = factory({})[key]
    _var_type, options, _shown = convert_settings_dict_for_gui(
        factory({}))[key]

    assert any(_normalise(opt) == _normalise(declared) for opt in options)


def test_the_curated_options_are_still_offered():
    """Preselecting the module's value must not shorten the menu.

    The point is which entry starts selected, not which entries exist: a fix
    that replaced the list would take away every other choice the user has.
    """
    _t, options, _d = convert_settings_dict_for_gui(
        get_identify_masks_finetune_default_settings({}))["channels"]

    for expected in ("[0,1,2,3]", "[0,1,2]", "[0,1]", "[0]"):
        assert expected in options


def test_a_module_that_agrees_with_the_table_is_unchanged():
    """The ordinary case must not move.

    Mask declares exactly what the table does, and this is the assertion
    that a fix aimed at the disagreeing modules did not perturb the
    agreeing ones.
    """
    settings = set_default_settings_preprocess_generate_masks({})
    out = convert_settings_dict_for_gui(settings)

    assert out["metadata_type"][2] == "cellvoyager"
    assert _normalise(out["channels"][2]) == _normalise(settings["channels"])


def test_a_value_the_module_does_not_declare_keeps_the_shared_default():
    """An empty or absent value falls back rather than inventing an option."""
    out = convert_settings_dict_for_gui({"metadata_type": None})
    assert out["metadata_type"][2] == "cellvoyager"


def test_a_none_that_is_a_real_choice_is_preselected_not_replaced():
    """``cov_type`` and ``transform`` list ``None`` as a genuine option.

    Falling back to the shared default for every None would be wrong for
    these: None is the answer, not the absence of one.
    """
    out = convert_settings_dict_for_gui({"transform": None})
    var_type, options, shown = out["transform"]
    assert var_type == "combo"
    assert None in options
    assert shown is None
