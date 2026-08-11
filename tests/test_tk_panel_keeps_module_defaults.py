"""The Tk panel showed a canned value where the module declared its own.

`convert_settings_dict_for_gui` replaces the caller's value whenever the key
is in its ``special_cases`` table::

    if key in special_cases:
        variables[key] = special_cases[key]      # the supplied value is dropped

That table is ONE ROW PER KEY FOR THE WHOLE APP, so every module gets the
same answer regardless of what it declared. Qt corrects this locally in
`SettingsWidgets._make_widget`; Tk never got the equivalent, so opening a Tk
panel and pressing Run without touching anything ran with values the module
never asked for:

    Cellpose Masks  channels                 [0, 0]     shown as '[0,1,2,3]'
    Mask / Measure  summarize_organelles_by  'cell'     shown as None
    Classify        model_type               'maxvit_t' shown as 'resnet50'

The last is a different network architecture. The first segments on four
channels instead of one. The second is the key whose mangling the golden
pipeline already records as an xfail.
"""

import pytest

import spacr.settings as S
from spacr.gui_core import _restore_module_defaults
from spacr.gui_utils import convert_settings_dict_for_gui


FACTORIES = {
    "cellpose_masks": S.get_identify_masks_finetune_default_settings,
    "mask": S.set_default_settings_preprocess_generate_masks,
    "measure": S.get_measure_crop_settings,
    "classify": S.deep_spacr_defaults,
    "umap": S.set_default_umap_image_settings,
}


def _normal(value):
    return str(value).replace(" ", "")


@pytest.mark.parametrize("module", sorted(FACTORIES))
def test_every_panel_value_is_the_module_s_own_default(module):
    """The whole point of a defaults factory is that it decides."""
    settings = FACTORIES[module](settings={})
    variables = _restore_module_defaults(
        convert_settings_dict_for_gui(settings), settings)

    drift = {
        key: (settings[key], spec[2])
        for key, spec in variables.items()
        if key in settings and isinstance(spec, tuple) and len(spec) == 3
        and _normal(spec[2]) != _normal(settings[key])
    }
    assert not drift, f"{module} panel disagrees with its own defaults: {drift}"


@pytest.mark.parametrize(("module", "key", "declared"), [
    ("cellpose_masks", "channels", [0, 0]),
    ("mask", "summarize_organelles_by", "cell"),
    ("measure", "summarize_organelles_by", "cell"),
    ("classify", "model_type", "maxvit_t"),
])
def test_the_four_that_were_actually_wrong(module, key, declared):
    """Named individually, so a regression says WHICH one came back."""
    settings = FACTORIES[module](settings={})
    assert _normal(settings[key]) == _normal(declared), (
        "the module's declared default moved; update this test deliberately")

    before = convert_settings_dict_for_gui(settings)[key][2]
    after = _restore_module_defaults(
        convert_settings_dict_for_gui(settings), settings)[key][2]

    assert _normal(before) != _normal(declared), (
        "special_cases no longer overrides this key, so the guard is untested")
    assert _normal(after) == _normal(declared)


def test_a_restored_default_is_offered_rather_than_substituted():
    """create_input_field falls back to options[0] for a value it does not
    know, and options[0] for `channels` is '[0,1,2,3,4,5,6,7,8]' -- worse
    than the value it replaced. So the declared default has to be IN the
    list, not merely selected."""
    settings = S.set_default_settings_preprocess_generate_masks(settings={})
    variables = _restore_module_defaults(
        convert_settings_dict_for_gui(settings), settings)

    kind, options, value = variables["summarize_organelles_by"]
    assert any(_normal(o) == _normal(value) for o in options), (
        "the restored default is not among the options the combo offers")


def test_a_spelling_difference_is_not_treated_as_drift():
    """'[0,1,2,3]' and [0, 1, 2, 3] are one choice written two ways.

    Restoring across that difference would add an option differing only in
    whitespace, and grow the list on every call.
    """
    settings = S.get_measure_crop_settings(settings={})
    before = convert_settings_dict_for_gui(settings)
    after = _restore_module_defaults(before, settings)

    assert len(after["channels"][1]) == len(before["channels"][1]), (
        "a whitespace-only difference added a duplicate option")


def test_keys_the_module_never_declared_are_left_alone():
    settings = {"src": "path"}
    variables = {"unrelated": ("entry", [], "kept")}
    assert _restore_module_defaults(variables, settings) == variables
