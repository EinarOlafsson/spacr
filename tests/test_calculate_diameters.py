"""Item 696, 2026-10-26: "Calculate diameters" under Magnification in Make Masks.

The maintainer: "when i pick a new magnefication number in mask generation,
the default diamiters for cell, nucleus and pathogen should change
acordingly ... there should be a new setting under Magnefication called
something like Calculate diamiters which is a boolean and if true the
diamiters of cell, nucleus and pathogen should change with changing
magnefication. the user may change any of these diamiters after a
magnification calculation is set of course."

The equation is spaCR's own (``_get_diam``, 2025-03-14 commit 5d959ac889,
fitted to the 2024 20x/40x/60x table): cell 2*mag+80, nucleus 0.75*mag+45,
pathogen mag, in pixels. These tests pin it, the switch's settings plumbing,
the rule for a diameter the user set, and that the run gets the values.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.object_settings_table import (_blank_diameters_held_still,
                                         _calculated_diameters,
                                         _calculates_diameters,
                                         _diameter_from_magnification,
                                         _diameter_is_pinned,
                                         _resolved_diameter)
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS


@pytest.mark.parametrize("mag, cell, nucleus, pathogen", [
    (10, 100, 52, 10),
    (20, 120, 60, 20),
    (40, 160, 75, 40),
    (60, 200, 90, 60),
    (100, 280, 120, 100),
])
def test_the_equation_at_each_objective(mag, cell, nucleus, pathogen):
    from spacr.utils import _get_diam

    assert _calculated_diameters(mag) == {"cell_diameter": cell,
                                          "nucleus_diameter": nucleus,
                                          "pathogen_diameter": pathogen}
    for obj, expected in (("cell", cell), ("nucleus", nucleus),
                          ("pathogen", pathogen)):
        assert _diameter_from_magnification(mag, obj) == expected
        assert _get_diam(mag, obj) == expected
        assert _resolved_diameter(None, mag, obj) == float(expected)


def test_without_a_magnification_nothing_is_calculated():
    assert _calculated_diameters(None) == {}
    assert _calculated_diameters("x") == {}


def test_the_switch_is_a_mask_setting_on_by_default():
    from spacr.settings import (categories, expected_types,
                                set_default_settings_preprocess_generate_masks,
                                get_timelapse_settings)

    assert set_default_settings_preprocess_generate_masks({})[
        "calculate_diameters"] is True
    assert get_timelapse_settings()["calculate_diameters"] is True
    assert expected_types["calculate_diameters"] is bool
    general = categories["General"]
    assert general[general.index("magnification") + 1] == "calculate_diameters"


def test_it_sits_directly_under_magnification_on_the_form():
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    for app in ("mask", "timelapse"):
        keys = _APP_CATEGORY_SPECS[app][0][1]
        assert keys[keys.index("magnification") + 1] == "calculate_diameters"


def test_the_tooltip_states_the_rule():
    from spacr.settings import tooltips

    text = tooltips["calculate_diameters"]
    for words in ("2*magnification+80", "0.75*magnification+45", "pinned",
                  "clear its cell", "off and on again", "Default True"):
        assert words in text


@pytest.mark.parametrize("raw, on", [(True, True), (False, False),
                                     (None, True), ("false", False),
                                     ("False", False), ("0", False),
                                     ("true", True), (1, True), (0, False)])
def test_the_switch_reads_files_and_text(raw, on):
    assert _calculates_diameters({"calculate_diameters": raw}) is on


def test_an_older_settings_file_without_it_keeps_calculating():
    assert _calculates_diameters({}) is True
    assert _calculates_diameters(None) is True


def test_on_a_set_diameter_is_pinned_and_a_blank_one_is_not():
    settings = {"calculate_diameters": True, "cell_diameter": 90,
                "nucleus_diameter": None, "pathogen_diameter": "auto"}
    assert _diameter_is_pinned(settings, "cell")
    assert not _diameter_is_pinned(settings, "nucleus")
    assert _diameter_is_pinned(settings, "pathogen")
    settings["calculate_diameters"] = False
    assert not any(_diameter_is_pinned(settings, obj)
                   for obj in ("cell", "nucleus", "pathogen"))


def test_on_nothing_is_held_still():
    settings = {"calculate_diameters": True, "magnification": 20,
                "cell_diameter": None, "nucleus_diameter": None,
                "pathogen_diameter": None}
    assert _blank_diameters_held_still(settings) == {}


def test_off_fills_only_the_blanks_at_the_magnification_they_showed():
    settings = {"calculate_diameters": False, "magnification": 40,
                "cell_diameter": None, "nucleus_diameter": 33,
                "pathogen_diameter": "auto"}
    assert _blank_diameters_held_still(settings, 20) == {"cell_diameter": 120}
    assert _blank_diameters_held_still(settings) == {"cell_diameter": 160}


def test_the_command_line_takes_the_switch():
    from spacr.cli import coerce_value
    from spacr.settings import expected_types

    assert coerce_value("calculate_diameters", "false", True,
                        expected_types, "mask") is False
    assert coerce_value("calculate_diameters", "true", False,
                        expected_types, "mask") is True


def test_a_settings_file_round_trips_the_switch_and_the_diameters(tmp_path):
    from spacr.cli import (load_settings_file, resolve_module,
                           resolve_settings, write_settings_file)

    path = tmp_path / "mask_settings.csv"
    path.write_text("Key,Value\nmagnification,60\ncalculate_diameters,False\n"
                    "cell_diameter,120\nnucleus_diameter,\n",
                    encoding="utf-8")
    module = resolve_module("mask")
    resolved = resolve_settings(module, str(path))
    assert resolved["calculate_diameters"] is False
    assert resolved["cell_diameter"] == 120
    assert resolved["nucleus_diameter"] is None
    written = write_settings_file(resolved, tmp_path / "again.csv")
    again = resolve_settings(module, written[0])
    assert again["calculate_diameters"] is False
    assert again["cell_diameter"] == 120
    assert load_settings_file(written[0])["calculate_diameters"] is False
    default = resolve_settings(module, None)
    assert default["calculate_diameters"] is True
    overridden = resolve_settings(module, None,
                                  ["calculate_diameters=false"])
    assert overridden["calculate_diameters"] is False


def test_no_double_scaling_with_auto_or_a_calculated_value():
    """A calculated diameter reaches Cellpose once, as itself; auto has no
    magnification in it at all."""
    from spacr.object_settings_table import _diameter_for_eval

    assert _diameter_for_eval(_resolved_diameter(None, 40, "cell")) == 160.0
    assert _diameter_for_eval(_resolved_diameter(160, 40, "cell")) == 160.0
    assert _diameter_for_eval(_resolved_diameter("auto", 40, "cell")) is None
    assert _diameter_for_eval(_resolved_diameter("auto", 40, "cell"),
                              estimates=True) == 0.0


def test_the_run_segments_at_the_values_the_switch_left(tmp_path,
                                                        monkeypatch):
    """The V1 Cellpose-SAM generator: on, a blank cell diameter follows the
    run's magnification; off, the value written when it was turned off at
    20x stays 120 px after the magnification moves to 60x."""
    from tests.test_cov_object_masks_sam import _settings, _write_npz
    import spacr.object as O

    seen = []

    class _Model:
        def __init__(self, **kwargs):
            pass

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            seen.append(diameter)
            masks = [np.zeros(np.asarray(i).shape[:2], dtype=np.uint16)
                     for i in x]
            return masks, [np.zeros(m.shape, np.float32) for m in masks], None

    import torch
    import types
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(O.accelerator, "cellpose_kwargs",
                        lambda: {"gpu": False, "device": torch.device("cpu")})
    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(
        CellposeModel=_Model))

    on = {"calculate_diameters": True, "cell_diameter": None}
    for mag in (20, 60):
        src = tmp_path / f"on_{mag}"
        _write_npz(src, n=2)
        O.generate_cellpose_masks_sam(
            str(src), _settings(src, magnification=mag, **on), "cell")

    off = {"calculate_diameters": False, "magnification": 20,
           "cell_diameter": None, "nucleus_diameter": None,
           "pathogen_diameter": None}
    off.update(_blank_diameters_held_still(off))
    src = tmp_path / "off_60"
    _write_npz(src, n=2)
    O.generate_cellpose_masks_sam(
        str(src), _settings(src, **dict(off, magnification=60)), "cell")
    assert seen == [120.0, 200.0, 120.0]


@pytest.mark.parametrize('invalid', ['not a diameter', -1, True, float('inf')])
def test_invalid_explicit_diameters_are_neither_filled_nor_pinned(invalid):
    """Magnification never silently repairs a malformed explicit user value."""
    settings = {'calculate_diameters': False, 'magnification': 40,
                'cell_diameter': invalid}
    before = settings.copy()
    assert _blank_diameters_held_still(settings) == {}
    assert settings == before
    settings['calculate_diameters'] = True
    assert not _diameter_is_pinned(settings, 'cell')
    assert 'nucleus_diameter' not in settings and 'pathogen_diameter' not in settings
