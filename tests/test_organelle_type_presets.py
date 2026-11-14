"""Organelle-type presets: what ``preset_for`` recommends and ``apply_preset`` says.

A preset turns a biological name into segmentation settings, and the same
name can mean a dot at one size and a ring at another.  ``preset_for`` must
never recommend a method the run-time validator rejects (including for
presets that do not ship yet), and ``apply_preset(explain=True)`` must print
exactly what it filled and what it kept, so a preset that kept the user's
method and one that overrode it can be told apart.
"""
from __future__ import annotations

import pytest

from spacr.object import _validate_organelle_settings
from spacr.organelle_types import (LEGAL_METHODS, ORGANELLE_TYPES,
                                   RING_RESOLVABLE_PX, OrganelleType,
                                   apply_preset, preset_for)


def test_preset_for_recommends_a_method_legal_for_the_morphology_size_picks(
        subtests, monkeypatch):
    """Shipped presets recommend their own method and params; ``custom``
    recommends nothing; a method illegal for the half the diameter picks, or
    no method at all, becomes the first legal method for that morphology,
    keeping the extra params; an unknown name raises."""
    shipped = [
        ("custom", None, {}),
        ("vesicular below the ring threshold", RING_RESOLVABLE_PX - 1,
         {"organelle_morphology": "spots", "organelle_method": "log",
          "organelle_watershed_spots": True, "organelle_log_min_sigma": 1,
          "organelle_log_max_sigma": 6}),
        ("vesicular at the ring threshold", RING_RESOLVABLE_PX,
         {"organelle_morphology": "ring", "organelle_method": "log",
          "organelle_watershed_spots": True, "organelle_log_min_sigma": 1,
          "organelle_log_max_sigma": 6}),
        ("filamentous", None,
         {"organelle_morphology": "network", "organelle_method": "ridge",
          "organelle_ridge_filter": "frangi",
          "organelle_ridge_sigmas": [1, 2, 3],
          "organelle_skeletonize": True}),
    ]
    for case_id, diameter, expected in shipped:
        name = case_id.split()[0]
        with subtests.test(msg=case_id):
            assert preset_for(name, diameter) == expected

    monkeypatch.setitem(ORGANELLE_TYPES, "tipping", OrganelleType(
        label="Tipping", members=("a thing that swells",), method="ridge",
        size_split=("network", "irregular"),
        params={"organelle_fill_holes": 8}))
    monkeypatch.setitem(ORGANELLE_TYPES, "silent", OrganelleType(
        label="Silent", members=(), method="", morphology="ring"))
    built = [
        ("tipping", RING_RESOLVABLE_PX - 1, "network", "ridge"),
        ("tipping", RING_RESOLVABLE_PX + 1, "irregular",
         LEGAL_METHODS["irregular"][0]),
        ("silent", None, "ring", LEGAL_METHODS["ring"][0]),
    ]
    for name, diameter, morphology, method in built:
        with subtests.test(msg=f"{name} at {diameter}"):
            out = preset_for(name, diameter)
            assert out["organelle_morphology"] == morphology
            assert out["organelle_method"] == method
            assert _validate_organelle_settings(morphology, method) is None
            if name == "tipping":
                assert out["organelle_fill_holes"] == 8
                assert sorted(out) == ["organelle_fill_holes",
                                       "organelle_method",
                                       "organelle_morphology"]
    assert LEGAL_METHODS["irregular"][0] != "ridge"

    with subtests.test(msg="unknown name"):
        with pytest.raises(ValueError, match="mitochondria"):
            preset_for("mitochondria")


def test_apply_preset_fills_unset_settings_and_explains_exactly_what_it_did(
        subtests, capsys):
    """Unset (or None) keys are filled, set ones kept; with ``explain`` the
    printout names the preset, its members, the size rule (split presets
    only), every value set and kept, and the caveat; without ``explain``, or
    for a preset that decides nothing, nothing is printed; a misspelled type
    raises."""
    vesicular = ORGANELLE_TYPES["vesicular"]
    size_rule = ("  size decides the detector: below 15 px 'spots', at or "
                 "above it 'ring'; organelle_diameter is {}")
    covers = ("  covers: vacuoles, autophagosomes, transport vesicles, "
              "secretory vesicles, early endosomes, lysosomes")
    note = f"  note: {vesicular.caveat}"
    cases = [
        ("vesicular, method kept",
         {"organelle_type": "vesicular", "organelle_diameter": 40,
          "organelle_method": "cellpose"}, True,
         {"organelle_morphology": "ring", "organelle_method": "cellpose"},
         ["organelle_type = Vesicular", covers, size_rule.format(40),
          "  set    organelle_log_max_sigma = 6",
          "  set    organelle_log_min_sigma = 1",
          "  set    organelle_morphology = 'ring'",
          "  set    organelle_watershed_spots = True",
          "  KEPT   organelle_method = 'cellpose' (yours, not overridden)",
          note]),
        ("vesicular, small, None method is filled",
         {"organelle_type": "vesicular", "organelle_diameter": 12,
          "organelle_method": None}, True,
         {"organelle_morphology": "spots", "organelle_method": "log"},
         ["organelle_type = Vesicular", covers, size_rule.format(12),
          "  set    organelle_log_max_sigma = 6",
          "  set    organelle_log_min_sigma = 1",
          "  set    organelle_method = 'log'",
          "  set    organelle_morphology = 'spots'",
          "  set    organelle_watershed_spots = True", note]),
        ("filamentous has no size rule",
         {"organelle_type": "filamentous"}, True,
         {"organelle_morphology": "network", "organelle_method": "ridge"},
         ["organelle_type = Filamentous",
          "  covers: microtubules, F-actin filaments, intermediate filaments",
          "  set    organelle_method = 'ridge'",
          "  set    organelle_morphology = 'network'",
          "  set    organelle_ridge_filter = 'frangi'",
          "  set    organelle_ridge_sigmas = [1, 2, 3]",
          "  set    organelle_skeletonize = True"]),
        ("silent without explain",
         {"organelle_type": "vesicular", "organelle_diameter": 12}, False,
         {"organelle_morphology": "spots", "organelle_method": "log"}, []),
        ("custom decides nothing", {"organelle_type": "custom"}, True, {}, []),
    ]
    for case_id, settings, explain, expected, printed in cases:
        with subtests.test(msg=case_id):
            given = dict(settings)
            out = apply_preset(settings, explain=explain)
            assert settings == given
            for key, value in expected.items():
                assert out[key] == value
            if not expected:
                assert out == settings
            assert capsys.readouterr().out.splitlines() == printed
            for line in printed:
                if line.startswith("  set    "):
                    assert line.split()[1] in out

    with subtests.test(msg="misspelled type"):
        with pytest.raises(ValueError, match="lysosome"):
            apply_preset({"organelle_type": "lysosome"})
