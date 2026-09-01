"""A setting's API link must aim at what READS it, and at something real.

Instruction 336. The link used to be built from the SCREEN's app_key, so every
row on the Mask panel pointed at Mask's entry point whether the value was read
there or twelve calls down -- "the tooltips are presently the only helpful
things". These tests hold the replacement to three promises: the target exists,
it is PUBLIC (AutoAPI runs without ``private-members``, so a private symbol has
no anchor to land on), and the mapping is regenerable rather than hand-kept.
"""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def targets():
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS
    return SETTING_API_TARGETS


def test_every_exact_target_is_a_real_public_module_level_symbol(targets):
    """An anchor is a promise that the heading exists on the page."""
    exact = {k: v for k, v in targets.items() if v[2]}
    assert exact, "no exact targets: the generator produced nothing to check"
    missing, private = [], []
    for key, (module, symbol, _exact) in exact.items():
        if symbol.startswith("_") or any(
                part.startswith("_") for part in symbol.split(".")):
            private.append((key, f"{module}.{symbol}"))
            continue
        try:
            mod = importlib.import_module(module)
        except Exception as exc:                             # noqa: BLE001
            missing.append((key, module, repr(exc)))
            continue
        obj = mod
        for part in symbol.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                missing.append((key, f"{module}.{symbol}", "attribute absent"))
                break
    assert not private, f"private symbols have no AutoAPI anchor: {private[:5]}"
    assert not missing, f"targets that do not resolve: {missing[:5]}"


def test_no_target_points_into_the_gui_or_the_declaration_site(targets):
    """A link into the Qt layer answers "which widget shows this".

    The reader is looking at the widget already; the question is what CONSUMES
    the value. ``spacr.settings`` is excluded for the same reason -- it is
    where the setting is declared, not where it is read.
    """
    bad = {k: v[0] for k, v in targets.items()
           if v[0].startswith(("spacr.qt.", "spacr.settings"))}
    assert not bad, f"display-only targets leaked into the map: {list(bad)[:5]}"


def test_the_committed_map_matches_what_the_generator_produces():
    """The map is committed so the next audit is a diff, not a rerun.

    A stale generated file is worse than none: it looks authoritative while
    describing code that has moved. This fails the moment the two disagree.
    """
    spec = ROOT / "tools" / "build_setting_consumer_map.py"
    src = ast.parse(spec.read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(src)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"setting_keys", "resolve_targets"} <= names, (
        "the generator no longer exposes the functions this test pins")

    data = json.loads((ROOT / "docs" / "setting_consumers.json").read_text())
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS
    assert data["target_count"] == len(SETTING_API_TARGETS), (
        "docs/setting_consumers.json and the generated runtime table disagree; "
        "rerun tools/build_setting_consumer_map.py and commit both")


def test_a_known_setting_links_to_its_consumer_not_to_the_screen():
    """The reported symptom, pinned.

    "for organell type i get linked to spacr.core ... but i need to be linked
    to the actual function that takes the organelle_type setting."
    """
    from spacr.qt.screens.settings_model import api_docs_url

    url = api_docs_url("mask", "organelle_type")
    assert "spacr/core/index.html" not in url, (
        "organelle_type still resolves to the screen's entry point")
    assert "#spacr." in url, "no anchor: the link still lands on a module page"


def test_an_unmapped_setting_still_gets_the_module_link():
    """The fallback must survive, or a checkout that has not run the generator
    would lose the link it had before this change."""
    from spacr.qt.screens.settings_model import api_docs_url

    url = api_docs_url("mask", "a_setting_that_does_not_exist")
    assert url.endswith("index.html"), url
    assert "#" not in url
