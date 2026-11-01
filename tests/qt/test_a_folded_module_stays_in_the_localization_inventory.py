"""A module folded into a host is still translated UI.

A folded module has no registry row, so ``spacr.qt.app.APPS`` alone is not
the list of screens whose settings a user reads. Its form still opens as a
page on its host, and every label and help paragraph on that form still
needs a translation. When Anndata Export, Barcode QC and Explain CV folded,
the catalog generator lost their authored tooltips, which turned every
reviewed translation bound to one of them into a hard "stale reviewed
runtime source" error and failed the docs build before Sphinx ever ran.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _builder():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        return import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)


@pytest.fixture
def pinned_reviewed_runtime_payloads():
    """Every reviewed runtime file exactly as the committed pin names it.

    ``docs/i18n/reviewed/`` is not tracked by this repository. It is
    hydrated from the pinned translation-review repository
    (``docs/translation_reviews_revision.json``) by
    ``tools/bootstrap_translation_reviews.py``, and only when the catalog
    builder runs as a script; importing the builder, as this test does,
    never hydrates. A translation closure moves the pin forward while
    other checkouts keep the files of the pin they last hydrated, so the
    directory on disk can lag the source it is compared with. That
    happened on 2026-11-01: the settings strip stopped showing the
    "Export settings…" caption, the next pin retired its nine reviewed
    records, and a checkout still holding the earlier hydration reported
    those nine records as orphaned.

    The fixture therefore reads the snapshot the committed pin fixes:
    each runtime path in the pinned manifest, from disk when its SHA-256
    matches the manifest and otherwise from the pinned revision in the
    local review cache. Like the builder, it reads the files directly in
    each language folder, not the ``archive`` folders beneath them. Files
    on disk that the pin no longer lists are stale hydrations and are not
    read.
    """
    bootstrap = sys.modules.get("bootstrap_translation_reviews")
    if bootstrap is None:
        _builder()
        bootstrap = sys.modules["bootstrap_translation_reviews"]
    pin, _identity = bootstrap._read_pin(ROOT)
    cache = bootstrap._cache_root(None)
    files = bootstrap._manifest(pin, cache, True)
    prefix = "docs/i18n/reviewed/runtime/"
    repository = None
    payloads = []
    for path in sorted(files):
        parts = path[len(prefix):].split("/")
        if not (path.startswith(prefix) and len(parts) == 2
                and parts[1].endswith(".json")):
            continue
        local = ROOT / path
        raw = local.read_bytes() if local.is_file() else b""
        if bootstrap._sha256(raw) != files[path]["sha256"]:
            if repository is None:
                repository = bootstrap._repository(pin, cache, True)
            raw = bootstrap._git(
                repository, "show", f"{pin['revision']}:{path}").stdout
            assert bootstrap._sha256(raw) == files[path]["sha256"], (
                f"the pinned review cache does not hold {path} as pinned"
            )
        payloads.append((parts[0], json.loads(raw.decode("utf-8"))))
    assert payloads, "the pinned manifest names no reviewed runtime file"
    return payloads


def test_every_folded_module_contributes_its_settings_to_the_inventory():
    from spacr.qt.screens.settings_model import (
        _FOLDED_DEFAULTS_MODULES,
        resolve_default_settings,
    )

    sources = _builder().canonical_sources()
    labels = sources["setting_labels"]
    tooltips = sources["setting_tooltips"]

    assert _FOLDED_DEFAULTS_MODULES, "no module is folded; the fold seam moved"
    for app_key in _FOLDED_DEFAULTS_MODULES:
        settings = resolve_default_settings(app_key)
        assert settings, f"{app_key} resolves to no settings at all"
        absent = [key for key in settings if str(key) not in labels]
        assert not absent, (
            f"{app_key} is folded, so its form has no registry row -- and "
            f"{len(absent)} of its settings left the localization "
            f"inventory: {sorted(absent)[:5]}"
        )
        assert any(str(key) in tooltips for key in settings), (
            f"{app_key} contributes no authored help text to the inventory"
        )


def test_no_reviewed_translation_is_bound_to_a_key_that_left_the_inventory(
        pinned_reviewed_runtime_payloads):
    """The exact gate the docs job fails on when the inventory shrinks.

    Reviewed evidence is bound to a table and a key. Rewording a tooltip
    is ordinary drift the catalog rebuild resolves; a key that is no
    longer inventoried at all cannot be resolved by any rebuild, because
    the generator can no longer see the source it would translate.
    """
    builder = _builder()
    sources = builder.canonical_sources()
    orphaned = []
    for language, payload in pinned_reviewed_runtime_payloads:
        for record in payload.get("records", []):
            table = sources.get(str(record["table"]))
            key = str(record["key"])
            if isinstance(table, dict):
                present = key in table
            elif isinstance(table, (tuple, list, set, frozenset)):
                present = key in table
            else:
                present = False
            if not present:
                orphaned.append(f"{language}/{record['table']}/{key}")

    assert not orphaned, (
        f"{len(orphaned)} reviewed translations name a source the catalog "
        f"generator can no longer see: {sorted(set(orphaned))[:5]}"
    )
