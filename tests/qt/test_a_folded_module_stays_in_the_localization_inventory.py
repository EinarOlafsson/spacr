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
import sys


ROOT = Path(__file__).resolve().parents[2]


def _builder():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        return import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)


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


def test_no_reviewed_translation_is_bound_to_a_key_that_left_the_inventory():
    """The exact gate the docs job fails on when the inventory shrinks.

    Reviewed evidence is bound to a table and a key. Rewording a tooltip
    is ordinary drift the catalog rebuild resolves; a key that is no
    longer inventoried at all cannot be resolved by any rebuild, because
    the generator can no longer see the source it would translate.
    """
    import json

    builder = _builder()
    sources = builder.canonical_sources()
    orphaned = []
    for language_dir in sorted(builder.REVIEWED_RUNTIME_DIR.iterdir()):
        if not language_dir.is_dir():
            continue
        for path in sorted(language_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
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
                    orphaned.append(f"{language_dir.name}/{record['table']}/{key}")

    assert not orphaned, (
        f"{len(orphaned)} reviewed translations name a source the catalog "
        f"generator can no longer see: {sorted(set(orphaned))[:5]}"
    )
