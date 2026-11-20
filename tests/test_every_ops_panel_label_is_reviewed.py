"""Every OPS panel label (item 969's labels without the leading "Ops") has a reviewed record in every language.

Item 969 shows each OPS setting on the OPS panel under its own label without
the "Ops" every label starts with, catalogued as ``setting_labels`` row
``ops.<setting>``. A setting added after that change gets the row in the
English catalog automatically, but no reviewed translation: the catalog build
then falls back to an unreviewed machine translation, or fails only where that
fallback happens to equal the English. This test names the missing rows, so an
item that adds an OPS setting knows to include its ``ops.<setting>`` label in
its translation task (landing train 2, 2026-11-19).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


def _english_ops_panel_labels():
    """The ``ops.<setting>`` rows of the committed English runtime catalog."""
    namespace: dict = {}
    exec(compile((ROOT / "spacr/qt/i18n_catalogs/en.py").read_text(encoding="utf-8"),
                 "en.py", "exec"), namespace)
    return {key: source for key, source in namespace["SETTING_LABELS"].items()
            if key.startswith("ops.")}


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_ops_panel_label_has_a_reviewed_record(language):
    directory = ROOT / "docs/i18n/reviewed/runtime" / language
    if not directory.is_dir():
        pytest.skip("the pinned reviewed records are not fetched in this checkout")
    labels = _english_ops_panel_labels()
    assert len(labels) > 400
    reviewed = set()
    for path in directory.glob("*.json"):
        for record in json.loads(path.read_text(encoding="utf-8")).get("records", []):
            if record.get("table") == "setting_labels" and str(record.get("key", "")).startswith("ops."):
                reviewed.add((record["key"], record.get("source")))
    missing = sorted(f"{key} ({source!r})" for key, source in labels.items()
                     if (key, source) not in reviewed)
    assert not missing, (
        f"{language}: {len(missing)} OPS panel label(s) have no reviewed setting_labels record; "
        f"add an ops.<setting> label record for each to the item's translation task: "
        + ", ".join(missing))
