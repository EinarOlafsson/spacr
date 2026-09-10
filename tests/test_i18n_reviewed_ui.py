"""Contracts for the reviewed, context-sensitive runtime UI vocabulary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from i18n_reviewed_ui import LANGUAGES, REVIEWED_UI_TRANSLATIONS  # noqa: E402

# 86 -> 87 on 2026-09-06, +1/-0: "Source", the empty-state card's label for the
# data a module reads.  It is here rather than in the generated layer because
# the generated one had it as a ROAD in Korean (도로), a LANE in French (Voie)
# and a ROUTE in Icelandic (leiðin) -- all three are "path" in the sense of a
# way to walk, and none is the sense of a source directory.  The reviewed row
# is the fix, and the pin moves with it rather than after it.
REVIEWED_UI_SOURCE_COUNT = 87
REVIEWED_UI_SOURCE_SHA256 = "d0931a39009b4916bc4a247c1803c92316951318182bbd6b5815cc3094fe25a6"
REVIEWED_UI_CONTENT_SHA256 = "f24504b25664a03e2328c8eb6cc0d693837ab6cea291b3fb01fc38c42ea51d50"


def test_reviewed_ui_vocabulary_is_complete_and_pinned():
    """Every reviewed source must provide a nonblank value in every locale."""
    assert len(REVIEWED_UI_TRANSLATIONS) == REVIEWED_UI_SOURCE_COUNT
    digest = hashlib.sha256("\0".join(sorted(REVIEWED_UI_TRANSLATIONS)).encode("utf-8")).hexdigest()
    assert digest == REVIEWED_UI_SOURCE_SHA256
    content_digest = hashlib.sha256(
        json.dumps(
            REVIEWED_UI_TRANSLATIONS,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert content_digest == REVIEWED_UI_CONTENT_SHA256
    expected = set(LANGUAGES)
    assert len(expected) == 9
    for source, row in REVIEWED_UI_TRANSLATIONS.items():
        assert source.strip() == source and source
        assert set(row) == expected, source
        assert all(isinstance(value, str) and value.strip() == value for value in row.values()), (
            source
        )
        if source.endswith("…"):
            assert all(value.endswith("…") for value in row.values()), source


def test_reviewed_ui_sources_still_belong_to_a_runtime_catalog_surface():
    """Do not accumulate translations for captions removed from spaCR."""
    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import en

    surface = (
        set(_ROWS)
        | set(en.UI_SOURCES)
        | set(en.SETTING_LABELS.values())
        | set(en.MODULE_SUMMARIES.values())
    )
    stale = sorted(set(REVIEWED_UI_TRANSLATIONS) - surface)
    assert not stale, f"reviewed translations no longer used by the UI: {stale}"


def test_reviewed_ui_vocabulary_rejects_the_known_false_sense_families():
    """Pin the scientific/UI meanings that generic models repeatedly missed."""
    rows = REVIEWED_UI_TRANSLATIONS
    assert rows["Figure"]["de"] == "Abbildung"
    assert rows["Figure"]["fr"] == "Figure"
    assert rows["Well"]["de"] == "Well"
    assert rows["Well"]["es"] == "Pocillo"
    assert rows["Plate"]["es"] == "Placa"
    assert rows["Axes…"]["de"] == "Achsen…"
    assert rows["Axes…"]["es"] == "Ejes…"
    assert rows["Sweeping…"]["sv"] == "Söker parametrar…"
    assert rows["Crop settings…"]["ko"] == "이미지 자르기 설정…"
    assert rows["Radius"]["sv"] == "Radie"
    assert rows["Annotate…"]["ko"] == "주석 지정…"
    assert rows["plane"]["is"] == "myndflötur"
    assert rows["Score the masks now"]["de"] == "Masken jetzt bewerten"
    assert rows["Open source…"]["es"] == "Abrir fuente…"
    assert rows["Folds"]["fr"] == "Partitions"
    assert rows["power"]["ko"] == "통계적 검정력"
    assert rows["Gate"]["es"] == "Región de selección"
    empty = rows[
        "Nothing to plot — build a table with at least one non-empty cell first."
    ]
    assert "Tabelle" in empty["de"] and "Zelle" in empty["de"]
    assert "表格" in empty["zh_CN"] and "单元格" in empty["zh_CN"]
    assert "tableau" in empty["fr"] and "cellule" in empty["fr"]

    contamination = (
        "ordförande",
        "herr präsident",
        "@info:",
        "description in lists",
        "빈 공간",
        "färdskrivaren",
        "na schön",
        "hachas",
        "plato",
        "sudar",
        "svett",
        "chiffre",
        "type de véhicule",
        "budget-programme",
    )
    failures = []
    for source, row in rows.items():
        for language, value in row.items():
            folded = value.casefold()
            if any(marker in folded for marker in contamination):
                failures.append(f"{language}:{source!r} -> {value!r}")
    assert not failures, "reviewed false-sense contamination:\n" + "\n".join(failures)
