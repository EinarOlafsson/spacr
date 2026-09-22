"""Prose cleanup must not rename the actual Gate Editor control."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_i18n_catalogs as builder


def test_portuguese_keeps_repeated_control_label_but_cleans_ordinary_prose(monkeypatch):
    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    source = "Rectangle through view works through a projected outline."
    target = "Rectangle through view funciona through um contorno projetado."
    expected = "Rectangle through view funciona por meio de um contorno projetado."
    assert builder._contextualize(target, "pt", source) == expected
    assert builder._contextualize(expected, "pt", source) == expected


def test_protection_does_not_require_an_english_label(monkeypatch):
    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    source = "Rectangle through view works through a projected outline."
    target = "O retângulo na vista funciona por meio de um contorno projetado."
    assert builder._contextualize(target, "pt", source) == target
