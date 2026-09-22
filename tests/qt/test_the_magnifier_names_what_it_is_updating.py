"""The magnifier's Updating mark names the model or method it is running.

Asked for 2026-09-21: "Updating Otsu …, or Updating toxoplasma_plaque_v1 …"
rather than a bare "Updating…", so a slow box says what it is waiting for.
"""
from types import SimpleNamespace

import pytest

from spacr.qt import i18n
from spacr.qt.screens import make_masks


def _box(mode, model):
    """A stand-in carrying just the settings :meth:`running_name` reads."""
    settings = (mode, 1.0, True, 0, model) + (0,) * 9
    return SimpleNamespace(_model_settings=lambda: settings)


@pytest.mark.parametrize("mode, model, shown", [
    ("cellpose", "toxoplasma_plaque_v1", "toxoplasma_plaque_v1"),
    ("cellpose", "cpsam", "cpsam"),
    ("otsu", "cpsam", "Otsu"),
    ("dinocell", "cpsam", "DINOCell"),
])
def test_the_mark_names_the_model_or_the_method(monkeypatch, mode, model, shown):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    name = make_masks._LiveMagnifier.running_name(_box(mode, model))
    assert name == shown
    assert make_masks._updating_caption(name) == f"Updating {shown}…"


def test_a_language_without_the_new_row_keeps_its_own_word(monkeypatch):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "sv")
    caption = make_masks._updating_caption("cpsam")
    assert caption.endswith("cpsam")
    assert "Updating" not in caption
