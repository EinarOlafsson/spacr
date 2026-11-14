"""A settings category at three edges the coverage ratchet found untested.

A body whose destroy link was already cut is still put back; an alpha
category's title is taken from the row its plain or "(Alpha)" name already
has in the catalog, with the badge in place of the bracket; and the body is
handed to the theme's surface seal as the ``surface`` role.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


from spacr.qt import theme                                   # noqa: E402
from spacr.qt.widgets import section as section_module       # noqa: E402
from spacr.qt.widgets.section import Section                 # noqa: E402

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("known,expected", [
    ({"Cloud": "Nuage"}, "Nuage"),
    ({"Confluency (Alpha)": "Confluence (Alpha)"}, "Confluence"),
    ({"Confluency (Alpha)": "コンフルエンス（アルファ）"}, "コンフルエンス"),
])
def test_an_alpha_title_uses_the_row_its_name_already_has(monkeypatch, known,
                                                          expected):
    monkeypatch.setattr(section_module, "tr",
                        lambda text, language=None: known.get(text, text))
    base = next(iter(known)).replace(" (Alpha)", "")
    source = f"{base} {theme.ALPHA_MARK}"
    assert Section._translated_alpha_name(source, "xx") == (
        f"{expected} {theme.ALPHA_MARK}")


def test_an_alpha_title_with_no_row_stays_in_english(monkeypatch):
    monkeypatch.setattr(section_module, "tr",
                        lambda text, language=None: text)
    source = f"Cloud {theme.ALPHA_MARK}"
    assert Section._translated_alpha_name(source, "xx") == source


def test_the_body_is_sealed_as_a_surface(qtbot, monkeypatch):
    """Built, and again on a style change, the body goes to the theme's seal
    with the ``surface`` role -- the card's own margin is left out of it."""
    from PySide6.QtCore import QEvent

    sealed = []
    monkeypatch.setattr(theme, "seal_surface",
                        lambda widget, role: sealed.append((widget, role)),
                        raising=False)
    section = Section("Sealed")
    qtbot.addWidget(section)
    assert sealed == [(section._body, "surface")]
    section.changeEvent(QEvent(QEvent.StyleChange))
    assert sealed[-1] == (section._body, "surface") and len(sealed) == 2


