"""Fold buttons keep their words and verdict dots when a lookup table is gone.

Pinned behaviour of :mod:`spacr.qt.widgets.fold_strip`:

* :func:`fold_label` names a key only the declared app catalogue knows with
  the catalogue's name, description and stage;
* with the catalogue unimportable, an unknown key still gets its
  title-cased name, no description and stage ``stable``;
* a :class:`FoldButton` whose regression-QC palette cannot be imported
  still paints its verdict dot, in the matching fallback colour.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor  # noqa: E402

from spacr.qt import app_catalog  # noqa: E402
from spacr.qt.widgets import fold_strip  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_key_only_the_catalogue_declares_is_named_from_it(monkeypatch):
    declared = SimpleNamespace(
        key="zz_only_declared", name="Only Declared",
        desc="Known only to the catalogue.", stage="alpha")
    monkeypatch.setattr(app_catalog, "DECLARED_APPS",
                        (*app_catalog.DECLARED_APPS, declared))

    assert fold_strip.fold_label("zz_only_declared") == (
        "Only Declared", "Known only to the catalogue.", "alpha")


def test_without_the_catalogue_an_unknown_key_is_title_cased(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.app_catalog", None)

    assert fold_strip.fold_label("zz_never_heard_of") == (
        "Zz Never Heard Of", "", "stable")


def test_the_verdict_dot_is_painted_without_the_qc_palette(qtbot, monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.regression_qc", None)
    button = fold_strip.FoldButton("regression_diagnostics")
    qtbot.addWidget(button)
    button.set_verdict("fail", "two checks failed")

    image = button.grab().toImage()
    size = max(6.0, button.height() * 0.22)
    centre_x = int(button.width() - size / 2 - 3.0)
    centre_y = int(3.0 + size / 2)
    ratio = image.devicePixelRatio()
    colour = QColor(image.pixel(int(centre_x * ratio), int(centre_y * ratio)))

    assert colour.name().upper() == "#F85149"
