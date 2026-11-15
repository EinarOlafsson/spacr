"""A fold button's restated maturity, and the previews a fold strip offers.

`FoldButton.set_stage` moves both the ``stage`` property and (for a switch)
the widget-local checked fill; a blank or repeated stage changes nothing.
`mask._offer_fold_previews` attaches a preview only for folds that have a
button on the strip and declare a preview, and one broken preview costs only
its own panel.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QCheckBox, QWidget                 # noqa: E402

from spacr.qt import preview_registry                            # noqa: E402
from spacr.qt.screens import mask as mask_module                 # noqa: E402
from spacr.qt.screens.map_barcodes import CategoryFoldSet        # noqa: E402
from spacr.qt.theme import STAGE_HOVER                           # noqa: E402
from spacr.qt.widgets.fold_strip import FoldButton, FoldStrip    # noqa: E402

pytestmark = pytest.mark.qt


def test_restating_a_fold_buttons_stage(qtbot, subtests):
    """The property and the checked fill move together; blank and repeats are no-ops."""
    with subtests.test(msg="a switch moves both the property and the checked fill"):
        button = FoldButton("timelapse", checkable=True)
        qtbot.addWidget(button)
        was = button.property("stage")
        assert STAGE_HOVER[was].lower() in button.styleSheet().lower()
        becomes = next(name for name in STAGE_HOVER if name != was)
        button.set_stage(becomes)
        assert button.property("stage") == becomes
        fill = button.styleSheet().lower()
        assert STAGE_HOVER[becomes].lower() in fill
        assert STAGE_HOVER[was].lower() not in fill

    for case, stages, expected in (
            ("restating the same stage", ("beta", "beta"), "beta"),
            ("a blank stage is ignored", ("alpha", ""), "alpha"),
            ("None is ignored", ("alpha", None), "alpha"),
    ):
        with subtests.test(msg=case):
            button = FoldButton("timelapse", checkable=True)
            qtbot.addWidget(button)
            button.set_stage(stages[0])
            before = button.styleSheet()
            button.set_stage(stages[1])
            assert button.property("stage") == expected
            assert button.styleSheet() == before

    with subtests.test(msg="a plain button takes the stage without a checked fill"):
        button = FoldButton("timelapse", checkable=False)
        qtbot.addWidget(button)
        button.set_stage("alpha")
        assert (button.property("stage"), button.styleSheet()) == ("alpha", "")


class _Screen:
    """A screen stand-in: `_offer_fold_previews` only passes it through."""


class _Host:
    """What ``attach_folded`` hands back: a card and its toggle."""

    def __init__(self):
        self.card = QWidget()
        self.toggle = QCheckBox(self.card)
        self.toggle.setChecked(True)


def _broken(key):
    raise RuntimeError("the preview builder is broken")


def test_a_fold_strip_offers_previews_only_where_it_can(qtbot, monkeypatch,
                                                        caplog, subtests):
    """Offered keys and built keys for missing buttons, no preview and a broken one."""
    folds = CategoryFoldSet(_Screen(), {key: (key,) for key in
                                        ("timelapse", "motility")})
    both = (("timelapse", lambda on: None, True),
            ("motility", lambda on: None, True))
    for case, buttons, attach, offered_keys, built_keys, logged in (
            ("a fold with no button gets no preview", both[:1],
             lambda key: _Host(), {"timelapse"}, ["timelapse"], False),
            ("a fold that declares no preview is skipped", both,
             lambda key: None if key == "timelapse" else _Host(),
             {"motility"}, ["timelapse", "motility"], False),
            ("a preview that will not build costs only that panel", both,
             lambda key: _broken(key) if key == "timelapse" else _Host(),
             {"motility"}, ["timelapse", "motility"], True),
    ):
        with subtests.test(msg=case):
            built = []

            def attach_folded(screen, key, attach=attach, built=built):
                built.append(key)
                return attach(key)

            monkeypatch.setattr(preview_registry, "attach_folded", attach_folded)
            strip = FoldStrip(list(buttons))
            qtbot.addWidget(strip)
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="spacr.qt.screens.mask"):
                offered = mask_module._offer_fold_previews(_Screen(), folds, strip)
            assert set(offered) == offered_keys
            assert built == built_keys
            assert any("Could not attach timelapse" in record.getMessage()
                       for record in caplog.records) is logged
