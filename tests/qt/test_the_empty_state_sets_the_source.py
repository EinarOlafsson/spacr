"""The card on an empty screen takes you to your data.

IT USED TO OPEN THE DEMOS MENU. That is one way to get data and not the way
most people arrive: somebody who already has images pressed the only button
on the card and was offered a synthetic dataset instead of a folder chooser.
The demo is still offered -- in the card's sentence, which names the one demo
that lands on THIS screen rather than sending the reader to another module --
and the button now does the thing the card is about.

AND THE FIELD IS CALLED SOURCE. `src` is an abbreviation of an abbreviation;
the humaniser capitalised it and stopped, so the field that asks for the
images read "Src". It was renamed once to "Path" and now to "Source", to
standardise one word across every surface that names where a run reads its
input. The KEY is untouched -- every settings CSV in existence uses `src`.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_the_field_that_asks_for_the_images_is_called_source():
    """One word, at the single label source every surface reads."""
    from spacr.object_roles import setting_label

    assert setting_label("src") == "Source"


def test_a_model_path_is_not_renamed_to_source():
    """Only `src` is a source, and calling a model one would be a lie.

    Measured across all 508 keys in `spacr.settings`: `src` is the only
    source among them. `model_path`, `custom_model_path` and
    `organelle_unet_model_path` all name a MODEL.
    """
    from spacr.object_roles import setting_label

    assert setting_label("model_path") == "Model path"
    assert "Source" not in setting_label("custom_model_path")


def test_regression_keeps_its_more_specific_name(qtbot, qt_theme_applied):
    """Regression's `src` is where results are WRITTEN, not read.

    Not an abbreviation but a more specific true statement, so it survives
    the standardisation rather than being flattened by it.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert screen._settings_model._label_for("src") == "Output directory"


def test_the_card_offers_to_choose_the_source(qtbot, qt_theme_applied,
                                              monkeypatch):
    """The button sets src, and it goes through the MODEL to do it.

    `src` is a plain line edit on most screens and a set of plate databases
    on Classify; the model is what knows the difference, and writing text
    straight into the second one would put a folder where a set goes.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *a, **k: "/tmp/some/images")

    chosen = screen.choose_source_folder()

    assert chosen == "/tmp/some/images"
    assert screen._settings_model.collect()["src"] == "/tmp/some/images"


def test_cancelling_the_chooser_leaves_the_source_alone(qtbot, qt_theme_applied,
                                                        monkeypatch):
    """A cancelled dialog must not clear what is already there."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._settings_model.set_value_for_key("src", "/already/chosen")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *a, **k: "")

    assert screen.choose_source_folder() == ""
    assert screen._settings_model.collect()["src"] == "/already/chosen"
