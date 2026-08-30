"""What a fold keeps when its registry row is gone, and what it switches.

Folding a module ends in its row being dropped from the app registry, so
everything a user recognised it by -- the name, the sentence, the maturity
colour -- has to come from somewhere else afterwards. The other half is the
gate: a fold that is switched on has to make the run do what the form is
showing, including the folds it cannot mean anything without.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QPushButton, QVBoxLayout,  # noqa: E402
                               QWidget)

from spacr.qt.screens import map_barcodes as mb  # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------
# where the name, the sentence and the stage come from
# --------------------------------------------------------------------------

def test_a_module_the_registry_still_holds_is_read_from_the_registry():
    from spacr.qt import app as app_module

    key = app_module.APPS[0][0]

    name, description, stage = mb.fold_description(key)

    assert name == app_module.APPS[0][1]
    assert description == app_module.APPS[0][2]
    assert stage == app_module.app_stage(key)


def test_a_registry_that_cannot_be_read_falls_back_to_the_record(monkeypatch,
                                                                  caplog):
    from spacr.qt import app as app_module

    def refuse(_key):
        raise RuntimeError("the registry is mid-rebuild")

    monkeypatch.setattr(app_module, "app_stage", refuse)
    key = next(iter(mb.FOLD_FALLBACK))
    monkeypatch.setattr(
        app_module, "APPS",
        [(key, "Live name", "Live sentence", "data"), *app_module.APPS])

    with caplog.at_level(logging.DEBUG, logger=mb.LOG.name):
        name, description, stage = mb.fold_description(key)

    assert (name, description) == ("Live name", "Live sentence"), (
        "what was read before the failure is kept")
    assert stage == mb.FOLD_FALLBACK[key][2], (
        "the maturity colour comes from the record the tile left behind")
    assert any("app registry" in record.getMessage()
               for record in caplog.records)


def test_a_key_nobody_has_a_record_for_answers_with_blanks(monkeypatch,
                                                            caplog):
    from spacr.qt.widgets import fold_strip

    def refuse(_key):
        raise RuntimeError("the shared records have gone")

    monkeypatch.setattr(fold_strip, "folded_fallback", refuse)

    with caplog.at_level(logging.DEBUG, logger=mb.LOG.name):
        assert mb.fold_description("not-a-module") == ("", "", "")


# --------------------------------------------------------------------------
# what that identity does to the button
# --------------------------------------------------------------------------

def test_a_button_for_a_module_with_no_stage_is_left_uncoloured(qapp,
                                                                monkeypatch):
    monkeypatch.setattr(mb, "fold_description",
                        lambda _key: ("Timelapse", "Track objects", ""))
    button = QPushButton()

    mb.restate_fold_button(button, "timelapse")

    assert button.toolTip() == "Timelapse\nTrack objects"
    assert button.accessibleName() == "Timelapse"
    assert not button.property("stage"), (
        "an unknown maturity must not be painted as one")


def test_a_plain_button_carries_the_stage_as_a_property(qapp, monkeypatch):
    monkeypatch.setattr(mb, "fold_description",
                        lambda _key: ("Motility", "Measure movement", "beta"))
    button = QPushButton()

    mb.restate_fold_button(button, "motility")

    assert button.property("stage") == "beta", (
        "the hover colour is computed from this property")


def test_a_button_that_knows_its_own_stage_is_asked_rather_than_set(
        qapp, monkeypatch):
    monkeypatch.setattr(mb, "fold_description",
                        lambda _key: ("Motility", "Measure movement", "beta"))

    class _StageAwareButton(QPushButton):
        def __init__(self):
            super().__init__()
            self.told = None

        def set_stage(self, stage):
            self.told = stage

    button = _StageAwareButton()
    mb.restate_fold_button(button, "motility")

    assert button.told == "beta", (
        "the property alone leaves the checked fill lighting the wrong colour")


def test_a_button_that_is_not_there_is_not_an_error(qapp, monkeypatch):
    looked_up = []
    monkeypatch.setattr(
        mb, "fold_description", lambda key: looked_up.append(key))

    result = mb.restate_fold_button(None, "motility")

    assert result is None
    assert looked_up == [], "a missing button should cost no registry lookup"


# --------------------------------------------------------------------------
# the page strip
# --------------------------------------------------------------------------

def _host_with_a_body():
    host = QWidget()
    host.app_key = "mask"
    layout = QVBoxLayout(host)
    body = QWidget(host)
    layout.addWidget(body, 1)
    return host, body


def test_a_style_block_that_cannot_be_registered_costs_no_page(monkeypatch,
                                                               caplog):
    from spacr.qt import theme

    def refuse(*_args, **_kwargs):
        raise RuntimeError("no stylesheet yet")

    monkeypatch.setattr(theme, "register_widget_qss", refuse)

    with caplog.at_level(logging.DEBUG, logger=mb.LOG.name):
        # The screen scopes a successfully registered late style.  This path
        # refuses during registration, so no real widget is needed, but the
        # call must still honour the production helper's screen contract.
        mb._ensure_pages_qss(None)

    assert any("fold page QSS" in record.getMessage()
               for record in caplog.records)


def test_a_body_the_layout_does_not_hold_makes_no_page_strip(qapp,
                                                             monkeypatch):
    host, _body = _host_with_a_body()
    stranger = QWidget()
    monkeypatch.setattr(mb, "_page_body", lambda _screen: stranger)
    try:
        assert mb.host_pages(host) is None, (
            "the host's own page has to keep the place its body had")
    finally:
        host.deleteLater()


def test_a_page_whose_module_has_no_mark_still_opens(qapp, monkeypatch,
                                                     caplog):
    from spacr.qt import iconset

    def refuse(_key):
        raise RuntimeError("the icon set is not loaded")

    monkeypatch.setattr(iconset, "app_icon", refuse)
    host, _body = _host_with_a_body()
    folded = QWidget()
    folded.app_key = "timelapse"

    try:
        with caplog.at_level(logging.DEBUG, logger=mb.LOG.name):
            assert mb.show_as_page(folded, host, "Timelapse") is folded

        pages = getattr(host, "_fold_pages")
        assert pages.indexOf(folded) > 0, "the host's own page stays first"
        assert pages.currentWidget() is folded
    finally:
        host.deleteLater()

    assert any("no mark for the timelapse page" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------

@pytest.fixture()
def fold_set(qapp):
    screen = QWidget()
    folds = mb.CategoryFoldSet(
        screen,
        {"timelapse": ("timelapse",), "motility": ("measure_motility",)},
        implies={"motility": ("timelapse",)})
    yield folds
    screen.deleteLater()


def test_a_fold_that_cannot_stand_alone_switches_on_what_it_needs(fold_set):
    fold_set.set_active("motility", True)

    assert fold_set.is_active("motility") is True
    assert fold_set.is_active("timelapse") is True, (
        "the assay runs inside the timelapse branch; asking for one asks "
        "for the other")
    assert fold_set.apply_gates() == {"timelapse": True,
                                      "measure_motility": True}


def test_switching_off_a_dependency_switches_off_what_needed_it(fold_set):
    fold_set.set_active("motility", True)

    fold_set.set_active("timelapse", False)

    assert fold_set.is_active("timelapse") is False
    assert fold_set.is_active("motility") is False, (
        "a form showing the assay's knobs without tracking describes a run "
        "that cannot happen")


def test_a_key_that_is_not_a_fold_here_switches_nothing(fold_set):
    fold_set.set_active("sequencing", True)

    assert fold_set.is_active("sequencing") is False
    assert fold_set.apply_gates() == {"timelapse": False,
                                      "measure_motility": False}


# --------------------------------------------------------------------------
# what one fold contributes on its own
# --------------------------------------------------------------------------

class _HostModel:
    def __init__(self, values):
        self._values = values

    def collect(self):
        return dict(self._values)


def test_a_fold_with_no_host_form_mounts_nothing(qapp):
    fold = mb.CategoryFold(QWidget(), "timelapse", ("timelapse",))

    assert fold.mount() is False, (
        "no form to mount into leaves the host exactly as it was")
    assert fold.collect() == {}


def test_a_fold_reports_only_the_settings_it_brought(qapp):
    screen = QWidget()
    screen._settings_model = _HostModel({"timelapse": True, "src": "/data",
                                         "fps": 4})
    fold = mb.CategoryFold(screen, "timelapse", ("timelapse",))
    fold.settings_keys = ("timelapse", "fps", "not_on_this_form")

    try:
        assert fold.collect() == {"timelapse": True, "fps": 4}, (
            "the host's own keys are not this fold's to report")
    finally:
        screen.deleteLater()
