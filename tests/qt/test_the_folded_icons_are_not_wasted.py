"""A folded module's icon goes where its settings went.

"remomber where possible integrate the modules and if possible make use of
the folded in moduals icons somwhere relevant."

A module that was folded into another one gives up its tile on Home. Where
the fold left it a BUTTON on the host's masthead, the button IS the icon and
nothing is owed. Where the fold turned it into SETTINGS CATEGORIES there is
no button, so the picture a user learned the module by has nowhere obvious
to go -- and a group of settings that arrived from somewhere else says
nothing at all about where.

What these tests pin:

  * the icon lands on the category HEADING, read off the real widget rather
    than off a helper's return value -- the pixmap on the heading is
    compared against the module's own icon, and against another module's, so
    "a mark is drawn" cannot pass for "the right mark is drawn";
  * both halves of a fold are marked: the cards Timelapse's fold mounts on
    Mask Generation, and the time-axis category Mask Generation drew itself;
  * Measure marks the Illumination category it has always had of its own;
  * the heading is not spent to do it -- the chevron and the category name
    are still there, and the mark is transparent to the mouse, so hovering
    it still hovers the heading the screen filters events on;
  * a key with no artwork of its own gets NO mark, rather than the generic
    glyph ``app_icon`` answers every unknown key with;
  * and a category the host wrote itself is left unclaimed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel

from spacr.qt import iconset
from spacr.qt.screens import mask as mask_folds
from spacr.qt.screens import measure as measure_folds
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.fold_strip import (
    mark_folded_categories, mark_folded_sections,
)
from spacr.qt.widgets.section import (
    SOURCE_ICON_NAME, SOURCE_ICON_PX, Section, module_mark,
)

#: ``app key -> the module that installs that screen's folds``. The folds are
#: hung on a screen by the main window as the stack reaches it, so a screen
#: built alone has none; installing them here is what makes these tests look
#: at the form a user sees rather than at half of it.
HOSTS = {"mask": mask_folds, "measure": measure_folds}


def _screen(qtbot, app_key: str) -> AppScreen:
    """One module screen, shown, with its folds installed."""
    screen = AppScreen(app_key=app_key)
    qtbot.addWidget(screen)
    HOSTS[app_key].install_folds(screen)
    screen.show()
    qtbot.waitExposed(screen)
    return screen


def _sections(screen: AppScreen) -> list:
    """Every settings category on the screen -- the host's and the folds'.

    A fold mounts its cards on the host's settings column but deliberately
    keeps them out of ``_settings_sections``, so a walk of that list alone
    would miss exactly the categories this instruction is about.
    """
    found = list(getattr(screen, "_settings_sections", []) or [])
    folds = getattr(mask_folds, "fold_set")(screen)
    if folds is not None:
        for fold in folds.folds.values():
            found.extend(getattr(fold, "sections", ()) or ())
    return found


def _by_title(screen: AppScreen) -> dict:
    """``UPPERCASED heading -> section`` for everything on the form."""
    return {str(section.title()).strip().upper(): section
            for section in _sections(screen)}


def _mark(section) -> QLabel:
    """The module mark drawn on this category's heading.

    Found under the HEADER, not under the section: the heading is the row
    the mark has to be on, and a label anywhere else in the card would be
    a decoration on the body rather than an attribution of the group.
    """
    return section.header().findChild(QLabel, SOURCE_ICON_NAME)


def _drawn(section):
    """The image actually painted on this heading's mark."""
    mark = _mark(section)
    assert mark is not None, f"{section.title()} carries no module mark"
    assert mark.isVisibleTo(section.header())
    # The section answers with the same widget a search of the header
    # finds, so a screen has one way to ask and it is the drawn one.
    assert section.source_mark() is mark
    pixmap = mark.pixmap()
    assert not pixmap.isNull(), f"{section.title()}'s mark is empty"
    return pixmap.toImage()


def _icon_image(key: str, mark: QLabel):
    """``key``'s own icon, rendered the size the heading draws it."""
    return iconset.app_icon(key).pixmap(
        QSize(SOURCE_ICON_PX, SOURCE_ICON_PX),
        mark.devicePixelRatioF()).toImage()


# ---------------------------------------------------------------------------
# The mark is the module's own, on the heading its settings became
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading", ["TRACKING SETUP", "TRACKING BACKENDS"])
def test_the_mounted_tracking_categories_carry_timelapses_icon(qtbot, heading):
    """The cards the fold mounted are marked with the module that sent them."""
    screen = _screen(qtbot, "mask")
    section = _by_title(screen).get(heading)
    assert section is not None, f"Mask Generation has no {heading} category"
    assert section.source_app() == "timelapse"
    drawn = _drawn(section)
    assert drawn == _icon_image("timelapse", _mark(section))
    # ... and it is not merely SOME icon: the host's own would pass a test
    # that only asked whether a picture was there.
    assert drawn != _icon_image("mask", _mark(section))


def test_mask_generations_own_time_category_carries_timelapses_icon(qtbot):
    """The half of the module the host had already written is marked too.

    Mask Generation has drawn the time-axis settings itself since before the
    fold, so marking only the mounted cards would leave one module's
    settings attributed in two cards and anonymous in a third.
    """
    screen = _screen(qtbot, "mask")
    section = _by_title(screen)["TIME AXES & TRACKING (BETA)"]
    assert section.source_app() == "timelapse"
    assert _drawn(section) == _icon_image("timelapse", _mark(section))


def test_measures_illumination_category_carries_illuminations_icon(qtbot):
    """Illumination is a button here AND a category; both wear its mark."""
    screen = _screen(qtbot, "measure")
    section = _by_title(screen)["ILLUMINATION CORRECTION"]
    assert section.source_app() == "illumination"
    drawn = _drawn(section)
    assert drawn == _icon_image("illumination", _mark(section))
    assert drawn != _icon_image("measure", _mark(section))
    # The same picture the masthead switch for that module carries, which is
    # the whole point: one mark, learned once.
    button = screen._fold_strip.button_for("illumination")
    assert button is not None
    assert drawn == button.icon().pixmap(
        QSize(SOURCE_ICON_PX, SOURCE_ICON_PX),
        _mark(section).devicePixelRatioF()).toImage()


def test_every_fold_that_became_categories_marks_at_least_one_heading(qtbot):
    """No folded module is left with its icon nowhere.

    Read back off the screen rather than off the tables, so a table entry
    that names a category the form does not have fails here instead of
    silently marking nothing.
    """
    for app_key, host in HOSTS.items():
        screen = _screen(qtbot, app_key)
        marked = host.mark_fold_sources(screen)
        for key in getattr(host, "FOLD_CATEGORIES", {}):
            assert marked.get(key), (
                f"{key} names categories on {app_key} but marked none")


def test_a_named_category_belongs_to_a_module_that_is_actually_folded():
    """A host cannot attribute its settings to a module it never folded.

    Mask Generation's table names its category folds; Measure's names a
    module on its strip. Either way the key has to be one this host holds,
    or the mark says a plate went somewhere it never went.
    """
    assert set(mask_folds.FOLD_CATEGORIES) <= set(mask_folds.FOLDED_APPS)
    assert set(measure_folds.FOLD_CATEGORIES) <= set(measure_folds.FOLDED_APPS)


# ---------------------------------------------------------------------------
# What the mark must not cost
# ---------------------------------------------------------------------------

def test_the_marked_heading_keeps_its_chevron_and_its_name(qtbot):
    """A QToolButton draws its arrow instead of its icon, so the mark is not
    in the icon slot -- and the proof is that both are still there."""
    screen = _screen(qtbot, "mask")
    section = _by_title(screen)["TRACKING SETUP"]
    header = section.header()
    assert header.arrowType() == Qt.RightArrow
    assert "TRACKING SETUP" in header.text()
    assert header.icon().isNull()
    section.set_expanded(True)
    assert header.arrowType() == Qt.DownArrow


def test_hovering_the_mark_still_hovers_the_heading(qtbot):
    """The screen filters hover events on the header to decide which
    category the pointer is over; a mark that took the mouse would report
    the wrong one over its own 16 px."""
    screen = _screen(qtbot, "mask")
    section = _by_title(screen)["TRACKING SETUP"]
    mark = _mark(section)
    assert mark.testAttribute(Qt.WA_TransparentForMouseEvents)
    # `childAt` answers with what a mouse press at that point would reach,
    # and skips a child that is transparent to it.
    assert section.header().childAt(mark.geometry().center()) is None


def test_the_mark_sits_at_the_trailing_end_of_the_heading(qtbot):
    """Beside the name rather than over it: the heading text is painted from
    the leading edge, so the mark belongs on the other one.

    Measured with the switch ON, because that is the only state in which
    the card is on screen at all -- and a position read off a card that has
    never been laid out is the widget's default, not its layout's answer.
    """
    screen = _screen(qtbot, "mask")
    switch = screen._fold_strip.button_for("timelapse")
    switch.setChecked(True)
    section = _by_title(screen)["TRACKING SETUP"]
    qtbot.waitUntil(lambda: section.isVisible())
    header, mark = section.header(), _mark(section)
    assert header.width() > 4 * SOURCE_ICON_PX
    assert mark.geometry().right() <= header.width()
    assert mark.x() > header.width() // 2
    assert mark.size() == QSize(SOURCE_ICON_PX, SOURCE_ICON_PX)


def test_installing_the_folds_twice_leaves_one_mark(qtbot):
    """The strip is installed as the stack reaches the screen, which happens
    every time a user comes back to it."""
    screen = _screen(qtbot, "mask")
    mask_folds.install_folds(screen)
    mask_folds.mark_fold_sources(screen)
    section = _by_title(screen)["TRACKING SETUP"]
    marks = section.header().findChildren(QLabel, SOURCE_ICON_NAME)
    assert len(marks) == 1


# ---------------------------------------------------------------------------
# A key with no picture of its own
# ---------------------------------------------------------------------------

def test_a_key_with_no_artwork_gets_no_mark(qtbot):
    """`app_icon` answers an unknown key with a generic glyph, which is right
    for a toolbar button that must show something and wrong for a heading:
    a mark that names no module a user can recognise is worse than none."""
    section = Section("Nowhere In Particular")
    qtbot.addWidget(section)
    assert module_mark("no_such_module_anywhere") is None
    assert section.set_source_app("no_such_module_anywhere") is False
    assert _mark(section) is None
    assert section.source_mark() is None


def test_a_mark_that_is_taken_away_stops_being_drawn(qtbot):
    """Re-attributing a category to a module with no picture must not leave
    the old module's mark standing over settings that are no longer its."""
    section = Section("Tracking Setup")
    qtbot.addWidget(section)
    assert section.set_source_app("timelapse", "Timelapse") is True
    assert section.set_source_app("no_such_module_anywhere") is False
    assert section.source_app() == "no_such_module_anywhere"
    assert not _mark(section).isVisibleTo(section.header())
    assert section.source_mark() is None


def test_a_category_the_host_wrote_itself_is_left_unclaimed(qtbot):
    """Most categories are the host's own, and marking them all would make
    the mark mean nothing."""
    screen = _screen(qtbot, "mask")
    section = _by_title(screen)["INPUT & METADATA"]
    assert section.source_app() == ""
    assert _mark(section) is None


# ---------------------------------------------------------------------------
# Finding the heading to mark
# ---------------------------------------------------------------------------

def test_a_heading_is_found_by_the_name_it_was_written_with(qtbot):
    """A heading is drawn uppercased and written mixed-case, and the table
    that names it is written the way a person would write it.

    Every settings category carries the written name as
    ``settingsCategorySource``, which is why the table can be spelled that
    way rather than in the shouting the header paints.
    """
    section = Section("Illumination Correction")
    section.setProperty("settingsCategorySource", "Illumination Correction")
    qtbot.addWidget(section)
    marked = mark_folded_categories(
        [section], {"illumination": ("illumination correction",)})
    assert marked == {"illumination": ("Illumination Correction",)}
    assert section.source_app() == "illumination"


def test_a_heading_with_no_written_name_falls_back_to_its_caption(qtbot):
    """A section built outside the settings form still gets marked."""
    section = Section("Illumination Correction")
    qtbot.addWidget(section)
    assert mark_folded_categories(
        [section], {"illumination": ("Illumination Correction",)}) == {
            "illumination": ("ILLUMINATION CORRECTION",)}
    assert section.source_app() == "illumination"


def test_a_heading_the_form_does_not_have_marks_nothing(qtbot):
    """A table entry naming a category that was renamed or removed leaves
    the form alone rather than marking whatever was nearest."""
    section = Section("Illumination Correction")
    qtbot.addWidget(section)
    assert mark_folded_categories(
        [section], {"illumination": ("Flat Field",)}) == {}
    assert section.source_app() == ""


def test_marking_skips_anything_that_is_not_a_settings_category(qtbot):
    """The sections are walked off a screen, and a screen holds widgets that
    are not categories."""
    from PySide6.QtWidgets import QWidget

    stranger = QWidget()
    qtbot.addWidget(stranger)
    assert mark_folded_sections("timelapse", [stranger]) == ()


# ---------------------------------------------------------------------------
# A mark that cannot be drawn costs the mark and nothing else
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_the_host_is_left_alone(qtbot):
    """The marking runs off a walk of the window's screen stack, so it is
    handed screens that host nothing."""
    from PySide6.QtWidgets import QWidget

    stranger = QWidget()
    qtbot.addWidget(stranger)
    assert measure_folds.mark_fold_sources(stranger) == {}
    assert mask_folds.mark_fold_sources(stranger) == {}


def test_an_icon_that_cannot_be_read_leaves_the_heading_alone(qtbot,
                                                              monkeypatch):
    """Icon lookup reads files off disk and re-inks them, and neither is
    something a settings heading should be able to fail on."""
    def _explode(*_args, **_kwargs):
        raise OSError("the icon cache is not readable")

    monkeypatch.setattr(iconset, "app_icon", _explode)
    section = Section("Tracking Setup")
    qtbot.addWidget(section)
    assert module_mark("timelapse") is None
    assert section.set_source_app("timelapse", "Timelapse") is False
    assert _mark(section) is None


def test_an_empty_icon_is_not_drawn_as_an_empty_square(qtbot, monkeypatch):
    """`app_icon` can answer softly with a null icon when its font or its
    artwork went missing at runtime."""
    from PySide6.QtGui import QIcon

    monkeypatch.setattr(iconset, "app_icon", lambda *a, **k: QIcon())
    section = Section("Tracking Setup")
    qtbot.addWidget(section)
    assert module_mark("timelapse") is None
    assert section.set_source_app("timelapse", "Timelapse") is False


def test_one_heading_that_refuses_the_mark_does_not_stop_the_rest(qtbot):
    """The categories are marked in a loop over a whole module's settings,
    and a screen that opens without one mark beats a screen that does not
    open."""
    class _Awkward(Section):
        def set_source_app(self, key, name=""):
            raise RuntimeError("this heading is already gone")

    awkward = _Awkward("Tracking Setup")
    willing = Section("Tracking Backends")
    qtbot.addWidget(awkward)
    qtbot.addWidget(willing)
    assert mark_folded_sections("timelapse", [awkward, willing]) == (
        "TRACKING BACKENDS",)


def test_the_marking_never_takes_a_screen_down_with_it(qtbot, monkeypatch):
    """Whatever goes wrong inside it, the host still gets its screen."""
    def _explode(*_args, **_kwargs):
        raise RuntimeError("no marks today")

    screen = _screen(qtbot, "mask")
    monkeypatch.setattr(mask_folds, "mark_folded_sections", _explode)
    monkeypatch.setattr(mask_folds, "mark_folded_categories", _explode)
    assert mask_folds.mark_fold_sources(screen) == {}
    monkeypatch.setattr(measure_folds, "mark_folded_categories", _explode)
    assert measure_folds.mark_fold_sources(_screen(qtbot, "measure")) == {}
