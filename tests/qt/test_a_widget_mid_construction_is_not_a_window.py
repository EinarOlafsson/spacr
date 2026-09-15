"""The parentless widget is not a window, and must not be sheeted as one.

QT CALLS A PARENTLESS WIDGET A WINDOW. `QWidget::isWindow` is
`window_flags & Qt::Window` and Qt forces that flag on when the parent is
null, so a `QSpinBox()` built on its own line -- one statement before
`layout.addWidget(it)` -- is a window for as long as it takes to reach the
next statement.  `_SheetsEveryWindowThatAppears` is on the QApplication and
sees that widget's `Polish`, so it hands it all ~49 KB of the composed
sheet for a window it is about to stop being.

MEASURED, one offscreen registry sweep of 45 modules:

    the whole filter                                  10,518 ms
    widgets that are windows only for want of a parent 3,817 ms

and over half of that is one module screen sheeted three times over -- the
screen is built parentless, so it collects `Polish` events all through its
own construction, and the second application costs 255 ms rather than the
first's 7 ms because the screen has grown ~1,500 widgets in between.

WHAT THIS FILE GUARDS IS BOTH HALVES, because the saving is only worth
having if nothing goes unsheeted.  The filter exists so a window that
appears after the theme was composed still wears it -- instruction 380
moved the sheet off the QApplication and this is what catches the rest --
and a menu, a tooltip, a dialog or a shown window opening in the previous
theme is the regression it was written against.  So: the mid-construction
widget is left alone, and every window that is really appearing is not.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QEvent, Qt                          # noqa: E402
from PySide6.QtGui import QPalette                             # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QLabel,  # noqa: E402
                               QMenu, QVBoxLayout, QWidget)

from spacr.qt import theme                                     # noqa: E402

#: A sheet with one colour nothing else uses, so a widget wearing it is
#: unambiguous rather than a near-miss.
SHEET = "QLabel { color: rgb(11, 22, 33); }"
SHEET_HEX = "#0b1621"


@pytest.fixture
def sheeted_app(qtbot):
    """A QApplication carrying the per-window sheet and its filter."""
    app = QApplication.instance()
    theme.apply_stylesheet_per_window(app, SHEET)
    yield app


def _wears_the_sheet(widget) -> bool:
    """True when the filter has put the composed sheet on ``widget``."""
    return SHEET in (widget.styleSheet() or "")


def _resolved(widget) -> str:
    """The colour the label actually paints with, after a polish."""
    widget.ensurePolished()
    QApplication.instance().processEvents()
    return widget.palette().color(QPalette.WindowText).name()


# -- the control -------------------------------------------------------------

def test_the_harness_can_see_a_window_being_sheeted_at_all(sheeted_app,
                                                           qtbot):
    """THE CONTROL.

    A test that cannot see the filter sheet anything cannot see it stop,
    and would pass for ever.  A `QDialog` is a real window by its own
    window type, so it is sheeted at its `Polish` exactly as before.
    """
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.ensurePolished()
    assert _wears_the_sheet(dialog)


# -- what is left alone ------------------------------------------------------

def test_a_parentless_widget_is_not_sheeted_while_it_is_still_being_built(
        sheeted_app, qtbot):
    """The 3,817 ms.

    Parentless, no native window, not visible, and the plain `Qt.Window`
    type Qt forced on it -- which is every widget between its constructor
    and the `addWidget` on the next line.
    """
    orphan = QWidget()
    qtbot.addWidget(orphan)
    assert orphan.isWindow(), "the premise: Qt calls this a window"
    orphan.ensurePolished()
    assert not _wears_the_sheet(orphan)


def test_a_screen_sized_orphan_is_not_sheeted_once_per_child_it_grows(
        sheeted_app, qtbot):
    """WHY THE REPEAT IS THE EXPENSIVE HALF.

    A module screen is built parentless and polished again as it fills, so
    under the old filter the same object was sheeted two and three times --
    and the repeat costs 255 ms, not the first application's 7 ms, because
    by then there are ~1,500 widgets under it to repolish.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.setLayout(QVBoxLayout())
    for _ in range(8):
        child = QLabel("x")
        screen.layout().addWidget(child)
        screen.ensurePolished()
        child.ensurePolished()
    assert not _wears_the_sheet(screen)


# -- what is still caught ----------------------------------------------------

def test_a_menu_is_still_sheeted_before_it_can_be_shown(sheeted_app, qtbot):
    """MENUS STAY COVERED.

    380 recorded menus and tooltips as the part "a test cannot open a
    native menu"; they are covered because a QMenu is an ordinary
    top-level widget that gets a `Polish`.  Its window type is `Qt.Popup`,
    not `Qt.Window`, so it is not mistaken for a widget mid-construction.
    """
    menu = QMenu()
    qtbot.addWidget(menu)
    assert menu.windowType() != Qt.Window
    menu.ensurePolished()
    assert _wears_the_sheet(menu)


def test_a_parentless_widget_that_is_shown_is_sheeted_after_all(sheeted_app,
                                                                qtbot):
    """A `Show` means the widget is appearing, whatever its parentage.

    The skip is applied to `Polish` only, so a widget that really does open
    as its own window is sheeted at the moment it opens.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.ensurePolished()
    assert not _wears_the_sheet(window)
    window.show()
    qtbot.waitExposed(window)
    assert _wears_the_sheet(window)


def test_a_show_sheets_a_widget_the_polish_skip_would_have_passed_over(
        sheeted_app, qtbot):
    """THE `Polish`-ONLY GATE, ASSERTED DIRECTLY -- AND WHY IT HAD TO BE.

    Every other test here leaves this gate free: `QWidget::show` creates
    the native window BEFORE it sends the `Show`, so by the time the filter
    sees a real one `WA_WState_Created` is already set and the
    mid-construction test says False on its own.  Removing `kind ==
    QEvent.Polish` therefore changes no observable behaviour through any
    ordinary path -- it was watched surviving that mutation before this
    test was written.

    So the event is delivered here the way Qt delivers it, to a widget
    still in the state the skip is aimed at.  The gate is kept and tested
    rather than deleted because it is the rule the filter is written
    around -- a `Show` is always the moment -- and because what makes it
    unobservable is an ordering inside Qt rather than anything this module
    controls.
    """
    orphan = QWidget()
    qtbot.addWidget(orphan)
    assert theme._a_window_for_want_of_a_parent(orphan)
    QApplication.sendEvent(orphan, QEvent(QEvent.Show))
    assert _wears_the_sheet(orphan)


def test_a_widget_marked_as_a_sheet_target_is_still_sheeted_on_its_show(
        sheeted_app, qtbot):
    """THE PATH THE MODULE SCREENS ACTUALLY TAKE.

    `MainWindow._a_page_joined_the_stack` marks a page before it joins the
    stack precisely so "the pages that are not showing are marked instead,
    and sheeted on their own `showEvent` before they are painted".  That
    branch is what makes leaving the page alone during construction safe,
    so it is asserted here rather than assumed.
    """
    host = QWidget()
    qtbot.addWidget(host)
    host.setLayout(QVBoxLayout())
    page = QWidget()
    theme.mark_as_a_sheet_target(page)
    host.layout().addWidget(page)
    host.show()
    qtbot.waitExposed(host)
    assert _wears_the_sheet(page)


def test_a_child_added_later_wears_its_windows_sheet_by_inheritance(
        sheeted_app, qtbot):
    """The other half of why nothing goes unsheeted.

    A widget that gets a parent instead of a `Show` is a descendant of a
    window that already carries the sheet, and QSS reaches descendants --
    which is how every one of these widgets was styled when the sheet lived
    on the QApplication.  The colour is read out of the palette rather than
    off the widget, because inheritance leaves no text on the child.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.setLayout(QVBoxLayout())
    window.show()
    qtbot.waitExposed(window)
    late = QLabel("Cell diameter")
    window.layout().addWidget(late)
    assert not _wears_the_sheet(late)
    assert _resolved(late) == SHEET_HEX


# -- the discriminator itself ------------------------------------------------

def test_a_real_window_has_its_native_handle_by_the_time_it_is_polished(
        sheeted_app, qtbot):
    """WHY `WA_WState_Created` IS IN THE TEST AND NOT JUST THE PARENT.

    A top-level window that is genuinely opening has been created by the
    time Qt polishes it, so it never looks mid-construction even though its
    window type is the same plain `Qt.Window`.
    """
    window = QWidget()
    qtbot.addWidget(window)
    assert theme._a_window_for_want_of_a_parent(window)
    window.show()
    qtbot.waitExposed(window)
    assert window.testAttribute(Qt.WA_WState_Created)
    assert not theme._a_window_for_want_of_a_parent(window)


def test_a_window_that_has_been_shown_once_is_never_mid_construction_again(
        sheeted_app, qtbot):
    """`WA_WState_Created`, ON ITS OWN.

    A dialog that is closed and opened again is parentless, invisible and
    plainly `Qt.Window` -- every other clause of the test says
    mid-construction -- and it is nothing of the kind.  Qt never takes the
    native handle back, so that is the clause that tells the two apart, and
    a theme change between the two openings has to reach it.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.hide()
    assert window.parent() is None
    assert window.windowType() == Qt.Window
    assert not window.isVisible()
    assert not theme._a_window_for_want_of_a_parent(window)


def test_a_child_window_with_a_parent_is_still_a_window_to_sheet(sheeted_app,
                                                                 qtbot):
    """`parent() is None`, ON ITS OWN.

    `QWidget(parent, Qt.Window)` is a top-level window that keeps an owner
    -- which is how a tool window or a detached panel is built.  It is a
    window that appears, so it is sheeted at its `Polish` like any other,
    and the parent clause is what stops it being read as a widget waiting
    to be added to a layout.
    """
    owner = QWidget()
    qtbot.addWidget(owner)
    child_window = QWidget(owner, Qt.Window)
    assert child_window.isWindow()
    assert child_window.parent() is owner
    child_window.ensurePolished()
    assert _wears_the_sheet(child_window)


# -- what the app actually paints, which is not only a timing question -------

#: A registered block naming one widget, in a colour the sheet above does
#: not contain, so the two can never be confused for one another.
BLOCK = "QLabel#TheOneTheBlockNames { color: rgb(200, 100, 50); }"
BLOCK_HEX = "#c86432"


def test_a_registered_block_reaches_the_widget_it_names(sheeted_app, qtbot):
    """THE VISIBLE HALF OF THE CHANGE, AND THE REASON IT IS AN IMPROVEMENT.

    Sheeting a mid-construction widget does not only cost time. Qt resolves
    QSS FROM THE NEAREST STYLESHEET FIRST, so a copy of the sheet left on an
    intermediate widget beats a more specific rule on the screen root. And
    those copies carry the GLOBAL sheet ONLY -- the blocks
    :func:`register_widget_qss` appends live on the screen root alone -- so
    every widget a registered block names was painted the plain `QLabel`
    colour instead of its own.

    MEASURED ON THE REAL WINDOW with mask, measure and regression open, one
    process each way: 31 widgets change colour when the copies go, every one
    of them from `fg` to the colour its own block asks for --
    `QLabel#ChainingStale` #0d0e10 -> #8f4e00 (`warning`), `#ChainingSource`
    -> #4b5460 (`fg_muted`), `#ChainingFix`, `#ChainingPinned` and
    `#SettingsSearchCount` -> #68707e (`fg_dim`) -- and 42 widgets move,
    seven of them visibly, because a label sized by its own block is a
    different height.

    NO TEST HELD ANY OF THOSE COLOURS, which is why nothing went red in
    EITHER direction and why this test exists. The bug was invisible to the
    suite while it was there, and the fix would be equally invisible if it
    were reverted.

    The tree here is the one spaCR builds: a widget made parentless, given
    its `Polish` in that state, and only then added to the screen.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    theme.register_widget_qss("TheOneTheBlockNames",
                              lambda palette, opacity: BLOCK, replace=True)
    try:
        theme.ensure_widget_qss_applied("TheOneTheBlockNames", root=window)
        assert BLOCK in (window.styleSheet() or ""), (
            "the block never reached the root, so this test would be about "
            "nothing at all")

        middle = QWidget()
        middle.ensurePolished()          # the Polish, while it has no parent
        named = QLabel("x", middle)
        named.setObjectName("TheOneTheBlockNames")
        layout = QVBoxLayout(window)
        layout.addWidget(middle)
        QApplication.instance().processEvents()

        assert _resolved(named) == BLOCK_HEX, (
            "the widget the block names is painting the generic colour, so "
            "an intermediate widget is carrying a copy of the global sheet "
            "and winning on proximity")
    finally:
        theme.unregister_widget_qss("TheOneTheBlockNames")


def test_the_block_probe_can_tell_the_two_colours_apart(sheeted_app, qtbot):
    """PROOF THE TEST ABOVE IS NOT ASSERTING A CONSTANT.

    Same probe, same registration, but the intermediate widget is handed a
    copy of the global sheet BY HAND -- which is exactly what the filter
    used to do to it. If the assertion above can fail, it fails here.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    theme.register_widget_qss("TheOneTheBlockNames",
                              lambda palette, opacity: BLOCK, replace=True)
    try:
        theme.ensure_widget_qss_applied("TheOneTheBlockNames", root=window)
        middle = QWidget()
        named = QLabel("x", middle)
        named.setObjectName("TheOneTheBlockNames")
        layout = QVBoxLayout(window)
        layout.addWidget(middle)
        middle.setStyleSheet(window.styleSheet().replace(BLOCK, ""))
        QApplication.instance().processEvents()

        assert _resolved(named) == SHEET_HEX, (
            "a copy of the global sheet one layout deep did NOT beat the "
            "root's block, so the mechanism this file is about does not "
            "exist and the test above proves nothing")
    finally:
        theme.unregister_widget_qss("TheOneTheBlockNames")


# -- the contract a renderer has to know about ------------------------------

def test_an_unshown_orphan_renders_without_the_sheet_and_that_is_the_deal(
        sheeted_app, qtbot):
    """THE BEHAVIOUR CHANGE, STATED, because it cost a bisect to rediscover.

    The skip's premise is that a widget which is neither shown nor parented
    is not rendered by anything. `QWidget.render()` DISPROVES THAT: a test,
    a thumbnailer or a report generator can render an orphan directly, and
    it now renders with no stylesheet at all.

    `tests/qt/test_field_fade.py` did exactly that -- built a bare
    `QLineEdit()`, never shown, never parented, and rendered it -- and
    twelve of its assertions read a flat 255 where they expected a fade.
    Nothing about fields had changed; the widget simply had no sheet. It was
    found by bisect, from the other session, hours after the change landed.

    MEASURED, all three shapes:

        parentless and never shown   left 255  right 255   no fade
        parented into a shown window left 255  right  10   fades
        parentless but shown         left 255  right  10   fades

    So the application is unaffected -- every field a user sees is parented
    into a form, or shown, or both -- and the only shape that loses the
    sheet is one no screen builds.

    THE RULE, for whoever renders a widget next: SHOW IT OR PARENT IT. One
    line, either one, and the sheet arrives. This test exists so that rule
    is discoverable by grep instead of by bisect.
    """
    orphan = QLabel("x")
    qtbot.addWidget(orphan)
    orphan.ensurePolished()
    assert not _wears_the_sheet(orphan), (
        "an unshown orphan carries a sheet, so the skip is not applying and "
        "this test no longer describes the code")

    orphan.show()
    QApplication.instance().processEvents()
    assert _wears_the_sheet(orphan), (
        "showing an orphan did not sheet it -- the escape route this "
        "contract promises does not work, which is worse than the skip")


def test_parenting_is_the_other_escape_route(sheeted_app, qtbot):
    """The second half of "show it or parent it", so both are held.

    A parented widget is a descendant of a window that carries the sheet,
    and QSS reaches descendants -- so it needs no sheet of its own. Asserting
    the RESOLVED colour rather than `styleSheet()` is the point: the widget
    is styled without carrying anything.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    QApplication.instance().processEvents()

    late = QLabel("x", window)
    QApplication.instance().processEvents()
    assert _resolved(late) == SHEET_HEX, (
        "a label parented into a sheeted window did not resolve to the "
        "sheet's colour, so parenting is not an escape route after all")
