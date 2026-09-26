"""Screen widgets — one per spacr app.

NOTHING IS IMPORTED HERE. This package used to import nine screens at the top,
because a screen that owned its registry row had to be executed for the row to
exist, and this package is on the path of every screen the window builds — so
importing one screen imported nine, and with them pandas and everything under
it, before the window had drawn.

A row does not need the screen any more. :mod:`spacr.qt.app_catalog` declares
the key, the name, the sentence, the section, the stage and the NAME of the
screen factory, and :func:`spacr.qt.app_catalog.register_declared` registers
all of that without importing anything; the screen is imported the first time
somebody opens it. A screen that contributes a stylesheet block registers it
at that import, and ``MainWindow`` applies missing registered blocks to the
new screen's own QSS scope before inserting it into the visible stack.
:data:`spacr.qt.theme.WIDGET_QSS_MODULES` remains the inventory used when an
exhaustive/static stylesheet is explicitly requested.

So: to add a screen, declare its row in ``app_catalog`` and — if it registers
a QSS block — name it in ``WIDGET_QSS_MODULES`` so exhaustive sheets remain
complete. Do not import it here merely for startup styling.
"""

#: Whether an ``AppScreen`` constructed now leaves each closed settings
#: category unbuilt until it is opened. Here rather than on ``AppScreen``
#: because the window sets it around EVERY screen it builds, Home included,
#: and reaching the class would import the module screens -- pandas and all
#: -- while Home is being drawn. See ``AppScreen._build_a_waiting_heading``.
_categories_wait_to_be_opened = False

_a_window_is_opening_a_screen = False
_last_breath_at = 0.0
BREATH_AFTER_S = 0.15
_SECOND_PASS_AFTER_S = 0.1


def _start_breathing_while_a_window_opens(started_at: float) -> None:
    """Allow breaths for the open that began at ``started_at``.

    Called only by ``MainWindow._on_nav_selected``, which also holds back
    any navigation that arrives during a breath. A screen built anywhere
    else -- a test, a rebuild after a preference change, a folded panel --
    is built in one piece exactly as before.

    :param started_at: ``time.perf_counter()`` when the open began, which
        counts as the last breath.
    """
    global _a_window_is_opening_a_screen, _last_breath_at
    _a_window_is_opening_a_screen = True
    _last_breath_at = started_at


def _stop_breathing_while_a_window_opens() -> None:
    """End the breaths; screens built after this are built in one piece."""
    global _a_window_is_opening_a_screen
    _a_window_is_opening_a_screen = False


def _breathe_while_a_window_opens() -> None:
    """Run the event loop, if the window has asked for breaths and it is due.

    FOR THE LARGE STEPS OF A SCREEN'S CONSTRUCTION, not for every widget:
    a breath is one pass of pending timers, paints and posted events, which
    is what keeps the ambient backdrop drawing and the event loop answering
    while a heavy module is assembled (item 284). It is taken only when the
    work since the last one ran for :data:`BREATH_AFTER_S` or more. User
    input is held back, so nothing can be clicked on a half-built page, as
    in the settings panel's own breaths.

    A breath costs a paint of the whole window with the backdrop behind it,
    so a light screen whose steps are all shorter than
    :data:`BREATH_AFTER_S` opens with none and pays nothing for them.

    A SECOND PASS WHEN THE FIRST WAS LONG (:data:`_SECOND_PASS_AFTER_S`), because one pass can run the
    event loop's own timers BEFORE the posted work the step left behind --
    a new page's first ``Polish``, which is when it is given the window's
    stylesheet, or the late caption pass. That work then joined the next
    step in one freeze (measured on a Mask open: the sheet and the
    translation pass after it in one gap, with no timer between them).
    """
    global _last_breath_at
    if not _a_window_is_opening_a_screen:
        return
    import time

    if time.perf_counter() - _last_breath_at < BREATH_AFTER_S:
        return
    from PySide6.QtCore import QCoreApplication, QEventLoop

    flags = QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
    started = time.perf_counter()
    QCoreApplication.processEvents(flags)
    if time.perf_counter() - started >= _SECOND_PASS_AFTER_S:
        QCoreApplication.processEvents(flags)
    _last_breath_at = time.perf_counter()
