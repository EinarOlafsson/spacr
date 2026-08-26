"""Standardize movement and resizing of spaCR modal dialogs.

Some desktop window managers attach a parented modal ``QDialog`` to its main
window. :func:`detach_from_window_manager` preserves ownership and modality
while presenting it as an independently movable top-level window.

The resizing helpers remove explicit minimum sizes when appropriate, wrap
content in a scroll area only when its layout prevents useful shrinking, add
a visible size grip, and preserve the dialog's natural opening size. Qt's own
specialized dialogs, simple message dialogs, and content that already scrolls
are left unchanged. :func:`install_the_dialog_filters` applies these rules to
new dialogs at application level.
"""

from __future__ import annotations

import logging
from typing import Any

_DIALOG_IMPLEMENTATION_NOTES = r"""

TWO THINGS EVERY DIALOG IN THE APPLICATION GETS, and neither is asked for
by the dialogs themselves: a window the WM will let the user DRAG, and a
window the user can RESIZE both ways. One application-wide filter does
both, because a rule applied by hand is a rule that holds until the next
dialog is written -- the first of these was called from six files while
more than twenty others opened dialogs without it.

--------------------------------------------------------------------------
ONE: A WINDOW THE USER CAN DRAG
--------------------------------------------------------------------------

A ``QDialog`` created with a parent is *transient-for* that parent, and
``exec()`` makes it modal. A window manager that implements **attached modal
dialogs** -- GNOME/Mutter does by default, and it is not alone -- treats that
combination as an invitation to glue the dialog to its parent: it is drawn
centred on the parent, it cannot be dragged anywhere else, and pulling at it
manipulates the parent window instead. On a maximised main window that reads
as "the app un-maximised itself when I tried to move the settings".

The tell is which dialogs misbehave. In spaCR, Preferences
(``PreferencesDialog(...).exec()``) and Annotate's settings
(``_SettingsDialog(...).exec()``) are attached; the UMAP search settings
(``UmapSearchSettingsDialog(panel).show()``) and the Mask and crop live
preview settings are not. The first two are modal and the rest are modeless
-- which is exactly the WM's rule, and nothing to do with how any of them
are built.

The fix is to stop advertising them as dialogs. Clearing ``Qt.Dialog`` from
the window type and setting ``Qt.Window`` makes the WM see an ordinary
top-level window with its own frame, which nothing attaches. The parent is
kept, so the dialog still stacks above the app and is still owned by it, and
the modality is untouched -- ``exec()`` still blocks and still returns
``Accepted``.

Not done instead:

* **Dropping the parent.** It detaches, but the dialog then stops stacking
  above the main window and can be lost behind it, and Qt no longer destroys
  it with its owner.
* **Making the dialogs modeless.** Every caller reads the ``exec()`` result
  to decide whether to apply the settings. Changing that is a different and
  much larger piece of work.
* **A platform check.** The flags are harmless on Windows and macOS -- a
  dialog with a normal window type is a normal window there too -- and a
  conditional would mean the layout differs by platform for no gain.

--------------------------------------------------------------------------
TWO: A WINDOW THE USER CAN RESIZE
--------------------------------------------------------------------------

Asked for as "i should be able to resize all settings windows horizontally
and verticaly", and nothing was stopping it in the way anyone expected.
There is one ``setFixedWidth`` in the application and it is not on a
settings window; no dialog sets a layout size constraint; the WM glue above
was already fixed.

THE FLOOR IS THE CONTENT. Qt sets a window's minimum size from its
layout's total minimum, so a dialog cannot be dragged in past the point
where every field is fully visible. Measured on ``PictureSettingsDialog``:
size 645x318, minimumSize 645x318, a resize to 300x200 answered 645x318. It
grew and could not shrink, and with no size grip there was not even a
handle saying it should be possible.

Three steps, in the order of what they cost, and the cheap one is enough
for a third of them:

* an explicit minimum the dialog set on ITSELF comes off first
  (:func:`drop_the_explicit_floor`) -- six dialogs have one, and it
  outranks anything the contents say;
* the contents move into a scroll area only if the dialog is STILL stuck
  (:func:`let_the_content_scroll`). A window smaller than its contents can
  only mean a window showing part of them;
* a size grip goes on every one of them (:func:`give_it_a_size_grip`), so
  the affordance is visible on a frameless window that has no corner drawn.

WHAT IS NOT TOUCHED, and each is a decision rather than an omission: Qt's
own dialogs, which lay out their own internals; a dialog that is a message
-- a sentence and two buttons has nothing to scroll and no room to give;
and a dialog whose contents are already a scroll area of its own, which
would gain a second set of scroll bars around the first.

AND IT OPENS AT THE SIZE IT ALWAYS DID (:func:`open_at_its_natural_size`).
The floor was load-bearing for that, and taking it away silently made wide
dialogs open narrow. Checked against every dialog in the sweep.
"""

LOG = logging.getLogger("spacr.qt.dialogs")


def detach_from_window_manager(dialog: Any) -> Any:
    """Prevent a window manager from attaching ``dialog`` to its parent.

    Call this before ``exec()`` when a modal dialog must remain independently
    movable. Repeated calls and dialogs without parents are supported.

    :param dialog: the ``QDialog`` (or any ``QWidget`` shown as a window).
    :returns: the same object, so it can be used inline.
    """
    try:
        from PySide6.QtCore import Qt
    except Exception:                       # pragma: no cover - headless import
        return dialog
    try:
        flags = dialog.windowFlags()
        # `Qt.Dialog` is `Qt.Window | 0x2`, so clearing it clears the Window
        # bit too and it has to be put back. Setting `Qt.Window` alone would
        # leave the dialog bit standing and change nothing.
        dialog.setWindowFlags((flags & ~Qt.WindowType.Dialog)
                              | Qt.WindowType.Window)
    except Exception:                       # pragma: no cover
        # Decoration must never be load-bearing (INVARIANTS 10): a dialog
        # that cannot be detached is still a dialog the user can use.
        pass
    return dialog


#: Marks a dialog this module has already taken in hand, so a dialog shown,
#: hidden and shown again is not rebuilt on every show.
RESIZABLE = "spacrMadeResizable"

#: Marks a dialog whose contents were moved into a scroll area. Separate
#: from :data:`RESIZABLE` because only some of them need it: a dialog with
#: a list or a text box in it already has slack, and the wrap is the
#: expensive half of this.
SCROLLS = "spacrContentScrolls"

#: Set on a dialog that must keep the layout its author gave it. Nothing in
#: spaCR sets it today; it is the same escape hatch ``spacrNoGlass`` is,
#: for the next dialog whose contents cannot survive being moved.
NO_SCROLL = "spacrNoScroll"

#: Marks a dialog that has not yet been given its opening size back, and
#: holds the floor it used to have. See :func:`open_at_its_natural_size`.
OPENS_AT = "spacrOpensAt"

#: How much of each side a settings window has to be able to lose before
#: this leaves its layout alone.
#:
#: A THIRD. The question a threshold answers is not "did the number
#: change" but "can the user put this window where they want it", and a
#: window that gives back fifty pixels answers no -- `SettingsAdvisorDialog`
#: measured 900 wide with a floor of 847, `UmapDisplaySettings` 234 with a
#: floor of 214. Both look resizable to a test that only asks whether the
#: floor is lower than the ceiling, and neither is resizable to a person.
SLACK = 1.0 / 3.0

#: The smallest a scrolled dialog's contents area may be made, per side.
#:
#: NOT ZERO, and not the content's own minimum either. Zero lets a window
#: be dragged down to nothing at all; the content's minimum is the floor
#: this exists to remove. This is a floor the user cannot get stuck at:
#: wide enough for the scroll bar and a readable strip beside it.
SMALLEST = 120


def _field_types():
    """The widget classes that count as a field, and the ones that nest.

    :returns: ``(fields, nesting)`` -- what to count, and what to look
        inside for children that must NOT be counted again.
    """
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QLineEdit,
        QPlainTextEdit,
        QRadioButton,
        QSlider,
        QTextEdit,
    )

    fields = (QAbstractItemView, QAbstractSpinBox, QCheckBox, QComboBox,
              QLineEdit, QPlainTextEdit, QRadioButton, QSlider, QTextEdit)
    # A spin box OWNS a line edit, an editable combo owns another, and an
    # item view owns a scroll bar and an editor. Counting those would make
    # one field look like three.
    nesting = (QAbstractItemView, QAbstractSpinBox, QComboBox, QLineEdit,
               QPlainTextEdit, QTextEdit)
    return fields, nesting


def fields_in(dialog) -> int:
    """Return the number of independent data-entry fields in ``dialog``.

    Editors nested inside another field, such as a spin box's line editor,
    are not counted separately. Push buttons are not considered fields.
    """
    from PySide6.QtWidgets import QWidget

    fields, nesting = _field_types()
    found = 0
    for child in dialog.findChildren(QWidget):
        if not isinstance(child, fields):
            continue
        parent = child.parentWidget()
        owned = False
        while parent is not None and parent is not dialog:
            if isinstance(parent, nesting):
                owned = True
                break
            parent = parent.parentWidget()
        if not owned:
            found += 1
    return found


def more_than_a_message(dialog) -> bool:
    """Return whether ``dialog`` contains resizable interactive content.

    A dialog qualifies when it contains at least one data-entry field or an
    existing scroll area. Simple confirmation and message dialogs do not.
    """
    from PySide6.QtWidgets import QAbstractScrollArea

    if fields_in(dialog):
        return True
    return bool(dialog.findChildren(QAbstractScrollArea))


def window_floor(dialog):
    """Return the explicit minimum size enforced for ``dialog``.

    Unlike ``minimumSizeHint()``, this value directly constrains manual and
    initial window resizing.
    """
    return dialog.minimumSize()


def content_floor(dialog):
    """Return the layout's minimum size for rendering without clipping.
    """
    return dialog.minimumSizeHint()


def is_stuck_at_its_contents(dialog) -> bool:
    """Return whether content prevents useful shrinking in either dimension.

    :data:`SLACK` defines the minimum proportional reduction required for a
    dialog to be considered usefully resizable. Call this before changing its
    layout or minimum size.
    """
    opening = dialog.sizeHint().expandedTo(window_floor(dialog))
    content = content_floor(dialog)
    return (content.width() > opening.width() * (1.0 - SLACK)
            or content.height() > opening.height() * (1.0 - SLACK))


def _already_scrolls(layout) -> bool:
    """Whether ``layout`` holds nothing but a scroll area of the dialog's own.

    Such a dialog scrolls already, and wrapping it would put a second set
    of scroll bars around the first.
    """
    from PySide6.QtWidgets import QAbstractScrollArea

    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    widgets = [w for w in widgets if w is not None]
    return len(widgets) == 1 and isinstance(widgets[0], QAbstractScrollArea)


#: Built on first use, because this module is imported where PySide6 is
#: not -- see the guarded import in `detach_from_window_manager`.
_SCROLL_CLASS = None


def _form_scroll_class():
    """The scroll area a wrapped dialog gets. Defined once, on first use."""
    global _SCROLL_CLASS
    if _SCROLL_CLASS is not None:
        return _SCROLL_CLASS
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QScrollArea

    class _FormScroll(QScrollArea):
        """A scroll area that keeps the dialog's size and drops its floor.

        BOTH HINTS ARE OVERRIDDEN, and each for a measured reason.

        `QScrollArea.sizeHint` is its widget's hint **bounded to 36x24 font
        heights** -- about 576x384 here. Wrapping a 645-pixel-wide settings
        dialog in a stock scroll area therefore makes it want to open 70
        pixels narrower than the form inside it, with a scroll bar already
        showing on a window nobody has touched. Passing the inner widget's
        hint through keeps the window the size it was.

        `QAbstractScrollArea.minimumSizeHint` is built from its scroll
        bars, which is small -- but small is not the same as *known*, and
        the floor is the whole point of this class. :data:`SMALLEST` says
        what it is.
        """

        def sizeHint(self):               # noqa: N802 - Qt naming
            inner = self.widget()
            if inner is None:
                return super().sizeHint()
            frame = 2 * self.frameWidth()
            hint = inner.sizeHint()
            return QSize(hint.width() + frame, hint.height() + frame)

        def minimumSizeHint(self):        # noqa: N802 - Qt naming
            return QSize(SMALLEST, SMALLEST)

    _SCROLL_CLASS = _FormScroll
    return _SCROLL_CLASS


_DRAG_CLASS = None


def _drag_class():
    """The filter that keeps a wrapped dialog draggable. Defined once."""
    global _DRAG_CLASS
    if _DRAG_CLASS is not None:
        return _DRAG_CLASS
    from PySide6.QtCore import QEvent, QObject, Qt

    class _DragTheWindowByTheForm(QObject):
        """Move the window by dragging the empty space between its fields.

        WITHOUT THIS, WRAPPING TAKES THE HANDLE AWAY. A glassed dialog is
        frameless -- there is no title bar -- and
        `spacr.qt.widgets.glass._DragByBackground` gives that back by
        starting a drag wherever `dialog.childAt(...)` answers None, which
        is the empty background between the controls.

        Putting the contents in a scroll area makes that answer the scroll
        area for the whole window, so every press lands on "a child" and
        the window stops moving. The empty space did not go anywhere -- it
        belongs to the holder widget now, so the same rule is applied
        there and the handle is back where it was.
        """

        def __init__(self, holder):
            super().__init__(holder)
            self._holder = holder
            self._grab = None
            holder.installEventFilter(self)

        def eventFilter(self, watched, event):    # noqa: N802 - Qt naming
            # `getattr`, for `_DragByBackground`'s reason: Qt goes on
            # delivering to a filter whose Python attributes are cleared.
            holder = getattr(self, "_holder", None)
            if holder is None or watched is not holder:
                return False
            try:
                kind = event.type()
                if (kind == QEvent.Type.MouseButtonPress
                        and event.button() == Qt.MouseButton.LeftButton):
                    if holder.childAt(event.position().toPoint()) is not None:
                        return False
                    window = holder.window()
                    self._grab = (event.globalPosition().toPoint()
                                  - window.frameGeometry().topLeft())
                    return True
                if kind == QEvent.Type.MouseMove and self._grab is not None:
                    holder.window().move(
                        event.globalPosition().toPoint() - self._grab)
                    return True
                if (kind == QEvent.Type.MouseButtonRelease
                        and self._grab is not None):
                    self._grab = None
                    return True
            except Exception:                     # noqa: BLE001
                LOG.debug("a drag from the form went wrong", exc_info=True)
                self._grab = None
            return False

    _DRAG_CLASS = _DragTheWindowByTheForm
    return _DRAG_CLASS


def _qts_own_dialogs():
    """The Qt dialogs that lay out their own internals. Not ours to move.

    BY TYPE, NOT BY MODULE, and the difference is Preferences. The first
    draft asked whether the class came from a ``spacr`` module, which
    reads well and excludes the most important settings window in the
    application: ``PreferencesDialog`` is a factory that builds a PLAIN
    ``QDialog`` and fills it, so its class is Qt's while every one of its
    thirty controls is spaCR's. Seven more windows are built the same way
    -- the shortcut sheet, the settings diff, the sweep panel's editor,
    the montage view's and the drag-and-drop prompts.

    What actually must be left alone is a dialog whose CONTENTS are Qt's:
    a file chooser, a message box, a colour or font picker, an input
    prompt, a progress dialog, a wizard. Each of those arranges its own
    children and documents behaviour that moving them would break.
    """
    from PySide6.QtWidgets import (
        QColorDialog,
        QErrorMessage,
        QFileDialog,
        QFontDialog,
        QInputDialog,
        QMessageBox,
        QProgressDialog,
        QWizard,
    )

    return (QColorDialog, QErrorMessage, QFileDialog, QFontDialog,
            QInputDialog, QMessageBox, QProgressDialog, QWizard)


def wants_resizing(dialog) -> bool:
    """Return whether spaCR should add standardized resizing to ``dialog``.

    The dialog must contain interactive content and must not be a specialized
    Qt dialog, explicitly exempt, or already processed.
    """
    from PySide6.QtWidgets import QDialog

    if not isinstance(dialog, QDialog):
        return False
    if isinstance(dialog, _qts_own_dialogs()):
        return False
    if dialog.property(NO_SCROLL) or dialog.property(RESIZABLE):
        return False
    layout = dialog.layout()
    if layout is None or layout.count() == 0:
        return False
    return more_than_a_message(dialog)


def drop_the_explicit_floor(dialog) -> bool:
    """Clear an explicit minimum size and report whether one was present.

    The original value can be retained separately and passed to
    :func:`open_at_its_natural_size` so the initial window size is preserved.
    """
    from PySide6.QtCore import QSize

    explicit = dialog.minimumSize()
    if explicit == QSize(0, 0):
        return False
    # ZERO CLEARS IT rather than setting one of its own: Qt tracks whether
    # a minimum was set explicitly, and a zero on either axis takes that
    # mark off, which puts the axis back under the layout's control.
    dialog.setMinimumSize(0, 0)
    return True


def let_the_content_scroll(dialog) -> bool:
    """Move a dialog's layout into a resizable scroll area.

    The existing layout and widgets are transferred to a transparent holder,
    while the original outer margins remain on the dialog. The holder expands
    with the viewport, and content exceeding the current window size scrolls.

    :returns: ``True`` after the content has been moved.
    """
    from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

    from .theme import make_transparent

    layout = dialog.layout()
    margins = layout.getContentsMargins()
    layout.setContentsMargins(0, 0, 0, 0)

    holder = QWidget()
    # Steals the layout from the dialog, and the fields come with it.
    holder.setLayout(layout)

    scroll = _form_scroll_class()(dialog)
    # NO FRAME. A sunken border round the whole form is a box drawn inside
    # a card that already has one, and its straight edges are the most
    # visible thing on a translucent window.
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setWidget(holder)

    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(*margins)
    outer.setSpacing(0)
    outer.addWidget(scroll)

    # THE NEW CONTAINERS PAINT NOTHING. `glass.clear_the_containers` walks
    # the dialog when the card goes in, and it has already run by the time
    # this does -- a scroll area added afterwards is an untagged QWidget,
    # and in a palette whose `bg` is #000000 that is a black rectangle over
    # the card. The viewport is tagged with it; `make_transparent` knows.
    make_transparent(scroll, holder)
    _drag_class()(holder)
    dialog.setProperty(SCROLLS, True)
    return True


def open_at_its_natural_size(dialog) -> bool:
    """Apply a stored natural opening size once and clear the stored value.

    This is called after Qt's initial size adjustment so removing a minimum
    size does not cause a dialog to open smaller than its original layout.
    Later shows retain the size selected by the user.

    :returns: ``True`` if a stored size was applied.
    """
    floor = dialog.property(OPENS_AT)
    dialog.setProperty(OPENS_AT, None)
    if floor is None:
        return False
    dialog.resize(dialog.size().expandedTo(floor))
    return True


def give_it_a_size_grip(dialog) -> bool:
    """Enable a transparent corner size grip on ``dialog``.

    :returns: ``True`` if the dialog contains a size-grip widget.
    """
    from PySide6.QtWidgets import QSizeGrip

    from .theme import make_transparent

    dialog.setSizeGripEnabled(True)
    grips = dialog.findChildren(QSizeGrip)
    # The grip is a child of the DIALOG, added after glass tagged the
    # containers, so without this it paints the palette's flat background
    # as a square in the one corner a rounded card is most visible.
    make_transparent(*grips)
    return bool(grips)


def make_the_window_resizable(dialog) -> bool:
    """Apply standardized resizing behavior to one eligible dialog.

    Explicit minimum sizes are cleared first. If the content still prevents
    useful shrinking and does not already scroll, it is moved into a scroll
    area. A size grip is then enabled. Repeated calls leave the dialog
    unchanged.

    :returns: ``True`` if the dialog was eligible and processed.
    """
    if not wants_resizing(dialog):
        return False
    dialog.setProperty(RESIZABLE, True)
    try:
        # THE LAYOUT HAS TO HAVE RUN before its floor can be read. Qt sets
        # a window's minimum size from `QLayout.activate`, and on Polish
        # that has not always happened yet: reading it first answered 0x0
        # for six of these dialogs, and a floor of zero is the same as not
        # remembering one -- they opened at two thirds of the screen.
        dialog.layout().activate()
        floor = window_floor(dialog)
        stuck = is_stuck_at_its_contents(dialog)
        lowered = drop_the_explicit_floor(dialog)
        if stuck and not _already_scrolls(dialog.layout()):
            lowered = let_the_content_scroll(dialog) or lowered
        if lowered:
            # The floor it used to open at, kept for the Show that is on
            # its way. See `open_at_its_natural_size`.
            dialog.setProperty(OPENS_AT, floor)
        give_it_a_size_grip(dialog)
        return True
    except Exception:                       # noqa: BLE001
        # INVARIANTS 10: a dialog that cannot be made smaller is still a
        # dialog the user can use. The marker is set above rather than
        # here, so a dialog this failed on half-way is not tried again on
        # its next show -- the half it did is already done.
        LOG.debug("could not make a dialog resizable", exc_info=True)
        return False


#: The property `spacr.qt.widgets.glass` sets on a dialog whose flags it
#: has already rewritten. Named here rather than imported, because
#: importing the widgets package from this one would be a cycle.
_GLASS_DETACHED = "spacrDetached"


class _DetachEveryDialog:
    """Application-wide filter: every dialog is a window the user can drag,
    and one the user can resize.

    Asked 2026-08-21: "the settings for your data settings window should be
    movable without moving the main window. this should be tru of all
    settings windows or any popup window from spacr."

    ONE PLACE, NOT ONE CALL SITE PER DIALOG. :func:`detach_from_window_manager`
    already existed and was being called from six files while more than
    twenty others opened dialogs without it -- which is what "all settings
    windows" cannot be built out of. A rule applied by hand is a rule that
    holds until the next dialog is written.

    HOOKED ON `Polish`, WHICH IS THE ONLY MOMENT THAT WORKS.
    `setWindowFlags` on a widget that is already visible destroys and
    recreates its native window, so Qt hides it -- detaching on `Show` makes
    the dialog flash or vanish. `Polish` is delivered during the first show
    sequence and BEFORE the window is mapped, so the flags are already right
    when it appears. Measured order on PySide6: WinIdChange, Polish, Show,
    ShowToParent.

    Each dialog is detached ONCE. The flag change is idempotent, but doing
    it repeatedly on a dialog that is shown, hidden and shown again would
    recreate the native window every time and lose its position -- which is
    the thing this exists to protect.

    AND THE RESIZING RIDES ON THE SAME EVENT, for the same reason there is
    one filter and not one call per dialog. It is NOT held to Polish,
    though: moving a layout into a scroll area is an ordinary layout change
    rather than a window recreation, so Show is a safe fallback for a
    dialog that builds its form after its first polish. The size a dialog
    OPENS at is restored on Show and only on Show -- see
    :func:`open_at_its_natural_size`, where the measurement is.
    """

    def __init__(self):
        self._done: set = set()

    def eventFilter(self, obj, event):        # noqa: N802 - Qt naming
        try:
            from PySide6.QtCore import QEvent
            from PySide6.QtWidgets import QDialog

            polished = event.type() == QEvent.Type.Polish
            if polished and isinstance(obj, QDialog):
                key = id(obj)
                # ALREADY DONE BY THE GLASS INSTALLER, which detaches and
                # goes frameless in ONE flags change. Doing it again here
                # recreates the native window a second time, and on some
                # window managers what comes back has square opaque
                # corners behind the rounded card.
                if obj.property(_GLASS_DETACHED):
                    self._done.add(key)
                elif key not in self._done:
                    self._done.add(key)
                    detach_from_window_manager(obj)
            # AND THE SAME EVENT MAKES IT RESIZABLE. Polish OR Show: the
            # flags above may only be rewritten before the window is
            # mapped, but moving the contents into a scroll area is an
            # ordinary layout change and is safe either way -- which
            # matters for a dialog that builds its form after its first
            # polish, and would otherwise never be reached.
            if (polished or event.type() == QEvent.Type.Show) \
                    and wants_resizing(obj):
                make_the_window_resizable(obj)
            # AND THE SIZE IT OPENS AT IS PUT BACK ON Show, which is the
            # first moment one sticks. See `open_at_its_natural_size`.
            if event.type() == QEvent.Type.Show \
                    and obj.property(OPENS_AT) is not None:
                open_at_its_natural_size(obj)
        except Exception:                     # pragma: no cover
            # INVARIANTS 10 again: a dialog that cannot be detached is still
            # a dialog. This filter sees every event in the application and
            # must never be the reason one of them is lost.
            pass
        return False


#: Kept alive for the life of the application. An event filter that is
#: garbage collected stops filtering, silently.
_DETACHER = None

#: WHICH application it was installed on. Tracking the filter alone was a
#: bug: a filter belongs to one `QApplication`, so when the application is
#: torn down and rebuilt -- which every Qt test session does, and which a
#: relaunch inside one process does too -- the filter goes with it while
#: `_DETACHER` stayed non-None. The next call then reported "already
#: installed" and returned, leaving nothing filtering and nothing saying so.
#:
#: Found by the suite: two of this module's own tests passed alone and
#: failed in a full run, which is the signature of state surviving a
#: teardown it should not have.
_DETACHED_APP = None


def detach_all_dialogs(app) -> bool:
    """Install the application-wide detacher. Returns True if it installed.

    Idempotent PER APPLICATION: calling it twice on the same app leaves one
    filter, and calling it on a NEW app installs again, because the old
    filter died with the old app.
    """
    global _DETACHER, _DETACHED_APP
    if app is None:
        return False
    if _DETACHER is not None and _DETACHED_APP is app:
        return False
    try:
        from PySide6.QtCore import QObject

        # ONE CLASS, NOT A MIX-IN, and that is a correctness fix rather than
        # a style one. `class F(QObject, _DetachEveryDialog)` puts QObject
        # first in the MRO, so QObject's own `eventFilter` -- which returns
        # False and does nothing -- wins over the one below it. The filter
        # installed, reported success, and silently never fired.
        class _Filter(QObject):
            def __init__(self):
                super().__init__()
                self._inner = _DetachEveryDialog()

            def eventFilter(self, obj, event):    # noqa: N802 - Qt naming
                return self._inner.eventFilter(obj, event)

        _DETACHER = _Filter()
        app.installEventFilter(_DETACHER)
        _DETACHED_APP = app
        return True
    except Exception:                         # pragma: no cover
        _DETACHER = None
        _DETACHED_APP = None
        return False
