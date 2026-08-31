"""Find WHICH connect triggers libpyside's "No Wrapper found" warning.

Run this on the machine that shows the warning:

    python tools/diagnose_pyside_slot_warning.py

Background. Opening Mask Generation on macOS prints a dozen lines like

    libpyside: addMetaMethod: Cannot add dynamic method "_on_use_offered()"
    (2) to QWidget/"ChainingBar": No Wrapper found.

and the same build on Linux -- same spaCR commit, same PySide6 6.11.2 --
prints none. So it is the platform, not the version, and it cannot be
reproduced away from a Mac.

The message names the SLOT and the receiver but not the line that asked
for it, which is the one thing needed to fix it. This script supplies
that: the Qt message handler runs synchronously, in the emitting thread,
at the moment of the warning, so the Python stack at that instant still
contains the `.connect(...)` call underneath.

It also answers the question the warning raises. PySide registers a
dynamic slot on the receiver so a connection dies with the receiver;
"cannot add" means that registration failed. Whether the connection is
then genuinely untracked decides whether this is cosmetic or a crash
waiting for the receiver to outlive its sender, and it can only be
measured where the warning actually fires.

CAUSE FOUND, 2026-08-31 -- see Part 5, which is the record for it. It is
not the child at all: a ``ChildAdded`` event filter on the PARENT calls
``event.child()`` while the child's C++ constructor is still running, and
that mints a bare ``QWidget`` wrapper which permanently displaces the real
one. Parts 1 to 4 are kept because they are what ruled the child out, and
because they are the shape anyone re-opening this will reach for first.

Prints a report and exits. Opens no window you have to close.
"""
from __future__ import annotations

import sys
import traceback

WARNING_MARK = "addMetaMethod"


def main() -> int:
    from PySide6.QtCore import (QEvent, QObject, Qt, QTimer, Signal,
                                qInstallMessageHandler)
    from PySide6.QtWidgets import QApplication, QWidget

    import PySide6

    print(f"PySide6 {PySide6.__version__} on {sys.platform}\n")

    #: (message, formatted Python stack) for every warning seen.
    caught: list[tuple[str, str]] = []

    def handler(_mode, _context, message):
        text = message or ""
        if WARNING_MARK in text:
            # -1 drops this handler's own frame.
            caught.append((text, "".join(traceback.format_stack()[:-1])))

    qInstallMessageHandler(handler)
    app = QApplication.instance() or QApplication([])

    # ---- Part 1: where does it come from --------------------------------
    print("=" * 72)
    print("PART 1  Building Mask Generation and watching for the warning")
    print("=" * 72)
    try:
        from spacr.qt.app import MainWindow

        window = MainWindow()
        window.show()
        app.processEvents()
        for key in ("mask", "make_masks", "measure"):
            try:
                window._on_nav_selected(key)
                app.processEvents()
            except Exception as error:                   # noqa: BLE001
                print(f"  (could not open {key}: {error})")
    except Exception as error:                           # noqa: BLE001
        print(f"  Could not build the window: {error!r}")

    if not caught:
        print("\n  NO WARNING SEEN. Nothing to diagnose on this machine --\n"
              "  run this where the warning actually appears.")
    else:
        print(f"\n  {len(caught)} warning(s). The Python frames under each,")
        print("  innermost last -- the `.connect(...)` is the spacr line\n"
              "  closest to the bottom.\n")
        for index, (message, stack) in enumerate(caught, 1):
            print("-" * 72)
            print(f"[{index}] {message.strip()}")
            spacr_frames = [line for line in stack.splitlines()
                            if "spacr" in line and "diagnose_pyside" not in line]
            if spacr_frames:
                for line in spacr_frames[-6:]:
                    print("   ", line.strip())
            else:
                print("    (no spacr frame -- raised from inside Qt's own C++,")
                print("     so the connect that set it up has already returned)")

    # ---- Part 2: is the connection actually untracked --------------------
    print()
    print("=" * 72)
    print("PART 2  Does a connection survive its receiver being destroyed")
    print("=" * 72)
    print("  If PySide could not register the tracking slot, the connection")
    print("  may outlive the receiver -- which is a crash, not noise.\n")

    class Sender(QObject):
        fired = Signal()

    class Receiver(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.hits = 0

        def _on_fired(self) -> None:
            self.hits += 1

    sender = Sender()
    receiver = Receiver()
    sender.fired.connect(receiver._on_fired)
    sender.fired.emit()
    print(f"  before destroy: receiver saw {receiver.hits} emit(s)"
          f"  {'OK' if receiver.hits == 1 else 'UNEXPECTED'}")

    receiver.deleteLater()
    app.processEvents()
    del receiver
    app.processEvents()

    try:
        sender.fired.emit()
    except Exception as error:                           # noqa: BLE001
        print(f"  after destroy:  emit RAISED {type(error).__name__}: {error}")
        print("\n  VERDICT: the connection outlived its receiver. NOT cosmetic.")
        return 1
    print("  after destroy:  emit survived, connection cleanly gone")

    trailing = [m for m, _s in caught]
    if any(WARNING_MARK in m for m in trailing[len(caught):]):
        print("\n  ...but note the warning fired during THIS test too.")
    print("\n  VERDICT: no crash in the simple case. Read with care -- this")
    print("  only proves the SIMPLE shape is safe. If Part 1 found warnings")
    print("  and this part did not, the failing registration is specific to")
    print("  how those widgets are built, not to connecting in general.")

    # ---- Part 3: is it connecting from inside __init__ -------------------
    print()
    print("=" * 72)
    print("PART 3  Does connecting from inside __init__ cause it")
    print("=" * 72)
    print("  Every warning reported so far names a slot connected from")
    print("  inside a widget's own __init__ -- the ambient backdrop's")
    print("  timer, ChainingBar's buttons, DiameterPanel's, MeasureQC's.")
    print("  If Shiboken has not registered the wrapper until __init__")
    print("  RETURNS, that one fact explains all of them, and moving the")
    print("  connect one step later is the whole fix.\n")

    class ConnectsInInit(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._on_tick)   # <- inside __init__

        def _on_tick(self) -> None:
            pass

    class ConnectsAfterInit(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.timer = QTimer(self)

        def wire(self) -> None:
            self.timer.timeout.connect(self._on_tick)   # <- after __init__

        def _on_tick(self) -> None:
            pass

    before = len(caught)
    inside = ConnectsInInit()
    during_init = len(caught) - before
    print(f"  connect INSIDE __init__ : {during_init} warning(s)"
          f"   {'<-- REPRODUCED' if during_init else 'clean'}")

    before = len(caught)
    outside = ConnectsAfterInit()
    outside.wire()
    after_init = len(caught) - before
    print(f"  connect AFTER  __init__ : {after_init} warning(s)"
          f"   {'still warns' if after_init else '<-- CLEAN'}")

    print()
    if during_init and not after_init:
        print("  VERDICT: CONFIRMED. The wrapper does not exist until")
        print("  __init__ returns, and every reported warning is a connect")
        print("  made before that. The fix is to wire signals one step")
        print("  later -- from the factory that builds the widget, or from")
        print("  showEvent -- not to silence the warning.")
    elif during_init and after_init:
        print("  VERDICT: NOT the whole story. Both shapes warn, so the")
        print("  timing of __init__ is not what decides it. Do not move the")
        print("  connects; that would be churn for nothing.")
    elif not during_init:
        print("  VERDICT: neither shape warns HERE, yet Part 1 did. So it")
        print("  is not connect-in-__init__ on its own -- something about")
        print("  how those particular widgets are built matters too.")
        print("  Compare against what Part 1 found before concluding.")

    # ---- Part 4: the shape the real code actually uses -------------------
    print()
    print("=" * 72)
    print("PART 4  A child built while its PARENT is still in __init__")
    print("=" * 72)
    print("  Part 3's widgets were standalone. The real one is not:")
    print("  AppScreen.__init__ calls _install_ambient, which calls")
    print("  install_ambient(self, ...), which does AmbientWidget(host).")
    print("  So the child is constructed WITH A PARENT that has not")
    print("  finished its own __init__ -- and a parent mid-construction")
    print("  may not be in Shiboken's binding manager yet.\n")

    class Child(QWidget):
        """Built by Host, connects to itself, exactly like AmbientWidget."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._on_tick)

        def _on_tick(self) -> None:
            pass

    class HostBuildsChildInInit(QWidget):
        """AppScreen's shape: builds the child from inside __init__."""

        def __init__(self) -> None:
            super().__init__()
            self.child = Child(self)

    class HostBuildsChildAfterInit(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.child = None

        def build(self) -> None:
            self.child = Child(self)

    shapes = (
        ("child with NO parent, standalone      ", lambda: Child(None)),
        ("child WITH parent, parent already made",
         lambda: Child(QWidget())),
        ("child built INSIDE parent's __init__  ",
         HostBuildsChildInInit),
        ("child built AFTER parent's __init__   ",
         lambda: (lambda h: (h.build(), h)[1])(HostBuildsChildAfterInit())),
    )
    results = []
    for label, make in shapes:
        before = len(caught)
        keep = make()                                    # noqa: F841
        count = len(caught) - before
        results.append((label, count))
        print(f"  {label} : {count} warning(s)"
              f"   {'<-- REPRODUCED' if count else 'clean'}")

    print()
    guilty = [label.strip() for label, count in results if count]
    if not guilty:
        print("  VERDICT: none of the four shapes reproduces it, yet Part 1")
        print("  does. So the trigger is something more specific than how")
        print("  the widget is parented -- report these numbers rather")
        print("  than changing any connect.")
    elif len(guilty) == len(results):
        print("  VERDICT: every shape warns. The trigger is connecting to")
        print("  self at all on this build, not where or when.")
    else:
        print("  VERDICT: reproduced by exactly these shapes:")
        for label in guilty:
            print(f"    - {label}")
        print("  That names the condition, and the fix is to stop building")
        print("  the widget that way -- not to silence the warning.")

    # ---- Part 5: the cause ----------------------------------------------
    print()
    print("=" * 72)
    print("PART 5  event.child() inside a ChildAdded filter")
    print("=" * 72)
    print("  Parts 3 and 4 asked what is different about the CHILD. Nothing")
    print("  is. The difference is the PARENT, and the mechanism is this:")
    print()
    print("    ChildAdded is delivered SYNCHRONOUSLY from inside the child's")
    print("    C++ constructor, before Shiboken has registered the wrapper")
    print("    for the Python class being built. A filter that calls")
    print("    event.child() at that moment forces Shiboken to mint a BARE")
    print("    QWidget wrapper and register THAT under the pointer. Shiboken")
    print("    never overwrites an existing key, so the real wrapper is")
    print("    never registered; when the bare one is collected it takes the")
    print("    entry with it, and the pointer is left mapped to nothing for")
    print("    the rest of the process.")
    print()
    print("  `AppScreen._watch_for_late_captions` installs exactly such a")
    print("  filter (`_LateCaptionTranslator`) on the screen itself, and it")
    print("  runs BEFORE `_install_ambient` in the same __init__ -- which is")
    print("  why the backdrop, parented into the screen, is the one that")
    print("  warns. app_screen.py's own docstring already describes the bare")
    print("  wrapper; it was read as a quirk to work around rather than as")
    print("  this warning's cause.\n")

    class TouchesTheChild(QObject):
        """What _LateCaptionTranslator.eventFilter does."""

        def __init__(self) -> None:
            super().__init__()
            self.held = []

        def eventFilter(self, watched, event):          # noqa: N802
            if event.type() == QEvent.ChildAdded:
                self.held.append(event.child())
            return False

    class DropsTheChild(QObject):
        """Asks for the child and throws it away again immediately.

        THIS IS THE CONTROL THAT DECIDES IT, and the one a reconstruction
        of this bug is most likely to write by accident. Calling
        event.child() is NOT enough on its own: the bare wrapper it mints
        is collected straight away, and collecting it releases the key it
        took, so the real registration that follows succeeds. The damage
        needs the bare wrapper to still be ALIVE when Shiboken registers
        the real one -- which is what `partial(self._on_arrival, child)`
        inside a singleShot does.
        """

        def eventFilter(self, watched, event):          # noqa: N802
            if event.type() == QEvent.ChildAdded:
                event.child()
            return False

    class IgnoresTheChild(QObject):
        """The same filter, without asking for the child."""

        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def eventFilter(self, watched, event):          # noqa: N802
            if event.type() == QEvent.ChildAdded:
                self.seen += 1
            return False

    class Ticker(QWidget):
        """A child of the reported shape: its own signal, its own timer.

        It also overrides a Qt virtual, because that is the consequence
        that actually broke a feature -- see below.
        """

        poked = Signal(int)

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.delivered = []
            self.shown = 0
            self.poked.connect(self.delivered.append)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._on_tick)

        def showEvent(self, event):                     # noqa: N802
            self.shown += 1
            super().showEvent(event)

        def _on_tick(self) -> None:
            pass

    keep = []                       # nothing here may be collected mid-test

    def under(filter_object):
        """Build a Ticker under a host watched by ``filter_object``."""
        host = QWidget()
        keep.append(host)
        if filter_object is not None:
            keep.append(filter_object)
            host.installEventFilter(filter_object)
        before = len(caught)
        child = Ticker(host)
        keep.append(child)
        return child, len(caught) - before

    plain, plain_warns = under(None)
    touched, touched_warns = under(TouchesTheChild())
    dropped, dropped_warns = under(DropsTheChild())
    ignoring = IgnoresTheChild()
    ignored, ignored_warns = under(ignoring)

    print(f"  no filter at all               : {plain_warns} warning(s)"
          f"   {'REPRODUCED' if plain_warns else 'clean'}")
    print(f"  child() and KEPT               : {touched_warns} warning(s)"
          f"   {'<-- REPRODUCED' if touched_warns else 'clean'}")
    print(f"  child() and DROPPED            : {dropped_warns} warning(s)"
          f"   {'REPRODUCED' if dropped_warns else 'clean'}")
    print(f"  child() never called           : {ignored_warns} warning(s)"
          f"   {'REPRODUCED' if ignored_warns else 'clean'}"
          f"   (it saw {ignoring.seen} ChildAdded)")
    print()
    print("  IT IS THE KEEPING, NOT THE CALL. A reconstruction that calls")
    print("  event.child() and lets it go is CLEAN, which is the most")
    print("  likely way to fail to reproduce this and conclude, wrongly,")
    print("  that it is platform-specific.")

    print()
    print("  WHAT THE CHILD LOSES. The warning names the tracking slot, but")
    print("  the wrapper takes the whole dynamic metaobject with it:\n")
    for label, widget in (("unwatched host", plain),
                          ("touched by the filter", touched)):
        meta = widget.metaObject()
        # indexOfSignal, not a slice from methodOffset(): PySide keeps a
        # class's own signals in a metaobject layer BELOW that offset, so
        # slicing reports a healthy widget as having no signals.
        declared = meta.indexOfSignal("poked(int)")
        widget.poked.emit(7)
        parent = widget.parent()
        found = len(parent.findChildren(Ticker)) if parent else -1
        if parent is not None:
            parent.show()
        widget.show()
        app.processEvents()
        print(f"    {label}:")
        print(f"      metaObject().className() : {meta.className()!r}")
        print(f"      indexOfSignal poked(int) : {declared}"
              f"{'  (absent)' if declared == -1 else ''}")
        print(f"      parent.findChildren      : {found}")
        print(f"      poked.emit(7) delivered  : {widget.delivered}")
        print(f"      showEvent override fired : {widget.shown}"
              f"{'   <-- SILENTLY SKIPPED' if not widget.shown else ''}")

    print()
    if touched_warns and not dropped_warns and not ignored_warns \
            and not plain_warns:
        print("  VERDICT: CONFIRMED, and this is NOT a cosmetic warning.")
        print("  A bound-method connection does still fire on its own (58")
        print("  ticks against 58 for a control), which is what made this")
        print("  look harmless. Everything else about the child is broken:")
        print("  its OWN signals are absent from its metaobject, so emitting")
        print("  one is a silent no-op; findChildren() by type cannot see")
        print("  it; and its PYTHON OVERRIDES OF QT VIRTUALS ARE SKIPPED,")
        print("  because Shiboken looks the override up through the wrapper")
        print("  that is missing.")
        print()
        print("  THAT LAST ONE COST A FEATURE. AmbientWidget starts its")
        print("  timer from showEvent, so the override never ran and the")
        print("  backdrop never animated. Measured in the real app, same")
        print("  script both ways: without the fix 2 warnings, className")
        print("  QWidget, findChildren 0, frames_painted 0 -- STALLED; with")
        print("  it 0 warnings, className AmbientWidget, findChildren 1,")
        print("  frames_painted 35 and 32 -- animating.")
        print()
        print("  THE FIX is in the filter, not in any child: capture")
        print("  `watched` -- which is fully wrapped already -- and look at")
        print("  its children one turn later, instead of calling")
        print("  event.child() while the child is still being constructed.")
        print("  Deferring that way also hands back the child's REAL class,")
        print("  which removes the reason `_on_arrival` has to ask Qt with")
        print("  `inherits` rather than Python with `isinstance`.")
    else:
        print("  VERDICT: this build does NOT behave as measured on the")
        print("  reporting Mac. Report these three numbers before changing")
        print("  anything -- the cause recorded in instruction 320 was")
        print("  established from them.")

    QTimer.singleShot(0, app.quit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
