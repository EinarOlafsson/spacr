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

Prints a report and exits. Opens no window you have to close.
"""
from __future__ import annotations

import sys
import traceback

WARNING_MARK = "addMetaMethod"


def main() -> int:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal, qInstallMessageHandler
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

    QTimer.singleShot(0, app.quit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
