"""Open the Overview's Add folders and Add plates pickers in the real app and report them.

A manual check for the maintainer's desktop (or any real window manager):
it launches spaCR with the saved preferences, opens the Overview on a small
synthetic plate, presses "Add folders…" on the metadata step and "Add
plates…" on the plate browser, and for each picker prints its geometry,
window flags and button visibility when shown, after 500 ms and after 2 s,
saves a screenshot of the screen at 2 s, then closes the picker and spaCR.

    python tools/check_folder_dialog.py [--out DIR] [--keep-open SECONDS] [--drag]

Nothing is chosen in either picker and no preference is written by the
check itself. Exit status 0 means both pickers stayed big, kept every button
visible and can be resized; 1 means at least one did not.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


def make_plate(root: Path) -> Path:
    """Write a two-well, two-channel synthetic plate under ``root``."""
    import numpy as np
    import tifffile

    plate = root / "plate1"
    plate.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(("A01", "A02")):
        for channel in (1, 2):
            tifffile.imwrite(plate / f"p1-{well}-f1-c{channel}.tif",
                             np.full((64, 64), 100 * (index + 1) + channel, dtype=np.uint16))
    return plate


def describe(dialog) -> dict:
    """Geometry, flags and button visibility of one picker."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton

    screen = dialog.screen()
    available = screen.availableGeometry() if screen is not None else None
    frame = dialog.frameGeometry()
    buttons = []
    for button in dialog.findChildren(QPushButton):
        if button.isHidden():
            continue
        corner = button.mapTo(dialog, button.rect().bottomRight())
        buttons.append({
            "text": button.text(), "visible": button.isVisible(),
            "region_empty": button.visibleRegion().isEmpty(),
            "unmasked": dialog.mask().isEmpty() or dialog.mask().contains(corner),
            "inside": dialog.rect().contains(button.mapTo(dialog, button.rect().topLeft()))
            and dialog.rect().contains(corner)})
    flags = dialog.windowFlags()
    handle = dialog.windowHandle()
    mask = dialog.mask().boundingRect()
    return {
        "native": [handle.x(), handle.y(), handle.width(), handle.height()] if handle is not None else None,
        "alpha_buffer": handle.format().alphaBufferSize() if handle is not None else None,
        "mask": [mask.x(), mask.y(), mask.width(), mask.height()],
        "x11": native_size(dialog),
        "visible": dialog.isVisible(),
        "exposed": bool(dialog.windowHandle() is not None and dialog.windowHandle().isExposed()),
        "active": dialog.isActiveWindow(),
        "geometry": [dialog.x(), dialog.y(), dialog.width(), dialog.height()],
        "frame": [frame.x(), frame.y(), frame.width(), frame.height()],
        "minimum": [dialog.minimumWidth(), dialog.minimumHeight()],
        "maximum": [dialog.maximumWidth(), dialog.maximumHeight()],
        "available": [available.width(), available.height()] if available is not None else None,
        "frameless": bool(flags & Qt.FramelessWindowHint),
        "translucent": dialog.testAttribute(Qt.WA_TranslucentBackground),
        "size_grip": dialog.isSizeGripEnabled(),
        "edge_resizer": getattr(dialog, "_spacr_resizer", None) is not None,
        "flags": hex(int(flags.value if hasattr(flags, "value") else flags)),
        "buttons": buttons,
    }


def native_size(dialog):
    """The platform window's own width and height, from ``xwininfo`` on X11."""
    import shutil
    import subprocess

    from PySide6.QtWidgets import QApplication

    if QApplication.platformName() != "xcb" or shutil.which("xwininfo") is None:
        return None
    try:
        text = subprocess.run(["xwininfo", "-id", str(int(dialog.winId()))], capture_output=True,
                              text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    sizes = {}
    for line in text.splitlines():
        name, _, value = line.strip().partition(":")
        if name in ("Width", "Height"):
            sizes[name] = int(value)
    return [sizes.get("Width"), sizes.get("Height")] if sizes else None


def x_pointer(steps):
    """Move and press the real X pointer through ``steps`` with XTest.

    :param steps: ``(x, y, button)`` tuples in screen pixels; ``button`` is
        ``"press"``, ``"release"`` or ``None`` for a move.
    :returns: whether XTest was available.
    """
    import ctypes
    import ctypes.util

    names = {"x11": ctypes.util.find_library("X11"), "xtst": ctypes.util.find_library("Xtst")}
    if not all(names.values()):
        return False
    x11, xtst = ctypes.CDLL(names["x11"]), ctypes.CDLL(names["xtst"])
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
    display = x11.XOpenDisplay(None)
    if not display:
        return False
    try:
        for x, y, button in steps:
            xtst.XTestFakeMotionEvent(display, -1, int(x), int(y), 0)
            if button is not None:
                xtst.XTestFakeButtonEvent(display, 1, 1 if button == "press" else 0, 0)
            x11.XFlush(display)
            time.sleep(.08)
    finally:
        x11.XCloseDisplay(display)
    return True


def verdict(record: dict) -> list:
    """What is wrong with a picker described by :func:`describe`, if anything."""
    problems = []
    width, height = record["geometry"][2:]
    available = record["available"]
    if available and (width < available[0] * .6 or height < available[1] * .6):
        problems.append(f"only {width}x{height} on a {available[0]}x{available[1]} screen")
    mask = record.get("mask") or [0, 0, 0, 0]
    if mask[2] and mask[3] and (mask[2] < width - 2 or mask[3] < height - 2):
        problems.append(f"a window mask cuts it to {mask[2]}x{mask[3]}")
    native = record.get("x11")
    if native and native[0] and native[1] and (native[0] < width - 2 or native[1] < height - 2):
        problems.append(f"the X11 window is only {native[0]}x{native[1]}")
    if not record["visible"]:
        problems.append("not visible")
    if record["maximum"][0] <= record["minimum"][0] or record["maximum"][1] <= record["minimum"][1]:
        problems.append("fixed size")
    if not (record["size_grip"] or record["edge_resizer"]):
        problems.append("no way to resize")
    shown = [b for b in record["buttons"] if b["visible"] and not b["region_empty"] and b["inside"]
             and b.get("unmasked", True)]
    texts = " ".join(b["text"].replace("&", "").lower() for b in shown)
    if "cancel" not in texts or not any(word in texts for word in ("choose", "open", "select")):
        problems.append("the Choose/Cancel buttons are not on screen: " + repr([b["text"] for b in shown]))
    return problems


class Driver:
    """Steps through the Overview and the two pickers on the app's own event loop."""

    def __init__(self, window, plate: Path, out: Path, keep_open: float, drag: bool = False):
        self.window, self.plate, self.out, self.keep_open, self.drag = window, plate, out, keep_open, drag
        self.report, self.started = {}, time.monotonic()

    def start(self):
        from PySide6.QtCore import QTimer

        QTimer.singleShot(3000, self.open_overview)

    def wait(self, test, then, timeout=60.0):
        from PySide6.QtCore import QTimer

        deadline = time.monotonic() + timeout

        def poll():
            try:
                done = test()
            except Exception:
                done = False
            if done:
                then()
            elif time.monotonic() > deadline:
                self.report.setdefault("errors", []).append(f"timed out waiting for {then.__name__}")
                self.finish()
            else:
                QTimer.singleShot(200, poll)

        poll()

    def open_overview(self):
        from spacr.qt.widgets import qc_field_browser as browser

        self.window.showMaximized()
        self.wizard = browser._open_field_overview(self.window, {
            "source": str(self.plate), "src": str(self.plate), "custom_regex": PATTERN,
            "source_scope": "folder", "channels": (1, 2)})
        self.wait(lambda: self.wizard.isVisible(), self.press_add_folders)

    def press(self, button, key):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication, QFileDialog

        def watch(step=0):
            dialog = QApplication.activeModalWidget()
            if not isinstance(dialog, QFileDialog):
                if step < 100:
                    QTimer.singleShot(50, lambda: watch(step + 1))
                return
            self.report[key] = {"at_show": describe(dialog)}
            QTimer.singleShot(500, lambda: self.report[key].__setitem__("after_500ms", describe(dialog)))
            QTimer.singleShot(2000, lambda: self.snap(dialog, key))

        QTimer.singleShot(0, watch)
        button.click()

    def snap(self, dialog, key):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        record = self.report[key]
        record["after_2s"] = describe(dialog)
        shot = self.out / f"{key}.png"
        QApplication.primaryScreen().grabWindow(0).save(str(shot))
        record["screenshot"] = str(shot)
        record["problems"] = verdict(record["after_2s"])
        if self.drag:
            QTimer.singleShot(100, lambda: self.drag_edge(dialog, key))
        else:
            QTimer.singleShot(int(self.keep_open * 1000), dialog.reject)

    def drag_edge(self, dialog, key):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        ratio = dialog.devicePixelRatioF()
        edge = dialog.mapToGlobal(dialog.rect().center())
        x = (dialog.mapToGlobal(dialog.rect().topRight()).x() - 2) * ratio
        y = edge.y() * ratio
        before = dialog.width()
        steps = [(x, y, None), (x, y, "press")] + [(x - d, y, None) for d in (20, 60, 120, 200)] + [(x - 200, y, "release")]

        def drive(index=0):
            if index < len(steps):
                x_pointer(steps[index:index + 1])
                QTimer.singleShot(80, lambda: drive(index + 1))
                return
            QTimer.singleShot(400, finish)

        def finish():
            self.report[key]["edge_drag"] = {"before": before, "after": dialog.width()}
            if dialog.width() >= before:
                self.report[key]["problems"].append(
                    f"dragging the right edge left by 200 px left the width at {dialog.width()}")
            QApplication.primaryScreen().grabWindow(0).save(str(self.out / f"{key}_dragged.png"))
            QTimer.singleShot(int(self.keep_open * 1000), dialog.reject)

        drive()

    def press_add_folders(self):
        from PySide6.QtWidgets import QPushButton

        button = self.wizard.findChild(QPushButton, "OverviewWizardAddFolders")
        self.press(button, "add_folders")
        self.wait(lambda: "problems" in self.report.get("add_folders", {}) and not self.modal(),
                  self.open_browser)

    def modal(self):
        from PySide6.QtWidgets import QApplication

        return QApplication.activeModalWidget() is not None

    def open_browser(self):
        self.wait(lambda: bool(self.wizard._fields), self.accept_wizard)

    def accept_wizard(self):
        self.wizard.accept()
        self.wait(lambda: self.overview() is not None and self.overview().isVisible(), self.press_add_plates)

    def overview(self):
        return getattr(self.window, "_field_overview", None)

    def press_add_plates(self):
        from PySide6.QtWidgets import QPushButton
        from spacr.qt.i18n import tr

        wanted = tr("Add plates…")
        button = next(b for b in self.overview().findChildren(QPushButton) if b.text() == wanted)
        self.press(button, "add_plates")
        self.wait(lambda: "problems" in self.report.get("add_plates", {}) and not self.modal(), self.finish)

    def finish(self):
        from PySide6.QtWidgets import QApplication

        (self.out / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")
        QApplication.instance().exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="Folder for report.json and the screenshots")
    parser.add_argument("--keep-open", type=float, default=1.0, help="Seconds each picker stays open after its screenshot")
    parser.add_argument("--drag", action="store_true",
                        help="Also drag each picker's right edge with the real X pointer (X11 only; it moves your mouse)")
    args = parser.parse_args()
    out = (args.out or Path(tempfile.mkdtemp(prefix="spacr-folder-dialog-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plate = make_plate(out / "plate")

    from spacr.qt import app as qt_app

    original = qt_app.MainWindow
    drivers = []

    def build(*a, **k):
        window = original(*a, **k)
        driver = Driver(window, plate, out, args.keep_open, args.drag)
        drivers.append(driver)
        driver.start()
        return window

    qt_app.MainWindow = build
    try:
        qt_app.launch(["--no-setup"])
    finally:
        qt_app.MainWindow = original
    report = drivers[0].report if drivers else {"errors": ["spaCR did not open"]}
    failed = bool(report.get("errors"))
    for key in ("add_folders", "add_plates"):
        record = report.get(key)
        if record is None:
            print(f"{key}: never opened")
            failed = True
            continue
        for moment in ("at_show", "after_500ms", "after_2s"):
            state = record.get(moment)
            if state:
                print(f"{key} {moment}: {state['geometry'][2]}x{state['geometry'][3]} at "
                      f"{state['geometry'][0]},{state['geometry'][1]} frameless={state['frameless']} "
                      f"translucent={state['translucent']} flags={state['flags']}")
        print(f"{key} screenshot: {record.get('screenshot')}")
        if "edge_drag" in record:
            print(f"{key} right edge dragged 200 px left: width {record['edge_drag']['before']} -> "
                  f"{record['edge_drag']['after']}")
        print(f"{key}: " + ("; ".join(record.get("problems") or ()) or "OK"))
        failed = failed or bool(record.get("problems"))
    for error in report.get("errors", ()):
        print("error:", error)
    print("report:", out / "report.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
