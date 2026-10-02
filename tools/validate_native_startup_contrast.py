"""Measure N415 on an actual, disposable GitHub-hosted Windows desktop.

Each worker starts after changing the OS's real personalization registry
values. It must observe that scheme through Qt's native style hints; neither
the OS query, painter nor application palette is mocked. Captures come from
QScreen.grabWindow, not QWidget.grab or an offscreen platform. The unchanged
production preferences and widgets choose every displayed colour.

This establishes Windows technical acceptance, not a maintainer's personal
hands-on approval or acceptance on another Windows build. Failure to obtain
a native exposed window, the requested OS scheme or readable captured text
is a failure, never a skip. Only ephemeral GitHub-hosted Windows runners may
change their OS settings. The controller restores the original registry data
in its finally block. spaCR preferences use only that disposable runner's
native registry store, reset for each worker and restored in its finally
block. QSettings(org, app) does not honor setDefaultFormat(IniFormat).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MINIMUM_TEXT_CONTRAST = 4.5
MINIMUM_ROLE_CONTRAST = 7.0
_JOB = None


def _require_hosted_windows():
    if not (sys.platform == "win32"
            and os.environ.get("GITHUB_ACTIONS") == "true"
            and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"):
        raise RuntimeError("This acceptance changes OS preferences and requires "
                           "a disposable GitHub-hosted Windows runner.")


def _bounded_worker():
    """Apply native hard committed-memory and processor-affinity bounds."""
    from ctypes import wintypes

    import psutil

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                    ("max_working", ctypes.c_size_t), ("active", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("read", "write", "other", "read_bytes", "write_bytes", "other_bytes")]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", Counters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    global _JOB
    _JOB = kernel.CreateJobObjectW(None, None)
    limits = Extended()
    # JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    limits.basic.flags = 0x200 | 0x2000
    limits.job_memory = 4 * 1024 ** 3
    if not _JOB or not kernel.SetInformationJobObject(_JOB, 9, ctypes.byref(limits),
                                                     ctypes.sizeof(limits)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(_JOB, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    actual = process.cpu_affinity()
    if len(actual) != 1:
        raise RuntimeError(f"Expected one CPU, got {actual}")
    return {"cpu_affinity": actual, "job_committed_memory_limit": limits.job_memory,
            "pagefile": "Windows-managed; this is not a Linux zero-swap cgroup"}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@contextmanager
def _runner_preferences():
    """Keep production registry reads; touch only the guarded hosted profile."""
    _require_hosted_windows()
    from PySide6.QtCore import QSettings

    store = QSettings("spacr", "qt")
    before = {key: store.value(key) for key in store.allKeys()}
    try:
        store.clear()
        store.sync()
        if store.status() != QSettings.Status.NoError:
            raise RuntimeError("Cannot initialize the runner's native preference store")
        yield store
    finally:
        store.clear()
        for key, value in before.items():
            store.setValue(key, value)
        store.sync()
        if store.status() != QSettings.Status.NoError:
            raise RuntimeError("Cannot restore the runner's native preference store")
        if {key: store.value(key) for key in store.allKeys()} != before:
            raise RuntimeError("Runner preference restoration did not round-trip")


def _contrast(a, b):
    def luminance(rgb):
        values = [v / 255 for v in rgb[:3]]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
                  for v in values]
        return sum(v * weight for v, weight in zip(linear, (.2126, .7152, .0722)))
    low, high = sorted((luminance(a), luminance(b)))
    return (high + .05) / (low + .05)


def _text_pixels(image, box, background):
    """Require a visible body of high-contrast ink, not one stray bright pixel."""
    ratio = image.devicePixelRatio()
    left, top, right, bottom = [round(value * ratio) for value in box]
    if left < 0 or top < 0 or right > image.width() or bottom > image.height():
        raise AssertionError(f"Text bounding box falls outside capture: {box}")
    counts = Counter(image.pixelColor(x, y).getRgb()[:3]
                     for y in range(top, bottom) for x in range(left, right))
    readable = sum(count for colour, count in counts.items()
                   if _contrast(colour, background) >= MINIMUM_TEXT_CONTRAST)
    best = max((_contrast(colour, background) for colour in counts), default=1)
    if readable < 8:
        raise AssertionError(f"Only {readable} readable ink pixels; best contrast {best:.3f}")
    return {"box": box, "readable_pixels": readable, "best_pixel_contrast": best}


def _capture(app, widget, path):
    from PySide6.QtTest import QTest

    widget.show()
    widget.raise_()
    widget.activateWindow()
    deadline = time.monotonic() + 10
    while not widget.windowHandle().isExposed() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(20)
    if not widget.windowHandle().isExposed():
        raise AssertionError("The actual native window was never exposed")
    QTest.qWait(120)
    screen = widget.windowHandle().screen()
    if not screen.geometry().contains(widget.frameGeometry()):
        raise AssertionError("Native window must fit entirely on the hosted desktop")
    capture = screen.grabWindow(int(widget.winId()))
    if capture.isNull() or not capture.save(str(path)):
        raise AssertionError("Native QScreen capture failed")
    image = capture.toImage()
    if abs(image.width() / image.devicePixelRatio() - widget.width()) > 1:
        raise AssertionError("Native capture does not cover the client width")
    return image


def _splash_measurements(screen, image, palette):
    from PySide6.QtGui import QColor, QFont, QFontMetrics

    from spacr.qt.preferences import scaled_px
    from spacr.qt.widgets.loading_screen import strap_phrases

    background = QColor(palette["splash_bg"]).getRgb()[:3]
    for point in ((3, 3), (screen.width() - 4, 3), (3, screen.height() - 4)):
        ratio = image.devicePixelRatio()
        actual = image.pixelColor(round(point[0] * ratio), round(point[1] * ratio)).getRgb()[:3]
        if actual != background:
            raise AssertionError(f"Native splash background {actual} != expected {background}")
    font = QFont(screen.font())
    font.setPixelSize(scaled_px(15))
    metrics = QFontMetrics(font)
    phases = strap_phrases()
    widths = [metrics.horizontalAdvance(text) for text in phases]
    arrow = "  →  "
    arrow_width = metrics.horizontalAdvance(arrow)
    side, gap = scaled_px(140), scaled_px(28)
    x = (screen.width() - side - gap - sum(widths) - 2 * arrow_width) / 2 + side + gap
    baseline = int(screen.height() / 2 + metrics.ascent() / 2 - metrics.descent() / 2)
    results = []
    for index, (text, width) in enumerate(zip(phases, widths)):
        for label, span, lit in ((text, width, index < screen.lit_phases()),
                                 (arrow, arrow_width, index + 1 < screen.lit_phases())):
            if label == arrow and index == len(phases) - 1:
                continue
            role = "splash_ink" if lit else "splash_ink_dim"
            ratio = _contrast(QColor(palette[role]).getRgb()[:3], background)
            if ratio < MINIMUM_ROLE_CONTRAST - .01:
                raise AssertionError(f"{role} contrast is only {ratio:.3f}")
            box = [int(x), baseline - metrics.ascent(), int(x) + span,
                   baseline + metrics.descent()]
            result = _text_pixels(image, box, background)
            results.append(dict(result, text=label, role=role, role_contrast=ratio))
            x += span
    return results


def _worker(os_scheme, stored_theme, output):
    _require_hosted_windows()
    receipt = {"passed": False, "os_scheme_requested": os_scheme, "stored_theme": stored_theme,
               "platform": platform.platform(), "python": sys.version, "captures": []}
    try:
        receipt["bounds"] = _bounded_worker()
        from PySide6 import __version__ as qt_binding_version
        from PySide6.QtCore import qVersion
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication

        with _runner_preferences() as settings:
            app = QApplication([])
            app.setQuitOnLastWindowClosed(False)
            if app.platformName() != "windows":
                raise AssertionError(f"Not native Windows QPA: {app.platformName()}")
            from spacr.qt import preferences, theme
            reported = theme.system_colour_scheme(app)
            receipt.update(qt=qVersion(), pyside=qt_binding_version,
                           qpa=app.platformName(), os_scheme_reported=reported,
                           style_hint=str(app.styleHints().colorScheme()),
                           screen_geometry=app.primaryScreen().geometry().getRect())
            if reported != os_scheme:
                raise AssertionError(f"Actual OS scheme {reported!r}, requested {os_scheme!r}")
            receipt["preferences"] = {"store": settings.fileName(),
                                      "isolation": "disposable GitHub-hosted runner profile"}
            preferences.set_language("en")
            preferences.set_theme(stored_theme)
            preferences.set_ambient_enabled(False)
            preferences.set_refresh_news(False)
            preferences.set_preload_policy("on_demand")
            from spacr.qt.first_run import mark_tour_seen
            mark_tour_seen()
            effective = os_scheme if stored_theme == "system" else stored_theme
            if preferences.resolve_effective_theme() != effective:
                raise AssertionError("Stored theme did not resolve against the actual OS")
            receipt["effective_theme"] = effective
            from spacr.qt.app import MainWindow, _load_bundled_fonts, _use_open_sans
            from spacr.qt.widgets.loading_screen import LoadingScreen
            _load_bundled_fonts()
            _use_open_sans(app)
            palette = theme.palette_for(effective)
            for applied in (False, True):
                if applied:
                    preferences.apply_preferences_to_app(app)
                splash = LoadingScreen(total=6)
                splash.resize(1200, 450)
                splash.move(30, 30)
                for done in (0, 2, 6):
                    splash.advance(done)
                    name = f"splash-{'applied' if applied else 'early'}-{done}.png"
                    image = _capture(app, splash, output / name)
                    measurements = _splash_measurements(splash, image, palette)
                    receipt["captures"].append({"file": name, "progress": done,
                                                "preferences_applied": applied,
                                                "device_pixel_ratio": image.devicePixelRatio(),
                                                "text": measurements})
                splash.close()
            window = MainWindow()
            window.resize(1200, 720)
            window.move(30, 30)
            # Inspect the real constructor's backing palette before the first show.
            background = window.palette().color(QPalette.ColorRole.Window)
            ink = app.palette().color(QPalette.ColorRole.WindowText)
            first_contrast = _contrast(ink.getRgb()[:3], background.getRgb()[:3])
            if background != QColor(palette["bg"]) or first_contrast < MINIMUM_TEXT_CONTRAST:
                raise AssertionError("The actual MainWindow first backing palette is unreadable")
            receipt["main_window"] = {"background": background.name(), "ink": ink.name(),
                                      "first_backing_contrast": first_contrast,
                                      "preload_policy": preferences.get_preload_policy()}
            image = _capture(app, window, output / "home-native.png")
            receipt["main_window"]["capture"] = "home-native.png"
            receipt["main_window"]["device_pixel_ratio"] = image.devicePixelRatio()
            window.close()
            app.processEvents()
            receipt["passed"] = True
    except Exception:
        receipt["passed"] = False
        receipt["error"] = traceback.format_exc()
    finally:
        _write(output / "receipt.json", receipt)
    return 0 if receipt["passed"] else 1


def _broadcast_scheme():
    from ctypes import wintypes
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                                        wintypes.LPCWSTR, wintypes.UINT, wintypes.UINT,
                                        ctypes.POINTER(ctypes.c_size_t)]
    result = ctypes.c_size_t()
    user.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "ImmersiveColorSet", 2, 5000, ctypes.byref(result))


def _controller(output):
    _require_hosted_windows()
    import winreg
    key_name = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    keys = ("AppsUseLightTheme", "SystemUsesLightTheme")
    cases, original = [], {}
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_name, 0, winreg.KEY_ALL_ACCESS) as key:
        for name in keys:
            try:
                original[name] = winreg.QueryValueEx(key, name)
            except FileNotFoundError:
                original[name] = None
        try:
            for os_scheme in ("light", "dark"):
                for name in keys:
                    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, int(os_scheme == "light"))
                _broadcast_scheme()
                for stored_theme in ("light", "dark", "system"):
                    case = output / f"os-{os_scheme}-spacr-{stored_theme}"
                    case.mkdir()
                    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(case),
                               "--case", os_scheme, stored_theme]
                    try:
                        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
                        (case / "stdout.log").write_text(run.stdout, encoding="utf-8")
                        (case / "stderr.log").write_text(run.stderr, encoding="utf-8")
                        passed = run.returncode == 0
                        error = None if passed else f"worker exit {run.returncode}"
                    except subprocess.TimeoutExpired as exc:
                        passed, error = False, str(exc)
                    cases.append({"case": case.name, "passed": passed, "error": error})
                    print(json.dumps(cases[-1]), flush=True)
        finally:
            for name, value in original.items():
                if value is None:
                    winreg.DeleteValue(key, name)
                else:
                    data, kind = value
                    winreg.SetValueEx(key, name, 0, kind, data)
            _broadcast_scheme()
            for name, expected in original.items():
                try:
                    actual = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    actual = None
                if actual != expected:
                    raise RuntimeError(f"OS personalization was not restored: {name}")
    passed = len(cases) == 6 and all(case["passed"] for case in cases)
    sources = ("spacr/qt/widgets/loading_screen.py", "spacr/qt/theme.py",
               "spacr/qt/preferences.py", "spacr/qt/app.py", "tools/validate_native_startup_contrast.py")
    _write(output / "summary.json", {
        "passed": passed, "cases": cases, "registry_restored": True,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "acceptance_scope": "Hosted Windows native Qt technical acceptance; not maintainer hands-on approval",
    })
    return 0 if passed else 1


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", nargs=2, metavar=("OS_SCHEME", "SPACR_THEME"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return _worker(*args.case, args.output) if args.case else _controller(args.output)


if __name__ == "__main__":
    raise SystemExit(_main())
