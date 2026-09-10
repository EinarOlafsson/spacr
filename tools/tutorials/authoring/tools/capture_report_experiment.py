#!/usr/bin/env python3
"""Capture the Report screen and its generated real-data HTML at native 4K."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle, wait_until


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
SOURCE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/test"
)
GENERATED = ROOT / "generated" / "report"
REPORT_HTML = GENERATED / "tutorial_test_report.html"
LESSON = "29_report"


def _browser_frame(output: Path, stem: str, section: str = "") -> None:
    source = REPORT_HTML
    if section:
        # Chrome's command-line screenshot can capture an empty viewport after
        # a very long fragment jump.  Hide sibling chapters in a temporary
        # view of the same generated document so the requested real section is
        # at the top; the report itself is not modified.
        html = REPORT_HTML.read_text()
        focus_css = (
            "<style>.banner,nav.toc,footer.doc,"
            f"main>section.chapter:not(#{section})"
            "{display:none!important}main{padding-top:24px}</style>"
        )
        source = GENERATED / f"focused_{section}.html"
        source.write_text(html.replace("</head>", focus_css + "</head>"))
    target = source.resolve().as_uri()
    path = output / f"{stem}.png"
    subprocess.run([
        "google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
        "--hide-scrollbars", "--force-device-scale-factor=1",
        "--window-size=3840,2160", "--virtual-time-budget=3000",
        f"--screenshot={path}", target,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    GENERATED.mkdir(parents=True, exist_ok=True)
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-report-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-report-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    import spacr.qt.screens.report as report_screen_module
    report_screen_module.QDesktopServices.openUrl = staticmethod(lambda _url: True)
    from spacr.qt.preferences import apply_preferences_to_app

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")
    window._on_nav_selected("report")
    wait_until(app, lambda: window._screens.get("report") is not None,
               15.0, "Report screen")
    screen = window._screens["report"]
    settle(app, 35)

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav_button(window, "report"),
        "screen": screen,
        "source": [screen._path_edit, screen._btn_pick_src, screen._btn_scan],
        "verdict": screen._verdict,
        "sections": screen._sections,
        "options": [screen._format, screen._figure_cap],
        "output": [screen._out_edit, screen._btn_pick_out,
                   screen._btn_generate, screen._btn_open],
        "status": screen._status,
    })

    screen.set_source(str(SOURCE))
    screen._figure_cap.setValue(12)
    settle(app, 20)
    capture.save("02_source", {
        "source": [screen._path_edit, screen._btn_pick_src, screen._btn_scan],
        "scan": screen._btn_scan,
        "status": screen._status,
    })
    screen._btn_scan.click()
    wait_until(app, lambda: not screen._busy and screen.report is not None,
               60.0, "real report scan")
    settle(app, 35)

    capture.save("03_verdict", {
        "verdict": screen._verdict,
        "sections": screen._sections,
        "status": screen._status,
        "scan_result": [screen._verdict, screen._sections, screen._status],
    })
    capture.save("04_sections", {
        "sections": screen._sections,
        "verdict": screen._verdict,
    })
    capture.save("05_options", {
        "options": [screen._format, screen._figure_cap],
        "format": screen._format,
        "figure_cap": screen._figure_cap,
    })

    screen.set_output(str(REPORT_HTML))
    settle(app, 15)
    capture.save("06_output", {
        "output": [screen._out_edit, screen._btn_pick_out,
                   screen._btn_generate, screen._btn_open],
        "generate": screen._btn_generate,
    })
    capture.save("07_generate", {
        "generate": screen._btn_generate,
        "output": [screen._out_edit, screen._btn_generate],
    })
    screen._btn_generate.click()
    wait_until(app, lambda: not screen._busy and bool(screen.written),
               90.0, "HTML report generation")
    if not REPORT_HTML.exists():
        raise FileNotFoundError(REPORT_HTML)
    settle(app, 35)
    capture.save("08_generated", {
        "output": [screen._out_edit, screen._btn_generate, screen._btn_open],
        "status": screen._status,
        "result": [screen._out_edit, screen._btn_generate,
                   screen._btn_open, screen._status],
    })
    capture.save("09_open", {
        "open": screen._btn_open,
        "output": [screen._out_edit, screen._btn_open],
    })
    screen._btn_open.click()
    settle(app, 20)
    capture.save("10_opened", {
        "status": screen._status,
        "open": screen._btn_open,
    })
    capture.write_geometry()

    screen.close()
    settle(app, 20)
    window.close()
    settle(app, 20)
    app.quit()

    browser_frames = (
        ("11_report_top", ""),
        ("12_report_figures", "figures"),
        ("13_report_statistics", "statistics"),
        ("14_report_settings", "settings"),
    )
    for stem, section in browser_frames:
        _browser_frame(output, stem, section)
        capture.frames[stem] = {
            "file": f"{stem}.png",
            "device_pixel_ratio": 1.0,
            "report": [0, 0, 3840, 2160],
        }
    capture.write_geometry()

    report = screen.report
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "report",
        "source": str(SOURCE),
        "output": str(REPORT_HTML),
        "capture_size": [3840, 2160],
        "status": report.status if report is not None else None,
        "status_detail": report.status_detail if report is not None else None,
        "found_sections": list(report.found_sections) if report is not None else [],
        "missing_sections": list(report.missing_sections) if report is not None else [],
        "figures_found": report.n_figures_found if report is not None else None,
        "figures_embedded": report.n_figures_embedded if report is not None else None,
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
