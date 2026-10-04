"""Item 638: verbose logging is usable in the GUI.

The maintainer: "verbose logging last time I used it would crash the
program; make sure this feature is usable".

* A module screen opens and finishes a run with Verbose logging on and
  DEBUG shown in the console, while the run prints and logs thousands of
  lines.
* The console's output block keeps its height as a running sum. Walking
  every paragraph on each new line cost 35 ms a line once the block was
  full, so fast output froze the window while the queue of lines grew.
"""
from __future__ import annotations

import logging
import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

ALL_LEVELS = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
              logging.CRITICAL)


@pytest.fixture
def verbose_on(tmp_path, monkeypatch):
    """Verbose logging and console DEBUG on; logger state put back after."""
    from spacr.qt import verbose_logger as vl
    from spacr.qt import preferences as prefs

    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    names = vl._ATTACHED_LOGGERS + ("cellpose", "")
    levels = {name: logging.getLogger(name).level for name in names}
    saved = {name: getattr(vl, name) for name in
             ("_console_ref", "_handler", "_relay", "_file_handler",
              "_verbose")}
    handlers = {name: list(logging.getLogger(name).handlers)
                for name in vl._ATTACHED_LOGGERS}
    prefs.set_verbose_logging(True)
    vl.apply_verbose_logging(True)
    vl.apply_console_levels(ALL_LEVELS)
    try:
        yield vl
    finally:
        for name, value in saved.items():
            setattr(vl, name, value)
        for name, kept in handlers.items():
            logging.getLogger(name).handlers[:] = kept
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock
    return "\n".join(b.text() for b in console.findChildren(_StdoutBlock))


def test_a_module_opens_and_finishes_a_noisy_run_with_verbose_on(
        qtbot, monkeypatch, verbose_on):
    from spacr.qt import bridge
    from spacr.qt.screens.app_screen import AppScreen

    def _noisy(settings):
        log = logging.getLogger("spacr.measure")
        for i in range(1500):
            print(f"Progress: {i}/1500, operation_type: measure")
            log.debug("measured field %d", i)
        print("NOISY-RUN-DONE")

    entry = lambda key: _noisy
    monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
    monkeypatch.setattr(
        "spacr.qt.screens.app_screen.resolve_pipeline_entry", entry)

    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    screen.resize(1200, 720)
    screen.show()
    verbose_on.register_console_target(screen._console)
    assert verbose_on.is_verbose()

    screen._on_run()
    qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=60000)
    qtbot.waitUntil(
        lambda: "NOISY-RUN-DONE" in _console_text(screen._console),
        timeout=30000)
    assert "Progress: 1499/1500" in _console_text(screen._console)


def test_the_output_block_height_is_kept_without_walking_every_line(qtbot):
    from spacr.qt.widgets.console_panel import _StdoutBlock

    block = _StdoutBlock()
    qtbot.addWidget(block)
    block.resize(700, 400)
    block.show()
    block.sizeHint()
    line = "x" * 90 + "\n"
    for _ in range(2000):
        block.append(line)
    block.sizeHint()

    started = time.perf_counter()
    for _ in range(200):
        block.append(line)
        block.sizeHint()
    per_line = (time.perf_counter() - started) / 200
    assert per_line < 0.003, per_line

    running = block.sizeHint().height()
    block._height_key = ()
    assert block.sizeHint().height() == running


def test_a_trimmed_block_still_reports_its_real_height(qtbot):
    from spacr.qt.widgets.console_panel import _StdoutBlock

    block = _StdoutBlock()
    qtbot.addWidget(block)
    block.resize(600, 300)
    block.show()
    block.sizeHint()
    block.MAX_CHARS = 5000
    for i in range(300):
        block.append(f"line {i} " + "y" * (i % 70) + "\n")
    assert block._chars <= 5000
    running = block.sizeHint().height()
    block._height_key = ()
    assert block.sizeHint().height() == running
