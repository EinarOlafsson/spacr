"""GUI end-to-end wiring tests:

* T1 — pressing Run on an AppScreen actually starts a pipeline via the
  real PipelineWorker/QThread path and reaches the finished state.
* T2 — pipeline prints (stdout) are captured and land in the console
  panel, visible via the console's stdout blocks.
* A @gpu variant drives the REAL mask pipeline end-to-end through the
  button so we know a real run finishes and figures/prints flow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")


def _console_text(console) -> str:
    """Concatenate every stdout/error block's text in a ConsolePanel."""
    from spacr.qt.widgets.console_panel import _StdoutBlock
    return "\n".join(b.text() for b in console.findChildren(_StdoutBlock))


# ---------------------------------------------------------------------------
# T1/T2 — stubbed pipeline through the real worker path (fast, no GPU)
# ---------------------------------------------------------------------------

class TestRunFinishesAndPrints:
    """Completion is observed via the Run button re-enabling (which the
    real _on_finished does), NOT by monkeypatching _on_finished — doing
    that replaces the bound-method slot with a plain function, which
    breaks Qt's thread-affinity detection and forces the finished slot
    to run on the worker thread (GUI mutation off-thread → abort)."""

    def _patch_entry(self, monkeypatch, fn):
        from spacr.qt import bridge
        entry = lambda k: fn
        monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.resolve_pipeline_entry", entry)

    def test_run_starts_worker_and_finishes(self, qtbot, monkeypatch):
        """The whole make_thread → PipelineWorker → finished chain runs
        for real (no Cellpose). Finished ⇒ Run button re-enabled +
        console shows the ✓ Finished banner."""
        from spacr.qt.screens.app_screen import AppScreen

        def _fn(settings):
            print("PIPELINE-STARTED")
            print(f"got src={settings.get('src')}")
            print("PIPELINE-DONE")
        self._patch_entry(monkeypatch, _fn)

        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._on_run()
        # While running, the Run button is disabled; on finish it
        # re-enables. Wait for that transition.
        assert not scr._btn_run.isEnabled()
        qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=5000)
        qtbot.wait(100)
        assert "✓ Finished" in _console_text(scr._console)

    def test_prints_land_in_console(self, qtbot, monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        def _fn(settings):
            print("HELLO-FROM-PIPELINE")
            print("second line")
        self._patch_entry(monkeypatch, _fn)

        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._on_run()
        qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=5000)
        qtbot.wait(150)
        text = _console_text(scr._console)
        assert "HELLO-FROM-PIPELINE" in text
        assert "second line" in text

    def test_run_button_shows_starting_breadcrumb(self, qtbot, monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen
        self._patch_entry(monkeypatch, lambda s: print("ok"))
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._on_run()
        qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=5000)
        qtbot.wait(100)
        # The improved Run breadcrumb names the entry point + src.
        assert "Starting mask" in _console_text(scr._console)


# ---------------------------------------------------------------------------
# GPU — the REAL mask pipeline driven through the Run button
# ---------------------------------------------------------------------------

def _require_gpu_cellpose():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — real-run GUI test is GPU-only")
    try:
        import cellpose  # noqa: F401
    except Exception as e:
        pytest.skip(f"cellpose unavailable: {e}")


def _make_plate(dst: Path, size: int = 96) -> Path:
    import tifffile
    plate = dst / "plate1"; plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    y, x = np.ogrid[:size, :size]
    for ch in range(2):
        img = rng.integers(80, 160, size=(size, size)).astype(np.uint16)
        for cy, cx in rng.integers(15, size - 15, size=(8, 2)):
            g = np.exp(-((x - int(cx)) ** 2 + (y - int(cy)) ** 2)
                           / (2 * 5 ** 2)) * 2500
            img = np.clip(img.astype(np.float32) + g, 0, 65535
                             ).astype(np.uint16)
        tifffile.imwrite(
            str(plate / f"plate1_A01_T01F01L01A01Z01C0{ch}.tif"), img)
    return plate


@pytest.mark.slow
@pytest.mark.gpu
def test_real_mask_run_through_button_finishes(qtbot, tmp_path):
    """The full path: set src on the Mask app, press Run, wait for the
    real Cellpose pipeline to finish, and confirm the console shows
    output + the finished banner."""
    _require_gpu_cellpose()
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt import synthetic as syn

    plate = _make_plate(tmp_path)
    scr = AppScreen("mask")
    qtbot.addWidget(scr)

    s = syn.demo_settings("mask", str(plate))
    s.update({
        "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": None, "organelle_channel": None,
        "consolidate": False, "remove_background": False,
        "normalize": True, "test_mode": False, "plot": False,
        "batch_size": 2,
    })
    scr.apply_settings_dict(s)

    scr._on_run()
    # Wait for the real Cellpose run to finish (Run button re-enables).
    qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=120000)
    qtbot.wait(200)
    text = _console_text(scr._console)
    assert "Starting mask" in text
    assert "✓ Finished" in text
    # Cellpose masks should exist on disk.
    assert list(plate.rglob("*cell_mask*"))
