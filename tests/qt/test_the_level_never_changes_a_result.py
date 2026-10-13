"""286: a performance level may change scheduling, never a result.

"Scientific computation and results do not change between levels; only
scheduling, caching, memory retention and interface decoration do."

WHAT A LEVEL CAN REACH IN A COMPUTATION, found by reading every reader of
the level: Laptop and Extra Performance run `resource_cleanup`'s launch and
pre-run cleanups, which drop spaCR's caches, release model weights and --
the one thing a running computation can feel -- lower torch's and OpenCV's
thread counts. The retention scale and the memory budget govern figures and
caches in the interface. No module outside ``spacr/qt`` reads the level at
all, which the first test holds.

The second runs a representative measurement at the two ends of the scale,
after each end's real cleanups, and asserts the scheduling DID change (or
the comparison proves nothing) and the numbers did not. CPU only: the VRAM
pass is replaced by a recorder, so nothing here touches a GPU.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import laptop_mode as LM
from spacr.qt import preferences as P
from spacr.qt import resource_cleanup as rc

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]

#: The names through which a module could read a level or a value derived
#: from it.
_LEVEL_READERS = {
    "get_performance_level", "retention_scale", "live_figure_allowance",
    "spacr_mode_for_level", "get_spacr_mode", "get_laptop_mode",
    "recommended_for", "PERFORMANCE_RETENTION", "PERFORMANCE_LEVELS",
}


def test_no_module_outside_the_interface_reads_the_level():
    offenders = []
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ("spacr", "qt"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = (node.attr if isinstance(node, ast.Attribute)
                    else node.id if isinstance(node, ast.Name)
                    else node.name if isinstance(node, ast.alias)
                    else None)
            if name in _LEVEL_READERS:
                offenders.append(
                    f"{relative}:{getattr(node, 'lineno', '?')} {name}")
    assert offenders == [], (
        "a computation module reads the performance level, so a level "
        "could change a result:\n" + "\n".join(offenders))


@pytest.fixture(autouse=True)
def _process_backdrop_state_is_reset():
    saved = os.environ.get("SPACR_NO_BACKDROP")

    def _clear():
        LM._suppressed_here = False
        os.environ.pop("SPACR_NO_BACKDROP", None)

    _clear()
    try:
        yield
    finally:
        _clear()
        if saved is not None:
            os.environ["SPACR_NO_BACKDROP"] = saved


@pytest.fixture
def store(monkeypatch, tmp_path):
    real = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(P, "_settings", lambda: real)
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return real


@pytest.fixture
def library_threads():
    """torch and OpenCV, with their thread counts put back afterwards."""
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    saved = (torch.get_num_threads(), cv2.getNumThreads())
    try:
        yield torch, cv2
    finally:
        torch.set_num_threads(saved[0])
        cv2.setNumThreads(saved[1])


def _representative_measurement(torch):
    """Per-object intensity features, a focus measure and a convolution:
    skimage, OpenCV and torch, the three libraries a level's cleanup reaches.
    """
    from spacr.measure import _estimate_blur, _extended_regionprops_table

    rng = np.random.default_rng(286)
    image = rng.normal(100.0, 20.0, (256, 256)).astype(np.float32)
    labels = np.zeros((256, 256), np.int32)
    labels[20:80, 20:90] = 1
    labels[120:200, 40:110] = 2
    labels[150:230, 150:240] = 3

    table = _extended_regionprops_table(
        labels, image, ["label", "area", "intensity_mean", "intensity_max"])
    focus = [_estimate_blur(image, labels == index) for index in (1, 2, 3)]
    kernel = torch.ones(1, 1, 5, 5, dtype=torch.float32) / 25.0
    smoothed = torch.nn.functional.conv2d(
        torch.from_numpy(image)[None, None], kernel).numpy()
    return table, np.asarray(focus), smoothed


def test_a_representative_measurement_is_identical_at_both_ends(
        store, monkeypatch, library_threads):
    torch, cv2 = library_threads
    monkeypatch.setattr(rc, "clear_vram",
                        lambda **_kw: rc.Reclaim("vram"))

    threads, results = {}, {}
    for level in ("workstation", "laptop"):
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        P.set_performance_level(level)
        rc.run_launch_cleanup()
        rc.run_pre_run_cleanup("measure")
        threads[level] = (torch.get_num_threads(), cv2.getNumThreads())
        results[level] = _representative_measurement(torch)

    assert threads["workstation"] != threads["laptop"], (
        f"the two ends ran with the same thread counts {threads}, so this "
        "comparison proves nothing about scheduling")

    table_w, focus_w, smooth_w = results["workstation"]
    table_l, focus_l, smooth_l = results["laptop"]
    import pandas.testing as pdt

    pdt.assert_frame_equal(table_w, table_l, check_exact=True)
    np.testing.assert_array_equal(focus_w, focus_l)
    # A convolution's float sums may be split across threads differently;
    # "equivalent" is agreement far inside any measurement's precision.
    np.testing.assert_allclose(smooth_w, smooth_l, rtol=1e-6, atol=0)
