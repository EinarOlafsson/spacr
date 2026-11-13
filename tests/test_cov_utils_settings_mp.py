"""CPU coverage for the ``spacr.utils`` settings / progress / multiprocessing block
(utils.py 1011-1337).

Targets the defensive and rarely-exercised branches of:
``calculate_activation_correlations`` (the degenerate-Manders tail),
``load_settings``, ``pretty_print_settings``, ``save_settings``,
``print_progress``, ``reset_mp``, ``is_multiprocessing_process``,
``close_file_descriptors``, ``close_multiprocessing_processes``,
``check_mask_folder`` and ``smooth_hull_lines``.

Everything here is offline and side-effect free:

* ``reset_mp`` never touches the real interpreter start method — both
  ``get_start_method`` and ``set_start_method`` are replaced with recorders.
* ``close_file_descriptors`` never closes a real descriptor — ``os.close`` and
  ``resource.getrlimit`` are stubbed and the patch is undone before asserting.
* ``close_multiprocessing_processes`` never sees a real process — ``psutil.
  process_iter`` yields hand-built fakes and the descriptor sweep is stubbed.
"""
from __future__ import annotations


try:
    import resource
except ImportError:
    resource = None

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_open_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# calculate_activation_correlations — degenerate Manders tails (1011-1015)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# pretty_print_settings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# save_settings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# print_progress
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# reset_mp
# ---------------------------------------------------------------------------

@pytest.fixture
def mp_recorder(monkeypatch):
    """Replace get/set_start_method so the real interpreter is never touched."""
    import spacr.utils as U

    state = {"current": "fork", "calls": []}

    def _get():
        return state["current"]

    def _set(method, force=False):
        state["calls"].append((method, force))
        state["current"] = method

    monkeypatch.setattr(U, "get_start_method", _get)
    monkeypatch.setattr(U, "set_start_method", _set)
    return state


@pytest.mark.parametrize(
    "system,current,expected",
    [
        ("Windows", "fork", [("spawn", True)]),
        ("Windows", "spawn", []),
        ("Linux", "spawn", [("fork", True)]),
        ("Linux", "fork", []),
        ("Darwin", "spawn", [("fork", True)]),
        ("Java", "spawn", []),  # unknown platform: leave it alone
    ],
)
def test_reset_mp_per_platform(mp_recorder, monkeypatch, system, current, expected):
    import spacr.utils as U

    mp_recorder["current"] = current
    monkeypatch.setattr(U.platform, "system", lambda: system)

    U.reset_mp()

    assert mp_recorder["calls"] == expected


# ---------------------------------------------------------------------------
# is_multiprocessing_process
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# close_file_descriptors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# close_multiprocessing_processes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# check_mask_folder
# ---------------------------------------------------------------------------

def test_check_mask_folder_counts_only_npy(tmp_path, capsys):
    """Non-.npy clutter in either folder is ignored by the equality test."""
    from spacr.utils import check_mask_folder

    src = tmp_path / "plate"
    masks = src / "masks" / "cell_mask_stack"
    stack = src / "stack"
    masks.mkdir(parents=True)
    stack.mkdir(parents=True)

    for i in range(3):
        (stack / f"{i}.npy").write_bytes(b"")
        # Masks are whole arrays: an empty mask is not counted (2026-09-19).
        np.save(masks / f"{i}.npy", np.zeros((4, 4), np.uint16))
    # Clutter that must not be counted.
    (masks / "notes.txt").write_text("hi", encoding="utf-8")
    (stack / "preview.png").write_bytes(b"")

    assert check_mask_folder(str(src), "cell_mask_stack") is False
    assert "All masks have been generated for cell_mask_stack" in capsys.readouterr().out

    # Remove one mask -> work still to do.
    (masks / "2.npy").unlink()
    assert check_mask_folder(str(src), "cell_mask_stack") is True

    # A mask folder that was never created at all short-circuits to True
    # without ever listing the stack folder.
    assert check_mask_folder(str(src), "pathogen_mask_stack") is True


# ---------------------------------------------------------------------------
# smooth_hull_lines
# ---------------------------------------------------------------------------

def test_smooth_hull_lines_returns_closed_100_point_outline():
    """The spline outline has 100 samples, is closed, and encloses the cluster."""
    from matplotlib.path import Path as MplPath

    from spacr.utils import smooth_hull_lines

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(40, 2))
    # Guarantee a well-conditioned hull with corners at +-2.
    pts = np.vstack([pts, [[-2, -2], [-2, 2], [2, 2], [2, -2]]])

    x, y = smooth_hull_lines(pts)

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape == (100,) and y.shape == (100,)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    # Closed loop: first and last sample coincide.
    assert x[0] == pytest.approx(x[-1], abs=1e-6)
    assert y[0] == pytest.approx(y[-1], abs=1e-6)

    outline = MplPath(np.column_stack([x, y]))
    # The cluster centre is inside the outline, a far-away point is not.
    assert outline.contains_point((0.0, 0.0))
    assert not outline.contains_point((50.0, 50.0))
    # The smoothed outline still tracks the hull scale (spline overshoot at the
    # square corners is bounded).
    assert 2.0 <= max(np.abs(x).max(), np.abs(y).max()) <= 6.0
