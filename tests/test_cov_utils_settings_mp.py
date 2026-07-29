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

import os
import resource

import numpy as np
import pandas as pd
import psutil
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

def test_activation_correlations_manders_nan_when_no_pixel_passes_both():
    """Threshold masks that never overlap -> M1/M2 are NaN, Pearson is not."""
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations

    # Two pixels; the bright pixel of the input is the dark pixel of the
    # activation map, so (input >= p50) & (act >= p50) is empty everywhere.
    inputs = torch.tensor([[[[0.0, 1.0]]]])          # (B=1, C=1, H=1, W=2)
    maps = torch.tensor([[[[1.0, 0.0]]]])            # same shape

    out = calculate_activation_correlations(
        inputs, maps, ["anti.png"], manders_thresholds=[50]
    )

    assert list(out["file_name"]) == ["anti.png"]
    # Pearson is well defined (perfect anti-correlation) ...
    assert out["channel_0_activation_0_pearsons"].iloc[0] == pytest.approx(-1.0)
    # ... but the Manders coefficients fall into the empty-mask branch.
    assert np.isnan(out["channel_0_activation_0_50_M1"].iloc[0])
    assert np.isnan(out["channel_0_activation_0_50_M2"].iloc[0])

    # Control: when the two channels agree, the mask is non-empty and the
    # coefficients are finite numbers, not NaN.
    same = calculate_activation_correlations(
        torch.tensor([[[[0.0, 1.0]]]]),
        torch.tensor([[[[0.0, 1.0]]]]),
        ["same.png"],
        manders_thresholds=[50],
    )
    assert same["channel_0_activation_0_50_M1"].iloc[0] == pytest.approx(1.0)
    assert same["channel_0_activation_0_50_M2"].iloc[0] == pytest.approx(1.0)


def test_activation_correlations_all_nan_channel_gives_nan_everywhere():
    """A channel that is entirely non-finite is filtered to size 0 -> all NaN."""
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations

    inputs = torch.full((1, 1, 2, 2), float("nan"))
    maps = torch.ones((1, 1, 2, 2))

    out = calculate_activation_correlations(
        inputs, maps, ["empty.png"], manders_thresholds=[15, 75]
    )

    assert len(out) == 1
    assert np.isnan(out["channel_0_activation_0_pearsons"].iloc[0])
    for thr in (15, 75):
        assert np.isnan(out[f"channel_0_activation_0_{thr}_M1"].iloc[0])
        assert np.isnan(out[f"channel_0_activation_0_{thr}_M2"].iloc[0])


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

def test_load_settings_parses_every_literal_type(tmp_path):
    """Each supported spelling round-trips into the right Python type."""
    from spacr.utils import load_settings

    csv = tmp_path / "s.csv"
    pd.DataFrame(
        {
            "setting_key": [
                "flag_t", "flag_f", "an_int", "a_float", "a_list",
                "a_tuple", "a_dict", "a_str", "blank", "broken_literal",
            ],
            "setting_value": [
                "True", "False", "42", "3.5", "[1, 2, 3]",
                "(4, 5)", "{'k': 'True', 'n': '7'}", "cell", "",
                "[1, 2",
            ],
        }
    ).to_csv(csv, index=False)

    out = load_settings(str(csv))

    assert out["flag_t"] is True
    assert out["flag_f"] is False
    assert out["an_int"] == 42 and isinstance(out["an_int"], int)
    assert out["a_float"] == pytest.approx(3.5)
    assert out["a_list"] == [1, 2, 3]
    assert out["a_tuple"] == (4, 5)
    # dict values are parsed recursively
    assert out["a_dict"] == {"k": True, "n": 7}
    assert out["a_str"] == "cell"
    assert out["blank"] is None
    # A malformed literal falls through literal_eval AND the numeric parse and
    # comes back verbatim.
    assert out["broken_literal"] == "[1, 2"


def test_load_settings_show_true_uses_display(tmp_path, monkeypatch):
    """``show=True`` routes the DataFrame through ``display`` exactly once."""
    import spacr.utils as U

    seen = []
    monkeypatch.setattr(U, "display", lambda df: seen.append(df))

    csv = tmp_path / "s.csv"
    pd.DataFrame(
        {"setting_key": ["a", "b"], "setting_value": ["1", "cell"]}
    ).to_csv(csv, index=False)

    out = U.load_settings(str(csv), show=True)

    assert out == {"a": 1, "b": "cell"}
    assert len(seen) == 1
    assert isinstance(seen[0], pd.DataFrame)
    assert list(seen[0]["setting_key"]) == ["a", "b"]


def test_load_settings_wrong_columns_raises(tmp_path):
    from spacr.utils import load_settings

    csv = tmp_path / "s.csv"
    pd.DataFrame({"Key": ["a"], "Value": ["cell"]}).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="setting_key"):
        load_settings(str(csv))

    # ...but the same file loads fine once the column names are declared.
    assert load_settings(str(csv), setting_key="Key", setting_value="Value") == {
        "a": "cell"
    }


def test_load_settings_all_numeric_column(tmp_path):
    """A settings CSV whose values are *all* numeric must still load.

    ``pd.read_csv`` types such a column as int64/float64, so ``parse_value``
    receives a numpy scalar and ``value.startswith(('(', '[', '{'))`` raises
    AttributeError. Correct behaviour: the numbers come back as numbers.
    """
    from spacr.utils import load_settings

    csv = tmp_path / "numeric.csv"
    pd.DataFrame(
        {"setting_key": ["nucleus_min_size", "cell_min_size"],
         "setting_value": [5, 100]}
    ).to_csv(csv, index=False)

    out = load_settings(str(csv))

    assert out == {"nucleus_min_size": 5, "cell_min_size": 100}


# ---------------------------------------------------------------------------
# pretty_print_settings
# ---------------------------------------------------------------------------

def test_pretty_print_settings_groups_by_category(capsys):
    """Known keys are printed under their category, unknown ones under 'Other'."""
    from spacr.settings import categories
    from spacr.utils import pretty_print_settings

    # Pick a real key out of a real category so the grouping branch is taken.
    cat_name, cat_keys = next(
        (c, ks) for c, ks in categories.items() if ks
    )
    known = cat_keys[0]

    pretty_print_settings({known: "value-x", "zz_unknown_key": 7}, title="My Run")
    out = capsys.readouterr().out

    assert "My Run" in out
    assert f"▸ {cat_name}" in out
    assert "▸ Other" in out
    assert "zz_unknown_key" in out
    assert "value-x" in out
    # Each key is printed exactly once (the `shown` de-dup set).
    assert out.count(f" {known} ") <= 1 or out.count("value-x") == 1


def test_pretty_print_settings_clips_long_values_and_no_other_header(
    capsys, monkeypatch
):
    """Missing spacr.settings.categories -> empty dict, everything is leftover."""
    import spacr.settings as S
    from spacr.utils import pretty_print_settings

    # Force `from .settings import categories` to raise ImportError so the
    # `except Exception: categories = {}` fallback executes.
    monkeypatch.delattr(S, "categories")

    long_value = "z" * 200
    pretty_print_settings({"src": long_value}, title="Fallback")
    out = capsys.readouterr().out

    assert "Fallback" in out
    # No category matched -> nothing was 'shown' -> no "Other" header is printed.
    assert "▸" not in out
    assert "src" in out
    # The value is clipped to 41 chars + an ellipsis.
    assert "…" in out
    assert long_value not in out
    assert "z" * 41 in out
    assert "z" * 42 not in out


def test_pretty_print_settings_empty_dict(capsys):
    """An empty settings dict still prints a well-formed box (key_w default)."""
    from spacr.utils import pretty_print_settings

    pretty_print_settings({}, title="Nothing")
    out = capsys.readouterr().out

    assert "Nothing" in out
    assert out.startswith("┌")
    # title(7) + 4 = 11 vs key_w(10) + 46 = 56 -> 56 box characters.
    assert "─" * 56 in out


# ---------------------------------------------------------------------------
# save_settings
# ---------------------------------------------------------------------------

def test_save_settings_forces_test_mode_and_plot_false(tmp_path):
    """The persisted copy is always a full headless run, and `settings` is not
    mutated in place."""
    from spacr.utils import load_settings, save_settings

    src = tmp_path / "plate01"
    src.mkdir()
    settings = {"src": str(src), "test_mode": True, "plot": True, "channels": [0, 1]}

    save_settings(settings, name="gen_mask_settings")

    out = src / "settings" / "gen_mask_settings.csv"
    assert out.is_file()
    reloaded = load_settings(str(out), setting_key="Key", setting_value="Value")
    assert reloaded["test_mode"] is False
    assert reloaded["plot"] is False
    assert reloaded["channels"] == [0, 1]
    # caller's dict untouched
    assert settings["test_mode"] is True and settings["plot"] is True


def test_save_settings_list_src_appends_list_suffix(tmp_path, capsys):
    """A list `src` writes into src[0] under a `<name>_list.csv` file."""
    from spacr.utils import save_settings

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    save_settings({"src": [str(a), str(b)], "test_mode": True}, name="exp", show=True)

    assert (a / "settings" / "exp_list.csv").is_file()
    assert not (b / "settings").exists()
    printed = capsys.readouterr().out
    # show=True routes through pretty_print_settings with a title-cased name
    assert "Exp List" in printed
    assert "Saving settings to" in printed


def test_save_settings_survives_unwritable_src(tmp_path, capsys):
    """A src that is a *file* makes makedirs raise OSError; the run continues."""
    from spacr.utils import save_settings

    bogus = tmp_path / "not_a_dir.txt"
    bogus.write_text("i am a file\n")

    # Must not raise.
    assert save_settings({"src": str(bogus)}, name="settings") is None

    printed = capsys.readouterr().out
    assert "Warning: could not save settings to" in printed
    assert "Continuing without writing the settings copy." in printed
    # Nothing was written next to the file.
    assert not (tmp_path / "not_a_dir.txt" / "settings").exists()


def test_save_settings_permission_error_is_swallowed(tmp_path, monkeypatch, capsys):
    """A read-only src (PermissionError from makedirs) is reported, not raised."""
    import spacr.utils as U

    src = tmp_path / "ro"
    src.mkdir()

    def _boom(*a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(U.os, "makedirs", _boom)

    U.save_settings({"src": str(src), "test_mode": False}, name="settings")

    printed = capsys.readouterr().out
    assert "Warning: could not save settings" in printed
    assert not (src / "settings").exists()


# ---------------------------------------------------------------------------
# print_progress
# ---------------------------------------------------------------------------

def test_print_progress_list_batch_size_uses_its_length(capsys):
    """A list batch_size is reduced to len() before the per-image ETA divide."""
    from spacr.utils import print_progress

    print_progress(
        files_processed=10,
        files_to_process=20,
        n_jobs=2,
        time_ls=[4.0, 4.0],
        batch_size=["f1", "f2", "f3", "f4"],
        operation_type="measure",
    )
    out = capsys.readouterr().out

    # average_time = 4.0, batch_size = 4 -> time/image = 1.0
    assert "Time/batch: 4.000sec" in out
    assert "Time/image: 1.000sec" in out
    # (20-10) * 4.0 / 2 / 60 = 0.333 min
    assert "Time_left: 0.333 min." in out
    assert "Progress: 10/20" in out
    assert "operation_type: measure" in out


def test_print_progress_zero_workers_keeps_eta_finite(capsys):
    from spacr.utils import print_progress

    print_progress(
        2, 10, n_jobs=0, time_ls=[1.0, 3.0], batch_size=0,
        operation_type="in-process",
    )
    out = capsys.readouterr().out
    assert "Time/image: 2.000sec" in out
    assert "Time_left: 0.267 min." in out
    assert "nan" not in out.lower()
    assert "inf" not in out.lower()


def test_print_progress_coerces_numeric_strings_and_floats(capsys):
    """Non-int counters that *can* be coerced go through int()."""
    from spacr.utils import print_progress

    print_progress("7", 12.9, n_jobs=1, time_ls=None, operation_type="coerce")
    out = capsys.readouterr().out

    assert "Progress: 7/12," in out
    assert "None" in out  # time_ls=None -> time_info is None


def test_print_progress_uncoercible_counters_fall_back_to_zero(capsys):
    """Garbage counters degrade to 0 instead of crashing the pipeline."""
    from spacr.utils import print_progress

    print_progress(object(), "not-a-number", n_jobs=1, time_ls=[1.0],
                   operation_type="junk")
    out = capsys.readouterr().out

    assert "Progress: 0/0," in out
    assert "operation_type: junk" in out
    # (0-0)*1.0/1/60 == 0
    assert "Time_left: 0.000 min." in out


def test_print_progress_lists_are_deduplicated(capsys):
    """List inputs are counted as *unique* items."""
    from spacr.utils import print_progress

    print_progress([1, 1, 2], [1, 2, 3, 3, 4], n_jobs=1, time_ls=[])
    out = capsys.readouterr().out

    assert "Progress: 2/4," in out
    # empty time_ls -> average_time 0
    assert "Time/image: 0.000sec" in out


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

class _FakeProc:
    """Minimal psutil.Process stand-in for the mp helpers."""

    def __init__(self, pid, cmdline, cmdline_exc=None, terminate_exc=None):
        self.info = {"pid": pid, "cmdline": cmdline}
        self._cmdline = cmdline
        self._cmdline_exc = cmdline_exc
        self._terminate_exc = terminate_exc
        self.terminated = False
        self.waited_timeout = None

    def cmdline(self):
        if self._cmdline_exc is not None:
            raise self._cmdline_exc
        return self._cmdline

    def terminate(self):
        if self._terminate_exc is not None:
            raise self._terminate_exc
        self.terminated = True

    def wait(self, timeout=None):
        self.waited_timeout = timeout


@pytest.mark.parametrize(
    "exc",
    [
        psutil.NoSuchProcess(4242),
        psutil.AccessDenied(4242),
        psutil.ZombieProcess(4242),
    ],
    ids=["no_such_process", "access_denied", "zombie"],
)
def test_is_multiprocessing_process_swallows_psutil_errors(exc):
    """A process that vanishes / is unreadable is reported as *not* mp."""
    from spacr.utils import is_multiprocessing_process

    proc = _FakeProc(4242, [], cmdline_exc=exc)
    assert is_multiprocessing_process(proc) is False


def test_is_multiprocessing_process_matches_substring():
    from spacr.utils import is_multiprocessing_process

    hit = _FakeProc(1, ["python", "-c", "from multiprocessing.spawn import x"])
    miss = _FakeProc(2, ["python", "-m", "pytest"])

    assert is_multiprocessing_process(hit) is True
    assert is_multiprocessing_process(miss) is False


# ---------------------------------------------------------------------------
# close_file_descriptors
# ---------------------------------------------------------------------------

def test_close_file_descriptors_sweeps_3_to_soft_limit(monkeypatch):
    """Every fd in [3, soft) is closed; an already-closed fd is ignored."""
    import spacr.utils as U

    closed = []

    def _fake_close(fd):
        if fd == 4:
            raise OSError(9, "Bad file descriptor")
        closed.append(fd)

    monkeypatch.setattr(resource, "getrlimit", lambda which: (6, 1024))
    monkeypatch.setattr(U.os, "close", _fake_close)
    try:
        U.close_file_descriptors()
    finally:
        # Restore os.close before touching anything else (asserts, capture...).
        monkeypatch.undo()

    # fd 4 raised OSError and was swallowed; 0-2 are never touched.
    assert closed == [3, 5]


def test_close_file_descriptors_noop_when_soft_limit_is_three(monkeypatch):
    """soft == 3 -> nothing to sweep, os.close is never called."""
    import spacr.utils as U

    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda which: (3, 1024))
    monkeypatch.setattr(U.os, "close", lambda fd: calls.append(fd))
    try:
        U.close_file_descriptors()
    finally:
        monkeypatch.undo()

    assert calls == []


# ---------------------------------------------------------------------------
# close_multiprocessing_processes
# ---------------------------------------------------------------------------

def test_close_multiprocessing_processes_terminates_only_mp_children(
    monkeypatch, capsys
):
    """Self is skipped, mp children are terminated+waited, others untouched,
    and the fd sweep runs last."""
    import spacr.utils as U

    me = os.getpid()
    self_proc = _FakeProc(me, ["python", "-c", "from multiprocessing import x"])
    mp_child = _FakeProc(me + 1, ["python", "-c", "from multiprocessing.spawn import y"])
    other = _FakeProc(me + 2, ["bash", "-lc", "sleep 1"])

    monkeypatch.setattr(
        U.psutil, "process_iter", lambda attrs=None: iter([self_proc, mp_child, other])
    )
    swept = []
    monkeypatch.setattr(U, "close_file_descriptors", lambda: swept.append(True))

    U.close_multiprocessing_processes()

    assert self_proc.terminated is False  # `continue` on our own pid
    assert mp_child.terminated is True
    assert mp_child.waited_timeout == 5
    assert other.terminated is False
    assert swept == [True]

    out = capsys.readouterr().out
    assert f"Terminated process {me + 1}" in out
    assert f"Terminated process {me}" not in out


def test_close_multiprocessing_processes_reports_failures(monkeypatch, capsys):
    """A child that dies / denies access between iteration and terminate is
    reported, and the sweep still finishes."""
    import spacr.utils as U

    me = os.getpid()
    gone = _FakeProc(
        me + 11,
        ["python", "multiprocessing.forkserver"],
        terminate_exc=psutil.NoSuchProcess(me + 11),
    )
    denied = _FakeProc(
        me + 12,
        ["python", "multiprocessing.resource_tracker"],
        terminate_exc=psutil.AccessDenied(me + 12),
    )
    survivor = _FakeProc(me + 13, ["python", "-c", "import multiprocessing"])

    monkeypatch.setattr(
        U.psutil, "process_iter", lambda attrs=None: iter([gone, denied, survivor])
    )
    swept = []
    monkeypatch.setattr(U, "close_file_descriptors", lambda: swept.append(True))

    # Must not propagate the psutil errors.
    U.close_multiprocessing_processes()

    out = capsys.readouterr().out
    assert f"Failed to terminate process {me + 11}" in out
    assert f"Failed to terminate process {me + 12}" in out
    # The loop kept going after both failures.
    assert survivor.terminated is True
    assert f"Terminated process {me + 13}" in out
    assert swept == [True]


def test_close_multiprocessing_processes_empty_iter_still_sweeps(monkeypatch):
    import spacr.utils as U

    monkeypatch.setattr(U.psutil, "process_iter", lambda attrs=None: iter([]))
    swept = []
    monkeypatch.setattr(U, "close_file_descriptors", lambda: swept.append(True))

    U.close_multiprocessing_processes()

    assert swept == [True]


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
        (masks / f"{i}.npy").write_bytes(b"")
    # Clutter that must not be counted.
    (masks / "notes.txt").write_text("hi")
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
