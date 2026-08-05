"""What ``spacr.core.preprocess_generate_masks`` actually orchestrates.

The Cellpose/SAM segmentation itself is a GPU model fit and is stubbed, but
everything spaCR's mask pipeline is *responsible for* is exercised for real
and asserted on by value:

* which object types get segmented, in which order, and with which folder;
* the "masks already generated" skip (:func:`spacr.utils.check_mask_folder`);
* the exact channel arguments handed to ``_load_and_concatenate_arrays``;
* which cell-mask adjustment folders are passed, and when adjustment is
  skipped entirely;
* the overlay-plot budget (``examples_to_plot``) and per-example failure
  isolation through the plot :class:`~spacr.errors.RunLedger`;
* the keep_intermediate / delete_intermediate truth table handed to cleanup;
* the run-status stamp written into ``measurements.db``.

Everything is CPU-only, offline and deterministic.
"""
from __future__ import annotations

import os
import random
import sqlite3

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


class Spy:
    """Records every call; returns ``ret``. Used for the GPU/plot externals."""

    def __init__(self, ret=None, fail_on=None):
        self.calls = []
        self.ret = ret
        self.fail_on = fail_on

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.fail_on is not None and self.fail_on(*args, **kwargs):
            raise RuntimeError("boom")
        return self.ret

    @property
    def n(self):
        return len(self.calls)


def _mask_settings(src, **over):
    s = {
        "src": str(src),
        "metadata_type": "cellvoyager",
        "channels": [0, 1, 2],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": None,
        "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False,
        "batch_size": 10, "save": True, "custom_regex": None,
        "randomize": True, "examples_to_plot": 2,
    }
    s.update(over)
    return s


@pytest.fixture
def stubs(monkeypatch):
    """Stub the GPU segmentation + the disk-heavy merge/plot collaborators.

    Everything here is a true externality for this module: a Cellpose-SAM
    forward pass, the array-merging IO pass and matplotlib overlay rendering.
    The orchestration under test is core's own.
    """
    import spacr.object as sobj
    import spacr.io as sio
    import spacr.plot as splot
    import spacr.utils as su

    sp = {
        "cellpose": Spy(),
        "organelle": Spy(),
        "concat": Spy(),
        "overlay": Spy(),
        "arrays": Spy(),
        "adjust": Spy(),
        "pivot": Spy(),
        "cleanup": Spy(ret=[]),
    }
    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam", sp["cellpose"])
    monkeypatch.setattr(sobj, "generate_organelle_masks_sam", sp["organelle"])
    monkeypatch.setattr(sio, "_load_and_concatenate_arrays", sp["concat"])
    monkeypatch.setattr(splot, "plot_image_mask_overlay", sp["overlay"])
    monkeypatch.setattr(splot, "plot_arrays", sp["arrays"])
    monkeypatch.setattr(su, "adjust_cell_masks", sp["adjust"])
    monkeypatch.setattr(su, "_pivot_counts_table", sp["pivot"])
    monkeypatch.setattr(su, "cleanup_pipeline_folders", sp["cleanup"])
    return sp


@pytest.fixture
def run_dir(tmp_path):
    """A v1 run folder with a 3-field stack/ and no masks yet."""
    src = tmp_path / "plate1"
    (src / "stack").mkdir(parents=True)
    for i in range(3):
        np.save(src / "stack" / f"f{i}.npy", np.zeros((3, 8, 8), np.uint16))
    return src


# ---------------------------------------------------------------------------
# which objects get segmented
# ---------------------------------------------------------------------------

def test_only_the_configured_channels_are_segmented(run_dir, stubs):
    """One segmentation call per non-None object channel, in cell → nucleus →
    pathogen → organelle order, all against <src>/masks."""
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(
        run_dir, cell_channel=1, nucleus_channel=0, pathogen_channel=2,
        organelle_channel=None))

    objs = [c[0][2] for c in stubs["cellpose"].calls]
    assert objs == ["cell", "nucleus", "pathogen"]
    mask_src = os.path.join(str(run_dir), "masks")
    assert {c[0][0] for c in stubs["cellpose"].calls} == {mask_src}
    # organelle_channel is None → the organelle segmenter is never touched
    assert stubs["organelle"].n == 0


def test_organelle_channel_uses_the_organelle_segmenter(run_dir, stubs):
    """organelle_channel routes to generate_organelle_masks_sam, not cellpose."""
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(
        run_dir, cell_channel=1, nucleus_channel=None,
        pathogen_channel=None, organelle_channel=2))

    assert [c[0][2] for c in stubs["cellpose"].calls] == ["cell"]
    assert stubs["organelle"].n == 1
    args, _ = stubs["organelle"].calls[0]
    assert args[0] == os.path.join(str(run_dir), "masks")
    assert args[2] == "organelle"


def test_existing_masks_are_not_regenerated(run_dir, stubs):
    """check_mask_folder: a cell_mask_stack with one .npy per stack field is
    complete, so cell is skipped while nucleus still runs."""
    from spacr.core import preprocess_generate_masks

    done = run_dir / "masks" / "cell_mask_stack"
    done.mkdir(parents=True)
    for i in range(3):                      # matches the 3 stack/*.npy
        np.save(done / f"f{i}.npy", np.zeros((8, 8), np.uint16))

    preprocess_generate_masks(_mask_settings(
        run_dir, cell_channel=1, nucleus_channel=0))

    assert [c[0][2] for c in stubs["cellpose"].calls] == ["nucleus"]


def test_partial_mask_folder_is_regenerated(run_dir, stubs):
    """Fewer masks than stack fields → the object is segmented again."""
    from spacr.core import preprocess_generate_masks

    partial = run_dir / "masks" / "cell_mask_stack"
    partial.mkdir(parents=True)
    np.save(partial / "f0.npy", np.zeros((8, 8), np.uint16))   # 1 of 3

    preprocess_generate_masks(_mask_settings(
        run_dir, cell_channel=1, nucleus_channel=None))

    assert [c[0][2] for c in stubs["cellpose"].calls] == ["cell"]


# ---------------------------------------------------------------------------
# merge step
# ---------------------------------------------------------------------------

def test_merge_receives_every_channel_index_positionally(run_dir, stubs):
    """_load_and_concatenate_arrays is positional — a channel handed over in
    the wrong slot silently merges the wrong plane into merged/."""
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(
        run_dir, channels=[0, 1, 2, 3], cell_channel=1, nucleus_channel=0,
        pathogen_channel=2, organelle_channel=3, resume=True))

    assert stubs["concat"].n == 1
    args, kwargs = stubs["concat"].calls[0]
    assert args[0] == str(run_dir)
    assert args[1] == [0, 1, 2, 3]
    assert args[2] == 1      # cell
    assert args[3] == 0      # nucleus
    assert args[4] == 2      # pathogen
    assert args[5] == 3      # organelle
    assert kwargs == {"resume": True}


def test_merge_resume_defaults_to_false(run_dir, stubs):
    from spacr.core import preprocess_generate_masks
    preprocess_generate_masks(_mask_settings(run_dir))
    assert stubs["concat"].calls[0][1] == {"resume": False}


def test_pivot_runs_only_when_a_measurements_folder_exists(run_dir, stubs):
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(run_dir))
    assert stubs["pivot"].n == 0, "no measurements/ folder → nothing to pivot"

    (run_dir / "measurements").mkdir()
    preprocess_generate_masks(_mask_settings(run_dir))
    assert stubs["pivot"].n == 1
    assert stubs["pivot"].calls[0][1] == {
        "db_path": os.path.join(str(run_dir), "measurements", "measurements.db")}


# ---------------------------------------------------------------------------
# cell mask adjustment
# ---------------------------------------------------------------------------

def test_adjust_cells_passes_the_three_mask_folders(run_dir, stubs):
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(
        run_dir, adjust_cells=True, cell_channel=1, nucleus_channel=0,
        pathogen_channel=2, n_jobs=3))

    assert stubs["adjust"].n == 1
    args, kwargs = stubs["adjust"].calls[0]
    masks = os.path.join(str(run_dir), "masks")
    assert args[0] == os.path.join(masks, "pathogen_mask_stack")
    assert args[1] == os.path.join(masks, "cell_mask_stack")
    assert args[2] == os.path.join(masks, "nucleus_mask_stack")
    assert args[3] is None, "no organelle_mask_stack on disk → None"
    assert kwargs == {"overlap_threshold": 5, "perimeter_threshold": 30,
                      "n_jobs": 3}


def test_adjust_cells_includes_organelle_folder_only_when_present(run_dir, stubs):
    from spacr.core import preprocess_generate_masks

    (run_dir / "masks" / "organelle_mask_stack").mkdir(parents=True)
    preprocess_generate_masks(_mask_settings(
        run_dir, adjust_cells=True, cell_channel=1, nucleus_channel=0,
        pathogen_channel=2, organelle_channel=3))

    assert stubs["adjust"].calls[0][0][3] == os.path.join(
        str(run_dir), "masks", "organelle_mask_stack")


@pytest.mark.parametrize("over", [
    {"pathogen_channel": None},
    {"nucleus_channel": None},
    {"cell_channel": None},
    {"timelapse": True},
])
def test_adjust_cells_is_skipped_without_all_three_objects(run_dir, stubs, over):
    """Adjustment reconciles cell vs nuclei vs pathogen; missing any of the
    three (or a timelapse run) must skip it rather than call with a folder
    that was never written."""
    from spacr.core import preprocess_generate_masks

    s = _mask_settings(run_dir, adjust_cells=True, cell_channel=1,
                       nucleus_channel=0, pathogen_channel=2)
    s.update(over)
    preprocess_generate_masks(s)
    assert stubs["adjust"].n == 0


# ---------------------------------------------------------------------------
# overlay plots
# ---------------------------------------------------------------------------

def _merged(run_dir, n=5):
    merged = run_dir / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.save(merged / f"m{i}.npy", np.zeros((5, 8, 8), np.uint16))
    return merged


def test_examples_to_plot_caps_the_overlay_count(run_dir, stubs):
    """5 merged fields, examples_to_plot=2 → exactly 2 overlays, each of a
    real merged file, each with the pipeline's channel arguments."""
    from spacr.core import preprocess_generate_masks

    merged = _merged(run_dir, n=5)
    random.seed(0)
    preprocess_generate_masks(_mask_settings(
        run_dir, plot=True, examples_to_plot=2, channels=[0, 1, 2],
        cell_channel=1, nucleus_channel=0, pathogen_channel=2))

    assert stubs["overlay"].n == 2
    for args, kwargs in stubs["overlay"].calls:
        assert os.path.dirname(args[0]) == str(merged)
        assert os.path.isfile(args[0])
        assert args[1] == [0, 1, 2]
        assert (args[2], args[3], args[4]) == (1, 0, 2)
        assert kwargs["organelle_channel"] is None
        assert kwargs["figuresize"] == 10
        assert kwargs["percentiles"] == (1, 99)
        assert kwargs["thickness"] == 3
        assert kwargs["save_pdf"] is True


def test_test_mode_plots_every_merged_field(run_dir, stubs):
    """BUG (fixed): test_mode set examples_to_plot to len() of the merged
    *path string*, a number that depends on how deep tmp_path is nested, not
    on how many fields exist. It must be the number of merged .npy files."""
    from spacr.core import preprocess_generate_masks

    _merged(run_dir, n=4)
    s = _mask_settings(run_dir, plot=True, test_mode=True, examples_to_plot=1)
    random.seed(0)
    preprocess_generate_masks(s)

    assert s["examples_to_plot"] == 4
    assert stubs["overlay"].n == 4


def test_one_unplottable_field_does_not_cancel_the_rest(run_dir, stubs, capsys):
    """The per-example ledger: a field that fails to render is recorded and
    the remaining fields are still plotted."""
    from spacr.core import preprocess_generate_masks

    stubs["overlay"].fail_on = lambda *a, **k: a[0].endswith("m2.npy")
    _merged(run_dir, n=5)
    random.seed(0)
    preprocess_generate_masks(_mask_settings(
        run_dir, plot=True, examples_to_plot=5))

    assert stubs["overlay"].n == 5, "every field was attempted"
    printed = capsys.readouterr().out
    assert "Failed to plot image mask overly. Error: boom" in printed
    assert "RUN INCOMPLETE" in printed
    # the run itself still completes: an overlay is cosmetic
    assert "Successfully completed run" in printed


def test_missing_merged_folder_reports_and_continues(run_dir, stubs, capsys):
    """No merged/ at all → the listing failure is reported, no overlay is
    attempted, and the run still finishes."""
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(run_dir, plot=True))
    printed = capsys.readouterr().out
    assert "Failed to plot image mask overly." in printed
    assert stubs["overlay"].n == 0
    assert "Successfully completed run" in printed


def test_timelapse_plots_arrays_instead_of_overlays(run_dir, stubs):
    """A timelapse run renders the merged stack with plot_arrays, using the
    figure settings, and never calls the per-field overlay."""
    from spacr.core import preprocess_generate_masks

    _merged(run_dir, n=3)
    preprocess_generate_masks(_mask_settings(
        run_dir, plot=True, timelapse=True, examples_to_plot=3,
        figuresize=7, cmap="viridis", normalize=False))

    assert stubs["overlay"].n == 0
    assert stubs["arrays"].n == 1
    _args, kwargs = stubs["arrays"].calls[0]
    assert kwargs["src"] == os.path.join(str(run_dir), "merged")
    assert kwargs["figuresize"] == 7
    assert kwargs["cmap"] == "viridis"
    assert kwargs["nr"] == 3
    assert kwargs["normalize"] is False
    assert (kwargs["q1"], kwargs["q2"]) == (1, 99)


# ---------------------------------------------------------------------------
# cleanup + run accounting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keep_int,keep_orig,delete_int,exp_int,exp_orig", [
    (False, False, False, False, False),
    (True, True, False, True, True),
    # the legacy delete_intermediate flag overrides both keep flags
    (True, True, True, False, False),
    (False, True, False, False, True),
])
def test_cleanup_keep_flags(run_dir, stubs, keep_int, keep_orig, delete_int,
                            exp_int, exp_orig):
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(
        run_dir, keep_intermediate=keep_int, keep_original_images=keep_orig,
        delete_intermediate=delete_int))

    args, kwargs = stubs["cleanup"].calls[0]
    assert args[0] == str(run_dir)
    assert kwargs == {"keep_intermediate": exp_int, "keep_original": exp_orig}


def test_run_status_is_stamped_into_measurements_db(run_dir, stubs):
    """A completed run records itself in the db it produced."""
    from spacr.core import preprocess_generate_masks
    from spacr.errors import read_run_status

    meas = run_dir / "measurements"
    meas.mkdir()
    db = meas / "measurements.db"
    sqlite3.connect(str(db)).close()

    preprocess_generate_masks(_mask_settings(run_dir))

    runs = read_run_status(str(db))
    assert len(runs) == 1
    assert runs[0]["status"] == "complete"
    assert runs[0]["n_succeeded"] == 1
    assert runs[0]["n_failed"] == 0


def test_two_folders_are_both_segmented_and_stamped(tmp_path, stubs):
    """A list src runs each folder and stamps each folder's own db."""
    from spacr.core import preprocess_generate_masks
    from spacr.errors import read_run_status

    dbs = []
    for name in ("p1", "p2"):
        d = tmp_path / name
        (d / "stack").mkdir(parents=True)
        np.save(d / "stack" / "f0.npy", np.zeros((3, 8, 8), np.uint16))
        (d / "measurements").mkdir()
        db = d / "measurements" / "measurements.db"
        sqlite3.connect(str(db)).close()
        dbs.append(db)

    s = _mask_settings(tmp_path / "p1")
    s["src"] = [str(tmp_path / "p1"), str(tmp_path / "p2")]
    preprocess_generate_masks(s)

    # 2 folders x (cell + nucleus)
    assert stubs["cellpose"].n == 4
    for db in dbs:
        runs = read_run_status(str(db))
        assert len(runs) == 1 and runs[0]["n_succeeded"] == 2


# ---------------------------------------------------------------------------
# strict mode
# ---------------------------------------------------------------------------

def test_no_object_channel_raises_under_strict_errors(run_dir, monkeypatch):
    """SPACR_STRICT_ERRORS turns the "nothing to segment" print into a stop."""
    from spacr.core import preprocess_generate_masks
    from spacr.errors import ConfigurationError

    monkeypatch.setenv("SPACR_STRICT_ERRORS", "1")
    with pytest.raises(ConfigurationError) as ei:
        preprocess_generate_masks(_mask_settings(
            run_dir, cell_channel=None, nucleus_channel=None,
            pathogen_channel=None, organelle_channel=None))
    assert "organelle_channel" in str(ei.value)


def test_converter_failure_raises_under_strict_errors(run_dir, monkeypatch):
    """A folder whose files cannot be renamed is a hard stop in strict mode —
    every later stage would be reading an unrecognised folder."""
    import spacr.core as core
    import spacr.io as sio
    from spacr.errors import ConfigurationError

    def _boom(*a, **k):
        raise RuntimeError("no metadata")

    monkeypatch.setattr(sio, "convert_to_yokogawa", _boom)
    monkeypatch.setenv("SPACR_STRICT_ERRORS", "1")
    with pytest.raises(ConfigurationError) as ei:
        core.preprocess_generate_masks(_mask_settings(
            run_dir, metadata_type="auto", masks=False))
    assert str(run_dir) in str(ei.value)


# ---------------------------------------------------------------------------
# the Timelapse entry point
# ---------------------------------------------------------------------------

def test_timelapse_entry_forces_timelapse_on(run_dir, stubs, capsys):
    """preprocess_generate_masks_timelapse overrides an incoming
    timelapse=False, says so, and switches randomization off."""
    import pandas as pd
    from spacr.core import preprocess_generate_masks_timelapse

    s = _mask_settings(run_dir, timelapse=False, randomize=True, masks=False)
    preprocess_generate_masks_timelapse(s)

    printed = capsys.readouterr().out
    assert "forcing it" in printed
    # the settings dict is canonicalised in place, so the caller sees both
    assert s["timelapse"] is True
    assert s["randomize"] is False
    saved = pd.read_csv(run_dir / "settings" / "gen_mask_settings.csv")
    vals = dict(zip(saved["Key"], saved["Value"].astype(str)))
    assert vals["timelapse"] == "True"
    # NOTE: gen_mask_settings.csv is written *before* the timelapse branch
    # switches randomize off, so the CSV still records the incoming True.
    # Reported, not changed here: moving save_settings past the
    # normalisations would also change what the repro/diff CLI records.
    assert vals["randomize"] == "True"


def test_timelapse_entry_accepts_none(tmp_path, monkeypatch, stubs, capsys):
    """settings=None is normalised to {} and canonicalised, rather than
    raising AttributeError on None.get."""
    import spacr.core as core

    seen = {}
    real = core.preprocess_generate_masks

    def _capture(settings):
        seen.update(settings)
        return None

    monkeypatch.setattr(core, "preprocess_generate_masks", _capture)
    core.preprocess_generate_masks_timelapse(None)
    assert seen["timelapse"] is True
    assert seen["src"] == "path"        # the settings default, untouched
    assert core.preprocess_generate_masks is _capture
    monkeypatch.undo()
    assert core.preprocess_generate_masks is real


# ---------------------------------------------------------------------------
# module import fallback
# ---------------------------------------------------------------------------

def test_display_falls_back_to_a_noop_without_ipython(monkeypatch):
    """spacr.core must import even when IPython.display cannot be imported
    (it is imported mid-init by another thread often enough to matter); the
    fallback display() is a silent no-op, not a missing name."""
    import importlib.util
    import sys

    import spacr.core as core

    monkeypatch.setitem(sys.modules, "IPython.display", None)
    spec = importlib.util.spec_from_file_location(
        "spacr._core_no_ipython", core.__file__)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "spacr"
    spec.loader.exec_module(mod)

    assert mod.display is not core.display
    assert mod.display("anything", key=1) is None
    assert callable(mod.preprocess_generate_masks)
