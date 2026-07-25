"""Tests for the ``dry_run`` (validate-only) setting.

The contract dry_run has to keep is narrow and absolute: parse the settings,
check the inputs, print the report and the plan, and return **before** any
model loads, any GPU work starts, or any file is written. The tests below
assert all three halves of that:

  * the temp tree is byte-for-byte unchanged afterwards,
  * the heavy modules the pipeline would import are never touched (they are
    replaced by stubs that raise on any attribute access),
  * with dry_run off, those same stubs *do* fire — which is what proves the
    guard, rather than something else, is what stopped the run.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

import spacr.core as core
import spacr.measure as measure
import spacr.settings as S
from spacr.validate import Problem, run_preflight


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _ExplodingModule(types.ModuleType):
    """Stand-in for a module that a dry run must never reach."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(
            f"dry_run reached {self.__name__}.{name} — compute was not skipped")


def _stub_heavy_modules(monkeypatch, *names):
    """Replace ``names`` in sys.modules with modules that raise when used."""
    for name in names:
        stub = _ExplodingModule(name)
        stub.__spec__ = None
        stub.__loader__ = None
        stub.__package__ = "spacr"
        stub.__file__ = f"<stub {name}>"
        monkeypatch.setitem(sys.modules, name, stub)


def snapshot(root):
    """Every path under ``root`` with its size and mtime, for exact comparison."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(str(root)):
        for name in dirnames:
            out[os.path.join(dirpath, name)] = "dir"
        for name in filenames:
            path = os.path.join(dirpath, name)
            stat = os.stat(path)
            out[path] = (stat.st_size, stat.st_mtime_ns)
    return out


@pytest.fixture
def merged_plate(tmp_path):
    """A plate with a populated ``merged/`` of 7-plane arrays."""
    plate = tmp_path / "plate1"
    merged = plate / "merged"
    merged.mkdir(parents=True)
    for i in range(3):
        np.save(str(merged / f"plate1_A01_{i}_1.npy"),
                np.zeros((8, 8, 7), dtype=np.uint16))
    return plate, merged


@pytest.fixture
def raw_plate(tmp_path):
    """A plate of CellVoyager-named raw tifs: 4 fields x 3 channels."""
    plate = tmp_path / "plate1"
    plate.mkdir()
    for field in range(1, 5):
        for chan in range(1, 4):
            (plate / f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif").write_bytes(b"")
    return plate


def measure_settings(merged, **overrides):
    settings = {
        "src": str(merged),
        "dry_run": True,
        "timelapse": False,
        "cell_mask_dim": 4,
        "nucleus_mask_dim": 5,
        "pathogen_mask_dim": 6,
        "channels": [0, 1, 2, 3],
        "crop_mode": ["cell"],
        "save_png": True,
        "png_size": [224, 224],
        "normalize": [1, 99],
        "normalize_by": "png",
        "n_jobs": 2,
    }
    settings.update(overrides)
    return settings


def mask_settings(plate, **overrides):
    settings = {
        "src": str(plate),
        "dry_run": True,
        "metadata_type": "cellvoyager",
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": 2,
        "organelle_channel": None,
        "channels": [0, 1, 2],
        "magnification": 20,
    }
    settings.update(overrides)
    return settings


# ---------------------------------------------------------------------------
# the setting itself
# ---------------------------------------------------------------------------

def test_dry_run_is_a_declared_boolean_setting():
    assert S.expected_types["dry_run"] is bool


def test_dry_run_has_a_tooltip():
    assert "dry_run" in S.tooltips
    assert S.tooltips["dry_run"].startswith("(bool) - ")


def test_dry_run_tooltip_meets_the_house_bar():
    """Same floors tests/test_settings_tooltip_quality.py enforces."""
    body = S.tooltips["dry_run"].split(" - ", 1)[1]
    assert len(body.split()) >= 15
    assert "\n" not in body and "**" not in body and "`" not in body


def test_dry_run_tooltip_says_what_it_actually_prevents():
    text = S.tooltips["dry_run"].lower()
    assert "before any compute" in text
    assert "nothing is written" in text


@pytest.mark.parametrize("setter", [
    "set_default_settings_preprocess_generate_masks",
    "get_measure_crop_settings",
])
def test_dry_run_defaults_to_false(setter):
    assert getattr(S, setter)({})["dry_run"] is False


@pytest.mark.parametrize("setter", [
    "set_default_settings_preprocess_generate_masks",
    "get_measure_crop_settings",
])
def test_dry_run_respects_a_caller_supplied_value(setter):
    assert getattr(S, setter)({"dry_run": True})["dry_run"] is True


def test_dry_run_is_offered_in_the_settings_panel():
    grouped = {k for group in S.categories.values() for k in group}
    assert "dry_run" in grouped, "dry_run must land in a settings category to be reachable in the GUI"


# ---------------------------------------------------------------------------
# measure_crop
# ---------------------------------------------------------------------------

def test_measure_crop_dry_run_writes_nothing(tmp_path, merged_plate, capsys):
    _plate, merged = merged_plate
    before = snapshot(tmp_path)

    measure.measure_crop(measure_settings(merged))

    assert snapshot(tmp_path) == before, "dry_run modified the source tree"


def test_measure_crop_dry_run_creates_no_output_folders(tmp_path, merged_plate, capsys):
    plate, merged = merged_plate
    measure.measure_crop(measure_settings(merged))

    for created in ("measurements", "data", "settings", "test"):
        assert not (plate / created).exists(), f"dry_run created {created}/"


def test_measure_crop_dry_run_returns_before_any_heavy_import(monkeypatch, merged_plate, capsys):
    """measure_crop's first act is `from .io import _save_settings_to_db`."""
    _plate, merged = merged_plate
    _stub_heavy_modules(monkeypatch, "spacr.io", "spacr.timelapse")

    measure.measure_crop(measure_settings(merged))  # must not raise


def test_measure_crop_without_dry_run_does_reach_the_heavy_imports(monkeypatch, merged_plate):
    """The inverse of the test above: proves the guard is what stops the run."""
    _plate, merged = merged_plate
    _stub_heavy_modules(monkeypatch, "spacr.io", "spacr.timelapse")

    with pytest.raises(AssertionError, match="compute was not skipped"):
        measure.measure_crop(measure_settings(merged, dry_run=False))


def test_measure_crop_dry_run_never_starts_a_worker_pool(monkeypatch, merged_plate, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("dry_run started a multiprocessing pool")

    monkeypatch.setattr(measure.mp, "Pool", boom)
    monkeypatch.setattr(measure, "_measure_crop_core", boom)
    _plate, merged = merged_plate

    measure.measure_crop(measure_settings(merged))


def test_measure_crop_dry_run_returns_the_problem_list(merged_plate, capsys):
    _plate, merged = merged_plate
    result = measure.measure_crop(measure_settings(merged))
    assert isinstance(result, list)
    assert all(isinstance(p, Problem) for p in result)


def test_measure_crop_dry_run_surfaces_a_bad_mask_dim(merged_plate, capsys):
    _plate, merged = merged_plate
    problems = measure.measure_crop(measure_settings(merged, pathogen_mask_dim=11))
    assert [p for p in problems if p.setting == "pathogen_mask_dim" and p.severity == "error"]
    assert "pathogen_mask_dim" in capsys.readouterr().out


def test_measure_crop_dry_run_prints_the_report_and_the_plan(merged_plate, capsys):
    _plate, merged = merged_plate
    measure.measure_crop(measure_settings(merged))
    out = capsys.readouterr().out

    assert "spaCR pre-flight check" in out
    assert "Plan" in out
    assert "spacr.measure.measure_crop" in out
    assert "3 merged arrays" in out
    assert "dry_run=True" in out


def test_measure_crop_dry_run_reports_a_missing_merged_folder(tmp_path, capsys):
    plate = tmp_path / "plate1"
    plate.mkdir()
    problems = measure.measure_crop(measure_settings(plate))
    assert [p for p in problems if "merged" in p.message and p.severity == "error"]
    assert snapshot(tmp_path) == snapshot(tmp_path)
    assert not (plate / "merged").exists()


# ---------------------------------------------------------------------------
# preprocess_generate_masks
# ---------------------------------------------------------------------------

def test_preprocess_generate_masks_dry_run_writes_nothing(tmp_path, raw_plate, capsys):
    before = snapshot(tmp_path)
    core.preprocess_generate_masks(mask_settings(raw_plate))
    assert snapshot(tmp_path) == before, "dry_run modified the source tree"


def test_preprocess_generate_masks_dry_run_creates_no_output_folders(raw_plate, capsys):
    core.preprocess_generate_masks(mask_settings(raw_plate))
    for created in ("masks", "merged", "stack", "orig", "settings", "test", "consolidated"):
        assert not (raw_plate / created).exists(), f"dry_run created {created}/"


def test_preprocess_generate_masks_dry_run_loads_no_model(monkeypatch, raw_plate, capsys):
    """The first local import is `from .object import generate_cellpose_masks...`."""
    _stub_heavy_modules(monkeypatch, "spacr.object", "spacr.io", "spacr.plot", "spacr.utils")
    core.preprocess_generate_masks(mask_settings(raw_plate))  # must not raise


def test_preprocess_generate_masks_without_dry_run_reaches_the_model_imports(monkeypatch, raw_plate):
    _stub_heavy_modules(monkeypatch, "spacr.object", "spacr.io", "spacr.plot", "spacr.utils")
    with pytest.raises(AssertionError, match="compute was not skipped"):
        core.preprocess_generate_masks(mask_settings(raw_plate, dry_run=False))


def test_preprocess_generate_masks_dry_run_flags_the_out_of_range_channel(raw_plate, capsys):
    problems = core.preprocess_generate_masks(mask_settings(raw_plate, organelle_channel=3))
    offenders = [p for p in problems
                 if p.setting == "organelle_channel" and p.severity == "error"]
    assert offenders
    out = capsys.readouterr().out
    assert "organelle_channel" in out
    assert "fix:" in out


def test_preprocess_generate_masks_dry_run_prints_the_plan(raw_plate, capsys):
    core.preprocess_generate_masks(mask_settings(raw_plate))
    out = capsys.readouterr().out
    assert "spacr.core.preprocess_generate_masks" in out
    assert "12 raw image files" in out
    assert "cell: channel 0" in out
    assert os.path.join(str(raw_plate), "masks") in out


def test_preprocess_generate_masks_dry_run_reports_a_missing_src_instead_of_raising(tmp_path, capsys):
    """Without dry_run this raises ValueError; a validate-only run must report."""
    problems = core.preprocess_generate_masks({"dry_run": True})
    assert [p for p in problems if p.setting == "src" and p.severity == "error"]


def test_preprocess_generate_masks_still_raises_on_a_missing_src_when_not_dry(capsys):
    with pytest.raises(ValueError, match="src is a required parameter"):
        core.preprocess_generate_masks({"dry_run": False})


def test_dry_run_absent_from_settings_behaves_as_false(monkeypatch, raw_plate):
    """`settings.get('dry_run', False)` — an old settings dict must be unaffected."""
    _stub_heavy_modules(monkeypatch, "spacr.object", "spacr.io", "spacr.plot", "spacr.utils")
    settings = mask_settings(raw_plate)
    del settings["dry_run"]
    with pytest.raises(AssertionError, match="compute was not skipped"):
        core.preprocess_generate_masks(settings)


# ---------------------------------------------------------------------------
# run_preflight, the shared implementation
# ---------------------------------------------------------------------------

def test_run_preflight_sends_everything_to_the_given_printer(merged_plate):
    _plate, merged = merged_plate
    captured = []
    problems = run_preflight(measure_settings(merged), "measure", printer=captured.append)

    text = "\n".join(captured)
    assert "spaCR pre-flight check" in text
    assert "Plan" in text
    assert text.rstrip().endswith("Set dry_run=False to run for real.")
    assert isinstance(problems, list)


def test_run_preflight_writes_nothing(tmp_path, merged_plate):
    _plate, merged = merged_plate
    before = snapshot(tmp_path)
    run_preflight(measure_settings(merged), "measure", printer=lambda *_: None)
    assert snapshot(tmp_path) == before
