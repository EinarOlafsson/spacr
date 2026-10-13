"""A Mask run whose regex conversion fails must not number the wells itself.

``metadata_type='auto'`` with a ``custom_regex`` converts the folder with
:func:`spacr.io.convert_separate_files_to_yokogawa`, which keeps the wells
the file names carry. Until 2026-09-19 any exception from it was caught
without a word and the folder was converted again by
:func:`spacr.io.convert_to_yokogawa`, which reads no regex and hands every
file the next free well in file order. A plate whose files said B03 and C07
came out as A01, A02, ... with one channel file per well, and only
``rename_log.csv`` said so. Found while fixing GitHub #117 / #121 (item 429),
where a macOS sidecar was what made the regex conversion fail; any other
unreadable file does the same.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

REGEX = (r"(?P<plateID>.*)_(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)"
         r"_w(?P<chanID>\d)\.tif")
WELLS_IN_THE_NAMES = {"B03", "C07", "D05"}


def _plate(root: Path, wells=("B03", "C07"), broken="exp1_D05_s1_w1.tif"):
    """Four channel tiffs per well named for the regex, and one that is not a TIFF."""
    import tifffile

    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for well in wells:
        for channel in (1, 2, 3, 4):
            image = rng.integers(100, 3000, size=(32, 32)).astype(np.uint16)
            tifffile.imwrite(root / f"exp1_{well}_s1_w{channel}.tif", image)
    if broken:
        (root / broken).write_bytes(b"not a tiff at all" * 64)
    return root


def _converted_wells(folder: Path):
    """Wells of every Yokogawa-named file under ``folder``, ``orig/`` included."""
    return {p.name.split("_")[1] for p in folder.rglob("plate*_*.tif")}


def _settings(src, **over):
    settings = {
        "src": str(src), "metadata_type": "auto", "custom_regex": REGEX,
        "channels": [0, 1, 2, 3], "cell_channel": 3, "nucleus_channel": 0,
        "pathogen_channel": None, "organelle_channel": None,
        "preprocess": True, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "batch_size": 10,
        "save": True, "randomize": False, "pipeline_style": "v1",
    }
    settings.update(over)
    return settings


def test_a_mask_run_whose_regex_conversion_fails_relabels_no_well(
        tmp_path, capsys, monkeypatch):
    """The user's path: the regex conversion stops on an unreadable file.

    Before the fix the run fell back to the regex-less conversion, wrote
    files for wells no file name carried (plate1_A01, plate1_A02, ...), and
    then died stacking them: "ValueError: all input arrays must have the
    same shape", one channel file per well.
    """
    from spacr.core import preprocess_generate_masks

    monkeypatch.delenv("SPACR_STRICT_ERRORS", raising=False)
    plate = _plate(tmp_path / "plate")

    try:
        result = preprocess_generate_masks(_settings(plate))
    except Exception as exc:                                 # noqa: BLE001
        result = exc

    out = capsys.readouterr().out
    wells = _converted_wells(plate)
    assert wells <= WELLS_IN_THE_NAMES, (
        f"wells no file name carries: {sorted(wells - WELLS_IN_THE_NAMES)}")
    assert result is None
    assert "exp1_D05_s1_w1.tif" in out
    assert "did not fall back" in out
    assert "RUN INCOMPLETE" in out
    assert not (plate / "stack").exists(), "the run went on past the refusal"


def test_strict_errors_turn_the_refusal_into_a_configuration_error(
        tmp_path, monkeypatch):
    """Under ``SPACR_STRICT_ERRORS`` the refusal is a hard stop naming the file.

    Before the fix the fallback conversion finished, so with preprocessing
    off nothing was raised.
    """
    from spacr.core import preprocess_generate_masks
    from spacr.errors import ConfigurationError

    plate = _plate(tmp_path / "plate")
    monkeypatch.setenv("SPACR_STRICT_ERRORS", "1")

    with pytest.raises(ConfigurationError) as caught:
        preprocess_generate_masks(_settings(plate, preprocess=False,
                                            masks=False))

    message = str(caught.value)
    assert "exp1_D05_s1_w1.tif" in message
    assert "did not fall back" in message
    assert _converted_wells(plate) <= WELLS_IN_THE_NAMES


def test_the_plain_converter_is_never_called_after_a_regex_failure(
        tmp_path, monkeypatch, capsys):
    """Whatever the regex conversion raised, the regex-less one does not run."""
    import spacr.core as core
    import spacr.io as sio

    def _regex_fails(folder, regex):
        raise RuntimeError("regex conversion failed")

    def _plain(folder):
        raise AssertionError("convert_to_yokogawa ran after a regex failure")

    monkeypatch.setattr(sio, "convert_separate_files_to_yokogawa", _regex_fails)
    monkeypatch.setattr(sio, "convert_to_yokogawa", _plain)
    monkeypatch.delenv("SPACR_STRICT_ERRORS", raising=False)
    src = tmp_path / "plate"
    src.mkdir()

    assert core.preprocess_generate_masks(_settings(src)) is None
    out = capsys.readouterr().out
    assert "RuntimeError: regex conversion failed" in out
    assert "did not fall back" in out


def test_the_converter_names_the_file_it_stopped_on(tmp_path):
    """The error says which file, what it was writing, and what it left.

    A bare ``tifffile`` error does not name the file it failed to read.
    """
    from spacr.io import convert_separate_files_to_yokogawa

    plate = _plate(tmp_path / "plate")

    with pytest.raises(ValueError) as caught:
        convert_separate_files_to_yokogawa(str(plate), REGEX)

    message = str(caught.value)
    assert "exp1_D05_s1_w1.tif" in message
    assert "plate1_D05_T0001F001L01C01.tif" in message
    assert "8 of 9 converted file(s) had been written" in message
    assert "rename_log.csv was not written" in message
    assert not (plate / "rename_log.csv").exists()
    assert _converted_wells(plate) == {"B03", "C07"}


def test_a_field_that_is_not_a_number_stops_before_anything_is_written(
        tmp_path):
    """A non-numeric ``fieldID`` used to fail only when its region was reached,
    after the regions before it had been converted."""
    import tifffile
    from spacr.io import convert_separate_files_to_yokogawa

    plate = tmp_path / "plate"
    plate.mkdir()
    for name in ("exp1_B03_1_w1.tif", "exp1_C07_x_w1.tif"):
        tifffile.imwrite(plate / name, np.ones((8, 8), np.uint16))
    regex = (r"(?P<plateID>.*)_(?P<wellID>[A-P]\d{2})_(?P<fieldID>[^_]+)"
             r"_w(?P<chanID>\d)\.tif")

    with pytest.raises(ValueError, match=re.escape("exp1_C07_x_w1.tif")):
        convert_separate_files_to_yokogawa(str(plate), regex)

    assert not list(plate.glob("plate*_*.tif"))


def test_a_regex_conversion_that_succeeds_is_unchanged(tmp_path, capsys):
    """Without a broken file the same plate converts to its own wells."""
    from spacr.io import convert_separate_files_to_yokogawa

    plate = _plate(tmp_path / "plate", broken=None)
    convert_separate_files_to_yokogawa(str(plate), REGEX)

    assert _converted_wells(plate) == {"B03", "C07"}
    assert (plate / "rename_log.csv").is_file()


def test_the_refusal_does_not_send_the_user_back_over_its_own_leftovers(
        tmp_path, capsys, monkeypatch):
    """Clearing the regex is only safe once the part-converted files are gone.

    The converter writes each region as it goes, so the refusal leaves the
    plate holding the ``plate*_*.tif`` files written before it stopped, with
    no ``rename_log.csv``. The plain converter now refuses that folder;
    the guidance must still explain how to retry from the original inputs.
    """
    from spacr.core import preprocess_generate_masks

    monkeypatch.delenv("SPACR_STRICT_ERRORS", raising=False)
    plate = _plate(tmp_path / "plate")

    preprocess_generate_masks(_settings(plate))
    out = capsys.readouterr().out

    assert list(plate.glob("plate*_*.tif")), (
        "premise: the refusal leaves the plate part-converted")
    assert not (plate / "rename_log.csv").exists()
    assert "moved out of" in out, out
    assert "refuses folders with converted images" in out, out
