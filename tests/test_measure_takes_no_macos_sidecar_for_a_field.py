"""Measure, its resume and the streamed dataset take no ``._`` sidecar for a field.

Item 429 (GitHub #117 / #121) found that macOS writes an AppleDouble
sidecar, ``._<name>``, beside every file on an exFAT, FAT or SMB volume, with
the same ``.npy`` ending, and fixed the Mask path's listings. Measure's were
left: on a ``merged/`` from that run, ``measure_crop`` measured every field
and still reported "spaCR RUN INCOMPLETE - measure_crop, failed: 1 (50.0%)"
for the sidecar. The same folder is listed by the Measure resume
(``spacr.resume``), by Measure's test mode (``spacr.utils.measure_test_mode``)
and by the streamed dataset (``spacr.stream_dataset``).
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_measure_crop_core_synth import (  # noqa: E402
    _build_merged_stack, _settings_for, _write_stack,
)


def _appledouble_bytes() -> bytes:
    """The 4096-byte AppleDouble header macOS writes for a file's metadata."""
    header = struct.pack(">II16sH", 0x00051607, 0x00020000,
                         b"Mac OS X        ", 2)
    header += struct.pack(">III", 9, 50, 3760)
    header += struct.pack(">III", 2, 3810, 286)
    return header + b"\x00" * (4096 - len(header))


APPLEDOUBLE = _appledouble_bytes()
FIELD = "plate1_A01_F001.npy"


def _with_sidecar(folder: Path, name: str = FIELD) -> Path:
    side = Path(folder) / f"._{name}"
    side.write_bytes(APPLEDOUBLE)
    return side


def test_measure_crop_on_a_macos_volume_reports_a_complete_run(
        tmp_path, synth_masks_multi, rng, capsys, caplog):
    """One measured field and its sidecar is a complete run, not 1 of 2."""
    from spacr.measure import measure_crop

    data = _build_merged_stack(synth_masks_multi, rng)
    merged, _ = _write_stack(tmp_path, data)
    _with_sidecar(merged)

    caplog.set_level(logging.WARNING)
    measure_crop(dict(_settings_for(merged, n_jobs=1)))

    printed = capsys.readouterr().out + caplog.text
    assert "RUN INCOMPLETE" not in printed
    assert "._plate1_A01_F001" not in printed
    assert (tmp_path / "measurements" / "measurements.db").is_file()


def test_the_measure_resume_plans_no_sidecar(tmp_path, synth_masks_multi, rng):
    """``plan_measure_resume`` used to plan ``._plate1_A01_F001`` as a field.

    It failed validation as unreadable, so every resume re-queued it and
    reported it rejected.
    """
    from spacr.resume import plan_measure_resume

    data = _build_merged_stack(synth_masks_multi, rng)
    merged, _ = _write_stack(tmp_path, data)
    _with_sidecar(merged)

    state = plan_measure_resume(
        dict(_settings_for(merged, resume=True)), verbose=False)

    planned = set(state.done) | set(state.pending) | set(state.reasons)
    assert planned == {"plate1_A01_F001"}
    assert state.total == 1


def test_completed_fields_in_merged_scans_no_sidecar(tmp_path):
    from spacr.resume import completed_fields_in_merged

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / FIELD, np.zeros((8, 8, 5), np.uint16))
    _with_sidecar(merged)

    rejected = {}
    done = completed_fields_in_merged(str(merged), reasons=rejected)

    assert done == {"plate1_A01_F001"}
    assert rejected == {}


def test_measure_test_mode_samples_only_fields(tmp_path, capsys):
    """Test mode sampled every file in ``merged/``, sidecars and the plane
    layout sidecar included, so a test set could hold fewer fields than asked."""
    from spacr.utils import measure_test_mode

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / FIELD, np.zeros((8, 8, 5), np.uint16))
    _with_sidecar(merged)
    (merged / ".spacr_plane_layout.json").write_text("{}")

    settings = measure_test_mode({"src": str(merged), "test_mode": True,
                                  "test_nr": 3})

    copied = sorted(p.name for p in Path(settings["src"]).iterdir())
    assert copied == [FIELD]
    assert "measuring all 1" in capsys.readouterr().out


def test_a_cellpose_training_set_counts_no_mask_sidecar_as_unreadable(
        tmp_path):
    """``generate_cellpose_train_set`` read ``masks/._x.tif`` with cv2, got
    None, and recorded a failure for a file that is not a mask."""
    import tifffile
    from spacr.measure import generate_cellpose_train_set

    folder = tmp_path / "exp1"
    (folder / "masks").mkdir(parents=True)
    mask = np.zeros((32, 32), np.uint16)
    for label in range(1, 7):
        mask[label * 4:label * 4 + 3, 2:6] = label
    tifffile.imwrite(folder / "masks" / "field1.tif", mask)
    tifffile.imwrite(folder / "field1.tif", np.ones((32, 32), np.uint16))
    _with_sidecar(folder / "masks", "field1.tif")

    ledger = generate_cellpose_train_set([str(folder)], str(tmp_path / "dst"))

    assert ledger.n_failed == 0
    assert sorted(p.name for p in (tmp_path / "dst" / "masks").iterdir()) == [
        "exp1_field1.tif"]


def test_a_streamed_selection_from_sidecars_alone_says_there_is_no_stack(
        tmp_path):
    """Sidecars used to be counted as stacks: "1 .npy stack(s) ... hold no
    object labels", about a folder that holds no stack at all."""
    from spacr.stream_dataset import selection_from_arrays

    merged = tmp_path / "merged"
    merged.mkdir()
    _with_sidecar(merged)

    with pytest.raises(FileNotFoundError, match="holds no .npy stack"):
        selection_from_arrays(str(merged))


def test_a_streamed_selection_counts_no_sidecar(tmp_path):
    from spacr.stream_dataset import selection_from_arrays

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / FIELD, np.zeros((8, 8, 5), np.uint16))
    _with_sidecar(merged)

    with pytest.raises(FileNotFoundError, match=r"^1 \.npy stack\(s\)"):
        selection_from_arrays(str(merged))


def test_the_stream_never_resolves_a_field_to_a_sidecar(tmp_path):
    """A sidecar sorts before every field. Only a stem that prefixes every
    name (the empty stem of a row with no identifiers) reached one."""
    from spacr.stream_dataset import _stack_for

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / FIELD, np.zeros((8, 8, 5), np.uint16))
    _with_sidecar(merged)

    for stem in ("", "plate1", "plate1_A01_F001"):
        found = _stack_for(str(merged), stem)
        assert found is not None and not Path(found).name.startswith("."), stem
