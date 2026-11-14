"""The genotyping decode speed-up of item 932, and what it must not change.

Item 932 made the sequencing (SBS) field step about twice as fast: the
registration transforms run in float32 through ``scipy.fft`` (or batched on
a CUDA card) with each reference's transform computed once, no stack of
moved planes is copied, the per-field diagnostics are off unless asked for,
and the library matcher looks a one-mismatch read up in an index instead of
scanning the library.

The index is exact by construction, and so is removing the copies. float32
against float64 and the GPU against the CPU are NOT exact by construction:
on 443 fields of the Funk et al. 2022 plates they gave the same integer
shifts and the same calls, and these tests hold that on a small fixture so
a change that breaks it is seen.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import ndimage

from spacr import ops_cycles, ops_engine, ops_settings
from spacr.ops_cycles import (FFT_BACKEND, FFT_PRECISION, _FFTEngine,
                              _fft_backend_used, _translate, align_field,
                              sbs_mean_target)
from spacr.ops_register import phase_correlate
from spacr.ops_sbs import (LibraryMatcher, called_bases,
                           estimate_read_locations, extract_bases, find_peaks)

SIZE = 192
#: Each cycle's stage drift and each channel's offset within a cycle.
CYCLE_DRIFT = {1: (0, 0), 2: (5, -3), 3: (-7, 4), 4: (2, 9)}
CHANNEL_OFFSET = [(0, 0), (2, -1), (-3, 2), (1, 4)]


def _field(seed=0, cycles=(1, 2, 3, 4)):
    """A sequencing field: a shared cell background and reads, one base per
    cycle, every cycle drifted and every channel offset within its cycle."""
    rng = np.random.default_rng(seed)
    background = ndimage.gaussian_filter(rng.random((SIZE + 40, SIZE + 40)), 6)
    background = (background - background.min()) / np.ptp(background) * 2500 + 400
    reads = rng.integers(30, SIZE + 10, size=(260, 2))
    bases = rng.integers(0, 4, size=(260, len(cycles)))
    planes = {}
    for index, cycle in enumerate(cycles):
        channels = []
        for k in range(4):
            dy = CYCLE_DRIFT[cycle][0] + CHANNEL_OFFSET[k][0]
            dx = CYCLE_DRIFT[cycle][1] + CHANNEL_OFFSET[k][1]
            spots = np.zeros_like(background)
            for (y, x), base in zip(reads, bases[:, index]):
                if base == k:
                    spots[y, x] = 1.0
            image = background + ndimage.gaussian_filter(spots, 1.0) * 30000
            image = np.roll(image, (dy, dx), axis=(0, 1))[20:20 + SIZE, 20:20 + SIZE]
            channels.append((image + rng.normal(0, 4, image.shape)).astype(np.uint16))
        planes[cycle] = channels
    return planes


def _calls(field):
    """The field's reads and their base calls, as the decode makes them."""
    filtered = np.empty_like(field.stack)
    for c in range(field.stack.shape[0]):
        for k in range(field.stack.shape[1]):
            filtered[c, k] = np.clip(-ndimage.gaussian_laplace(
                field.stack[c, k], 1.0), 0, None)
    score = estimate_read_locations(filtered)
    peaks = find_peaks(score, min_distance=2, gpu=False)
    values = extract_bases(filtered, peaks, window=1)
    _data, codes, _margin = called_bases(values, gpu=False)
    return peaks.tolist(), list(codes)


def _same(first, second):
    """Assert two aligned fields agree on every shift, the stack and the calls."""
    assert first.kept == second.kept and first.refused == second.refused
    assert first.cycle_shifts == second.cycle_shifts
    assert first.channel_shifts == second.channel_shifts
    assert first.refused_channels == second.refused_channels
    assert first.spot_replaced == second.spot_replaced
    assert np.array_equal(first.stack, second.stack)
    assert _calls(first) == _calls(second)


@pytest.mark.parametrize("mode", ops_cycles.CHANNEL_MODES)
def test_float32_and_float64_give_the_same_shifts_and_calls(mode):
    """Identical here, not by construction: the guard on the float32 default."""
    planes = _field()
    single = align_field(planes, reference=1, gpu=False, channel_mode=mode,
                         precision="float32", backend="cpu")
    double = align_field(planes, reference=1, gpu=False, channel_mode=mode,
                         precision="float64", backend="cpu")
    _same(single, double)
    assert single.kept == (1, 2, 3, 4)
    assert single.cycle_shifts[3] == (7, -4)


def _cuda():
    return _fft_backend_used("gpu", True, "float32") == "gpu"


@pytest.mark.skipif(not _cuda(), reason="no CUDA card usable through torch")
@pytest.mark.parametrize("precision", ops_cycles.FFT_PRECISIONS)
def test_the_gpu_gives_the_cpu_shifts_and_calls(precision):
    planes = _field(seed=3)
    for mode in ops_cycles.CHANNEL_MODES:
        on_card = align_field(planes, reference=1, gpu=True, channel_mode=mode,
                              precision=precision, backend="gpu")
        on_cpu = align_field(planes, reference=1, gpu=False, channel_mode=mode,
                             precision=precision, backend="cpu")
        _same(on_card, on_cpu)


def test_the_gpu_backend_falls_back_to_the_cpu_without_cuda():
    """CI runs with CUDA_VISIBLE_DEVICES=''; the GPU default must still run."""
    code = (
        "import numpy as np\n"
        "from spacr.ops_cycles import _FFTEngine, _fft_backend_used\n"
        "assert _fft_backend_used('gpu', True, 'float32') == 'cpu'\n"
        "assert _fft_backend_used('gpu', True, 'float64') == 'cpu'\n"
        "engine = _FFTEngine('float32', 'gpu')\n"
        "assert engine.used == 'cpu' and engine.torch is None\n"
        "image = np.random.default_rng(0).random((64, 64))\n"
        "found = engine.register(image, [np.roll(image, (3, -2), (0, 1))], tolerance=8)\n"
        "assert (found[0].dy, found[0].dx) == (-3, 2), found\n"
        "print('fallback ok')\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert "fallback ok" in done.stdout


def test_a_gpu_backend_with_no_card_aligns_as_the_cpu(monkeypatch):
    monkeypatch.setattr(ops_cycles, "_cuda_torch", lambda precision: None)
    planes = _field(seed=1)
    _same(align_field(planes, reference=1, channel_mode="spots_refine",
                      backend="gpu"),
          align_field(planes, reference=1, channel_mode="spots_refine",
                      backend="cpu"))


def test_a_card_that_fails_mid_field_hands_the_field_to_the_cpu(caplog):
    """The GPU path raising (out of memory, a driver fault) costs time only."""
    engine = _FFTEngine("float32", "cpu")
    engine.used, engine.torch = "gpu", object()
    image = np.random.default_rng(2).random((64, 64)).astype(np.float32)
    found = engine.register(image, [np.roll(image, (4, 1), (0, 1))], tolerance=8)
    assert (found[0].dy, found[0].dx) == (-4, -1)
    assert engine.used == "cpu"
    assert "continues on the CPU" in caplog.text


def test_float64_is_the_arithmetic_before_item_932_bit_for_bit():
    """The old background surface came from phase_correlate on numpy, the
    old spot surface from two float64 numpy transforms: float64 keeps both."""
    rng = np.random.default_rng(4)
    first = rng.random((96, 96)).astype(np.float32)
    others = [np.roll(first, (s, -s), (0, 1)) + rng.random((96, 96)) * 0.1
              for s in (1, 3, 6)]
    engine = _FFTEngine("float64", "cpu")
    for found, other in zip(engine.register(first, others, tolerance=16), others):
        old = phase_correlate(first, other, expected=(0, 0), tolerance=16,
                              backend="numpy")
        assert (found.dy, found.dx, found.accepted) == (old.dy, old.dx, old.accepted)
        assert found.peak_ratio == old.peak_ratio
    for (dy, dx, peak, mean, _near), other in zip(
            engine.spot_peaks(first, others, [(0, 0)] * 3), others):
        cross = np.fft.rfft2(first.astype(np.float64)) * np.conj(
            np.fft.rfft2(other.astype(np.float64)))
        cross /= np.abs(cross) + 1e-12
        surface = np.fft.irfft2(cross, s=first.shape)
        assert peak == float(surface.max())
        assert mean == float(np.abs(surface).mean())


def _old_background_align(planes, reference):
    """``align_field(channel_mode='background')`` as written before item 932:
    every channel moved into a new array, the cycle's moved stack stacked
    and moved again."""
    moved, channel_shifts = {}, {}
    for cycle, offered in planes.items():
        first = np.asarray(offered[0], np.float32)
        base = np.clip(first, *np.percentile(first, [0.0, 90.0]))
        out, shifts = [first], [(0, 0)]
        for plane in offered[1:]:
            plane = np.asarray(plane, np.float32)
            found = phase_correlate(base, np.clip(plane, *np.percentile(plane, [0.0, 90.0])),
                                    expected=(0, 0), tolerance=16, backend="numpy")
            shift = (found.dy, found.dx) if found.accepted else (0, 0)
            out.append(_translate(plane, *shift) if found.accepted else plane)
            shifts.append(shift)
        moved[cycle], channel_shifts[cycle] = out, tuple(shifts)
    target = sbs_mean_target(moved[reference])
    layers = []
    for cycle in sorted(planes):
        stack = np.stack(moved[cycle])
        if cycle != reference:
            found = phase_correlate(target, sbs_mean_target(moved[cycle]),
                                    expected=(0, 0), tolerance=64, backend="numpy")
            stack = _translate(stack, found.dy, found.dx)
        layers.append(stack)
    return np.stack(layers), channel_shifts


def test_no_copy_alignment_gives_the_old_stack_in_background_mode():
    """The channel move and the cycle move stay two moves: a row the first
    pushes out is still empty after the second brings it back."""
    planes = _field(seed=5)
    stack, channel_shifts = _old_background_align(planes, 1)
    field = align_field(planes, reference=1, gpu=False, channel_mode="background",
                        precision="float64")
    assert field.channel_shifts == channel_shifts
    assert np.array_equal(field.stack, stack)


def test_the_registration_log_lists_every_background_registration():
    planes = _field(seed=6)
    record = []
    field = align_field(planes, reference=1, gpu=False, channel_mode="spots",
                        registrations=record)
    assert len(record) == 4 * 3 + 3
    assert all(len(entry) == 4 for entry in record)
    assert [tuple(entry[:2]) for entry in record[-3:]] == [
        field.cycle_shifts[c] for c in (2, 3, 4)]


def test_an_unknown_precision_or_backend_is_refused():
    planes = _field()
    with pytest.raises(ValueError, match="precision"):
        align_field(planes, precision="float16")
    with pytest.raises(ValueError, match="backend"):
        align_field(planes, backend="tpu")
    with pytest.raises(ValueError, match="ops_registration_backend"):
        ops_engine._registration_choice("tpu", "ops_registration_backend")
    assert ops_engine._registration_choice(None, "ops_registration_precision") \
        == FFT_PRECISION


def _scan(entries, read, max_distance):
    """The library scan the index replaces, written out."""
    if read in entries:
        return read, 0
    best, found = None, []
    for entry in entries:
        if len(entry) != len(read):
            continue
        distance = sum(a != b and b != "N" for a, b in zip(entry, read))
        if best is None or distance < best:
            best, found = distance, [entry]
        elif distance == best:
            found.append(entry)
    if best is None or best > max_distance or len(found) > 1 \
            or (max_distance == 0 and "N" not in read):
        return None, -1
    return found[0], best


def test_the_one_mismatch_index_answers_exactly_as_the_scan():
    rng = np.random.default_rng(7)
    letters = np.array(list("ACGT"))
    entries = sorted({"".join(rng.choice(letters, 6)) for _ in range(300)}
                     - {"AAAAAG"})
    entries += ["AAAAAA", "AAAAAC"]           # AAAAAG is one from both: a tie
    matcher = LibraryMatcher(entries)
    reads = ["AAAAAG", "AAAAAA", "AANAAA", "AAXAAA", "AAAAA"]
    for entry in entries[:150]:
        for _ in range(3):
            read = list(entry)
            for position in rng.choice(6, rng.integers(1, 3), replace=False):
                read[position] = rng.choice(list("ACGTN"))
            reads.append("".join(read))
    for read in reads:
        for max_distance in (0, 1, 2):
            assert matcher.match(read, max_distance) == _scan(
                set(entries), read, max_distance), (read, max_distance)
    assert matcher.match("AAAAAG", 1) == (None, -1)


def test_the_settings_defaults_are_the_engine_defaults():
    defaults = ops_settings.OPS_DEFAULTS
    assert defaults["ops_registration_precision"] == FFT_PRECISION == "float32"
    assert defaults["ops_registration_backend"] == FFT_BACKEND
    assert defaults["ops_field_diagnostics"] is False
    for key in ("ops_registration_precision", "ops_registration_backend",
                "ops_field_diagnostics"):
        assert key in ops_settings.OPS_TOOLTIPS
        assert any(key in keys for keys in ops_settings.OPS_CATEGORIES.values())


@pytest.fixture(scope="module")
def decoded(tmp_path_factory):
    """The synthetic well of the end-to-end test, decoded under rule (a) + S2
    in float64 with the diagnostics on, and again with the defaults."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_932")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1,
                "ops_decode_preset": "rule_a_s2"}
    ops_engine.run_ops(settings, library=library)
    out = {}
    for name, extra in (("before", {"ops_registration_precision": "float64",
                                    "ops_registration_backend": "cpu",
                                    "ops_field_diagnostics": True}),
                        ("default", {})):
        result = ops_engine.run_ops({**settings, **extra}, phases=("decode",),
                                    library=library)
        with sqlite3.connect(result["db"]) as conn:
            rows = conn.execute("SELECT object_id, barcode, mapped_guide, "
                                "n_reads FROM ops_barcodes ORDER BY object_id"
                                ).fetchall()
        folder = os.path.join(os.path.dirname(result["db"]), "A1",
                              "decode_diagnostics")
        files = (sorted(name for name in os.listdir(folder)
                        if name.startswith("site")) if os.path.isdir(folder)
                 else [])
        out[name] = (result["wells"]["A1"]["decode"], rows, folder, files)
        if files:
            shutil.copytree(folder, root / f"diagnostics_{name}")
            shutil.rmtree(folder)
    yield out, root
    monkeypatch.undo()


def test_the_old_and_new_defaults_give_the_same_calls(decoded):
    out, _root = decoded
    before, after = out["before"], out["default"]
    assert before[1] and before[1] == after[1]
    assert before[0]["registration"]["precision"] == "float64"
    assert after[0]["registration"] == {
        "precision": FFT_PRECISION, "backend": FFT_BACKEND,
        "fields_by_backend": {"cpu": after[0]["fields_decoded"]},
        "field_diagnostics": None, "field_diagnostics_table": None}


def test_the_field_diagnostics_are_written_only_when_asked(decoded):
    out, root = decoded
    assert out["default"][3] == []
    files = out["before"][3]
    assert files and all(name.endswith(".json") for name in files)
    assert (root / "diagnostics_before" / "field_diagnostics.csv").is_file()
    with open(root / "diagnostics_before" / files[0], encoding="utf-8") as handle:
        record = json.load(handle)
    assert {"diag", "log_p999", "registrations", "kept", "cycle_shifts",
            "channel_shifts", "registration"} <= set(record)
    cycle = str(record["kept"][0])
    assert {"p50", "p999", "focus_lapvar"} <= set(record["diag"][cycle])
    assert len(record["log_p999"][cycle]) == 4
    assert record["registration"]["used"] == "cpu"
    assert record["registrations"]


def test_the_diagnostics_table_reads_what_the_decode_wrote(decoded):
    out, root = decoded
    table = ops_engine._field_diagnostics_table(str(root / "diagnostics_before"))
    assert len(table) == len(out["before"][3])
    assert out["before"][0]["registration"]["field_diagnostics_table"].endswith(
        "field_diagnostics.csv")
    assert np.isfinite(table["chan_peak_min"]).all()
    assert {"dapi_focus", "max_chan_shift", "cycle_peak_median"} <= set(table)


def test_a_reader_of_the_diagnostics_copes_with_their_absence(tmp_path):
    """Off by default since item 932: no folder, an empty folder, a field
    JSON without diagnostics and an unreadable file must not break it."""
    assert ops_engine._field_diagnostics_table(None) is None
    assert ops_engine._field_diagnostics_table(str(tmp_path / "missing")) is None
    assert ops_engine._field_diagnostics_table(str(tmp_path)) is None
    (tmp_path / "site0001.json").write_text(json.dumps(
        {"site": 1, "kept": [1, 2], "refused": [], "registrations": [],
         "diag": {}}), encoding="utf-8")
    (tmp_path / "site0002.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "site0003.json").write_text(json.dumps(
        {"site": 3, "kept": [1, 2], "refused": [],
         "registrations": [[0, 0, 50.0, True]] * 6 + [[2, 1, 200.0, True]],
         "channel_shifts": {"1": [[0, 0], [1, -2]]},
         "diag": {"1": {"p50": [1.0], "p999": [10.0, 30.0], "focus_lapvar": 0.5}}}),
        encoding="utf-8")
    table = ops_engine._field_diagnostics_table(str(tmp_path))
    assert list(table["site"]) == [1, 3]
    assert np.isnan(table.loc[0, "chan_peak_min"])
    assert np.isnan(table.loc[0, "dapi_focus"])
    assert table.loc[1, "cycle_peak_min"] == 200.0
    assert table.loc[1, "max_chan_shift"] == 2
    assert table.loc[1, "p999_c1"] == 20.0


def test_a_settings_file_from_before_932_reruns_with_float64_on_the_cpu(tmp_path, caplog):
    """The reproducibility rule: a saved file reruns as it ran. Files without
    a stamp, and files stamped 931, get the defaults from before item 932;
    a 932 stamp gets today's, and a key the file sets always wins."""
    import csv
    import logging

    from spacr.cli import resolve_module, resolve_settings
    from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                SETTINGS_VERSION_KEY)

    assert DEFAULT_CHANGES[932] == {"ops_registration_precision": "float64",
                                    "ops_registration_backend": "cpu"}
    assert SETTINGS_SCHEMA_VERSION >= 932
    assert "ops_field_diagnostics" not in DEFAULT_CHANGES[932]

    def written(name, rows):
        path = tmp_path / name
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Key", "Value"])
            writer.writerows(rows)
        return str(path)

    ops = resolve_module("ops")
    with caplog.at_level(logging.WARNING, logger="spacr.settings"):
        old = resolve_settings(ops, written("old.csv", [("plate", "P1")]))
    assert old["ops_registration_precision"] == "float64"
    assert old["ops_registration_backend"] == "cpu"
    assert old["ops_field_diagnostics"] is False
    assert "ops_registration_precision='float64'" in caplog.text
    from_931 = resolve_settings(ops, written("931.csv", [
        (SETTINGS_VERSION_KEY, "931"), ("plate", "P1")]))
    assert (from_931["ops_registration_precision"],
            from_931["ops_registration_backend"]) == ("float64", "cpu")
    assert from_931["ops_compose_blend"] == ops_settings.OPS_DEFAULTS["ops_compose_blend"]
    new = resolve_settings(ops, written("932.csv", [
        (SETTINGS_VERSION_KEY, "932"), ("plate", "P1")]))
    assert (new["ops_registration_precision"],
            new["ops_registration_backend"]) == (FFT_PRECISION, FFT_BACKEND)
    explicit = resolve_settings(ops, written("explicit.csv", [
        ("plate", "P1"), ("ops_registration_precision", "float32")]))
    assert explicit["ops_registration_precision"] == "float32"
    assert explicit["ops_registration_backend"] == "cpu"
