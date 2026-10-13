"""Tests for :mod:`spacr.diameter`, the Cellpose diameter auto-estimator.

The core test is the first one: synthetic discs of a *known* diameter must be
recovered to within a stated tolerance. Everything else guards the ways the
answer can be wrong in a way the user would not notice — a confluent field
that silently collapses to a small number, a background-only channel that
gets thresholded anyway, an off-by-one channel index that raises IndexError
instead of saying what is wrong.

Measured accuracy on the synthetic fixtures below (median over 5 fields,
disc radii 6/10/15/25 px, Gaussian read noise, illumination gradient): the
error never exceeds 3%. The assertions allow 15% so that a future change to
the thresholding is caught without the suite becoming a hash of the current
implementation.

Everything here is CPU-only, offline and deterministic (fixed seeds).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from spacr.diameter import (
    OBJECT_TYPES,
    SETTING_KEYS,
    DiameterEstimate,
    _analyse_plane,
    _crop_to,
    _illumination,
    _load_array_plane,
    _load_raw_plane,
    _measure,
    _region_diameters,
    _roots,
    _sample_indices,
    _to_2d,
    channels_from_settings,
    estimate_diameters,
    format_estimates,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------

def _disc_plane(
    shape=(256, 256),
    radius=12,
    step=None,
    amp=3000.0,
    background=200.0,
    noise=25.0,
    jitter=0,
    seed=0,
):
    """A field of discs of exactly known radius on a noisy background.

    :param step: centre-to-centre spacing. ``2 * radius`` makes the discs
        exactly tangent (the confluent case); the default of ``4 * radius``
        leaves them well separated.
    """
    rng = np.random.default_rng(seed)
    step = int(4 * radius) if step is None else int(step)
    img = np.full(shape, background, np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    margin = radius + 2
    for cy in range(margin, shape[0] - margin, step):
        for cx in range(margin, shape[1] - margin, step):
            oy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            ox = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            img[(yy - cy - oy) ** 2 + (xx - cx - ox) ** 2 <= radius ** 2] += amp
    if noise:
        img += rng.normal(0.0, noise, shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _flat_plane(shape=(256, 256), background=200.0, noise=25.0, seed=0):
    """A field with nothing in it but background and read noise."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(background, noise, shape), 0, 65535).astype(np.uint16)


def _write_stack(root, planes_per_field, wells=None, plate="plate1"):
    """Write ``stack/<plate>_<well>_1_t0.npy`` arrays of shape (H, W, C).

    Mirrors the naming ``spacr.io._rename_and_organize_image_files`` produces,
    so the sorted file order really is plate/well order.
    """
    stack = Path(root) / "stack"
    stack.mkdir(parents=True, exist_ok=True)
    wells = wells or [f"A{i + 1:02d}" for i in range(len(planes_per_field))]
    for well, planes in zip(wells, planes_per_field):
        arr = np.stack(planes, axis=-1)
        np.save(stack / f"{plate}_{well}_1_t0.npy", arr)
    return str(root)


@pytest.fixture(scope="module")
def clean_source(tmp_path_factory):
    """5 fields, cells (radius 15) in channel 0, nuclei (radius 6) in channel 1."""
    root = tmp_path_factory.mktemp("clean_src")
    fields = []
    for i in range(5):
        cells = _disc_plane(radius=15, seed=100 + i, jitter=4)
        nuclei = _disc_plane(radius=6, step=60, seed=200 + i, jitter=4)
        empty = _flat_plane(seed=300 + i)
        fields.append([cells, nuclei, empty])
    return _write_stack(root, fields)


# ---------------------------------------------------------------------------
# THE core test: known diameter in, known diameter out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [6, 10, 15, 25])
def test_discs_of_known_diameter_are_recovered_within_15_percent(tmp_path, radius):
    """The one thing this module exists to do.

    Measured error on these fixtures is under 3%; the assertion allows 15%.
    """
    fields = [[_disc_plane(shape=(384, 384), radius=radius, seed=i, jitter=radius // 3)]
              for i in range(5)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=5)["cell"]

    truth = 2.0 * radius
    assert est.usable, est.note
    assert est.diameter == pytest.approx(truth, rel=0.15), (
        f"radius {radius}: proposed {est.diameter:.2f} px for a true diameter of {truth}"
    )
    assert est.low <= est.diameter <= est.high
    assert est.n_fields == 5
    assert est.n_objects > 20
    assert est.method == "threshold_otsu"
    assert est.confidence == "high", est.note


def test_recovery_survives_an_illumination_gradient(tmp_path):
    """A vignetted plate must not shift the estimate; that is what flattening is for."""
    yy, xx = np.mgrid[0:384, 0:384]
    fields = []
    for i in range(4):
        plane = _disc_plane(shape=(384, 384), radius=15, seed=i).astype(np.float32)
        plane = plane + 2500.0 * (yy / 384.0) + 1500.0 * (xx / 384.0)
        fields.append([plane.astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=4)["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(30.0, rel=0.15), est.diameter


# ---------------------------------------------------------------------------
# two object types, two channels
# ---------------------------------------------------------------------------

def test_two_object_types_in_different_channels_get_different_estimates(clean_source):
    """Cells at d=30 and nuclei at d=12 must come back apart, and correct."""
    est = estimate_diameters(clean_source, {"cell": 0, "nucleus": 1}, n_fields=5)

    assert set(est) == {"cell", "nucleus"}
    assert est["cell"].usable and est["nucleus"].usable
    assert est["cell"].diameter == pytest.approx(30.0, rel=0.15), est["cell"].diameter
    assert est["nucleus"].diameter == pytest.approx(12.0, rel=0.15), est["nucleus"].diameter
    assert est["cell"].diameter > 2 * est["nucleus"].diameter
    assert est["cell"].object_type == "cell"
    assert est["nucleus"].object_type == "nucleus"


def test_object_types_are_reported_in_a_stable_order(clean_source):
    """cell/nucleus/pathogen/organelle order, whatever order the caller asked in."""
    est = estimate_diameters(clean_source, {"nucleus": 1, "cell": 0}, n_fields=2)
    assert list(est) == ["cell", "nucleus"]


# ---------------------------------------------------------------------------
# the failure mode that matters: confluence
# ---------------------------------------------------------------------------

def test_confluent_objects_do_not_silently_collapse(tmp_path):
    """Tangent discs fuse under a plain threshold; the estimate must not collapse.

    Plain labelling welds the whole packing into one component, which is then
    dropped for being oversized/border-touching — leaving nothing, or leaving
    debris that would read far too small. The distance-transform watershed has
    to take over, the confidence has to drop, and the note has to say why.
    """
    radius = 14
    fields = [[_disc_plane(shape=(384, 384), radius=radius, step=2 * radius, seed=i)]
              for i in range(4)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=4)["cell"]

    assert est.usable, est.note
    assert est.diameter == pytest.approx(2.0 * radius, rel=0.15), (
        f"confluent estimate collapsed to {est.diameter:.2f} px (truth {2 * radius})"
    )
    assert est.method == "watershed_edt"
    assert est.confidence != "high"
    low = est.note.lower()
    assert "confluent" in low or "touching" in low, est.note
    assert "distance transform" in low, est.note
    assert "foreground covers" in low, est.note


def test_dense_but_separated_objects_are_flagged_without_switching_method(tmp_path):
    """High foreground alone downgrades confidence; it does not hijack the method."""
    radius = 12
    fields = [[_disc_plane(shape=(384, 384), radius=radius, step=int(2.3 * radius), seed=i)]
              for i in range(4)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=4, fused_fraction=0.10)["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(2.0 * radius, rel=0.15)
    assert est.confidence != "high"
    assert "foreground covers" in est.note.lower(), est.note


# ---------------------------------------------------------------------------
# nothing usable -> no fabricated number
# ---------------------------------------------------------------------------

def test_background_only_channel_yields_no_number(clean_source):
    """Otsu will happily bisect pure noise; the SNR gate must stop it first."""
    est = estimate_diameters(clean_source, {"pathogen": 2}, n_fields=5)["pathogen"]

    assert not est.usable
    assert math.isnan(est.diameter)
    assert math.isnan(est.low) and math.isnan(est.high)
    assert est.n_objects == 0
    assert est.method == "none"
    assert est.confidence == "low"
    assert "noise" in est.note.lower() or "background" in est.note.lower(), est.note


def test_all_zero_channel_yields_no_number(tmp_path):
    """A dead channel has no variation at all — a different rejection path."""
    fields = [[_disc_plane(seed=i), np.zeros((256, 256), np.uint16)] for i in range(3)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0, "nucleus": 1}, n_fields=3)
    assert est["cell"].usable
    assert not est["nucleus"].usable
    assert math.isnan(est["nucleus"].diameter)
    assert "flat" in est["nucleus"].note.lower() or "no usable signal" in est["nucleus"].note.lower()


def test_empty_source_folder_is_reported_not_guessed(tmp_path):
    """Nowhere to sample from is a message, not a number and not a traceback."""
    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=5)
    assert not est["cell"].usable
    assert est["cell"].n_fields == 0
    assert "stack" in est["cell"].note or "no fields" in est["cell"].note


def test_missing_src_folder_is_reported_not_raised(tmp_path):
    est = estimate_diameters(str(tmp_path / "does-not-exist"), {"cell": 0})
    assert not est["cell"].usable
    assert "no such folder" in est["cell"].note


def test_no_channels_requested_returns_an_empty_mapping(clean_source):
    assert estimate_diameters(clean_source, {}) == {}
    assert estimate_diameters(clean_source, {"cell": None, "nucleus": "not-an-int"}) == {}


# ---------------------------------------------------------------------------
# field sampling
# ---------------------------------------------------------------------------

def test_fewer_fields_available_than_requested(tmp_path):
    """Asking for 10 fields from a 3-field plate uses 3 and says so."""
    fields = [[_disc_plane(radius=12, seed=i)] for i in range(3)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=10)["cell"]
    assert est.usable
    assert est.n_fields == 3
    assert "3 of the 10 requested fields" in est.note, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_a_single_field_downgrades_confidence(tmp_path):
    src = _write_stack(tmp_path, [[_disc_plane(radius=12, seed=0)]])
    est = estimate_diameters(src, {"cell": 0}, n_fields=5)["cell"]
    assert est.n_fields == 1
    assert est.confidence != "high"
    assert "only one field contributed" in est.note


def test_sample_indices_spread_across_the_source_and_never_take_the_first_n():
    """Plates vary down rows and across columns; the first N wells are not a sample."""
    picks = _sample_indices(100, 5, None)
    assert picks[0] == 0 and picks[-1] == 99
    assert picks != [0, 1, 2, 3, 4]
    assert len(set(picks)) == 5
    # roughly even stride
    gaps = np.diff(picks)
    assert gaps.min() >= 20

    assert _sample_indices(3, 5, None) == [0, 1, 2]
    assert _sample_indices(0, 5, None) == []


def test_seeded_sampling_is_random_but_reproducible():
    a = _sample_indices(200, 6, random_state=7)
    b = _sample_indices(200, 6, random_state=7)
    c = _sample_indices(200, 6, random_state=8)
    assert a == b
    assert len(a) == 6 and a != c
    assert a != _sample_indices(200, 6, None)


def test_sampling_actually_reaches_the_far_end_of_the_plate(tmp_path):
    """End-to-end proof that the stride, not the first N files, drives the answer.

    Fields 0-2 hold small objects (d=12), fields 3-9 hold large ones (d=40).
    Taking the first three fields would answer ~12; an even stride over ten
    fields takes indices 0, 4 and 9, so the large objects dominate.
    """
    fields = []
    for i in range(10):
        radius = 6 if i < 3 else 20
        fields.append([_disc_plane(shape=(320, 320), radius=radius, step=80, seed=i)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=3)["cell"]
    assert est.n_fields == 3
    assert est.diameter == pytest.approx(40.0, rel=0.15), (
        f"got {est.diameter:.1f}; a first-N sampler would have answered about 12"
    )


def test_src_may_be_a_list_of_folders(tmp_path):
    """expected_types['src'] is (str, list); both plates get pooled."""
    a = _write_stack(tmp_path / "plateA", [[_disc_plane(radius=12, seed=1)]], plate="plateA")
    b = _write_stack(tmp_path / "plateB", [[_disc_plane(radius=12, seed=2)]], plate="plateB")
    est = estimate_diameters([a, b], {"cell": 0}, n_fields=5)["cell"]
    assert est.n_fields == 2
    assert est.diameter == pytest.approx(24.0, rel=0.15)


# ---------------------------------------------------------------------------
# bad channel indices
# ---------------------------------------------------------------------------

def test_channel_index_out_of_range_is_reported_not_an_index_error(clean_source):
    """The off-by-one that costs a GPU run must read as a sentence."""
    est = estimate_diameters(clean_source, {"cell": 0, "pathogen": 7}, n_fields=2)

    assert est["cell"].usable
    bad = est["pathogen"]
    assert not bad.usable
    assert "out of range" in bad.note
    assert "3 channel" in bad.note
    assert "0-2" in bad.note
    assert SETTING_KEYS["pathogen"] in bad.note


def test_negative_channel_index_is_reported(clean_source):
    est = estimate_diameters(clean_source, {"cell": -1}, n_fields=2)["cell"]
    assert not est.usable
    assert "negative" in est.note


def test_two_dimensional_arrays_only_have_channel_zero(tmp_path):
    """A plain 2-D .npy is a one-channel field, and asking for channel 1 says so."""
    stack = tmp_path / "stack"
    stack.mkdir()
    for i in range(3):
        np.save(stack / f"plate1_A{i + 1:02d}_1_t0.npy", _disc_plane(radius=12, seed=i))

    est = estimate_diameters(str(tmp_path), {"cell": 0, "nucleus": 1}, n_fields=3)
    assert est["cell"].usable
    assert est["cell"].diameter == pytest.approx(24.0, rel=0.15)
    assert not est["nucleus"].usable
    assert "out of range" in est["nucleus"].note or "2-D" in est["nucleus"].note


# ---------------------------------------------------------------------------
# raw acquisition files, parsed with the metadata regex
# ---------------------------------------------------------------------------

def test_raw_cellvoyager_files_are_grouped_into_fields(tmp_path):
    """No stack/ yet: fall back to the filename metadata, as spacr.io does.

    Channel index i is the i-th sorted chanID, which is exactly the ordering
    spacr.io._rename_and_organize_image_files bakes into stack/.
    """
    from PIL import Image

    for field in range(4):
        for chan, radius in (("01", 15), ("02", 6)):
            plane = _disc_plane(shape=(320, 320), radius=radius, step=4 * radius,
                                seed=field * 10 + int(chan))
            name = f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C{chan}.tif"
            Image.fromarray(plane).save(tmp_path / name)

    est = estimate_diameters(str(tmp_path), {"cell": 0, "nucleus": 1},
                             n_fields=4, metadata_type="cellvoyager")

    assert est["cell"].usable, est["cell"].note
    assert est["nucleus"].usable, est["nucleus"].note
    assert est["cell"].n_fields == 4
    assert est["cell"].diameter == pytest.approx(30.0, rel=0.15), est["cell"].diameter
    assert est["nucleus"].diameter == pytest.approx(12.0, rel=0.15), est["nucleus"].diameter


def test_raw_files_report_an_out_of_range_channel(tmp_path):
    from PIL import Image

    for field in range(2):
        for chan in ("01", "02"):
            name = f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C{chan}.tif"
            Image.fromarray(_disc_plane(radius=10, seed=field)).save(tmp_path / name)

    est = estimate_diameters(str(tmp_path), {"cell": 5}, n_fields=2,
                             metadata_type="cellvoyager")["cell"]
    assert not est.usable
    assert "out of range" in est.note
    assert "2 channel" in est.note


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_format_estimates_contains_the_numbers_and_the_confidence(clean_source):
    est = estimate_diameters(clean_source, {"cell": 0, "nucleus": 1, "pathogen": 2}, n_fields=5)
    text = format_estimates(est)

    assert isinstance(text, str) and "\n" in text
    for obj in ("cell", "nucleus", "pathogen"):
        assert obj in text
        assert est[obj].confidence in text
        assert SETTING_KEYS[obj] in text
    # the proposed numbers themselves, to one decimal
    assert f"{est['cell'].diameter:.1f}" in text
    assert f"{est['nucleus'].diameter:.1f}" in text
    # the unusable channel is shown as "no estimate", never as a number
    assert "no estimate" in text
    assert "nan" not in text.lower()
    # the plausible range is carried through
    assert f"{est['cell'].low:.1f}-{est['cell'].high:.1f}" in text


def test_format_estimates_handles_an_empty_mapping():
    text = format_estimates({})
    assert "nothing requested" in text


def test_diameter_estimate_str_is_readable(clean_source):
    est = estimate_diameters(clean_source, {"cell": 0}, n_fields=2)["cell"]
    assert "cell" in str(est) and "confidence" in str(est)

    from spacr.diameter import _no_estimate
    assert "no estimate" in str(_no_estimate("cell", "because"))


def test_note_tells_the_user_what_to_check(clean_source):
    est = estimate_diameters(clean_source, {"cell": 0}, n_fields=5)["cell"]
    assert "channel 0" in est.note
    assert "saturated" in est.note
    assert "by hand" in est.note


# ---------------------------------------------------------------------------
# settings glue
# ---------------------------------------------------------------------------

def test_channels_from_settings_reads_the_four_object_channels():
    settings = {
        "cell_channel": 2,
        "nucleus_channel": "0",          # settings CSVs hand back strings
        "pathogen_channel": None,
        "organelle_channel": 3.0,
        "unrelated": 9,
    }
    assert channels_from_settings(settings) == {"cell": 2, "nucleus": 0, "organelle": 3}
    assert channels_from_settings({}) == {}
    assert channels_from_settings(None) == {}


def test_a_boolean_is_not_mistaken_for_channel_zero_or_one():
    assert channels_from_settings({"cell_channel": True}) == {}
    assert channels_from_settings({"cell_channel": False}) == {}


@pytest.mark.parametrize("value", [[0], {"a": 1}, 1.5, object()])
def test_channel_values_that_are_not_an_index_are_ignored(value):
    assert channels_from_settings({"cell_channel": value}) == {}


def test_setting_keys_cover_every_object_type():
    assert set(SETTING_KEYS) == set(OBJECT_TYPES)
    assert SETTING_KEYS["cell"] == "cell_diameter"


def test_settings_declares_diameter_estimate_n_fields():
    """The knob is typed, defaulted, tooltipped and reachable from the GUI."""
    import spacr.settings as S

    assert S.expected_types["diameter_estimate_n_fields"] is int
    assert S.set_default_settings_preprocess_generate_masks({})["diameter_estimate_n_fields"] == 5

    tip = S.tooltips["diameter_estimate_n_fields"]
    assert tip.startswith("(int) - ")
    assert len(tip.split(" - ", 1)[1].split()) >= 15

    assert any("diameter_estimate_n_fields" in keys for keys in S.categories.values())


def test_an_explicit_setting_value_is_respected():
    import spacr.settings as S

    out = S.set_default_settings_preprocess_generate_masks({"diameter_estimate_n_fields": 12})
    assert out["diameter_estimate_n_fields"] == 12


# ---------------------------------------------------------------------------
# the cost guarantee: no torch, no cellpose
# ---------------------------------------------------------------------------

def test_a_call_does_not_pull_in_torch_or_cellpose(clean_source):
    """In-process guard: the call must not *add* torch or cellpose."""
    before = {m.split(".")[0] for m in list(sys.modules)}
    estimate_diameters(clean_source, {"cell": 0, "nucleus": 1}, n_fields=2)
    after = {m.split(".")[0] for m in list(sys.modules)}
    added = (after - before) & {"torch", "torchvision", "cellpose"}
    assert not added, f"estimate_diameters imported {sorted(added)}"


def test_neither_torch_nor_cellpose_is_in_sys_modules_after_a_call(tmp_path):
    """The real guarantee, checked in a subprocess with a clean interpreter.

    Running in-process cannot prove this: the pytest session has already
    imported half of spaCR, and the coverage runner deliberately pre-imports
    torch via a sitecustomize shim. So a fresh interpreter is started with
    PYTHONPATH set to the repo alone, and it is asked what it ended up with.

    The point is not purity for its own sake. Importing torch and cellpose
    costs seconds and hundreds of MB, and this estimator's entire value is
    being cheap enough to run *before* committing to a segmentation run.
    """
    src = _write_stack(tmp_path, [[_disc_plane(radius=12, seed=i)] for i in range(3)])

    code = textwrap.dedent(
        """
        import sys
        from spacr.diameter import estimate_diameters, format_estimates

        est = estimate_diameters(sys.argv[1], {"cell": 0}, n_fields=3)
        assert est["cell"].usable, est["cell"].note
        format_estimates(est)

        heavy = sorted({m.split(".")[0] for m in sys.modules}
                       & {"torch", "torchvision", "cellpose", "tensorflow"})
        print("HEAVY:" + ",".join(heavy))
        print("DIAM:%.3f" % est["cell"].diameter)
        """
    )

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.run(
        [sys.executable, "-c", code, src],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "HEAVY:\n" in proc.stdout or "HEAVY:" in proc.stdout
    heavy_line = next(l for l in proc.stdout.splitlines() if l.startswith("HEAVY:"))
    assert heavy_line == "HEAVY:", f"heavy modules imported: {heavy_line}"
    diam = float(next(l for l in proc.stdout.splitlines() if l.startswith("DIAM:"))[5:])
    assert diam == pytest.approx(24.0, rel=0.15)


def test_the_module_source_does_not_import_torch_or_cellpose():
    """Belt and braces: a lazy import inside a helper would still be a cost."""
    import spacr.diameter as D

    source = Path(D.__file__).read_text()
    for banned in ("import torch", "from torch", "import cellpose", "from cellpose",
                   "import tensorflow", "from tensorflow"):
        assert banned not in source, f"spacr/diameter.py contains {banned!r}"


# ---------------------------------------------------------------------------
# dataclass contract
# ---------------------------------------------------------------------------

def test_estimate_carries_every_declared_field(clean_source):
    est = estimate_diameters(clean_source, {"cell": 0}, n_fields=3)["cell"]
    assert isinstance(est, DiameterEstimate)
    for name in ("object_type", "diameter", "low", "high", "n_objects",
                 "n_fields", "method", "confidence", "note"):
        assert hasattr(est, name)
    assert est.confidence in ("high", "medium", "low")
    assert est.method in ("threshold_otsu", "watershed_edt", "none")
    assert isinstance(est.n_objects, int) and isinstance(est.n_fields, int)
    assert est.note and est.note[-1] == "."


def test_estimates_are_deterministic(clean_source):
    """Same folder, same answer — the default stride sampler uses no randomness."""
    a = estimate_diameters(clean_source, {"cell": 0}, n_fields=4)["cell"]
    b = estimate_diameters(clean_source, {"cell": 0}, n_fields=4)["cell"]
    assert a == b


def test_verbose_prints_each_estimate(clean_source, capsys):
    estimate_diameters(clean_source, {"cell": 0}, n_fields=2, verbose=True)
    assert "cell:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# confidence: every downgrade has to be earned and named
# ---------------------------------------------------------------------------

def test_a_handful_of_objects_drops_confidence_to_low(tmp_path):
    """Four objects is not a size distribution."""
    fields = [[_disc_plane(shape=(256, 256), radius=20, step=110, seed=i)] for i in range(2)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=2)["cell"]
    assert est.usable
    assert est.n_objects < 10
    assert est.confidence == "low"
    assert f"only {est.n_objects} objects measured" in est.note


def test_a_moderate_object_count_drops_confidence_to_medium(tmp_path):
    fields = [[_disc_plane(shape=(280, 280), radius=18, step=110, seed=i)] for i in range(2)]
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=2)["cell"]
    assert est.usable
    assert 10 <= est.n_objects < 30
    assert est.confidence == "medium"
    assert "objects measured" in est.note


def test_a_very_wide_size_spread_drops_confidence_and_says_so(tmp_path):
    """Discs of 12 px and 60 px in the same field are not one population."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:400, 0:400]
    fields = []
    for f in range(4):
        img = np.full((400, 400), 200.0, np.float32)
        for radius, step in ((6, 40), (30, 130)):
            for cy in range(radius + 4, 400 - radius - 4, step):
                for cx in range(radius + 4, 400 - radius - 4, step):
                    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 3200.0
        img += rng.normal(0, 25, (400, 400))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=4)["cell"]
    assert est.usable
    assert est.confidence != "high"
    assert "size spread" in est.note, est.note
    assert "two populations" in est.note, est.note
    # the reported range has to show both populations even though the median
    # sits on the more numerous one
    assert est.low <= est.diameter < est.high
    assert est.high / est.low > 3.0


def test_a_moderate_size_spread_drops_confidence_one_step(tmp_path):
    """A genuinely heterogeneous population is medium, not low and not high."""
    rng = np.random.default_rng(29)
    yy, xx = np.mgrid[0:500, 0:500]
    radii = [8, 10, 12, 14, 18, 24]
    fields = []
    for f in range(3):
        img = np.full((500, 500), 200.0, np.float32)
        k = 0
        for cy in range(35, 466, 70):
            for cx in range(35, 466, 70):
                r = radii[k % len(radii)]
                k += 1
                img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] += 3000.0
        img += rng.normal(0, 25, (500, 500))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=3)["cell"]
    assert est.usable
    assert est.confidence == "medium", est.note
    assert "wide size spread" in est.note
    assert "two populations" not in est.note


def test_confluence_wins_over_a_handful_of_surviving_specks(tmp_path):
    """The nastiest version of the collapse: debris survives and lies quietly.

    Two thirds of the field is a tangent packing (d=28) and the rest holds a
    few isolated specks (d=16). Plain thresholding drops the fused block and
    keeps only the specks, so it would answer 16 with nothing obviously wrong.
    The distance transform outnumbers it in a field that is nearly half
    foreground, which is the signature that has to trigger.
    """
    radius = 14
    rng = np.random.default_rng(31)
    yy, xx = np.mgrid[0:420, 0:420]
    fields = []
    for f in range(3):
        img = np.full((420, 420), 200.0, np.float32)
        for cy in range(radius + 2, 420 - radius - 2, 2 * radius):
            for cx in range(radius + 2, 280, 2 * radius):
                img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] += 3000.0
        for cy in range(40, 400, 90):
            img[(yy - cy) ** 2 + (xx - 350) ** 2 <= 8 ** 2] += 3000.0
        img += rng.normal(0, 25, (420, 420))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=3)["cell"]
    assert est.method == "watershed_edt"
    assert est.diameter == pytest.approx(2.0 * radius, rel=0.15), est.note
    assert est.confidence != "high"
    assert "resolves" in est.note and "plain thresholding kept" in est.note, est.note


def test_hollow_objects_make_the_two_measurements_disagree_out_loud(tmp_path):
    """A membrane-only ring reads full width by area and thin by distance transform.

    This is the mirror image of confluence, and the trap a naive "the distance
    transform found more objects, so they must be fused" rule falls into: one
    ring is a single correct component by area but shatters into dozens of arcs
    under the transform. The threshold answer has to survive, and the
    disagreement has to be said out loud rather than resolved silently.
    """
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:400, 0:400]
    fields = []
    for f in range(4):
        img = np.full((400, 400), 200.0, np.float32)
        for cy in range(40, 370, 80):
            for cx in range(40, 370, 80):
                r2 = (yy - cy) ** 2 + (xx - cx) ** 2
                img[(r2 <= 30 ** 2) & (r2 >= 25 ** 2)] = 3200.0
        img += rng.normal(0, 20, (400, 400))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=4)["cell"]
    assert est.usable
    assert est.method == "threshold_otsu", est.note
    assert est.diameter == pytest.approx(60.0, rel=0.15), est.note
    assert est.confidence != "high"
    assert "disagree" in est.note, est.note


def test_a_fully_confluent_field_says_thresholding_kept_nothing(tmp_path):
    """Discs packed right up to the field edge leave plain labelling empty-handed."""
    radius = 14
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:384, 0:384]
    fields = []
    for f in range(3):
        img = np.full((384, 384), 200.0, np.float32)
        for cy in range(0, 384 + radius, 2 * radius):
            for cx in range(0, 384 + radius, 2 * radius):
                img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] += 3000.0
        img += rng.normal(0, 25, (384, 384))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=3)["cell"]
    assert est.usable, est.note
    assert est.method == "watershed_edt"
    assert est.diameter == pytest.approx(2.0 * radius, rel=0.15), est.diameter
    assert "kept no whole object at all" in est.note, est.note


def test_one_object_filling_the_field_yields_no_number(tmp_path):
    """A single blob covering the plate is not an object measurement."""
    rng = np.random.default_rng(13)
    yy, xx = np.mgrid[0:256, 0:256]
    fields = []
    for f in range(3):
        img = np.full((256, 256), 200.0, np.float32)
        img[(yy - 128) ** 2 + (xx - 128) ** 2 <= 100 ** 2] += 3000.0
        img += rng.normal(0, 20, (256, 256))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=3)["cell"]
    assert not est.usable
    assert "no object that survived" in est.note, est.note


def test_thin_filaments_give_no_distance_transform_seeds(tmp_path):
    """A 1 px filament has no inscribed circle; the threshold path carries alone."""
    rng = np.random.default_rng(17)
    fields = []
    for f in range(3):
        img = np.full((256, 256), 200.0, np.float32)
        img[::16, 8:248] = 3200.0            # horizontal hairlines, 1 px wide
        img += rng.normal(0, 15, (256, 256))
        fields.append([np.clip(img, 0, 65535).astype(np.uint16)])
    src = _write_stack(tmp_path, fields)

    est = estimate_diameters(src, {"cell": 0}, n_fields=3, min_object_diameter=4.0)["cell"]
    # Either it measures the filaments by area or it declines; what it must not
    # do is raise, and it must not report a watershed split it never made.
    if est.usable:
        assert est.method == "threshold_otsu"


# ---------------------------------------------------------------------------
# ragged and damaged sources
# ---------------------------------------------------------------------------

def test_a_field_with_fewer_channels_than_the_first_is_reported(tmp_path):
    """A ragged plate must not raise IndexError halfway through the sample."""
    stack = tmp_path / "stack"
    stack.mkdir()
    three = np.stack([_disc_plane(radius=12, seed=1)] * 3, axis=-1)
    two = np.stack([_disc_plane(radius=12, seed=2)] * 2, axis=-1)
    np.save(stack / "plate1_A01_1_t0.npy", three)
    np.save(stack / "plate1_A02_1_t0.npy", two)

    est = estimate_diameters(str(tmp_path), {"pathogen": 2}, n_fields=2)["pathogen"]
    assert not est.usable
    assert "valid channels are 0-1" in est.note or "out of range" in est.note


def test_an_unreadable_field_is_skipped_not_fatal(tmp_path):
    """One corrupt .npy costs that field, not the estimate."""
    stack = tmp_path / "stack"
    stack.mkdir()
    for i in range(4):
        np.save(stack / f"plate1_A{i + 1:02d}_1_t0.npy", _disc_plane(radius=12, seed=i))
    (stack / "plate1_A03_1_t0.npy").write_bytes(b"not a numpy file at all")

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=4)["cell"]
    assert est.usable, est.note
    assert est.n_fields == 3
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_a_corrupt_first_array_does_not_stop_the_channel_count_probe(tmp_path):
    """The header probe reads the first few files; damaged ones cost it nothing."""
    stack = tmp_path / "stack"
    stack.mkdir()
    (stack / "plate1_A00_1_t0.npy").write_bytes(b"junk")
    for i in range(3):
        np.save(stack / f"plate1_A{i + 1:02d}_1_t0.npy", _disc_plane(radius=12, seed=i))

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=4)["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_the_channel_count_can_stay_unknown_and_loading_still_reports_the_range(tmp_path):
    """When every probed header is damaged the up-front check cannot run.

    The load itself has to produce the same sentence rather than an
    IndexError, which is the backstop this exercises.
    """
    stack = tmp_path / "stack"
    stack.mkdir()
    for i in range(3):                       # sorted first, so the probe sees only these
        (stack / f"plate1_A0{i}_1_t0.npy").write_bytes(b"junk")
    for i in range(3):
        np.save(stack / f"plate1_B{i + 1:02d}_1_t0.npy",
                np.stack([_disc_plane(radius=12, seed=i)] * 2, axis=-1))

    good = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=6)["cell"]
    assert good.usable, good.note
    assert good.diameter == pytest.approx(24.0, rel=0.15)

    bad = estimate_diameters(str(tmp_path), {"cell": 9}, n_fields=6)["cell"]
    assert not bad.usable
    assert "valid channels are 0-1" in bad.note


def test_raw_field_missing_one_channel_is_noted_not_fatal(tmp_path):
    from PIL import Image

    for field in range(4):
        Image.fromarray(_disc_plane(radius=12, seed=field)).save(
            tmp_path / f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C01.tif"
        )
        if field != 2:                      # field 2 never acquired channel 2
            Image.fromarray(_disc_plane(radius=6, step=50, seed=field)).save(
                tmp_path / f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C02.tif"
            )

    est = estimate_diameters(str(tmp_path), {"nucleus": 1}, n_fields=4,
                             metadata_type="cellvoyager")["nucleus"]
    assert est.usable, est.note
    assert est.n_fields == 3
    assert est.diameter == pytest.approx(12.0, rel=0.15)


def test_raw_z_slices_are_maximum_projected(tmp_path):
    """Several Z planes per (field, channel) collapse the way spacr.io does."""
    from PIL import Image

    for field in range(3):
        bright = _disc_plane(radius=12, seed=field)
        for z, plane in enumerate((np.zeros_like(bright), bright), start=1):
            Image.fromarray(plane).save(
                tmp_path / f"plate1_A{field + 1:02d}_T0001F001L01A01Z{z:02d}C01.tif"
            )

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=3,
                             metadata_type="cellvoyager")["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_filenames_that_match_no_metadata_regex_are_reported(tmp_path):
    from PIL import Image

    for i in range(3):
        Image.fromarray(_disc_plane(radius=12, seed=i)).save(tmp_path / f"random_name_{i}.tif")

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=3)["cell"]
    assert not est.usable
    assert "metadata regex" in est.note or "no fields" in est.note


def test_files_that_do_not_match_the_winning_pattern_are_skipped(tmp_path):
    """A stray README.tif in the acquisition folder is not a field."""
    from PIL import Image

    for field in range(3):
        Image.fromarray(_disc_plane(radius=12, seed=field)).save(
            tmp_path / f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C01.tif"
        )
    Image.fromarray(_disc_plane(radius=30, seed=99)).save(tmp_path / "thumbnail.png")

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=5,
                             metadata_type="cellvoyager")["cell"]
    assert est.usable, est.note
    assert est.n_fields == 3
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_a_custom_regex_with_an_optional_channel_group_skips_the_files_without_one(tmp_path):
    from PIL import Image

    for field in range(3):
        Image.fromarray(_disc_plane(radius=12, seed=field)).save(
            tmp_path / f"plate1_A{field + 1:02d}_1_C01.tif"
        )
        Image.fromarray(_disc_plane(radius=30, seed=99)).save(
            tmp_path / f"plate1_A{field + 1:02d}_2.tif"      # no channel group
        )

    est = estimate_diameters(
        str(tmp_path), {"cell": 0}, n_fields=5, metadata_type="custom",
        custom_regex=r"(?P<plateID>[^_]+)_(?P<wellID>[^_]+)_(?P<fieldID>\d+)(?:_C(?P<chanID>\d+))?",
    )["cell"]
    assert est.usable, est.note
    assert est.n_fields == 3
    assert est.diameter == pytest.approx(24.0, rel=0.15), est.note


def test_an_uncompilable_custom_regex_falls_back_to_the_known_patterns(tmp_path):
    """A bad custom_regex must not take the whole estimate down with it."""
    from PIL import Image

    for field in range(3):
        Image.fromarray(_disc_plane(radius=12, seed=field)).save(
            tmp_path / f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C01.tif"
        )

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=3,
                             metadata_type="custom", custom_regex="(unclosed[")["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_a_merged_folder_is_used_when_there_is_no_stack(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    for i in range(3):
        arr = np.stack([_disc_plane(radius=12, seed=i)] * 2, axis=-1)
        np.save(merged / f"plate1_A{i + 1:02d}_1_t0.npy", arr)

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=3)["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.15)


def test_raw_files_are_found_in_the_orig_backup_folder(tmp_path):
    """After preprocessing with save_original_images, the raws live in orig/."""
    from PIL import Image

    orig = tmp_path / "orig"
    orig.mkdir()
    for field in range(3):
        Image.fromarray(_disc_plane(radius=12, seed=field)).save(
            orig / f"plate1_A{field + 1:02d}_T0001F001L01A01Z01C01.tif"
        )

    est = estimate_diameters(str(tmp_path), {"cell": 0}, n_fields=3,
                             metadata_type="cellvoyager")["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.15)


# ---------------------------------------------------------------------------
# helper units
# ---------------------------------------------------------------------------

def test_roots_accepts_a_path_a_string_a_list_and_nothing(tmp_path):
    assert _roots(str(tmp_path)) == [str(tmp_path)]
    assert _roots(tmp_path) == [str(tmp_path)]
    assert _roots([str(tmp_path), tmp_path]) == [str(tmp_path), str(tmp_path)]
    assert _roots(None) == []
    assert _roots(17) == []
    assert _roots([1, 2]) == []


def test_a_src_that_is_not_a_path_is_reported(tmp_path):
    est = estimate_diameters(17, {"cell": 0})["cell"]
    assert not est.usable
    assert "src is empty or not a folder path" in est.note


def test_to_2d_reduces_rgb_and_z_stacks():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 60
    rgb[..., 2] = 90
    assert _to_2d(rgb).shape == (8, 8)
    assert _to_2d(rgb)[0, 0] == pytest.approx(60.0)

    zstack = np.zeros((4, 8, 8), np.uint16)
    zstack[2, 3, 3] = 700
    flat = _to_2d(zstack)
    assert flat.shape == (8, 8) and flat[3, 3] == 700

    assert _to_2d(np.zeros((8, 8))).shape == (8, 8)


def test_crop_to_takes_a_centred_window_only_when_oversized():
    big = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
    assert _crop_to(big, 400 * 400) is big
    assert _crop_to(big, 0) is big
    small = _crop_to(big, 100 * 100)
    assert small.shape == (100, 100)
    assert small[0, 0] == big[150, 150]


def test_illumination_uses_a_direct_filter_when_the_field_is_small():
    img = np.ones((40, 40), np.float32) * 5.0
    out = _illumination(img, 8.0)
    assert out.shape == (40, 40)
    assert out == pytest.approx(np.full((40, 40), 5.0), abs=1e-3)


def test_illumination_matches_a_direct_gaussian_on_a_smooth_field():
    """The decimation shortcut must not change the answer it is a shortcut for."""
    from scipy.ndimage import gaussian_filter

    yy, xx = np.mgrid[0:512, 0:512]
    field = (1000.0 + 3.0 * yy + 2.0 * xx).astype(np.float32)
    fast = _illumination(field, 128.0)
    slow = gaussian_filter(field, 128.0, mode="nearest")
    inner = (slice(64, 448), slice(64, 448))
    assert np.abs(fast[inner] - slow[inner]).max() < 0.02 * np.ptp(field)  # ndarray.ptp() removed in numpy 2.0


def test_region_diameters_handles_the_empty_cases():
    empty = np.zeros((8, 8), np.int32)
    assert _region_diameters(empty, 0, 4.0, 100.0).size == 0

    # one region, and it touches the border -> dropped as truncated
    border = np.zeros((8, 8), np.int32)
    border[0, 0:3] = 1
    assert _region_diameters(border, 1, 1.0, 100.0).size == 0

    # label ids that were counted but claim no pixels at all
    assert _region_diameters(np.zeros((8, 8), np.int32), 2, 0.5, 100.0).size == 0

    # a label id that exists in the count but claims no pixels
    ghost = np.zeros((8, 8), np.int32)
    ghost[3:5, 3:5] = 2
    diams = _region_diameters(ghost, 2, 0.5, 100.0)
    assert diams.size == 1
    assert diams[0] == pytest.approx(2.0 * math.sqrt(4.0 / math.pi))

    # oversized regions are dropped rather than measured
    assert _region_diameters(ghost, 2, 0.5, 1.0).size == 0


def test_analyse_plane_rejects_shapes_it_cannot_measure():
    assert not _analyse_plane(np.zeros((4, 4, 4)), 4.0, 4.0, 0.25).ok
    assert "2-D" in _analyse_plane(np.zeros((4, 4, 4)), 4.0, 4.0, 0.25).reason
    assert "too small" in _analyse_plane(np.zeros((8, 8)), 4.0, 4.0, 0.25).reason


def test_measure_degrades_an_unmeasurable_field_instead_of_raising():
    bad = np.array([["a", "b"], ["c", "d"]])
    result = _measure(bad, 4.0, 4.0, 0.25)
    assert not result.ok
    assert "could not measure field" in result.reason


def test_a_bright_field_with_one_dark_speck_is_declined_as_noise(tmp_path):
    """There is no object population here, and none may be invented."""
    rng = np.random.default_rng(23)
    img = np.full((256, 256), 6000.0, np.float32)
    img[10:14, 10:14] = 0.0
    img += rng.normal(0, 5, (256, 256))
    result = _analyse_plane(img.astype(np.uint16), 4.0, 4.0, 0.25)
    assert not result.ok
    assert "noise" in result.reason or "flat" in result.reason


def test_load_array_plane_rejects_channels_that_do_not_exist(tmp_path):
    two_d = tmp_path / "flat.npy"
    np.save(two_d, np.zeros((8, 8), np.uint16))
    assert _load_array_plane(str(two_d), 0).shape == (8, 8)
    with pytest.raises(IndexError, match="only channel 0 exists"):
        _load_array_plane(str(two_d), 1)

    three_d = tmp_path / "stackish.npy"
    np.save(three_d, np.zeros((8, 8, 2), np.uint16))
    assert _load_array_plane(str(three_d), 1).shape == (8, 8)
    with pytest.raises(IndexError, match="valid channels are 0-1"):
        _load_array_plane(str(three_d), 5)


def test_load_raw_plane_needs_at_least_one_readable_file():
    with pytest.raises(ValueError, match="no readable image"):
        _load_raw_plane([])


def test_oversized_fields_are_cropped_without_shifting_the_estimate(tmp_path):
    """The crop is a speed measure; it must not move the number."""
    fields = [[_disc_plane(shape=(600, 600), radius=15, seed=i)] for i in range(2)]
    src = _write_stack(tmp_path, fields)

    full = estimate_diameters(src, {"cell": 0}, n_fields=2)["cell"]
    cropped = estimate_diameters(src, {"cell": 0}, n_fields=2, max_pixels=300 * 300)["cell"]
    assert cropped.n_objects < full.n_objects
    assert cropped.diameter == pytest.approx(full.diameter, rel=0.05)
    assert cropped.diameter == pytest.approx(30.0, rel=0.15)
