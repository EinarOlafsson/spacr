"""Vendor flat-field profiles replace the estimated illumination field.

An Operetta / Opera Phenix export carries Harmony's own flat-field correction
in ``Index.idx.xml``: per channel a foreground (gain) and background (offset)
polynomial. Harmony's corrected image is ``(raw - background) / foreground``.
These tests write a synthetic export, compute Harmony's corrected output
independently of spaCR, and check spaCR's correction lands on the same
pixels. A ZEN-style shading reference image is read too.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr import illumination as ill
from spacr.measure_hooks import PreprocessingContext

W, H = 40, 30

FOREGROUND = {
    1: [[1.0], [0.08, -0.05], [-0.3, 0.02, -0.25]],
    2: [[0.9], [-0.04, 0.06], [-0.2, 0.0, -0.35]],
}
BACKGROUND = {1: [[95.0], [3.0, -2.0]], 2: [[110.0]]}
ORIGIN = [(W - 1) / 2.0, (H - 1) / 2.0]
SCALE = [2.0 / W, 2.0 / H]


def _profile(coefficients):
    return {"Character": "NonFlat",
            "Profile": {"Coefficients": coefficients, "Dims": [W, H],
                        "Origin": ORIGIN, "Scale": SCALE,
                        "Type": "Polynomial"}}


def _blob(channel):
    return {"Background": _profile(BACKGROUND[channel]), "Channel": channel,
            "ChannelName": f"ch{channel}",
            "Foreground": _profile(FOREGROUND[channel]),
            "ProfileSource": "Measurement", "Version": "1.0"}


def _bare(value):
    """Harmony's other spelling: bare keys and bare-word values."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_bare(v)}" for k, v in value.items()) \
            + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_bare(v) for v in value) + "]"
    return str(value) if not isinstance(value, str) else value


def _write_index(path, *, bare=False):
    entries = "".join(
        f'<Entry ChannelID="{c}"><FlatfieldProfile>'
        f'{_bare(_blob(c)) if bare else json.dumps(_blob(c))}'
        f'</FlatfieldProfile></Entry>' for c in (1, 2))
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<EvaluationInputData xmlns="http://www.perkinelmer.com/PEHH/'
        'HarmonyV5"><Maps><Map>' + entries + '</Map></Maps>'
        '</EvaluationInputData>')
    return path


def _harmony_polynomial(coefficients):
    """Harmony's surface, written out term by term without spaCR's code."""
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float64)
    x = (cols - ORIGIN[0]) * SCALE[0]
    y = (rows - ORIGIN[1]) * SCALE[1]
    out = np.zeros((H, W))
    for i in range(len(coefficients)):
        for j in range(i + 1):
            if j < len(coefficients[i]):
                out += coefficients[i][j] * x ** (i - j) * y ** j
    return out


def _harmony_corrected(raw):
    return np.stack([
        (raw[..., k] - _harmony_polynomial(BACKGROUND[c]))
        / _harmony_polynomial(FOREGROUND[c])
        for k, c in enumerate((1, 2))], axis=-1)


def _raw_field(seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(200, 3000, size=(H, W, 2))
    return np.stack([
        truth[..., k] * _harmony_polynomial(FOREGROUND[c])
        + _harmony_polynomial(BACKGROUND[c])
        for k, c in enumerate((1, 2))], axis=-1).astype(np.float32)


def _correct(model, raw, channels=(0, 1)):
    corrector = ill.IlluminationCorrector(model, verbose=False)
    context = PreprocessingContext(file_name="plate1_r01c01_f01",
                                   channels=list(channels), settings={})
    return corrector(raw, context)


@pytest.mark.parametrize("bare", [False, True])
def test_harmony_profile_matches_harmonys_corrected_output(tmp_path, bare):
    index = _write_index(tmp_path / "Index.idx.xml", bare=bare)
    model = ill._vendor_illumination(str(index), [0, 1], verbose=False)
    raw = _raw_field()

    corrected = _correct(model, raw)

    np.testing.assert_allclose(corrected, _harmony_corrected(raw),
                               rtol=1e-4, atol=1e-2)
    item = model.field_for("any_plate")
    assert item.estimator == "harmony"
    assert item.degree == 2
    assert model.meta["vendor_profile"] == str(index)


def test_prepare_reads_the_profile_and_the_saved_model_keeps_the_background(
        tmp_path):
    src = tmp_path / "plate1" / "merged"
    src.mkdir(parents=True)
    index = _write_index(tmp_path / "Index.idx.xml")
    settings = {"src": str(src), "channels": [0, 1],
                "illumination_correction": True,
                "illumination_vendor_profile": str(index),
                "illumination_qc": False, "verbose": False}

    prepared = ill.prepare_illumination_model(settings)

    reloaded = ill.load_illumination_model(prepared.model_path)
    assert reloaded.fields[ill.ALL_PLATES].darkfield is not None
    raw = _raw_field(1)
    np.testing.assert_allclose(_correct(reloaded, raw),
                               _harmony_corrected(raw), rtol=1e-4, atol=1e-2)


def test_the_estimate_is_used_when_no_profile_is_set():
    settings = ill.illumination_settings()
    assert settings["illumination_vendor_profile"] == ""


def test_a_channel_harmony_never_profiled_is_refused(tmp_path):
    index = _write_index(tmp_path / "Index.idx.xml")
    with pytest.raises(ill.IlluminationError, match="Harmony channel"):
        ill._vendor_illumination(str(index), [0, 2], verbose=False)


def test_xml_without_a_profile_is_refused(tmp_path):
    empty = tmp_path / "Index.idx.xml"
    empty.write_text("<Root><Images/></Root>")
    with pytest.raises(ill.IlluminationError, match="no Harmony"):
        ill._vendor_illumination(str(empty), [0], verbose=False)


def test_a_shading_reference_image_is_normalised_and_applied(tmp_path):
    import tifffile

    rows, cols = np.mgrid[0:H, 0:W]
    shading = 1000.0 + 400.0 * np.cos((cols - W / 2) / W * np.pi) \
        + 5.0 * rows
    path = tmp_path / "shading_reference.tif"
    tifffile.imwrite(str(path), (shading + 100.0).astype(np.float32))

    model = ill._vendor_illumination(str(path), [0, 1], dark=100.0,
                                     verbose=False)
    item = model.fields[ill.ALL_PLATES]
    assert item.darkfield is None
    assert item.flatfield.shape == (2, H, W)
    np.testing.assert_allclose(item.flatfield[0].mean(), 1.0, rtol=1e-5)

    truth = np.full((H, W, 2), 500.0, dtype=np.float32)
    flat = shading / shading.mean()
    raw = (truth * flat[..., None] + 100.0).astype(np.float32)
    np.testing.assert_allclose(_correct(model, raw), truth, rtol=1e-4)


def test_an_unknown_file_kind_is_refused(tmp_path):
    other = tmp_path / "profile.bin"
    other.write_bytes(b"\0")
    with pytest.raises(ill.IlluminationError, match="not a vendor"):
        ill._vendor_illumination(str(other), [0], verbose=False)
