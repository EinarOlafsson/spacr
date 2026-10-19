"""Item 696: the magnification sets the default diameter; 0 and "auto" are one value.

The maintainer, 2026-10-13: older spaCR used the magnification to work out
the default Cellpose diameter, and that should be used again; 0 must mean
"Cellpose works the diameter out" and be the same as typing auto.

So, for every model path, Cellpose-SAM included:

* a blank diameter is ``_get_diam(magnification, object)`` -- 20x gives a
  cell 120, a nucleus 60 and a pathogen 20 px; 40x gives 160, 75 and 40;
* 0 and "auto" (any case) are stored as "auto" and reach Cellpose as
  "estimate it": 0 for a Cellpose 3 model with a size model, None for
  Cellpose-SAM, which has none;
* settings files, the command line and validation take 0, auto and blank;
* the run journal names the diameter used, Cellpose's estimate included.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr.object_settings_table import (
    _AUTO_DIAMETER,
    _canonical_diameters,
    _diameter_from_magnification,
    _diameter_setting,
    _diameter_used_text,
    _resolved_diameter,
)
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS, eval_arguments


@pytest.mark.parametrize("mag, cell, nucleus, pathogen", [
    (20, 120, 60, 20),
    (40, 160, 75, 40),
])
def test_the_acceptance_table(mag, cell, nucleus, pathogen):
    from spacr.utils import _get_diam

    assert [_get_diam(mag, obj) for obj in ("cell", "nucleus", "pathogen")] \
        == [cell, nucleus, pathogen]
    assert [_resolved_diameter(None, mag, obj)
            for obj in ("cell", "nucleus", "pathogen")] \
        == [cell, nucleus, pathogen]


def test_get_diam_is_the_shared_formula():
    from spacr.utils import _get_diam

    for mag in (4, 10, 20, 40, 60, 100):
        for obj in ("cell", "cell_large", "nucleus", "pathogen"):
            assert _get_diam(mag, obj) == _diameter_from_magnification(mag, obj)
    with pytest.raises(ValueError, match="unsupported object type"):
        _get_diam(20, "organelle")


@pytest.mark.parametrize("raw", [0, 0.0, "0", "0.0", " 0 ", "auto", "Auto",
                                 "AUTO", " auto "])
def test_zero_and_auto_are_one_stored_value(raw):
    assert _diameter_setting(raw) == _AUTO_DIAMETER


@pytest.mark.parametrize("raw", [None, "", "   ", "None"])
def test_blank_is_the_magnification_default_not_auto(raw):
    assert _diameter_setting(raw) is None
    assert _resolved_diameter(raw, 20, "cell") == 120.0


@pytest.mark.parametrize("raw, stored", [(60, 60), ("60", 60), ("60.0", 60),
                                         (17.5, 17.5), ("17.5", 17.5)])
def test_a_number_is_itself(raw, stored):
    assert _diameter_setting(raw) == stored
    assert _resolved_diameter(raw, 20, "cell") == float(stored)


@pytest.mark.parametrize("raw", [-1, "-3", "thirty", True, float("nan"),
                                 float("inf"), "auto-ish", [], {}])
def test_anything_else_is_refused(raw):
    with pytest.raises(ValueError):
        _diameter_setting(raw)


def test_no_magnification_and_no_formula_fall_back_to_native_scale():
    assert _resolved_diameter(None, None, "cell") is None
    assert _resolved_diameter(None, 20, "organelle") is None


def test_legacy_unreadable_diameters_use_the_same_documented_default():
    assert _resolved_diameter("thirty", 20, "cell") == 120.0
    assert _diameter_used_text("thirty", 20, "cell") == \
        "120 px (magnification 20x default)"


def test_missing_magnification_keeps_native_scale_in_eval_and_the_journal():
    from spacr.object_settings_table import _default_diameter_of, _diameter_for_eval

    assert _default_diameter_of({}, "cell") is None
    assert _default_diameter_of({"magnification": 20}, "cell") == 120.0
    assert _diameter_for_eval(_resolved_diameter(None, None, "cell")) is None
    assert _diameter_used_text(None, None, "cell") == \
        "native scale (no magnification default)"


# ---------------------------------------------------------------------------
# What Cellpose is handed
# ---------------------------------------------------------------------------

def test_eval_diameter_applies_the_rule():
    from spacr.object import _eval_diameter

    assert _eval_diameter(None, "cell", 120) == 120.0
    assert _eval_diameter("", "cell", 120) == 120.0
    assert _eval_diameter("45", "cell", 120) == 45.0
    for auto in (0, "0", "auto", "Auto"):
        assert _eval_diameter(auto, "cell", 120) is None
        assert _eval_diameter(auto, "cell", 120, estimates=True) == 0.0


def test_an_unreadable_diameter_falls_back_to_the_magnification_default(capsys):
    from spacr.object import _eval_diameter

    assert _eval_diameter("thirty", "nucleus", 60) == 60.0
    assert "nucleus_diameter" in capsys.readouterr().out


@pytest.mark.parametrize("raw", [0, "auto", "Auto"])
def test_cellpose3_estimates_at_zero_and_auto(raw):
    from spacr.object import _cellpose3_eval_settings

    keywords, _ = _cellpose3_eval_settings(
        {"cell_diameter": raw}, "cell", default_diameter=120)
    assert keywords["diameter"] == 0.0


def test_cellpose3_blank_is_the_magnification_default():
    from spacr.object import _cellpose3_eval_settings

    keywords, _ = _cellpose3_eval_settings(
        {"cell_diameter": None}, "cell", default_diameter=120)
    assert keywords["diameter"] == 120.0


class _RecordingModel:
    """A Cellpose-shaped model that records what ``eval`` was handed."""

    def __init__(self, estimated=()):
        self.calls = []
        self.estimated_diameters = list(estimated)

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
        self.calls.append(eval_arguments(locals()))
        masks = [np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
                 for image in x]
        flows = [[np.zeros(m.shape + (3,), dtype=np.uint8), None, None, None]
                 for m in masks]
        return masks, flows, None


@pytest.mark.parametrize("raw, expected", [(None, 120.0), ("auto", None),
                                           (0, None), (45, 45.0)])
def test_the_dino_route_reads_the_magnification_default(raw, expected):
    """It used to ignore the default: "V1's Cellpose-SAM call does not read
    the magnification diameter"."""
    from spacr.object import _cellpose_dino_masks

    model = _RecordingModel()
    _cellpose_dino_masks(model, [np.zeros((8, 8, 1))], {"cell_diameter": raw},
                         "cell", min_size=15, default_diameter=120)
    assert model.calls[-1]["diameter"] == expected


def test_the_v1_cellpose_sam_call_uses_the_magnification_default(
        tmp_path, monkeypatch):
    """The generator itself: blank at 20x is 120 px, auto is None."""
    from tests.test_cov_object_masks_sam import _settings, _write_npz
    import spacr.object as O

    seen = []

    class _Model:
        def __init__(self, **kwargs):
            pass

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None, flow3D_smooth=0,
                 stitch_threshold=0.0, min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            seen.append(diameter)
            masks = [np.zeros(np.asarray(i).shape[:2], dtype=np.uint16)
                     for i in x]
            return masks, [np.zeros(m.shape, np.float32) for m in masks], None

    import torch
    import types
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(O.accelerator, "cellpose_kwargs",
                        lambda: {"gpu": False, "device": torch.device("cpu")})
    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(
        CellposeModel=_Model))
    for raw in (None, "auto", 0, 70):
        src = tmp_path / f"stack_{raw}"
        _write_npz(src, n=2)
        O.generate_cellpose_masks_sam(
            str(src), _settings(src, cell_diameter=raw), "cell")
    assert seen == [120.0, None, None, 70.0]


# ---------------------------------------------------------------------------
# Settings, files and the command line
# ---------------------------------------------------------------------------

def test_the_mask_defaults_store_zero_as_auto():
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks(
        {"cell_diameter": 0, "nucleus_diameter": "Auto",
         "pathogen_diameter": None})
    assert settings["cell_diameter"] == "auto"
    assert settings["nucleus_diameter"] == "auto"
    assert settings["pathogen_diameter"] is None


def test_object_settings_keep_the_magnification_size_bounds_for_auto():
    from spacr.settings import _get_object_settings

    base = {"magnification": 20, "verbose": False, "cell_diameter": "auto",
            "cell_model_name": "cpsam"}
    assert _get_object_settings("cell", base)["diameter"] == 120
    numeric = dict(base, cell_diameter="60")
    assert _get_object_settings("cell", numeric)["diameter"] == 60.0


def test_canonical_diameters_leaves_what_it_cannot_read_for_validation():
    settings = {"cell_diameter": "thirty", "nucleus_diameter": "0"}
    _canonical_diameters(settings)
    assert settings == {"cell_diameter": "thirty", "nucleus_diameter": "auto"}


@pytest.mark.parametrize("text, expected", [("0", "auto"), ("auto", "auto"),
                                            ("Auto", "auto"), ("none", None),
                                            ("", None), ("80", 80),
                                            ("12.5", 12.5)])
def test_the_command_line_takes_zero_auto_and_none(text, expected):
    from spacr.cli import coerce_value
    from spacr.settings import expected_types

    assert coerce_value("cell_diameter", text, None, expected_types,
                        "mask") == expected


def test_the_command_line_refuses_what_is_not_a_diameter():
    from spacr.cli import SettingsError, coerce_value
    from spacr.settings import expected_types

    for text in ("-5", "thirty"):
        with pytest.raises(SettingsError, match="is not a diameter"):
            coerce_value("cell_diameter", text, None, expected_types, "mask")


@pytest.mark.parametrize("written", ["0", "auto", "AUTO"])
def test_a_settings_file_round_trips_auto(tmp_path, written):
    from spacr.cli import (load_settings_file, resolve_module,
                           resolve_settings, write_settings_file)

    path = tmp_path / "mask_settings.csv"
    path.write_text(f"Key,Value\ncell_diameter,{written}\n"
                    f"nucleus_diameter,\n")
    module = resolve_module("mask")
    resolved = resolve_settings(module, str(path))
    assert resolved["cell_diameter"] == "auto"
    assert resolved["nucleus_diameter"] is None

    written_paths = write_settings_file(resolved, tmp_path / "again.csv")
    again = resolve_settings(module, written_paths[0])
    assert again["cell_diameter"] == "auto"
    assert again["nucleus_diameter"] is None
    assert load_settings_file(written_paths[0])["cell_diameter"] == "auto"


def test_validation_accepts_zero_auto_and_blank_and_refuses_the_rest():
    from spacr.validate import (_check_object_diameters, _check_numeric_sanity,
                                _check_types, coerce_expected_types)

    good = {"cell_diameter": "auto", "nucleus_diameter": 0,
            "pathogen_diameter": None}
    assert _check_object_diameters(good) == []
    assert not [p for p in _check_types(good, "mask")
                if p.setting.endswith("_diameter")]
    assert not [p for p in _check_numeric_sanity(good)
                if p.setting.endswith("_diameter")]

    bad = {"cell_diameter": -4, "nucleus_diameter": "thirty"}
    problems = _check_object_diameters(bad)
    assert {p.setting for p in problems} == {"cell_diameter",
                                            "nucleus_diameter"}
    assert all(p.severity == "error" for p in problems)

    coerced = coerce_expected_types({"cell_diameter": "0",
                                     "nucleus_diameter": "60"}, "mask")
    assert coerced == {"cell_diameter": "auto", "nucleus_diameter": 60}


def test_the_dry_run_names_the_diameter_it_will_use():
    from spacr.validate import _describe_objects

    lines = _describe_objects({"cell_channel": 0, "nucleus_channel": 1,
                               "pathogen_channel": 2, "magnification": 20,
                               "cell_diameter": None,
                               "nucleus_diameter": "auto",
                               "pathogen_diameter": 18}, "mask")
    assert "diameter 120 px (magnification 20x default)" in lines[0]
    assert "diameter auto" in lines[1]
    assert "diameter 18 px" in lines[2]


# ---------------------------------------------------------------------------
# The run journal
# ---------------------------------------------------------------------------

def test_the_journal_text_says_which_diameter_was_used():
    assert _diameter_used_text(None, 20, "cell") == \
        "120 px (magnification 20x default)"
    assert _diameter_used_text(60, 20, "cell") == "60 px"
    assert _diameter_used_text(0, 20, "cell") == "auto"
    assert _diameter_used_text("auto", 20, "cell", 57.25) == \
        "auto (Cellpose estimated 57.25 px)"


def test_the_manifest_records_the_diameters_and_the_estimate(
        tmp_path, monkeypatch):
    from spacr import run_journal
    from spacr.object import _cellpose3_masks, _note_the_diameter

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path)
    settings = {"magnification": 20, "cell_diameter": None,
                "nucleus_diameter": "auto"}
    with run_journal.open_run("mask", dict(settings)) as run:
        _note_the_diameter(settings, "cell")
        _note_the_diameter(settings, "nucleus")
        for estimate in (50.0, 60.0, 70.0):
            _cellpose3_masks(_RecordingModel([estimate]),
                             [np.zeros((8, 8, 2))], settings, "nucleus",
                             min_size=15, default_diameter=60)
        _note_the_diameter(settings, "nucleus")
        run_dir = run.dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["diameters"] == {
        "cell": "120 px (magnification 20x default)",
        "nucleus": "auto (Cellpose estimated 60 px)",
    }


def test_noting_a_diameter_without_a_run_is_harmless():
    from spacr.run_journal import _record_diameter, current_run

    assert current_run() is None
    _record_diameter("cell", "120 px")


def test_the_cellpose3_worker_reply_carries_its_estimates(tmp_path):
    """The size model's estimate crosses the backend's process boundary."""
    from spacr import _segmentation_backends as B

    class _Adapter:
        ignored, translated = set(), set()
        estimated_diameters = [41.5]

        def eval(self, images, channel_axis=-1, diameter=None, normalize=True,
                 flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
                 resample=True, batch_size=None):
            return [np.zeros((4, 4), np.uint16) for _ in images], [], None

    image = tmp_path / "image_0.npy"
    np.save(image, np.zeros((4, 4), np.uint16))
    adapters = {("cyto3", "cpu", json.dumps({}, sort_keys=True)): _Adapter()}
    reply = B._worker_segment(
        "cellpose3", {"device": "cpu", "model": "cyto3", "options": {},
                      "inputs": [str(image)], "outputs": str(tmp_path),
                      "params": {"diameter": 0.0}}, adapters)
    assert reply["estimated_diameters"] == [41.5]


def test_the_backend_client_keeps_the_estimates_it_was_sent():
    from spacr import _segmentation_backends as B

    class _Worker:
        def request(self, kind, *, outputs, inputs, **_):
            mask = f"{outputs}/mask_0.npy"
            np.save(mask, np.zeros((4, 4), np.uint16))
            return {"outputs": [{"mask": mask, "flows": [None] * 4}],
                    "estimated_diameters": [38.0]}

    client = B._RemoteBackend.__new__(B._RemoteBackend)
    client.name, client.label, client.model = "cellpose3", "Cellpose 3", "cyto3"
    client.env, client.device, client.options = "env", "cpu", {}
    client._worker_for = lambda name, env: _Worker()
    client._said = set()
    client.eval([np.zeros((4, 4), np.uint16)], diameter=0.0)
    assert client.estimated_diameters == [38.0]


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
@pytest.mark.parametrize('diagnostic', [0, 1, 2])
def test_diameter_failures_are_localized_without_changing_the_fallback(
        language, diagnostic, monkeypatch, capsys):
    from pathlib import Path
    from spacr.settings import _own_diameter
    from spacr.object import _eval_diameter, _note_the_diameter
    import spacr.run_journal as journal

    monkeypatch.setenv('SPACR_LANGUAGE', language)
    record_path = Path(__file__).resolve().parents[1] / (
        f'docs/i18n/reviewed/runtime/{language}/2026-10-14-diameter-diagnostics.json')
    record = json.loads(record_path.read_text(encoding="utf-8"))['records'][diagnostic]
    values = {'setting': 'nucleus_diameter', 'value': repr('thirty'),
              'object_type': 'nucleus', 'error': 'disk unavailable'}
    if diagnostic == 0:
        settings = {'diameter': 60.0, 'minimum_size': 900, 'maximum_size': 36000}
        original = settings.copy()
        _own_diameter(settings, {'nucleus_diameter': 'thirty'}, 'nucleus')
        assert settings == original
    elif diagnostic == 1:
        assert _eval_diameter('thirty', 'nucleus', 60) == 60.0
    else:
        def unavailable(*args, **kwargs):
            raise OSError('disk unavailable')

        monkeypatch.setattr(journal, '_record_diameter', unavailable)
        _note_the_diameter({'magnification': 20}, 'nucleus')
    assert capsys.readouterr().out.strip() == record['translation'].format(**values)
