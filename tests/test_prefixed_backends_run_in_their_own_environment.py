"""Items 551-554: StarDist, InstanSeg, Omnipose and Spotiflow, each in an
environment of its own and chosen by a model-setting prefix.

The maintainer, 2026-09-26: "(in its own env like cellpose 3)". What is
pinned here, for each backend:

1. the spec -- pins, licence, prefix, default model, the alpha mark -- and
   that it is no ``segmentation_backend`` value;
2. the install command, run with a fake runner: its own venv, the pins,
   the self-test, model downloads kept inside the environment;
3. the routing of ``<prefix><model>`` from Mask generation (V1 and the
   shared route table v2 and the previews use), and the parallel
   preparation that must refuse it;
4. the worker adapter, with the package faked: labels and probability in
   the shapes the Cellpose-SAM path reads, through the real ``.npy``
   hand-off, and the settings it cannot honour named;
5. the zoo rows: one per model, needing the backend until it is installed.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import spacr._segmentation_backends as SB
import spacr.object as O
from spacr import model_zoo as zoo
from spacr.spacr_cellpose import parse_cellpose4_output
import tests.test_object_tstack_wiring as _wiring
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.cellpose_api_contract import eval_arguments

_base_settings = _wiring._base_settings
_write_npz = _wiring._write_npz
_artifacts = _wiring._artifacts

PREFIXED = ("stardist", "instanseg", "omnipose", "spotiflow")


@pytest.fixture(autouse=True)
def _sandboxed(tmp_path, monkeypatch):
    """Own backends folder and home; no real environment counts."""
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(SB._TORCH_INDEX_ENV, raising=False)
    monkeypatch.delenv(SB._DEVICE_ENV, raising=False)
    monkeypatch.setattr(SB, "_PROBED", {})
    monkeypatch.setattr(SB, "_CANDIDATES", {})
    return tmp_path


def _finish(tmp_path, name):
    env = tmp_path / "backends" / name
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": name})
    return env


def _known_labels(shape):
    out = np.zeros(shape, np.int32)
    out[2:9, 3:10] = 1
    out[14:20, 12:22] = 2
    out[0, 0] = 3
    return out


# ===========================================================================
# 1. The specs
# ===========================================================================

def test_each_prefixed_backend_is_listed_by_its_prefix():
    assert SB._prefixed_names() == PREFIXED
    assert zoo.PREFIXED_KINDS == PREFIXED
    for name in PREFIXED:
        spec = SB._SPECS[name]
        assert spec.prefix == f"{name}:"
        assert spec.default_model == spec.models[0]
        assert spec.alpha, "future-list work ships behind item 569's switch"
        assert spec.segments and spec.licence and spec.licence_note
        assert "spaCR ships none of it" in spec.licence_note
        assert name not in SB._BACKEND_NAMES
        assert name not in zoo.INSTALLABLE_BACKENDS
        assert name in zoo.KINDS


def test_stardist_pins_tensorflow_and_no_pytorch():
    spec = SB._SPECS["stardist"]
    assert spec.requirements == ("stardist==0.9.2", "csbdeep==0.8.2",
                                 "tensorflow==2.21.0")
    assert spec.torch == ()
    assert spec.models == ("2D_versatile_fluo", "2D_versatile_he",
                           "2D_paper_dsb2018")
    assert "BSD-3-Clause" in spec.licence
    assert "stardist-models" in spec.licence_note


def test_the_zoo_lists_each_backend_with_its_state_and_licence(tmp_path):
    rows = {e.key: e for e in zoo.installable_backend_entries()}
    for name in PREFIXED:
        row = rows[f"{name}_v1"]
        assert (row.kind, row.uri) == ("backend", f"backend:{name}")
        assert row.source in ("installable", "not installable here")
        if row.source == "not installable here":
            assert "needs Python" in row.notes[0]
        assert row.notes[1] == SB._SPECS[name].licence_note
    env = _finish(tmp_path, "stardist")
    ready = {e.key: e for e in zoo.installable_backend_entries()}
    assert ready["stardist_v1"].source == "installed"
    assert ready["stardist_v1"].path == str(env)


def test_omnipose_uses_python_with_a_wheel_for_its_lxml_dependency():
    spec = SB._SPECS["omnipose"]
    assert spec.python == ((3, 11), (3, 12))
    candidates = SB._interpreter_candidates(
        spec, executable="/python3.13", version=(3, 13), frozen=False,
        which=lambda name: "/python3.12" if name == "python3.12" else None,
        windows=False)
    assert candidates == [("/python3.12",)]


# ===========================================================================
# 2. The install command, faked
# ===========================================================================

def _fake_runner(record):
    def _run(argv, env=None, cwd=None, on_line=None, cancel=None):
        record.append((tuple(argv), env))
        if argv[1:3] == ("-m", "venv"):
            python = Path(SB._env_python(argv[3]))
            python.parent.mkdir(parents=True)
            python.write_text("")
        on_line("ok")
        if "--selftest" in argv:
            return 0, [json.dumps({
                "ok": True, "python": "3.12.4", "device": "cpu",
                "packages": {"numpy": "2.5.3"}})]
        return 0, ["ok"]
    return _run


def test_stardist_installs_without_a_pytorch_step(tmp_path):
    record = []
    state = SB._install_backend(
        "stardist", runner=_fake_runner(record),
        preflight=lambda spec, root: (sys.executable,),
        torch_index="https://download.pytorch.org/whl/cpu")
    assert state.ready and not state.in_process
    env = str(tmp_path / "backends" / "stardist")
    argvs = [argv for argv, _env in record]
    assert argvs[0] == (sys.executable, "-m", "venv", env)
    assert len(argvs) == 4, "venv, pip, StarDist, self-test: no torch step"
    assert argvs[2][-3:] == SB._SPECS["stardist"].requirements
    assert argvs[3][-2:] == ("--selftest", "stardist")
    assert state.record["torch"] == []
    assert SB._stale_requirements("stardist", state.record) == []
    keras = os.path.join(env, "keras")
    assert all(e["KERAS_HOME"] == keras for _argv, e in record[1:])


def test_a_worker_without_pytorch_still_names_its_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    assert SB._worker_device("auto") == "cpu"
    assert SB._worker_device("cuda") == "cuda"


# ===========================================================================
# 3. Routing
# ===========================================================================

@pytest.mark.parametrize("value, expected", [
    ("stardist:2D_versatile_he", ("stardist", "2D_versatile_he")),
    ("  StarDist: 2D_paper_dsb2018 ", ("stardist", "2D_paper_dsb2018")),
    ("stardist:", ("stardist", "")),
    ("cellpose3:cyto3", (None, None)),
    ("cellpose_dino:/m/w", (None, None)),
    ("2D_versatile_fluo", (None, None)),
    ("cpsam", (None, None)),
    (None, (None, None)),
])
def test_a_prefixed_choice_is_read_from_its_prefix(value, expected):
    backend = SB._prefixed_backend(value)
    assert backend == expected[0]
    if backend:
        assert SB._prefixed_choice(backend, value) == expected[1]


@pytest.mark.parametrize("name", PREFIXED)
def test_the_model_is_a_name_the_default_or_a_path_that_exists(
        tmp_path, name):
    spec = SB._SPECS[name]
    prefix = spec.prefix
    assert SB._prefixed_value(name, spec.models[-1]) == prefix + spec.models[-1]
    assert SB._prefixed_value(name, prefix + "x") == prefix + "x"
    assert SB._prefixed_model(name, prefix) == spec.default_model
    for model in spec.models:
        assert SB._prefixed_model(name, prefix + model) == model
    folder = tmp_path / "my_model"
    folder.mkdir()
    assert SB._prefixed_model(name, f"{prefix}{folder}") == str(folder)
    with pytest.raises(FileNotFoundError, match="no file or folder"):
        SB._prefixed_model(name, f"{prefix}{tmp_path / 'gone'}")
    assert SB._prefixed_model_ok(prefix) is True
    assert SB._prefixed_model_ok(f"{prefix}{folder}") is True
    assert SB._prefixed_model_ok(f"{prefix}nope") is False
    assert SB._prefixed_model_ok("cpsam") is None


def test_a_run_knows_when_an_object_chose_a_prefixed_backend():
    assert SB._prefixed_is_chosen({"nucleus_model_name": "stardist:"})
    assert SB._prefixed_is_chosen({"pathogen_model": "stardist:x"})
    assert not SB._prefixed_is_chosen({"cell_model_name": "cpsam"})
    assert not SB._prefixed_is_chosen({"cell_model_name": "cellpose3:cyto3"})


def test_the_route_table_sends_each_prefix_to_its_backend():
    for name in PREFIXED:
        prefix = SB._SPECS[name].prefix
        assert O._prefixed_model_route(prefix) == (name, O._prefixed_masks)
        assert O._prefixed_model_route(
            prefix + "x", {"segmentation_backend": "cellpose3"}) == (
                name, O._prefixed_masks), "a prefix beats the backend"
    assert O._prefixed_model_route("cpsam") is None
    assert O._prefixed_model_route("cellpose3:cyto3")[0] == "cellpose3"


def test_load_backend_builds_the_remote_model(tmp_path, capsys):
    env = _finish(tmp_path, "stardist")
    model = SB._load_backend("cellpose", model_name="stardist:2D_versatile_he",
                             object_type="nucleus")
    assert isinstance(model, SB._RemoteBackend)
    assert (model.name, model.model, model.env, model.options) == (
        "stardist", "2D_versatile_he", str(env), {})
    assert "Segmentation backend: stardist" in capsys.readouterr().out
    bare = SB._load_backend("stardist", model_name="2D_paper_dsb2018")
    assert bare.model == "2D_paper_dsb2018"
    default = SB._load_backend("stardist")
    assert default.model == "2D_versatile_fluo"


def test_load_backend_refuses_what_it_cannot_run(tmp_path):
    with pytest.raises(ImportError, match="StarDist is not installed"):
        SB._load_backend("stardist", model_name="stardist:")
    _finish(tmp_path, "stardist")
    with pytest.raises(FileNotFoundError):
        SB._load_backend("stardist",
                         model_name=f"stardist:{tmp_path / 'gone'}")
    with pytest.raises(ValueError, match="single 2-D planes"):
        SB._load_backend("stardist", model_name="stardist:", z_plan=object())


def test_parallel_preparation_refuses_a_prefixed_object():
    from spacr._mask_workers import _prepare_mask_model

    with pytest.raises(ValueError, match="requires the Cellpose backend"):
        _prepare_mask_model({"nucleus_model_name": "stardist:"}, "nucleus")


def _no_cellpose(*args, **kwargs):
    raise AssertionError("Cellpose-SAM was built for a prefixed object")


def test_mask_generation_sends_the_object_through_the_sam_call(
        tmp_path, monkeypatch):
    """The object is loaded by ``_load_backend`` and given the very
    ``eval`` call, and output handling, a Cellpose-SAM model gets; the
    masks are saved as a Cellpose-SAM run saves them."""
    seen, calls = {}, []

    def _eval(x, **kwargs):
        calls.append(dict(kwargs, shapes=[np.shape(i) for i in x]))
        masks = [_known_labels(np.shape(i)[:2]).astype(np.uint16) for i in x]
        flows = [[None, None, np.zeros(np.shape(i)[:2], np.float32), None]
                 for i in x]
        return masks, flows, None

    def _load(name, **kwargs):
        seen.update(kwargs, name=name)
        return types.SimpleNamespace(eval=_eval)

    monkeypatch.setattr(SB, "_load_backend", _load)
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (2, 24, 24, 2))
    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, cell_model_name="stardist:"), "cell")
    assert seen["name"] == "stardist"
    assert seen["model_name"] == "stardist:"
    assert seen["object_type"] == "cell"
    [call] = calls
    assert call["normalize"] is False and call["channel_axis"] == -1
    assert call["batch_size"] == 2
    masks, rows = _artifacts(src)
    assert sorted(masks) == ["plate1_A01_1.npy", "plate1_A01_2.npy"]
    for name in masks:
        saved = np.load(src / "cell_mask_stack" / name)
        assert saved.max() >= 2


def test_prefixed_masks_makes_v1s_call_and_returns_the_probability():
    calls = []

    def _eval(x, **kwargs):
        calls.append(dict(kwargs, shapes=[np.shape(i) for i in x]))
        return ([_known_labels(np.shape(i)[:2]) for i in x],
                [[None, None, np.full(np.shape(i)[:2], 0.25, np.float32),
                  None] for i in x], None)

    model = types.SimpleNamespace(eval=_eval)
    settings = {"nucleus_diameter": None, "nucleus_flow_threshold": 0.4,
                "nucleus_cellprob_threshold": 0.0}
    masks, flows, probability = O._prefixed_masks(
        model, [np.zeros((20, 24), np.float32)], settings, "nucleus",
        min_size=5, default_diameter=17, probabilities=True)
    [call] = calls
    assert call["shapes"] == [(20, 24, 1)]
    assert call["min_size"] == 5 and call["diameter"] is None
    assert call["resample"] is True and call["normalize"] is False
    assert probability[0].shape == (20, 24)
    assert float(probability[0].max()) == 0.25


# ===========================================================================
# 4. The worker adapters, with each package faked
# ===========================================================================

class _FakeStarDist2D:
    """``stardist.models.StarDist2D`` as far as the adapter can see it."""

    built = []

    def __init__(self, config=None, name=None, basedir=None):
        self.where = (config, name, basedir)
        self.calls = []
        type(self).built.append(self)

    @classmethod
    def from_pretrained(cls, key):
        model = cls(name=key)
        model.pretrained = key
        return model

    def _guess_n_tiles(self, image):
        return (1, 1)

    def predict_instances(self, image, prob_thresh=None, n_tiles=None,
                          scale=None, show_tile_progress=True,
                          return_predict=False, verbose=True):
        self.calls.append(dict(prob_thresh=prob_thresh, n_tiles=n_tiles,
                               scale=scale,
                               lo=float(np.min(image)),
                               hi=float(np.max(image))))
        h, w = image.shape
        labels = _known_labels((h, w))
        probability = np.full((h // 2, w // 2), 0.75, np.float32)
        return (labels, {}), (probability, None)


def _fake_stardist(monkeypatch):
    _FakeStarDist2D.built = []
    csbdeep = types.ModuleType("csbdeep")
    utils = types.ModuleType("csbdeep.utils")

    def normalize(x, pmin, pmax, axis=None):
        lo, hi = np.percentile(x, (pmin, pmax))
        return (x - lo) / max(hi - lo, 1e-20)

    utils.normalize = normalize
    csbdeep.utils = utils
    monkeypatch.setitem(sys.modules, "csbdeep", csbdeep)
    monkeypatch.setitem(sys.modules, "csbdeep.utils", utils)
    return types.SimpleNamespace(StarDist2D=_FakeStarDist2D)


def test_stardist_loads_its_own_model_or_a_folder(tmp_path, monkeypatch):
    models = _fake_stardist(monkeypatch)
    adapter = SB._StarDistAdapter("2D_versatile_he", "cpu",
                                  models_module=models)
    assert adapter._model.pretrained == "2D_versatile_he"
    folder = tmp_path / "my_stardist"
    folder.mkdir()
    own = SB._StarDistAdapter(str(folder), "cpu", models_module=models)
    assert own._model.where == (None, "my_stardist", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no StarDist model"):
        SB._StarDistAdapter(str(tmp_path / "gone"), "cpu",
                            models_module=models)


def test_stardist_answers_in_cellpose_sams_shapes(monkeypatch):
    adapter = SB._StarDistAdapter("2D_versatile_fluo", "cpu",
                                  models_module=_fake_stardist(monkeypatch))
    rng = np.random.default_rng(0)
    images = [rng.random((24, 30, 2)).astype(np.float32) * 900
              for _ in range(2)]
    masks, flows, styles = adapter.eval(
        images, channel_axis=-1, normalize=False, diameter=17.0,
        min_size=2, batch_size=2, resample=True, flow_threshold=0.4,
        cellprob_threshold=0.0, progress=True)
    assert styles is None
    assert [m.shape for m in masks] == [(24, 30), (24, 30)]
    assert all(m.dtype == np.uint16 for m in masks)
    assert all(int(m.max()) == 2 for m in masks), "the 1-pixel object went"
    for entry in flows:
        assert len(entry) == 4 and entry[0] is None and entry[1] is None
        assert entry[2].shape == (24, 30) and entry[2].dtype == np.float32
    call = _FakeStarDist2D.built[0].calls[0]
    assert call["prob_thresh"] is None, "0 keeps StarDist's tuned threshold"
    assert call["n_tiles"] == (1, 1)
    assert call["scale"] == pytest.approx(SB._STARDIST_DIAMETER / 17.0)
    assert call["lo"] < 0.1 and call["hi"] > 0.9, "percentile-normalised"
    assert adapter.ignored == {"flow_threshold", "resample"}
    assert any("normalize=False" in t for t in adapter.translated)


def test_stardist_turns_a_cellpose_logit_into_its_probability(monkeypatch):
    adapter = SB._StarDistAdapter("2D_versatile_fluo", "cpu",
                                  models_module=_fake_stardist(monkeypatch))
    adapter.eval([np.ones((16, 16), np.float32)], cellprob_threshold=1.0)
    call = _FakeStarDist2D.built[0].calls[0]
    assert call["scale"] is None, "a blank diameter keeps the plane's scale"
    assert call["prob_thresh"] == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert any("prob_thresh=0.731" in t for t in adapter.translated)


class _InProcessWorker:
    """A worker that answers through the worker's own request handler."""

    def __init__(self, name, adapter):
        self.name = name
        self.adapter = adapter

    def request(self, op, should_cancel=None, **body):
        request = dict(body, op=op, protocol=SB._PROTOCOL, id=1)
        key = (body["model"], "cpu", json.dumps(body["options"],
                                                sort_keys=True))
        reply = SB._handle(self.name, request, {key: self.adapter})
        assert reply["ok"], reply
        return reply


def test_stardist_crosses_the_npy_hand_off_into_parse_cellpose4_output(
        tmp_path, monkeypatch):
    _finish(tmp_path, "stardist")
    adapter = SB._StarDistAdapter("2D_versatile_fluo", "cpu",
                                  models_module=_fake_stardist(monkeypatch))
    worker = _InProcessWorker("stardist", adapter)
    backend = SB._RemoteBackend("stardist", model="2D_versatile_fluo",
                                device="cpu",
                                worker_for=lambda name, env: worker)
    images = [np.ones((24, 30, 1), np.float32) for _ in range(2)]
    output = backend.eval(x=images, batch_size=2, normalize=False,
                          channel_axis=-1, min_size=0, progress=True,
                          diameter=None, flow_threshold=0.4,
                          cellprob_threshold=0.0, resample=True)
    masks, flows0, flows1, flows2, flows3 = parse_cellpose4_output(output)
    for mask in masks:
        np.testing.assert_array_equal(mask, _known_labels((24, 30)))
    assert flows0 == [None, None] and flows1 == [None, None]
    assert [f.shape for f in flows2] == [(24, 30), (24, 30)]
    assert flows3 == [None, None]


def test_the_worker_builds_the_adapter_on_the_default_model(monkeypatch):
    built = []
    monkeypatch.setitem(SB._PREFIXED_ADAPTERS, "stardist",
                        lambda model, device, **options: built.append(
                            (model, device, options)) or "adapter")
    assert SB._worker_adapter("stardist", "", "cpu", {}) == "adapter"
    assert built == [("2D_versatile_fluo", "cpu", {})]


# ===========================================================================
# 5. The zoo rows
# ===========================================================================

def test_each_model_is_a_zoo_row_that_needs_its_backend(tmp_path):
    rows = {e.key: e for e in zoo._prefixed_model_entries()}
    for name in PREFIXED:
        spec = SB._SPECS[name]
        for model in spec.models:
            row = rows[f"{name}_{model}"]
            assert (row.kind, row.source, row.name, row.path) == (
                name, "stock", model, "")
            assert row.uri == f"backend:{name}"
            assert row.licence == spec.licence
            assert "needs the" in row.notes[0]
            assert zoo._backend_for(row) == name
            assert zoo.source_of(row) == "spaCR"
    _finish(tmp_path, "stardist")
    ready = {e.key: e for e in zoo._prefixed_model_entries()}
    assert ready["stardist_2D_versatile_fluo"].path == "2D_versatile_fluo"
    assert ready["stardist_2D_versatile_fluo"].notes == ()
    keys = {e.key for e in zoo.catalogue(remote=False)}
    assert "stardist_2D_versatile_fluo" in keys


def test_mask_generation_fields_ask_the_zoo_for_every_prefixed_kind():
    assert zoo._mask_model_kinds() == (
        "cellpose", "cellpose3", "cellpose_dino") + PREFIXED


# ===========================================================================
# InstanSeg (item 552)
# ===========================================================================

def test_instanseg_pins_its_package_and_names_its_models():
    spec = SB._SPECS["instanseg"]
    assert spec.requirements == ("instanseg-torch==0.1.1",)
    assert spec.torch == ("torch",)
    assert spec.models == ("fluorescence_nuclei_and_cells",
                           "brightfield_nuclei")
    assert spec.licence == "Apache-2.0"
    assert "instanseg_models_v0.1.1" in spec.licence_note


def test_instanseg_keeps_its_downloads_inside_its_environment(tmp_path):
    env = str(tmp_path / "env")
    assert SB._worker_env("instanseg", env)["INSTANSEG_BIOIMAGEIO_PATH"] == (
        os.path.join(env, "instanseg_models"))


@pytest.mark.parametrize("value, object_type, target", [
    ("instanseg:", "nucleus", "nuclei"),
    ("instanseg:", "cell", "cells"),
    ("instanseg:", "pathogen", "cells"),
    ("instanseg:fluorescence_nuclei_and_cells#nuclei", "cell", "nuclei"),
    ("instanseg:fluorescence_nuclei_and_cells#cells", "nucleus", "cells"),
])
def test_instanseg_keeps_the_objects_output(tmp_path, value, object_type,
                                            target):
    _finish(tmp_path, "instanseg")
    model = SB._load_backend("cellpose", model_name=value,
                             object_type=object_type)
    assert model.model == "fluorescence_nuclei_and_cells"
    assert model.options == {"target": target}


class _FakeInstanSeg:
    """``instanseg.InstanSeg`` as far as the adapter can see it."""

    built = []

    def __init__(self, model_type, device=None, verbosity=1):
        self.model_type, self.device = model_type, device
        self.instanseg = types.SimpleNamespace(pixel_size=0.5)
        self.calls = []
        type(self).built.append(self)

    def _get_eval_function_to_use(self, num_pixels):
        return "small" if num_pixels < 1000 else "medium"

    def _answer(self, image, how, **kwargs):
        self.calls.append(dict(kwargs, how=how, shape=np.shape(image)))
        h, w = np.shape(image)[-2:]
        return np.asarray(_known_labels((h, w)))[None, None]

    def eval_small_image(self, image, **kwargs):
        return self._answer(image, "small", **kwargs)

    def eval_medium_image(self, image, **kwargs):
        return self._answer(image, "medium", **kwargs)


def test_instanseg_answers_in_cellpose_sams_shapes(tmp_path):
    _FakeInstanSeg.built = []
    adapter = SB._InstanSegAdapter("fluorescence_nuclei_and_cells", "cpu",
                                   target="nuclei",
                                   instanseg_class=_FakeInstanSeg)
    assert _FakeInstanSeg.built[0].model_type == (
        "fluorescence_nuclei_and_cells")
    images = [np.ones((24, 30, 2), np.float32), np.ones((40, 40), np.float32)]
    masks, flows, styles = adapter.eval(
        images, channel_axis=-1, normalize=False, diameter=17.0,
        min_size=2, flow_threshold=0.4, cellprob_threshold=0.0,
        resample=True, batch_size=2)
    assert styles is None
    assert [m.shape for m in masks] == [(24, 30), (40, 40)]
    assert all(m.dtype == np.uint16 and int(m.max()) == 2 for m in masks)
    assert all(entry == [None, None, None, None] for entry in flows)
    calls = _FakeInstanSeg.built[0].calls
    assert [c["how"] for c in calls] == ["small", "medium"]
    assert all(c["target"] == "nuclei" and c["normalise"] is True
               for c in calls)
    assert calls[0]["pixel_size"] == pytest.approx(
        0.5 * SB._INSTANSEG_DIAMETER / 17.0), "the diameter, as a pixel size"
    assert calls[0]["shape"] == (1, 24, 30)
    assert calls[1]["tile_size"] == 512
    assert adapter.ignored == {"flow_threshold", "cellprob_threshold",
                               "resample"}
    adapter.eval([np.ones((8, 8), np.float32)], diameter=None)
    assert _FakeInstanSeg.built[0].calls[-1]["pixel_size"] is None
    assert any("normalize=False" in t for t in adapter.translated)


def test_instanseg_loads_a_torchscript_file_and_refuses_a_missing_one(
        tmp_path, monkeypatch):
    loaded = []
    fake_torch = types.SimpleNamespace(
        jit=types.SimpleNamespace(load=lambda path, map_location=None:
                                  loaded.append(path) or "network"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    folder = tmp_path / "my_instanseg"
    folder.mkdir()
    (folder / "instanseg.pt").write_bytes(b"ts")
    _FakeInstanSeg.built = []
    adapter = SB._InstanSegAdapter(str(folder), "cpu",
                                   instanseg_class=_FakeInstanSeg)
    assert loaded == [str(folder / "instanseg.pt")]
    assert _FakeInstanSeg.built[0].model_type == "network"
    assert adapter.target == "cells"
    with pytest.raises(FileNotFoundError, match="no InstanSeg model"):
        SB._InstanSegAdapter(str(tmp_path / "gone"), "cpu",
                             instanseg_class=_FakeInstanSeg)


# ===========================================================================
# Omnipose (item 553)
# ===========================================================================

def test_omnipose_pins_its_package_and_says_its_licence_is_noncommercial():
    spec = SB._SPECS["omnipose"]
    assert spec.requirements == ("omnipose==1.1.4", "ncolor==1.5.3")
    assert spec.torch == ("torch", "torchvision")
    assert spec.models[0] == "bact_phase_omni"
    assert "NonCommercial" in spec.licence
    assert "NOT open source" in spec.licence_note
    assert "noncommercial purposes only" in spec.licence_note


def test_omnipose_keeps_its_downloads_inside_its_environment(tmp_path):
    env = str(tmp_path / "env")
    assert SB._worker_env("omnipose", env)["CELLPOSE_LOCAL_MODELS_PATH"] == (
        os.path.join(env, "models"))


class _FakeOmniModel:
    """``cellpose_omni.models.CellposeModel`` as far as the adapter sees it."""

    built = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        type(self).built.append(self)

    def eval(self, x, batch_size=8, indices=None, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, do_3D=False,
             anisotropy=None, net_avg=True, augment=False, tile=False,
             tile_overlap=0.1, bsize=224, num_workers=8, loader_batch_size=1,
             resample=True, interp=True, cluster=False, hdbscan=False,
             suppress=None, boundary_seg=False, affinity_seg=False,
             despur=False, flow_threshold=0.4, mask_threshold=0.0,
             diam_threshold=12.0, niter=None, cellprob_threshold=None,
             dist_threshold=None, flow_factor=5.0, compute_masks=True,
             min_size=15, max_size=None, stitch_threshold=0.0, progress=None,
             show_progress=True, omni=False, calc_trace=False, verbose=False,
             transparency=False, loop_run=False, model_loaded=False,
             hysteresis=True):
        kwargs = eval_arguments(locals())
        check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
        self.calls.append(dict(kwargs, shape=np.shape(x)))
        h, w = np.shape(x)[:2]
        return (_known_labels((h, w)),
                [np.full((h, w, 3), 5, np.uint8),
                 np.ones((2, h, w), np.float32),
                 np.full((h, w), -1.5, np.float32),
                 np.zeros((h, w), np.float32)], None)


def _fake_omni():
    _FakeOmniModel.built = []
    return types.SimpleNamespace(CellposeModel=_FakeOmniModel)


def test_omnipose_answers_in_cellpose_sams_shapes():
    adapter = SB._OmniposeAdapter("bact_fluor_omni", "cpu",
                                  models_module=_fake_omni())
    built = _FakeOmniModel.built[0]
    assert built.kwargs["model_type"] == "bact_fluor_omni"
    assert built.kwargs["gpu"] is False
    masks, flows, styles = adapter.eval(
        [np.ones((24, 30, 2), np.float32)], channel_axis=-1,
        normalize=False, diameter=17.0, min_size=2, flow_threshold=0.6,
        cellprob_threshold=-1.0, resample=False, batch_size=1)
    assert styles is None
    assert masks[0].shape == (24, 30) and int(masks[0].max()) == 2
    rgb, d_p, distance, last = flows[0]
    assert rgb.shape == (24, 30, 3) and d_p.shape == (2, 24, 30)
    assert distance.shape == (24, 30) and last is None
    [call] = built.calls
    assert call["shape"] == (24, 30), "the object's own plane"
    assert call["channels"] == [0, 0] and call["omni"] is True
    assert call["rescale"] is None and call["normalize"] is True
    assert (call["flow_threshold"], call["mask_threshold"]) == (0.6, -1.0)
    assert call["resample"] is False
    assert adapter.ignored == {"diameter"}
    assert any("normalize=False" in t for t in adapter.translated)


def test_omnipose_reads_a_checkpoints_shape_and_refuses_a_missing_file(
        tmp_path, monkeypatch):
    path = tmp_path / "my_omni_model"
    path.write_bytes(b"weights")
    monkeypatch.setattr(SB, "_omnipose_shape", lambda p: (1, 3))
    SB._OmniposeAdapter(str(path), "cpu", models_module=_fake_omni())
    kwargs = _FakeOmniModel.built[0].kwargs
    assert kwargs["pretrained_model"] == str(path)
    assert (kwargs["nchan"], kwargs["nclasses"], kwargs["omni"]) == (1, 3, True)
    with pytest.raises(FileNotFoundError, match="no Omnipose model"):
        SB._OmniposeAdapter(str(tmp_path / "gone"), "cpu",
                            models_module=_fake_omni())


# ===========================================================================
# Spotiflow (item 554)
# ===========================================================================

def test_spotiflow_pins_its_package_and_names_its_2d_models():
    spec = SB._SPECS["spotiflow"]
    assert spec.requirements == ("spotiflow==0.6.5",)
    assert spec.torch == ("torch", "torchvision")
    assert spec.models == ("general", "hybiss", "synth_complex", "fluo_live")
    assert spec.default_model == "general" and spec.prefix == "spotiflow:"
    assert spec.licence == "BSD-3-Clause" and spec.alpha
    assert "weigertlab/spotiflow-models" in spec.licence_note


def test_spotiflow_keeps_its_downloads_inside_its_environment(tmp_path):
    env = str(tmp_path / "env")
    assert SB._worker_env("spotiflow", env)["SPOTIFLOW_CACHE_DIR"] == (
        os.path.join(env, "models"))


class _FakeSpotiflow:
    """``spotiflow.model.Spotiflow`` as far as the adapter sees it."""

    built = []
    points = [[5.2, 6.4], [15.0, 20.0], [5.0, 7.0], [-3.0, 2.0]]

    def __init__(self, how, where, **kwargs):
        self.how, self.where, self.kwargs = how, where, kwargs
        self.calls = []
        type(self).built.append(self)

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls("pretrained", name, **kwargs)

    @classmethod
    def from_folder(cls, folder, **kwargs):
        return cls("folder", folder, **kwargs)

    def predict(self, image, **kwargs):
        self.calls.append(dict(kwargs, shape=np.shape(image)))
        heatmap = np.full(np.shape(image), 0.25, np.float32)
        return (np.asarray(self.points, float),
                types.SimpleNamespace(prob=np.full(len(self.points), 0.9),
                                      heatmap=heatmap))


def _fake_spotiflow():
    _FakeSpotiflow.built = []
    return _FakeSpotiflow


def test_spotiflow_answers_in_cellpose_sams_shapes():
    adapter = SB._SpotiflowAdapter("hybiss", "cpu",
                                   spotiflow_class=_fake_spotiflow())
    built = _FakeSpotiflow.built[0]
    assert (built.how, built.where) == ("pretrained", "hybiss")
    assert built.kwargs["map_location"] == "cpu"
    masks, flows, styles = adapter.eval(
        [np.ones((24, 30, 2), np.float32)], channel_axis=-1,
        normalize=False, min_size=0, flow_threshold=0.6,
        cellprob_threshold=0.0, resample=False, batch_size=1)
    assert styles is None
    labels = masks[0]
    assert labels.shape == (24, 30)
    assert int(labels.max()) == 3, "the spot outside the field is dropped"
    assert labels[5, 6] == 1 and labels[15, 20] == 2 and labels[5, 7] == 3
    assert (labels == 2).sum() == 13, "a disc of radius 2"
    rgb, d_p, heatmap, last = flows[0]
    assert rgb is None and d_p is None and last is None
    assert heatmap.shape == (24, 30)
    [call] = built.calls
    assert call["shape"] == (24, 30), "the object's own plane"
    assert call["prob_thresh"] is None, "0 keeps the model's own threshold"
    assert call["normalizer"] == "auto"
    assert adapter.ignored == {"flow_threshold", "resample"}
    assert any("normalize=False" in t for t in adapter.translated)


def test_spotiflow_turns_a_logit_and_a_diameter_into_its_own_terms():
    adapter = SB._SpotiflowAdapter("general", "cpu",
                                   spotiflow_class=_fake_spotiflow())
    masks, _flows, _ = adapter.eval(
        [np.ones((24, 30), np.float32)], cellprob_threshold=2.0,
        diameter=8.0)
    [call] = _FakeSpotiflow.built[0].calls
    assert call["prob_thresh"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert (masks[0] == 2).sum() == 49, "a disc of radius 4"
    assert any("radius" in t for t in adapter.translated)


def test_spotiflow_loads_a_folder_and_refuses_a_missing_one(tmp_path):
    SB._SpotiflowAdapter(str(tmp_path), "cpu",
                         spotiflow_class=_fake_spotiflow())
    assert (_FakeSpotiflow.built[0].how,
            _FakeSpotiflow.built[0].where) == ("folder", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no Spotiflow model"):
        SB._SpotiflowAdapter(str(tmp_path / "gone"), "cpu",
                             spotiflow_class=_fake_spotiflow())


def test_spot_labels_give_every_spot_inside_the_field_one_object():
    labels = SB._spot_labels([[0, 0], [0, 1], [9, 9], [20, 20]], (10, 10), 1)
    assert labels.dtype == np.uint16
    assert sorted(np.unique(labels)) == [0, 1, 2, 3]
    assert labels[0, 1] == 2, "a touching spot keeps its own pixel"
    assert (labels == 3).sum() == 3, "a disc cut by the border"
    assert SB._spot_labels(np.zeros((0, 2)), (4, 4), 2).max() == 0


def test_spotiflow_refuses_a_threshold_outside_0_to_1():
    network = _fake_spotiflow()("pretrained", "general")
    with pytest.raises(ValueError, match="between 0 and 1"):
        SB._spotiflow_predict(network, np.ones((4, 4)), 1.5)
    with pytest.raises(ValueError, match="finite"):
        SB._spotiflow_predict(network, np.full((4, 4), np.nan))
