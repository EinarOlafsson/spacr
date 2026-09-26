"""Items 551-553: StarDist, InstanSeg and Omnipose, each in an environment
of its own and chosen by a model-setting prefix.

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

_base_settings = _wiring._base_settings
_write_npz = _wiring._write_npz
_artifacts = _wiring._artifacts

PREFIXED = ("stardist",)


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
        assert (row.kind, row.source, row.uri) == (
            "backend", "installable", f"backend:{name}")
        assert row.notes[1] == SB._SPECS[name].licence_note
    env = _finish(tmp_path, "stardist")
    ready = {e.key: e for e in zoo.installable_backend_entries()}
    assert ready["stardist_v1"].source == "installed"
    assert ready["stardist_v1"].path == str(env)


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
    assert zoo.mask_model_kinds() == (
        "cellpose", "cellpose3", "cellpose_dino") + PREFIXED
