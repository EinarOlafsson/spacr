"""Item 525: Cellpose-DINO models run in an environment of their own.

The maintainer, 2026-09-25: "dinov3 ... should get its own env". What is
pinned here:

1. the ``cellpose_dino`` backend spec -- Cellpose 4.2.1.1 and DINOv3 from
   one pinned GitHub commit, the DINOv3 licence said before installing, and
   no ``segmentation_backend`` value of its own;
2. the install command, run with a fake runner: its own venv, the
   installer's torch index, the two pins, the worker's self-test;
3. the routing of ``cellpose_dino:<path>`` from Mask generation, Make Masks
   and the parallel preparation that must refuse it;
4. the worker's output: masks, flows and cell probability in the shapes the
   Cellpose-SAM path already reads, through the real ``.npy`` hand-off;
5. the bioimage.io DINO rows: downloadable ``cellpose_dino`` rows that name
   their backend, no longer refused.
"""
from __future__ import annotations

import inspect
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


def _finish(tmp_path, name="cellpose_dino"):
    env = tmp_path / "backends" / name
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": name})
    return env


def _checkpoint(tmp_path, name="cellposedino_vit_b"):
    path = tmp_path / name
    path.write_bytes(b"weights")
    return path


# ===========================================================================
# 1. The spec
# ===========================================================================

def test_the_spec_pins_cellpose_4_and_one_dinov3_commit():
    spec = SB._spec("cellpose_dino")
    assert spec is SB._SPECS["cellpose_dino"]
    assert spec.requirements[0] == "cellpose==4.2.1.1"
    assert spec.requirements[1] == (
        "dinov3 @ https://github.com/facebookresearch/dinov3/archive/"
        "6876159a11b4df116f30f667f8c9888617df0751.zip")
    assert SB._requirement_name(spec.requirements[1]) == "dinov3"
    assert "dinov3.hub.backbones" in spec.probe
    assert "cellpose.vit" in spec.probe
    assert spec.torch == ("torch", "torchvision")
    assert spec.python[0] >= (3, 11), "dinov3 declares python_requires>=3.11"
    assert spec.segments


def test_the_dinov3_licence_is_said_before_anything_is_installed():
    spec = SB._SPECS["cellpose_dino"]
    assert "DINOv3 License" in spec.licence
    note = spec.licence_note
    assert "NOT open source" in note
    assert "agreeing to it" in note
    assert "acknowledges DINOv3" in note
    assert "spaCR ships none of it" in note


def test_it_is_chosen_by_a_model_setting_not_by_segmentation_backend():
    assert "cellpose_dino" not in SB._BACKEND_NAMES
    with pytest.raises(ValueError):
        SB._backend_name("cellpose_dino")
    assert "cellpose_dino" not in zoo.INSTALLABLE_BACKENDS
    assert list(zoo.INSTALLABLE_BACKENDS) == ["cellpose3", "dinocell",
                                              "samcell"]


def test_the_zoo_lists_the_backend_with_its_state_and_licence(tmp_path):
    rows = {e.key: e for e in zoo.installable_backend_entries()}
    row = rows["cellpose_dino_v1"]
    assert (row.kind, row.source, row.uri) == (
        "backend", "installable", "backend:cellpose_dino")
    assert "DINOv3 License" in row.notes[1]
    env = _finish(tmp_path)
    ready = {e.key: e for e in zoo.installable_backend_entries()}
    assert ready["cellpose_dino_v1"].source == "installed"
    assert ready["cellpose_dino_v1"].path == str(env)


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
                "packages": {"cellpose": "4.2.1.1", "torch": "2.14.0"}})]
        return 0, ["ok"]
    return _run


def test_the_install_builds_its_own_environment_with_both_pins(tmp_path):
    record = []
    state = SB._install_backend(
        "cellpose_dino", runner=_fake_runner(record),
        preflight=lambda spec, root: (sys.executable,),
        torch_index="https://download.pytorch.org/whl/cpu")
    assert state.ready and not state.in_process
    env = str(tmp_path / "backends" / "cellpose_dino")
    argvs = [argv for argv, _env in record]
    assert argvs[0] == (sys.executable, "-m", "venv", env)
    python = SB._env_python(env)
    assert all(argv[0] == python for argv in argvs[1:])
    torch_step = argvs[2]
    assert torch_step[-4:] == ("torch", "torchvision", "--index-url",
                               "https://download.pytorch.org/whl/cpu")
    assert argvs[3][-2:] == SB._SPECS["cellpose_dino"].requirements
    assert argvs[4][-2:] == ("--selftest", "cellpose_dino")
    assert state.record["requirements"] == list(
        SB._SPECS["cellpose_dino"].requirements)
    assert state.record["torch_index"] == "https://download.pytorch.org/whl/cpu"
    assert SB._stale_requirements("cellpose_dino", state.record) == []
    models = os.path.join(env, "models")
    assert all(e["CELLPOSE_LOCAL_MODELS_PATH"] == models
               for _argv, e in record[1:])


def test_its_worker_keeps_cellpose_downloads_inside_its_environment(tmp_path):
    env = str(tmp_path / "env")
    assert SB._worker_env("cellpose_dino", env)[
        "CELLPOSE_LOCAL_MODELS_PATH"] == os.path.join(env, "models")


# ===========================================================================
# 3. Routing
# ===========================================================================

@pytest.mark.parametrize("value, expected", [
    ("cellpose_dino:/m/cpdino-vitb", "/m/cpdino-vitb"),
    ("  CELLPOSE_DINO: /m/x ", "/m/x"),
    ("cellpose_dino:", ""),
    ("cellpose3:cyto3", None),
    ("/m/cpdino-vitb", None),
    ("cpsam", None),
    (None, None),
])
def test_a_cellpose_dino_choice_is_read_from_its_prefix(value, expected):
    assert SB._cellpose_dino_choice(value) == expected


def test_the_value_round_trips_and_names_an_existing_checkpoint(tmp_path):
    path = _checkpoint(tmp_path)
    value = SB._cellpose_dino_value(str(path))
    assert value == f"cellpose_dino:{path}"
    assert SB._cellpose_dino_value(value) == value
    assert SB._cellpose_dino_model(value) == str(path)
    with pytest.raises(FileNotFoundError, match="not there"):
        SB._cellpose_dino_model(f"cellpose_dino:{tmp_path / 'gone'}")
    with pytest.raises(FileNotFoundError):
        SB._cellpose_dino_model("cellpose_dino:")


def test_a_run_knows_when_an_object_chose_cellpose_dino():
    assert SB._cellpose_dino_is_chosen(
        {"nucleus_model_name": "cellpose_dino:/m/w"})
    assert SB._cellpose_dino_is_chosen({"pathogen_model": "cellpose_dino:/m"})
    assert not SB._cellpose_dino_is_chosen({"cell_model_name": "cpsam"})
    assert not SB._cellpose_dino_is_chosen(
        {"cell_model_name": "cellpose3:cyto3"})


def test_load_backend_builds_the_remote_model_on_the_checkpoint(tmp_path,
                                                                capsys):
    env = _finish(tmp_path)
    path = _checkpoint(tmp_path)
    model = SB._load_backend("cellpose", model_name=f"cellpose_dino:{path}",
                             object_type="cell")
    assert isinstance(model, SB._RemoteBackend)
    assert (model.name, model.model, model.env) == (
        "cellpose_dino", str(path), str(env))
    assert "Segmentation backend: cellpose_dino" in capsys.readouterr().out
    again = SB._load_backend("cellpose_dino", model_name=str(path))
    assert again.model == str(path)


def test_load_backend_refuses_what_it_cannot_run(tmp_path):
    path = _checkpoint(tmp_path)
    with pytest.raises(ImportError, match="Cellpose-DINO is not installed"):
        SB._load_backend("cellpose_dino", model_name=f"cellpose_dino:{path}")
    _finish(tmp_path)
    with pytest.raises(FileNotFoundError):
        SB._load_backend("cellpose_dino",
                         model_name=f"cellpose_dino:{tmp_path / 'gone'}")
    with pytest.raises(ValueError, match="single 2-D planes"):
        SB._load_backend("cellpose_dino", model_name=str(path),
                         z_plan=object())


def test_parallel_preparation_refuses_a_cellpose_dino_object():
    from spacr._mask_workers import _prepare_mask_model

    with pytest.raises(ValueError, match="requires the Cellpose backend"):
        _prepare_mask_model({"cell_model_name": "cellpose_dino:/m/w"}, "cell")


def _no_cellpose(*args, **kwargs):
    raise AssertionError("Cellpose-SAM was built for a Cellpose-DINO object")


def _known_labels(shape):
    out = np.zeros(shape, np.int32)
    out[2:9, 3:10] = 1
    out[14:20, 12:22] = 2
    return out


def test_mask_generation_sends_the_object_through_the_sam_call(
        tmp_path, monkeypatch):
    """The DINO object is loaded by ``_load_backend`` and then given the
    very ``eval`` call, and the very output handling, a Cellpose-SAM model
    gets; the masks are saved as a Cellpose-SAM run saves them."""
    seen, calls = {}, []

    def _eval(x, **kwargs):
        calls.append(dict(kwargs, shapes=[np.shape(i) for i in x]))
        masks = [_known_labels(np.shape(i)[:2]).astype(np.uint16) for i in x]
        flows = [[np.zeros(np.shape(i)[:2] + (3,), np.uint8),
                  np.zeros((2,) + np.shape(i)[:2], np.float32),
                  np.zeros(np.shape(i)[:2], np.float32), None] for i in x]
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
        str(src), _base_settings(src, cell_model_name="cellpose_dino:/m/w"),
        "cell")
    assert seen["name"] == "cellpose_dino"
    assert seen["model_name"] == "cellpose_dino:/m/w"
    [call] = calls
    assert call["normalize"] is False and call["channel_axis"] == -1
    assert call["batch_size"] == 2 and "percentile" not in str(call)
    masks, rows = _artifacts(src)
    assert sorted(masks) == ["plate1_A01_1.npy", "plate1_A01_2.npy"]
    for name in masks:
        saved = np.load(src / "cell_mask_stack" / name)
        np.testing.assert_array_equal(saved, _known_labels((24, 24)))
    assert {count for _name, _kind, count in rows} == {2}


# ===========================================================================
# 4. The worker's output shapes, with a faked Cellpose 4
# ===========================================================================

class _FakeCellposeModel:
    """Cellpose 4's ``CellposeModel`` as far as the adapter can see it."""

    built = []

    def __init__(self, gpu=False, pretrained_model=None, device=None,
                 use_bfloat16=True):
        self.kwargs = dict(gpu=gpu, pretrained_model=pretrained_model,
                           device=device, use_bfloat16=use_bfloat16)
        self.backbone = ("sam_vitl" if "sam" in os.path.basename(
            str(pretrained_model)) else "dino_vitb")
        self.calls = []
        type(self).built.append(self)

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        self.calls.append(dict(batch_size=batch_size, resample=resample,
                               channel_axis=channel_axis, normalize=normalize,
                               diameter=diameter, min_size=min_size,
                               flow_threshold=flow_threshold,
                               cellprob_threshold=cellprob_threshold))
        masks, flows = [], []
        for image in x:
            h, w = image.shape[:2]
            masks.append(_known_labels((h, w)).astype(np.uint16))
            flows.append([np.full((h, w, 3), 7, np.uint8),
                          np.ones((2, h, w), np.float32),
                          np.full((h, w), 0.5, np.float32)])
        return masks, flows, np.zeros((len(x), 256), np.float32)


def _fake_models():
    _FakeCellposeModel.built = []
    return types.SimpleNamespace(CellposeModel=_FakeCellposeModel)


def test_the_adapter_builds_the_checkpoint_in_float32_on_a_cpu(tmp_path):
    path = _checkpoint(tmp_path)
    adapter = SB._CellposeDinoAdapter(str(path), "cpu",
                                      models_module=_fake_models())
    [built] = _FakeCellposeModel.built
    assert built.kwargs["pretrained_model"] == str(path)
    assert built.kwargs["gpu"] is False
    assert built.kwargs["use_bfloat16"] is False
    assert adapter.name == "cellpose_dino"


def test_the_adapter_refuses_a_missing_file_and_a_sam_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="cpsam_v2 in its place"):
        SB._CellposeDinoAdapter(str(tmp_path / "gone"), "cpu",
                                models_module=_fake_models())
    sam = _checkpoint(tmp_path, "cellpose_sam")
    with pytest.raises(ValueError, match="not a Cellpose-DINO one"):
        SB._CellposeDinoAdapter(str(sam), "cpu", models_module=_fake_models())


def test_the_adapter_returns_cellpose_sams_shapes_and_names_what_it_drops(
        tmp_path):
    adapter = SB._CellposeDinoAdapter(str(_checkpoint(tmp_path)), "cpu",
                                      models_module=_fake_models())
    images = [np.zeros((24, 30, 2), np.float32), np.zeros((24, 30, 2),
                                                          np.float32)]
    masks, flows, styles = adapter.eval(
        images, channel_axis=-1, normalize=False, diameter=44.0,
        min_size=15, batch_size=2, resample=True, flow_threshold=0.4,
        cellprob_threshold=0.0, net_avg=True)
    assert styles is None
    assert [m.shape for m in masks] == [(24, 30), (24, 30)]
    assert all(np.issubdtype(m.dtype, np.integer) for m in masks)
    for entry in flows:
        assert len(entry) == 4 and entry[3] is None
        assert entry[0].shape == (24, 30, 3)
        assert entry[1].shape == (2, 24, 30)
        assert entry[2].shape == (24, 30)
    assert adapter.ignored == {"net_avg"}, "Cellpose 3's, not Cellpose 4's"
    [call] = _FakeCellposeModel.built[0].calls
    assert call["normalize"] is False and call["diameter"] == 44.0


class _InProcessWorker:
    """A worker that answers through the worker's own request handler."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.requests = []

    def request(self, op, should_cancel=None, **body):
        self.requests.append((op, body))
        request = dict(body, op=op, protocol=SB._PROTOCOL, id=1)
        key = (body["model"], "cpu", json.dumps(body["options"],
                                                sort_keys=True))
        reply = SB._handle("cellpose_dino", request, {key: self.adapter})
        assert reply["ok"], reply
        return reply


def test_masks_flows_and_cellprob_cross_the_npy_hand_off_unchanged(
        tmp_path, monkeypatch):
    """Through ``_RemoteBackend`` and the worker's ``segment`` op, the
    output reads exactly as a Cellpose-SAM model's does."""
    _finish(tmp_path)
    path = _checkpoint(tmp_path)
    adapter = SB._CellposeDinoAdapter(str(path), "cpu",
                                      models_module=_fake_models())
    worker = _InProcessWorker(adapter)
    backend = SB._RemoteBackend("cellpose_dino", model=str(path),
                                device="cpu",
                                worker_for=lambda name, env: worker)
    images = [np.random.default_rng(0).random((24, 30, 2)).astype(np.float32)
              for _ in range(2)]
    output = backend.eval(x=images, batch_size=2, normalize=False,
                          channel_axis=-1, min_size=15, progress=True,
                          diameter=44.0, flow_threshold=0.4,
                          cellprob_threshold=0.0, resample=True)
    masks, flows0, flows1, flows2, flows3 = parse_cellpose4_output(output)
    sam = _FakeCellposeModel(pretrained_model="dino").eval(images)
    s_masks, s0, s1, s2, _s3 = parse_cellpose4_output(sam)
    for got, want in zip(masks, s_masks):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(flows0 + flows1 + flows2, s0 + s1 + s2):
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)
    assert flows3 == [None, None]
    [(op, body)] = worker.requests
    assert op == "segment" and body["model"] == str(path)
    assert body["params"]["normalize"] is False
    assert body["params"]["diameter"] == 44.0


def test_the_worker_builds_the_dino_adapter_for_its_backend(monkeypatch):
    built = []

    class _Adapter:
        def __init__(self, model, device):
            built.append((model, device))

    monkeypatch.setattr(SB, "_CellposeDinoAdapter", _Adapter)
    SB._worker_adapter("cellpose_dino", "/m/w", "cpu", {})
    assert built == [("/m/w", "cpu")]


def test_the_adapter_reads_the_real_cellpose_eval_signature():
    """What the adapter is handed must be what Cellpose 4's eval takes;
    checked against the Cellpose spaCR itself installs, 4.2.1.1."""
    cellpose = pytest.importorskip("cellpose.models")
    parameters = inspect.signature(cellpose.CellposeModel.eval).parameters
    for key in ("batch_size", "normalize", "channel_axis", "min_size",
                "diameter", "flow_threshold", "cellprob_threshold",
                "resample", "progress"):
        assert key in parameters
    init = inspect.signature(cellpose.CellposeModel.__init__).parameters
    assert {"gpu", "pretrained_model", "device", "use_bfloat16"} <= set(init)


# ===========================================================================
# 5. The bioimage.io rows
# ===========================================================================

_DINO_VITB = {
    "alias": "passionate-bug",
    "manifest": {
        "type": "model",
        "name": "CellposeDINO ViT-B 2D Microscopy Instance Segmenter",
        "license": "BSD-3-Clause", "tags": ["cellpose", "cellpose-dino"],
        "weights": {"pytorch_state_dict": {
            "source": "https://huggingface.co/mouseland/cellpose-sam/"
                      "resolve/main/cpdino-vitb",
            "sha256": "3ED4C06A",
            "architecture": {"callable": "CellposeInstanceLabelsModel",
                             "kwargs": {"bsize": 384}}}},
    },
}


def test_a_dino_row_downloads_and_names_its_backend():
    [row] = zoo._bioimageio_rows([_DINO_VITB], {})
    assert row.kind == "cellpose_dino"
    assert row.uri.endswith("/resolve/main/cpdino-vitb")
    assert row.sha256 == "3ed4c06a"
    assert zoo._bioimageio_cannot_run(row) == ""
    assert "cellpose_dino:<its path>" in row.notes[0]
    assert zoo._backend_for(row) == "cellpose_dino"
    assert zoo.source_of(row) == "bioimage.io"
    assert "cellpose_dino" in zoo.KINDS
