"""Item 503: cyto, cyto2, cyto3 and nuclei are usable from Mask generation.

An object's model setting that reads ``cellpose3:<model or weights path>``
sends that object to the Cellpose 3 backend through ``_cellpose3_masks``,
which owns Cellpose 3's input shape, its legacy settings and its output
shape. Everything after that call is the Cellpose-SAM path's own code.

What is pinned here:

1. the ``cellpose3:`` spelling -- parsed, written and resolved to a model;
2. the legacy settings exist, have defaults, and reach Cellpose 3 as the
   keywords it takes, with a real diameter unless the size model is asked
   for;
3. the input shapes: ``[cyto, nucleus]`` as two planes, or one plane;
4. a Cellpose 3 object runs out of process and writes the same kind of
   arrays and database rows a Cellpose-SAM object does, while the other
   objects stay on Cellpose-SAM;
5. a Cellpose-SAM run writes exactly the bytes it wrote before this item
   (the digest below was taken on nightly c2def6baa, before any 503 edit);
6. the backend plumbing: ``augment`` and the normalization dict reach the
   worker, and a named model's keywords are checked against the inner
   ``CellposeModel.eval`` rather than the ``**kwargs`` wrapper.
"""
from __future__ import annotations

import hashlib
import types

import numpy as np
import pytest

import spacr._segmentation_backends as SB
import spacr.object as O
import tests.test_object_tstack_wiring as _wiring
from tests.conftest import MISSING_CHANNEL_AXIS

fake_model = _wiring.fake_model
force_cpu = _wiring.force_cpu
_artifacts = _wiring._artifacts
_base_settings = _wiring._base_settings
_write_npz = _wiring._write_npz

#: sha256 over what a Cellpose-SAM run of the fixture below leaves behind
#: -- every saved mask's bytes, the object_counts rows, and the arguments
#: CellposeModel was built and evaluated with -- measured on nightly
#: c2def6baa, before item 503 touched anything.
SAM_DIGEST_BEFORE_503 = {
    "cell": "0498fbe0354b0aa92f0d96ed02c9e0018fca71b85777d672ddce864e7a255a9a",
    "nucleus": "2fe19ca5a9972dfe0c1959e108faeb9846769f9b950432ffe222f9294624f5dd",
}


@pytest.fixture(autouse=True)
def _no_backend_environments(tmp_path, monkeypatch):
    """No test here may reach a real backend under ``~/.spacr/backends``."""
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))


def _known_labels(shape):
    out = np.zeros(shape, np.int32)
    out[2:9, 3:10] = 1
    out[14:20, 12:22] = 2
    return out


def _FakeCellpose3():
    """What ``_load_backend('cellpose3', ...)`` returns, recording the call.

    A namespace rather than a class with an ``eval`` method: it stands in
    for ``_RemoteBackend``, not for a Cellpose, so it is not one of the
    Cellpose doubles tests/test_cellpose_api_contract.py sweeps for.
    """
    calls = []

    def _eval(x, **kwargs):
        calls.append(dict(kwargs, shapes=[np.shape(i) for i in x]))
        masks = [_known_labels(np.shape(i)[:2]).astype(np.uint16) for i in x]
        flows = [[np.zeros(np.shape(i)[:2] + (3,)), None, None, None]
                 for i in x]
        return masks, flows, None

    return types.SimpleNamespace(calls=calls, eval=_eval)


def _no_cellpose(*args, **kwargs):
    raise AssertionError("Cellpose-SAM was built for a Cellpose 3 object")


class _InputLabellingCellpose:
    """A ``CellposeModel`` whose masks depend on the images it is given.

    The digest below has to move if anything before or after ``eval``
    moves: what the images are prepared as, what ``eval`` is asked for, and
    what is done with the masks. So the masks are the connected components
    of each image's first plane above its median, and every call is
    recorded with a hash of its images.
    """

    built = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        type(self).built.append(self)

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=256,
             compute_masks=True, progress=None):
        from skimage.measure import label

        bound = {k: v for k, v in locals().items()
                 if k not in ("self", "x", "label")}
        images = [np.asarray(i) for i in x]
        self.calls.append((
            [hashlib.sha256(np.ascontiguousarray(i)).hexdigest()
             for i in images],
            [(i.shape, str(i.dtype)) for i in images],
            sorted((k, repr(v)) for k, v in bound.items())))
        masks = []
        for image in images:
            plane = image[..., 0] if image.ndim == 3 else image
            masks.append(label(plane > np.median(plane)).astype(np.uint16))
        flows = [[np.zeros(m.shape + (3,), np.uint8),
                  np.zeros((2,) + m.shape, np.float32),
                  np.zeros(m.shape, np.float32)] for m in masks]
        return masks, flows, None


def _digest(src, object_type, model):
    masks, rows = _artifacts(src, object_type)
    blob = hashlib.sha256()
    for name, data in masks.items():
        blob.update(name.encode())
        blob.update(data)
    blob.update(repr(rows).encode())
    blob.update(repr(sorted((k, repr(v))
                            for k, v in model.init_kwargs.items())).encode())
    blob.update(repr(model.calls).encode())
    return blob.hexdigest()


# ===========================================================================
# 1. The spelling
# ===========================================================================

@pytest.mark.parametrize("value, expected", [
    ("cellpose3:cyto3", "cyto3"), ("cellpose3:cyto2", "cyto2"),
    ("Cellpose3: nuclei ", "nuclei"), ("cellpose3:/w/cp3.pth", "/w/cp3.pth"),
    ("cellpose3:", ""), ("cyto3", None), ("cpsam", None), (None, None),
    ("/models/cellpose3_toxo.pth", None),
])
def test_a_cellpose3_choice_is_read_from_its_prefix(value, expected):
    assert SB._cellpose3_choice(value) == expected


@pytest.mark.parametrize("model", ["cyto", "cyto2", "cyto3", "nuclei"])
def test_each_stock_model_round_trips_through_its_setting(model):
    value = SB._cellpose3_value(model)
    assert value == f"cellpose3:{model}"
    assert SB._cellpose3_value(value) == value
    assert SB._cellpose3_model(value, "cell") == model


def test_any_cellpose3_weights_file_is_a_model(tmp_path):
    weights = tmp_path / "my_cp3_model"
    weights.write_bytes(b"weights")
    value = SB._cellpose3_value(str(weights))
    assert SB._cellpose3_model(value, "pathogen") == str(weights)


def test_a_weights_path_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="not there"):
        SB._cellpose3_model(f"cellpose3:{tmp_path}/gone.pth", "cell")


def test_a_bare_prefix_means_the_objects_own_default():
    assert SB._cellpose3_model("cellpose3:", "nucleus") == "nuclei"
    assert SB._cellpose3_model("cellpose3:", "cell") == "cyto3"


@pytest.mark.parametrize("settings, chosen", [
    ({}, False),
    ({"segmentation_backend": "cellpose"}, False),
    ({"segmentation_backend": "cellpose3"}, True),
    ({"cell_model_name": "cpsam", "nucleus_model_name": "cyto3"}, False),
    ({"nucleus_model_name": "cellpose3:nuclei"}, True),
    ({"pathogen_model": "cellpose3:/w/cp3.pth"}, True),
    ({"organelle_model_name": "cellpose3:cyto2"}, True),
])
def test_a_run_chooses_cellpose3_by_backend_or_by_model(settings, chosen):
    assert SB._cellpose3_is_chosen(settings) is chosen


# ===========================================================================
# 2. The legacy settings
# ===========================================================================

LEGACY = {
    "cellpose3_add_nucleus_channel": True, "cellpose3_size_model": False,
    "cellpose3_resample": True, "cellpose3_augment": False,
    "cellpose3_percentile_low": 1.0, "cellpose3_percentile_high": 99.0,
}


def test_the_legacy_settings_have_their_defaults_and_a_category():
    from spacr.settings import (categories, expected_types,
                                set_default_settings_preprocess_generate_masks)

    settings = set_default_settings_preprocess_generate_masks({})
    for key, value in LEGACY.items():
        assert settings[key] == value
        assert key in expected_types
    assert set(categories["Cellpose 3"]) == set(LEGACY)


def test_a_blank_diameter_is_the_magnification_default_not_zero():
    """Item 507, plate1_A02_12: diameter 0 took 153 s on the CPU and matched
    7 of 33 vacuoles; diameter 44 took 11.9 s and matched 24."""
    keywords, use_nucleus = O._cellpose3_eval_settings(
        dict(LEGACY, cell_diameter=None, cell_flow_threshold=0.5,
             cell_cellprob_threshold=-1.0), "cell", 120.0)
    assert keywords == dict(
        diameter=120.0, flow_threshold=0.5, cellprob_threshold=-1.0,
        resample=True, augment=False,
        normalize={"normalize": True, "percentile": [1.0, 99.0]})
    assert use_nucleus is True


def test_the_objects_own_diameter_wins_and_the_size_model_means_zero():
    settings = dict(LEGACY, pathogen_diameter="44")
    assert O._cellpose3_eval_settings(
        settings, "pathogen", 20.0)[0]["diameter"] == 44.0
    settings["cellpose3_size_model"] = True
    assert O._cellpose3_eval_settings(
        settings, "pathogen", 20.0)[0]["diameter"] == 0.0


def test_the_other_legacy_settings_reach_cellpose3():
    keywords, use_nucleus = O._cellpose3_eval_settings(
        dict(LEGACY, cellpose3_resample=False, cellpose3_augment=True,
             cellpose3_percentile_low=2, cellpose3_percentile_high=99.8,
             cellpose3_add_nucleus_channel=False), "cell", 120.0)
    assert keywords["resample"] is False
    assert keywords["augment"] is True
    assert keywords["normalize"] == {"normalize": True,
                                     "percentile": [2.0, 99.8]}
    assert use_nucleus is False


@pytest.mark.parametrize("low, high", [(99, 1), (5, 5), (-1, 99), (1, 101)])
def test_percentiles_out_of_order_are_refused(low, high):
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        O._cellpose3_eval_settings(
            dict(LEGACY, cellpose3_percentile_low=low,
                 cellpose3_percentile_high=high), "cell", 120.0)


# ===========================================================================
# 3. Input and output shapes
# ===========================================================================

@pytest.mark.parametrize("planes, use_nucleus, sent", [
    (2, True, (16, 16, 2)), (2, False, (16, 16)), (1, True, (16, 16)),
])
def test_cellpose3_is_given_cyto_then_nucleus_or_one_plane(planes,
                                                           use_nucleus, sent):
    model = _FakeCellpose3()
    images = [np.random.default_rng(0).random((16, 16, planes),
                                              dtype=np.float32)]
    masks, flows = O._cellpose3_masks(
        model, images, dict(LEGACY, cellpose3_add_nucleus_channel=use_nucleus),
        "cell", min_size=5, default_diameter=30.0)
    assert model.calls[0]["shapes"] == [sent]
    assert model.calls[0]["channel_axis"] == -1
    assert model.calls[0]["min_size"] == 5
    assert len(masks) == 1 and masks[0].shape == (16, 16)
    assert masks[0].dtype == np.uint16
    assert len(flows) == 1


# ===========================================================================
# 4. Through the mask generator
# ===========================================================================

def _spy_backend(monkeypatch):
    seen = {}
    model = _FakeCellpose3()

    def _load(name, **kwargs):
        seen.update(kwargs, name=name)
        return model

    monkeypatch.setattr(SB, "_load_backend", _load)
    return seen, model


def test_a_cellpose3_object_runs_out_of_process_and_saves_its_masks(
        tmp_path, monkeypatch):
    seen, model = _spy_backend(monkeypatch)
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (2, 24, 24, 2))
    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, cell_model_name="cellpose3:cyto2"),
        "cell")

    assert seen["name"] == "cellpose3"
    assert seen["model_name"] == "cellpose3:cyto2"
    assert seen["z_plan"] is None and seen["t_plan"] is None
    [call] = model.calls
    assert call["shapes"] == [(24, 24, 2), (24, 24, 2)]
    assert call["diameter"] == 2 * 20 + 80
    assert call["normalize"] == {"normalize": True, "percentile": [1.0, 99.0]}
    assert call["augment"] is False and call["resample"] is True

    masks, rows = _artifacts(src)
    assert sorted(masks) == ["plate1_A01_1.npy", "plate1_A01_2.npy"]
    for name in masks:
        saved = np.load(src / "cell_mask_stack" / name)
        assert saved.dtype == np.uint16 and saved.shape == (24, 24)
        np.testing.assert_array_equal(saved, _known_labels((24, 24)))
    assert {count for _name, _kind, count in rows} == {2}


def test_segmentation_backend_cellpose3_takes_the_same_route(
        tmp_path, monkeypatch):
    seen, model = _spy_backend(monkeypatch)
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (1, 24, 24, 2))
    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, segmentation_backend="cellpose3",
                                 cell_model_name="cyto2"), "cell")
    assert seen["name"] == "cellpose3"
    assert model.calls[0]["normalize"]["percentile"] == [1.0, 99.0]


def test_the_other_objects_stay_on_cellpose_sam(tmp_path, fake_model,
                                                monkeypatch):
    seen, model = _spy_backend(monkeypatch)
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (1, 24, 24, 2))
    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, cell_model_name="cellpose3:cyto3"),
        "nucleus")
    assert not seen and not model.calls
    assert fake_model["model"] is not None
    assert len(fake_model["model"].eval_kwargs) == 1


# ===========================================================================
# 5. Cellpose-SAM writes what it wrote before
# ===========================================================================

@pytest.mark.parametrize("object_type", ["cell", "nucleus"])
def test_a_cellpose_sam_run_is_byte_identical_to_before_503(
        tmp_path, monkeypatch, object_type):
    def _no_backend(*args, **kwargs):
        raise AssertionError("a Cellpose-SAM run reached a backend loader")

    monkeypatch.setattr(SB, "_load_backend", _no_backend)
    monkeypatch.setattr(_InputLabellingCellpose, "built", [])
    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(
        CellposeModel=_InputLabellingCellpose))
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (3, 32, 32, 2), seed=503)
    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, cell_diameter=40, cell_min_area=4,
                                 nucleus_min_area=4), object_type)
    [model] = _InputLabellingCellpose.built
    assert len(model.calls) == 1
    digest = _digest(src, object_type, model)
    assert digest == SAM_DIGEST_BEFORE_503[object_type]


# ===========================================================================
# 6. The backend plumbing
# ===========================================================================

class _Worker:
    def __init__(self):
        self.requests = []

    def request(self, op, **kwargs):
        self.requests.append((op, kwargs))
        outputs = []
        for index, _path in enumerate(kwargs.get("inputs") or ()):
            mask = kwargs["outputs"] + f"/mask_{index}.npy"
            np.save(mask, np.zeros((8, 8), dtype=np.uint16))
            outputs.append({"mask": mask, "flows": [None] * 4})
        return {"outputs": outputs}


def _remote(monkeypatch, tmp_path):
    worker = _Worker()
    state = SB._BackendState(name="cellpose3", state=SB._INSTALLED,
                             env=str(tmp_path / "env"), in_process=False)
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None: state)
    backend = SB._RemoteBackend("cellpose3", model="cyto3",
                                worker_for=lambda name, env: worker)
    return backend, worker


def test_augment_and_the_normalization_dict_reach_the_worker(monkeypatch,
                                                             tmp_path):
    backend, worker = _remote(monkeypatch, tmp_path)
    backend.eval([np.zeros((8, 8), np.float32)], augment=True,
                 normalize={"normalize": True,
                            "percentile": [np.float64(1), 99.0]})
    params = worker.requests[0][1]["params"]
    assert params["augment"] is True
    assert params["normalize"] == {"normalize": True,
                                   "percentile": [1.0, 99.0]}


def test_a_run_that_does_not_set_augment_does_not_send_it(monkeypatch,
                                                          tmp_path):
    backend, worker = _remote(monkeypatch, tmp_path)
    backend.eval([np.zeros((8, 8), np.float32)])
    assert "augment" not in worker.requests[0][1]["params"]


def test_a_named_models_keywords_are_checked_against_the_inner_model():
    """cellpose 3.1.1.3's ``Cellpose.eval`` ends in ``**kwargs``; its
    ``CellposeModel`` (``.cp``) is what takes or refuses them. Plain
    functions on namespaces, because these are Cellpose 3's signatures, not
    stand-ins for the installed Cellpose 4 the contract test sweeps for."""

    def inner_eval(x, batch_size=8, augment=False, resample=True):
        raise AssertionError("not called")

    def wrapper_eval(x, batch_size=8, channels=None, **kwargs):
        raise AssertionError("not called")

    adapter = SB._Cellpose3Adapter.__new__(SB._Cellpose3Adapter)
    adapter._model = types.SimpleNamespace(
        cp=types.SimpleNamespace(eval=inner_eval), eval=wrapper_eval)
    adapter.ignored = set()
    taken = adapter._accepted({"augment": True, "net_avg": True,
                               "batch_size": 4})
    assert taken == {"augment": True, "batch_size": 4}
    assert adapter.ignored == {"net_avg"}


# ===========================================================================
# 7. pipeline_style 'v2' takes the same route (503, second pass)
# ===========================================================================

def _InputDependentCellpose3():
    """A stand-in for ``_RemoteBackend`` whose masks follow its input.

    The masks are the connected components of each image's first plane
    above its median, so two callers agree on the masks only when they
    hand over the same images. Every call keeps its keywords and a hash of
    each image, and the images themselves are kept. Its flows carry a cell
    probability, as Cellpose 3's do.
    """
    from skimage.measure import label

    calls, images_seen = [], []

    def _eval(x, **kwargs):
        images = [np.ascontiguousarray(np.asarray(i)) for i in x]
        images_seen.append(images)
        calls.append(dict(
            kwargs, shapes=[i.shape for i in images],
            dtypes=[str(i.dtype) for i in images],
            hashes=[hashlib.sha256(i).hexdigest() for i in images]))
        masks, flows = [], []
        for image in images:
            plane = image[..., 0] if image.ndim == 3 else image
            masks.append(label(plane > np.median(plane)).astype(np.uint16))
            flows.append([np.full(plane.shape + (3,), 7, np.uint8),
                          np.zeros((2,) + plane.shape, np.float32),
                          (plane - np.median(plane)).astype(np.float32),
                          None])
        return masks, flows, None

    return types.SimpleNamespace(calls=calls, images=images_seen,
                                 eval=_eval)


def _v2_stack(root, field):
    from spacr import pipeline_v2 as PV

    merged = root / "merged"
    merged.mkdir(parents=True)
    path = merged / "stack_A01_F001.npy"
    np.save(path, field)
    return [PV.StackFile("A01_F001", path, field.shape, ["cell", "nucleus"])]


@pytest.mark.parametrize("value, route", [
    ("cellpose3:cyto3", "cellpose3"),
    ("cellpose3:/models/w.pth", "cellpose3"),
    ("cpsam", None),
    ("cyto3", None),
])
def test_one_table_routes_a_prefixed_model(value, route):
    answer = O._prefixed_model_route(value)
    if route is None:
        assert answer is None
    else:
        assert answer == (route, O._cellpose3_masks)


def test_segmentation_backend_cellpose3_routes_a_bare_name_too():
    assert O._prefixed_model_route(
        "cyto2", {"segmentation_backend": "cellpose3"})[0] == "cellpose3"


def _spy_on_both_pipelines(monkeypatch):
    import cellpose.models

    seen = []
    model = _InputDependentCellpose3()

    def _load(name, **kwargs):
        seen.append(dict(kwargs, name=name))
        return model

    monkeypatch.setattr(SB, "_load_backend", _load)
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    monkeypatch.setattr(cellpose.models, "CellposeModel", _no_cellpose)
    return seen, model


def _field():
    rng = np.random.default_rng(3)
    field = rng.integers(100, 4000, size=(32, 32, 2)).astype(np.uint16)
    field[4:12, 4:12, 0] += 3000
    field[18:28, 16:30, 0] += 2500
    return field


def _run_v2(stacks, settings, **over):
    from spacr import pipeline_v2 as PV
    from spacr.object import _eval_diameter
    from spacr.qt.mask_engine import object_filter_area_floor

    PV.stream_masks_from_stack(
        stacks, model_name=settings["cell_model_name"],
        channels_for_cellpose=(0, 1),
        diameter=_eval_diameter(settings.get("cell_diameter"), "cell"),
        cellprob_threshold=float(settings["cell_cellprob_threshold"]),
        flow_threshold=float(settings["cell_flow_threshold"]),
        min_size=object_filter_area_floor(settings, "cell"),
        postprocess_settings=dict(settings, **over), object_type="cell")


def test_v2_segments_a_cellpose3_model_exactly_as_v1_does(tmp_path,
                                                         monkeypatch):
    """The same field, the same model: V1's saved mask and v2's appended
    mask plane are the same array, from the same images and keywords.

    v2 normalizes its raw field on the way to the model; V1's npz already
    holds the normalized field, which is what V1's preprocessing leaves
    there. So V1's npz is what v2 handed the model, laid out in V1's role
    order (nucleus before cell, see ``dense_mask_channel_positions``), and
    everything from there on -- the model, its call, the settings, the
    masks -- must agree.
    """
    from spacr.settings import set_default_settings_preprocess_generate_masks

    seen, model = _spy_on_both_pipelines(monkeypatch)
    v1_src = tmp_path / "v1" / "masks"
    settings = set_default_settings_preprocess_generate_masks(
        _base_settings(v1_src, cell_model_name="cellpose3:cyto3"))

    stacks = _v2_stack(tmp_path / "v2", _field())
    _run_v2(stacks, settings)
    [v2_call] = model.calls
    v2_saved = np.load(stacks[0].path)
    assert v2_saved.shape == (32, 32, 3)

    v1_src.mkdir(parents=True)
    handed = model.images[0][0]
    v1_stack = np.stack([handed[..., 1], handed[..., 0]], axis=-1)
    np.savez(v1_src / "batch1.npz", data=v1_stack[None],
             filenames=np.array(["plate1_A01_1.npy"]))
    O.generate_cellpose_masks_sam(str(v1_src), dict(settings), "cell")
    v1_call = model.calls[1]

    assert [s["name"] for s in seen] == ["cellpose3", "cellpose3"]
    assert [s["model_name"] for s in seen] == ["cellpose3:cyto3"] * 2
    assert v1_call == v2_call
    assert v2_call["shapes"] == [(32, 32, 2)]
    assert v2_call["diameter"] == 2 * 20 + 80
    assert v2_call["normalize"] == {"normalize": True,
                                    "percentile": [1.0, 99.0]}
    v1_mask = np.load(v1_src / "cell_mask_stack" / "plate1_A01_1.npy")
    assert v1_mask.dtype == v2_saved.dtype == np.uint16
    np.testing.assert_array_equal(v1_mask, v2_saved[..., -1])
    assert v1_mask.max() >= 2


def test_v2_passes_its_own_diameter_and_thresholds_to_cellpose3(
        tmp_path, monkeypatch):
    from spacr.settings import set_default_settings_preprocess_generate_masks

    seen, model = _spy_on_both_pipelines(monkeypatch)
    settings = set_default_settings_preprocess_generate_masks(
        _base_settings(tmp_path, cell_model_name="cellpose3:cyto2",
                       cell_diameter=55, cell_flow_threshold=0.7,
                       cell_cellprob_threshold=-1.5,
                       cellpose3_add_nucleus_channel=False))
    _run_v2(_v2_stack(tmp_path / "v2", _field()), settings)
    [call] = model.calls
    assert seen[0]["model_name"] == "cellpose3:cyto2"
    assert call["diameter"] == 55.0
    assert call["flow_threshold"] == 0.7
    assert call["cellprob_threshold"] == -1.5
    assert call["shapes"] == [(32, 32)]


def test_v2_on_cellpose_sam_never_loads_a_backend(tmp_path, monkeypatch):
    import cellpose.models

    from spacr import accelerator

    def _no_backend(*args, **kwargs):
        raise AssertionError("a backend was loaded for a Cellpose-SAM model")

    built = []

    def _sam(**kwargs):
        built.append(kwargs)
        return types.SimpleNamespace(eval=lambda x, **k: (
            [np.zeros(np.shape(i)[:2], np.uint16) for i in x], [], None))

    monkeypatch.setattr(SB, "_load_backend", _no_backend)
    monkeypatch.setattr(cellpose.models, "CellposeModel", _sam)
    monkeypatch.setattr(accelerator, "_CACHED", accelerator._CPU)
    from spacr import pipeline_v2 as PV

    PV.stream_masks_from_stack(_v2_stack(tmp_path, _field()),
                               model_name="cpsam",
                               channels_for_cellpose=(0, 1))
    assert len(built) == 1
