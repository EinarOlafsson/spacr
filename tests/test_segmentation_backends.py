"""One optional segmentation-backend seam: DINOCell (item 404), SAMCell (405).

``segmentation_backend`` picks what ``generate_cellpose_masks_sam`` builds.
``'cellpose'`` -- or the key absent -- must be the existing path exactly;
``'dinocell'`` and ``'samcell'`` build a :class:`spacr._segmentation_backends.
_PlaneBackend` that answers the same ``model.eval`` call, so everything after
it (parsing, merge/split/filter, counts, saving) is shared.

What is pinned here:

1. names canonicalise, and an unknown name is refused;
2. the dispatch builds the backend it names, and refuses 3-D/4-D runs;
3. the default path builds Cellpose, never a backend, and writes the same
   bytes with the key absent or set to ``'cellpose'``;
4. a missing package raises an ImportError naming the Model Zoo, which
   installs it into an environment of its own (item 423), before any mask
   is written;
5. a stub backend's labels reach the saved masks with Cellpose's shape and
   dtype, computed from the object's own channel;
6. the seam imports neither package, nor torch, at import time (item 282);
7. one real end-to-end run per backend on a synthetic 256x256 image, which
   SKIPS unless the package is importable -- from this interpreter or from the
   directory named by ``SPACR_SEGMENTATION_BACKEND_SITE`` -- and its weights
   are already cached. It never downloads and always runs on the CPU.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

import spacr._segmentation_backends as SB
import spacr.object as O
import tests.test_object_tstack_wiring as _wiring

# The CellposeModel double and the npz/settings/artifact helpers the 4-D wiring
# tests already use, so this file adds no second Cellpose double to keep in
# step with tests/cellpose_api_contract.py. Bound as module attributes, not
# `from ... import`ed: pytest registers a fixture it finds in the module
# namespace either way, and a test parameter named like an imported fixture is
# a redefinition to ruff (F811). `force_cpu` is autouse there and here.
fake_model = _wiring.fake_model
force_cpu = _wiring.force_cpu
_artifacts = _wiring._artifacts
_base_settings = _wiring._base_settings
_write_npz = _wiring._write_npz

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directory holding the backend packages outside the main environment, e.g.
#: one filled by ``pip install --no-deps --target <dir> dinocell samcell``.
SITE_ENV = "SPACR_SEGMENTATION_BACKEND_SITE"


@pytest.fixture(autouse=True)
def _no_backend_environments(tmp_path, monkeypatch):
    """Item 423: an environment under the real ``~/.spacr/backends`` would
    route these runs out of process; every test here sees an empty one."""
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))


def _known_labels(shape):
    """Two rectangles, labelled 1 and 2."""
    out = np.zeros(shape, np.int32)
    out[2:9, 3:10] = 1
    out[14:20, 12:22] = 2
    return out


def _no_cellpose(*args, **kwargs):
    raise AssertionError("Cellpose was built on a non-Cellpose backend run")


@pytest.fixture
def stub_backends(monkeypatch):
    """Replace both backends with stubs that label every plane the same way.

    The stubs override ``_segment_plane`` only, so the batch handling under
    test is the real ``_PlaneBackend.eval``.
    """
    built = []

    def _make(name):
        class _Stub(SB._PlaneBackend):
            def __init__(self, device=None, **options):
                super().__init__(device="cpu")
                self.name = name
                self.options = options
                self.planes = []
                built.append(self)

            def _segment_plane(self, image, cellprob_threshold=None):
                self.planes.append(image)
                return _known_labels(image.shape), [image, None, None, None]

        return _Stub

    monkeypatch.setitem(SB._BACKEND_CLASSES, "dinocell", _make("dinocell"))
    monkeypatch.setitem(SB._BACKEND_CLASSES, "samcell", _make("samcell"))
    return built


# ===========================================================================
# 1. Names
# ===========================================================================

@pytest.mark.parametrize("value, expected", [
    (None, "cellpose"), ("", "cellpose"), ("cellpose", "cellpose"),
    (" Cellpose ", "cellpose"), ("dinocell", "dinocell"),
    ("DINOCell", "dinocell"), ("samcell", "samcell"), ("SAMCell", "samcell"),
    ("cellpose3", "cellpose3"), ("Cellpose3", "cellpose3"),
])
def test_a_backend_name_is_canonicalised(value, expected):
    assert SB._backend_name(value) == expected


def test_an_unknown_backend_is_refused_and_the_choices_are_named():
    with pytest.raises(ValueError, match="stardist") as exc:
        SB._backend_name("stardist")
    for name in SB._BACKEND_NAMES:
        assert repr(name) in str(exc.value)


def test_cellpose_is_the_first_choice_so_it_reads_as_the_default():
    assert SB._BACKEND_NAMES == ("cellpose", "cellpose3", "dinocell",
                                 "samcell")


# ===========================================================================
# 2. The dispatch
# ===========================================================================

@pytest.mark.parametrize("name", ["dinocell", "samcell", "DINOCell", "SAMCell"])
def test_the_dispatch_builds_the_backend_it_names(stub_backends, name):
    model = SB._load_backend(name)
    assert isinstance(model, SB._BACKEND_CLASSES[name.lower()])
    assert [m.name for m in stub_backends] == [name.lower()]


def test_the_real_backend_classes_are_registered_under_their_names():
    assert SB._BACKEND_CLASSES["dinocell"].__name__ == "_DinoCellBackend"
    assert SB._BACKEND_CLASSES["samcell"].__name__ == "_SamCellBackend"


def test_the_loader_does_not_build_cellpose(stub_backends):
    with pytest.raises(ValueError, match="cellpose"):
        SB._load_backend("cellpose")
    assert stub_backends == []


@pytest.mark.parametrize("plan", ["z_plan", "t_plan"])
def test_a_3d_or_4d_run_is_refused_before_anything_is_built(stub_backends,
                                                            plan):
    with pytest.raises(ValueError, match="2-D") as exc:
        SB._load_backend("dinocell", **{plan: object()})
    assert "segmentation_backend='cellpose'" in str(exc.value)
    assert stub_backends == []


def test_the_generator_passes_its_z_and_t_plans_to_the_loader(
        tmp_path, monkeypatch):
    seen = {}

    def _spy(name, **kwargs):
        seen.update(kwargs, name=name)
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(SB, "_load_backend", _spy)
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (1, 24, 24, 2))
    with pytest.raises(RuntimeError, match="stop after dispatch"):
        O.generate_cellpose_masks_sam(
            str(src), _base_settings(src, segmentation_backend="samcell"),
            "cell")
    assert seen == {"name": "samcell", "z_plan": None, "t_plan": None,
                    "model_name": "cpsam", "object_type": "cell"}


# ===========================================================================
# 3. The default path is the existing path
# ===========================================================================

def _forbid_backends(monkeypatch):
    def _no_backend(*args, **kwargs):
        raise AssertionError("a non-Cellpose backend was built on the "
                             "default path")

    monkeypatch.setattr(SB, "_load_backend", _no_backend)


@pytest.mark.parametrize("over", [
    {}, {"segmentation_backend": "cellpose"}, {"segmentation_backend": None},
], ids=["absent", "cellpose", "none"])
def test_the_default_path_builds_cellpose_and_no_backend(
        tmp_path, fake_model, monkeypatch, over):
    _forbid_backends(monkeypatch)
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (2, 24, 24, 2))
    O.generate_cellpose_masks_sam(str(src), _base_settings(src, **over),
                                  "cell")
    model = fake_model["model"]
    assert model is not None, "CellposeModel was not built"
    assert len(model.eval_kwargs) == 1
    assert sorted(p.name for p in (src / "cell_mask_stack").iterdir()) == [
        "plate1_A01_1.npy", "plate1_A01_2.npy"]


def test_the_default_path_writes_the_same_bytes_with_and_without_the_key(
        tmp_path, fake_model, monkeypatch):
    _forbid_backends(monkeypatch)
    runs = []
    for index, over in enumerate(({}, {"segmentation_backend": "cellpose"})):
        src = tmp_path / f"run{index}" / "masks"
        _write_npz(src, (2, 24, 24, 2))
        O.generate_cellpose_masks_sam(str(src), _base_settings(src, **over),
                                      "cell")
        model = fake_model["model"]
        runs.append((_artifacts(src), model.init_kwargs,
                     model.eval_configured))
    assert runs[0] == runs[1]


# ===========================================================================
# 4. A missing package
# ===========================================================================

def _hide_package(monkeypatch, name):
    for module in [m for m in sys.modules if m == name
                   or m.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setitem(sys.modules, name, None)


@pytest.mark.parametrize("name", ["dinocell", "samcell"])
def test_a_missing_package_names_the_model_zoo_that_installs_it(monkeypatch,
                                                                name):
    _hide_package(monkeypatch, name)
    with pytest.raises(ImportError) as exc:
        SB._load_backend(name, device="cpu")
    message = str(exc.value)
    assert "Install it from the Model Zoo" in message
    assert "an environment of its own" in message
    assert "segmentation_backend='cellpose'" in message
    assert isinstance(exc.value.__cause__, ImportError)


def test_a_missing_package_fails_before_any_mask_is_written(tmp_path,
                                                            monkeypatch):
    _hide_package(monkeypatch, "samcell")
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (1, 24, 24, 2))
    with pytest.raises(ImportError, match="Model Zoo"):
        O.generate_cellpose_masks_sam(
            str(src), _base_settings(src, segmentation_backend="samcell"),
            "cell")
    assert not (src / "cell_mask_stack").exists()


# ===========================================================================
# 5. A backend's labels become spaCR's masks
# ===========================================================================

def test_a_backends_labels_flow_through_to_the_saved_masks(
        tmp_path, stub_backends, monkeypatch):
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    src = tmp_path / "plate" / "masks"
    data = _write_npz(src, (3, 24, 24, 2))
    settings = _base_settings(src, segmentation_backend="dinocell")
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    [backend] = stub_backends
    saved = sorted((src / "cell_mask_stack").iterdir())
    assert [p.name for p in saved] == [
        "plate1_A01_1.npy", "plate1_A01_2.npy", "plate1_A01_3.npy"]
    for path in saved:
        mask = np.load(path)
        assert mask.shape == (24, 24)
        assert mask.dtype == np.uint16
        np.testing.assert_array_equal(mask, _known_labels((24, 24)))

    _masks, rows = _artifacts(src)
    assert rows and {count for _name, _kind, count in rows} == {2}

    # The backend saw one uint8 plane per field: the CELL channel, not the
    # nucleus channel the Cellpose batch carries beside it. Which npz index
    # that is comes from spaCR's own dense channel mapping, not from the raw
    # `cell_channel` value -- measured 2026-09-14, this fixture resolves the
    # cell channel to index 1 and the nucleus to 0 -- so the expectation is
    # derived the way the generator derives it rather than assumed.
    from spacr.utils import _get_cellpose_channels

    _extract, cellpose_channels = _get_cellpose_channels(settings)
    cell_index, nucleus_index = cellpose_channels["cell"]
    assert cell_index != nucleus_index
    assert len(backend.planes) == 3
    for index, plane in enumerate(backend.planes):
        assert plane.dtype == np.uint8 and plane.shape == (24, 24)
        np.testing.assert_array_equal(
            plane, SB._to_uint8(data[index, :, :, cell_index]))
        assert not np.array_equal(
            plane, SB._to_uint8(data[index, :, :, nucleus_index]))


def test_eval_returns_what_parse_cellpose4_output_reads(stub_backends):
    from spacr.spacr_cellpose import parse_cellpose4_output

    model = SB._load_backend("samcell")
    batch = [np.random.default_rng(i).random((16, 20, 2), dtype=np.float32)
             for i in range(2)]
    masks, flows0, flows1, flows2, flows3 = parse_cellpose4_output(
        model.eval(x=batch, batch_size=2, normalize=False, channel_axis=-1,
                   min_size=15, diameter=None, flow_threshold=0.4,
                   cellprob_threshold=0.0, resample=True, progress=True))
    assert [m.shape for m in masks] == [(16, 20), (16, 20)]
    assert all(m.dtype == np.uint16 for m in masks)
    assert len(flows0) == len(flows1) == len(flows2) == len(flows3) == 2


def test_eval_refuses_labels_of_the_wrong_shape():
    class _Wrong(SB._PlaneBackend):
        def _segment_plane(self, image, cellprob_threshold=None):
            return np.zeros((3, 3), np.int32), None

    with pytest.raises(ValueError, match="shape"):
        _Wrong(device="cpu").eval([np.zeros((8, 8, 1), np.float32)])


def test_eval_refuses_a_volume():
    with pytest.raises(ValueError, match="2-D"):
        SB._object_plane(np.zeros((2, 4, 4, 1)))


# ===========================================================================
# 5b. The helpers the real backends are built from
# ===========================================================================

def _dinocell_v2_origins(length, crop, overlap):
    """One axis of ``SlidingWindowHelper.seperate_into_crops_v2`` (dinocell 0.74),
    transcribed as the oracle."""
    origins = []
    for start in range(0, length, crop - overlap * 2):
        if start + crop > length:
            start = length - crop
        origins.append(start)
    return origins


@pytest.mark.parametrize("length", [512, 513, 600, 767, 768, 1000, 2048])
def test_tiles_are_dinocells_own_tiles_without_the_repeats(length):
    crop, overlap = SB._DINOCELL_CROP, SB._DINOCELL_OVERLAP
    starts = SB._tile_starts(length, crop, overlap)
    assert starts == sorted(set(_dinocell_v2_origins(length, crop, overlap)))
    covered = np.zeros(length, bool)
    for start in starts:
        covered[start:start + crop] = True
    assert covered.all()


def test_a_plane_no_wider_than_a_tile_is_one_tile():
    assert SB._tile_starts(512, 512, 128) == [0]
    assert SB._tile_starts(300, 512, 128) == [0]


def test_the_cellpose_logit_threshold_becomes_dinocells_probability():
    assert SB._probability_threshold(0) == 0.5
    assert SB._probability_threshold(None) == 0.5
    assert SB._probability_threshold(-1) == pytest.approx(0.26894, abs=1e-5)
    assert 0.0 <= SB._probability_threshold(-1e9) < 1e-20
    assert SB._probability_threshold(1e9) == pytest.approx(1.0)


def test_nearest_resampling_keeps_labels_labels():
    labels = _known_labels((24, 24))
    up = SB._resize_nearest(labels, (48, 48))
    assert up.dtype == labels.dtype and set(np.unique(up)) == {0, 1, 2}
    np.testing.assert_array_equal(SB._resize_nearest(up, (24, 24)), labels)
    stack = np.stack([labels, labels * 2])
    assert SB._resize_nearest(stack, (12, 30)).shape == (2, 12, 30)


def test_the_uint8_stretch_uses_the_full_range_and_survives_a_flat_plane():
    ramp = SB._to_uint8(np.linspace(10.0, 20.0, 16).reshape(4, 4))
    assert ramp.dtype == np.uint8 and (ramp.min(), ramp.max()) == (0, 255)
    assert not SB._to_uint8(np.full((4, 4), 7.0)).any()
    with_nan = np.array([[np.nan, 1.0], [2.0, 3.0]])
    assert SB._to_uint8(with_nan).tolist() == [[0, 0], [128, 255]]


def test_labels_come_back_sequential_in_cellposes_dtype():
    gappy = np.array([[0, 5], [9, 9]])
    out = SB._as_label_image(gappy)
    assert out.dtype == np.uint16 and out.tolist() == [[0, 1], [2, 2]]
    many = np.arange(70_000).reshape(280, 250)
    assert SB._as_label_image(many).dtype == np.uint32


def test_a_field_with_no_objects_is_an_empty_mask_in_cellposes_dtype():
    """An empty field is a real result that spaCR saves and counts as zero
    objects, so it must come back in Cellpose's dtype with nothing
    relabelled -- including a zero-size array, which has no maximum."""
    empty = SB._as_label_image(np.zeros((4, 5), np.int32))
    assert empty.dtype == np.uint16 and empty.shape == (4, 5)
    assert not empty.any()
    assert SB._as_label_image(np.zeros((0, 0), np.int64)).dtype == np.uint16


def test_a_plane_with_no_finite_pixel_stretches_to_black_not_to_garbage():
    """A field of NaN or inf has no range to stretch, and casting a
    non-finite float to uint8 is undefined; the backend must get zeros."""
    for plane in (np.full((3, 3), np.nan), np.array([[np.inf, -np.inf]])):
        out = SB._to_uint8(plane)
        assert out.dtype == np.uint8 and out.shape == plane.shape
        assert not out.any()


def test_resampling_labels_to_their_own_shape_leaves_them_untouched():
    """The common case -- a plane already at the model's size -- must hand
    back the label image itself, not an indexed copy."""
    labels = _known_labels((24, 24))
    assert SB._resize_nearest(labels, (24, 24)) is labels


def test_a_2d_image_is_its_own_plane_and_a_batch_of_one(stub_backends):
    """A single-channel field arrives as ``(H, W)``: it is the object's
    plane as it stands, and ``eval`` given one bare 2-D array segments it as
    one image rather than iterating over its rows. For ``(H, W, C)`` the
    first channel along the axis ``eval`` was told is the plane."""
    plane = np.arange(576.0).reshape(24, 24)
    assert SB._object_plane(plane) is plane
    channels_first = np.stack([plane, plane + 1000.0])
    np.testing.assert_array_equal(
        SB._object_plane(channels_first, channel_axis=0), plane)
    np.testing.assert_array_equal(
        SB._object_plane(np.moveaxis(channels_first, 0, -1),
                         channel_axis=None), plane)

    model = SB._load_backend("dinocell")
    masks, flows, _styles = model.eval(plane)
    assert len(masks) == len(flows) == 1
    assert masks[0].shape == (24, 24)
    [seen] = model.planes
    np.testing.assert_array_equal(seen, SB._to_uint8(plane))


def test_a_backend_without_a_segmenter_fails_loudly():
    """``_PlaneBackend`` is only the batch half of a backend. A subclass
    that forgets ``_segment_plane`` must raise, not return an empty mask."""
    with pytest.raises(NotImplementedError):
        SB._PlaneBackend(device="cpu").eval([np.zeros((4, 4, 1), np.float32)])


# ===========================================================================
# 6. Lazy imports and packaging
# ===========================================================================

def _subprocess_env(extra_path=None):
    env = dict(os.environ)
    parts = [p for p in (extra_path, str(REPO_ROOT), env.get("PYTHONPATH"))
             if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def test_importing_the_seam_imports_no_backend_and_no_torch():
    """Item 282. Run with the backend site on the path when there is one, so
    the packages are importable and the test proves they were not imported."""
    code = (
        "import sys, json\n"
        "import spacr\n"
        "before = set(sys.modules)\n"
        "import spacr._segmentation_backends\n"
        "roots = ('torch', 'cellpose', 'transformers', 'huggingface_hub',\n"
        "         'dinocell', 'samcell', 'cv2', 'skimage')\n"
        "new = sorted(m for m in set(sys.modules) - before\n"
        "             if m.split('.')[0] in roots)\n"
        "print(json.dumps(new))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
        text=True, timeout=300,
        env=_subprocess_env(os.environ.get(SITE_ENV)))
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def _extras_require():
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
            for keyword in node.keywords:
                if keyword.arg == "extras_require":
                    return ast.literal_eval(keyword.value)
    raise AssertionError("setup(extras_require=...) not found")


@pytest.mark.parametrize("name", ["dinocell", "samcell"])
def test_each_backend_has_its_own_extra_and_stays_out_of_all(name):
    from packaging.requirements import Requirement

    extras = _extras_require()
    assert name in extras, f"no `{name}` extra for the ImportError to name"
    assert [Requirement(s).name for s in extras[name]] == [name]
    assert name not in {Requirement(s).name for s in extras["all"]}


# ===========================================================================
# 7. Real end-to-end runs (skip without the package or cached weights)
# ===========================================================================

_SMOKE_SCRIPT = r'''
import json, resource, sys, time, types
from pathlib import Path

import numpy as np

backend, root = sys.argv[1], Path(sys.argv[2])
src = root / "plate" / "masks"
src.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
yy, xx = np.mgrid[:256, :256]
image = rng.normal(300.0, 20.0, (256, 256))
for cy in (48, 128, 208):
    for cx in (48, 128, 208):
        image += 3000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                 / (2 * 14.0 ** 2))
data = np.clip(image, 0, 65535).astype(np.uint16)[None, :, :, None]
np.savez(src / "batch1.npz", data=data,
         filenames=np.array(["plate1_A01_1.npy"]))

import spacr.object as O


def _no_cellpose(*args, **kwargs):
    raise AssertionError("Cellpose was built on a non-Cellpose backend run")


O.cp_models = types.SimpleNamespace(CellposeModel=_no_cellpose)
settings = {
    "src": str(src), "cell_channel": 0, "nucleus_channel": None,
    "pathogen_channel": None, "magnification": 20, "batch_size": 50,
    "verbose": False, "plot": False, "save": True, "timelapse": False,
    "n_jobs": 1, "seg_qc": "off", "segmentation_backend": backend,
}
start = time.perf_counter()
O.generate_cellpose_masks_sam(str(src), settings, "cell")
mask = np.load(src / "cell_mask_stack" / "plate1_A01_1.npy")
print("SMOKE-JSON " + json.dumps({
    "shape": list(mask.shape), "dtype": str(mask.dtype),
    "objects": int(np.unique(mask[mask > 0]).size),
    "seconds": round(time.perf_counter() - start, 1),
    "package_imported": backend in sys.modules,
    "peak_rss_gib": round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
}), flush=True)
'''


def _package_importable(name):
    site = os.environ.get(SITE_ENV)
    if site and (Path(site) / name / "__init__.py").is_file():
        return True
    return find_spec(name) is not None


def _hf_cached(repo_id, filename):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, filename)
    return isinstance(path, str) and os.path.isfile(path)


def _weights_cached(name):
    if name == "dinocell":
        return _hf_cached("KadenStillwagon/DINOCell", "DINOCell_demo_model.pt")
    base = SB._SAMCELL_BASE_MODEL
    return (_hf_cached(base, "config.json")
            and _hf_cached(base, "preprocessor_config.json")
            and (_hf_cached(base, "model.safetensors")
                 or _hf_cached(base, "pytorch_model.bin"))
            and os.path.isfile(SB._samcell_weights_path("generalist")))


@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.integration
@pytest.mark.parametrize("name", ["dinocell", "samcell"])
def test_a_real_backend_segments_a_synthetic_field_end_to_end(tmp_path, name):
    if not _package_importable(name):
        pytest.skip(f"{name} is not importable here; install "
                    f"`spacr[{name}]` or point {SITE_ENV} at a directory "
                    f"holding it")
    if not _weights_cached(name):
        pytest.skip(f"{name} weights are not cached, and this test never "
                    f"downloads")
    env = _subprocess_env(os.environ.get(SITE_ENV))
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-c", _SMOKE_SCRIPT, name, str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600, env=env)
    assert result.returncode == 0, (result.stdout[-2000:]
                                    + result.stderr[-4000:])
    [line] = [ln for ln in result.stdout.splitlines()
              if ln.startswith("SMOKE-JSON ")]
    report = json.loads(line[len("SMOKE-JSON "):])
    print(f"{name}: {report}")
    assert report["package_imported"], f"{name} was never imported"
    assert report["shape"] == [256, 256]
    assert report["dtype"] == "uint16"
    # Nine bright discs on a dark field. Measured 2026-09-14 on CPU: both
    # backends found all nine. "At least one" is the floor, not the score --
    # the scorecard items 404/405 ask for is a separate, labelled evaluation.
    assert report["objects"] >= 1, f"{name} found no object at all"
