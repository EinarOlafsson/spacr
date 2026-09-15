"""DINOCell and SAMCell drive their own packages the way those packages do.

``tests/test_segmentation_backends.py`` pins the seam with stub backends: the
names, the dispatch, the default path and the batch handling. It cannot reach
the two real classes, because neither package is installed in the development
environment and both need weights -- its end-to-end test skips without them.
Yet everything those classes do WITH a package is spaCR's own code, and a
mistake there would first surface on a user's machine:

1. the checkpoint each backend loads, and where it looks for one;
2. the model and pipeline each backend builds, with its package's constants;
3. DINOCell's tiling, which must average overlapping tiles, not overwrite them;
4. DINOCell's flows reaching Cellpose's dynamics in Cellpose's ``(dy, dx)``
   order with the threshold converted, on a plane upscaled to one tile and
   resampled back to its own size;
5. SAMCell's distance map becoming labels through its own pipeline, and a
   failed prediction raising rather than saving an empty field;
6. a package that is installed but fails to load still naming its extra.

Each package is replaced in ``sys.modules`` by stand-in modules that record
what spaCR built and called. Cellpose's ``compute_masks`` and torch's
``download_url_to_file`` are replaced too, but every call is first bound
against the INSTALLED signature, so a renamed keyword fails here rather than
in a user's run. Nothing downloads and nothing touches a GPU: every backend is
built with ``device='cpu'``.
"""
from __future__ import annotations

import builtins
import inspect
import sys
import types

import numpy as np
import pytest

import spacr._segmentation_backends as SB


def _drop_modules(monkeypatch, name):
    """Remove package ``name`` and its submodules from ``sys.modules``."""
    for module in [m for m in sys.modules
                   if m == name or m.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, module)


def _install_package(monkeypatch, name, **submodules):
    """Stand in for package ``name``; each keyword maps a submodule to the
    attributes it exports."""
    _drop_modules(monkeypatch, name)
    package = types.ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)
    for sub, attributes in submodules.items():
        module = types.ModuleType(f"{name}.{sub}")
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        setattr(package, sub, module)
        monkeypatch.setitem(sys.modules, f"{name}.{sub}", module)


# ===========================================================================
# DINOCell
# ===========================================================================

class _FakeDinoCell:
    """``dinocell.model.DINOCell``: records its arguments, weights and mode."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = None
        self.device = None
        self.evaluating = False

    def load_state_dict(self, state):
        self.state = state

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self


class _FakeFlowsPipeline:
    """What ``dinocell.pipeline.get_pipeline`` returns.

    Each prediction is ``[dx, dy, probability]`` as batch-of-one tensors: dx
    repeats the tile's own pixels, dy is ones, and the probability is the
    tile's 1-based position in the order it was predicted.
    """

    def __init__(self, objective, model, device, **kwargs):
        self.objective = objective
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.tiles = []

    def get_model_prediction(self, tile):
        import torch

        self.tiles.append(tile)
        pixels = torch.from_numpy(tile.astype(np.float32))[None]
        return [pixels, torch.ones_like(pixels),
                torch.full_like(pixels, float(len(self.tiles)))]


@pytest.fixture
def dinocell(monkeypatch, tmp_path):
    """A stand-in ``dinocell`` whose weight resolver hands back a real
    checkpoint written with ``torch.save``."""
    import torch

    checkpoint = tmp_path / "DINOCell_demo_model.pt"
    torch.save({"head.weight": torch.arange(6.0).reshape(2, 3)}, checkpoint)
    resolved = []

    def get_weights_path():
        resolved.append(str(checkpoint))
        return str(checkpoint)

    _install_package(
        monkeypatch, "dinocell",
        main={"get_weights_path": get_weights_path},
        model={"DINOCell": _FakeDinoCell},
        pipeline={"get_pipeline": _FakeFlowsPipeline})
    return types.SimpleNamespace(checkpoint=checkpoint, resolved=resolved)


@pytest.fixture
def compute_masks(monkeypatch):
    """Cellpose's ``compute_masks``, bound against its installed signature.

    Returns labels 1 (left half) and 2 (right half) at the flows' own size,
    so what comes back out shows how the labels were resampled.
    """
    from cellpose import dynamics

    signature = inspect.signature(dynamics.compute_masks)
    calls = []

    def _compute_masks(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        calls.append(arguments)
        labels = np.zeros(arguments["dP"].shape[1:], np.uint16)
        half = labels.shape[1] // 2
        labels[:, :half] = 1
        labels[:, half:] = 2
        return labels

    monkeypatch.setattr(dynamics, "compute_masks", _compute_masks)
    return calls


def test_dinocell_loads_its_checkpoint_into_its_flows_model_and_pipeline(
        dinocell, capsys):
    """The checkpoint only fits the architecture it was trained as, so the
    model must be built with the constants ``dinocell.main.segment`` uses,
    hold the checkpoint's tensors, sit on the run's device in eval mode, and
    be wrapped in DINOCell's flows pipeline at its own crop and overlap."""
    import torch

    backend = SB._load_backend("dinocell", device="cpu")

    assert isinstance(backend, SB._DinoCellBackend)
    assert backend.device == torch.device("cpu")
    assert dinocell.resolved == [str(dinocell.checkpoint)]
    pipeline = backend._pipeline
    model = pipeline.model
    assert isinstance(model, _FakeDinoCell)
    assert model.kwargs == {
        "dino_model": None, "decoder_type": "upsample",
        "objective_type": "flows", "use_dino_weights": False,
        "patch_size": 8, "feat_size": 64, "crop_size": 512,
        "drop_rate": 0.05, "dropout_in_encoder": True,
        "finetune_vision": True, "finetune_decoder": True,
        "finetune_prediction_head": True}
    assert list(model.state) == ["head.weight"]
    torch.testing.assert_close(model.state["head.weight"],
                               torch.arange(6.0).reshape(2, 3))
    assert model.device == torch.device("cpu")
    assert model.evaluating
    assert pipeline.objective == "flows"
    assert pipeline.device == torch.device("cpu")
    assert pipeline.kwargs == {"crop_size": 512,
                               "use_advanced_augmentations": False,
                               "overlap_size": 128}
    assert capsys.readouterr().out == (
        f"Segmentation backend: dinocell -- {SB._DinoCellBackend.note}.\n")


def test_a_named_dinocell_checkpoint_is_used_without_asking_the_hub(
        dinocell, tmp_path):
    """``get_weights_path`` resolves DINOCell's published checkpoint through
    the Hugging Face cache. A user who names their own checkpoint must get
    that one, and the hub must not be consulted at all."""
    import torch

    own = tmp_path / "own.pt"
    torch.save({"own.weight": torch.ones(1)}, own)

    backend = SB._load_backend("dinocell", device="cpu", weights_path=str(own))

    assert dinocell.resolved == []
    assert list(backend._pipeline.model.state) == ["own.weight"]


def test_dinocell_averages_overlapping_tiles_instead_of_overwriting_them(
        dinocell):
    """A plane wider than one tile is predicted tile by tile. Where tiles
    overlap the prediction is their mean, as in DINOCell's own sliding
    window, so a pixel's value cannot depend on which tile ran last; and
    every tile's prediction must land back where that tile was cut."""
    backend = SB._load_backend("dinocell", device="cpu")
    image = np.random.default_rng(0).integers(0, 256, (600, 700),
                                              dtype=np.uint8)

    dx, dy, probability = backend._predict(image)

    # 600 rows start tiles at 0 and 88, 700 columns at 0 and 188.
    tiles = backend._pipeline.tiles
    assert len(tiles) == 4
    np.testing.assert_array_equal(tiles[0], image[:512, :512])
    np.testing.assert_array_equal(tiles[3], image[88:600, 188:700])
    np.testing.assert_array_equal(dx, image.astype(np.float32))
    np.testing.assert_array_equal(dy, np.ones(image.shape, np.float32))
    assert probability[0, 0] == 1.0          # inside the first tile only
    assert probability[599, 699] == 4.0      # inside the last tile only
    assert probability[300, 350] == 2.5      # inside all four: their mean


def test_a_small_plane_is_upscaled_keeping_its_aspect_and_labelled_at_its_own_size(
        dinocell, compute_masks):
    """DINOCell sees 512-pixel tiles, so a 64x128 field is upscaled until its
    SHORT side is one tile -- 512x1024, not a squashed square -- and labels,
    flows and probability all come back at 64x128, the size spaCR saves.
    The flows must reach Cellpose as ``(dy, dx)``, with spaCR's logit
    threshold turned into DINOCell's probability."""
    import torch

    backend = SB._load_backend("dinocell", device="cpu")
    seen = []

    def _predict(work):
        seen.append(work)
        return (np.full(work.shape, 0.25, np.float32),
                np.full(work.shape, -0.5, np.float32),
                np.full(work.shape, 0.75, np.float32))

    backend._predict = _predict
    image = np.random.default_rng(1).integers(0, 256, (64, 128),
                                              dtype=np.uint8)

    labels, flow = backend._segment_plane(image, cellprob_threshold=-1.0)

    [work] = seen
    assert work.shape == (512, 1024) and work.dtype == np.uint8
    [call] = compute_masks
    np.testing.assert_array_equal(call["dP"][0], np.full((512, 1024), -0.5))
    np.testing.assert_array_equal(call["dP"][1], np.full((512, 1024), 0.25))
    np.testing.assert_array_equal(call["cellprob"], np.full((512, 1024), 0.75))
    assert call["cellprob_threshold"] == pytest.approx(0.26894, abs=1e-5)
    assert {key: call[key] for key in ("niter", "flow_threshold", "do_3D",
                                        "min_size", "max_size_fraction")} == {
        "niter": 250, "flow_threshold": 0, "do_3D": False, "min_size": 15,
        "max_size_fraction": 0.4}
    assert call["device"] == torch.device("cpu")
    expected = np.zeros((64, 128), np.uint16)
    expected[:, :64] = 1
    expected[:, 64:] = 2
    np.testing.assert_array_equal(labels, expected)
    display, d_p, cell_probability, last = flow
    assert display.shape == (64, 128, 3)
    assert d_p.shape == (2, 64, 128)
    assert cell_probability.shape == (64, 128)
    assert last is None


def test_a_plane_at_least_one_tile_wide_reaches_the_model_unresized(
        dinocell, compute_masks):
    """Resampling costs detail. A field whose short side is already a tile
    goes to the model as the very pixels spaCR stretched from the object's
    channel, and its labels come back at that size with nothing resampled."""
    backend = SB._load_backend("dinocell", device="cpu")
    seen = []

    def _predict(work):
        seen.append(work)
        zeros = np.zeros(work.shape, np.float32)
        return zeros, zeros, zeros

    backend._predict = _predict
    field = np.random.default_rng(2).random((512, 600, 2), dtype=np.float32)

    masks, flows, styles = backend.eval([field], channel_axis=-1,
                                        cellprob_threshold=0.0)

    [work] = seen
    np.testing.assert_array_equal(work, SB._to_uint8(field[:, :, 0]))
    [call] = compute_masks
    assert call["dP"].shape == (2, 512, 600)
    assert call["cellprob_threshold"] == 0.5
    assert masks[0].shape == (512, 600) and masks[0].dtype == np.uint16
    assert len(flows) == 1 and styles is None


# ===========================================================================
# SAMCell
# ===========================================================================

@pytest.fixture
def torch_hub(monkeypatch, tmp_path):
    """torch's hub cache moved into ``tmp_path``, and a download that writes
    a file instead of fetching one."""
    import torch

    signature = inspect.signature(torch.hub.download_url_to_file)
    downloads = []

    def _download(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        downloads.append(arguments)
        with open(arguments["dst"], "wb") as handle:
            handle.write(b"checkpoint")

    root = tmp_path / "hub"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(root))
    monkeypatch.setattr(torch.hub, "download_url_to_file", _download)
    return types.SimpleNamespace(root=root, downloads=downloads)


def test_asking_where_a_samcell_checkpoint_lives_downloads_nothing(torch_hub):
    """The end-to-end test asks where the checkpoint is to decide whether to
    skip. That question must answer with torch's hub cache path for the
    variant, and must neither fetch the checkpoint nor create a directory."""
    path = SB._samcell_weights_path("generalist")

    assert path == str(torch_hub.root / "checkpoints" / "samcell-generalist.pt")
    assert SB._samcell_weights_path("cyto") == str(
        torch_hub.root / "checkpoints" / "samcell-cyto.pt")
    assert torch_hub.downloads == []
    assert not torch_hub.root.exists()


def test_a_missing_samcell_checkpoint_is_downloaded_once_from_the_release(
        torch_hub, capsys):
    """SAMCell publishes its checkpoints as GitHub release assets. The first
    build fetches the asset into the cache and says so; every later build
    finds it there and fetches nothing."""
    path = SB._samcell_weights_path("cyto", download=True)

    [download] = torch_hub.downloads
    assert download == {"url": SB._SAMCELL_RELEASE + "samcell-cyto.pt",
                        "dst": path, "progress": True}
    assert "Downloading SAMCell weights" in capsys.readouterr().out

    assert SB._samcell_weights_path("cyto", download=True) == path
    assert len(torch_hub.downloads) == 1


class _FakeFinetunedSAM:
    """``samcell.model.FinetunedSAM``: records its base model and weights."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.weights = None

    def load_weights(self, path, map_location=None):
        self.weights = (path, map_location)


class _FakeSamPipeline:
    """``samcell.pipeline.SlidingWindowPipeline``.

    The distance map is the image scaled to [0, 1]; its cells are the pixels
    above one half, labelled 7 so relabelling shows.
    """

    def __init__(self, model, device, crop_size=None):
        self.model = model
        self.device = device
        self.crop_size = crop_size
        self.images = []
        self.failure = None

    def predict_on_full_img(self, image):
        if self.failure is not None:
            raise self.failure
        self.images.append(image)
        return image.astype(np.float32) / 255.0

    def cells_from_dist_map(self, dist_map):
        return (dist_map > 0.5).astype(np.int32) * 7

    def run(self, *args, **kwargs):
        raise AssertionError("spaCR called SlidingWindowPipeline.run")


@pytest.fixture
def samcell(monkeypatch, tmp_path):
    """A stand-in ``samcell``, with the checkpoint lookup recorded."""
    checkpoint = str(tmp_path / "samcell-generalist.pt")
    requested = []

    def _weights_path(variant="generalist", download=False):
        requested.append((variant, download))
        return checkpoint

    monkeypatch.setattr(SB, "_samcell_weights_path", _weights_path)
    _install_package(
        monkeypatch, "samcell",
        model={"FinetunedSAM": _FakeFinetunedSAM},
        pipeline={"SlidingWindowPipeline": _FakeSamPipeline})
    return types.SimpleNamespace(checkpoint=checkpoint, requested=requested)


@pytest.mark.parametrize("variant", ["generalist", "cyto"])
def test_samcell_builds_sam_vit_b_with_the_checkpoint_for_its_variant(
        samcell, capsys, variant):
    """SAMCell is SAM ViT-B with a fine-tuned checkpoint. The build must ask
    for the chosen variant's checkpoint (downloading it when absent), load it
    onto the run's device, and wrap the model in SAMCell's sliding window at
    its own 256-pixel crop."""
    import torch

    backend = SB._load_backend("samcell", device="cpu", variant=variant)

    assert isinstance(backend, SB._SamCellBackend)
    assert samcell.requested == [(variant, True)]
    pipeline = backend._pipeline
    assert pipeline.model.base_model == "facebook/sam-vit-base"
    assert pipeline.model.weights == (samcell.checkpoint, torch.device("cpu"))
    assert pipeline.device == torch.device("cpu")
    assert pipeline.crop_size == 256
    assert capsys.readouterr().out == (
        f"Segmentation backend: samcell -- {SB._SamCellBackend.note}.\n")


def test_a_named_samcell_checkpoint_is_used_without_a_download(samcell,
                                                               tmp_path):
    """A user who names a checkpoint may be offline or on a private
    fine-tune; the release asset must not be looked up or fetched."""
    own = str(tmp_path / "own.pt")

    backend = SB._load_backend("samcell", device="cpu", weights_path=own)

    assert samcell.requested == []
    assert backend._pipeline.model.weights[0] == own


def test_samcells_distance_map_becomes_labels_through_its_own_pipeline(
        samcell):
    """SAMCell predicts a distance map, not flows. The labels must be what
    its own ``cells_from_dist_map`` finds on the object's channel, made
    sequential, and the map is the flow entry spaCR displays and reads as
    the probability."""
    backend = SB._load_backend("samcell", device="cpu")
    field = np.zeros((40, 50, 2), np.float32)
    field[5:15, 5:20, 0] = 100.0      # the object's own channel
    field[20:30, 30:45, 1] = 100.0    # a second channel the backend ignores

    masks, flows, styles = backend.eval([field], channel_axis=-1)

    [image] = backend._pipeline.images
    np.testing.assert_array_equal(image, SB._to_uint8(field[:, :, 0]))
    expected = np.zeros((40, 50), np.uint16)
    expected[5:15, 5:20] = 1
    np.testing.assert_array_equal(masks[0], expected)
    assert masks[0].dtype == np.uint16
    [(dist_map, d_p, probability, last)] = flows
    np.testing.assert_array_equal(dist_map, image.astype(np.float32) / 255.0)
    assert d_p is None and probability is dist_map and last is None
    assert styles is None


def test_a_failed_samcell_prediction_stops_the_run_instead_of_saving_an_empty_field(
        samcell):
    """SAMCell's ``run`` logs a failure and returns an all-zero label image,
    which spaCR would save as a field with no cells: a wrong count, not an
    error. The prediction must go through ``predict_on_full_img``, which
    raises, so the failure reaches the user."""
    backend = SB._load_backend("samcell", device="cpu")
    backend._pipeline.failure = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        backend.eval([np.ones((16, 16, 1), np.float32)])
    assert backend._pipeline.images == []


# ===========================================================================
# A package that is installed but will not load
# ===========================================================================

@pytest.mark.parametrize("name", ["dinocell", "samcell"])
def test_a_package_that_fails_to_load_names_its_extra_like_a_missing_one(
        monkeypatch, name):
    """A package can be installed and still fail to import: a shared library
    that will not load raises OSError, not ImportError. The user needs the
    same advice as for a missing package, and the loader's own message so
    they can see what actually broke."""
    real_import = builtins.__import__

    def _import(module, *args, **kwargs):
        if module == name or module.startswith(name + "."):
            raise OSError(f"lib{name}.so: cannot open shared object file")
        return real_import(module, *args, **kwargs)

    _drop_modules(monkeypatch, name)
    monkeypatch.setattr(builtins, "__import__", _import)

    with pytest.raises(ImportError) as exc:
        SB._load_backend(name, device="cpu")

    message = str(exc.value)
    assert f'`pip install "spacr[{name}]"`' in message
    assert "segmentation_backend='cellpose'" in message
    assert f"lib{name}.so: cannot open shared object file" in message
    assert isinstance(exc.value.__cause__, OSError)
