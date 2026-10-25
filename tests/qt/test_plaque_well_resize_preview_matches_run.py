"""Plaque mode's live preview and its batch run give identical well-resized masks.

Both go through :func:`spacr.plaque.segment_plaque_image`; this pins that the
preview's plan (read from the settings without a download) and the run's
plan (resolved once, with the detector fetched) segment the same wells the
same way, for a model whose zoo record declares well resize.
"""
import numpy as np
import pytest

from tests.conftest import MISSING_CHANNEL_AXIS
from tests.cellpose_api_contract import configured_eval_arguments

tifffile = pytest.importorskip("tifffile")
ppv = pytest.importorskip("spacr.qt.widgets.plaque_preview")

from spacr import model_zoo, plaque, submodules
from spacr.settings import get_analyze_plaque_settings

V6_RECORD = {"key": "toxoplasma_plaque_v6_test", "name": "cpsam_plaque_v6_test",
             "kind": "cellpose",
             "preprocess": {"well_resize": {"side": 1500, "max_factor": 6}}}
DETECTIONS = [(40.2, 60.7, 380.4, 390.1, .9), (430.5, 40.0, 790.3, 400.8, .8)]


class _ThresholdModel:
    """A Cellpose stand-in whose labels depend on the pixels it is given."""

    def __init__(self):
        self.calls = []

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
        kwargs = configured_eval_arguments(locals())
        from skimage.measure import label

        self.calls.append(kwargs)
        x = np.asarray(x)
        if x.ndim == 2:
            plane = x
        else:
            plane = x[0] if kwargs.get("channel_axis") == 0 else x[..., 0]
        labels = label(plane > 2.0).astype(np.int32)
        return labels, [np.zeros((*plane.shape, 3), np.uint8), None,
                        plane.astype(np.float32)], None


def _field():
    rng = np.random.default_rng(7)
    image = (rng.random((420, 820, 3)) * 80).astype(np.uint8)
    yy, xx = np.ogrid[:420, :820]
    for cy, cx, r in ((150, 150, 6), (300, 250, 3), (120, 600, 10), (330, 700, 2)):
        image[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 250
    return image


def test_preview_and_batch_masks_are_identical_with_well_resize(monkeypatch, tmp_path):
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS",
                        tuple(model_zoo.BUNDLED_REMOTE_MODELS) + (V6_RECORD,))
    monkeypatch.setattr(plaque, "_resolve_well_resize_detector",
                        lambda requested, fetch: "detector.pt")
    monkeypatch.setattr(plaque, "_raw_well_detections", lambda *_a: DETECTIONS)
    src, dst = tmp_path / "src", tmp_path / "masks"
    src.mkdir()
    tifffile.imwrite(src / "field.tif", _field())
    settings = get_analyze_plaque_settings(
        {"src": str(src), "dst": str(dst), "plaque_model": V6_RECORD["key"]})

    preview_model = _ThresholdModel()
    monkeypatch.setattr(ppv, "resolve_plaque_model", lambda s: ("model.pt", "", None))
    monkeypatch.setattr(ppv, "_cellpose_model", lambda path: preview_model)
    shown = ppv.plaque_pass(src / "field.tif", dict(settings))

    run_model = _ThresholdModel()
    monkeypatch.setattr(submodules, "_plaque_cellpose_model", lambda path: run_model)
    assert submodules._segment_plaque_folder(dict(settings), "model.pt") == 1
    saved = tifffile.imread(dst / "field.tif")

    assert len(preview_model.calls) == 2 and len(run_model.calls) == 2
    assert all(call["normalize"] is False for call in run_model.calls)
    assert shown["count"] == 4
    np.testing.assert_array_equal(np.asarray(shown["labels"]).astype(np.uint32),
                                  saved.astype(np.uint32))


def test_preview_and_batch_agree_without_well_resize(monkeypatch, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "masks"
    src.mkdir()
    tifffile.imwrite(src / "field.tif", _field())
    settings = get_analyze_plaque_settings({"src": str(src), "dst": str(dst)})
    preview_model, run_model = _ThresholdModel(), _ThresholdModel()
    monkeypatch.setattr(ppv, "resolve_plaque_model", lambda s: ("model.pt", "", None))
    monkeypatch.setattr(ppv, "_cellpose_model", lambda path: preview_model)
    monkeypatch.setattr(submodules, "_plaque_cellpose_model", lambda path: run_model)
    shown = ppv.plaque_pass(src / "field.tif", dict(settings))
    submodules._segment_plaque_folder(dict(settings), "model.pt")
    assert "normalize" not in preview_model.calls[0]
    assert preview_model.calls == run_model.calls
    np.testing.assert_array_equal(np.asarray(shown["labels"]).astype(np.uint32),
                                  tifffile.imread(dst / "field.tif").astype(np.uint32))


def _write(kind, folder):
    """One field of ``kind`` on disk, and the file name."""
    rgb = _field()
    if kind == "rgb_uint8":
        tifffile.imwrite(folder / "field.tif", rgb)
        return "field.tif"
    grey = rgb[..., 0]
    if kind == "grey_uint8":
        tifffile.imwrite(folder / "field.tif", grey)
        return "field.tif"
    if kind == "grey_uint16":
        tifffile.imwrite(folder / "field.tif", grey.astype(np.uint16) * 257)
        return "field.tif"
    from tests.test_scn_reader import write_scn

    write_scn(folder / "field.scn", [(4095 - grey.astype(np.uint16) * 16)])
    return "field.scn"


@pytest.mark.parametrize("resize", [True, False], ids=["resize", "off"])
@pytest.mark.parametrize("kind", ["grey_uint8", "grey_uint16", "rgb_uint8", "scn"])
def test_preview_and_batch_agree_for_every_image_kind(monkeypatch, tmp_path, kind, resize):
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS",
                        tuple(model_zoo.BUNDLED_REMOTE_MODELS) + (V6_RECORD,))
    monkeypatch.setattr(plaque, "_resolve_well_resize_detector",
                        lambda requested, fetch: "detector.pt")
    monkeypatch.setattr(plaque, "_raw_well_detections", lambda *_a: DETECTIONS)
    src, dst = tmp_path / "src", tmp_path / "masks"
    src.mkdir()
    name = _write(kind, src)
    settings = get_analyze_plaque_settings(
        {"src": str(src), "dst": str(dst),
         "plaque_model": V6_RECORD["key"] if resize else "toxoplasma_plaque_v5"})
    preview_model, run_model = _ThresholdModel(), _ThresholdModel()
    monkeypatch.setattr(ppv, "resolve_plaque_model", lambda s: ("model.pt", "", None))
    monkeypatch.setattr(ppv, "_cellpose_model", lambda path: preview_model)
    monkeypatch.setattr(submodules, "_plaque_cellpose_model", lambda path: run_model)
    shown = ppv.plaque_pass(src / name, dict(settings))
    assert submodules._segment_plaque_folder(dict(settings), "model.pt") == 1
    saved = tifffile.imread(dst / "field.tif")
    assert len(preview_model.calls) == len(run_model.calls) == (2 if resize else 1)
    for seen, ran in zip(preview_model.calls, run_model.calls):
        assert seen.keys() == ran.keys()
        assert {k: v for k, v in seen.items()} == {k: v for k, v in ran.items()}
    np.testing.assert_array_equal(np.asarray(shown["labels"]).astype(np.uint32),
                                  saved.astype(np.uint32))
    if resize:
        assert shown["count"] == 4


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_rgb_uint8_is_read_byte_for_byte_as_before(tmp_path, suffix):
    """The v5 path keeps reading 8-bit RGB exactly as cellpose.io.imread did."""
    import cellpose.io
    from PIL import Image

    path = tmp_path / f"field{suffix}"
    if suffix == ".tif":
        tifffile.imwrite(path, _field())
    else:
        Image.fromarray(_field()).save(path)
    before = cellpose.io.imread(str(path))
    after = plaque._read_plaque_image(path)
    assert after.dtype == before.dtype == np.uint8
    np.testing.assert_array_equal(after, before)


def test_loader_keeps_grey_and_16_bit_as_v6_was_trained(tmp_path):
    from tests.test_scn_reader import write_scn

    grey = _field()[..., 0]
    tifffile.imwrite(tmp_path / "g16.tif", grey.astype(np.uint16) * 257)
    read = plaque._read_plaque_image(tmp_path / "g16.tif")
    assert read.dtype == np.uint16 and read.shape == grey.shape
    tifffile.imwrite(tmp_path / "rgba.tif", np.dstack([_field(), np.full(grey.shape, 255, np.uint8)]))
    assert plaque._read_plaque_image(tmp_path / "rgba.tif").shape == (*grey.shape, 3)
    scn = plaque._read_plaque_image(write_scn(tmp_path / "plate.scn",
                                              [np.full((4, 6), 4095, np.uint16)]))
    assert scn.shape == (4, 6, 3) and scn.dtype == np.uint8
    assert plaque._grey_if_equal_channels(scn).shape == (4, 6)
    assert plaque._grey_if_equal_channels(_field()).shape == _field().shape
