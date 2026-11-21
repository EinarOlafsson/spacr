"""The V2 streaming pipeline (``spacr.pipeline_v2``) on small synthetic plates.

V2 reads the originals straight into one ``merged/stack_<field>.npy`` per
field, segments the stacks in batches and appends the mask plane back into the
same files.  Anything it guesses wrong is baked into those files, so these
tests pin the exact on-disk result: the regex it picks, the planes it reads or
refuses (and that a refusal leaves the source bytes alone), the shape and
contents each stack ends with, and the calibrated route's refusals before any
model is built.  Cellpose is replaced by small doubles that go through
``check_cellpose_eval_call``.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from spacr import pipeline_v2 as PV

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

#: Postprocess settings that switch on V1 normalisation without filtering.
_NORMALISE = {
    "lower_percentile": 2, "background": 100, "Signal_to_noise": 10,
    "remove_background": False, "cell_background": 100,
    "cell_signal_to_noise": 10, "remove_background_cell": False,
    "nucleus_background": 100, "nucleus_signal_to_noise": 10,
    "remove_background_nucleus": False,
}
#: Settings that select the calibrated route (the file is never opened).
_CALIBRATED = {"unmix": True, "unmix_calibration": "not-opened.json"}


def _make_plate(dst: Path, name="plate1", fields=(1, 2), channels=3, size=12,
                sizes=None) -> Path:
    """A CellVoyager-named plate of random uint16 planes, well A01."""
    import tifffile

    plate = dst / name
    plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for field in fields:
        side = sizes[field] if sizes else size
        for channel in range(channels):
            arr = rng.integers(1, 2000, size=(side, side)).astype(np.uint16)
            tifffile.imwrite(
                str(plate / f"{name}_A01_T01F0{field}L01A01Z01C0{channel}.tif"),
                arr)
    return plate


def _snapshot(folder: Path):
    """Every file's bytes under ``folder`` (top level only)."""
    return {path: path.read_bytes() for path in folder.iterdir()
            if path.is_file()}


def _model(fill="square", *, bare=False, seen=None):
    """A Cellpose double: one mask per image, or a bare array for one image.

    ``fill`` is ``"square"`` (label 1 at ``[1:4, 1:4]``), ``"ones"`` or
    ``"zeros"``; ``seen`` collects copies of the images it was handed.
    """
    class _Model:
        def __init__(self, *args, **kwargs):
            self.pretrained_model = kwargs.get("pretrained_model")

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            images = x if isinstance(x, list) else [x]
            if seen is not None:
                seen.extend(np.asarray(image).copy() for image in images)
            masks = []
            for image in images:
                shape = np.asarray(image).shape[:2]
                if fill == "ones":
                    mask = np.ones(shape, np.uint16)
                else:
                    mask = np.zeros(shape, np.uint16)
                    if fill == "square":
                        mask[1:4, 1:4] = 1
                masks.append(mask)
            return (masks[0] if bare else masks), None, None

    return _Model


def _stacks_for(tmp_path, name="plate1", channels=(0, 1, 2), **plate):
    """Discover a fresh plate and write its field stacks."""
    plate_dir = _make_plate(tmp_path, name=name, channels=len(channels),
                            **plate)
    mapper = PV.FilenameMapper.discover(plate_dir, metadata_type="cellvoyager")
    return plate_dir, PV.stream_originals_to_stack(plate_dir, mapper,
                                                   channels=channels)


def test_read_plane_returns_a_uint16_view_or_refuses_unacquired_axes(
        tmp_path, subtests, monkeypatch):
    """The legacy reader keeps plane 0 as uint16; the calibrated reader keeps
    the acquired dtype and refuses multi-band, animated, volume, multi-series
    and over-budget sources (TIFF budget before decoding), never changing the
    source bytes.  ``_partition_plane_header`` refuses the same sources."""
    from PIL import Image
    import tifffile

    pixels = np.arange(20, dtype=np.uint8).reshape(4, 5)
    single_channel = "single-channel 2D"

    def gray_png(path):
        Image.fromarray(pixels).save(path)

    def rgb_png(path):
        Image.fromarray(np.stack([pixels, pixels + 1, pixels + 2], -1)).save(path)

    def gif(path):
        Image.fromarray(pixels).save(path, save_all=True, duration=100,
                                     append_images=[Image.fromarray(pixels + 1)])

    def volume(path):
        tifffile.imwrite(path, np.stack([pixels] * 2), photometric="minisblack")

    def two_series(path):
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((4, 5), np.uint16), photometric="minisblack")
            writer.write(np.ones((6, 7), np.uint16), photometric="minisblack")

    def tif16(path):
        tifffile.imwrite(path, pixels.astype(np.uint16) * 100)

    legacy = [
        ("2-D tiff", "a.tif", tif16, pixels.astype(np.uint16) * 100),
        ("gray png", "a.png", gray_png, pixels.astype(np.uint16)),
        ("rgb png keeps band 0", "b.png", rgb_png, pixels.astype(np.uint16)),
    ]
    for case_id, name, write, expected in legacy:
        with subtests.test(msg=f"legacy {case_id}"):
            path = tmp_path / name
            write(path)
            out = PV._read_plane(str(path))
            assert out.dtype == np.uint16
            np.testing.assert_array_equal(out, expected)

    with subtests.test(msg="calibrated gray png keeps its dtype"):
        path = tmp_path / "gray.png"
        gray_png(path)
        header = PV._partition_plane_header(path)
        assert header["shape"] == [4, 5]
        assert np.dtype(header["dtype"]) == np.uint8
        out = PV._read_plane(str(path), calibrated=True)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, pixels)
        np.testing.assert_array_equal(
            PV._read_plane(str(path), calibrated=True, max_pixels=20), pixels)

    refusals = [
        ("png over budget", "budget.png", gray_png, {"max_pixels": 19},
         "source-pixel budget", True),
        ("rgb png", "rgb.png", rgb_png, {}, single_channel, False),
        ("animated gif", "anim.gif", gif, {}, single_channel, False),
        ("volume tiff", "vol.tif", volume, {}, single_channel, False),
        ("two-series tiff", "series.tif", two_series, {}, single_channel,
         False),
    ]
    for case_id, name, write, kwargs, message, header_ok in refusals:
        with subtests.test(msg=f"calibrated refuses {case_id}"):
            path = tmp_path / name
            write(path)
            before = path.read_bytes()
            with pytest.raises(ValueError, match=message):
                PV._read_plane(str(path), calibrated=True, **kwargs)
            if not header_ok:
                with pytest.raises(ValueError):
                    PV._partition_plane_header(path)
            assert path.read_bytes() == before

    with subtests.test(msg="tiff budget refuses before decoding"):
        path = tmp_path / "budget.tif"
        tifffile.imwrite(path, np.arange(20, dtype=np.uint16).reshape(4, 5))
        before = path.read_bytes()
        monkeypatch.setattr(tifffile, "imread",
                            lambda *a, **k: pytest.fail("plane was decoded"))
        with pytest.raises(ValueError, match="source-pixel budget"):
            PV._read_plane(path, calibrated=True, max_pixels=19)
        assert path.read_bytes() == before


def test_resolve_regex_picks_the_named_layout_or_the_best_fit(tmp_path,
                                                              subtests,
                                                              caplog):
    """``custom`` needs and returns its own regex; a named layout is used
    when every file matches; otherwise the candidate matching most files wins
    (the first on a tie) and a warning names it."""
    cellvoyager = [Path("plate1_A01_T01F01L01A01Z01C00.tif"),
                   Path("plate1_A01_T01F01L01A01Z01C01.tif")]
    yokogawa = [Path("p_A01_T0001F001L01A01Z01C01.tif")]
    garbage = [Path("garbage_name_1.tif"), Path("garbage_2.tif")]
    cases = [
        ("custom", "custom", cellvoyager, r".*", (r".*", "custom"), None),
        ("cellvoyager", "cellvoyager", cellvoyager, None,
         (PV._CELLVOYAGER, "cellvoyager"), None),
        ("yokogawa exact", "yokogawa", yokogawa, None,
         (PV._YOKOGAWA, "yokogawa"), None),
        ("yokogawa named, cellvoyager files", "yokogawa", cellvoyager, None,
         (PV._YOKOGAWA, "yokogawa"), "best fit was yokogawa (0/2)"),
        ("auto, all cellvoyager", "auto", cellvoyager, None,
         (PV._CELLVOYAGER, "cellvoyager"), None),
        ("unknown type acts as auto", "zeiss", yokogawa, None,
         (PV._CELLVOYAGER, "cellvoyager"), None),
        ("auto, nothing matches", "auto", garbage, None,
         (PV._CELLVOYAGER, "cellvoyager"), "best fit was cellvoyager (0/2)"),
    ]
    for case_id, kind, files, regex, expected, warning in cases:
        with subtests.test(msg=case_id):
            caplog.clear()
            with caplog.at_level("WARNING", logger="spacr.pipeline_v2"):
                assert PV._resolve_regex(kind, files, regex) == expected
            messages = [r.getMessage() for r in caplog.records]
            if warning:
                assert any(warning in m for m in messages), messages
            else:
                assert not messages
    with subtests.test(msg="custom without a regex"):
        with pytest.raises(ValueError, match="needs custom_regex"):
            PV._resolve_regex("custom", [Path("x.tif")], None)


def test_stream_originals_to_stack_writes_one_hwc_stack_per_field(tmp_path,
                                                                 subtests):
    """Each field becomes an (H, W, C) uint16 stack in channel order with a
    ``channel_order.json`` sidecar and ``filename_map.csv``; a missing channel
    is a zero plane of the field's shape, a field with none of the channels a
    256x256 zero plane; the calibrated route refuses a missing channel or a
    mismatched YX shape before writing anything."""
    import tifffile

    plate, stacks = _stacks_for(tmp_path, channels=(0, 1, 2))
    with subtests.test(msg="one stack per field, planes in channel order"):
        assert [s.field_id for s in stacks] == sorted(s.field_id for s in stacks)
        assert len(stacks) == 2
        for stack in stacks:
            arr = np.load(stack.path)
            assert arr.dtype == np.uint16 and arr.shape == (12, 12, 3)
            assert stack.shape == (12, 12, 3)
            assert stack.channels == ["ch0", "ch1", "ch2"]
            assert stack.path == plate / "merged" / f"stack_{stack.field_id}.npy"
        first = sorted(plate.glob("*F01*C01.tif"))[0]
        np.testing.assert_array_equal(np.load(stacks[0].path)[..., 1],
                                      tifffile.imread(first))
        sidecar = json.loads((plate / "merged" / "channel_order.json").read_text())
        assert sidecar["image_channels"] == ["ch0", "ch1", "ch2"]
        assert sidecar["mask_channels"] == []
        assert (plate / "filename_map.csv").is_file()

    with subtests.test(msg="missing channel becomes zeros"):
        mapper = PV.FilenameMapper.discover(plate, metadata_type="cellvoyager")
        out = PV.stream_originals_to_stack(plate, mapper, channels=(9, 1),
                                           channel_names=["gfp", "dapi"],
                                           dst=tmp_path / "other")
        arr = np.load(out[0].path)
        assert arr.shape == (12, 12, 2) and not arr[..., 0].any()
        assert arr[..., 1].any() and out[0].channels == ["gfp", "dapi"]

    with subtests.test(msg="no requested channel at all"):
        mapper = PV.FilenameMapper.discover(plate, metadata_type="cellvoyager")
        out = PV.stream_originals_to_stack(plate, mapper, channels=(7,),
                                           dst=tmp_path / "none")
        assert len(out) == 2
        for stack in out:
            arr = np.load(stack.path)
            assert arr.shape == (256, 256, 1) and not arr.any()

    for damage, message in (("missing-channel", "physical channel"),
                            ("different-shape", "same acquired YX")):
        with subtests.test(msg=f"calibrated refuses {damage}"):
            source = tmp_path / damage
            source.mkdir()
            records = []
            for channel, shape in [(0, (4, 5)), (1, (5, 5) if damage ==
                                                 "different-shape" else (4, 5))]:
                path = source / f"c{channel}.tif"
                tifffile.imwrite(path, np.full(shape, channel + 11, np.uint16))
                records.append(PV.FilenameRecord(str(path), "plate1", "A01", 1,
                                                 channel, stack_field_id="f1"))
            mapper = PV.FilenameMapper(
                records[:1] if damage == "missing-channel" else records,
                "custom", "")
            before = _snapshot(source)
            destination = tmp_path / f"{damage}-out"
            with pytest.raises(ValueError, match=message):
                PV.stream_originals_to_stack(source, mapper, channels=(0, 1),
                                             dst=destination, calibrated=True)
            assert not list(destination.glob("*.npy"))
            assert _snapshot(source) == before


def test_stream_masks_from_stack_refuses_before_building_a_model(
        tmp_path, subtests, monkeypatch):
    """No stacks returns the same list with no scratch or model; a missing
    Cellpose is a RuntimeError; a calibrated session refuses a non-cell
    object, illumination, a stack outside its merged folder, an out-of-range
    channel and non-finite pixels, all before a model exists and without
    touching the stack."""
    from cellpose import models

    with subtests.test(msg="no stacks"):
        def _never(*args, **kwargs):
            raise AssertionError("cellpose must not be loaded for no stacks")

        monkeypatch.setattr(models, "CellposeModel", _never)
        empty = []
        assert PV.stream_masks_from_stack(empty, model_name="cyto") is empty
        assert not list(tmp_path.iterdir())

    root = tmp_path / "calibrated"
    (root / "merged").mkdir(parents=True)
    session = types.SimpleNamespace(calibrated=True, root=root,
                                    configuration={})
    nonfinite = np.ones((4, 5, 1), np.float32)
    nonfinite[1, 2, 0] = np.nan
    cases = [
        ("non-cell object", {"object_type": "nucleus"}, None,
         "cell objects without illumination"),
        ("illumination", {"illumination_session": object()}, None,
         "cell objects without illumination"),
        ("foreign parent", {}, tmp_path, "separate output merged directory"),
        ("channel out of range", {"channels_for_cellpose": (1,)}, None,
         "valid model positions"),
        ("non-finite pixels", {}, nonfinite, "finite 2D channel-last"),
    ]
    for case_id, kwargs, where, message in cases:
        with subtests.test(msg=f"calibrated: {case_id}"):
            values = where if isinstance(where, np.ndarray) else \
                np.ones((4, 5, 1), np.float32)
            folder = where if isinstance(where, Path) else root / "merged"
            path = folder / "stack_f1.npy"
            np.save(path, values)
            before = path.read_bytes()
            stack = PV.StackFile("f1", path, (4, 5, 1), ["ch0"])
            monkeypatch.setattr(models, "CellposeModel",
                                lambda **k: pytest.fail("a model was built"))
            call = {"psf_session": session, "channels_for_cellpose": (0,),
                    **kwargs}
            with pytest.raises(ValueError, match=message):
                PV.stream_masks_from_stack([stack], **call)
            assert path.read_bytes() == before
            assert stack.shape == (4, 5, 1) and stack.channels == ["ch0"]

    with subtests.test(msg="cellpose cannot be imported"):
        _plate, stacks = _stacks_for(tmp_path, name="noCellpose")
        import builtins

        real = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == "cellpose" or name.startswith("cellpose."):
                raise ImportError("no cellpose")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block)
        with pytest.raises(RuntimeError,
                           match="cellpose is required for v2 mask streaming"):
            PV.stream_masks_from_stack(stacks, model_name="cyto")


def test_stream_masks_from_stack_appends_exactly_one_mask_plane_per_field(
        tmp_path, subtests, monkeypatch, caplog):
    """Whatever shape the stack or the model's return takes (a list, a bare
    2-D array, one channel, a stack saved as a bare plane), every field ends
    as its own planes plus one uint16 mask plane, the ``StackFile`` and the
    sidecar say so, no scratch NPZ is ever written, and a missing sidecar
    costs only a warning."""
    def run(name, model, channels=(0, 1, 2), drop=False, batch_fields=1):
        monkeypatch.setattr("cellpose.models.CellposeModel", model)
        _plate, stacks = _stacks_for(tmp_path, name=name, channels=channels)
        originals = [np.load(s.path) for s in stacks]
        sidecar = stacks[0].path.parent / "channel_order.json"
        if drop:
            sidecar.unlink()
        returned = PV.stream_masks_from_stack(
            stacks, model_name="cyto", batch_fields=batch_fields)
        assert returned is stacks
        return stacks, originals, sidecar

    def assert_appended(stacks, originals, mask_max):
        for stack, original in zip(stacks, originals):
            arr = np.load(stack.path)
            n = original.shape[-1]
            assert arr.dtype == np.uint16
            assert arr.shape == (12, 12, n + 1)
            assert stack.shape == (12, 12, n + 1)
            assert stack.channels == [f"ch{i}" for i in range(n)] + ["mask"]
            np.testing.assert_array_equal(arr[..., :n], original)
            assert arr[..., -1].max() == mask_max

    cases = [
        ("three channels, list of masks", _model("square"), (0, 1, 2), 1),
        ("two channels, empty masks", _model("zeros"), (0, 1), 0),
        ("one channel", _model("square"), (0,), 1),
        ("bare 2-D mask for one image", _model("square", bare=True),
         (0, 1, 2), 1),
    ]
    for index, (case_id, model, channels, mask_max) in enumerate(cases):
        with subtests.test(msg=case_id):
            stacks, originals, sidecar = run(f"p{index}", model, channels)
            assert_appended(stacks, originals, mask_max)
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            assert meta["mask_channels"] == ["mask"]
            assert not (sidecar.parent / "_scratch").exists()

    with subtests.test(msg="no batch is written to disk"):
        stacks, _originals, sidecar = run("batches", _model("ones"),
                                          batch_fields=2)
        assert not (sidecar.parent / "_scratch").exists()
        assert not list(sidecar.parent.rglob("*.npz"))

    with subtests.test(msg="a stack saved as a bare plane"):
        monkeypatch.setattr("cellpose.models.CellposeModel", _model("ones"))
        merged = tmp_path / "bare" / "merged"
        merged.mkdir(parents=True)
        path = merged / "stack_f1.npy"
        np.save(path, np.zeros((12, 12), dtype=np.uint16))
        stack = PV.StackFile(field_id="f1", path=path, shape=(12, 12, 1),
                             channels=["ch0"])
        PV.stream_masks_from_stack([stack], model_name="cyto", batch_fields=1)
        written = np.load(path)
        assert written.shape == (12, 12, 2)
        assert not written[..., 0].any() and written[..., 1].min() == 1

    with subtests.test(msg="missing sidecar is a warning naming it"):
        caplog.clear()
        with caplog.at_level("WARNING", logger="spacr.pipeline_v2"):
            stacks, originals, sidecar = run("nosidecar", _model("ones"),
                                             drop=True, batch_fields=2)
        assert_appended(stacks, originals, 1)
        assert not sidecar.exists()
        assert any("channel_order.json" in r.getMessage()
                   for r in caplog.records)


def test_stream_masks_from_stack_normalises_like_v1_with_postprocess_settings(
        tmp_path, subtests, monkeypatch):
    """With postprocess settings Cellpose sees exactly V1's
    ``_normalize_img_batch`` of the selected planes (padded to the batch's
    largest field, fields keep their own shapes), and only a cell run maps
    the second selected plane to ``nucleus_channel``."""
    from spacr.io import _normalize_img_batch

    with subtests.test(msg="pixels equal V1's normalisation"):
        seen = []
        monkeypatch.setattr("cellpose.models.CellposeModel",
                            _model("zeros", seen=seen))
        _plate, stacks = _stacks_for(tmp_path, name="v1", channels=(0, 1))
        raw = [np.load(stack.path)[..., [1, 0]] for stack in stacks]
        PV.stream_masks_from_stack(
            stacks, model_name="cpsam", channels_for_cellpose=(1, 0),
            batch_fields=2, postprocess_settings=dict(_NORMALISE),
            object_type="cell")
        settings = dict(_NORMALISE, cell_channel=0, nucleus_channel=1,
                        pathogen_channel=None, organelle_channel=None)
        expected = _normalize_img_batch(np.stack(raw).copy(),
                                        channels=range(2),
                                        save_dtype=np.float32,
                                        settings=settings)
        assert np.array_equal(np.asarray(seen), expected)

    with subtests.test(msg="heterogeneous field shapes survive padding"):
        monkeypatch.setattr("cellpose.models.CellposeModel", _model("square"))
        _plate, stacks = _stacks_for(tmp_path, name="sizes", channels=(0, 1),
                                     sizes={1: 8, 2: 12})
        PV.stream_masks_from_stack(
            stacks, model_name="cpsam", channels_for_cellpose=(1, 0),
            batch_fields=2, postprocess_settings=dict(_NORMALISE),
            object_type="cell")
        assert [np.load(s.path).shape for s in stacks] == [(8, 8, 3),
                                                           (12, 12, 3)]

    with subtests.test(msg="nucleus slot only for a cell run"):
        import spacr.io as spacr_io

        roles = []

        def recording_normalise(*, stack, channels, save_dtype, settings, **kwargs):
            roles.append({key: settings[key] for key in
                          ("cell_channel", "nucleus_channel")})
            return np.asarray(stack, dtype=np.float32)

        monkeypatch.setattr(spacr_io, "_normalize_img_batch",
                            recording_normalise)
        monkeypatch.setattr("cellpose.models.CellposeModel", _model("ones"))
        for object_type in ("nucleus", "cell"):
            _plate, stacks = _stacks_for(tmp_path, name=f"role{object_type}",
                                         fields=(1,))
            PV.stream_masks_from_stack(
                stacks, model_name="cyto", batch_fields=1,
                channels_for_cellpose=(0, 1), object_type=object_type,
                postprocess_settings={"percentiles": [2, 98]})
        assert roles == [{"cell_channel": None, "nucleus_channel": 0},
                         {"cell_channel": 0, "nucleus_channel": 1}]


def test_direct_prefixed_backend_keeps_normalization_and_diameter_in_caller_settings(
        tmp_path, monkeypatch):
    """Direct V2 callers share one resolution context across backend settings copies."""
    from spacr import _segmentation_backends
    from spacr.settings import RESOLVED_PARAMETERS_KEY

    class Backend:
        """Provide the normal Cellpose 3 return structure without a model download."""

        def __init__(self):
            """Expose a named model and record its actual diameter calls."""
            self.model = 'cyto3'
            self.estimated_diameters = []
            self.diameters = []

        def eval(self, images, *, diameter=None, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
            """Return one spatial mask and full flow slots for every input."""
            self.channel_axes = getattr(self, 'channel_axes', []) + [channel_axis]
            self.diameters.append(diameter)
            return ([np.zeros(image.shape[:2], np.uint16) for image in images],
                    [[None, None, None, None] for image in images], None)

    model = Backend()

    def load_backend(name, **kwargs):
        """Keep backend routing real while replacing only the inference model."""
        assert name == 'cellpose3'
        assert kwargs['model_name'] == 'cellpose3:cyto3'
        return model

    monkeypatch.setattr(_segmentation_backends, '_load_backend', load_backend)
    _plate, stacks = _stacks_for(tmp_path, name='shared', channels=(0,))
    originals = [np.load(stack.path).copy() for stack in stacks]
    identities = [stack.field_id for stack in stacks]
    settings = dict(_NORMALISE)
    assert RESOLVED_PARAMETERS_KEY not in settings
    PV.stream_masks_from_stack(
        stacks, model_name='cellpose3:cyto3', channels_for_cellpose=(0,),
        diameter=17.25, batch_fields=2, postprocess_settings=settings,
        object_type='cell')
    fields = settings[RESOLVED_PARAMETERS_KEY]['fields']
    assert set(fields) == set(identities)
    for identity in identities:
        assert set(fields[identity]) == {'normalization:cell', 'diameter:cell'}
        assert fields[identity]['diameter:cell']['value'] == 17.25
        assert fields[identity]['normalization:cell']['value']['bounds']['0']
    assert model.diameters == [17.25]
    assert {key: value for key, value in settings.items()
            if key != RESOLVED_PARAMETERS_KEY} == _NORMALISE
    for stack, original in zip(stacks, originals):
        pixels = np.load(stack.path)
        np.testing.assert_array_equal(pixels[..., :1], original)
        assert not np.any(pixels[..., -1])


def test_run_v2_segments_a_plate_or_refuses_the_calibrated_route_early(
        tmp_path, subtests, monkeypatch):
    """The plain route maps, stacks and segments every field into
    ``merged/``; the calibrated route refuses illumination or non-cell runs
    before any output, refuses time series without decoding, reports an
    empty custom mapping as no fields, and refuses when every output name is
    taken, never touching the source files."""
    monkeypatch.setattr("cellpose.models.CellposeModel", _model("square"))
    for model_name, batch in (("cyto", 1), ("cpsam", 2)):
        with subtests.test(msg=f"plain run, {model_name}"):
            plate = _make_plate(tmp_path / model_name)
            result = PV.run_v2(plate, channels=(0, 1, 2), model_name=model_name,
                               metadata_type="cellvoyager", batch_fields=batch)
            assert sorted(result) == ["dst", "mapper", "stacks"]
            assert result["dst"] == plate / "merged"
            assert len(result["mapper"].records) == 6
            assert [s.path.name for s in result["stacks"]] == sorted(
                p.name for p in (plate / "merged").glob("stack_*.npy"))
            for stack in result["stacks"]:
                arr = np.load(stack.path)
                assert arr.shape == (12, 12, 4)
                assert stack.channels == ["ch0", "ch1", "ch2", "mask"]
                assert arr[..., -1].max() == 1

    def calibrated(name):
        source = _make_plate(tmp_path / name)
        return source, _snapshot(source)

    for invalid in ("illumination", "object"):
        with subtests.test(msg=f"calibrated refuses {invalid}"):
            source, before = calibrated(invalid)
            with pytest.raises(ValueError, match="cell objects"):
                PV.run_v2(
                    source, channels=(0, 1), channels_for_cellpose=(0,),
                    postprocess_settings=dict(_CALIBRATED),
                    illumination_settings=({"illumination_correction": True}
                                           if invalid == "illumination"
                                           else None),
                    object_type="nucleus" if invalid == "object" else "cell")
            assert _snapshot(source) == before
            assert sorted(p.name for p in source.iterdir() if p.is_dir()) == []

    with subtests.test(msg="calibrated refuses a time series"):
        source = _make_plate(tmp_path / "times")
        original = next(source.glob("*.tif"))
        original.with_name(original.name.replace("T01", "T02")).write_bytes(
            original.read_bytes())
        before = _snapshot(source)
        with monkeypatch.context() as patch:
            patch.setattr(PV, "_read_plane",
                          lambda *a, **k: pytest.fail("time series decoded"))
            with pytest.raises(ValueError, match="time series or Z stacks"):
                PV.run_v2(source, channels=(0, 1), channels_for_cellpose=(0,),
                          postprocess_settings=dict(_CALIBRATED))
        assert _snapshot(source) == before
        marker = (source / "calibrated_segmentation" / "psf"
                  / "segmentation_application.json")
        assert json.loads(marker.read_text())["complete"] is False
        assert not list((marker.parent.parent / "merged").glob("*.npy"))

    with subtests.test(msg="empty custom mapping"):
        source, before = calibrated("empty")
        result = PV.run_v2(
            source, channels=(0,), channels_for_cellpose=(0,),
            metadata_type="custom",
            custom_regex=r"^unmatched_(?P<channel>\d+)\.tif$",
            postprocess_settings=dict(_CALIBRATED))
        assert result["stacks"] == [] and result["mapper"].records == []
        root = result["output_root"]
        assert root == source / "calibrated_segmentation"
        assert result["dst"] == root / "merged"
        record = json.loads(
            (root / "psf" / "segmentation_application.json").read_text())
        assert record["complete"] is False
        assert not list(root.rglob("*.npy"))
        assert _snapshot(source) == before

    with subtests.test(msg="every calibrated output name taken"):
        source, before = calibrated("exhausted")
        occupied = [source / ("calibrated_segmentation" if index == 0 else
                              f"calibrated_segmentation_{index:04d}")
                    for index in range(10000)]
        for path in occupied:
            path.mkdir()
        owner = occupied[-1] / "preserved.csv"
        owner.write_bytes(b"original result owner")
        with monkeypatch.context() as patch:
            patch.setattr(PV, "_read_plane",
                          lambda *a, **k: pytest.fail("images decoded"))
            with pytest.raises(ValueError, match="No unused calibrated "
                                                 "segmentation output"):
                PV.run_v2(source, channels=(0, 1), channels_for_cellpose=(0,),
                          postprocess_settings=dict(_CALIBRATED))
        assert all(path.is_dir() for path in occupied)
        assert owner.read_bytes() == b"original result owner"
        assert not list(source.glob(
            "calibrated_segmentation*/psf/segmentation_application.json"))
        assert _snapshot(source) == before
