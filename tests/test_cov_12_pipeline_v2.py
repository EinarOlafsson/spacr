"""V2 streaming on the inputs that do not match the happy shape.

The V2 pipeline writes one npy per field and appends the mask channel back into
it, so anything it guesses wrong is baked into the file rather than reported.
These cover the guesses: a mapping reloaded without its sidecar, a field with
none of the requested channels, a Cellpose that hands back a bare array, and the
provenance recorder failing in each of the ways it is written to survive.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from spacr import pipeline_v2 as PV

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


def make_plate(dst: Path, channels=3, size=12) -> Path:
    import tifffile

    plate = dst / 'plate1'
    plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for field in (1, 2):
        for channel in range(channels):
            arr = rng.integers(0, 2000, size=(size, size)).astype(np.uint16)
            tifffile.imwrite(
                str(plate / f'plate1_A01_T01F0{field}L01A01Z01C0{channel}.tif'),
                arr)
    return plate


def test_a_mapping_reloaded_without_its_sidecar_says_the_regex_is_unknown(
        tmp_path):
    """A CSV with no companion JSON loads with placeholder provenance.

    The sidecar is what ``spacr repro`` replays; when it is missing the mapping
    must still load -- the file names are the data -- but it must not claim a
    metadata type or a regex it cannot know.
    """
    plate = make_plate(tmp_path)
    mapper = PV.FilenameMapper.discover(plate, metadata_type='cellvoyager')
    csv_path = mapper.save_csv(tmp_path / 'mapping.csv')
    csv_path.with_suffix('.json').unlink()

    reloaded = PV.FilenameMapper.load_csv(csv_path)
    assert reloaded.metadata_type == '?'
    assert reloaded.regex == ''
    assert len(reloaded.records) == len(mapper.records)


def test_a_field_with_none_of_the_requested_channels_gets_a_default_plane(
        tmp_path):
    """No readable channel means a 256x256 zero plane rather than a crash.

    ``np.stack`` needs every plane the same shape, and with the reference
    channel absent there is nothing to copy the shape from. A synthesised
    field is visibly empty downstream; a raised exception here would take the
    whole plate with it.
    """
    plate = make_plate(tmp_path)
    mapper = PV.FilenameMapper.discover(plate, metadata_type='cellvoyager')

    stacks = PV.stream_originals_to_stack(plate, mapper, channels=(7,))
    assert stacks
    for stack in stacks:
        arr = np.load(stack.path)
        assert arr.shape == (256, 256, 1)
        assert not arr.any()


def test_a_bare_mask_array_from_cellpose_is_still_appended(tmp_path,
                                                           monkeypatch):
    """A single 2-D mask is wrapped into a batch of one before it is written.

    Cellpose returns a bare array rather than a list when it is handed one
    image; iterating that array would treat each ROW as a field's mask and
    append nonsense to the stack.
    """
    class SingleMaskModel:
        def __init__(self, *args, **kwargs):
            self.pretrained_model = None

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            shape = np.asarray(x[0]).shape[:2]
            mask = np.zeros(shape, dtype=np.uint16)
            mask[1:4, 1:4] = 1
            return mask, None, None

    monkeypatch.setattr('cellpose.models.CellposeModel', SingleMaskModel)
    plate = make_plate(tmp_path)
    mapper = PV.FilenameMapper.discover(plate, metadata_type='cellvoyager')
    stacks = PV.stream_originals_to_stack(plate, mapper, channels=(0, 1, 2))

    PV.stream_masks_from_stack(stacks[:1], model_name='cyto', batch_fields=1)

    arr = np.load(stacks[0].path)
    assert arr.shape[-1] == 4
    assert arr[..., -1].max() == 1
    assert stacks[0].channels[-1] == 'mask'


def test_a_missing_channel_order_sidecar_does_not_fail_the_mask_stage(
        tmp_path, monkeypatch, caplog):
    """With no sidecar to update the masks are still written, and it is logged.

    The sidecar tells every later reader which plane is a mask. Failing the
    stage would throw away the segmentation; failing silently would leave a
    stack that describes itself wrongly, so the warning has to name the file.
    """
    class ConstantMaskModel:
        def __init__(self, *args, **kwargs):
            self.pretrained_model = None

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            return ([np.ones(np.asarray(image).shape[:2], dtype=np.uint16)
                     for image in x], None, None)

    monkeypatch.setattr('cellpose.models.CellposeModel', ConstantMaskModel)
    plate = make_plate(tmp_path)
    mapper = PV.FilenameMapper.discover(plate, metadata_type='cellvoyager')
    stacks = PV.stream_originals_to_stack(plate, mapper, channels=(0, 1, 2))
    sidecar = stacks[0].path.parent / 'channel_order.json'
    sidecar.unlink()

    with caplog.at_level('WARNING', logger='spacr.pipeline_v2'):
        PV.stream_masks_from_stack(stacks, model_name='cyto', batch_fields=2)

    assert np.load(stacks[0].path).shape[-1] == 4
    assert not sidecar.exists()
    assert any('channel_order.json' in record.getMessage()
               for record in caplog.records)


class TestCheckpointProvenance:
    """Every way the model fingerprint can fail is still only a warning."""

    def test_a_nested_list_of_checkpoints_is_recorded(self, tmp_path,
                                                      monkeypatch):
        """``model.cp.pretrained_model`` as a list has each entry hashed.

        Cellpose exposes the checkpoint one level down and as a list depending
        on the version; a list flattened to nothing leaves the manifest unable
        to name the weights that produced the masks.
        """
        import spacr.run_journal as rj

        first = tmp_path / 'a.pth'
        first.write_bytes(b'weights-a')
        inner = types.SimpleNamespace(pretrained_model=[str(first)])
        model = types.SimpleNamespace(pretrained_model=None, cp=inner)

        monkeypatch.setattr(rj, 'runs_root', lambda: tmp_path)
        with rj.open_run('mask', {'src': '/x'}) as run:
            PV._record_cellpose_hash(model, 'cyto')

        manifest = json.loads((run.dir / 'manifest.json').read_text())
        assert manifest['model_hashes']['cyto'].startswith('a.pth:')

    def test_a_journal_that_refuses_the_record_does_not_stop_segmentation(
            self, tmp_path, monkeypatch, caplog):
        """A raising run journal is logged and stepped over, not propagated.

        Provenance is not the result. Losing the whole segmentation because the
        manifest could not be written would be the expensive half failing for
        the cheap half.
        """
        import spacr.run_journal as rj

        ckpt = tmp_path / 'model.pth'
        ckpt.write_bytes(b'weights')

        def boom():
            raise RuntimeError('journal is closed')

        monkeypatch.setattr(rj, 'current_run', boom)
        with caplog.at_level('WARNING', logger='spacr.pipeline_v2'):
            PV._record_cellpose_hash(
                types.SimpleNamespace(pretrained_model=[str(ckpt)]), 'cyto')

        assert any('was not recorded in the run journal' in
                   record.getMessage() for record in caplog.records)

    def test_a_model_that_cannot_be_inspected_is_logged_not_raised(
            self, caplog):
        """An attribute lookup that raises leaves the run without provenance.

        A wrapped or lazily-loaded Cellpose model can raise from
        ``pretrained_model``; that must cost the manifest entry and nothing
        else.
        """
        class HostileModel:
            @property
            def pretrained_model(self):
                raise RuntimeError('model is not loaded')

        with caplog.at_level('WARNING', logger='spacr.pipeline_v2'):
            PV._record_cellpose_hash(HostileModel(), 'cyto')

        assert any('could not work out which checkpoint' in
                   record.getMessage() for record in caplog.records)


def test_a_single_plane_stack_gets_its_mask_appended(tmp_path, monkeypatch):
    """A stack saved as one plane must come back as plane plus mask.

    The marker came off once every field was promoted to (H, W, C) at load.
    Before that a single-plane stack was squeezed instead, which Cellpose
    rejects for a ``channel_axis=-1`` call, and the write-back then asked
    numpy to concatenate a (H, W, 1) mask onto a (H, W) array -- either way
    the whole batch was lost rather than one field.
    """
    class ConstantMaskModel:
        def __init__(self, *args, **kwargs):
            self.pretrained_model = None

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            return ([np.ones(np.asarray(image).shape[:2], dtype=np.uint16)
                     for image in x], None, None)

    monkeypatch.setattr('cellpose.models.CellposeModel', ConstantMaskModel)
    merged = tmp_path / 'merged'
    merged.mkdir()
    path = merged / 'stack_f1.npy'
    np.save(path, np.zeros((12, 12), dtype=np.uint16))
    stack = PV.StackFile(field_id='f1', path=path, shape=(12, 12, 1),
                         channels=['ch0'])

    PV.stream_masks_from_stack([stack], model_name='cyto', batch_fields=1)

    written = np.load(path)
    assert written.shape == (12, 12, 2)
    assert written[..., -1].max() == 1
