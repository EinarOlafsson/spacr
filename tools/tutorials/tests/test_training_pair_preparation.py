"""Pin the tutorial's cell compartment and preserve historical datasets."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
import tifffile

PATH = Path(__file__).resolve().parents[1] / 'authoring/tools/prepare_train_cellpose_tutorial_data.py'
spec = importlib.util.spec_from_file_location('tutorial_cell_training_preparation', PATH)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def original(tmp_path, *, cell_plane=4, dtype=np.uint16):
    source = tmp_path / 'original/merged'; source.mkdir(parents=True)
    settings = source.parent / 'settings'; settings.mkdir()
    (settings/'measure_crop_settings.csv').write_text(
        f'Key,Value\ncell_mask_dim,{cell_plane}\nnucleus_mask_dim,5\npathogen_mask_dim,6\n')
    (settings/'gen_mask_settings.csv').write_text('Key,Value\ncell_channel,1\n')
    data = np.zeros((4,4,7),dtype=dtype)
    data[1:3,1:3,1] = [[11,12],[13,14]]
    data[1:3,1:3,4] = [[0,10],[20,30]]
    data[1:3,1:3,6] = [[101,102],[103,104]]
    np.save(source/'field.npy',data)
    return source


def prepare(source,destination,**kwargs):
    return preparation.prepare(source,destination,pairs=(('field.npy',1,1),),crop_size=2,**kwargs)


def test_exact_cell_image_mask_pair_is_not_pathogen_and_preserves_source(tmp_path):
    source=original(tmp_path);before=preparation.digest(source/'field.npy')
    destination=tmp_path/'corrected'
    manifest=prepare(source,destination)
    image=tifffile.imread(destination/'train/images/cell_pair_01.tif')
    mask=tifffile.imread(destination/'train/masks/cell_pair_01.tif')
    assert image.tolist()==[[11,12],[13,14]]
    assert mask.tolist()==[[0,10],[20,30]]
    assert mask.tolist()!=[[101,102],[103,104]]
    assert image.dtype==mask.dtype==np.uint16
    assert preparation.digest(source/'field.npy')==before
    assert manifest['mask_plane']==4 and manifest['image_channel']==1
    assert manifest['pairs'][0]['objects']==3
    assert manifest['independent_annotation_review'] is False
    assert manifest['held_out_accuracy_validated'] is False
    assert json.loads((destination/'source_manifest.json').read_text())==manifest
    assert json.loads((destination/'tutorial_settings.json').read_text())['src']==str(destination)


def test_changed_recorded_compartment_refused_after_valid_counterpart(tmp_path):
    good=original(tmp_path/'good');assert prepare(good,tmp_path/'good-output')['mask_plane']==4
    bad=original(tmp_path/'bad',cell_plane=6)
    with pytest.raises(ValueError,match='Recorded compartment mapping changed'):
        prepare(bad,tmp_path/'bad-output')
    assert not (tmp_path/'bad-output').exists()


def test_existing_destination_and_its_real_file_preserved(tmp_path):
    source=original(tmp_path);destination=tmp_path/'retained'
    prepare(source,destination)
    existing=destination/'train/masks/cell_pair_01.tif'
    before=existing.read_bytes()
    with pytest.raises(FileExistsError):prepare(source,destination)
    assert existing.read_bytes()==before


def test_wrong_dtype_refused_after_uint16_counterpart(tmp_path):
    good=original(tmp_path/'good');prepare(good,tmp_path/'good-output')
    bad=original(tmp_path/'bad',dtype=np.float32)
    with pytest.raises(ValueError,match='uint16'):
        prepare(bad,tmp_path/'bad-output')


def test_incomplete_crop_refused_after_complete_counterpart(tmp_path):
    source=original(tmp_path);prepare(source,tmp_path/'complete')
    with pytest.raises(ValueError,match='incomplete crop'):
        preparation.prepare(source,tmp_path/'incomplete',pairs=(('field.npy',3,3),),crop_size=2)
