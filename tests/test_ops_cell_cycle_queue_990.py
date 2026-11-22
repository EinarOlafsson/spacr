"""Blind v3 context tiles keep exact identities, fixed channel bounds and annotation contracts."""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _runtime_readers():
    """Import runtime readers only for collected queue tests, keeping arrival adapters source-only."""
    global cc, read_table

    from spacr import ops_cell_cycle as cc
    from spacr.tabular import read_table


class _Source:
    """Synthetic unmasked fields with own nuclei and an explicit toy calibration and bar layout."""

    context = True
    pixel_size_um = 1.0

    def __init__(self, images):
        """Keep each channel-first synthetic field by label."""
        self.images = images

    def get_raw(self, row):
        """Return one unmasked field and only its own central nucleus."""
        image = self.images[row['label']]
        mask = np.zeros(image.shape[1:], bool)
        cy, cx = np.array(mask.shape) // 2
        mask[cy-2:cy+2, cx-2:cx+2] = True
        return image, mask


def _queue(labels=(301, 902)):
    """Return identifying names that must remain separate from every PNG basename."""
    return pd.DataFrame({'plate': 'secret_plate', 'well': 'B09', 'site': 147,
                         'label': labels, 'model_stage': ['metaphase'] * len(labels),
                         'rank': range(len(labels))})


def _write(tmp_path, queue, images, name='queue', **kwargs):
    """Write through the production queue operation and return mapping and PNG pixels."""
    folder = tmp_path / name
    cc.write_label_queue(queue, str(tmp_path), str(folder),
                         labels=str(tmp_path / 'answers.csv'), crop_source=_Source(images), **kwargs)
    mapping = read_table(str(folder / 'queue.csv'), canonicalise=False, report=None)
    pixels = []
    for row in mapping.itertuples():
        with Image.open(row.png_path) as image:
            pixels.append(np.asarray(image))
    return mapping, pixels


def test_blind_names_mapping_and_annotate_roundtrip_preserve_queue_order(tmp_path):
    """Opaque names reveal no identity while labels retain each exact source cell and queue order."""
    queue = _queue((902, 301))
    images = {key: np.full((2, 128, 128), key, dtype=np.uint16) for key in (301, 902)}
    mapping, _ = _write(tmp_path, queue, images)
    pd.testing.assert_frame_equal(mapping[cc.KEY], queue[cc.KEY])
    for row in mapping.itertuples():
        assert re.fullmatch(r'[0-9a-f]{32}\.png', row.image_name)
        assert Path(row.png_path).name == row.image_name
        assert all(text not in row.image_name for text in ('secret_plate', 'B09', 'metaphase', '_s', '_l'))
    with sqlite3.connect(tmp_path / 'queue/measurements/measurements.db') as db:
        db.execute('UPDATE png_list SET annotate = ? WHERE png_path = ?', (4, mapping.iloc[0].png_path))
        assert db.execute('SELECT COUNT(*) FROM png_list').fetchone()[0] == 2
    assert cc.collect_label_queue(str(tmp_path / 'queue'), user='agent') == 1
    answers = cc.read_cell_cycle_labels(str(tmp_path / 'answers.csv'))
    assert answers[cc.KEY].to_dict('records') == queue.iloc[[0]][cc.KEY].to_dict('records')
    assert answers.stage.tolist() == ['metaphase']


def test_repeated_labels_across_plates_and_wells_map_exactly(tmp_path):
    """Plate and well remain part of cell identity even when site and label coincide."""
    queue = _queue((301, 301, 301))
    queue['plate'] = ['P2', 'P1', 'P2']
    queue['well'] = ['B09', 'B09', 'A01']
    mapping, _ = _write(tmp_path, queue, {301:np.full((2,128,128),1500,np.uint16)})
    pd.testing.assert_frame_equal(mapping[cc.KEY], queue[cc.KEY])
    assert mapping.image_name.nunique() == 3
    with sqlite3.connect(tmp_path / 'queue/measurements/measurements.db') as db:
        db.execute('UPDATE png_list SET annotate = 4')
    assert cc.collect_label_queue(str(tmp_path / 'queue'), user='agent') == 3
    answers = cc.read_cell_cycle_labels(str(tmp_path / 'answers.csv'))
    assert set(map(tuple, answers[cc.KEY].to_numpy())) == set(map(tuple, queue[cc.KEY].to_numpy()))


def test_channel_selection_uses_the_two_chosen_planes(tmp_path):
    """Other stains can occupy either slot, while the composite red channel remains zero."""
    image = np.stack([np.full((128,128),value,np.uint16) for value in (100,200,300)])
    _, (tile,) = _write(tmp_path, _queue((301,)), {301:image}, channels=[2,0],
                        ranges=[(0,400),(0,800)])
    assert tile[145,145].tolist() == [191,191,191]
    assert tile[145,292+145].tolist() == [32,32,32]
    assert tile[145,584+145].tolist() == [0,32,191]
    metadata=json.loads((tmp_path/'queue/queue.json').read_text())
    assert metadata['channels'] == [2,0] and metadata['ranges'] == [[0,400],[0,800]]


@pytest.mark.parametrize('kwargs, message', [
    ({'radius':-1},'radius'), ({'radius':2.5},'radius'), ({'radius':True},'radius'),
    ({'channels':[]},'channel'), ({'channels':[0,0]},'channel'),
    ({'channels':[0,1,2]},'channel'), ({'channels':[5]},'absent'),
    ({'ranges':[(1,1),(0,1)]},'range'), ({'ranges':[(0,np.nan)]},'range'),
    ({'ranges':[(0,1)]},'range'),
])
def test_invalid_display_recipes_are_refused(tmp_path, kwargs, message):
    """Invalid bounds, selections and dilation radii give actionable errors."""
    with pytest.raises(ValueError, match=message):
        _write(tmp_path, _queue((301,)), {301:np.zeros((2,128,128),np.uint16)}, **kwargs)


def test_funk_ranges_are_fixed_and_other_pixel_units_require_bounds(tmp_path):
    """The Funk defaults are recorded exactly and never inferred from selected cells."""
    image=np.full((2,128,128),1500,np.uint16)
    _write(tmp_path, _queue((301,)), {301:image})
    recipe=json.loads((tmp_path/'queue/queue.json').read_text())
    assert recipe['ranges'] == [[550,23039],[1084,5002]]
    with pytest.raises(ValueError,match='raw uint16'):
        _write(tmp_path,_queue((301,)),{301:image.astype(float)},name='float')
    _write(tmp_path,_queue((301,)),{301:image.astype(float)},name='float-explicit',
           ranges=[(0,2000),(0,3000)])


def test_native_context_pool_and_non_ops_nucleus_table_adapter_share_pixels(tmp_path):
    """Default native OPS context and a nucleus-table source render the same unchanged field pixels."""
    from spacr.crop_store import CropPoolWriter, OmeZarrCropSource, committed_rows

    pixels=np.full((2,128,128),1500,np.uint16)
    nucleus=np.zeros((128,128),bool)
    nucleus[62:66,62:66]=True
    attrs={'pixel_size_um':1.0}
    writer=CropPoolWriter(tmp_path,'secret_plate','B09','cell',2,mask_dtype=np.uint8,
        attributes={**attrs,'source':'ops_ph_crops','options':{'margin_px':64,'zero_outside':False}})
    writer.add([{'object_id':'native_cell','ops_object_id':None,'prcfo':'native_cell',
        'ops_crop':True,'object_label':301,'label':301,'field':147,'site':147,'has_mask':True,
        'pixels':pixels,'mask':nucleus.astype(np.uint8)*2,'bbox_y0':0,'bbox_x0':0,
        'bbox_y1':128,'bbox_x1':128,'margin_px':64,'clipped_at_edge':False}])
    writer.commit()
    cc.write_label_queue(_queue((301,)),str(tmp_path),str(tmp_path/'native'),labels='unused.csv')
    native=read_table(str(tmp_path/'native/queue.csv'),canonicalise=False,report=None)
    with Image.open(native.iloc[0].png_path) as image:
        expected=np.array(image)
    writer=CropPoolWriter(tmp_path/'external','secret_plate','B09','nucleus',2,attributes=attrs)
    writer.add([{'object_id':'external_nucleus','object_label':301,'field':147,
        'pixels':pixels,'mask':nucleus,'bbox_y0':0,'bbox_x0':0,'bbox_y1':128,'bbox_x1':128,
        'margin_px':64,'clipped_at_edge':False}])
    writer.commit()
    table=pd.DataFrame(committed_rows(tmp_path/'external'))
    queue=_queue((301,)).merge(table.drop(columns=['plate','well']),
                               left_on=['site','label'],right_on=['field','object_label'])
    source=OmeZarrCropSource(tmp_path/'external')
    source.pixel_size_um=1.0
    cc.write_label_queue(queue,str(tmp_path/'external'),str(tmp_path/'adapted'),
                         labels='unused.csv',crop_source=source)
    adapted=read_table(str(tmp_path/'adapted/queue.csv'),canonicalise=False,report=None)
    with Image.open(adapted.iloc[0].png_path) as image:
        assert np.array_equal(expected,np.array(image))


def test_channel_setting_arrival_and_job_pass_through(tmp_path, monkeypatch):
    """The new setting is alpha, typed, grouped and passed from the Home job into its one writer."""
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TYPES, OPS_CATEGORIES, OPS_TOOLTIPS
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS
    from spacr.settings import ALPHA_FEATURES
    from types import SimpleNamespace

    key = 'ops_cell_cycle_label_channels'
    assert OPS_DEFAULTS[key] is None
    assert OPS_TYPES[key] == (list, type(None))
    assert key in OPS_CATEGORIES['Cell-cycle stages'] and key in ALPHA_FEATURES[990]['settings']
    assert 'DNA first' in OPS_TOOLTIPS[key]
    assert SETTING_API_TARGETS[key] == ('spacr.ops_cell_cycle', 'run_cell_cycle_phase', True)
    calls = []

    def write(queue, root, folder, **kwargs):
        """Record the concrete display controls supplied by the job opener."""
        calls.append(kwargs)

    def read(*args, **kwargs):
        """Return the selected synthetic cell without file reads."""
        return _queue((301,))

    from spacr import tabular

    monkeypatch.setattr(cc, 'write_label_queue', write)
    monkeypatch.setattr(tabular, 'read_table', read)
    window = SimpleNamespace(open_module=lambda key: key,
                              _screens={'annotate': SimpleNamespace(_open_source=lambda folder: None)})
    cc.open_label_job(window, {'folder': str(tmp_path), 'pending': 'unused.csv', 'crop_root': '',
                               'labels': 'answers.csv', 'channels': [2, 0],
                               'ranges': [[0, 200], [0, 500]], 'radius': 17})
    assert calls == [{'labels': 'answers.csv', 'channels': [2, 0],
                      'ranges': [[0, 200], [0, 500]], 'radius': 17, 'crop_source': None}]


def _current_queue_api(entries):
    """Apply one exact queue API successor using only its source docstring and AST signature."""
    import ast
    import hashlib
    import inspect

    root = Path(__file__).resolve().parents[1]
    path = root / 'tests/data/release_contracts/990queue_stage_tiles_api_changes_2026-11-21.json'
    changes = json.loads(path.read_text())
    key = 'spacr.ops_cell_cycle.write_label_queue'
    assert set(changes) == {key}
    change = changes[key]
    assert hashlib.sha256(entries[key].encode()).hexdigest() == change['before_sha256']
    tree = ast.parse((root / 'spacr/ops_cell_cycle.py').read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'write_label_queue')
    assert inspect.cleandoc(ast.get_docstring(owner, clean=False)).strip() == change['docstring']
    args = owner.args
    expected = change['signature']
    assert not args.posonlyargs and args.vararg is None and args.kwarg is None
    assert [value.arg for value in args.args + args.kwonlyargs] == expected['parameters']
    positional_required = args.args[:len(args.args) - len(args.defaults)]
    required = [value.arg for value in positional_required]
    required += [value.arg for value, default in zip(args.kwonlyargs, args.kw_defaults)
                 if default is None]
    assert required == expected['required']
    assert [value.arg for value in args.kwonlyargs] == expected['keyword_only']
    defaults = {value.arg: ast.literal_eval(default)
                for value, default in zip(args.args[len(positional_required):], args.defaults)}
    defaults.update({value.arg: ast.literal_eval(default)
                     for value, default in zip(args.kwonlyargs, args.kw_defaults)
                     if default is not None})
    assert defaults == expected['defaults']
    assert all(type(defaults[name]) is type(value)
               for name, value in expected['defaults'].items())
    return {**entries, key: change['docstring']}


def test_api_arrival_keeps_the_exact_historical_boundary():
    """Only the queue writer can replace its original arrival, with a checked source hash."""
    path = Path(__file__).parent / 'data/release_contracts/990_ops_cell_cycle_api_arrivals_2026-11-21.json'
    original = json.loads(path.read_text())
    current = _current_queue_api(original)
    assert len(current) == len(original) == 20
    assert {key for key in original if original[key] != current[key]} == {
        'spacr.ops_cell_cycle.write_label_queue'}
    corrupted = {**original, 'spacr.ops_cell_cycle.write_label_queue': 'changed original'}
    with pytest.raises(AssertionError):
        _current_queue_api(corrupted)


def test_setting_arrival_is_exact():
    """The setting arrival binds its default, types, tooltip, panel and API owner."""
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TYPES, OPS_TOOLTIPS, OPS_CATEGORIES
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS

    path = Path(__file__).parent / 'data/release_contracts/990queue_stage_tiles_setting_arrival_2026-11-21.json'
    arrival = json.loads(path.read_text())
    key = arrival['key']
    assert OPS_DEFAULTS[key] == arrival['default']
    assert [value.__name__ for value in OPS_TYPES[key]] == arrival['types']
    assert OPS_TOOLTIPS[key] == arrival['tooltip']
    assert key in OPS_CATEGORIES[arrival['category']]
    assert list(SETTING_API_TARGETS[key]) == arrival['api_target']


def test_api_signature_binding_rejects_a_changed_radius_default(monkeypatch):
    """A changed optional default cannot pass the documented-parameter and source-text checks."""
    root = Path(__file__).resolve().parents[1]
    path = root / 'tests/data/release_contracts/990_ops_cell_cycle_api_arrivals_2026-11-21.json'
    original = json.loads(path.read_text())
    source = root / 'spacr/ops_cell_cycle.py'
    read = Path.read_text

    def changed(path, *args, **kwargs):
        """Change only the apparent source default while preserving source prose and names."""
        text = read(path, *args, **kwargs)
        if path == source:
            assert 'radius: int = 30, crop_source' in text
            return text.replace('radius: int = 30, crop_source',
                                'radius: int = 31, crop_source', 1)
        return text

    monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(AssertionError):
        _current_queue_api(original)


def _current_inherited_module_api(entries):
    """Align only the exact inherited module prose using its source and both source-bound hashes."""
    import ast
    import hashlib
    import inspect

    root = Path(__file__).resolve().parents[1]
    path = root / 'tests/data/release_contracts/990queue_inherited_module_doc_alignment_2026-11-21.json'
    changes = json.loads(path.read_text())
    key = 'spacr.ops_cell_cycle'
    assert set(changes) == {key}
    change = changes[key]
    assert hashlib.sha256(entries[key].encode()).hexdigest() == change['before_sha256']
    assert hashlib.sha256(change['docstring'].encode()).hexdigest() == change['after_sha256']
    tree = ast.parse((root / 'spacr/ops_cell_cycle.py').read_text())
    assert inspect.cleandoc(ast.get_docstring(tree, clean=False)).strip() == change['docstring']
    assert change['unchanged_source_base'] == '79343f5b59f62507adce96feb68b912ebfa0222f'
    return {**entries, key: change['docstring']}


def test_inherited_module_alignment_changes_only_the_known_annotation():
    """Only the removed historical date annotation is aligned; every other API entry stays exact."""
    path = Path(__file__).parent / 'data/release_contracts/990_ops_cell_cycle_api_arrivals_2026-11-21.json'
    original = json.loads(path.read_text())
    aligned = _current_inherited_module_api(original)
    key = 'spacr.ops_cell_cycle'
    assert len(aligned) == len(original) == 20
    assert {name for name in original if aligned[name] != original[name]} == {key}
    assert original[key].replace('; maintainer 2026-11-20', '') == aligned[key]
    with pytest.raises(AssertionError):
        _current_inherited_module_api({**original, key: original[key] + ' extra'})
    other = 'spacr.ops_cell_cycle.collect_label_queue'
    preserved = _current_inherited_module_api({**original, other: 'not normalized'})
    assert preserved[other] == 'not normalized'
