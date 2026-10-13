"""CSV plate dilution preflight and real batch/preview persistence boundaries."""
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import plaque


@pytest.mark.parametrize('value', [1, 0.001, '100', '', None, {'plate': 100}])
def test_existing_scalar_and_dictionary_inputs_are_preserved(value):
    assert plaque._load_colony_dilutions(value) is value


def test_csv_bom_quoting_filename_priority_and_missing_plate(tmp_path):
    path = tmp_path / 'dilutions.csv'
    path.write_text('file,dilution,note\n"pläte,1.tif",1e-4,one\nplate,100,two\nplate.tif,200,three\n',
                    encoding='utf-8-sig')
    before = path.read_bytes()
    mapping = plaque._load_colony_dilutions(path)
    assert mapping == {'pläte,1.tif': 0.0001, 'plate': 100, 'plate.tif': 200}
    assert plaque._dilution_for('plate.tif', mapping) == 200
    assert plaque._dilution_for('plate.png', mapping) == 100
    assert plaque._dilution_for('unlisted.tif', mapping) is None
    assert plaque._cfu_per_ml(2, plaque._dilution_for('pläte,1.tif', path), 50) == 400000
    assert path.read_bytes() == before


@pytest.mark.parametrize('content', [
    '', 'file,dilution\n', 'filename,dilution\np,1\n',
    'file,dilution,dilution\np,1,2\n', 'file,dilution,\np,1,x\n',
    'file,dilution\np,100\np,100\n', 'file,dilution\np,100\np,200\n',
    'file,dilution\np\n', 'file,dilution\np,100,extra\n',
    'file,dilution\n,100\n', 'file,dilution\n../p,100\n',
    'file,dilution\nfolder\\p,100\n', 'file,dilution\n.,100\n',
    'file,dilution\np,100\n"unterminated,1\n',
])
def test_malformed_csv_is_rejected_completely(tmp_path, content):
    path = tmp_path / 'bad.csv'
    path.write_text(content)
    with pytest.raises(ValueError, match='colony_dilution CSV'):
        plaque._load_colony_dilutions(path)


@pytest.mark.parametrize('dilution', ['', 'no', 'True', '0', '-1', 'nan', 'inf', '1e-320'])
def test_invalid_later_dilution_prevents_even_first_plate_processing(tmp_path, monkeypatch, dilution):
    path = tmp_path / 'bad.csv'
    path.write_text(f'file,dilution\np,100\nq,{dilution}\n')

    def forbidden(*args, **kwargs):
        pytest.fail('image processing began before complete dilution validation')

    monkeypatch.setattr(plaque, '_find_dish', forbidden)
    with pytest.raises(ValueError, match='row 3: dilution'):
        plaque._count_colony_plate(np.zeros((8, 8)), name='p.tif',
                                  settings={'colony_dilution': str(path)})


@pytest.mark.parametrize('kind', ['encoding', 'oversize', 'missing'])
def test_unreadable_or_oversized_csv_is_actionable(tmp_path, kind):
    path = tmp_path / 'bad.csv'
    if kind == 'encoding':
        path.write_bytes(b'file,dilution\np,\xff\n')
    elif kind == 'oversize':
        path.write_bytes(b'x' * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match='colony_dilution CSV'):
        plaque._load_colony_dilutions(path)


@pytest.fixture
def two_colonies(monkeypatch):
    def segment(crop, *, centre, radius, **kwargs):
        labels = np.zeros(crop.shape[:2], dtype=np.int32)
        labels[4:8, 4:8] = 1
        labels[16:20, 16:20] = 2
        return {'labels': labels, 'factor': 1.0, 'polarity': 'dark',
                'centre': centre, 'radius': radius}

    monkeypatch.setattr(plaque, '_find_dish', lambda image: (
        plaque.Well(0, 0, image.shape[1], image.shape[0]), 'frame'))
    monkeypatch.setattr(plaque, '_segment_colonies', segment)


def test_real_batch_persists_one_csv_snapshot_and_preview_agrees(tmp_path, monkeypatch, two_colonies):
    from spacr import submodules
    from spacr.qt.widgets import plaque_preview as preview

    monkeypatch.setattr(submodules, '_resolve_well_detector', lambda settings: None)
    originals = {}
    for name in ['p1.tif', 'p2.tif', 'unlisted.tif']:
        path = tmp_path / name
        tifffile.imwrite(path, np.arange(1024, dtype=np.uint16).reshape(32, 32))
        originals[name] = path.read_bytes()
    csv_path = tmp_path / 'dilutions.csv'
    source = 'file,dilution\np1.tif,100\np2,0.001\n'
    csv_path.write_text(source)
    settings = {'src': str(tmp_path), 'save': False, 'well_detection': False,
                'colony_dilution': str(csv_path), 'colony_plated_volume_ul': 50}
    original_count = plaque._count_colony_plate
    calls = []

    def count(*args, **kwargs):
        result = original_count(*args, **kwargs)
        calls.append(kwargs['name'])
        if len(calls) == 1:
            csv_path.write_text('file,dilution\np1.tif,900\np2,900\n')
        return result

    monkeypatch.setattr(plaque, '_count_colony_plate', count)
    result = submodules._analyze_colony_plates(settings).set_index('file')
    assert calls == ['p1.tif', 'p2.tif', 'unlisted.tif']
    assert result.loc['p1.tif', 'cfu_per_ml'] == 4000
    assert result.loc['p2.tif', 'cfu_per_ml'] == 40000
    assert pd.isna(result.loc['unlisted.tif', 'cfu_per_ml'])
    assert settings['colony_dilution'] == str(csv_path)
    with sqlite3.connect(tmp_path / 'colonies' / 'colonies.db') as connection:
        saved = pd.read_sql_query('SELECT * FROM per_plate', connection).set_index('file')
        assert connection.execute('SELECT count(*) FROM per_colony').fetchone()[0] == 6
    exported = pd.read_csv(tmp_path / 'colonies' / 'per_plate.csv').set_index('file')
    for table in [saved, exported]:
        np.testing.assert_allclose(table['cfu_per_ml'], result['cfu_per_ml'], equal_nan=True)
        np.testing.assert_allclose(table['dilution'], result['dilution'], equal_nan=True)
    csv_path.write_text(source)
    monkeypatch.setattr(plaque, '_count_colony_plate', original_count)
    shown = preview._colony_preview_pass(tmp_path / 'p2.tif', settings)
    assert shown['count'] == 2
    assert shown['colony_summaries'][0]['cfu_per_ml'] == 40000
    assert shown['colony_summaries'][0]['dilution'] == 1000
    for name, content in originals.items():
        assert (tmp_path / name).read_bytes() == content


@pytest.mark.parametrize('existing_outputs', [False, True])
def test_batch_invalid_csv_preserves_outputs_and_never_resolves_models(tmp_path, monkeypatch, existing_outputs):
    from spacr import submodules

    path = tmp_path / 'bad.csv'
    path.write_text('file,dilution\np1,100\np2,nan\n')
    out = tmp_path / 'colonies'
    if existing_outputs:
        out.mkdir()
        (out / 'colonies.db').write_bytes(b'previous database')
        (out / 'per_plate.csv').write_bytes(b'previous results')
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('model resolution began before dilution preflight')

    monkeypatch.setattr(submodules, '_resolve_well_detector', forbidden)
    with pytest.raises(ValueError, match='row 3: dilution'):
        submodules._analyze_colony_plates({'src': str(tmp_path), 'colony_dilution': path})
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    assert out.exists() == existing_outputs


def test_preview_invalid_csv_precedes_detector_resolution(tmp_path, monkeypatch):
    from spacr.qt.widgets import plaque_preview as preview

    path = tmp_path / 'bad.csv'
    path.write_text('file,dilution\np1,100\np2,inf\n')

    def forbidden(*args, **kwargs):
        pytest.fail('detector resolution began before dilution preflight')

    monkeypatch.setattr(preview, 'resolve_detector', forbidden)
    with pytest.raises(ValueError, match='row 3: dilution'):
        preview._colony_preview_pass(tmp_path / 'nonexistent.tif', {
            'colony_dilution': path, 'colony_detector': 'never-resolve'})
