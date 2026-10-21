"""Capture the real phase dataset/trainer boundary without training a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from spacr import deep_spacr, measure
from spacr.classifier_evaluation import split_group_values


@pytest.fixture
def phase_case(tmp_path, monkeypatch):
    records = []
    for plate in range(4):
        for well in range(3):
            for field in range(2):
                for phase in ('G1', 'S'):
                    number = len(records)
                    records.append(dict(
                        plateID=f'screen_day_{plate}', rowID='r1',
                        columnID=f'c{well}', fieldID=f'f{field}',
                        object_label=number + 1,
                        prcfo=f'screen_day_{plate}_r1_c{well}_f{field}_o{number}',
                        phase=phase))
    frame = pd.DataFrame(records, index=np.arange(len(records)) * 3 + 101)
    labels = frame.pop('phase').copy()
    originals = tmp_path / 'originals'
    originals.mkdir()
    paths = pd.Series(index=frame.index, dtype=object)
    for number, index in enumerate(frame.index):
        path = originals / f'{frame.at[index, "prcfo"]}.png'
        Image.fromarray(np.full((4, 5, 3), number, np.uint8)).save(path)
        paths.at[index] = str(path)
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    calls = []
    monkeypatch.setattr(measure, '_nucleus_crop_size', lambda _: 32)
    monkeypatch.setattr(measure, '_write_nucleus_crops', lambda *a, **kw: paths.copy())

    def train(settings):
        captured = dict(settings)
        captured['assigned'] = {
            split: {p.name for p in (Path(settings['src']) / split).glob('*/*.png')}
            for split in ('train', 'test')}
        calls.append(captured)
        model = tmp_path / 'fake_trainer' / 'model.pth'
        model.parent.mkdir(exist_ok=True)
        model.write_bytes(b'not model weights; intercepted trainer')
        return str(model)

    def apply(folder, model, **kwargs):
        return pd.DataFrame({'path': paths.dropna().tolist(),
                             'pred': np.full(paths.notna().sum(), 0.75)})

    monkeypatch.setattr(deep_spacr, 'train_test_model', train)
    monkeypatch.setattr(deep_spacr, 'apply_model', apply)
    measured = pd.DataFrame(index=frame.index)
    settings = {'cell_cycle_epochs': 1}
    work = tmp_path / 'private_work'

    def run():
        saved_frame = frame.copy(deep=True)
        saved_labels = labels.copy(deep=True)
        saved_settings = dict(settings)
        result = measure._phases_by_torch(
            str(tmp_path), frame, measured, labels, settings, channel=0,
            work=str(work))
        pd.testing.assert_frame_equal(frame, saved_frame)
        pd.testing.assert_series_equal(labels, saved_labels)
        assert settings == saved_settings
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                   for p, digest in before.items())
        return result

    return frame, labels, paths, settings, calls, work, run, before


@pytest.mark.parametrize('requested,expected', [
    ('default', 'well'), ('well', 'well'), ('field', 'field'),
    ('plate', 'plate'), ('cell', 'cell'), (None, 'cell')])
def test_actual_torch_dataset_and_trainer_use_requested_grouping(phase_case, requested, expected):
    frame, labels, paths, settings, calls, work, run, before = phase_case
    if requested != 'default':
        settings['cv_group_by'] = requested
    phases, confidence, report = run()
    assert len(calls) == 1
    captured = calls[0]
    name_to_index = {Path(path).name: index for index, path in paths.items()}
    positions = {
        split: [frame.index.get_loc(name_to_index[name]) for name in names]
        for split, names in captured['assigned'].items()}
    assert captured['assigned']['train'].isdisjoint(captured['assigned']['test'])
    assert captured['assigned']['train'] | captured['assigned']['test'] == set(name_to_index)
    level, groups = split_group_values(frame=frame, group_by=expected)
    assert set(groups[positions['train']]).isdisjoint(groups[positions['test']])
    assert captured['cv_group_by'] == report['split_group_by'] == level
    assert report['split']['group_by'] == level
    assert report['split']['train_cells'] == len(positions['train'])
    assert report['split']['test_cells'] == len(positions['test'])
    assert report['split']['total_groups'] == len(set(groups))
    metadata = json.loads(Path(report['model']).with_name('cell_cycle_phases.json').read_text(encoding="utf-8"))
    assert metadata['split_group_by'] == expected
    assert metadata['split'] == report['split']
    assert captured['random_seed'] == 42
    assert captured['test'] is True
    assert phases.index.equals(frame.index) and confidence.index.equals(frame.index)
    for split in ('train', 'test'):
        selected = [name_to_index[name] for name in captured['assigned'][split]]
        assert set(labels.loc[selected]) == {'G1', 'S'}


@pytest.mark.parametrize('design', ['one_well', 'class_confounded', 'missing_identity', 'unknown_level'])
def test_unprovable_holdout_refuses_before_trainer_and_dataset_replacement(phase_case, design):
    frame, labels, paths, settings, calls, work, run, before = phase_case
    if design == 'one_well':
        frame.loc[:, ['plateID', 'rowID', 'columnID']] = ['p', 'r1', 'c1']
    elif design == 'class_confounded':
        frame.loc[:, 'plateID'] = 'p'
        frame.loc[:, 'columnID'] = labels.map({'G1': 'c1', 'S': 'c2'})
    elif design == 'missing_identity':
        frame.loc[frame.index[0], 'plateID'] = None
    else:
        settings['cv_group_by'] = 'unknown'
    previous = work / 'dataset' / 'previous.txt'
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b'previous private dataset')
    with pytest.raises(ValueError):
        run()
    assert calls == []
    assert previous.read_bytes() == b'previous private dataset'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
               for p, digest in before.items())


def test_only_labelled_existing_crops_participate_in_shared_split(phase_case):
    frame, labels, paths, settings, calls, work, run, before = phase_case
    labels.iloc[0] = None
    paths.iloc[1] = None
    frame.loc[frame.index[:2], 'plateID'] = None
    run()
    assert len(calls) == 1
    assigned = calls[0]['assigned']['train'] | calls[0]['assigned']['test']
    assert len(assigned) == len(frame) - 2
    assert report_count(calls[0]['src']) == len(frame) - 2


def report_count(dataset):
    return len(list(Path(dataset).glob('*/*/*.png')))


def test_saved_model_application_does_not_invent_new_split(phase_case, tmp_path):
    frame, labels, paths, settings, calls, work, run, before = phase_case
    model = tmp_path / 'saved' / 'model.pth'
    model.parent.mkdir()
    model.write_bytes(b'intercepted saved model')
    model.with_name('cell_cycle_phases.json').write_text(json.dumps(
        {'phases': ['G1', 'S'], 'crop_size': 32}), encoding="utf-8")
    settings['cell_cycle_model'] = str(model)
    settings['cv_group_by'] = 'not consulted for existing model'
    frame.loc[:, 'plateID'] = None
    phases, confidence, report = run()
    assert calls == []
    assert 'split' not in report and 'split_group_by' not in report
    assert report['model'] == str(model)
    assert phases.notna().all() and confidence.notna().all()


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_actual_torch_holdout_refusal_uses_existing_reviewed_translation(phase_case, monkeypatch, language):
    frame, labels, paths, settings, calls, work, run, before = phase_case
    frame.loc[:, ['plateID', 'rowID', 'columnID']] = ['p', 'r1', 'c1']
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    source = 'The requested acquisition groups cannot place every class in both training and held-out data.'
    reviews = Path(__file__).resolve().parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-14-cell-cycle-grouped-evaluation.json'
    record = next(row for row in json.loads(reviews.read_text(encoding="utf-8"))['records'] if row['source'] == source)
    assert record['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    with pytest.raises(ValueError) as error:
        run()
    assert str(error.value) == record['translation']
    assert isinstance(error.value.__cause__, ValueError)
    assert calls == []
