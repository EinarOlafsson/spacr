"""Training samples retain their file pairs and report every download ending."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QWidget

from spacr.qt import make_masks_datasets as md


@pytest.fixture
def sample(tmp_path, monkeypatch):
    dataset = replace(md.MASK_DATASETS[0], counts='')
    cache = tmp_path / 'remote-cache'
    cache.mkdir()
    remote = {}
    for stem in ('a', 'b'):
        for folder in ('images', 'masks'):
            name = f'{folder}/{stem}.tif'
            path = cache / f'{folder}-{stem}.tif'
            path.write_bytes(name.encode())
            remote[name] = path
    requests = []

    def download(repo, name, *, repo_type):
        assert repo == dataset.repo and repo_type == 'dataset'
        requests.append(name)
        return str(remote[name])

    def listing(repo, *, repo_type):
        assert repo == dataset.repo and repo_type == 'dataset'
        return list(remote)

    api = SimpleNamespace(list_repo_files=listing)
    hub = SimpleNamespace(HfApi=lambda: api, hf_hub_download=download)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    worker = md._SampleWorker(dataset, tmp_path / 'sample')
    endings, progress = [], []
    worker.finished.connect(lambda *values: endings.append(values))
    worker.progress.connect(lambda *values: progress.append(values))
    return SimpleNamespace(dataset=dataset, worker=worker, endings=endings,
                           progress=progress, requests=requests, hub=hub,
                           api=api, remote=remote)


def test_sample_copies_image_and_mask_bytes_and_resumes_existing_files(sample):
    sample.worker.dest.mkdir()
    existing = sample.worker.dest / 'a.tif'
    existing.write_bytes(b'previously downloaded image')
    sample.worker.run()  # no real QThread is started
    assert sample.endings == [(True, str(sample.worker.dest), '', '')]
    assert sample.progress == [('a.tif', 1, 2), ('b.tif', 2, 2)]
    assert sample.requests == ['masks/a.tif', 'images/b.tif', 'masks/b.tif']
    assert existing.read_bytes() == b'previously downloaded image'
    assert (sample.worker.dest / 'b.tif').read_bytes() == b'images/b.tif'
    assert (sample.worker.dest / 'masks/a.tif').read_bytes() == b'masks/a.tif'
    assert (sample.worker.dest / 'masks/b.tif').read_bytes() == b'masks/b.tif'
    assert md.is_present(sample.worker.dest, expected=2)


def test_images_only_download_does_not_create_masks_or_fetch_box_labels(sample):
    sample.worker.dataset = replace(sample.dataset, masks='')
    sample.worker.run()
    assert sample.endings == [(True, str(sample.worker.dest), '', '')]
    assert sample.requests == ['images/a.tif', 'images/b.tif']
    assert not (sample.worker.dest / 'masks').exists()
    assert md.is_present(sample.worker.dest, expected=2)


def test_missing_download_dependency_reports_one_failure(sample, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    sample.worker.run()
    assert len(sample.endings) == 1
    ok, folder, payload, error = sample.endings[0]
    assert (ok, folder, payload) == (False, '', '')
    assert 'huggingface_hub is missing' in error
    assert not sample.worker.dest.exists()


def test_listing_failure_names_the_error_without_creating_a_sample(sample):
    def fail(*args, **kwargs):
        raise ConnectionError('repository unavailable')
    sample.api.list_repo_files = fail
    sample.worker.run()
    assert sample.endings == [(False, '', '', 'repository unavailable')]
    assert sample.requests == [] and not sample.worker.dest.exists()


def test_no_matching_masks_reports_the_repository_and_expected_folders(sample):
    sample.api.list_repo_files = lambda *a, **kw: ['images/unpaired.tif']
    sample.worker.run()
    assert len(sample.endings) == 1
    assert sample.endings[0][:3] == (False, '', '')
    assert sample.dataset.repo in sample.endings[0][3]
    assert 'images/ and masks/' in sample.endings[0][3]
    assert sample.requests == []


def test_bad_foreground_manifest_warns_and_still_downloads_real_pairs(sample, caplog):
    sample.worker.dataset = replace(sample.dataset, counts='missing.csv')
    sample.worker.run()
    assert sample.endings == [(True, str(sample.worker.dest), '', '')]
    assert 'drawn from all fields alike' in caplog.text
    assert md.is_present(sample.worker.dest, expected=2)


@pytest.mark.parametrize('after_first_pair', [False, True])
def test_cancel_stops_before_the_next_pair_without_reporting_success(sample, after_first_pair):
    original = sample.hub.hf_hub_download

    def download(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] == 'masks/a.tif':
            sample.worker.cancel()
        return result

    sample.hub.hf_hub_download = download
    if not after_first_pair:
        sample.worker.cancel()
    sample.worker.run()
    assert sample.endings == [(False, '', '', 'cancelled')]
    assert sample.requests == (['images/a.tif', 'masks/a.tif'] if after_first_pair else [])
    assert not (sample.worker.dest / 'b.tif').exists()


def test_download_error_keeps_completed_files_and_names_the_failed_field(sample):
    del sample.remote['masks/b.tif']
    sample.api.list_repo_files = lambda *a, **kw: [
        'images/a.tif', 'masks/a.tif', 'images/b.tif', 'masks/b.tif']
    sample.worker.run()
    assert len(sample.endings) == 1 and sample.endings[0][:3] == (False, '', '')
    assert 'b.tif' in sample.endings[0][3]
    assert (sample.worker.dest / 'masks/a.tif').read_bytes() == b'masks/a.tif'
    assert not md.is_present(sample.worker.dest, expected=2)


def test_default_foreground_readers_handle_bad_csv_counts_and_empty_yolo_files(sample, tmp_path):
    manifest = tmp_path / 'fields.csv'
    manifest.write_text('stem,name,n_objects\na.tif,,2\n,b.png,1\nempty,,0\n'
                        'invalid,,not-a-number\nmissing,,\n,,3\n')
    sample.remote['fields.csv'] = manifest
    assert md.foreground_stems(replace(sample.dataset, counts='fields.csv')) == {'a', 'b'}
    calls = []

    def tree(repo, **kwargs):
        calls.append((repo, kwargs))
        return [SimpleNamespace(path='labels/a.txt', size=8),
                SimpleNamespace(path='labels/empty.txt', size=0),
                SimpleNamespace(path='labels/subdirectory')]

    sample.api.list_repo_tree = tree
    assert md.foreground_stems(replace(sample.dataset, counts='labels/')) == {'a'}
    assert calls == [(sample.dataset.repo, dict(path_in_repo='labels',
                                               repo_type='dataset', recursive=True))]


@pytest.mark.parametrize('error,opened', [('', True), ('', False), ('download interrupted', False)])
def test_download_completion_restores_the_button_and_reports_opening(qtbot, tmp_path, error, opened):
    screen = QWidget()
    qtbot.addWidget(screen)
    screen._status_label = QLabel(screen)
    screen._btn_training_datasets = QPushButton(screen)
    pending, used = [], []
    dataset = md.MASK_DATASETS[0]

    def fetch(parent, selected, folder, done):
        assert parent is screen and selected is dataset
        pending.append(done)

    def use(folder):
        used.append(folder)
        return opened

    assert not md.open_a_training_dataset(screen, pick=lambda s: dataset,
                                          fetch=fetch, root=tmp_path, use=use)
    assert not screen._btn_training_datasets.isEnabled()
    folder = md.sample_folder(tmp_path, dataset)
    pending[0](None if error else str(folder), error)
    assert screen._btn_training_datasets.isEnabled()
    if error:
        assert error in screen._status_label.text() and used == []
    else:
        assert used == [folder]
        assert ('fields and the masks' if opened else 'folder would not open') in screen._status_label.text()


def test_picker_selection_and_cancel_are_real_dialog_results(qtbot, monkeypatch):
    screen = QWidget()
    qtbot.addWidget(screen)
    real_picker = md.DatasetPicker
    dialogs = []
    decision = [QDialog.Accepted]

    def picker(parent, datasets):
        dialog = real_picker(parent, datasets)
        dialogs.append(dialog)
        dialog._list.setCurrentRow(1)
        dialog.exec = lambda: decision[0]
        return dialog

    monkeypatch.setattr(md, 'DatasetPicker', picker)
    chosen = md._ask_which(screen, 'analyze_plaques')
    assert chosen.key == 'plaque_figures'
    assert dialogs[-1]._list.count() == 2
    dialogs[-1]._list.clearSelection()
    dialogs[-1]._list.setCurrentRow(-1)
    assert dialogs[-1].chosen() is None
    decision[0] = QDialog.Rejected
    assert md._ask_which(screen, 'analyze_plaques') is None


def test_shared_progress_adapter_builds_the_right_worker_without_starting_a_thread(sample, monkeypatch):
    from spacr.qt import hf_download
    captured = []
    callback = lambda *args: None
    parent = object()

    def download(screen, folder, on_done, **kwargs):
        captured.append((screen, folder, on_done, kwargs))

    monkeypatch.setattr(hf_download, 'download_toxo_mito_demo', download)
    md._fetch_sample(parent, sample.dataset, sample.worker.dest, callback)
    screen, folder, on_done, kwargs = captured.pop()
    assert (screen, folder, on_done) == (parent, sample.worker.dest, callback)
    worker = kwargs['worker_factory'](folder)
    assert isinstance(worker, md._SampleWorker)
    assert worker.dataset is sample.dataset and worker.dest == folder
    assert sample.dataset.title in kwargs['title']
