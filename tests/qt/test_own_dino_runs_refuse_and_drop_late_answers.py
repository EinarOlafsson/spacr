"""Own-DINO training, classifier and demo runs refuse cleanly and ignore late answers.

Each test drives the real screen. Background runners are replaced only to
hold a completion back, so a test can change what is current (a newer run,
a closed screen, a new crop selection) before the stale answer arrives, or
to make a submission fail the way a refused worker does.
"""
from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('PySide6')
from PySide6.QtWidgets import QDialog, QFileDialog, QLineEdit, QPushButton

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, monkeypatch):
    from spacr.qt import preferences
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class Held:
    """A runner stand-in that keeps each submission instead of running it."""

    def __init__(self):
        self.calls = []

    def __call__(self, work, done):
        self.calls.append((work, done))

    def run(self, index=-1):
        """Run a kept job's work, returning its answer without delivering it."""
        return self.calls[index][0]()


def refused(*_args, **_kwargs):
    raise RuntimeError('the worker pool is shutting down')


def crops(n=6):
    return np.arange(n * 4 * 4 * 3, dtype=np.uint16).reshape(n, 4, 4, 3)


def training_data(screen, tmp_path):
    screen.set_crops(crops())
    teacher = tmp_path / 'teacher.pt'
    teacher.write_text('Inert trainer boundary; this file must never be decoded')
    screen._select_dino_checkpoint(str(teacher))
    table = tmp_path / 'training.csv'
    with table.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['label', 'acquisition_group'])
        writer.writerows([('A', 'p1:A01'), ('B', 'p1:A01'), ('A', 'p1:B01'),
                          ('B', 'p1:B01'), ('', ''), ('', '')])
    return teacher, table


def application(n):
    return {'features': np.ones((n, 2)), 'logits': np.zeros((n, 2)),
            'probabilities': np.full((n, 2), .5), 'predictions': ['A'] * n,
            'classes': ['A', 'B'], 'source_channels': [0], 'checkpoint_sha256': 'b' * 64}


def test_a_refused_pretraining_submission_restores_the_controls(screen, monkeypatch, tmp_path):
    screen.set_crops(crops())
    monkeypatch.setattr(screen._dino_jobs, 'submit', refused)
    assert screen._pretrain_dino(str(tmp_path / 'backbone.pt')) == ''
    assert screen._status.text() == 'the worker pool is shutting down'
    assert not screen._dino_running
    assert screen._dino.isEnabled()


def test_an_older_pretraining_completion_does_not_choose_its_checkpoint(screen, monkeypatch, tmp_path):
    held = Held()
    monkeypatch.setattr(screen._dino_jobs, 'submit', held)
    screen.set_crops(crops())
    chosen = screen._foundation.currentData()
    assert screen._pretrain_dino(str(tmp_path / 'old.pt')) == str(tmp_path / 'old.pt')
    screen._dino_generation += 1
    held.calls[0][1]({'path': str(tmp_path / 'old.pt'), 'epochs': 2, 'loss': [.5]})
    assert screen._foundation.currentData() == chosen
    assert not hasattr(screen, '_dino_summary') or screen._dino_summary is None


def test_a_closed_screen_ignores_pretraining_failure_and_completion(screen, tmp_path):
    screen.set_crops(crops())
    screen.close()
    status = screen._status.text()
    screen._on_dino_failed('late failure')
    screen._on_dino_done({'path': str(tmp_path / 'late.pt'), 'epochs': 3, 'loss': []})
    assert screen._status.text() == status
    assert getattr(screen, '_dino_summary', None) is None


def test_the_training_table_browser_fills_the_field_only_when_a_table_is_chosen(screen, monkeypatch, tmp_path):
    answers = iter([(str(tmp_path / 'labels.csv'), ''), ('', '')])
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: next(answers))
    dialog = screen._dino_classifier_dialog()
    try:
        field = dialog.findChild(QLineEdit, 'EmbeddingsDinoClassifierTable')
        browse = dialog.findChild(QPushButton, 'EmbeddingsDinoClassifierTableBrowse')
        browse.click()
        assert field.text() == str(tmp_path / 'labels.csv')
        browse.click()
        assert field.text() == str(tmp_path / 'labels.csv')
    finally:
        dialog.close()


def test_the_classifier_action_opens_its_settings_and_discards_them_after(screen, monkeypatch):
    shown = []

    def execute(dialog):
        shown.append(dialog.objectName())
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, 'exec', execute)
    screen._choose_dino_classifier()
    assert shown == ['EmbeddingsDinoClassifierDialog']
    screen._classifier_running = True
    screen._choose_dino_classifier()
    assert shown == ['EmbeddingsDinoClassifierDialog']


def test_cancel_without_a_running_classifier_changes_nothing(screen):
    screen._status.setText('ready')
    screen._cancel_dino_classifier()
    assert not screen._classifier_stop.is_set()
    assert screen._status.text() == 'ready'


@pytest.mark.parametrize('problem, reason', [
    ('no_crops', 'Load crops first.'),
    ('no_teacher', 'Choose an existing own-DINO teacher checkpoint first.'),
    ('no_table', 'Choose the training label and acquisition-group table.'),
])
def test_classifier_training_refuses_before_asking_for_an_output(
        screen, monkeypatch, tmp_path, problem, reason):
    import spacr.embeddings as backend
    monkeypatch.setattr(backend, '_dino_finetune_classifier',
                        lambda *a, **k: pytest.fail('a refused run reached the trainer'))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *a, **k: pytest.fail('a refused run asked for a file'))
    _teacher, table = training_data(screen, tmp_path)
    if problem == 'no_crops':
        screen._crops = None
    elif problem == 'no_teacher':
        screen._foundation.setCurrentIndex(0)
    path = screen._train_dino_classifier(
        table_path='' if problem == 'no_table' else str(table), source_channels='0')
    assert path == ''
    assert screen._status.text() == reason
    assert not screen.is_busy()


def test_progress_from_a_deleted_screen_signal_does_not_fail_training(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    _teacher, table = training_data(screen, tmp_path)

    def deleted(*_args):
        raise RuntimeError('Signal source has been deleted')

    monkeypatch.setattr(screen, '_classifier_progress', SimpleNamespace(emit=deleted))

    def train(raw, teacher, output, **kwargs):
        kwargs['progress'](1, 2, .7, .8)
        return {'path': output, 'classes': ['A', 'B']}

    monkeypatch.setattr(backend, '_dino_finetune_classifier', train)
    out = str(tmp_path / 'classifier.pt')
    assert screen._train_dino_classifier(out, table_path=str(table), source_channels='0') == out
    assert screen._dino_classifier_summary == {'path': out, 'classes': ['A', 'B']}
    assert 'Classifier saved to' in screen._status.text()


def test_an_older_classifier_completion_is_not_published(screen, monkeypatch, tmp_path):
    held = Held()
    monkeypatch.setattr(screen._classifier_jobs, 'submit', held)
    _teacher, table = training_data(screen, tmp_path)
    out = str(tmp_path / 'classifier.pt')
    assert screen._train_dino_classifier(out, table_path=str(table), source_channels='0') == out
    screen._classifier_generation += 1
    held.calls[0][1]({'path': out, 'classes': ['A', 'B']})
    assert getattr(screen, '_dino_classifier_summary', None) is None
    assert screen._classifier_running


def test_a_refused_classifier_submission_restores_the_controls(screen, monkeypatch, tmp_path):
    monkeypatch.setattr(screen._classifier_jobs, 'submit', refused)
    _teacher, table = training_data(screen, tmp_path)
    assert screen._train_dino_classifier(
        str(tmp_path / 'c.pt'), table_path=str(table), source_channels='0') == ''
    assert screen._status.text() == 'the worker pool is shutting down'
    assert not screen._classifier_running
    assert screen._classifier_train.isEnabled()


def test_a_failure_after_close_or_a_new_selection_keeps_the_current_message(screen):
    screen.set_crops(crops())
    screen._classifier_running = True
    screen._classifier_crop_generation = screen._crop_generation - 1
    screen._status.setText('newer crops loaded')
    screen._on_classifier_failed('old run failed')
    assert screen._status.text() == 'newer crops loaded'
    assert not screen._classifier_running
    screen.close()
    screen._classifier_running = True
    screen._on_classifier_failed('late failure')
    assert screen._classifier_running


def test_an_application_failure_for_old_crops_is_not_reported(screen):
    screen.set_crops(crops())
    screen._classifier_apply_running = True
    screen._classifier_apply_delivery = True
    screen._classifier_apply_crop_generation = screen._crop_generation - 1
    screen._status.setText('newer crops loaded')
    screen._on_classifier_application_failed('old application failed')
    assert screen._status.text() == 'newer crops loaded'
    assert not screen._classifier_apply_running


@pytest.mark.parametrize('change', ['newer_run', 'new_crops', 'closed'])
def test_a_late_application_result_is_dropped(screen, monkeypatch, tmp_path, change):
    import spacr.embeddings as backend
    held = Held()
    monkeypatch.setattr(screen._classifier_apply_jobs, 'submit', held)
    monkeypatch.setattr(backend, '_dino_classifier_apply',
                        lambda raw, path, **k: application(len(raw)))
    screen.set_crops(crops(4))
    artifact = str(tmp_path / 'classifier.pt')
    assert screen._apply_dino_classifier(artifact, batch_size=2) == artifact
    answer = held.run()
    assert answer['path'] == artifact
    if change == 'newer_run':
        screen._classifier_apply_generation += 1
    elif change == 'new_crops':
        screen._crop_generation += 1
    else:
        screen.close()
    held.calls[0][1](answer)
    assert screen._classifier_application_result is None
    assert not screen._classifier_results.isEnabled()
    if change == 'new_crops':
        assert not screen._classifier_apply_running


def test_a_refused_application_submission_restores_the_controls(screen, monkeypatch, tmp_path):
    monkeypatch.setattr(screen._classifier_apply_jobs, 'submit', refused)
    screen.set_crops(crops(4))
    assert screen._apply_dino_classifier(str(tmp_path / 'c.pt'), batch_size=2) == ''
    assert screen._status.text() == 'the worker pool is shutting down'
    assert not screen._classifier_apply_running
    assert screen._classifier_apply.isEnabled()


def test_results_and_export_need_an_application_and_a_chosen_file(screen, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: asked.append(1) or ('', ''))
    assert screen._show_classifier_application() is None
    assert screen._save_classifier_application() == ''
    assert asked == []
    screen._classifier_application_result = dict(application(2), path='c.pt')
    assert screen._save_classifier_application() == ''
    assert asked == [1]
    assert not (tmp_path / 'results.csv').exists()


def test_an_export_of_a_replaced_result_does_not_claim_the_current_one(screen, monkeypatch, tmp_path):
    held = Held()
    monkeypatch.setattr(screen._jobs, 'submit', held)
    first = dict(application(2), path='first.pt')
    screen._classifier_application_result = first
    target = str(tmp_path / 'results.csv')
    assert screen._save_classifier_application(target) == target
    saved = held.run()
    assert pd.read_csv(target)['checkpoint_path'].tolist() == ['first.pt', 'first.pt']
    screen._classifier_application_result = dict(application(2), path='second.pt')
    screen._status.setText('second result ready')
    held.calls[0][1](saved)
    assert screen._status.text() == 'second result ready'
    screen._classifier_application_result = first
    held.calls[0][1](saved)
    assert screen._status.text() == f'Classifier results saved to {saved}.'


@pytest.fixture
def demo(screen, monkeypatch, tmp_path):
    from spacr import example_archives
    prepare = example_archives._prepare_embeddings_example
    monkeypatch.setattr(example_archives, '_prepare_embeddings_example',
                        lambda: prepare(tmp_path / 'demo'))
    return screen


def test_the_demo_is_not_started_twice_or_after_close(demo, monkeypatch):
    held = Held()
    monkeypatch.setattr(demo._jobs, 'submit', held)
    demo._load_demo_crops()
    demo._load_demo_crops()
    assert len(held.calls) == 1
    demo.close()
    demo._loading = False
    demo._load_demo_crops()
    assert len(held.calls) == 1


def test_a_demo_stopped_while_reading_is_headed_as_a_stop(demo, monkeypatch):
    held = Held()
    monkeypatch.setattr(demo._jobs, 'submit', held)
    demo._load_demo_crops()
    answer = held.run()
    demo._stop.set()
    held.calls[0][1](answer)
    assert not demo.is_loading()
    assert demo._status.text() == 'Loading synthetic demo crops was stopped.'
    assert getattr(demo, '_crops', None) is None


def test_a_demo_finishing_after_close_is_not_loaded(demo, monkeypatch):
    held = Held()
    monkeypatch.setattr(demo._jobs, 'submit', held)
    demo._load_demo_crops()
    answer = held.run()
    demo.close()
    held.calls[0][1](answer)
    assert getattr(demo, '_crops', None) is None
    assert not demo._demo_active


def test_an_unreadable_demo_label_table_is_reported_not_claimed(screen, monkeypatch, tmp_path):
    from spacr import example_archives
    prepare = example_archives._prepare_embeddings_example

    def damaged():
        example = prepare(tmp_path / 'demo')
        open(example['labels'], 'w').close()
        return example

    monkeypatch.setattr(example_archives, '_prepare_embeddings_example', damaged)
    screen._load_demo_crops()
    assert screen._crops.shape[0] == 24
    assert screen._labels == {}
    assert 'Could not read the label table' in screen._status.text()
    assert 'not biological validation' in screen._caveat.text()


@pytest.mark.parametrize('has_result', [False, True])
def test_a_failed_job_outside_loading_is_said_and_keeps_save_for_a_result(screen, has_result):
    screen.set_crops(crops(2))
    if has_result:
        screen._result = object()
        screen._result_identity = ('folder', ('0', '1'))
    screen._on_job_failed('the embedding could not be used')
    assert screen._status.text() == 'the embedding could not be used'
    assert screen._save_similar.isEnabled() is has_result
    assert not screen.is_loading()


def test_close_finishes_even_when_every_runner_fails_to_shut_down(screen, monkeypatch):
    attempts = []

    def broken(name):
        def shutdown(*_args, **_kwargs):
            attempts.append(name)
            raise RuntimeError(f'{name} could not be drained')
        return shutdown

    for name in ('_classifier_apply_jobs', '_classifier_jobs', '_dino_jobs', '_jobs'):
        monkeypatch.setattr(getattr(screen, name), 'shutdown', broken(name))
    screen.show()
    assert screen.close()
    assert attempts == ['_classifier_apply_jobs', '_classifier_jobs', '_dino_jobs', '_jobs']
    assert screen._dino_closed and screen._stop.is_set() and screen._classifier_stop.is_set()
    assert not screen.isVisible()
