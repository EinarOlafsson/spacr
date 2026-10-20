"""Own-backbone training UI sends raw, explicitly aligned data to its trainer."""
from __future__ import annotations

import csv
import threading

import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, monkeypatch):
    from spacr.qt import preferences
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def training_data(screen, tmp_path, rows=None):
    crops = np.arange(6 * 4 * 4 * 3, dtype=np.uint16).reshape(6, 4, 4, 3)
    screen.set_crops(crops)
    checkpoint = tmp_path / 'teacher.pt'
    checkpoint.write_text('Inert trainer boundary; this file must never be decoded', encoding="utf-8")
    screen._select_dino_checkpoint(str(checkpoint))
    table = tmp_path / 'training.csv'
    with table.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['label', 'acquisition_group'])
        writer.writerows(rows or [('A', 'plate1:A01'), ('', ''), ('B', 'plate1:A01'),
                                  ('A', 'plate1:B01'), ('B', 'plate1:B01'), ('', '')])
    return crops, checkpoint, table


def answer(path):
    return {'path': path, 'classes': ['A', 'B'], 'split': {'group_by': 'well'},
            'loss': [{'epoch': 1, 'train_loss': 0.7, 'validation_loss': 0.8}]}


def test_alpha_container_has_distinct_classifier_action_and_settings(screen):
    from spacr.qt import preferences
    button = screen.findChild(QPushButton, 'EmbeddingsDinoClassifierButton')
    assert button is not None and screen._dino_controls.isAncestorOf(button)
    dialog = screen._dino_classifier_dialog()
    try:
        assert dialog.findChild(QSpinBox, 'EmbeddingsDinoClassifierEpochs').value() == 20
        assert dialog.findChild(QSpinBox, 'EmbeddingsDinoClassifierBatch').value() == 32
        assert dialog.findChild(QDoubleSpinBox, 'EmbeddingsDinoClassifierLearningRate').value() == 1e-4
        assert dialog.findChild(QCheckBox, 'EmbeddingsDinoClassifierFreeze').isChecked()
        assert dialog.findChild(QCheckBox, 'EmbeddingsDinoClassifierNormalize').isChecked()
        assert dialog.findChild(QLineEdit, 'EmbeddingsDinoClassifierChannels').text() == ''
    finally:
        dialog.close()


def test_training_filters_labelled_rows_before_passing_raw_full_channels(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    crops, checkpoint, table = training_data(screen, tmp_path)
    seen = []
    before = crops.copy()
    screen._labels = {'0': 'existing logistic label'}
    def train(raw, teacher, output, **kwargs):
        seen.append((raw.copy(), teacher, output, kwargs))
        kwargs['progress'](1, kwargs['epochs'], 0.7, 0.8)
        return answer(output)
    monkeypatch.setattr(backend, '_dino_finetune_classifier', train)
    path = str(tmp_path / 'classifier.pt')
    assert screen._train_dino_classifier(path, table_path=str(table), source_channels='2,0',
                                        epochs=7, batch_size=3, lr=.002,
                                        freeze_encoder=False, normalize_features=False,
                                        holdout=.25, seed=4) == path
    raw, teacher, output, options = seen[0]
    np.testing.assert_array_equal(raw, crops[[0, 2, 3, 4]])
    np.testing.assert_array_equal(crops, before)
    assert raw.dtype == np.uint16 and raw.shape[-1] == 3
    assert teacher == str(checkpoint) and output == path
    assert options['source_channels'] == (2, 0)
    assert options['labels'] == ('A', 'B', 'A', 'B')
    assert options['groups'] == ('plate1:A01', 'plate1:A01', 'plate1:B01', 'plate1:B01')
    assert {k: options[k] for k in ('epochs', 'batch_size', 'lr', 'freeze_encoder', 'normalize_features', 'holdout', 'seed', 'device')} == dict(epochs=7,batch_size=3,lr=.002,freeze_encoder=False,normalize_features=False,holdout=.25,seed=4,device='cpu')
    assert screen._labels == {'0': 'existing logistic label'}
    assert screen._dino_classifier_summary == answer(path)
    assert not screen._classifier_running


def test_training_refuses_unaligned_metadata_without_starting_worker(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    _, _, table = training_data(screen, tmp_path, rows=[('A', 'wellA'), ('B', 'wellB')])
    monkeypatch.setattr(backend, '_dino_finetune_classifier', lambda *a, **kw: pytest.fail('Invalid metadata reached trainer'))
    assert screen._train_dino_classifier(str(tmp_path/'out.pt'), table_path=str(table), source_channels='0') == ''
    assert 'row' in screen._status.text().lower()
    assert not screen.is_busy()


def test_classifier_action_inherits_real_alpha_container(screen, monkeypatch):
    from spacr.qt import preferences
    from spacr.settings import ALPHA_FEATURES
    assert 'EmbeddingsDinoControls' in ALPHA_FEATURES[561]['widgets']
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: False)
    preferences._apply_alpha_widgets(screen)
    assert not screen._classifier_train.isVisibleTo(screen)
    assert not screen._classifier_cancel.isVisibleTo(screen)
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    preferences._apply_alpha_widgets(screen)
    assert screen._classifier_train.isVisibleTo(screen)


def test_actual_dialog_supplies_supervised_controls_and_retains_invalid_form(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    from PySide6.QtWidgets import QDialog, QFileDialog
    _, _, table = training_data(screen, tmp_path)
    seen = []
    monkeypatch.setattr(backend, '_dino_finetune_classifier', lambda raw, teacher, path, **kw: seen.append(kw) or answer(path))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(tmp_path/'new.pt'), ''))
    dialog = screen._dino_classifier_dialog()
    try:
        dialog.findChild(QLineEdit, 'EmbeddingsDinoClassifierTable').setText(str(table))
        start = dialog.findChild(QPushButton, 'EmbeddingsDinoClassifierStart')
        start.click()
        assert dialog.result() != QDialog.DialogCode.Accepted and not seen
        dialog.findChild(QLineEdit, 'EmbeddingsDinoClassifierChannels').setText('2,0')
        dialog.findChild(QSpinBox, 'EmbeddingsDinoClassifierEpochs').setValue(4)
        dialog.findChild(QSpinBox, 'EmbeddingsDinoClassifierBatch').setValue(8)
        dialog.findChild(QDoubleSpinBox, 'EmbeddingsDinoClassifierLearningRate').setValue(.003)
        dialog.findChild(QDoubleSpinBox, 'EmbeddingsDinoClassifierHoldout').setValue(.4)
        dialog.findChild(QSpinBox, 'EmbeddingsDinoClassifierSeed').setValue(11)
        dialog.findChild(QCheckBox, 'EmbeddingsDinoClassifierFreeze').setChecked(False)
        dialog.findChild(QCheckBox, 'EmbeddingsDinoClassifierNormalize').setChecked(False)
        start.click()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert len(seen) == 1
        for key, value in dict(epochs=4,batch_size=8,lr=.003,holdout=.4,seed=11,
                               freeze_encoder=False,normalize_features=False,device='cpu',source_channels=(2,0)).items():
            assert seen[0][key] == value
    finally:
        dialog.close()


@pytest.mark.parametrize('channels', ['', '0,0', '-1', '3', '0,', '0.5'])
def test_invalid_physical_channels_refuse_before_trainer(screen, monkeypatch, tmp_path, channels):
    import spacr.embeddings as backend
    _, _, table = training_data(screen, tmp_path)
    monkeypatch.setattr(backend, '_dino_finetune_classifier', lambda *a, **kw: pytest.fail('Invalid channels reached trainer'))
    assert screen._train_dino_classifier(str(tmp_path/'new.pt'), table_path=str(table), source_channels=channels) == ''
    assert not screen.is_busy()
    assert 'channel' in screen._status.text().lower()


@pytest.mark.parametrize('kind', ['missing_columns', 'unknown_group', 'one_class', 'one_group'])
def test_metadata_requires_explicit_groups_and_labelled_classes(screen, monkeypatch, tmp_path, kind):
    import spacr.embeddings as backend
    rows = [('A','wellA'),('A','wellA'),('B','wellA'),('A','wellB'),('B','wellB'),('B','wellB')]
    if kind == 'unknown_group': rows[0] = ('A','unknown')
    if kind == 'one_class': rows = [('A',group) for _, group in rows]
    if kind == 'one_group': rows = [(label,'wellA') for label, _ in rows]
    _, _, table = training_data(screen, tmp_path, rows)
    if kind == 'missing_columns': table.write_text('label,well\nA,a\n', encoding="utf-8")
    monkeypatch.setattr(backend, '_dino_finetune_classifier', lambda *a, **kw: pytest.fail('Invalid metadata reached trainer'))
    assert screen._train_dino_classifier(str(tmp_path/'new.pt'), table_path=str(table), source_channels='0') == ''
    assert not screen.is_busy()
    assert screen._classifier_train.isEnabled()


def test_cancelled_save_or_existing_output_never_starts_or_overwrites(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    from PySide6.QtWidgets import QFileDialog
    _, checkpoint, table = training_data(screen, tmp_path)
    monkeypatch.setattr(backend, '_dino_finetune_classifier', lambda *a, **kw: pytest.fail('Refused output reached trainer'))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: ('', ''))
    assert screen._train_dino_classifier(table_path=str(table), source_channels='0') == ''
    existing = tmp_path/'existing.pt'
    existing.write_bytes(b'keep this classifier')
    for path in (existing, checkpoint):
        before = path.read_bytes()
        assert screen._train_dino_classifier(str(path),table_path=str(table),source_channels='0') == ''
        assert path.read_bytes() == before
    assert not screen.is_busy()


def test_backend_failure_preserves_previous_artifact_and_logistic_labels(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    _, _, table = training_data(screen, tmp_path)
    previous = answer('previous.pt')
    screen._dino_classifier_summary = previous
    screen._labels = {'0':'old'}
    selected = screen._foundation.currentData()
    callbacks=[]
    def refuse(raw, teacher, output, **options):
        callbacks.append(options['progress'])
        raise ValueError('teacher scale is incompatible')
    monkeypatch.setattr(backend, '_dino_finetune_classifier', refuse)
    screen._train_dino_classifier(str(tmp_path/'new.pt'),table_path=str(table),source_channels='0')
    assert 'teacher scale' in screen._status.text()
    assert screen._dino_classifier_summary is previous
    assert screen._labels == {'0':'old'} and screen._foundation.currentData() == selected
    assert screen._classifier_train.isEnabled() and screen._dino.isEnabled()
    assert not screen._classifier_cancel.isEnabled()
    callbacks[0](99,100,.1,.2)
    assert screen._classifier_progress_bar.value() == 0


def test_old_progress_cannot_overwrite_a_new_classifier_run(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    _, _, table = training_data(screen, tmp_path)
    callbacks=[]
    def train(raw, teacher, output, **options):
        callbacks.append(options['progress'])
        options['progress'](1,2,.7,.8)
        if len(callbacks)>1:
            callbacks[0](99,100,.1,.1)
            assert screen._classifier_progress_bar.value() == 1
        return answer(output)
    monkeypatch.setattr(backend, '_dino_finetune_classifier', train)
    for name in ('first.pt','second.pt'):
        screen._train_dino_classifier(str(tmp_path/name),table_path=str(table),source_channels='0')
    assert screen._dino_classifier_summary['path'].endswith('second.pt')


def test_real_worker_reports_on_gui_thread_and_stops_cooperatively(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    started, release = threading.Event(), threading.Event()
    observed=[]
    class Observed(em.EmbeddingsScreen):
        def _on_classifier_progress(self, generation, epoch, total, train, validation):
            observed.append(QThread.currentThread())
            super()._on_classifier_progress(generation,epoch,total,train,validation)
    widget=Observed(threaded=True)
    qtbot.addWidget(widget)
    _, _, table=training_data(widget,tmp_path)
    def train(raw, teacher, output, **options):
        assert QThread.currentThread() != widget.thread()
        options['progress'](1,2,.7,.8)
        started.set()
        while not release.wait(.01):
            if options['cancelled']():
                raise InterruptedError('Classifier training stopped.')
        return answer(output)
    monkeypatch.setattr(backend,'_dino_finetune_classifier',train)
    try:
        widget._train_dino_classifier(str(tmp_path/'new.pt'),table_path=str(table),source_channels='0')
        qtbot.waitUntil(lambda: started.is_set() and len(observed)==1,timeout=5000)
        assert observed == [widget.thread()]
        assert widget.is_busy() and not widget._dino.isEnabled()
        assert widget._train_dino_classifier(str(tmp_path/'other.pt'),table_path=str(table),source_channels='0') == ''
        widget._classifier_cancel.click()
        qtbot.waitUntil(lambda: not widget.is_busy(),timeout=5000)
        assert widget._classifier_train.isEnabled() and widget._dino.isEnabled()
        assert not hasattr(widget,'_dino_classifier_summary')
    finally:
        release.set()
        widget.close()
        qtbot.waitUntil(lambda: not widget._classifier_jobs.active_jobs(),timeout=5000)


def test_new_crop_selection_stops_training_and_rejects_old_completion(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    started, release = threading.Event(), threading.Event()
    options_seen=[]
    widget=em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    _, _, table=training_data(widget,tmp_path)
    def train(raw, teacher, output, **options):
        options_seen.append(options)
        started.set()
        assert release.wait(5)
        options['progress'](2,2,.1,.1)
        return answer(output)
    monkeypatch.setattr(backend,'_dino_finetune_classifier',train)
    try:
        widget._train_dino_classifier(str(tmp_path/'old.pt'),table_path=str(table),source_channels='0')
        qtbot.waitUntil(started.is_set,timeout=5000)
        widget.set_crops(np.ones((2,4,4,3)))
        assert options_seen[0]['cancelled']()
        release.set()
        qtbot.waitUntil(lambda: not widget.is_busy(),timeout=5000)
        assert not hasattr(widget,'_dino_classifier_summary')
        assert widget._classifier_progress_bar.value() == 0
    finally:
        release.set()
        widget.close()
        qtbot.waitUntil(lambda: not widget._classifier_jobs.active_jobs(),timeout=5000)


def test_close_cancels_active_classifier_and_discards_late_reports(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    started, release = threading.Event(), threading.Event()
    options_seen=[]
    widget=em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    _, _, table=training_data(widget,tmp_path)
    def train(raw, teacher, output, **options):
        options_seen.append(options)
        started.set()
        assert release.wait(5)
        options['progress'](2,2,.1,.1)
        return answer(output)
    monkeypatch.setattr(backend,'_dino_finetune_classifier',train)
    try:
        widget._train_dino_classifier(str(tmp_path/'closed.pt'),table_path=str(table),source_channels='0')
        qtbot.waitUntil(started.is_set,timeout=5000)
        timer=threading.Timer(.05,release.set)
        timer.start()
        widget.close()
        timer.join()
        qtbot.waitUntil(lambda: not widget._classifier_jobs.active_jobs(),timeout=5000)
        assert options_seen[0]['cancelled']()
        assert widget._classifier_progress_bar.value() == 0
        assert not hasattr(widget,'_dino_classifier_summary')
    finally:
        release.set()
        widget.close()


def test_cancel_before_start_restores_classifier_controls(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    from spacr.qt import job_runner
    original=job_runner.make_thread
    def stopped(*args,**kwargs):
        thread,worker=original(*args,**kwargs)
        worker.request_cancel('cancel before trainer starts')
        return thread,worker
    monkeypatch.setattr(job_runner,'make_thread',stopped)
    monkeypatch.setattr(backend,'_dino_finetune_classifier',lambda *a,**kw: pytest.fail('Cancelled worker called trainer'))
    widget=em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    _, _, table=training_data(widget,tmp_path)
    widget._train_dino_classifier(str(tmp_path/'cancelled.pt'),table_path=str(table),source_channels='0')
    qtbot.waitUntil(lambda: not widget.is_busy(),timeout=5000)
    assert widget._classifier_train.isEnabled() and widget._dino.isEnabled()
    assert not widget._classifier_cancel.isEnabled()
    assert not hasattr(widget,'_dino_classifier_summary')


def test_numeric_looking_label_and_group_identities_are_not_inferred(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    _, _, table = training_data(screen, tmp_path, rows=[
        ('01','001'),('1','001'),('01','1'),('1','1'),('',''),('','')])
    seen=[]
    monkeypatch.setattr(backend,'_dino_finetune_classifier',lambda raw,teacher,path,**kw: seen.append(kw) or answer(path))
    path=str(tmp_path/'ids.pt')
    assert screen._train_dino_classifier(path,table_path=str(table),source_channels='0') == path
    assert seen[0]['labels'] == ('01','1','01','1')
    assert seen[0]['groups'] == ('001','001','1','1')
