"""Saved classifier application consumes raw crops without the teacher encoder."""
from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QTableWidget

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, monkeypatch):
    from spacr.qt import preferences
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def loaded(widget, tmp_path):
    crops = np.arange(4 * 3 * 3 * 3, dtype=np.uint16).reshape(4, 3, 3, 3)
    widget.set_crops(crops)
    artifact = tmp_path / 'own-classifier.pt'
    artifact.write_text('Inert adapter boundary; never decode this fixture', encoding="utf-8")
    return crops, str(artifact)


def result(crops):
    n = len(crops)
    return {'features': np.arange(n*3,dtype=np.float32).reshape(n,3),
            'logits': np.tile([1.,2.],(n,1)),
            'probabilities': np.tile([.25,.75],(n,1)),
            'predictions':['B']*n,'classes':['A','B'],
            'source_channels':[2,0], 'checkpoint_sha256':'a'*64}


def test_saved_classifier_has_explicit_alpha_application_action(screen):
    button=screen.findChild(QPushButton,'EmbeddingsDinoClassifierApply')
    assert button is not None and screen._dino_controls.isAncestorOf(button)
    assert not screen.findChild(QPushButton,'EmbeddingsDinoClassifierApplyCancel').isEnabled()


def test_application_passes_full_raw_stack_and_exposes_actual_outputs(screen,monkeypatch,tmp_path):
    import spacr.embeddings as backend
    crops,path=loaded(screen,tmp_path);seen=[];before=crops.copy()
    foundation=screen._foundation.currentData();screen._labels={'0':'keep logistic label'}
    def apply(raw,artifact,**kwargs):
        seen.append((raw.copy(),artifact,kwargs));return result(raw)
    monkeypatch.setattr(backend,'_dino_classifier_apply',apply)
    monkeypatch.setattr(backend,'_embed_plate',lambda *a,**k:pytest.fail('Teacher preprocessing'))
    monkeypatch.setattr(backend,'_dino_encoder',lambda *a,**k:pytest.fail('Teacher loading'))
    assert screen._apply_dino_classifier(path,batch_size=2)==path
    np.testing.assert_array_equal(seen[0][0],before)
    np.testing.assert_array_equal(crops,before)
    assert seen[0][0].dtype==np.uint16 and seen[0][0].shape[-1]==3
    assert seen[0][1:]==(path,{'batch_size':2,'device':'cpu'})
    answer=screen._classifier_application_result
    np.testing.assert_array_equal(answer['features'],result(crops)['features'])
    assert answer['predictions']==['B']*4 and answer['classes']==['A','B']
    assert answer['source_channels']==[2,0] and answer['checkpoint_sha256']=='a'*64
    assert screen._foundation.currentData()==foundation
    assert screen._labels=={'0':'keep logistic label'}
    assert screen._classifier_results.isEnabled()


def test_result_dialog_shows_actual_matrices_and_artifact_identity(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    crops, path = loaded(screen, tmp_path)
    monkeypatch.setattr(backend, '_dino_classifier_apply', lambda raw, *a, **k: result(raw))
    screen._apply_dino_classifier(path)
    dialog = screen._show_classifier_application()
    try:
        table = dialog.findChild(QTableWidget, 'EmbeddingsDinoClassifierResultTable')
        picker = dialog.findChild(QComboBox, 'EmbeddingsDinoClassifierResultView')
        assert table.rowCount() == len(crops) and table.item(0, 0).text() == 'B'
        assert table.editTriggers() == QTableWidget.NoEditTriggers
        for key, headers, first in [('features', ['feature_0', 'feature_1', 'feature_2'], [0, 1, 2]),
                                    ('logits', ['A', 'B'], [1, 2]),
                                    ('probabilities', ['A', 'B'], [.25, .75])]:
            picker.setCurrentIndex(picker.findData(key))
            assert [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())] == headers
            assert [float(table.item(0, i).text()) for i in range(table.columnCount())] == first
        text = '\n'.join(label.text() for label in dialog.findChildren(QLabel))
        assert path in text and 'a' * 64 in text and '2, 0' in text
    finally:
        dialog.close()


def test_export_writes_all_crops_dimensions_and_string_classes_via_tabular(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    from spacr.tabular import read_table
    crops = np.zeros((53, 2, 2, 3), dtype=np.uint16)
    screen.set_crops(crops)
    answer = result(crops)
    answer.update(features=np.arange(53 * 12).reshape(53, 12),
                  predictions=['01'] * 53, classes=['01', '1'])
    monkeypatch.setattr(backend, '_dino_classifier_apply', lambda *a, **k: answer)
    artifact, output = str(tmp_path / 'selected.pt'), str(tmp_path / 'complete.csv')
    screen._apply_dino_classifier(artifact)
    assert screen._save_classifier_application(output) == output
    frame = read_table(output, canonicalise=False, report=None,
                       repair_plate_ids=False, dtype=str, keep_default_na=False)
    assert len(frame) == 53
    assert frame['prediction'].tolist() == ['01'] * 53
    assert set(['probability:01', 'probability:1', 'logit:01', 'logit:1']).issubset(frame.columns)
    assert frame['feature_11'].astype(int).tolist() == list(range(11, 53 * 12, 12))
    assert frame['checkpoint_path'].tolist() == [artifact] * 53
    assert frame['checkpoint_sha256'].tolist() == ['a' * 64] * 53
    assert frame['source_channels'].tolist() == ['2,0'] * 53


def test_missing_crops_and_cancelled_picker_do_not_invoke_adapter(screen, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    monkeypatch.setattr(backend, '_dino_classifier_apply', lambda *a, **k: pytest.fail('No input'))
    assert screen._apply_dino_classifier('not-read.pt') == ''
    loaded(screen, tmp_path)
    screen._dino_classifier_summary = {'path': 'recent-classifier.pt'}
    seen = []
    monkeypatch.setattr(em.QFileDialog, 'getOpenFileName',
                        lambda *a: seen.append(a) or ('', ''))
    before = screen._foundation.currentData()
    assert screen._apply_dino_classifier() == ''
    assert seen[0][2] == 'recent-classifier.pt'
    assert screen._foundation.currentData() == before


@pytest.mark.parametrize('failure', ['exception', 'row_alignment'])
def test_failed_application_preserves_existing_outputs_and_other_routes(screen, monkeypatch, tmp_path, failure):
    import spacr.embeddings as backend
    crops, path = loaded(screen, tmp_path)
    old = dict(result(crops), path='previous.pt')
    screen._classifier_application_result = old
    screen._classifier_results.setEnabled(True)
    screen._labels = {'0': 'old logistic label'}
    old_frame = screen._frame
    teacher = screen._foundation.currentData()
    def apply(*args, **kwargs):
        if failure == 'exception':
            raise ValueError('Rejected artifact fixture')
        return dict(result(crops), probabilities=np.zeros((1, 2)))
    monkeypatch.setattr(backend, '_dino_classifier_apply', apply)
    screen._apply_dino_classifier(path)
    assert screen._classifier_application_result is old and screen._classifier_results.isEnabled()
    assert screen._frame is old_frame and screen._foundation.currentData() == teacher
    assert screen._labels == {'0': 'old logistic label'}
    assert not screen.is_busy() and screen._classifier_apply.isEnabled()
    assert ('Rejected' if failure == 'exception' else 'aligned') in screen._status.text()


def test_real_worker_uses_adapter_off_gui_and_delivers_on_gui(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    seen = []
    class Observed(em.EmbeddingsScreen):
        def _sync_classifier_application_controls(self, *args):
            seen.append(QThread.currentThread())
            super()._sync_classifier_application_controls(*args)
    widget = Observed(threaded=True)
    qtbot.addWidget(widget)
    crops, path = loaded(widget, tmp_path)
    def apply(raw, artifact, **kwargs):
        assert QThread.currentThread() != widget.thread()
        assert raw is crops
        return result(raw)
    monkeypatch.setattr(backend, '_dino_classifier_apply', apply)
    try:
        widget._apply_dino_classifier(path)
        qtbot.waitUntil(lambda: not widget.is_busy(), timeout=5000)
        assert widget._classifier_application_result['path'] == path
        assert seen and all(thread == widget.thread() for thread in seen)
    finally:
        widget.close()


@pytest.mark.parametrize('action', ['cancel', 'replace_crops', 'close'])
def test_cancel_crop_change_and_close_discard_late_adapter_results(qtbot, monkeypatch, tmp_path, action):
    import spacr.embeddings as backend
    started, release = threading.Event(), threading.Event()
    widget = em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    crops, path = loaded(widget, tmp_path)
    old = dict(result(crops), path='previous.pt')
    widget._classifier_application_result = old
    def apply(raw, *args, **kwargs):
        assert 'cancelled' not in kwargs and 'progress' not in kwargs
        started.set()
        assert release.wait(5)
        return result(raw)
    monkeypatch.setattr(backend, '_dino_classifier_apply', apply)
    try:
        widget._apply_dino_classifier(path)
        qtbot.waitUntil(started.is_set, timeout=5000)
        assert widget.is_busy() and not widget._classifier_train.isEnabled()
        assert widget._pretrain_dino(str(tmp_path / 'not-started.pt')) == ''
        assert widget._train_dino_classifier(str(tmp_path / 'not-started.pt')) == ''
        monkeypatch.setattr(em.QFileDialog, 'getOpenFileName', lambda *a: pytest.fail('Busy picker'))
        widget._choose_dino_checkpoint()
        monkeypatch.setattr(widget, '_dino_classifier_dialog', lambda: pytest.fail('Busy training dialog'))
        widget._choose_dino_classifier()
        if action == 'replace_crops':
            widget.set_crops(np.ones((2, 2, 2, 3), dtype=np.uint16))
        elif action == 'close':
            timer = threading.Timer(.05, release.set)
            timer.start()
            widget.close()
            timer.join()
        else:
            widget._cancel_classifier_application()
        status = widget._status.text()
        if action != 'close':
            assert widget.is_busy() and not widget._classifier_apply.isEnabled()
            assert widget._apply_dino_classifier(path) == ''
        release.set()
        qtbot.waitUntil(lambda: not widget._classifier_apply_jobs.active_jobs(), timeout=5000)
        assert widget._classifier_application_result is (None if action == 'replace_crops' else old)
        assert widget._status.text() == status
        if action != 'close':
            qtbot.waitUntil(lambda: widget._classifier_apply.isEnabled()
                            and widget._classifier_train.isEnabled(), timeout=5000)
    finally:
        release.set()
        widget.close()
        qtbot.waitUntil(lambda: not widget._classifier_apply_jobs.active_jobs(), timeout=5000)


def test_cancel_before_worker_start_restores_application_controls(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as backend
    from spacr.qt import job_runner
    original = job_runner.make_thread
    def stopped(*args, **kwargs):
        thread, worker = original(*args, **kwargs)
        worker.request_cancel('cancel before adapter starts')
        return thread, worker
    monkeypatch.setattr(job_runner, 'make_thread', stopped)
    monkeypatch.setattr(backend, '_dino_classifier_apply', lambda *a, **k: pytest.fail('Cancelled adapter'))
    widget = em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    _, path = loaded(widget, tmp_path)
    try:
        widget._apply_dino_classifier(path)
        qtbot.waitUntil(lambda: not widget.is_busy(), timeout=5000)
        qtbot.waitUntil(widget._classifier_apply.isEnabled, timeout=5000)
        assert not widget._classifier_apply_cancel.isEnabled()
        assert widget._classifier_application_result is None
    finally:
        widget.close()


def test_both_training_progress_captions_are_visible_outside_slim_bars(screen):
    screen._set_dino_running(True)
    screen._on_dino_progress(screen._dino_generation, 2, 7, 1.25)
    screen._set_dino_running(False)
    screen._classifier_crop_generation = screen._crop_generation
    screen._set_classifier_running(True)
    screen._on_classifier_progress(screen._classifier_generation, 3, 9, .7, .8)
    for bar, label, fragments in [
        (screen._dino_progress_bar, screen._dino_progress_label, ['2/7', '1.250']),
        (screen._classifier_progress_bar, screen._classifier_progress_label, ['3/9', '0.700', '0.800'])]:
        assert not bar.isTextVisible()
        assert label.isVisibleTo(screen._dino_controls) and label.wordWrap()
        assert label.text() == bar.format() == bar.accessibleDescription()
        assert all(value in label.text() for value in fragments)
    screen._set_classifier_running(False)
