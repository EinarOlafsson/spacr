"""Annotate similarity options rank only eligible crops using current labels."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.active_learning import _SimilarityIndex
from spacr.qt.screens.annotate import _SimilarityWorker
from spacr.suggest import SUGGESTION_OFFSET
from tests.qt.test_find_similar_alpha_gate import alpha, annotate, plate  # noqa: F401


def test_worker_filters_current_column_table_zero_proposals_and_pending_labels(qtbot, tmp_path):
    path = tmp_path / 'labels.db'
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(8, 4)), index=[f'p{i}' for i in range(8)])
    index = _SimilarityIndex(frame, backend='numpy')
    table, column = 'crops " selected', 'label " current'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE "crops "" selected" (png_path TEXT, "label "" current" INTEGER)')
        db.executemany('INSERT INTO "crops "" selected" VALUES (?, ?)',
                       [('p0', 2), ('p1', 1), ('p2', None), ('p3', 0),
                        ('p4', SUGGESTION_OFFSET + 2), ('p5', 2), ('p6', None)])
    results, failures = [], []
    worker = _SimilarityWorker(str(path), None, 'p0', index=index, k=4, unlabelled_only=True,
                               annotation_column=column, png_table=table,
                               pending_labels={'p5': None, 'p6': 2})
    worker.done.connect(results.append)
    worker.failed.connect(failures.append)
    worker.start()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=5000)
    qtbot.waitUntil(lambda: bool(results) or bool(failures), timeout=5000)
    assert not failures
    assert set(results[0]['hits'].key) == {'p2', 'p3', 'p4', 'p5'}
    assert results[0]['hits'].similarity.is_monotonic_decreasing
    # Labels are read again even when embeddings are reused.
    with sqlite3.connect(path) as db:
        db.execute('UPDATE "crops "" selected" SET "label "" current"=1 WHERE png_path=\'p2\'')
    results.clear()
    worker.run()
    assert set(results[0]['hits'].key) == {'p3', 'p4', 'p5'}


def test_real_screen_count_unlabelled_filter_and_reference(qtbot, annotate):  # noqa: F811
    annotate._set_focus_slot(0)
    query = annotate._similar_query_key()
    annotate._similar_k.setValue(3)
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None and annotate._object_request is not None, timeout=10000)
    assert len(annotate._object_request.keys) == 4
    excluded = list(annotate._object_request.keys)[1:3]
    with sqlite3.connect(annotate._settings.db_path) as db:
        db.executemany('UPDATE png_list SET annotate=1 WHERE png_path=?', [(key,) for key in excluded + [query]])
    annotate._similar_unlabelled.setChecked(True)
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None, timeout=10000)
    keys = list(annotate._object_request.keys)
    assert keys[0] == query
    assert len(keys) == 4
    assert not set(excluded) & set(keys)
    assert annotate._object_request.context['requested_k'] == 3
    assert annotate._object_request.context['unlabelled_only']
    assert '3 unlabelled matches' in annotate._status_label.text()


def test_options_follow_alpha_registration(annotate, alpha, monkeypatch):  # noqa: F811
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.settings import ALPHA_FEATURES
    # Registration is integrated by the root agent in the shared settings file.
    monkeypatch.setitem(ALPHA_FEATURES, 565, {'widgets': ('AnnotateFindSimilar', 'AnnotateSimilarityOptions')})
    panel = annotate._similar_k.parentWidget()
    _apply_alpha_widgets(annotate)
    assert panel.isHidden()
    alpha['on'] = True
    _apply_alpha_widgets(annotate)
    assert not panel.isHidden()
    assert annotate._similar_k.value() == 100
    assert not annotate._similar_unlabelled.isChecked()


def test_failed_pending_label_save_refuses_unlabelled_search(qtbot, tmp_path):
    class FailedWriter:
        pending_batches = 1
        last_error = 'locked'
    frame = pd.DataFrame({'a': [1., 2., 3.], 'b': [3., 1., 2.]}, index=['a', 'b', 'c'])
    worker = _SimilarityWorker(str(tmp_path / 'unused.db'), None, 'a',
                               index=_SimilarityIndex(frame, backend='numpy'),
                               unlabelled_only=True, writer=FailedWriter())
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert 'resolve the save error' in failures[0]


@pytest.mark.parametrize('changed', ['image_type', 'annotation_column', 'png_table'])
def test_stale_search_options_do_not_replace_current_view(annotate, changed):  # noqa: F811
    before = annotate._object_request
    result = {'db_path': annotate._settings.db_path, changed: 'different'}
    annotate._on_similar_done(result)
    assert annotate._object_request is before
