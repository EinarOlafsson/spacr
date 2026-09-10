"""Record the real Power / Design controls and bounded simulator runs.

No injected fits, lowered optimiser steps, invented metrics or hidden export.
The smaller library and replicate count are visible teaching inputs, not a
claim that these assumptions describe an acquired experiment.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import time

from power_evidence import check_curves, check_recommendation


def record_power(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QScrollArea, QSplitter

    forms = [w for w in screen.findChildren(QScrollArea) if w.widget().isAncestorOf(screen._genes)]
    if len(forms) != 1:
        raise RuntimeError('Expected one actual scrollable planning form')
    form = forms[0]

    def reveal(widget):
        bar = form.verticalScrollBar()
        bar.setFocus()
        QTest.keyClick(bar, Qt.Key_Home)
        for _ in range(20):
            center = widget.mapTo(form.viewport(), widget.rect().center())
            if form.viewport().rect().contains(center):
                return
            QTest.keyClick(bar, Qt.Key_PageDown)
            settle(0.1)
        raise RuntimeError('The actual form cannot scroll the requested control into view')

    def fill(box, value):
        reveal(box)
        edit = box.lineEdit()
        QTest.mouseClick(edit, Qt.LeftButton)
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, str(value))
        QTest.keyClick(edit, Qt.Key_Tab)
        settle(0.3)
        if abs(float(box.value()) - float(value)) > 0.001:
            raise RuntimeError('The real numeric editor did not accept its requested value')

    def choose(box, value):
        reveal(box)
        index = box.findText(value)
        if index < 0:
            raise RuntimeError(f'Missing actual choice: {value}')
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)
        settle(0.3)
        if box.currentText() != value:
            raise RuntimeError('The real selector did not change')

    # A genuine splitter drag gives readable controls without changing fonts
    # or fabricating a layout inside the recording.
    splitters = [s for s in screen.findChildren(QSplitter) if s.widget(0) is form]
    if len(splitters) != 1:
        raise RuntimeError('Expected the actual form/output splitter')
    handle = splitters[0].handle(1)
    start = handle.rect().center()
    current = handle.mapTo(window, start).x()
    QTest.mousePress(handle, Qt.LeftButton, pos=start)
    end = start + QPoint(1450 - current, 0)
    QTest.mouseMove(handle, end, delay=100)
    QTest.mouseRelease(handle, Qt.LeftButton, pos=end)
    settle(0.5)
    capture('02_readable_defaults')

    fill(screen._genes, 40)
    fill(screen._grnas, 4)
    fill(screen._constructs, 2)
    choose(screen._score_per, 'guide')
    if screen.spec().n_library_units != 160:
        raise RuntimeError('Guide scoring does not reflect the actual 40 x 4 library')
    capture('03_guide_library')
    choose(screen._score_per, 'gene')
    choose(screen._plate_format, '96')
    fill(screen._plates, 1)
    fill(screen._prevalence, 0.1)
    fill(screen._cells, 120)
    fill(screen._reads, 30000)
    fill(screen._replicates, 2)
    fill(screen._threshold, 0.8)
    fill(screen._seed, 0)
    choose(screen._backend, 'torch')
    if screen.fit_kwargs or not screen._threaded:
        raise RuntimeError('Do not use hidden fit overrides or an inline substitute')
    reveal(screen._genes)
    capture('04_bounded_library_plate')

    fill(screen._effect, 0.5)
    if not any('LESS' in text for text in screen.spec().validate()) or screen._btn_run.isEnabled():
        raise RuntimeError('The protective-effect warning must prevent running this unsupported direction')
    capture('05_unsupported_effect_warning')
    fill(screen._effect, 6.667)
    if screen.spec().validate() or not screen._btn_run.isEnabled():
        raise RuntimeError('The actual valid effect did not restore Run')
    capture('06_effect_restored')
    reveal(screen._cells)
    capture('07_acquisition_and_held_values')
    reveal(screen._btn_run)
    capture('08_run_settings')

    spec = screen.spec()
    results, progress = [], []
    screen.job_finished.connect(lambda ok: results.append(bool(ok)))
    screen.progressed.connect(lambda done, total: progress.append([done, total]))
    QTest.mouseClick(screen._btn_run, Qt.LeftButton)
    settle(0.8)
    capture('09_real_simulation_running')
    deadline = time.monotonic() + timeout
    while screen.is_busy() or screen.active_jobs():
        if time.monotonic() >= deadline:
            reveal(screen._btn_stop)
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
            raise TimeoutError('The bounded real simulation exceeded its recording budget')
        settle(0.2)
    settle(0.7)
    result = screen.result()
    if results != [True] or not result or result.get('cancelled') or result['spec'] != spec:
        raise RuntimeError('The real sweep did not complete with its recorded inputs')
    records = {name: json.loads(result[name].to_json(orient='records')) for name in
               ('cells_scan', 'wells_scan', 'cells_curve', 'wells_curve')}
    write_json(captures / 'simulation_result.json', {
        'spec': asdict(spec), 'fit_kwargs': dict(screen.fit_kwargs),
        **records, 'status': screen.status_text(), 'answer': screen.answer_text(),
        'table': screen.table_rows(), 'progress': progress,
        'n_clipped_screens': result.get('n_clipped_screens'),
        'clip_message': result.get('clip_message'),
        'cells_plot': screen._cells_view.describe(),
        'wells_plot': screen._wells_view.describe(),
    })
    if len(records['cells_scan']) != 10 or len(records['wells_scan']) != 8:
        raise RuntimeError('Expected five cell points and four well points, two fits each')
    check_curves(records, spec.detection_auroc, spec.n_replicates)
    capture('10_actual_results')
    if not any(r['status'] == 'ok' for r in records['cells_scan'] + records['wells_scan']):
        raise RuntimeError('No usable fit: do not freeze a tutorial claiming a working power estimate')
    if screen._cells_view.is_empty() or screen._wells_view.is_empty():
        raise RuntimeError('The actual result curves were not drawn')
    before = screen.visible_caveat_text()
    QTest.mouseClick(screen._caveats._more, Qt.LeftButton)
    settle(0.4)
    if screen.visible_caveat_text() == before or not screen._caveats._rest.isVisible():
        raise RuntimeError('The actual caveat toggle did not reveal the additional caveats')
    capture('11_full_caveats')
    QTest.mouseClick(screen._caveats._more, Qt.LeftButton)
    settle(0.4)
    if screen.visible_caveat_text() != before:
        raise RuntimeError('Hiding the extra caveats did not restore the original view')
    capture('12_results_and_limitations')
    check_recommendation(screen.answer_text(), spec.n_wells)
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'real_simulator': True, 'spec': asdict(spec),
        'fit_kwargs_overridden': False, 'scans_count': 2, 'fits': 18,
        'plotted_points': 9, 'replicates_per_point': 2,
        'every_replicate_in_detection_denominator': True,
        'unsupported_effect_guard_then_recovery': True,
        'guide_and_gene_units_demonstrated': True,
        'caveat_toggle_then_restoration': True,
        'measured_biological_results': False,
        'recommended_experimental_sample_size': False,
        'gui_export_demonstrated': False,
        'published': False,
    })
    print('Accepted actual Power / Design: 18 real fits, two marginal scans, nine points', flush=True)
