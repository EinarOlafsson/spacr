"""Record the real Regression Hits fold with existing unmodified results."""
from pathlib import Path
import shutil
import tempfile
import time
from capture_database import _digest
from hit_list_evidence import expected_hits, filtered, check_screen, check_rows, read_rows


def record_hits(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QPushButton, QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.hit_list import HitListScreen

    source = Path(stage) / 'regression_runs/example-9wut6lcv/results/guide_permutation'
    parent = Path(stage) / 'hit_list_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-gene-family-', dir=parent))
    data = work / 'results'; data.mkdir()
    originals = {str(p): _digest(p) for p in source.rglob('*') if p.is_file()}
    names = ('results_gene.csv', 'results_grna.csv', 'results.csv', 'results_significant.csv', 'model_summary.txt')
    copies = {}
    for name in names:
        if (source / name).is_file():
            shutil.copy2(source / name, data / name)
            copies[name] = _digest(data / name)
            if copies[name] != originals[str(source / name)]:
                raise ValueError('Real Hit List source changed during copying')
    expected = expected_hits(data)
    if len(expected) != 325 or len(read_rows(data / 'results_grna.csv')) != 434:
        raise ValueError('Expected checked Regression gene and guide families')
    proof = dict(lesson='48_hit_list', accepted=False, source_folder=str(source), source_files=originals,
                 private_folder=str(work), private_input_hashes=copies, synthetic_results=False,
                 app_source_modified=False, regression_recomputed=False, published=False)
    screen = None; deadline = time.monotonic() + timeout

    def click(widget):
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded Hit List recording timed out')
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Hit List control unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center()); settle(.25)

    def fill(widget, value):
        click(widget); QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        if str(value): QTest.keyClicks(widget, str(value))
        else: QTest.keyClick(widget, Qt.Key_Backspace)
        QTest.keyClick(widget, Qt.Key_Tab); settle(.3)

    def combo(widget, value):
        index = widget.findText(value)
        if index < 0: raise ValueError('No native choice ' + value)
        click(widget); QTest.keyClick(widget.view(), Qt.Key_Home)
        for _ in range(index): QTest.keyClick(widget.view(), Qt.Key_Down)
        QTest.keyClick(widget.view(), Qt.Key_Return); settle(.3)
        if widget.currentText() != value: raise ValueError('Native choice not applied')

    def picker(button, path, name):
        accepted = []; errors = []; timer = QTimer(window); watch = QTimer(window)
        timer.setSingleShot(True); watch.setSingleShot(True)
        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog): raise ValueError('Expected native Qt file picker')
                dialog.accepted.connect(lambda: accepted.append(True)); dialog.resize(1500, 1000)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path); capture(name, desktop=True)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1: raise ValueError('No unique picker accept button')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None: dialog.reject()
        def abort():
            errors.append('File picker timed out'); dialog = app.activeModalWidget()
            if dialog is not None: dialog.reject()
        timer.timeout.connect(handle); watch.timeout.connect(abort); timer.start(300); watch.start(15000)
        try: click(button)
        finally: timer.stop(); watch.stop(); timer.deleteLater(); watch.deleteLater()
        if errors or not accepted: raise ValueError('; '.join(errors) or 'No accepted path')
        settle(.4)

    def check(name, **options):
        settle(.3); wanted = filtered(expected, **options)
        proof.setdefault('checks', {})[name] = check_screen(screen, wanted)
        capture(name); return wanted

    def action(text):
        buttons = [b for b in screen.findChildren(QPushButton) if b.text() == text]
        if len(buttons) != 1: raise ValueError('No unique action ' + text)
        return buttons[0]

    try:
        tiles = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
                 (w.property('moduleAppKey') == 'regression' or w.property('navKey') == 'regression')]
        if not tiles: raise ValueError('No real Regression Home tile')
        click(max(tiles, key=lambda w: w.width() * w.height())); settle(.8)
        host = window._screens['regression']
        folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'hit_list']
        if len(folds) != 1: raise ValueError('No unique Hit List fold')
        capture('01_regression_hit_list_host'); click(folds[0]); settle(.8)
        panel = getattr(host, '_results_panel', None)
        card = getattr(host, '_figures_card', None)
        hits = getattr(panel, 'hits', None)
        tabs = getattr(host, '_results_tabs', None)
        proof['native_fold_state'] = dict(
            results_panel_exists=panel is not None,
            figures_card_visible=card.isVisible() if card is not None else None,
            hits_exists=hits is not None,
            hits_visible=hits.isVisible() if hits is not None else None,
            hits_selected=panel.tabs.currentWidget() is hits if panel is not None and hits is not None else None,
            results_page_selected=tabs.currentWidget() is getattr(host, '_results_page', None) if tabs is not None else None)
        found = [w for w in app.allWidgets() if isinstance(w, HitListScreen) and w.isVisible()]
        if len(found) != 1:
            capture('01b_hit_list_route_failure', desktop=True)
            proof['route_failure'] = [dict(type=type(w).__name__, visible=w.isVisible(),
                title=w.windowTitle()) for w in app.topLevelWidgets()]
            raise ValueError('No actual visible Hit List')
        screen = found[0]
        picker(screen._browse_button, data, '02_choose_completed_results_folder')
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic() > deadline: raise TimeoutError('Hit List worker did not finish')
            settle(.1)
        check('03_all_325_ranked_genes_not_discoveries')
        proof['family'] = dict(genes=325, guides=434, mixed_result_rows=len(read_rows(data / 'results.csv')),
                              min_gene_q=min(r['q_value'] for r in expected),
                              significant_genes=sum(r['q_value'] <= .05 for r in expected),
                              first_gene=expected[0]['gene'], first_effect=expected[0]['effect'],
                              source_well_support=[int(r['wells_with_gene']) for r in read_rows(data / 'results_gene.csv')],
                              hit_n_obs_values=sorted({h.n_obs for h in screen.hits()}))
        fill(screen._q_spin.lineEdit(), '.05'); check('04_no_genes_pass_q_point05', max_q=.05)
        fill(screen._q_spin.lineEdit(), '1'); check('05_all_candidates_restored')
        fill(screen._guides_spin.lineEdit(), '2'); check('06_at_least_two_tested_guides', min_guides=2)
        fill(screen._agreement_spin.lineEdit(), '1'); check('07_all_tested_guides_same_sign', min_guides=2, min_agreement=1)
        click(screen._drop_controls); check('08_hide_controls_not_recalculate_q', min_guides=2, min_agreement=1, exclude_controls=True)
        click(screen._drop_controls); fill(screen._agreement_spin.lineEdit(), '0'); fill(screen._guides_spin.lineEdit(), '0')
        combo(screen._direction, 'up'); check('09_positive_effect_not_significance', direction='up')
        fill(screen._effect_spin.lineEdit(), '.1'); check('10_positive_effect_magnitude_cut', direction='up', min_effect=.1)
        fill(screen._effect_spin.lineEdit(), '0'); combo(screen._direction, 'any')
        fill(screen._query, '239740'); check('11_search_actual_positive_control', query='239740')
        fill(screen._query, 'not_a_gene_in_this_example'); check('12_no_search_match', query='not_a_gene_in_this_example')
        fill(screen._query, '239740'); check('13_real_gene_returns', query='239740')
        fill(screen._query, ''); check('14_all_genes_restored')
        fill(screen._guides_spin.lineEdit(), '2'); fill(screen._agreement_spin.lineEdit(), '1')
        export_rows = check('15_candidate_subset_for_export', min_guides=2, min_agreement=1)
        exports = {}
        for suffix, label in [('csv', 'Export CSV…'), ('md', 'Export Markdown…'), ('html', 'Export HTML…')]:
            target = work / ('ranked_candidates.' + suffix)
            picker(action(label), target, '16_export_' + suffix + '_picker')
            if not target.is_file() or target.stat().st_size == 0: raise ValueError('Missing genuine export')
            exports[suffix] = dict(path=str(target), sha256=_digest(target), bytes=target.stat().st_size)
            if suffix == 'csv': exports[suffix]['values_checked'] = check_rows(read_rows(target), export_rows)
            capture('17_export_' + suffix + '_complete')
        proof['exports'] = exports
        proof['export_filters'] = screen.current_filters()
        fill(screen._guides_spin.lineEdit(), '0'); fill(screen._agreement_spin.lineEdit(), '0')
        check('18_original_ranked_population_restored')
        proof['accepted'] = True
    finally:
        if screen is not None: screen.close()
        window.close(); settle(.3)
        proof['all_originals_unchanged'] = all(Path(p).is_file() and _digest(p) == h for p, h in originals.items())
        proof['all_private_inputs_unchanged'] = all(_digest(data / p) == h for p, h in copies.items())
        proof['active_jobs_after_close'] = screen.active_jobs() if screen is not None else 0
        write_json(captures / 'hit-list-workflow.json', proof)
    if not proof['all_originals_unchanged'] or not proof['all_private_inputs_unchanged']:
        raise ValueError('Original or private Hit List input changed')
