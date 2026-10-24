"""Check the real Methods & Results digest against a completed run's seed.

No provider is contacted, checkpoint loaded or application source modified.
The explicit refusal is intentional while the recorded seed contradicts the
deterministic prose. A successful-looking digest alone cannot close a lesson.
"""
import json
from pathlib import Path
import shutil
import tempfile
import time
import capture_ui
from capture_ui_drv_c import Scenes
from capture_database import _digest


def check_recorded_seed(declared, digest, methods):
    """Require both the digest and its prose to preserve this run's seed."""
    actual = digest['run'].get('seed')
    if actual != declared or f'The random seed was {declared}.' not in methods:
        raise ValueError(f'Draft seed {actual} contradicts recorded permutation seed {declared} or its prose')


def record_methods(app, window, stage, captures, capture, settle, write_json, timeout, *, review_export=False, export_picker_only=False):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.methods_export import MethodsExportScreen

    if export_picker_only and not review_export:
        raise ValueError('Picker-only capture requires unchanged draft export verification')
    native_capture = capture
    recorded_frames = []

    def capture(name, **kwargs):
        if not export_picker_only or name == '08_export_unreviewed_markdown_picker':
            native_capture(name, **kwargs)
            recorded_frames.append(name)

    def _begin(name):
        if export_picker_only and name != '08_export_unreviewed_markdown_picker':
            return False
        recorded_frames.append(name)
        return native_capture.begin(name)

    capture.begin = _begin
    capture.end = lambda: native_capture.end()
    capture.host = getattr(native_capture, 'host', None)
    scenes = Scenes(capture, settle)

    stage = Path(stage)
    sources = dict(project=stage/'regression_runs/example-9wut6lcv',
                   run_dir=stage/'runs/2026-09-09_154327_c79bb2f6__regression')
    parent = stage/'methods_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-regression-digest-', dir=parent))
    originals = {}; copies = {}
    for key, root in sources.items():
        for p in root.rglob('*'):
            if p.is_symlink(): raise ValueError('No symlinks in the tutorial source copy')
            if p.is_file(): originals[str(p)] = _digest(p)
        shutil.copytree(root, work/key)
        for p in (work/key).rglob('*'):
            if p.is_file():
                copies[str(p.relative_to(work))] = _digest(p)
                if _digest(p) != originals[str(root/p.relative_to(work/key))]:
                    raise ValueError('Methods source changed during copying')
    manifest = json.loads((work/'run_dir/manifest.json').read_text(encoding="utf-8"))
    declared = manifest['seeds']['declared']['guide_permutation_seed']
    proof = dict(lesson='49_methods_results', accepted=False, source_files=originals,
                 private_folder=str(work), private_input_hashes=copies,
                 declared_permutation_seed=declared, provider_contacted=False,
                 app_source_modified=False, published=False,
                 export_picker_only=export_picker_only, recorded_frames=recorded_frames)
    deadline = time.monotonic()+timeout; screen = None

    def click(widget):
        if time.monotonic()>deadline: raise TimeoutError('Methods recording timed out')
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Methods control unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center()); settle(.3)

    def fill(widget, value):
        click(widget); QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        if str(value): QTest.keyClicks(widget, str(value))
        else: QTest.keyClick(widget, Qt.Key_Backspace)
        QTest.keyClick(widget, Qt.Key_Tab); settle(.2)

    def tab(index):
        capture_ui.select_tab(capture, screen._tabs, index); settle(.4)
        if screen._tabs.currentIndex()!=index: raise ValueError('Native Methods tab did not change')

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='regression' or w.property('navKey')=='regression')]
        if not tiles: raise ValueError('No real Regression tile')
        scenes.begin('01_regression_methods_host')
        click(max(tiles,key=lambda w:w.width()*w.height())); settle(.8)
        host=window._screens['regression']
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='methods_export']
        if len(folds)!=1: raise ValueError('No unique Methods fold')
        scenes.end('01_regression_methods_host', host)
        scenes.begin('02_real_project_journal_and_result_sources')
        click(folds[0]); settle(.8)
        found=[w for w in app.allWidgets() if isinstance(w,MethodsExportScreen) and w.isVisible()]
        if len(found)!=1: raise ValueError('No actual visible Methods screen')
        screen=found[0]
        paths=dict(project=work/'project',run_dir=work/'run_dir',
                   results=work/'project/results/guide_permutation',model='')
        for key,path in paths.items(): fill(screen._fields[key],path)
        if screen.sources()!={k:str(v) for k,v in paths.items()}: raise ValueError('Wrong entered sources')
        scenes.end('02_real_project_journal_and_result_sources', screen._fields['project'], screen._fields['results'], desktop=True)
        scenes.begin('03_actual_deterministic_methods')
        click(screen._build_button)
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic()>deadline: raise TimeoutError('Methods digest worker did not finish')
            settle(.1)
        if screen.last_error or screen.digest() is None: raise ValueError('Actual digest build failed')
        proof['digest']=screen.digest(); proof['methods']=screen._methods_view.toPlainText()
        proof['results']=screen._results_view.toPlainText(); proof['caveats']=screen._caveats_view.toPlainText()
        proof['status']=screen._provenance.text()
        scenes.end('03_actual_deterministic_methods', screen._methods_view, desktop=True)
        scenes.begin('04_actual_seed_caveat')
        tab(2); scenes.end('04_actual_seed_caveat', screen._caveats_view, desktop=True)
        scenes.begin('05_digest_run_record')
        tab(3); scenes.end('05_digest_run_record', screen._tabs, desktop=True)
        scenes.begin('06_zero_significant_genes_and_ranked_candidates')
        tab(1); scenes.end('06_zero_significant_genes_and_ranked_candidates', screen._results_view, desktop=True)
        scenes.begin('07_copy_sections_is_not_scientific_approval')
        click(screen._copy_button)
        proof['clipboard_matches_visible_sections']=app.clipboard().text()==screen.text()
        if not proof['clipboard_matches_visible_sections']: raise ValueError('Copied sections differ')
        scenes.end('07_copy_sections_is_not_scientific_approval', screen._copy_button, desktop=True)
        actual=proof['digest']['run'].get('seed')
        proof['resolved_seed']=actual; proof['seed_matches_run']=actual==declared
        if review_export:
            # This is a distinct review/export lesson, not a relaxation of the
            # original scientific-accuracy hold. Never change the visible draft.
            from methods_review_evidence import inspect_review
            target = work / 'UNREVIEWED_methods_results.md'
            failures = []; chosen = []
            def choose_export():
                dialog = app.activeModalWidget()
                try:
                    if not isinstance(dialog, QFileDialog):
                        raise ValueError('Expected the genuine native export picker')
                    fill(dialog.findChild(QLineEdit, 'fileNameEdit'), target)
                    scenes.end('08_export_unreviewed_markdown_picker', dialog, desktop=True)
                    scenes.begin('09_actual_export_completed_not_approved')
                    box = dialog.findChild(QDialogButtonBox)
                    buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                    if len(buttons) != 1:
                        raise ValueError('No unique visible Save action')
                    capture_ui.accept_file_dialog(capture, dialog, buttons[0]); chosen.append(str(target))
                except Exception as exc:
                    failures.append(str(exc))
                    if dialog is not None: dialog.reject()
            watch = QTimer(window); watch.setSingleShot(True)
            def expired():
                failures.append('Export picker timed out')
                dialog = app.activeModalWidget()
                if dialog is not None: dialog.reject()
            watch.timeout.connect(expired)
            capture_ui.when_modal(capture, 400, choose_export)
            watch.start(120000 if capture_ui.live(capture) else 15000)
            scenes.begin('08_export_unreviewed_markdown_picker')
            click(screen._export_button); watch.stop(); settle(.5)
            if failures or chosen != [str(target)] or not target.is_file():
                raise ValueError(f'Actual export did not finish: {failures}')
            proof['review'] = inspect_review(proof, target)
            proof['original_draft_scientifically_approved'] = False
            proof['scope'] = 'Recorded review and unchanged export, NOT a scientifically approved manuscript'
            scenes.end('09_actual_export_completed_not_approved', screen._export_button, desktop=True)
        else:
            check_recorded_seed(declared, proof['digest'], proof['methods'])
        proof['accepted']=True
    finally:
        if screen is not None: screen.close()
        window.close(); settle(.3)
        proof['all_originals_unchanged']=all(Path(p).is_file() and _digest(p)==h for p,h in originals.items())
        proof['all_private_inputs_unchanged']=all((work/p).is_file() and _digest(work/p)==h for p,h in copies.items())
        proof['active_jobs_after_close']=screen.active_jobs() if screen is not None else 0
        write_json(captures/'methods-workflow.json',proof)
    if not proof['all_originals_unchanged'] or not proof['all_private_inputs_unchanged']:
        raise ValueError('Methods inputs changed')
    if review_export:
        write_json(captures/'scientific_acceptance.json', proof)
