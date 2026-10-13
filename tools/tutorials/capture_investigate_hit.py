"""Record Investigate Hit on the real screen example through actual controls.

Route: Regression -> Hits -> Browse the Regression run -> select a gene ->
Investigate selected… -> choose the database, predictions and guide
fractions with the real file pickers -> Investigate hit -> result tabs.
The example is a fresh unzip of Investigate_Hit_real_screen_example.zip
(tools/tutorials/build_investigate_hit_example.py); nothing is injected into
the form except through its visible widgets.
"""
from pathlib import Path
import json
import shutil
import time
import zipfile

from capture_database import _digest

GENE = '225160'
ZIP_NAME = 'Investigate_Hit_real_screen_example.zip'


def record_investigate_hit(app, window, host, stage, captures, capture,
                           settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QDialogButtonBox, QFileDialog, QLineEdit,
                                   QPushButton, QTabWidget)
    from spacr.qt.screens.hit_list import HitListScreen
    from spacr.qt.screens.model_explanation import InvestigateHitScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    stage = Path(stage)
    archive = stage / ZIP_NAME
    work = stage / 'tutorial'
    if work.exists():
        raise FileExistsError('Use a fresh stage: the example is modified by a run')
    work.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(work)
    example = work / 'Investigate_Hit_real_screen_example'
    run = example / 'regression_run/results/guide_permutation'
    database = example / 'measurements/measurements.db'
    predictions = example / 'cv_predictions.csv'
    fractions = run / 'regression_data.csv'
    manifest = json.loads((example / 'example_manifest.json').read_text())
    proof = dict(accepted=False, zip_sha256=_digest(archive), synthetic_data=False,
                 inputs={str(p.relative_to(work)): _digest(p) for p in
                         (database, predictions, fractions, run / 'results_gene.csv')},
                 example_cells=manifest['cells'], example_wells=manifest['wells'],
                 app_source_modified=False, published=False)
    deadline = time.monotonic() + timeout

    def click(widget):
        if time.monotonic() > deadline:
            raise TimeoutError('Investigate Hit recording timed out')
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual control unavailable: ' + (widget.objectName() or type(widget).__name__))
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    def fill(widget, value, key=Qt.Key_Tab):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        if str(value):
            QTest.keyClicks(widget, str(value))
        else:
            QTest.keyClick(widget, Qt.Key_Backspace)
        QTest.keyClick(widget, key)
        settle(.3)

    def picker(button, path, name):
        accepted, errors = [], []
        timer, watch = QTimer(window), QTimer(window)
        timer.setSingleShot(True); watch.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('Expected the Qt file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1600, 1000)
                dialog.setDirectory(str(Path(path).parent))
                settle(.4)
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                fill(edit, Path(path).name if Path(path).is_file() else str(path), key=Qt.Key_End)
                capture(name, desktop=True)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons()
                           if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1:
                    raise ValueError('No unique picker accept button')
                click(buttons[0])
            except Exception as error:  # reported below; never leave a modal open
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def abort():
            errors.append('File picker timed out')
            dialog = app.activeModalWidget()
            if dialog is not None:
                dialog.reject()

        timer.timeout.connect(handle); watch.timeout.connect(abort)
        timer.start(400); watch.start(20000)
        try:
            click(button)
        finally:
            timer.stop(); watch.stop()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'No accepted path')
        settle(.4)

    # Hits: the Regression host's own Hit List.
    folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'hit_list']
    if len(folds) != 1:
        raise ValueError('No unique Hit List fold')
    click(folds[0]); settle(.8)
    lists = [w for w in app.allWidgets() if isinstance(w, HitListScreen) and w.isVisible()]
    if len(lists) != 1:
        raise ValueError('Hit List is not visible')
    hits = lists[0]
    capture('02_hit_list_open')
    picker(hits._browse_button, run, '03_choose_regression_run')
    while hits.is_busy() or hits.active_jobs():
        if time.monotonic() > deadline:
            raise TimeoutError('Hit List did not load')
        settle(.1)
    settle(.5)
    loaded = hits.hits()
    proof['hit_list_genes'] = len(loaded) if loaded is not None else 0
    capture('04_ranked_genes')
    fill(hits._q_spin.lineEdit(), '.05'); settle(.4)
    proof['genes_at_q_0_05'] = hits._table.topLevelItemCount()
    if proof['genes_at_q_0_05'] != 0:
        raise ValueError('This example is described as having no gene at q <= 0.05')
    capture('05_no_gene_passes_q')
    fill(hits._q_spin.lineEdit(), '1'); settle(.4)
    fill(hits._query, GENE); settle(.4)
    items = [hits._table.topLevelItem(i) for i in range(hits._table.topLevelItemCount())]
    rows = [item for item in items if str(item.data(0, Qt.UserRole)) == GENE]
    if len(rows) != 1:
        raise ValueError('Expected one row for ' + GENE)
    hits._table.scrollToItem(rows[0])
    rect = hits._table.visualItemRect(rows[0])
    QTest.mouseClick(hits._table.viewport(), Qt.LeftButton, pos=rect.center()); settle(.4)
    if hits._table.currentItem() is not rows[0]:
        raise ValueError('The gene row was not selected')
    capture('06_selected_gene')
    buttons = [b for b in hits.findChildren(QPushButton) if b.text() == 'Investigate selected…']
    if len(buttons) != 1:
        raise ValueError('No unique Investigate selected… action')
    click(buttons[0]); settle(1.5)
    screens = [w for w in app.allWidgets() if isinstance(w, InvestigateHitScreen) and w.isVisible()]
    if len(screens) != 1:
        raise ValueError('Investigate Hit did not open from Hit List')
    panel = screens[0].investigate
    seeded = dict(folder=panel.regression_folder.text(), gene=panel.gene.text(),
                  guides=panel.guides.text(), direction=panel.direction.currentText(),
                  fdr=panel.gene.property('source_fdr'),
                  effect=panel.gene.property('source_effect'))
    proof['seeded_from_hit_list'] = seeded
    if seeded['gene'] != GENE or Path(seeded['folder']).resolve() != run.resolve():
        raise ValueError('Hit List hand-off did not carry the selected result')
    capture('07_investigate_prefilled')
    picker(panel.database.button, database, '08_choose_database')
    picker(panel.predictions.button, predictions, '09_choose_predictions')
    # Browsing fills the field without re-reading its header; Enter does.
    click(panel.predictions.edit); QTest.keyClick(panel.predictions.edit, Qt.Key_End)
    QTest.keyClick(panel.predictions.edit, Qt.Key_Return); settle(.5)
    options = [panel.score.itemText(i) for i in range(panel.score.count())]
    proof['score_choices'] = options
    if panel.score.currentText() != 'pred':
        raise ValueError(f'Expected pred as the score column, got {options}')
    picker(panel.fractions.button, fractions, '10_choose_fractions')
    capture('11_form_ready')
    proof['form'] = dict(database=panel.database.text(), predictions=panel.predictions.text(),
                         fractions=panel.fractions.text(), folder=panel.regression_folder.text(),
                         gene=panel.gene.text(), guides=panel.guides.text(),
                         score=panel.score.currentText(), direction=panel.direction.currentText(),
                         features=panel.features.text(), annotation=panel.annotation.text())
    if panel.results_section.shut:
        panel.results_section.set_folded(False)
    started = time.monotonic()
    click(panel.run_button); settle(.6)
    capture('12_running')
    while panel.result is None:
        if 'Could not' in panel.status.text():
            raise RuntimeError(panel.status.text())
        if time.monotonic() > deadline:
            raise TimeoutError('Investigation did not finish')
        settle(.5)
    settle(1)
    proof['run_seconds'] = round(time.monotonic() - started, 1)
    result = panel.result
    proof['result'] = dict(
        cells=len(result.cells), wells=len(result.wells), split_level=result.split_level,
        features=result.feature_columns, validation=result.validation,
        guide_evidence=json.loads(result.guide_evidence.to_json(orient='records')),
        threshold_sensitivity=json.loads(result.threshold_sensitivity.to_json(orient='records')),
        status=panel.status.text(), attribution_run_id=panel.attribution_run_id,
        gallery_rows=int(len(panel.investigation['gallery'])))
    outputs = panel.investigation['paths']
    proof['outputs'] = {name: dict(path=str(Path(p).relative_to(work)), sha256=_digest(p))
                        for name, p in outputs.items()}
    if len(result.cells) != manifest['cells'] or len(result.wells) != manifest['wells']:
        raise ValueError('Investigation did not use every example cell and well')
    tabs = panel.tabs
    names = {tabs.tabText(i): i for i in range(tabs.count())}
    for number, title in (('13', 'Evidence'), ('14', 'Wells'), ('15', 'Guides'),
                          ('16', 'Threshold sensitivity'), ('17', 'Candidate cells'),
                          ('18', 'Control-fitted embedding')):
        tab_bar = tabs.tabBar()
        QTest.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(names[title]).center())
        settle(.5)
        if tabs.currentIndex() != names[title]:
            raise ValueError('Tab did not open: ' + title)
        capture(f'{number}_{title.lower().replace(" ", "_").replace("-", "_")}')
    proof['accepted'] = True
    write_json(Path(captures) / 'scientific_acceptance.json', proof)
    screens[0].close(); settle(.3)
