"""Record the actual Annotate fold using clearly artificial demonstration labels."""
from pathlib import Path
import time

from agreement_demo import COLUMNS, prepare, digest, expected, verify


def record_agreement(app, window, host, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtTest import QTest
    from spacr.qt.screens.agreement import AgreementScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    manifest = prepare(stage)
    database = Path(manifest['database'])
    proof = dict(lesson='23_agreement', accepted=False, data=manifest, runs=[],
        app_source_modified=False, labels_are_artificial=True, human_performance_claim=False,
        crop_checks=[], published=False)
    write_json(captures / 'agreement_acceptance.json', proof)
    screen = None
    deadline = time.monotonic() + timeout

    def wait(predicate, message):
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError(message)
            settle(.1)
        settle(.3)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Agreement control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    def select(names):
        for index in range(screen._columns_list.count()):
            item = screen._columns_list.item(index)
            desired = item.text() in names
            if (item.checkState() == Qt.Checked) != desired:
                screen._columns_list.scrollToItem(item)
                rect = screen._columns_list.visualItemRect(item)
                from PySide6.QtCore import QPoint
                QTest.mouseClick(screen._columns_list.viewport(), Qt.LeftButton,
                                 pos=QPoint(rect.left()+10, rect.center().y()))
                settle(.1)
        if screen.selected_columns() != list(names):
            raise ValueError('Native checkbox clicks did not select the intended columns')

    def run(name, columns):
        click(screen._btn_compute)
        wait(lambda: not screen.is_busy() and screen.active_jobs() == 0,
             'Agreement computation did not settle')
        report = screen.report()
        if report is None or report.columns != list(columns):
            raise ValueError('No report for the requested comparison')
        wanted = expected(manifest['rows'], columns)
        actual = {key:getattr(report,key) for key in wanted if key not in ('pairs','disagreement_paths')}
        actual['pairs'] = [{key:(pair.confusion.values.tolist() if key == 'confusion' else getattr(pair,key))
                           for key in wanted['pairs'][0]} for pair in report.pairs]
        actual['disagreement_paths'] = screen.disagreement_paths()
        verify(actual,wanted)
        table = screen.kappa_rows()
        expected_rows = [[p['column_a'],p['column_b'],str(p['n_compared']),str(p['n_abstained']),
                          f"{p['percent_agreement']:.1%}",f"{p['kappa']:+.3f}"] for p in wanted['pairs']]
        if [row[:6] for row in table] != expected_rows:
            raise ValueError('Displayed pair identities or numbers differ from the independent calculation')
        for index,pair in enumerate(wanted['pairs']):
            screen._pair_combo.setCurrentIndex(index)
            settle(.2)
            if screen.confusion_rows() != [[str(label),*[str(n) for n in row]]
                                          for label,row in zip((1,2),pair['confusion'])]:
                raise ValueError('Displayed confusion matrix differs from the independent counts')
        screen._pair_combo.setCurrentIndex(0)
        settle(.2)
        capture(name)
        proof['runs'].append(dict(name=name, columns=list(columns), independent=wanted,
            actual=actual, displayed_pairs=table, summary=screen._summary.text(),
            warnings=report.warnings, all_values_verified=True))
        write_json(captures / 'agreement_acceptance.json',proof)

    try:
        folds = [button for button in host.findChildren(FoldButton)
                 if button.isVisible() and button.app_key == 'agreement']
        if len(folds) != 1:
            raise ValueError('No unique Annotate -> Annotator Agreement fold')
        QTest.mouseMove(folds[0]);settle(.5);capture('02_agreement_fold')
        click(folds[0])
        wait(lambda: any(s.isVisible() for s in host.findChildren(AgreementScreen)),
             'Actual Agreement fold did not open')
        screens = [s for s in host.findChildren(AgreementScreen) if s.isVisible()]
        if len(screens) != 1:
            raise ValueError('More than one visible Agreement screen')
        screen = screens[0]
        capture('03_agreement_empty')
        click(screen._path_edit)
        QTest.keyClick(screen._path_edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(screen._path_edit, str(database))
        QTest.keyClick(screen._path_edit, Qt.Key_Return)
        wait(lambda: screen.database_path() == str(database), 'Typed database did not open')
        if not set(COLUMNS).issubset(screen.available_columns()):
            raise ValueError('The three artificial annotation passes were not discovered')
        capture('04_demo_database')
        select(COLUMNS[:1])
        if screen._btn_compute.isEnabled():
            raise ValueError('A single column incorrectly enables a comparison')
        capture('05_one_column_not_a_comparison')
        select(COLUMNS[:2])
        if not screen._btn_compute.isEnabled():
            raise ValueError('The positive two-column comparison is disabled')
        capture('06_two_columns')
        run('07_pairwise_results',COLUMNS[:2])
        capture('08_confusion_matrix')
        for index,path in enumerate(screen.disagreement_paths()):
            screen._review_table.setCurrentCell(index,0)
            if not screen.select_disagreement(index) or screen.current_crop_path() != path:
                raise ValueError('The selected disagreement does not show its own crop')
            pixmap = screen._crop_label.pixmap()
            original = QPixmap(str(database.parent.parent/path))
            reference = original.scaled(pixmap.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            got = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            wanted = reference.toImage().convertToFormat(QImage.Format_RGBA8888)
            if got.size() != wanted.size() or bytes(got.constBits()) != bytes(wanted.constBits()):
                raise ValueError('Displayed crop pixels do not match the selected original image')
            settle(.3);capture(f'09_disagreement_crop_{index+1}')
            proof['crop_checks'].append(dict(path=path, all_rgba_pixels_equal=True,
                                             width=got.width(),height=got.height()))
        select(COLUMNS)
        capture('10_three_columns')
        run('11_three_pass_results',COLUMNS)
        select(COLUMNS[:2]);run('12_pairwise_restored',COLUMNS[:2])
        proof['accepted'] = True
    finally:
        if screen is not None:
            screen.close();settle(.3)
        proof['source_database_unchanged'] = digest(manifest['source_database']) == manifest['source_database_sha256']
        proof['private_database_unchanged'] = digest(database) == manifest['database_sha256']
        proof['all_crop_files_unchanged'] = all(digest(row['original']) == row['sha256'] == digest(row['copied'])
                                                for row in manifest['crops'])
        if not all(proof[key] for key in ['source_database_unchanged','private_database_unchanged','all_crop_files_unchanged']):
            proof['accepted'] = False
        write_json(captures / 'agreement_acceptance.json',proof)
    if not proof['accepted']:
        raise ValueError('Agreement tutorial capture or preservation failed')
