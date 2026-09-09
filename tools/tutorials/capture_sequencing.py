"""Drive the actual archive picker and verify its bounded paired FASTQ files."""
from __future__ import annotations

import gzip
import hashlib
import time
from pathlib import Path


def inspect_pair(paths, expected_reads):
    """Require complete, equal-length mate records with matching read identities."""
    if len(paths) != 2 or len(set(map(str, paths))) != 2:
        raise RuntimeError('A paired example needs two distinct FASTQ files')
    records, reports = [], []
    for raw in paths:
        path = Path(raw)
        identities = []
        with gzip.open(path, 'rt') as handle:
            while header := handle.readline():
                sequence, separator, quality = (handle.readline().rstrip('\n') for _ in range(3))
                if (not header.startswith('@') or not separator.startswith('+')
                        or not sequence or len(sequence) != len(quality)):
                    raise RuntimeError(f'Incomplete or malformed FASTQ record in {path.name}')
                identities.append(header.split()[0].removesuffix('/1').removesuffix('/2'))
        if len(identities) != expected_reads:
            raise RuntimeError(f'Expected {expected_reads} reads, got {len(identities)} in {path.name}')
        records.append(identities)
        reports.append({'file': path.name, 'reads': len(identities),
                        'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if records[0] != records[1]:
        raise RuntimeError('The paired FASTQ read identities do not match in order')
    return reports


def schedule_picker(app, captures, capture, settle, write_json, timeout):
    """Schedule real dialog actions before the caller clicks Load test data."""
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from spacr.qt.widgets.sra_picker import SraPicker

    result = {}

    def choose():
        dialogs = [d for d in app.topLevelWidgets() if isinstance(d, SraPicker) and d.isVisible()]
        if len(dialogs) != 1:
            result['error'] = 'The actual archive picker did not open'
            return
        dialog = dialogs[0]
        try:
            if not dialog._files:
                raise RuntimeError(dialog._blurb.text())
            first_run = dialog._files[0].run
            selected = [f for f in dialog._files if f.run == first_run]
            if len(selected) != 2 or {f.mate for f in selected} != {1, 2}:
                raise RuntimeError('The first archive run does not have exactly two mates')
            capture('02_archive_all_runs')
            for row in range(dialog._list.count()):
                item = dialog._list.item(row)
                if item.data(Qt.UserRole).run != first_run and item.checkState() == Qt.Checked:
                    dialog._list.scrollToItem(item)
                    area = dialog._list.visualItemRect(item)
                    QTest.mouseClick(dialog._list.viewport(), Qt.LeftButton,
                                     pos=QPoint(area.left() + 10, area.center().y()))
            dialog._reads.setFocus()
            QTest.keyClick(dialog._reads, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(dialog._reads, '10000')
            QTest.keyClick(dialog._reads, Qt.Key_Tab)
            if dialog.max_reads() != 10000 or dialog.chosen_files() != selected:
                raise RuntimeError('The visible bounded archive choices were not retained')
            dialog._list.scrollToTop()
            settle()
            capture('02_archive_bounded_choice')
            result.update(run=first_run, library=selected[0].library,
                          reads_per_file=10000, source_urls=[f.url for f in selected],
                          full_files_requested=False, estimate=dialog._estimate.text())
            QTest.mouseClick(dialog._download, Qt.LeftButton)
            deadline = time.monotonic() + timeout
            while dialog.isVisible() and dialog._worker is not None:
                if time.monotonic() > deadline:
                    raise TimeoutError('The bounded archive download exceeded the time limit')
                settle(0.2)
            if dialog.isVisible() or len(dialog.written) != 2:
                raise RuntimeError(dialog._estimate.text())
            result['files'] = inspect_pair(dialog.written, 10000)
            result['paired_read_identities_match'] = True
        except Exception as error:
            result['error'] = str(error)
            capture('02_archive_error')
            dialog.reject()
        write_json(captures / 'sequencing_download.json', result)

    QTimer.singleShot(1200, choose)
    return result
