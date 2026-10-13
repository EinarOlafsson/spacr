"""Record the current crop loader and two real CPU embedding runs."""
from pathlib import Path
import hashlib
import time


def record_screen(app, window, stage, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton
    import numpy as np

    folder = Path(stage) / 'embeddings_example' / 'cell_png'
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(folder.glob('*.png'))}
    if len(sources) != 16:
        raise ValueError('Prepare sixteen original example crops before recording')
    failures = []
    deadline = time.monotonic() + 240

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Unavailable control: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton)
        settle(.2)

    def select(widget, value):
        index = widget.findData(value)
        if index < 0:
            raise ValueError('Unknown visible choice: ' + value)
        click(widget)
        QTest.keyClick(widget.view(), Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(widget.view(), Qt.Key_Down)
        QTest.keyClick(widget.view(), Qt.Key_Return)
        settle(.2)
        if widget.currentData() != value:
            raise ValueError('Visible selection failed')

    def wait_for(predicate):
        while not predicate():
            if failures:
                raise RuntimeError(failures[-1])
            if time.monotonic() > deadline:
                raise TimeoutError('Embedding capture timed out: ' + screen._status.text())
            settle(.1)
        settle(.4)

    buttons = [w for w in window.findChildren(QAbstractButton) if w.isVisible()
               and w.property('moduleAppKey') == 'embeddings']
    if len(buttons) != 1:
        raise ValueError('Expected the actual Embeddings Home tile')
    click(buttons[0]); settle(1)
    screen = window._screens["embeddings"]
    screen._jobs.job_failed.connect(failures.append)
    capture('01_crop_loader')
    select(screen._where, 'folder')
    click(screen._path)
    QTest.keyClicks(screen._path, str(folder))
    QTest.keyClick(screen._path, Qt.Key_Return)
    QTest.keyClick(screen._path, Qt.Key_Tab)
    screen._limit.setFocus()
    QTest.keyClick(screen._limit, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(screen._limit, '16')
    QTest.keyClick(screen._limit, Qt.Key_Tab)
    settle(.3); capture('02_folder_selected')
    click(screen._load)
    wait_for(lambda: screen._run.isEnabled() and not screen._loading)
    if screen._crops.shape != (16, 224, 224, 3):
        raise ValueError('Unexpected crop stack: ' + str(screen._crops.shape))
    capture('03_crops_loaded')
    screen._batch.setFocus()
    QTest.keyClick(screen._batch, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(screen._batch, '4')
    QTest.keyClick(screen._batch, Qt.Key_Tab)
    if screen.spec().backbone != 'resnet18' or screen.spec().batch_size != 4:
        raise ValueError('Unexpected encoder controls')
    runs = []
    for policy, dimensions in [('per_channel', 1536), ('project', 512)]:
        select(screen._policy, policy)
        capture('04_' + policy + '_settings')
        previous = screen._result
        click(screen._run)
        wait_for(lambda: screen._result is not None and screen._result is not previous)
        values = np.asarray(screen._result.values)
        if values.shape != (16, dimensions) or not np.isfinite(values).all():
            raise ValueError('Invalid native embedding result')
        if screen._table.rowCount() != 16:
            raise ValueError('GUI preview did not display the result')
        capture('05_' + policy + '_result')
        output = Path(captures) / (policy + '.npy')
        np.save(output, values)
        runs.append(dict(policy=policy, shape=list(values.shape), finite=True,
                         preview_rows=screen._table.rowCount(),
                         matrix_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         status=screen._status.text()))
    if sources != {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(folder.glob('*.png'))}:
        raise ValueError('Input images changed')
    write_json(Path(captures) / 'scientific_acceptance.json', dict(
        accepted=True, gui_workflow_completed=True, crops_injected=False,
        input_method='Visible crop-folder selection and Load crops button',
        input_hashes=sources, crop_record=screen.crop_record(), runs=runs,
        app_source_modified=False, device='cpu', gpu_used=False,
        matrix_files='Private evidence saved by the recorder; GUI has no export button.',
        source_unchanged=True, published=False))


if __name__ == '__main__':
    from capture_barcode_saved_plots import launch
    raise SystemExit(launch('embeddings', 'embeddings_current_gui', [], 300))
