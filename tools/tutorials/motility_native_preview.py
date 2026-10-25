"""Drive the unmodified Motility preview and retain its numerical results."""
from dataclasses import asdict
import time
from contextlib import nullcontext

import capture_ui


def record_preview(app, window, screen, root, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.widgets.card import Card

    def click(widget, *, settle_after=True):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('A genuine Motility preview control is unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        if settle_after:
            settle(.25)

    def fill(widget, value):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(.3)
        if float(widget.value()) != float(value):
            raise ValueError('The native preview field differs from the requested value')

    panel = screen._motility_preview
    if panel is None:
        raise ValueError('The Motility screen has no native preview panel')
    with capture_ui.clip(capture, '07_actual_preview'):
        for folder in [screen._console_folder, *[c.folder for c in screen.findChildren(Card)
                if c.title_label is not None and c.title_label.text().lstrip('▾▸▼▶ ') == 'System'
                and c.folder is not None]]:
            if not folder.shut:
                click(folder.heading)
        if not screen._motility_preview_card.isVisible():
            click(screen._preview_switch)
        capture_ui.point_at(capture, panel, dwell=1.5)
    live = capture_ui.live(capture)
    errors, accepted = [], []
    timer, watchdog = QTimer(window), QTimer(window)
    timer.setSingleShot(True)
    watchdog.setSingleShot(True)

    def choose():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise ValueError('The real preview directory picker did not open')
            dialog.accepted.connect(lambda: accepted.append(True))
            capture_ui.dialog_geometry(capture, dialog, (1400, 950))
            if not live:
                dialog.setDirectory(str(root.parent))
                dialog.setSidebarUrls([QUrl.fromLocalFile(str(root.parent))])
                dialog.setHistory([str(root.parent)])
            settle(.3)
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            click(edit)
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(root))
            if not live:
                capture('08_actual_input_picker')
            box = dialog.findChild(QDialogButtonBox)
            choices = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
            if len(choices) != 1:
                raise ValueError('No unique folder acceptance button')
            click(choices[0])
        except Exception as error:
            errors.append(str(error))
            if dialog is not None:
                dialog.reject()

    def abort():
        errors.append('The Motility folder dialog timed out')
        if app.activeModalWidget() is not None:
            app.activeModalWidget().reject()

    watchdog.timeout.connect(abort)
    with (capture_ui.clip(capture, '08_actual_input_picker') if live else nullcontext()):
        capture_ui.when_modal(capture, 400, choose)
        watchdog.start(90000 if live else 20000)
        try:
            click(panel._pick_btn)
        finally:
            watchdog.stop()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'The folder selection was not accepted')
        deadline = time.monotonic() + 30
        while not panel._groups or panel._plane_busy or panel._plane_jobs.pending_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError('Native Motility input scanning did not settle')
            settle(.1)
    with capture_ui.clip(capture, '09_actual_plane_and_unit_settings'):
        fill(panel._n_channels, 2)
        settle(.5)
        fill(panel._tracked_plane, 2)
        fill(panel._pathogen_plane, -1)
        fill(panel._max_frames, 8)
        fill(panel._min_len, 3)
        fill(panel._max_disp, 50)
        fill(panel._pixels_per_um, 0)
        fill(panel._seconds_per_frame, 0)
        if panel._propagate_btn.isChecked():
            click(panel._propagate_btn)
        if panel._straightness_filter.isChecked():
            click(panel._straightness_filter)
    with capture_ui.clip(capture, '10_actual_uncalibrated_preview'):
        completed = []
        click(panel._run_btn, settle_after=False)
        worker = panel._worker
        for _ in range(100):            # a real click reaches the button over a later event-loop turn
            if worker is not None:
                break
            settle(.1)
            worker = panel._worker
        if worker is None:
            raise ValueError('The actual preview Run button started no worker (run enabled=%s, visible=%s)'
                             % (panel._run_btn.isEnabled(), panel._run_btn.isVisible()))
        worker.finished_result.connect(lambda points, error: completed.append(dict(
            rows=0 if points is None else len(points), error=error)))
        deadline = time.monotonic() + timeout
        while panel._worker is not None or not completed:
            if time.monotonic() > deadline:
                click(panel._cancel_btn)
                raise TimeoutError('The native Motility preview timed out')
            settle(.1)
        settle(.4)
        if completed[-1]['error'] or panel._points is None:
            raise ValueError('The native Motility preview failed: ' + repr(completed))
        cached = panel._points.copy(deep=True)
        results = {}

        def snapshot(name):
            if not cached.equals(panel._points) or panel._worker is not None or len(completed) != 1:
                raise ValueError('A live metric change reread or changed the cached points')
            capture_ui.point_at(capture, panel._stats_label, dwell=1.2)
            value = dict(params=panel.current_params(), summary=asdict(panel._summary),
                         knobs=dict(n_channels=panel._n_channels.value(), tracked_plane=panel._tracked_plane.value(),
                             pathogen_plane=panel._pathogen_plane.value(), min_length=panel._min_len.value(),
                             max_displacement=panel._max_disp.value(), straightness=panel._straightness.value(),
                             straightness_filter=panel._straightness_filter.isChecked(),
                             pixels_per_um=panel._pixels_per_um.value(), seconds_per_frame=panel._seconds_per_frame.value(),
                             propagate=panel._propagate_btn.isChecked()),
                         stats_text=panel._stats_label.text(), status_text=panel._status.text(),
                         points=panel._points.to_dict('records'), tracks=panel._tracks.to_dict('records'),
                         plot_visible=panel._plot.isVisible(), plot_is_null=panel._plot.pixmap().isNull())
            results[name] = value
            write_json(captures / (name + '.json'), value)

        snapshot('10_actual_uncalibrated_preview')
    def scene(name, *actions):
        with capture_ui.clip(capture, name):
            for action in actions:
                action()
            snapshot(name)

    scene('11_actual_all_short', lambda: fill(panel._min_len, 9))
    scene('12_actual_length_restored', lambda: fill(panel._min_len, 3))
    scene('13_actual_straightness_filter', lambda: fill(panel._straightness, .95),
          lambda: click(panel._straightness_filter))
    scene('14_actual_filter_restored', lambda: click(panel._straightness_filter))
    scene('15_actual_one_unit_unknown', lambda: fill(panel._pixels_per_um, 2))
    scene('16_actual_hypothetical_calibration', lambda: fill(panel._seconds_per_frame, 60))
    scene('17_actual_unknown_units_restored', lambda: fill(panel._pixels_per_um, 0),
          lambda: fill(panel._seconds_per_frame, 0))
    return dict(worker_results=completed, snapshots=results, source=str(root),
                acquired_calibration_claim=False, hypothetical_calibration=dict(pixels_per_um=2, seconds_per_frame=60),
                inference_rerun=False, published=False)
