"""Record a bounded native CV run with the explicit prepared-split workaround."""
import capture_ui
from capture_barcode_saved_plots import launch
from capture_ui_drv_a import Rec, modal_wait, type_text
from stage_lesson import DEFAULT_STAGE


def choose_existing_folder(app, screen, source, capture, settle):
    """Use the genuine source-picker action, including its empty-state update."""
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox, QAbstractButton
    previous = app.testAttribute(Qt.AA_DontUseNativeDialogs)
    app.setAttribute(Qt.AA_DontUseNativeDialogs, True)
    outcome = {}
    # Host: the clip starts before the pointer reaches Choose source data.
    rec = Rec(capture, '18_source_folder_choice', settle)

    def select():
        dialogs = [d for d in app.topLevelWidgets() if isinstance(d, QFileDialog) and d.isVisible()]
        if len(dialogs) != 1:
            outcome['error'] = 'Expected the actual directory chooser'
            for d in dialogs:
                d.reject()
            return
        dialog = dialogs[0]
        try:
            # The real picker can remember a personal folder and account-name
            # bookmarks. Keep this recording's visible navigation neutral.
            capture_ui.file_dialog_dir(capture, dialog, source.parent.resolve())
            dialog.setSidebarUrls([QUrl.fromLocalFile(str(source.parent.resolve()))])
            settle()
            line = dialog.findChild(QLineEdit, 'fileNameEdit')
            if line is None or not line.isVisible():
                raise RuntimeError('The actual directory entry is unavailable')
            type_text(capture, line, source.resolve(), tab=False)
            settle()
            dialog.accepted.connect(lambda: outcome.update(accepted=True))
            box = dialog.findChild(QDialogButtonBox)
            button = next(b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole)
            QTest.mouseClick(button, Qt.LeftButton)
        except Exception as error:
            outcome['error'] = str(error)
            dialog.reject()

    try:
        buttons = [b for b in screen.findChildren(QAbstractButton)
                   if b.isVisible() and b.text() == 'Choose source data']
        if len(buttons) != 1:
            raise RuntimeError('Expected the source-picker action on the fresh screen')
        rec.start()
        modal_wait(capture, 500, select)
        QTest.mouseClick(buttons[0], Qt.LeftButton)
        settle()
        if (outcome.get('error') or not outcome.get('accepted')
                or screen._settings_src_path() != str(source.resolve())):
            raise RuntimeError('The actual folder choice did not set the source: ' + str(outcome))
        if screen._empty_state_card.isVisible():
            raise RuntimeError('The source-picker did not dismiss the genuine empty-state banner')
        rec.stop()  # The actual acceptance and UI readback are still in canonical scene18.
        capture('18_source_folder_selected')
    finally:
        rec.abort()  # Failure cleanup, or a no-op after the successful single stop.
        app.setAttribute(Qt.AA_DontUseNativeDialogs, previous)


if __name__ == '__main__':
    raise SystemExit(launch('classify_merged', 'classify_canonical_existing_v2',
                           ['--run', '--settings-tour', '--ai-controls',
                            '--classifier-existing-split', str(DEFAULT_STAGE / 'classify_canonical_split_v2')],
                           300, stage=DEFAULT_STAGE / 'classify_canonical_capture_v2'))
