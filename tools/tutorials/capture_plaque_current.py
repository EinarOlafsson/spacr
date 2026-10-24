"""Record the current Plaque Assay preview: overlay display, Help route and Figure mode.

The synthetic example (four 768 px grayscale plaque images with label masks)
is copied into the private stage. Figure mode reads a SYNTHETIC figure built
from those same four images: a 2 x 2 panel figure with a condition label above
each panel, saved as a PDF. It is labelled synthetic on the figure itself and
is never presented as published data.
"""
import capture_ui
from capture_ui_drv_a import dismiss_menu, modal_wait
from capture_ui_drv_c import Scenes
from capture_ui_drv_e import guarded
from capture_ui_drv_f import open_overlay_settings
from pathlib import Path
import os
import time

from plaque_demo import prepare
from stage_lesson import read


def make_synthetic_figure(source: Path, folder: Path) -> Path:
    """Write SYNTHETIC_plaque_figure.pdf: the four example images as labelled panels."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    names = [("plate1_A01_control_1.tif", "Control 1"),
             ("plate1_A02_control_2.tif", "Control 2"),
             ("plate1_B01_treatment_1.tif", "Treatment 1"),
             ("plate1_B02_treatment_2.tif", "Treatment 2")]
    panel, gap, label_h, title_h = 600, 60, 70, 110
    width = 2 * panel + 3 * gap
    height = title_h + 2 * (label_h + panel) + 3 * gap
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    try:
        import matplotlib
        font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), 44)
        title_font = ImageFont.truetype(str(font_path), 40)
    except Exception:                                   # noqa: BLE001
        font = title_font = ImageFont.load_default()
    draw.text((gap, 30), "SYNTHETIC example figure (spaCR tutorial)", fill="black", font=title_font)
    for index, (name, label) in enumerate(names):
        row, column = divmod(index, 2)
        x = gap + column * (panel + gap)
        y = title_h + gap + row * (label_h + panel + gap)
        draw.text((x, y), label, fill="black", font=font)
        with Image.open(source / name) as image:
            array = np.asarray(image).astype(float)
        low, high = np.percentile(array, (1, 99.5))
        scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1) * 255
        tile = Image.fromarray(scaled.astype("uint8")).convert("RGB").resize((panel, panel))
        # Show each image as a round well of a plate: the image inside a
        # circle, a dark plate rim around it.
        well = Image.new("L", (panel, panel), 0)
        ImageDraw.Draw(well).ellipse((12, 12, panel - 12, panel - 12), fill=255)
        plate = Image.new("RGB", (panel, panel), (60, 60, 64))
        plate.paste(tile, (0, 0), well)
        figure.paste(plate, (x, y + label_h))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SYNTHETIC_plaque_figure.pdf"
    figure.save(path, "PDF", resolution=150)
    return path


def _wait(settle, condition, timeout, what):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        settle(0.2)


def _run_with_download(panel, settle, timeout, state, key):
    """Press Run preview; if a model must be downloaded, use the panel's own button and run again."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    settle(3)
    _wait(settle, lambda: not panel.preview_running(), timeout, "Preview did not settle")
    if panel._download_btn.isVisible():
        state[key] = panel._download_btn.text()
        QTest.mouseClick(panel._download_btn, Qt.LeftButton)
        settle(2)
        _wait(settle, lambda: panel._download_btn.isHidden() or panel._download_btn.isEnabled(),
              timeout, "Download did not finish: " + panel._status.text())
        settle(2)
        QTest.mouseClick(panel._run_btn, Qt.LeftButton)
        settle(1)
    _wait(settle, lambda: not panel.preview_running(), timeout, "Preview did not finish")


@guarded
def record_plaque_current(app, window, screen, stage, captures, capture, settle,
                          write_json, timeout, figure_mode=True):
    from capture_geometry import capture_rect
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.widgets import plaque_preview as ppv

    import faulthandler
    import sys
    # 24 a02 hung twice at 100% CPU for 2h40m with no output after 01_module: a stall that no
    # settle()-based timeout can see because the GUI thread itself is stuck. Dump every thread
    # stack and exit when the step budget is exceeded, so the hang becomes a failure with a cause.
    # Re-armed at every step: a single step that stalls for 7 minutes dumps every thread stack and exits
    # (the whole-tour budget of 2*timeout+300 s was longer than the turn limit, so no dump was ever written).
    def arm():
        # file=sys.stderr failed with "UnsupportedOperation: fileno" (the host replaces sys.stderr with an object
        # that has no file descriptor); the real descriptor is sys.__stderr__. A watchdog must never crash the run.
        try:
            faulthandler.dump_traceback_later(420, exit=True, file=sys.__stderr__)
        except Exception as exc:                                  # noqa: BLE001
            print('plaque tour: no faulthandler watchdog: %r' % (exc,), flush=True)

    arm()

    def step(text):
        print('plaque tour step: ' + text, file=sys.stderr, flush=True)
        arm()

    step('prepare')
    manifest = prepare(stage)
    root = Path(manifest["root"])
    catalog = Path(__file__).resolve().parent / "authoring/catalog/24_plaque_settings.json"
    requested = read(catalog)
    requested.pop('plaque_model', None)   # the current default model
    requested.update(src=str(root), well_detection=False,
                     plate_format=None, well_diameter_mm=None)
    step('fill settings')
    for key, value in requested.items():
        step('setting ' + key)
        if not capture_ui.fill_setting(capture, screen, key, value):
            raise ValueError("No Plaque setting for " + key)
        settle(.03)
    state = {"manifest": manifest, "frames": {}}
    S = Scenes(capture, settle)         # host: each clip starts before its actions
    panel = getattr(screen, "_live_preview", None)
    switch = getattr(screen, "_preview_switch", None)
    if panel is None or switch is None:
        raise ValueError("The Plaque preview or its Live switch is missing")
    if not panel.isVisible():
        QTest.mouseClick(switch, Qt.LeftButton)
        settle(1.5)
    if not panel.isVisible():
        raise ValueError("The Plaque preview is hidden")
    if panel.mode() != ppv.PLAQUE_MODE:
        panel.set_mode(ppv.PLAQUE_MODE)
        settle()
    step('list the example images')
    panel.load_source_async(str(root))
    _wait(settle, lambda: panel.current_path() is not None, 60, "Plaque preview did not list the example")
    step('run the preview')
    run = panel._run_btn
    if not run.isVisible() or not run.isEnabled():
        raise ValueError('The preview Run button is not usable: visible=%s enabled=%s status=%r'
                         % (run.isVisible(), run.isEnabled(), panel._status.text()))
    step('click Run at ' + repr(run.mapToGlobal(run.rect().center())) + ' (window ' + repr(window.geometry()) + ')')
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    step('Run clicked')
    settle(3)
    step('wait for the preview to settle')
    _wait(settle, lambda: not panel.preview_running(), timeout, "Plaque preview did not settle")
    if panel._download_btn.isVisible():
        # The Model Zoo plaque model is not in the private profile yet: use the
        # preview's own Download button, then run the preview again.
        state["model_download"] = panel._download_btn.text()
        QTest.mouseClick(panel._download_btn, Qt.LeftButton)
        settle(2)
        _wait(settle, lambda: panel._download_btn.isHidden() or panel._download_btn.isEnabled(),
              timeout, "Model download did not finish: " + panel._status.text())
        settle(2)
        QTest.mouseClick(panel._run_btn, Qt.LeftButton)
        settle(1)
    _wait(settle, lambda: not panel.preview_running() and (
          panel._plaque_result is not None or "fail" in panel._status.text().lower()),
          timeout, "Plaque preview did not finish: " + panel._status.text())
    if panel._plaque_result is None:
        capture("16_preview_failed")
        raise ValueError("Plaque preview failed: " + panel._status.text())
    settle(2)
    fit = [b for b in panel.findChildren(QPushButton) if b.text().replace("&", "") == "Fit image"
           and b.isVisible()]
    if fit:
        QTest.mouseClick(fit[0], Qt.LeftButton)
        settle(1.5)
    state["plaque_preview"] = {"status": panel._status.text(),
                               "count": panel._plaque_result.get("count")}
    state["frames"]["16_overlay_preview"] = capture_rect(panel, window)
    S.end("16_overlay_preview", panel)
    S.begin("16b_overlay_settings")

    dialog = open_overlay_settings(capture, settle, panel)
    settle(1)
    target = window.mapToGlobal(QPoint(0, 0)) + QPoint(2300, 300)
    capture_ui.dialog_geometry(capture, dialog, None, target)
    index = dialog.display.findData(ppv.OVERLAY_FILL)
    original = dialog.display.currentIndex()
    capture_ui.select_combo(capture, dialog.display, index)
    settle(1.5)
    state["frames"]["16b_overlay_settings"] = capture_rect(dialog, window)
    S.end("16b_overlay_settings", dialog.display)
    S.begin("20_help_route")
    capture_ui.select_combo(capture, dialog.display, original)
    settle(.5)
    capture_ui.close_dialog(capture, dialog)
    settle(.5)

    help_action = next(a for a in window.menuBar().actions()
                       if a.text().replace("&", "") == "Help")
    menu = help_action.menu()
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(help_action).center())
    settle(.5)
    state["frames"]["20_help_route"] = capture_rect(menu, window)
    S.end("20_help_route", menu)
    if figure_mode:
        figures = make_synthetic_figure(root, stage / "plaque_figures" / root.name)
        S.begin("30_figure_mode")
    dismiss_menu(capture, menu)
    settle(.5)

    if not figure_mode:
        # Figure mode's wells table currently shows item 501's estimate
        # columns with alpha features off, so it is not recorded.
        write_json(captures / "plaque_current.json", state)
        return state
    # Figure mode, through the panel's own switch.
    button = panel._mode_switch._buttons[ppv.FIGURE_MODE]
    QTest.mouseClick(button, Qt.LeftButton)
    settle(2)
    if panel.mode() != ppv.FIGURE_MODE:
        raise ValueError("The Figure switch did not change the preview's mode")
    state["frames"]["30_figure_mode"] = capture_rect(panel, window)
    S.end("30_figure_mode", button)
    paper = os.environ.get("SPACR_TUTORIAL_PLAQUE_PAPER")
    answered = []
    S.begin("30b_from_a_paper" if paper else "31_figure_wells")
    if paper:
        # From a paper…: an open-access paper by DOI, fetched into the stage.
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication, QDialog
        papers = stage / "plaque_papers"
        papers.mkdir(exist_ok=True)
        filled = []

        def fill():
            dialogs = [w for w in QApplication.topLevelWidgets()
                       if isinstance(w, QDialog) and w.objectName() == "PlaquePaperDialog" and w.isVisible()]
            if not dialogs:
                if len(filled) < 100:
                    filled.append(None)
                    QTimer.singleShot(200, fill)
                return
            dialog = dialogs[0]
            capture_ui.set_text(capture, dialog.reference, paper)
            capture_ui.set_text(capture, dialog.folder, str(papers))
            settle(0.5)
            S.end("30b_from_a_paper", dialog)
            S.begin("31_figure_wells", lead=0.3)
            filled.append(dialog)
            capture_ui.accept_dialog(capture, dialog)
        modal_wait(capture, 300, fill)
        QTest.mouseClick(panel._paper_btn, Qt.LeftButton)
        if not any(filled):
            raise ValueError("The From a paper dialog did not open")
        state["paper"] = paper
        settle(3)
        _wait(settle, lambda: getattr(panel, "_paper_batch", None) is None
              and not panel._paper_jobs.is_busy() and panel.current_path() is not None,
              timeout, "The paper's figures were not read: " + panel._status.text())
        settle(2)
        wanted = os.environ.get("SPACR_TUTORIAL_PLAQUE_FIGURE", "")
        if wanted:
            names = [panel._picker.itemText(i) for i in range(panel._picker.count())]
            state["paper_figures"] = names
            match = [i for i, name in enumerate(names) if wanted in name]
            if not match:
                raise ValueError(f"No figure named like {wanted!r}: {names}")
            capture_ui.select_combo(capture, panel._picker, match[0])
            settle(2)
    else:
        # Drop the PDF on the screen, as a user does: Figure mode reads the
        # paper into a folder of figures and lists them. A question box, if one
        # opens, is answered with its Figure-mode choice and recorded.
        from PySide6.QtCore import QMimeData, QPointF, QTimer, QUrl
        from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
        from PySide6.QtWidgets import QApplication, QMessageBox

        answered = []

        def answer():
            for box in QApplication.topLevelWidgets():
                if isinstance(box, QMessageBox) and box.isVisible():
                    for button in box.buttons():
                        if button.property("plaque_mode") == ppv.FIGURE_MODE:
                            answered.append(button.text())
                            capture_ui.press(capture, button)
                            return
            if len(answered) == 0 and time.monotonic() < answer.until:
                QTimer.singleShot(300, answer)
        answer.until = time.monotonic() + 60
        QTimer.singleShot(300, answer)

        def mime():
            data = QMimeData()
            data.setUrls([QUrl.fromLocalFile(str(figures))])
            return data
        point = screen.rect().center()
        # Keep every mime object and event alive until Qt is done with it.
        state["_keep"] = held = [mime(), mime(), mime()]
        events = [QDragEnterEvent(point, Qt.CopyAction, held[0], Qt.LeftButton, Qt.NoModifier),
                  QDragMoveEvent(point, Qt.CopyAction, held[1], Qt.LeftButton, Qt.NoModifier),
                  QDropEvent(QPointF(point), Qt.CopyAction, held[2], Qt.LeftButton, Qt.NoModifier)]
        capture_ui.code_action(capture, 'PDF dropped by synthetic drag events: a file-manager drag is '
                               'not available in the capture host; set SPACR_TUTORIAL_PLAQUE_PAPER '
                               'to use From a paper instead')
        for event in events:
            QApplication.sendEvent(screen, event)
        held.extend(events)
        settle(3)
        _wait(settle, lambda: getattr(panel, "_paper_batch", None) is None
              and not panel._paper_jobs.is_busy() and panel.current_path() is not None,
              timeout, "Figure mode did not read the PDF: " + panel._status.text())
        settle(2)
    state["figure_question"] = answered
    state.pop("_keep", None)
    state["figure_listing"] = str(panel.current_path())
    settle(2)
    _run_with_download(panel, settle, timeout, state, "detector_download")
    settle(3)
    state["figure_preview"] = {"status": panel._status.text(),
                               "figure": figures.name,
                               "legend_visible": panel._legend_box.isVisible(),
                               "wells": len((getattr(panel, "_figure", None) or {}).get("regions", []) or [])
                               if isinstance(getattr(panel, "_figure", None), dict) else None}
    fit = [b for b in panel.findChildren(QPushButton) if b.text().replace("&", "") == "Fit image"
           and b.isVisible()]
    if fit:
        QTest.mouseClick(fit[0], Qt.LeftButton)
        settle(1.5)
    S.end("31_figure_wells", panel)
    write_json(captures / "plaque_current.json", state)
    if panel._all_btn.isVisible() and panel._all_btn.isEnabled():
        S.begin("32_figure_all_wells")
        QTest.mouseClick(panel._all_btn, Qt.LeftButton)
        settle(2)
        _wait(settle, lambda: not panel.preview_running(), timeout,
              "Find plaques in all wells did not finish: " + panel._status.text())
        settle(3)
        state["all_wells"] = panel._status.text()
        S.end("32_figure_all_wells", panel)
        write_json(captures / "plaque_current.json", state)
    if panel._legend_box.isVisible():
        S.begin("32_figure_annotate")
        QTest.mouseClick(panel._legend_skip, Qt.LeftButton)
        settle(2)
        S.end("32_figure_annotate", panel._legend_box)
    write_json(captures / "plaque_current.json", state)
    QTest.mouseClick(panel._mode_switch._buttons[ppv.PLAQUE_MODE], Qt.LeftButton)
    settle(1)
    faulthandler.cancel_dump_traceback_later()
    return state
