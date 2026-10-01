"""Record the current Plaque Assay preview: overlay display, Help route and Figure mode.

The synthetic example (four 768 px grayscale plaque images with label masks)
is copied into the private stage. Figure mode reads a SYNTHETIC figure built
from those same four images: a 2 x 2 panel figure with a condition label above
each panel, saved as a PDF. It is labelled synthetic on the figure itself and
is never presented as published data.
"""
from pathlib import Path
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
        figure.paste(tile, (x, y + label_h))
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


def record_plaque_current(app, window, screen, stage, captures, capture, settle,
                          write_json, timeout, figure_mode=False):
    from capture_geometry import capture_rect
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.widgets import plaque_preview as ppv

    manifest = prepare(stage)
    root = Path(manifest["root"])
    # The bundled checkpoint is a Cellpose 3 file that Cellpose 4 refuses;
    # the preview downloads the Model Zoo plaque model instead.
    model = "toxoplasma_plaque_v1"
    catalog = Path(__file__).resolve().parent / "authoring/catalog/24_plaque_settings.json"
    requested = read(catalog)
    requested.update(src=str(root), plaque_model=str(model), well_detection=False,
                     plate_format=None, well_diameter_mm=None)
    for key, value in requested.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError("No Plaque setting for " + key)
        settle(.03)
    state = {"manifest": manifest, "frames": {}}
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
    panel.load_source_async(str(root))
    _wait(settle, lambda: panel.current_path() is not None, 60, "Plaque preview did not list the example")
    panel._model_box.setCurrentText(str(model))
    settle(1)
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    settle(3)
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
    capture("16_overlay_preview")

    dialog = panel.open_overlay_settings()
    settle(1)
    target = window.mapToGlobal(QPoint(0, 0)) + QPoint(2300, 300)
    dialog.move(target)
    index = dialog.display.findData(ppv.OVERLAY_FILL)
    original = dialog.display.currentIndex()
    dialog.display.setCurrentIndex(index)
    settle(1.5)
    state["frames"]["16b_overlay_settings"] = capture_rect(dialog, window)
    capture("16b_overlay_settings")
    dialog.display.setCurrentIndex(original)
    settle(.5)
    dialog.close()
    settle(.5)

    help_action = next(a for a in window.menuBar().actions()
                       if a.text().replace("&", "") == "Help")
    menu = help_action.menu()
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(help_action).center())
    settle(.5)
    state["frames"]["20_help_route"] = capture_rect(menu, window)
    capture("20_help_route")
    menu.hide()
    settle(.5)

    if not figure_mode:
        # Figure mode's wells table currently shows item 501's estimate
        # columns with alpha features off, so it is not recorded.
        write_json(captures / "plaque_current.json", state)
        return state
    # Figure mode, through the panel's own switch.
    figures = make_synthetic_figure(root, stage / "plaque_figures" / root.name)
    button = panel._mode_switch._buttons[ppv.FIGURE_MODE]
    QTest.mouseClick(button, Qt.LeftButton)
    settle(2)
    if panel.mode() != ppv.FIGURE_MODE:
        raise ValueError("The Figure switch did not change the preview's mode")
    state["frames"]["30_figure_mode"] = capture_rect(panel, window)
    capture("30_figure_mode")
    panel.load_source_async(str(figures.parent))
    _wait(settle, lambda: panel.current_path() is not None, 120, "Figure mode listed no figure")
    settle(2)
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    settle(1)
    _wait(settle, lambda: not panel.preview_running(), timeout,
          "Figure preview did not finish: " + panel._status.text())
    settle(3)
    state["figure_preview"] = {"status": panel._status.text(),
                               "figure": figures.name,
                               "legend_visible": panel._legend_box.isVisible(),
                               "wells": len((getattr(panel, "_figure", None) or {}).get("regions", []) or [])
                               if isinstance(getattr(panel, "_figure", None), dict) else None}
    capture("31_figure_wells")
    write_json(captures / "plaque_current.json", state)
    if panel._legend_box.isVisible():
        QTest.mouseClick(panel._legend_skip, Qt.LeftButton)
        settle(2)
        capture("32_figure_annotate")
    write_json(captures / "plaque_current.json", state)
    QTest.mouseClick(panel._mode_switch._buttons[ppv.PLAQUE_MODE], Qt.LeftButton)
    settle(1)
    return state
