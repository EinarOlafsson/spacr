"""Capture current Gate chrome through real controls on a private completed table."""
from pathlib import Path
import hashlib
import time


def record_gate_chrome(app, window, screen, stage, captures, capture,
                       settle, write_json, database, timeout):
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QMenu, QPushButton

    database = Path(database).resolve()
    if not database.is_relative_to(Path(stage).resolve()) or not database.is_file():
        raise RuntimeError("Gate chrome recording requires a private existing database")
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    errors, picked = [], []

    def pick():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise RuntimeError("Load table did not open its actual file picker")
            watchdog = QTimer(dialog)
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(dialog.reject)
            watchdog.start(12000)
            dialog.setDirectory(str(database.parent))
            settle(.3)
            edit = dialog.findChild(QLineEdit, "fileNameEdit")
            edit.setFocus()
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, database.name)
            QTest.keyClick(edit, Qt.Key_Tab)
            capture("02_current_database_picker")
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open),
                             Qt.LeftButton)
            picked.append(True)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(350, pick)
    loaders = [button for button in screen.findChildren(QPushButton)
               if button.isVisible() and button.text() == "Load table…"]
    if len(loaders) != 1:
        raise RuntimeError("Gate Editor has no unique visible Load table control")
    QTest.mouseClick(loaders[0], Qt.LeftButton)
    if errors or not picked:
        raise RuntimeError("; ".join(errors) or "Database picker was not accepted")
    deadline = time.monotonic() + timeout
    while screen.is_busy() or screen.active_jobs():
        if time.monotonic() >= deadline:
            raise TimeoutError("Gate table load did not finish")
        settle(.1)
    if screen._frame is None or screen._frame.empty:
        raise RuntimeError("Gate Editor loaded no real measurements")

    def choose(box, value):
        index = box.findText(value)
        if index < 0:
            raise RuntimeError(f"Downloaded table has no {value}")
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)
        settle(.4)
        if box.currentText() != value:
            raise RuntimeError(f"Axis picker requested {value} but selected {box.currentText()}")

    choose(screen._x, "cell_area")
    choose(screen._y, "cell_channel_1_mean_intensity")
    QTest.mouseClick(screen.gates._mode_buttons["2D"], Qt.LeftButton)
    settle(1)
    capture("03_current_opaque_2d_plot")
    QTest.mouseClick(screen.gates._mode_buttons["3D"], Qt.LeftButton)
    choose(screen._z, "cell_perimeter")
    settle(1)
    capture("04_current_opaque_3d_plot")
    QTest.mouseClick(screen.gates._mode_buttons["2D"], Qt.LeftButton)
    settle(.5)
    canvas = screen.gates.canvas
    plot = canvas._canvas
    point = plot.mapTo(canvas, plot.rect().center())
    menus = []

    def record_menu():
        menu = app.activePopupWidget()
        if not isinstance(menu, QMenu):
            visible = [widget for widget in app.topLevelWidgets()
                       if isinstance(widget, QMenu) and widget.isVisible()]
            menu = visible[0] if len(visible) == 1 else None
        try:
            if not isinstance(menu, QMenu):
                raise RuntimeError("Right-click did not open the actual Gate graph menu")
            actions = [{"text": a.text(), "enabled": a.isEnabled()}
                       for a in menu.actions() if not a.isSeparator()]
            required = {"Change graph type", "Statistics…", "Save figure (zip)…"}
            if not required <= {a["text"] for a in actions}:
                raise RuntimeError("The graph menu lacks current figure actions")
            menus.extend(actions)
            capture("05_current_graph_figure_menu")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            if menu is not None:
                menu.close()

    QTimer.singleShot(350, record_menu)
    watchdog = QTimer(screen)
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: [widget.close() for widget in app.topLevelWidgets()
                                     if isinstance(widget, QMenu) and widget.isVisible()])
    watchdog.start(12000)
    # Offscreen QTest mouse clicks do not synthesize the native context event.
    # Deliver that event to the actual Gate canvas, whose policy owns this menu.
    app.sendEvent(canvas, QContextMenuEvent(QContextMenuEvent.Mouse, point,
                                           canvas.mapToGlobal(point)))
    watchdog.stop()
    if errors or not menus:
        raise RuntimeError("; ".join(errors) or "No recorded graph menu")
    figure = canvas.figure()
    box = figure.get_axes()[0].bbox
    ratio = float(getattr(plot, "device_pixel_ratio", 0) or plot.devicePixelRatioF())
    axis_point = plot.mapTo(canvas, QPoint(int((box.x0 + box.x1) / (2 * ratio)),
                                          int((figure.bbox.height - box.y0 + 15) / ratio)))
    if screen.axis_under(axis_point) != "x":
        raise RuntimeError("The actual axis context position does not identify X")
    axis_actions = []

    def record_axis_menu():
        visible = [widget for widget in app.topLevelWidgets()
                   if isinstance(widget, QMenu) and widget.isVisible()]
        menu = visible[0] if len(visible) == 1 else None
        try:
            if menu is None:
                raise RuntimeError("Axis context event opened no unique actual menu")
            axis_actions.extend({"text": action.text(), "enabled": action.isEnabled(),
                                 "checked": action.isChecked()}
                                for action in menu.actions() if not action.isSeparator())
            labels = {row["text"] for row in axis_actions}
            if not {"X axis: cell_area", "linear", "Set cutoffs…", "Clear cutoffs"} <= labels:
                raise RuntimeError("Actual X axis menu lacks measurement/scale/cutoff controls")
            if {"Change graph type", "Statistics…"} & labels:
                raise RuntimeError("X axis context incorrectly opened the graph menu")
            capture("06_current_axis_menu")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            if menu is not None:
                menu.close()

    QTimer.singleShot(350, record_axis_menu)
    watchdog.start(12000)
    app.sendEvent(canvas, QContextMenuEvent(QContextMenuEvent.Mouse, axis_point,
                                           canvas.mapToGlobal(axis_point)))
    watchdog.stop()
    if errors or not axis_actions:
        raise RuntimeError("; ".join(errors) or "No recorded axis menu")
    if hashlib.sha256(database.read_bytes()).hexdigest() != before:
        raise RuntimeError("Chrome recording modified the completed measurement database")
    write_json(captures / "gate_chrome_review.json", {
        "database_sha256": before, "database_unchanged": True,
        "loaded_rows": len(screen._frame), "table": screen._table,
        "axes": [screen._x.currentText(), screen._y.currentText(), screen._z.currentText()],
        "menu_actions": menus, "gates_created": False,
        "axis_menu_actions": axis_actions,
        "context_event_route": "Native QContextMenuEvent delivered to the actual GateCanvas owner; qualified offscreen Qt",
        "scope": "Current chrome and menu companion; separate real table, no replacement of prior scientific walkthrough",
        "published": False,
    })
