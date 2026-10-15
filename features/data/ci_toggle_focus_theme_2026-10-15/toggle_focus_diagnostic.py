from pathlib import Path
import json
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed or "test_keyboard_focus_remains_visible_when_the_field_paints_the_switch" not in item.nodeid:
        return
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPainter, QColor
        from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionFocusRect
        import PySide6
        switch = None
        tb = call.excinfo.tb
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "test_keyboard_focus_remains_visible_when_the_field_paints_the_switch":
                switch = tb.tb_frame.f_locals["switch"]
                break
            tb = tb.tb_next
        if switch is None:
            return
        option = QStyleOptionFocusRect()
        option.initFrom(switch)
        result = {"nodeid":item.nodeid,"PySide6":PySide6.__version__,"style_name":switch.style().objectName(),"style_class":switch.style().metaObject().className(),"application_style":QApplication.style().objectName(),"has_focus":switch.hasFocus(),"keyboard_focus_attribute":switch.testAttribute(Qt.WA_KeyboardFocusChange),"state":option.state.value,"keyboard_state":bool(option.state & QStyle.State_KeyboardFocusChange),"has_focus_state":bool(option.state & QStyle.State_HasFocus),"background_valid":option.backgroundColor.isValid(),"variants":{}}
        for variant in ("original", "keyboard_flag", "background", "keyboard_and_background"):
            image = QImage(switch.size(), QImage.Format_ARGB32_Premultiplied)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                focus = QStyleOptionFocusRect()
                focus.initFrom(switch)
                if "keyboard" in variant:
                    focus.state |= QStyle.State_KeyboardFocusChange
                if "background" in variant:
                    focus.backgroundColor = QColor("black")
                switch.style().drawPrimitive(QStyle.PE_Widget, focus, painter, switch)
                focus.rect = switch.rect().adjusted(3, 3, -3, -3)
                switch.style().drawPrimitive(QStyle.PE_FrameFocusRect, focus, painter, switch)
            finally:
                painter.end()
            result["variants"][variant] = sum(image.pixelColor(x,y).alpha() > 0 for y in range(image.height()) for x in range(image.width()))
        path = Path("/mnt/wd4tb/scratch/ci-toggle-focus-20261015/native-style-events.jsonl")
        with path.open("a") as stream:
            stream.write(json.dumps(result)+"\n")
    except Exception as error:
        with Path("/mnt/wd4tb/scratch/ci-toggle-focus-20261015/diagnostic-errors.txt").open("a") as stream:
            stream.write(type(error).__name__+": "+str(error)+"\n")
