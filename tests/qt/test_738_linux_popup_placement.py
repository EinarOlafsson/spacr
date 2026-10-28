"""Initial dialog geometry keeps native ownership and deliberate positions."""
import pytest
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QLineEdit, QMainWindow, QMenu, QVBoxLayout, QWidget
from spacr.qt import dialogs

pytestmark = pytest.mark.qt


@pytest.fixture
def placement_filter(qapp, request):
    saved = dialogs._DETACHER, dialogs._DETACHED_APP
    if dialogs.detach_all_dialogs(qapp):
        installed = dialogs._DETACHER
        def restore():
            from spacr.qt.gil_priority import _stop_watching_application_events
            _stop_watching_application_events(qapp, installed)
            dialogs._DETACHER, dialogs._DETACHED_APP = saved
        request.addfinalizer(restore)


def _owner(qtbot, main=False):
    owner = QMainWindow() if main else QWidget()
    qtbot.addWidget(owner)
    owner.resize(480, 340)
    owner.move(170, 120)
    owner.show()
    qtbot.waitExposed(owner)
    return owner


@pytest.mark.parametrize('file_browser', [False, True])
def test_initial_dialog_is_centred_after_sizing_with_real_transient_owner(qtbot, placement_filter, file_browser):
    owner = _owner(qtbot)
    parent = QWidget(owner)
    if file_browser:
        dialog = QFileDialog(parent)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    else:
        dialog = QDialog(parent)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLineEdit('owned setting', dialog))
        dialog.resize(260, 180)
    qtbot.addWidget(dialog)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.show()
    qtbot.waitExposed(dialog)
    from spacr.qt.hidpi import screen_for_widget
    work = screen_for_widget(owner).availableGeometry()
    frame = dialog.frameGeometry()
    expected = owner.frameGeometry().center()
    assert work.contains(frame)
    if (expected.x() - frame.width() // 2 >= work.left()
            and expected.y() - frame.height() // 2 >= work.top()):
        assert (frame.center() - expected).manhattanLength() <= 2
    assert dialog.windowHandle().transientParent() is owner.windowHandle()
    assert dialog.parentWidget() is parent
    assert dialog.windowModality() == Qt.WindowModal
    assert dialog.windowType() == Qt.Window
    if file_browser:
        assert dialog.testOption(QFileDialog.DontUseNativeDialog)


@pytest.mark.parametrize('restore', [False, True])
def test_explicit_or_restored_position_and_later_user_moves_are_preserved(qtbot, placement_filter, restore):
    owner = _owner(qtbot)
    dialog = QDialog(owner)
    qtbot.addWidget(dialog)
    dialog.resize(200, 140)
    dialog.move(50, 60)
    if restore:
        saved = dialog.saveGeometry()
        dialog.move(300, 280)
        assert dialog.restoreGeometry(saved)
    position = dialog.geometry().topLeft() if restore else dialog.pos()
    dialog.show()
    qtbot.waitExposed(dialog)
    assert (dialog.geometry().topLeft() if restore else dialog.pos()) == position
    dialog.move(95, 105)
    dialog.hide()
    owner.move(250, 220)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.pos() == QPoint(95, 105)


@pytest.mark.parametrize('hidden_parent', [False, True])
def test_unowned_and_hidden_owner_file_browsers_get_the_visible_main_window(qtbot, placement_filter, hidden_parent):
    owner = _owner(qtbot, main=True)
    hidden = QDialog(owner) if hidden_parent else None
    if hidden is not None:
        qtbot.addWidget(hidden)
    browser = QFileDialog(hidden)
    qtbot.addWidget(browser)
    assert not browser.testOption(QFileDialog.DontUseNativeDialog)
    browser.ensurePolished()
    assert browser.parentWidget() is owner
    assert not browser.testOption(QFileDialog.DontUseNativeDialog)
    browser.show()
    qtbot.waitExposed(browser)
    assert browser.nativeParentWidget().window() is owner
    assert browser.windowHandle().transientParent() is owner.windowHandle()


@pytest.mark.parametrize('owner,size,work,expected', [
    (QRect(-1600, 100, 800, 600), QSize(400, 200), QRect(-1920, 30, 1920, 1050), QPoint(-1399, 300)),
    (QRect(-2050, -60, 400, 300), QSize(500, 400), QRect(-1920, 30, 1920, 1050), QPoint(-1920, 30)),
    (QRect(-1200, 100, 400, 300), QSize(1800, 1000), QRect(-1600, 0, 1600, 900), QPoint(-1600, 0)),
])
def test_logical_geometry_handles_negative_monitors_edges_and_oversized_frames(owner, size, work, expected):
    frame = dialogs._centred_dialog_frame(owner, size, work)
    assert frame.topLeft() == expected
    assert frame.size() == size


def test_embedded_pages_and_anchored_menus_keep_their_placement(qtbot, placement_filter):
    owner = _owner(qtbot)
    layout = QVBoxLayout(owner)
    page = QDialog(owner)
    page.setWindowFlags(Qt.Widget)
    layout.addWidget(page)
    page.show()
    assert page.window() is owner and not page.property('spacrInitiallyPlaced')
    menu = QMenu(owner)
    menu.addAction('anchored action')
    menu.popup(QPoint(220, 190))
    qtbot.waitUntil(menu.isVisible)
    assert menu.pos() == QPoint(220, 190)
    assert not menu.property('spacrInitiallyPlaced')
    menu.close()
