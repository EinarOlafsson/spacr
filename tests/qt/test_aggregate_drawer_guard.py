"""A disabled hot edge cannot arm a delayed drawer opening."""
from PySide6.QtWidgets import QWidget


def test_disabled_edge_keeps_the_open_timer_stopped(qapp, qtbot):
    from spacr.qt.widgets.drawer import EdgeDrawer

    host, panel = QWidget(), QWidget()
    qtbot.addWidget(host)
    drawer = EdgeDrawer(host, panel, width=180)
    drawer.set_enabled(False)
    drawer.arm()
    assert not drawer.is_open()
    assert not drawer._open_timer.isActive()
    assert not drawer._close_timer.isActive()
