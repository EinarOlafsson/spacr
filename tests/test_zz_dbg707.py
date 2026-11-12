from tests.qt.test_707_the_window_fits_a_720p_screen import _open_make_masks, folder  # noqa
from PySide6.QtWidgets import QWidget, QLabel

def test_dbg(qtbot, folder):
    window, screen = _open_make_masks(qtbot, folder, (1280, 680))
    bar = screen._title_bar
    print("BAR", bar.size().toTuple(), bar.minimumHeight(), bar.minimumSizeHint().toTuple())
    for l in bar.findChildren(QLabel):
        if l.isVisibleTo(bar):
            print("LBL", l.objectName(), repr(l.text()[:30]), l.geometry().getRect(), l.hasHeightForWidth(), l.heightForWidth(l.width()) if l.hasHeightForWidth() else l.sizeHint().height())
    print("SCREEN min", screen.minimumSizeHint().toTuple(), window.minimumSizeHint().toTuple())
    assert False
