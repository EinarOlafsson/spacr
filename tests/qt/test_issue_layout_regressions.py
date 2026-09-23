"""Reported clipping and whole-row selection in real Qt layouts."""
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage
from PySide6.QtWidgets import QMainWindow, QPushButton, QStackedWidget, QWidget
from spacr.qt.first_run import _TourOverlay, TourStep, _widget_rect_in_window


@pytest.mark.parametrize('size', [(380, 300), (900, 600)])
def test_tour_text_scrolls_with_actions_in_view(qtbot, qt_theme_applied, size):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.resize(*size)
    window.move(10, 10)
    window.show()
    overlay = _TourOverlay(window, [TourStep('Long translated title ' * 4, 'Long explanation ' * 150)])
    overlay.show()
    qtbot.wait(50)
    available = overlay.screen().availableGeometry()
    visible = overlay.rect().intersected(QRect(overlay.mapFromGlobal(available.topLeft()), available.size()))
    assert visible.contains(overlay._card.geometry())
    for button in (overlay._skip_btn, overlay._next_btn):
        assert button.visibleRegion().contains(button.rect())
    bar = overlay._text_scroll.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    qtbot.wait(10)
    assert overlay._next_btn.visibleRegion().contains(overlay._next_btn.rect())
    overlay._skip_btn.click()


def test_menu_highlight_uses_menu_bar_label_not_hidden_popup(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().setNativeMenuBar(False)
    menu = window.menuBar().addMenu('Demos')
    menu.addAction('Example')
    window.resize(600, 450)
    window.show()
    qtbot.wait(20)
    rect = _widget_rect_in_window(menu, window)
    expected = window.menuBar().actionGeometry(menu.menuAction())
    assert rect == QRect(window.menuBar().mapTo(window, expected.topLeft()), expected.size())
    assert not menu.isVisible()


@pytest.mark.parametrize('width', [900, 1400])
@pytest.mark.parametrize('factor', [1., 1.5])
def test_annotate_actions_wrap_without_clipping(qtbot, qt_theme_applied, width, factor):
    from spacr.qt.screens.annotate import AnnotateScreen
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    screen = AnnotateScreen()
    stack.addWidget(screen)
    font = QFont(screen.font())
    font.setPointSizeF(font.pointSizeF() * factor)
    screen.setFont(font)
    stack.setFixedSize(width, 800)
    stack.show()
    qtbot.wait(80)
    toolbar = screen._btn_open.parentWidget()
    buttons = toolbar.findChildren(QPushButton)
    assert len(buttons) >= 16
    for button in buttons:
        assert button.width() >= button.sizeHint().width(), button.text()
        assert toolbar.rect().contains(button.geometry()), button.text()
        assert button.visibleRegion().contains(button.rect()), button.text()
    assert len({button.y() for button in buttons}) > 1
    screen.close()


@pytest.mark.parametrize('target', ['title', 'margin', 'blank_diagram', 'node', 'edge'])
def test_click_any_pipeline_area_selects_its_details(qtbot, target):
    from spacr.qt.widgets.sample_project import SampleProjectDialog
    dialog = SampleProjectDialog()
    qtbot.addWidget(dialog)
    dialog.resize(1100, 850)
    dialog.show()
    index = 1
    item = dialog.list.item(index)
    dialog.list.scrollToItem(item)
    qtbot.wait(50)
    row = dialog.list.itemWidget(item)
    graph = dialog.diagrams[index]
    dialog.list.setCurrentRow(0)
    if target == 'title':
        widget = row.layout().itemAt(0).widget()
        position = widget.rect().center()
    elif target == 'margin':
        widget, position = row, QPoint(2, 2)
    else:
        widget = graph.viewport()
        if target == 'blank_diagram':
            position = QPoint(widget.width() // 2, 2)
            assert graph.itemAt(position) is None
        elif target == 'node':
            node = next(iter(graph.nodes.values()))
            position = graph.mapFromScene(node.sceneBoundingRect().center())
        else:
            edge = graph.edges[0]
            position = graph.mapFromScene(edge.path().pointAtPercent(.5))
    qtbot.mouseClick(widget, Qt.LeftButton, pos=position)
    assert dialog.list.currentRow() == index
    assert dialog.selected()['id'] == dialog._entries[index]['id']
    assert set(dialog.details.cards) >= set(graph.nodes)
