"""Well image-size controls change painted image bounds at stable decode quality."""
import hashlib

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

from tests.qt.test_725_overview_image_settings import source, view


def _decoration(fields, item):
    option = QStyleOptionViewItem()
    fields.initViewItemOption(option)
    option.rect = fields.visualItemRect(item)
    index = fields.indexFromItem(item)
    fields.itemDelegate().initStyleOption(option, index)
    return fields.style().subElementRect(QStyle.SE_ItemViewItemDecoration, option, fields)


@pytest.mark.parametrize("quality", [0, 128], ids=["original", "capped"])
def test_multiwell_size_slider_scales_real_image_decorations(view, source, qtbot, quality):
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source.iterdir()}
    view.preview_resolution.setCurrentIndex(view.preview_resolution.findData(quality))
    for well in ("A01", "A02"):
        view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), well)]))
    qtbot.waitUntil(lambda: len(view._well_panels) == 2
                    and all(panel.field_list.count() for panel in view._well_panels), timeout=10000)
    panels = list(view._well_panels)
    identities = [panel.field_list.item(0).data(Qt.UserRole).identity for panel in panels]
    for panel in panels:
        panel.field_list.item(0).setSelected(True)
    sizes = []
    for edge in (128, 256):
        panels[0].tile_slider.setValue(edge)
        qtbot.waitUntil(lambda: all(panel.tile_slider.value() == edge for panel in panels))
        for panel in panels:
            fields = panel.field_list
            fields.scrollToItem(fields.item(0))
        qtbot.waitUntil(lambda: all(_decoration(panel.field_list, panel.field_list.item(0)).width() == edge
                                   for panel in panels), timeout=10000)
        sizes.append([_decoration(panel.field_list, panel.field_list.item(0)).size()
                      for panel in panels])
        assert [panel.field_list.item(0).data(Qt.UserRole).identity for panel in panels] == identities
        assert all(panel.field_list.iconSize().width() == edge for panel in panels)
        assert all(panel.field_list.item(0).isSelected() for panel in panels)
        assert all(size.width() == edge and size.height() == round(edge * 40 / 64)
                   for size in sizes[-1])
    assert all(after.width() == 2 * before.width() and after.height() == 2 * before.height()
               for before, after in zip(*sizes))
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.iterdir()} == before
