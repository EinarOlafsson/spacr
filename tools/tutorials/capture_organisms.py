"""Record organism guides using the visible Home, diagram and assay controls."""
from __future__ import annotations


def record_organism(app, window, stage, captures, capture, settle, write_json, key):
    """Capture a guide without installing external applications or starting analysis."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QLabel

    if key not in ('toxoplasma', 'plasmodium', 'candida'):
        raise ValueError('Unknown organism guide')

    def click(button):
        visible = button.visibleRegion()
        if not button.isEnabled() or visible.isEmpty():
            raise RuntimeError('A required navigation control is unavailable')
        QTest.mouseClick(button, Qt.LeftButton, pos=visible.boundingRect().center())
        settle(.6)

    def home():
        if not window._startup.isVisible():
            choices = [button for button in window.findChildren(QAbstractButton)
                       if button.property('navKey') == '__home__' and button.isVisible()]
            click(max(choices, key=lambda button: button.width() * button.height()))
        if not window._startup.isVisible():
            raise RuntimeError('The Home control did not open Home')

    def open_guide():
        home()
        tiles = [button for button in window._startup.findChildren(QAbstractButton)
                 if button.property('moduleAppKey') == key]
        tabs = window._startup._tabs
        for index in range(tabs.count()):
            choices = [tile for tile in tiles if tabs.widget(index).isAncestorOf(tile)]
            if choices:
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                 pos=tabs.tabBar().tabRect(index).center())
                settle(.4)
                capture('00_home')
                click(max(choices, key=lambda tile: tile.width() * tile.height()))
                break
        else:
            raise RuntimeError('The organism Home tile is absent')
        screen = window._screens.get(key)
        if screen is None or not screen.isVisible():
            raise RuntimeError('The Home tile did not open the organism guide')
        return screen

    screen = open_guide()
    capture('01_organism')
    handle = screen._splitter.handle(1)
    origin = handle.rect().center()
    QTest.mousePress(handle, Qt.LeftButton, pos=origin)
    QTest.mouseMove(handle, origin + QPoint(220, 0), delay=100)
    QTest.mouseRelease(handle, Qt.LeftButton, pos=origin + QPoint(220, 0))
    settle(.8)
    diagram = screen._diagram
    screen._scroll.ensureWidgetVisible(diagram)
    settle(.5)
    selector = diagram.selector
    selected = []
    for index in (0, 2):
        item = selector.item(index)
        selector.scrollToItem(item)
        region = selector.visualItemRect(item)
        QTest.mouseMove(selector.viewport(), region.center())
        QTest.mouseClick(selector.viewport(), Qt.LeftButton,
                         pos=QPoint(region.left() + 10, region.center().y()))
        settle(.3)
        if item.checkState() != Qt.Checked:
            raise RuntimeError('The visible compartment checkbox did not select its component')
        selected.append(item.text())
    if len(diagram.artwork.selected) != 2:
        raise RuntimeError('The diagram did not retain both selected components')
    capture('02_diagram')
    click(diagram.clear_button)
    if diagram.artwork.selected:
        raise RuntimeError('Clear components did not reset the diagram')
    screen._module_scroll.verticalScrollBar().setValue(0)
    capture('03_assays' if key == 'toxoplasma' else '03_modules')
    evidence = {'organism': key, 'home_tile_clicked': True,
                'selected_components': selected, 'clear_components_verified': True,
                'analysis_started': False, 'external_application_started': False,
                'tiles': [{'key': tile.property('organismModuleKey'),
                           'enabled': tile.isEnabled()} for tile in screen._tiles]}
    if key == 'toxoplasma':
        tile = next(tile for tile in screen._tiles
                    if tile.property('organismModuleKey') == 'host_pathogen')
        screen._module_scroll.ensureWidgetVisible(tile)
        settle(.4)
        click(tile)
        target = window._screens.get('host_pathogen')
        if target is None or not target.isVisible():
            raise RuntimeError('The Host–Pathogen assay tile did not open its module')
        capture('04_host_pathogen')
        evidence['host_pathogen_tile_clicked'] = True
        screen = open_guide()
        starplast = next(tile for tile in screen._tiles
                         if tile.property('organismModuleKey') == 'starplast')
        screen._module_scroll.ensureWidgetVisible(starplast)
        settle(.4)
        capture('05_starplast')
        home()
        capture('06_home_return')
    else:
        if any(tile.isEnabled() for tile in screen._tiles):
            raise RuntimeError('The narration expects planned, disabled assay tiles')
        labels = screen._intro.findChildren(QLabel)
        external = [label for label in labels if label.openExternalLinks()]
        if not external:
            raise RuntimeError('The organism reference links are absent')
        screen._scroll.ensureWidgetVisible(external[-1])
        settle(.5)
        capture('04_references')
        evidence['reference_links'] = [label.text() for label in external]
    write_json(captures / 'organism_navigation.json', evidence)
