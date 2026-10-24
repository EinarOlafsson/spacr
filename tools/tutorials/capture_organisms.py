"""Record organism guides using the visible Home, diagram and assay controls."""
from __future__ import annotations

import capture_ui
from capture_ui_drv_a import Rec, pick_list_row
from capture_ui_drv_e import guarded


@guarded
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

    initial_home_recorded = False

    def open_guide():
        nonlocal initial_home_recorded
        # Host: the 00_home clip starts on Home, before the pointer picks the tab,
        # and ends after the tile click; Xvfb: the still before the tile click.
        home_clip = None if initial_home_recorded else Rec(capture, '00_home', settle).start()
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
                tile = max(choices, key=lambda tile: tile.width() * tile.height())
                if home_clip is not None:
                    capture_ui.point_at(capture, tile, dwell=1.2)
                    if not capture_ui.live(capture):
                        home_clip.stop()
                initial_home_recorded = True
                click(tile)
                if home_clip is not None and capture_ui.live(capture):
                    home_clip.stop()
                break
        else:
            if home_clip is not None:
                home_clip.abort()
            raise RuntimeError('The organism Home tile is absent')
        screen = window._screens.get(key)
        if screen is None or not screen.isVisible():
            raise RuntimeError('The Home tile did not open the organism guide')
        return screen

    screen = open_guide()
    handle = screen._splitter.handle(1)

    def drag_divider():
        origin = handle.rect().center()
        QTest.mousePress(handle, Qt.LeftButton, pos=origin)
        QTest.mouseMove(handle, origin + QPoint(220, 0), delay=100)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=origin + QPoint(220, 0))
        settle(.8)

    with capture_ui.clip(capture, '01_organism'):
        # Host: the pointer reads the introduction, then drags the divider.
        if capture_ui.live(capture):
            capture_ui.point_at(capture, screen._intro, handle, dwell=1.5)
            drag_divider()
    if not capture_ui.live(capture):
        drag_divider()
    diagram = screen._diagram
    selector = diagram.selector
    selected = []
    with capture_ui.clip(capture, '02_diagram'):
        capture_ui.scroll_to(capture, screen._scroll, diagram)
        settle(.5)
        for index in (0, 2):
            item = selector.item(index)
            pick_list_row(capture, selector, index)
            settle(.3)
            if item.checkState() != Qt.Checked:
                raise RuntimeError('The visible compartment checkbox did not select its component')
            selected.append(item.text())
        if len(diagram.artwork.selected) != 2:
            raise RuntimeError('The diagram did not retain both selected components')
        capture_ui.point_at(capture, diagram.artwork, dwell=1.5)
    with capture_ui.clip(capture, '03_assays' if key == 'toxoplasma' else '03_modules'):
        click(diagram.clear_button)
        if diagram.artwork.selected:
            raise RuntimeError('Clear components did not reset the diagram')
        capture_ui.scroll_to_value(capture, screen._module_scroll, 0)
        capture_ui.point_at(capture, *[t for t in screen._tiles][:3], dwell=1.5)
    evidence = {'organism': key, 'home_tile_clicked': True,
                'selected_components': selected, 'clear_components_verified': True,
                'analysis_started': False, 'external_application_started': False,
                'tiles': [{'key': tile.property('organismModuleKey'),
                           'enabled': tile.isEnabled()} for tile in screen._tiles]}
    if key == 'toxoplasma':
        tile = next(tile for tile in screen._tiles
                    if tile.property('organismModuleKey') == 'analyze_plaques')
        with capture_ui.clip(capture, '04_plaque_assay'):
            capture_ui.scroll_to(capture, screen._module_scroll, tile)
            settle(.4)
            click(tile)
            target = window._screens.get('analyze_plaques')
            if target is None or not target.isVisible():
                raise RuntimeError('The Plaque Assay tile did not open its module')
            capture_ui.point_at(capture, getattr(target, '_btn_run', None), dwell=1.5)
        evidence['plaque_assay_tile_clicked'] = True
        with capture_ui.clip(capture, '05_references'):
            screen = open_guide()
            references = [label for label in screen._intro.findChildren(QLabel)
                          if label.openExternalLinks() and 'toxodb.org' in label.text()]
            if len(references) != 1:
                raise RuntimeError('The ToxoDB organism reference link is absent')
            external = [label for label in screen._intro.findChildren(QLabel)
                        if label.openExternalLinks()]
            capture_ui.scroll_to(capture, screen._scroll, external[-1])
            settle(.4)
            capture_ui.point_at(capture, external[-1], dwell=1.5)
        evidence['reference_links_shown'] = True
        with capture_ui.clip(capture, '06_home_return'):
            home()
            capture_ui.wait(capture, 1.5, settle)
    else:
        if any(tile.isEnabled() for tile in screen._tiles):
            raise RuntimeError('The narration expects planned, disabled assay tiles')
        labels = screen._intro.findChildren(QLabel)
        external = [label for label in labels if label.openExternalLinks()]
        if not external:
            raise RuntimeError('The organism reference links are absent')
        with capture_ui.clip(capture, '04_references'):
            capture_ui.scroll_to(capture, screen._scroll, external[-1])
            settle(.5)
            capture_ui.point_at(capture, external[-1], dwell=1.5)
        evidence['reference_links'] = [label.text() for label in external]
    write_json(captures / 'organism_navigation.json', evidence)
