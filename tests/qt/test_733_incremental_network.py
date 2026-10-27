"""Navigator expands real child paths without replacing the original root."""
import pytest
from PySide6.QtCore import Qt

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _network(qtbot, private_ai_profile, tmp_path, motion=False):
    from spacr.qt.app import make_home_page
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences.set_reduce_motion(not motion)
    preferences.set_ambient_enabled(motion)
    page = make_home_page()
    qtbot.addWidget(page)
    page.resize(1440, 1000)
    page.show()
    page.activateWindow()
    qtbot.waitUntil(page.isActiveWindow)
    page.field.set_cards([('mask', 'Mask Generation', '', str(tmp_path)),
                          ('make_masks', 'Make Masks', '', str(tmp_path))])
    for card in page.field.cards:
        card.activated.connect(page._select_suggestion)
    page._last_result = {'actions': [('mask', 'Mask Generation', '', str(tmp_path))]}
    page._select_suggestion('mask', str(tmp_path))
    return page, page._workflow


def test_child_exploration_keeps_root_and_old_edges_and_second_click_opens(
        qtbot, private_ai_profile, tmp_path):
    page, graph = _network(qtbot, private_ai_profile, tmp_path)
    opened = []
    page.module_requested.connect(lambda *args: opened.append(args))
    root = graph.central
    initial = {(edge.edge['from'], edge.edge['to']) for edge in graph.view.edges}
    positions = {key: graph.view.nodes[key].pos() for key in graph.view._node_order}
    try:
        node = graph.view.nodes['measure']
        graph.view.ensureVisible(node)
        point = graph.view.mapFromScene(node.sceneBoundingRect().center())
        qtbot.mouseClick(graph.view.viewport(), Qt.LeftButton, pos=point)
        assert opened == []
        assert graph.central is root and root._key == 'mask' and root.isVisible()
        assert graph._chosen_key == 'measure' and node.isVisible() and node.highlighted
        assert initial <= {(edge.edge['from'], edge.edge['to']) for edge in graph.view.edges}
        assert ('measure', 'annotate') in {(edge.edge['from'], edge.edge['to']) for edge in graph.view.edges}
        assert all(graph.view.nodes[key].pos() == position for key, position in positions.items())
        child_edge = next(edge for edge in graph.view.edges
                          if edge.edge['from'] == 'measure' and edge.edge['to'] == 'annotate')
        assert child_edge.start.x() == pytest.approx(node.pos().x() + graph.view._node_side - 5)
        graph.view.ensureVisible(node)
        qtbot.mouseClick(graph.view.viewport(), Qt.LeftButton,
                         pos=graph.view.mapFromScene(node.sceneBoundingRect().center()))
        assert opened == [('measure', '')]
    finally:
        page.close()


@pytest.mark.parametrize('exit_kind', ['back', 'escape'])
def test_exit_clears_entire_graph_and_restores_search_and_cards(
        qtbot, private_ai_profile, tmp_path, exit_kind):
    page, graph = _network(qtbot, private_ai_profile, tmp_path)
    try:
        graph.select('measure')
        assert not page.path_edit.isVisible() and not page.field.rim.isVisible()
        assert graph.central._key == 'mask'
        if exit_kind == 'back':
            qtbot.mouseClick(graph.back, Qt.LeftButton)
        else:
            page.setFocus()
            qtbot.keyClick(page, Qt.Key_Escape)
        assert page._workflow is None
        assert page.path_edit.parentWidget() is page.field and page.path_edit.isVisible()
        assert page.field.rim.isVisible()
        assert all(card.isEnabled() and card._content_opacity.opacity() == 1
                   for card in page.field.cards)
        page._select_suggestion('mask', str(tmp_path))
        assert page._workflow.view._expanded_keys == ['mask']
        assert not any(edge.edge['from'] == 'measure' for edge in page._workflow.view.edges)
    finally:
        page.close()


def test_network_uses_short_transition_without_selected_field_compression(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    from spacr.qt.widgets import ambient
    calls = []
    monkeypatch.setattr(ambient, '_field_visit_node', lambda *args: calls.append(args))
    page, graph = _network(qtbot, private_ai_profile, tmp_path, motion=True)
    try:
        assert graph._transition.duration() == 500
        graph.select('measure')
        assert graph.view._growth.duration() == 350
        qtbot.waitUntil(lambda: graph._progress == 1.0, timeout=1500)
        assert not page.path_edit.isVisible()
        assert calls == []
        page._show_suggestions()
        assert calls == [] and page.path_edit.isVisible()
    finally:
        page.close()


def test_repeated_child_choice_uses_real_native_handoff(qtbot, private_ai_profile, tmp_path):
    from spacr.qt.app import MainWindow
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1440, 1000)
    window.show()
    page = window._startup
    try:
        page.field.set_cards([('mask', 'Mask Generation', '', str(tmp_path))])
        page.field.cards[0].activated.connect(page._select_suggestion)
        page._last_result = {'actions': [('mask', 'Mask Generation', '', str(tmp_path))]}
        page._select_suggestion('mask', str(tmp_path))
        graph = page._workflow
        graph.select('measure')
        assert window._stack.currentWidget() is page and 'measure' not in window._screens
        graph.select('measure')
        assert window._stack.currentWidget() is window._screens['measure']
        assert window._screens['measure'].app_key == 'measure'
    finally:
        window.close()


def test_immediate_back_before_first_animation_tick_restores_search(
        qtbot, private_ai_profile, tmp_path):
    page, graph = _network(qtbot, private_ai_profile, tmp_path, motion=True)
    try:
        assert graph._transition.currentTime() == 0 and graph._progress == 0.0
        assert not hasattr(page.path_edit, '_content_opacity')
        page._show_suggestions()
        assert page._workflow is None
        assert page.path_edit.parentWidget() is page.field and page.path_edit.isVisible()
        assert page.field.rim.isVisible()
        assert all(card.isEnabled() and card._content_opacity.opacity() == 1.0
                   for card in page.field.cards)
    finally:
        page.close()
