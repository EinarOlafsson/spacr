"""Navigator retains original inputs while scientific network choices change."""
import pytest
from PySide6.QtCore import Qt

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.mark.parametrize('source_kind', ['file', 'folder'])
def test_pdf_network_keeps_input_across_descendants_and_back(
        qtbot, private_ai_profile, tmp_path, source_kind):
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences.set_reduce_motion(True)
    pdf = tmp_path / 'plaques.pdf'
    pdf.write_bytes(b'%PDF-1.4\n%%EOF\n')
    original = pdf.read_bytes()
    source = pdf if source_kind == 'file' else tmp_path
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    opened = []
    page.module_requested.connect(lambda key, path: opened.append((key, path)))
    try:
        page.accept_paths([str(source)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        card = next(card for card in page.field.cards if card._key == 'analyze_plaques')
        assert card._path == str(pdf)
        card.click()
        graph = page._workflow
        assert graph is not None and graph.central is card and opened == []
        card.click()
        assert opened == [('analyze_plaques', str(pdf))]
        graph.select('regression')
        graph.select('regression')
        assert opened[-1] == ('regression', str(pdf))
        graph.select('measure')
        graph.select('measure')
        assert opened[-1] == ('measure', str(pdf))
        graph.select('run_history')
        graph.select('run_history')
        assert opened[-1] == ('run_history', '')
        assert graph.central is card and card._path == str(pdf)
        qtbot.mouseClick(graph.back, Qt.LeftButton)
        assert page._workflow is None and page.path_edit.text() == str(source)
        card.click()
        page._workflow.select('regression')
        page._workflow.select('regression')
        assert opened[-1] == ('regression', str(pdf))
        assert pdf.read_bytes() == original
    finally:
        page.close()


def test_network_prefers_existing_destination_source_over_original_input(
        qtbot, private_ai_profile, tmp_path):
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences.set_reduce_motion(True)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    image = tmp_path / 'image.tif'
    database = tmp_path / 'measurements.db'
    actions = [('mask', 'Mask Generation', '', str(image)),
               ('regression', 'Regression', '', str(database))]
    opened = []
    page.module_requested.connect(lambda key, path: opened.append((key, path)))
    try:
        page._last_result = {'actions': actions}
        page.field.set_cards(actions)
        for card in page.field.cards:
            card.activated.connect(page._select_suggestion)
        page.field.cards[0].click()
        graph = page._workflow
        graph.select('regression')
        graph.select('regression')
        assert opened == [('regression', str(database))]
        graph.select('measure')
        graph.select('measure')
        assert opened[-1] == ('measure', str(image))
    finally:
        page.close()
