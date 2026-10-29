"""Assistant instructions name actual native routes and published reference targets."""
from spacr.qt.ai import prompts


def test_navigation_catalog_includes_core_folded_and_page_tools(monkeypatch):
    from spacr.qt import app
    from spacr.qt.help_search import api_url
    from spacr.qt.tutorials import tutorial_url

    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    rows = {row[0]: row for row in prompts._navigation_modules()}
    assert {'mask', 'measure', 'make_masks', 'annotate', 'classify_merged',
            'graph_builder', 'gate_editor', 'regression', 'overview', '3d',
            'ops', 'spatial_masks', 'spatial_map'} <= rows.keys()
    assert rows['measure'][3] == api_url('spacr.measure', language='en')
    assert rows['mask'][4] == tutorial_url('mask')
    assert 'cellpose_masks' not in rows
    assert 'simulation' not in rows
    assert len(rows) == len({row[0] for row in rows.values()})


def test_navigation_catalog_obeys_tool_and_host_visibility(monkeypatch):
    from spacr.qt import app

    monkeypatch.setattr(app, 'app_is_visible', lambda key: key not in ('measure', 'mask'))
    keys = {row[0] for row in prompts._navigation_modules()}
    assert not {'mask', 'measure', '3d', 'ops', 'spatial_masks', 'illumination'} & keys
    assert 'annotate' in keys


def test_default_and_error_instructions_share_actionable_gui_context(monkeypatch):
    monkeypatch.setattr(prompts, '_navigation_modules', lambda: (
        ('mask', 'Mask', '', 'https://einarolafsson.github.io/spacr/api/spacr/core/index.html',
         'https://einarolafsson.github.io/spacr/tutorials/#lesson=07_mask'),
        ('measure', 'Measure', '', 'https://einarolafsson.github.io/spacr/api/spacr/measure/index.html', ''),
    ))
    context = prompts._navigation_context()
    for answer_prompt in (prompts.default_system_prompt(), prompts.error_explainer_prompt()):
        assert context in answer_prompt
        assert '[Mask](spacr://module/mask)' in answer_prompt
        assert 'Keep all links out of the main answer text' in answer_prompt
        assert 'once in first-mention order' in answer_prompt
        assert 'footer is required' in answer_prompt
        assert '[API](exact API URL) | ' in answer_prompt
        assert '[Tutorial](exact Tutorial URL)' in answer_prompt
        assert 'Navigator' in answer_prompt
        assert 'Actions > Run' in answer_prompt
        assert 'Propagate settings' in answer_prompt
        assert 'Preview and propagation do not process the full dataset' in answer_prompt
        assert 'never run analysis or change settings' in answer_prompt
        assert 'API=spacr/core/index.html' in answer_prompt
        assert 'Tutorial=#lesson=07_mask' in answer_prompt
    assert 'Train CV' not in prompts.default_system_prompt()
    assert 'Assign training labels to crops in Annotate' in prompts.default_system_prompt()
    assert 'at most 6 steps' in prompts.error_explainer_prompt()
    assert '; Tutorial=' not in context.split('measure: Measure')[1]


def test_catalog_context_stays_bounded_and_has_no_synthetic_command_routes():
    rows = prompts._navigation_modules()
    context = prompts._navigation_context()
    assert len(context) < 12000
    assert len(context.splitlines()) >= len(rows)
    assert all(not key.startswith('__') for key, *_rest in rows)
    assert all(not api or api.startswith('https://einarolafsson.github.io/spacr/api/')
               for _key, _name, _description, api, _tutorial in rows)
    assert all(not tutorial or tutorial.startswith('https://einarolafsson.github.io/spacr/tutorials/')
               for _key, _name, _description, _api, tutorial in rows)
