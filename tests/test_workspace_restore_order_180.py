"""Settings restore before dependent sections within their saved app namespace."""
import copy

from spacr import workspace


def test_settings_reset_precedes_same_app_panel_restore():
    selection = []
    calls = []

    class Settings:
        def apply_workspace_state(self, state):
            calls.append("settings")
            selection.clear()
            return True

    class Panel:
        def apply_workspace_state(self, state):
            calls.append("databases")
            selection.extend(state["paths"])
            return True

    document = {"sections": {"umap:databases": {"paths": ["carried.db"]},
                             "umap:settings": {"src": ["original"]}}}
    before = copy.deepcopy(document)
    report = workspace.restore({"umap:databases": Panel(), "umap:settings": Settings()}, document)
    assert selection == ["carried.db"]
    assert calls == ["settings", "databases"]
    assert report["restored"] == ["umap:settings", "umap:databases"]
    assert report["skipped"] == []
    assert document == before


def test_only_matching_namespaced_settings_move_and_skip_reports_survive():
    calls = []

    class Provider:
        def __init__(self, name):
            self.name = name

        def apply_workspace_state(self, state):
            calls.append(self.name)
            return state.get("accept", True)

    names = ["a:databases", "a:extra:panel", "a:extra:settings", "a:settings",
             "a0", "b:databases", "b:settings", "settings", "z"]
    states = {name: {} for name in reversed(names)}
    states["b:databases"] = {"accept": False}
    states["a:missing"] = {}
    states["invalid"] = None
    before = copy.deepcopy(states)
    report = workspace.restore({name: Provider(name) for name in names}, {"sections": states})
    assert calls == ["a0", "a:settings", "a:databases", "a:extra:settings", "a:extra:panel",
                     "b:settings", "b:databases", "settings", "z"]
    assert report["restored"] == [name for name in calls if name != "b:databases"]
    assert report["skipped"] == [
        {"section": "a:missing", "why": "nothing on screen owns it"},
        {"section": "b:databases", "why": "the panel declined it"},
        {"section": "invalid", "why": "not a state document"},
    ]
    assert states == before


def test_unknown_numeric_section_cannot_replace_live_named_state():
    live = {'src': ['existing.csv']}
    before = copy.deepcopy(live)

    class Settings:
        def apply_workspace_state(self, state):
            live.clear()
            live.update(state)

    document = {'sections': {7: {'src': ['unrecognized.csv']}}}
    saved = copy.deepcopy(document)
    report = workspace.restore({'settings': Settings()}, document)
    assert report['restored'] == []
    assert report['skipped'] == [{'section': 7, 'why': 'nothing on screen owns it'}]
    assert live == before
    assert document == saved


def test_settings_without_a_saved_dependent_panel_restore_normally():
    live = {'src': ['existing.csv']}

    class Settings:
        def apply_workspace_state(self, state):
            live.clear()
            live.update(state)
            return True

    document = {'sections': {'mask:settings': {'src': ['restored.csv']}}}
    before = copy.deepcopy(document)
    report = workspace.restore({'mask:settings': Settings()}, document)
    assert report == {'restored': ['mask:settings'], 'skipped': [], 'files': []}
    assert live == {'src': ['restored.csv']}
    assert document == before
