"""Shared style captions enter the normal runtime scan without importing plotting producers."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
from tools import build_i18n_catalogs as builder

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT/'spacr/figures/style.py'
HANDOFF = ROOT/'tr_graphstyle_g1_sources.json'


def _handoff(field):
    """The root English handoff for Codex translation, or None once a landing train archived it (the record then lives in the train's tr_record)."""
    if not HANDOFF.exists():
        return None
    return json.loads(HANDOFF.read_text())[field]


def test_every_literal_shared_style_caption_is_registered_and_has_an_exact_root_handoff():
    """Every real literal tr/replay-message template, including existing ones used by Home G2, must be source-bound in the scanner and root English handoff with exact locations and hashes."""
    source = STYLE.read_text()
    syntax = ast.parse(source)
    expected = {(node.lineno, node.args[0].value)
        for node in ast.walk(syntax) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id in ('tr','_replay_message')
        and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}
    assert expected
    registered = builder._static_ui_sources_of_file(STYLE, source, ROOT/'spacr', ROOT/'spacr/qt/preferences.py')
    assert {text.strip() for _line, text in expected if builder._looks_translatable(text)} <= registered
    handoff = _handoff('shared_style_runtime')
    if handoff is None:
        return
    assert {(record['line'], record['source']) for record in handoff} == expected
    for record in handoff:
        assert record['file'] == 'spacr/figures/style.py'
        assert record['source_sha256'] == hashlib.sha256(record['source'].encode()).hexdigest()


def test_shared_style_owner_is_in_the_normal_runtime_scan(monkeypatch):
    """Calling the shipped extractor must actually visit style.py rather than relying on a separate test-only or handoff-only list."""
    seen = []
    original = builder._static_ui_sources_of_file

    def observe(path, *args):
        """Retain the scanned owner while executing the original per-file extractor."""
        if path == STYLE:
            seen.append(path)
        return original(path, *args)

    builder._STATIC_UI_FILE_CACHE.clear()
    monkeypatch.setattr(builder, '_static_ui_sources_of_file', observe)
    found = set(builder.extract_static_ui_sources())
    assert seen == [STYLE]
    assert 'Validate the updated native plot before certifying its appearance' in found


def test_replay_caption_admission_is_limited_to_the_real_shared_owner():
    """An unrelated same-name helper must not expose arbitrary internal strings as figure captions; real style replay literals remain registered."""
    text = '_replay_message("Use the measured scalar range")'
    source_root = ROOT/'spacr'
    preferences = source_root/'qt/preferences.py'
    assert 'Use the measured scalar range' in builder._static_ui_sources_of_file(STYLE, text, source_root, preferences)
    assert 'Use the measured scalar range' not in builder._static_ui_sources_of_file(source_root/'unrelated.py', text, source_root, preferences)


@pytest.mark.parametrize('relative,field', [('spacr/figures/bundle.py','bundle_replay_runtime'),('spacr/qt/widgets/feature_explorer.py','feature_explorer_runtime')])
def test_replay_and_feature_captions_have_exact_normal_runtime_registration_and_handoff(relative, field):
    """The real bundle adapter and feature producer need source registration and exact English templates, with no generated catalogue edits."""
    path = ROOT/relative
    source = path.read_text()
    expected = {(node.lineno,node.args[0].value) for node in ast.walk(ast.parse(source))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('tr','_replay_message')
        and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)}
    assert expected
    registered = builder._static_ui_sources_of_file(path,source,ROOT/'spacr',ROOT/'spacr/qt/preferences.py')
    assert {text.strip() for _line,text in expected if builder._looks_translatable(text)} <= registered
    handoff = _handoff(field)
    if handoff is None:
        return
    assert {(record['line'],record['source']) for record in handoff} == expected
    assert all(record['file']==relative and record['source_sha256']==hashlib.sha256(record['source'].encode()).hexdigest() for record in handoff)


def test_bundle_replay_owner_is_visited_by_the_normal_runtime_extractor(monkeypatch):
    """The production extractor must visit bundle.py and find the actual class-binding refusal without a parallel test-only inventory."""
    seen = []
    original = builder._static_ui_sources_of_file
    path = ROOT/'spacr/figures/bundle.py'

    def observe(candidate, *args):
        """Record the real bundle visit and execute the shipped collector."""
        if candidate==path:
            seen.append(candidate)
        return original(candidate,*args)

    builder._STATIC_UI_FILE_CACHE.clear()
    monkeypatch.setattr(builder,'_static_ui_sources_of_file',observe)
    found = set(builder.extract_static_ui_sources())
    assert seen == [path]
    assert 'Captured feature class identities do not match the source table; retain the original class labels' in found
