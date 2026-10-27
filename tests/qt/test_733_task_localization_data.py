"""Every approved English task keeps its intent in each Navigator input language."""
import re
from pathlib import Path

import pytest

from spacr.qt import app

_LANGUAGES = ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
_ITEM = Path(__file__).resolve().parents[2] / 'features/new/733_easy_mode_content_guided_home.txt'


def test_localized_data_covers_all_original_task_pairs_and_current_aliases():
    pairs = re.findall(r'^\d{3}\. (.+) \| (.+) -> (.+)$', _ITEM.read_text(), re.M)
    assert len(pairs) == 100
    english = {phrase: keys for keys, phrases in app._EASY_TASK_GROUPS for phrase in phrases}
    for original, alternate, raw_keys in pairs:
        expected = tuple(raw_keys.split(', '))
        assert english[original] == expected
        assert english[alternate] == expected
    assert len(app._EASY_TASK_GROUPS) == 21
    assert sum(len(phrases) for _keys, phrases in app._EASY_TASK_GROUPS) == 218
    assert set(app._EASY_TASK_LOCALIZED) == set(_LANGUAGES)
    for language in _LANGUAGES:
        localized = app._EASY_TASK_LOCALIZED[language]
        assert len(localized) == len(app._EASY_TASK_GROUPS)
        for (_keys, source), targets in zip(app._EASY_TASK_GROUPS, localized):
            assert len(targets) == len(source), language
            assert all(isinstance(target, str) and target.strip() for target in targets)


@pytest.mark.parametrize('language', _LANGUAGES)
def test_every_localized_noun_verb_and_short_task_keeps_its_canonical_modules(language):
    for (expected, sources), targets in zip(app._EASY_TASK_GROUPS, app._EASY_TASK_LOCALIZED[language]):
        assert len(sources) == len(targets)
        for source, target in zip(sources, targets):
            assert app._easy_task_keys(target) == expected, (language, source, target)
            assert app._easy_task_keys(target.upper() + '!') == expected, (language, source, target)


def test_localized_task_and_associated_aliases_have_no_cross_intent_collisions():
    seen = {}
    for keys, phrases in [*app._EASY_TASK_GROUPS, *app._easy_localized_alias_groups()]:
        for phrase in phrases:
            normalized = app._easy_command_text(phrase)
            assert normalized, phrase
            previous = seen.setdefault(normalized, tuple(keys))
            assert previous == tuple(keys), (phrase, previous, tuple(keys))


@pytest.mark.parametrize('language', _LANGUAGES)
def test_every_associated_short_alias_keeps_its_native_intent(language):
    assert len(app._EASY_ASSOCIATED_GROUPS) == 27
    translated = app._EASY_ASSOCIATED_LOCALIZED[language]
    assert len(translated) == len(app._EASY_ASSOCIATED_GROUPS)
    for (expected, source), target in zip(app._EASY_ASSOCIATED_GROUPS, translated):
        assert target.strip()
        assert app._easy_task_keys(target) == expected, (language, source, target)
