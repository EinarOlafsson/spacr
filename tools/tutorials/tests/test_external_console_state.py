"""Native preference changes must never erase or replace captured results."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_external_masks import check_retained_console_state


def state():
    return {
        'settings': {'cytoplasm': True, 'channels': [0], 'dst': '/private/project'},
        'figure_count': 6,
        'figures': [bytes([index]) for index in range(6)],
        'console_blocks': [
            {'kind': 'stdout', 'text': 'Preview: 2 fields, nothing written'},
            {'kind': 'stdout', 'text': 'Prepared 2 fields. tables: cell, cytoplasm'},
            {'kind': 'stdout', 'text': 'Finished'},
        ],
    }


def test_exact_results_survive_a_native_refresh_without_mutation():
    before = state()
    after = deepcopy(before)
    snapshot = deepcopy((before, after))
    check_retained_console_state(before, after)
    assert (before, after) == snapshot


@pytest.mark.parametrize('which', ['before', 'after'])
@pytest.mark.parametrize('count', [0, 5, 7])
def test_both_states_require_exactly_six_figures(which, count):
    states = {'before': state(), 'after': state()}
    states[which]['figure_count'] = count
    with pytest.raises(RuntimeError, match='all six figures'):
        check_retained_console_state(**states)


@pytest.mark.parametrize('which', ['before', 'after'])
def test_reported_count_cannot_hide_a_missing_image(which):
    states = {'before': state(), 'after': state()}
    states[which]['figures'].pop()
    with pytest.raises(RuntimeError, match='all six figures'):
        check_retained_console_state(**states)


@pytest.mark.parametrize('change', ['content', 'order'])
def test_same_count_cannot_hide_changed_figure_content_or_order(change):
    before, after = state(), state()
    if change == 'content':
        after['figures'][3] = b'not the original pixels'
    else:
        after['figures'].reverse()
    with pytest.raises(RuntimeError, match='ordered figure images'):
        check_retained_console_state(before, after)


@pytest.mark.parametrize('change', ['content', 'order', 'kind', 'missing', 'added'])
def test_exact_ordered_console_history_is_required(change):
    before, after = state(), state()
    if change == 'content':
        after['console_blocks'][1]['text'] = 'Finished'
    elif change == 'order':
        after['console_blocks'].reverse()
    elif change == 'kind':
        after['console_blocks'][1]['kind'] = 'traceback'
    elif change == 'missing':
        after['console_blocks'].pop(0)
    else:
        after['console_blocks'].append({'kind': 'stdout', 'text': 'replacement'})
    with pytest.raises(RuntimeError, match='console history'):
        check_retained_console_state(before, after)


@pytest.mark.parametrize('blocks', [[], [{'kind': 'stdout', 'text': ''}]])
def test_empty_history_is_not_a_valid_baseline(blocks):
    before = state()
    before['console_blocks'] = blocks
    with pytest.raises(RuntimeError, match='actual console text'):
        check_retained_console_state(before, deepcopy(before))


def test_preferences_may_not_change_a_nested_measurement_setting():
    before, after = state(), state()
    after['settings']['channels'].append(1)
    with pytest.raises(RuntimeError, match='measurement settings'):
        check_retained_console_state(before, after)
