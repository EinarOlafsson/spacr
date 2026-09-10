"""Linkless retained lessons pass, but missing/extra required links never do."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_staged_lesson import check_related_links


@pytest.mark.parametrize('actual,expected', [([], []), (['a'], ['a']),
    (['b', 'a', 'a'], ['a', 'b'])])
def test_exact_authored_link_set_passes(actual, expected):
    check_related_links(actual, expected)


@pytest.mark.parametrize('actual,expected', [([], ['a']), (['a'], []),
    (['a'], ['b']), (['a'], ['a', 'b']), (['a', 'b'], ['a'])])
def test_missing_wrong_and_extra_links_fail(actual, expected):
    with pytest.raises(ValueError, match='Related lesson links differ'):
        check_related_links(actual, expected)
