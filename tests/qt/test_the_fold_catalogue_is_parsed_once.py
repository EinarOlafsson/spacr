"""Asking what is folded must not re-parse every host's source each time.

`fold_strip._host_declarations` reads a host module's SOURCE and runs
`ast.parse` on it, deliberately: importing all fifteen hosts to learn which
folds they declare would put pandas and scipy in the process before Home had
painted, and two other tests say so in as many words.

Reading is cheap; reading FIFTEEN FILES ON EVERY QUESTION is not.
`folded_fallback` asks `folded_modules` once per key and `folded_modules`
walks every host, so one Mask screen open ran `ast.parse` 73 times and spent
0.71 s of its 2.05 s compiling Python that was never going to be executed --
a third of the open, and item 284 is open on exactly that kind of stall.

COUNTED, NOT TIMED. The defect is "how many times", which is an integer the
machine cannot influence, so there is nothing here for a loaded runner to
make flaky.
"""

from __future__ import annotations

import ast

import pytest

from spacr.qt.widgets import fold_strip


@pytest.fixture
def parse_counter(monkeypatch):
    """Count `ast.parse` calls, starting from a cold cache.

    The cache is cleared FIRST so the count is this test's own work, and
    cleared again afterwards so a test that edits a host's source is not
    handed this test's answers.
    """
    fold_strip._forget_host_declarations()
    calls = []
    real = ast.parse

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ast, "parse", counted)
    yield calls
    fold_strip._forget_host_declarations()


def test_asking_twice_parses_nothing_the_second_time(parse_counter):
    """The second question is answered from what the first one read."""
    first = fold_strip.folded_modules()
    after_first = len(parse_counter)
    assert after_first > 0, (
        "no source was parsed at all, so this test is not reaching the "
        "reader it exists to measure")

    second = fold_strip.folded_modules()

    assert len(parse_counter) == after_first, (
        f"the second call parsed {len(parse_counter) - after_first} more "
        f"files; the host sources cannot have changed between them")
    assert second == first


def test_one_host_is_read_once_however_many_keys_are_asked_about(
        parse_counter):
    """`folded_fallback` per key must not mean every host per key.

    This is the shape the screen actually calls: a fold strip asks about each
    of its keys in turn. Before the cache that was one full walk of fifteen
    host files per key.
    """
    catalogue = fold_strip.folded_modules()
    baseline = len(parse_counter)
    keys = list(catalogue)[:8]
    assert len(keys) >= 4, "too few folded keys to make this meaningful"

    for key in keys:
        fold_strip.folded_fallback(key)

    assert len(parse_counter) == baseline, (
        f"asking about {len(keys)} keys parsed "
        f"{len(parse_counter) - baseline} files")


def test_the_answer_is_the_same_one_the_uncached_reader_gave(parse_counter):
    """A cache that changes the answer is not a cache.

    Compared against a genuinely cold read rather than against a second
    cached call, which would agree with itself whatever it held.
    """
    warm = fold_strip.folded_modules()
    fold_strip._forget_host_declarations()
    cold = fold_strip.folded_modules()

    assert warm == cold
    assert warm, "the catalogue is empty, so agreement proves nothing"
