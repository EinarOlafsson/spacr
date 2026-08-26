"""A settings file naming a withdrawn setting is told, not ignored in silence.

The typo check only speaks when a live setting is within a close match of
the name it was handed, which is deliberate -- newer pipelines carry keys it
has never heard of, and warning about all of them would be noise. But a
deleted setting usually loses its neighbours in the same breath, so nothing
is left within matching distance: the value is dropped, the default is used,
and the run differs from the file describing it without a word.

An old settings file is exactly where a retired name turns up, and its
author has every reason to think it still applies.
"""

from __future__ import annotations

import pytest

from spacr.validate import (                                     # noqa: E402
    RETIRED_SETTINGS, _check_retired_keys, _check_unknown_keys,
)

#: A retired name whose neighbours went with it, so fuzzy matching is blind
#: to it. This is the case the whole table exists for.
UNREACHABLE_BY_FUZZY = "upscale_factor"


def _messages(problems):
    return [getattr(p, "message", str(p)) for p in problems]


def test_the_typo_check_really_is_blind_to_it():
    """The premise. If this ever fails, the table has become unnecessary."""
    assert _check_unknown_keys({"src": "/tmp", UNREACHABLE_BY_FUZZY: 2},
                               "mask") == []


def test_a_withdrawn_setting_is_reported():
    problems = _check_retired_keys({"src": "/tmp", UNREACHABLE_BY_FUZZY: 2})
    assert len(problems) == 1
    assert UNREACHABLE_BY_FUZZY in _messages(problems)[0]


def test_the_report_says_the_value_does_nothing():
    """A warning that does not say the value is ignored has not helped."""
    problem = _check_retired_keys({UNREACHABLE_BY_FUZZY: 2})[0]
    advice = " ".join(str(getattr(problem, f, "")) for f in
                      ("fix", "advice", "hint", "remedy", "suggestion"))
    assert "no effect" in advice or "does not read" in advice, advice


def test_a_renamed_setting_names_its_replacement():
    problem = _check_retired_keys({"minimum_cell_count": 30})[0]
    text = " ".join(str(getattr(problem, f, "")) for f in
                    ("message", "fix", "advice", "hint", "remedy"))
    assert "min_cell_count" in text


def test_every_named_replacement_is_a_live_setting():
    """A rename pointing at another dead name sends the user nowhere."""
    from spacr.settings import expected_types

    live = set(expected_types)
    wrong = {old: new for old, new in RETIRED_SETTINGS.items()
             if new and new not in live}
    assert wrong == {}, f"these replacements are not live settings: {wrong}"


def test_no_retired_name_is_also_a_live_setting():
    """A key cannot be both withdrawn and offered."""
    from spacr.settings import expected_types

    both = sorted(set(RETIRED_SETTINGS) & set(expected_types))
    assert both == [], f"declared and retired at once: {both}"


def test_a_healthy_settings_file_is_quiet():
    assert _check_retired_keys({"src": "/tmp", "cell_channel": 0}) == []


def test_a_non_string_key_does_not_raise():
    assert _check_retired_keys({7: "x", None: "y"}) == []


@pytest.mark.parametrize("key", sorted(RETIRED_SETTINGS))
def test_each_retired_name_is_reported(key):
    assert len(_check_retired_keys({key: 1})) == 1
