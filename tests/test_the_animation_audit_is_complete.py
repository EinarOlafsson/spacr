"""Every animation must carry a recorded verdict.

Instruction 62 is 94 judgements and a machine cannot make them: it can check
that a pair differs only where it should, not that the illustration is a good
one. What it CAN do is refuse to let the record rot -- so the audit file has
to name every shipped animation, and only shipped animations, or the next
pass starts from zero the way this one nearly did.

The file is the deliverable, not a scratch note: it carries the verdict, the
setting keys it was judged against, and the pixel measurements taken at the
time, so a later reader can tell a judgement that was made from one that was
assumed.
"""

import json
from pathlib import Path

import pytest

from spacr.setting_animations import setting_animations

AUDIT = Path(__file__).resolve().parent.parent / "instructions" / "62_anim_audit.json"

VERDICTS = ("GOOD", "WEAK")


@pytest.fixture(scope="module")
def audit():
    if not AUDIT.exists():
        pytest.skip("the audit record ships with the instructions, not the package")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_every_shipped_animation_has_a_verdict(audit):
    recorded = {entry["slug"] for entry in audit["animations"]}
    shipped = {animation.slug for animation in setting_animations()}
    assert shipped - recorded == set(), "animations with no recorded verdict"


def test_no_verdict_names_an_animation_that_does_not_ship(audit):
    recorded = {entry["slug"] for entry in audit["animations"]}
    shipped = {animation.slug for animation in setting_animations()}
    assert recorded - shipped == set(), "verdicts for animations that were removed"


def test_each_verdict_is_one_of_the_documented_values(audit):
    assert set(audit["verdicts"]) == set(VERDICTS)
    for entry in audit["animations"]:
        assert entry["verdict"] in VERDICTS, entry["slug"]


def test_each_verdict_says_why(audit):
    """A bare GOOD is not a judgement anyone can check or overturn."""
    for entry in audit["animations"]:
        assert len(entry["note"].split()) >= 6, entry["slug"]


def test_the_verdict_was_judged_against_the_settings_that_ship(audit):
    keys = {a.slug: list(a.settings) for a in setting_animations()}
    for entry in audit["animations"]:
        assert entry["settings"] == keys[entry["slug"]], (
            f"{entry['slug']} was judged against setting keys it no longer has; "
            "re-watch it rather than editing the record"
        )


def test_the_counts_match_the_entries(audit):
    for verdict in VERDICTS:
        got = sum(1 for e in audit["animations"] if e["verdict"] == verdict)
        assert audit["counts"][verdict] == got
