"""``plan_import`` proposes; nothing is written until someone agrees.

Instruction 363 calls the reviewable table "the single most important
requirement in this file", and the reason is that ``metadata_type`` is
unusable today not because its regular expressions are bad but because the
user cannot see what they did until masks come out wrong.

So these tests are about the PROPOSAL: that it says what it found, says what
it could not work out, says both in sentences rather than tracebacks, accepts
a correction without touching the disk, and never writes anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_corpus import BUILDERS, build_all  # noqa: E402

from spacr.image_import import plan_import  # noqa: E402

#: The one tree whose channel axis nothing in the files can name: folders are
#: dye names, and only a person knows DAPI is meant to be channel 1.
ANSWERS = {"per_channel_folder": {0: {"DAPI": 1, "GFP": 2}}}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return {t.name: t for t in build_all(tmp_path_factory.mktemp("plan"))}


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS])
def test_a_plan_writes_nothing(name, corpus):
    """Planning must not change the folder it is planning for.

    The whole design rests on a user being able to look before anything
    happens, so a plan that wrote even a cache file would break the promise.
    """
    tree = corpus[name]
    before = {p: p.stat().st_mtime_ns for p in tree.root.rglob("*") if p.is_file()}
    plan_import(tree.root, mapping=ANSWERS.get(name))
    after = {p: p.stat().st_mtime_ns for p in tree.root.rglob("*") if p.is_file()}
    assert before == after, f"{name}: planning modified the folder"


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS])
def test_every_layout_resolves_with_at_most_one_answer(name, corpus):
    """Ten real layouts, two wells, two fields and two channels each.

    The fixed convention list resolves two of these. This is the number that
    says whether instruction 363 is being delivered.
    """
    plan = plan_import(corpus[name].root, mapping=ANSWERS.get(name))
    counts = plan.counts()
    assert counts.get("well") == 2, f"{name}: wells {counts.get('well')}"
    assert counts.get("field") == 2, f"{name}: fields {counts.get('field')}"
    if "channel" in counts:
        assert counts["channel"] == 2, f"{name}: channels {counts['channel']}"
    else:
        # The channel is pages inside the file: a COUNT per file, not a
        # position, and the plan must keep those distinct.
        sizes = {e["c_count"] for e in plan.files.values() if "c_count" in e}
        assert sizes == {2}, f"{name}: channel pages {sizes}"
    assert not plan.problems(), f"{name}: {plan.problems()}"


def test_an_axis_nobody_can_name_is_a_stated_problem(corpus):
    """Before the answer: a sentence, not a silence and not a traceback."""
    plan = plan_import(corpus["per_channel_folder"].root)
    problems = plan.problems()
    assert problems, "the dye folders raised no complaint at all"
    assert any("DAPI" in p and "GFP" in p for p in problems), problems
    assert "channel" not in plan.counts(), "a channel was invented"


def test_an_answer_resolves_it_without_touching_the_disk(corpus):
    """`with_mapping` returns a NEW plan, resolved in memory.

    Editing has to be instant and free, or a user will not explore a wrong
    guess to find out it was wrong.
    """
    tree = corpus["per_channel_folder"]
    plan = plan_import(tree.root)
    before = {p: p.stat().st_mtime_ns for p in tree.root.rglob("*") if p.is_file()}

    mapped = plan.with_mapping({0: {"DAPI": 1, "GFP": 2}})

    assert mapped is not plan, "with_mapping mutated the plan in place"
    assert not plan.counts().get("channel"), "the original plan was changed"
    assert mapped.counts()["channel"] == 2
    assert not mapped.problems()
    after = {p: p.stat().st_mtime_ns for p in tree.root.rglob("*") if p.is_file()}
    assert before == after


def test_every_problem_is_reported_not_just_the_first(tmp_path):
    """Several faults produce several sentences.

    A plan that stops at the first complaint makes the user fix things one
    round-trip at a time -- the same fault an earlier translation audit had,
    where it raised on its first finding and reported one problem where there
    were three.
    """
    import numpy as np
    import tifffile

    root = tmp_path / "messy"
    root.mkdir()
    # An unlabelled multi-page file AND a truncated one AND a dye folder.
    for well in ("A01", "B02"):
        for dye in ("DAPI", "GFP"):
            p = root / dye / f"{well}_f1.tif"
            p.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(p), np.zeros((3, 4, 4), np.uint16),
                             photometric="minisblack")
    (root / "DAPI" / "B02_f2.tif").write_bytes(b"II*\x00 broken")

    problems = plan_import(root).problems()
    assert len(problems) >= 2, problems
    assert any("pages" in p for p in problems), problems
    assert all(p.endswith(".") for p in problems), (
        "problems should read as sentences", problems)


def test_the_table_shows_the_filename_beside_what_was_parsed(corpus):
    """A wrong guess is only visible next to the name it came from.

    Reading a column of numbers cannot tell you the field and the channel
    were swapped; reading them beside the filename can.
    """
    plan = plan_import(corpus["cellvoyager"].root)
    table = plan.table(limit=3)
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in table
    for axis in ("well", "field", "channel"):
        assert axis in table, f"{axis} missing from the table"
    assert "and 5 more" in table, "the table did not say it was truncated"
