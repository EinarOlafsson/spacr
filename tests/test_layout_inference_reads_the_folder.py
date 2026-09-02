"""``spacr.image_import.infer_layout`` against the ten-layout corpus.

The fixed convention list parses 2 of 10 (see
``test_the_import_corpus_is_the_specification.py``). Inference reads what
VARIES across the folder instead, and these tests pin what that buys and,
more importantly, what it refuses to guess.

THE REFUSALS ARE THE POINT. Three trees are not fully resolved and all three
are honest: a channel that lives inside the file cannot be read from a name,
and a folder called ``DAPI`` cannot be turned into channel 1 without someone
saying so. Each is asserted as a REFUSAL rather than left as a gap, because
the failure this module exists to prevent is a plausible wrong answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_corpus import BUILDERS, build_all  # noqa: E402

from spacr.image_import import infer_layout  # noqa: E402


#: Trees whose NAMES carry well, field and channel, and from which inference
#: recovers all three. A ratchet: this may grow, and a shrink is a regression.
FULLY_RESOLVED_FROM_NAMES = (
    "cellvoyager", "cq1", "harmony", "imagexpress",
    "per_well_folder", "z_stack_in_file", "time_in_file", "tiled",
)

#: Trees where something real is genuinely not in the names.
NEEDS_MORE_THAN_NAMES = {
    "flat_ome": "the channel is a page inside the file",
    "per_channel_folder": "the channel is a dye name, not an index",
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return {t.name: t for t in build_all(tmp_path_factory.mktemp("infer"))}


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS])
def test_no_file_is_silently_dropped(name, corpus):
    """Every image is accounted for, in every layout.

    THE FIRST VERSION FAILED THIS and it is why the test exists. The shape
    that groups files of one convention was computed from the alphabetic runs
    as well as the digits -- but the WELL LETTER is alphabetic and varies, so
    ``plate1_A01_...`` and ``plate1_B02_...`` were two different conventions,
    the majority won, and half of every plate vanished before anything was
    inferred. Placed-plus-skipped is the only total that can catch that.
    """
    tree = corpus[name]
    layout = infer_layout(tree.root)
    assert len(layout.per_file) + len(layout.skipped) == len(tree.files), (
        f"{name}: {len(tree.files)} files in, "
        f"{len(layout.per_file)} placed + {len(layout.skipped)} skipped")
    assert not layout.skipped, (
        f"{name}: {layout.skipped} were treated as a different convention")


@pytest.mark.parametrize("name", FULLY_RESOLVED_FROM_NAMES)
def test_a_layout_whose_names_carry_everything_is_fully_resolved(name, corpus):
    """Well, field and channel all recovered, with the right cardinality."""
    tree = corpus[name]
    layout = infer_layout(tree.root)
    wells = {m["well"] for m in layout.per_file.values() if "well" in m}
    fields = {m["field"] for m in layout.per_file.values() if "field" in m}
    channels = {m["channel"] for m in layout.per_file.values() if "channel" in m}
    assert len(wells) == 2, f"{name}: wells {sorted(wells)}"
    assert len(fields) == 2, f"{name}: fields {sorted(fields)}"
    assert len(channels) == 2, f"{name}: channels {sorted(channels)}"


@pytest.mark.parametrize("name", sorted(NEEDS_MORE_THAN_NAMES))
def test_what_the_names_do_not_say_is_not_invented(name, corpus):
    """A tree missing an axis from its names must not acquire one.

    Resolving well and field and stopping is the correct outcome. Inventing a
    channel would be the dangerous one: it would look like a complete import
    and produce a plate with the wrong number of channels.
    """
    tree = corpus[name]
    layout = infer_layout(tree.root)
    channels = {m["channel"] for m in layout.per_file.values() if "channel" in m}
    assert not channels, (
        f"{name}: invented channels {sorted(channels)}; "
        f"{NEEDS_MORE_THAN_NAMES[name]}")
    # It must still have got what the names DO say.
    wells = {m["well"] for m in layout.per_file.values() if "well" in m}
    assert len(wells) == 2, f"{name}: lost the wells too: {sorted(wells)}"


def test_an_axis_that_cannot_be_named_is_reported_not_dropped(corpus):
    """``DAPI``/``GFP`` folders are an axis. Say so.

    Instruction 363: a partly-unparseable tree must import what it can and
    REPORT the rest. A varying folder name is a real axis that nothing in the
    names maps to a channel index, so it belongs in ``unplaced`` with its
    values -- which is what lets a caller ask "is DAPI channel 1?" instead of
    deciding for the user.
    """
    layout = infer_layout(corpus["per_channel_folder"].root)
    assert layout.unplaced, "the dye folders were dropped silently"
    values = {v for vals in layout.unplaced.values() for v in vals}
    assert values == {"DAPI", "GFP"}, values


def test_wells_are_not_manufactured_from_punctuation(corpus):
    """The greedy-well bug, pinned.

    ``plate1_A01_T0001F001L01A01Z01C01`` contains four single-letter-plus-two-
    digit tokens: ``A01``, ``L01``, ``A01`` again and ``C01``. Matching wells
    on that shape made all four wells, the last overwriting the channel, and
    every plate came out with four wells and no channels.

    Only ``A`` VARIES across the folder. A constant letter is the convention's
    punctuation; a varying one is an axis.
    """
    layout = infer_layout(corpus["cellvoyager"].root)
    wells = {m["well"] for m in layout.per_file.values() if "well" in m}
    assert wells == {"A01", "B02"}, sorted(wells)
    channels = {m["channel"] for m in layout.per_file.values() if "channel" in m}
    assert channels == {1, 2}, sorted(channels)


def test_row_and_column_become_a_well(corpus):
    """Opera Phenix keeps them apart; spaCR's vocabulary is a well name."""
    layout = infer_layout(corpus["harmony"].root)
    wells = {m["well"] for m in layout.per_file.values() if "well" in m}
    assert wells == {"A01", "B02"}, sorted(wells)


def test_inference_reads_a_sample_not_the_whole_tree(corpus, tmp_path):
    """Inspecting a large archive must not cost the archive.

    363 requires inspecting a 400-plate archive to be as fast as inspecting
    one plate, so the sample bound has to actually bound the reading.
    """
    root = tmp_path / "big"
    for i in range(50):
        p = root / f"plate{i}" / f"plate{i}_A01_T0001F001L01A01Z01C01.tif"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    layout = infer_layout(root, sample=10)
    assert layout.sampled == 10, layout.sampled
