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


# ---------------------------------------------------------------------------
# What the names cannot carry: axes inside the file.
# ---------------------------------------------------------------------------

def test_the_channel_inside_an_ome_file_is_read(corpus):
    """``flat_ome``'s channels are pages, and the file says so."""
    from spacr.image_import import read_axes_inside

    tree = corpus["flat_ome"]
    for path in sorted(tree.root.rglob("*.tif")):
        inside = read_axes_inside(path)
        assert inside.declared, f"{path.name}: the OME axes were not read"
        assert inside.sizes.get("c") == 2, inside.sizes
        assert not inside.is_ambiguous


def test_a_z_stack_and_a_timelapse_are_told_apart(corpus):
    """The pair the corpus exists to separate.

    ``z_stack_in_file`` and ``time_in_file`` have IDENTICAL filenames and the
    same page count. Nothing about the name distinguishes them, and a page
    index alone means nothing -- so if this passes, it is the file's own
    metadata doing the work and nothing else.
    """
    from spacr.image_import import read_axes_inside

    z = read_axes_inside(sorted(corpus["z_stack_in_file"].root.rglob("*.tif"))[0])
    t = read_axes_inside(sorted(corpus["time_in_file"].root.rglob("*.tif"))[0])
    assert z.pages == t.pages == 5, (z.pages, t.pages)
    assert z.sizes == {"z": 5}, z.sizes
    assert t.sizes == {"t": 5}, t.sizes


def test_pages_with_no_metadata_are_reported_as_unknown(tmp_path):
    """The honest unknown, and the one that must never be guessed.

    A multi-page TIFF carrying no axis metadata could be Z, T or C. Picking
    one would produce a plate with a confidently wrong shape -- so the page
    count is reported and ``declared`` stays False.
    """
    import numpy as np
    import tifffile

    from spacr.image_import import read_axes_inside

    path = tmp_path / "mystery.tif"
    tifffile.imwrite(str(path), np.zeros((3, 4, 4), dtype=np.uint16),
                     photometric="minisblack")
    inside = read_axes_inside(path)
    assert inside.pages == 3
    assert inside.is_ambiguous, "three unexplained pages were treated as known"
    assert not inside.sizes, f"an axis was invented: {inside.sizes}"


def test_an_unreadable_file_does_not_stop_the_scan(tmp_path):
    """One truncated file must not fail a folder of thousands."""
    from spacr.image_import import read_axes_inside

    broken = tmp_path / "truncated.tif"
    broken.write_bytes(b"II*\x00 not really a tiff")
    inside = read_axes_inside(broken)
    assert inside.pages == 0
    assert not inside.declared
