"""``create_consensus`` is safe only because every caller pads first.

This file pins an invariant rather than a branch, because the branch it was
written for turns out to be unreachable and the invariant is what actually
protects the code.

``create_consensus`` loops over ``len(seq1)`` and indexes ``seq2`` at the same
offset, so a shorter second sequence raises IndexError from inside a chunk
worker. Nothing in the function checks. It is safe today because all three
callers pad both reads to ``expected_end`` before calling it -- and if a later
change moves, shortens or removes that padding, the failure is an IndexError
in a subprocess halfway through a sequencing run.
"""
from __future__ import annotations

import random

import pytest


def test_the_consensus_is_exactly_as_long_as_the_first_sequence():
    """Fuzzed over three thousand equal-length pairs.

    This is the invariant the callers' ``len(consensus_seq) >= expected_end``
    guards depend on. Because the consensus is always exactly the padded
    length, those guards are unconditionally true -- which is why their false
    side cannot be covered and is recorded in instruction 310 rather than
    tested here.
    """
    from spacr.sequencing import create_consensus

    rng = random.Random(0)
    for _ in range(3000):
        n = rng.randint(0, 20)
        seq1 = "".join(rng.choice("ACGTN") for _ in range(n))
        seq2 = "".join(rng.choice("ACGTN") for _ in range(n))
        assert len(create_consensus(seq1, "I" * n, seq2, "I" * n)) == n


def test_a_shorter_second_read_raises_rather_than_truncating():
    """The unguarded indexing, pinned so a caller that stops padding is caught.

    Truncating would be worse than raising: a consensus built from the first
    two bases of a barcode still matches a permissive regex, and the read
    would be counted into the wrong well rather than discarded.
    """
    from spacr.sequencing import create_consensus

    with pytest.raises(IndexError):
        create_consensus("ACGT", "IIII", "AC", "II")


def test_a_longer_second_read_is_ignored_past_the_first_ones_length():
    """The other asymmetry, which does NOT raise -- and that is the danger.

    A longer seq2 silently contributes nothing past ``len(seq1)``. Since the
    callers pad to a common length this never happens today, but it is the
    half of the asymmetry that would fail quietly rather than loudly.
    """
    from spacr.sequencing import create_consensus

    assert len(create_consensus("AC", "II", "ACGT", "IIII")) == 2


def test_the_higher_quality_base_wins_and_n_is_avoided():
    """What the consensus is FOR, so the length tests are not the whole story."""
    from spacr.sequencing import create_consensus

    # Second read has the better quality at position 0.
    assert create_consensus("A", "!", "G", "I") == "G"
    # A real base beats an N even when the N scores higher.
    assert create_consensus("N", "I", "T", "!") == "T"
