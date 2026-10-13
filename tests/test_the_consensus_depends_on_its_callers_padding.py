"""``create_consensus`` is safe only because every caller pads first.

This file pins an invariant rather than a branch, because the branch it was
written for turns out to be unreachable and the invariant is what actually
protects the code.

``create_consensus`` used to loop over ``len(seq1)`` and index ``seq2`` at the
same offset. A shorter second sequence raised IndexError from inside a chunk
worker, while a longer one was silently truncated. All three callers pad both
reads to ``expected_end`` first, but the function now also names and rejects
the invariant itself so a later caller cannot build a partial barcode.
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


@pytest.mark.parametrize("seq1, qual1, seq2, qual2, fragment", [
    ("ACGT", "IIII", "AC", "II", "seq2=2"),
    ("AC", "II", "ACGT", "IIII", "seq2=4"),
    ("ACGT", "II", "ACGT", "IIII", "qual1=2"),
    ("ACGT", "IIII", "ACGT", "II", "qual2=2"),
])
def test_uneven_sequences_or_qualities_are_rejected_with_their_lengths(
        seq1, qual1, seq2, qual2, fragment):
    """Neither truncation nor an internal IndexError is a scientific answer."""
    from spacr.sequencing import create_consensus

    with pytest.raises(ValueError, match=fragment):
        create_consensus(seq1, qual1, seq2, qual2)


def test_the_higher_quality_base_wins_and_n_is_avoided():
    """What the consensus is FOR, so the length tests are not the whole story."""
    from spacr.sequencing import create_consensus

    # Second read has the better quality at position 0.
    assert create_consensus("A", "!", "G", "I") == "G"
    # A real base beats an N even when the N scores higher.
    assert create_consensus("N", "I", "T", "!") == "T"
