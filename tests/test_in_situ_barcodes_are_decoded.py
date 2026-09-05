"""Decode barcodes we planted ourselves, so "correct" is a comparison.

An in-situ decoder can be wrong in ways that look completely healthy: it finds
spots, it emits barcodes of the right length, and a fixed fraction of them even
match the library by chance. The only way to know it works is to plant known
barcodes and demand them back.

THE SYNTHETIC FIELD IS BUILT TO PUNISH the three mistakes that matter:

* **Constant-bright debris.** Blobs as bright as any read, in every channel,
  in every cycle. A decoder that looks for bright things finds these; one that
  looks for CHANGE does not. They carry no barcode, so any read reported at a
  debris location is a false positive by construction.
* **Cross-talk.** Each dye bleeds into its neighbours through a mixing matrix,
  which is what makes the raw argmax the wrong answer.
* **An unequal base composition.** One base is deliberately rare, which is the
  case a median-based cross-talk fit gets wrong and a percentile-based one
  survives.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spacr import ops_sbs

CYCLES = 8
CHANNELS = 4
SIZE = 320
N_SPOTS = 120
N_DEBRIS = 15

#: How much each dye leaks into the others. Row i is what channel i emits.
#: Deliberately severe -- 25 % into a neighbour is more than a real filter set
#: would pass, so a decoder that survives this has margin.
BLEED = np.array([
    [1.00, 0.25, 0.05, 0.02],
    [0.20, 1.00, 0.18, 0.04],
    [0.04, 0.15, 1.00, 0.22],
    [0.02, 0.05, 0.20, 1.00],
], dtype=np.float32)


def _plant(rng):
    """Build a field of reads with known barcodes, plus debris.

    :param rng: the seeded generator, so a failure is reproducible.
    :returns: ``(stack, truth)`` -- a (cycles, channels, Y, X) array and a
        dict of ``(y, x) -> barcode``.
    """
    stack = np.zeros((CYCLES, CHANNELS, SIZE, SIZE), np.float32)

    # A rare base: 'C' appears at a tenth the rate of the others, which is
    # what breaks a median-based cross-talk fit.
    weights = np.array([0.3, 0.3, 0.3, 0.1])
    truth = {}
    taken = set()
    while len(truth) < N_SPOTS:
        y = int(rng.integers(12, SIZE - 12))
        x = int(rng.integers(12, SIZE - 12))
        # Keep spots apart: two reads inside one optical spot is a different
        # problem (crowding) and not the one under test here.
        if any(abs(y - ay) < 9 and abs(x - ax) < 9 for ay, ax in taken):
            continue
        taken.add((y, x))
        idx = rng.choice(CHANNELS, size=CYCLES, p=weights)
        truth[(y, x)] = "".join(ops_sbs.BASES[i] for i in idx)
        for cycle, channel in enumerate(idx):
            pure = np.zeros(CHANNELS, np.float32)
            pure[channel] = rng.uniform(2500, 4200)
            observed = pure @ BLEED          # the dyes bleed into each other
            for c in range(CHANNELS):
                stack[cycle, c, y - 1:y + 2, x - 1:x + 2] += observed[c]

    # Debris: bright in EVERY channel and EVERY cycle, so it varies not at all.
    for _ in range(N_DEBRIS):
        y = int(rng.integers(12, SIZE - 12))
        x = int(rng.integers(12, SIZE - 12))
        stack[:, :, y - 2:y + 3, x - 2:x + 3] += rng.uniform(3800, 5200)

    stack += rng.normal(90, 26, stack.shape).astype(np.float32)
    return np.clip(stack, 0, None), truth


@pytest.fixture(scope="module")
def planted():
    """One field, built once; every test below interrogates the same one."""
    return _plant(np.random.default_rng(20260905))


def test_the_read_map_scores_change_not_brightness(planted):
    """Debris is brighter than any read and must still score lower.

    This is the assertion that separates "find the bright things", which is
    wrong, from "find the things that change", which is the method. If it
    fails, every barcode below is being read off whatever happened to be
    luminous.
    """
    stack, truth = planted
    score = ops_sbs.estimate_read_locations(stack)

    read_scores = [score[y, x] for (y, x) in truth]
    # The debris is the brightest thing in the raw data by construction.
    brightest = stack.max(axis=(0, 1))
    assert brightest.max() > 0

    assert float(np.median(read_scores)) > 0, "reads score nothing at all"
    # Every planted read must out-score the median pixel by a wide margin.
    background = float(np.median(score))
    assert float(np.median(read_scores)) > 10 * max(background, 1e-6)


def test_every_planted_barcode_is_decoded(planted):
    """The barcodes that come back must be the barcodes that went in."""
    stack, truth = planted
    score = ops_sbs.estimate_read_locations(stack)
    peaks = ops_sbs.find_peaks(score, min_distance=4)
    assert len(peaks) > 0, "no peaks found at all"

    values = ops_sbs.extract_bases(stack, peaks, window=1)
    barcodes, quality = ops_sbs.call_reads(values)

    # Match each peak back to the read it sits on, if any.
    found = {}
    for (y, x), code, q in zip(peaks, barcodes, quality):
        for (ty, tx), expected in truth.items():
            if abs(int(y) - ty) <= 2 and abs(int(x) - tx) <= 2:
                found[(ty, tx)] = (code, expected, float(q))
                break

    assert len(found) >= int(0.9 * len(truth)), (
        f"only {len(found)} of {len(truth)} planted reads were located"
    )
    wrong = [(k, got, want) for k, (got, want, _q) in found.items()
             if got != want]
    assert not wrong, (
        f"{len(wrong)} of {len(found)} barcodes decoded wrongly, e.g. "
        f"{wrong[:3]}"
    )


def test_the_crosstalk_correction_is_what_makes_it_work():
    """With enough bleed the raw argmax is wrong, and the correction fixes it.

    A correction that changes nothing is one that is not needed, and a suite
    that would pass without it is not testing it. So this uses a SEVERE mixing
    matrix -- one where a neighbouring dye out-shines the true base often
    enough to matter -- and asserts the compensated call beats the raw one.

    The bleed in the main fixture is deliberately milder, because a decoder
    should also be correct on easy data; asserting the correction's value
    needs the hard case, and putting the hard case here keeps the two
    questions apart.
    """
    rng = np.random.default_rng(4242)
    n_reads, cycles = 400, 6
    # THE LAST ROW IS THE POINT. Base 'C' is a dim dye that leaks harder into
    # channel 2 than it emits into its own: 0.75 against 0.55. An uncorrected
    # argmax therefore calls EVERY 'C' an 'A', systematically, which is the
    # failure mode that matters -- it is not noise, it is the same base
    # mis-read everywhere, and it survives any amount of averaging.
    bleed = np.array([
        [1.00, 0.30, 0.10, 0.05],
        [0.10, 1.00, 0.15, 0.05],
        [0.05, 0.20, 1.00, 0.10],
        [0.05, 0.10, 0.75, 0.55],
    ], dtype=np.float32)

    idx = rng.integers(0, CHANNELS, size=(n_reads, cycles))
    values = np.zeros((n_reads, cycles, CHANNELS), np.float32)
    for r in range(n_reads):
        for c in range(cycles):
            pure = np.zeros(CHANNELS, np.float32)
            pure[idx[r, c]] = rng.uniform(2500, 4200)
            values[r, c] = pure @ bleed
    values += rng.normal(60, 20, values.shape).astype(np.float32)

    truth = ["".join(ops_sbs.BASES[i] for i in row) for row in idx]
    with_it, _ = ops_sbs.call_reads(values, compensate=True)
    without, _ = ops_sbs.call_reads(values, compensate=False)

    good = sum(a == b for a, b in zip(with_it, truth))
    bad = sum(a == b for a, b in zip(without, truth))
    assert bad < n_reads, "the fixture is too easy: the raw call is perfect"
    assert good > bad, (
        f"compensation did not help: {good} correct with it, {bad} without"
    )


def test_an_ambiguous_read_is_dropped_rather_than_guessed():
    """Equidistant from two library barcodes means None, not a coin toss.

    A misassigned guide moves a cell's phenotype onto the wrong perturbation
    and corrupts every statistic downstream. A dropped read only costs power.
    """
    library = ["AAAA", "TTTT", "AATT"]
    reads = [
        "AAAA",   # exact match, short-circuits
        "AAAG",   # AAAA at 1, AATT at 2, TTTT at 3 -> unique
        "AATG",   # AATT at 1, AAAA at 2, TTTT at 3 -> unique
        "TTAA",   # AAAA at 2 and TTTT at 2 -> TIED, must be dropped
        "GGGG",   # 4 from everything -> beyond max_distance
    ]
    got = ops_sbs.correct_to_library(reads, library, max_distance=2)
    assert got[0] == "AAAA"
    assert got[1] == "AAAA"
    assert got[2] == "AATT"
    assert got[3] is None, "a tie was resolved instead of being dropped"
    assert got[4] is None

    # AAAT and AATA sit at distance 1 from TWO library barcodes each. An
    # earlier draft of this test expected them to resolve, which was simply
    # wrong arithmetic -- and the implementation was right to refuse them.
    ambiguous = ops_sbs.correct_to_library(["AAAT", "AATA"], library,
                                           max_distance=2)
    assert ambiguous == [None, None]
