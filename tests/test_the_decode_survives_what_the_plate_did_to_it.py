"""372 PART 14-L's Phase C failures, each planted on synthetic data first.

The first real decode (well A1 of screenA/20200202_6W-LaC024A) ran the
modules end to end and decoded noise, and every failure was of the kind that
leaves the tables full:

* the four base channels of ONE cycle sat up to 14 px apart and nothing
  registered them (library match 2.8 % without, 77.9 % with);
* ``call_reads``' default cross-talk compensation scored 59.3 % against
  77.9 % for a per-cycle, per-channel median normalisation;
* ``gpu=False`` could not keep the call off the card;
* reads sit AROUND the nucleus, so sampling at its centroid decoded at
  chance (0.7 %) and PART 9's nucleus + 3 px footprint held 67 % of spots
  against its own 0.8 gate, while nucleus + 10 px held 98 %;
* one truncated cycle file cost a field every call it had.

Each test below reproduces one of those shapes on planted truth, so the
assertion is against an answer decided before the code ran.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy import ndimage

from spacr import ops_sbs

BASES = ops_sbs.BASES


def _field(seed, *, offsets=None, cycles=4, size=256, n_reads=60, pad=40):
    """A field of reads on a textured background, cropped as a camera sees it.

    Every plane is cut out of one larger canvas, so a shifted channel shows
    real content at its border rather than a zero band -- a zero band is an
    edge only the shifted plane has, which no acquisition produces and which
    a whitened correlation would lock onto. The background is shared by every
    channel of every cycle, which is what cells give a real acquisition and
    what a channel-to-channel registration has to lock onto; each read is
    bright in one channel per cycle.

    :param seed: the generator seed, so the same field can be cut twice.
    :param offsets: ``{(cycle, channel): (dy, dx)}``, how far the content of
        that plane moved; absent planes did not move.
    :param cycles: how many cycles to plant, numbered from 1.
    :param size: the field edge.
    :param n_reads: how many reads.
    :param pad: the canvas margin the offsets may use.
    :returns: ``(planes, truth)`` -- ``{cycle: [4 planes]}`` and
        ``{(y, x): barcode}`` in field coordinates.
    """
    rng = np.random.default_rng(seed)
    offsets = offsets or {}
    edge = size + 2 * pad
    # Sigma 3 is cell-edge scale. A sigma-6 background is so smooth that a
    # whitened correlation's peak is several pixels wide and one channel came
    # back 5 px out, which says the fixture had no structure, not the method.
    texture = ndimage.gaussian_filter(rng.random((edge, edge)), 3)
    texture = (texture - texture.min()) / np.ptp(texture) * 600 + 200
    truth = {}
    while len(truth) < n_reads:
        y, x = (int(v) for v in rng.integers(40, size - 40, 2))
        if any(abs(y - a) < 10 and abs(x - b) < 10 for a, b in truth):
            continue
        truth[(y, x)] = "".join(BASES[i] for i in rng.integers(0, 4, cycles))
    planes = {}
    for cycle in range(1, cycles + 1):
        stack = []
        for channel in range(4):
            # Independent noise per plane, drawn in a fixed order so the same
            # seed cuts the same pixels whatever the offsets are. Without it
            # every channel shows the identical background and a read whose
            # spot moved away is called by the channels' medians rather than
            # by chance.
            canvas = (texture + rng.normal(0, 10, texture.shape)).astype(np.float32)
            for (y, x), code in truth.items():
                if code[cycle - 1] == BASES[channel]:
                    canvas[pad + y - 1:pad + y + 2, pad + x - 1:pad + x + 2] += 3000
            dy, dx = offsets.get((cycle, channel), (0, 0))
            stack.append(canvas[pad - dy:pad - dy + size,
                                pad - dx:pad - dx + size].copy())
        planes[cycle] = stack
    return planes, truth


def _decode_at(stack, truth, **kwargs):
    """Call the planted positions' barcodes out of a ``(C, K, Y, X)`` stack.

    :param stack: the aligned stack.
    :param truth: ``{(y, x): barcode}``.
    :param kwargs: passed to :func:`spacr.ops_sbs.call_reads`.
    :returns: ``{(y, x): whether that read came back exactly}``.
    """
    peaks = np.array(list(truth))
    values = ops_sbs.extract_bases(stack, peaks, window=1)
    calls, _ = ops_sbs.call_reads(values, gpu=False, **kwargs)
    return {place: got == want
            for place, got, want in zip(truth, calls, truth.values())}


# -- channels within one cycle -------------------------------------------------

def test_a_channel_shifted_14_px_within_its_cycle_is_put_back():
    """PART 14-L measured within-cycle channel shifts up to 14 px.

    The cycles are ALSO displaced from each other, as the stage does, so the
    test asks for both corrections and checks the aligned planes against the
    planted geometry pixel for pixel away from the border.
    """
    from spacr.ops_cycles import align_field

    whole_cycle = {(3, channel): (-6, 4) for channel in range(4)}
    moved, _ = _field(372, offsets={**whole_cycle, (2, 3): (14, -9),
                                    (3, 1): (-11, 15)})
    still, _ = _field(372)

    field = align_field(moved, gpu=False)

    assert field.kept == (1, 2, 3, 4)
    assert field.channel_shifts[2][3] == (-14, 9)
    assert field.channel_shifts[3][1] == (5, -11)
    assert field.cycle_shifts[3] == (6, -4)
    inner = (slice(40, -40), slice(40, -40))
    for index, cycle in enumerate(field.kept):
        for channel in range(4):
            np.testing.assert_array_equal(
                field.stack[index, channel][inner],
                still[cycle][channel][inner],
                err_msg=f"cycle {cycle} channel {channel} is not back in register")


def test_without_the_channel_alignment_the_same_reads_decode_wrongly():
    """The measured failure itself: every table fills and the calls are wrong.

    A 14 px channel shift moves one base's spot off the read, so that cycle
    calls whichever channel is left. The aligned field reads every planted
    barcode back; the same field stacked as acquired does not.
    """
    from spacr.ops_cycles import align_field

    planes, truth = _field(1414, offsets={(2, 2): (14, 0), (3, 3): (14, 0)})

    as_acquired = np.stack([np.stack(planes[c]) for c in sorted(planes)])
    aligned = align_field(planes, gpu=False)

    # Channel 2 of cycle 2 is base A and channel 3 of cycle 3 is base C: the
    # reads carrying those bases in those cycles lost their spot.
    hit = {place for place, code in truth.items()
           if code[1] == BASES[2] or code[2] == BASES[3]}
    assert hit and len(hit) < len(truth)
    assert all(_decode_at(aligned.stack, truth).values())
    before = _decode_at(as_acquired, truth)
    # A read whose spot moved away shows the local background in three
    # channels and the background from 14 px away in the shifted one, so that
    # cycle's call is the right letter whenever the distant background happens
    # to be brighter -- 15 of 29 at this seed. And a moved spot can land on a
    # neighbouring read and outshine its base: 2 of the other 31 here. So a
    # third of the affected reads must fail, and at most a tenth of the rest.
    wrong = sum(not before[place] for place in hit)
    assert wrong >= len(hit) / 3, (
        f"only {wrong} of {len(hit)} reads whose spot moved 14 px decoded "
        "wrongly without the alignment")
    others = [place for place in truth if place not in hit]
    assert sum(not before[place] for place in others) <= len(others) / 10


def test_a_cycle_with_no_planes_is_reported_missing_not_fatal():
    """A cycle whose files could not be read is dropped, and said to be."""
    from spacr.ops_cycles import align_field

    planes, _ = _field(9)
    planes[3] = None

    field = align_field(planes, gpu=False)

    assert field.kept == (1, 2, 4)
    assert field.missing == (3,)
    assert field.stack.shape[0] == 3


# -- the base call -------------------------------------------------------------

def _drifting_values(rng, *, n=3000, cycles=11):
    """Reads whose channel gains drift with cycle, as the plate's did.

    PART 14-L: base frequencies drifted toward C (CY7) with cycle number,
    reaching 60-70 % by cycle 7 before normalisation. Here the C channel's
    gain and its background both rise threefold over the run, and every
    channel carries an autofluorescent floor, with no cross-talk at all.

    :param rng: seeded generator.
    :param n: reads.
    :param cycles: cycles.
    :returns: ``(values, truth)``.
    """
    idx = rng.integers(0, 4, size=(n, cycles))
    gain = np.ones((cycles, 4))
    gain[:, 3] = np.linspace(1.0, 3.0, cycles)
    floor = rng.uniform(900, 1300, size=(1, cycles, 4)) * gain
    values = floor + rng.normal(0, 120, size=(n, cycles, 4))
    rows, cols = np.meshgrid(np.arange(n), np.arange(cycles), indexing="ij")
    values[rows, cols, idx] += rng.uniform(900, 1800, size=(n, cycles)) \
        * gain[cols, idx]
    truth = ["".join(BASES[i] for i in row) for row in idx]
    return values.astype(np.float32), truth


def test_the_default_call_normalises_and_beats_the_old_compensated_default():
    """``call_reads`` used to compensate by default and scored worst of three."""
    values, truth = _drifting_values(np.random.default_rng(593))

    default, _ = ops_sbs.call_reads(values, gpu=False)
    compensated, _ = ops_sbs.call_reads(values, compensate=True,
                                        normalise=False, gpu=False)
    exact = lambda calls: float(np.mean([a == b for a, b in zip(calls, truth)]))

    assert exact(default) > 0.9
    # Measured at seed 593: 1.000 against 0.862. Today's code called the
    # compensated path by default and so failed the first assertion.
    assert exact(default) > exact(compensated) + 0.1


def test_gpu_false_keeps_the_compensation_off_the_card(monkeypatch):
    """``call_reads(gpu=False)`` must reach the multiply as ``gpu=False``."""
    import spacr.ops_accel as accel

    seen = []
    real = accel.accelerated_backends

    def spy(gpu=True):
        """Record what the multiply was asked for, then answer honestly."""
        seen.append(gpu)
        return real(gpu)

    monkeypatch.setattr(accel, "accelerated_backends", spy)
    values, _ = _drifting_values(np.random.default_rng(1), n=200)
    ops_sbs.call_reads(values, compensate=True, gpu=False)

    assert seen and not any(seen), f"the multiply was asked for gpu={seen}"


def test_a_cycle_that_was_not_measured_is_called_N_and_the_rest_survive():
    """NaN for a lost cycle: that base is N, the other ten are still called."""
    values, truth = _drifting_values(np.random.default_rng(7), n=500)
    values[:, 4, :] = np.nan

    calls, quality = ops_sbs.call_reads(values, gpu=False)

    assert all(code[4] == "N" for code in calls)
    kept = [a[:4] + a[5:] == b[:4] + b[5:] for a, b in zip(calls, truth)]
    assert np.mean(kept) > 0.9
    assert np.isfinite(quality).all()


def test_a_read_with_a_lost_base_still_maps_but_never_to_two_guides():
    """N is a base that was not measured: it matches any letter, never two guides.

    At ``max_distance=0`` the old code counted the N as a mismatch and could
    map nothing with a lost cycle, which is what a field with one truncated
    file hands the library.
    """
    library = ["GTACGTACGTA", "GTACGTACGTC", "AAAACCCCGGG"]
    exact = ops_sbs.correct_to_library(
        ["AAAANCCCGGG",      # one guide once the N is allowed
         "GTACGTACGTN",      # two guides differ only at the N: ambiguous
         "AAAANCCCGGT",      # an N AND a mismatch: not an exact fit
         "GTACGTACGTA"],     # exact
        library, max_distance=0)
    assert exact == ["AAAACCCCGGG", None, None, "GTACGTACGTA"]

    one_off = ops_sbs.correct_to_library(
        ["AAAANCCCGGT", "GTACGTACGTN", "NNAACCCCGGG", "NNNNNNNNNNN"],
        library, max_distance=1)
    assert one_off == ["AAAACCCCGGG", None, "AAAACCCCGGG", None]


def test_library_correction_agrees_with_brute_force_and_scales_to_a_screen():
    """20,445 guides and a field's reads must not be a quadratic Python loop."""
    import time

    rng = np.random.default_rng(20445)
    letters = np.array(list(BASES))
    library = sorted({"".join(row) for row in
                      letters[rng.integers(0, 4, size=(20445, 11))]})
    reads = []
    for _ in range(500):
        code = list(library[int(rng.integers(0, len(library)))])
        for _ in range(int(rng.integers(0, 3))):
            code[int(rng.integers(0, 11))] = BASES[int(rng.integers(0, 4))]
        reads.append("".join(code))

    started = time.perf_counter()
    fast = ops_sbs.correct_to_library(reads, library, max_distance=1)
    elapsed = time.perf_counter() - started

    def brute(read):
        """The old loop, on a sample only."""
        best, tied, distance = None, False, 2
        for candidate in library:
            d = sum(a != b for a, b in zip(read, candidate))
            if d < distance:
                best, tied, distance = candidate, False, d
            elif d == distance:
                tied = True
        return None if best is None or tied or distance > 1 else best

    sample = rng.choice(len(reads), 60, replace=False)
    assert [fast[i] for i in sample] == [brute(reads[i]) for i in sample]
    # The old loop costs reads x guides x letters in Python: tens of
    # seconds for these 500 reads, and a whole field holds thousands.
    assert elapsed < 1.0, f"{elapsed:.2f} s for 500 reads"


# -- where the reads are -------------------------------------------------------

def _nuclei():
    """Three planted nuclei: centroids ``(y, x)`` and areas (radius 6 px)."""
    centroids = np.array([[50.0, 50.0], [50.0, 90.0], [120.0, 60.0]])
    areas = np.full(3, math.pi * 36.0)
    return centroids, areas


def test_a_perinuclear_read_belongs_to_the_nucleus_within_ten_px_not_three():
    """Reads 7 px beyond the boundary: owned at +10 px, lost at +3 px."""
    centroids, areas = _nuclei()
    peaks = np.array([[50.0 - 13.0, 50.0],       # 7 px above nucleus 0's edge
                      [120.0, 60.0 + 12.0],      # 6 px right of nucleus 2
                      [200.0, 200.0]])           # near nothing
    owner, ambiguous = ops_sbs.attribute_reads(peaks, centroids, areas,
                                               footprint=10.0)
    assert owner.tolist() == [0, 2, -1]
    assert not ambiguous.any()

    tight, _ = ops_sbs.attribute_reads(peaks, centroids, areas, footprint=3.0)
    assert tight.tolist() == [-1, -1, -1]


def test_a_read_between_two_boundaries_is_given_to_neither_and_counted():
    """PART 9: a spot inside two footprints is assigned to neither."""
    centroids, areas = _nuclei()
    # Nuclei 0 and 1 are 40 px apart with 6 px radii: a read midway is 14 px
    # beyond both boundaries, so the footprint is widened to reach it.
    peaks = np.array([[50.0, 70.0],              # 14 px from both boundaries
                      [50.0, 64.0]])             # 8 px from one, 20 from other
    owner, ambiguous = ops_sbs.attribute_reads(peaks, centroids, areas,
                                               footprint=15.0)
    assert owner.tolist() == [-1, 0]
    assert ambiguous.tolist() == [True, False]


def test_sampling_at_the_centroid_misses_reads_that_attribution_finds():
    """The C1 design sampled at the centroid; the reads are not there.

    Each nucleus carries three agreeing reads around its rim and nothing at
    its centre. Sampling the stack at the centroid reads background and calls
    noise; detecting the spots, calling them and attributing them to the
    nearest boundary gives every nucleus its barcode.
    """
    from spacr.ops_objects import PlateObject
    from spacr.ops_sample import barcode_rows, decode_input, sample_objects

    rng = np.random.default_rng(15)
    centroids, areas = _nuclei()
    codes = ["GTACGTAC", "CATGCATG", "AACCGGTT"]
    stack = rng.normal(100, 10, size=(8, 4, 180, 180)).astype(np.float32)
    peaks, labels = [], []
    for index, ((cy, cx), code) in enumerate(zip(centroids, codes)):
        for angle in (0.3, 2.4, 4.4):
            y = int(round(cy + 9 * math.sin(angle)))
            x = int(round(cx + 9 * math.cos(angle)))
            for cycle, base in enumerate(code):
                stack[cycle, BASES.index(base), y - 1:y + 2, x - 1:x + 2] += 3000
            peaks.append((y, x))
            labels.append(code)

    objects = [PlateObject(object_id=i + 1, centroid_x=float(x),
                           centroid_y=float(y), area=int(a),
                           bbox=(0, 0, 1, 1), window=(0, 0))
               for i, ((y, x), a) in enumerate(zip(centroids, areas))]
    sampled = decode_input([[sample_objects(objects, stack[c, k], radius=1)
                             for k in range(4)] for c in range(8)])
    at_centroid = [row["barcode"] for row in barcode_rows(objects, sampled)]
    assert sum(a == b for a, b in zip(at_centroid, codes)) == 0

    values = ops_sbs.extract_bases(stack, np.array(peaks), window=1)
    calls, quality = ops_sbs.call_reads(values, gpu=False)
    owner, _ = ops_sbs.attribute_reads(np.array(peaks, float), centroids,
                                       areas, footprint=10.0)
    ids = np.where(owner >= 0, owner + 1, 0)
    got = ops_sbs.assign_reads_to_objects(ids, calls, quality=quality)
    assert {k: v["barcode"] for k, v in got.items()} == {
        1: codes[0], 2: codes[1], 3: codes[2]}


def test_objects_vote_on_their_reads_exactly_as_cells_do():
    """The object vote keeps the cell vote's refusals: one read, a split."""
    ids = np.array([1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 0])
    calls = ["GT", "GT", "TT", "GT", "AA", "CC", "AA", "AA", "CC", "CC", "GG"]
    got = ops_sbs.assign_reads_to_objects(ids, calls)
    assert set(got) == {1}
    assert got[1]["barcode"] == "GT" and got[1]["reads"] == 3
    assert got[1]["agreeing"] == 2
    lenient = ops_sbs.assign_reads_to_objects(ids, calls, min_fraction=0.5)
    assert set(lenient) == {1}, "a 1-1 or 2-2 split was broken arbitrarily"
