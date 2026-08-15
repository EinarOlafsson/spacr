"""spacr.align — what the filename decides, and what the solve weighs.

``tests/test_align.py`` and ``tests/test_align_cov.py`` between them already
execute every line of :mod:`spacr.align`. Executing a line is not the same as
checking what it computed, and this file goes after two things those suites
step over:

* **the name is the key.** :func:`~spacr.align.scan_tiles` decides from a
  filename alone which files are the same stage position and which are
  different ones. Get that wrong and tiles are dropped, or two wells are
  composited into one canvas as if they were two channels — silently, with
  a clean-looking plan and a full coordinates table. Five naming shapes are
  pinned here; three of them are broken today and carry
  ``xfail(strict=True)`` asserting the correct answer, in this repo's usual
  form.
* **the numbers the least-squares solve actually produces.** The existing
  suite proves the global solve beats sequential accumulation. It never
  checks that a pair's ``confidence`` reaches the solver as a *weight*, nor
  that ``DEFAULT_ANCHOR_WEIGHT`` is small enough to leave a registered chain
  alone — both are single-constant regressions that would degrade every
  stitch on the plate while every existing test still passed.

Plus the two documented settings the suite never passes (``save_stack``,
``group_by_well``) and the reader-cache size, which turns out to unregister
an entire plate when it is set to its own documented minimum.

Everything is deterministic (fixed seeds), CPU-only, offline, and no tile
here is larger than 128x128.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import align
from spacr.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Builders — same shape as the two existing align suites use.
# ---------------------------------------------------------------------------

def _texture(height, width, seed=0, sigma=2.0):
    """A smooth, non-repeating uint16 field — registrable, unlike noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    raw = rng.random((height, width)).astype(np.float32)
    smooth = gaussian_filter(raw, sigma)
    smooth -= smooth.min()
    smooth /= max(float(smooth.max()), 1e-9)
    return (smooth * 30000 + 1000).astype(np.uint16)


TILE = 128
STEP = 80
OVERLAP = 1 - STEP / TILE


def _write_2x2(folder, plate='plate1', well='B07', seed=7):
    """Cut one texture into a 2x2 grid of ``TILE`` tiles stepped ``STEP``."""
    os.makedirs(folder, exist_ok=True)
    big = _texture(TILE + STEP + 8, TILE + STEP + 8, seed=seed)
    k = 0
    for row in range(2):
        for col in range(2):
            np.save(os.path.join(folder, f'{plate}_{well}_{k + 1:03d}.npy'),
                    big[row * STEP:row * STEP + TILE,
                        col * STEP:col * STEP + TILE])
            k += 1
    return big


# ---------------------------------------------------------------------------
# 1. The filename is the key: which files are one stage position
# ---------------------------------------------------------------------------

YOKOGAWA_1536_BUG = (
    "scan_tiles collapses every field of a 1536-format well into ONE tile "
    "and silently drops the rest. _YOKOGAWA's (?P<well>[A-Za-z]?\\d+) allows "
    "at most one letter, so 'AA01' never matches; the last-resort branch of "
    "_parse_name then takes the trailing digits of the *channel* token "
    "('...C01' -> field 1) as the field, so F001..F004 all key to "
    "('', '', 1) and sites[key]['paths'][1] keeps only the last file read. "
    "spacr.schema.parse_well('AA01') == ('r27', 'c1'), so the rest of spaCR "
    "handles 1536 plates; only align's regex does not."
)


def test_a_1536_well_export_keeps_one_tile_per_field(tmp_path):
    """Four fields of well AA01 are four stage positions, not one.

    A 1536-plate Yokogawa export names its wells with two letters. Every
    field of such a well must still come back as its own tile: dropping
    three of four fields is not a stitch that is slightly off, it is three
    quarters of the well missing from the canvas and from
    ``align_coordinates``, with nothing in either saying so.
    """
    folder = tmp_path / 'yoko1536'
    folder.mkdir()
    for field in range(1, 5):
        np.save(folder / f'plate1_AA01_T0001F{field:03d}L01A01Z01C01.npy',
                _texture(32, 32, seed=field))

    tiles = align.scan_tiles(str(folder), grid=(2, 2), overlap=0.25)

    assert len(tiles) == 4, 'one tile per field'
    assert {t.well for t in tiles} == {'AA01'}
    assert sorted(t.field for t in tiles) == [1, 2, 3, 4]
    assert all(t.channel_paths == () for t in tiles), \
        'four fields of one channel are not four channels of one field'


CROSS_WELL_MERGE_BUG = (
    "scan_tiles merges two different wells into one site when the plate "
    "token contains '_'. _MERGED's (?P<plate>[^_]+) cannot match "
    "'exp1_plate1', so 'exp1_plate1_C08_001' falls to the last-resort "
    "branch, where _CHANNEL_TOKEN reads the WELL token 'C08' as 'channel 8' "
    "and _TRAILING_FIELD reads '001' as field 1 — the same key as B07's "
    "field 1. write_stack then composites well C08 into well B07's canvas "
    "as plane 2, and every align_coordinates row has an empty "
    "plateID/rowID/columnID, so nothing downstream can tell them apart."
)


def test_two_wells_never_become_two_channels_of_one_site(tmp_path):
    """Fields of different wells are different sites, whatever the plate is called.

    ``group_tiles`` exists precisely because fields of different wells are
    not neighbours. That guarantee is only as good as the parse that fills
    in ``Tile.well``: if two wells key to the same site, the wells are
    merged before grouping ever runs, and one well's pixels end up in the
    other well's canvas.
    """
    folder = tmp_path / 'underscore_plate'
    folder.mkdir()
    for well, value in (('B07', 111), ('C08', 999)):
        for field in (1, 2):
            np.save(folder / f'exp1_plate1_{well}_{field:03d}.npy',
                    np.full((32, 32), value, np.uint16))

    tiles = align.scan_tiles(str(folder), grid=(2, 2), overlap=0.25)

    assert len(tiles) == 4, 'two wells x two fields are four stage positions'
    for tile in tiles:
        sources = {os.path.basename(p) for p in
                   (tile.channel_paths or (tile.path,))}
        wells = {name.split('_')[2] for name in sources}
        assert len(wells) == 1, f'{sources} mixes wells into one site'
    assert len(align.group_tiles(tiles)) == 2, 'one stitchable set per well'


def test_a_1536_well_is_keyed_the_way_the_rest_of_spacr_keys_it(tmp_path):
    """``AA01`` reaches the database as ``r27``/``c1``, not as ``AA01``.

    This is the half of the 1536 story that works, and the reason the
    scan-side failure above is a defect rather than an unsupported format:
    the key layer delegates to :mod:`spacr.schema`, so a two-letter well
    joins to ``cell`` exactly like any other. Row 27 is ``AA`` — a copy of
    this mapping that stopped at ``Z`` is the bug the module docstring
    records having already been bitten by once.
    """
    tile = align.Tile(path=str(tmp_path / 'x.npy'), index=0, plate='plate1',
                      well='AA01', field=3, shape=(16, 16, 1))
    plan = align.AlignPlan(
        tiles=[tile],
        placements=[align.Placement(tile=tile, y=0.0, x=0.0, confidence=1.0,
                                    method=align.METHOD_SINGLE)],
        dtype='uint16')

    db_path = tmp_path / 'm.db'
    assert align.save_coordinates(plan, db_path) == 1

    frame = align.read_coordinates(db_path)
    assert list(frame['rowID']) == ['r27']
    assert list(frame['columnID']) == ['c1']
    assert list(frame['fieldID']) == ['f3']
    assert list(frame['prcf']) == ['plate1_r27_c1_f3']
    # And the well filter finds it by its human name too.
    assert len(align.read_coordinates(db_path, well='AA01')) == 1


REFERENCE_CHANNEL_BUG = (
    "scan_tiles(reference_channel=N) is documented as 'which channel drives "
    "registration' and stores it as Tile.channel ('the channel that drives "
    "registration for this site'), but nothing in spacr/align.py ever reads "
    "Tile.channel — estimate_offsets uses its own reference_channel "
    "parameter, which defaults to 0. A caller who selects the reference "
    "channel at scan time silently registers on channel 0 instead."
)


@pytest.mark.xfail(strict=True, reason=REFERENCE_CHANNEL_BUG)
def test_the_reference_channel_chosen_at_scan_time_drives_registration(tmp_path):
    """Asking for channel 2 must register on channel 2, blank or not.

    Channel 2 here is a flat constant: an overlap with no variance, which
    the module is careful never to believe. So "did the scan-time choice
    reach the registration?" has a visible answer — honouring it gives two
    nominal placements and a note saying the overlap was blank, ignoring it
    gives two registered tiles measured off channel 1. Getting the *wrong*
    reference channel is silent and, on a plate where one channel is a
    sparse marker, is exactly how a stitch quietly stops registering.
    """
    folder = tmp_path / 'refchan'
    folder.mkdir()
    big = _texture(64, 200, seed=15)
    for k, x0 in enumerate((0, 40)):
        np.save(folder / f'plate1_B07_T0001F{k + 1:03d}L01A01Z01C01.npy',
                big[:, x0:x0 + 100])
        np.save(folder / f'plate1_B07_T0001F{k + 1:03d}L01A01Z01C02.npy',
                np.full((64, 100), 777, np.uint16))

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.5,
                             reference_channel=2)
    assert [t.channel for t in tiles] == [1, 1], 'C02 is plane 1 of the site'

    plan = align.estimate_offsets(tiles)

    assert plan.n_nominal == 2, 'the blank channel 2 cannot be registered'
    assert plan.n_registered == 0
    assert 'blank' in plan.placements[0].note


PER_WELL_GRID_BUG = (
    "align_folder applies 'grid' to the whole folder and only then splits "
    "by well, so the per-well acquisition grid cannot be expressed: two "
    "wells acquired as 2x2 are refused with 'grid 2x2 has room for 4 tiles "
    "but 8 were found', and grid=None infers one 2x4 layout for the folder, "
    "which lays each well's four fields out as a 1x4 strip and puts their "
    "vertical neighbours out of overlap. scan_tiles is called before "
    "group_tiles in align_folder."
)


@pytest.mark.xfail(strict=True, reason=PER_WELL_GRID_BUG)
def test_grid_describes_one_wells_acquisition_not_the_whole_folder(tmp_path):
    """``grid=(2, 2)`` on a two-well plate means 2x2 *per well*.

    A grid is a property of the acquisition, and the acquisition is
    per well — the same 2x2 field pattern is repeated in every well on the
    plate. With ``group_by_well`` on (the default) there is no other thing
    ``grid`` could mean, and no other value a user could pass: (2, 4) is not
    the layout of anything that was imaged.
    """
    src = tmp_path / 'plate'
    src.mkdir()
    for well, seed in (('B07', 7), ('C08', 11)):
        _write_2x2(str(src), well=well, seed=seed)

    results = align.align_folder(src=str(src), dst=str(tmp_path / 'out'),
                                 grid=(2, 2), overlap=OVERLAP,
                                 save_stack=False)

    assert len(results) == 2, 'one stitch per well'
    for result in results:
        # A 2x2 mosaic is about TILE + STEP on a side; a 1x4 strip would be
        # TILE + 3 * STEP wide and one tile tall.
        assert result.canvas.height == pytest.approx(TILE + STEP, abs=4)
        assert result.canvas.width == pytest.approx(TILE + STEP, abs=4)


# ---------------------------------------------------------------------------
# 2. The reader cache: a documented memory knob that unregisters the plate
# ---------------------------------------------------------------------------

TINY_READER_CACHE_BUG = (
    "estimate_offsets(max_open_tiles=1) registers nothing. _ReaderCache "
    "allows max_open=1 (max(1, int(max_open))), but a pair needs BOTH tiles "
    "open at once: cache.get(tile_b) evicts and close()s tile_a's reader "
    "while _register_pair still holds it, so every pair dies with an "
    "internal \"AttributeError: 'NoneType' object has no attribute 'shape'\" "
    "that is swallowed into PairResult.note. The stitch is not refused — "
    "every tile falls back to its stage position and the run looks like a "
    "plate that simply would not register."
)


@pytest.mark.xfail(strict=True, reason=TINY_READER_CACHE_BUG)
def test_a_tiny_reader_cache_does_not_silently_unregister_the_plate(tmp_path):
    """A memory knob may cost speed. It may not change the answer.

    ``max_open_tiles`` is documented as "how many tiles may be
    memory-mapped at once" — a RAM/throughput trade-off. Nothing says it has
    a floor, and the class accepts 1. Either the cache must keep the two
    readers a pair is being registered from, or the value must be refused up
    front; what it must not do is turn a registrable plate into a nominal
    one and report the reason as an AttributeError.
    """
    folder = tmp_path / 'tiny_cache'
    _write_2x2(str(folder))
    tiles = align.scan_tiles(str(folder), grid=(2, 2), overlap=OVERLAP)

    try:
        plan = align.estimate_offsets(tiles, max_open_tiles=1)
    except ConfigurationError:
        return          # refusing the value outright is a legitimate fix

    assert plan.n_registered == 4
    assert not any('AttributeError' in p.note for p in plan.overlaps), \
        [p.note for p in plan.overlaps]


def test_a_two_tile_cache_places_and_writes_exactly_what_the_default_does(
        tmp_path):
    """Evicting between bands must not move a pixel.

    Two is the smallest cache a pair fits in, so this run evicts and
    re-opens a reader on nearly every pair and every band — the LRU path
    that keeps a thousand-tile folder from becoming resident. If a re-opened
    reader ever returned a stale window, an offset window, or a stale
    ``shape``, the placements or the canvas would differ from the run that
    never evicted anything. They are compared exactly.
    """
    folder = tmp_path / 'lru'
    _write_2x2(str(folder))
    tiles = align.scan_tiles(str(folder), grid=(2, 2), overlap=OVERLAP)

    roomy = align.estimate_offsets(tiles, max_open_tiles=8)
    cramped = align.estimate_offsets(tiles, max_open_tiles=2)

    assert roomy.n_registered == 4 and cramped.n_registered == 4
    assert [(p.y, p.x) for p in cramped.placements] == \
           [(p.y, p.x) for p in roomy.placements]

    wide = align.write_stack(roomy, tmp_path / 'wide.npy', max_open_tiles=8)
    narrow = align.write_stack(cramped, tmp_path / 'narrow.npy',
                               max_open_tiles=2, band_rows=17)
    assert wide.n_written == narrow.n_written == 4
    assert np.array_equal(np.load(wide.stack_path),
                          np.load(narrow.stack_path))


# ---------------------------------------------------------------------------
# 3. What the global solve weighs
# ---------------------------------------------------------------------------

def test_a_confident_pair_outweighs_a_pair_that_disagrees_with_it():
    """``confidence`` reaches the solver as a weight, not as decoration.

    Three tiles in a row with one redundant edge: 0-1 and 1-2 each say
    "100 px apart", 0-2 says "225", and the three cannot all be true. Which
    answer the solve leans toward is the whole reason
    :func:`estimate_offsets` passes ``PairResult.confidence`` through as the
    edge weight — a well-textured overlap should pull harder than a marginal
    one. Drop the weight (or pass a constant) and all three cases below
    collapse onto the equal-weight answer, 216.67.
    """
    nominal = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])

    def span(direct_w, chain_w):
        edges = [(0, 1, 0.0, 100.0, chain_w),
                 (1, 2, 0.0, 100.0, chain_w),
                 (0, 2, 0.0, 225.0, direct_w)]
        positions, _residuals, _degrees = align.solve_positions(
            3, edges, nominal)
        return float(positions[2, 1] - positions[0, 1])

    equal = span(0.5, 0.5)
    trust_direct = span(0.9, 0.1)
    trust_chain = span(0.1, 0.9)

    assert equal == pytest.approx(216.667, abs=0.01)
    assert trust_direct == pytest.approx(224.85, abs=0.05)
    assert trust_chain == pytest.approx(200.60, abs=0.05)
    assert trust_chain < equal < trust_direct, \
        'the confident edge must win, and it must win in that direction'


def test_an_inconsistent_triangle_leaves_a_residual_on_every_tile():
    """The residual is an RMS over a tile's own edges, and it is not zero.

    Three measurements that cannot all hold have to leave a trace
    *somewhere*: this is the number :class:`~spacr.align.Placement` tells a
    user to sort on. The 25 px of disagreement is spread evenly over the
    three edges, so every tile carries 8.33 px of it — a residual that came
    back as a plain sum, or that was averaged over all edges instead of the
    tile's own, would not be this number.
    """
    nominal = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    edges = [(0, 1, 0.0, 100.0, 0.5),
             (1, 2, 0.0, 100.0, 0.5),
             (0, 2, 0.0, 225.0, 0.5)]

    _positions, residuals, degrees = align.solve_positions(3, edges, nominal)

    assert list(degrees) == [2, 2, 2]
    assert residuals == pytest.approx([25 / 3, 25 / 3, 25 / 3], abs=0.01)

    # And a consistent triangle leaves none of it.
    consistent = [(0, 1, 0.0, 100.0, 0.5),
                  (1, 2, 0.0, 100.0, 0.5),
                  (0, 2, 0.0, 200.0, 0.5)]
    _p, clean, _d = align.solve_positions(3, consistent, nominal)
    assert clean == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_the_default_anchor_never_drags_a_registered_chain_back_to_nominal():
    """``DEFAULT_ANCHOR_WEIGHT`` fixes the gauge and contributes nothing else.

    The anchors exist so a pure pair graph has an origin and so a tile with
    no pairs lands somewhere sensible. If they weigh anything appreciable
    they also *bend* the answer: here the pairs measure a 90 px step where
    the stage claimed 100, and at the default the solve returns 90 to four
    decimal places. Raise the weight to 10 and the same measurements come
    back as 99.9 — the stage's answer, wearing a registered tile's
    ``method='registration'`` label. That is a one-constant regression no
    existing test would see.
    """
    nominal = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    edges = [(0, 1, 0.0, 90.0, 0.8), (1, 2, 0.0, 90.0, 0.8)]

    at_default, _r, _d = align.solve_positions(
        3, edges, nominal, anchor_weight=align.DEFAULT_ANCHOR_WEIGHT)
    assert float(at_default[1, 1] - at_default[0, 1]) == \
        pytest.approx(90.0, abs=1e-3)
    assert float(at_default[2, 1] - at_default[1, 1]) == \
        pytest.approx(90.0, abs=1e-3)

    heavy, _r, _d = align.solve_positions(3, edges, nominal, anchor_weight=10.0)
    assert float(heavy[1, 1] - heavy[0, 1]) > 99.0, \
        'a heavy anchor overrides the measurements — the default must not'
    assert align.DEFAULT_ANCHOR_WEIGHT <= 1e-2


# ---------------------------------------------------------------------------
# 4. Two settings the suite never passes
# ---------------------------------------------------------------------------

def test_save_stack_false_records_where_the_tiles_go_and_writes_no_pixels(
        tmp_path):
    """The plan-and-record run: coordinates in the database, no canvas on disk.

    This is the setting that makes "show me the plan before you write
    800 MB" usable from the settings dict rather than from the API, and it
    has to do both halves: nothing written, *and* a complete
    ``align_coordinates`` table whose ``stack_path`` is honestly empty
    rather than pointing at a file that does not exist. The canvas geometry
    is still reported, because the number a user is deciding on is what the
    write would have cost.
    """
    src = tmp_path / 'tiles'
    src.mkdir()
    big = _texture(64, 160, seed=17)
    np.save(src / 'plate1_B07_001.npy', big[:, 0:100])
    np.save(src / 'plate1_B07_002.npy', big[:, 60:160])
    db_path = tmp_path / 'm.db'

    results = align.align_folder(src=str(src), dst=str(tmp_path / 'out'),
                                 grid=(1, 2), overlap=0.40,
                                 save_stack=False, db_path=str(db_path))

    assert len(results) == 1
    result = results[0]
    assert result.stack_path == ''
    assert result.n_written == 0
    assert result.canvas.shape == (64, 160, 1)
    assert result.canvas.nbytes == 64 * 160 * 2
    assert result.band_rows >= 1

    written = [p for p in tmp_path.rglob('*.npy') if p.parent != src]
    assert written == [], f'save_stack=False wrote {written}'
    assert not (tmp_path / 'out').exists()

    frame = align.read_coordinates(db_path)
    assert len(frame) == 2
    assert list(frame['stack_path']) == ['', '']
    assert list(frame['method']) == [align.METHOD_REGISTRATION] * 2
    assert list(frame['canvas_width']) == [160, 160]


def test_group_by_well_false_stitches_every_well_onto_one_canvas(tmp_path):
    """Turning the split off puts both wells in one output, and says so per row.

    ``group_by_well=False`` is the escape hatch for a folder whose "wells"
    are not wells — a montage cut into named pieces. The two things that
    have to hold are that there is exactly one canvas (not one per well),
    and that the coordinates table still keys every row to the well it came
    from, so the rows remain joinable even though the pixels are shared.
    """
    src = tmp_path / 'tiles'
    src.mkdir()
    big = _texture(64, 160, seed=23)
    for well, x0 in (('B07', 0), ('C08', 60)):
        np.save(src / f'plate1_{well}_001.npy', big[:32, x0:x0 + 100])
        np.save(src / f'plate1_{well}_002.npy', big[32:, x0:x0 + 100])
    db_path = tmp_path / 'm.db'

    results = align.align_folder(src=str(src), dst=str(tmp_path / 'out'),
                                 grid=(2, 2), overlap=0.25,
                                 group_by_well=False, db_path=str(db_path))

    assert len(results) == 1, 'one canvas for the whole folder'
    result = results[0]
    assert result.n_written == 4
    assert os.path.basename(result.stack_path) == 'tiles_stitched.npy'
    assert np.load(result.stack_path).shape == result.canvas.shape

    frame = align.read_coordinates(db_path)
    assert len(frame) == 4
    assert sorted(set(frame['well'])) == ['B07', 'C08']
    assert set(frame['stack_path']) == {result.stack_path}
    assert sorted(set(frame['rowID'])) == ['r2', 'r3']
