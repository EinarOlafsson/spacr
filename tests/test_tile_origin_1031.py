"""Item 1031: one tile origin for every well <-> tile conversion.

``compose_window`` pastes each sequencing tile at the whole pixel nearest its
stored (fractional) position, so the nuclei segmented on the composite, and
the tile-affine transforms fitted to them, live in that rounded frame. Before
item 1031 the decode moved the nuclei into a tile by the stored fraction
while the reads came from the raw tile, leaving the two up to half a pixel
apart on each axis. ``ops_tile_origin`` 'paste' (schema 971) uses the paste
origin everywhere; 'float' replays the old offset for files stamped before.
"""
from __future__ import annotations

import ast
import inspect
import math
import textwrap

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_compose, ops_decode, ops_engine, ops_live
from spacr.ops_compose import (TILE_ORIGINS, Window, compose_window,
                               paste_origin, tile_centres, tile_origin_rule,
                               tile_origins, tile_to_well, well_to_tile)
from spacr.ops_sbs import BASES
from spacr.ops_store import write_table

SIZE = 160
#: The tile's stored position: fractions on both axes, one of them a tie.
PLACED = (10.4, 20.5)
PASTED = (10, 20)
#: Nucleus centres in the tile's own pixels, where its reads are drawn.
CENTRES = [(50, 50), (50, 110), (110, 80)]
AREA = math.pi * 36
LIBRARY = ["GTAC", "CATG", "AAGG", "TTCC"]
READS = {1: ["GTAC"], 2: ["CATG", "CATG"], 3: ["AAGG", "AAGG"]}


def test_paste_origin_is_the_rounding_compose_window_used():
    """Python's round and numpy.rint agree, ties to even, scalars and arrays."""
    values = [0.0, 0.4999, 0.5, 1.5, 2.5, -0.5, -1.5, 10.4, 20.5, 333.50001,
              -7.49, 1e6 + 0.5]
    for top in values:
        for left in values:
            assert paste_origin(top, left) == (int(round(float(top))),
                                               int(round(float(left))))
            assert all(type(v) is int for v in paste_origin(top, left))
    rows, cols = paste_origin(np.array(values), np.array(values[::-1]))
    assert rows.tolist() == [int(round(v)) for v in values]
    assert cols.tolist() == [int(round(v)) for v in values[::-1]]


def test_the_rule_is_checked_and_defaults_to_paste():
    assert TILE_ORIGINS[0] == "paste"
    assert tile_origin_rule(None) == "paste"
    assert tile_origin_rule(" Float ") == "float"
    assert tile_origin_rule({"ops_tile_origin": "float"}) == "float"
    with pytest.raises(ValueError, match="ops_tile_origin"):
        tile_origin_rule("nearest")
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION
    assert OPS_DEFAULTS["ops_tile_origin"] == "paste"
    assert DEFAULT_CHANGES[971] == {"ops_tile_origin": "float"}
    assert SETTINGS_SCHEMA_VERSION >= 971


def test_the_conversions_round_trip_and_float_is_the_stored_fraction():
    points = np.array([[3.25, 7.0], [100.0, 0.5]])
    for rule in TILE_ORIGINS:
        back = tile_to_well(well_to_tile(points, PLACED, rule), PLACED, rule)
        assert np.array_equal(back, points)
    assert np.array_equal(well_to_tile(points, PLACED, "float"),
                          np.column_stack([points[:, 0] - PLACED[0],
                                           points[:, 1] - PLACED[1]]))
    assert np.array_equal(well_to_tile(points, PLACED, "paste"),
                          points - np.array(PASTED, float))
    assert tile_origins({7: PLACED}, "paste") == {7: (10.0, 20.0)}
    assert tile_origins({7: PLACED}, "float") == {7: PLACED}
    peaks = np.array([[5, 6]], np.int32)
    assert tile_to_well(peaks, PLACED, "float").dtype == np.float64
    assert well_to_tile(np.zeros((1, 2), np.float32), PLACED, "float").dtype == np.float32


def _marked_tile(points):
    """A tile with one bright 3 x 3 square on each point of its own pixels."""
    tile = np.zeros((SIZE, SIZE), np.float32)
    for y, x in points:
        tile[y - 1:y + 2, x - 1:x + 2] = 1000.0
    return tile


def test_nuclei_found_on_a_composite_line_up_with_the_raw_tile_only_under_paste():
    """Two tiles at fractional origins; marks found on the composite are moved
    back into each raw tile. 'paste' lands on the raw pixel exactly; 'float'
    misses by the rounding fraction, up to half a pixel on each axis."""
    placements = {0: (0.3, 0.0), 1: (2.6, SIZE - 30.5)}
    marks = {0: [(40, 40), (120, 60)], 1: [(40, 80), (120, 140)]}
    tiles = {site: _marked_tile(points) for site, points in marks.items()}
    image, _count = compose_window(Window(0, 0, SIZE + 4, 2 * SIZE),
                                   placements, tiles.__getitem__,
                                   blend="average")
    labels, n = ndimage.label(image > 500)
    assert n == 4
    found = np.array(ndimage.center_of_mass(image, labels, range(1, n + 1)))
    centres = tile_centres([placements[s] for s in (0, 1)], (SIZE, SIZE), "paste")
    from scipy.spatial import cKDTree
    _, owner = cKDTree(centres).query(found)
    for rule in TILE_ORIGINS:
        for point, site in zip(found, owner):
            local = well_to_tile(point[None], placements[site], rule)[0]
            nearest = min(marks[site], key=lambda m: np.hypot(*(np.array(m) - local)))
            offset = local - np.array(nearest, float)
            frac = np.array(paste_origin(*placements[site]), float) - np.array(placements[site])
            if rule == "paste":
                assert np.allclose(offset, 0.0, atol=1e-9)
            else:
                assert np.allclose(offset, frac, atol=1e-9)
    assert any(abs(v) > 0.25 for site in placements
               for v in np.array(paste_origin(*placements[site])) - np.array(placements[site]))


def test_tile_centres_agree_with_the_composite():
    """The merge's tile centres sit on the composite pixel the tile centre was
    pasted to under 'paste', and on the stored fraction under 'float'."""
    placements = {0: (0.4, 0.6), 1: (5.5, SIZE - 20.5)}
    mark = (SIZE // 2, SIZE // 2)
    tiles = {site: _marked_tile([mark]) for site in placements}
    image, _ = compose_window(Window(0, 0, SIZE + 8, 2 * SIZE), placements,
                              tiles.__getitem__, blend="average")
    labels, n = ndimage.label(image > 500)
    found = np.array(ndimage.center_of_mass(image, labels, range(1, n + 1)))
    found = found[np.argsort(found[:, 1])]
    paste = tile_centres([placements[0], placements[1]], (SIZE, SIZE), "paste")
    assert np.allclose(paste, found, atol=1e-9)
    stored = tile_centres([placements[0], placements[1]], (SIZE, SIZE), "float")
    assert np.allclose(stored, np.array([placements[0], placements[1]]) + SIZE / 2)
    assert not np.allclose(stored, found, atol=0.05)


def _write_geometry(db, placed=PLACED, cycle_shape=(SIZE, SIZE)):
    write_table(db, "ops_geometry", pd.DataFrame([{
        "plate": "plate", "well": "A1", "cycle": 1, "site": 0,
        "y": placed[0], "x": placed[1], "origin_y": 0.0, "origin_x": 0.0,
        "tile_height": cycle_shape[0], "tile_width": cycle_shape[1]}]))


def test_the_merge_tile_centres_follow_the_setting(tmp_path):
    db = str(tmp_path / "measurements.db")
    _write_geometry(db)
    paste, sites = ops_engine._ph_tile_centres(db, "plate", "A1", "paste")
    stored, _ = ops_engine._ph_tile_centres(db, "plate", "A1", "float")
    assert sites.tolist() == [0]
    assert np.array_equal(paste, [[PASTED[0] + SIZE / 2, PASTED[1] + SIZE / 2]])
    assert np.array_equal(stored, [[PLACED[0] + SIZE / 2.0, PLACED[1] + SIZE / 2.0]])
    target = np.array([[50.0, 60.0]])
    stage = {0: np.array([1000.0, 2000.0])}
    moved = ops_engine._merge_sbs_mosaic(db, "plate", "A1", target, stage, "paste")
    assert np.array_equal(moved, target - np.array(PASTED, float) + stage[0])
    moved = ops_engine._merge_sbs_mosaic(db, "plate", "A1", target, stage, "float")
    assert np.array_equal(moved, target + (stage[0] - np.array(PLACED)))


def _reads():
    out = []
    for (cy, cx), (_obj, codes) in zip(CENTRES, sorted(READS.items())):
        for k, code in enumerate(codes):
            angle = 0.4 + k * 2.1
            out.append((int(round(cy + 9 * math.sin(angle))),
                        int(round(cx + 9 * math.cos(angle))), code))
    return out


def _write_field(root, cycles=4):
    rng = np.random.default_rng(0)
    texture = ndimage.gaussian_filter(rng.random((SIZE, SIZE)), 6)
    texture = (texture - texture.min()) / np.ptp(texture) * 600 + 200
    for cycle in range(1, cycles + 1):
        planes = []
        for channel in range(4):
            plane = texture + rng.normal(0, 10, texture.shape)
            for y, x, code in _reads():
                if code[cycle - 1] == BASES[channel]:
                    plane[y - 1:y + 2, x - 1:x + 2] += 6000
            planes.append(np.clip(plane, 0, 65535).astype(np.uint16))
        folder = root / f"c{cycle}"
        folder.mkdir(parents=True, exist_ok=True)
        if cycle == 1:
            dapi = np.full((SIZE, SIZE), 500, np.uint16)
            tifffile.imwrite(folder / "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-0.tif",
                             np.stack([dapi] + planes))
            continue
        for name, plane in zip(ops_engine._BASE_CHANNELS, planes):
            tifffile.imwrite(folder / f"10X_c{cycle}_A1_{name}_Site-0.tif", plane)


@pytest.fixture()
def well(tmp_path):
    """One field stored at a fractional origin; its nuclei in the composite
    frame (the raw centre plus the paste origin), as the objects phase stores
    them."""
    _write_field(tmp_path / "raw")
    out = tmp_path / "out"
    out.mkdir()
    db = str(out / "measurements.db")
    _write_geometry(db)
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": "plate", "well": "A1", "object_id": [1, 2, 3],
        "centroid_y": [float(y + PASTED[0]) for y, _ in CENTRES],
        "centroid_x": [float(x + PASTED[1]) for _, x in CENTRES],
        "area": np.full(3, AREA)}))
    settings = {"genotype_source": str(tmp_path / "raw"), "dst_root": str(out),
                "plate": "plate", "ops_gpu": False, "n_workers": 1,
                "ops_reuse_unprovenanced_objects": True,
                "ops_decode_threshold_mode": "manual", "ops_decode_preset": "none",
                "ops_store_reads": True}
    return settings, db


def _decode(settings, db, monkeypatch, **extra):
    """Decode with ``extra``; the tasks' centroids, the reads and the calls."""
    from tests.ops_store_sql import ops_connect, read_sql

    seen = []
    original = ops_engine._decode_field

    def spy(task):
        seen.append(np.array(task["centroids"], copy=True))
        return original(task)

    monkeypatch.setattr(ops_engine, "_decode_field", spy)
    ops_engine.run_ops({**settings, **extra}, phases=("decode",), library=LIBRARY)
    with ops_connect(db) as conn:
        calls = read_sql("SELECT * FROM ops_barcodes ORDER BY object_id", conn)
        reads = read_sql("SELECT * FROM ops_reads", conn)
    return seen[0], reads, calls


def test_the_decode_puts_nuclei_on_their_reads_under_paste(well, monkeypatch):
    settings, db = well
    centroids, reads, calls = _decode(settings, db, monkeypatch)
    assert np.array_equal(centroids, np.array(CENTRES, float))
    assert len(calls) == 3
    raw = {(y, x) for y, x, _ in _reads()}
    lifted = {(int(round(y - PASTED[0])), int(round(x - PASTED[1])))
              for y, x in reads[["y", "x"]].drop_duplicates().to_numpy()}
    assert lifted <= {(y + dy, x + dx) for y, x in raw
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    assert len(reads)
    assert np.array_equal(reads["y"].to_numpy(), np.round(reads["y"].to_numpy()))
    assert np.array_equal(reads["x"].to_numpy(), np.round(reads["x"].to_numpy()))


def test_float_replays_the_old_offset(well, monkeypatch):
    """'float' moves nuclei by the stored fraction exactly as the decode did
    before item 1031 (``oy - top``), and lifts reads by it (``peak + top``)."""
    settings, db = well
    centroids, reads, _calls = _decode(settings, db, monkeypatch,
                                       ops_tile_origin="float")
    oy = np.array([y + PASTED[0] for y, _ in CENTRES], float)
    ox = np.array([x + PASTED[1] for _, x in CENTRES], float)
    assert np.array_equal(centroids, np.column_stack([oy - PLACED[0], ox - PLACED[1]]))
    assert not np.array_equal(centroids, np.array(CENTRES, float))
    for column, origin in (("y", PLACED[0]), ("x", PLACED[1])):
        local = reads[column].to_numpy(np.float64) - origin
        assert len(local) and np.allclose(local, np.round(local), atol=1e-3)
        assert not np.allclose(reads[column].to_numpy(np.float64),
                               np.round(reads[column].to_numpy(np.float64)),
                               atol=1e-3) or origin == round(origin)


def test_a_file_stamped_before_971_gets_float():
    from spacr.settings import SETTINGS_SCHEMA_VERSION, _fold_older_defaults

    known = {"plate", "ops_tile_origin"}
    for version in (0, 965, 970):
        old = {"plate": "P1"}
        applied = _fold_older_defaults(old, version, known=known)
        assert applied["ops_tile_origin"] == ("float", 971)
        assert old["ops_tile_origin"] == "float"
    current = {"plate": "P1"}
    assert _fold_older_defaults(current, SETTINGS_SCHEMA_VERSION, known=known) == {}
    assert "ops_tile_origin" not in current
    chosen = {"plate": "P1", "ops_tile_origin": "paste"}
    _fold_older_defaults(chosen, 965, known=known)
    assert chosen["ops_tile_origin"] == "paste"


#: Every function that converts between the well and a tile's own pixels.
CONVERTERS = [
    (ops_compose, "compose_window"),
    (ops_engine, "_touches"),
    (ops_engine, "_objects_window_key"),
    (ops_engine, "_phenotype"),
    (ops_engine, "_reads_frame"),
    (ops_engine, "_decode"),
    (ops_engine, "_ph_tile_centres"),
    (ops_engine, "_merge_sbs_mosaic"),
    (ops_decode, "compare_region"),
    (ops_decode, "run_compare_calls"),
    (ops_live, "_run_align"),
]
HELPERS = {"paste_origin", "tile_origins", "tile_centres", "well_to_tile",
           "tile_to_well"}


@pytest.mark.parametrize("module,name", CONVERTERS,
                         ids=[f"{m.__name__}.{n}" for m, n in CONVERTERS])
def test_every_converter_goes_through_the_one_helper(module, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(module, name))))
    called = {(_n.func.id if isinstance(_n.func, ast.Name) else
               getattr(_n.func, "attr", None))
              for _n in ast.walk(tree) if isinstance(_n, ast.Call)}
    assert called & HELPERS, f"{module.__name__}.{name} converts without the helper"


def test_no_tile_origin_is_rounded_outside_the_owner():
    """The one-implementation ratchet sees no ``int(round(top))`` outside
    spacr/ops_compose.py, and its detector catches a new one."""
    from tests.test_one_implementation_per_operation import count_sites

    counts = count_sites()
    assert not any(rel.startswith("spacr/ops_") for rel in counts["tile_origin"])


def test_the_paste_round_detector_catches_a_copy(tmp_path):
    from tests.test_one_implementation_per_operation import count_sites

    pkg = tmp_path / "spacr"
    pkg.mkdir()
    (pkg / "ops_compose.py").write_text(
        "def paste_origin(top, left):\n    return int(round(float(top))), int(round(left))\n",
        encoding="utf-8")
    (pkg / "copy.py").write_text(
        "def place(top, left, size):\n"
        "    a = int(round(float(top)))\n"
        "    b = int(round(placement_left[0]))\n"
        "    return a, b, int(round(size))\n", encoding="utf-8")
    assert count_sites(tmp_path)["tile_origin"] == {"spacr/copy.py": 2}
