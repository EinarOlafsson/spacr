"""The sequencing engine on a synthetic well, from tile files to joined tables.

A five-field round well (the plus shape `round_well_layout(5)` gives) is cut
from one planted field, with an overlap wider than the stitch's strip, and
written to disk
the way the acquisition names it: cycle 1 as a DAPI-CY3-A594-CY5-CY7 stack,
later cycles one file per base channel. Every nucleus carries three reads
around its rim, one base channel of cycle 2 is displaced as the plate's were,
and one file of the centre field's cycle 3 is truncated.

Cellpose is replaced by a threshold on the composite, because the engine's
contract with the segmenter is "a label image per window", and a model would
make this a test of the model. Everything else is the shipped code.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_engine
from spacr.ops_layout import round_well_layout
from spacr.ops_sbs import BASES

TILE = 384
#: Wider than the 30 % strip the stitch correlates (115 px), so every strip
#: holds only content its neighbour shares. At the real plate's 14 % -- 53 px
#: on a 384 px tile -- half of each strip cannot match, and a five-field tree
#: of edges lost an edge or landed 1-4 px off on most seeds while the real
#: plate's 213 px overlaps register 624 of 624 to 0.2 px. Measured: 120 px
#: gave 4 of 4 edges at 0.0 px on six seeds of six.
OVERLAP = 120
STEP = TILE - OVERLAP
CYCLES = 4
MARGIN = 20
#: One base channel of cycle 2 sits this far from its cycle's first channel.
CHANNEL_OFFSET = {(2, 3): (6, -4)}


class _ThresholdModel:
    """A stand-in for Cellpose: nuclei are the bright discs of the composite."""

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Label the composite's discs.

        The installed cellpose's eval parameters are written out, not taken
        as ``**kwargs``, so an argument cellpose removed fails here too
        (tests/test_cellpose_api_contract.py).

        :param image: one composed window.
        :returns: ``(labels,)``, as Cellpose returns masks first.
        """
        return (ndimage.label(np.asarray(image) > 7000)[0],)


def _plate(root):
    """Write the synthetic well and return what was planted.

    :param root: the acquisition folder.
    :returns: ``(nuclei, codes, owner_of_truncated)`` -- planted centres in
        well pixels relative to the field origin, their barcodes, and the
        site whose cycle-3 A594 file was cut short.
    """
    rng = np.random.default_rng(372)
    layout = round_well_layout(5)
    rows = [row for _column, row in layout.positions()]
    top_row = min(rows)
    height = STEP * (max(rows) - top_row) + TILE + 2 * MARGIN
    width = STEP * (layout.columns - 1) + TILE + 2 * MARGIN
    tops = {site: ((row - top_row) * STEP + MARGIN, column * STEP + MARGIN)
            for site, (column, row) in enumerate(layout.positions())}

    nuclei, codes = [], []
    while len(nuclei) < 40:
        y, x = (float(v) for v in rng.uniform(0, [height, width]))
        inside = any(t + 40 <= y < t + TILE - 40 and l + 40 <= x < l + TILE - 40
                     for t, l in tops.values())
        if not inside or any(math.hypot(y - a, x - b) < 34 for a, b in nuclei):
            continue
        code = "".join(BASES[i] for i in rng.integers(0, 4, CYCLES))
        if len(set(code)) == 1:
            # A barcode whose base never changes has no variance across
            # cycles, which is exactly what the read detector ignores as
            # debris; no guide library carries one.
            continue
        nuclei.append((y, x))
        codes.append(code)

    # The stitcher's own fixture: a dot every 300 px, each peaking at 1,200.
    # Overlapping dots reach about 5,400, under the segmenter's 7,000, so only
    # the planted discs (8,000 above the field) become objects.
    dots = np.zeros((height, width))
    count = (height * width) // 300
    dots[rng.integers(0, height, count), rng.integers(0, width, count)] = 1.0
    dapi = ndimage.gaussian_filter(dots, 3.0) * (1200 * 2 * math.pi * 9.0) + 40.0
    yy, xx = np.mgrid[:height, :width]
    for y, x in nuclei:
        dapi[(yy - y) ** 2 + (xx - x) ** 2 <= 36] += 8000.0
    texture = ndimage.gaussian_filter(rng.random((height, width)), 3)
    texture = (texture - texture.min()) / np.ptp(texture) * 600 + 200

    canvases = {}
    for cycle in range(1, CYCLES + 1):
        for channel in range(4):
            plane = texture + rng.normal(0, 10, texture.shape)
            for (cy, cx), code in zip(nuclei, codes):
                if code[cycle - 1] != BASES[channel]:
                    continue
                for angle in (0.4, 2.5, 4.5):
                    y = int(round(cy + 9 * math.sin(angle)))
                    x = int(round(cx + 9 * math.cos(angle)))
                    plane[y - 1:y + 2, x - 1:x + 2] += 6000.0
            canvases[(cycle, channel)] = np.clip(plane, 0, 65535)

    def cut(canvas, top, left, offset=(0, 0)):
        """One tile of a canvas, its content moved by ``offset``."""
        dy, dx = offset
        return canvas[top - dy:top - dy + TILE,
                      left - dx:left - dx + TILE].astype(np.uint16)

    names = ("CY3", "A594", "CY5", "CY7")
    for site, (top, left) in tops.items():
        folder = root / "c1"
        folder.mkdir(exist_ok=True)
        stack = [np.clip(rng.poisson(cut(dapi, top, left)), 0, 65535).astype(np.uint16)]
        stack += [cut(canvases[(1, k)], top, left) for k in range(4)]
        tifffile.imwrite(folder / f"10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif",
                         np.stack(stack))
        for cycle in range(2, CYCLES + 1):
            folder = root / f"c{cycle}"
            folder.mkdir(exist_ok=True)
            for k, name in enumerate(names):
                tifffile.imwrite(
                    folder / f"10X_c{cycle}_A1_{name}_Site-{site}.tif",
                    cut(canvases[(cycle, k)], top, left,
                        CHANNEL_OFFSET.get((cycle, k), (0, 0))))
    centre = layout.site(*layout.centre)
    cut_short = root / "c3" / f"10X_c3_A1_A594_Site-{centre}.tif"
    data = cut_short.read_bytes()
    cut_short.write_bytes(data[: len(data) // 3])
    return nuclei, codes, centre, tops


@pytest.fixture(scope="module")
def engine_run(tmp_path_factory):
    """Run all three phases once on the synthetic well.

    :param tmp_path_factory: pytest's module-scoped temporary directories.
    :returns: ``(result, nuclei, codes, centre, tops, settings)``.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: _ThresholdModel())
    root = tmp_path_factory.mktemp("ops_engine")
    raw = root / "raw"
    raw.mkdir()
    nuclei, codes, centre, tops = _plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    result = ops_engine.run_ops(settings, library=library)
    yield result, nuclei, codes, centre, tops, settings, library, monkeypatch
    monkeypatch.undo()


def test_the_well_is_stitched_from_its_nuclear_plane(engine_run):
    """Five of five fields, four of four edges, every field where it was cut.

    The placements are compared with the planted ones, not only the residual:
    four edges over five fields form a tree, so an edge measured 2 px short
    still closes with a residual of zero.
    """
    result, tops = engine_run[0], engine_run[4]
    stitch = result["wells"]["A1"]["stitch"]
    assert stitch["placed"] == 5 and stitch["edges_accepted"] == 4
    assert stitch["canvas_agrees"]
    with sqlite3.connect(result["db"]) as conn:
        placed = {site: (y, x) for site, y, x in
                  conn.execute("SELECT site, y, x FROM ops_geometry")}
    for site, (top, left) in tops.items():
        y, x = placed[site]
        assert abs(y - (top - MARGIN)) <= 1 and abs(x - (left - MARGIN)) <= 1, (
            site, (y, x), (top - MARGIN, left - MARGIN))


def test_every_planted_nucleus_is_one_object(engine_run):
    """Sewn across 256 px windows, numbered once, stored with a unique key."""
    result, nuclei = engine_run[0], engine_run[1]
    objects = result["wells"]["A1"]["objects"]
    assert objects["objects"] == len(nuclei)
    assert objects["objects_ready"][0] is True
    with sqlite3.connect(result["db"]) as conn:
        rows = conn.execute("SELECT centroid_y, centroid_x FROM ops_objects").fetchall()
    found = np.array(rows)
    for y, x in nuclei:
        # The well frame starts at the top-left tile, MARGIN px into the plate.
        away = np.hypot(found[:, 0] - (y - MARGIN), found[:, 1] - (x - MARGIN))
        # A radius-6 disc, composed from Poisson tiles pasted at rounded
        # positions, keeps its centroid within a couple of pixels.
        assert away.min() < 2.5, (y, x, away.min())


def test_every_nucleus_gets_its_barcode_and_the_lost_cycle_is_an_N(engine_run):
    """Reads around each rim vote; the centre field's objects read N at cycle 3."""
    result, nuclei, codes, centre, tops = engine_run[:5]
    decode = result["wells"]["A1"]["decode"]
    assert decode["fields_decoded"] == 5
    assert decode["cycles_missing"] == {str(centre): [3]}
    assert any(reason.startswith("ValueError") for _path, reason in decode["unreadable"])
    assert decode["containment"] > 0.95 and not decode["containment_below_gate"]

    with sqlite3.connect(result["db"]) as conn:
        barcodes = conn.execute(
            "SELECT o.centroid_y, o.centroid_x, b.barcode, b.n_cycles, b.mapped_guide "
            "FROM ops_barcodes b JOIN ops_objects o ON b.object_id = o.object_id "
            "AND b.plate = o.plate AND b.well = o.well").fetchall()
        orphans = conn.execute(
            "SELECT COUNT(*) FROM ops_barcodes b LEFT JOIN ops_objects o "
            "ON b.object_id = o.object_id WHERE o.object_id IS NULL").fetchone()[0]
    assert orphans == 0
    assert len(barcodes) == len(nuclei)
    lost = 0
    for y, x, barcode, n_cycles, mapped in barcodes:
        index = int(np.argmin([math.hypot(y - (a - MARGIN), x - (b - MARGIN))
                               for a, b in nuclei]))
        want = codes[index]
        if n_cycles == CYCLES - 1:
            lost += 1
            assert barcode == want[:2] + "N" + want[3:]
        else:
            assert barcode == want
        assert mapped == want or (n_cycles == CYCLES - 1 and mapped == "")
    assert lost > 0, "no object was owned by the field that lost a file"


def test_decoding_again_in_worker_processes_gives_the_same_table(engine_run):
    """The decode phase alone, in two spawned workers, replaces the well's rows."""
    import pandas as pd

    result, settings, library = engine_run[0], engine_run[5], engine_run[6]
    with sqlite3.connect(result["db"]) as conn:
        before = pd.read_sql_query("SELECT * FROM ops_barcodes ORDER BY object_id", conn)
    again = ops_engine.run_ops({**settings, "n_workers": 2}, wells=["a1"],
                               phases=("decode",), library=library)
    with sqlite3.connect(again["db"]) as conn:
        after = pd.read_sql_query("SELECT * FROM ops_barcodes ORDER BY object_id", conn)
    assert again["wells"]["A1"]["decode"]["workers"] == 2
    pd.testing.assert_frame_equal(before, after)


def test_what_cannot_run_says_why(tmp_path, engine_run):
    """A missing source, an unknown phase, an unstitched well."""
    settings = engine_run[5]
    with pytest.raises(ValueError, match="genotype_source"):
        ops_engine.run_ops({"genotype_source": str(tmp_path / "nowhere")})
    with pytest.raises(ValueError, match="unknown phases"):
        ops_engine.run_ops(settings, phases=("stich",))
    with pytest.raises(ValueError, match="no tiles for wells"):
        ops_engine.run_ops(settings, wells=["H12"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="named like"):
        ops_engine.run_ops({"genotype_source": str(empty)})
    fresh = {**settings, "dst_root": str(tmp_path / "fresh")}
    with pytest.raises(ValueError, match="stitch phase first"):
        ops_engine.run_ops(fresh, phases=("objects",))
