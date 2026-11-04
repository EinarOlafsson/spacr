"""The cell rule: reads are matched to the library first, then vote.

The decode-yield study of 2026-11-04 (item 752) found that the first rule --
reads vote on their RAW called strings, two at least -- assigned a third of
the nuclei on the Funk et al. 2022 plates while 83 % carried a read: a
nucleus with one read can never vote, and two reads of one guide split the
vote whenever either has a wrong base. ``ops_cell_rule='mapped'`` matches
every read to the library exactly and votes among the matched reads.

One synthetic field carries three nuclei, built so each rule's answer is
known: one with a single clean read, one with two clean reads and one with a
wrong base, and one whose two reads are a clean one and one a single base off
its guide.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_engine
from spacr.ops_sbs import BASES
from spacr.ops_store import write_table

SIZE = 160
CENTRES = [(50, 50), (50, 110), (110, 80)]
AREA = math.pi * 36
LIBRARY = ["GTAC", "CATG", "AAGG", "TTCC"]
READS = {
    1: ["GTAC"],                    # one clean read
    2: ["CATG", "CATG", "CATC"],    # two clean, one wrong base
    3: ["AAGG", "AAGT"],            # one clean, one a base off
}


def _reads():
    """``[(y, x, code)]``, each read on its nucleus's rim, 3 px outside it."""
    out = []
    for (cy, cx), (_obj, codes) in zip(CENTRES, sorted(READS.items())):
        for k, code in enumerate(codes):
            angle = 0.4 + k * 2.1
            out.append((int(round(cy + 9 * math.sin(angle))),
                        int(round(cx + 9 * math.cos(angle))), code))
    return out


def _write_field(root, cycles=4):
    """One field, cycle 1 as a five-channel stack and the rest one file per base."""
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
    """Settings and database for a one-field well with the three nuclei stored."""
    _write_field(tmp_path / "raw")
    out = tmp_path / "out"
    out.mkdir()
    db = str(out / "measurements.db")
    write_table(db, "ops_geometry", pd.DataFrame([{
        "plate": "plate", "well": "A1", "cycle": 1, "site": 0, "y": 0.0, "x": 0.0,
        "origin_y": 0.0, "origin_x": 0.0, "tile_height": SIZE, "tile_width": SIZE}]))
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": "plate", "well": "A1", "object_id": [1, 2, 3],
        "centroid_y": [float(y) for y, _ in CENTRES],
        "centroid_x": [float(x) for _, x in CENTRES],
        "area": np.full(3, AREA)}))
    settings = {"genotype_source": str(tmp_path / "raw"), "dst_root": str(out),
                "plate": "plate", "ops_gpu": False, "n_workers": 1}
    return settings, db


def _decode(settings, db, **extra):
    """Decode the well with ``extra`` settings; the report and the table."""
    import sqlite3

    report = ops_engine.run_ops({**settings, **extra}, phases=("decode",),
                                library=LIBRARY)["wells"]["A1"]["decode"]
    with sqlite3.connect(db) as conn:
        table = pd.read_sql_query("SELECT * FROM ops_barcodes ORDER BY object_id",
                                  conn)
    return report, table.set_index("object_id")


def test_the_mapped_rule_assigns_a_single_clean_read_and_ignores_a_wrong_one(well):
    """Exact matches, one read allowed: all three nuclei get their guide."""
    report, table = _decode(*well)
    assert report["cell_rule"] == "mapped" and report["read_mismatches"] == 0
    assert report["min_reads"] == 1
    assert table["mapped_guide"].to_dict() == {1: "GTAC", 2: "CATG", 3: "AAGG"}
    # n_reads counts every read the nucleus owns; the vote is over matched ones.
    assert table.loc[2, "n_reads"] == 3 and table.loc[2, "n_agreeing"] == 2
    assert table.loc[2, "fraction"] == 1.0
    assert table.loc[3, "n_reads"] == 2 and table.loc[3, "n_agreeing"] == 1
    assert report["reads_mapped"] == 4 and report["objects_with_a_mapped_read"] == 3


def test_the_agree_rule_is_the_earlier_one(well):
    """Raw-string votes, two reads at least: only the nucleus whose reads agree."""
    report, table = _decode(*well, ops_cell_rule="agree")
    assert report["cell_rule"] == "agree" and report["min_reads"] == 2
    assert table["mapped_guide"].to_dict() == {2: "CATG"}
    assert report["reads_mapped"] is None


def test_more_reads_or_a_mismatch_change_what_is_counted(well):
    """min_reads 2 drops the one-read nucleus and the one with one clean read;
    a mismatch allowed lets the off-by-one read count for its guide."""
    _r, strict = _decode(*well, ops_min_reads=2)
    assert strict["mapped_guide"].to_dict() == {2: "CATG"}
    report, loose = _decode(*well, ops_min_reads=2, ops_read_mismatches=1)
    assert loose["mapped_guide"].to_dict() == {2: "CATG", 3: "AAGG"}
    assert loose.loc[2, "n_agreeing"] == 3
    assert report["reads_mapped"] == 6


def test_without_a_library_the_rule_falls_back_to_agree(well, capsys):
    """'mapped' has nothing to match against, so it says so and votes raw."""
    settings, _db = well
    rule = ops_engine._cell_rule(settings, [])
    assert rule == ("agree", 0, 2, 0.6)
    assert "needs a guide library" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [{"ops_cell_rule": "plurality"},
                                 {"ops_read_mismatches": -1},
                                 {"ops_min_reads": 0},
                                 {"ops_min_fraction": 0.0},
                                 {"ops_min_fraction": 1.5}])
def test_a_rule_setting_out_of_range_is_refused(bad):
    """Each refusal names the setting, so the run stops before it starts."""
    with pytest.raises(ValueError, match=next(iter(bad))):
        ops_engine._cell_rule(bad, LIBRARY)
