"""372 PART 14-L's file failures, planted on disk, through the engine.

The plate on the NAS carried sixteen truncated sequencing files. tifffile
raises ValueError on a short read, not OSError, so PART 14-D's "catch
OSError" rule let the first 105-field run die at field 51; and the driver
that survived it decoded a field only when all eleven cycles were there, so
the two fields with one truncated file got no calls at all. PART 9 hole 1
says a lost cycle must not cost the well.

Every test here writes real TIFF files named as the acquisition names them.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

from spacr import ops_engine
from spacr.ops_sbs import BASES, assign_reads_to_objects, correct_to_library


def _truncate(path, keep=0.3):
    """Cut a file short, as an interrupted download leaves it.

    :param path: the file.
    :param keep: the fraction of its bytes to keep.
    """
    data = open(path, "rb").read()
    with open(path, "wb") as handle:
        handle.write(data[: int(len(data) * keep)])


def test_a_truncated_tiff_raises_value_error_which_an_oserror_rule_misses(tmp_path):
    """The measured exception, so the handler's breadth is not a guess."""
    path = tmp_path / "10X_c8_A1_CY5_Site-172.tif"
    tifffile.imwrite(path, np.ones((64, 64), np.uint16))
    _truncate(path)
    with pytest.raises(ValueError) as caught:
        tifffile.imread(path)
    assert not isinstance(caught.value, OSError)


def test_an_unreadable_file_is_an_outcome_with_its_reason(tmp_path):
    """Truncated, a directory, a missing plane: None each time, and why."""
    good = tmp_path / "good.tif"
    tifffile.imwrite(good, np.full((2, 16, 16), 7, np.uint16))
    short = tmp_path / "short.tif"
    tifffile.imwrite(short, np.ones((256, 256), np.uint16))
    _truncate(short)
    folder = tmp_path / "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-3.tif"
    folder.mkdir()

    unreadable = []
    assert ops_engine._read_plane((str(good), 1), unreadable).shape == (16, 16)
    assert ops_engine._read_plane((str(short), None), unreadable) is None
    assert ops_engine._read_plane((str(folder), 0), unreadable) is None
    assert ops_engine._read_plane((str(good), 5), unreadable) is None
    reasons = [reason for _path, reason in unreadable]
    assert reasons[0].startswith("ValueError")
    assert reasons[1] == "not a file"
    assert reasons[2] == "has no plane 5"


def test_a_zero_byte_tiff_raises_an_error_that_is_not_value_error(tmp_path):
    """372, 2026-09-26: well B3 died on c4/10X_c4_B3_CY3_Site-59.tif, 0 bytes.

    tifffile's TiffFileError derives from Exception alone, so the ValueError
    rule that covers a truncated file does not cover an empty one.
    """
    path = tmp_path / "10X_c4_B3_CY3_Site-59.tif"
    path.write_bytes(b"")
    with pytest.raises(Exception) as caught:
        tifffile.TiffFile(str(path))
    assert "not a TIFF file" in str(caught.value)
    assert not isinstance(caught.value, (ValueError, OSError, IndexError))


def test_a_zero_byte_or_non_tiff_file_is_unreadable_with_its_reason(tmp_path):
    """Empty and not-a-TIFF are both "no image here", never a raised error."""
    empty = tmp_path / "10X_c4_B3_CY3_Site-59.tif"
    empty.write_bytes(b"")
    text = tmp_path / "10X_c4_B3_CY5_Site-59.tif"
    text.write_bytes(b"not an image at all")

    unreadable = []
    assert ops_engine._read_plane((str(empty), None), unreadable) is None
    assert ops_engine._read_plane((str(text), 0), unreadable) is None
    assert [path for path, _reason in unreadable] == [str(empty), str(text)]
    assert all(reason.startswith("TiffFileError: not a TIFF file")
               for _path, reason in unreadable)


def test_both_file_layouts_are_indexed_by_channel(tmp_path):
    """Cycle 1 is one five-plane stack; later cycles are one file per base."""
    (tmp_path / "c1").mkdir()
    (tmp_path / "c2").mkdir()
    tifffile.imwrite(tmp_path / "c1" / "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-4.tif",
                     np.zeros((5, 8, 8), np.uint16))
    for channel in ("CY3", "A594", "CY5", "CY7"):
        tifffile.imwrite(tmp_path / "c2" / f"10X_c2_A1_{channel}_Site-4.tif",
                         np.zeros((8, 8), np.uint16))
    (tmp_path / "c2" / "10X_c2_A1_CY5_Site-9.tif").mkdir()   # a folder, not a tile

    index = ops_engine._index_tiles(str(tmp_path))
    assert sorted(index) == ["A1"] and sorted(index["A1"]) == [1, 2]
    assert sorted(index["A1"][2]) == [4]
    stack = ops_engine._plane_sources(index["A1"][1][4])
    assert stack["DAPI"][1] == 0 and stack["CY7"][1] == 4
    single = ops_engine._plane_sources(index["A1"][2][4])
    assert single["A594"][1] is None


def test_a_tile_no_edge_reached_is_not_placed_at_the_origin():
    """solve_placements pins a lone tile at (0, 0); the engine drops it."""
    class Edge:
        """A registration that was or was not accepted."""

        def __init__(self, accepted):
            """:param accepted: the verdict."""
            self.accepted = accepted

    edges = {(0, 1): Edge(True), (1, 2): Edge(True), (2, 3): Edge(False),
             (3, 4): Edge(True)}
    assert ops_engine._largest_component(edges, [0, 1, 2, 3, 4, 5]) == [0, 1, 2]


def _write_field(root, site, cycles, reads, size=200, seed=0):
    """Write one field's cycles as the acquisition names them.

    :param root: the acquisition folder.
    :param site: the site number.
    :param cycles: how many cycles.
    :param reads: ``[(y, x, barcode)]``.
    :param size: the field edge.
    :param seed: the noise seed.
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    # The cell background every channel of every cycle shares: it is what
    # the within-cycle and cycle-to-cycle registrations lock onto, and a
    # field of pure noise and spots is rightly refused by both.
    texture = ndimage.gaussian_filter(rng.random((size, size)), 6)
    texture = (texture - texture.min()) / np.ptp(texture) * 600 + 200
    planes = {}
    for cycle in range(1, cycles + 1):
        per = []
        for channel in range(4):
            plane = texture + rng.normal(0, 10, (size, size))
            for y, x, code in reads:
                if code[cycle - 1] == BASES[channel]:
                    plane[y - 1:y + 2, x - 1:x + 2] += 6000
            per.append(np.clip(plane, 0, 65535).astype(np.uint16))
        planes[cycle] = per
    folder = os.path.join(root, "c1")
    os.makedirs(folder, exist_ok=True)
    dapi = np.full((size, size), 500, np.uint16)
    tifffile.imwrite(os.path.join(folder, f"10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif"),
                     np.stack([dapi] + planes[1]))
    for cycle in range(2, cycles + 1):
        folder = os.path.join(root, f"c{cycle}")
        os.makedirs(folder, exist_ok=True)
        for name, plane in zip(("CY3", "A594", "CY5", "CY7"), planes[cycle]):
            tifffile.imwrite(os.path.join(folder, f"10X_c{cycle}_A1_{name}_Site-{site}.tif"), plane)


@pytest.mark.parametrize("keep, reason", [(0.3, "ValueError"),
                                           (0.0, "TiffFileError")],
                         ids=["truncated", "zero-byte"])
def test_one_truncated_cycle_file_costs_that_cycle_and_not_the_field(
        tmp_path, keep, reason):
    """Three nuclei, three reads each, cycle 3's A594 file cut short or empty.

    The field still decodes: the kept cycles are 1, 2 and 4, every read
    carries N for cycle 3, every nucleus gets its barcode from the reads
    around its rim, and the library maps each one back through the N. The
    zero-byte case is B3's Site-59 from the 2026-09-25 plate run.
    """
    codes = ["GTAC", "CATG", "AAGG"]
    centres = [(60, 60), (60, 130), (140, 90)]
    reads = []
    for (cy, cx), code in zip(centres, codes):
        for angle in (0.3, 2.4, 4.4):
            reads.append((int(round(cy + 9 * math.sin(angle))),
                          int(round(cx + 9 * math.cos(angle))), code))
    _write_field(str(tmp_path), 7, 4, reads)
    _truncate(tmp_path / "c3" / "10X_c3_A1_A594_Site-7.tif", keep=keep)

    index = ops_engine._index_tiles(str(tmp_path))["A1"]
    cycles = sorted(index)
    task = {
        "site": 7, "cycles": cycles, "reference": 1, "gpu": False,
        "planes": {cycle: [ops_engine._plane_sources(index[cycle][7]).get(name)
                           for name in ops_engine._BASE_CHANNELS]
                   for cycle in cycles},
        "centroids": np.array(centres, float), "areas": np.full(3, math.pi * 36),
        "ids": np.array([11, 12, 13]), "owned": np.array([True, True, True]),
    }
    ops_engine._init_decode_worker(frozenset(codes))
    result = ops_engine._decode_field(task)

    assert "skipped" not in result
    assert result["kept"] == [1, 2, 4] and result["missing"] == [3]
    assert result["n_cycles"] == 3
    assert result["unreadable"] and result["unreadable"][0][1].startswith(reason)
    assert result["calls"] and all(code[2] == "N" for code in result["calls"])

    got = assign_reads_to_objects(result["ids"], result["calls"],
                                  quality=result["quality"])
    assert {k: v["barcode"] for k, v in got.items()} == {
        11: "GTNC", 12: "CANG", 13: "AANG"}
    assert correct_to_library([got[k]["barcode"] for k in (11, 12, 13)], codes,
                              max_distance=0) == codes
