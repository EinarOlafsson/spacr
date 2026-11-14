"""The extraction preview: planes a container or folder would expand to, and the summary line.

The planners run over real folders of tiny TIFFs so the rows are the ones the
extraction would write, and over a described container so the Yokogawa names
follow ``convert_to_yokogawa``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from spacr.qt import folder_metadata as fm
from spacr.qt import ingest_preview as ip


@dataclass
class _Described:
    """What :mod:`spacr.qt.multi_format` reports for a container."""
    path: str
    n_fields: int = 1
    n_channels: int = 1
    n_timepoints: int = 1


def _tiny_tif(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))


def _row(original, plate, well, field, channel, time, canonical):
    return {"original": original, "plate": plate, "well": well,
            "field": field, "channel": channel, "time": time,
            "canonical": canonical}


def test_plan_container_extraction_enumerates_time_field_channel(subtests):
    """One row per (time, field, channel), canonical names in Yokogawa form."""
    with subtests.test(msg="2 fields x 3 channels x 4 timepoints"):
        rows = ip.plan_container_extraction(
            _Described("/data/movie.nd2", n_fields=2, n_channels=3,
                       n_timepoints=4))
        assert len(rows) == 24
        assert rows[0] == _row("/data/movie.nd2", "plate1", "plate1_A01",
                               1, 1, 1, "plate1_A01_T0001F001L01C01.tif")
        assert rows[1]["canonical"] == "plate1_A01_T0001F001L01C02.tif"
        assert rows[3]["canonical"] == "plate1_A01_T0001F002L01C01.tif"
        assert rows[6]["canonical"] == "plate1_A01_T0002F001L01C01.tif"
        assert rows[-1] == _row("/data/movie.nd2", "plate1", "plate1_A01",
                                2, 3, 4, "plate1_A01_T0004F002L01C03.tif")
        assert all(tuple(r) == ip.ROW_COLUMNS for r in rows)
        assert {(r["time"], r["field"], r["channel"]) for r in rows} == {
            (t, f, c) for t in range(1, 5) for f in (1, 2) for c in (1, 2, 3)}

    cases = [
        ("a header reporting zeros still yields one plane",
         _Described("/data/x.czi", 0, 0, 0), {},
         [_row("/data/x.czi", "plate1", "plate1_A01", 1, 1, 1,
               "plate1_A01_T0001F001L01C01.tif")]),
        ("a description with no counts at all is one plane",
         type("Bare", (), {"path": "/data/bare.tif"})(), {},
         [_row("/data/bare.tif", "plate1", "plate1_A01", 1, 1, 1,
               "plate1_A01_T0001F001L01C01.tif")]),
        ("the caller's plate and well reach the name",
         _Described("/data/x.lif"), {"plate": "plate7", "well": "B11"},
         [_row("/data/x.lif", "plate7", "plate7_B11", 1, 1, 1,
               "plate7_B11_T0001F001L01C01.tif")]),
    ]
    for case_id, desc, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert ip.plan_container_extraction(desc, **kwargs) == expected


def test_plan_folder_extraction_plans_one_row_per_image(subtests, tmp_path,
                                                         monkeypatch):
    """Missing roots and empty folders plan nothing; caps keep the smallest paths."""
    with subtests.test(msg="a path that is not a folder"):
        assert ip.plan_folder_extraction(tmp_path / "absent") == []

    with subtests.test(msg="a folder with no images"):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "notes.txt").write_text("nothing here", encoding="utf-8")
        assert ip.plan_folder_extraction(notes) == []

    wells = tmp_path / "wells"
    for well in ("A01", "A02"):
        for field in (1, 2):
            _tiny_tif(wells / well / f"f{field}.tif")

    with subtests.test(msg="the wells the tree names reach the rows"):
        rows = ip.plan_folder_extraction(wells, plate="plate3")
        assert rows == [
            _row(str(wells / "A01" / "f1.tif"), "plate3", "A01", 1, 1, 1,
                 "plate3_A01_F001_C01.tif"),
            _row(str(wells / "A01" / "f2.tif"), "plate3", "A01", 2, 1, 1,
                 "plate3_A01_F002_C01.tif"),
            _row(str(wells / "A02" / "f1.tif"), "plate3", "A02", 3, 1, 1,
                 "plate3_A02_F003_C01.tif"),
            _row(str(wells / "A02" / "f2.tif"), "plate3", "A02", 4, 1, 1,
                 "plate3_A02_F004_C01.tif"),
        ]

    with subtests.test(msg="template=None means detection found nothing"):
        def _must_not_run(_root):
            raise AssertionError("the tree was walked a second time")

        with monkeypatch.context() as m:
            m.setattr(fm, "detect_folder_metadata", _must_not_run)
            rows = ip.plan_folder_extraction(wells, template=None)
        assert [(r["well"], r["field"], r["canonical"]) for r in rows] == [
            ("A01", 1, "plate1_A01_F001_C01.tif"),
            ("A01", 2, "plate1_A01_F002_C01.tif"),
            ("A01", 3, "plate1_A01_F003_C01.tif"),
            ("A01", 4, "plate1_A01_F004_C01.tif")]

    ten = tmp_path / "ten"
    for i in reversed(range(10)):
        _tiny_tif(ten / "A01" / f"f{i:02d}.tif")

    for case_id, limit, names in [
            ("a cap keeps the smallest paths", 3,
             ["f00.tif", "f01.tif", "f02.tif"]),
            ("no cap plans the whole folder in order", None,
             [f"f{i:02d}.tif" for i in range(10)]),
            ("the default cap is above ten rows", ip._PREVIEW_ROWS,
             [f"f{i:02d}.tif" for i in range(10)])]:
        with subtests.test(msg=case_id):
            rows = ip.plan_folder_extraction(ten, limit=limit)
            assert [Path(r["original"]).name for r in rows] == names
            assert [r["field"] for r in rows] == list(range(1, len(names) + 1))

    with subtests.test(msg="files a caller already walked are used as given"):
        paths = [ten / "A01" / f"f{i:02d}.tif" for i in (2, 0, 1)]
        rows = ip.plan_folder_extraction(tmp_path / "absent",
                                         files=iter(paths), template=None)
        assert [Path(r["original"]).name for r in rows] == [
            "f00.tif", "f01.tif", "f02.tif"]

    with subtests.test(msg="an empty file list plans nothing"):
        assert ip.plan_folder_extraction(tmp_path / "absent", files=[],
                                         template=None) == []


def test_summarize_rows_counts_wells_fields_channels_and_timepoints(subtests):
    """Timepoints are only mentioned when there is more than one."""
    cases = [
        ("an empty preview", [], "no images to extract"),
        ("a single timepoint",
         ip.plan_container_extraction(_Described("/data/x.nd2", 2, 2)),
         "4 images, 1 well(s), 2 field(s), 2 channel(s)"),
        ("a time series",
         ip.plan_container_extraction(_Described("/data/x.nd2", 2, 2, 5)),
         "20 images, 1 well(s), 2 field(s), 2 channel(s), 5 timepoint(s)"),
        ("rows from two wells",
         [{"well": "A01", "field": 1, "channel": 1, "time": 1},
          {"well": "B02", "field": 1, "channel": 2, "time": 1}],
         "2 images, 2 well(s), 1 field(s), 2 channel(s)"),
    ]
    for case_id, rows, expected in cases:
        with subtests.test(msg=case_id):
            assert ip.summarize_rows(rows) == expected
