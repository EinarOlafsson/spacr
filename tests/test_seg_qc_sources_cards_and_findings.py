"""Segmentation QC from mask source to scorecard to finding.

The rule throughout ``spacr.seg_qc`` is refuse rather than default: a typo'd
mask path is an error, not zero fields; a card missing its verdict columns is
refused, not read as "ok"; a positional claim needs a position; a stack folder
that cannot be dated dates as zero. These tests walk case tables over:

* :func:`score_masks` -- every source shape it accepts, and the plate
  comparison that calls a field split apart or merged, once;
* :func:`read_scorecard` -- every way a card on disk can be damaged;
* :func:`diagnose` -- which flags and plate patterns become findings;
* :func:`find_mask_stacks` / :func:`mask_stack_mtime` -- locating and dating
  the stacks a card is checked against;
* :class:`Scorecard` -- its verdict and one-line rendering.

CPU-only, offline and deterministic.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from spacr import seg_qc
from spacr.seg_qc import (CARD_DIR, CARD_PREFIX, FLAG_EMPTY, FLAG_OVER,
                          FLAG_UNDER, FLAG_UNREADABLE, MASK_STACK_SUFFIX,
                          FieldQC, Scorecard, diagnose, find_mask_stacks,
                          mask_stack_mtime, read_scorecard, score_field,
                          score_masks)


# ---------------------------------------------------------------------------
# mask builders
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    """Paint one filled disc of label ``value`` into ``labels``."""
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hit = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    labels[y0:y1, x0:x1][hit] = value
    return labels


def _grid_field(shape=(256, 256), radius=8, spacing=40, margin=30, count=None):
    """A grid of well-separated discs (25 by default), none on the border."""
    labels = np.zeros(shape, np.int32)
    value = 0
    for cy in range(margin, shape[0] - margin, spacing):
        for cx in range(margin, shape[1] - margin, spacing):
            if count is not None and value >= count:
                return labels
            value += 1
            _disc(labels, cy, cx, radius, value)
    return labels


def _fused_field(shape=(384, 384), radius=16, step=26, cluster=3, block=96):
    """Clusters of overlapping discs, each cluster welded into ONE label."""
    labels = np.zeros(shape, np.int32)
    value = 0
    for by in range(0, shape[0], block):
        for bx in range(0, shape[1], block):
            value += 1
            cy0 = by + block // 2 - (cluster - 1) * step // 2
            cx0 = bx + block // 2 - (cluster - 1) * step // 2
            for i in range(cluster):
                for j in range(cluster):
                    _disc(labels, cy0 + i * step, cx0 + j * step, radius, value)
    return labels


_CLEAN_25 = ("25 cell(s), median diameter 15.8 px, 0% on the border, "
             "foreground 7.5%. Nothing wrong with this field.")


def _rows(scored):
    return [(q.field, q.n_objects, q.flags, q.severity) for q in scored]


# ---------------------------------------------------------------------------
# score_masks
# ---------------------------------------------------------------------------

def test_score_masks_scores_every_shape_a_mask_source_arrives_in(subtests,
                                                                 tmp_path):
    """score_masks: file, folder, 2-D, 3-D, list and mapping sources, by name."""
    single = tmp_path / "plate1_A01_f1.npy"
    np.save(single, _grid_field().astype(np.uint16))
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()
    np.save(folder / "plate1_A02_f1.npy", _grid_field(count=12))
    np.save(folder / "plate1_A01_f1.npy", _grid_field().astype(np.uint16))
    (folder / "plate1_A03_f1.npy").write_bytes(b"junk")
    (folder / ".partial.npy").write_bytes(b"junk")
    (folder / "notes.txt").write_text("not a mask", encoding="utf-8")

    cases = [
        # (id, source, expected (field, n_objects, flags, severity) rows)
        ("a single npy file is one field named after it", str(single),
         [("plate1_A01_f1", 25, [], "ok")]),
        ("a folder is scored file by file, sorted, hidden and non-npy "
         "skipped, an unreadable mask flagged", str(folder),
         [("plate1_A01_f1", 25, [], "ok"), ("plate1_A02_f1", 12, [], "ok"),
          ("plate1_A03_f1", 0, [FLAG_UNREADABLE], "fail")]),
        ("a bare 2-D mask is one field, not a stack of rows", _grid_field(),
         [("field_0000", 25, [], "ok")]),
        ("a list is scored field by field",
         [_grid_field(), _grid_field(count=3)],
         [("field_0000", 25, [], "ok"),
          ("field_0001", 3, ["near_empty_field"], "warn")]),
        ("a 3-D stack is split on its first axis",
         np.stack([_grid_field(), _grid_field(count=4)]),
         [("field_0000", 25, [], "ok"),
          ("field_0001", 4, ["near_empty_field"], "warn")]),
        ("a mapping keeps its names and order, and calls loaders",
         {"b_field": _grid_field(), "a_field": lambda: _grid_field(count=12)},
         [("b_field", 25, [], "ok"), ("a_field", 12, [], "ok")]),
        ("a source with no masks is no fields, not an error", [], []),
    ]
    for case_id, source, expected in cases:
        with subtests.test(msg=case_id):
            scored = score_masks(source, object_type="cell")
            assert _rows(scored) == expected
            assert {q.object_type for q in scored} <= {"cell"}

    with subtests.test(msg="a clean field's note states its numbers"):
        [qc] = score_masks(str(single), object_type="cell")
        assert qc.note == _CLEAN_25

    with subtests.test(msg="an unreadable mask says why it was not scored"):
        unreadable = score_masks(str(folder), object_type="cell")[2]
        assert unreadable.metrics == {}
        assert unreadable.note.startswith(
            "could not read the mask: ValueError: ")

    with subtests.test(msg="a path that is neither folder nor file is refused"):
        with pytest.raises(FileNotFoundError,
                           match="no mask folder or file at .*nowhere"):
            score_masks(str(tmp_path / "nowhere"), object_type="cell")


def test_the_plate_comparison_flags_split_and_merged_fields_once(subtests):
    """score_masks: plate-relative over/under segmentation, never stapled twice."""
    with subtests.test(msg="small and numerous objects are called split apart"):
        normal = {f"plate1_A0{i}_f1": _grid_field(shape=(400, 400), radius=10,
                                                  spacing=60, margin=40)
                  for i in range(1, 6)}
        shattered = _grid_field(shape=(400, 400), radius=4, spacing=25,
                                margin=20)
        # Alone, the field is unremarkable: the flag can only come from
        # comparing it with the plate.
        assert score_field(shattered, object_type="cell", field="x").flags == []

        scored = {q.field: q
                  for q in score_masks({**normal, "plate1_A06_f1": shattered},
                                       object_type="cell")}
        odd = scored["plate1_A06_f1"]
        assert (odd.n_objects, odd.flags, odd.severity) == (
            225, [FLAG_OVER, "high_object_count"], "fail")
        assert odd.note.endswith(
            "Nothing wrong with this field. 225 objects where the plate median "
            "is 36 (6.2x); objects are 0.39x the plate's median diameter at "
            "6.2x its count, the signature of objects split apart.")
        assert [(scored[name].n_objects, scored[name].flags)
                for name in sorted(normal)] == [(36, [])] * 5

    with subtests.test(msg="a field its pixels called fused is flagged once"):
        grid = dict(shape=(512, 512), radius=10, spacing=50, margin=40)
        fields = {
            "plate1_A01_f1": _grid_field(**grid),
            "plate1_A02_f1": _grid_field(**grid),
            "plate1_A03_f1": _grid_field(**grid),
            "plate1_B01_f1": _fused_field(),
            "plate1_B02_f1": _grid_field(shape=(512, 512), radius=15,
                                         spacing=170, margin=60),
        }
        scored = {q.field: q for q in score_masks(fields, object_type="cell")}
        both = scored["plate1_B01_f1"]
        plate_only = scored["plate1_B02_f1"]

        assert both.flags == [FLAG_UNDER, "low_object_count"]
        assert both.note == (
            "16 cell(s), median diameter 89.5 px, 0% on the border, "
            "foreground 68.3%. objects look fused (foreground covers 68% of "
            "the field; the distance transform resolves 9.0 objects per mask "
            "object). 16 objects where the plate median is 81 (0.20x).")
        assert plate_only.flags == ["near_empty_field", FLAG_UNDER,
                                    "low_object_count"]
        assert plate_only.note.endswith(
            "9 objects where the plate median is 81 (0.11x); objects are 1.50x "
            "the plate's median diameter at 0.11x its count, the signature of "
            "objects merged in pairs.")
        assert [scored[f"plate1_A0{i}_f1"].flags for i in (1, 2, 3)] == [[]] * 3


# ---------------------------------------------------------------------------
# read_scorecard
# ---------------------------------------------------------------------------

_HEADER = "field,object_type,n_objects,severity,flags,note\n"


def _card(tmp_path, text):
    path = tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return str(path)


class _NulRefusingReader:
    """What python 3.9's DictReader does with a NUL: raise while iterating."""

    fieldnames = ["field", "n_objects"]

    def __init__(self, *_args, **_kwargs):
        pass

    def __iter__(self):
        raise csv.Error("line contains NUL")


class _LimitRefusingReader(_NulRefusingReader):
    """A DictReader that refuses an oversized field."""

    def __iter__(self):
        raise csv.Error("field larger than field limit (131072)")


def test_read_scorecard_refuses_a_damaged_card_rather_than_defaulting(
        subtests, tmp_path, monkeypatch):
    """read_scorecard: (rows, "") for a sound card, ([], reason) for a damaged one."""
    name = f"{CARD_PREFIX}cell.csv"
    refused = [
        # (id, card text or None for no file, DictReader stand-in, reason)
        ("a card that cannot be opened", None, None,
         "card.csv unreadable (FileNotFoundError)"),
        ("a card with no verdict columns",
         "object_type,severity\ncell,fail\n", None,
         f"{name} has no field, n_objects column, so it carries no verdict "
         "to read"),
        ("a card missing only its count column",
         "field,severity\nplate1_A01_f1,fail\n", None,
         f"{name} has no n_objects column, so it carries no verdict to read"),
        ("a NUL byte in a row", _HEADER + "plate1_A01_f1,cell,\x00,fail,,\n",
         None, f"{name} is not CSV (NUL byte)"),
        ("a NUL byte in the header", "field,n_obj\x00ects\nx,1\n", None,
         f"{name} is not CSV (NUL byte)"),
        ("a csv runtime that raises for NUL", _HEADER + "a,cell,1,ok,,\n",
         _NulRefusingReader, f"{name} is not CSV (NUL byte)"),
        ("a card the csv module refuses", _HEADER + "a,cell,1,ok,,\n",
         _LimitRefusingReader,
         f"{name} is not readable as CSV (field larger than field limit "
         "(131072))"),
    ]
    for case_id, text, reader, reason in refused:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            path = (str(tmp_path / "nowhere" / "card.csv") if text is None
                    else _card(tmp_path, text))
            if reader is not None:
                patch.setattr(csv, "DictReader", reader)
            assert read_scorecard(path) == ([], reason)

    with subtests.test(msg="an empty file is no rows and no error"):
        assert read_scorecard(_card(tmp_path, "")) == ([], "")

    with subtests.test(msg="a sound card is rebuilt row by row"):
        path = _card(tmp_path, (
            "field,object_type,n_objects,severity,flags,note,border_fraction\n"
            f"plate1_A01_f1,cell,many,fail,{FLAG_EMPTY};{FLAG_UNDER},"
            "hand edited,n/a\n"
            "plate1_A02_f1,cell,12.0,ok,,,0.25\n"
            ",,,,,,\n"))
        rows, error = read_scorecard(path)
        assert error == ""
        assert [(q.field, q.object_type, q.n_objects, q.flags, q.severity,
                 q.note) for q in rows] == [
            ("plate1_A01_f1", "cell", 0, [FLAG_EMPTY, FLAG_UNDER], "fail",
             "hand edited"),
            ("plate1_A02_f1", "cell", 12, [], "ok", ""),
            ("", "", 0, [], "ok", ""),
        ]
        assert np.isnan(rows[0].metrics["border_fraction"])
        assert rows[0].metrics["n_objects"] == 0.0
        assert rows[1].metrics == {"border_fraction": 0.25, "n_objects": 12.0}


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

def _qc(field, flags=(), n_objects=10, severity="ok", diameter=12.0,
        object_type="cell"):
    return FieldQC(field=field, object_type=object_type, n_objects=n_objects,
                   flags=list(flags), metrics={"median_diameter": diameter},
                   severity=severity)


def _plate_of_rows(top, bottom, columns=(1, 2, 3, 4)):
    """Rows A-D built by ``top(name)``, rows E-H by ``bottom(name)``."""
    rows = [top(f"plate1_{row}0{column}_f1")
            for row in "ABCD" for column in columns]
    rows += [bottom(f"plate1_{row}0{column}_f1")
             for row in "EFGH" for column in columns]
    return rows


def _finding_tuple(finding):
    return (finding.severity, finding.kind, finding.headline, finding.flag,
            finding.plate, finding.object_type, finding.fields,
            finding.n_fields)


def test_diagnose_reports_only_what_the_rows_can_support(subtests):
    """diagnose: flag findings, demotion, and positional gradients with exclusions."""
    wells = ("plate1_A01_f1", "plate1_A02_f1", "plate1_A03_f1")
    uniform = [_qc(f"plate1_{row}0{column}_f1", n_objects=100)
               for row in "ABCDEFGH" for column in (1, 2, 3)]
    cases = [
        # (id, rows, expected finding tuples)
        ("nothing at all produces nothing", [], []),
        ("a flag nothing can explain produces no finding",
         [_qc("plate1_A01_f1", flags=["a_flag_from_the_future"],
              severity="warn")], []),
        ("fields without wells are located by their names",
         [_qc("field_0000", flags=[FLAG_EMPTY], n_objects=0, severity="fail"),
          _qc("field_0001", flags=[FLAG_EMPTY], n_objects=0, severity="fail")],
         [("fail", "flag",
           "2 cell field(s) on field, fields field_0000, field_0001: no "
           "object at all in the field", FLAG_EMPTY, "field", "cell",
           ("field_0000", "field_0001"), 2)]),
        ("a flag demoted on a sparse plate is reported as a warning",
         [_qc(w, flags=[FLAG_EMPTY], n_objects=0, severity="warn")
          for w in wells],
         [("warn", "flag",
           "3 cell field(s) on plate1, wells A01, A02, A03: no object at all "
           "in the field", FLAG_EMPTY, "plate1", "cell", wells, 3)]),
        ("a flag no field demoted stays a failure",
         [_qc(w, flags=[FLAG_EMPTY], n_objects=0, severity="fail")
          for w in wells],
         [("fail", "flag",
           "3 cell field(s) on plate1, wells A01, A02, A03: no object at all "
           "in the field", FLAG_EMPTY, "plate1", "cell", wells, 3)]),
        ("unreadable fields are left out of the positional comparison",
         uniform + [_qc(f"plate1_A0{c}_f1", flags=[FLAG_UNREADABLE],
                        n_objects=0, severity="fail") for c in (4, 5)],
         [("fail", "flag",
           "2 cell field(s) on plate1, wells A04, A05: the mask file could "
           "not be opened", FLAG_UNREADABLE, "plate1", "cell",
           ("plate1_A04_f1", "plate1_A05_f1"), 2)]),
        ("fields whose names carry no well are left out of the gradient",
         [_qc(f"plate1_{row}01_f1", n_objects=100) for row in "ABCDEFGH"]
         + [_qc(f"field_{i:04d}", n_objects=0) for i in range(6)], []),
        ("a size step across the plate is a warning",
         _plate_of_rows(lambda n: _qc(n, n_objects=100, diameter=10.0),
                        lambda n: _qc(n, n_objects=100, diameter=30.0)),
         [("warn", "size_gradient",
           "plate1: cells in rows E-H measure 3.0x the diameter of those in "
           "rows A-D (30.0 vs 10.0 px)", "", "plate1", "cell", (), 32)]),
        ("a count step across the plate is a failure",
         _plate_of_rows(lambda n: _qc(n, n_objects=100),
                        lambda n: _qc(n, n_objects=10)),
         [("fail", "count_gradient",
           "plate1: rows A-D hold 10.0x the cell count of rows E-H (100 vs 10 "
           "objects per field)", "", "plate1", "cell", (), 32)]),
    ]
    for case_id, rows, expected in cases:
        with subtests.test(msg=case_id):
            assert [_finding_tuple(f) for f in diagnose(rows)] == expected


# ---------------------------------------------------------------------------
# finding and dating mask stacks
# ---------------------------------------------------------------------------

def _stack(path, *names):
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        np.save(path / name, np.zeros((4, 4), np.uint16))
    return path


def test_find_mask_stacks_offers_only_folders_it_could_list(subtests, tmp_path,
                                                            monkeypatch):
    """find_mask_stacks: two levels, real folders only, unlistable levels skipped."""
    real_listdir = os.listdir

    def refusing(*refused):
        def listdir(path):
            if os.path.basename(str(path)) in refused or str(path) in refused:
                raise PermissionError(path)
            return real_listdir(path)
        return listdir

    cases = [
        # (id, refused listdir targets or None, expected relative stacks)
        ("stacks at the root and one level down", None,
         {"cell": "cell_mask_stack",
          "nucleus": "norm_channel_stack/nucleus_mask_stack",
          "pathogen": "plate2/pathogen_mask_stack"}),
        ("a folder that cannot be listed hides only itself", ("plate2",),
         {"cell": "cell_mask_stack",
          "nucleus": "norm_channel_stack/nucleus_mask_stack"}),
        ("a root that cannot be listed finds no stacks", ("root",), {}),
    ]
    root = tmp_path / "root"
    _stack(root / "cell_mask_stack", "plate1_A01_f1.npy")
    _stack(root / "norm_channel_stack" / "nucleus_mask_stack")
    # the root-level cell stack wins over a deeper one of the same type
    _stack(root / "norm_channel_stack" / "cell_mask_stack")
    _stack(root / "plate2" / "pathogen_mask_stack")
    # a FILE named like a stack is not a stack
    (root / "plate2" / f"nucleus{MASK_STACK_SUFFIX}").write_bytes(b"tarball")
    (root / f"organelle{MASK_STACK_SUFFIX}").write_bytes(b"lock")
    for case_id, refused, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            if refused is not None:
                patch.setattr(os, "listdir", refusing(*refused))
            assert find_mask_stacks(str(root)) == {
                kind: str(root / rel) for kind, rel in expected.items()}


class _EntryWrapper:
    def __init__(self, entry, refuse):
        self._entry = entry
        self.name = entry.name
        self._refuse = refuse

    def stat(self):
        if self.name in self._refuse:
            raise FileNotFoundError(self.name)
        return self._entry.stat()


def _scandir_double(pulled, refuse=()):
    """An os.scandir stand-in counting the entries pulled from it."""
    real_scandir = os.scandir

    class _Scandir:
        def __init__(self, path):
            self._inner = real_scandir(path)

        def __enter__(self):
            def entries():
                for entry in self._inner:
                    pulled.append(entry.name)
                    yield _EntryWrapper(entry, refuse)
            return entries()

        def __exit__(self, *exc):
            self._inner.close()
            return False

    return _Scandir


def test_mask_stack_mtime_dates_a_stack_by_its_newest_mask(subtests, tmp_path,
                                                           monkeypatch):
    """mask_stack_mtime: newest .npy or the folder's own date; 0.0 when unstatable."""
    def refuse(path, *args, **kwargs):
        raise PermissionError(path)

    def build(name, files):
        folder = tmp_path / name / "cell_mask_stack"
        folder.mkdir(parents=True)
        for filename, when in files:
            target = folder / filename
            if filename.lower().endswith(".npy"):
                np.save(target, np.zeros((4, 4), np.uint16))
                if target.suffix != ".npy":       # np.save appends .npy
                    os.replace(str(target) + ".npy", target)
            else:
                target.write_text("not a mask", encoding="utf-8")
            os.utime(target, (when, when))
        return folder

    cases = [
        # (id, files (name, mtime), folder mtime, patch, cap, expected,
        #  entries pulled or None)
        ("the newest mask is found past files that are not masks",
         [("plate1_A01_f1.npy", 10 ** 6), ("notes.txt", 10 ** 9)], 10 ** 5,
         None, 512, 10 ** 6, None),
        ("an upper-case suffix is still a mask",
         [("plate1_A01_f1.NPY", 2 * 10 ** 6)], 10 ** 5, None, 512,
         2 * 10 ** 6, None),
        ("the folder's own date is the floor",
         [("plate1_A01_f1.npy", 10 ** 5)], 10 ** 6, None, 512, 10 ** 6, None),
        ("a mask that cannot be stat-ed is skipped",
         [("gone.npy", 3 * 10 ** 6), ("here.npy", 10 ** 6)], 10 ** 5,
         "entry", 512, 10 ** 6, None),
        ("the scan stops one entry past the cap",
         [(f"plate1_A0{i}_f1.npy", 10 ** 6) for i in range(6)], 10 ** 5,
         "count", 2, None, 3),
        ("an uncapped scan visits every entry",
         [(f"plate1_A0{i}_f1.npy", 10 ** 6) for i in range(6)], 10 ** 5,
         "count", 512, 10 ** 6, 6),
        ("a folder that cannot be scanned falls back to its own date",
         [("plate1_A01_f1.npy", 3 * 10 ** 6)], 10 ** 6, "scandir", 512,
         10 ** 6, None),
        ("a folder that cannot be stat-ed dates as zero",
         [("plate1_A01_f1.npy", 3 * 10 ** 6)], 10 ** 6, "stat", 512, 0.0,
         None),
    ]
    for index, (case_id, files, folder_time, patch_kind, cap, expected,
                n_pulled) in enumerate(cases):
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            folder = build(f"case{index}", files)
            os.utime(folder, (folder_time, folder_time))
            pulled = []
            patch.setattr(seg_qc, "_MAX_MTIME_STATS", cap)
            if patch_kind == "entry":
                patch.setattr(os, "scandir",
                              _scandir_double(pulled, refuse=("gone.npy",)))
            elif patch_kind == "count":
                patch.setattr(os, "scandir", _scandir_double(pulled))
            elif patch_kind == "scandir":
                patch.setattr(os, "scandir", refuse)
            elif patch_kind == "stat":
                patch.setattr(os, "stat", refuse)
            dated = mask_stack_mtime(str(folder))
            if expected is not None:
                assert dated == pytest.approx(expected)
            if n_pulled is not None:
                assert len(pulled) == n_pulled

    with subtests.test(msg="a folder that does not exist dates as zero"):
        assert mask_stack_mtime(str(tmp_path / "missing")) == 0.0


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def test_a_scorecard_reports_its_verdict_in_one_line(subtests):
    """Scorecard: error beats the summary, no summary is empty, stale is said."""
    path = "/runs/plate1/qc/segmentation_qc_cell.csv"
    cases = [
        # (id, card, verdict, str)
        ("a card that could not be read reports error",
         Scorecard(path=path, object_type="cell",
                   summary={"verdict": "ok"},
                   error="card.csv unreadable (OSError)"),
         "error", "cell: error"),
        ("a stale card says it is out of date",
         Scorecard(path="", object_type="nucleus", summary={"verdict": "warn"},
                   stale=True),
         "warn", "nucleus: warn (out of date)"),
        ("a card with no summary is empty",
         Scorecard(path=path, object_type="cell"), "empty", "cell: empty"),
        ("a failed card says fail", Scorecard(
            path=path, object_type="pathogen", summary={"verdict": "fail"}),
         "fail", "pathogen: fail"),
    ]
    for case_id, card, verdict, text in cases:
        with subtests.test(msg=case_id):
            assert card.verdict == verdict
            assert str(card) == text
