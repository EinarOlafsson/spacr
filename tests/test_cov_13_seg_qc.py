"""Segmentation QC when the mask, the card or the plate layout is not what it says.

A QC verdict is only worth having if it cannot be produced by accident. Every
path here is one where something is missing or damaged, and the rule is the
same throughout: refuse rather than default. A card with no verdict column
must not read as "ok"; a threshold that is not a number must not silently
become one; a plate with one row must not be reported as a gradient across
its halves; a mask holding NaN is not a label image.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from spacr import seg_qc
from spacr.seg_qc import (
    CARD_DIR,
    CARD_PREFIX,
    FLAG_EMPTY,
    FLAG_UNREADABLE,
    FieldQC,
    Finding,
    QCDigest,
    Scorecard,
    diagnose,
    explain_flag,
    format_digest,
    format_findings,
    parse_field_name,
    read_scorecard,
    score_field,
    score_masks,
    thresholds_from_settings,
    write_scorecard,
)


# ---------------------------------------------------------------------------
# mask builders
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hit = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    sub = labels[y0:y1, x0:x1]
    sub[hit] = value
    return labels


def _grid_field(shape=(256, 256), radius=8, spacing=40, margin=30, count=None):
    labels = np.zeros(shape, np.int32)
    value = 0
    for cy in range(margin, shape[0] - margin, spacing):
        for cx in range(margin, shape[1] - margin, spacing):
            if count is not None and value >= count:
                return labels
            value += 1
            _disc(labels, cy, cx, radius, value)
    return labels


# ---------------------------------------------------------------------------
# thresholds that are not numbers
# ---------------------------------------------------------------------------

def test_a_threshold_that_is_not_a_number_is_ignored_rather_than_guessed():
    """A blank or mistyped ``seg_qc_*`` setting must fall back to the default.

    The settings dict is filled from GUI fields and from user CSVs, so a
    threshold arriving as ``""``, ``"none"`` or a list is ordinary. Coercing
    any of those to a number would move the line between pass and fail
    without saying so.
    """
    key = "seg_qc_border_fraction"

    assert thresholds_from_settings({key: "not a number"}) == {}
    assert thresholds_from_settings({key: [0.4]}) == {}
    assert thresholds_from_settings({key: None}) == {}
    assert thresholds_from_settings({key: float("nan")}) == {}
    # A real number still gets through, so the guard is not simply dropping
    # everything.
    assert thresholds_from_settings({key: "0.4"}) == {"border_fraction": 0.4}
    assert thresholds_from_settings({key: 0.4}) == {"border_fraction": 0.4}


# ---------------------------------------------------------------------------
# masks that are not label images
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask, expected", [
    (np.full((16, 16), np.nan, dtype=np.float32), "NaN or infinite"),
    (np.zeros((16, 16), dtype=np.complex64), "not a label type"),
    (np.array([[0, -1], [2, 3]], dtype=np.int32), "negative labels"),
])
def test_a_mask_that_is_not_a_label_image_is_flagged_and_says_why(mask,
                                                                  expected):
    """One corrupt field must not cost the plate its scorecard.

    Each of these is a real shape a mask arrives in: a float array a
    normalisation divided by zero, a complex array from the wrong reader, and
    a signed array where a downstream step wrote -1 for "unassigned". None of
    them can be counted, and each has to say which one it was.
    """
    qc = score_field(mask, object_type="cell", field="plate1_A01_f1")

    assert qc.flags == [FLAG_UNREADABLE]
    assert qc.severity == "fail"
    assert expected in qc.note
    assert qc.n_objects == 0


def test_a_foreground_too_small_to_hold_a_seed_is_not_called_fused():
    """The fusion cross-check needs seeds; with none it must claim nothing.

    A field holding one four-pixel speck has a foreground the distance
    transform cannot find a single peak in. Reporting zero seeds is the honest
    answer; anything else would compare a count against a number that was
    never computed.
    """
    speck = np.zeros((64, 64), np.int32)
    speck[30:32, 30:32] = 1

    assert seg_qc._split_seed_count(speck > 0, min_diameter=30.0) == 0


# ---------------------------------------------------------------------------
# the shapes a mask source arrives in
# ---------------------------------------------------------------------------

def test_a_single_npy_file_is_scored_as_one_field_named_after_it(tmp_path):
    """A user pointing at one mask file must get that file scored.

    The field name is what every downstream address is parsed from, so it has
    to be the file's stem rather than a positional ``field_0000``.
    """
    path = tmp_path / "plate1_A01_f1.npy"
    np.save(path, _grid_field().astype(np.uint16))

    scored = score_masks(str(path), object_type="cell")

    assert [q.field for q in scored] == ["plate1_A01_f1"]
    assert scored[0].n_objects > 0


def test_a_path_that_is_neither_a_folder_nor_a_file_is_refused(tmp_path):
    """A typo in a mask path must say so, not score zero fields.

    An empty result is what ``save=False`` looks like, and reporting a missing
    folder the same way would tell the user their masks were fine.
    """
    with pytest.raises(FileNotFoundError, match="no mask folder or file"):
        score_masks(str(tmp_path / "nowhere"), object_type="cell")


def test_a_bare_two_dimensional_mask_is_one_field_not_a_stack():
    """A 2-D array handed straight in is a single field.

    Iterating it would treat each row of pixels as a field and produce 256
    unreadable ones.
    """
    scored = score_masks(_grid_field(), object_type="cell")

    assert [q.field for q in scored] == ["field_0000"]
    assert scored[0].n_objects > 0


def test_a_list_of_masks_is_scored_field_by_field():
    """A plain sequence is the shape a caller with masks in memory has."""
    scored = score_masks([_grid_field(), _grid_field(count=3)],
                         object_type="cell")

    assert [q.field for q in scored] == ["field_0000", "field_0001"]
    assert scored[0].n_objects > scored[1].n_objects


# ---------------------------------------------------------------------------
# formatting a number nobody computed
# ---------------------------------------------------------------------------

def test_a_pixel_size_that_was_never_computed_prints_as_a_dash():
    """An empty field has no median diameter, and 0.0 px would be a claim.

    The scorecard is read by eye. A column of ``0.0`` invites the reader to
    conclude the objects were tiny rather than that there were none.
    """
    assert seg_qc._px(None) == "-"
    assert seg_qc._px(float("nan")) == "-"
    assert seg_qc._px(12.34) == "12.3"


def test_writing_a_scorecard_for_nothing_writes_no_file(tmp_path):
    """No fields is not an empty card; it is no card.

    An empty CSV beside the masks would be read back later as a plate that
    scored clean.
    """
    assert write_scorecard([], str(tmp_path), "cell") is None
    assert not (tmp_path / CARD_DIR).exists()


# ---------------------------------------------------------------------------
# how the flag guidance and the findings print
# ---------------------------------------------------------------------------

def test_a_flag_explanation_prints_as_its_flag_and_headline():
    """``str()`` on guidance is what a log line and a tooltip both use."""
    guidance = explain_flag(FLAG_EMPTY)

    assert str(guidance) == f"{guidance.flag}: {guidance.headline}"
    assert guidance.flag == FLAG_EMPTY


def test_a_finding_prints_its_severity_and_reads_as_one_paragraph():
    """The two renderings a finding has: a list line and a block of prose."""
    finding = Finding(severity="fail", kind="flag",
                      headline="plate1: 3 cell field(s) empty",
                      detail="Nothing was segmented there.",
                      fix="Check the diameter.")

    assert str(finding) == "[fail] plate1: 3 cell field(s) empty"
    assert finding.text() == ("plate1: 3 cell field(s) empty "
                              "Nothing was segmented there. "
                              "Check the diameter.")


def test_a_finding_with_nothing_but_a_headline_does_not_pad_it_with_spaces():
    """The joiner has to drop the empty parts, not join around them."""
    assert Finding(severity="warn", kind="flag",
                   headline="plate1: something").text() == "plate1: something"


def test_nothing_flagged_says_so_rather_than_printing_an_empty_report():
    """A blank report reads as a report that failed to run."""
    assert format_findings([]) == "Nothing flagged."


# ---------------------------------------------------------------------------
# plate addresses
# ---------------------------------------------------------------------------

def test_a_field_address_prints_the_well_when_it_has_one():
    """The address is what a finding names, so it has to degrade cleanly.

    A hand-assembled folder has no well in its file names, and printing
    ``plate1/`` for it would look like a well whose name went missing.
    """
    assert str(parse_field_name("plate1_A01_f1.npy")) == "plate1/A01"
    assert str(parse_field_name("just_a_name.npy")) == "just"
    assert str(parse_field_name("plateonly.npy")) == "plateonly"


def test_naming_no_wells_at_all_produces_no_list():
    """An empty list must not print as an empty pair of parentheses."""
    assert seg_qc._name_list([], max_named=3) == ""
    assert seg_qc._name_list(["", None], max_named=3) == ""
    assert seg_qc._name_list(["A01", "A02"], max_named=3) == "A01, A02"


def test_a_single_row_or_column_prints_as_itself_not_as_a_range():
    """``A-A`` and ``3-3`` are ranges of one, which read as a mistake."""
    assert seg_qc._row_range([]) == ""
    assert seg_qc._row_range(["c"]) == "C"
    assert seg_qc._row_range(["A", "B", "C"]) == "A-C"
    assert seg_qc._row_range(["A", "C"]) == "A, C"

    assert seg_qc._column_range([]) == ""
    assert seg_qc._column_range([3]) == "3"
    assert seg_qc._column_range([1, 2, 3]) == "1-3"
    assert seg_qc._column_range([1, 4]) == "1, 4"


# ---------------------------------------------------------------------------
# what the findings refuse to claim
# ---------------------------------------------------------------------------

def _qc(field, flags=(), n_objects=10, severity="ok", diameter=12.0,
        object_type="cell"):
    return FieldQC(field=field, object_type=object_type, n_objects=n_objects,
                   flags=list(flags), metrics={"median_diameter": diameter},
                   severity=severity)


def test_a_flag_nothing_knows_how_to_explain_produces_no_finding():
    """A finding with no cause and no fix is a line of noise.

    Flags are written by whatever version of spaCR produced the card, and a
    card from a newer run can carry one this build has no guidance for.
    """
    findings = diagnose([_qc("plate1_A01_f1", flags=["a_flag_from_the_future"],
                             severity="warn")])

    assert findings == []


def test_a_plate_with_no_wells_in_its_names_is_located_by_its_fields():
    """A hand-assembled folder still has to say WHERE the problem is.

    Without a well the finding would name the plate and stop, and the user
    would have to open every mask to find which ones were flagged.
    """
    findings = diagnose([
        _qc("field_0000", flags=[FLAG_EMPTY], n_objects=0, severity="fail"),
        _qc("field_0001", flags=[FLAG_EMPTY], n_objects=0, severity="fail"),
    ])

    assert findings
    where = findings[0].headline
    assert "fields" in where
    assert "field_0000" in where


def test_a_blank_field_name_does_not_print_an_empty_location():
    """A location phrase is useful only when there is a location to print."""
    finding = seg_qc._flag_findings([
        _qc("", flags=[FLAG_EMPTY], n_objects=0, severity="fail"),
    ], max_named=3)[0]

    assert finding.fields == ("",)
    assert finding.headline == (
        "1 cell field(s) on this project: no object at all in the field")
    assert "fields :" not in finding.headline


def test_a_flag_demoted_on_a_sparse_plate_is_not_reported_as_a_failure():
    """The finding must not contradict the verdict the card already printed.

    ``_apply_plate_context`` demotes empty fields on a plate whose median
    count is a handful, because there a field with none is the assay. The
    flag's own severity is still "fail", and reporting it as one here would
    put a failure in the digest that no field claims.
    """
    members = [_qc(f"plate1_A0{i}_f1", flags=[FLAG_EMPTY], n_objects=0,
                   severity="warn") for i in range(1, 4)]

    findings = diagnose(members)

    assert findings
    assert [f.severity for f in findings] == ["warn"]


def test_a_flag_no_field_demoted_stays_a_failure():
    """The other half of the same rule, so the demotion is not unconditional."""
    members = [_qc(f"plate1_A0{i}_f1", flags=[FLAG_EMPTY], n_objects=0,
                   severity="fail") for i in range(1, 4)]

    findings = diagnose(members)

    assert [f.severity for f in findings] == ["fail"]


def test_an_unreadable_field_is_left_out_of_the_positional_comparison():
    """A field that could not be read has no count to compare.

    Its ``n_objects`` is 0 by construction, so leaving it in would make its
    half of the plate look empty and fabricate a gradient.
    """
    rows = []
    for row in "ABCDEFGH":
        for column in (1, 2, 3):
            rows.append(_qc(f"plate1_{row}0{column}_f1", n_objects=100))
    rows.append(_qc("plate1_A04_f1", flags=[FLAG_UNREADABLE], n_objects=0,
                    severity="fail"))
    rows.append(_qc("plate1_A05_f1", flags=[FLAG_UNREADABLE], n_objects=0,
                    severity="fail"))

    gradients = [f for f in diagnose(rows) if f.kind.endswith("gradient")]

    assert gradients == []


def test_a_field_whose_name_carries_no_well_is_left_out_of_the_gradient():
    """A positional claim needs a position.

    ``field_0003`` is somewhere on the plate and nothing says where, so it
    cannot be assigned to a half.
    """
    rows = [_qc(f"plate1_{row}01_f1", n_objects=100) for row in "ABCD"]
    rows += [_qc(f"plate1_{row}01_f1", n_objects=100) for row in "EFGH"]
    rows += [_qc(f"field_{i:04d}", n_objects=0) for i in range(6)]

    gradients = [f for f in diagnose(rows) if f.kind.endswith("gradient")]

    assert gradients == []


def test_a_plate_with_one_row_has_no_halves_to_compare():
    """Splitting one key into halves gives an empty half, not a comparison."""
    assert seg_qc._halves([]) == ([], [])
    assert seg_qc._halves(["A"]) == ([], [])
    assert seg_qc._halves(["A", "B", "C"]) == (["A"], ["B", "C"])

    one_row = {"A": [10.0] * 8}
    assert seg_qc._axis_step(one_row, ratio=2.0, min_fields=3) is None


def test_a_half_with_too_few_fields_is_not_compared():
    """One field's count is not a median, however far it is from the other."""
    thin = {"A": [1.0], "B": [100.0]}

    assert seg_qc._axis_step(thin, ratio=2.0, min_fields=3) is None


def test_a_half_that_is_simply_empty_is_not_reported_as_a_gradient():
    """A median of zero is an empty half, and the empty flag already said so.

    Dividing by it would give an infinite fold change and a headline claiming
    a ratio nobody can act on.
    """
    empty_half = {"A": [0.0, 0.0, 0.0], "B": [100.0, 110.0, 90.0]}

    assert seg_qc._axis_step(empty_half, ratio=2.0, min_fields=3) is None


def test_objects_that_measure_twice_as_wide_on_one_half_are_reported(tmp_path):
    """A size step across the plate is a warning in its own right.

    Object size stepping by row is what an uneven lamp does to a threshold:
    the brighter half segments generously and its objects come out larger,
    with every field individually unremarkable.
    """
    rows = []
    for row in "ABCD":
        for column in range(1, 5):
            rows.append(_qc(f"plate1_{row}0{column}_f1", n_objects=100,
                            diameter=10.0))
    for row in "EFGH":
        for column in range(1, 5):
            rows.append(_qc(f"plate1_{row}0{column}_f1", n_objects=100,
                            diameter=30.0))

    findings = diagnose(rows)
    size = [f for f in findings if f.kind == "size_gradient"]

    assert size, [f.kind for f in findings]
    assert size[0].severity == "warn"
    assert "3.0x the diameter" in size[0].headline
    assert "A-D" in size[0].headline and "E-H" in size[0].headline


# ---------------------------------------------------------------------------
# reading a card back off disk
# ---------------------------------------------------------------------------

def _write_card(path, rows, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = header or ["field", "object_type", "n_objects", "severity",
                        "flags", "note"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_a_card_with_no_verdict_column_is_refused_rather_than_defaulted(
        tmp_path):
    """Half a card is a different verdict, and this module does not invent one.

    Every field of ``FieldQC`` has a default and ``severity`` defaults to
    "ok", so a card missing its columns would read back as a plate that
    passed. That is the exact failure this check exists for.
    """
    path = _write_card(str(tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv"),
                       [{"object_type": "cell", "severity": "fail"}],
                       header=["object_type", "severity"])

    rows, error = read_scorecard(path)

    assert rows == []
    assert "has no field, n_objects column" in error


def test_a_card_with_a_nul_byte_in_a_row_is_refused(tmp_path):
    """A truncated write leaves NUL bytes, and python 3.12 parses them through.

    The header check catches a NUL in a column name; this one catches a NUL in
    the data, where the row would otherwise be read as a field whose name
    contains a null.
    """
    path = str(tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("field,object_type,n_objects,severity,flags,note\n")
        handle.write("plate1_A01_f1,cell,\x00,fail,,\n")

    rows, error = read_scorecard(path)

    assert rows == []
    assert "not CSV (NUL byte)" in error


def test_a_csv_runtime_that_raises_for_nul_uses_the_same_diagnosis(
        tmp_path, monkeypatch):
    """Python 3.9 raises where 3.12 returns a row; the contract is identical."""
    path = _write_card(str(tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv"),
                       [{"field": "plate1_A01_f1", "n_objects": 1}])

    class _RejectsNul:
        fieldnames = ["field", "n_objects"]

        def __iter__(self):
            raise csv.Error("line contains NUL")

    monkeypatch.setattr(seg_qc.csv, "DictReader", lambda _handle: _RejectsNul())

    rows, error = read_scorecard(path)

    assert rows == []
    assert error.endswith("is not CSV (NUL byte)")


def test_a_count_that_is_not_a_number_reads_back_as_no_objects(tmp_path):
    """A hand-edited card can carry text where the count belongs.

    Zero is the safe reading: the field's own flags and severity are what the
    verdict is taken from, and refusing the whole card over one cell would
    lose the other ninety-five fields.
    """
    path = _write_card(str(tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv"), [
        {"field": "plate1_A01_f1", "object_type": "cell",
         "n_objects": "many", "severity": "fail", "flags": FLAG_EMPTY,
         "note": ""},
        {"field": "plate1_A02_f1", "object_type": "cell",
         "n_objects": "12", "severity": "ok", "flags": "", "note": ""},
    ])

    rows, error = read_scorecard(path)

    assert error == ""
    assert [q.n_objects for q in rows] == [0, 12]
    assert rows[0].flags == [FLAG_EMPTY]


def test_a_card_that_cannot_be_opened_says_so_instead_of_raising(tmp_path):
    """A screen draws the verdict; a missing card must not take it down."""
    rows, error = read_scorecard(str(tmp_path / "nowhere" / "card.csv"))

    assert rows == []
    assert "unreadable (FileNotFoundError)" in error


def test_a_card_the_csv_module_refuses_says_so(tmp_path, monkeypatch):
    """``csv.Error`` is a different failure from an unreadable file.

    It is reported separately because it means the bytes arrived and are not
    a table, which points at the writer rather than at the disk.
    """
    path = _write_card(str(tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv"), [
        {"field": "plate1_A01_f1", "object_type": "cell", "n_objects": "3",
         "severity": "ok", "flags": "", "note": ""},
    ])

    class _Refusing:
        def __init__(self, *args, **kwargs):
            self.fieldnames = ["field", "n_objects"]

        def __iter__(self):
            return self

        def __next__(self):
            raise csv.Error("field larger than field limit")

    monkeypatch.setattr(csv, "DictReader", _Refusing)

    rows, error = read_scorecard(path)

    assert rows == []
    assert "not readable as CSV" in error


# ---------------------------------------------------------------------------
# looking for mask stacks and their dates
# ---------------------------------------------------------------------------

def test_a_project_root_that_cannot_be_listed_finds_no_stacks(tmp_path,
                                                              monkeypatch):
    """A permission error on a project folder is not a project with no masks.

    It comes back empty either way, but it must not raise: the screen calls
    this to decide whether it can offer to score.
    """
    def refuse(path):
        raise PermissionError(path)

    monkeypatch.setattr(os, "listdir", refuse)

    assert seg_qc.find_mask_stacks(str(tmp_path)) == {}


def test_a_plate_folder_that_cannot_be_listed_is_skipped_not_fatal(tmp_path,
                                                                   monkeypatch):
    """One unreadable plate must not hide the plates beside it."""
    good = tmp_path / "plate1" / "cell_mask_stack"
    good.mkdir(parents=True)
    np.save(good / "plate1_A01_f1.npy", np.zeros((8, 8), np.uint16))
    bad = tmp_path / "plate2"
    bad.mkdir()

    real = os.listdir

    def refuse(path):
        if os.path.basename(str(path)) == "plate2":
            raise PermissionError(path)
        return real(path)

    monkeypatch.setattr(os, "listdir", refuse)

    assert seg_qc.find_mask_stacks(str(tmp_path)) == {"cell": str(good)}


def test_the_newest_mask_is_found_past_files_that_are_not_masks(tmp_path):
    """Only ``.npy`` files date a stack.

    A stack folder holds a README, a lock file, whatever the user dropped in.
    Dating the stack from those would make a card look out of date because
    somebody opened the folder.
    """
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()
    np.save(folder / "plate1_A01_f1.npy", np.zeros((8, 8), np.uint16))
    (folder / "notes.txt").write_text("not a mask")
    os.utime(folder / "notes.txt", (10 ** 9, 10 ** 9))
    os.utime(folder / "plate1_A01_f1.npy", (10 ** 6, 10 ** 6))
    os.utime(folder, (10 ** 5, 10 ** 5))

    assert seg_qc.mask_stack_mtime(str(folder)) == pytest.approx(10 ** 6)


def test_dating_a_stack_stops_after_a_bounded_number_of_files(tmp_path,
                                                              monkeypatch):
    """A plate can hold tens of thousands of masks and this runs on a click.

    The cap is what keeps drawing the verdict off the whole-folder stat path.
    Counted rather than compared by date, because the order a directory is
    scanned in is the filesystem's business and not something to assert on.
    """
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()
    for index in range(6):
        np.save(folder / f"plate1_A0{index}_f1.npy",
                np.zeros((4, 4), np.uint16))
    os.utime(folder, (10 ** 5, 10 ** 5))

    real_scandir = os.scandir
    pulled: list = []

    class _Counting:
        def __init__(self, path):
            self._inner = real_scandir(path)

        def __enter__(self):
            def gen():
                for entry in self._inner:
                    pulled.append(entry.name)
                    yield entry
            return gen()

        def __exit__(self, *exc):
            self._inner.close()
            return False

    monkeypatch.setattr(os, "scandir", _Counting)

    monkeypatch.setattr(seg_qc, "_MAX_MTIME_STATS", 2)
    seg_qc.mask_stack_mtime(str(folder))
    capped = len(pulled)

    pulled.clear()
    monkeypatch.setattr(seg_qc, "_MAX_MTIME_STATS", 512)
    seg_qc.mask_stack_mtime(str(folder))
    whole = len(pulled)

    assert capped == 3, "the scan did not stop one entry past the cap"
    assert whole == 6


def test_a_mask_file_that_cannot_be_stat_ed_is_skipped(tmp_path, monkeypatch):
    """A file deleted between the listing and the stat is not an error."""
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()
    np.save(folder / "gone.npy", np.zeros((4, 4), np.uint16))
    np.save(folder / "here.npy", np.zeros((4, 4), np.uint16))
    os.utime(folder / "here.npy", (10 ** 6, 10 ** 6))
    os.utime(folder, (10 ** 5, 10 ** 5))

    real_scandir = os.scandir

    class _Entry:
        def __init__(self, entry):
            self._entry = entry
            self.name = entry.name

        def stat(self):
            if self.name == "gone.npy":
                raise FileNotFoundError(self.name)
            return self._entry.stat()

    class _Scandir:
        def __init__(self, path):
            self._inner = real_scandir(path)

        def __enter__(self):
            return (_Entry(e) for e in self._inner)

        def __exit__(self, *exc):
            self._inner.close()
            return False

    monkeypatch.setattr(os, "scandir", _Scandir)

    assert seg_qc.mask_stack_mtime(str(folder)) == pytest.approx(10 ** 6)


def test_a_stack_folder_that_cannot_be_scanned_falls_back_to_its_own_date(
        tmp_path, monkeypatch):
    """The directory's own mtime is the floor, and it survives a failed scan.

    Masks are written by atomic rename, which touches the directory, so the
    folder's date already catches a stack whose files were all replaced. A
    scan that raises must leave that answer standing rather than take the
    whole verdict down.
    """
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()
    np.save(folder / "plate1_A01_f1.npy", np.zeros((4, 4), np.uint16))
    os.utime(folder, (10 ** 6, 10 ** 6))

    def refuse(path):
        raise PermissionError(path)

    monkeypatch.setattr(os, "scandir", refuse)

    assert seg_qc.mask_stack_mtime(str(folder)) == pytest.approx(10 ** 6)


def test_a_stack_folder_that_cannot_be_stat_ed_dates_as_zero(tmp_path,
                                                             monkeypatch):
    """With no date at all, nothing can be called newer than the card."""
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()

    def refuse(path, *args, **kwargs):
        raise PermissionError(path)

    monkeypatch.setattr(os, "stat", refuse)

    assert seg_qc.mask_stack_mtime(str(folder)) == 0.0


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------

def test_a_scorecard_that_could_not_be_read_reports_error_as_its_verdict():
    """The verdict and the one-line rendering both have to say so."""
    card = Scorecard(path="/runs/plate1/qc/segmentation_qc_cell.csv",
                     object_type="cell", error="card.csv unreadable (OSError)")

    assert card.verdict == "error"
    assert str(card) == "cell: error"
    assert str(Scorecard(path="", object_type="nucleus",
                         summary={"verdict": "warn"},
                         stale=True)) == "nucleus: warn (out of date)"


def test_a_digest_prints_its_verdict_and_headline():
    """``str()`` on a digest is what a log line carries."""
    digest = QCDigest(root="/runs", verdict="fail", headline="plate1 is empty")

    assert str(digest) == "fail: plate1 is empty"


def test_a_source_that_is_not_a_path_contributes_no_project_root(tmp_path):
    """``settings['src']`` arrives as a list that can hold anything.

    A ``None`` or a number in it is a settings-file typo, and dropping it is
    what lets the real entries beside it still be searched.
    """
    roots = seg_qc.qc_roots([None, 17, str(tmp_path)])

    assert roots == (str(tmp_path),)


def test_a_digest_of_cards_with_no_verdict_and_no_findings_is_missing():
    """A card that scored no fields is not a card that passed.

    ``summarize_qc`` reports "empty" for a run that produced no fields, which
    is not a severity; with no findings either there is nothing to grade, and
    "missing" is the answer that sends the user to score the masks.
    """
    card = Scorecard(path="/runs/plate1/qc/segmentation_qc_cell.csv",
                     object_type="cell", field_qcs=[], summary={})
    assert card.verdict == "empty"

    digest = seg_qc._digest_from_cards("/runs", [card])

    assert digest.verdict == "missing"
    assert "Nothing has scored these masks" in digest.headline


def test_the_subhead_counts_the_failures_when_there_are_any():
    """The line under the verdict is a count, and it has to be the real one."""
    card = Scorecard(
        path="/runs/plate1/qc/segmentation_qc_cell.csv", object_type="cell",
        field_qcs=[_qc("plate1_A01_f1", flags=[FLAG_EMPTY], n_objects=0,
                       severity="fail"),
                   _qc("plate1_A02_f1", n_objects=50)],
        summary={"verdict": "fail", "n_fail": 1, "n_warn": 1})

    digest = seg_qc._digest_from_cards("/runs", [card])

    assert digest.verdict == "fail"
    assert "1 of 2 cell field(s) failed and 1 need a look" in digest.subhead


def test_a_card_whose_date_cannot_be_read_is_dated_zero(tmp_path, monkeypatch):
    """A card on a filesystem that will not stat is still readable content.

    Dating it zero means it is treated as older than its masks -- reported as
    out of date rather than as clean, which is the safe direction.
    """
    root = tmp_path / "plate1"
    _write_card(str(root / CARD_DIR / f"{CARD_PREFIX}cell.csv"), [
        {"field": "plate1_A01_f1", "object_type": "cell", "n_objects": "5",
         "severity": "ok", "flags": "", "note": ""},
    ])
    real_stat = os.stat
    seen: set = set()

    def refuse(path, *args, **kwargs):
        # The card is stat-ed once to check it is a file and again to date it.
        # Only the dating call fails, which is the race a card being rewritten
        # underneath the reader produces.
        if str(path).endswith(".csv"):
            if str(path) in seen:
                raise PermissionError(path)
            seen.add(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", refuse)

    digest = seg_qc.read_digest(str(root))

    assert [card.mtime for card in digest.scorecards] == [0.0]


def test_scoring_a_project_can_be_limited_to_one_object_type(tmp_path):
    """Re-scoring every stack on a plate is the expensive path.

    A user who asks about cells must not pay for the pathogen masks beside
    them.
    """
    root = tmp_path / "plate1"
    for object_type in ("cell", "pathogen"):
        folder = root / f"{object_type}_mask_stack"
        folder.mkdir(parents=True)
        np.save(folder / "plate1_A01_f1.npy", _grid_field().astype(np.uint16))

    digest = seg_qc.score_digest(str(root), object_types=["cell"], write=False)

    assert [card.object_type for card in digest.scorecards] == ["cell"]


def test_a_stack_folder_holding_no_masks_contributes_no_card(tmp_path):
    """An empty stack folder is what ``save=False`` leaves behind.

    A card with no rows would be read back later as a plate that scored
    clean, so nothing is recorded for it at all.
    """
    root = tmp_path / "plate1"
    (root / "cell_mask_stack").mkdir(parents=True)

    digest = seg_qc.score_digest(str(root), write=False)

    assert digest.scorecards == []
    assert digest.verdict == "missing"


def test_a_digest_printout_shows_a_card_that_could_not_be_read(tmp_path):
    """The printed digest is what the console shows and what a test reads.

    A card in error has no summary message, and printing an empty one would
    make the failure look like a clean line.
    """
    digest = QCDigest(
        root="/runs", verdict="error", headline="could not be read",
        scorecards=[Scorecard(path="/runs/qc/segmentation_qc_cell.csv",
                              object_type="cell",
                              error="segmentation_qc_cell.csv is not CSV",
                              stale=True)])

    printed = format_digest(digest)

    assert "cell: segmentation_qc_cell.csv is not CSV" in printed
    assert "OUT OF DATE" in printed


def test_a_field_of_small_numerous_objects_is_called_split_apart(tmp_path):
    """Smaller AND more numerous than the plate is the shatter signature.

    Neither half is enough on its own: a field with more objects is ordinary
    well-to-well variation, and a field of smaller objects is a different cell
    type or a different focal plane. Together, against a plate that agrees
    with itself, they are one population cut into pieces -- and the note has
    to carry both ratios, because that pairing is the whole argument.

    The per-field rules leave this field alone (its objects are well over
    ``min_diameter``), so the flag can only come from the plate comparison.
    """
    fields = {f"plate1_A0{i}_f1": _grid_field(shape=(400, 400), radius=10,
                                              spacing=60, margin=40)
              for i in range(1, 6)}
    shattered = _grid_field(shape=(400, 400), radius=4, spacing=25, margin=20)
    fields["plate1_A06_f1"] = shattered

    assert score_field(shattered, object_type="cell", field="x").flags == [], (
        "the field is already flagged without the plate for comparison")

    scored = {q.field: q for q in score_masks(fields, object_type="cell")}
    odd = scored["plate1_A06_f1"]

    assert seg_qc.FLAG_OVER in odd.flags
    assert "the signature of objects split apart" in odd.note
    assert "0.39x the plate's median diameter" in odd.note
    assert "6.2x its count" in odd.note
    # The fields it is being compared against stay clean.
    assert all(not scored[f"plate1_A0{i}_f1"].flags for i in range(1, 6))
