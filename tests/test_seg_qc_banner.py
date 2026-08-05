"""The verdict a user reads: plain language, plate addresses, and freshness.

``tests/test_seg_qc.py`` covers the scoring — does a fused field get called
fused. This file covers everything built on top of it for the Measure banner,
and the questions here are different ones:

* **Does it say something a user can act on?** "3 plates failed QC" is nearly
  useless. Every flag has to come with what it does to the measurements, what
  usually causes it and what to do, and every finding has to name the plate
  and the wells it is about.
* **Does it see what no single field can?** The motivating case is a plate
  whose object count steps four-fold between one half of its rows and the
  other while every individual field scores perfectly clean. That is uneven
  illumination or a threshold set too low, and it is invisible per field.
* **Is it cheap?** ``read_digest`` must read the card the mask run already
  wrote and must not score a mask. :func:`test_read_digest_never_opens_a_mask`
  proves it the only way that cannot rot: by making scoring raise.
* **Is it honest about being out of date?** A card older than the masks it
  describes is not a verdict about the masks on disk, and must not be shown
  as one.

Every fixture below is a real label mask with a defect put there on purpose.
CPU-only, offline, deterministic, unmarked.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from spacr import seg_qc
from spacr.seg_qc import (
    FLAG_BORDER,
    FLAG_EMPTY,
    FLAG_GUIDANCE,
    FLAGS,
    diagnose,
    explain_flag,
    find_mask_stacks,
    find_scorecards,
    format_digest,
    mask_stack_mtime,
    parse_field_name,
    qc_roots,
    read_digest,
    read_scorecard,
    score_digest,
    score_masks,
)


# ---------------------------------------------------------------------------
# mask builders — the defects are deliberate
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    """Paint one filled disc of label ``value``."""
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sub = labels[y0:y1, x0:x1]
    sub[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = value
    return labels


def _field_of_n(n, shape=(512, 512), radius=6, seed=0):
    """Exactly ``n`` well-separated discs, none of them touching the border.

    Laid out on a jittered grid so the count is exact and the size
    distribution is not degenerate.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int32)
    margin = radius + 4
    step = 2 * radius + 8
    slots = [
        (cy, cx)
        for cy in range(margin, shape[0] - margin, step)
        for cx in range(margin, shape[1] - margin, step)
    ]
    if n > len(slots):
        raise AssertionError(f"cannot fit {n} discs in a {shape} field")
    picked = rng.permutation(len(slots))[:n]
    for value, index in enumerate(sorted(picked), start=1):
        cy, cx = slots[index]
        _disc(labels, cy, cx, radius, value)
    return labels


def _border_heavy_field(shape=(300, 300), radius=8):
    """Sixteen objects on the edge and five in the middle: 76% truncated."""
    labels = np.zeros(shape, np.int32)
    value = 0
    step = (shape[1] - 2 * radius) // 8
    for k in range(8):
        value += 1
        _disc(labels, 0, radius + k * step, radius, value)
        value += 1
        _disc(labels, shape[0] - 1, radius + k * step, radius, value)
    for k in range(5):
        value += 1
        _disc(labels, shape[0] // 2, 40 + k * 50, radius, value)
    return labels


def _write_plate(root, fields, object_type="cell", inner="norm_channel_stack"):
    """Write ``{field_name: mask}`` where ``spacr.object`` writes them.

    ``<plate>/<inner>/<object_type>_mask_stack/<field>.npy`` — one level below
    the plate folder the scorecard is written at, which is the layout
    ``spacr.object._run_seg_qc`` produces and the one the readers must find.
    """
    folder = Path(root) / inner / f"{object_type}_mask_stack"
    folder.mkdir(parents=True, exist_ok=True)
    for name, mask in fields.items():
        np.save(folder / f"{name}.npy", mask.astype(np.uint16))
    return str(folder)


def _even_plate(plate="plate1", n=20, rows="ABCDEFGH", seed=0):
    """Every well the same: no gradient, nothing to flag."""
    fields = {}
    for i, row in enumerate(rows):
        for col in (1, 2):
            fields[f"{plate}_{row}{col:02d}_1"] = _field_of_n(
                n, seed=seed + i * 10 + col)
    return fields


def _stepped_plate(plate="plate2", low=10, high=40, rows="ABCDEFGH", seed=500):
    """Rows E-H hold four times the objects of rows A-D. Nothing else is wrong.

    This is the case the whole positional half of the module exists for: not
    one of these fields is individually remarkable, and the plate is broken.
    """
    fields = {}
    for i, row in enumerate(rows):
        n = high if row in "EFGH" else low
        for col in (1, 2):
            fields[f"{plate}_{row}{col:02d}_1"] = _field_of_n(
                n, seed=seed + i * 10 + col)
    return fields


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stepped_project(tmp_path):
    """A scored plate whose rows E-H hold 4x the objects of rows A-D."""
    root = tmp_path / "plate2"
    _write_plate(root, _stepped_plate())
    score_digest(str(root))
    return str(root)


@pytest.fixture
def clean_project(tmp_path):
    """A scored plate with nothing wrong with it."""
    root = tmp_path / "plate1"
    _write_plate(root, _even_plate())
    score_digest(str(root))
    return str(root)


# ---------------------------------------------------------------------------
# 1. plain language
# ---------------------------------------------------------------------------

def test_every_flag_seg_qc_can_raise_has_a_plain_language_entry():
    """A flag with no explanation is a nine-letter identifier a user must guess."""
    assert set(FLAGS) == set(FLAG_GUIDANCE), (
        f"unexplained: {sorted(set(FLAGS) - set(FLAG_GUIDANCE))}"
    )
    for flag in FLAGS:
        guidance = explain_flag(flag)
        assert guidance.severity in ("warn", "fail")
        assert guidance.severity == seg_qc._FLAG_SEVERITY[flag], (
            f"{flag}: the explanation and the scorer disagree about severity"
        )
        assert guidance.headline and not guidance.headline.endswith(".")
        # The three questions a user actually has.
        assert len(guidance.means) > 40, f"{flag}: no 'what does it mean'"
        assert len(guidance.causes) >= 2, f"{flag}: fewer than two causes"
        assert len(guidance.fix) > 30, f"{flag}: no 'what do I do'"
        assert guidance.text().count("(1)") == 1


def test_illumination_is_named_exactly_where_it_is_a_cause():
    """The pointer to spacr.illumination is a claim, not decoration.

    Where uneven illumination really is one of the usual causes the fix has to
    say so and name the module. Where it is not, saying so anyway would train
    the user to ignore it.
    """
    named = {flag for flag, g in FLAG_GUIDANCE.items() if g.illumination}
    assert named == {
        seg_qc.FLAG_LOW_COUNT,
        seg_qc.FLAG_HIGH_COUNT,
        seg_qc.FLAG_OUTLIERS,
    }
    for flag, guidance in FLAG_GUIDANCE.items():
        mentions = "spacr.illumination" in guidance.fix
        assert mentions is guidance.illumination, (
            f"{flag}: illumination={guidance.illumination} but "
            f"{'mentions' if mentions else 'does not mention'} the module"
        )
    # And it must not over-promise: the correction runs in the measurement
    # path, so it fixes intensity features and not object counts.
    assert "counts only change" in seg_qc.ILLUMINATION_ADVICE


def test_the_cellpose_diameter_rescaling_is_explained_where_it_matters():
    """Under- and over-segmentation both trace back to one number."""
    for flag in (seg_qc.FLAG_UNDER, seg_qc.FLAG_OVER, seg_qc.FLAG_EMPTY):
        text = explain_flag(flag).text()
        assert "30/diameter" in text, f"{flag} does not explain the rescaling"


# ---------------------------------------------------------------------------
# 2. where on the plate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,plate,well,row,column", [
    ("plate1_E07_3", "plate1", "E07", "E", 7),
    ("plate1_E07_3.npy", "plate1", "E07", "E", 7),
    ("/data/x/plate1_A01_1_t0.npy", "plate1", "A01", "A", 1),
    ("exp2_AA12_5", "exp2", "AA12", "AA", 12),
])
def test_parse_field_name_reads_the_plate_and_the_well(name, plate, well, row, column):
    address = parse_field_name(name)
    assert (address.plate, address.well, address.row, address.column) == (
        plate, well, row, column)
    assert address.known


@pytest.mark.parametrize("name", ["field_0000", "handmade", "plate1_notawell_1"])
def test_a_name_without_a_well_makes_no_positional_claim(name):
    """Never guess. A folder assembled by hand still has to work."""
    address = parse_field_name(name)
    assert not address.known
    assert address.well == "" and address.row == "" and address.column is None
    assert address.plate


# ---------------------------------------------------------------------------
# 3. THE case: a plate that steps, whose every field is clean
# ---------------------------------------------------------------------------

def test_a_fourfold_row_step_is_named_with_its_plate_its_rows_and_its_cause():
    """The finding this module exists for.

    plate2 rows E-H hold four times the objects of rows A-D. Every field is
    individually unremarkable — that is the point — so nothing per-field can
    raise this, and a user told only "the plate is fine" would run Measure on
    it and get a row effect that looks like biology.
    """
    field_qcs = score_masks(
        {name: mask for name, mask in _stepped_plate().items()},
        object_type="cell",
    )
    assert all(q.severity == "ok" for q in field_qcs), (
        "the fixture is meant to be clean field by field: "
        f"{[str(q) for q in field_qcs if q.severity != 'ok']}"
    )

    findings = diagnose(field_qcs)
    steps = [f for f in findings if f.kind == "count_gradient"]
    assert len(steps) == 1, [f.headline for f in findings]
    step = steps[0]

    assert step.plate == "plate2"
    assert "plate2" in step.headline
    assert "rows E-H" in step.headline, step.headline
    assert "rows A-D" in step.headline, step.headline
    assert "4.0x" in step.headline, step.headline
    assert step.severity == "fail"

    # The cause, in the words the user needs, and both of the two suspects.
    assert "illumination" in step.detail
    assert "threshold" in step.detail
    assert step.illumination is True
    assert "spacr.illumination" in step.fix
    # And the wells on the implicated half, so there is somewhere to look.
    assert set(step.wells) >= {"E01", "F01", "G01", "H01"}


def test_an_even_plate_raises_no_positional_finding():
    """The other direction: a QC report that cries wolf is one nobody reads."""
    field_qcs = score_masks(_even_plate(), object_type="cell")
    assert diagnose(field_qcs) == []


def test_a_step_below_the_ratio_is_seeding_variation_and_is_not_reported():
    """Seeding varies with a CV of 10-30%; 1.5x is not a gradient."""
    fields = {}
    for i, row in enumerate("ABCDEFGH"):
        n = 30 if row in "EFGH" else 20
        for col in (1, 2):
            fields[f"plate3_{row}{col:02d}_1"] = _field_of_n(n, seed=i * 7 + col)
    findings = diagnose(score_masks(fields, object_type="cell"))
    assert [f for f in findings if f.kind == "count_gradient"] == []


def test_a_column_step_is_found_on_the_column_axis():
    """Edge columns are as common a defect as edge rows."""
    fields = {}
    for row in "ABCD":
        for col in range(1, 9):
            n = 40 if col >= 5 else 10
            fields[f"plate4_{row}{col:02d}_1"] = _field_of_n(
                n, seed=col * 13 + ord(row))
    steps = [f for f in diagnose(score_masks(fields, object_type="cell"))
             if f.kind == "count_gradient"]
    assert len(steps) == 1
    assert "columns 5-8" in steps[0].headline, steps[0].headline
    assert "columns 1-4" in steps[0].headline


# ---------------------------------------------------------------------------
# 4. per-flag findings name plates and wells
# ---------------------------------------------------------------------------

def test_a_flag_finding_names_the_plate_the_wells_and_what_to_do():
    """Two wells of one plate are full of truncated objects; the rest are fine."""
    fields = _even_plate(plate="plate7", n=20)
    for well in ("C01", "C02"):
        fields[f"plate7_{well}_1"] = _border_heavy_field()

    findings = diagnose(score_masks(fields, object_type="cell"))
    border = [f for f in findings if f.flag == FLAG_BORDER]
    assert len(border) == 1
    finding = border[0]

    assert finding.plate == "plate7"
    assert finding.n_fields == 2
    assert set(finding.wells) == {"C01", "C02"}
    assert "C01" in finding.headline and "C02" in finding.headline
    assert finding.severity == "warn"
    assert "truncated" in finding.detail or "cut off" in finding.detail
    assert "remove_border_objects" in finding.fix


def test_empty_fields_are_named_and_explained_as_a_channel_or_diameter_problem():
    fields = _even_plate(plate="plate8", n=20)
    fields["plate8_D01_1"] = np.zeros((512, 512), np.int32)
    findings = diagnose(score_masks(fields, object_type="cell"))
    empty = [f for f in findings if f.flag == FLAG_EMPTY]
    assert len(empty) == 1
    assert empty[0].wells == ("D01",)
    assert empty[0].severity == "fail"
    assert "channel" in empty[0].detail


def test_findings_come_back_worst_first():
    fields = _even_plate(plate="plate9", n=20)
    fields["plate9_D01_1"] = np.zeros((512, 512), np.int32)
    fields["plate9_C01_1"] = _border_heavy_field()
    findings = diagnose(score_masks(fields, object_type="cell"))
    severities = [f.severity for f in findings]
    assert severities == sorted(
        severities, key=lambda s: -seg_qc._SEVERITY_ORDER.index(s))


# ---------------------------------------------------------------------------
# 5. reading the card back — the cheap path
# ---------------------------------------------------------------------------

def test_score_digest_writes_the_card_where_report_and_object_expect_it(tmp_path):
    root = tmp_path / "plate1"
    _write_plate(root, _even_plate())
    digest = score_digest(str(root))

    expected = root / "qc" / "segmentation_qc_cell.csv"
    assert expected.is_file()
    assert digest.scorecards[0].path == str(expected)
    assert find_scorecards(str(root)) == (str(expected),)
    assert find_mask_stacks(str(root)) == {
        "cell": str(root / "norm_channel_stack" / "cell_mask_stack")}


def test_read_digest_never_opens_a_mask(clean_project, monkeypatch):
    """The whole cost argument, asserted rather than asserted-in-a-docstring.

    A screen that rescored a plate on every visit would be switched off. If
    ``read_digest`` ever starts scoring, this test fails immediately.
    """
    def _explode(*_a, **_k):
        raise AssertionError("read_digest scored a mask")

    monkeypatch.setattr(seg_qc, "score_masks", _explode)
    monkeypatch.setattr(seg_qc, "score_field", _explode)
    monkeypatch.setattr(np, "load", _explode)

    digest = read_digest(clean_project)
    assert digest.verdict == "ok"
    assert digest.n_fields == 16


def test_a_clean_project_reads_back_as_a_clean_verdict(clean_project):
    digest = read_digest(clean_project)
    assert digest.verdict == "ok"
    assert digest.findings == []
    assert digest.stale is False
    assert "passed" in digest.headline
    assert "none flagged" in digest.subhead
    assert digest.object_types == ("cell",)
    assert digest.plates == ("plate1",)
    assert digest.failing_fields == ()


def test_a_stepped_project_reads_back_naming_the_plate_and_the_rows(stepped_project):
    digest = read_digest(stepped_project)
    assert digest.verdict == "fail"
    assert "plate2" in digest.headline
    assert "rows E-H" in digest.headline
    # No field failed on its own; the banner has to say so rather than
    # contradict the per-field card.
    assert "no single field was flagged" in digest.subhead
    assert "spacr.illumination" in format_digest(digest)


def test_a_verdict_read_back_is_the_verdict_the_run_printed(clean_project):
    """Same rows, same summarize_qc — not a second opinion that could differ."""
    card = read_digest(clean_project).scorecards[0]
    rows, error = read_scorecard(card.path)
    assert error == ""
    assert [q.field for q in rows] == [q.field for q in card.field_qcs]
    assert card.summary["verdict"] == seg_qc.summarize_qc(rows)["verdict"]


# ---------------------------------------------------------------------------
# 6. freshness
# ---------------------------------------------------------------------------

def _age_card(root, seconds=600):
    """Backdate the scorecard, i.e. re-mask the plate without re-scoring it."""
    card = os.path.join(root, "qc", "segmentation_qc_cell.csv")
    when = os.stat(card).st_mtime - seconds
    os.utime(card, (when, when))
    return card


def test_a_card_older_than_its_masks_is_reported_out_of_date(clean_project):
    """Re-masking without re-scoring must not leave a stale verdict standing."""
    assert read_digest(clean_project).stale is False

    _age_card(clean_project)

    digest = read_digest(clean_project)
    assert digest.stale is True
    assert digest.scorecards[0].stale is True
    assert digest.scorecards[0].masks_mtime > digest.scorecards[0].mtime
    assert "OUT OF DATE" in format_digest(digest)


def test_a_card_written_moments_after_its_masks_is_not_called_stale(clean_project):
    """Equal-ish mtimes are the normal case, not a staleness."""
    stack = find_mask_stacks(clean_project)["cell"]
    card = os.path.join(clean_project, "qc", "segmentation_qc_cell.csv")
    when = os.stat(card).st_mtime
    for name in os.listdir(stack):
        os.utime(os.path.join(stack, name), (when, when))
    assert read_digest(clean_project).stale is False


def test_mask_stack_mtime_is_zero_for_a_folder_that_is_not_there(tmp_path):
    assert mask_stack_mtime(str(tmp_path / "nope")) == 0.0


def test_scoring_again_clears_the_staleness(clean_project):
    """Which is what makes the banner's one expensive button worth pressing."""
    _age_card(clean_project)
    assert read_digest(clean_project).stale is True

    assert score_digest(clean_project).stale is False
    assert read_digest(clean_project).stale is False


# ---------------------------------------------------------------------------
# 7. the shapes src actually takes, and the failure modes
# ---------------------------------------------------------------------------

def test_the_merged_folder_measure_is_pointed_at_finds_the_plates_card(clean_project):
    """Measure's src is usually ``<plate>/merged``; the card is one level up."""
    merged = Path(clean_project) / "merged"
    merged.mkdir()
    assert clean_project in qc_roots(str(merged))
    assert read_digest(str(merged)).verdict == "ok"


def test_a_project_root_of_plates_is_walked_one_level(tmp_path):
    for plate, fields in (("plate1", _even_plate("plate1")),
                          ("plate2", _stepped_plate("plate2"))):
        _write_plate(tmp_path / plate, fields)
        score_digest(str(tmp_path / plate))

    digest = read_digest(str(tmp_path))
    assert len(digest.scorecards) == 2
    assert digest.plates == ("plate1", "plate2")
    assert digest.verdict == "fail"
    assert "plate2" in digest.headline


def test_a_list_of_plates_is_read_as_a_list(tmp_path):
    roots = []
    for plate in ("plateA", "plateB"):
        root = tmp_path / plate
        _write_plate(root, _even_plate(plate))
        score_digest(str(root))
        roots.append(str(root))
    assert read_digest(roots).n_fields == 32


@pytest.mark.parametrize("src", [None, "", "path", "/path/to/src", 17, []])
def test_no_source_is_missing_and_never_raises(src):
    digest = read_digest(src)
    assert digest.verdict == "missing"
    assert digest.findings == []


def test_no_card_is_missing_which_is_not_the_same_as_clean(tmp_path):
    """The distinction that matters: nothing has looked at these masks."""
    _write_plate(tmp_path / "plate1", _even_plate())
    digest = read_digest(str(tmp_path / "plate1"))
    assert digest.verdict == "missing"
    assert "not the same as clean" in digest.headline


def test_a_damaged_card_is_an_error_not_an_invented_verdict(clean_project):
    """Half a scorecard is a different verdict; this module does not guess."""
    card = Path(clean_project) / "qc" / "segmentation_qc_cell.csv"
    card.write_bytes(b"field,object_type\x00,n_objects\nbroken")
    digest = read_digest(clean_project)
    assert digest.verdict == "error"
    assert digest.scorecards[0].error
    assert "could not be read" in digest.headline


# ---------------------------------------------------------------------------
# 8. it informs, it does not block
# ---------------------------------------------------------------------------

def test_a_digest_never_blocks_a_run(stepped_project):
    """The worst verdict this module can reach still gates nothing."""
    digest = read_digest(stepped_project)
    assert digest.verdict == "fail"
    assert digest.blocks_run is False
    assert read_digest(None).blocks_run is False
