"""Instruction 145's ratchet — a reader that does not canonicalise.

The size of the problem, counted when 145 was filed: 248 tabular reads and
writes across `spacr/`, of which 13 normalised column names. There is no
funnel; there are 248 doors. `spacr/tabular.py` is the funnel and a large
number of call sites now go through it, but the rest cannot be converted in
one pass and a partial conversion silently un-converts itself as new code is
written.

THE ARGUMENT FOR A RATCHET IS IN 145's OWN FINDINGS. Four readers that did not
canonicalise were found on the maintainer's screen, and NOT ONE OF THEM
FAILED. Each returned a number:

* `fractions_from_counts` pooled four plates' `r1/c1` into one well — 384
  wells instead of 1,536 — and the fractions still summed to 1;
* `load_montage_objects` composed no well key on three plates of four and the
  caption said "no object comes from this well" for wells holding 244;
* the score CSVs' `pplate1` matched nothing and the all-NaN result was
  reported as "0 of 5,959 hits are circular", which is the most confident
  possible way to say nothing.

A reader that does not canonicalise does not raise. That is why counting is
worth doing at all: the defect has no symptom.

WHAT THIS TEST IS AND IS NOT. It is not a coverage target and it does not
demand zero — some of these are genuinely raw (a settings CSV, a log, a file
with no metadata columns). It says the number MAY NOT GO UP. Lowering the
ceiling as sites are converted is the point; a new raw reader has to either
go through `spacr.tabular` or be justified in this file, where the next
person can read the justification.
"""
from __future__ import annotations

import collections
import pathlib
import re

import pytest

#: Every way a module opens or writes a table without saying anything about
#: its column names.
RAW_CALL = re.compile(r"pd\.read_csv\(|pd\.read_sql\w*\(|\.to_csv\(|\.to_sql\(")

#: The ceiling, measured 2026-08-20. LOWER IT when sites are converted; never
#: raise it without saying here why the new call cannot go through
#: `spacr.tabular`.
#:
#: It was 248 when 145 was filed and the funnel did not exist. The number
#: below is higher than that only because it counts writes and reads in
#: modules the original census did not reach, not because anything regressed;
#: what matters from here is the direction.
#:
#: 261 -> 254 on 2026-08-20: seven readers in `submodules.py` moved onto
#: `tabular.read_table`, chosen for damage rather than for count. Two of them
#: sat directly above a comment recording that the helper "was left half-way
#: through the column_name -> columnID rename" -- it filtered on one spelling
#: and grouped on the other, so the frame it produced could never match. That
#: is what canonicalising at the read fixes, and it is why counting is worth
#: doing: the reader did not FAIL, it returned an empty answer.
#:
#: 257 -> 253 on 2026-08-22: classifier test-split input and the reproducible
#: figure-bundle and streaming-dataset outputs now use the shared reader or
#: writer. These are scientific tables with schema-bearing columns, so there
#: is no reason for them to bypass canonicalisation.
CEILING = 253

#: Files allowed to hold raw calls without argument, and why.
EXPECTED_HOMES = {
    # THE FUNNEL ITSELF. `read_table` and `write_table` are where the raw
    # calls are SUPPOSED to be -- that is what makes them a funnel.
    "tabular.py",
}


def _spacr_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "spacr"


def _counts() -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for path in sorted(_spacr_root().rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        found = len(RAW_CALL.findall(path.read_text(encoding="utf-8")))
        if found:
            counts[str(path.relative_to(_spacr_root()))] = found
    return counts


def test_the_ratchet_is_reading_the_tree_it_thinks_it_is():
    """A checker pointed at nothing passes forever."""
    root = _spacr_root()
    assert (root / "tabular.py").is_file()
    assert sum(_counts().values()) > 0


def test_the_number_of_raw_tabular_calls_does_not_go_up():
    counts = _counts()
    total = sum(counts.values())
    assert total <= CEILING, (
        f"{total} raw tabular calls, up from {CEILING}. A reader that does "
        f"not canonicalise does not FAIL -- it returns a number (145's four "
        f"findings all did). Route it through spacr.tabular.read_table / "
        f"write_table, or lower this ceiling deliberately and say why.\\n"
        + "\\n".join(f"  {n:3}  {name}" for name, n in counts.most_common(12)))


def test_the_ceiling_is_not_left_far_above_the_truth():
    """A ratchet with slack in it is not a ratchet.

    If the count has fallen well below the ceiling, the ceiling should come
    down with it -- otherwise the gap is room for a new raw reader to be added
    without anybody noticing.
    """
    total = sum(_counts().values())
    assert total > CEILING - 15, (
        f"only {total} raw calls against a ceiling of {CEILING}: lower "
        f"CEILING to {total} so the ratchet keeps holding")


def test_the_funnel_is_where_the_raw_calls_belong():
    counts = _counts()
    assert counts.get("tabular.py", 0) > 0, (
        "spacr/tabular.py holds no raw tabular call, which means it is not "
        "the funnel any more and something else is")


def test_the_reader_and_the_writer_are_importable_and_canonicalise(tmp_path):
    """The thing the ratchet is pushing everything towards."""
    pd = pytest.importorskip("pandas")

    from spacr.tabular import read_table, write_table

    path = tmp_path / "counts.csv"
    pd.DataFrame({"Plate": ["p1"], "row_name": ["r1"], "COLUMN": ["c1"],
                  "count": [7]}).to_csv(path, index=False)

    frame = read_table(path)
    assert {"plateID", "rowID", "columnID"} <= set(frame.columns)
    assert "row_name" not in frame.columns

    out = tmp_path / "out.csv"
    write_table(frame, out)
    assert {"plateID", "rowID", "columnID"} <= set(read_table(out).columns)
