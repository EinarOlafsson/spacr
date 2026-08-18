"""The bundled Toxoplasma annotation reaches an exported table, once per row.

Instruction 133, asked for on 2026-08-17: "if it is on all the exported tables
should be merged with the relevant Toxoplasma information (gene name, signal
peptide, transmembrane domain, the phenotype scores from the screens we have
downloaded, tachyzoite expression, bradyzoite expression, sexual stages
expression, hyperLopit expression)".

THE TEST THAT MATTERS MOST IS THE ROW COUNT. This project has already shipped
a join that turned one coefficient into four rows on a user's volcano, and it
looked entirely plausible while it was wrong. Every case here checks that the
annotated table is exactly as long as the one that went in.
"""

import pandas as pd
import pytest

from spacr import annotation


@pytest.mark.parametrize("value, expected", [
    # The four spellings that reach this from the screen itself.
    ("TGGT1_224750", "224750"),          # the guide library's strain
    ("TGME49_224750", "224750"),         # every annotation table's strain
    ("gene_fraction:gene[224750]", "224750"),   # a patsy gene term
    ("fraction:grna[224750_2]", "224750"),      # a patsy guide term
    ("224750_2", "224750"),              # a bare guide id
    ("224750", "224750"),                # the key itself
    ("TGME49_201180A", "201180"),        # a split gene model -> its parent
    ("TGME49_201180B", "201180"),
    # And everything that names no gene. None, not a guess.
    ("Intercept", None),
    ("rowID[T.r03]", None),
    ("columnID[T.c11]", None),
    ("", None),
    ("nan", None),
    ("<NA>", None),
    (None, None),
    (float("nan"), None),
])
def test_every_spelling_of_a_gene_reaches_one_key(value, expected):
    assert annotation.gene_number(value) == expected


def test_the_prefix_digit_is_not_the_gene():
    """`TGGT1` ends in a 1 and `224750_2` ends in a 2; neither is the gene.

    The floor of four digits is what rejects both, and it is the whole reason
    the parse is a shared function rather than a `split("_")` at each site.
    """
    assert annotation.gene_number("TGGT1_224750") == "224750"
    assert annotation.gene_number("TGGT1") is None


def _coefficients():
    return pd.DataFrame({
        "feature": ["gene_fraction:gene[224750]",
                    "fraction:grna[201180_2]",
                    "fraction:grna[224750_1]",
                    "Intercept"],
        "coefficient": [1.0, 2.0, 3.0, 4.0],
    })


def test_the_join_never_multiplies_rows():
    """One row in, one row out -- including for a split gene model.

    `201180` is `TGME49_201180`, `_201180A` and `_201180B` in the published
    phenotype table. All three collapse to one key, so an unguarded join
    would return three rows where the export has one.
    """
    frame = _coefficients()
    out = annotation.annotate(frame, quiet=True)
    assert len(out) == len(frame)
    assert list(out["feature"]) == list(frame["feature"])
    assert list(out["coefficient"]) == list(frame["coefficient"])


def test_the_annotation_arrives_and_says_something_true():
    out = annotation.annotate(_coefficients(), quiet=True)
    row = out.loc[out["feature"] == "fraction:grna[201180_2]"].iloc[0]
    # MSF: a signal peptide AND three transmembrane helices, in the dense
    # granules. Both booleans come from the one `dtm_type` field.
    assert bool(row["signal_peptide"]) is True
    assert bool(row["transmembrane"]) is True
    assert row["n_transmembrane"] == 3
    assert row["topology"] == "SP+TM"
    assert row["hyperlopit"] == "dense granules"
    assert row["gene_name"] == "MSF"
    # And the phenotype scores from the published screens.
    assert pd.notna(row["fit_invitro_hff"])
    assert pd.notna(row["fit_ifng"])


def test_a_term_that_names_no_gene_gets_no_annotation():
    """The intercept is not a gene, so its annotation cells are empty.

    Empty, and still present in the row -- the alternative is dropping the
    intercept from the export, and a coefficient table without its intercept
    is not the table the run produced.
    """
    out = annotation.annotate(_coefficients(), quiet=True)
    row = out.loc[out["feature"] == "Intercept"].iloc[0]
    assert pd.isna(row["gene_name"])
    assert pd.isna(row["hyperlopit"])


def test_every_requested_field_is_among_the_columns():
    """The request named eight things. Each has a column."""
    names = annotation.columns()
    for wanted in ("gene_name", "signal_peptide", "transmembrane",
                   "fit_invitro_hff", "expr_tachyzoite", "expr_bradyzoite",
                   "expr_ees1", "hyperlopit"):
        assert wanted in names, wanted


def test_the_five_sexual_stages_are_five_columns_not_an_average():
    """EES1-5 arrive as five columns.

    A mean of the five would be a number no published table contains, and a
    reader who wanted it can take it; a reader handed it cannot get the five
    back.
    """
    names = annotation.columns()
    assert [n for n in names if n.startswith("expr_ees")] == [
        "expr_ees1", "expr_ees2", "expr_ees3", "expr_ees4", "expr_ees5"]


def test_the_input_table_is_not_touched():
    frame = _coefficients()
    before = frame.copy()
    annotation.annotate(frame, quiet=True)
    pd.testing.assert_frame_equal(frame, before)


def test_a_table_naming_no_gene_comes_back_unchanged(capsys):
    """Not a block of empty columns, which reads as "found nothing"."""
    frame = pd.DataFrame({"plate": ["p1", "p2"], "value": [1.0, 2.0]})
    out = annotation.annotate(frame)
    pd.testing.assert_frame_equal(out, frame)
    assert "names no gene column" in capsys.readouterr().out


def test_a_column_the_table_already_has_is_not_overwritten(capsys):
    """A run that computed its own `gene_name` keeps it."""
    frame = _coefficients()
    frame["gene_name"] = ["mine"] * len(frame)
    out = annotation.annotate(frame)
    assert list(out["gene_name"]) == ["mine"] * len(frame)
    # The rest of that source still arrives; only the clashing column is held
    # back, and the console says so.
    assert "expr_tachyzoite" in out.columns
    assert "already on this table" not in capsys.readouterr().out


def test_a_source_entirely_present_is_skipped_with_a_reason(capsys):
    frame = _coefficients()
    for name in ("hyperlopit",):
        frame[name] = "mine"
    annotation.annotate(frame)
    assert "already on this table" in capsys.readouterr().out


def test_an_explicit_gene_column_is_preferred_over_the_design_term():
    """A table carrying both has already parsed the term once.

    Joining on the term as well is how two columns of the same export start
    naming different genes.
    """
    frame = pd.DataFrame({
        "feature": ["gene_fraction:gene[224750]"],
        "gene": ["TGME49_201180"],
        "coefficient": [1.0],
    })
    out = annotation.annotate(frame, quiet=True)
    assert out.iloc[0]["gene_name"] == "MSF"


def test_the_key_column_can_be_named_outright():
    frame = pd.DataFrame({"a": ["TGME49_201180"], "b": ["TGME49_224750"]})
    out = annotation.annotate(frame, key_column="b", quiet=True)
    assert out.iloc[0]["gene_name"] == "SRS40F"


def test_a_named_key_column_that_is_not_there_falls_back(capsys):
    frame = pd.DataFrame({"plate": ["p1"]})
    out = annotation.annotate(frame, key_column="nope")
    pd.testing.assert_frame_equal(out, frame)
    assert "names no gene column" in capsys.readouterr().out


def test_an_empty_table_is_returned_as_is():
    empty = pd.DataFrame()
    assert annotation.annotate(empty) is empty
    assert annotation.annotate(None) is None


def test_a_missing_bundled_table_is_absent_with_a_reason(monkeypatch, capsys):
    """Not a column of NaN. The console names the file and the export shrinks."""
    annotation.clear_cache()
    monkeypatch.setattr(annotation, "_DATA", "/nonexistent/spacr/data")
    try:
        out = annotation.annotate(_coefficients())
        printed = capsys.readouterr().out
        assert "lopit.csv is not bundled" in printed
        assert "deeptmhmm.csv is not bundled" in printed
        assert "no bundled table could be read" in printed
        assert "hyperlopit" not in out.columns
        assert "signal_peptide" not in out.columns
        assert len(out) == 4
        assert annotation.columns() == []
    finally:
        annotation.clear_cache()


def test_an_unreadable_bundled_table_is_absent_with_a_reason(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    annotation.clear_cache()
    (tmp_path / "lopit.csv").write_bytes(b'"a,b\n1')
    monkeypatch.setattr(annotation, "_DATA", str(tmp_path))
    try:
        annotation.annotate(_coefficients())
        assert "lopit.csv could not be read" in capsys.readouterr().out
    finally:
        annotation.clear_cache()


def test_a_source_missing_its_key_or_its_columns_is_dropped():
    assert annotation._keyed(None, "gene_nr", (("a", "a"),)) is None
    frame = pd.DataFrame({"gene_nr": ["224750"], "other": [1]})
    assert annotation._keyed(frame, "gene_nr", (("absent", "a"),)) is None
    assert annotation._keyed(frame, "absent", (("other", "a"),)) is None


def test_the_console_line_counts_what_matched(capsys):
    annotation.annotate(_coefficients())
    printed = capsys.readouterr().out
    # Three of the four terms name a gene; the intercept does not.
    assert "onto 3 of 4 row(s)" in printed


def test_the_bundled_tables_have_one_row_per_gene():
    """The guard the `many_to_one` merge relies on, checked directly.

    A bundled table that grew a duplicate key would raise inside `annotate`
    with a pandas message about merge validation. This says which file.
    """
    for label, source in annotation.SOURCES:
        frame = source()
        assert frame is not None, label
        assert not frame["gene_nr"].duplicated().any(), label


# ---------------------------------------------------------------------------
# The hit list -- the deliverable a collaborator actually receives
# ---------------------------------------------------------------------------

def _gene_frame():
    return pd.DataFrame({
        "feature": ["gene_fraction:gene[224750]",
                    "gene_fraction:gene[201180]",
                    "gene_fraction:gene[999999]"],
        "coefficient": [1.5, -2.0, 0.1],
        "p_value": [1e-5, 1e-4, 0.9],
        "std_err": [0.2, 0.3, 0.4],
    })


def test_the_hit_list_carries_the_annotation_when_asked():
    from spacr.hits import build_hit_list

    hits = build_hit_list({"gene": _gene_frame(), "__source__": "(test)"},
                          toxoplasma=True)
    frame = hits.to_frame()
    assert frame.loc[frame["gene"] == "201180", "gene_name"].iloc[0] == "MSF"
    assert frame.loc[frame["gene"] == "201180",
                     "hyperlopit"].iloc[0] == "dense granules"
    assert pd.notna(frame.loc[frame["gene"] == "201180",
                              "fit_invitro_hff"].iloc[0])
    assert any("Toxoplasma annotation joined" in note for note in hits.notes)


def test_the_hit_list_is_unannotated_by_default():
    """Off unless asked. A non-Toxoplasma screen gets no Toxoplasma columns."""
    from spacr.hits import build_hit_list

    frame = build_hit_list({"gene": _gene_frame(),
                            "__source__": "(test)"}).to_frame()
    assert "gene_name" not in frame.columns
    assert "hyperlopit" not in frame.columns


def test_the_annotation_does_not_change_the_number_of_hits():
    """One row per gene, before and after. The guard, on the real path."""
    from spacr.hits import build_hit_list

    plain = build_hit_list({"gene": _gene_frame(), "__source__": "(test)"})
    rich = build_hit_list({"gene": _gene_frame(), "__source__": "(test)"},
                          toxoplasma=True)
    assert len(rich.hits) == len(plain.hits) == 3
    assert [h.gene for h in rich.hits] == [h.gene for h in plain.hits]


def test_a_users_own_column_beats_the_bundle(tmp_path):
    """`metadata_files` is applied first and its columns are not replaced."""
    from spacr.hits import build_hit_list

    mine = tmp_path / "mine.csv"
    mine.write_text("Gene ID,gene_name\nTGME49_201180,my own name\n")
    hits = build_hit_list({"gene": _gene_frame(), "__source__": "(test)"},
                          metadata_files=[str(mine)], toxoplasma=True)
    frame = hits.to_frame()
    assert frame.loc[frame["gene"] == "201180",
                     "gene_name"].iloc[0] == "my own name"
    # And the rest of the bundle still arrives.
    assert frame.loc[frame["gene"] == "201180",
                     "hyperlopit"].iloc[0] == "dense granules"
