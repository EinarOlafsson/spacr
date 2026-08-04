"""Real tests for :mod:`spacr.hits` — the hit-list deliverable.

The assertions here are about the four things that make a hit list usable
rather than merely present: the effect size and its interval, a q-value that
corrects for how many genes were actually tested, gRNA agreement computed
from the per-guide table, and a metadata join that CANNOT multiply a gene.

That last one is a regression test with a history. The bundled
``toxoplasma_metadata.csv`` lists a gene once per transcript — 30 Gene IDs
repeat between 2 and 32 times — and joined as-is every one of those genes
came back that many times, with every downstream consumer counting each copy
as an independent hit. ``test_a_metadata_file_with_one_row_per_transcript_
cannot_multiply_a_gene`` is that bug, planted at its worst observed
multiplicity.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import hits as hits_module                          # noqa: E402
from spacr.hits import (DEFAULT_ALPHA, FLAG_CONTROL,            # noqa: E402
                        FLAG_GUIDES_DISAGREE, FLAG_NO_GUIDES,
                        FLAG_NO_METADATA, FLAG_SINGLE_GUIDE, HitList,
                        benjamini_hochberg, build_hit_list, gene_of,
                        grna_agreement, join_metadata, load_gene_metadata,
                        load_results)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _gene_frame():
    """Six genes: two strong hits, a control, a null and two weak ones."""
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in
                    ("100", "200", "300", "400", "233460", "600")],
        "coefficient": [2.4, -1.8, 0.9, 0.05, 0.02, -0.4],
        "std_err": [0.30, 0.25, 0.40, 0.30, 0.10, 0.35],
        "p_value": [1e-6, 4e-5, 0.02, 0.86, 0.84, 0.31],
        "condition": ["other", "other", "other", "other", "nc", "other"],
        "n_gene": [48, 44, 30, 40, 60, 22],
    })


def _grna_frame():
    """Guides: gene 100 fully agrees, 200 splits, 300 has one, 400 none."""
    rows = [
        ("100_1", 2.2), ("100_2", 2.9), ("100_3", 1.7), ("100_4", 2.5),
        ("200_1", -1.9), ("200_2", 1.4), ("200_3", 0.8),
        ("300_1", 0.9),
        ("233460_1", 0.02), ("233460_2", -0.01),
        ("600_1", 0.0), ("600_2", -0.5),
    ]
    return pd.DataFrame({
        "feature": [f"fraction:grna[{g}]" for g, _ in rows],
        "grna": [g for g, _ in rows],
        "coefficient": [c for _, c in rows],
        "p_value": [0.01] * len(rows),
    })


@pytest.fixture
def frames():
    """The ``{role: DataFrame}`` mapping a results folder would give."""
    gene = _gene_frame()
    return {"gene": gene, "grna": _grna_frame(),
            "all": pd.concat([gene, _grna_frame()], ignore_index=True)}


@pytest.fixture
def results_folder(tmp_path):
    """A real folder laid out the way ``perform_regression`` writes one."""
    folder = tmp_path / "results" / "pred" / "ols" / "list"
    folder.mkdir(parents=True)
    gene = _gene_frame()
    gene.to_csv(folder / "results_gene.csv", index=False)
    _grna_frame().to_csv(folder / "results_grna.csv", index=False)
    pd.concat([gene, _grna_frame()], ignore_index=True).to_csv(
        folder / "results.csv", index=False)
    return str(folder)


def _metadata(path, *, repeats=1, genes=("100", "200", "300")):
    """A metadata CSV with each gene repeated ``repeats`` times."""
    rows = []
    for gene in genes:
        for transcript in range(repeats):
            rows.append({
                "Gene ID": f"TGME49_{gene}",
                "Gene Name": f"name-{gene}",
                "Product Description": f"product {gene} t{transcript}",
                "Protein Length": 100 + transcript,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# parsing and statistics
# ---------------------------------------------------------------------------

def test_gene_of_maps_a_gene_term_and_its_guides_to_one_key():
    assert gene_of("gene_fraction:gene[233460]") == "233460"
    assert gene_of("gene_fraction:gene[T.233460]") == "233460"
    assert gene_of("fraction:grna[233460_1]") == "233460"
    assert gene_of("fraction:grna[T.233460_12]") == "233460"


def test_gene_of_returns_none_for_a_term_that_names_no_gene():
    assert gene_of("Intercept") is None
    assert gene_of("rowID[T.B]") == "B"       # bracketed, so it parses
    assert gene_of(None) is None
    assert gene_of(float("nan")) is None
    assert gene_of("") is None


def test_gene_of_matches_the_rule_the_shipped_metadata_merge_uses():
    """The join key must be the same one ``spacr.utils`` derives.

    Two different rules would attach the right annotation to the wrong row
    for every gene id that contains an underscore.
    """
    import re

    def shipped(feature):
        match = re.search(r"\[(.*?)\]", feature)
        if not match:
            return None
        gene = re.sub(r"^T\.", "", match.group(1))
        return gene.split("_")[0]

    for feature in ("gene_fraction:gene[T.233460]", "fraction:grna[239740_7]",
                    "gene_fraction:gene[000000]"):
        assert gene_of(feature) == shipped(feature)


def test_benjamini_hochberg_matches_the_worked_example():
    q = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.06])

    assert q[0] == pytest.approx(0.006)
    assert q[1] == pytest.approx(0.024)
    # Monotone: a larger p can never get a smaller q.
    assert list(q) == sorted(q)
    assert all(0.0 <= value <= 1.0 for value in q)


def test_benjamini_hochberg_ignores_untested_terms():
    with_nan = benjamini_hochberg([0.01, 0.02, np.nan])
    without = benjamini_hochberg([0.01, 0.02])

    assert math.isnan(with_nan[2])
    assert with_nan[0] == pytest.approx(without[0]), (
        "a term that was never tested must not inflate everyone else's q")
    assert list(benjamini_hochberg([])) == []
    assert all(math.isnan(v) for v in benjamini_hochberg([np.nan, np.nan]))


def test_grna_agreement_counts_only_guides_that_push_the_same_way():
    effects = {"100": 2.4, "200": -1.8}
    agreement = grna_agreement(effects, _grna_frame())

    assert agreement["100"] == (4, 4, ["100_1", "100_2", "100_3", "100_4"])
    n_agree, n_guides, names = agreement["200"]
    assert (n_agree, n_guides) == (1, 3)
    assert names == ["200_1"]


def test_a_guide_shrunk_to_exactly_zero_is_not_corroboration():
    frame = pd.DataFrame({
        "feature": ["fraction:grna[600_1]", "fraction:grna[600_2]"],
        "grna": ["600_1", "600_2"],
        "coefficient": [0.0, -0.5]})

    n_agree, n_guides, _ = grna_agreement({"600": -0.4}, frame)["600"]

    assert (n_agree, n_guides) == (1, 2), (
        "a penalised backend zeroes non-contributing guides; counting those "
        "as agreement turns sparsity into evidence")


def test_grna_agreement_without_a_guide_table_reports_nothing_known():
    assert grna_agreement({"100": 1.0}, None) == {"100": (0, 0, [])}
    assert grna_agreement({"100": 1.0}, pd.DataFrame()) == {"100": (0, 0, [])}


# ---------------------------------------------------------------------------
# metadata: the fan-out bug
# ---------------------------------------------------------------------------

def test_a_metadata_file_with_one_row_per_transcript_cannot_multiply_a_gene(
        frames, tmp_path):
    """The bug: 30 Gene IDs repeated 2-32x turned each into up to 32 hits."""
    path = _metadata(tmp_path / "toxo.csv", repeats=32)

    hits = build_hit_list(frames, metadata_files=[path])

    genes = [hit.gene for hit in hits]
    assert len(genes) == len(set(genes)), "a gene appeared more than once"
    assert len(hits) == 6, (
        f"32 transcripts per gene produced {len(hits)} rows instead of 6")
    assert hits.gene("100").annotation["Gene Name"] == "name-100"


def test_the_bundled_toxoplasma_metadata_cannot_multiply_a_gene(frames):
    """The same regression, against the file the bug was found in.

    ``spacr/resources/data/toxoplasma_metadata.csv`` really does list 30 Gene
    IDs between 2 and 32 times, one row per transcript. A synthetic fixture
    proves the code path; this proves the shipped data does not break it.
    """
    import spacr

    path = os.path.join(os.path.dirname(os.path.abspath(spacr.__file__)),
                        "resources", "data", "toxoplasma_metadata.csv")
    if not os.path.isfile(path):
        pytest.skip("the bundled toxoplasma metadata is not installed")
    raw = pd.read_csv(path)
    repeats = raw["Gene ID"].value_counts()
    assert repeats.max() > 1, (
        "this test is only meaningful while the file repeats a gene")

    collapsed, notes = load_gene_metadata(path)

    assert collapsed["gene"].is_unique
    assert len(collapsed) == raw["Gene ID"].nunique()
    assert notes and "one row per transcript" in notes[0]

    genes = collapsed["gene"].head(40).tolist()
    table = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{gene}]" for gene in genes],
        "coefficient": [0.1 * index for index in range(len(genes))],
        "p_value": [0.01] * len(genes)})

    hits = build_hit_list({"gene": table}, metadata_files=[path])

    assert len(hits) == len(table), (
        "joining the real annotation changed the row count")


def test_the_collapse_is_reported_rather_than_hidden(tmp_path):
    path = _metadata(tmp_path / "toxo.csv", repeats=4)

    frame, notes = load_gene_metadata(path)

    assert len(frame) == 3
    assert frame["gene"].is_unique
    assert notes and "one row per transcript" in notes[0]
    assert "12 rows share 3 gene id(s)" in notes[0]
    assert "not carried over" in notes[0]


def test_a_metadata_file_with_no_duplicates_is_left_alone(tmp_path):
    path = _metadata(tmp_path / "clean.csv", repeats=1)

    frame, notes = load_gene_metadata(path)

    assert len(frame) == 3
    assert notes == []


def test_metadata_rows_with_no_parsable_gene_are_dropped_not_joined(tmp_path):
    path = tmp_path / "ragged.csv"
    pd.DataFrame({"Gene ID": ["TGME49_100", "NOUNDERSCORE", "TGME49_200"],
                  "Gene Name": ["a", "b", "c"]}).to_csv(path, index=False)

    frame, notes = load_gene_metadata(path)

    assert sorted(frame["gene"]) == ["100", "200"]
    assert any("no parsable gene" in note for note in notes), (
        "a NaN key joins against every NaN key, so it must be reported")


def test_the_join_refuses_an_annotation_that_is_not_one_row_per_gene(tmp_path):
    """The guard behind the collapse: a fan-out must fail, never fan out."""
    path = tmp_path / "bad.csv"
    frame = pd.DataFrame({"gene": ["100", "100"], "Gene Name": ["a", "b"]})
    frame.to_csv(path, index=False)
    left = pd.DataFrame({"gene": ["100"], "coefficient": [1.0]})

    with pytest.raises(pd.errors.MergeError):
        left.merge(frame, on="gene", how="left", validate="many_to_one")


def test_join_metadata_applies_several_files_without_losing_rows(frames,
                                                                 tmp_path):
    first = _metadata(tmp_path / "a.csv", repeats=3, genes=("100", "200"))
    second = _metadata(tmp_path / "b.csv", repeats=2, genes=("300",))
    table = pd.DataFrame({"gene": ["100", "200", "300", "999"]})

    joined, notes = join_metadata(table, [first, second])

    assert len(joined) == 4
    assert len(notes) == 2
    assert joined.loc[joined["gene"] == "100", "Gene Name"].iloc[0] == "name-100"
    # The second file's colliding columns are suffixed rather than
    # overwriting the first's: two annotations that disagree must both
    # survive, or the join silently picks a winner.
    assert joined.loc[joined["gene"] == "300",
                      "Gene Name_meta2"].iloc[0] == "name-300"
    assert math.isnan(joined.loc[joined["gene"] == "999",
                                 "Protein Length"].iloc[0])


def test_join_metadata_needs_a_gene_column():
    with pytest.raises(KeyError):
        join_metadata(pd.DataFrame({"feature": ["x"]}), [])


def test_a_metadata_file_without_the_key_column_is_refused(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"something": [1]}).to_csv(path, index=False)

    with pytest.raises(KeyError) as caught:
        load_gene_metadata(path)

    assert "Gene ID" in str(caught.value)


def test_a_missing_metadata_file_names_itself(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gene_metadata(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# building the list
# ---------------------------------------------------------------------------

def test_the_list_is_ranked_by_significance_then_by_magnitude(frames):
    hits = build_hit_list(frames, regression_type="ols")

    assert [hit.gene for hit in hits][:3] == ["100", "200", "300"]
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    q_values = [hit.q_value for hit in hits]
    assert q_values == sorted(q_values), "the ranking must follow the q-values"


def test_every_row_carries_an_effect_size_with_an_interval(frames):
    hit = build_hit_list(frames).gene("100")

    assert hit.effect == pytest.approx(2.4)
    assert hit.std_err == pytest.approx(0.30)
    assert hit.ci_low == pytest.approx(2.4 - 1.96 * 0.30)
    assert hit.ci_high == pytest.approx(2.4 + 1.96 * 0.30)
    assert hit.direction == "up"
    assert build_hit_list(frames).gene("200").direction == "down"


def test_a_backend_with_no_standard_error_gets_no_interval():
    frame = pd.DataFrame({"feature": ["gene_fraction:gene[100]"],
                          "coefficient": [1.0], "p_value": [0.01]})

    hit = build_hit_list({"gene": frame}).gene("100")

    assert math.isnan(hit.std_err)
    assert math.isnan(hit.ci_low) and math.isnan(hit.ci_high)


def test_the_q_value_corrects_across_the_genes_actually_tested(frames):
    hits = build_hit_list(frames)

    expected = benjamini_hochberg(_gene_frame()["p_value"])
    by_gene = dict(zip(_gene_frame()["feature"].map(gene_of), expected))
    for hit in hits:
        assert hit.q_value == pytest.approx(by_gene[hit.gene])
    assert hits.gene("300").p_value < DEFAULT_ALPHA
    assert hits.gene("300").q_value > hits.gene("300").p_value, (
        "correcting for six tests must make a p of 0.02 less impressive")


def test_guide_agreement_reaches_every_row(frames):
    hits = build_hit_list(frames)

    assert (hits.gene("100").n_agree, hits.gene("100").n_guides) == (4, 4)
    assert hits.gene("100").agreement == pytest.approx(1.0)
    assert (hits.gene("200").n_agree, hits.gene("200").n_guides) == (1, 3)
    assert hits.gene("200").agreement == pytest.approx(1 / 3)
    assert hits.gene("200").agreeing_guides == ("200_1",)


def test_the_flags_name_every_reason_to_look_twice(frames):
    hits = build_hit_list(frames)

    assert FLAG_GUIDES_DISAGREE in hits.gene("200").flags
    assert FLAG_SINGLE_GUIDE in hits.gene("300").flags
    assert FLAG_NO_GUIDES in hits.gene("400").flags
    assert FLAG_CONTROL in hits.gene("233460").flags
    assert hits.gene("100").flags == ()
    assert set(hits.flag_counts()) >= {FLAG_CONTROL, FLAG_SINGLE_GUIDE}


def test_a_gene_with_no_metadata_row_is_flagged_not_dropped(frames, tmp_path):
    path = _metadata(tmp_path / "partial.csv", genes=("100",))

    hits = build_hit_list(frames, metadata_files=[path])

    assert len(hits) == 6, "an unannotated gene must stay in the list"
    assert FLAG_NO_METADATA in hits.gene("400").flags
    assert FLAG_NO_METADATA not in hits.gene("100").flags
    assert hits.gene("100").name == "name-100"
    assert hits.gene("400").name == "400", "no annotation falls back to the id"


def test_controls_are_listed_by_default_and_can_be_dropped(frames):
    assert build_hit_list(frames).gene("233460") is not None
    assert build_hit_list(frames, include_controls=False).gene("233460") is None


def test_a_penalised_backend_ranks_by_selection_frequency_and_says_so():
    frame = pd.DataFrame({
        "feature": ["gene_fraction:gene[100]", "gene_fraction:gene[200]"],
        "coefficient": [0.4, 1.9],
        "selection_frequency": [0.95, 0.30]})

    hits = build_hit_list({"gene": frame}, regression_type="lasso")

    assert hits.ranking == "selection-frequency"
    assert [hit.gene for hit in hits] == ["100", "200"], (
        "a bigger coefficient selected 30% of the time is not the better hit")
    assert all(math.isnan(hit.q_value) for hit in hits)
    assert any("not a hypothesis test" in note for note in hits.notes)
    assert [h.gene for h in hits.significant(0.6)] == ["100"]


def test_the_no_p_value_list_matches_the_one_in_ml():
    from spacr.ml import NO_P_VALUE_TYPES as shipped

    assert hits_module.NO_P_VALUE_TYPES == tuple(shipped)


def test_a_missing_guide_table_is_a_note_not_a_silent_zero(frames):
    hits = build_hit_list({"gene": frames["gene"]})

    assert any("guide agreement could not be computed" in note
               for note in hits.notes)
    assert all(hit.n_guides == 0 for hit in hits)
    assert all(math.isnan(hit.agreement) for hit in hits)


def test_a_folder_with_no_gene_terms_is_refused_by_name(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(ValueError) as caught:
        build_hit_list(folder)

    assert "results_gene.csv" in str(caught.value)


def test_a_folder_that_is_not_there_is_reported_as_such(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_hit_list(tmp_path / "nope")


def test_an_old_run_without_the_split_files_still_yields_a_list():
    """A results.csv holding both gene and gRNA terms is the legacy layout."""
    everything = pd.concat([_gene_frame(), _grna_frame()], ignore_index=True)

    hits = build_hit_list({"all": everything, "grna": _grna_frame()})

    assert len(hits) == 6
    assert all("gene[" in hit.feature for hit in hits)


def test_the_output_invariant_is_asserted_not_assumed(frames, monkeypatch):
    """A duplicate that slips past every guard must still fail loudly."""
    doubled = pd.concat([frames["gene"], frames["gene"].head(1)],
                        ignore_index=True)
    monkeypatch.setattr(hits_module.pd.DataFrame, "drop_duplicates",
                        lambda self, **_kw: self)

    with pytest.raises(ValueError) as caught:
        build_hit_list({"gene": doubled})

    assert "appears more than once" in str(caught.value)


# ---------------------------------------------------------------------------
# reading a real folder
# ---------------------------------------------------------------------------

def test_a_results_folder_is_read_off_disk(results_folder):
    frames = load_results(results_folder)

    assert set(frames) == {"all", "gene", "grna"}
    hits = build_hit_list(results_folder, regression_type="ols")
    assert len(hits) == 6
    assert hits.source == os.path.abspath(results_folder)


def test_a_folder_with_none_of_the_files_is_empty_not_an_error(tmp_path):
    folder = tmp_path / "other"
    folder.mkdir()
    (folder / "notes.txt").write_text("hello", encoding="utf-8")

    assert load_results(folder) == {}


def test_an_empty_csv_is_skipped_rather_than_taking_the_folder_down(tmp_path):
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "results_gene.csv").write_text("", encoding="utf-8")
    _gene_frame().to_csv(folder / "results.csv", index=False)

    frames = load_results(folder)

    assert "gene" not in frames and "all" in frames


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def test_every_filter_narrows_and_leaves_the_original_alone(frames):
    hits = build_hit_list(frames)

    strong = hits.filter(max_q=0.01)

    assert len(hits) == 6, "filter must not mutate the receiver"
    assert [hit.gene for hit in strong] == ["100", "200"]
    assert [hit.rank for hit in strong] == [1, 2], "ranks renumber"
    assert strong.filters["max_q"] == 0.01


def test_filters_compose(frames):
    hits = build_hit_list(frames)

    narrowed = hits.filter(max_q=0.05).filter(min_agreement=0.5)

    assert [hit.gene for hit in narrowed] == ["100", "300"]
    assert narrowed.filters["max_q"] == 0.05
    assert narrowed.filters["min_agreement"] == 0.5


def test_a_row_with_a_missing_value_fails_the_filter_rather_than_passing(frames):
    hits = build_hit_list({"gene": frames["gene"]})

    assert len(hits.filter(min_agreement=0.0)) == 0, (
        "a gene with no guide evidence has not been shown to agree with "
        "itself")
    assert len(hits.filter(min_guides=1)) == 0


def test_filtering_by_direction_condition_and_controls(frames):
    hits = build_hit_list(frames)

    assert all(h.direction == "up" for h in hits.filter(direction="up"))
    assert [h.gene for h in hits.filter(conditions=["nc"])] == ["233460"]
    assert "233460" not in [h.gene for h in hits.filter(exclude_controls=True)]
    assert [h.gene for h in hits.filter(genes=["300", "600"])] == ["300", "600"]


def test_the_text_query_searches_the_annotation_too(frames, tmp_path):
    path = _metadata(tmp_path / "m.csv", genes=("100", "200", "300"))

    hits = build_hit_list(frames, metadata_files=[path])

    assert [h.gene for h in hits.filter(query="name-200")] == ["200"]
    assert [h.gene for h in hits.filter(query="PRODUCT 300")] == ["300"]
    assert len(hits.filter(query="nothing here")) == 0


def test_min_effect_uses_the_magnitude_not_the_sign(frames):
    hits = build_hit_list(frames)

    kept = hits.filter(min_effect=1.5)

    assert sorted(h.gene for h in kept) == ["100", "200"]


def test_a_custom_predicate_is_honoured(frames):
    hits = build_hit_list(frames)

    kept = hits.filter(predicate=lambda hit: hit.n_obs >= 44)

    assert sorted(h.gene for h in kept) == ["100", "200", "233460"]


def test_top_and_slicing_return_hit_lists(frames):
    hits = build_hit_list(frames)

    assert isinstance(hits.top(2), HitList)
    assert [h.gene for h in hits.top(2)] == ["100", "200"]
    assert [h.gene for h in hits[:2]] == ["100", "200"]
    assert hits[0].gene == "100"
    assert len(hits.top(0)) == 0


def test_significant_uses_the_alpha_the_list_was_built_with(frames):
    hits = build_hit_list(frames, alpha=0.001)

    assert [h.gene for h in hits.significant()] == ["100", "200"]
    assert [h.gene for h in hits.significant(1e-5)] == ["100"]
    assert [h.gene for h in hits.significant(0.05)] == ["100", "200", "300"]


# ---------------------------------------------------------------------------
# the deliverable
# ---------------------------------------------------------------------------

def test_the_frame_has_one_row_per_gene_and_the_columns_a_reader_wants(frames,
                                                                       tmp_path):
    path = _metadata(tmp_path / "m.csv", repeats=5)

    frame = build_hit_list(frames, metadata_files=[path]).to_frame()

    assert len(frame) == 6
    assert frame["gene"].is_unique
    for column in ("rank", "gene", "name", "effect", "p_value", "q_value",
                   "n_guides", "n_agree", "agreement", "flags"):
        assert column in frame.columns
    assert "Product Description" in frame.columns, (
        "the annotation must reach the deliverable")
    assert list(frame["rank"]) == sorted(frame["rank"])


def test_the_csv_round_trips(frames, tmp_path):
    hits = build_hit_list(frames)

    path = hits.write_csv(tmp_path / "out" / "hits.csv")

    assert os.path.isfile(path)
    reread = pd.read_csv(path)
    assert len(reread) == 6
    assert reread["gene"].astype(str).tolist() == [h.gene for h in hits]


def test_the_markdown_carries_the_counts_and_the_flag_legend(frames):
    text = build_hit_list(frames, regression_type="ols").to_markdown(limit=3)

    assert text.startswith("# Hit list")
    assert "genes tested clear FDR 0.05" in text
    assert "Model: ols." in text
    assert text.count("\n| 1 |") == 1
    assert "…and 3 more rows." in text
    assert "single-guide" in text
    assert not text.endswith("\n")


def test_the_html_is_self_contained_and_escapes_its_content(frames, tmp_path):
    frame = frames["gene"].copy()
    frame.loc[0, "feature"] = "gene_fraction:gene[<script>]"
    hits = build_hit_list({"gene": frame, "grna": frames["grna"]})

    path = hits.write_html(tmp_path / "hits.html")
    text = Path(path).read_text(encoding="utf-8")

    assert text.startswith("<!doctype html>")
    assert "<script>" not in text, "gene ids must be escaped, not executed"
    assert "&lt;script&gt;" in text
    assert "http://" not in text and "https://" not in text, (
        "the file a collaborator opens must not fetch anything")


def test_the_summary_is_the_block_a_results_paragraph_quotes(frames):
    summary = build_hit_list(frames, regression_type="ols").summary()

    assert summary["n_genes_tested"] == 6
    assert summary["n_significant"] == 3
    assert summary["n_up"] == 2 and summary["n_down"] == 1
    # Of the three, only gene 100 has two or more guides that mostly agree:
    # 200's guides split 1-of-3 and 300 has a single guide.
    assert summary["n_corroborated"] == 1, (
        "only genes with two or more mostly-agreeing guides are corroborated")
    assert summary["max_abs_effect"] == pytest.approx(2.4)
    assert summary["top_genes"][:3] == ["100", "200", "300"]
    assert summary["alpha"] == DEFAULT_ALPHA
    assert summary["regression_type"] == "ols"


def test_the_summary_records_the_filters_that_produced_the_list(frames):
    summary = build_hit_list(frames).filter(max_q=0.01,
                                            exclude_controls=True).summary()

    assert summary["filters"]["max_q"] == 0.01
    assert summary["filters"]["exclude_controls"] is True
    assert summary["n_listed"] == 2


def test_the_summary_is_json_serializable(frames, tmp_path):
    import json

    path = _metadata(tmp_path / "m.csv", repeats=3)
    payload = json.dumps(
        build_hit_list(frames, metadata_files=[path]).summary(), default=str)

    assert "n_significant" in payload
