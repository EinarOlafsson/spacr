"""A UniProt query that fails partway through pagination.

The annotation is an extra: a screen is still analysable without it. So the
question this file answers is what happens when the network drops between
page two and page three -- and the answer the code gives is that the pages
already fetched are KEPT and returned, rather than the whole query being
discarded.

That is the branch nothing had exercised, and it is the one that decides
whether a flaky connection costs a user their annotations or merely some of
them.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def organism():
    from spacr.uniprot import Resolution

    return Resolution(kind="organism", text="Toxoplasma gondii ME49",
                      taxon="508771", accession=None, near=())


def _page(names):
    """A parsed page as ``_parse_tsv`` would return it."""
    return pd.DataFrame({"Entry": names, "Gene Names": names})


def test_a_failure_after_the_first_page_keeps_that_page(organism, tmp_path,
                                                        monkeypatch):
    """Arc 489 -> 491: ``frames`` is not empty, so None is NOT returned.

    A partial annotation is worth having. Discarding two fetched pages
    because a third request timed out would turn a slow network into no
    annotation at all, and the user would have no way to tell that from an
    organism UniProt does not know.
    """
    from spacr import uniprot

    calls = {"n": 0}

    def flaky(_url):
        calls["n"] += 1
        if calls["n"] == 1:
            return "irrelevant", "https://example.invalid/page2"
        raise RuntimeError("the connection dropped")

    monkeypatch.setattr(uniprot, "_get", flaky)
    monkeypatch.setattr(uniprot, "_parse_tsv",
                        lambda _text: _page(["TGGT1_231640", "TGGT1_231650"]))

    frame = uniprot.fetch_genes(organism, ["TGGT1_231640", "TGGT1_231650"],
                                cache_dir=str(tmp_path))

    assert frame is not None
    assert len(frame) == 2
    assert calls["n"] == 2, "it really did try the second page"


def test_a_failure_before_any_page_returns_nothing(organism, tmp_path,
                                                   monkeypatch):
    """Line 490, the taken side: nothing fetched means nothing to return.

    None here means "the query could not be made", which the caller reports
    differently from an empty result -- an organism with no matching genes is
    a finding, and a failed request is not.
    """
    from spacr import uniprot

    def refuse(_url):
        raise RuntimeError("the connection dropped")

    monkeypatch.setattr(uniprot, "_get", refuse)

    assert uniprot.fetch_genes(organism, ["TGGT1_231640"],
                               cache_dir=str(tmp_path)) is None


def test_a_clean_single_page_query_returns_it(organism, tmp_path, monkeypatch):
    """The whole-happy path, so the two failures above are visibly different."""
    from spacr import uniprot

    monkeypatch.setattr(uniprot, "_get", lambda _url: ("irrelevant", None))
    monkeypatch.setattr(uniprot, "_parse_tsv",
                        lambda _text: _page(["TGGT1_231640"]))

    frame = uniprot.fetch_genes(organism, ["TGGT1_231640"],
                                cache_dir=str(tmp_path))

    assert frame is not None and len(frame) == 1


def test_a_resolution_that_is_not_an_organism_is_not_queried(tmp_path):
    """The guard above everything: no organism, no query, no network call."""
    from spacr.uniprot import Resolution, fetch_genes

    other = Resolution(kind="accession", text="P12345", taxon=None,
                       accession="P12345", near=())

    assert fetch_genes(other, ["TGGT1_231640"], cache_dir=str(tmp_path)) is None
