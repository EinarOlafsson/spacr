"""Every way a UniProt lookup can come back empty, and what it says then.

The module's own promise is that nothing here is required to open it: an
unreachable UniProt, an unknown organism and an empty proteome all have to
end as ``(None, note)`` with the run carrying on unannotated. That promise is
made entirely out of failure paths, so a test suite that only walks the happy
one proves none of it.

The network is cut at exactly one seam -- :func:`spacr.uniprot._get`, the
single function that opens a socket. Everything above it is the real thing:
the query is really built, the pages are really followed, the TSV is really
parsed by pandas, and the answer is really written to and read back from the
cache directory.
"""

import json
import os

import pytest

import spacr.uniprot as uniprot
from spacr.uniprot import (ORGANISMS, TARGETED_MAX, Resolution, _parse_tsv,
                           _read_cache, _write_cache, annotation_for, fetch,
                           fetch_genes, near_misses, organisms_for, resolve)


# ---------------------------------------------------------------------------
# a stand-in for the one function that opens a socket
# ---------------------------------------------------------------------------

def _tsv(rows):
    """UniProt's own TSV shape: the asked-for fields, tab separated."""
    header = "\t".join(uniprot.FIELDS)
    body = "\n".join("\t".join(str(cell) for cell in row) for row in rows)
    return f"{header}\n{body}\n" if rows else f"{header}\n"


def _row(accession, gene):
    return (accession, f"{gene}_HUMAN", f"the {gene} protein", gene,
            "Homo sapiens", "393", "Nucleus", "apoptosis", "nucleus",
            "DNA binding", "", "", "guards the genome", "AF-P0-F1")


def _serve(monkeypatch, pages):
    """Answer `_get` from `pages`, recording the URLs it was asked for.

    :param pages: ``[(body, next_url_or_exception), ...]``
    :returns: the list of requested URLs.
    """
    asked = []
    remaining = list(pages)

    def fake_get(url):
        asked.append(url)
        if not remaining:
            raise AssertionError(f"an unexpected extra request: {url}")
        body, nxt = remaining.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return body, nxt

    monkeypatch.setattr(uniprot, "_get", fake_get)
    return asked


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this module may reach the network, even by accident."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test opened a socket to UniProt")

    monkeypatch.setattr(uniprot.urllib.request, "urlopen", refuse)


# ---------------------------------------------------------------------------
# what a piece of text turns out to mean
# ---------------------------------------------------------------------------

def test_a_resolution_is_true_only_when_it_resolved():
    """`if resolve(text):` is the natural way to ask, so it has to be right."""
    assert bool(resolve("P04637")) is True
    assert bool(resolve("plasmodium falciparum")) is True
    assert bool(resolve("")) is True                # the bundled default
    assert bool(resolve("not an organism at all")) is False
    assert bool(Resolution("unknown", "x")) is False


def test_a_genus_on_its_own_still_gets_a_did_you_mean():
    """The common miss is a bare genus, and difflib does not rate a prefix.

    Without the prefix fallback "aedes" comes back with no suggestions at
    all, which is the difference between a typo the user can fix and a
    silent absence of annotation.
    """
    import difflib

    assert difflib.get_close_matches("aedes", sorted(ORGANISMS), n=5,
                                     cutoff=0.6) == []
    assert near_misses("aedes") == ("aedes aegypti",)
    assert near_misses("ixodes") == ("ixodes scapularis",)


def test_a_genus_nothing_starts_with_offers_nothing_rather_than_guessing():
    """No prefix match and no close match is an empty tuple, not a wild guess."""
    assert near_misses("qqqqqqqq") == ()


def test_nothing_typed_has_no_near_misses():
    """An empty name is not a typo, so it gets no suggestions."""
    assert near_misses("") == ()
    assert near_misses("   ") == ()
    assert near_misses(None) == ()


def test_the_organism_list_can_be_narrowed_to_a_genus():
    """`organisms_for` is the picker's list, sorted, filtered by prefix."""
    everything = organisms_for()
    assert everything == tuple(sorted(ORGANISMS))

    plasmodium = organisms_for("Plasmodium")
    assert plasmodium, "the genus filter returned nothing"
    assert all(name.startswith("plasmodium") for name in plasmodium)
    assert set(plasmodium) < set(everything)
    # the filter is canonicalised, so spacing and case do not matter
    assert organisms_for("  PLASMODIUM  ") == plasmodium


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

def test_with_nowhere_to_cache_nothing_is_cached(tmp_path):
    """No cache directory reads None and writes nothing, silently."""
    assert _read_cache("", "key") is None
    assert _read_cache(None, "key") is None
    _write_cache("", "key", [{"a": 1}])
    _write_cache(None, "key", [{"a": 1}])
    assert list(tmp_path.iterdir()) == []


def test_a_cache_that_cannot_be_written_does_not_stop_the_answer(tmp_path):
    """A cache directory that is really a file is logged, not raised.

    Caching is an optimisation; failing the fetch because the answer could
    not be filed would make it a requirement.
    """
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("in the way")
    _write_cache(str(blocked), "key", [{"a": 1}])     # must not raise
    assert blocked.read_text() == "in the way"


def test_an_unreadable_cache_entry_is_a_miss_not_a_crash(tmp_path):
    """Truncated JSON reads as "not cached" and the fetch happens again."""
    path = tmp_path / "uniprot_key.json"
    path.write_text("{not json")
    assert _read_cache(str(tmp_path), "key") is None


def test_a_cached_answer_is_returned_without_asking_uniprot(tmp_path,
                                                            monkeypatch):
    """A rerun is offline: the cached rows come back and `_get` is not called."""
    def refuse(_url):
        raise AssertionError("a cached answer still hit the network")

    monkeypatch.setattr(uniprot, "_get", refuse)

    resolution = resolve("9606")
    key = f"tax_{resolution.taxon}_sp"
    (tmp_path / f"uniprot_{key}.json").write_text(
        json.dumps([{"accession": "P04637", "gene_names": "TP53"}]))

    frame = fetch(resolution, cache_dir=str(tmp_path))
    assert list(frame["accession"]) == ["P04637"]


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def test_there_is_nothing_to_ask_uniprot_about_a_bundled_name():
    """The bundled Toxoplasma path never touches this module's network."""
    assert fetch(resolve("toxoplasma")) is None
    assert fetch(resolve("")) is None
    assert fetch(resolve("nonsense organism")) is None


def test_every_page_is_followed_and_the_answer_is_cached(tmp_path,
                                                         monkeypatch):
    """UniProt caps a page at 500 rows, so the cursor has to be followed.

    Stopping at the first page is how a "proteome" became the first five
    hundred entries of one, and a human screen looking for TP53 matched
    nothing.
    """
    asked = _serve(monkeypatch, [
        (_tsv([_row("P04637", "TP53")]), "https://rest.uniprot.org/page2"),
        (_tsv([_row("P38398", "BRCA1")]), ""),
    ])

    frame = fetch(resolve("homo sapiens"), cache_dir=str(tmp_path))

    assert list(frame["accession"]) == ["P04637", "P38398"]
    assert len(asked) == 2
    assert "organism_id%3A9606" in asked[0]
    assert "reviewed%3Atrue" in asked[0]
    assert asked[1] == "https://rest.uniprot.org/page2"

    # and it is on disk, so the next run is offline
    cached = tmp_path / "uniprot_tax_9606_sp.json"
    assert cached.is_file()
    assert len(json.loads(cached.read_text())) == 2


def test_an_empty_page_ends_the_paging(monkeypatch):
    """A page with no rows stops the walk rather than looping on the cursor."""
    asked = _serve(monkeypatch, [
        (_tsv([_row("P04637", "TP53")]), "https://rest.uniprot.org/page2"),
        (_tsv([]), "https://rest.uniprot.org/page3"),
    ])

    frame = fetch(resolve("homo sapiens"))
    assert list(frame["accession"]) == ["P04637"]
    assert len(asked) == 2, "the walk carried on past an empty page"


def test_uniprot_going_away_mid_walk_keeps_what_arrived(monkeypatch, caplog):
    """A failure after some pages returns the pages that did arrive.

    Throwing away a partial proteome would turn a transient outage into no
    annotation at all.
    """
    _serve(monkeypatch, [
        (_tsv([_row("P04637", "TP53")]), "https://rest.uniprot.org/page2"),
        ("", OSError("connection reset")),
    ])

    frame = fetch(resolve("homo sapiens"))
    assert list(frame["accession"]) == ["P04637"]


def test_uniprot_being_unreachable_is_a_warning_and_no_frame(monkeypatch,
                                                             caplog):
    """A failure on the first page returns None and names the organism."""
    _serve(monkeypatch, [("", OSError("name resolution failed"))])

    with caplog.at_level("WARNING"):
        assert fetch(resolve("homo sapiens")) is None
    assert "homo sapiens" in caplog.text
    assert "name resolution failed" in caplog.text


def test_an_organism_with_no_entries_at_all_is_no_frame(monkeypatch):
    """An empty first page is None rather than an empty DataFrame."""
    _serve(monkeypatch, [(_tsv([]), "")])
    assert fetch(resolve("babesia bovis")) is None


def test_an_accession_asks_for_exactly_one_entry(monkeypatch):
    """A single accession is a one-row query, not a proteome."""
    asked = _serve(monkeypatch, [(_tsv([_row("P04637", "TP53")]), "")])

    frame = fetch(resolve("P04637"))
    assert list(frame["accession"]) == ["P04637"]
    assert "accession%3AP04637" in asked[0] or "accession:P04637" in asked[0]
    assert "size=1" in asked[0]


# ---------------------------------------------------------------------------
# parsing what came back
# ---------------------------------------------------------------------------

def test_an_empty_body_is_not_a_table():
    """Blank text is None, not a zero-row frame."""
    assert _parse_tsv("") is None
    assert _parse_tsv("   \n  ") is None
    assert _parse_tsv(None) is None


def test_a_body_that_is_not_a_table_is_not_forced_into_one():
    """A response cut off mid-field is refused rather than half-parsed.

    A truncated page ends inside a quoted description; parsing it would put
    half a protein name into the annotation column and carry on.
    """
    truncated = ("accession\tprotein_name\n"
                 "P04637\t\"Cellular tumor antigen p53, cut off mid-")
    assert _parse_tsv(truncated) is None


# ---------------------------------------------------------------------------
# targeted gene queries
# ---------------------------------------------------------------------------

def test_a_handful_of_genes_is_asked_for_by_name(tmp_path, monkeypatch):
    """A coefficient table naming three genes gets a three-clause query.

    The whole proteome is 20,431 entries and four minutes; this is the path
    that keeps a small table cheap.
    """
    asked = _serve(monkeypatch, [(_tsv([_row("P04637", "TP53")]), "")])

    frame = fetch_genes(resolve("homo sapiens"), ["TP53", "BRCA1", "P38398"],
                        cache_dir=str(tmp_path))

    assert frame is not None
    url = asked[0]
    assert "gene%3ATP53" in url or "gene:TP53" in url
    assert "gene%3ABRCA1" in url or "gene:BRCA1" in url
    # an accession is asked for as an accession, not as a gene name
    assert "accession%3AP38398" in url or "accession:P38398" in url
    assert "organism_id%3A9606" in url or "organism_id:9606" in url


def test_a_targeted_query_is_only_made_when_it_can_be_made():
    """No genes, or no organism, is None rather than a malformed query."""
    assert fetch_genes(resolve("homo sapiens"), []) is None
    assert fetch_genes(resolve("homo sapiens"), ["", "  "]) is None
    assert fetch_genes(resolve("P04637"), ["TP53"]) is None
    assert fetch_genes(resolve("nonsense organism"), ["TP53"]) is None


def test_too_many_genes_falls_back_to_the_whole_proteome():
    """Past `TARGETED_MAX` the targeted query is declined.

    A query URL with a thousand clauses in it is more expensive than the
    proteome the caller will otherwise fetch once and cache.
    """
    many = [f"GENE{i}" for i in range(TARGETED_MAX + 1)]
    assert fetch_genes(resolve("homo sapiens"), many) is None


def test_a_targeted_query_that_fails_is_no_frame(monkeypatch):
    """An unreachable UniProt on the targeted path returns None quietly."""
    _serve(monkeypatch, [("", OSError("connection reset"))])
    assert fetch_genes(resolve("homo sapiens"), ["TP53"]) is None


def test_a_targeted_query_that_matches_nothing_is_no_frame(monkeypatch):
    """Genes UniProt does not have come back as None, not an empty frame."""
    _serve(monkeypatch, [(_tsv([]), "")])
    assert fetch_genes(resolve("homo sapiens"), ["NOSUCHGENE"]) is None


def test_a_cached_targeted_answer_skips_the_network(tmp_path, monkeypatch):
    """The gene set is part of the cache key, so the same set is offline."""
    _serve(monkeypatch, [(_tsv([_row("P04637", "TP53")]), "")])
    first = fetch_genes(resolve("homo sapiens"), ["TP53"],
                        cache_dir=str(tmp_path))
    assert first is not None

    def refuse(_url):
        raise AssertionError("the second identical query hit the network")

    monkeypatch.setattr(uniprot, "_get", refuse)
    again = fetch_genes(resolve("homo sapiens"), ["TP53"],
                        cache_dir=str(tmp_path))
    assert list(again["accession"]) == ["P04637"]


# ---------------------------------------------------------------------------
# the one call a pipeline makes
# ---------------------------------------------------------------------------

def test_a_bundled_name_annotates_from_the_bundled_tables():
    """`(None, "")` means "not this module's job", with nothing to report."""
    assert annotation_for("toxoplasma") == (None, "")
    assert annotation_for("") == (None, "")


def test_an_unknown_organism_says_so_and_suggests_the_near_misses():
    """The note is the whole difference between a typo and silent absence."""
    frame, note = annotation_for("plasmodum falciparum")
    assert frame is None
    assert "plasmodum falciparum" in note
    assert "did you mean" in note
    assert "plasmodium falciparum" in note
    assert "not annotated" in note


def test_an_unknown_name_with_no_near_misses_still_explains_itself():
    """No suggestion to offer does not mean no explanation."""
    frame, note = annotation_for("qqqqqqqq")
    assert frame is None
    assert "did you mean" not in note
    assert "is not an organism spaCR knows" in note


def test_a_named_gene_is_answered_without_the_whole_proteome(monkeypatch):
    """The targeted query answers, so the proteome is never asked for."""
    asked = _serve(monkeypatch, [(_tsv([_row("P04637", "TP53")]), "")])

    frame, note = annotation_for("homo sapiens", genes=["TP53"])
    assert list(frame["accession"]) == ["P04637"]
    assert note == ""
    assert len(asked) == 1


def test_an_organism_with_no_reviewed_entries_says_the_rows_are_predictions(
        monkeypatch):
    """The TrEMBL fallback is taken AND said.

    Swiss-Prot has nothing at all for some strain-level taxa; refusing to
    annotate those would make the field useless for exactly the organisms it
    was asked for. But an unreviewed entry is a prediction, and the reader is
    entitled to know which they have.
    """
    asked = _serve(monkeypatch, [
        (_tsv([]), ""),                                   # reviewed: nothing
        (_tsv([_row("A0A0G2K3N4", "BBOV")]), ""),         # unreviewed: rows
    ])

    frame, note = annotation_for("babesia bovis")

    assert len(frame) == 1
    assert "no reviewed (Swiss-Prot) entries" in note
    assert "1 are unreviewed TrEMBL predictions" in note
    assert "reviewed%3Atrue" in asked[0] or "reviewed:true" in asked[0]
    assert "reviewed%3Atrue" not in asked[1]
    assert "reviewed:true" not in asked[1]


def test_an_organism_uniprot_has_nothing_for_leaves_the_run_unannotated(
        monkeypatch):
    """Both queries empty is `(None, note)`, and the run carries on."""
    _serve(monkeypatch, [(_tsv([]), ""), (_tsv([]), "")])

    frame, note = annotation_for("babesia bovis")
    assert frame is None
    assert "UniProt returned nothing" in note
    assert "babesia bovis" in note
    assert "not annotated" in note


def test_an_unreachable_uniprot_leaves_the_run_unannotated(monkeypatch):
    """It never raises: a fitted run must not be lost to a network failure."""
    _serve(monkeypatch, [
        ("", OSError("no route to host")),
        ("", OSError("no route to host")),
    ])

    frame, note = annotation_for("homo sapiens")
    assert frame is None
    assert "UniProt returned nothing" in note


# ---------------------------------------------------------------------------
# the paging header, and the one call that opens a socket
# ---------------------------------------------------------------------------

def test_the_next_page_is_read_out_of_a_link_header_that_contains_commas():
    """The header is comma separated and the URL inside it has commas too.

    Splitting on commas cut the URL into fragments, the fragment holding
    `rel="next"` lost its opening bracket, and every fetch stopped at 500
    rows -- which is why a human screen looking for TP53 matched nothing.
    """
    from spacr.uniprot import _next_page

    header = ('<https://rest.uniprot.org/uniprotkb/search?cursor=abc123'
              '&fields=accession,id,protein_name,gene_names&format=tsv'
              '&size=500>; rel="next"')
    assert _next_page(header) == (
        "https://rest.uniprot.org/uniprotkb/search?cursor=abc123"
        "&fields=accession,id,protein_name,gene_names&format=tsv&size=500")


def test_a_last_page_says_there_is_no_next_one():
    """No `rel="next"` is "", and so is no header at all."""
    from spacr.uniprot import _next_page

    assert _next_page("") == ""
    assert _next_page(None) == ""
    assert _next_page('<https://rest.uniprot.org/x>; rel="last"') == ""


def test_one_request_asks_for_tsv_and_identifies_itself(monkeypatch):
    """`_get` sets the Accept and User-Agent headers and returns the cursor.

    UniProt rate-limits anonymous traffic harder, and a request that does not
    say what it is gets throttled first.
    """
    from spacr.uniprot import _get

    seen = {}

    class _Response:
        headers = {"Link": '<https://rest.uniprot.org/next>; rel="next"'}

        def read(self):
            return "accession\nP04637\n".encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(uniprot.urllib.request, "urlopen", fake_urlopen)

    body, nxt = _get("https://rest.uniprot.org/uniprotkb/search?query=x")

    assert body == "accession\nP04637\n"
    assert nxt == "https://rest.uniprot.org/next"
    assert seen["url"] == "https://rest.uniprot.org/uniprotkb/search?query=x"
    assert seen["timeout"] == uniprot.TIMEOUT
    lowered = {k.lower(): v for k, v in seen["headers"].items()}
    assert lowered["accept"] == "text/plain"
    assert "spaCR" in lowered["user-agent"]


def test_a_body_that_is_not_utf8_is_read_anyway(monkeypatch):
    """A stray byte in a description does not lose the whole page."""
    from spacr.uniprot import _get

    class _Response:
        headers = {}

        def read(self):
            return b"accession\nP0\xff4637\n"

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(uniprot.urllib.request, "urlopen",
                        lambda request, timeout=None: _Response())

    body, nxt = _get("https://rest.uniprot.org/x")
    assert "accession" in body
    assert nxt == ""
