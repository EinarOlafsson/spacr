"""A busy UniProt is waited for, briefly, and then said to be busy.

A GitHub runner sharing an address with many others got UniProt's throttle:
every lookup in one job came back with nothing, and the note told the user
"UniProt returned nothing for 'human'" -- which reads as "UniProt has no
human proteins". A 429, a 503 or a dropped connection is a statement about
UniProt's load, not about the organism, so it is retried with a short
backoff that honours ``Retry-After``, and if it does not clear the user is
told UniProt is busy. An empty answer after a 200 is still "returned
nothing".

The network is cut at ``urllib.request.urlopen`` so the real retry loop in
:func:`spacr.uniprot._get` runs; the waits go to a fake ``_sleep``.
"""
from __future__ import annotations

import http.client
import urllib.error
from email.message import Message
from io import BytesIO

import pandas as pd
import pytest

import spacr.uniprot as uniprot
from spacr.uniprot import annotation_for, fetch, fetch_genes, resolve


def _tsv(rows):
    """UniProt's TSV with the asked-for fields."""
    header = "\t".join(uniprot.FIELDS)
    body = "\n".join("\t".join(row) for row in rows)
    return f"{header}\n{body}\n" if rows else f"{header}\n"


def _row(accession, gene):
    """One entry in UniProt's TSV shape."""
    cells = [accession, f"{gene}_HUMAN", f"the {gene} protein", gene,
             "Homo sapiens", "393"]
    return cells + [""] * (len(uniprot.FIELDS) - len(cells))


class _Response(BytesIO):
    """What ``urlopen`` hands back on a 200."""

    def __init__(self, text, link=""):
        super().__init__(text.encode("utf-8"))
        self.headers = {"Link": link} if link else {}


def _http_error(code, retry_after=None):
    """An ``HTTPError`` as urllib raises it, optionally with Retry-After."""
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://rest.uniprot.org/x", code,
                                  "busy", headers, None)


@pytest.fixture
def waits(monkeypatch):
    """Every wait the retry loop asks for, without waiting."""
    seen = []
    monkeypatch.setattr(uniprot, "_sleep", seen.append)
    return seen


def _answer(monkeypatch, replies):
    """Make ``urlopen`` raise or return each reply in turn; count the calls."""
    remaining = list(replies)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if not remaining:
            raise AssertionError("an unexpected extra request")
        reply = remaining.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(uniprot.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_a_429_then_an_answer_is_the_answer(monkeypatch, waits):
    """One throttled request is waited out and the entry arrives."""
    calls = _answer(monkeypatch, [_http_error(429),
                                  _Response(_tsv([_row("P04637", "TP53")]))])

    frame, note = annotation_for("human", genes=["TP53"])

    assert list(frame["accession"]) == ["P04637"]
    assert note == ""
    assert len(calls) == 2
    assert waits == [uniprot._BACKOFF]


def test_a_dropped_connection_is_retried_too(monkeypatch, waits):
    """A reset or a cut-off read is UniProt being unavailable, briefly."""
    calls = _answer(monkeypatch, [
        urllib.error.URLError("connection reset by peer"),
        http.client.IncompleteRead(b"partial"),
        _Response(_tsv([_row("P04637", "TP53")])),
    ])

    frame = fetch_genes(resolve("human"), ["TP53"])

    assert list(frame["accession"]) == ["P04637"]
    assert len(calls) == 3
    assert waits == [uniprot._BACKOFF, uniprot._BACKOFF * 2]


def test_a_503_that_never_clears_says_uniprot_is_busy(monkeypatch, waits,
                                                      caplog):
    """Exhausted retries are "busy, try again later", not "returned nothing".

    And the proteome is not then asked for as well: it would be just as busy,
    and the user would wait the whole budget twice more for the same answer.
    """
    calls = _answer(monkeypatch, [_http_error(503)] * uniprot._ATTEMPTS)

    with caplog.at_level("WARNING"):
        frame, note = annotation_for("human", genes=["TP53"])

    assert frame is None
    assert "UniProt is busy or unreachable, try again later" in note
    assert "'human'" in note and "not annotated" in note
    assert "returned nothing" not in note
    assert len(calls) == uniprot._ATTEMPTS
    assert sum(waits) <= uniprot._BUSY_WAIT
    assert "busy or unreachable for human" in caplog.text


def test_a_busy_proteome_after_an_empty_gene_query_says_busy(monkeypatch,
                                                             waits):
    """The targeted query found nothing; the proteome walk is then throttled."""
    _answer(monkeypatch, [_Response(_tsv([]))]
            + [_http_error(429)] * uniprot._ATTEMPTS)

    frame, note = annotation_for("human", genes=["NOTAGENE"])

    assert frame is None
    assert "UniProt is busy" in note


def test_retry_after_is_honoured(monkeypatch, waits):
    """The server says how long; that is how long is waited."""
    _answer(monkeypatch, [_http_error(429, retry_after=7),
                          _Response(_tsv([_row("P04637", "TP53")]))])

    assert fetch_genes(resolve("human"), ["TP53"]) is not None
    assert waits == [7.0]


def test_retry_after_as_a_date_is_honoured(monkeypatch, waits):
    """The header's other form is a date; a date in the past means now."""
    _answer(monkeypatch, [_http_error(503, "Wed, 21 Oct 2015 07:28:00 GMT"),
                          _Response(_tsv([_row("P04637", "TP53")]))])

    assert fetch_genes(resolve("human"), ["TP53"]) is not None
    assert waits == [0.0]


def test_an_unreadable_retry_after_falls_back_to_the_backoff(monkeypatch,
                                                             waits):
    """A header that is neither seconds nor a date is ignored."""
    _answer(monkeypatch, [_http_error(429, retry_after="soon"),
                          _Response(_tsv([_row("P04637", "TP53")]))])

    assert fetch_genes(resolve("human"), ["TP53"]) is not None
    assert waits == [uniprot._BACKOFF]


def test_a_retry_after_past_the_cap_is_not_waited_for(monkeypatch, waits):
    """Asked to come back in two minutes, the lookup gives up now.

    The cap is what keeps a regression run from sitting on UniProt's queue;
    a wait longer than it is a "busy" answer already.
    """
    calls = _answer(monkeypatch, [_http_error(429, retry_after=120)])

    frame, note = annotation_for("human", genes=["TP53"])

    assert frame is None and "UniProt is busy" in note
    assert waits == []
    assert len(calls) == 1


def test_an_empty_answer_after_a_200_still_returned_nothing(monkeypatch,
                                                            waits):
    """UniProt answered and had nothing: that is a finding, not a busy server."""
    _answer(monkeypatch, [_Response(_tsv([])) for _ in range(3)])

    frame, note = annotation_for("human", genes=["NOTAGENE"])

    assert frame is None
    assert "UniProt returned nothing for 'human'" in note
    assert "busy" not in note
    assert waits == []


def test_a_rejected_query_is_not_retried(monkeypatch, waits, caplog):
    """A 400 is about the query; asking again would get the same 400."""
    calls = _answer(monkeypatch, [_http_error(400)])

    with caplog.at_level("WARNING"):
        assert fetch(resolve("human")) is None
    assert len(calls) == 1
    assert waits == []
    assert "could not be reached for human" in caplog.text


def test_a_busy_page_after_good_ones_keeps_the_good_ones(monkeypatch, waits):
    """Throttled mid-walk, the pages that arrived are still the answer."""
    _answer(monkeypatch, [
        _Response(_tsv([_row("P04637", "TP53")]),
                  link='<https://rest.uniprot.org/page2>; rel="next"'),
    ] + [_http_error(503)] * uniprot._ATTEMPTS)

    frame = fetch(resolve("human"))

    assert list(frame["accession"]) == ["P04637"]


def test_a_busy_uniprot_leaves_the_coefficients_as_they_were(monkeypatch,
                                                             waits):
    """The annotation step hands the busy note on and keeps every row."""
    from spacr.annotation import annotate_with

    _answer(monkeypatch, [_http_error(429)] * uniprot._ATTEMPTS)
    coefficients = pd.DataFrame({"gene": ["TP53", "EGFR"],
                                 "coefficient": [1.2, -0.4]})

    annotated, note = annotate_with(coefficients, "human")

    pd.testing.assert_frame_equal(annotated, coefficients)
    assert "UniProt is busy" in note
