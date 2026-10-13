"""Two Model Zoo row rules that no bundled record exercises today.

* A bundled record that SAYS whether it is a candidate is believed over
  its display name: "candidate" in a name is only the fallback, so a
  promoted model whose name still reads "v2 candidate" is a default again
  the moment its record says ``candidate: false``.
* A MitoNet row is someone else's model wherever its file came from, so a
  copy the user keeps locally is still listed under ``Third-party`` and not
  under ``spaCR``, whose local rows are spaCR's own.
"""
from __future__ import annotations

from types import SimpleNamespace

from spacr import model_zoo


def test_a_record_that_says_it_is_no_candidate_is_believed_over_its_name():
    named_candidate = {"key": "x", "display_name": "Nucleus v2 candidate"}
    assert model_zoo._record_is_candidate(named_candidate) is True

    promoted = dict(named_candidate, candidate=False)
    assert model_zoo._record_is_candidate(promoted) is False

    declared = {"key": "y", "display_name": "Nucleus v3", "candidate": 1}
    assert model_zoo._record_is_candidate(declared) is True


def test_a_locally_kept_mitonet_is_still_listed_as_third_party():
    for source, uri in (("", ""), ("local", "/models/mitonet_mini.pth"),
                        ("remote", "https://example.org/mitonet.pth")):
        row = SimpleNamespace(key="mitonet-somebody-else", kind="mitonet",
                              source=source, uri=uri)
        assert model_zoo.is_third_party(row)
        assert model_zoo.source_of(row) == "Third-party", (source, uri)

    spacr_row = SimpleNamespace(key="spacr-own", kind="backend",
                                source="local", uri="/models/own.pth")
    assert model_zoo.source_of(spacr_row) == "spaCR"
