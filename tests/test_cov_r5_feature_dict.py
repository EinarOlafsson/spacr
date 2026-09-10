"""``spacr.feature_dict``: the suffix rules that decline, and a query of nothing.

``parse_column`` peels a column name apart by trying rules in order. Two of
those rules match on the *shape* of the name and then have to back out when
the thing they uncovered is not a feature after all:

* a trailing ``_<object>`` looks like the pandas merge suffix
  ``_read_and_join_tables`` adds -- but only if what is left is a real stat;
* a trailing ``__dup<n>`` looks like ``_check_integrity``'s duplicate marker
  -- likewise.

Backing out is what keeps an unknown column *unknown* instead of being
re-described as some neighbouring feature. The third case is ``search_features``
with a query made only of stopwords, which must match on nothing rather than
on everything.
"""
from __future__ import annotations

from spacr.feature_dict import parse_column, search_features


def test_a_merge_suffix_is_only_stripped_when_a_real_stat_is_left():
    """``_<object>`` is a merge suffix only if the rest names a feature.

    ``cell_area_nucleus`` is ``cell_area`` joined onto the nucleus table, and
    is described as such. ``cell_zzznotastat_nucleus`` has the same shape and
    nothing behind it, so the rule has to put the suffix back and let the
    column fall through to "unknown" -- describing it as *some* nucleus-joined
    feature would be an invented answer.
    """
    joined = parse_column("cell_area_nucleus")
    assert joined.key == "area"
    assert joined.object_type == "cell"
    assert joined.object_type_2 == "nucleus"
    assert "merge suffix" in (joined.notes or "")

    declined = parse_column("cell_zzznotastat_nucleus")
    assert declined.family == "unknown"
    assert declined.key is None
    assert declined.object_type_2 is None
    assert "merge suffix" not in (declined.notes or "")


def test_a_dup_suffix_is_only_stripped_when_a_real_stat_is_left():
    """``__dup<n>`` is ``_check_integrity``'s marker, not part of a name.

    Same shape, same discipline: ``cell_area__dup2`` is a second copy of
    ``cell_area`` and says so, while ``cell_zzznotastat__dup2`` stays unknown
    because the base is not a feature the dictionary knows.
    """
    duplicate = parse_column("cell_area__dup2")
    assert duplicate.key == "area"
    assert "Occurrence 2 of a duplicated column name" in (duplicate.notes or "")

    declined = parse_column("cell_zzznotastat__dup2")
    assert declined.family == "unknown"
    assert declined.key is None
    assert "duplicated column name" not in (declined.notes or "")


def test_a_query_of_nothing_but_stopwords_matches_nothing():
    """"of" is not a search; scoring every entry for it would be a lie.

    ``_query_terms`` drops stopwords, so the free-text rule has no terms left
    and must contribute no score at all. The contrast is a query with one real
    term, which reaches the same rule and does score entries on their
    definitions -- so "no hits" here is a decision, not a dead code path.
    """
    everything = search_features("")
    assert len(everything) > 100, "the dictionary should list its whole self"

    assert search_features("of") == []

    real = search_features("homogeneity")
    assert 0 < len(real) < len(everything)
    assert any("mentioned in the definition" in hit.reason for hit in real), (
        "the free-text rule never fired, so the empty-terms case proves nothing")
