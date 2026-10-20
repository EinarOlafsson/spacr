"""Typing in the feature dictionary search box recycles list items."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.feature_dict import search_features
from spacr.qt.widgets import feature_dictionary as fd


def _shown(panel):
    """Texts and keys of the visible list items."""
    lst = panel._list
    return [(lst.item(i).text(), lst.item(i).data(Qt.UserRole))
            for i in range(lst.count()) if not lst.item(i).isHidden()]


def test_typing_never_deletes_items(qapp):
    panel = fd.FeatureDictionaryPanel()
    seen = []
    last = panel._list.count()
    for query in ["area", "intensity", "zzzqqq no such thing", "", "are", "shape"]:
        panel._search.setText(query)
        lst = panel._list
        assert lst.count() >= last
        last = lst.count()
        for old in seen:
            assert lst.item(old[0]) is old[1]
        seen = [(i, lst.item(i)) for i in range(lst.count())]
        hits = search_features(query, concept=None, object_type=None, limit=300)
        shown = _shown(panel)
        assert [k for _t, k in shown] == [h.doc.key for h in hits]
        assert len(shown) == len(hits)
    assert panel.result_keys() == [h.doc.key for h in hits]


def test_selection_works_on_a_shown_hit(qapp):
    panel = fd.FeatureDictionaryPanel()
    panel._search.setText("area")
    panel._search.setText("intensity")
    assert panel._list.currentRow() == 0
    keys = panel.result_keys()
    got = []
    panel.feature_selected.connect(got.append)
    panel._list.setCurrentRow(1)
    assert got[-1] == keys[1]
    assert panel.current_doc().key == keys[1]
    panel._search.setText("zzzqqq no such thing")
    assert panel._list.currentItem() is None
    assert panel.current_doc() is None
