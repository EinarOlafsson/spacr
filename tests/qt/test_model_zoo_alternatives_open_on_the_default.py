"""The zoo's family rows open on the default version, not the alternative.

Publishing an alternative must not change what a family row hands out: the
Toxoplasma PV row opens on v4 although v5 is newer, and the plaque row on
whatever ``spacr.submodules.DEFAULT_PLAQUE_MODEL`` names. Both sit under spaCR.
"""
from __future__ import annotations

import pytest

from spacr import model_zoo as mz
from spacr.qt.screens.model_zoo import ModelZooScreen
from spacr.qt.widgets import model_zoo_picker as mzp


def _rows():
    return [mz._entry_from_mapping(record)
            for record in mz.BUNDLED_REMOTE_MODELS]


def _chosen(view, stem):
    for s, pairs in view._groups:
        if s == stem:
            return pairs[view._chosen[s]][1], pairs
    raise AssertionError(stem)


@pytest.fixture(autouse=True)
def _alpha_visible(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_is_alpha_visible", lambda *a, **k: True)


def test_the_screen_opens_each_family_on_its_default(qtbot):
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_entries(_rows())
    pv, pv_pairs = _chosen(screen, "toxoplasma_pv")
    plaque, plaque_pairs = _chosen(screen, "toxoplasma_plaque")
    assert pv.key == "toxoplasma_pv_v4"
    from spacr import submodules

    assert plaque.key == submodules.DEFAULT_PLAQUE_MODEL
    assert mz.source_of(pv) == mz.source_of(plaque) == "spaCR"
    for group, (stem, pairs) in enumerate(screen._groups):
        if stem not in ("toxoplasma_pv", "toxoplasma_plaque"):
            continue
        combo = screen._table.cellWidget(screen._row_of_group(group), 4)
        assert combo.currentIndex() == screen._chosen[stem]
        alts = [i for i, (_l, e) in enumerate(pairs) if mz.is_alternative(e)]
        if not alts:
            continue
        alt = alts[0]
        from PySide6.QtCore import Qt

        tip = combo.itemData(alt, Qt.ToolTipRole)
        assert tip and "ALTERNATIVE" in tip.upper()


def test_the_picker_opens_each_family_on_its_default(qapp, tmp_path,
                                                    monkeypatch):
    monkeypatch.setattr(mzp.ModelZooPicker, "_warm_the_community_catalogue",
                        lambda self: None)
    monkeypatch.setattr(mzp, "DEFAULT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(tmp_path))
    picker = mzp.ModelZooPicker()
    try:
        picker._rebuild(_rows())
        assert _chosen(picker, "toxoplasma_pv")[0].key == "toxoplasma_pv_v4"
        from spacr import submodules

        assert (_chosen(picker, "toxoplasma_plaque")[0].key
                == submodules.DEFAULT_PLAQUE_MODEL)
        for group, (stem, _pairs) in enumerate(picker._groups):
            combo = picker.table.cellWidget(picker._row_of_group(group), 4)
            assert combo.currentIndex() == picker._chosen[stem]
    finally:
        picker.reject()
        picker.deleteLater()
