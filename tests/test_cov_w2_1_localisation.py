"""Compartment lookup behaviour when the reference table is unusable.

A screen of a different organism, a table saved without a location column,
and a frame that has no feature column at all all reach the same place: the
volcano loses its compartment colouring and keeps working.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import localisation


@pytest.fixture(autouse=True)
def _forget_the_cached_table():
    """The bundled table is cached process-wide; do not leak a fake one."""
    localisation.table.cache_clear()
    yield
    localisation.table.cache_clear()


def _bundle(monkeypatch, tmp_path, frame):
    """Point the bundled-table constant at ``frame`` written to disk."""
    path = tmp_path / "lopit.csv"
    frame.to_csv(path, index=False)
    import spacr.gene_tile as gene_tile

    monkeypatch.setattr(gene_tile, "BUNDLED_LOCALISATION", str(path))
    return path


def test_a_table_without_a_location_column_colours_nothing(monkeypatch,
                                                           tmp_path):
    """A readable CSV that names no compartment yields no lookup at all."""
    _bundle(monkeypatch, tmp_path,
            pd.DataFrame({"gene_nr": ["233460"], "notes": ["cytosol"]}))

    assert localisation.table() == {}


def test_a_table_without_a_gene_column_colours_nothing(monkeypatch, tmp_path):
    """A compartment with nothing to join it to is equally unusable."""
    _bundle(monkeypatch, tmp_path,
            pd.DataFrame({"accession": ["233460"], "location": ["cytosol"]}))

    assert localisation.table() == {}


def test_a_blank_gene_row_is_skipped(monkeypatch, tmp_path):
    """Rows whose gene is empty or literally 'nan' name no gene."""
    _bundle(monkeypatch, tmp_path, pd.DataFrame({
        "gene_nr": ["", "nan", "233460"],
        "location": ["cytosol", "nucleus", "rhoptry"],
    }))

    assert localisation.table() == {"233460": "rhoptry"}


def test_a_float_formatted_gene_number_joins_to_the_screen(monkeypatch,
                                                           tmp_path):
    """`gene_nr` read as a float arrives as '244480.0' and must join anyway."""
    _bundle(monkeypatch, tmp_path, pd.DataFrame({
        "gene_nr": [244480.0, 233460.0],
        "tagm_location": ["dense granule", "cytosol"],
    }))

    places = localisation.table()

    assert places == {"244480": "dense granule", "233460": "cytosol"}
    assert "244480.0" not in places


def test_a_numeric_compartment_is_not_a_compartment(monkeypatch, tmp_path):
    """A location column holding numbers describes no biology."""
    _bundle(monkeypatch, tmp_path, pd.DataFrame({
        "gene_nr": ["233460", "244480"],
        "location": ["12.5", "cytosol"],
    }))

    assert localisation.table() == {"244480": "cytosol"}


def test_a_frame_without_the_key_column_has_no_compartments():
    """`of` answers with an empty Series rather than raising a KeyError."""
    frame = pd.DataFrame({"coef": [1.0, 2.0]})

    places = localisation.of(frame)

    assert isinstance(places, pd.Series)
    assert len(places) == 0


def test_no_frame_at_all_has_no_compartments():
    """`None` is a legitimate 'nothing loaded yet' state for a figure."""
    assert len(localisation.of(None)) == 0


def test_a_frame_without_the_key_column_offers_no_menu():
    """`present` is what fills the colour-by menu; it must not raise."""
    assert localisation.present(pd.DataFrame({"coef": [1.0]})) == []


def test_a_frame_without_the_key_column_highlights_nothing():
    """`mask` returns all-False aligned to the frame, not an exception."""
    frame = pd.DataFrame({"coef": [1.0, 2.0, 3.0]}, index=[7, 8, 9])

    chosen = localisation.mask(frame, "cytosol")

    assert list(chosen.index) == [7, 8, 9]
    assert not chosen.any()


def test_a_missing_bundled_file_is_not_an_error(monkeypatch, tmp_path):
    """A screen of another organism has no reason to carry the LOPIT file."""
    import spacr.gene_tile as gene_tile

    monkeypatch.setattr(gene_tile, "BUNDLED_LOCALISATION",
                        str(tmp_path / "absent.csv"))

    assert localisation.table() == {}


def _screen(monkeypatch, tmp_path):
    """A three-compartment reference table and a matching screen frame."""
    _bundle(monkeypatch, tmp_path, pd.DataFrame({
        "gene_nr": [str(200000 + n) for n in range(12)],
        "location": (["cytosol"] * 6) + (["rhoptry"] * 5) + ["micronemes"],
    }))
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{200000 + n}]" for n in range(12)]
                   + ["fraction:grna[999999_1]"],
        "coef": list(range(13)),
    })


def test_a_guide_term_resolves_to_its_genes_compartment(monkeypatch, tmp_path):
    """Both gene and guide terms join, and unknown genes come back blank."""
    frame = _screen(monkeypatch, tmp_path)

    places = localisation.of(frame)

    assert list(places[:6]) == ["cytosol"] * 6
    assert list(places[6:11]) == ["rhoptry"] * 5
    assert places.iloc[12] == ""


def test_the_menu_drops_a_compartment_too_small_to_read(monkeypatch, tmp_path):
    """One gene is not a pattern; the single-gene compartment is not offered."""
    frame = _screen(monkeypatch, tmp_path)

    assert localisation.present(frame) == ["cytosol", "rhoptry"]
    assert "micronemes" not in localisation.present(frame)


def test_choosing_nothing_highlights_nothing(monkeypatch, tmp_path):
    """The menu's 'no compartment' state passes straight through."""
    frame = _screen(monkeypatch, tmp_path)

    chosen = localisation.mask(frame, None)

    assert not chosen.any()
    assert list(chosen.index) == list(frame.index)


def test_choosing_a_compartment_highlights_exactly_its_genes(monkeypatch,
                                                             tmp_path):
    """The highlight is the compartment's rows and nothing else."""
    frame = _screen(monkeypatch, tmp_path)

    chosen = localisation.mask(frame, "rhoptry")

    assert list(frame.loc[chosen, "coef"]) == [6, 7, 8, 9, 10]
