"""Attention MIL learns which cells carry a well label from well labels only.

Validated on synthetic wells with planted responder cells: the held-out
attention ranks responders above the other cells of positive wells, and the
held-out well prediction beats a classifier on each well's mean cell.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from spacr.embeddings import (_mil_bags, _mil_fit, _mil_from_table,
                              _mil_predict, _mil_scorecard,
                              _synthetic_mil_bags)

_FIT = {"epochs": 40}


def test_attention_finds_planted_responders_and_beats_the_mean_baseline():
    bags, labels, responders = _synthetic_mil_bags(
        wells=32, cells=50, features=8, fraction=0.1, seed=3)
    card = _mil_scorecard(bags, labels, responders=responders, folds=4,
                          **_FIT)
    assert card["attention_auroc"] > 0.85
    assert card["evidence_auroc"] > 0.85
    assert card["mil_auroc"] > 0.8
    assert card["mil_auroc"] > card["mean_auroc"]


def test_attention_is_an_equal_share_scale_per_well():
    bags, labels, _ = _synthetic_mil_bags(wells=8, cells=20, features=4)
    fitted = _mil_fit(bags, labels, hidden=8, epochs=2)
    probs, attention, evidence = _mil_predict(fitted, bags)
    assert probs.shape == (8,) and np.all((probs >= 0) & (probs <= 1))
    assert all(np.isclose(a.mean(), 1.0, atol=1e-4) for a in attention)
    assert [e.shape for e in evidence] == [(20,)] * 8


def _table(seed=0):
    bags, labels, responders = _synthetic_mil_bags(
        wells=16, cells=30, features=6, fraction=0.2, seed=seed)
    rows = []
    for well, (bag, label, hit) in enumerate(zip(bags, labels, responders)):
        frame = pd.DataFrame(bag, columns=[f"emb_{i}" for i in range(6)])
        frame.insert(0, "well_label", int(label))
        frame.insert(0, "well", f"A{well:02d}")
        frame["area"] = 100.0
        frame["planted"] = hit
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_the_table_route_uses_embedding_columns_and_keeps_every_cell():
    frame = _table()
    bags, labels, wells, rows, kept = _mil_bags(frame)
    assert bags[0].shape == (30, 6)
    assert labels.sum() == 8 and len(wells) == 16
    cells, well_frame, card = _mil_from_table(frame, folds=2, **_FIT)
    assert len(cells) == len(frame)
    assert {"mil_attention", "mil_evidence", "planted", "area"} <= set(cells.columns)
    assert list(well_frame.columns) == ["well", "well_label",
                                        "mil_probability", "cells"]
    assert set(card) >= {"mil_auroc", "mean_auroc", "mean_sd_auroc"}


def test_refuses_labels_it_cannot_read():
    frame = _table()
    mixed = frame.copy()
    mixed.loc[0, "well_label"] = 1 - mixed.loc[0, "well_label"]
    with pytest.raises(ValueError, match="more than one label"):
        _mil_bags(mixed)
    named = frame.assign(well_label=frame["well_label"].map(
        {0: "control", 1: "treated"}))
    with pytest.raises(ValueError, match="name the positive label"):
        _mil_bags(named)
    assert _mil_bags(named, positive="treated")[1].sum() == 8
    with pytest.raises(ValueError, match="no 'wellID' column"):
        _mil_bags(frame.drop(columns="well"))


def _embedded_project(tmp_path):
    """A measurements.db whose crops carry stored embeddings.

    Row A's four wells are positive; a third of their cells have a raised
    first embedding dimension. Row B's four wells are negative.
    """
    import sqlite3
    from types import SimpleNamespace

    from spacr import active_learning as al
    from tests.test_cov_active_learning_rounds import _make_project

    wells = [(r, str(c)) for r in "AB" for c in range(1, 5)]
    project = _make_project(tmp_path, per_well=12, wells=wells)
    with sqlite3.connect(project["db"]) as con:
        rows = con.execute("SELECT prcfo, rowID FROM png_list").fetchall()
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, (len(rows), 4)).astype(np.float32)
    for i, (key, row) in enumerate(rows):
        if row == "A" and int(key.split("_o")[-1]) % 3 == 0:
            values[i, 0] += 4.0
    al._store_crop_embeddings(project["db"], [r[0] for r in rows],
                              SimpleNamespace(values=values, columns=tuple(
                                  f"emb_{i}" for i in range(4))))
    labels = pd.DataFrame({"well": [f"{r}{c}" for r, c in wells],
                           "well_label": [int(r == "A") for r, c in wells]})
    return project["db"], labels


def test_stored_embeddings_join_object_ids_and_wells(tmp_path):
    from spacr.embeddings import _mil_frame_from_db, _mil_from_table

    db, labels = _embedded_project(tmp_path)
    frame = _mil_frame_from_db(db)
    assert len(frame) == 96 and frame["wellID"].nunique() == 8
    assert {"prcfo", "png_path", "plateID", "rowID", "columnID", "object",
            "emb_0"} <= set(frame.columns)
    assert (frame["wellID"] == "plate1_" + frame["rowID"] + "_"
            + frame["columnID"]).all()
    assert all(key.endswith("_" + obj)
               for key, obj in zip(frame["prcfo"], frame["object"]))
    labelled = _mil_frame_from_db(db, labels)
    assert labelled["well_label"].notna().all()
    assert (labelled["well_label"] == (labelled["rowID"] == "A")).all()
    cells, wells, card = _mil_from_table(labelled, folds=2, epochs=30)
    assert len(wells) == 8 and "prcfo" in cells.columns
    planted = cells["rowID"].eq("A") & cells["object"].str[1:].astype(
        int).mod(3).eq(0)
    assert cells.loc[planted, "mil_evidence"].mean() > cells.loc[
        ~planted, "mil_evidence"].mean()


def test_a_database_without_stored_embeddings_is_refused(tmp_path):
    from spacr.embeddings import _mil_frame_from_db
    from tests.test_cov_active_learning_rounds import _make_project

    db = _make_project(tmp_path)["db"]
    with pytest.raises(ValueError, match="no stored crop embeddings"):
        _mil_frame_from_db(db)
    db, labels = _embedded_project(tmp_path / "e")
    with pytest.raises(ValueError, match="no 'well_label' column"):
        _mil_frame_from_db(db, labels.drop(columns="well_label"))


def test_mil_labels_resolve_mixed_full_short_and_compact_keys_without_losing_wells(tmp_path):
    from spacr.embeddings import _mil_frame_from_db
    db, _ = _embedded_project(tmp_path)
    labels = pd.DataFrame({'well': ['plate1_A_1', 'A_2', 'A3', 'plate1_B_1', 'B2'], 'well_label': [0, 1, 0, 1, 0]})
    before = labels.copy(deep=True)
    database_before = Path(db).read_bytes()
    frame = _mil_frame_from_db(db, labels)
    assert Path(db).read_bytes() == database_before
    expected = {'plate1_A_1': 0, 'plate1_A_2': 1, 'plate1_A_3': 0, 'plate1_B_1': 1, 'plate1_B_2': 0}
    for well, group in frame.groupby('wellID'):
        if well in expected:
            assert group.well_label.eq(expected[well]).all()
        else:
            assert group.well_label.isna().all()
    pd.testing.assert_frame_equal(labels, before)


@pytest.mark.parametrize('keys', [['A1', 'A1'], ['A1', 'plate1_A_1'], ['A_1', 'A1']])
def test_mil_conflicting_duplicate_or_alias_labels_are_refused(tmp_path, keys):
    from spacr.embeddings import _mil_frame_from_db
    db, _ = _embedded_project(tmp_path)
    before = Path(db).read_bytes()
    labels = pd.DataFrame({'well': keys, 'well_label': [0, 1]})
    with pytest.raises(ValueError, match='conflicting labels'):
        _mil_frame_from_db(db, labels)
    assert Path(db).read_bytes() == before


def test_mil_agreeing_duplicate_and_alias_labels_and_missing_values_are_safe(tmp_path):
    from spacr.embeddings import _mil_frame_from_db
    db, _ = _embedded_project(tmp_path)
    labels = pd.DataFrame({'well': ['A1', 'A1', 'A_1', 'plate1_A_1', 'A1', None, 'Z99'], 'well_label': [1, 1, 1, 1, np.nan, 0, 1]})
    frame = _mil_frame_from_db(db, labels)
    assert frame.loc[frame.wellID.eq('plate1_A_1'), 'well_label'].eq(1).all()
    assert frame.loc[~frame.wellID.eq('plate1_A_1'), 'well_label'].isna().all()


def _two_plate_mil_project(tmp_path):
    import sqlite3
    db, _ = _embedded_project(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE png_list SET prcfo=replace(prcfo, 'plate1_B_', 'plate2_A_'), plateID='plate2', rowID='A' WHERE rowID='B'")
        connection.execute("UPDATE crop_embedding SET prcfo=replace(prcfo, 'plate1_B_', 'plate2_A_')")
    return db


@pytest.mark.parametrize('key', ['A1', 'A_1'])
def test_mil_short_well_labels_never_cross_plate_boundaries(tmp_path, key):
    from spacr.embeddings import _mil_frame_from_db
    db = _two_plate_mil_project(tmp_path)
    before = Path(db).read_bytes()
    with pytest.raises(ValueError, match='ambiguous across plates'):
        _mil_frame_from_db(db, pd.DataFrame({'well': [key], 'well_label': [1]}))
    assert Path(db).read_bytes() == before


def test_mil_full_well_keys_keep_two_plates_separate(tmp_path):
    from spacr.embeddings import _mil_frame_from_db
    db = _two_plate_mil_project(tmp_path)
    labels = pd.DataFrame({'wellID': ['plate1_A_1', 'plate2_A_1'], 'well_label': [0, 1]})
    frame = _mil_frame_from_db(db, labels)
    assert frame.loc[frame.wellID.eq('plate1_A_1'), 'well_label'].eq(0).all()
    assert frame.loc[frame.wellID.eq('plate2_A_1'), 'well_label'].eq(1).all()
    assert frame.loc[~frame.wellID.isin(labels.wellID), 'well_label'].isna().all()


def test_mil_missing_database_refuses_without_creating_it(tmp_path):
    from spacr.embeddings import _mil_frame_from_db
    missing = tmp_path / 'missing.db'
    with pytest.raises(ValueError, match='no stored crop embeddings'):
        _mil_frame_from_db(missing)
    assert not missing.exists()


@pytest.mark.parametrize('layout', ['absent', 'null', 'blank'])
def test_mil_recovers_underscored_plate_metadata_with_canonical_parser(tmp_path, layout):
    import sqlite3
    from spacr import tabular
    from spacr.embeddings import _mil_frame_from_db
    db, _ = _embedded_project(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE png_list SET prcfo=replace(prcfo, 'plate1_', 'screen_day_1_')")
        connection.execute("UPDATE crop_embedding SET prcfo=replace(prcfo, 'plate1_', 'screen_day_1_')")
        connection.execute("UPDATE png_list SET plateID='screen_day_1'")
    crops = tabular.read_table(db, table='png_list', canonicalise=False, repair_plate_ids=False, migrate=False, read_only=True, report=None)
    if layout == 'absent':
        crops = crops.drop(columns=['plateID', 'rowID', 'columnID', 'fieldID'])
    elif layout == 'null':
        crops.loc[:, ['plateID', 'rowID', 'columnID', 'fieldID']] = None
    else:
        crops.loc[:, ['plateID', 'rowID', 'columnID', 'fieldID']] = ' '
    tabular.write_database(crops, db, 'png_list', canonicalise=False, if_exists='replace')
    before = Path(db).read_bytes()
    frame = _mil_frame_from_db(db)
    assert frame.plateID.eq('screen_day_1').all()
    assert set(frame.rowID) == {'A', 'B'} and set(frame.columnID) == {'1', '2', '3', '4'}
    assert frame.fieldID.eq('f1').all()
    assert frame.wellID.str.startswith('screen_day_1_').all()
    assert Path(db).read_bytes() == before


def test_mil_invalid_fallback_key_refuses_without_writing_or_guessing(tmp_path):
    import sqlite3
    from spacr.embeddings import _mil_frame_from_db
    db, _ = _embedded_project(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE png_list SET prcfo='invalid-key', plateID=NULL WHERE prcfo=(SELECT prcfo FROM png_list LIMIT 1)")
        connection.execute("UPDATE crop_embedding SET prcfo='invalid-key' WHERE prcfo=(SELECT prcfo FROM crop_embedding LIMIT 1)")
    before = Path(db).read_bytes()
    with pytest.raises(ValueError, match='Cannot recover acquisition metadata'):
        _mil_frame_from_db(db)
    assert Path(db).read_bytes() == before


def test_mil_explicit_metadata_remains_authoritative(tmp_path):
    import sqlite3
    from spacr.embeddings import _mil_frame_from_db
    db, _ = _embedded_project(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE png_list SET plateID='explicit_plate', fieldID='f9'")
    frame = _mil_frame_from_db(db)
    assert frame.plateID.eq('explicit_plate').all() and frame.fieldID.eq('f9').all()


def _direct_mil_identity_table(well_column='well'):
    records = []
    for plate in ('plate_1', 'plate_2'):
        for well in range(4):
            for field in range(2):
                for cell in range(2):
                    records.append({
                        'plateID': plate, 'rowID': 'r1', 'columnID': f'c{well + 1}',
                        'fieldID': f'f{field + 1}', well_column: f'A{well + 1}',
                        'well_label': int(well < 2), 'emb_0': float(well + cell),
                        'emb_1': float(field)})
    return pd.DataFrame(records)


@pytest.mark.parametrize('column', ['plateID', 'rowID', 'columnID'])
@pytest.mark.parametrize('well_column', ['well', 'wellID', 'custom_well'])
def test_direct_mil_refuses_one_identifier_with_conflicting_acquisition_metadata(column, well_column):
    frame = _direct_mil_identity_table(well_column)
    second = frame.plateID.eq('plate_2')
    if column != 'plateID':
        frame.loc[:, 'plateID'] = 'plate_1'
        frame.loc[second, column] = 'another_identity'
    original = frame.copy(deep=True)
    kwargs = {'well_column': well_column} if well_column == 'custom_well' else {}
    with pytest.raises(ValueError, match='spans different acquisition wells'):
        _mil_bags(frame, **kwargs)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('layout', ['mixed_blanks', 'partial_metadata'])
def test_direct_mil_known_cross_plate_conflict_survives_incomplete_metadata(layout):
    frame = _direct_mil_identity_table()
    if layout == 'mixed_blanks':
        frame.loc[frame.index[::4], 'plateID'] = None
        frame.loc[frame.index[1::4], 'plateID'] = '   '
    else:
        frame = frame.drop(columns=['rowID', 'columnID'])
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match='spans different acquisition wells'):
        _mil_bags(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('well_column', ['well', 'wellID', 'custom_well'])
def test_direct_mil_complete_unique_identifiers_keep_independent_wells_and_fields(well_column):
    frame = _direct_mil_identity_table(well_column)
    frame[well_column] = frame.plateID + '_' + frame[well_column]
    original = frame.copy(deep=True)
    kwargs = {'well_column': well_column} if well_column == 'custom_well' else {}
    bags, labels, wells, positions, kept = _mil_bags(frame, **kwargs)
    assert len(bags) == len(wells) == 8 and labels.sum() == 4
    assert [bag.shape for bag in bags] == [(4, 2)] * 8
    assert all(kept.iloc[index].plateID.nunique() == 1 for index in positions)
    assert all(kept.iloc[index].fieldID.nunique() == 2 for index in positions)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('layout', ['absent', 'blank', 'known_and_blank'])
def test_direct_mil_unknown_metadata_does_not_invent_identity_or_change_legacy_input(layout):
    frame = _direct_mil_identity_table()
    if layout == 'absent':
        frame = frame.drop(columns=['plateID', 'rowID', 'columnID'])
    elif layout == 'blank':
        frame.loc[:, ['plateID', 'rowID', 'columnID']] = None
    else:
        frame.loc[:, 'plateID'] = 'one_known_plate'
        frame.loc[frame.index[::2], ['plateID', 'rowID', 'columnID']] = ' '
    original = frame.copy(deep=True)
    bags, labels, wells, positions, kept = _mil_bags(frame)
    assert len(bags) == 4 and labels.sum() == 2
    assert all(len(bag) == 8 for bag in bags)
    pd.testing.assert_frame_equal(frame, original)


def test_direct_mil_only_participating_rows_can_contradict_well_identity():
    frame = _direct_mil_identity_table()
    frame.well = frame.plateID + '_' + frame.well
    excluded = frame.iloc[[0]].copy()
    excluded.plateID = 'another_plate'
    excluded.emb_0 = np.nan
    frame = pd.concat([frame, excluded], ignore_index=True)
    original = frame.copy(deep=True)
    bags, labels, wells, positions, kept = _mil_bags(frame)
    assert len(bags) == 8 and len(kept) == len(frame) - 1
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_direct_mil_conflict_uses_actual_reviewed_locale_callback(language, monkeypatch):
    import hashlib
    import json

    monkeypatch.setenv('SPACR_LANGUAGE', language)
    source = 'Well identifier {well} spans different acquisition wells; use complete, unique well identifiers.'
    reviews = (Path(__file__).resolve().parents[1] / 'docs/i18n/reviewed/runtime'
               / language / '2026-10-15-mil-direct-well-identity.json')
    row = next(record for record in json.loads(reviews.read_text(encoding="utf-8"))['records']
               if record['source'] == source)
    assert row['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    frame = _direct_mil_identity_table()
    original = frame.copy(deep=True)
    with pytest.raises(ValueError) as error:
        _mil_bags(frame)
    assert str(error.value) == row['translation'].format(well='A1')
    pd.testing.assert_frame_equal(frame, original)
