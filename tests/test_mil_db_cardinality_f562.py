"""Real read-only SQLite joins preserve one unambiguous vector per object."""
import sqlite3

import pandas as pd
import pytest

from spacr.embeddings import _mil_frame_from_db


@pytest.fixture(autouse=True)
def english_errors(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


def crop():
    return dict(prcfo='plate1_r1_c1_f1_o1', png_path='/synthetic/cell.png',
                plateID='plate1', rowID='r1', columnID='c1', fieldID='f1')


def embedding():
    return dict(prcfo='plate1_r1_c1_f1_o1', emb_0=1.0, emb_1=2.0,
                _embedding_fingerprint='declared-synthetic-encoder')


def read_database(tmp_path, crops, stored):
    path = tmp_path / 'measurements.db'
    with sqlite3.connect(path) as connection:
        pd.DataFrame(crops).to_sql('png_list', connection, index=False)
        pd.DataFrame(stored).to_sql('crop_embedding', connection, index=False)
    before = path.read_bytes()
    try:
        return _mil_frame_from_db(path)
    finally:
        assert path.read_bytes() == before


@pytest.mark.parametrize('column', ['plateID', 'rowID', 'columnID', 'fieldID'])
def test_conflicting_object_acquisition_metadata_refuses_before_join(tmp_path, column):
    with pytest.raises(ValueError, match='Crop acquisition metadata disagrees'):
        read_database(tmp_path, [crop(), dict(crop(), **{column: 'other'})], [embedding()])


@pytest.mark.parametrize('column,value', [('emb_0', 99.0), ('_embedding_fingerprint', 'other-encoder')])
def test_conflicting_object_embeddings_or_provenance_refuse(tmp_path, column, value):
    with pytest.raises(ValueError, match='Stored embeddings disagree'):
        read_database(tmp_path, [crop()], [embedding(), dict(embedding(), **{column: value})])


def test_identical_embedding_rows_coalesce_without_inflating_cell_count(tmp_path):
    result = read_database(tmp_path, [crop()], [embedding(), embedding()])
    assert len(result) == 1
    assert result.emb_0.tolist() == [1.0]
    assert result._embedding_fingerprint.tolist() == ['declared-synthetic-encoder']


def test_alternative_crop_views_keep_first_valid_path_and_one_object(tmp_path):
    result = read_database(tmp_path, [crop(), dict(crop(), png_path='/synthetic/nucleus.png')], [embedding()])
    assert len(result) == 1 and result.png_path.tolist() == ['/synthetic/cell.png']


@pytest.mark.parametrize('column', ['plateID', 'rowID', 'columnID', 'fieldID'])
@pytest.mark.parametrize('missing', [None, '  '])
def test_later_known_metadata_is_not_discarded_or_replaced_by_parsed_key(tmp_path, column, missing):
    result = read_database(tmp_path, [dict(crop(), **{column: missing}), dict(crop(), **{column: 'explicit_override'})], [embedding()])
    assert len(result) == 1 and result[column].tolist() == ['explicit_override']


def test_blank_first_path_keeps_later_valid_alternative_view(tmp_path):
    result = read_database(tmp_path, [dict(crop(), png_path=None), crop()], [embedding()])
    assert len(result) == 1 and result.png_path.tolist() == ['/synthetic/cell.png']


def test_unique_rows_and_explicit_metadata_preserve_input_order(tmp_path):
    crops = [dict(crop(), prcfo='plate1_r1_c1_f1_o2', plateID='override'), crop()]
    stored = [dict(embedding(), prcfo='plate1_r1_c1_f1_o2', emb_0=3.0), embedding()]
    result = read_database(tmp_path, crops, stored)
    assert result.prcfo.tolist() == [row['prcfo'] for row in crops]
    assert result.emb_0.tolist() == [3.0, 1.0]
    assert result.plateID.tolist() == ['override', 'plate1']


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
@pytest.mark.parametrize('conflict', ['embedding', 'acquisition'])
def test_db_object_conflict_uses_installed_reviewed_language(tmp_path, language, conflict, monkeypatch):
    import json
    from pathlib import Path

    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-mil-db-cardinality.json'
    record = json.loads(review.read_text())['records'][0 if conflict == 'embedding' else 1]
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    crops = [crop()]
    stored = [embedding()]
    if conflict == 'embedding':
        stored.append(dict(embedding(), emb_0=99.0))
    else:
        crops.append(dict(crop(), fieldID='other'))
    with pytest.raises(ValueError) as error:
        read_database(tmp_path, crops, stored)
    assert str(error.value) == record['translation'].format(key=crop()['prcfo'])
    assert str(error.value) != record['source'].format(key=crop()['prcfo'])
