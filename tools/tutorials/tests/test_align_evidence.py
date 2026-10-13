"""A reconstructed image does not prove its coordinate metadata is correct."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from align_evidence import check_coordinate_rows


def example():
    tiles = [{'path': 'one.npy', 'field': 1, 'source_yx': [0, 0]},
             {'path': 'two.npy', 'field': 2, 'source_yx': [0, 600]}]
    rows = [{'source': t['path'], 'plateID': 'tutorial', 'rowID': 'r1',
             'columnID': 'c1', 'fieldID': f"f{t['field']}",
             'canvas_y': t['source_yx'][0], 'canvas_x': t['source_yx'][1],
             'stack_path': 'new.npy'} for t in tiles]
    return tiles, rows


def test_reordered_rows_still_identify_the_correct_tiles():
    tiles, rows = example()
    assert check_coordinate_rows(tiles, rows[::-1], 'new.npy') == 2


@pytest.mark.parametrize('key,value', [('plateID', 'original'), ('rowID', 'r2'),
                                      ('columnID', 'c2'), ('fieldID', 'f3')])
def test_same_geometry_does_not_excuse_a_different_identity(key, value):
    tiles, rows = example()
    rows[0][key] = value
    with pytest.raises(ValueError, match='identities'):
        check_coordinate_rows(tiles, rows, 'new.npy')


@pytest.mark.parametrize('axis', ['canvas_y', 'canvas_x'])
def test_correct_ids_do_not_excuse_shifted_coordinates(axis):
    tiles, rows = example()
    rows[0][axis] += 1
    with pytest.raises(ValueError, match='canvas coordinates'):
        check_coordinate_rows(tiles, rows, 'new.npy')


def test_equal_row_count_on_another_source_is_not_the_same_export():
    tiles, rows = example()
    rows[0]['source'] = 'wrong.npy'
    with pytest.raises(ValueError, match='source files'):
        check_coordinate_rows(tiles, rows, 'new.npy')


def test_duplicate_source_is_not_a_new_tile():
    tiles, rows = example()
    with pytest.raises(ValueError, match='Duplicate'):
        check_coordinate_rows(tiles, rows + rows[:1], 'new.npy')


def test_coordinates_must_reference_the_actual_new_output():
    tiles, rows = example()
    rows[0]['stack_path'] = 'old.npy'
    with pytest.raises(ValueError, match='different output'):
        check_coordinate_rows(tiles, rows, 'new.npy')
