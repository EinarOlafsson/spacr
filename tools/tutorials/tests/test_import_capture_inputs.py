"""The bounded demo must preserve image bytes AND acquisition identity."""
import importlib.util
import re
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'capture_image_import', Path(__file__).parents[1] / 'capture_image_import.py')
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def seed(stage, well, field):
    folder = stage / 'example_data/plate1'
    folder.mkdir(parents=True, exist_ok=True)
    for channel, acquisition in ((1, 2), (2, 1), (3, 3), (4, 2)):
        name = f'plate1_{well}_T0001F{field:03}L01A{acquisition:02}Z01C{channel:02}.tif'
        (folder / name).write_bytes(f'{well}-{field}-{channel}'.encode())


def test_prepared_copies_preserve_all_bytes_and_name_the_changed_token(tmp_path):
    seed(tmp_path, 'E01', 1)
    seed(tmp_path, 'E01', 9)
    seed(tmp_path, 'E02', 9)
    seed(tmp_path, 'L01', 1)
    seed(tmp_path, 'L01', 10)
    run, raw, prepared, images = capture.prepare_example(tmp_path)
    assert len(images) == 8
    assert {item['well'] for item in images} == {'E01', 'L01'}
    assert {item['field'] for item in images} == {1, 10}
    assert {item['channel'] for item in images} == {1, 2, 3, 4}
    assert not (run / 'spacr_project').exists()
    for item in images:
        source = Path(item['downloaded_source'])
        copied = Path(item['input'])
        original = Path(item['raw_copy'])
        assert copied.parent == prepared and original.parent == raw
        assert copied.read_bytes() == original.read_bytes() == source.read_bytes()
        assert capture.digest(copied) == item['sha256']
        assert original.name == source.name
        assert copied.name == re.sub(r'A\d+(?=Z\d+C\d+\.tif$)', '', source.name)
        assert not copied.is_symlink()


def test_one_well_is_not_misrepresented_as_a_complete_import_example(tmp_path):
    seed(tmp_path, 'E01', 1)
    seed(tmp_path, 'E01', 9)
    with pytest.raises(RuntimeError, match='Download the real Mask example'):
        capture.prepare_example(tmp_path)


def test_field_axis_must_also_vary_in_the_selected_example(tmp_path):
    seed(tmp_path, 'E01', 1)
    seed(tmp_path, 'L01', 1)
    with pytest.raises(RuntimeError, match='Download the real Mask example'):
        capture.prepare_example(tmp_path)
