"""Private capture paths keep the real bundled weights and privacy guard."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtWidgets import QApplication, QLineEdit

from capture_plaque import prepare_bundled_model
from capture_policy import verify_visible_paths
from replication_demo import digest


def test_actual_bundled_bytes_pass_privacy_only_after_private_preparation(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = Path(__file__).resolve().parents[3] / (
        'spacr/resources/models/toxo_plaque_cyto_e25000_X1120_Y1120.CP_model')
    original = digest(source)
    stage = tmp_path / 'capture'
    stage.mkdir()
    field = QLineEdit(str(source))
    field.show()
    app.processEvents()
    try:
        with pytest.raises(RuntimeError, match='outside the prepared capture directory'):
            verify_visible_paths([field], stage)
        proof = prepare_bundled_model(stage, source)
        copied = Path(proof['copy'])
        assert copied == stage / 'plaque_models/bundled' / source.name
        assert not copied.samefile(source)
        assert proof['sha256'] == digest(copied) == digest(source) == original
        assert proof['bytes'] == source.stat().st_size
        field.setText(str(copied))
        verify_visible_paths([field], stage)
        assert prepare_bundled_model(stage, source) == proof
    finally:
        field.close()
        field.deleteLater()
        app.processEvents()


def test_conflicting_private_copy_is_preserved_and_refused(tmp_path):
    source = tmp_path / 'weights.CP_model'
    source.write_bytes(b'actual original checkpoint bytes')
    stage = tmp_path / 'capture'
    proof = prepare_bundled_model(stage, source)
    target = Path(proof['copy'])
    target.write_bytes(b'conflicting private checkpoint')
    with pytest.raises(ValueError, match='differs or aliases'):
        prepare_bundled_model(stage, source)
    assert target.read_bytes() == b'conflicting private checkpoint'
    assert source.read_bytes() == b'actual original checkpoint bytes'


def test_private_model_folder_cannot_escape_through_a_symlink(tmp_path):
    source = tmp_path / 'weights.CP_model'
    source.write_bytes(b'checkpoint')
    stage, outside = tmp_path / 'capture', tmp_path / 'outside'
    stage.mkdir()
    outside.mkdir()
    (stage / 'plaque_models').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='inside the capture stage'):
        prepare_bundled_model(stage, source)
    assert list(outside.iterdir()) == []
