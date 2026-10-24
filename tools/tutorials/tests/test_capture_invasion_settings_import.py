"""Nested Invasion metadata enters through the real AppScreen CSV picker."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from capture_invasion import import_invasion_settings
from spacr.qt.screens.app_screen import AppScreen


def test_real_invasion_import_picker_applies_nested_metadata_and_all_requested_settings(tmp_path):
    import json
    app = QApplication.instance() or QApplication([])
    screen = AppScreen('invasion')
    screen.resize(1500, 1000)
    screen.show()
    QTest.qWait(100)
    path = Path(__file__).resolve().parents[1] / 'authoring/catalog/26_invasion_settings.json'
    requested = json.loads(path.read_text())
    requested['stain_baseline_wells'] = requested.pop('control_wells')
    requested['src'] = str(tmp_path)
    requested['bimodality_cutoff'] = .555556
    try:
        proof = import_invasion_settings(app, screen, tmp_path, requested, None,
            lambda seconds: QTest.qWait(max(1, int(seconds * 1000))))
        actual = screen._settings_model.collect()
        assert all(actual[key] == value for key, value in requested.items())
        assert actual['pathogen_plate_metadata'] == [['c2'], ['c3']]
        assert proof['actual_file_picker'] and proof['requested_settings_read_back'] == len(requested)
        assert 'Loaded ' in screen._console.as_text()
        assert not screen._worker_thread_is_running()
        assert not hasattr(screen, '_run_started_at')
        assert Path(proof['path']).read_text().startswith('Key,Value')
        assert Path(proof['path']).with_suffix('.json').is_file()
        # A repeated request cannot overwrite the prepared settings evidence.
        original = Path(proof['path']).read_bytes()
        with pytest.raises(ValueError, match='fresh'):
            import_invasion_settings(app, screen, tmp_path, requested, None, lambda _: None)
        assert Path(proof['path']).read_bytes() == original
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
