"""Actual Map Barcodes default paths enter private storage through Qt edits."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import capture_ui
from capture_barcode_search import prepare_default_references
from capture_policy import verify_visible_paths
from map_barcodes_data import digest
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.settings_search import install as install_settings_search


def test_all_actual_bundled_csvs_are_identical_private_and_selected_through_ui(tmp_path, monkeypatch):
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)  # capture_refresh setup
    app = QApplication.instance() or QApplication([])
    screen = AppScreen('map_barcodes')
    install_settings_search(screen)  # The normal MainWindow post-build hook.
    screen.resize(1800, 1800)
    screen.show()
    screen._settings_search.set_level('all')
    screen._settings_search.set_query('csv')
    QTest.qWait(100)
    before = screen._settings_model.collect()
    nonempty_csvs = {key for key, value in before.items() if key.endswith('_csv') and value}
    assert nonempty_csvs == {'column_csv', 'row_csv', 'grna_csv'}
    sources = {key: digest(before[key]) for key in nonempty_csvs}
    edits = []
    class QtInput:
        def click(self, widget, pos=None):
            QTest.mouseClick(widget, Qt.LeftButton, pos=pos or widget.rect().center())
        def key(self, widget, key, modifiers=Qt.NoModifier):
            QTest.keyClick(widget, key, modifiers)
        def type_text(self, widget, value):
            edits.append((widget, value))
            QTest.keyClicks(widget, value)
        def pause(self, capture, seconds):
            QTest.qWait(5)
        def scroll_into_view(self, capture, scroll, widget, **kwargs):
            scroll.ensureWidgetVisible(widget)
            app.processEvents()
        def toggle_section(self, capture, section, expanded):
            if section.is_expanded() != expanded:
                self.click(section.header())
            app.processEvents()
    monkeypatch.setattr(capture_ui, '_h', lambda: QtInput())
    capture = SimpleNamespace(host=object())
    try:
        with pytest.raises(RuntimeError, match='outside the prepared capture directory'):
            verify_visible_paths([screen], tmp_path)
        proof = prepare_default_references(screen, tmp_path, capture, lambda _: QTest.qWait(10))
        after = screen._settings_model.collect()
        assert len([value for _, value in edits if value.endswith('.csv')]) == 3
        assert proof['initial_default_bytes_preserved'] and proof['other_settings_unchanged']
        for row in proof['files']:
            assert row['key'] in nonempty_csvs
            assert after[row['key']] == row['copy']
            assert not Path(row['copy']).samefile(row['source'])
            assert digest(row['source']) == digest(row['copy']) == sources[row['key']] == row['sha256']
            assert row['bytes'] == Path(row['source']).stat().st_size
        assert {key: value for key, value in before.items() if key not in nonempty_csvs} == {
            key: value for key, value in after.items() if key not in nonempty_csvs}
        verify_visible_paths([screen], tmp_path)
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('symlink', [False, True])
def test_private_reference_destination_is_exclusive_and_never_follows_existing_target(tmp_path, symlink):
    bundled = Path(__file__).resolve().parents[3] / 'spacr/resources/data'
    values = {key: str(bundled / name) for key, name in (
        ('column_csv', 'barcodes_column.csv'), ('row_csv', 'barcodes_row.csv'),
        ('grna_csv', 'barcodes_grna.csv'))}
    screen = SimpleNamespace(_settings_model=SimpleNamespace(collect=lambda: values))
    destination = tmp_path / 'bundled_barcode_references'
    if symlink:
        outside = tmp_path / 'outside'
        outside.mkdir()
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir()
        (destination / 'keep.csv').write_bytes(b'preserve existing proof')
    with pytest.raises(FileExistsError):
        prepare_default_references(screen, tmp_path, None, lambda _: None)
    if symlink:
        assert list(outside.iterdir()) == []
    else:
        assert (destination / 'keep.csv').read_bytes() == b'preserve existing proof'
