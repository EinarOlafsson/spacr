"""Import preview exposes the same explicit ScanR map as Mask settings."""
import numpy as np
import tifffile
from PySide6.QtWidgets import QLineEdit


def test_scanr_import_preview_blocks_missing_map_and_invalidates_changed_map(tmp_path, qtbot):
    """Actual preview stays read-only and requires a freshly agreed mapping."""
    from spacr.qt.screens.convert import ConvertScreen
    from spacr.settings import set_default_settings_preprocess_img_data

    source = tmp_path / 'plate'
    source.mkdir()
    for value, channel in enumerate(('nucleus-dapi', 'pm-647'), 11):
        tifffile.imwrite(source / f'--W00002--P00001--Z00000--T00000--{channel}.tif',
                         np.full((5, 7), value, dtype=np.uint16))
    originals = {path.name: path.read_bytes() for path in source.iterdir()}
    screen = ConvertScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_source(str(source))
    screen.set_destination(str(tmp_path / 'converted'))
    assert screen.preview()
    assert not screen.can_convert()
    assert screen.plan().errors
    edit = screen.findChild(QLineEdit, 'MetadataWellMap')
    assert edit is not None
    mapping = tmp_path / 'wells.csv'
    mapping.write_text('plate,index,well\nplate,W00002,B07\n', encoding='utf-8')
    edit.setText(str(mapping))
    assert screen.plan() is None
    assert screen.preview()
    assert screen.can_convert()
    assert {row.well for row in screen.plan().mappings} == {'B07'}
    edit.clear()
    assert screen.plan() is None and not screen.can_convert()
    assert not (tmp_path / 'converted').exists()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == originals
    assert set_default_settings_preprocess_img_data({})['metadata_well_map'] == ''
