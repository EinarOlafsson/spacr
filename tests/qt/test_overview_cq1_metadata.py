"""Yokogawa CQ1 well indices, acquisition XMLs and Overview metadata autofill."""
import re
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from spacr import regex_infer
from spacr.qt.widgets import qc_field_browser as browser

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "yokogawa_cq1"
WELLS = (49, 52, 73)


def _cq1_source(tmp_path, xml=True, wells=WELLS, fields=(1, 2), channels=(1, 2, 3), folder="Projection"):
    import tifffile
    root = tmp_path / "20261015T112457_measurement"
    images = root / folder if folder else root
    images.mkdir(parents=True)
    if xml:
        for path in FIXTURE.iterdir():
            shutil.copy(path, root / path.name)
    for well in wells:
        for field in fields:
            for channel in channels:
                tifffile.imwrite(images / f"W{well:04}F{field:04}T0001Z000C{channel}.tif",
                                 np.full((6, 8), well * 10 + channel, dtype=np.uint16))
    (root / "MeasurementResultMIP.ome.tif").write_bytes(b"")
    return root, images


def _column_major_protocol(root):
    path = root / "MeasurementProtocol.xml"
    text = path.read_text(encoding="utf-8")

    def renumber(match):
        column, row = int(match.group(2)), int(match.group(3))
        return f'{match.group(1)}"{(column - 1) * 16 + row}" icm:Column="{column}" icm:Row="{row}"'
    text = re.sub(r'(<icm:Well icm:UniqueID="\d+" icm:Number=)"\d+" icm:Column="(\d+)" icm:Row="(\d+)"',
                  renumber, text)
    path.write_text(text, encoding="utf-8")


def test_cq1_acquisition_xml_reads_plate_channels_axes_and_pixel_size():
    acquisition = regex_infer._yokogawa_acquisition(FIXTURE)
    assert acquisition["instrument"] == "CQ1"
    assert (acquisition["rows"], acquisition["columns"]) == (16, 24)
    assert acquisition["well_order"] == "row"
    assert acquisition["wells"][49] == (3, 1)
    assert acquisition["pixel_size_um"] == pytest.approx(0.1625)
    assert (acquisition["fields"], acquisition["time_points"], acquisition["z_slices"]) == (9, 1, 10)
    channels = acquisition["channels"]
    assert sorted(channels) == [1, 2, 3]
    assert [channels[key]["name"] for key in (1, 2, 3)] == [
        "640nm BP685/40", "488nm BP525/50", "405nm BP447/60"]
    assert [channels[key]["excitation_nm"] for key in (1, 2, 3)] == [640., 488., 405.]
    assert [channels[key]["emission_nm"] for key in (1, 2, 3)] == [685., 525., 447.]
    assert [channels[key]["color"] for key in (1, 2, 3)] == ["#ff00ff", "#19ff06", "#00ffff"]


def test_cq1_well_index_maps_both_index_orders_and_explicit_overrides():
    acquisition = regex_infer._yokogawa_acquisition(FIXTURE)
    assert [regex_infer._yokogawa_well(index, acquisition) for index in (1, 49, 52, 73, 384)] == [
        (1, 1), (3, 1), (3, 4), (4, 1), (16, 24)]
    assert regex_infer._yokogawa_well(385, acquisition) is None
    no_table = dict(acquisition, wells={})
    assert regex_infer._yokogawa_well(49, no_table) == (3, 1)
    assert regex_infer._yokogawa_well(49, dict(no_table, well_order="column")) == (1, 4)
    assert regex_infer._yokogawa_well(13, acquisition, 8, 12, "row") == (2, 1)
    assert regex_infer._yokogawa_well(13, acquisition, 8, 12, "column") == (5, 2)
    assert regex_infer._yokogawa_well(97, None, 8, 12, "row") is None
    assert regex_infer._yokogawa_well(49) == (3, 1)


def test_cq1_column_major_well_table_is_recognised(tmp_path):
    root, images = _cq1_source(tmp_path)
    _column_major_protocol(root)
    acquisition = regex_infer._yokogawa_acquisition(images)
    assert acquisition["well_order"] == "column"
    assert regex_infer._yokogawa_well(49, dict(acquisition, wells={})) == (1, 4)
    fields = browser._overview_index(images, source_scope="folder")
    assert {field.well for field in fields} == {"A04", "D04", "I05"}


def test_overview_index_places_cq1_projection_wells_from_parent_xmls(tmp_path):
    root, images = _cq1_source(tmp_path)
    fields = browser._overview_index(images, source_scope="folder")
    assert len(fields) == 6
    assert {field.well for field in fields} == {"C01", "C04", "D01"}
    assert all(not field.unresolved for field in fields)
    assert {(field.row, field.column) for field in fields} == {("3", 1), ("3", 4), ("4", 1)}
    assert {field.field_id for field in fields} == {"0001", "0002"}
    assert all([channel for channel, _ in field.files] == ["1", "2", "3"] for field in fields)
    explicit = browser._overview_index(images, source_scope="folder", well_layout=(8, 12, "column"))
    assert {field.well for field in explicit} == {"A07", "D07", "A10"}


def test_cq1_raw_folder_and_measurement_root_use_the_same_xmls(tmp_path):
    root, images = _cq1_source(tmp_path, folder="")
    fields, info = browser._overview_wizard_probe(str(root), "", "auto", threading.Event(), "folder")
    matched = browser._overview_matched_fields(fields, info["detected_pattern"])
    assert info["detected_label"] == "cq1"
    assert {field.well for field in matched} == {"C01", "C04", "D01"}
    assert "MeasurementResultMIP.ome.tif" not in {Path(path).name for field in matched for _, path in field.files}
    assert info["acquisition"]["instrument"] == "CQ1"


def test_cq1_without_xml_uses_24_columns_and_reports_it(tmp_path):
    root, images = _cq1_source(tmp_path, xml=False, wells=(13, 25))
    fields, info = browser._overview_wizard_probe(str(images), "", "auto", threading.Event(), "folder")
    assert {field.well for field in fields} == {"A13", "B01"}
    acquisition = info["acquisition"]
    assert acquisition["numeric_wells"] and not acquisition["sources"]
    assert acquisition["rows"] is None
    assert any("24 columns" in sentence for sentence in
               browser._OverviewMetadataWizard._acquisition_sentences(acquisition))
    fields, _ = browser._overview_wizard_probe(str(images), "", "auto", threading.Event(), "folder",
                                               well_layout=(8, 12, "row"))
    assert {field.well for field in fields} == {"B01", "C01"}


def test_metadata_wizard_autofills_cq1_geometry_channels_and_pixel_size(qtbot, tmp_path):
    _root, images = _cq1_source(tmp_path)
    wizard = browser._OverviewMetadataWizard(str(images))
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    config = wizard.configuration()
    assert wizard.preset.currentData() == "cq1"
    assert config["geometry"] == (16, 24)
    assert config["channels"] == (1, 2, 3)
    assert config["channel_names"] == {1: "640nm BP685/40", 2: "488nm BP525/50", 3: "405nm BP447/60"}
    assert config["channel_colors"] == {1: "#ff00ff", 2: "#19ff06", 3: "#00ffff"}
    assert config["pixel_size_um"] == pytest.approx(0.1625)
    assert config["well_layout"] is None
    assert wizard.well_order.isEnabled() and wizard.well_order.currentData() == "row"
    assert wizard.channel_table.rowCount() == 3
    assert wizard.channel_table.item(0, 2).text() == "640 nm / BP685/40"
    columns = {wizard.preview.horizontalHeaderItem(index).text(): index
               for index in range(wizard.preview.columnCount())}
    first = {caption: wizard.preview.item(0, columns[browser.tr(caption)]).text()
             for caption in ("Well", "Row", "Column")}
    assert first == {"Well": "C01", "Row": "C", "Column": "1"}
    assert "CQ1" in wizard.axes.text() and "16 × 24" in wizard.axes.text()
    wizard.channel_table.item(1, 1).setText("LysoTracker")
    wizard.channel_table.item(1, 3).setText("yellow")
    wizard.pixel_size.setValue(0.325)
    config = wizard.configuration()
    assert config["channel_names"][2] == "LysoTracker"
    assert config["channel_colors"][2] == "#ffff00"
    assert config["pixel_size_um"] == pytest.approx(0.325)
    wizard.channel_table.item(1, 3).setText("not-a-colour")
    with pytest.raises(browser._OverviewValidationError, match="valid colour"):
        wizard.configuration()
    wizard.channel_table.item(1, 3).setText("#ffff00")
    wizard.well_order.setCurrentIndex(wizard.well_order.findData("column"))
    wizard._choose_well_order()
    qtbot.waitUntil(lambda: wizard.next.isEnabled() and wizard._fields
                    and {field.well for field in wizard._fields} == {"A04", "D04", "I05"}, timeout=10000)
    config = wizard.configuration()
    assert config["well_layout"] == (16, 24, "column")
    assert config["channel_names"][2] == "LysoTracker"
    assert config["pixel_size_um"] == pytest.approx(0.325)
    wizard.reject()


def test_cellvoyager_metadata_and_composite_are_unchanged(qtbot, tmp_path):
    import tifffile
    for channel in (1, 2):
        tifffile.imwrite(tmp_path / f"plate1_E01_T0001F001L01A01Z01C{channel:02}.tif",
                         np.full((6, 8), channel * 100, dtype=np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    config = wizard.configuration()
    assert wizard.preset.currentData() == "cellvoyager"
    assert config["geometry"] == (6, 8)
    assert config["channels"] == (1, 2)
    assert config["channel_colors"] == {1: "#ff0000", 2: "#00ff00"}
    assert config["pixel_size_um"] is None
    assert not wizard.well_order.isEnabled()
    wizard._choose_geometry()
    assert wizard._geometry_explicit and not wizard._layout_explicit
    assert {field.well for field in wizard._fields} == {"E01"}
    wizard.reject()
    field = browser._overview_index(tmp_path, source_scope="folder")[0]
    plain = browser._overview_pixels(field, (1, 2), (0, 100), edge=8)
    coloured = browser._overview_pixels(field, (1, 2), (0, 100), edge=8,
                                        channel_colors=config["channel_colors"])
    np.testing.assert_array_equal(plain, coloured)
    magenta = browser._overview_pixels(field, (1, 2), None, edge=8,
                                       channel_colors={1: "#ff00ff", 2: "#00ff00"})
    assert magenta.shape == plain.shape and magenta.dtype == np.uint8


def test_cellvoyager_measurement_detail_autofills_geometry_and_pixel_size(qtbot, tmp_path):
    import tifffile
    (tmp_path / "MeasurementDetail.mrf").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<bts:MeasurementDetail xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0" '
        'bts:RowCount="8" bts:ColumnCount="12" bts:FieldCount="4" bts:ZCount="1" bts:TimePointCount="1">\n'
        '  <bts:MeasurementChannel bts:Ch="1" bts:HorizontalPixelDimension="0.65" '
        'bts:VerticalPixelDimension="0.65" />\n'
        '  <bts:MeasurementChannel bts:Ch="2" bts:HorizontalPixelDimension="0.65" />\n'
        '</bts:MeasurementDetail>\n', encoding="utf-8")
    for channel in (1, 2):
        tifffile.imwrite(tmp_path / f"plate1_B03_T0001F001L01A01Z01C{channel:02}.tif",
                         np.ones((6, 8), dtype=np.uint16))
    acquisition = regex_infer._yokogawa_acquisition(tmp_path)
    assert acquisition["instrument"] == "CellVoyager"
    assert (acquisition["rows"], acquisition["columns"], acquisition["pixel_size_um"]) == (8, 12, 0.65)
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    config = wizard.configuration()
    assert config["geometry"] == (8, 12)
    assert config["pixel_size_um"] == pytest.approx(0.65)
    assert {field.well for field in wizard._fields} == {"B03"}
    wizard.reject()


def test_pipeline_cq1_well_conversion_reads_the_acquisition_plate(tmp_path):
    from spacr.utils import _convert_cq1_well_id, _extract_filename_metadata
    from spacr.qt.regex_detect import BUILTIN_REGEXES
    assert _convert_cq1_well_id("0049") == "C01"
    plate96 = dict(rows=8, columns=12, wells={}, well_order="row")
    assert _convert_cq1_well_id("0013", plate96) == "B01"
    _root, images = _cq1_source(tmp_path)
    _column_major_protocol(_root)
    names = sorted(path.name for path in images.iterdir())
    grouped = _extract_filename_metadata(names, str(images), re.compile(BUILTIN_REGEXES["cq1"]), "cq1")
    assert {key[1] for key in grouped} == {"A04", "D04", "I05"}


def test_default_channel_colours_follow_wavelengths_then_names_then_position():
    colour = regex_infer._channel_display_colour
    assert colour(emission_nm=447) == "#0000ff"
    assert colour(excitation_nm=488) == "#00ff00"
    assert colour(emission_nm=685) == "#ff00ff"
    assert colour("Hoechst 33342") == "#0000ff"
    assert colour("mCherry") == "#ff0000"
    assert colour("Brightfield") == "#ffffff"
    assert [colour(position=index) for index in range(3)] == ["#ff0000", "#00ff00", "#0000ff"]


def test_tiff_pixel_size_comes_from_imagej_or_centimetre_resolution_not_inches(tmp_path):
    import tifffile
    tifffile.imwrite(tmp_path / "imagej.tif", np.zeros((4, 4), np.uint16), imagej=True,
                     resolution=(1 / 0.5, 1 / 0.5), metadata={"unit": "um"})
    tifffile.imwrite(tmp_path / "cm.tif", np.zeros((4, 4), np.uint16),
                     resolution=(20000., 20000.), resolutionunit=3)
    tifffile.imwrite(tmp_path / "inch.tif", np.zeros((4, 4), np.uint16),
                     resolution=(96., 96.), resolutionunit=2)
    sizes = {}
    for name in ("imagej", "cm", "inch"):
        with tifffile.TiffFile(tmp_path / f"{name}.tif") as image:
            sizes[name] = browser._overview_tiff_pixel_size(image)
    assert sizes["imagej"] == pytest.approx(0.5)
    assert sizes["cm"] == pytest.approx(0.5)
    assert sizes["inch"] is None


def test_acquisition_xml_edge_cases_stay_explicit(tmp_path, monkeypatch):
    assert regex_infer._yokogawa_acquisition(tmp_path / "missing") is None
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "MeasurementDetail.mrf").write_text("<not xml", encoding="utf-8")
    assert regex_infer._yokogawa_acquisition(broken) is None
    monkeypatch.setattr(regex_infer, "_YOKOGAWA_XML_LIMIT", 1)
    assert regex_infer._xml_tree(FIXTURE / "ImagingResult.xml") is None
    monkeypatch.undo()
    mes = tmp_path / "mes"
    mes.mkdir()
    (mes / "plate.mes").write_text(
        '<?xml version="1.0"?>\n<bts:MeasurementSetting xmlns:bts="urn:bts">'
        '<bts:Channel bts:Ch="1" bts:Target="Hoechst" bts:LaserWavelength="405" bts:Filter="BP445/45"/>'
        '<bts:Channel bts:Ch="x" bts:Target="ignored"/>'
        '<bts:Channel bts:Ch="2" bts:Fluorophore="mCherry" bts:Filter="none"/>'
        '</bts:MeasurementSetting>', encoding="utf-8")
    first = regex_infer._yokogawa_acquisition(mes)
    assert first["instrument"] == "CellVoyager" and first["sources"] == ["plate.mes"]
    assert first["channels"][1]["excitation_nm"] == 405. and first["channels"][1]["emission_nm"] == 445.
    assert first["channels"][1]["color"] == "#0000ff"
    assert first["channels"][2]["color"] == "#ff0000"
    first["channels"].clear()
    assert regex_infer._yokogawa_acquisition(mes)["channels"][2]["name"] == "mCherry"
    plate_only = tmp_path / "plate_only"
    plate_only.mkdir()
    (plate_only / "ImagingResult.xml").write_text('<r xmlns:icm="urn:i"/>', encoding="utf-8")
    (plate_only / "MP_Greiner_12x8_1.xml").write_text("<x/>", encoding="utf-8")
    shape = regex_infer._yokogawa_acquisition(plate_only)
    assert (shape["rows"], shape["columns"]) == (8, 12)
    table = tmp_path / "table"
    table.mkdir()
    (table / "MeasurementProtocol.xml").write_text(
        '<p xmlns:icm="urn:i"><icm:Well icm:Number="7" icm:Row="2" icm:Column="3"/>'
        '<icm:Well icm:Number="bad"/><icm:Channel icm:Number="z"/></p>', encoding="utf-8")
    shaped = regex_infer._yokogawa_acquisition(table)
    assert (shaped["rows"], shaped["columns"], shaped["well_order"]) == (2, 3, "table")
    assert regex_infer._well_index_order({}, 2, 3) is None
    assert regex_infer._argb_colour("#AABBCC") == "#aabbcc"
    assert regex_infer._argb_colour("red") is None
    assert regex_infer._wavelength_nm("no wavelength") is None
    assert regex_infer._xml_number("abc") is None
    assert regex_infer._yokogawa_well("x") is None and regex_infer._yokogawa_well(0) is None
    assert regex_infer._yokogawa_well(5, None, None, None, "column") is None
    assert regex_infer._yokogawa_metadata_folder(Path(tmp_path.anchor), depth=5) is None


def test_acquisition_partial_files_and_cache_limits(tmp_path, monkeypatch):
    assert [regex_infer._wavelength_colour(value) for value in (490, 560, 580, 620, 700)] == [
        "#00ffff", "#00ff00", "#ffff00", "#ff0000", "#ff00ff"]
    folder = tmp_path / "partial"
    folder.mkdir()
    (folder / "MeasurementProtocol.xml").write_text(
        '<p xmlns:icm="urn:i"><icm:ContainerPath>MP_Broken_12x8_1.xml</icm:ContainerPath>'
        '<icm:ZSetting icm:Range="0" icm:Slices="1"/></p>', encoding="utf-8")
    (folder / "MP_Broken_12x8_1.xml").write_text("<broken", encoding="utf-8")
    (folder / "ImageViewSetting.xml").write_text(
        '<v xmlns:icm="urn:i"><icm:ChannelViewSetting icm:Ch="4"><icm:ViewSetting icm:Color="none"/>'
        '<icm:ViewSetting icm:Color="#FF102030"/></icm:ChannelViewSetting></v>', encoding="utf-8")
    partial = regex_infer._yokogawa_acquisition(folder)
    assert (partial["rows"], partial["columns"], partial["z_slices"]) == (8, 12, 1)
    assert "z_step_um" not in partial
    assert partial["channels"][4]["color"] == "#102030"
    mrf = tmp_path / "mrf"
    mrf.mkdir()
    (mrf / "MeasurementDetail.mrf").write_text(
        '<bts:MeasurementDetail xmlns:bts="urn:b" bts:RowCount="16" bts:ColumnCount="24" bts:ZCount="0">'
        '<bts:MeasurementChannel bts:Ch="1" bts:LaserNumber="1" bts:LaserWavelength="n/a"/>'
        '</bts:MeasurementDetail>', encoding="utf-8")
    detail = regex_infer._yokogawa_acquisition(mrf)
    assert "z_slices" not in detail and detail["channels"][1]["excitation_nm"] is None
    monkeypatch.setattr(regex_infer, "_YOKOGAWA_FOLDER_CACHE", {str(index): None for index in range(33)})
    assert regex_infer._yokogawa_acquisition(mrf)["rows"] == 16
    assert list(regex_infer._YOKOGAWA_FOLDER_CACHE) == [str(mrf)]


def test_pipeline_cq1_index_outside_the_plate_is_kept():
    from spacr.utils import _convert_cq1_well_id
    assert _convert_cq1_well_id("0097", dict(rows=8, columns=12, wells={})) == "0097"


def test_tiff_pixel_size_reads_ome_and_refuses_unusable_metadata(tmp_path):
    import tifffile
    from types import SimpleNamespace
    tifffile.imwrite(tmp_path / "ome.tif", np.zeros((4, 4), np.uint16),
                     description='<OME><Pixels PhysicalSizeX="325" PhysicalSizeXUnit="nm"/></OME>',
                     metadata=None)
    tifffile.imwrite(tmp_path / "parsec.tif", np.zeros((4, 4), np.uint16),
                     description='<OME><Pixels PhysicalSizeX="1" PhysicalSizeXUnit="pc"/></OME>', metadata=None)
    with tifffile.TiffFile(tmp_path / "parsec.tif") as image:
        assert browser._overview_tiff_pixel_size(image) is None
    tifffile.imwrite(tmp_path / "zero.tif", np.zeros((4, 4), np.uint16),
                     description='<OME><Pixels PhysicalSizeX="0"/></OME>', metadata=None)
    with tifffile.TiffFile(tmp_path / "ome.tif") as image:
        assert browser._overview_tiff_pixel_size(image) == pytest.approx(0.325)
    with tifffile.TiffFile(tmp_path / "zero.tif") as image:
        assert browser._overview_tiff_pixel_size(image) is None
    assert browser._overview_tiff_pixel_size(SimpleNamespace(pages=[])) is None
    page = SimpleNamespace(description="", tags={"XResolution": SimpleNamespace(value=(0, 1))})
    assert browser._overview_tiff_pixel_size(SimpleNamespace(pages=[page])) is None
    page = SimpleNamespace(description="", tags={"XResolution": SimpleNamespace(value=4.)})
    assert browser._overview_tiff_pixel_size(SimpleNamespace(pages=[page], imagej_metadata={"unit": "micron"})) == .25


def test_channel_colour_weights_refuse_incomplete_or_invalid_colours():
    assert browser._overview_channel_rgb((1, 2), {1: "#ff0000"}) is None
    assert browser._overview_channel_rgb((1,), {"1": "#zz0000"}) is None
    assert browser._overview_channel_rgb((1,), {"1": "#ff8000"}) == [(1., 128 / 255., 0.)]


def test_browser_uses_wizard_names_colours_layout_and_pixel_size(qtbot, tmp_path):
    _root, images = _cq1_source(tmp_path, wells=(13,), fields=(1,), channels=(1, 2))
    dialog = browser._FieldOverview(settings={
        "source": str(images), "source_scope": "folder", "custom_regex": regex_infer._CQ1_FILENAME,
        "channels": (1, 2), "geometry": (8, 12), "well_layout": (8, 12, "row"),
        "channel_names": {1: "640nm BP685/40", "2": "LysoTracker"},
        "channel_colors": {1: "#ff00ff", 2: "#00ff00"}, "pixel_size_um": 0.1625})
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        dialog._scan()
        qtbot.waitUntil(lambda: bool(dialog._fields) and not dialog._index_jobs.is_busy(), timeout=10000)
        assert {field.well for field in dialog._fields} == {"B01"}
        assert dialog._display_kwargs()["channel_colors"] == {1: "#ff00ff", 2: "#00ff00"}
        dialog._install_channel_buttons({"channels": (1, 2)})
        assert [control.text() for control in dialog._channel_buttons.values()] == [
            "1 640nm BP685/40", "2 LysoTracker"]
        dialog._update_image_metadata(dialog._fields[0])
        assert "Pixel size: 0.1625 µm" in dialog._image_metadata.text()
    finally:
        dialog._cancel()
        for runner in (dialog._index_jobs, dialog._image_jobs, dialog._thumbnail_jobs,
                       dialog._field_thumbnail_jobs, dialog._montage_jobs, dialog._export_jobs,
                       dialog._metadata_jobs, dialog._measurement_jobs, dialog._native_jobs):
            qtbot.waitUntil(lambda runner=runner: not runner.is_busy(), timeout=10000)
        dialog.close()


def test_wizard_restores_saved_layout_pixel_size_and_unlisted_geometry(qtbot, tmp_path):
    root, images = _cq1_source(tmp_path, wells=(49,), fields=(1,), channels=(1,))
    plate = root / "MP_PerkinElmer_#6007430_24x16_1.xml"
    plate.write_text(plate.read_text(encoding="utf-8-sig").replace('ColumnCount="24" icm:RowCount="16"',
                     'ColumnCount="20" icm:RowCount="10"'), encoding="utf-8")
    protocol = root / "MeasurementProtocol.xml"
    protocol.write_text(re.sub(r"<icm:Well .*?</icm:Well>", "", protocol.read_text(encoding="utf-8"),
                               flags=re.S), encoding="utf-8")
    wizard = browser._OverviewMetadataWizard(str(images), {"pixel_size_um": 0.5})
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    config = wizard.configuration()
    assert config["geometry"] == (10, 20)
    assert wizard.plate_geometry.currentText() == "10 × 20 plate"
    assert config["pixel_size_um"] == pytest.approx(0.5)
    assert {field.well for field in wizard._fields} == {"C09"}
    wizard.channel_table.takeItem(0, 3)
    wizard._paint_colour_cell(0)
    assert wizard.configuration()["channel_colors"] == {}
    wizard.plate_geometry.setCurrentIndex(next(index for index in range(wizard.plate_geometry.count())
                                               if wizard.plate_geometry.itemData(index) == (16, 24)))
    wizard._choose_geometry()
    qtbot.waitUntil(lambda: wizard.next.isEnabled() and {f.well for f in wizard._fields} == {"C01"}, timeout=10000)
    assert wizard.configuration()["well_layout"] == (16, 24, "row")
    wizard.reject()
    restored = browser._OverviewMetadataWizard(str(images), {"well_layout": (8, 12, "column")})
    qtbot.addWidget(restored)
    restored.show()
    qtbot.waitUntil(restored.next.isEnabled, timeout=10000)
    assert restored.well_order.currentData() == "column"
    assert {field.well for field in restored._fields} == {"A07"}
    restored.reject()
