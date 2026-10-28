"""Round-5 Overview: per-channel normalization scope (image, well, plate, all loaded plates).

Item 725, maintainer feedback 2026-10-25 round 5: "there should be normalization
options for normalizing on different levels. like the image ..., on the well
level, plate level, and all." and "the user must be able to treat each channel
individually."
"""
import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr.qt.widgets import qc_field_browser as browser
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


def _plate(folder, gains):
    folder.mkdir(parents=True, exist_ok=True)
    ramp = np.arange(32 * 40, dtype=np.float64).reshape(32, 40)
    for well, gain in gains.items():
        tifffile.imwrite(folder / f"p1-{well}-f1-c1.tif", (ramp * gain).astype(np.uint16))
        tifffile.imwrite(folder / f"p1-{well}-f1-c2.tif", (ramp[::-1] * gain + 7).astype(np.uint16))
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, tmp_path):
    first = _plate(tmp_path / "plateA", {"A01": 1, "A02": 4})
    second = _plate(tmp_path / "plateB", {"A01": 10})
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": f"{first}; {second}", "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.low.setValue(0.)
    dialog.high.setValue(100.)
    return dialog


def _field(view, plate, well):
    return next(field for field in view._fields if field.plate == plate and field.well == well)


def _limits(view, plate, well):
    return view._channel_limits.get(browser._overview_image_key(_field(view, plate, well)), {})


def _scope(view, qtbot, value):
    view._set_normalization_scopes(browser._overview_channel_scopes(value))
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)


def _pixels(view, plate, well, channels=(1,)):
    options = view._display_kwargs(view=True)
    options.pop("enhancement", None)
    return browser._overview_pixels(_field(view, plate, well), channels, (0., 100.), edge=None, **options)


def test_the_scope_setting_reads_and_writes_per_channel():
    assert browser._overview_channel_scopes("plate") == {"*": "plate"}
    assert browser._overview_channel_scopes("") == {"*": "image"}
    assert browser._overview_channel_scopes("1=plate, 2=image, 3=bogus") == {"1": "plate", "2": "image", "3": "image"}
    assert browser._overview_scope_text({"1": "plate", "2": "image"}) == "1=plate, 2=image"
    assert browser._overview_scope_text({"1": "well", "2": "well"}) == "well"
    assert browser._overview_scope_of({"1": "plate"}, 2) == "image"


def test_two_wells_share_one_mapping_under_plate_scope_and_not_under_image_scope(view, qtbot):
    _scope(view, qtbot, "image")
    assert view._channel_limits == {}
    dim, bright = _pixels(view, "plateA", "A01"), _pixels(view, "plateA", "A02")
    assert (dim == bright).all()
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert _limits(view, "plateA", "A01") == _limits(view, "plateA", "A02")
    low, high = _limits(view, "plateA", "A01")[1]
    assert low == 0. and high == pytest.approx(4 * (32 * 40 - 1), rel=1e-3)
    dim, bright = _pixels(view, "plateA", "A01"), _pixels(view, "plateA", "A02")
    assert bright[..., 0].max() == 255 and dim[..., 0].max() == pytest.approx(255 / 4, abs=1)
    _scope(view, qtbot, "well")
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01") != _limits(view, "plateA", "A02"), timeout=15000)


def test_all_scope_spans_plates(view, qtbot):
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateB", "A01")), timeout=15000)
    assert _limits(view, "plateA", "A01") != _limits(view, "plateB", "A01")
    _scope(view, qtbot, "all")
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01") == _limits(view, "plateB", "A01")
                    == _limits(view, "plateA", "A02"), timeout=15000)
    assert _limits(view, "plateB", "A01")[1][1] == pytest.approx(10 * (32 * 40 - 1), rel=1e-3)


def test_each_channel_keeps_its_own_scope(view, qtbot):
    image_scoped = _pixels(view, "plateA", "A01", (1, 2))
    _scope(view, qtbot, "1=plate, 2=image")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert set(_limits(view, "plateA", "A01")) == {1}
    mixed = _pixels(view, "plateA", "A01", (1, 2))
    assert (mixed[..., 1] == image_scoped[..., 1]).all()
    assert not (mixed[..., 0] == image_scoped[..., 0]).all()


def test_the_scope_cache_is_reused_and_invalidated(view, qtbot, monkeypatch):
    calls = []
    real = browser._overview_scope_statistics

    def counted(groups, *args, **kwargs):
        calls.append(len(groups))
        return real(groups, *args, **kwargs)

    monkeypatch.setattr(browser, "_overview_scope_statistics", counted)
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert calls == [4]
    view._refresh_scope_limits()
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    assert calls == [4]
    view.high.setValue(50.)
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=15000)
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A02").get(1, (0, 0))[1]
                    == pytest.approx(np.percentile(np.concatenate([np.arange(32 * 40), np.arange(32 * 40) * 4.]), 50.),
                                     rel=1e-2), timeout=15000)
    view._set_load_limits(1, None)
    qtbot.waitUntil(lambda: len(calls) == 3, timeout=15000)
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01").get(1, (0, 0))[1]
                    == pytest.approx(np.percentile(np.arange(32 * 40), 50.), rel=1e-2), timeout=15000)
    _scope(view, qtbot, "image")
    assert view._channel_limits == {}


def test_thumbnails_follow_the_scope(view, qtbot):
    view.plates.setCurrentIndex(view.plates.findData("plateA"))
    item = view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")])
    qtbot.waitUntil(lambda: not item.icon().isNull(), timeout=20000)
    before = item.data(Qt.UserRole + 1)
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    qtbot.waitUntil(lambda: view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]).data(Qt.UserRole + 1) not in (None, before)
                    and not view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]).icon().isNull(), timeout=20000)
    after = view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]).data(Qt.UserRole + 1)
    assert after is not None and after != before
    assert _limits(view, "plateA", "A01") == _limits(view, "plateA", "A02")


def test_a_scoped_channel_with_its_own_percentiles_is_stretched_at_them(view, qtbot):
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert _limits(view, "plateA", "A01")[1][1] == pytest.approx(4 * (32 * 40 - 1), rel=1e-3)
    view._channel_overrides[1] = {"percentile_low": 0., "percentile_high": 50.}
    view._channel_display_changed()
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    ramp = np.arange(32 * 40, dtype=np.float64)
    expected = float(np.percentile(np.concatenate([ramp, ramp * 4]), 50.))
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01")[1][1] == pytest.approx(expected, rel=1e-3),
                    timeout=15000)
    assert _limits(view, "plateA", "A01")[1] == _limits(view, "plateA", "A02")[1]
    assert _limits(view, "plateA", "A01")[2][1] == pytest.approx(4 * (32 * 40 - 1) + 7, rel=1e-3)


def test_the_canvas_snapshot_draws_the_shown_display_without_the_enhancement(view, qtbot):
    _scope(view, qtbot, "1=plate, 2=image")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    view._channel_overrides[2] = {"percentile_low": 10., "percentile_high": 60.}
    view._channel_display_changed()
    view._set_channel_targets(None, (2,))
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    snapshot = view._canvas_snapshot()
    shown = view._display_kwargs(view=True)
    for key in ("channel_limits", "percentile_channels", "channel_percentiles"):
        assert snapshot._display[key] == shown[key]
    assert "enhancement" not in snapshot._display and "channel_enhancement" not in snapshot._display
    field = _field(view, "plateA", "A02")
    drawn = snapshot(browser._overview_canvas_image(field), None)
    shown.pop("enhancement", None)
    expected = browser._overview_pixels(field, snapshot._channels, snapshot._limits, edge=None, **shown)
    assert (drawn == expected[..., :3]).all()


@pytest.mark.parametrize('value, expected', [
    (None, None), (True, None), (3, (3,)), ([1, '2'], (1, 2)),
    ('  ', None), ('all', None), ('2,1', (2, 1)),
])
def test_channel_target_input_keeps_typed_physical_ids(value, expected):
    """Typed settings preserve channel identity instead of indexing channel positions."""
    assert browser._overview_channel_targets(value) == expected


def test_native_setting_controls_commit_values_and_ignore_unrepresentable_input(qtbot):
    """The supported native editors keep their last representable value on bad input."""
    from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QLineEdit, QSpinBox, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    text, check, combo, spin = QLineEdit(host), QCheckBox(host), QComboBox(host), QSpinBox(host)
    label = QLabel(host)
    combo.addItem('Full image', None)
    combo.addItem('Two pixels', 2)
    spin.setRange(0, 10)
    browser._overview_control_changed(text, lambda: label.setText(text.text()))
    browser._overview_set_control(text, [2, 1])
    assert text.text() == label.text() == '2,1'
    browser._overview_set_control(text, None)
    assert text.text() == label.text() == ''
    browser._overview_set_control(check, True)
    assert browser._overview_control_value(check) is True
    assert browser._overview_control_value(combo) == 'full'
    browser._overview_set_control(combo, 2)
    browser._overview_set_control(combo, 'unavailable')
    assert browser._overview_control_value(combo) == 2
    browser._overview_set_control(spin, 7)
    browser._overview_set_control(spin, 'not a number')
    assert browser._overview_control_value(spin) == 7
    assert browser._overview_control_value(host) is None
    assert browser._overview_channel_list_error(2, (1, 2)) == ''
    assert browser._overview_channel_list_error(3, (1, 2))


@pytest.fixture
def workspace(qtbot, tmp_path):
    """An actual Overview module with tiny physical plate/channel inputs and owned cleanup."""
    from spacr.qt.screens.app_screen import AppScreen

    first = _plate(tmp_path / 'workspaceA', {'A01': 1, 'A02': 4})
    second = _plate(tmp_path / 'workspaceB', {'A01': 10})
    screen = AppScreen('overview')

    def close(widget):
        page = widget._overview_page
        if page.browser is not None:
            page.stop()
            runners = (*_runners(page.browser), page.browser._levels_jobs)
            qtbot.waitUntil(lambda: all(r.active_jobs() == 0 for r in runners), timeout=15000)
            for runner in runners:
                runner.shutdown(timeout_ms=3000)
        page.retire()
        widget.close()

    qtbot.addWidget(screen, before_close_func=close)
    screen.resize(1500, 1000)
    screen.show()
    screen._settings_model.set_value_for_key('src', f'{first}; {second}')
    return screen, (first, second)


def _workspace_browser(screen, qtbot):
    """Take the real metadata Next transition before exercising a mounted workspace."""
    page = screen._overview_page
    wizard = page.run(screen._settings_model.collect())
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 3
                    and page.browser._normalization_ready and len(page.browser._channel_buttons) == 2,
                    timeout=15000)
    return page.browser


def test_workspace_metadata_reuse_cancel_and_new_source_keep_module_settings(workspace, qtbot):
    """Cancel and source replacement never borrow a stale metadata step or edit the source."""
    screen, folders = workspace
    model, page = screen._settings_model, screen._overview_page
    original = model._read_value('src')
    assert page.export_menu() is None and page._current_image_paths() == []
    page.stop()
    page._back()
    assert page.stack.currentWidget() is page.placeholder
    wizard = page.run(model.collect())
    assert page.run(model.collect()) is wizard
    assert page._metadata_step_shown()
    wizard.reject()
    assert page.wizard is None and page.stack.currentWidget() is page.placeholder
    assert model._read_value('src') == original
    another = page.run(dict(model.collect(), src=str(folders[0])))
    replacement = page.run(dict(model.collect(), src=str(folders[1])))
    assert replacement is not another and page.wizard is replacement
    assert replacement.source.text() == str(folders[1])
    page.stop()
    assert replacement._cancelled.is_set()
    replacement.reject()
    assert page.wizard is None and model._read_value('src') == original


def test_workspace_invalid_apply_detach_and_channel_drafts_preserve_physical_pixels(workspace, qtbot):
    """Invalid staged input and retired bridges cannot change source pixels or channel ownership."""
    screen, folders = workspace
    model, page = screen._settings_model, screen._overview_page
    before = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    view = _workspace_browser(screen, qtbot)
    bridge = page._bridge
    source = model._read_value('src')
    channels = tuple(view._channel_buttons)
    model.set_value_for_key('overview_ink_intensity', .6)
    model.set_value_for_key('overview_normalize_channels', '1,7')
    assert bridge.apply() is False
    assert view._ink_intensity == 1. and view._normalize_channels is None
    assert '7' in view.status.text() and bridge.pending()
    model.set_value_for_key('overview_normalize_channels', '2')
    assert bridge.apply() is True
    assert view._ink_intensity == .6 and view._normalize_channels == (2,)
    view.norm_channel.setCurrentIndex(view.norm_channel.findData(1))
    model.set_value_for_key('overview_ink_intensity', .2)
    view._apply_channel_normalization()
    assert view._channel_overrides[1]['ink'] == .2 and view._ink_intensity == .6
    view.norm_channel.setCurrentIndex(view.norm_channel.findData(None))
    assert model._read_value('overview_ink_intensity') == .6
    assert tuple(view._channel_buttons) == channels
    assert {path: path.read_bytes() for path in before} == before
    view._cancel()
    runners = (*_runners(view), view._levels_jobs)
    qtbot.waitUntil(lambda: all(r.active_jobs() == 0 for r in runners), timeout=15000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    page._retire_browser()
    assert page.browser is None and page._bridge is None and not page._rows and not page._hidden_keys
    assert bridge.apply() is False
    model.set_value_for_key('overview_ink_intensity', .8)
    assert model._read_value('src') == source
    assert model._read_value('overview_ink_intensity') == .8
    assert {path: path.read_bytes() for path in before} == before


def test_channel_widgets_remount_choices_without_losing_the_model_contract(workspace, qtbot, monkeypatch):
    """Real switches, scope choices and colour picks commit to their own physical channel."""
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker

    screen, _ = workspace
    view = _workspace_browser(screen, qtbot)
    model, page = screen._settings_model, screen._overview_page
    fields = [field for _section, field, _widgets in page._rows]
    toggles = next(field for field in fields if isinstance(field, browser._OverviewChannelToggles))
    scopes = next(field for field in fields if isinstance(field, browser._OverviewChannelScopes))
    colours = next(field for field in fields if isinstance(field, browser._OverviewCustomColors))
    assert tuple(toggles._toggles) == tuple(scopes._choices) == (1, 2)
    toggles.set_channels((1, 2))
    toggles._toggles[1].setChecked(False)
    toggles._toggles[2].setChecked(False)
    assert model._read_value(toggles._key) == 'none'
    toggles._toggles[1].setChecked(True)
    assert model._read_value(toggles._key) == '1'
    toggles._toggles[2].setChecked(True)
    assert model._read_value(toggles._key) == 'all'
    scopes._choices[1].setCurrentIndex(scopes._choices[1].findData('plate'))
    scopes._choices[2].setCurrentIndex(scopes._choices[2].findData('well'))
    scopes.set_channels((2,))
    assert tuple(scopes._choices) == (2,) and scopes._choices[2].currentData() == 'well'
    scopes.set_channels((1, 2))
    assert scopes._choices[1].currentData() == 'plate'
    model.set_value_for_key('overview_color_mode', 'custom')
    colours.set_channels((2,))
    colours.set_channels((1, 2))
    colours.set_channels((1, 2))
    old = model._read_value('overview_custom_colors')
    monkeypatch.setattr(colour_picker, 'pick_colour', lambda *_: QColor())
    qtbot.mouseClick(colours._buttons['1'], Qt.LeftButton)
    assert model._read_value('overview_custom_colors') == old
    monkeypatch.setattr(colour_picker, 'pick_colour', lambda *_: QColor('#123abc'))
    qtbot.mouseClick(colours._buttons['2'], Qt.LeftButton)
    assert colours.colours()['2'] == '#123abc'
    assert '#123abc' in model._read_value('overview_custom_colors')
    assert tuple(view._channel_buttons) == (1, 2)


def test_workspace_height_grip_drag_and_reset_preserve_category_expansion(qtbot):
    """Native pointer events resize real category geometry, then restore content height."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QLabel
    from spacr.qt.widgets.section import Section

    section = Section('Export', expanded=True)
    section.add_prose(QLabel('A real export row'))
    qtbot.addWidget(section)
    section.resize(400, 160)
    grip = browser._overview_height_grip(section)
    assert browser._overview_height_grip(section) is grip
    section.show()
    qtbot.waitUntil(lambda: grip.isVisible())
    start = section._body.height()
    local = QPointF(grip.rect().center())
    global_point = QPointF(grip.mapToGlobal(grip.rect().center()))
    for event in (
        QMouseEvent(QEvent.MouseButtonPress, local, global_point, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier),
        QMouseEvent(QEvent.MouseMove, local + QPointF(0, 45), global_point + QPointF(0, 45),
                    Qt.NoButton, Qt.LeftButton, Qt.NoModifier),
        QMouseEvent(QEvent.MouseButtonRelease, local + QPointF(0, 45), global_point + QPointF(0, 45),
                    Qt.LeftButton, Qt.NoButton, Qt.NoModifier),
    ):
        from PySide6.QtWidgets import QApplication
        QApplication.sendEvent(grip, event)
    assert section._body.minimumHeight() == start + 45
    qtbot.mouseDClick(grip, Qt.LeftButton)
    assert section._body.minimumHeight() == 0 and section.is_expanded()


def test_workspace_back_returns_the_actual_field_through_well_to_plate(workspace, qtbot):
    """Back navigates real populated views without changing plate, field or source identity."""
    screen, folders = workspace
    view = _workspace_browser(screen, qtbot)
    page, model = screen._overview_page, screen._settings_model
    source = model._read_value('src')
    before = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    plate = view.plates.currentData()
    well = next(field.well for field in view._fields if field.plate == plate)
    item = view.wells.item(*view._well_positions[(plate, well)])
    view._well(item)
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=15000)
    field_item = view.fields.item(0)
    field = field_item.data(Qt.UserRole)
    view._field(field_item)
    assert view._individual and view._main_view_field.identity == field.identity
    assert view._scope == (plate, well)
    page._back()
    assert not view._individual and view._scope == (plate, well)
    assert not page.back_button.isHidden()
    qtbot.mouseClick(page.back_button, Qt.LeftButton)
    assert view._scope == (plate, None) and not view._individual
    assert page.back_button.isHidden()
    assert field.identity in {entry.identity for entry in view._fields}
    assert view.plates.currentData() == plate and model._read_value('src') == source
    assert {path: path.read_bytes() for path in before} == before


def test_workspace_stop_cancels_only_its_browser_and_keeps_loaded_source_identity(workspace, qtbot):
    """Stop cancels the browser generation while preserving its loaded metadata and files."""
    screen, folders = workspace
    view = _workspace_browser(screen, qtbot)
    page, model = screen._overview_page, screen._settings_model
    source = model._read_value('src')
    fields = tuple(field.identity for field in view._fields)
    channels = tuple(view._channel_buttons)
    before = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    assert not view._cancelled.is_set()
    page.stop()
    assert view._cancelled.is_set() and page.browser is view
    runners = (*_runners(view), view._levels_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=15000)
    assert tuple(field.identity for field in view._fields) == fields
    assert tuple(view._channel_buttons) == channels and model._read_value('src') == source
    assert {path: path.read_bytes() for path in before} == before


def test_workspace_rejects_a_real_nonmask_folder_without_replacing_the_saved_path(
        workspace, qtbot, tmp_path, monkeypatch):
    """A native directory selection must pass the real mask validator before it is committed."""
    from PySide6.QtWidgets import QFileDialog, QLineEdit

    screen, folders = workspace
    view = _workspace_browser(screen, qtbot)
    model, page = screen._settings_model, screen._overview_page
    model.set_value_for_key('overview_objects_path', str(folders[0]))
    row = model._widgets['overview_objects_path']
    edit = row if isinstance(row, QLineEdit) else row.findChildren(QLineEdit)[0]
    saved = edit.text()
    invalid = tmp_path / 'not-masks'
    invalid.mkdir()
    tifffile.imwrite(invalid / 'ordinary.tif', np.ones((8, 8), dtype=np.uint16))
    original = (invalid / 'ordinary.tif').read_bytes()
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *_: str(invalid))
    assert browser._overview_choose_object_masks(edit, page) is None
    assert edit.text() == saved and model._read_value('overview_objects_path') == saved
    assert (invalid / 'ordinary.tif').read_bytes() == original
    assert str(invalid) in view.status.text()


def test_opening_overview_copies_only_explicit_source_and_percentiles_from_its_owner(qtbot, tmp_path):
    """A real module supplies Overview context without rewriting its own saved settings."""
    from spacr.qt.screens.app_screen import AppScreen

    source = _plate(tmp_path / 'owner-source', {'A01': 1})
    owner = AppScreen('mask')

    def close(widget):
        tab = getattr(widget, '_overview_tab', None)
        if tab is not None:
            tab._overview_page.retire()
            tab.close()
        widget.close()

    qtbot.addWidget(owner, before_close_func=close)
    owner.resize(1300, 900)
    owner.show()
    owner._settings_model.set_value_for_key('src', str(source))
    original = owner._settings_model._read_value('src')
    wizard = browser._open_overview_module(owner, {
        'src': str(source), 'normalization_percentiles': (2.5, 97.5)})
    tab = owner._overview_tab
    model = tab._settings_model
    assert wizard is tab._overview_page.wizard and wizard.source.text() == str(source)
    assert model._read_value('src') == str(source)
    assert model._read_value('overview_percentile_low') == 2.5
    assert model._read_value('overview_percentile_high') == 97.5
    assert browser._open_overview_module(owner) is wizard
    assert model._read_value('src') == str(source)
    assert model._read_value('overview_percentile_low') == 2.5
    assert model._read_value('overview_percentile_high') == 97.5
    assert owner._settings_model._read_value('src') == original


def test_prerun_placeholder_and_original_controls_leave_the_workspace_unstarted(workspace):
    screen, folders = workspace
    page, model = screen._overview_page, screen._settings_model
    before = model.collect()
    sources = {path: path.read_bytes() for root in folders for path in root.glob('*.tif')}
    page.psf_controls.operation.setCurrentIndex(page.psf_controls.operation.findData('convolve'))
    assert page._full_resolution_fields()['psf_operation'] == 'convolve'
    page.export_pdfs()
    page.export_measurements()
    assert page.browser is None and page.export_menu() is None
    wizard = page.run(dict(before, src='path'))
    assert wizard.source.text() == '' and page.browser is None
    assert model.collect() == before
    assert all(path.read_bytes() == raw for path, raw in sources.items())


def test_canceling_object_masks_before_run_keeps_the_committed_path(workspace, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QLineEdit

    screen, _ = workspace
    page, model = screen._overview_page, screen._settings_model
    row = model._widgets['overview_objects_path']
    edit = row if isinstance(row, QLineEdit) else row.findChildren(QLineEdit)[0]
    before, text = model.collect(), edit.text()
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '')
    assert browser._overview_choose_object_masks(edit, page) is None
    assert page.browser is None and model.collect() == before and edit.text() == text


@pytest.mark.parametrize('mounted', [False, True])
def test_valid_object_masks_commit_real_header_identity_before_or_after_next(
        workspace, qtbot, tmp_path, monkeypatch, mounted):
    from PySide6.QtWidgets import QFileDialog, QLineEdit

    screen, folders = workspace
    page, model = screen._overview_page, screen._settings_model
    view = _workspace_browser(screen, qtbot) if mounted else None
    fields = tuple(field.identity for field in view._fields) if view is not None else ()
    source = model._read_value('src')
    root = tmp_path / 'segmented'
    stack = root / 'masks' / 'cell_mask_stack'
    stack.mkdir(parents=True)
    labels = np.zeros((16, 20), dtype=np.uint32)
    labels[2:7, 3:9] = 70001
    mask = stack / 'p1-A01-f1.tif'
    tifffile.imwrite(mask, labels)
    originals = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    originals[mask] = mask.read_bytes()
    row = model._widgets['overview_objects_path']
    edit = row if isinstance(row, QLineEdit) else row.findChildren(QLineEdit)[0]
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(root))
    found = browser._overview_choose_object_masks(edit, page)
    assert found['count'] == 1 and found['roles'] == {'cell': str(stack)}
    assert found['files'] == {'cell': {'p1-A01-f1': str(mask)}}
    assert model._read_value('overview_objects_path') == edit.text() == str(root)
    assert model._read_value('src') == source
    if view is None:
        assert page.browser is None
    else:
        assert tuple(field.identity for field in view._fields) == fields
        assert tuple(view._channel_buttons) == (1, 2)
        assert '1 files' in view.status.text() and 'cell' in view.status.text()
    assert all(path.read_bytes() == raw for path, raw in originals.items())


def test_page_contact_sheet_export_preserves_selected_physical_fields_and_originals(workspace, qtbot):
    import json
    from pathlib import Path
    from pypdf import PdfReader
    import spacr.plot

    screen, folders = workspace
    view = _workspace_browser(screen, qtbot)
    page, model = screen._overview_page, screen._settings_model
    source = model._read_value('src')
    originals = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    plate_index = view.plates.findData(folders[0].name)
    assert plate_index >= 0
    view.plates.setCurrentIndex(plate_index)
    model.set_value_for_key('overview_export_scope', 'plate')
    model.set_value_for_key('overview_measurements_csv', False)
    expected = tuple(field for field in view._fields if field.plate == folders[0].name)
    assert {field.well for field in expected} == {'A01', 'A02'}
    menu = page.export_menu()
    try:
        assert menu.parentWidget() is view and len(menu.actions()) == 3
    finally:
        menu.close()
    page.export_pdfs()
    qtbot.waitUntil(lambda: not view._export_jobs.is_busy(), timeout=30000)
    selections = list((Path(view._metadata_source) / 'results').glob('overview_*/overview_selection.json'))
    assert len(selections) == 1
    record = json.loads(selections[0].read_text(encoding='utf-8'))
    assert record['channels'] == [1, 2]
    assert record['requested_field_count'] == record['written_field_count'] == 2
    assert not record['cancelled'] and record['errors'] == []
    assert {(row['plate'], row['well'], row['field']) for row in record['fields']} == {
        (folders[0].name, 'A01', '1'), (folders[0].name, 'A02', '1')}
    assert {path for row in record['fields'] for _channel, path in row['files']} == {
        str(path) for path in folders[0].glob('*.tif')}
    pdfs = list(selections[0].parent.glob('overview_*.pdf'))
    assert len(pdfs) == 1
    document = PdfReader(pdfs[0])
    assert len(document.pages) == 1
    text = document.pages[0].extract_text()
    assert 'A01' in text and 'A02' in text
    assert model._read_value('src') == source
    assert all(path.read_bytes() == raw for path, raw in originals.items())


def test_page_measurement_export_refuses_an_empty_explicit_selection_without_a_database(workspace, qtbot):
    screen, folders = workspace
    view = _workspace_browser(screen, qtbot)
    model, page = screen._settings_model, screen._overview_page
    before = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    model.set_value_for_key('overview_export_scope', 'selection')
    assert view._export_selection() == ()
    page.export_measurements()
    assert 'Select fields' in view.status.text()
    assert not view._measurement_jobs.is_busy() and view._measurement_jobs.active_jobs() == 0
    assert not [path for folder in folders for path in folder.rglob('*.db')]
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_invalid_global_channel_text_during_a_channel_draft_keeps_the_last_valid_target(workspace, qtbot):
    screen, folders = workspace
    view = _workspace_browser(screen, qtbot)
    model, page = screen._settings_model, screen._overview_page
    before = {path: path.read_bytes() for folder in folders for path in folder.glob('*.tif')}
    model.set_value_for_key('overview_normalize_channels', '2')
    assert page._bridge.apply() and view._normalize_channels == (2,)
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    prior = _pixels(view, folders[0].name, 'A01', (1, 2)).copy()
    view.norm_channel.setCurrentIndex(view.norm_channel.findData(1))
    assert view._editing_channel == 1
    model.set_value_for_key('overview_normalize_channels', '1,9')
    assert view._normalize_channels == (2,)
    assert model._read_value('overview_normalize_channels') == '1,9'
    assert page._check_channel_list() and '9' in view.status.text()
    np.testing.assert_array_equal(_pixels(view, folders[0].name, 'A01', (1, 2)), prior)
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('level', ['essentials', 'all'])
def test_declared_workspace_headings_survive_lazy_disclosure_and_collapse(
        qtbot, private_ai_profile, monkeypatch, level):
    from spacr.qt import screens, settings_search
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.settings_model import categories_for_app, get_categories

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    preferences.set_reduce_motion(True)
    preferences._set_spacr_ai_enabled(False)
    monkeypatch.setattr(settings_search, '_settings', preferences._settings)
    settings_search.remember_disclosure('overview', level)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    previous_policy = screens._categories_wait_to_be_opened
    screen = window._build_screen('overview')
    qtbot.addWidget(screen, before_close_func=lambda widget: widget._overview_page.retire())
    try:
        assert screens._categories_wait_to_be_opened == previous_policy
        assert screen._categories_wait and screen._settings_search._level == level
        assert screen._overview_page.browser is None
        model = screen._settings_model
        widgets = model._widgets
        export_keys = categories_for_app('overview', get_categories())['Export']
        assert set(export_keys) <= set(widgets)
        waiting = next(section for section in screen._settings_sections
                       if section.property('settingsCategorySource') == 'Export')
        assert screen._heading_is_waiting(waiting) and not waiting.is_expanded()
        assert set(export_keys) & set(widgets.keys_to_come())
        section = browser._overview_section(screen, 'Export')
        assert section is waiting and not screen._heading_is_waiting(section)
        controls = {key: widgets[key] for key in export_keys}
        assert model.set_value_for_key('overview_pdf_dpi', 600)
        assert model._read_value('overview_pdf_dpi') == 600
        for title in ('Image', 'Channels', 'Enhancement', 'Advanced normalization'):
            heading = browser._overview_section(screen, title)
            assert heading is not None
            heading.set_expanded(False)
            assert browser._overview_section(screen, title) is heading
            heading.set_expanded(True)
            assert browser._overview_section(screen, title) is heading
        section.set_expanded(True)
        section.set_expanded(False)
        assert browser._overview_section(screen, 'Export') is section
        assert all(widgets[key] is control for key, control in controls.items())
        assert model._read_value('overview_pdf_dpi') == 600
        declared = [key for key, _, _ in section._spacr_declared_rows]
        assert set(declared) == set(export_keys) and len(declared) == len(export_keys)
        assert screen._overview_page.browser is None
    finally:
        screen._overview_page.retire()
        screen.close()
        window.close()
