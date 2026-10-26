"""Round-5 Overview, M6: every display operation can be set per channel.

Item 725, maintainer feedback 2026-10-25 round 5, point 5: "for any
normalization or other operations to the image, the user must be able to
treat each channel individually." The percentile stretch, every enhancement
step and the CMY/CMYK ink intensity follow the advanced normalization's
Channel to edit; a channel without its own value uses the global one.
"""
import json

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")


def _raw(channel, index):
    rng = np.random.default_rng(10 * channel + index)
    yy, xx = np.mgrid[0:48, 0:64]
    signal = np.exp(-((xx - 20 - 8 * channel) ** 2 + (yy - 24) ** 2) / 120.) * 3000
    return (signal + rng.integers(0, 400, size=(48, 64)) + 100 * channel).astype(np.uint16)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", _raw(channel, index))
    return folder


def _field(source):
    fields = browser._overview_index(str(source), PATTERN, source_scope="folder")
    return next(field for field in fields if field.well == "A01" and field.field == "1")


def _views(field, **options):
    return dict(browser._overview_pixels(field, (1, 2), (1., 99.), edge=None, channel_views=True, **options))


def _chain(**values):
    return browser._overview_chain(values)


OPERATIONS = (
    ("percentiles", dict(channel_percentiles={1: (20., 80.)})),
    ("enhancement", dict(channel_enhancement={1: _chain(enhance_gamma=.4)})),
    ("enhancement over a global chain", dict(enhancement=_chain(enhance_sharpen=True),
                                             channel_enhancement={1: _chain(enhance_gamma=2.5)})),
    ("no enhancement on one channel", dict(enhancement=_chain(enhance_gamma=.5),
                                           channel_enhancement={1: None})),
)


@pytest.mark.parametrize("name, options", OPERATIONS, ids=[name for name, _ in OPERATIONS])
def test_changing_channel_one_leaves_channel_two_unchanged(source, name, options):
    field = _field(source)
    base = {key: value for key, value in options.items() if key == "enhancement"}
    plain = _views(field, **base)
    changed = _views(field, **options)
    assert not np.array_equal(plain[1], changed[1]), name
    assert np.array_equal(plain[2], changed[2]), name


def test_a_channel_without_an_override_matches_the_global_value(source):
    field = _field(source)
    global_only = _views(field)
    assert np.array_equal(_views(field, channel_percentiles={1: (5., 95.)})[2], global_only[2])
    assert np.array_equal(_views(field, channel_percentiles={})[1], global_only[1])
    chain = _chain(enhance_gamma=.6)
    enhanced = _views(field, enhancement=chain)
    assert np.array_equal(_views(field, enhancement=chain, channel_enhancement={1: None})[2], enhanced[2])
    own = _views(field, channel_percentiles={1: (1., 99.)})
    assert np.array_equal(own[1], global_only[1])


def test_ink_intensity_scales_only_its_own_channel_in_cmy(source):
    field = _field(source)
    views = _views(field)
    inks = browser._overview_inks((1, 2), None, "cmy")
    planes = [views[1][..., 0], views[2][..., 0]]

    def printed(strengths):
        absorbed = sum(plane[..., None].astype(np.float32) * strength * (1 - np.asarray(ink, np.float32))
                       for plane, ink, strength in zip(planes, inks, strengths))
        return np.clip(255. - absorbed, 0, 255).astype(np.uint8)

    plain = browser._overview_pixels(field, (1, 2), (1., 99.), edge=None, color_mode="cmy")
    assert np.array_equal(plain, printed((1., 1.)))
    half = browser._overview_pixels(field, (1, 2), (1., 99.), edge=None, color_mode="cmy",
                                    ink_intensity={1: .5})
    assert np.array_equal(half, printed((.5, 1.)))
    assert np.array_equal(browser._overview_pixels(field, (1, 2), (1., 99.), edge=None, color_mode="cmy",
                                                   ink_intensity={2: 1.}), plain)


def test_export_records_each_channels_own_display_values():
    record = browser._overview_channel_display_record((1, 2, 3), (1., 99.), {2: (5., 95.)}, {3: .5})
    assert record == {"1": {"percentiles": [1., 99.], "ink_intensity": 1.},
                      "2": {"percentiles": [5., 95.], "ink_intensity": 1.},
                      "3": {"percentiles": [1., 99.], "ink_intensity": .5}}
    assert browser._overview_channel_display_record((1,), None, {1: (5., 95.)})["1"]["percentiles"] is None
    lines = browser._overview_channel_display_lines((1, 2), {2: (5., 95.)}, {1: .5}, "cmy")
    assert "1: ink intensity 0.5" in " ".join(lines) and "2: percentiles 5-95" in " ".join(lines)
    assert browser._overview_channel_display_lines((1, 2)) == ()


def test_the_exported_selection_file_holds_the_channel_values(source, tmp_path):
    fields = browser._overview_index(str(source), PATTERN, source_scope="folder")[:1]
    folder = tmp_path / "export"
    browser._overview_export(fields, str(folder), (1, 2), (1., 99.), quality_edge=64, dpi=72,
                             color_mode="cmy", channel_percentiles={2: (10., 90.)}, ink_intensity={1: .25})
    saved = json.loads((folder / "overview_selection.json").read_text())
    assert saved["channel_display"]["2"]["percentiles"] == [10., 90.]
    assert saved["channel_display"]["1"]["ink_intensity"] == .25


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs, view._readout_jobs,
            view._levels_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 2), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy() and dialog.norm_channel.count() == 3, timeout=15000)
    return dialog


def _pick(view, channel):
    view.norm_channel.setCurrentIndex(view.norm_channel.findData(channel))


def _rendered(view, field):
    channels, limits = view._choices()
    options = dict(view._display_kwargs(view=True))
    return dict(browser._overview_pixels(field, channels, limits, edge=None, channel_views=True, **options))


def test_a_picked_channel_gets_its_own_percentiles_marker_and_reset(view):
    field = view._fields[0]
    before = _rendered(view, field)
    _pick(view, 1)
    view.low.setValue(25.)
    assert view._choices()[1] == (1., 99.9)
    assert np.array_equal(_rendered(view, field)[1], before[1])
    view._apply_channel_normalization()
    assert view._channel_overrides[1]["percentile_low"] == 25.
    assert "percentile_high" not in view._channel_overrides[1]
    assert view._display_kwargs(view=True)["channel_percentiles"] == {1: (25., 99.9)}
    assert "channel_percentiles" in view._display_kwargs()
    after = _rendered(view, field)
    assert not np.array_equal(after[1], before[1])
    assert np.array_equal(after[2], before[2])
    assert view.norm_channel.itemText(view.norm_channel.findData(1)) == "1 (own settings)"
    assert view.norm_channel.itemText(view.norm_channel.findData(2)) == "2"
    _pick(view, None)
    assert view.low.value() == 1.
    _pick(view, 1)
    assert view.low.value() == 25.
    assert view._reset_channel_button.isEnabled()
    view._reset_channel_button.click()
    assert 1 not in view._channel_overrides
    assert view.low.value() == 1.
    assert view.norm_channel.itemText(view.norm_channel.findData(1)) == "1"
    assert np.array_equal(_rendered(view, field)[1], before[1])


def test_ink_and_enhancement_overrides_follow_the_global_values_they_do_not_set(view):
    view._set_channel_display_globals(.8, {"enhance_gamma": .7})
    view._color_mode = "cmy"
    assert view._display_kwargs(view=True)["ink_intensity"] == {1: .8, 2: .8}
    assert "channel_enhancement" not in view._display_kwargs(view=True)
    _pick(view, 2)
    view._channel_display_draft({"ink": .3, "enhance_gamma": .7, "enhance_sharpen": True})
    view._apply_channel_normalization()
    entry = view._channel_overrides[2]
    assert entry["ink"] == .3 and entry["enhance_sharpen"] is True and "enhance_gamma" not in entry
    options = view._display_kwargs(view=True)
    assert options["ink_intensity"] == {1: .8, 2: .3}
    assert options["channel_enhancement"][2] == browser._overview_chain(
        {"enhance_gamma": .7, "enhance_sharpen": True})
    assert "channel_enhancement" not in view._display_kwargs()
    view._set_channel_display_globals(.9, {"enhance_gamma": .5})
    assert view._display_kwargs(view=True)["ink_intensity"] == {1: .9, 2: .3}
    assert view._display_kwargs(view=True)["channel_enhancement"][2].gamma == .5
    assert view._normalization_options()["per_channel"]["2"].keys() == set(view._editor_options())


def _close_mask(mask, qtbot):
    tab = getattr(mask, "_overview_tab", None)
    page = getattr(tab, "_overview_page", None)
    if page is not None and page.browser is not None:
        overview = page.browser
        overview._cancel()
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(overview)), timeout=15000)
        page.retire()
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0, timeout=15000)
    if tab is not None:
        tab.close()
    mask.close()


@pytest.fixture
def module(qtbot, source):
    from spacr.qt.screens.app_screen import AppScreen

    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    mask = AppScreen("mask")
    qtbot.addWidget(mask, before_close_func=lambda widget: _close_mask(widget, qtbot))
    mask.resize(1500, 1000)
    mask.show()
    mask._settings_model.set_value_for_key("src", str(source))
    row = mask._settings_model._widgets.get("src")
    next(action for action in row.actions() if action.objectName() == "GenerateMasksFieldOverview").trigger()
    tab = mask._overview_tab
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 2 * len(WELLS)
                    and page.browser._normalization_ready and page.browser.norm_channel.count() >= 3,
                    timeout=15000)
    return tab, page.browser


def test_the_enhancement_and_ink_rows_edit_the_picked_channel_and_show_the_global_values_again(module):
    tab, view = module
    model = tab._settings_model
    assert model._widgets.get("overview_ink_intensity") is not None
    model.set_value_for_key("overview_ink_intensity", .6)
    assert view._ink_intensity == .6
    global_gamma = model._read_value("enhance_gamma")
    _pick(view, 1)
    model.set_value_for_key("enhance_gamma", .45)
    model.set_value_for_key("overview_ink_intensity", .2)
    assert view._enhancement == browser._overview_chain(dict(view._enhance_values))
    assert view._enhance_values["enhance_gamma"] == global_gamma and view._ink_intensity == .6
    model.set_value_for_key("overview_color_mode", "cmy")
    assert view._color_mode == "cmy"
    view._apply_channel_normalization()
    entry = view._channel_overrides[1]
    assert entry["enhance_gamma"] == .45 and entry["ink"] == .2
    options = view._display_kwargs(view=True)
    assert options["channel_enhancement"][1].gamma == .45
    assert options["ink_intensity"][1] == .2 and options["ink_intensity"][2] == .6
    _pick(view, None)
    assert model._read_value("enhance_gamma") == global_gamma
    assert model._read_value("overview_ink_intensity") == .6
    _pick(view, 1)
    assert model._read_value("enhance_gamma") == .45
    assert view.norm_channel.currentText() == "1 (own settings)"
