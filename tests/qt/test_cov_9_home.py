"""Home when the things it reports on are unavailable.

Home is a dashboard over other people's state -- the run journal, the plate
queue, the GPU, the disk, the theme preferences -- and it is the first screen
the application shows. Every branch here is one of those readings failing.
None of them may keep the window from opening, and none may print a number it
did not measure: "n/a" and an empty list are answers, a traceback on launch is
not.
"""
from __future__ import annotations

import builtins
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.widgets import home as home_mod
from spacr.qt.widgets.home import (
    HomePage, NewsPanel, QueuedPanel, RecentRunsPanel, RunningBanner,
    SystemPanel, TotalsPanel, _find_logo_pixmap, _fmt_elapsed, active_palette)

pytestmark = pytest.mark.qt

APPS = [("mask", "Mask", "Generate masks", "Core")]


def _block(name_fragment, monkeypatch):
    """Make ``import <name_fragment>...`` fail for the duration of a test."""
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name_fragment in name:
            raise ImportError(f"{name} is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)


class _Gate:
    """A pause gate that is never paused."""

    @staticmethod
    def is_paused():
        return False


class _Handle:
    """A run handle with the surface Home reads."""

    def __init__(self, app_key="mask", running=True):
        self.app_key = app_key
        self.thread = None
        self.worker = None
        self.cancels = []
        self.supports_pause = False
        self.gate = _Gate()
        self.progress = None
        self.last_line = ""
        self._running = running

    def is_running(self):
        return self._running

    def request_cancel(self, why):
        self.cancels.append(why)

    def fraction(self):
        return None

    def elapsed(self):
        return 12.0


# ---------------------------------------------------------------------------
# Reading the theme and the assets
# ---------------------------------------------------------------------------

def test_a_theme_that_cannot_be_read_still_yields_a_palette(qapp, monkeypatch):
    """Home is the first screen shown, so it must open before preferences do.

    An unreadable preferences file would otherwise take the application down
    on launch. Dark is the fallback because a light page under an unset theme
    is the brighter mistake.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(preferences, "resolve_effective_theme", _explode)

    palette = active_palette()

    assert palette["bg"] == home_mod.palette_for("dark")["bg"]


def test_a_missing_logo_leaves_the_hero_without_one(qapp, monkeypatch):
    """The wordmark is drawn from a packaged file that may not be installed.

    A source checkout without the resources, or a trimmed wheel, must give a
    hero with no mark rather than a null pixmap scaled into the layout.
    """
    real_isfile = os.path.isfile

    def _no_logos(path):
        return (False if str(path).endswith(("logo_spacr.png",
                                             "logo_spacr_v1.png"))
                else real_isfile(path))

    monkeypatch.setattr(os.path, "isfile", _no_logos)

    assert _find_logo_pixmap() is None


def test_an_hour_long_run_is_reported_in_hours_and_minutes():
    """Past an hour, seconds are noise and the hour is the number that matters.

    A run reported as "5400s" is one a reader has to do arithmetic on before
    they can tell whether it is worth waiting for.
    """
    assert _fmt_elapsed(45) == "45s"
    assert _fmt_elapsed(125) == "2m 05s"
    assert _fmt_elapsed(5400) == "1h 30m"


# ---------------------------------------------------------------------------
# The running banner
# ---------------------------------------------------------------------------

@pytest.fixture
def banner(qtbot):
    widget = RunningBanner(lambda key: None, {"mask": "Mask"})
    qtbot.addWidget(widget)
    return widget


def test_a_banner_with_no_job_refreshes_to_nothing(banner):
    """The ticker fires on a hidden banner too, and there is nothing to read.

    Reading the handle regardless would raise on every tick after the job it
    was showing retired, and a stale subtitle would keep describing a run
    that has already finished.
    """
    banner.bind(_Handle())
    banner.refresh()
    assert banner._sub.text(), "a bound job puts something in the subtitle"
    before = banner._sub.text()

    banner.bind(None)
    banner.refresh()

    assert banner.isHidden()
    assert banner._sub.text() == before, "nothing was re-read"


def test_quitting_with_no_job_asks_nothing(banner, monkeypatch):
    """The button can outlive the job it was bound to.

    A click arriving after the run finished must not prompt about stopping a
    job that has already stopped -- a dialog about nothing is worse than no
    dialog.
    """
    import spacr.qt.shutdown as shutdown

    asked = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: asked.append(what))
    banner.bind(None)

    banner._on_quit()

    assert asked == []


def test_a_cancelled_quit_prompt_leaves_the_run_alone(banner, monkeypatch):
    """Choosing Cancel in the "how should this stop" prompt stops nothing.

    This is the one control on Home that can end somebody's analysis, so a
    dismissed dialog has to be exactly a no-op.
    """
    import spacr.qt.shutdown as shutdown

    handle = _Handle()
    banner.bind(handle)
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: shutdown.CANCEL)

    banner._on_quit()

    assert handle.cancels == []


def test_choosing_force_stops_the_job_outright(banner, monkeypatch):
    """Force still asks the worker to stop before the thread is taken away.

    A worker that IS still checking gets the chance to stop on its own terms
    in the moment before it is parked, which is the difference between a
    clean exit and a half-written file.
    """
    import spacr.qt.shutdown as shutdown

    handle = _Handle()
    banner.bind(handle)
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: shutdown.FORCE)

    banner._on_quit()

    assert handle.cancels == ["force quit from the Home screen"]


def test_a_graceful_quit_asks_the_run_to_stop_and_keeps_watching(
        banner, monkeypatch):
    """The watcher lives on the banner, not on the handle.

    The handle is retired the moment the job stops, and a timer parented to a
    dead object is a crash rather than a missed prompt.
    """
    import spacr.qt.shutdown as shutdown

    handle = _Handle()
    banner.bind(handle)
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: shutdown.GRACEFUL)

    banner._on_quit()

    assert handle.cancels == ["quit from the Home screen"]
    assert banner._quit_watcher.parent() is banner
    banner._quit_watcher.stop()


def test_force_stopping_a_job_with_no_thread_is_not_an_error(banner):
    """A handle whose thread has already gone has nothing left to park.

    The job finished between the prompt and the click, which is the good
    outcome; treating it as an error would report a failure to the user for
    getting what they asked for.
    """
    handle = _Handle()
    handle.thread = None

    RunningBanner._terminate(handle)

    assert handle.cancels == ["force quit from the Home screen"]


def test_force_stopping_survives_a_handle_that_refuses_to_cancel(banner,
                                                                 monkeypatch):
    """``request_cancel`` is a courtesy call and may fail on a dead handle.

    Force quit exists precisely for a job that is not responding, so a
    failure here must not stop the thread being parked -- which is the half
    that actually ends the run.
    """
    import spacr.qt.bridge as bridge

    class _Refusing(_Handle):
        def request_cancel(self, why):
            raise RuntimeError("the handle is already retired")

    drained = []
    monkeypatch.setattr(bridge, "drain_thread",
                        lambda thread, worker=None, timeout_ms=3000:
                            drained.append(thread))
    handle = _Refusing()
    handle.thread = object()

    RunningBanner._terminate(handle)

    assert drained == [handle.thread]


def test_force_stopping_a_thread_that_is_already_gone_is_not_an_error(
        banner, monkeypatch):
    """The drain call raises when the QThread's C++ half has been deleted.

    That is the job having finished, not a failure, so nothing is reported.
    """
    import spacr.qt.bridge as bridge

    def _gone(thread, worker=None, timeout_ms=3000):
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(bridge, "drain_thread", _gone)
    handle = _Handle()
    handle.thread = object()

    RunningBanner._terminate(handle)

    assert handle.cancels == ["force quit from the Home screen"]


# ---------------------------------------------------------------------------
# The aside panels, when what they report on cannot be read
# ---------------------------------------------------------------------------

def test_an_unreadable_plate_queue_shows_an_empty_queue(qtbot, monkeypatch):
    """A queue file that will not load is an empty queue on screen.

    Home must open without it; the queue is one panel of six and none of them
    is allowed to be the reason the window does not appear.
    """
    import spacr.qt.plate_queue as plate_queue

    def _explode(*args, **kwargs):
        raise OSError("the queue file is corrupt")

    monkeypatch.setattr(plate_queue, "PlateQueue", _explode)
    panel = QueuedPanel()
    qtbot.addWidget(panel)

    assert panel.queue_items() == []


def test_an_unreadable_journal_shows_no_recent_runs(qtbot, monkeypatch):
    """The run journal is thousands of manifests and any of them can be bad.

    An empty list is honest -- nothing could be read -- and it keeps the walk
    from taking down the page it was being drawn on.
    """
    import spacr.run_journal as run_journal

    def _explode(limit=4):
        raise OSError("the journal is unreadable")

    monkeypatch.setattr(run_journal, "recent_runs", _explode)
    panel = RecentRunsPanel()
    qtbot.addWidget(panel)

    assert panel.read() == []


def test_unreadable_journal_totals_are_reported_as_zeroes(qtbot, monkeypatch):
    """Every count is present and zero rather than the panel being absent.

    A panel with missing keys would raise while drawing; zeroes say "nothing
    could be counted", which is what happened.
    """
    import spacr.run_journal as run_journal

    def _explode():
        raise OSError("the journal is unreadable")

    monkeypatch.setattr(run_journal, "journal_totals", _explode)
    panel = TotalsPanel()
    qtbot.addWidget(panel)

    assert panel.read() == {"total_runs": 0, "mask_runs": 0,
                            "measure_runs": 0, "models_recorded": 0}


def test_a_machine_without_nvml_reports_no_gpu(monkeypatch):
    """One lightweight probe, and when it is unavailable the answer is n/a.

    Reporting "0%" or "0.0 GB" would state a measurement of a device that was
    never found.
    """
    monkeypatch.setattr(home_mod, "_nvml", lambda: None)

    assert SystemPanel.gpu_util() == "n/a"
    assert SystemPanel.gpu_vram() == "n/a"


def test_reading_the_home_gpu_card_does_not_import_torch(monkeypatch):
    """The dashboard must not allocate a model runtime to decorate itself.

    A CPU-only runner and a laptop without NVML both take this path. Importing
    Torch there cost hundreds of megabytes before the user chose an operation
    that needs it.
    """
    monkeypatch.setattr(home_mod, "_nvml", lambda: None)
    _block("torch", monkeypatch)

    assert SystemPanel.gpu_util() == "n/a"
    assert SystemPanel.gpu_vram() == "n/a"


def test_a_disk_that_cannot_be_measured_reports_no_percentage(monkeypatch):
    """A home directory on a stale mount cannot be sized.

    "n/a" is the honest cell; a zero would read as an empty disk.
    """
    import shutil

    def _explode(path):
        raise OSError("the mount went away")

    monkeypatch.setattr(shutil, "disk_usage", _explode)

    assert SystemPanel.disk_used() == "n/a"


def test_replacing_the_news_content_releases_the_old_widget(qtbot):
    """The slot holds one widget, and the previous one has to go.

    Left parented it would keep painting under the new content and hold
    whatever it references for the life of the page.
    """
    panel = NewsPanel()
    qtbot.addWidget(panel)
    first = QLabel("release 1")
    second = QLabel("release 2")

    panel.set_content(first)
    panel.set_content(second)

    assert panel.content is second
    assert first.parent() is None


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------

@pytest.fixture
def page(qtbot):
    widget = HomePage(APPS, lambda key: None)
    qtbot.addWidget(widget)
    return widget


def test_an_image_theme_paints_no_flat_page_fill(page, monkeypatch):
    """A wallpaper theme has a picture behind Home; a slab would hide it.

    The fill exists to replace the stylesheet background on the flat themes.
    Under an image theme there is nothing to replace.
    """
    import spacr.qt.preferences as preferences
    from spacr.qt.theme import IMAGE_THEMES

    page._ambient = None
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: sorted(IMAGE_THEMES)[0])

    assert page.page_fill() is None


def test_an_unreadable_theme_paints_no_page_fill(page, monkeypatch):
    """Without a theme there is no colour to fill with.

    Guessing one would paint a slab of the wrong colour over the whole page,
    which is the exact failure the fill exists to undo.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise OSError("preferences are unreadable")

    page._ambient = None
    monkeypatch.setattr(preferences, "resolve_effective_theme", _explode)

    assert page.page_fill() is None


def test_an_unreadable_theme_leaves_the_ambient_without_a_backdrop(monkeypatch):
    """Only the image themes have a wallpaper, and the lookup can fail.

    None means "paint over the flat window colour", which is what every other
    theme does anyway.
    """
    import spacr.qt.preferences as preferences

    def _explode(theme):
        raise OSError("the theme is unreadable")

    monkeypatch.setattr(preferences, "theme_background_path", _explode)

    assert HomePage._ambient_backdrop() is None


def test_nothing_is_discarded_when_the_ambient_module_is_absent(page,
                                                               monkeypatch):
    """If the import is what failed, nothing was ever constructed.

    Walking the children looking for a class that could not be imported is
    not possible, and there is nothing there to find.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    leftover = AmbientWidget(page)
    _block("ambient", monkeypatch)

    page._discard_ambient(leftover)

    assert leftover.parent() is page, "nothing was swept"


def test_an_aborted_ambient_install_leaves_no_live_child(page):
    """A widget parented before its wiring finished is still a live child.

    ``install_ambient`` parents first, so an installer that raises part way
    hands nothing back to unparent -- and an invisible leftover keeps its
    timer running behind every screen the user opens afterwards.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    leftover = AmbientWidget(page)

    page._discard_ambient(leftover)

    assert not [c for c in page.children() if isinstance(c, AmbientWidget)]


def test_a_leftover_that_refuses_to_be_freed_does_not_stop_the_sweep(page):
    """Both halves of the release are best-effort, and both can fail.

    A child whose C++ half is already gone raises on every call; the sweep
    has to reach the rest of the children regardless.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    class _Stubborn(AmbientWidget):
        def set_animating(self, on):
            raise RuntimeError("already deleted")

        def setParent(self, parent):
            raise RuntimeError("already deleted")

    stubborn = _Stubborn(page)
    ordinary = AmbientWidget(page)

    page._discard_ambient(stubborn)

    assert ordinary.parent() is None, "the sweep reached the rest"


def test_the_tabs_are_built_even_when_the_theme_cannot_be_read(qtbot,
                                                              monkeypatch):
    """Whether the theme is glass changes the tab styling, nothing more.

    Unreadable preferences must give the non-glass styling rather than a page
    with no tabs on it at all.
    """
    import spacr.qt.preferences as preferences

    real_resolve = preferences.resolve_effective_theme
    calls = {"n": 0}

    def _fails_once():
        calls["n"] += 1
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(preferences, "resolve_effective_theme", _fails_once)
    widget = HomePage(APPS, lambda key: None)
    qtbot.addWidget(widget)

    assert calls["n"] > 0
    assert widget._tabs.count() > 0


def test_unreadable_preferences_leave_the_panes_opaque(monkeypatch):
    """Opaque is what Home looked like before the transparency preference.

    A guessed alpha could put text on a background it cannot be read
    against, which is the one thing the preference's own floor prevents.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(preferences, "effective_pane_alpha", _explode)

    assert HomePage._pane_alpha() == 1.0


def test_a_build_with_no_importable_version_names_no_release(monkeypatch):
    """The News panel heads itself with a version only when there is one.

    "spaCR dev" says less than "News", and a package that cannot even be
    imported has no release to name.
    """
    with monkeypatch.context() as blocked_imports:
        _block("spacr", blocked_imports)
        assert HomePage._version() == ""


def test_closing_home_stops_its_ticker_even_when_the_registry_is_gone(page):
    """Releasing the registry connection may fail, and the ticker still stops.

    The run registry is process-wide and outlives no page in particular; a
    disconnect that raises must not leave a one-second timer running against
    a widget that is being torn down.
    """
    class _DeadSignal:
        @staticmethod
        def disconnect(_slot):
            raise RuntimeError("Internal C++ object already deleted.")

    class _DeadRegistry:
        changed = _DeadSignal()

    original = page._registry
    page._registry = _DeadRegistry()

    page.closeEvent(QCloseEvent())
    original.changed.emit()

    assert not page._ticker.isActive()


# ---------------------------------------------------------------------------
# Two pieces of Home that cannot run at all
# ---------------------------------------------------------------------------

def test_closing_home_shuts_down_its_journal_walk(qtbot, monkeypatch):
    """A journal walk must not outlive the page that asked for it.

    The walk reads thousands of manifests on a worker; a page closed while
    one is in flight leaves a thread running against a widget that is being
    torn down. ``closeEvent`` is meant to shut the runner down first.
    """
    widget = HomePage(APPS, lambda key: None)
    qtbot.addWidget(widget)
    stopped = []
    monkeypatch.setattr(widget._journal_jobs, "shutdown",
                        lambda: stopped.append(True))

    widget.closeEvent(QCloseEvent())

    assert stopped == [True]


def test_the_hero_labels_are_reachable_by_the_name_that_clears_them(page):
    """The masthead's type has to go transparent with the rest of the page.

    ``_clear_page_surfaces`` looks the hero up by object name to reach its
    labels; without that name the lookup finds nothing and a QLabel with no
    rule of its own takes the blanket window fill -- a black band across the
    masthead over the ambient backdrop.
    """
    hero = page.findChild(QWidget, "Hero")

    assert hero is not None
    assert hero.findChildren(QLabel)


def test_release_notes_render_markdown_links_headings_and_code():
    """The 1.5.1.5 notes showed ``### Upgrading`` and a link that ate its ``)``."""
    from spacr.qt.widgets.home import NewsPanel

    body = ("### Upgrading\n"
            "Install the `.pkg` from the [releases page](https://github.com/EinarOlafsson/spacr/releases).\n"
            "See https://example.org/notes, then <script>x</script>.")
    html = NewsPanel.render_body(body, "#123456")
    assert "<b>Upgrading</b>" in html and "###" not in html
    assert ('<a href="https://github.com/EinarOlafsson/spacr/releases" style="color: #123456;'
            ' text-decoration: none;">releases page</a>.') in html
    assert "[releases page]" not in html and "releases)" not in html
    assert "<code>.pkg</code>" in html
    assert '<a href="https://example.org/notes"' in html and "notes</a>," in html
    assert "&lt;script&gt;" in html and "<script>" not in html


def test_dashboard_watermarks_filter_real_manifests_without_deleting_history(
        qtbot, tmp_path, monkeypatch):
    """Clearing a dashboard preserves provenance while filtering dates and module keys."""
    import json
    from PySide6.QtCore import QSettings
    from spacr import run_journal
    from spacr.qt import preferences

    settings = QSettings(str(tmp_path / 'private.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: settings)
    root = tmp_path / 'runs'
    root.mkdir()
    monkeypatch.setattr(run_journal, 'runs_root', lambda: root)
    rows = [('2026-10-27_110000', 'mask', '2026-10-27T11:00:00Z'),
            ('2026-10-27_120100', 'measure', '2026-10-27T12:01:00Z'),
            ('2026-10-27_120200', 'unregistered', '2026-10-27T12:02:00Z'),
            ('2026-10-27_120300', 'mask', '2026-10-27T12:03:00Z')]
    originals = {}
    for folder, key, stamp in rows:
        path = root / folder / 'manifest.json'
        path.parent.mkdir()
        path.write_text(json.dumps(dict(app_key=key, start_utc=stamp, status='complete')))
        originals[path] = path.read_bytes()
    broken = root / '2026-10-27_120400' / 'manifest.json'
    broken.parent.mkdir()
    broken.write_text('{invalid')
    originals[broken] = broken.read_bytes()
    preferences.set_dashboard_watermark('runs', '2026-10-27T12:00:00Z')
    panel = RecentRunsPanel(limit=2, known_keys={'mask': 'Mask', 'measure': 'Measure'}, read_now=False)
    qtbot.addWidget(panel)
    assert [entry['app_key'] for entry in panel.read()] == ['mask', 'measure']
    assert panel.name_for('mask') == 'Mask'
    with qtbot.waitSignal(panel.cleared):
        panel.clear_list()
    assert panel.read() == []
    totals = TotalsPanel._totals_since('2026-10-27T12:00:00Z')
    assert totals == dict(total_runs=3, mask_runs=1, measure_runs=1,
                          classify_runs=0, models_recorded=0)
    assert TotalsPanel._totals_since('2026-10-27T12:00:00') == totals
    assert TotalsPanel._totals_since('') == dict(
        total_runs=4, mask_runs=2, measure_runs=1, classify_runs=0, models_recorded=0)
    assert all(path.read_bytes() == raw for path, raw in originals.items())


@pytest.mark.parametrize('format', ['json', 'csv', 'manifest'])
def test_saved_settings_are_bounded_read_only_proposals(tmp_path, format):
    """JSON, packed CSV and manifest-associated settings retain typed values only."""
    import json

    if format == 'csv':
        path = tmp_path / 'mask_settings.csv'
        path.write_text('setting,value\nsize,12\nenabled,True\nignored\n')
    else:
        path = tmp_path / ('settings.json' if format == 'manifest' else 'mask_settings.json')
        path.write_text(json.dumps({'size': 12, 'enabled': True}))
    if format == 'manifest':
        (tmp_path / 'manifest.json').write_text(json.dumps({'app_key': 'mask'}))
    before = {file: file.read_bytes() for file in tmp_path.iterdir()}
    result = home_mod._easy_saved_settings(path)
    assert result == dict(path=str(path), module='mask', settings={'size': 12, 'enabled': True})
    assert all(file.read_bytes() == raw for file, raw in before.items())


@pytest.mark.parametrize('contents', ['[]', '{}', 'x' * (65536 + 1)])
def test_invalid_or_oversized_saved_settings_never_become_actions(tmp_path, contents):
    """Invalid shapes and oversized data cannot seed a module's settings."""
    path = tmp_path / 'mask_settings.json'
    path.write_text(contents if len(contents) < 1000 else ' ' * (home_mod._EASY_METADATA_LIMIT + 1))
    before = path.read_bytes()
    assert home_mod._easy_saved_settings(path) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize('kind', ['symlink', 'directory', 'offset'])
def test_local_header_reader_refuses_nonregular_or_out_of_range_inputs(tmp_path, kind):
    """The bounded reader neither follows symlinks nor reads beyond a real file."""
    path = tmp_path / 'input.bin'
    path.write_bytes(b'original bytes')
    target = path
    offset = 0
    if kind == 'symlink':
        target = tmp_path / 'link.bin'
        target.symlink_to(path)
    elif kind == 'directory':
        target = tmp_path
    else:
        offset = path.stat().st_size + 1
    with pytest.raises(ValueError):
        home_mod._easy_read_header(target, 8, offset)
    assert path.read_bytes() == b'original bytes'


@pytest.mark.parametrize('reason', ['cancelled', 'pattern', 'samples', 'payload'])
def test_regex_limits_preserve_input_and_never_start_a_validation_child(monkeypatch, reason):
    """Expected cancellation or bounded-input refusal does not launch a process."""
    import copy
    import multiprocessing

    result = {'path': '/local/images', 'settings_overrides': {'mask': {'size': 12}},
              'regex': {'sample_filenames': ['image.tif'], 'example_path': '/local/image.tif'}}
    pattern = '(.*)'
    if reason == 'pattern':
        pattern = 'x' * 4097
    elif reason == 'samples':
        result['regex']['sample_filenames'] = []
    elif reason == 'payload':
        result['regex']['example_path'] = 'x' * 1048577
    before = copy.deepcopy(result)

    def unexpected(*args, **kwargs):
        raise AssertionError('bounded refusal must not start a child')

    monkeypatch.setattr(multiprocessing, 'get_context', unexpected)
    actual = home_mod._easy_validate_regex(result, pattern, cancelled=lambda: reason == 'cancelled')
    assert result == before
    assert actual['regex']['pattern'] == pattern and actual['regex']['required']
    assert 'settings_overrides' not in actual
    assert bool(actual['regex'].get('validation_error')) is (reason != 'cancelled')


@pytest.mark.parametrize('raw,expected', [
    (b'not metadata', {}),
    (b'{"axes":"TZYX","shape":[2,-1,4,4]}', {}),
    (b'<Pixels SizeT="3" SizeZ="4" SizeC="2"/>', {'t': 3, 'z': 4, 'axes': 'OME'}),
])
def test_image_dimension_hints_require_valid_structural_metadata(raw, expected):
    """Malformed metadata never becomes a claim of time or depth acquisition."""
    assert home_mod._easy_axes_metadata(raw) == expected


@pytest.mark.parametrize('content,family', [
    (b'<Workspace/>', 'flowjo'), (b'<GraphPadPrismFile/>', 'prism'),
    (b'<CDXML/>', 'chemdraw'),
])
def test_external_document_headers_offer_original_software_not_analysis(tmp_path, content, family):
    """Misnamed external documents are identified from bounded bytes, never executed."""
    path = tmp_path / 'source.dat'
    path.write_bytes(content)
    assert home_mod._easy_file_kind(path) == 'external_' + family
    assert path.read_bytes() == content
    result = home_mod._easy_inspect_path(str(path))
    assert result['actions'] == []
    assert result['context']['unsupported_documents'] == [dict(kind=family, path=str(path))]


@pytest.mark.parametrize('header', [
    b'II+\x00\x04\x00\x00\x00' + b'\x00' * 8,
    b'II*\x00\x08\x00\x00\x00\x01',
    b'II*\x00\x08\x00\x00\x00\x00\x00',
    b'II*\x00\x08\x00\x00\x00\xff\xff',
    b'II*\x00\x08\x00\x00\x00\x01\x00' + b'\x00' * 5,
], ids=['invalid-bigtiff-offset-width', 'short-count', 'empty-ifd', 'unbounded-ifd', 'short-entry'])
def test_truncated_or_unbounded_tiff_directories_never_offer_image_processing(tmp_path, header):
    """A magic prefix alone is not a validated microscope image."""
    path = tmp_path / 'source.tif'
    path.write_bytes(header)
    assert home_mod._easy_file_kind(path) == 'unsupported'
    assert home_mod._easy_inspect_path(str(path))['actions'] == []
    assert path.read_bytes() == header


def test_corrupt_compressed_sequence_is_not_misidentified_as_reads(tmp_path):
    """A corrupt gzip header has no sequence payload to classify or process."""
    path = tmp_path / 'source.fastq.gz'
    raw = b'\x1f\x8b' + b'not compressed sequence bytes'
    path.write_bytes(raw)
    assert home_mod._easy_file_kind(path) == 'unsupported'
    assert home_mod._easy_inspect_path(str(path))['actions'] == []
    assert path.read_bytes() == raw


def test_native_mask_archive_remains_a_native_input_without_unpacking_arrays(tmp_path):
    """A genuine ZIP-backed NumPy archive is routed as a container, not decoded."""
    import numpy as np

    path = tmp_path / 'masks.npz'
    np.savez(path, masks=np.zeros((2, 3, 4), dtype=np.uint16))
    before = path.read_bytes()
    assert home_mod._easy_file_kind(path) == 'native_container'
    assert path.read_bytes() == before


def test_empty_measurement_table_and_unsupported_csv_header_do_not_seed_values(tmp_path):
    """Single-row measurements and a settings CSV with the wrong heading are not data."""
    table = tmp_path / 'single.csv'
    table.write_text('field,value\n')
    assert home_mod._easy_table_evidence(table) == {}
    settings = tmp_path / 'mask_settings.csv'
    settings.write_text('wrong,heading\nsize,12\n')
    assert home_mod._easy_saved_settings(settings) is None


def test_bounded_run_discovery_keeps_failure_evidence_and_never_opens_extra_runs(tmp_path):
    """Run and coefficient quotas retain incomplete-output provenance without model execution."""
    import json

    files = []
    before = {}
    for index in range(10):
        folder = tmp_path / 'runs' / f'run-{index:02d}'
        folder.mkdir(parents=True)
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps(dict(app_key='mask', schema_version=1, status='partial')))
        coefficients = folder / 'results.csv'
        coefficients.write_text('feature,coefficient\narea,1.5\n')
        (folder / 'diagnostics').mkdir()
        report = folder / 'diagnostics' / 'fit.txt'
        report.write_text('owned diagnostic')
        files.extend((manifest, coefficients))
        before.update({path: path.read_bytes() for path in (manifest, coefficients, report)})
    result = home_mod._easy_run_evidence(tmp_path, files)
    assert len(result['manifests']) == 8
    assert all(item['missing_outputs'] == ['merged'] for item in result['manifests'])
    assert len(result['results']) == 4 and all(item['diagnostics'] for item in result['results'])
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0,
                    reason='requires POSIX permission enforcement for a non-root user')
def test_unreadable_and_malformed_local_evidence_leaves_accessible_results_intact(tmp_path):
    """Real filesystem permission failures are local refusals, not phantom successful runs."""
    import json

    runs = tmp_path / 'runs'
    runs.mkdir()
    broken = runs / 'broken'
    broken.mkdir()
    (broken / 'manifest.json').write_text('{invalid')
    denied = runs / 'denied'
    denied.mkdir()
    inaccessible = denied / 'manifest.json'
    inaccessible.write_text(json.dumps(dict(app_key='mask', schema_version=1, status='failed')))
    valid = tmp_path / 'results.csv'
    valid.write_text('feature,coefficient\narea,2\n')
    diagnostics = tmp_path / 'diagnostics'
    diagnostics.mkdir()
    (diagnostics / 'fit.txt').write_text('diagnostic')
    empty = tmp_path / 'empty' / 'results.csv'
    empty.parent.mkdir()
    empty.write_bytes(b'\xff')
    before = valid.read_bytes()
    inaccessible.chmod(0)
    diagnostics.chmod(0)
    try:
        result = home_mod._easy_run_evidence(tmp_path, [empty, valid])
        assert result['manifests'] == []
        assert result['results'] == [dict(path=str(valid), folder=str(tmp_path),
                                        diagnostics=False, diagnostics_folder=None)]
    finally:
        inaccessible.chmod(0o600)
        diagnostics.chmod(0o700)
    assert valid.read_bytes() == before


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0,
                    reason='requires POSIX permission enforcement for a non-root user')
def test_bounded_directory_sampling_preserves_selected_file_and_reports_real_permission_errors(tmp_path):
    """Reservoir sampling keeps the explicit input and reports unreadable immediate siblings."""
    root = tmp_path / 'source'
    root.mkdir()
    selected = root / 'explicit.csv'
    selected.write_text('value,other\n1,2\n')
    for index in range(180):
        (root / f'other-{index:03d}.csv').write_text('value,other\n1,2\n')
    denied = root / 'denied'
    denied.mkdir()
    denied.chmod(0)
    try:
        first = home_mod._easy_local_sample(selected, sample_limit=12)
        second = home_mod._easy_local_sample(selected, sample_limit=12)
        assert first == second
        files, folders, unreadable, limited = first
        assert selected in files and len(files) == 12
        assert unreadable == 1 and limited
    finally:
        denied.chmod(0o700)


def test_sidecar_budget_and_symlink_rules_preserve_only_structural_hints(tmp_path):
    """At most four sidecars are described; linked external metadata is never read."""
    import json

    folders = []
    for index in range(5):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / 'metadata.json').write_text(json.dumps({'channels': ['one'], 'private': 'excluded'}))
        folders.append(folder)
    hints = home_mod._easy_metadata_hints(folders)
    assert len(hints) == 4 and all(item['fields'] == ['channels'] for item in hints)
    linked = tmp_path / 'linked'
    linked.mkdir()
    (linked / 'metadata.json').symlink_to(folders[0] / 'metadata.json')
    assert home_mod._easy_small_json(linked / 'metadata.json') == {}
    assert home_mod._easy_metadata_hints([linked]) == []
    (folders[0] / 'metadata.json').write_text('{invalid')
    assert home_mod._easy_metadata_hints([folders[0]])[0]['unreadable']
