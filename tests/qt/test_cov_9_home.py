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


def test_header_reader_rejects_an_actual_atomic_file_replacement(tmp_path, monkeypatch):
    """Replacing the path between stat and open cannot pass a different file as the original."""
    path, previous = tmp_path / 'input.bin', tmp_path / 'original.bin'
    path.write_bytes(b'original header')
    original = home_mod.os

    class ReplacedFile:
        def __getattr__(self, name):
            return getattr(original, name)

        def open(self, selected, flags):
            assert selected == path
            original.replace(path, previous)
            path.write_bytes(b'replacement header')
            return original.open(selected, flags)

    monkeypatch.setattr(home_mod, 'os', ReplacedFile())
    with pytest.raises(ValueError, match='Not a regular file'):
        home_mod._easy_read_header(path, 8)
    assert previous.read_bytes() == b'original header'
    assert path.read_bytes() == b'replacement header'


@pytest.mark.parametrize('problem', ['missing_samples', 'unmatched_example'])
def test_inline_regex_validation_requires_both_sample_and_example_identity(problem):
    """A valid-looking expression cannot seed settings without all required local evidence."""
    import copy

    pattern = r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$'
    result = {'regex': dict(sample_filenames=['scan_1_C1.tif'], img_format='tif',
                            example_path='/local/not-the-sample.tif')}
    if problem == 'missing_samples':
        result['regex']['sample_filenames'] = []
    before = copy.deepcopy(result)
    actual = home_mod._easy_validate_regex_inline(result, pattern)
    assert actual['regex']['required'] and actual['regex']['validation_error']
    assert 'settings_overrides' not in actual
    assert result == before


@pytest.mark.parametrize('receiver_closed', [False, True])
def test_regex_worker_invalid_payload_closes_the_real_pipe(receiver_closed):
    """Bad serialized input and a retired receiver cannot leave an owned pipe open."""
    import multiprocessing

    receiver, sender = multiprocessing.Pipe(duplex=False)
    if receiver_closed:
        receiver.close()
    try:
        home_mod._easy_regex_worker('{invalid', '(', sender)
        assert sender.closed
        if not receiver_closed:
            assert receiver.recv_bytes() == b'null'
    finally:
        receiver.close()
        sender.close()


def test_regex_child_missing_interpreter_returns_local_refusal_and_retires_process(tmp_path):
    """A real failed spawn cannot leave a trusted mapping or a live owned child."""
    import copy
    import multiprocessing
    from multiprocessing import spawn

    result = {'regex': dict(sample_filenames=['scan_1_C1.tif'], img_format='tif',
                            example_path='/local/scan_1_C1.tif')}
    before = copy.deepcopy(result)
    executable = spawn.get_executable()
    children = {child.pid for child in multiprocessing.active_children()}
    try:
        multiprocessing.set_executable(str(tmp_path / 'missing-python'))
        actual = home_mod._easy_validate_regex(result, r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$')
    finally:
        multiprocessing.set_executable(executable)
    assert actual['regex']['required'] and actual['regex']['validation_error']
    assert result == before
    assert {child.pid for child in multiprocessing.active_children()} == children


def _input_header_tiff(path):
    from PIL import Image
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.arange(30, dtype=np.uint16).reshape(5, 6)).save(path)
    return path


def _deny_named_input_scan(monkeypatch, denied):
    from pathlib import Path

    original = home_mod.os

    class NamedScanBoundary:
        def __getattr__(self, name):
            return getattr(original, name)

        def scandir(self, path):
            if Path(path) == denied:
                raise PermissionError('the named acquisition directory is unavailable')
            return original.scandir(path)

    monkeypatch.setattr(home_mod, 'os', NamedScanBoundary())


def test_corrupt_merged_header_keeps_real_image_mask_pairs_unstructured(tmp_path):
    image = _input_header_tiff(tmp_path / 'field.tif')
    mask = _input_header_tiff(tmp_path / 'field_mask.tif')
    merged = tmp_path / 'merged' / 'field.npy'
    merged.parent.mkdir()
    merged.write_bytes(b'\x93NUMPYtruncated')
    before = {path: path.read_bytes() for path in (image, mask, merged)}
    result = home_mod._easy_paired_layout([image, mask], tmp_path)
    assert result == dict(kind='unstructured', pairs={str(image): str(mask)},
                          project=str(tmp_path), filename_evidence=True)
    assert home_mod._easy_merged_evidence(tmp_path) is False
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_manifest_discovers_its_coefficient_table_outside_the_supplied_sample(tmp_path):
    import json

    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(app_key='regression', schema_version=1,
                                        status='success')))
    table = tmp_path / 'results_gene.csv'
    table.write_text('feature,coefficient\narea,1.5\n')
    before = {path: path.read_bytes() for path in (manifest, table)}
    result = home_mod._easy_run_evidence(tmp_path, [])
    assert result == dict(manifests=[], results=[dict(path=str(table), folder=str(tmp_path),
                        diagnostics=False, diagnostics_folder=None)])
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_malformed_named_settings_remain_a_local_refusal_without_applying_values(tmp_path):
    path = tmp_path / 'mask_settings.json'
    path.write_text('{"src": broken')
    before = path.read_bytes()
    result = home_mod._easy_inspect_path(str(path))
    assert result['path'] == str(path) and result['actions'] == []
    assert 'saved_settings' not in result.get('context', {})
    assert path.read_bytes() == before


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='requires a real local FIFO')
def test_selected_fifo_is_refused_before_any_header_read(tmp_path, monkeypatch):
    from spacr.qt.i18n import tr

    path = tmp_path / 'input.pipe'
    os.mkfifo(path)

    def unexpected(*args, **kwargs):
        raise AssertionError('a FIFO must never reach the blocking header reader')

    monkeypatch.setattr(home_mod, '_easy_read_header', unexpected)
    result = home_mod._easy_inspect_path(str(path))
    assert result == dict(path=str(path), actions=[], message=tr(
        'Choose a regular file or folder, rather than a device or pipe.'))


def test_corrupt_sqlite_in_a_selected_folder_cannot_become_scientific_evidence(tmp_path):
    database = tmp_path / 'measurements.db'
    database.write_bytes(b'SQLite format 3\x00' + bytes(112))
    before = database.read_bytes()
    result = home_mod._easy_inspect_path(str(tmp_path))
    assert result['context']['database_schemas'] == []
    assert result['context']['analysis_inputs'] == {}
    assert not any(action[0] in ('db_browser', 'graph_builder', 'regression', 'umap')
                   for action in result['actions'])
    assert database.read_bytes() == before


def test_failed_plate_directory_scan_does_not_publish_partial_database_discovery(tmp_path, monkeypatch):
    import sqlite3

    database = tmp_path / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (cell_area REAL)')
    before = database.read_bytes()
    _deny_named_input_scan(monkeypatch, tmp_path)
    details = {}
    assert home_mod._easy_analysis_inputs(tmp_path, details) == []
    assert details == {} and database.read_bytes() == before


def test_unreadable_merged_directory_preserves_local_pairing_without_promoting_project(tmp_path, monkeypatch):
    image = _input_header_tiff(tmp_path / 'field.tif')
    mask = _input_header_tiff(tmp_path / 'field_mask.tif')
    merged = tmp_path / 'merged'
    merged.mkdir()
    before = {path: path.read_bytes() for path in (image, mask)}
    _deny_named_input_scan(monkeypatch, merged)
    result = home_mod._easy_paired_layout([image, mask], tmp_path)
    assert result['kind'] == 'unstructured' and result['project'] == str(tmp_path)
    assert result['pairs'] == {str(image): str(mask)}
    assert home_mod._easy_merged_evidence(tmp_path) is False
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('value', [{'unit': 'unknown'}, 'not a physical scale'])
def test_spatial_calibration_requires_a_positive_numeric_value(tmp_path, value):
    import json

    slide = _input_header_tiff(tmp_path / 'slide.tif')
    scale = tmp_path / 'spatial' / 'scalefactors_json.json'
    scale.parent.mkdir()
    scale.write_text(json.dumps({'microns_per_pixel': value}))
    before = {path: path.read_bytes() for path in (slide, scale)}
    assert home_mod._easy_specialized_inputs(tmp_path, [slide], [], []) == {}
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_empty_hd_bins_offer_only_calibrated_morphology_review(tmp_path):
    import json

    slide = _input_header_tiff(tmp_path / 'slide.tif')
    (tmp_path / 'binned_outputs').mkdir()
    scale = tmp_path / 'spatial' / 'scalefactors_json.json'
    scale.parent.mkdir()
    scale.write_text(json.dumps({'microns_per_pixel': .5}))
    before = {path: path.read_bytes() for path in (slide, scale)}
    hints = home_mod._easy_specialized_inputs(tmp_path, [slide], [], [])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['settings'] == dict(spatial_source=str(tmp_path),
                                                    spatial_platform='visium_hd')
    assert hints['spatial_masks']['image'] == str(slide)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_unreadable_hd_bins_do_not_guess_a_bin_or_spatial_mapping(tmp_path, monkeypatch):
    import json

    slide = _input_header_tiff(tmp_path / 'slide.tif')
    bins = tmp_path / 'binned_outputs'
    (bins / 'square_002um').mkdir(parents=True)
    scale = tmp_path / 'spatial' / 'scalefactors_json.json'
    scale.parent.mkdir()
    scale.write_text(json.dumps({'microns_per_pixel': .5}))
    before = {path: path.read_bytes() for path in (slide, scale)}
    _deny_named_input_scan(monkeypatch, bins)
    hints = home_mod._easy_specialized_inputs(tmp_path, [slide], [], [])
    assert set(hints) == {'spatial_masks'}
    assert 'spatial_bin_um' not in hints['spatial_masks']['settings']
    assert hints['spatial_masks']['image'] == str(slide)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_unreadable_xenium_focus_directory_uses_the_real_mip_fallback(tmp_path, monkeypatch):
    marker = tmp_path / 'experiment.xenium'
    marker.write_text('{}')
    slide = _input_header_tiff(tmp_path / 'morphology_mip.ome.tif')
    focus = tmp_path / 'morphology_focus'
    focus.mkdir()
    before = {path: path.read_bytes() for path in (marker, slide)}
    _deny_named_input_scan(monkeypatch, focus)
    hints = home_mod._easy_specialized_inputs(tmp_path, [slide], [], [])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['image'] == str(slide)
    assert hints['spatial_masks']['settings']['spatial_platform'] == 'xenium'
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_unreadable_count_directory_does_not_promote_spatial_mapping(tmp_path, monkeypatch):
    import json
    import h5py

    slide = _input_header_tiff(tmp_path / 'slide.tif')
    spatial = tmp_path / 'spatial'
    spatial.mkdir()
    scale = spatial / 'scalefactors_json.json'
    scale.write_text(json.dumps({'microns_per_pixel': .5}))
    positions = spatial / 'tissue_positions.csv'
    positions.write_text('barcode,in_tissue\na,1\n')
    counts = tmp_path / 'filtered_feature_bc_matrix.h5'
    with h5py.File(counts, 'w') as archive:
        archive.create_dataset('data', data=[1])
    before = {path: path.read_bytes() for path in (slide, scale, positions, counts)}
    _deny_named_input_scan(monkeypatch, tmp_path)
    hints = home_mod._easy_specialized_inputs(tmp_path, [slide], [], [])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['image'] == str(slide)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_spatial_schema_without_source_column_cannot_bind_an_acquisition(tmp_path):
    import sqlite3
    import numpy as np
    import pandas as pd
    from spacr.tabular import write_table

    acquisition = tmp_path / 'acquisition'
    slide = _input_header_tiff(acquisition / 'morphology_focus.ome.tif')
    transcripts = acquisition / 'transcripts.parquet'
    write_table(pd.DataFrame({'x_location': [1.0], 'y_location': [2.0]}), transcripts)
    project = tmp_path / 'project'
    merged = project / 'merged' / 'field.npy'
    merged.parent.mkdir(parents=True)
    np.save(merged, np.zeros((2, 3, 2), dtype=np.uint16))
    database = project / 'measurements' / 'measurements.db'
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE spatial_fields (field_id INTEGER)')
    before = {path: path.read_bytes() for path in (slide, transcripts, merged, database)}
    schemas = [dict(path=str(database), tables=['spatial_fields'])]
    hints = home_mod._easy_specialized_inputs(acquisition, [slide], [], schemas)
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['source'] == str(acquisition)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_ops_library_requires_local_readable_headers_without_changing_tiles(tmp_path):
    root = tmp_path / 'selected'
    tile = _input_header_tiff(root / 'genotype' / '10X_c2_A1_A594_Site-0.tif')
    malformed = root / 'library.csv'
    malformed.write_bytes(b'\xffinvalid table')
    outside = tmp_path / 'outside.csv'
    outside.write_text('barcode,sequence\na,ACGT\n')
    before = {path: path.read_bytes() for path in (tile, malformed, outside)}
    hints = home_mod._easy_specialized_inputs(root, [outside, malformed], [tile], [])
    assert set(hints) == {'ops'}
    assert hints['ops']['settings'] == dict(genotype_source=str(tile.parent))
    assert hints['ops']['source'] == str(tile.parent)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_mixed_local_images_keep_external_document_guidance_without_claiming_its_reader(tmp_path):
    image = _input_header_tiff(tmp_path / 'field.tif')
    workspace = tmp_path / 'analysis.wsp'
    workspace.write_bytes(b'<Workspace/>')
    before = {path: path.read_bytes() for path in (image, workspace)}
    result = home_mod._easy_inspect_path(str(tmp_path))
    assert [action[0] for action in result['actions']] == ['make_masks', 'mask']
    assert home_mod._easy_external_guidance('flowjo') in result['message']
    assert result['context']['unsupported_documents'] == [dict(kind='flowjo', path=str(workspace))]
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_selected_path_stat_failure_becomes_local_guidance_without_opening_it(tmp_path, monkeypatch):
    from pathlib import Path
    from spacr.qt.i18n import tr

    path = tmp_path / 'image.tif'
    path.write_bytes(b'untouched input')
    before = path.read_bytes()
    exists = Path.exists

    def denied(candidate):
        if candidate == path:
            raise PermissionError('the selected path is unavailable')
        return exists(candidate)

    monkeypatch.setattr(Path, 'exists', denied)
    result = home_mod._easy_inspect_path(str(path))
    assert result == dict(path=str(path), actions=[], message=tr(
        'This input could not be inspected. Check permissions and choose another path.'))
    assert path.read_bytes() == before


def test_regex_cancellation_after_real_spawn_retires_its_owned_child(monkeypatch):
    """A real cancellation event after start is handled before accepting a worker result."""
    import copy
    import multiprocessing
    import threading

    context = multiprocessing.get_context('spawn')
    factory = context.Process
    start = factory.start
    cancelled, created = threading.Event(), []

    def started(child):
        start(child)
        created.append(child)
        cancelled.set()

    monkeypatch.setattr(factory, 'start', started)
    result = {'regex': dict(sample_filenames=['scan_1_C1.tif'], img_format='tif',
                            example_path='/local/scan_1_C1.tif')}
    before = copy.deepcopy(result)
    children = {child.pid for child in multiprocessing.active_children()}
    actual = home_mod._easy_validate_regex(result, r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$',
                                          cancelled=cancelled.is_set)
    assert cancelled.is_set() and len(created) == 1 and created[0]._closed
    assert actual['regex']['required'] and 'validation_error' not in actual['regex']
    assert 'settings_overrides' not in actual and result == before
    assert {child.pid for child in multiprocessing.active_children()} == children


def _regex_protocol_worker_ignoring_termination(payload, pattern, connection, ready):
    import signal
    import threading

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    connection.send_bytes(b'null')
    try:
        threading.Event().wait(30)
    finally:
        connection.close()


@pytest.mark.skipif(os.name != 'posix', reason='requires real POSIX terminate/kill semantics')
def test_invalid_regex_worker_reply_escalates_real_child_retirement_to_kill(monkeypatch):
    """A protocol-only worker refuses SIGTERM; the owner must kill, join and close it."""
    import copy
    import multiprocessing
    import signal

    context = multiprocessing.get_context('spawn')
    process_class = context.Process
    ready = context.Event()
    children, pipes, operations, exits = [], [], [], []
    original_start = process_class.start
    original_terminate = process_class.terminate
    original_kill = process_class.kill
    original_close = process_class.close

    class ProtocolWorkerContext:
        def Pipe(self, **kwargs):
            endpoints = context.Pipe(**kwargs)
            pipes.extend(endpoints)
            return endpoints

        def Process(self, *, target, args):
            assert target is home_mod._easy_regex_worker
            child = process_class(target=_regex_protocol_worker_ignoring_termination,
                                  args=(*args, ready))
            children.append(child)
            return child

    def start(child):
        original_start(child)
        assert ready.wait(10), 'the real child must install its signal policy before the protocol reply'

    def terminate(child):
        operations.append('terminate')
        original_terminate(child)

    def kill(child):
        operations.append('kill')
        original_kill(child)

    def close(child):
        operations.append('close')
        exits.append(child.exitcode)
        original_close(child)

    monkeypatch.setattr(multiprocessing, 'get_context', lambda method: ProtocolWorkerContext())
    monkeypatch.setattr(process_class, 'start', start)
    monkeypatch.setattr(process_class, 'terminate', terminate)
    monkeypatch.setattr(process_class, 'kill', kill)
    monkeypatch.setattr(process_class, 'close', close)
    result = {'regex': dict(sample_filenames=['scan_1_C1.tif'], img_format='tif',
                            example_path='/local/scan_1_C1.tif')}
    before = copy.deepcopy(result)
    prior_children = {child.pid for child in multiprocessing.active_children()}
    try:
        actual = home_mod._easy_validate_regex(result, r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$')
        assert operations == ['terminate', 'kill', 'close']
        assert exits == [-signal.SIGKILL]
        assert len(children) == 1 and children[0]._closed
        assert all(connection.closed for connection in pipes)
        assert actual['regex']['required'] and actual['regex']['validation_error']
        assert result == before and 'settings_overrides' not in actual
        assert {child.pid for child in multiprocessing.active_children()} == prior_children
    finally:
        for child in children:
            if not child._closed:
                if child.is_alive():
                    original_kill(child)
                child.join()
                original_close(child)
        for connection in pipes:
            connection.close()


def test_queued_directory_replaced_by_a_symlink_does_not_expand_the_input(tmp_path, monkeypatch):
    from pathlib import Path

    root = tmp_path / 'selected'
    child = root / 'acquisition'
    child.mkdir(parents=True)
    local = child / 'local.txt'
    local.write_text('local acquisition notes')
    outside = tmp_path / 'outside'
    outside.mkdir()
    foreign = outside / 'private.txt'
    foreign.write_text('not part of the selected input')
    archived = tmp_path / 'original-acquisition'
    original_os = home_mod.os
    visited = []

    class AdmissionEntry:
        def __init__(self, entry):
            self.entry = entry

        def __getattr__(self, name):
            return getattr(self.entry, name)

        def is_dir(self, *, follow_symlinks=True):
            admitted = self.entry.is_dir(follow_symlinks=follow_symlinks)
            if admitted and Path(self.entry.path) == child:
                child.rename(archived)
                child.symlink_to(outside, target_is_directory=True)
            return admitted

    class SelectedScan:
        def __enter__(self):
            self.entries = original_os.scandir(root)
            return (AdmissionEntry(entry) for entry in self.entries)

        def __exit__(self, *args):
            self.entries.close()

    class DirectoryAdmissionBoundary:
        def __getattr__(self, name):
            return getattr(original_os, name)

        def scandir(self, folder):
            visited.append(Path(folder))
            return SelectedScan() if Path(folder) == root else original_os.scandir(folder)

    monkeypatch.setattr(home_mod, 'os', DirectoryAdmissionBoundary())
    files, directories, unreadable, limited = home_mod._easy_local_sample(root)
    assert files == [] and directories == (root,) and visited == [root]
    assert unreadable == 0 and not limited
    assert child.is_symlink() and child.resolve() == outside
    assert (archived / local.name).read_text() == 'local acquisition notes'
    assert foreign.read_text() == 'not part of the selected input'


def test_unreadable_representative_keeps_accessible_image_routes_and_permission_guidance(tmp_path, monkeypatch):
    image = _input_header_tiff(tmp_path / 'field.tif')
    denied = tmp_path / 'unreadable.dat'
    denied.write_bytes(b'private inaccessible acquisition metadata')
    before = {path: path.read_bytes() for path in (image, denied)}
    original_read = home_mod._easy_read_header
    refused = []

    def read(path, limit, offset=0):
        if path == denied:
            refused.append(path)
            raise PermissionError('the selected representative cannot be read')
        return original_read(path, limit, offset)

    monkeypatch.setattr(home_mod, '_easy_read_header', read)
    result = home_mod._easy_inspect_path(str(tmp_path))
    actions = {action[0]: action[3] for action in result['actions']}
    assert refused and actions['make_masks'] == str(tmp_path)
    assert actions['mask'] == str(tmp_path)
    assert 'Some files could not be read' in result['message']
    assert result['context']['database_schemas'] == []
    assert result['context']['image_dimensions'][0]['path'] == str(image)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_table_removed_after_classification_does_not_gain_numeric_analysis_routes(tmp_path, monkeypatch):
    image = _input_header_tiff(tmp_path / 'field.tif')
    table = tmp_path / 'observations.csv'
    table.write_text('field,value\n1,2.5\n2,3.5\n')
    original = table.read_bytes()
    image_bytes = image.read_bytes()
    archived = tmp_path / 'removed-observations.csv'
    original_read = home_mod._easy_read_header
    removed = []

    def read(path, limit, offset=0):
        if path == table and limit == home_mod._EASY_HEADER_LIMIT and not removed:
            table.rename(archived)
            removed.append(path)
        return original_read(path, limit, offset)

    monkeypatch.setattr(home_mod, '_easy_read_header', read)
    result = home_mod._easy_inspect_path(str(tmp_path))
    keys = {action[0] for action in result['actions']}
    assert removed == [table] and not table.exists()
    assert {'make_masks', 'mask'} <= keys
    assert keys.isdisjoint({'graph_builder', 'gate_editor', 'regression'})
    assert result['context']['analysis_inputs'] == {}
    assert archived.read_bytes() == original and image.read_bytes() == image_bytes


def test_xenium_counts_without_readable_morphology_do_not_offer_spatial_steps(tmp_path):
    import pandas as pd
    from spacr.tabular import write_table

    marker = tmp_path / 'experiment.xenium'
    marker.write_text('{}')
    transcripts = tmp_path / 'transcripts.parquet'
    write_table(pd.DataFrame({'x_location': [1.], 'y_location': [2.]}), transcripts)
    slide = tmp_path / 'morphology_focus.ome.tif'
    slide.write_bytes(b'II*\x00\x08\x00\x00\x00')
    before = {path: path.read_bytes() for path in (marker, transcripts, slide)}
    assert home_mod._easy_specialized_inputs(tmp_path, [slide, transcripts], [], []) == {}
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('missing_evidence', ['spatial_table', 'merged_images'])
def test_spatial_mapping_requires_both_its_measurement_table_and_generated_images(tmp_path, missing_evidence):
    import sqlite3
    import numpy as np
    import pandas as pd
    from spacr.tabular import write_table

    acquisition = tmp_path / 'acquisition'
    slide = _input_header_tiff(acquisition / 'morphology_focus.ome.tif')
    transcripts = acquisition / 'transcripts.parquet'
    write_table(pd.DataFrame({'x_location': [1.], 'y_location': [2.]}), transcripts)
    project = tmp_path / 'measured'
    database = project / 'measurements' / 'measurements.db'
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        if missing_evidence == 'spatial_table':
            connection.execute('CREATE TABLE image_annotations (condition TEXT)')
        else:
            connection.execute('CREATE TABLE spatial_fields (source TEXT)')
            connection.execute('INSERT INTO spatial_fields VALUES (?)', (str(acquisition),))
    if missing_evidence == 'spatial_table':
        merged = project / 'merged' / 'field.npy'
        merged.parent.mkdir()
        np.save(merged, np.zeros((2, 3, 2), np.uint16))
    schema = dict(path=str(database), **home_mod._easy_database_evidence(database))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    hints = home_mod._easy_specialized_inputs(acquisition, [slide, transcripts], [], [schema])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['source'] == str(acquisition)
    assert hints['spatial_masks']['image'] == str(slide)
    assert 'spatial_map' not in hints
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_hd_root_calibration_does_not_validate_a_malformed_selected_bin_calibration(tmp_path):
    import json
    import h5py

    slide = _input_header_tiff(tmp_path / 'slide.tif')
    root_scale = tmp_path / 'spatial' / 'scalefactors_json.json'
    root_scale.parent.mkdir()
    root_scale.write_text(json.dumps({'microns_per_pixel': .5}))
    selected = tmp_path / 'binned_outputs' / 'square_002um'
    spatial = selected / 'spatial'
    spatial.mkdir(parents=True)
    bin_scale = spatial / 'scalefactors_json.json'
    bin_scale.write_text(json.dumps({'microns_per_pixel': ['not a calibration']}))
    positions = spatial / 'tissue_positions.csv'
    positions.write_text('barcode,in_tissue\na,1\n')
    counts = selected / 'filtered_feature_bc_matrix.h5'
    with h5py.File(counts, 'w') as archive:
        archive.create_dataset('data', data=[1])
    before = {path: path.read_bytes() for path in (slide, root_scale, bin_scale, positions, counts)}
    hints = home_mod._easy_specialized_inputs(tmp_path, [slide], [], [])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['settings'] == dict(
        spatial_source=str(tmp_path), spatial_platform='visium_hd', spatial_bin_um=2)
    assert hints['spatial_masks']['image'] == str(slide)
    assert all(path.read_bytes() == raw for path, raw in before.items())
