"""The four resource buttons, and the line they are not allowed to cross.

The dangerous version of this feature writes itself: find the biggest
processes, kill them; drop the page cache; terminate the threads that will
not stop. Every one of those makes the number on screen look better and
every one of them is a catastrophe on a shared machine — the box this was
written on sat at load 120 with twenty concurrent test suites, and "free as
much as possible" would have taken somebody's eight-hour training run to
make a segmentation start four seconds sooner.

So this file tests two different kinds of claim:

*the outcome* — a freed figure is a measured before/after and nothing else,
a confirmation names the action it is confirming, and declining does not run
anything;

*the refusal* — read as source text, because "there is no code path that can
kill a process spaCR did not start" is a statement about what is **not**
there, and no amount of exercising the happy path can assert an absence.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt import resource_cleanup as rc


SOURCE = Path(rc.__file__).read_text()


# ---------------------------------------------------------------------------
# The refusal, read out of the source
# ---------------------------------------------------------------------------

#: Every way this module could reach a process it did not start, or the
#: operating system's own memory. Each is a regex over the source rather
#: than an import check, because the failure this guards against is somebody
#: adding one line in a hurry, and a line is what a regex sees.
FORBIDDEN = {
    "os.kill": r"\bos\.kill\b",
    "signal delivery": r"\bsignal\.(SIG|kill|raise_signal)",
    "a subprocess": r"\bsubprocess\b|\bos\.system\b|\bos\.popen\b",
    "psutil process control": r"\.(terminate|kill|send_signal|suspend)\s*\(",
    "QThread.terminate": r"\.terminate\s*\(",
    "dropping the page cache": r"drop_caches",
    "sysctl": r"\bsysctl\b",
    "sudo/root": r"\bsudo\b|\bsetuid\b|geteuid",
    "cancelling somebody's run": r"cancel_all|request_cancel",
    "emptying a queue of pending work": r"QThreadPool[^\n]*\.clear\s*\(",
}


@pytest.mark.parametrize("what,pattern", sorted(FORBIDDEN.items()))
def test_the_cleanup_module_cannot_do_this(what, pattern):
    """Read the source and prove the dangerous thing is not in it.

    Comments and docstrings are stripped first: this module *discusses*
    ``QThread.terminate()`` at length, and a test that could not tell an
    explanation apart from a call would force the explanation out — which
    is the wrong way round, because the explanation is the reason the call
    is not there.
    """
    code = _code_only(SOURCE)
    hits = [line for line in code.splitlines() if re.search(pattern, line)]
    assert not hits, f"{what} is reachable from resource_cleanup: {hits}"


def _code_only(text: str) -> str:
    """``text`` with triple-quoted blocks and ``#`` comments removed.

    Line structure is preserved, because the assertions above report the
    offending line back and a line number that means nothing is a worse
    error message than none.
    """
    without_docstrings = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", text)
    return re.sub(r"#[^\n]*", "", without_docstrings)


def test_the_stripper_would_have_caught_a_real_call():
    """The guard above is only as good as its blindness to prose."""
    sample = '"""os.kill is never called."""\nx = 1  # os.kill\n'
    assert "os.kill" not in _code_only(sample)
    assert "os.kill" in _code_only("os.kill(1, 9)\n")
    assert "os.kill" in _code_only('x = 1\nos.kill(p, 9)  # tidy up\n')


def test_the_cpu_cleanup_reads_the_run_registry_and_never_cancels_it():
    code = _code_only(SOURCE)
    assert "registry" in code, "it should ask what is running"
    assert "prune_parked_threads" in code, (
        "releasing a parked thread that has exited is the whole mechanism")
    # The words "cancel"/"stopped" appear in the confirmation text, which
    # promises the opposite; the calls are what must be absent.
    assert "cancel_all" not in code
    assert "request_cancel" not in code


# ---------------------------------------------------------------------------
# Reporting: measured, or said to be unmeasurable
# ---------------------------------------------------------------------------

def test_freed_is_the_measured_difference_and_nothing_else(monkeypatch):
    readings = iter([500 * 1024 * 1024, 300 * 1024 * 1024])
    monkeypatch.setattr(rc, "process_rss", lambda: next(readings))
    result = rc.clear_ram()
    assert result.before == 500 * 1024 * 1024
    assert result.after == 300 * 1024 * 1024
    assert result.freed == 200 * 1024 * 1024
    assert "200.0 MB" in result.summary()


def test_a_cleanup_that_freed_nothing_says_so(monkeypatch):
    monkeypatch.setattr(rc, "process_rss", lambda: 400 * 1024 * 1024)
    result = rc.clear_ram()
    assert result.freed == 0
    assert "freed nothing" in result.summary()
    assert "200" not in result.summary()


def test_memory_that_grew_is_not_reported_as_freed(monkeypatch):
    readings = iter([100, 180])
    monkeypatch.setattr(rc, "process_rss", lambda: next(readings))
    result = rc.clear_ram()
    assert result.freed == 0
    assert result.grew == 80
    assert "freed nothing" in result.summary()
    assert "more is in use" in result.summary()


def test_an_unmeasurable_cleanup_reports_that_rather_than_a_number(
        monkeypatch):
    monkeypatch.setattr(rc, "process_rss", lambda: 0)
    result = rc.clear_ram()
    assert result.measured is False
    assert result.freed == 0
    assert "could not be read" in result.summary()


def test_the_ram_cleanup_really_drops_spacrs_own_caches():
    import spacr.crops as crops
    crops._FIELD_CACHE[("x", 1, 2)] = object()
    crops._FORMAT_CACHE["y"] = (None, None)
    result = rc.clear_ram()
    assert not crops._FIELD_CACHE
    assert not crops._FORMAT_CACHE
    assert any("_FIELD_CACHE" in detail for detail in result.details)


def test_nothing_cached_is_reported_as_nothing_cached(monkeypatch):
    import spacr.crops as crops
    crops._FIELD_CACHE.clear()
    crops._FORMAT_CACHE.clear()
    crops._DB_FORMAT_CACHE.clear()
    monkeypatch.setattr(rc.gc, "collect", lambda *a: 0)
    monkeypatch.setattr(rc, "process_rss", lambda: 1024)
    result = rc.clear_ram()
    assert result.details == ()
    assert "nothing was cached" in result.summary().lower()


def test_the_vram_cleanup_says_what_it_cannot_do(monkeypatch):
    class _FakeCuda:
        def __init__(self):
            self.emptied = 0
            self.reserved = [8 * 1024 ** 3, 2 * 1024 ** 3]

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def memory_reserved(self):
            return self.reserved[min(self.emptied, 1)]

        def empty_cache(self):
            self.emptied += 1

    class _FakeTorch:
        def __init__(self):
            self.cuda = _FakeCuda()

    fake = _FakeTorch()
    monkeypatch.setattr(rc, "_torch_if_loaded", lambda: fake)
    result = rc.clear_vram()
    assert fake.cuda.emptied == 1
    assert result.freed == 6 * 1024 ** 3
    assert "another process" in result.note
    assert "torch.cuda.empty_cache()" in result.details


def test_the_vram_cleanup_never_initialises_cuda_just_to_look(monkeypatch):
    """A button whose job is to free VRAM must not allocate a context."""
    calls = []

    class _FakeCuda:
        def is_available(self):
            return True

        def is_initialized(self):
            return False

        def memory_reserved(self):
            calls.append("memory_reserved")
            return 0

        def empty_cache(self):
            calls.append("empty_cache")

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setattr(rc, "_torch_if_loaded", lambda: _FakeTorch())
    result = rc.clear_vram()
    assert calls == []
    assert result.measured is False
    assert "no initialised CUDA context" in result.note


def test_no_torch_means_no_vram_claim(monkeypatch):
    monkeypatch.setattr(rc, "_torch_if_loaded", lambda: None)
    result = rc.clear_vram()
    assert result.measured is False
    assert result.freed == 0
    assert "torch is not loaded" in result.note


def test_the_cpu_cleanup_leaves_a_running_job_alone(monkeypatch):
    """It reports what is running; it does not touch it."""
    class _Handle:
        app_key = "measure"

    class _Registry:
        def __init__(self):
            self.handles = [_Handle()]

        def active(self):
            return list(self.handles)

    registry = _Registry()
    import spacr.qt.bridge as bridge
    monkeypatch.setattr(bridge, "registry", lambda: registry)
    result = rc.clear_cpu()
    assert registry.handles, "the cleanup removed a running job"
    assert "1 spaCR job(s) are still running" in result.note


def test_library_thread_counts_never_go_below_the_floor(monkeypatch):
    import sys
    import types
    fake = types.ModuleType("torch")
    fake.state = 32
    fake.get_num_threads = lambda: fake.state
    def _set(n):
        fake.state = n
    fake.set_num_threads = _set
    monkeypatch.setitem(sys.modules, "torch", fake)
    rc._lower_library_threads(target=1)
    assert fake.state == rc.MIN_LIBRARY_THREADS >= 2


# ---------------------------------------------------------------------------
# Disk: read-only, and deduplicated by drive
# ---------------------------------------------------------------------------

def test_the_disk_check_only_reads(tmp_path):
    (tmp_path / "keep.txt").write_text("data")
    report = rc.disk_report([str(tmp_path)])
    assert len(report.entries) == 1
    entry = report.entries[0]
    assert entry.total > 0 and entry.free >= 0
    assert 0 <= entry.percent_used <= 100
    assert (tmp_path / "keep.txt").read_text() == "data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_one_line_per_drive_not_per_folder(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    report = rc.disk_report([str(a), str(b)])
    assert len(report.entries) == 1, (
        "two folders on one filesystem produced two identical lines")


def test_a_folder_that_is_gone_is_counted_not_crashed_on(tmp_path):
    report = rc.disk_report([str(tmp_path / "not-there"), str(tmp_path)])
    assert len(report.entries) == 1
    assert "could not be read" in report.note


def test_no_known_project_says_so_rather_than_showing_nothing():
    report = rc.disk_report([])
    assert report.entries == ()
    assert "No project folder is known yet" in report.summary()


def test_the_tightest_drive_is_the_one_worth_reading():
    entries = (rc.DiskEntry("/big", 100, 10, 90),
               rc.DiskEntry("/small", 100, 95, 5))
    assert rc.DiskReport(entries).tightest.path == "/small"


# ---------------------------------------------------------------------------
# The confirmations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action", rc.ACTIONS)
def test_every_confirmation_names_what_will_happen(action):
    text = rc.confirmation_text(action)
    assert rc.confirmation_title(action)
    assert len(text.split()) >= 25, "too thin to be a description of an action"
    assert "spaCR will" in text or "spaCR will" in rc.confirmation_title(action)
    lowered = text.lower()
    assert "are you sure" not in lowered, (
        "a user cannot consent to an unnamed action")
    assert "cannot" in lowered or "will not" in lowered or "only reads" \
        in lowered, "it must also say what the action cannot do"


def test_the_confirmations_name_the_specific_mechanism():
    assert "garbage collection" in rc.confirmation_text("ram")
    assert "empty_cache" in rc.confirmation_text("vram")
    assert "torch" in rc.confirmation_text("cpu")
    assert "read" in rc.confirmation_text("disk")


def test_the_ram_confirmation_admits_the_cost_of_pressing_it():
    assert "slower" in rc.confirmation_text("ram")


def test_the_cpu_confirmation_promises_not_to_kill_anything():
    text = rc.confirmation_text("cpu").lower()
    assert "no process is killed" in text
    assert "no running or queued job is stopped" in text
    assert "anybody else" in text, (
        "the promise is about other people's work, so it has to say so")


def test_the_vram_confirmation_admits_the_limit():
    assert "another process" in rc.confirmation_text("vram")


def test_the_disk_confirmation_promises_it_writes_nothing():
    assert "only reads" in rc.confirmation_text("disk").lower()


# ---------------------------------------------------------------------------
# The buttons, through the dialog
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action,object_name", [
    ("ram", "ClearRamButton"),
    ("vram", "ClearVramButton"),
    ("cpu", "ClearCpuButton"),
    ("disk", "CheckDiskButton"),
])
def test_preferences_offers_each_button(action, object_name, qtbot,
                                        qt_theme_applied):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    button = dlg.findChild(QPushButton, object_name)
    assert button is not None, f"no button for {action}"
    assert button.toolTip() == rc.confirmation_text(action)


@pytest.mark.parametrize("action", rc.ACTIONS)
def test_declining_the_confirmation_does_absolutely_nothing(action,
                                                            monkeypatch):
    from spacr.qt import preferences as prefs

    ran = []
    monkeypatch.setattr(prefs, "confirm_resource_action",
                        lambda *a, **k: False)
    for name in ("clear_ram", "clear_vram", "clear_cpu", "disk_report"):
        monkeypatch.setattr(rc, name,
                            lambda *a, _n=name, **k: ran.append(_n))
    monkeypatch.setattr(prefs, "_show_resource_result",
                        lambda *a, **k: ran.append("reported"))

    assert prefs.run_resource_action(action) is None
    assert ran == [], f"declining {action} still ran {ran}"


@pytest.mark.parametrize("action", rc.ACTIONS)
def test_accepting_runs_exactly_that_action_and_reports_it(action,
                                                           monkeypatch):
    from spacr.qt import preferences as prefs

    ran = []
    shown = []
    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    for name in ("clear_ram", "clear_vram", "clear_cpu", "disk_report"):
        monkeypatch.setattr(
            rc, name,
            lambda *a, _n=name, **k: (ran.append(_n) or rc.Reclaim("ram")))
    monkeypatch.setattr(prefs, "_show_resource_result",
                        lambda act, result, parent=None: shown.append(act))

    prefs.run_resource_action(action)
    expected = {"ram": "clear_ram", "vram": "clear_vram", "cpu": "clear_cpu",
                "disk": "disk_report"}[action]
    assert ran == [expected]
    assert shown == [action]


def test_the_button_confirms_before_it_acts_not_after(monkeypatch, qtbot,
                                                      qt_theme_applied):
    """Order matters: a confirmation asked afterwards is a notification."""
    from PySide6.QtWidgets import QPushButton
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import PreferencesDialog

    order = []
    monkeypatch.setattr(prefs, "confirm_resource_action",
                        lambda *a, **k: (order.append("asked"), True)[1])
    monkeypatch.setattr(rc, "clear_ram",
                        lambda *a, **k: (order.append("ran"),
                                         rc.Reclaim("ram"))[1])
    monkeypatch.setattr(prefs, "_show_resource_result",
                        lambda *a, **k: order.append("reported"))

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg.findChild(QPushButton, "ClearRamButton").click()
    assert order == ["asked", "ran", "reported"]


def test_the_reported_figure_is_the_one_that_was_measured(monkeypatch):
    """What the dialog shows is the Reclaim's own summary, not a retelling."""
    from spacr.qt import preferences as prefs

    result = rc.Reclaim("ram", before=10 * 1024 ** 2, after=4 * 1024 ** 2)
    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(rc, "clear_ram", lambda *a, **k: result)
    shown = {}
    monkeypatch.setattr(
        prefs, "_show_resource_result",
        lambda action, res, parent=None: shown.update(text=res.summary()))
    returned = prefs.run_resource_action("ram")
    assert returned is result
    assert shown["text"] == result.summary()
    assert "6.0 MB" in shown["text"]
