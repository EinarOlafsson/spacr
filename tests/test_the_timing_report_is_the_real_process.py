"""``spacr.qt.timing`` with the switch actually on.

The module reads ``SPACR_TIMING`` once, at import, into a module-level
``ENABLED``. The suite runs without it, so every recorder in here returns
early and the module sat at 55% -- not because the code is untested by
oversight, but because no test had ever turned it on.

That matters more than a coverage number. This module is the instrument: it
is what says whether a slow start-up was imports, a stylesheet rebuild or a
blocked event loop, and it is the only honest freeze measurement spaCR has,
because the stall watchdog is the event loop reporting on itself. An
instrument nobody has calibrated is not evidence.

Everything here re-imports the module with the variable set, and puts the
process back as it found it: the import timer patches
``SourceFileLoader.exec_module`` for the whole interpreter, so a test that
leaked it would slow and pollute every import in the rest of the run.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import sys
import threading
import time

import pytest


@pytest.fixture
def timing(monkeypatch):
    """``spacr.qt.timing`` re-imported with ``SPACR_TIMING=1``.

    Import attribution is left OFF by default: installing the loader wrapper
    is a global change and the tests that want it ask for it explicitly.
    """
    previous_environment = {
        name: os.environ.get(name)
        for name in ("SPACR_TIMING", "SPACR_TIMING_IMPORTS")
    }
    monkeypatch.setenv("SPACR_TIMING", "1")
    monkeypatch.setenv("SPACR_TIMING_IMPORTS", "0")
    saved = sys.modules.get("spacr.qt.timing")
    module = importlib.reload(importlib.import_module("spacr.qt.timing"))
    assert module.ENABLED is True, "the fixture did not actually enable it"
    try:
        yield module
    finally:
        # This fixture is finalized before its ``monkeypatch`` dependency.
        # Restore the two switches before reloading, or the module handed to
        # the rest of the process remains enabled after the environment is
        # restored a moment later.
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if saved is not None:
            sys.modules["spacr.qt.timing"] = saved
        importlib.reload(saved or module)


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------

def test_a_span_records_what_it_wrapped_and_how_long_it_took(timing):
    """The ordinary case, and the fields a report is built from."""
    with timing.span("build mask", "cellpose"):
        time.sleep(0.01)

    recorded = [s for s in timing._SPANS if s["name"] == "build mask"]
    assert len(recorded) == 1
    assert recorded[0]["detail"] == "cellpose"
    assert recorded[0]["took"] >= 0.009
    assert recorded[0]["failed"] == ""


def test_a_span_that_raised_is_still_recorded_and_says_so(timing):
    """A span that only appears on success hides the slow failures.

    The expensive thing about a failing step is that it was expensive; if the
    exception erased its span, the timeline would show a gap with nothing in
    it and the reader would go looking for a missing import.
    """
    with pytest.raises(ValueError):
        with timing.span("doomed"):
            raise ValueError("no")

    recorded = [s for s in timing._SPANS if s["name"] == "doomed"]
    assert len(recorded) == 1
    assert recorded[0]["failed"] == "ValueError"


def test_nested_spans_record_their_depth(timing):
    """The report indents by depth, so the nesting has to be real."""
    with timing.span("outer"):
        with timing.span("inner"):
            pass

    depths = {s["name"]: s["depth"] for s in timing._SPANS
              if s["name"] in ("outer", "inner")}
    assert depths["inner"] == depths["outer"] + 1


def test_a_span_on_another_thread_carries_that_thread_s_name(timing):
    """The preloader competing with a click is the case this exists for.

    A span with no thread on it cannot answer "was the main thread doing
    this?", which is the whole question when a click feels slow.
    """
    def work():
        with timing.span("preload"):
            pass

    thread = threading.Thread(target=work, name="preloader")
    thread.start()
    thread.join()

    recorded = [s for s in timing._SPANS if s["name"] == "preload"]
    assert recorded and recorded[0]["thread"] == "preloader"


def test_depth_is_per_thread_not_shared(timing):
    """``_DEPTH`` is a threading.local, and the indentation depends on it.

    Were it shared, a background span opening while the main thread is three
    deep would be reported as a child of work it has nothing to do with.
    """
    seen = {}

    def work():
        with timing.span("background"):
            seen["depth"] = timing._depth()

    with timing.span("foreground"):
        with timing.span("deeper"):
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()

    assert seen["depth"] == 1


# ---------------------------------------------------------------------------
# marks and the interaction clock
# ---------------------------------------------------------------------------

def test_a_mark_records_an_instant_with_its_detail(timing):
    timing.mark("window shown", "home")

    recorded = [m for m in timing._MARKS if m["name"] == "window shown"]
    assert len(recorded) == 1
    assert recorded[0]["detail"] == "home"
    assert recorded[0]["at"] >= 0.0


def test_an_interaction_start_also_leaves_a_mark_saying_it_was_requested(
        timing):
    """The pair is what makes a click measurable: requested, then ready."""
    started = timing.interval_started("open regression", "from the sidebar")

    assert isinstance(started, float)
    assert any(m["name"] == "open regression requested"
               for m in timing._MARKS)


def test_the_clock_origin_and_elapsed_agree(timing):
    assert timing.process_started_at() == timing._START
    assert timing.elapsed() >= 0.0


# ---------------------------------------------------------------------------
# the stall watchdog's arithmetic
# ---------------------------------------------------------------------------

def test_a_stall_is_charged_to_a_click_only_for_the_part_that_overlaps(timing):
    """The clipping, and the number it replaces.

    A watchdog beat can begin before a click and end after it. Charging the
    whole gap to the click reports a multi-second freeze for an interaction
    that lasted a few hundred milliseconds -- which is how a timing report
    starts accusing the wrong code. ``late_ms`` keeps the raw gap and
    ``overlap_ms`` carries the honest in-window part.
    """
    stalls = [{"at": 10.0, "started_at": 8.0, "late_ms": 2000.0}]

    overlapping = timing.stalls_between(9.5, 10.5, stalls)

    assert len(overlapping) == 1
    assert overlapping[0]["late_ms"] == 2000.0
    assert overlapping[0]["overlap_ms"] == pytest.approx(500.0)


def test_a_stall_that_missed_the_window_is_not_charged_at_all(timing):
    stalls = [{"at": 3.0, "started_at": 2.0, "late_ms": 1000.0}]

    assert timing.stalls_between(9.5, 10.5, stalls) == []


def test_an_interval_that_ends_before_it_starts_has_no_stalls(timing):
    """Two clocks read in the wrong order must not produce a negative window."""
    assert timing.stalls_between(10.0, 9.0, [{"at": 9.5, "late_ms": 100.0}]) == []


def test_a_stall_missing_its_start_has_one_inferred_from_its_length(timing):
    """Older traces carry ``late_ms`` and no ``started_at``.

    Reading such a row as a zero-length gap at ``at`` would silently drop it
    from every overlap, so the start is reconstructed from the duration.
    """
    stalls = [{"at": 10.0, "late_ms": 1000.0}]

    overlapping = timing.stalls_between(9.0, 11.0, stalls)

    assert len(overlapping) == 1
    assert overlapping[0]["started_at"] == pytest.approx(9.0)


def test_the_charged_overlap_never_exceeds_the_gap_itself(timing):
    """A window wider than the stall is charged the stall, not the window."""
    stalls = [{"at": 10.0, "started_at": 9.9, "late_ms": 100.0}]

    overlapping = timing.stalls_between(0.0, 100.0, stalls)

    assert overlapping[0]["overlap_ms"] == pytest.approx(100.0)


def test_with_no_watchdog_running_there_is_no_last_beat(timing):
    """``None`` is "nobody is watching", which a caller must not read as 0."""
    timing._GUI_WATCHDOG_ACTIVE = False

    assert timing.last_gui_beat_at() is None


def test_a_running_watchdog_reports_its_beat_on_the_report_clock(timing):
    timing._GUI_WATCHDOG_ACTIVE = True
    timing._LAST_GUI_BEAT_AT = timing._START + 2.5

    assert timing.last_gui_beat_at() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_the_report_says_the_interface_stayed_answerable_when_it_did(timing):
    """The empty case is a sentence, not a blank section.

    A report whose stall section is empty reads as "the instrument was not
    running". Saying so explicitly is the difference between evidence of
    absence and absence of evidence.
    """
    timing._STALLS.clear()

    text = timing.report()

    assert "the interface stayed answerable" in text


def test_the_report_ranks_the_stalls_it_found_worst_first(timing):
    """A reader looking for the freeze should not have to sort by eye."""
    timing._STALLS.clear()
    timing._STALLS.extend([
        {"at": 1.0, "started_at": 0.9, "late_ms": 90.0, "source": "w"},
        {"at": 2.0, "started_at": 1.0, "late_ms": 900.0, "source": "w"},
    ])

    text = timing.report()

    assert "2 stalls" in text
    assert text.index("900 ms") < text.index("90 ms")


def test_the_report_carries_every_kind_of_entry_it_collected(timing):
    """Spans, imports, marks and readiness all reach the page."""
    timing._SPANS.clear()
    timing._IMPORTS.clear()
    timing._MARKS.clear()
    timing._READINESS.clear()
    timing._SPANS.append({"at": 0.5, "took": 0.25, "name": "build",
                          "detail": "masks", "depth": 0, "failed": "",
                          "thread": "MainThread"})
    timing._IMPORTS.append({"at": 0.1, "took": 0.4, "name": "torch",
                            "by": "qt/app.py:12", "thread": "preloader"})
    timing._MARKS.append({"at": 0.2, "name": "window shown", "detail": "home",
                          "thread": "MainThread"})
    timing._READINESS.append({
        "at": 1.0, "duration_s": 0.5, "name": "home", "detail": "first paint",
        "painted_usable_controls": 7, "budget_s": 5.0, "within_budget": True})

    text = timing.report()

    assert "build" in text and "[masks]" in text
    assert "torch" in text and "asked by qt/app.py:12" in text
    assert "<preloader>" in text
    assert "window shown" in text
    assert "painted control(s)" in text and "OK" in text


def test_a_readiness_entry_over_its_budget_says_so_in_words(timing):
    """The budget is the release contract; a number alone hides the verdict."""
    timing._READINESS.clear()
    timing._READINESS.append({
        "at": 9.0, "duration_s": 7.5, "name": "regression",
        "detail": "first paint", "painted_usable_controls": 3,
        "budget_s": 5.0, "within_budget": False})

    assert "OVER 5.0s BUDGET" in timing.report()


def test_a_span_that_raised_is_named_in_the_report(timing):
    timing._SPANS.clear()
    timing._SPANS.append({"at": 0.1, "took": 0.2, "name": "doomed",
                          "detail": "", "depth": 0, "failed": "ValueError",
                          "thread": "MainThread"})

    assert "RAISED ValueError" in timing.report()


def test_the_report_says_so_when_nothing_reported_readiness(timing):
    timing._READINESS.clear()

    assert "none observed" in timing.report()


# ---------------------------------------------------------------------------
# writing it out
# ---------------------------------------------------------------------------

def test_the_json_snapshot_round_trips_through_a_file(timing, tmp_path):
    target = tmp_path / "timing.json"

    written = timing.write_json(str(target))

    assert written == str(target)
    loaded = json.loads(target.read_text())
    assert isinstance(loaded, dict)


def test_an_unwritable_json_path_returns_empty_rather_than_raising(timing,
                                                                   tmp_path):
    """A diagnostic that takes the run down with it is worse than no report."""
    assert timing.write_json(str(tmp_path / "no" / "such" / "dir" / "t.json")) == ""


def test_no_path_at_all_writes_no_json(timing):
    assert timing.write_json("") == ""


def test_the_text_report_goes_where_the_environment_says(timing, tmp_path,
                                                         monkeypatch):
    """``SPACR_TIMING_LOG`` is how a user gets the file somewhere they can find."""
    target = tmp_path / "from-the-environment.log"
    monkeypatch.setenv("SPACR_TIMING_LOG", str(target))

    written = timing.write_report()

    assert written == str(target)
    assert "spaCR timing report" in target.read_text()


def test_an_explicit_path_beats_the_environment(timing, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_TIMING_LOG", str(tmp_path / "ignored.log"))
    target = tmp_path / "explicit.log"

    assert timing.write_report(str(target)) == str(target)
    assert not (tmp_path / "ignored.log").exists()


def test_an_unwritable_report_path_returns_empty_rather_than_raising(
        timing, tmp_path):
    assert timing.write_report(str(tmp_path / "no" / "such" / "t.log")) == ""


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def test_peak_memory_comes_back_as_megabytes(timing):
    """A plausible figure, not merely a number.

    The unit is the whole point: getrusage answers KiB on Linux and bytes on
    macOS, and a report that mixed them would be wrong by a factor of 1024 in
    exactly one place.
    """
    peak = timing._peak_rss_mb()

    assert peak is not None
    assert 1.0 < peak < 1_000_000.0


def test_a_platform_without_getrusage_falls_back_to_psutil(timing,
                                                           monkeypatch):
    """The fallback exists for Windows, which has no ``resource`` module."""
    real_import = __import__

    def no_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("no resource module here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_resource)

    peak = timing._peak_rss_mb()

    assert peak is None or peak > 0.0


def test_neither_source_available_reports_nothing_rather_than_zero(
        timing, monkeypatch):
    """``None`` means "not measured"; 0.0 would be a claim about the process."""
    real_import = __import__

    def nothing(name, *args, **kwargs):
        if name in ("resource", "psutil"):
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", nothing)

    assert timing._peak_rss_mb() is None


# ---------------------------------------------------------------------------
# switched off, which is how it ships
# ---------------------------------------------------------------------------

def test_with_timing_off_nothing_is_recorded_and_nothing_is_written(
        monkeypatch, tmp_path):
    """The shipped path: a few attribute lookups and no allocation.

    Every recorder returning early is what lets these calls stay in hot paths,
    so "off" has to mean off -- not "records into a list nobody reads".
    """
    monkeypatch.delenv("SPACR_TIMING", raising=False)
    module = importlib.reload(importlib.import_module("spacr.qt.timing"))
    try:
        assert module.ENABLED is False

        before = len(module._SPANS), len(module._MARKS)
        module.mark("ignored")
        with module.span("ignored"):
            pass

        assert (len(module._SPANS), len(module._MARKS)) == before
        assert module.interval_started("ignored") is None
        assert module.process_started_at() is None
        assert module.write_report(str(tmp_path / "x.log")) == ""
        assert module.write_json(str(tmp_path / "x.json")) == ""
        assert module.watch_the_gui_thread() is None
        assert not (tmp_path / "x.log").exists()
    finally:
        importlib.reload(module)


def test_a_span_still_lets_an_exception_through_when_timing_is_off(monkeypatch):
    """The disabled path has its own yield, and it must not swallow anything."""
    monkeypatch.delenv("SPACR_TIMING", raising=False)
    module = importlib.reload(importlib.import_module("spacr.qt.timing"))
    try:
        with pytest.raises(KeyError):
            with module.span("ignored"):
                raise KeyError("through")
    finally:
        importlib.reload(module)
