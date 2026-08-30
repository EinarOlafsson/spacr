"""What the process tree cost, recorded without perturbing the run.

The load-bearing test is the first one. A child that allocates a KNOWN amount
must be seen by the tree reading and missed by a parent-only reading, and both
halves are asserted together: the second is the bug `spacr.resource_log`
exists to fix, and a test that only checked the first would pass against the
reading spaCR already had.

The rest hold the properties that make a record usable after a failure -- that
``off`` leaves no thread behind, that a child dying mid-run leaves a record
ending at its death rather than an exception or a corrupt file, that the file
names the measure it used and survives a truncated last line, that a figure
the platform cannot supply is recorded as unavailable and never as a zero, and
that sampling installs no profile hook, which is the whole reason this is not
part of verbose logging.

Nothing here waits and hopes. The sampler's reading is driven by calling it,
the clock is passed in, and every wait is on an event with a generous ceiling,
so the results do not depend on how fast the machine is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import types

import psutil
import pytest

from spacr import fit_resources, resource_log

MEGABYTE = 1024 * 1024

#: Big enough that no ordinary interpreter noise reaches it, small enough to
#: allocate on any machine that can run the suite.
KNOWN_ALLOCATION = 64 * MEGABYTE

#: A child that reports when it is up, allocates on command, and waits to be
#: told to go. ``os.urandom`` rather than a zero-filled buffer on purpose: a
#: calloc of fresh anonymous pages can stay unwritten and therefore unresident,
#: which would make the allocation invisible to exactly the measure under test.
CHILD_SOURCE = """
import os, sys
sys.stdout.write("ready\\n")
sys.stdout.flush()
sys.stdin.readline()
held = os.urandom({size})
sys.stdout.write("allocated %d\\n" % len(held))
sys.stdout.flush()
sys.stdin.readline()
"""


def _census():
    """The names of every live thread, so a sampler cannot hide in the count."""
    return sorted(thread.name for thread in threading.enumerate())


def _sampler_threads():
    """Every live sampler thread."""
    return [thread for thread in threading.enumerate()
            if thread.name == resource_log.THREAD_NAME]


def _rows(sample):
    """A sample's process rows, keyed by pid."""
    return {row["pid"]: row for row in sample["processes"]}


class Figures:
    """Whatever a platform's memory or CPU call handed back."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakeThread:
    """One thread of a fake process."""

    def __init__(self, ident=1, user_time=0.5, system_time=0.25):
        self.id = ident
        self.user_time = user_time
        self.system_time = system_time


class FakeProcess:
    """A stand-in for ``psutil.Process`` that fails exactly where asked.

    Real races -- a child exiting between being listed and being read -- cannot
    be provoked reliably from a test, so they are asked for by name through
    ``fail``, which maps a method to the exception it raises.
    """

    def __init__(self, pid, ppid=1, name="worker", full=None, info=None,
                 cpu=(1.5, 0.5), threads=(), children=(), fail=None):
        self.pid = pid
        self._ppid = ppid
        self._name = name
        self._full = full
        self._info = info
        self._cpu = cpu
        self._threads = list(threads)
        self._children = list(children)
        self._fail = dict(fail or {})

    def _check(self, key):
        error = self._fail.get(key)
        if error is not None:
            raise error

    def children(self, recursive=False):
        self._check("children")
        return list(self._children)

    def memory_full_info(self):
        self._check("memory_full_info")
        return self._full

    def memory_info(self):
        self._check("memory_info")
        return self._info

    def cpu_times(self):
        self._check("cpu_times")
        return Figures(user=self._cpu[0], system=self._cpu[1])

    def threads(self):
        self._check("threads")
        return list(self._threads)

    def name(self):
        self._check("name")
        return self._name

    def ppid(self):
        self._check("ppid")
        return self._ppid


def fake_psutil(monkeypatch, root):
    """Put a fake process tree behind the module's psutil lookup."""
    module = types.SimpleNamespace(
        Process=lambda: root,
        NoSuchProcess=psutil.NoSuchProcess,
        AccessDenied=psutil.AccessDenied,
    )
    monkeypatch.setattr(resource_log, "_psutil", lambda: module)
    return module


def no_preference(monkeypatch):
    """An install whose Qt preferences cannot be imported at all."""
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)


@pytest.fixture
def spawn_child():
    """Start short-lived children that are always reaped."""
    started = []

    def _start(size=0):
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD_SOURCE.format(size=size)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        started.append(child)
        assert child.stdout.readline().strip() == "ready"
        return child

    yield _start

    for child in started:
        child.kill()
        child.wait(timeout=60)
        child.stdin.close()
        child.stdout.close()


@pytest.mark.integration
def test_a_child_is_seen_by_the_tree_and_missed_by_a_parent_only_reading(
        spawn_child):
    """The whole reason the module exists, asserted from both sides."""
    child = spawn_child(KNOWN_ALLOCATION)
    before = resource_log.tree_sample("summary")
    parent_only_before = fit_resources.host_rss()

    child.stdin.write("go\n")
    child.stdin.flush()
    assert child.stdout.readline().split()[0] == "allocated"

    after = resource_log.tree_sample("summary")
    parent_only_after = fit_resources.host_rss()

    seen = _rows(after)
    assert child.pid in seen, "the tree reading must name the child"
    assert seen[child.pid]["memory"] >= 0.75 * KNOWN_ALLOCATION
    assert seen[child.pid]["ppid"] == os.getpid()
    assert after["measure"] in resource_log.MEASURES
    assert after["total"] - before["total"] >= 0.75 * KNOWN_ALLOCATION

    # The half that is the bug: every reading spaCR took before this module
    # counted the calling process, and the calling process did not grow.
    mine = _rows(before)[os.getpid()]["memory"]
    assert seen[os.getpid()]["memory"] - mine < 0.25 * KNOWN_ALLOCATION
    assert parent_only_after - parent_only_before < 0.25 * KNOWN_ALLOCATION


@pytest.mark.integration
def test_a_child_that_dies_leaves_a_record_ending_at_its_death(
        tmp_path, spawn_child):
    """A killed child ends the record; it does not raise and does not corrupt."""
    path = tmp_path / "resources.jsonl"
    child = spawn_child()
    sampler = resource_log.ResourceSampler(path=path, level="summary",
                                           interval=60.0)

    alive = sampler.sample_once()
    assert child.pid in _rows(alive)

    child.kill()
    child.wait(timeout=60)
    dead = sampler.sample_once()
    assert child.pid not in _rows(dead)
    assert dead["total"] is not None
    assert sampler.stop() is True

    log = resource_log.read_log(path)
    assert log["unreadable"] == 0
    assert len(log["samples"]) == 2
    assert child.pid in _rows(log["samples"][0])
    assert child.pid not in _rows(log["samples"][1])


def test_off_starts_no_sampler_thread_and_writes_no_file(tmp_path):
    """The thread census before and after is the assertion the item asks for."""
    path = tmp_path / "resources.jsonl"
    before = _census()

    sampler = resource_log.ResourceSampler(path=path, level="off")
    assert sampler.level_source == "argument"
    assert sampler.start() is False
    assert sampler.sample_once() is None
    assert sampler.is_running() is False
    assert _census() == before
    assert not path.exists()

    assert sampler.stop() is True
    assert sampler.samples() == []
    assert sampler.summary() == {}
    assert sampler.describe() == ""
    assert _census() == before


def test_a_sampler_thread_is_a_daemon_and_is_gone_after_stop(tmp_path):
    """It cannot hold the process open, and it is never the GUI thread."""
    path = tmp_path / "resources.jsonl"
    before = _census()
    sampler = resource_log.ResourceSampler(path=path, level="summary",
                                           interval=60.0)

    assert sampler.start() is True
    assert sampler.start() is True, "starting twice must not make two threads"
    running = _sampler_threads()
    assert len(running) == 1
    assert running[0].daemon is True
    assert running[0] is not threading.main_thread()
    assert sampler.is_running() is True

    assert sampler.stop() is True
    assert _sampler_threads() == []
    assert _census() == before
    assert sampler.is_running() is False

    # The loop takes its first reading before it waits, so a run killed after
    # half a second still leaves one.
    log = resource_log.read_log(path)
    assert log["header"]["level"] == "summary"
    assert len(log["samples"]) == 1


def test_sampling_installs_no_profile_hook(tmp_path):
    """Verbose logging's tracer costs twenty times the startup. This does not."""
    profile_before = sys.getprofile()
    trace_before = sys.gettrace()

    with resource_log.ResourceSampler(path=tmp_path / "resources.jsonl",
                                      level="detailed",
                                      interval=60.0) as sampler:
        assert sampler.is_running() is True
        assert sys.getprofile() is profile_before
        assert sys.gettrace() is trace_before

    assert sampler.is_running() is False
    assert sys.getprofile() is profile_before
    assert sys.gettrace() is trace_before


def test_the_written_file_names_the_measure_and_survives_truncation(tmp_path):
    """A kill leaves a truncated last line, which is not a corrupt file."""
    path = tmp_path / "resources.jsonl"
    ticks = iter([100.0, 101.0, 102.0])
    sampler = resource_log.ResourceSampler(path=path, level="summary",
                                           interval=60.0, label="trial-7",
                                           clock=lambda: next(ticks))
    sampler.sample_once()
    sampler.sample_once()
    assert sampler.stop() is True

    whole = resource_log.read_log(path)
    assert whole["header"]["measure"] in resource_log.MEASURES
    assert whole["header"]["measure"] == sampler.measure
    assert whole["header"]["unit"] == "bytes"
    assert whole["header"]["label"] == "trial-7"
    assert whole["header"]["level_source"] == "argument"
    assert whole["header"]["started"] == 100.0
    assert whole["unreadable"] == 0
    assert [sample["time"] for sample in whole["samples"]] == [101.0, 102.0]

    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-40], encoding="utf-8")
    killed = resource_log.read_log(path)
    assert killed["header"]["measure"] == whole["header"]["measure"]
    assert [sample["time"] for sample in killed["samples"]] == [101.0]
    assert killed["unreadable"] == 1


def test_per_thread_figures_appear_under_detailed_and_not_under_summary():
    """Per-thread rows are what ``detailed`` buys, and all it buys here."""
    mine = os.getpid()
    summary = _rows(resource_log.tree_sample("summary"))[mine]
    detailed = _rows(resource_log.tree_sample("detailed"))[mine]

    assert "threads" not in summary
    assert detailed["threads"] is not None
    idents = {row["thread_id"] for row in detailed["threads"]}
    assert threading.get_native_id() in idents
    assert all(row["cpu_user"] is not None for row in detailed["threads"])
    # A CUDA context belongs to a process, so no thread row claims GPU memory.
    assert all("gpu" not in row for row in detailed["threads"])


def test_a_figure_the_platform_cannot_supply_is_unavailable_not_zero(
        monkeypatch):
    """A zero would read as "this thread was free", which is a lie."""
    silent = FakeThread(ident=11, user_time=None, system_time=None)
    unnamed = FakeThread(ident="not-a-tid", user_time=2.0, system_time=1.0)
    child = FakeProcess(202, full=Figures(uss=2 * MEGABYTE),
                        threads=[silent, unnamed],
                        fail={"cpu_times": OSError("no CPU times here")})
    root = FakeProcess(201, full=Figures(uss=MEGABYTE),
                       threads=[FakeThread(ident=201)],
                       fail={"threads": NotImplementedError("no thread times"),
                             "cpu_times": OSError("no CPU times here")},
                       children=[child])
    fake_psutil(monkeypatch, root)

    rows = _rows(resource_log.tree_sample("detailed", now=5.0))
    assert rows[201]["threads"] is None, "unavailable, not an empty list"
    assert rows[202]["cpu_user"] is None and rows[202]["cpu_system"] is None
    assert rows[202]["threads"][0] == {"thread_id": 11, "cpu_user": None,
                                       "cpu_system": None}
    assert rows[202]["threads"][1]["thread_id"] is None
    assert rows[202]["threads"][1]["cpu_user"] == 2.0

    whole = resource_log.summarise([resource_log.tree_sample("detailed")])
    assert "cpu_seconds" not in whole, "no CPU figure is not a CPU figure of 0"


def test_a_child_that_vanishes_or_refuses_is_counted_not_raised(monkeypatch):
    """Three ways to lose one child, none of which may lose the other two."""
    gone = FakeProcess(302, fail={"memory_full_info": psutil.NoSuchProcess(302)})
    shut = FakeProcess(303, fail={"memory_full_info": psutil.AccessDenied(303),
                                  "memory_info": psutil.AccessDenied(303)})
    odd = FakeProcess(304, fail={"memory_full_info": psutil.AccessDenied(304),
                                 "memory_info": ValueError("nonsense")})
    kept = FakeProcess(305, full=Figures(uss=3 * MEGABYTE))
    root = FakeProcess(301, full=Figures(uss=MEGABYTE),
                       children=[gone, shut, odd, kept])
    fake_psutil(monkeypatch, root)

    sample = resource_log.tree_sample("summary", now=9.0)
    assert sorted(_rows(sample)) == [301, 305]
    assert sample["missed"] == 3
    assert sample["total"] == 4 * MEGABYTE
    assert "3 readings missed" in resource_log.describe([sample])


def test_a_tree_that_cannot_be_enumerated_reads_as_unmeasured(monkeypatch):
    """No tree is not an empty tree."""
    root = FakeProcess(401, full=Figures(uss=MEGABYTE),
                       fail={"children": psutil.AccessDenied(401)})
    fake_psutil(monkeypatch, root)

    sample = resource_log.tree_sample("summary", now=1.0)
    assert sample["total"] is None
    assert sample["measure"] is None
    assert sample["processes"] == []
    assert sample["missed"] == 0
    assert resource_log.describe([sample]).splitlines()[0].endswith(
        "measured as not measured")


def test_without_psutil_nothing_is_measured_and_nothing_raises(monkeypatch):
    """An install with no psutil records that it measured nothing."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert resource_log._psutil() is None
    assert resource_log.preferred_measure() is None

    sample = resource_log.tree_sample("detailed", now=2.0)
    assert sample == {"record": "sample", "time": 2.0, "level": "detailed",
                      "measure": None, "unit": "bytes", "total": None,
                      "processes": [], "missed": 0}


def test_the_measure_falls_back_from_uss_to_pss_to_rss(monkeypatch):
    """The record says which definition it used, because they differ."""
    private = FakeProcess(501, full=Figures(uss=MEGABYTE, pss=2 * MEGABYTE,
                                            rss=4 * MEGABYTE))
    shared = FakeProcess(502, full=Figures(pss=2 * MEGABYTE,
                                           rss=4 * MEGABYTE))
    resident = FakeProcess(503, info=Figures(rss=4 * MEGABYTE),
                           fail={"memory_full_info": psutil.AccessDenied(503)})
    nothing = FakeProcess(504, info=Figures(),
                          fail={"memory_full_info": psutil.AccessDenied(504)})
    root = FakeProcess(500, full=Figures(uss=MEGABYTE),
                       children=[private, shared, resident, nothing])
    fake_psutil(monkeypatch, root)

    assert resource_log.preferred_measure(private) == "uss"
    assert resource_log.preferred_measure(shared) == "pss"
    assert resource_log.preferred_measure(resident) == "rss"
    assert resource_log.preferred_measure(nothing) is None
    assert resource_log.preferred_measure() == "uss"

    sample = resource_log.tree_sample("summary", process=root, now=3.0)
    rows = _rows(sample)
    assert rows[501]["measure"] == "uss" and rows[501]["memory"] == MEGABYTE
    assert rows[502]["measure"] == "pss"
    assert rows[503]["measure"] == "rss"
    assert rows[504]["measure"] is None and rows[504]["memory"] is None
    # A total is only as comparable as its weakest member.
    assert sample["measure"] == "rss"
    assert sample["total"] == 8 * MEGABYTE


def test_a_label_that_cannot_be_read_does_not_lose_the_figure_beside_it(
        monkeypatch):
    """Names are labels; memory is the measurement. They fail separately."""
    root = FakeProcess(601, full=Figures(uss=MEGABYTE),
                       fail={"name": psutil.AccessDenied(601),
                             "ppid": psutil.NoSuchProcess(601)})
    fake_psutil(monkeypatch, root)

    row = _rows(resource_log.tree_sample("summary", now=4.0))[601]
    assert row["name"] is None and row["ppid"] is None
    assert row["memory"] == MEGABYTE
    assert "unnamed" in resource_log.describe(
        [resource_log.tree_sample("summary")])


def test_a_measure_probe_that_raises_reports_nothing(monkeypatch):
    """The probe is a measurement too, and may fail like one."""
    module = types.SimpleNamespace(
        Process=lambda: (_ for _ in ()).throw(psutil.NoSuchProcess(1)),
        NoSuchProcess=psutil.NoSuchProcess,
        AccessDenied=psutil.AccessDenied,
    )
    monkeypatch.setattr(resource_log, "_psutil", lambda: module)
    assert resource_log.preferred_measure() is None


def test_the_environment_variable_overrides_the_default_and_says_so(
        monkeypatch):
    """A support request can tell a chosen level from an assumed one."""
    monkeypatch.delenv(resource_log.ENV_VAR, raising=False)
    no_preference(monkeypatch)
    assert resource_log.resolve_level() == "summary"
    assert resource_log.DEFAULT_LEVEL == "summary"
    assert resource_log.level_source() == "default"

    monkeypatch.setenv(resource_log.ENV_VAR, "  Detailed \n")
    assert resource_log.resolve_level() == "detailed"
    assert resource_log.level_source() == "environment"

    monkeypatch.setenv(resource_log.ENV_VAR, "loud")
    assert resource_log.resolve_level() == "summary"
    assert resource_log.level_source() == "default"

    assert resource_log.level_source("off") == "argument"
    assert resource_log.resolve_level("off") == "off"


def test_the_preference_decides_when_the_environment_is_silent(monkeypatch):
    """The GUI can drive it, and a missing preference is not an error."""
    monkeypatch.delenv(resource_log.ENV_VAR, raising=False)
    stub = types.ModuleType("spacr.qt.preferences")
    stub.get_performance_logging = lambda: "detailed"
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", stub)
    assert resource_log.resolve_level() == "detailed"
    assert resource_log.level_source() == "preference"

    stub.get_performance_logging = lambda: object()
    assert resource_log.level_source() == "default"

    def unreadable():
        raise RuntimeError("no settings backend on this machine")

    stub.get_performance_logging = unreadable
    assert resource_log.resolve_level() == "summary"
    assert resource_log.level_source() == "default"

    del stub.get_performance_logging
    assert resource_log.level_source() == "default"

    stub.get_performance_logging = lambda: "off"
    monkeypatch.setenv(resource_log.ENV_VAR, "detailed")
    assert resource_log.resolve_level() == "detailed"
    assert resource_log.level_source() == "environment"


def test_an_unknown_level_is_refused_rather_than_guessed():
    """A caller holding the setting in its hand gets told, not corrected."""
    with pytest.raises(ValueError, match="Unknown performance-logging level"):
        resource_log.resolve_level("verbose")
    with pytest.raises(ValueError, match="off"):
        resource_log.ResourceSampler(level=17)


def test_a_summary_of_nothing_is_empty_not_zero():
    """"Nobody measured" and "nothing was used" are opposite findings."""
    assert resource_log.summarise([]) == {}
    assert resource_log.summarise(["not a sample", 5]) == {}
    assert resource_log.describe([]) == ""


def test_the_summary_names_the_pid_that_held_the_peak():
    """"Which trial was large" is the question the record has to answer."""
    samples = [
        {"record": "sample", "time": 1.0, "measure": "uss",
         "total": 40 * MEGABYTE, "missed": 0, "processes": [
             {"pid": 1, "name": "spacr", "memory": 10 * MEGABYTE,
              "measure": "uss", "cpu_user": 1.0, "cpu_system": 0.5},
             {"pid": 2, "name": "trial-a", "memory": 30 * MEGABYTE,
              "measure": "uss", "cpu_user": 2.0, "cpu_system": None}]},
        {"record": "sample", "time": 2.0, "measure": "rss",
         "total": 100 * MEGABYTE, "missed": 2, "processes": [
             {"pid": 1, "name": "spacr", "memory": 10 * MEGABYTE,
              "measure": "rss", "cpu_user": 1.0, "cpu_system": 0.5},
             {"pid": 3, "name": "trial-b", "memory": 90 * MEGABYTE,
              "measure": "rss", "cpu_user": 8.0, "cpu_system": 1.5}]},
    ]

    high = resource_log.summarise(samples)
    assert high["samples"] == 2
    assert high["pids"] == [1, 2, 3]
    assert high["measure"] == "rss"
    assert high["missed"] == 2
    assert high["peak_total"] == 100 * MEGABYTE
    assert high["peak_total_time"] == 2.0
    assert high["peak_process"] == {"pid": 3, "name": "trial-b",
                                    "memory": 90 * MEGABYTE,
                                    "measure": "rss", "time": 2.0}
    assert high["cpu_seconds"] == 11.0

    text = resource_log.describe(samples)
    assert "PEAK tree     100.0 MB" in text
    assert "pid 3 (trial-b)" in text
    assert "CPU           11.0 s" in text
    assert "2 readings missed" in text


def test_a_summary_of_samples_that_measured_nothing_keeps_its_shape():
    """Empty peaks are omitted rather than spelled as zero."""
    samples = [{"record": "sample", "time": 1.0, "measure": None,
                "total": None, "missed": "not a number"}]
    high = resource_log.summarise(samples)
    assert high == {"samples": 1, "measure": None, "missed": 0, "pids": []}
    assert resource_log.describe(samples) == (
        "  performance log: 1 sample over 0 processes, "
        "measured as not measured")


def test_reading_a_log_that_is_not_there_or_not_json(tmp_path):
    """A missing file reads empty; a junk line is counted, not raised."""
    assert resource_log.read_log(tmp_path / "absent.jsonl") == {
        "header": {}, "samples": [], "unreadable": 0}

    path = tmp_path / "messy.jsonl"
    path.write_text("\n".join([
        json.dumps({"record": "header", "measure": "uss"}),
        "",
        "5",
        "{oh dear",
        json.dumps({"record": "sample", "time": 1.0}),
    ]), encoding="utf-8")
    log = resource_log.read_log(path)
    assert log["header"]["measure"] == "uss"
    assert [sample["time"] for sample in log["samples"]] == [1.0]
    assert log["unreadable"] == 2


def test_the_ring_buffer_is_bounded_and_the_interval_has_a_floor():
    """A week-long run keeps the last samples rather than growing forever."""
    sampler = resource_log.ResourceSampler(level="summary", capacity=2,
                                           interval=0.0)
    assert sampler.capacity == 2
    assert sampler.interval == resource_log.MIN_INTERVAL_SECONDS
    ticks = iter([1.0, 2.0, 3.0, 4.0])
    sampler._clock = lambda: next(ticks)

    for _ in range(4):
        sampler.sample_once()
    assert [sample["time"] for sample in sampler.samples()] == [3.0, 4.0]
    assert sampler.summary()["samples"] == 2
    assert resource_log.ResourceSampler(capacity=0).capacity == 1


def test_a_log_that_cannot_be_opened_or_written_does_not_stop_sampling(
        tmp_path):
    """The measurement matters more than the file it was going to go in."""
    unwritable = tmp_path / "a-directory-not-a-file"
    unwritable.mkdir()
    blocked = resource_log.ResourceSampler(path=unwritable, level="summary")
    assert blocked.sample_once() is not None
    assert blocked.stop() is True

    path = tmp_path / "resources.jsonl"
    sampler = resource_log.ResourceSampler(path=path, level="summary")
    assert sampler.sample_once() is not None
    sampler._handle.close()          # the file goes away under the sampler
    assert sampler.sample_once() is not None
    assert len(sampler.samples()) == 2

    sampler._handle = types.SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(OSError("gone")))
    assert sampler.stop() is True


def test_a_stopped_sampler_never_reopens_and_truncates_its_record(tmp_path):
    """The record of a finished run is not overwritten by a late reading."""
    path = tmp_path / "resources.jsonl"
    sampler = resource_log.ResourceSampler(path=path, level="summary",
                                           interval=60.0)
    sampler.sample_once()
    assert sampler.stop() is True
    written = path.read_text(encoding="utf-8")

    sampler.sample_once()
    assert path.read_text(encoding="utf-8") == written
    assert len(resource_log.read_log(path)["samples"]) == 1


def test_the_loop_survives_a_reading_that_raises(monkeypatch):
    """A sampler that died on a bad reading would stop recording exactly when
    the run started going wrong."""
    sampler = resource_log.ResourceSampler(level="summary", interval=0.001)
    calls = []
    tried_again = threading.Event()

    def unreliable():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("a child vanished between two syscalls")
        tried_again.set()
        return None

    monkeypatch.setattr(sampler, "sample_once", unreliable)
    assert sampler.start() is True
    assert tried_again.wait(30) is True
    assert sampler.stop() is True
    assert len(calls) >= 2


def test_stop_reports_a_thread_that_will_not_end(monkeypatch):
    """A sampler that outlives its timeout says so rather than claiming to
    have stopped."""
    sampler = resource_log.ResourceSampler(level="summary", interval=0.01)
    entered = threading.Event()
    release = threading.Event()

    def blocking():
        entered.set()
        release.wait(60)
        return None

    monkeypatch.setattr(sampler, "sample_once", blocking)
    assert sampler.start() is True
    thread = sampler._thread
    assert entered.wait(30) is True
    assert sampler.stop(timeout=0.05) is False

    release.set()
    thread.join(30)
    assert thread.is_alive() is False


def test_the_context_manager_stops_the_sampler_when_the_block_raises(tmp_path):
    """An exception in the measured work must not leave a thread behind."""
    before = _census()
    sampler = resource_log.ResourceSampler(path=tmp_path / "resources.jsonl",
                                           level="summary", interval=60.0)
    with pytest.raises(ZeroDivisionError):
        with sampler:
            assert sampler.is_running() is True
            raise ZeroDivisionError("the run failed")
    assert sampler.is_running() is False
    assert _census() == before
    assert resource_log.read_log(tmp_path / "resources.jsonl")["samples"]
