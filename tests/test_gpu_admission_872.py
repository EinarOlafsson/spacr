"""GPU admission for long OPS runs, on a faked device (item 872, B11 and B19).

No GPU is needed: the device query, the sleep and the clock are replaced by
fakes. The policy must wait while a running compute process above the
threshold holds the card, start after the asked-for number of idle samples
in a row, never wait for a browser's GPU process or a stopped process, log
START and FINISH, and stop cleanly on a STOP file.
"""
from __future__ import annotations

import subprocess

import pytest

from spacr import gpu_admission as ga
from spacr import ops_engine

BROWSER = ga.GpuProcess(11, 600, "chrome", "/opt/chrome --type=gpu-process", False)
COMPUTE = ga.GpuProcess(22, 600, "python", "python train.py", False)
STOPPED = ga.GpuProcess(33, 9000, "python", "python paused.py", True)


def _sample(*processes, utilisation=0.0, used=1024.0):
    """One faked look at the card."""
    return ga.GpuSample([utilisation], [used], list(processes))


class _Clock:
    """A clock the fake sleep advances."""

    def __init__(self):
        """Start at zero."""
        self.now = 1_000_000.0
        self.sleeps = 0

    def sleep(self, seconds):
        """Advance the clock."""
        self.now += seconds
        self.sleeps += 1

    def time(self):
        """The current fake time."""
        return self.now


def _policy(samples, tmp_path, **kwargs):
    """A policy fed ``samples`` in turn, the last one repeated."""
    clock = _Clock()
    queue = list(samples)

    def query():
        """The next faked sample."""
        return queue.pop(0) if len(queue) > 1 else queue[0]

    policy = ga.GpuAdmission("test", query=query, sleep=clock.sleep,
                             clock=clock.time, log_path=str(tmp_path / "log"),
                             **kwargs)
    return policy, clock


def test_admission_waits_while_a_compute_process_holds_the_card(tmp_path):
    samples = [_sample(COMPUTE)] * 3 + [_sample()] * 6
    policy, clock = _policy(samples, tmp_path, idle_checks=6)
    result = policy.wait()
    assert result["admitted"] and not result["stopped"]
    assert result["samples"] == 9
    assert len(policy.waits) == 3 and "22(python, 600 MiB)" in policy.waits[0]
    assert result["waited_s"] == pytest.approx(8 * 10.0)


def test_a_busy_sample_restarts_the_idle_count(tmp_path):
    samples = [_sample()] * 5 + [_sample(COMPUTE)] + [_sample()] * 6
    policy, _clock = _policy(samples, tmp_path, idle_checks=6)
    assert policy.wait()["samples"] == 12


def test_a_browser_gpu_process_never_blocks_but_a_compute_one_does(tmp_path):
    assert ga.gpu_idle(_sample(BROWSER)) == (True, "idle")
    assert not ga.gpu_idle(_sample(COMPUTE))[0]
    assert ga.gpu_idle(_sample(STOPPED))[0]
    policy, _clock = _policy([_sample(BROWSER)], tmp_path, idle_checks=6)
    assert policy.wait() == {"admitted": True, "stopped": False,
                             "timed_out": False, "samples": 6,
                             "waited_s": 50.0}
    small = ga.GpuProcess(44, 400, "python", "python small.py")
    assert ga.gpu_idle(_sample(small))[0]
    assert not ga.gpu_idle(_sample(small), min_process_mib=300)[0]


def test_utilisation_and_a_missing_device_keep_it_waiting():
    idle, reason = ga.gpu_idle(_sample(utilisation=35.0))
    assert not idle and reason == "utilisation 35%"
    assert ga.gpu_idle(_sample(utilisation=35.0),
                       require_zero_utilisation=False)[0]
    assert ga.gpu_idle(None) == (False, "no GPU answered (nvidia-smi missing "
                                        "or failed)")


def test_ignore_patterns_match_names_and_command_lines():
    code = ga.GpuProcess(1, 700, "code", "/usr/share/code/code --type=renderer")
    custom = ga.GpuProcess(2, 700, "napari", "python -m napari")
    assert ga.is_display_process(code)
    assert not ga.is_display_process(custom)
    assert ga.is_display_process(custom, ("napari",))
    assert ga.is_display_process(custom, ("-m napari",))
    assert ga.busy_processes(_sample(code, custom)) == [custom]


def test_a_stop_file_ends_the_wait_cleanly(tmp_path):
    stop = tmp_path / "STOP"
    policy, clock = _policy([_sample(COMPUTE)], tmp_path, stop_file=str(stop))
    original = clock.sleep

    def sleep(seconds):
        """Drop the STOP file after the third sample."""
        original(seconds)
        if clock.sleeps == 3:
            stop.write_text("")

    policy.sleep = sleep
    result = policy.wait()
    assert result["stopped"] and not result["admitted"]
    assert result["samples"] == 3


def test_the_wait_follows_a_release_line_first_and_can_time_out(tmp_path):
    events = tmp_path / "cnn.events"
    policy, clock = _policy([_sample()], tmp_path, idle_checks=2,
                            release_file=str(events),
                            release_line="CNN FINISHED -- GPU released")
    original = clock.sleep

    def sleep(seconds):
        """The other job releases after four sleeps."""
        original(seconds)
        if clock.sleeps == 4:
            events.write_text("x\nCNN FINISHED -- GPU released\n")

    policy.sleep = sleep
    result = policy.wait()
    assert result["admitted"] and result["samples"] == 2
    assert clock.sleeps == 5

    waiting, _clock = _policy([_sample()], tmp_path, release_file=str(
        tmp_path / "never"), release_line="done", max_wait=30)
    assert waiting.wait()["timed_out"]
    busy, _clock = _policy([_sample(COMPUTE)], tmp_path, max_wait=30)
    assert busy.wait() == {"admitted": False, "stopped": False,
                           "timed_out": True, "samples": 3, "waited_s": 30.0}
    stopped, _clock = _policy([_sample()], tmp_path, release_file=str(
        tmp_path / "never"), release_line="done", stop_file=str(events))
    assert stopped.wait()["stopped"]


def test_start_and_finish_are_logged(tmp_path):
    policy, _clock = _policy([_sample()], tmp_path)
    policy.start()
    policy.finish(0)
    policy.finish("ValueError")
    lines = (tmp_path / "log").read_text().splitlines()
    assert len(lines) == 3
    assert "[test] START pid" in lines[0]
    assert lines[1].endswith("[test] FINISH rc=0")
    assert lines[2].endswith("FINISH rc=ValueError")


def test_fewer_workers_above_the_vram_cap(tmp_path):
    policy, _clock = _policy([_sample(used=23 * 1024.0)], tmp_path,
                             vram_cap_gb=22)
    assert policy.workers_under_cap(8) == 4
    assert policy.workers_under_cap(1) == 1
    assert policy.workers_under_cap(8, _sample(used=20 * 1024.0)) == 8
    uncapped, _clock = _policy([_sample(used=23 * 1024.0)], tmp_path)
    assert uncapped.workers_under_cap(8) == 8
    silent, _clock = _policy([None], tmp_path, vram_cap_gb=22)
    assert silent.workers_under_cap(8) == 8


def test_query_gpu_parses_nvidia_smi(monkeypatch):
    replies = {
        "--query-gpu=utilization.gpu,memory.used": "0, 1200\n3, 800\n",
        "--query-compute-apps=pid,used_memory": "101, 600\n202, 5000 MiB\n\n",
    }

    def run(command, **kwargs):
        """Answer as nvidia-smi would."""
        return subprocess.CompletedProcess(command, 0, replies[command[1]], "")

    details = {101: ("chrome", "chrome --type=gpu-process", False),
               202: ("python", "python job.py", False)}
    sample = ga.query_gpu(run=run, details=details.__getitem__)
    assert sample.utilisation == [0.0, 3.0]
    assert sample.memory_used_mib == [1200.0, 800.0]
    assert [(p.pid, p.mib, p.executable) for p in sample.processes] == [
        (101, 600.0, "chrome"), (202, 5000.0, "python")]
    assert ga.busy_processes(sample) == [sample.processes[1]]

    def missing(command, **kwargs):
        """No nvidia-smi on this machine."""
        raise FileNotFoundError("nvidia-smi")

    assert ga.query_gpu(run=missing) is None
    failing = lambda command, **kwargs: subprocess.CompletedProcess(  # noqa: E731
        command, 9, "", "no devices")
    assert ga.query_gpu(run=failing) is None


def test_process_details_read_this_process():
    import os

    executable, command, stopped = ga._process_details(os.getpid())
    assert executable.startswith("python") and command
    assert stopped is False
    assert ga._process_details(2 ** 30) == ("", "", False)


def test_the_settings_build_the_policy():
    policy = ga.admission_from_settings(
        {"ops_gpu_idle_checks": 3, "ops_gpu_min_process_mib": 4096,
         "ops_gpu_vram_cap_gb": 22, "ops_gpu_ignore_process_patterns":
         "napari, -m my_viewer"}, "run", stop_file="/x/STOP")
    assert (policy.idle_checks, policy.min_process_mib, policy.vram_cap_gb) == (
        3, 4096.0, 22.0)
    assert policy.ignore_patterns[-2:] == ("napari", "-m my_viewer")
    assert policy.stop_file == "/x/STOP"
    default = ga.admission_from_settings({}, "run")
    assert (default.idle_checks, default.min_process_mib,
            default.vram_cap_gb) == (6, 500.0, None)


@pytest.fixture
def faked_card(monkeypatch, tmp_path):
    """The default device query answers with an idle card; sleeps are free."""
    monkeypatch.setattr(ga, "query_gpu", lambda: _sample(BROWSER))
    monkeypatch.setattr(ga.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(ga, "DEFAULT_LOG", str(tmp_path / "gpu" / "log"))
    return tmp_path


def test_run_ops_waits_for_the_card_and_logs_the_run(faked_card, monkeypatch):
    log = faked_card / "gpu" / "log"
    seen = []

    def admission(settings, label, *, stop_file=None, **overrides):
        """The settings' policy, logging to the test folder."""
        seen.append((label, stop_file))
        return ga.GpuAdmission(label, idle_checks=2, stop_file=stop_file,
                               log_path=str(log))

    monkeypatch.setattr(ga, "admission_from_settings", admission)
    calls = []
    original = ops_engine.run_ops

    def spy(settings, **kwargs):
        """Record that the run itself was entered once admitted."""
        calls.append(ops_engine._OPS_ADMITTED.get())
        return original(settings, **kwargs)

    monkeypatch.setattr(ops_engine, "run_ops", spy)
    with pytest.raises(ValueError, match="no ops_phenotype rows"):
        original({"plate": "P", "dst_root": str(faked_card / "run"),
                  "ops_gpu_admission": True}, phases=("ph_segment",))
    assert calls == [True]
    assert seen == [("spacr-ops-P", str(faked_card / "run" / "STOP"))]
    lines = log.read_text().splitlines()
    assert "START" in lines[0] and lines[1].endswith("FINISH rc=ValueError")
    assert ops_engine._OPS_ADMITTED.get() is False


def test_run_ops_does_not_start_when_stopped(faked_card, monkeypatch):
    run = faked_card / "run"
    run.mkdir()
    (run / "STOP").write_text("")
    out = ops_engine.run_ops({"plate": "P", "dst_root": str(run),
                              "ops_gpu_admission": True},
                             phases=("ph_segment",))
    assert out["wells"] == {} and out["gpu_admission"]["stopped"] is True


def test_run_ops_finishes_cleanly_and_honours_stop_between_wells(
        faked_card, monkeypatch, tmp_path):
    import pandas as pd

    from spacr.ops_store import write_table

    run = tmp_path / "run"
    run.mkdir()
    frame = pd.DataFrame([{"plate": "P_1", "well": well, "site": 0,
                           "centre_y": 1000.0, "centre_x": 1000.0,
                           "sbs_site": 0, "is_anchor": 1}
                          for well in ("A1", "A2")])
    write_table(str(run / "measurements.db"), "ops_phenotype",
                ops_engine._ph_transform_columns(frame, 0.25, 0.0, (64, 64)))
    monkeypatch.setattr(ops_engine, "_ph_masks",
                        lambda *a, **k: (run / "STOP").write_text("") and {})
    out = ops_engine.run_ops({"plate": "P_1", "dst_root": str(run),
                              "ops_gpu_admission": True,
                              "ops_gpu_idle_checks": 1},
                             phases=("ph_masks",))
    assert out["gpu_admission"]["admitted"] is True
    assert list(out["wells"]) == ["A1"] and out["stopped_before"] == "A2"
    lines = (faked_card / "gpu" / "log").read_text().splitlines()
    assert lines[-1].endswith("FINISH rc=0")
    again = ops_engine.run_ops({"plate": "P_1", "dst_root": str(run)},
                               phases=("ph_masks",))
    assert list(again["wells"]) == ["A1", "A2"]
