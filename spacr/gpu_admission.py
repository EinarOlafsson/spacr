"""Wait until the GPU is free before a long run takes it (item 872, B11 and B19, alpha).

A plate-scale phenotype run holds the card for days, so it must not start
while another job is using it, and it must give the card back cleanly. The
reference launcher of the Funk et al. 2022 run did this in a shell script;
:class:`GpuAdmission` is that behaviour as a policy that can be tested
without a GPU:

* optionally wait for another job's release line in a file;
* then wait for ``idle_checks`` samples in a row in which the card shows 0 %
  utilisation and no running compute process uses more than
  ``min_process_mib``. Display and browser processes never count
  (``ignore_patterns``: a browser's ``--type=gpu-process`` held about
  600 MiB and blocked the reference launcher for hours), and nor do stopped
  processes;
* write ``START`` and ``FINISH`` lines to ``~/.spacr/gpu/log``;
* stop cleanly when a ``STOP`` file appears;
* hand out fewer measurement workers when the card's memory in use is above
  a cap (:meth:`GpuAdmission.workers_under_cap`).

The device is queried through ``nvidia-smi`` by :func:`query_gpu`; every
part of the policy takes the query, the clock and the sleep as arguments, so
a test fakes the device.
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

#: Processes that never block admission: browser and Electron GPU helpers,
#: compositors and display servers. A pattern that starts with ``-`` is
#: looked for in the command line; any other is a shell-style pattern for
#: the executable's name.
DEFAULT_IGNORE_PATTERNS = (
    "--type=gpu-process", "chrome*", "google-chrome*", "chromium*",
    "firefox*", "code", "electron*", "Xorg", "Xwayland", "gnome-shell",
    "kwin*", "mullvad*")

#: Where START and FINISH lines are written.
DEFAULT_LOG = os.path.join(os.path.expanduser("~"), ".spacr", "gpu", "log")


@dataclass
class GpuProcess:
    """One compute process on the card.

    :param pid: the process id.
    :param mib: the card memory it uses, in MiB.
    :param executable: the executable's base name.
    :param command: its command line.
    :param stopped: True when the process is stopped (state T), which holds
        memory but no compute.
    """

    pid: int
    mib: float
    executable: str = ""
    command: str = ""
    stopped: bool = False


@dataclass
class GpuSample:
    """One look at the card.

    :param utilisation: the utilisation of each card, in percent.
    :param memory_used_mib: the memory in use on each card, in MiB.
    :param processes: the compute processes.
    """

    utilisation: List[float] = field(default_factory=list)
    memory_used_mib: List[float] = field(default_factory=list)
    processes: List[GpuProcess] = field(default_factory=list)


def _process_details(pid: int) -> Tuple[str, str, bool]:
    """Executable, command line and stopped state of one process, from /proc."""
    executable = command = ""
    stopped = False
    try:
        executable = os.path.basename(os.readlink(f"/proc/{pid}/exe"))
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command = handle.read().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stopped = handle.read().rsplit(")", 1)[-1].split()[0] in ("T", "t")
    except (OSError, IndexError):
        pass
    return executable, command, stopped


def query_gpu(run: Callable[..., Any] = subprocess.run,
              details: Callable[[int], Tuple[str, str, bool]] = _process_details
              ) -> Optional[GpuSample]:
    """Look at the card through ``nvidia-smi``.

    :param run: the subprocess runner, for tests.
    :param details: ``pid -> (executable, command line, stopped)``.
    :returns: the sample, or None when ``nvidia-smi`` is missing or fails.
    """
    def ask(*arguments: str) -> Optional[str]:
        """One ``nvidia-smi`` query's output, or None."""
        try:
            done = run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    cards = ask("--query-gpu=utilization.gpu,memory.used")
    apps = ask("--query-compute-apps=pid,used_memory")
    if cards is None or apps is None:
        return None
    sample = GpuSample()
    for line in cards.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            sample.utilisation.append(float(parts[0]))
            sample.memory_used_mib.append(float(parts[1]))
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            pid = int(parts[0])
            executable, command, stopped = details(pid)
            sample.processes.append(GpuProcess(
                pid, float("".join(c for c in parts[1] if c.isdigit() or c == ".")
                           or 0), executable, command, stopped))
    return sample


def is_display_process(process: GpuProcess,
                       patterns: Sequence[str] = DEFAULT_IGNORE_PATTERNS) -> bool:
    """Whether a process is a display or browser one that never blocks the card.

    :param process: the process.
    :param patterns: :data:`DEFAULT_IGNORE_PATTERNS` or the user's own.
    :returns: True when a ``-`` pattern is in its command line or another
        pattern matches its executable's name.
    """
    for pattern in patterns:
        pattern = str(pattern)
        if pattern.startswith("-"):
            if pattern in process.command:
                return True
        elif process.executable and fnmatch.fnmatch(process.executable, pattern):
            return True
    return False


def busy_processes(sample: GpuSample, *, min_process_mib: float = 500,
                   ignore_patterns: Sequence[str] = DEFAULT_IGNORE_PATTERNS
                   ) -> List[GpuProcess]:
    """The compute processes that keep the card busy.

    :param sample: one look at the card.
    :param min_process_mib: a process using more than this blocks.
    :param ignore_patterns: processes matching these never block.
    :returns: the running, non-display processes above the threshold.
    """
    return [p for p in sample.processes
            if p.mib > float(min_process_mib) and not p.stopped
            and not is_display_process(p, ignore_patterns)]


def gpu_idle(sample: Optional[GpuSample], *, min_process_mib: float = 500,
             ignore_patterns: Sequence[str] = DEFAULT_IGNORE_PATTERNS,
             require_zero_utilisation: bool = True) -> Tuple[bool, str]:
    """Whether one sample shows a free card.

    :param sample: one look at the card; None (no ``nvidia-smi``) is not idle.
    :param min_process_mib: :func:`busy_processes`.
    :param ignore_patterns: :func:`busy_processes`.
    :param require_zero_utilisation: also ask for 0 % utilisation.
    :returns: ``(idle, reason)``; the reason says what keeps it busy.
    """
    if sample is None:
        return False, "no GPU answered (nvidia-smi missing or failed)"
    busy = busy_processes(sample, min_process_mib=min_process_mib,
                          ignore_patterns=ignore_patterns)
    if busy:
        return False, "busy: " + ", ".join(
            f"{p.pid}({p.executable or '?'}, {p.mib:g} MiB)" for p in busy)
    if require_zero_utilisation and any(u > 0 for u in sample.utilisation):
        return False, "utilisation " + ", ".join(
            f"{u:g}%" for u in sample.utilisation)
    return True, "idle"


class GpuAdmission:
    """The admission policy of one long GPU run.

    :param label: the run's name in the GPU log.
    :param idle_checks: idle samples in a row before the run starts.
    :param min_process_mib: :func:`busy_processes`.
    :param vram_cap_gb: :meth:`workers_under_cap`'s cap; None for none.
    :param ignore_patterns: :func:`is_display_process`.
    :param interval: seconds between samples.
    :param stop_file: a file whose presence stops the wait (and, between
        wells, the run); None for none.
    :param release_file: a file to wait in for ``release_line`` first.
    :param release_line: the text another job writes when it releases the
        card.
    :param log_path: where START and FINISH go; :data:`DEFAULT_LOG` when
        None.
    :param query: the device query; :func:`query_gpu` when None.
    :param sleep: the sleep function; :func:`time.sleep` when None.
    :param clock: the clock, for the log lines and the wait time;
        :func:`time.time` when None.
    :param require_zero_utilisation: :func:`gpu_idle`.
    :param max_wait: give up after this many seconds; None waits for ever.
    """

    def __init__(self, label: str, *, idle_checks: int = 6,
                 min_process_mib: float = 500,
                 vram_cap_gb: Optional[float] = None,
                 ignore_patterns: Sequence[str] = DEFAULT_IGNORE_PATTERNS,
                 interval: float = 10.0, stop_file: Optional[str] = None,
                 release_file: Optional[str] = None,
                 release_line: Optional[str] = None,
                 log_path: Optional[str] = None,
                 query: Optional[Callable[[], Optional[GpuSample]]] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 require_zero_utilisation: bool = True,
                 max_wait: Optional[float] = None) -> None:
        """Keep the policy's parameters."""
        self.label = str(label)
        self.idle_checks = max(1, int(idle_checks))
        self.min_process_mib = float(min_process_mib)
        self.vram_cap_gb = None if vram_cap_gb in (None, "") else float(vram_cap_gb)
        self.ignore_patterns = tuple(ignore_patterns)
        self.interval = float(interval)
        self.stop_file = stop_file
        self.release_file = release_file
        self.release_line = release_line
        self.log_path = log_path or DEFAULT_LOG
        self.query = query or (lambda: query_gpu())
        self.sleep = sleep or (lambda seconds: time.sleep(seconds))
        self.clock = clock or (lambda: time.time())
        self.require_zero_utilisation = bool(require_zero_utilisation)
        self.max_wait = max_wait
        self.waits: List[str] = []

    def stop_requested(self) -> bool:
        """Whether the STOP file is there."""
        return bool(self.stop_file) and os.path.exists(str(self.stop_file))

    def _released(self) -> bool:
        """Whether the other job's release line is in its file."""
        if not self.release_file or not self.release_line:
            return True
        try:
            with open(self.release_file, encoding="utf-8",
                      errors="replace") as handle:
                return any(self.release_line in line for line in handle)
        except OSError:
            return False

    def wait(self) -> Dict[str, Any]:
        """Wait until the card may be taken.

        :returns: ``admitted`` (True when the run may start), ``stopped``
            (the STOP file appeared), ``timed_out``, ``samples`` and
            ``waited_s``. The reasons each busy sample gave are kept in
            :attr:`waits`.
        """
        begun = self.clock()
        samples = 0

        def outcome(admitted: bool, stopped: bool = False,
                    timed_out: bool = False) -> Dict[str, Any]:
            """The result dict."""
            return {"admitted": admitted, "stopped": stopped,
                    "timed_out": timed_out, "samples": samples,
                    "waited_s": round(self.clock() - begun, 1)}

        def over() -> bool:
            """Whether the wait has run out of time."""
            return (self.max_wait is not None
                    and self.clock() - begun >= float(self.max_wait))

        while not self._released():
            if self.stop_requested():
                return outcome(False, stopped=True)
            if over():
                return outcome(False, timed_out=True)
            self.sleep(self.interval)
        ok = 0
        while ok < self.idle_checks:
            if self.stop_requested():
                return outcome(False, stopped=True)
            if over():
                return outcome(False, timed_out=True)
            idle, reason = gpu_idle(
                self.query(), min_process_mib=self.min_process_mib,
                ignore_patterns=self.ignore_patterns,
                require_zero_utilisation=self.require_zero_utilisation)
            samples += 1
            if idle:
                ok += 1
            else:
                ok = 0
                self.waits.append(reason)
            if ok < self.idle_checks:
                self.sleep(self.interval)
        return outcome(True)

    def _log(self, text: str) -> None:
        """Append one line to the GPU log."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z",
                              time.localtime(self.clock()))
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} [{self.label}] {text}\n")

    def start(self) -> None:
        """Write the START line."""
        self._log(f"START pid {os.getpid()}")

    def finish(self, status: Any = 0) -> None:
        """Write the FINISH line.

        :param status: what the run ended with (0, an exception name).
        """
        self._log(f"FINISH rc={status}")

    def workers_under_cap(self, workers: int,
                          sample: Optional[GpuSample] = None) -> int:
        """Fewer measurement workers while the card's memory is above the cap.

        :param workers: the workers asked for.
        :param sample: a look at the card; queried when None.
        :returns: ``workers`` when there is no cap, no answer or the memory
            in use is at or below the cap; else half of it (at least 1),
            as the reference watchdog restarted with fewer workers.
        """
        if self.vram_cap_gb is None:
            return workers
        sample = sample if sample is not None else self.query()
        if sample is None or not sample.memory_used_mib:
            return workers
        used_gb = max(sample.memory_used_mib) / 1024.0
        return workers if used_gb <= self.vram_cap_gb else max(1, workers // 2)


def admission_from_settings(settings: Mapping[str, Any], label: str, *,
                            stop_file: Optional[str] = None,
                            **overrides: Any) -> GpuAdmission:
    """The admission policy the OPS settings describe.

    :param settings: read for ``ops_gpu_idle_checks``,
        ``ops_gpu_min_process_mib``, ``ops_gpu_vram_cap_gb`` and
        ``ops_gpu_ignore_process_patterns`` (comma-separated, added to
        :data:`DEFAULT_IGNORE_PATTERNS`).
    :param label: the run's name in the GPU log.
    :param stop_file: the STOP file.
    :param overrides: any other :class:`GpuAdmission` parameter (the query
        and sleep, in tests).
    :returns: the policy.
    """
    extra = settings.get("ops_gpu_ignore_process_patterns") or ()
    if isinstance(extra, str):
        extra = [p.strip() for p in extra.split(",") if p.strip()]
    checks = settings.get("ops_gpu_idle_checks")
    threshold = settings.get("ops_gpu_min_process_mib")
    return GpuAdmission(
        label, idle_checks=6 if checks in (None, "") else int(checks),
        min_process_mib=500 if threshold in (None, "") else float(threshold),
        vram_cap_gb=settings.get("ops_gpu_vram_cap_gb"),
        ignore_patterns=(*DEFAULT_IGNORE_PATTERNS, *extra),
        stop_file=stop_file, **overrides)
