#!/usr/bin/env python
"""Stop a runaway process before it takes the desktop down with it.

WHY THIS EXISTS AND WHY IT IS NOT THE MAIN GUARD. On 2026-09-20 at
15:46:13 the kernel killed a python holding 113 GiB on this 125 GB machine
and VS Code went down with it. It was the third time. The maintainer asked
for a watchdog that acts when RAM passes 100 GB. Filed as 450 and
renumbered 453 the same day, because the work session filed its own 450.

READ THIS BEFORE TRUSTING IT. `tools/run_capped.sh` already records that a
POLLER WAS TRIED AND LOST: a daemon reading /proc/meminfo every three
seconds fired five times and still lost the machine, because a process
going from nothing to ninety gigabytes outruns a three-second poll. A
cgroup is checked by the kernel on every allocation; a poller is checked
when it happens to look. So:

    run_capped.sh is the guard. This is the net under it,
    for the processes somebody forgot to put through it.

WHAT IT DOES DIFFERENTLY FROM THE POLLER THAT LOST. Three things, and the
first two are why it can win where that one could not.

* IT FREEZES BEFORE IT KILLS. SIGSTOP stops a process allocating in the
  time it takes to deliver a signal. The old poller killed, then waited ten
  seconds before considering the next one; this stops the offender dead and
  leaves it stopped, so its memory is still there to look at and the
  decision to kill it can be made by a human -- or by this program, when
  memory keeps climbing anyway.
* IT MAKES THE KERNEL PICK THE RIGHT VICTIM. Writing 1000 to the
  offender's oom_score_adj means that if the kernel's own OOM killer beats
  us to it, it takes the runaway rather than the editor. This matters here
  more than it sounds: VS Code's own processes on this machine run at
  oom_score_adj=100, which biases the kernel TOWARDS them.
* IT ACTS WITH HEADROOM. 100 GB of 125 GB leaves 25 GB, which is a lot of
  allocating to do in one poll interval.

WHAT IT WILL NOT TOUCH: the desktop and the editor (see :data:`PROTECTED`),
anything under a memory cgroup that already has a limit -- that process is
somebody else's problem and will die in its own scope -- and itself.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Sequence

#: Names this never signals, however large they get. Killing the compositor
#: to save memory is not saving the session.
PROTECTED = frozenset({
    "code", "node", "Xorg", "gnome-shell", "mutter", "mutter-x11-fram",
    "systemd", "dbus-daemon", "pulseaudio", "pipewire", "chrome",
    "firefox", "ssh", "sshd", "tmux", "bash", "zsh",
})

#: Below this an offender is not worth acting on: nothing on a normal
#: desktop holds sixteen gigabytes by accident, and above it nothing does
#: by right either.
DEFAULT_FLOOR_GB = 16.0

#: Act here. The maintainer's number, 2026-09-20.
DEFAULT_ACT_GB = 100.0

#: Kill what was frozen if memory is still above this afterwards.
DEFAULT_KILL_GB = 112.0


class Process(NamedTuple):
    """One process, as much of it as this needs."""

    pid: int
    name: str
    rss_gb: float
    capped: bool


def memory_used_gb(meminfo: Optional[str] = None) -> float:
    """How much memory is in use, as the kernel reports it.

    MemTotal minus MemAvailable rather than MemFree: the page cache is not
    a leak, and counting it as used would make this fire on a machine that
    is merely busy.

    :param meminfo: the contents of /proc/meminfo, for tests.
    :returns: gigabytes in use.
    """
    text = meminfo if meminfo is not None else _read("/proc/meminfo")
    values: Dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal",
                                                        "MemAvailable"):
            values[parts[0].rstrip(":")] = float(parts[1])
    if len(values) != 2:
        return 0.0
    return (values["MemTotal"] - values["MemAvailable"]) / 1024 / 1024


def _read(path: str) -> str:
    """Read a file, or "" when it is gone -- a pid can die mid-scan."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def is_capped(cgroup: str) -> bool:
    """Whether this process already lives under a memory limit.

    A process inside a scope with `memory.max` set will be killed there,
    by the kernel, without the rest of the machine noticing. It is not this
    program's business.

    :param cgroup: the contents of /proc/<pid>/cgroup.
    :returns: True when a limit is set on its cgroup.
    """
    for line in cgroup.splitlines():
        path = line.rpartition(":")[2].strip()
        if not path or path == "/":
            continue
        limit = _read(f"/sys/fs/cgroup{path}/memory.max").strip()
        if limit and limit != "max":
            return True
    return False


def scan(pids: Optional[Sequence[int]] = None) -> List[Process]:
    """Every process this could act on, largest first."""
    found = []
    for entry in (pids if pids is not None else os.listdir("/proc")):
        try:
            pid = int(entry)
        except (TypeError, ValueError):
            continue
        status = _read(f"/proc/{pid}/status")
        if not status:
            continue
        name = ""
        rss_kb = 0.0
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.split(maxsplit=1)[1].strip() if len(
                    line.split()) > 1 else ""
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    rss_kb = float(parts[1])
                break
        if not name:
            continue
        found.append(Process(pid, name, rss_kb / 1024 / 1024,
                             is_capped(_read(f"/proc/{pid}/cgroup"))))
    return sorted(found, key=lambda p: p.rss_gb, reverse=True)


def choose(processes: Sequence[Process], floor_gb: float,
           me: Optional[int] = None) -> Optional[Process]:
    """The one to act on, or None when nothing qualifies.

    :param processes: what :func:`scan` found.
    :param floor_gb: ignore anything smaller.
    :param me: this program's own pid.
    :returns: the largest process that is not protected, not already
        capped, and not this program.
    """
    mine = os.getpid() if me is None else me
    for process in sorted(processes, key=lambda p: p.rss_gb, reverse=True):
        if process.pid == mine or process.pid == 1:
            continue
        if process.name in PROTECTED or process.capped:
            continue
        if process.rss_gb < floor_gb:
            continue
        return process
    return None


def make_the_kernel_prefer(pid: int) -> bool:
    """Ask the kernel to take ``pid`` first if it has to take something.

    Raising a score needs no privilege; lowering one does. So this raises
    the offender rather than protecting the editor, which is the half that
    works without root.

    :param pid: the offender.
    :returns: whether the score was written.
    """
    try:
        with open(f"/proc/{pid}/oom_score_adj", "w", encoding="utf-8") as h:
            h.write("1000")
        return True
    except OSError:
        return False


def freeze(pid: int) -> bool:
    """Stop a process allocating, without ending it.

    :param pid: the offender.
    :returns: whether the signal was delivered.
    """
    try:
        os.kill(pid, signal.SIGSTOP)
        return True
    except OSError:
        return False


def watch(act_gb: float, kill_gb: float, floor_gb: float, interval: float,
          once: bool = False, log=print) -> int:
    """Poll, and act when memory passes the threshold.

    :param act_gb: freeze the largest offender above this.
    :param kill_gb: kill an already-frozen offender above this.
    :param floor_gb: the smallest process worth acting on.
    :param interval: seconds between looks.
    :param once: check a single time and return, for tests.
    :returns: 0 always; this is a daemon.
    """
    frozen: Dict[int, Process] = {}
    while True:
        used = memory_used_gb()
        if used >= kill_gb and frozen:
            for pid, process in list(frozen.items()):
                log(f"memory still at {used:.1f} GB; killing frozen "
                    f"{process.name} ({pid}) at {process.rss_gb:.1f} GB")
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                frozen.pop(pid, None)
        elif used >= act_gb:
            offender = choose(scan(), floor_gb)
            if offender is None:
                log(f"memory at {used:.1f} GB and nothing above "
                    f"{floor_gb:.0f} GB is uncapped -- nothing to do")
            elif offender.pid not in frozen:
                made = make_the_kernel_prefer(offender.pid)
                stopped = freeze(offender.pid)
                frozen[offender.pid] = offender
                log(f"memory at {used:.1f} GB: {offender.name} "
                    f"({offender.pid}) holds {offender.rss_gb:.1f} GB "
                    f"uncapped. oom_score_adj={'1000' if made else 'unchanged'}, "
                    f"{'FROZEN with SIGSTOP' if stopped else 'could not be stopped'}. "
                    f"Nothing was killed; `kill -CONT {offender.pid}` resumes it.")
        if once:
            return 0
        time.sleep(interval)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--act-gb", type=float, default=DEFAULT_ACT_GB)
    parser.add_argument("--kill-gb", type=float, default=DEFAULT_KILL_GB)
    parser.add_argument("--floor-gb", type=float, default=DEFAULT_FLOOR_GB)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true",
                        help="look once and exit")
    parser.add_argument("--report", action="store_true",
                        help="print what it would act on and exit")
    args = parser.parse_args(argv)

    if args.report:
        used = memory_used_gb()
        print(f"memory in use: {used:.1f} GB (acts at {args.act_gb:.0f})")
        offender = choose(scan(), args.floor_gb)
        if offender is None:
            print(f"nothing uncapped is above {args.floor_gb:.0f} GB")
        else:
            print(f"would act on: {offender.name} ({offender.pid}), "
                  f"{offender.rss_gb:.1f} GB, capped={offender.capped}")
        return 0
    return watch(args.act_gb, args.kill_gb, args.floor_gb, args.interval,
                 once=args.once)


if __name__ == "__main__":
    sys.exit(main())
