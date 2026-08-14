"""Measure this machine, then choose the worker count from the measurement.

Instruction 86:

    "Add a small benchmark command that records throughput and peak RAM/VRAM
     for representative fields, making worker-count defaults specific to the
     current machine."

WHY A DEFAULT NEEDS THIS. spaCR's worker defaults are arithmetic on the core
count -- ``cpu_count() - 4``, ``cpu_count() // 2``, ``-1``. Cores are the one
thing that is never the binding constraint on a measurement run: a field is
hundreds of megabytes decompressed, and eight workers on a 16-core laptop
with 16 GB will swap long before they saturate the CPU. The number that
matters is how much MEMORY one worker needs for one representative field, and
that is a property of the plate and the machine, not of the core count.

So: run the real work over a few fields, watch peak RSS and peak VRAM, and
divide what is available by what one worker actually took.

**This measures; it does not tune.** It returns a recommendation and the
evidence behind it, and never writes a setting. A benchmark that silently
changed `n_jobs` would make a run's speed depend on when the benchmark last
ran, which is the opposite of reproducible.

Qt-free and torch-optional: VRAM is reported when torch is present with a
CUDA device and reported as ``None`` otherwise, which is not the same as 0.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Measurement:
    """What one benchmark run observed.

    :param items: how many representative units were processed.
    :param seconds: wall clock for all of them.
    :param peak_rss_bytes: the highest resident set size seen, for the whole
        process. Not per worker -- the benchmark runs serially on purpose, so
        this IS one worker's requirement.
    :param peak_vram_bytes: peak CUDA allocation, or ``None`` when there is
        no CUDA device. ``None`` and ``0`` are different answers: one means
        "not measured", the other "measured, and it used none".
    :param baseline_rss_bytes: RSS before the work started, so the caller can
        tell the interpreter's own footprint from the work's.
    """

    items: int
    seconds: float
    peak_rss_bytes: int
    peak_vram_bytes: Optional[int] = None
    baseline_rss_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def per_item_seconds(self) -> float:
        return self.seconds / self.items if self.items else float("nan")

    @property
    def items_per_second(self) -> float:
        return self.items / self.seconds if self.seconds > 0 else float("nan")

    @property
    def work_rss_bytes(self) -> int:
        """What the WORK needed, above the interpreter that was already there."""
        return max(0, int(self.peak_rss_bytes) - int(self.baseline_rss_bytes))


@dataclass(frozen=True)
class Recommendation:
    """A worker count, and every number that produced it."""

    workers: int
    reason: str
    measurement: Optional[Measurement] = None
    cores: int = 0
    available_bytes: int = 0

    def __str__(self) -> str:
        return f"{self.workers} worker(s): {self.reason}"


def _rss_bytes() -> int:
    """Resident set size now, in bytes, or 0 where it cannot be read."""
    try:
        import resource
    except ImportError:                      # pragma: no cover - Windows
        return 0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KILOBYTES on Linux and BYTES on macOS. Getting this
    # backwards is a factor of 1024 in a memory budget, which would either
    # recommend one worker on a large machine or forty on a small one.
    import sys
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


def _vram_bytes() -> Optional[int]:
    """Peak CUDA allocation since the counter was reset, or None."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _reset_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def available_memory_bytes() -> int:
    """Memory this machine can actually give to workers.

    ``MemAvailable`` from ``/proc/meminfo`` rather than total: total includes
    what is already in use, and sizing workers against it is how a run gets
    OOM-killed at field 900.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def benchmark(work: Callable[[Any], Any], items: Sequence[Any], *,
              warmup: int = 1) -> Measurement:
    """Run ``work`` over ``items`` serially and record what it cost.

    SERIALLY ON PURPOSE. The question is what ONE worker needs, and running
    them in parallel measures the sum while hiding the per-worker figure that
    the recommendation divides by.

    :param warmup: items processed before the clock starts. The first field
        pays for imports, CUDA context creation and page faults that no later
        field pays again, and counting it makes a short run look far slower
        than the plate it is predicting.
    :raises ValueError: no items to measure -- a benchmark over nothing would
        return a per-item cost of NaN and a recommendation built on it.
    """
    items = list(items)
    if not items:
        raise ValueError("benchmark needs at least one item to measure")

    notes: List[str] = []
    warmup = max(0, min(int(warmup), len(items) - 1))
    for item in items[:warmup]:
        work(item)
    if warmup:
        notes.append(f"{warmup} warm-up item(s) excluded from the timing")

    measured = items[warmup:]
    baseline = _rss_bytes()
    _reset_vram()
    started = time.perf_counter()
    for item in measured:
        work(item)
    elapsed = time.perf_counter() - started

    vram = _vram_bytes()
    if vram is None:
        notes.append("no CUDA device, so VRAM was not measured")
    return Measurement(
        items=len(measured),
        seconds=elapsed,
        peak_rss_bytes=_rss_bytes(),
        peak_vram_bytes=vram,
        baseline_rss_bytes=baseline,
        notes=notes,
    )


def recommend_workers(measurement: Optional[Measurement] = None, *,
                      cores: Optional[int] = None,
                      available_bytes: Optional[int] = None,
                      reserve_bytes: int = 2 * 1024 ** 3,
                      maximum: int = 32) -> Recommendation:
    """How many workers this machine can actually feed.

    The rule, in order:

    1. Never more than ``cores``. More workers than cores is contention.
    2. Never more than ``available memory - reserve`` divided by what ONE
       worker measurably needed. This is the term the core-count defaults
       omit, and it is usually the binding one.
    3. Never fewer than 1, and never more than ``maximum``.

    :param reserve_bytes: memory left for everything that is not a worker --
        the GUI, the page cache the readers depend on, and the operating
        system. Defaults to 2 GiB.
    :returns: a :class:`Recommendation` carrying the reason, so a number a
        user disagrees with can be argued with rather than just overridden.
    """
    cores = int(cores if cores is not None else (os.cpu_count() or 1))
    cores = max(1, cores)
    available = int(available_bytes if available_bytes is not None
                    else available_memory_bytes())

    if measurement is None or measurement.work_rss_bytes <= 0:
        # No measurement, or the work was too small to register above the
        # interpreter. Fall back to the core count rather than inventing a
        # memory bound from a number that is not there.
        workers = max(1, min(cores, maximum))
        return Recommendation(
            workers=workers,
            reason=(f"{cores} core(s); no usable memory measurement, so the "
                    f"core count is the only bound"),
            measurement=measurement, cores=cores, available_bytes=available)

    per_worker = measurement.work_rss_bytes
    budget = max(0, available - int(reserve_bytes))
    by_memory = int(budget // per_worker)

    if by_memory < 1:
        return Recommendation(
            workers=1,
            reason=(f"one worker needs {per_worker / 1024 ** 3:.2f} GiB and "
                    f"only {budget / 1024 ** 3:.2f} GiB is free after the "
                    f"reserve; one worker at a time is what fits"),
            measurement=measurement, cores=cores, available_bytes=available)

    workers = max(1, min(cores, by_memory, maximum))
    if workers == by_memory < cores:
        reason = (f"memory-bound: {budget / 1024 ** 3:.2f} GiB free after the "
                  f"reserve, {per_worker / 1024 ** 3:.2f} GiB per worker, "
                  f"{cores} cores available")
    elif workers == cores:
        reason = (f"core-bound: {cores} core(s), and memory would allow "
                  f"{by_memory}")
    else:
        reason = f"capped at the {maximum}-worker maximum"
    return Recommendation(workers=workers, reason=reason,
                          measurement=measurement, cores=cores,
                          available_bytes=available)


def format_report(measurement: Measurement,
                  recommendation: Recommendation) -> str:
    """The benchmark as a few lines a user can read and paste into an issue."""
    lines = [
        "spaCR benchmark",
        f"  items measured      {measurement.items}",
        f"  throughput          {measurement.items_per_second:.2f} item/s "
        f"({measurement.per_item_seconds:.2f} s each)",
        f"  peak RSS            {measurement.peak_rss_bytes / 1024 ** 3:.2f} GiB",
        f"  attributable to it  {measurement.work_rss_bytes / 1024 ** 3:.2f} GiB",
    ]
    if measurement.peak_vram_bytes is None:
        lines.append("  peak VRAM           not measured (no CUDA device)")
    else:
        lines.append("  peak VRAM           "
                     f"{measurement.peak_vram_bytes / 1024 ** 3:.2f} GiB")
    lines.append(f"  recommended workers {recommendation.workers}")
    lines.append(f"    because           {recommendation.reason}")
    for note in measurement.notes:
        lines.append(f"  note                {note}")
    return "\n".join(lines)
