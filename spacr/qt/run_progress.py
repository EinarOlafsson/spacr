"""How far a run is and how long it has left, read from its Progress lines.

Workers print ``Progress: n/total, operation_type: <stage>, ...,
Time_left: x min.`` (:func:`spacr.utils.print_progress`) or a bare
``Progress: n/total``. :class:`RunProgress` turns that stream into one
record per run: the overall fraction ``0..1`` and the seconds left, or
``None`` while either is unknown.

**Stages.** A module whose run passes through several counted stages (Make
Masks preprocesses, then segments, then merges arrays) declares them in
:data:`MODULE_STAGES`. Each stage names the ``operation_type`` labels that
belong to it and a weight, its share of the run time. The overall fraction
is then the weights of the stages already passed plus the current stage's
weight times its own fraction, and it never goes backwards. A stage the run
skips counts as passed. A label a declared module does not list (a nested
loop inside a stage) does not move the bar once a declared stage has
started. A module that declares nothing keeps the fraction of whichever
stage is printing and names it.

The weights are read from the code (which stage does the per-image heavy
work), not yet measured on a reference dataset; item 938 records that the
measurement is still owed.

**Time left.** The worker's own ``Time_left`` is used when it prints one and
it covers the rest of the run, that is when the module declares no stages or
the run is on its last declared stage. Otherwise the estimate is the
remaining fraction divided by a smoothed rate: an exponential moving average
of the fraction gained per active second, sampled at most every
:data:`RATE_SAMPLE_S` seconds. Nothing is shown before :data:`WARMUP_FRACTION`
of the run or :data:`WARMUP_S` active seconds, whichever comes first, so the
first number is not wild. Paused time is not active time: while a run is
paused the estimate stands still, and on resume it carries on from the last
rate.

The class is plain Python with an injectable clock so the estimator can be
tested without Qt or real time.
"""
from __future__ import annotations

import fnmatch
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Fraction of the run after which a time-left estimate may be shown.
WARMUP_FRACTION = 0.05
#: Active seconds after which a time-left estimate may be shown.
WARMUP_S = 30.0
#: Weight of the newest rate sample in the moving average.
RATE_SMOOTHING = 0.3
#: Shortest active interval, in seconds, that yields a rate sample.
RATE_SAMPLE_S = 2.0

_PROGRESS_LINE_RE = re.compile(r"\bProgress:\s*(\d+)\s*/\s*(\d+)")
_OPERATION_RE = re.compile(r"operation_type:\s*([^,]*?)\s*(?:,|$)")
_TIME_LEFT_RE = re.compile(r"Time_left:\s*([0-9]+(?:\.[0-9]+)?)\s*min")
_MASK_GPU_LINE_RE = re.compile(r"\[mask GPUs\] \w+ Progress:")

#: The label given to the parallel mask workers' ``[mask GPUs]`` lines,
#: which carry no ``operation_type`` of their own.
MASK_GPU_LABEL = "mask GPUs"

_MASK_STAGES: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("Consolidating images",), 0.04),
    (("Merging channels into npy stacks", "Preprocessing filenames"), 0.04),
    (("Concatinating",), 0.08),
    (("Normalizing",), 0.04),
    (("*_mask_gen", MASK_GPU_LABEL), 0.60),
    (("adjust_cell_masks",), 0.05),
    (("Merging Arrays",), 0.10),
    (("Plot mask outlines",), 0.05),
)
_CLASSIFY_STAGES: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("Copying files for Train dataset",), 0.05),
    (("Copying files for Test dataset",), 0.02),
    (("Training",), 0.58),
    (("generating .tar dataset",), 0.15),
    (("Tar dataset",), 0.20),
)

#: Declared stages per module key: ``((labels, weight), ...)`` in run order.
#: Labels are ``fnmatch`` patterns over the printed ``operation_type``.
#: Weights need not sum to one; they are normalised.
MODULE_STAGES: Dict[str, Tuple[Tuple[Tuple[str, ...], float], ...]] = {
    "mask": _MASK_STAGES,
    "timelapse": _MASK_STAGES,
    "measure": (
        (("Merging Arrays",), 0.10),
        (("Measure and Crop",), 0.90),
    ),
    "external_masks": (
        (("Writing merged arrays",), 0.30),
        (("Measure and Crop",), 0.70),
    ),
    "classify": _CLASSIFY_STAGES,
    "classify_merged": _CLASSIFY_STAGES,
}


class ProgressSample:
    """One parsed ``Progress:`` line.

    :param done: units finished.
    :param total: units in the stage.
    :param label: the ``operation_type``, or ``None`` when the line has none.
    :param seconds_left: the worker's own estimate in seconds, or ``None``.
    """

    __slots__ = ("done", "total", "label", "seconds_left")

    def __init__(self, done: int, total: int, label: Optional[str] = None,
                 seconds_left: Optional[float] = None):
        """Hold the four fields of one progress line."""
        self.done = int(done)
        self.total = int(total)
        self.label = label or None
        self.seconds_left = seconds_left

    def __repr__(self) -> str:
        """Show the fields, for test failure messages."""
        return (f"ProgressSample({self.done}/{self.total}, "
                f"label={self.label!r}, seconds_left={self.seconds_left!r})")


def parse_progress(text: str) -> List[ProgressSample]:
    """Every progress line in ``text``, in the order printed.

    ``Time_left`` is read only from a line that also counts at least one
    finished unit: ``print_progress`` reports ``0.000 min`` before the first
    unit, which is not an estimate.

    :param text: worker output, one or more lines.
    :returns: the samples; empty when no line carries ``Progress: n/total``.
    """
    samples: List[ProgressSample] = []
    for line in str(text or "").splitlines():
        match = _PROGRESS_LINE_RE.search(line)
        if match is None:
            continue
        done, total = int(match.group(1)), int(match.group(2))
        rest = line[match.end():]
        label = None
        operation = _OPERATION_RE.search(rest)
        if operation is not None and operation.group(1).strip():
            label = operation.group(1).strip()
        elif _MASK_GPU_LINE_RE.search(line):
            label = MASK_GPU_LABEL
        seconds_left = None
        left = _TIME_LEFT_RE.search(rest)
        if left is not None and done > 0:
            seconds_left = float(left.group(1)) * 60.0
        samples.append(ProgressSample(done, total, label, seconds_left))
    return samples


class RunProgress:
    """Overall fraction and time left for one run.

    :param app_key: the module running; its :data:`MODULE_STAGES` entry, if
        any, weights the stages.
    :param clock: returns the current time in seconds; :func:`time.monotonic`
        by default.
    :param stages: stage declarations to use instead of the module's.
    """

    def __init__(self, app_key: str = "", clock: Optional[Callable[[], float]] = None,
                 stages: Optional[Sequence[Tuple[Sequence[str], float]]] = None):
        """Start an empty record whose active clock starts now."""
        self._clock = clock or time.monotonic
        declared = MODULE_STAGES.get(app_key, ()) if stages is None else stages
        total_weight = sum(max(0.0, float(w)) for _labels, w in declared)
        self._stages: Tuple[Tuple[Tuple[str, ...], float], ...] = tuple(
            (tuple(labels), max(0.0, float(w)) / total_weight)
            for labels, w in declared) if total_weight > 0 else ()
        self._started = self._clock()
        self._paused_total = 0.0
        self._paused_at: Optional[float] = None
        self._stage_index: Optional[int] = None
        self._label: Optional[str] = None
        self._fraction: Optional[float] = None
        self._worker_left: Optional[float] = None
        self._worker_left_at = 0.0
        self._window_start = 0.0
        self._window_fraction = 0.0
        self._anchor_time = 0.0
        self._anchor_fraction = 0.0
        self._rate: Optional[float] = None
        self._rate_left: Optional[float] = None
        self._rate_left_at = 0.0
        self._last_seen = 0.0

    @property
    def declares_stages(self) -> bool:
        """Whether the module declared weighted stages."""
        return bool(self._stages)

    @property
    def stage_label(self) -> Optional[str]:
        """The ``operation_type`` of the newest counted line, or ``None``."""
        return self._label

    def active_seconds(self) -> float:
        """Seconds since the start, less the time spent paused."""
        now = self._clock()
        paused = self._paused_total
        if self._paused_at is not None:
            paused += now - self._paused_at
        return max(0.0, now - self._started - paused)

    def is_paused(self) -> bool:
        """Whether :meth:`pause` was called and not yet resumed."""
        return self._paused_at is not None

    def pause(self) -> None:
        """Stop the active clock; repeated calls change nothing."""
        if self._paused_at is None:
            self._paused_at = self._clock()

    def resume(self) -> None:
        """Restart the active clock; the last rate carries on."""
        if self._paused_at is not None:
            self._paused_total += max(0.0, self._clock() - self._paused_at)
            self._paused_at = None

    def sync_pause(self, paused: bool) -> None:
        """Follow a pause gate's state: pause or resume to match ``paused``.

        :param paused: whether the gate is paused now.
        """
        if paused:
            self.pause()
        else:
            self.resume()

    def fraction(self) -> Optional[float]:
        """Overall fraction in ``0..1``, or ``None`` before any count."""
        return self._fraction

    def _stage_of(self, label: Optional[str]) -> Optional[int]:
        """Index of the declared stage ``label`` belongs to, or ``None``."""
        if label is None:
            return None
        for index, (patterns, _weight) in enumerate(self._stages):
            if any(fnmatch.fnmatchcase(label, pattern) for pattern in patterns):
                return index
        return None

    def observe(self, sample: ProgressSample) -> None:
        """Fold one progress line into the record.

        :param sample: a parsed line; a total of zero or less is ignored.
        """
        if sample.total <= 0:
            return
        stage_fraction = max(0.0, min(1.0, sample.done / sample.total))
        index = self._stage_of(sample.label)
        if self._stages and index is None and self._stage_index is not None:
            return
        now = self.active_seconds()
        restart_at: Optional[float] = None
        if index is not None:
            passed = sum(w for _labels, w in self._stages[:index])
            overall = passed + self._stages[index][1] * stage_fraction
            if self._stage_index is None:
                restart_at = 0.0
            self._stage_index = index
            last = index == len(self._stages) - 1
        else:
            overall = stage_fraction
            if self._fraction is None:
                restart_at = 0.0
            elif sample.label != self._label:
                restart_at = self._last_seen
            last = True
        if restart_at is not None:
            self._rate = None
            self._rate_left = None
            self._worker_left = None
            self._window_start = self._anchor_time = restart_at
            self._window_fraction = self._anchor_fraction = 0.0
        elif self._fraction is not None:
            overall = max(self._fraction, overall)
        self._fraction = min(1.0, overall)
        self._label = sample.label
        self._last_seen = now
        self._update_rate(now)
        if sample.seconds_left is not None and last:
            self._worker_left = max(0.0, sample.seconds_left)
            self._worker_left_at = now
        elif not last:
            self._worker_left = None

    def _update_rate(self, now: float) -> None:
        """Take a rate sample when enough active time has passed."""
        span = now - self._anchor_time
        if span < RATE_SAMPLE_S:
            return
        gained = self._fraction - self._anchor_fraction
        sample = max(0.0, gained) / span
        if self._rate is None:
            self._rate = sample
        else:
            self._rate = (RATE_SMOOTHING * sample
                          + (1.0 - RATE_SMOOTHING) * self._rate)
        self._anchor_time = now
        self._anchor_fraction = self._fraction
        if self._rate > 0:
            self._rate_left = (1.0 - self._fraction) / self._rate
            self._rate_left_at = now

    def warmed_up(self) -> bool:
        """Whether enough of the run has passed to show an estimate."""
        if self._fraction is None:
            return False
        progressed = self._fraction - self._window_fraction
        waited = self.active_seconds() - self._window_start
        return progressed >= WARMUP_FRACTION or waited >= WARMUP_S

    def seconds_left(self) -> Optional[float]:
        """Seconds the run still needs, or ``None`` while unknown.

        Counts down between progress lines with active time, so a paused run
        holds its estimate.
        """
        if self._fraction is None:
            return None
        if self._fraction >= 1.0:
            return 0.0
        if not self.warmed_up():
            return None
        now = self.active_seconds()
        if self._worker_left is not None:
            return max(0.0, self._worker_left - (now - self._worker_left_at))
        if self._rate_left is None:
            return None
        return max(0.0, self._rate_left - (now - self._rate_left_at))


def format_time_left(seconds: Optional[float]) -> str:
    """Round ``seconds`` for display: "about 12 min left", "about 2 h 10 min left".

    Under an hour rounds to the minute; an hour or more rounds to five
    minutes. ``None`` reads "estimating…".

    :param seconds: the estimate, or ``None``.
    :returns: the translated phrase.
    """
    from .i18n import tr

    if seconds is None:
        return tr("estimating…")
    if seconds < 60:
        return tr("less than a minute left")
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return tr("about {minutes} min left", minutes=minutes)
    minutes = int(round(minutes / 5.0)) * 5
    hours, minutes = divmod(minutes, 60)
    if minutes == 0:
        return tr("about {hours} h left", hours=hours)
    return tr("about {hours} h {minutes} min left", hours=hours,
              minutes=minutes)


def describe_progress(handle) -> Tuple[Optional[int], str]:
    """The percent and time-left text the module bar and Home banner show.

    Both surfaces call this with the same run handle, so they always agree.

    :param handle: a run handle with ``fraction()`` and, optionally,
        ``seconds_left()``.
    :returns: ``(percent, text)``; ``percent`` is ``None`` when the run has
        no countable steps (keep the busy bar), and ``text`` is then empty.
    """
    fraction = handle.fraction()
    if fraction is None:
        return None, ""
    percent = int(round(max(0.0, min(1.0, fraction)) * 100))
    reader = getattr(handle, "seconds_left", None)
    seconds = reader() if callable(reader) else None
    return percent, format_time_left(seconds)


def show_on_bar(bar, percent: Optional[int]) -> None:
    """Put ``percent`` in a ``QProgressBar``, or make it a busy bar.

    :param bar: the progress bar.
    :param percent: ``0..100``, or ``None`` for a run with no countable steps.
    """
    from .i18n import tr

    if percent is None:
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        return
    bar.setRange(0, 100)
    bar.setValue(int(percent))
    bar.setFormat(tr("{percent} %", percent=int(percent)))
    bar.setTextVisible(True)
