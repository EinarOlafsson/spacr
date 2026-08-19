"""What a fit costs while it runs, recorded per stage. Instruction 160.

Filed after two regressions in a row made the machine unresponsive twice,
badly enough to need a restart. That report could not be acted on because
nothing recorded a number: a hung machine is not a hung application, and
telling memory exhaustion from a driver fault from unbounded process spawning
needs measurements taken WHILE the fit runs, not afterwards.

So each stage records what was in use when it began, and the peak is carried
into the run summary and the failure report. The next report then arrives with
"peak RSS 61.2 GB at 'fitting the mixed model'" instead of "it hung".

NOTHING HERE RAISES, and nothing here is required. A measurement that can fail
the run it is measuring is worse than no measurement -- psutil may be absent,
torch may not be importable, and a container may report neither.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "RESOURCE_KEY",
    "STAGE_KEY",
    "host_rss",
    "gpu_allocated",
    "readable",
    "record_stage",
    "peak",
    "describe_resources",
]

#: Where the per-stage readings accumulate on the settings dict.
RESOURCE_KEY = "_regression_resources"

#: Where the current stage name lives, for the failure report to name.
STAGE_KEY = "_regression_stage"


def host_rss() -> Optional[int]:
    """Resident bytes for this process, or ``None`` when unknowable.

    `/proc/self/statm` first because it needs no dependency and no import;
    psutil second. A container that reports neither gets ``None``, which the
    caller must not spell as zero -- "nothing was using memory" and "nobody
    measured" are opposite findings.
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:                                            # noqa: BLE001
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:                                            # noqa: BLE001
        return None


def gpu_allocated() -> Optional[int]:
    """Bytes torch has allocated on the current CUDA device, or ``None``.

    ASKED ONLY IF TORCH IS ALREADY IMPORTED. Importing it to take a
    measurement would make the measurement the most expensive thing in the
    stage, and on a settings panel it is the import this project has twice
    had to keep out (`tests/test_a_settings_panel_does_not_import_torch.py`).
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:                                            # noqa: BLE001
        return None


def readable(total: Optional[int]) -> str:
    """Bytes as the unit a person decides in, or "not measured"."""
    if total is None:
        return "not measured"
    size = float(max(0, int(total)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"                          # pragma: no cover - loop


def record_stage(settings: Any, name: str) -> Dict[str, Any]:
    """Name the stage a fit has reached, and note what it costs there.

    One call does both on purpose: a stage recorded without its cost is the
    state instruction 160 was filed about, and a cost recorded without its
    stage cannot say WHERE the fit was when it grew.

    :returns: the reading, so a caller can print it without asking twice.
    """
    reading = {"stage": str(name), "rss": host_rss(), "gpu": gpu_allocated()}
    try:
        settings[STAGE_KEY] = str(name)
        settings.setdefault(RESOURCE_KEY, []).append(reading)
    except Exception:                                            # noqa: BLE001
        pass
    return reading


def peak(settings: Any) -> Dict[str, Any]:
    """The largest reading recorded, and where it was taken.

    Empty when nothing was recorded -- NOT zero, for the reason `host_rss`
    gives.
    """
    try:
        readings: List[Mapping[str, Any]] = list(
            settings.get(RESOURCE_KEY) or [])
    except Exception:                                            # noqa: BLE001
        return {}
    out: Dict[str, Any] = {}
    for key in ("rss", "gpu"):
        seen = [r for r in readings if r.get(key) is not None]
        if not seen:
            continue
        worst = max(seen, key=lambda r: r[key])
        out[key] = worst[key]
        out[f"{key}_stage"] = worst.get("stage", "")
    return out


def describe_resources(settings: Any) -> str:
    """The per-stage table, for a summary or a failure report. "" when empty."""
    try:
        readings = list(settings.get(RESOURCE_KEY) or [])
    except Exception:                                            # noqa: BLE001
        return ""
    if not readings:
        return ""
    lines = [f"  {'stage':<34} {'resident':>12} {'GPU':>12}"]
    for reading in readings:
        lines.append(
            f"  {str(reading.get('stage', ''))[:34]:<34} "
            f"{readable(reading.get('rss')):>12} "
            f"{readable(reading.get('gpu')):>12}")
    high = peak(settings)
    if "rss" in high:
        lines.append(f"  PEAK resident {readable(high['rss'])} at "
                     f"{high.get('rss_stage', '')!r}")
    if "gpu" in high:
        lines.append(f"  PEAK GPU      {readable(high['gpu'])} at "
                     f"{high.get('gpu_stage', '')!r}")
    return "\n".join(lines)
