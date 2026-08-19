"""Record process and GPU memory use for each regression stage.

Stage readings are included in run summaries and failure reports so resource
exhaustion can be distinguished from other failures. Measurements are
best-effort: missing ``psutil``, an unavailable Torch runtime, or unsupported
container metrics produce an unavailable reading rather than failing the fit.
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
    """The HIGH-WATER mark of torch's CUDA allocation, or ``None``.

    ASKED ONLY IF TORCH IS ALREADY IMPORTED. Importing it to take a
    measurement would make the measurement the most expensive thing in the
    stage, and on a settings panel it is the import this project has twice
    had to keep out (`tests/test_a_settings_panel_does_not_import_torch.py`).

    Uses ``max_memory_allocated`` rather than the current allocation because
    fit tensors may already be released when a stage boundary is recorded.
    The high-water mark is cumulative across the process and therefore reports
    the largest allocation reached across a sequence of fits.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(max(torch.cuda.memory_allocated(),
                       torch.cuda.max_memory_allocated()))
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
    """Record the current fit stage and its memory use.

    Updates the stage and resource-history entries in ``settings`` when it is
    mutable. Measurement and storage failures are ignored so diagnostics do
    not interrupt the fit.

    :param settings: Mutable fit settings or another mapping-like object.
    :param name: Name of the stage being entered.
    :returns: Dictionary containing the stage, resident memory, and allocated
        GPU memory. Unavailable measurements are ``None``.
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
