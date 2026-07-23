"""
Bridge helpers between the v1 preprocess_generate_masks settings dict
and the v2 streaming pipeline (:mod:`spacr.pipeline_v2`).

Kept in its own module so :mod:`spacr.core` doesn't grow another 200
lines and so unit tests can hit the translation code without spinning
up Cellpose.

Two responsibilities:

* :func:`v2_channels_from_settings` — extract ``(channels,
  channel_names)`` in a stable order from the mask/cell/nucleus/
  pathogen/organelle settings keys.
* :func:`report_disk_savings` — after a v2 run, log an estimate of
  how much disk v1 would have used vs. what v2 actually used, so
  users can see the payoff.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

LOG = logging.getLogger("spacr.pipeline_v2.bridge")


# ---------------------------------------------------------------------------
# Settings → v2 kwargs
# ---------------------------------------------------------------------------

_CHANNEL_KEYS = [
    # (settings key,          human name)
    ("nucleus_channel",       "nucleus"),
    ("cell_channel",          "cell"),
    ("pathogen_channel",      "pathogen"),
    ("organelle_channel",     "organelle"),
]


def v2_channels_from_settings(settings: Dict[str, Any]
                                ) -> Tuple[List[int], List[str]]:
    """Pick out ``(channel_indices, channel_names)`` from a v1 settings dict.

    Order is fixed: nucleus, cell, pathogen, organelle (drops any that
    are None or absent). Uses the same C-axis convention that
    ``spacr.qt.synthetic.CHANNEL_LAYOUT`` uses so demo data flows
    end-to-end through v2 unchanged.

    :returns: two lists of equal length — indices to pass to
        :func:`stream_originals_to_stack`, and matching human names to
        record in ``channel_order.json``.
    """
    chans: List[int] = []
    names: List[str] = []
    for key, human in _CHANNEL_KEYS:
        v = settings.get(key)
        if v is None:
            continue
        try:
            chans.append(int(v))
            names.append(human)
        except (TypeError, ValueError):
            continue
    if not chans:
        # Fall back to a top-level `channels` list if the user set that
        raw = settings.get("channels")
        if isinstance(raw, (list, tuple)):
            for i, c in enumerate(raw):
                try:
                    chans.append(int(c))
                    names.append(f"ch{i}")
                except (TypeError, ValueError):
                    continue
    if not chans:
        # Last-ditch default — 4-channel plate
        chans = [0, 1, 2, 3]
        names = ["ch0", "ch1", "ch2", "ch3"]
    return chans, names


# ---------------------------------------------------------------------------
# Disk-savings reporter
# ---------------------------------------------------------------------------

def report_disk_savings(src: Path, stacks: Sequence[Any]) -> Dict[str, Any]:
    """After a v2 run, log an estimate of v1's disk footprint vs v2's.

    v1 keeps every intermediate: channel/, orig/, stack/ (per-channel
    npy), stack.npz (batch), masks/ (per-field mask npy), and merged/
    (final stack). We approximate v1 as roughly:

        v1 ≈ 4 × merged  (originals + orig backup + per-channel npy +
                            batch npz)

    (Real ratios in the field are 3-5× depending on channel count.)

    :param src: plate root.
    :param stacks: the list of :class:`StackFile` produced by the run.
    :returns: dict of ``{"v2_bytes", "v1_estimated_bytes",
        "saved_pct"}``; also logged at INFO.
    """
    src = Path(src)
    v2_bytes = 0
    for s in stacks:
        try:
            v2_bytes += Path(s.path).stat().st_size
        except Exception:
            continue
    # Add the filename map + channel-order sidecars
    for extra in (src / "filename_map.csv",
                    src / "filename_map.json",
                    src / "merged" / "channel_order.json"):
        try:
            if extra.exists():
                v2_bytes += extra.stat().st_size
        except Exception:
            pass

    v1_estimated_bytes = v2_bytes * 4   # see docstring for rationale
    saved = v1_estimated_bytes - v2_bytes
    saved_pct = round(100.0 * saved / max(1, v1_estimated_bytes), 1)

    LOG.info(
        "v2 pipeline finished — used %s. Estimated v1 disk use for the "
        "same run: %s. Saved: %s (%s%%).",
        _human(v2_bytes), _human(v1_estimated_bytes),
        _human(saved), saved_pct,
    )
    return {
        "v2_bytes":            v2_bytes,
        "v1_estimated_bytes":  v1_estimated_bytes,
        "saved_bytes":         saved,
        "saved_pct":           saved_pct,
    }


def _human(n_bytes: int) -> str:
    """Render byte count in a human-friendly unit."""
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6),
                       ("KB", 1e3)):
        if n_bytes >= div:
            return f"{n_bytes / div:.2f} {unit}"
    return f"{n_bytes} B"
