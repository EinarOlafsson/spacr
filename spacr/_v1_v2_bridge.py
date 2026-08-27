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
import numpy as np
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


def v2_mask_source(merged_dir, object_type: str = "cell"):
    """A lazy mask source for :func:`spacr.seg_qc.run_segmentation_qc`.

    v1 writes one ``.npy`` per field into a ``<object_type>_mask_stack``
    folder, which the scorecard globs directly. v2 has no such folder: the
    mask is a CHANNEL of ``merged/stack_<field>.npy``, shape
    ``(H, W, C_image + C_mask)``. This reads ``channel_order.json`` to learn
    which plane that is and hands back ``{field: thunk}``, so the same QC
    scores both layouts and neither has to know about the other.

    :param merged_dir: the ``merged/`` folder ``run_v2`` wrote.
    :param object_type: which mask channel to score, matched by name against
        ``channel_order.json``'s ``mask_channels``.
    :returns: ``{field_id: callable}`` -- one thunk per field, each loading
        its own stack and slicing one plane. Empty when the sidecar is
        missing, names no mask, or names none matching ``object_type``;
        scoring nothing is the honest answer to "there is no mask here", and
        the caller says so rather than this raising into a finished run.

    Loaded through ``mmap_mode='r'`` and copied plane-first, so a 1536-field
    plate is scored one field at a time rather than read whole.
    """
    import json
    import os

    merged = os.fspath(merged_dir)
    sidecar = os.path.join(merged, "channel_order.json")
    try:
        with open(sidecar) as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return {}

    image_channels = list(meta.get("image_channels") or [])
    mask_channels = list(meta.get("mask_channels") or [])
    if not mask_channels:
        return {}

    wanted = str(object_type)
    if wanted in mask_channels:
        offset = mask_channels.index(wanted)
    elif len(mask_channels) == 1:
        # One mask and a name that does not match: score it anyway. The
        # channel was written by the run being scored, and refusing over a
        # naming difference would report "no masks" about a plate that has
        # them.
        offset = 0
    else:
        return {}
    plane = len(image_channels) + offset

    def _reader(path):
        def read():
            stack = np.load(path, mmap_mode="r")
            if stack.ndim < 3 or plane >= stack.shape[2]:
                raise IndexError(
                    f"{os.path.basename(path)} has no plane {plane}: its "
                    f"shape is {getattr(stack, 'shape', None)}, so "
                    f"channel_order.json does not describe this stack")
            return np.asarray(stack[:, :, plane])
        return read

    out = {}
    for name in sorted(os.listdir(merged)):
        if not name.lower().endswith(".npy"):
            continue
        field = name[:-4]
        if field.startswith("stack_"):
            field = field[len("stack_"):]
        out[field] = _reader(os.path.join(merged, name))
    return out
