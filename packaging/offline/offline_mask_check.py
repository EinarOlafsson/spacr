#!/usr/bin/env python3
"""Run spaCR's Mask module on an offline bundle's test data.

Shipped inside every offline bundle and run by the installer's
``--check-mask`` option, or by hand after an install::

    <install root>/venv/bin/python offline_mask_check.py --data <install root>/test_data/plate1

The test images are copied to a scratch folder first, because Mask
reorganises the folder it is pointed at, so the installed copy stays
reusable. A model name in the settings that points at a model this bundle
does not carry (``<models>/...``) is replaced by ``--fallback-model``.
Previously generated merged and role-mask directories are excluded from the
scratch copy so only this invocation's outputs can satisfy the check.

Mask planes embedded in a merged stack are validated against its layout
sidecar when normal pipeline cleanup removes the intermediate mask folders.
Exit 0 when Mask ran and wrote masks, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def model_overrides(settings: Path, fallback: str) -> list[str]:
    """``--set`` overrides for model names that cannot resolve offline."""
    overrides = []
    with open(settings, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].endswith("_model_name") and "<" in row[1]:
                overrides.append(f"{row[0]}={fallback}")
    return overrides


def _merged_mask_count(src: Path, merged: list[Path]) -> int:
    """Validate retained merged label planes after normal intermediate cleanup."""
    import numpy as np

    sidecar = src / 'merged' / '.spacr_plane_layout.json'
    try:
        with sidecar.open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError('merged layout exceeds 64 KiB')
        layout = json.loads(raw)
        if not isinstance(layout, dict) or type(layout.get('version')) is not int or layout['version'] != 1:
            raise ValueError('unsupported merged layout')
        channels = layout.get('intensity_channels')
        order = layout.get('mask_plane_order')
        dims = layout.get('mask_dims')
        if (not isinstance(channels, list) or not channels
                or any(type(value) is not int or value < 0 for value in channels)
                or len(set(channels)) != len(channels)):
            raise ValueError('invalid intensity channel layout')
        if (not isinstance(order, list) or not order
                or any(not isinstance(role, str) or not role.strip() for role in order)
                or len(set(order)) != len(order) or not isinstance(dims, dict)
                or set(dims) != set(order)):
            raise ValueError('invalid mask role layout')
        expected = {role: len(channels) + index for index, role in enumerate(order)}
        if any(type(value) is not int for value in dims.values()) or dims != expected:
            raise ValueError('mask planes overlap intensities or disagree with role order')
        count = 0
        for path in merged:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if (array.ndim < 3 or any(size == 0 for size in array.shape)
                    or array.shape[-1] != len(channels) + len(order)
                    or array.dtype.kind not in 'uif'):
                raise ValueError(f'invalid merged array shape or dtype: {path}')
            for index in dims.values():
                for block in np.nditer(array[..., index], flags=['external_loop', 'buffered'],
                                       op_flags=['readonly'], buffersize=65536):
                    if (not np.isfinite(block).all() or (block < 0).any()
                            or (array.dtype.kind == 'f' and (block != np.floor(block)).any())):
                        raise ValueError(f'invalid labels in merged plane {index}: {path}')
                count += 1
        return count
    except (OSError, ValueError, TypeError) as exc:
        print(f'Cannot validate merged masks: {exc}', file=sys.stderr)
        return 0


def main(argv=None) -> int:
    """Copy the test plate, run ``spacr-run mask`` on it, check its masks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--fallback-model", default="cpsam")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the scratch copy and its masks.")
    args = parser.parse_args(argv)

    settings = args.data / "settings" / "gen_mask_settings.csv"
    if not settings.is_file():
        print(f"no Mask settings at {settings}", file=sys.stderr)
        return 1
    work = Path(tempfile.mkdtemp(prefix="spacr-offline-mask-", dir=args.work))
    try:
        src = work / "plate1"
        shutil.copytree(args.data, src,
                        ignore=shutil.ignore_patterns('merged', '*_mask_stack'))
        command = [sys.executable, "-m", "spacr.cli", "mask",
                   "--settings", str(src / "settings" / "gen_mask_settings.csv"),
                   "--set", f"src={src}", "--no-hash-inputs",
                   *[part for item in model_overrides(settings, args.fallback_model)
                     for part in ("--set", item)]]
        print("+", " ".join(command), flush=True)
        code = subprocess.run(command).returncode
        masks = [p for p in src.rglob('*_mask_stack/*.npy') if p.is_file()]
        merged = [p for p in src.glob('merged/*.npy') if p.is_file()]
        embedded = _merged_mask_count(src, merged) if code == 0 and merged and not masks else 0
        print(f"Mask exit code {code}; {len(masks)} mask arrays, "
              f"{len(merged)} merged stacks under {src}; "
              f"{embedded} validated embedded mask planes")
        return 0 if code == 0 and (masks or embedded) and merged else 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
