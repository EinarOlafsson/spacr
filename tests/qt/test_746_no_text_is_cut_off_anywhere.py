"""No user-facing text is cut off on any main screen (item 746 ratchet).

The maintainer (2026-10-31): text must never be cut off, for tooltips and
for all user-facing text. :mod:`tests.qt.text_fit` opens the main window
and every module screen offscreen at 100 % and 200 % Zoom, in English and
in German, and measures every visible label, button, placeholder, combo
box, group title, tab and header section against the room its widget
gives it.

``tests/data/text_fit_known_offenders.json`` lists, exactly, what is still
cut off. The list only shrinks: a new offender fails, and so does an entry
that no longer occurs -- delete it from the list when it is fixed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.qt.text_fit import CONFIGURATIONS

ROOT = Path(__file__).resolve().parents[2]
KNOWN = ROOT / "tests" / "data" / "text_fit_known_offenders.json"


def _sweep(config, folder):
    scale, language = config
    out = folder / f"{language}-{scale}.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-m", "tests.qt.text_fit", scale,
                    language, str(out)], cwd=ROOT, env=env, check=True,
                   timeout=900, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return [entry.rsplit("|@", 1)[0]
            for entry in json.loads(out.read_text(encoding="utf-8"))]


@pytest.mark.heavy
@pytest.mark.qt
def test_the_list_of_cut_off_text_only_shrinks(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda c: _sweep(c, tmp_path),
                                CONFIGURATIONS))
    found = {key for keys in results for key in keys}
    known = set(json.loads(KNOWN.read_text(encoding="utf-8")))
    new = sorted(found - known)
    gone = sorted(known - found)
    assert not new, (
        "text is cut off that was not before -- fix it at its source "
        "(size, size policy, word wrap or a minimum from the font):\n"
        + "\n".join(new))
    assert not gone, (
        "these are no longer cut off; delete them from "
        f"{KNOWN.relative_to(ROOT)}:\n" + "\n".join(gone))
