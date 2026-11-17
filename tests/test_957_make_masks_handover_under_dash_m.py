"""Item 957: ``python -m spacr.cli_make_masks`` hands its session to the screen.

Under ``-m`` the CLI runs as ``__main__``, a second module object beside
``spacr.cli_make_masks``. The Make Masks screen takes the point session, the
mask layers and the queue from ``spacr.cli_make_masks``, so a handover stored
in ``__main__`` was lost silently: Make Masks opened, the point window did
not. The ``__main__`` block now delegates to ``spacr.cli_make_masks.main``.

Each case runs ``python -m spacr.cli_make_masks`` in a subprocess with
offscreen Qt. A ``sitecustomize`` on the subprocess path replaces
``spacr.qt.run`` with a stand-in that takes the handovers exactly as the
screen does and prints what it found, so no window opens.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

_SITECUSTOMIZE = '''
import json, sys
import spacr.qt as _qt

def _fake_run(argv):
    from spacr.cli_make_masks import _take_layers, _take_points, take_handover
    points = _take_points()
    layers = _take_layers()
    queue = take_handover()
    print("HANDOVER " + json.dumps({
        "argv": list(argv),
        "points": None if points is None else str(points["queue"].folder),
        "layers": None if layers is None else layers.get("layers"),
        "queue": None if queue is None else str(queue.folder),
    }), flush=True)
    return 0

_qt.run = _fake_run
'''


def _run(tmp_path, *args):
    hook = tmp_path / "hook"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(hook), str(ROOT)]))
    result = subprocess.run([sys.executable, "-m", "spacr.cli_make_masks", *args],
                            cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("HANDOVER "))
    return json.loads(line[len("HANDOVER "):])


def _images(folder: Path, n: int = 2) -> None:
    folder.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((16, 16), 40 * i, np.uint8)).save(folder / f"f{i}.png")


def test_dash_m_hands_the_point_session_to_the_screen(tmp_path):
    clean = tmp_path / "crops" / "annotate_clean"
    _images(clean)
    seen = _run(tmp_path, "--folder", str(clean), "--points")
    assert seen["argv"] == ["make_masks"]
    assert seen["points"] == str(clean)


def test_dash_m_hands_the_layers_and_queue_to_the_screen(tmp_path):
    folder = tmp_path / "field"
    _images(folder)
    (folder / "masks").mkdir()
    layer = tmp_path / "layer_a"
    layer.mkdir()
    seen = _run(tmp_path, "--folder", str(folder), "--layers", str(layer))
    assert seen["queue"] == str(folder)
    assert seen["layers"] == [str(layer.resolve())]
    assert seen["points"] is None
