"""No text in the Navigator protocol steps is cut off at 100 % or 200 % Zoom, in English or German.

:func:`tests.qt.text_fit.protocol_sweep` chooses every family and each long dependent choice
(all filename conventions, image formats, sequencers, read pairings, barcode states and
organisms) through the real selectors and measures the protocol controls after each choice:
labels, including a single word wider than its label, and the choice text as it is drawn.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.qt.test_739_navigator_protocols import protocol_profile as protocol_profile
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile
from tests.qt.text_fit import CONFIGURATIONS

ROOT = Path(__file__).resolve().parents[2]


def _sweep(config, folder):
    """Run one protocol sweep in a fresh process and return its offender keys."""
    scale, language = config
    out = folder / f"protocols-{language}-{scale}.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-m", "tests.qt.text_fit", scale, language, str(out),
                    "--protocols"], cwd=ROOT, env=env, check=True, timeout=900,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.qt
def test_protocol_step_text_is_never_cut_off(tmp_path):
    """Every protocol label and choice shows its whole text in all four configurations."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda config: _sweep(config, tmp_path), CONFIGURATIONS))
    found = sorted(key for keys in results for key in keys)
    assert found == [], "protocol step text is cut off:\n" + "\n".join(found)


def _box_sweep(config, size, folder):
    """Run one description-box check in a fresh process and return its failures."""
    scale, language = config
    out = folder / f"box-{language}-{scale}-{size}.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-m", "tests.qt.text_fit", scale, language, str(out),
                    f"--protocol-box={size}"], cwd=ROOT, env=env, check=True, timeout=900,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.qt
def test_the_description_box_shows_the_whole_protocol_list_when_it_fits(tmp_path):
    """At 1366x768 and 1920x1080, every choice step and the finished protocol show the whole list.

    While Protocols is open the composer moves up into empty space above it, so the box
    may scroll only when even the highest composer leaves too little height.
    """
    runs = [(config, size) for size in ("1366x768", "1920x1080") for config in CONFIGURATIONS]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda run: _box_sweep(run[0], run[1], tmp_path), runs))
    found = sorted(entry for entries in results for entry in entries)
    assert found == [], "the description box scrolls although its text fits:\n" + "\n".join(found)


def test_the_composer_moves_up_only_while_protocols_is_open(qtbot, protocol_profile):
    """The composer gives the box room while Protocols is open and returns to its centre after."""
    from PySide6.QtCore import Qt

    from spacr.qt.widgets.home import _EasyHomePage
    from tests.qt.test_739_navigator_protocols import _choose

    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1366, 600)
    page.show()
    qtbot.waitExposed(page)
    centred = page.field.panel.geometry()
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'sequencing')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'fastq_gz')
    for name, key in (('sequencer', 'ont'), ('read_mode', 'single'), ('single_direction', 'R2')):
        _choose(qtbot, page._protocol_detail_combos[name], key)
    moved = page.field.panel.geometry()
    assert moved.y() < centred.y() and moved.x() == centred.x()
    assert page.field.details.verticalScrollBar().maximum() == 0
    assert page.field.details.geometry().top() > page.field.protocol_controls.geometry().bottom()
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    assert page.field.panel.geometry() == centred
