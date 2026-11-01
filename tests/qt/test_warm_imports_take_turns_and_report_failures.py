"""Background warm-ups import one at a time and report a failure as a warning.

A heavy package left half imported by one thread is imported again by the
next thread that needs it, and torch refuses a second start: "Only a single
TORCH_LIBRARY can be used to register the namespace triton" (the Make
Masks role-colour test, 2026-11-01). Make Masks' cold-import warm-up ran
beside the other warm-ups without the heavy-import lock, and every warm-up
logged its failure at debug level, where a test run never shows it.

These tests build a small circular package rather than importing torch, so
the timing is theirs to control. The package is not one the import gate
knows, so only the heavy-import lock keeps the two warm-ups apart.
"""
from __future__ import annotations

import importlib
import logging
import sys
import textwrap
import threading
import uuid
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_module  # noqa: E402,F401
from spacr.qt import bridge  # noqa: E402
from spacr.qt.screens import make_masks  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture
def modules(tmp_path, monkeypatch):
    """A torchvision-shaped circular package, a warm module, an entry module
    and a module that fails on import, all ungated."""
    tag = uuid.uuid4().hex[:8]
    names = {"cyc": f"wcyc_{tag}", "warm": f"wwarm_{tag}",
             "entry": f"wentry_{tag}", "signal": f"wsignal_{tag}",
             "broken": f"wbroken_{tag}"}
    cyc, signal = names["cyc"], names["signal"]
    _write(tmp_path, f"{signal}.py", "import threading\nentered = threading.Event()\n")
    _write(tmp_path, f"{cyc}/__init__.py", f"""
        import time
        import {signal}
        {signal}.entered.set()
        time.sleep(0.4)
        from . import models
        from .ops import nms
        READY = True
        """)
    _write(tmp_path, f"{cyc}/models.py", "from .ops import misc\n")
    _write(tmp_path, f"{cyc}/ops/__init__.py", "from .boxes import nms\nfrom . import misc\n")
    _write(tmp_path, f"{cyc}/ops/misc.py", "X = 1\n")
    _write(tmp_path, f"{cyc}/ops/boxes.py", f"import {cyc}.models\ndef nms():\n    return 1\n")
    _write(tmp_path, f"{names['warm']}.py", f"import {cyc}\n")
    _write(tmp_path, f"{names['entry']}.py",
           f"from {cyc}.ops.boxes import nms\nimport {cyc}\nassert {cyc}.READY\n")
    _write(tmp_path, f"{names['broken']}.py", "raise RuntimeError('half imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield names
    for name in list(sys.modules):
        if name.split(".")[0] in names.values():
            sys.modules.pop(name, None)


@pytest.fixture
def warnings_seen(caplog):
    """WARNING records of the bridge's logger, read from the logger itself.

    The ``spacr`` logger may keep its records from the root logger once
    spaCR's log file is configured, so the handler goes on the logger.
    Read live, since the handler's list is replaced between test phases,
    and once each, since the root logger may pass the same record on too.
    """
    caplog.set_level(logging.WARNING, logger=bridge.LOG.name)
    bridge.LOG.addHandler(caplog.handler)
    try:
        yield lambda: list({id(record): record for record in caplog.records
                            if record.name == bridge.LOG.name
                            and record.levelno >= logging.WARNING}.values())
    finally:
        bridge.LOG.removeHandler(caplog.handler)


def test_the_cold_import_warm_waits_for_another_warm_inside_a_package(
        modules, monkeypatch, warnings_seen):
    """Without the shared lock the two warm-ups break the package between them."""
    monkeypatch.setattr(make_masks, "_COLD_IMPORTS", (modules["entry"],))
    monkeypatch.setattr(make_masks, "_WARM_LANGUAGE", [])
    first = threading.Thread(target=bridge._import_the_entry_module,
                             args=(modules["warm"],))
    first.start()
    assert importlib.import_module(modules["signal"]).entered.wait(10)
    cold = threading.Thread(target=make_masks._import_the_cold_modules)
    cold.start()
    first.join(20)
    cold.join(20)

    assert not first.is_alive() and not cold.is_alive()
    assert [record.getMessage() for record in warnings_seen()] == []
    package = sys.modules[modules["cyc"]]
    assert package.READY is True
    assert sys.modules[modules["cyc"] + ".ops"].nms() == 1
    assert modules["entry"] in sys.modules
    assert modules["warm"] in sys.modules
    assert not app_module.HEAVY_IMPORT_LOCK.locked()
    assert bridge._BACKGROUND_IMPORTERS == {}


def test_a_failed_warm_is_a_warning_with_its_traceback(modules, warnings_seen):
    broken = modules["broken"]
    assert bridge._warm_import(broken) is False
    bridge._import_the_entry_module(broken)

    failures = [record for record in warnings_seen()
                if broken in record.getMessage()]
    assert len(failures) == 2
    for record in failures:
        assert record.exc_info is not None
        assert "half imported" in str(record.exc_info[1])
    assert not app_module.HEAVY_IMPORT_LOCK.locked()
    assert bridge._BACKGROUND_IMPORTERS == {}


def test_the_cold_import_warm_reports_a_failure_and_goes_on(
        modules, monkeypatch, warnings_seen):
    monkeypatch.setattr(make_masks, "_COLD_IMPORTS",
                        (modules["broken"], modules["signal"]))
    monkeypatch.setattr(make_masks, "_WARM_LANGUAGE", [])
    make_masks._import_the_cold_modules()
    assert any(modules["broken"] in record.getMessage()
               and record.exc_info is not None for record in warnings_seen())
    assert modules["signal"] in sys.modules


def test_a_good_warm_imports_and_says_so(modules, warnings_seen):
    assert bridge._warm_import(modules["signal"]) is True
    assert modules["signal"] in sys.modules
    assert warnings_seen() == []
