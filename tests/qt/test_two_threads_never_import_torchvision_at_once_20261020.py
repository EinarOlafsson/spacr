"""A background warm and a GUI-thread import never initialise one package together.

Make Masks warms ``spacr.io`` on a worker as a folder opens, Mask preloads
``spacr.core`` three seconds after it opens, and the GUI thread imports
``spacr.object`` when a screen validates its settings. torchvision is a
circular package: a second thread arriving inside it while the first is
still initialising it leaves one of them a partially initialised module,
``cannot import name 'batched_nms'`` (coverage shard 1, 2026-10-20).

Every background import registers itself (``_importing_in_the_background``)
and any other thread is held at a gated module by ``_HeavyImportGate`` until
it finishes. These tests build a small circular package with the same shape
rather than importing torch, so the timing is theirs to control.
"""
from __future__ import annotations

import ast
import importlib
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt import bridge  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture
def circular(tmp_path, monkeypatch):
    """A torchvision-shaped circular package, a warm module and an entry module.

    ``<cyc>/__init__`` signals that it has started, waits, and then imports
    ``models``, which imports ``ops``, whose ``boxes`` imports ``models``
    again: the cycle torchvision has. The warm module imports the package,
    the entry module (``spacr.object``'s part) imports ``<cyc>.ops.boxes``.
    """
    tag = uuid.uuid4().hex[:8]
    cyc, warm, entry, signal = (f"cyc_{tag}", f"warm_{tag}", f"entry_{tag}",
                                f"signal_{tag}")
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
    _write(tmp_path, f"{warm}.py", f"import {cyc}\n")
    _write(tmp_path, f"{entry}.py", f"from {cyc}.ops.boxes import nms\nimport {cyc}\nassert {cyc}.READY\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bridge, "_GATED_IMPORT_ROOTS",
                        bridge._GATED_IMPORT_ROOTS | {cyc, entry})
    names = {"cyc": cyc, "warm": warm, "entry": entry, "signal": signal}
    yield names
    for name in list(sys.modules):
        if name.split(".")[0] in names.values():
            sys.modules.pop(name, None)


def _import(name, errors, where):
    try:
        importlib.import_module(name)
    except Exception as error:                               # noqa: BLE001
        errors.append((where, error))


def test_without_the_gate_the_two_imports_break_each_other(circular):
    """The hazard is real: the same two imports, neither registered, fail."""
    errors = []
    warm = threading.Thread(target=_import, args=(circular["warm"], errors, "warm"))
    warm.start()
    signal = importlib.import_module(circular["signal"])
    assert signal.entered.wait(10)
    _import(circular["entry"], errors, "main")
    warm.join(10)
    assert errors, "the circular package survived two threads; this test proves nothing"


def test_a_warm_and_a_main_thread_import_both_finish_with_the_whole_package(circular):
    """The warm runs through the real background import; the GUI thread waits for it."""
    warm = threading.Thread(target=bridge._import_the_entry_module,
                            args=(circular["warm"],))
    warm.start()
    signal = importlib.import_module(circular["signal"])
    assert signal.entered.wait(10)
    assert bridge._BACKGROUND_IMPORTERS, "the warm did not register itself"

    errors = []
    _import(circular["entry"], errors, "main")
    warm.join(10)

    assert errors == []
    assert not warm.is_alive()
    package = sys.modules[circular["cyc"]]
    assert package.READY is True
    assert sys.modules[circular["cyc"] + ".ops"].nms() == 1
    assert not getattr(package.__spec__, "_initializing", False)
    assert sys.modules[circular["warm"]] is not None
    assert bridge._BACKGROUND_IMPORTERS == {}


def test_a_warm_that_needs_the_module_the_gui_thread_holds_does_not_deadlock(
        tmp_path, monkeypatch):
    """Waiting stops the moment the warm blocks on a lock the waiter holds."""
    tag = uuid.uuid4().hex[:8]
    shared, warm, signal = f"shared_{tag}", f"warmshared_{tag}", f"sig_{tag}"
    _write(tmp_path, f"{signal}.py", "import threading\nentered = threading.Event()\n")
    _write(tmp_path, f"{warm}.py", f"""
        import time
        import {signal}
        {signal}.entered.set()
        time.sleep(0.2)
        import {shared}
        """)
    _write(tmp_path, f"{shared}.py", "VALUE = 7\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bridge, "_GATED_IMPORT_ROOTS",
                        bridge._GATED_IMPORT_ROOTS | {shared})
    try:
        thread = threading.Thread(target=bridge._import_the_entry_module, args=(warm,))
        thread.start()
        assert importlib.import_module(signal).entered.wait(10)
        started = time.monotonic()
        module = importlib.import_module(shared)
        thread.join(10)
        assert time.monotonic() - started < bridge._GATE_WAIT_S / 2
        assert module.VALUE == 7
        assert not thread.is_alive()
        assert sys.modules[warm] is not None
    finally:
        for name in (shared, warm, signal):
            sys.modules.pop(name, None)


class _Lock:
    def __init__(self, owner):
        self.owner = owner


def test_the_deadlock_check_follows_the_import_systems_record():
    """Direct and chained waits on this thread count; anything else does not."""
    me = 100
    assert bridge._held_module_lock_blocks([1], me, {1: [_Lock(me)]}) is True
    assert bridge._held_module_lock_blocks([1], me, {1: _Lock(me)}) is True
    assert bridge._held_module_lock_blocks(
        [1], me, {1: [_Lock(2)], 2: [_Lock(me)]}) is True
    assert bridge._held_module_lock_blocks(
        [1], me, {1: [_Lock(3)], 3: [_Lock(1)]}) is False
    assert bridge._held_module_lock_blocks([1], me, {1: [_Lock(None)]}) is False
    assert bridge._held_module_lock_blocks([1], me, {}) is False
    assert bridge._held_module_lock_blocks([1], me, object()) is True
    assert bridge._held_module_lock_blocks([1], me) in (True, False)


def test_a_wait_that_never_ends_gives_up_and_imports_beside_it(monkeypatch, caplog):
    """A registered import that never finishes costs at most the bound."""
    never = threading.Event()
    monkeypatch.setattr(bridge, "_GATE_WAIT_S", 0.05)
    monkeypatch.setattr(bridge, "_held_module_lock_blocks", lambda *_a: False)
    monkeypatch.setitem(bridge._BACKGROUND_IMPORTERS, -1, never)
    assert bridge._wait_for_background_imports() is False
    assert "still running" in caplog.text


def test_a_waiter_the_warm_is_blocked_on_stops_at_once(monkeypatch):
    monkeypatch.setattr(bridge, "_held_module_lock_blocks", lambda *_a: True)
    monkeypatch.setitem(bridge._BACKGROUND_IMPORTERS, -1, threading.Event())
    assert bridge._wait_for_background_imports() is False


def test_nothing_running_means_no_wait_and_ungated_names_pass():
    assert bridge._wait_for_background_imports() is True
    assert bridge._HeavyImportGate.find_spec("json") is None
    assert bridge._HeavyImportGate.find_spec("torch.nn") is None


def test_a_nested_registration_leaves_the_inner_ones_entry_to_the_inner_one():
    tid = threading.get_ident()
    with bridge._importing_in_the_background():
        outer = bridge._BACKGROUND_IMPORTERS[tid]
        with bridge._importing_in_the_background():
            assert bridge._BACKGROUND_IMPORTERS[tid] is not outer
        assert tid not in bridge._BACKGROUND_IMPORTERS
    assert outer.is_set()
    assert sum(finder is bridge._HeavyImportGate for finder in sys.meta_path) == 1


def _top_level_imports(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in tree.body:
        for statement in (node.body if isinstance(node, ast.Try) else [node]):
            if isinstance(statement, ast.Import):
                found.update(alias.name for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                if statement.level:
                    base = "spacr" + (f".{statement.module}" if statement.module else "")
                    found.add(base)
                    if not statement.module:
                        found.update(f"spacr.{alias.name}" for alias in statement.names)
                else:
                    found.add(statement.module or "")
    return found


def test_every_spacr_module_that_imports_the_stack_at_load_is_gated():
    """The gated spaCR modules are exactly those that bring torch in on import."""
    package = Path(bridge.__file__).resolve().parents[1]
    imports = {f"spacr.{path.stem}": _top_level_imports(path)
               for path in package.glob("*.py") if path.stem != "__init__"}
    heavy: set = set()
    grew = True
    while grew:
        grew = False
        for name, found in imports.items():
            if name not in heavy and any(
                    item.split(".")[0] in bridge._GATED_IMPORT_ROOTS or item in heavy
                    for item in found):
                heavy.add(name)
                grew = True
    assert heavy == set(bridge._GATED_SPACR_MODULES)
