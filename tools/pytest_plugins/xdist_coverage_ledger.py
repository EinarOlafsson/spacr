"""pytest plugin: record whose coverage data a pytest-xdist session lost.

Loaded by ``tools/run_coverage_batches.py`` as ``-p xdist_coverage_ledger``
(item 288).  It does nothing unless ``SPACR_COVERAGE_LEDGER`` names a file,
and nothing on an xdist worker: only the controller sees every worker.

WHY IT EXISTS.  pytest-cov's workers save their coverage data when their
session FINISHES.  A worker that dies by a signal -- the segfault family of
item 43 -- never gets there, so every line it measured is gone, including the
lines of tests it had already reported as PASSED.  xdist replaces the worker,
reschedules what it had not started, and the run's coverage silently becomes
the survivors' coverage.  Measured on a toy package (2026-09-15): a test
passed on gw0, gw0 segfaulted in a later test, and the combined data held
none of the first test's lines.

WHAT IT WRITES, at the end of the controller's session, as JSON:

  workers           per worker id: the test files it reported on, whether it
                    went down, its error, whether pytest-cov's
                    ``cov_worker_node_id`` came back (the one proof that its
                    data file was written) and, when it did not, the worker
                    process's exit status.  xdist says "Not properly
                    terminated" for every dead worker; the status is what
                    tells a segfault (-11, SIGSEGV) from tests/conftest.py's
                    memory guard (3, ``os._exit(3)``)
  crashes           every (worker, test id) xdist reported as crashed
  unreported_files  test files with a collected test that no worker ever
                    reported, e.g. after "maximum crashed workers reached"

The batch runner turns that into the list of files to re-run.  A missing
ledger is itself evidence -- the controller did not finish -- and the runner
treats every file of that batch as lost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

LEDGER_ENV = "SPACR_COVERAGE_LEDGER"
LEDGER_SCHEMA = "spacr.xdist-coverage-ledger/v1"
CONTROLLER = "controller"


def _file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _worker_id(node: Any) -> str:
    gateway = getattr(node, "gateway", None)
    return str(getattr(gateway, "id", None) or CONTROLLER)


def _exit_status(node: Any) -> int | None:
    """The dead worker's exit status, from execnet's handle on its process.

    ``gateway._io.popen`` is execnet's, not a public API, so every step is
    guarded: without it the ledger still says the data was lost, only not
    how.  Asked only of a worker whose coverage did not come back, which has
    already closed its channel, so the wait does not block a live worker.
    """
    io = getattr(getattr(node, "gateway", None), "_io", None)
    popen = getattr(io, "popen", None)
    if popen is None:
        return None
    try:
        return int(popen.wait(timeout=10))
    except Exception:  # noqa: BLE001 -- TimeoutExpired, or a changed execnet
        return None


class CoverageLedger:
    """Controller-side record of which files each worker's data depends on."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.workers: dict[str, dict[str, Any]] = {}
        self.crashes: list[dict[str, str]] = []
        self.collected: set[str] = set()
        self.reported: set[str] = set()

    def _worker(self, worker: str) -> dict[str, Any]:
        return self.workers.setdefault(worker, {
            "files": set(),
            "down": False,
            "error": None,
            "coverage_returned": None,
            "exit_status": None,
        })

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: Any) -> None:
        self._worker(_worker_id(node))
        self.collected.update(str(nodeid) for nodeid in ids)

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        worker = _worker_id(getattr(report, "node", None))
        self._worker(worker)["files"].add(_file_of(report.nodeid))
        self.reported.add(report.nodeid)

    @pytest.hookimpl(optionalhook=True)
    def pytest_handlecrashitem(self, crashitem: str, report: Any, sched: Any) -> None:
        self.crashes.append({
            "worker": _worker_id(getattr(report, "node", None)),
            "nodeid": str(crashitem),
        })

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        entry = self._worker(_worker_id(node))
        output = getattr(node, "workeroutput", None) or {}
        entry["down"] = True
        entry["error"] = None if error is None else str(error)
        entry["coverage_returned"] = "cov_worker_node_id" in output
        if not entry["coverage_returned"]:
            entry["exit_status"] = _exit_status(node)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        unreported = sorted({
            _file_of(nodeid) for nodeid in self.collected - self.reported
        })
        document = {
            "schema": LEDGER_SCHEMA,
            "exitstatus": int(exitstatus),
            "workers": {
                worker: {**entry, "files": sorted(entry["files"])}
                for worker, entry in sorted(self.workers.items())
            },
            "crashes": list(self.crashes),
            "unreported_files": unreported,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        partial.replace(self.path)


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, "workerinput"):
        return
    target = os.environ.get(LEDGER_ENV)
    if not target:
        return
    config.pluginmanager.register(
        CoverageLedger(Path(target)), "spacr-xdist-coverage-ledger",
    )
