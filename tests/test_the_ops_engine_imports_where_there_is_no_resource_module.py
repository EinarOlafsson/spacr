"""Windows has no `resource`, and the engine still has to import there.

The module used it for one line of the well report: peak resident memory.
Importing it at module scope meant the whole OPS engine failed to import on
Windows, which is two smoke-test failures and, for a user, an application
that cannot open the module at all for the sake of a diagnostic number.
"""
import importlib
import sys

import spacr.ops_engine as ops_engine


def test_a_platform_without_the_resource_module_still_imports_the_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    try:
        reloaded = importlib.reload(ops_engine)
        assert reloaded.resource is None
        assert reloaded._peak_rss_gb() == 0.0
    finally:
        monkeypatch.undo()
        importlib.reload(ops_engine)


def test_where_the_platform_can_report_it_the_peak_is_a_real_measurement():
    assert ops_engine.resource is not None
    assert ops_engine._peak_rss_gb() > 0.0
