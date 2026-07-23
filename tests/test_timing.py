"""Tests for the timing utilities in spacr.logging_util."""
from __future__ import annotations

import logging
import time

import pytest


# ---------------------------------------------------------------------------
# Test isolation: reset timing enable + threshold per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_timing():
    from spacr import logging_util as lu
    lu.enable_timing()
    lu.set_timing_threshold_ms(0)   # log everything, even 0 ms calls
    yield
    lu.enable_timing()
    lu.set_timing_threshold_ms(5)


# ---------------------------------------------------------------------------
# @timed decorator
# ---------------------------------------------------------------------------

def test_timed_bare_decorator_logs_elapsed(caplog):
    from spacr.logging_util import timed

    @timed
    def slow_add(a, b):
        time.sleep(0.005)
        return a + b

    with caplog.at_level(logging.INFO):
        assert slow_add(2, 3) == 5

    # At least one INFO record mentioning "took" + "ms"
    matches = [r for r in caplog.records
               if "took" in r.message and "ms" in r.message]
    assert matches, f"expected a timing log; got {caplog.records!r}"


def test_timed_with_custom_name_appears_in_log(caplog):
    from spacr.logging_util import timed

    @timed(name="my.pipeline.step")
    def _work():
        return "ok"

    with caplog.at_level(logging.INFO):
        _work()
    assert any("my.pipeline.step" in r.message for r in caplog.records)


def test_timed_respects_threshold(caplog):
    from spacr.logging_util import timed, set_timing_threshold_ms

    @timed
    def _instant():
        pass

    set_timing_threshold_ms(10_000)   # 10 s — nothing this fast crosses
    with caplog.at_level(logging.INFO):
        _instant()
    assert not any("took" in r.message for r in caplog.records)


def test_timed_pass_through_when_disabled(caplog):
    from spacr.logging_util import timed, disable_timing

    @timed
    def _work():
        return 42

    disable_timing()
    with caplog.at_level(logging.INFO):
        assert _work() == 42
    assert not any("took" in r.message for r in caplog.records)


def test_timed_still_logs_on_exception(caplog):
    from spacr.logging_util import timed

    @timed
    def _boom():
        raise ValueError("bang")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            _boom()
    assert any("took" in r.message for r in caplog.records)


def test_timed_marks_wrapped_function(caplog):
    """The wrapper carries __spacr_timed__ so time_module can skip
    already-wrapped functions on subsequent calls."""
    from spacr.logging_util import timed

    @timed
    def _work():
        pass

    assert getattr(_work, "__spacr_timed__", False) is True
    # And preserves the original via __wrapped__ for introspection
    assert callable(_work.__wrapped__)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

def test_timer_context_manager_logs(caplog):
    from spacr.logging_util import Timer

    with caplog.at_level(logging.INFO):
        with Timer("my block") as t:
            time.sleep(0.003)
    assert any("my block took" in r.message for r in caplog.records)
    assert t.elapsed_ms is not None and t.elapsed_ms >= 0


def test_timer_disabled_context_no_log(caplog):
    from spacr.logging_util import Timer, disable_timing

    disable_timing()
    with caplog.at_level(logging.INFO):
        with Timer("silent"):
            pass
    assert not any("silent" in r.message for r in caplog.records)


def test_timer_nested_blocks_log_independently(caplog):
    from spacr.logging_util import Timer

    with caplog.at_level(logging.INFO):
        with Timer("outer"):
            with Timer("inner"):
                time.sleep(0.001)
    log_text = "\n".join(r.message for r in caplog.records)
    assert "outer" in log_text
    assert "inner" in log_text


# ---------------------------------------------------------------------------
# time_module bulk wrapper
# ---------------------------------------------------------------------------

def test_time_module_wraps_public_functions(caplog):
    """Create a dummy module in memory, wrap it, check every public
    function got the __spacr_timed__ marker."""
    import types
    mod = types.ModuleType("spacr._test_time_mod")

    def public_a(x): return x + 1
    def public_b(x): return x * 2
    def _private(x): return x
    class Klass: pass

    # These need to look like they live in the module
    for f in (public_a, public_b, _private):
        f.__module__ = mod.__name__
    mod.public_a = public_a
    mod.public_b = public_b
    mod._private = _private
    mod.Klass = Klass

    from spacr.logging_util import time_module
    n = time_module(mod)
    assert n == 2   # 2 public funcs wrapped; underscore + class skipped
    assert getattr(mod.public_a, "__spacr_timed__", False) is True
    assert getattr(mod.public_b, "__spacr_timed__", False) is True
    # Private stays untouched
    assert not getattr(mod._private, "__spacr_timed__", False)


def test_time_module_is_idempotent():
    import types
    mod = types.ModuleType("spacr._test_time_mod2")
    def f(): pass
    f.__module__ = mod.__name__
    mod.f = f

    from spacr.logging_util import time_module
    assert time_module(mod) == 1
    # Second call is a no-op — already marked
    assert time_module(mod) == 0


def test_time_module_exclude_skips_named_functions():
    import types
    mod = types.ModuleType("spacr._test_time_mod3")
    def a(): pass
    def b(): pass
    a.__module__ = mod.__name__
    b.__module__ = mod.__name__
    mod.a = a
    mod.b = b

    from spacr.logging_util import time_module
    time_module(mod, exclude=("a",))
    assert not getattr(mod.a, "__spacr_timed__", False)
    assert getattr(mod.b, "__spacr_timed__", False) is True
