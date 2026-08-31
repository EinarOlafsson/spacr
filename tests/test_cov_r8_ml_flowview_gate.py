"""`_flowview_event` -- reaching optional tracing without importing it.

The gate exists so that a Classify run does not import the FlowView
package merely to discover that tracing is switched off. Everything here
is a branch that only runs when tracing is unavailable, disabled, or
broken -- which is the ordinary case, and therefore the case that gets
no coverage from a suite where FlowView is installed and working.

The rule every one of these serves: an observation layer must never
change scientific output. A gate that raised would turn a tracing
problem into a failed run, and `_flowview_pipeline` re-raises the
SCIENTIFIC error after reporting, never the reporting error.
"""
from __future__ import annotations

import sys

import pytest

import spacr.ml as ml


@pytest.fixture(autouse=True)
def _no_ambient_trace(monkeypatch):
    """Start from "the trace module has not been imported"."""
    monkeypatch.delitem(sys.modules, "spacr.flowview.trace", raising=False)
    monkeypatch.delenv("SPACR_FLOWVIEW", raising=False)


class TestTheGateWhenTracingHasNotBeenImported:

    def test_an_unset_environment_variable_keeps_it_out(self):
        """The point of the gate: no import, no cost, no tracing."""
        assert ml._flowview_event("begin", {}, "classify") is False
        assert "spacr.flowview.trace" not in sys.modules

    @pytest.mark.parametrize("value", ["", "0", "off", "false", "no",
                                       "maybe", "  "])
    def test_a_value_that_is_not_a_yes_keeps_it_out(self, monkeypatch, value):
        monkeypatch.setenv("SPACR_FLOWVIEW", value)
        assert ml._flowview_event("begin", {}, "classify") is False

    @pytest.mark.parametrize("value", ["1", "on", "true", "yes",
                                       "TRUE", " Yes "])
    def test_a_yes_is_read_whatever_its_case_or_spacing(self, monkeypatch,
                                                        value):
        """The env value is stripped and case-folded before it is judged.

        This drives the import branch -- what it returns depends on the
        machine, and the assertion is that asking did not RAISE.
        """
        monkeypatch.setenv("SPACR_FLOWVIEW", value)
        assert ml._flowview_event("begin", {}, "classify") in (True, False)

    def test_an_import_that_fails_is_answered_no(self, monkeypatch):
        """FlowView is an optional extra; not having it is not an error."""
        import builtins

        monkeypatch.setenv("SPACR_FLOWVIEW", "1")
        real_import = builtins.__import__

        def refuse(name, globals=None, locals=None, fromlist=(), level=0):
            if "flowview" in name or "trace" in (fromlist or ()):
                raise ImportError("flowview is not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert ml._flowview_event("begin", {}, "classify") is False


class TestTheGateWhenTracingIsThereButUnusable:

    def test_a_module_that_says_it_is_disabled_is_answered_no(self,
                                                              monkeypatch):
        import types

        module = types.ModuleType("spacr.flowview.trace")
        module.is_enabled = lambda: False
        monkeypatch.setitem(sys.modules, "spacr.flowview.trace", module)
        assert ml._flowview_event("begin", {}, "classify") is False

    def test_a_module_that_raises_when_asked_is_answered_no(self,
                                                            monkeypatch):
        """`except BaseException` -- the outer guard, and it is broad
        on purpose: a KeyboardInterrupt during a run must not be turned
        into a tracing exception either."""
        import types

        module = types.ModuleType("spacr.flowview.trace")

        def explode():
            raise RuntimeError("the tracer is in a bad state")

        module.is_enabled = explode
        monkeypatch.setitem(sys.modules, "spacr.flowview.trace", module)
        assert ml._flowview_event("begin", {}, "classify") is False

    def test_an_unknown_action_is_answered_no_rather_than_raising(
            self, monkeypatch):
        """The action names a `_<action>` function on `_classify_stages`.

        One that is not there must read as "not traced", not as an
        AttributeError out of a pipeline.
        """
        import types

        module = types.ModuleType("spacr.flowview.trace")
        module.is_enabled = lambda: True
        monkeypatch.setitem(sys.modules, "spacr.flowview.trace", module)
        assert ml._flowview_event("no_such_action") is False


class TestThePipelineDecorator:
    """`_flowview_pipeline` reports, and never changes the result."""

    def test_a_failing_function_reports_the_failure_and_re_raises(
            self, monkeypatch):
        """THE UNCOVERED LINE.

        The scientific error is what the caller must see. Reporting is a
        side effect that happens on the way past.
        """
        events = []

        def fake_event(action, *args):
            events.append((action, args))
            return action == "begin"        # "active" only for begin

        monkeypatch.setattr(ml, "_flowview_event", fake_event)

        @ml._flowview_pipeline("classify")
        def explodes(_settings):
            raise ValueError("the model did not converge")

        with pytest.raises(ValueError, match="did not converge"):
            explodes({"a": 1})

        assert [a for a, _ in events] == ["begin", "fail"]
        assert isinstance(events[1][1][0], ValueError), (
            "the scientific error was not the one reported")

    def test_a_failure_while_inactive_reports_nothing(self, monkeypatch):
        """If the graph never began there is nothing to fail."""
        events = []
        monkeypatch.setattr(
            ml, "_flowview_event",
            lambda action, *args: events.append(action) or False)

        @ml._flowview_pipeline("classify")
        def explodes(_settings):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            explodes({})
        assert events == ["begin"], "it reported a failure it never began"

    def test_a_successful_function_returns_its_own_result(self,
                                                          monkeypatch):
        events = []

        def fake_event(action, *args):
            events.append(action)
            return action == "begin"

        monkeypatch.setattr(ml, "_flowview_event", fake_event)

        @ml._flowview_pipeline("classify")
        def works(_settings):
            return {"score": 0.9}

        assert works({}) == {"score": 0.9}
        assert events == ["begin", "finish"]

    def test_the_settings_are_read_from_a_keyword_too(self, monkeypatch):
        """`args[0] if args else kwargs.get("settings")` -- callers use both."""
        seen = []

        def fake_event(action, *args):
            if action == "begin":
                seen.append(args[0])
            return False

        monkeypatch.setattr(ml, "_flowview_event", fake_event)

        @ml._flowview_pipeline("classify")
        def works(settings=None):
            return settings

        works(settings={"from": "keyword"})
        assert seen == [{"from": "keyword"}]
