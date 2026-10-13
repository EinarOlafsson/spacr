"""A fit records what it costs, per stage. Instruction 160.

Filed after two regressions in a row made the machine unresponsive twice. That
report could not be acted on because nothing recorded a number: a hung machine
is not a hung application, and telling memory exhaustion from a driver fault
needs measurements taken WHILE the fit runs.
"""

import spacr


import pytest

from spacr.fit_resources import (RESOURCE_KEY, STAGE_KEY, describe_resources,
                                 gpu_allocated, host_rss, peak, readable,
                                 record_stage)


def test_resident_memory_is_measurable_here():
    assert host_rss() and host_rss() > 0


def test_one_call_records_the_stage_and_its_cost():
    """A stage without its cost is the state this was filed about, and a cost
    without its stage cannot say where the fit was when it grew."""
    settings = {}
    record_stage(settings, "building the design")
    assert settings[STAGE_KEY] == "building the design"
    assert len(settings[RESOURCE_KEY]) == 1
    assert settings[RESOURCE_KEY][0]["stage"] == "building the design"


def test_the_peak_names_where_it_was_taken():
    settings = {}
    record_stage(settings, "small")
    settings[RESOURCE_KEY].append({"stage": "huge", "rss": 10**12, "gpu": None})
    high = peak(settings)
    assert high["rss"] == 10**12
    assert high["rss_stage"] == "huge"


def test_nothing_measured_is_not_zero():
    """"Nothing was using memory" and "nobody measured" are opposite findings."""
    assert readable(None) == "not measured"
    assert peak({RESOURCE_KEY: [{"stage": "s", "rss": None, "gpu": None}]}) == {}


def test_gpu_is_not_asked_by_importing_torch(monkeypatch):
    """Importing torch to take a measurement would make the measurement the
    most expensive thing in the stage."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    assert gpu_allocated() is None


def test_the_table_reads_as_a_table():
    settings = {}
    record_stage(settings, "reading the counts")
    record_stage(settings, "fitting")
    text = describe_resources(settings)
    assert "stage" in text and "resident" in text
    assert "reading the counts" in text and "fitting" in text
    assert "PEAK resident" in text


def test_nothing_recorded_gives_no_table():
    assert describe_resources({}) == ""


def test_the_recorder_never_raises():
    """A measurement that can fail the run it measures is worse than none."""
    class Hostile:
        def __setitem__(self, key, value):
            raise RuntimeError("no")

        def setdefault(self, *a):
            raise RuntimeError("no")

        def get(self, *a):
            raise RuntimeError("no")

    record_stage(Hostile(), "x")
    assert peak(Hostile()) == {}
    assert describe_resources(Hostile()) == ""


def test_the_failure_report_carries_the_readings():
    """A failure that ran out of memory looks identical to one that did not,
    unless the readings are beside it."""
    from spacr.regression_failure import describe_failure

    settings = {"regression_type": "mixed"}
    record_stage(settings, "fitting the mixed model")
    try:
        raise MemoryError("out of memory")
    except Exception as error:
        text = describe_failure(error, stage="fitting the mixed model",
                                settings=settings, include_traceback=False)
    assert "WHAT IT COST, PER STAGE" in text
    assert "fitting the mixed model" in text


def test_the_fit_records_stages(monkeypatch):
    from spacr import ml

    settings = {}
    ml._stage(settings, "reading the counts")
    assert settings["_regression_stage"] == "reading the counts"
    assert settings["_regression_resources"][0]["rss"] is not None
