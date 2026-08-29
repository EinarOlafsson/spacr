"""A PEAK row appears only for the quantity that was actually measured.

`describe_resources` prints one PEAK line per quantity, and each of them is
conditional on that quantity having a reading. The interesting machine is the
one that answers for only one of the two: a container with no `/proc` and no
psutil still gets a GPU number out of an already-imported Torch, and the
resulting table must name the GPU peak while claiming no resident peak at
all. "Nothing was using memory" and "nobody measured" are opposite findings,
so a PEAK line that appears anyway would be reporting the wrong one.
"""

from spacr.fit_resources import RESOURCE_KEY, describe_resources, peak


def test_a_run_measured_only_on_the_gpu_names_the_gpu_peak_and_no_resident_peak():
    """With every resident reading unavailable, only the GPU peak is claimed."""
    settings = {RESOURCE_KEY: [
        {"stage": "build design", "rss": None, "gpu": 2 * 1024 ** 3},
        {"stage": "optimise", "rss": None, "gpu": 5 * 1024 ** 3},
        {"stage": "write results", "rss": None, "gpu": 1 * 1024 ** 3},
    ]}

    table = describe_resources(settings)

    assert peak(settings) == {"gpu": 5 * 1024 ** 3, "gpu_stage": "optimise"}
    assert "PEAK GPU      5.0 GB at 'optimise'" in table
    assert "PEAK resident" not in table
    # Every stage row still says the resident reading was never taken, which
    # is what distinguishes it from a stage that used no memory.
    assert table.count("not measured") == 3


def test_a_run_that_measured_nothing_still_lists_its_stages_and_claims_no_peak():
    """A table with no numbers in it is a table of stages, not an empty string."""
    settings = {RESOURCE_KEY: [
        {"stage": "load", "rss": None, "gpu": None},
        {"stage": "fit", "rss": None, "gpu": None},
    ]}

    table = describe_resources(settings)

    assert peak(settings) == {}
    assert "PEAK" not in table
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[1].split()[0] == "load"
    assert lines[2].split()[0] == "fit"
