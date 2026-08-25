"""A misbehaving ``nvidia-smi`` must not be able to break the usage panel.

``GPUtil.getGPUs()`` shells out to ``nvidia-smi`` and ``int()``s whatever
comes back, so a driver/library version mismatch surfaces as

    ValueError: invalid literal for int() with base 10:
    'Failed to initialize NVML: Driver/library version mismatch'

and it is not a one-off: whatever raises it raises it again on every tick of
the two-second usage timer, for the life of the session. The panel is
decoration. The run it decorates is not, and neither is the window.

WHERE THIS LOOKS NOW. The panel used to be built and polled inline on the
GUI thread, so the exception landed in the toolkit's callback handler once a
second, and the guard it needed was a try/except plus a latch that switched
the poll off after the first failure. The panel is now sampled on a worker
by :func:`spacr.qt.screens.app_screen._sample_usage` -- a plain function
that takes no widgets and returns a dict -- and painted on the GUI thread by
``_apply_usage``. That is why these are calls rather than greps over source
text: the failure can be produced instead of described.

That the sampling happens off the GUI thread at all, that an overlapping
tick is skipped rather than queued, and that closing a module mid-poll
leaves no thread behind are asserted in
``tests/qt/test_gui_responsiveness.py``. This file is only about what a
broken driver does.
"""
import sys
import types

import pytest

# The exact text nvidia-smi hands back on a driver/library mismatch, fed
# through the same int() that raises on it in GPUtil.
MISMATCH = ("Failed to initialize NVML: Driver/library version mismatch")


@pytest.fixture
def broken_driver(monkeypatch):
    """Install a GPUtil whose poll fails the way a real mismatch fails."""
    from spacr.qt.screens import app_screen

    def _raise():
        raise ValueError(
            f"invalid literal for int() with base 10: '{MISMATCH}'")

    fake = types.ModuleType("GPUtil")
    fake.getGPUs = _raise
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    # The host has an nvidia-smi -- that is the whole point. A machine
    # without one never enters GPUtil at all, which is a different case and
    # is covered in tests/qt/test_cov_app_screen.py.
    monkeypatch.setattr(app_screen, "_nvidia_smi_available", lambda: True)
    return app_screen


def test_a_driver_mismatch_does_not_escape_the_sampler(broken_driver):
    """The sample comes back. Whatever nvidia-smi did, it stays in here."""
    sample = broken_driver._sample_usage(False)
    assert isinstance(sample, dict)


def test_ram_and_cpu_readings_survive_a_dead_gpu(broken_driver):
    """The GPU is one of four things this panel shows, not all of them.

    An unguarded GPU poll -- or a guard that returned out of the sampler on
    failure -- would take the RAM and CPU bars down with it, on a machine
    where the only thing wrong is a driver that needs a reboot.
    """
    sample = broken_driver._sample_usage(True)
    assert 0 <= sample["ram"] <= 100
    assert 0 <= sample["cpu"] <= 100
    assert len(sample["per_core"]) >= 1


def test_a_broken_driver_is_not_reported_as_an_idle_gpu(broken_driver):
    """A reading that failed is left out, not turned into a plausible 0%.

    ``_apply_usage`` paints only the keys it is given, so an omitted GPU
    reading leaves the last real number on the bar. Reporting zero instead
    would draw a fully idle GPU -- the same picture as a card doing nothing,
    on a machine that cannot see its card at all.
    """
    sample = broken_driver._sample_usage(False)
    assert "gpu" not in sample
    assert "vram" not in sample


def test_a_permanently_broken_driver_stays_quiet_on_every_tick(
        broken_driver, capsys):
    """The failure repeats every two seconds; the complaint must not.

    The Tk panel printed its diagnosis and then latched itself off so the
    print could not repeat. There is no latch here because there is nothing
    to latch: the sampler says nothing at all, on any tick, and the cost of
    a failed poll is one subprocess on a worker thread. Ten ticks of console
    spew per session is the regression this pins.
    """
    for _ in range(10):
        broken_driver._sample_usage(False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
