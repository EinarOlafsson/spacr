"""Every branch of the GPU button's decision, on a machine with no GPU.

Instruction 60. ``spacr.gpu_reduce`` was the worst non-legacy module by
percentage (55%), and for an avoidable reason: almost all of it only runs
when cuML is importable, so a CPU test machine reaches none of it.

The fix is to control the IMPORT rather than the hardware. Every path below
is driven by putting a fake ``cuml`` / ``cupy`` in ``sys.modules``, or by
removing them, which is exactly what the function is branching on. Nothing
here needs a GPU and nothing here is skipped -- a test that skips on the
machine that runs it is a test that never runs.

What is actually being protected: this is the code behind a button that can
download SEVERAL GIGABYTES. Getting ``wrong_python`` when the answer was
``install``, or ``install`` when the answer was ``no_device``, wastes a user's
afternoon and their bandwidth.
"""
from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def no_rapids(monkeypatch):
    """cuML and cupy absent, and the env flag permissive."""
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    for name in ("cuml", "cupy"):
        monkeypatch.setitem(sys.modules, name, None)   # import -> ImportError
    return monkeypatch


def _fake_cuml(version="24.10"):
    module = types.ModuleType("cuml")
    module.__version__ = version
    return module


def _fake_cupy(devices):
    module = types.ModuleType("cupy")
    runtime = types.SimpleNamespace(getDeviceCount=lambda: devices)
    module.cuda = types.SimpleNamespace(runtime=runtime)
    return module


# --------------------------------------------------------------------------- #
#  rapids_available -- both halves, because either alone is a crash later
# --------------------------------------------------------------------------- #

def test_no_cuml_means_not_available(no_rapids):
    from spacr.gpu_reduce import rapids_available

    assert rapids_available() is False


def test_cuml_without_a_device_is_not_available(monkeypatch):
    """cuML imports happily with no GPU and then fails at fit time. Calling
    that 'available' turns an optional accelerator into a crash on exactly
    the machines that did not ask for one."""
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml())
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=0))

    from spacr.gpu_reduce import rapids_available

    assert rapids_available() is False


def test_cuml_without_a_working_cupy_is_not_available(monkeypatch):
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml())
    monkeypatch.setitem(sys.modules, "cupy", None)

    from spacr.gpu_reduce import rapids_available

    assert rapids_available() is False


def test_cuml_with_a_device_is_available(monkeypatch):
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml())
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=1))

    from spacr.gpu_reduce import rapids_available

    assert rapids_available() is True


# --------------------------------------------------------------------------- #
#  install_plan -- the four answers the button can give
# --------------------------------------------------------------------------- #

def test_plan_is_ready_when_a_device_answered(monkeypatch):
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml())
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=2))

    from spacr.gpu_reduce import install_plan

    plan = install_plan()
    assert plan["action"] == "ready"
    assert "2 device" in plan["message"]


def test_plan_is_no_device_when_cuml_is_there_and_the_gpu_is_not(monkeypatch):
    """Installing more cannot fix a missing device, so the message must not
    offer to."""
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml())
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=0))

    from spacr.gpu_reduce import install_plan

    plan = install_plan()
    assert plan["action"] == "no_device"
    assert "nvidia-smi" in plan["message"]
    # It must not OFFER an install -- more packages cannot conjure a device.
    assert "installing again cannot fix" in plan["message"]
    assert "GIGABYTES" not in plan["message"]


def test_plan_is_wrong_python_and_says_which(no_rapids, monkeypatch):
    """"Make a 3.11 environment" is actionable; a pip resolver error is
    not."""
    from spacr import gpu_reduce

    monkeypatch.setattr(gpu_reduce, "python_supported", lambda: False)

    plan = gpu_reduce.install_plan()
    assert plan["action"] == "wrong_python"
    assert "3.11" in plan["message"]
    assert "conda create" in plan["message"]


def test_plan_is_install_and_warns_about_the_size(no_rapids, monkeypatch):
    """A multi-gigabyte download with no warning reads as a hang, and the
    restart requirement is not optional -- pip can upgrade numpy underneath
    a process that has already imported it."""
    from spacr import gpu_reduce

    monkeypatch.setattr(gpu_reduce, "python_supported", lambda: True)

    plan = gpu_reduce.install_plan()
    assert plan["action"] == "install"
    assert "GIGABYTES" in plan["message"]
    assert "RESTARTED" in plan["message"]


def test_the_install_command_targets_this_interpreter():
    """Not `pip`, which may be another environment's."""
    from spacr.gpu_reduce import install_command

    command = install_command()
    assert command[0] == sys.executable
    assert command[1:4] == ["-m", "pip", "install"]
    assert "spacr[rapids]" in command


# --------------------------------------------------------------------------- #
#  describe -- the one line an About box or a log shows
# --------------------------------------------------------------------------- #

def test_describe_names_the_flag_that_disabled_it(monkeypatch):
    from spacr import gpu_reduce

    # The flag is opt-OUT: 0/false/no/off disable it, anything else allows.
    monkeypatch.setenv(gpu_reduce.ENV_FLAG, "0")
    assert gpu_reduce.ENV_FLAG in gpu_reduce.describe()
    assert "disabled" in gpu_reduce.describe()


def test_describe_tells_an_uninstalled_user_what_to_do(no_rapids):
    from spacr.gpu_reduce import describe

    text = describe()
    assert "not installed" in text
    assert "3.11" in text


def test_describe_distinguishes_no_device_from_not_installed(monkeypatch):
    """Two very different problems that would otherwise read the same."""
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml("24.10"))
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=0))

    from spacr.gpu_reduce import describe

    text = describe()
    assert "no CUDA device" in text
    assert "24.10" in text


def test_describe_counts_the_devices(monkeypatch):
    monkeypatch.delenv("SPACR_NO_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml("24.10"))
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=4))

    from spacr.gpu_reduce import describe

    assert "4 device(s)" in describe()


# --------------------------------------------------------------------------- #
#  make_reducer -- the fallback that must not surprise a caller
# --------------------------------------------------------------------------- #

def test_make_reducer_falls_back_to_cpu_and_says_so(no_rapids):
    """The backend is RETURNED, not just chosen: a caller that asked for GPU
    and silently got CPU would report the wrong provenance."""
    from spacr.gpu_reduce import make_reducer

    reducer, backend = make_reducer("umap", prefer_gpu=True, n_components=2)
    assert backend == "cpu"
    assert reducer is not None


def test_make_reducer_uses_cuml_when_it_is_really_there(monkeypatch):
    """The GPU branch, driven by a fake estimator so no device is needed."""
    from spacr import gpu_reduce

    made = {}

    def _fake_estimator(name, **kwargs):
        made["name"] = name
        made["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
    monkeypatch.setattr(gpu_reduce, "backend_for", lambda *a, **k: "cuml")
    monkeypatch.setattr(gpu_reduce, "_cuml_estimator", _fake_estimator)

    reducer, backend = gpu_reduce.make_reducer("umap", prefer_gpu=True,
                                               n_neighbors=15)
    # 'cuml', not 'gpu': the backend names the LIBRARY, which is what a
    # provenance record needs -- 'gpu' would not say which one.
    assert backend == "cuml"
    assert made["name"] == "umap"
    assert made["kwargs"]["n_neighbors"] == 15
    assert reducer is not None
