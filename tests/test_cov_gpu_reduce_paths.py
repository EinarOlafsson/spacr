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
    monkeypatch.delenv("SPACR_USE_RAPIDS", raising=False)
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
#  rapids_available and install_plan -- the answers the button can give
# --------------------------------------------------------------------------- #

_INSTALLED = object()      # a fake cuml is present
_NO_CUPY = object()        # importing cupy raises


def _the_machine(monkeypatch, *, cuml, cupy, env=None, python_ok=None):
    """Put fake cuml/cupy (or none) in sys.modules, the way the code branches on them."""
    from spacr import gpu_reduce

    monkeypatch.delenv(gpu_reduce.ENV_FLAG, raising=False)
    if env is not None:
        monkeypatch.setenv(gpu_reduce.ENV_FLAG, env)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml() if cuml is _INSTALLED else None)
    if cupy is _NO_CUPY or cupy is None:
        monkeypatch.setitem(sys.modules, "cupy", None)
    else:
        monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=cupy))
    if python_ok is not None:
        monkeypatch.setattr(gpu_reduce, "python_supported", lambda: python_ok)
    return gpu_reduce


@pytest.mark.parametrize("cuml, cupy, env, expected", [
    pytest.param(None, None, None, False, id="no-cuml"),
    pytest.param(_INSTALLED, 0, None, False, id="cuml-without-a-device"),
    pytest.param(_INSTALLED, _NO_CUPY, None, False, id="cuml-without-a-working-cupy"),
    pytest.param(_INSTALLED, 1, None, True, id="cuml-with-a-device"),
    pytest.param(_INSTALLED, 1, "0", False, id="flag-0-is-off"),
    pytest.param(_INSTALLED, 1, "false", False, id="flag-false-is-off"),
    pytest.param(_INSTALLED, 1, "no", False, id="flag-no-is-off"),
    pytest.param(_INSTALLED, 1, "off", False, id="flag-off-is-off"),
    pytest.param(_INSTALLED, 1, "", False, id="flag-empty-is-off"),
])
def test_rapids_is_available_only_with_cuml_a_device_and_the_flag_not_off(
        monkeypatch, cuml, cupy, env, expected):
    """cuML imports happily with no GPU and then fails at fit time. Calling
    that 'available' turns an optional accelerator into a crash on exactly
    the machines that did not ask for one."""
    gpu_reduce = _the_machine(monkeypatch, cuml=cuml, cupy=cupy, env=env)
    assert gpu_reduce.rapids_available() is expected


@pytest.mark.parametrize("cuml, cupy, python_ok, action, present, absent", [
    pytest.param(_INSTALLED, 2, None, "ready", ["2 device"], [],
                 id="a-device-answered"),
    pytest.param(_INSTALLED, 0, None, "no_device",
                 ["nvidia-smi", "installing again cannot fix"], ["GIGABYTES"],
                 id="cuml-without-a-device-offers-no-install"),
    pytest.param(None, None, False, "wrong_python", ["3.11", "conda create"], [],
                 id="wrong-python-says-which"),
    pytest.param(None, None, True, "install", ["GIGABYTES", "RESTARTED"], [],
                 id="supported-python-warns-about-size-and-restart"),
])
def test_the_install_plan_gives_each_situation_its_own_sentence(
        monkeypatch, cuml, cupy, python_ok, action, present, absent):
    """\"Make a 3.11 environment\" is actionable; a pip resolver error is not. A
    multi-gigabyte download with no warning reads as a hang, and pip can
    upgrade numpy underneath a process that has already imported it."""
    gpu_reduce = _the_machine(monkeypatch, cuml=cuml, cupy=cupy, python_ok=python_ok)
    plan = gpu_reduce.install_plan()
    assert plan["action"] == action
    assert [w for w in present if w in plan["message"]] == present
    assert [w for w in absent if w in plan["message"]] == []


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
    monkeypatch.delenv("SPACR_USE_RAPIDS", raising=False)
    monkeypatch.setitem(sys.modules, "cuml", _fake_cuml("24.10"))
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(devices=0))

    from spacr.gpu_reduce import describe

    text = describe()
    assert "no CUDA device" in text
    assert "24.10" in text


def test_describe_counts_the_devices(monkeypatch):
    monkeypatch.delenv("SPACR_USE_RAPIDS", raising=False)
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
