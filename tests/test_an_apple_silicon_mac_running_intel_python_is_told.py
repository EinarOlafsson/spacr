"""An Apple Silicon Mac running an Intel Python is told so, with a fix that
works.

Found 2026-10-24 on the maintainer's first Apple Silicon test machine (M2).
spaCR was installed into ``conda create -n spacr python=3.12`` from a 2023
Intel Anaconda, so the interpreter, NumPy, every wheel and torch ran under
Rosetta 2, and torch was capped at 2.2.2 -- the last Intel-Mac wheel. The
maintainer: "this is a bug and spacr should have auto installed the correct
version!"

The macOS installer already re-executes itself as arm64 when it is started
under Rosetta (``packaging/online/install_spacr_unix.sh``). Nothing at RUNTIME
looked, and nothing could have: a translated process is told
``platform.machine() == "x86_64"``, so it looked exactly like an Intel Mac.
And the doctor's own advice, a plain ``conda create``, rebuilds the problem
from an Intel conda; ``CONDA_SUBDIR=osx-arm64`` is what makes it native
(verified on that machine with Anaconda's x86_64 conda 23.7.4).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import accelerator as A
from spacr import doctor


def _forget_the_cached_answer_if_cached() -> None:
    """Guarded for the reason given in test_an_intel_mac_is_told_why_metal_is_no_use:
    teardown can run while a test's lambda is still bound there."""
    clear = getattr(A._rosetta_translated, "cache_clear", None)
    if clear is not None:
        clear()


@pytest.fixture(autouse=True)
def _forget_the_cached_answer():
    """`_rosetta_translated` is `lru_cache`d; each test sets its own machine."""
    _forget_the_cached_answer_if_cached()
    yield
    _forget_the_cached_answer_if_cached()


def _translated(monkeypatch, translated: bool) -> None:
    monkeypatch.setattr(A.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(A.platform, "machine",
                        lambda: "x86_64" if translated else "arm64")
    monkeypatch.setattr(A, "_rosetta_translated", lambda: translated)


@pytest.mark.skipif(sys.platform != "darwin", reason="asks the macOS kernel")
def test_the_answer_is_the_kernels():
    """Same verdict as ``sysctl``, whichever way this interpreter was built."""
    out = subprocess.run(["sysctl", "-in", "sysctl.proc_translated"],
                         capture_output=True, text=True).stdout.strip()
    assert A._rosetta_translated() is (out == "1")


def test_nothing_but_macos_is_ever_translated(monkeypatch):
    monkeypatch.setattr(A.platform, "system", lambda: "Linux")
    assert A._rosetta_translated() is False


def test_the_doctor_warns_and_gives_a_conda_that_builds_arm64(monkeypatch):
    _translated(monkeypatch, True)
    result = doctor._check_native_cpu(doctor.Context())
    assert result.status == doctor.WARN
    assert "Rosetta" in result.message
    assert "CONDA_SUBDIR=osx-arm64" in result.fix
    assert "--set subdir osx-arm64" in result.fix


def test_a_native_python_passes(monkeypatch):
    _translated(monkeypatch, False)
    result = doctor._check_native_cpu(doctor.Context())
    assert result.status == doctor.PASS


def test_the_check_sits_right_after_python():
    labels = [check.check_label for check in doctor.CHECKS]
    assert labels[labels.index("python") + 1] == "native CPU"


def test_every_conda_create_the_doctor_suggests_is_arm64_when_translated(monkeypatch):
    _translated(monkeypatch, True)
    assert doctor._conda_create("spacr").startswith("CONDA_SUBDIR=osx-arm64 ")
    _translated(monkeypatch, False)
    assert doctor._conda_create("spacr") == "conda create -n spacr python=3.12 -y"


def test_a_translated_m2_still_has_its_neural_engine(monkeypatch):
    _translated(monkeypatch, True)
    assert A.neural_engines() == ("Apple Neural Engine (CoreML only)",)


def test_a_translated_m2_is_not_called_an_intel_mac(monkeypatch):
    """The Intel-integrated-graphics explanation must not reach an M2."""
    _translated(monkeypatch, True)
    monkeypatch.setattr(A.platform, "mac_ver", lambda: ("26.0", ("", "", ""), ""))
    monkeypatch.setattr(A, "_metal_gpu_name", lambda: "Intel Iris Plus Graphics")
    _label, note = A._why_metal_is_unavailable()
    assert "Intel Mac" not in note


def test_the_app_says_it_at_launch(monkeypatch, capsys):
    from spacr import qt

    _translated(monkeypatch, True)
    qt._say_when_running_translated()
    assert "Rosetta 2" in capsys.readouterr().err

    _translated(monkeypatch, False)
    qt._say_when_running_translated()
    assert capsys.readouterr().err == ""


def _forget_gpu_name():
    clear = getattr(A._metal_gpu_name, "cache_clear", None)
    if clear is not None:
        clear()


def test_apple_silicon_names_its_gpu_without_system_profiler(monkeypatch):
    """The chip is the GPU; system_profiler cost 0.75 s on the launch path."""
    _forget_gpu_name()
    monkeypatch.setattr(A.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(A, "_apple_silicon", lambda: True)
    monkeypatch.setattr(A, "_sysctl_string", lambda name: "Apple M3 Pro")

    def no_subprocess(*args, **kwargs):
        raise AssertionError("system_profiler was spawned")

    monkeypatch.setattr("subprocess.run", no_subprocess)
    try:
        assert A._metal_gpu_name() == "Apple M3 Pro"
    finally:
        _forget_gpu_name()


@pytest.mark.skipif(sys.platform != "darwin" or not A._apple_silicon(),
                    reason="needs an Apple Silicon Mac")
def test_the_kernels_chip_name_is_system_profilers_gpu_name():
    out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                         capture_output=True, text=True, timeout=60).stdout
    chipset = next(line.split(":", 1)[1].strip() for line in out.splitlines()
                   if "Chipset Model:" in line)
    assert A._sysctl_string(b"machdep.cpu.brand_string") == chipset


@pytest.mark.parametrize("live", [True, False])
def test_a_torch_imported_before_spacr_is_named_not_assumed(monkeypatch, recwarn, live):
    """torch reads PYTORCH_ENABLE_MPS_FALLBACK once, at its own import."""
    import spacr

    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setattr(spacr, "_MPS_FALLBACK_LIVE", live)
    metal = A.Accelerator(kind="mps", device="mps", label="Apple M2 (Apple Metal)",
                          name="Apple M2", fallback=True)
    assert A._mps_fallback_live(metal) is live
    named = [w for w in recwarn if "imported before spaCR" in str(w.message)]
    assert bool(named) is (not live)


_METAL = False
try:
    import torch as _torch
    _METAL = bool(_torch.backends.mps.is_available())
except Exception:                                            # noqa: BLE001
    pass


@pytest.mark.skipif(not _METAL, reason="needs a Metal device")
@pytest.mark.parametrize("order, falls_back", [
    ("import spacr; import torch", True),
    ("import torch; import spacr", False),
])
def test_the_warning_matches_what_metal_actually_does(order, falls_back):
    script = (
        "import os, warnings\n"
        "os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)\n"
        f"{order}\n"
        "from spacr import accelerator\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    found = accelerator.resolve()\n"
        "try:\n"
        "    torch.linalg.eig(torch.eye(4, device='mps')); ran = True\n"
        "except NotImplementedError:\n"
        "    ran = False\n"
        "print(found.fallback, ran, any('imported before spaCR' in str(w.message) for w in caught))\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=300, env=env, cwd=str(Path(__file__).parent)).stdout.split()
    assert out == [str(falls_back), str(falls_back), str(not falls_back)]


@pytest.mark.parametrize("method", ["pca", "umap"])
def test_a_gpu_reduction_on_apple_silicon_runs_on_the_cpu_instead_of_refusing(
        monkeypatch, method):
    """PCA and KMeans stay on the CPU on an Apple GPU (it is faster there);
    asking for the GPU must run them, not raise "GPU was requested"."""
    pytest.importorskip("sklearn")
    from spacr import utils

    monkeypatch.setattr(A, "_CACHED", A.Accelerator(
        kind="mps", device="mps", label="Apple M2 (Apple Metal)", name="Apple M2",
        float64=False, autocast=False, fallback=True))
    values = __import__("numpy").random.default_rng(0).random((40, 4))
    embedding, labels, _reducer = utils.reduction_and_clustering(
        values, 5, 0.1, "euclidean", 0.5, 3, "kmeans",
        reduction_method=method, prefer_gpu=True, n_jobs=1)
    assert embedding.shape == (40, 2) and len(labels) == 40
