"""RAPIDS cuML where it helps, and the CPU implementation everywhere else.

Instruction 86's GPU insight, built as an EXTRA:

    pip install spacr[rapids]

and nothing else changes. The CPU path stays the default, stays tested, and
stays the answer whenever cuML is absent, the interpreter is wrong, there is
no CUDA device, or the caller asks for determinism.

WHY AN EXTRA AND NEVER A DEPENDENCY. ``cuml-cu12`` declares
``requires_python >= 3.11`` with classifiers for 3.11 and 3.12 ONLY, and
wants ``numpy>=2.0`` and ``scipy>=1.14``. spaCR promises 3.9 through 3.14, so
making it core would drop four of six interpreters. As an extra it constrains
nothing -- see the note beside it in setup.py.

WHERE IT ACTUALLY HELPS. cuML implements the algorithms spaCR already runs on
big tables: UMAP, t-SNE, PCA, DBSCAN and KMeans. Those are the ones offered
here. Everything else spaCR does -- barcode mapping, format conversion,
SQLite, report assembly, grouped statistics -- is decompression, filesystem
and small-table work, and moving it to a GPU would cost transfer time and buy
nothing. That is a finding from instruction 70, not a guess.

**Determinism is a real difference, not a footnote.** cuML's UMAP is not
bit-identical to umap-learn's, and its KMeans and DBSCAN can differ at the
boundaries. A figure regenerated on a different machine would move. So the
accelerator is OPT-IN per call, reports which backend ran, and any caller
that pins a seed for reproducibility should keep the CPU path.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("spacr.gpu_reduce")

#: The algorithms cuML is offered for. Each is one spaCR already runs and one
#: whose cuML implementation takes the same shaped input.
ACCELERATED: Tuple[str, ...] = ("umap", "tsne", "pca", "dbscan", "kmeans")

#: Set to anything falsy to force the CPU path regardless of what is
#: installed. The escape hatch for "the GPU answer looks wrong" that does not
#: require uninstalling anything.
ENV_FLAG = "SPACR_USE_RAPIDS"


def rapids_available() -> bool:
    """Is cuML importable AND is there a device for it?

    Both halves matter: cuML imports happily on a machine with no GPU and
    then fails at fit time, which would turn an optional accelerator into a
    crash on exactly the machines that did not ask for one.
    """
    if not _env_allows():
        return False
    try:
        import cuml  # noqa: F401
    except Exception:
        return False
    try:
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        # cuML present without a working cupy runtime is not a usable GPU.
        return False


def _env_allows() -> bool:
    raw = os.environ.get(ENV_FLAG)
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def backend_for(method: str, *, prefer_gpu: bool = False) -> str:
    """``'cuml'`` or ``'cpu'`` for ``method``, and never a surprise.

    :param prefer_gpu: opt in. Default False, so an existing caller keeps the
        CPU path and the reproducibility that goes with it.
    :returns: the backend that will actually run.
    """
    if not prefer_gpu:
        return "cpu"
    if str(method).strip().lower() not in ACCELERATED:
        return "cpu"
    return "cuml" if rapids_available() else "cpu"


def make_reducer(method: str, *, prefer_gpu: bool = False, **kwargs) -> Tuple[Any, str]:
    """Build the estimator for ``method``, on whichever backend is available.

    :param kwargs: passed to the estimator. The parameter names cuML shares
        with the CPU libraries -- ``n_neighbors``, ``min_dist``,
        ``n_components``, ``eps``, ``min_samples``, ``n_clusters`` -- carry
        through unchanged, which is what makes one call site serve both.
    :returns: ``(estimator, backend)``. The backend is returned rather than
        logged only, so a caller can record WHICH one produced a figure.
    :raises ImportError: the CPU library for ``method`` is missing. A missing
        optional GPU is a fallback; a missing required CPU library is a
        genuine setup problem and is not silently worked around.
    """
    name = str(method).strip().lower()
    backend = backend_for(name, prefer_gpu=prefer_gpu)

    if backend == "cuml":
        try:
            return _cuml_estimator(name, **kwargs), "cuml"
        except Exception:
            # A cuML that imports but cannot build the estimator -- a version
            # skew, a CUDA mismatch -- falls back rather than taking the run
            # down. The whole promise of an extra is that its absence, or its
            # misbehaviour, costs nothing.
            LOG.info("cuML could not build a %s estimator; using the CPU "
                     "implementation", name, exc_info=True)

    return _cpu_estimator(name, **kwargs), "cpu"


def _cuml_estimator(name: str, **kwargs):
    import cuml

    if name == "umap":
        return cuml.UMAP(**kwargs)
    if name == "tsne":
        return cuml.TSNE(**kwargs)
    if name == "pca":
        return cuml.PCA(**kwargs)
    if name == "dbscan":
        return cuml.DBSCAN(**kwargs)
    if name == "kmeans":
        return cuml.KMeans(**kwargs)
    raise ValueError(f"{name!r} has no cuML equivalent here")


def _cpu_estimator(name: str, **kwargs):
    if name == "umap":
        # The package-level ``umap`` import eagerly reaches parametric UMAP
        # and TensorFlow. spaCR's lazy proxy loads only ``umap.umap_``, which
        # is the CPU implementation this reducer actually needs.
        from .utils import umap
        return umap.UMAP(**kwargs)
    if name == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(**kwargs)
    if name == "pca":
        from sklearn.decomposition import PCA

        return PCA(**kwargs)
    if name == "dbscan":
        from sklearn.cluster import DBSCAN

        return DBSCAN(**kwargs)
    if name == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(**kwargs)
    raise ValueError(f"{name!r} is not one of {list(ACCELERATED)}")


#: The interpreters ``cuml-cu12`` declares. Not a guess -- read off the wheel
#: metadata, which carries classifiers for 3.11 and 3.12 ONLY. On anything
#: else pip produces a resolver error a user cannot act on, so spaCR says
#: what is needed instead of letting pip say what went wrong.
SUPPORTED_PYTHON = ((3, 11), (3, 12))


def python_supported() -> bool:
    """Can cuML be installed into the interpreter running this?"""
    import sys as _sys
    return _sys.version_info[:2] in SUPPORTED_PYTHON


def install_plan() -> Dict[str, Any]:
    """What pressing GPU should do, decided before anything is installed.

    :returns: ``{action, message}``. ``action`` is one of:

        ``ready``      cuML is importable and a device answered. Turn it on.
        ``install``    the interpreter can take it. The message says what it
                       will pull -- GIGABYTES of CUDA libraries, not a small
                       wheel, and a multi-gigabyte download with no progress
                       reads as a hang.
        ``wrong_python`` say exactly what is needed. "Make a 3.11 environment"
                       is actionable; a pip resolver error is not.
        ``no_device``  cuML is installed and there is no CUDA device, which
                       installing more cannot fix.

    NOTHING IS INSTALLED HERE. This function decides and reports; the caller
    installs, because installing is the part that needs a confirmation and a
    progress bar, and a function that did both could not be asked "what would
    happen" without it happening.
    """
    import sys as _sys

    version = f"{_sys.version_info.major}.{_sys.version_info.minor}"
    if rapids_available():
        return {"action": "ready", "message": describe()}
    try:
        import cuml  # noqa: F401
        return {"action": "no_device",
                "message": ("cuML is installed but no CUDA device answered. "
                            "Check the driver with nvidia-smi -- installing "
                            "again cannot fix a missing device.")}
    except Exception:
        pass
    if not python_supported():
        wanted = " or ".join(f"{a}.{b}" for a, b in SUPPORTED_PYTHON)
        return {"action": "wrong_python",
                "message": (f"cuML supports Python {wanted} only, and this is "
                            f"{version}. Make a {SUPPORTED_PYTHON[0][0]}."
                            f"{SUPPORTED_PYTHON[0][1]} environment and "
                            f"install spaCR there:\n\n"
                            f"    conda create -n spacr-gpu python="
                            f"{SUPPORTED_PYTHON[0][0]}."
                            f"{SUPPORTED_PYTHON[0][1]}\n"
                            f"    conda activate spacr-gpu\n"
                            f"    pip install 'spacr[rapids]'")}
    return {"action": "install",
            "message": ("Install cuML for GPU UMAP?\n\nThis downloads "
                        "SEVERAL GIGABYTES of CUDA libraries -- cuml-cu12 "
                        "pulls libcuml, cudf, cupy and the CUDA runtime. It "
                        "is not a small wheel and it will take a while.\n\n"
                        "spaCR must be RESTARTED afterwards: pip can upgrade "
                        "numpy and scipy underneath a process that has "
                        "already imported them, and this one has.")}


def install_command() -> List[str]:
    """The command that installs the extra. Separate so it can be shown."""
    import sys as _sys
    return [_sys.executable, "-m", "pip", "install", "spacr[rapids]"]


def describe() -> str:
    """One line for a log or an About box: what is available, and why not."""
    if not _env_allows():
        return f"RAPIDS disabled by {ENV_FLAG}"
    try:
        import cuml
    except Exception:
        return ("RAPIDS not installed (pip install 'spacr[rapids]', "
                "Python 3.11 or 3.12)")
    try:
        import cupy

        devices = cupy.cuda.runtime.getDeviceCount()
    except Exception:
        devices = 0
    if not devices:
        return f"cuML {getattr(cuml, '__version__', '?')} installed, no CUDA device"
    return f"cuML {getattr(cuml, '__version__', '?')} on {devices} device(s)"
