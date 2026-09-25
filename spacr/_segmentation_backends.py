"""Optional segmentation backends behind the Cellpose mask path.

NOT AN API PAGE. tools/build_documentation_i18n.py extracts the module
docstring of every file under spacr/, private modules included, and pins it
by hash in nine translated API catalogs -- unless the module is listed in its
AUTOAPI_NON_RENDERED_MODULES, and this one is (2026-09-15). AutoAPI renders
no leading-underscore module, and everything below is underscore-private, so
this docstring carries the design without adding a page that would need
translating.

Items 404 (DINOCell), 405 (SAMCell) and 423 (Cellpose 3, and one environment
per backend), built as ONE seam rather than three.
`generate_cellpose_masks_sam` builds a model and calls
`model.eval(x=batch_list, ...)`; every backend here answers that same call
with the `(masks, flows, styles)` triple Cellpose 4 returns, so the lines
after it -- `parse_cellpose4_output`, merge/split/filter, tracking, the
object-count database, saving and segmentation QC -- run unchanged. The only
dispatch is at model construction, and segmentation_backend='cellpose' (or
the key absent) never reaches this module's loaders at all.

WHERE A BACKEND RUNS. Each backend gets an isolated environment, and
Cellpose 3 is one of them, with its cyto, nuclei, cyto2 and cyto3 models.
Each optional backend is installed into an
environment of its own under ``~/.spacr/backends/<name>`` (or
``$SPACR_BACKENDS_DIR/<name>``), and spaCR calls it out of process. spaCR's
own environment is never modified: DINOCell pins exact versions of torch,
numpy and Cellpose 4 that conflict with spaCR's, and Cellpose 3 cannot share
a process with the Cellpose 4 spaCR runs at all.

THIS FILE IS ALSO THE WORKER. Inside a backend environment spaCR runs
``<env python> -I _segmentation_backends.py --serve <name>``. At module scope
this file imports only the standard library and numpy, which every backend
environment has, so the adapters below run there without spaCR installed.
``-I`` keeps the script's own folder -- spaCR's package directory, whose
``io.py`` and friends would shadow the standard library -- off ``sys.path``.

WHY VENV AND NOT CONDA. ``venv`` is in the standard library on Linux, macOS
and Windows, needs no conda on the machine, and builds an environment in
seconds; pip then fills it from PyPI, where all three projects publish. Its
one limit is that the environment's Python is a Python already on the
computer: spaCR's own when it is in the range a backend's pins install on,
otherwise a ``python3.X`` on PATH (or ``py -3.X`` on Windows). When there is
none, the row says so -- "not installable here" with the reason -- rather
than half-building something.

THE PROTOCOL, version :data:`_PROTOCOL`: one JSON object per line in each
direction over the worker's stdin and stdout. The worker moves its own
stdout to stderr before loading anything, so a library that prints cannot
corrupt a reply. Images and masks travel as ``.npy`` files in a temporary
folder, never inside the JSON. Requests: ``hello`` (versions and device),
``segment`` and ``shutdown``. A failure comes back as the exception's type,
message and traceback, and spaCR raises it with the message VERBATIM.

HOW EACH BACKEND BECOMES A MASK

* Cellpose 3 (``cellpose==3.1.1.3``) runs its own ``cyto``, ``cyto2``,
  ``cyto3`` or ``nuclei`` model through ``models.Cellpose``, which estimates
  the diameter with Cellpose's size model when none is given, or a
  Cellpose-format checkpoint file -- a bioimage.io download, say -- through
  ``models.CellposeModel``. An object whose batch carries a second channel
  (a cell with its nucleus) is segmented as ``channels=[1, 2]``.
* DINOCell predicts Cellpose-style flows: (dx, dy, cell probability). Masks
  come from `cellpose.dynamics.compute_masks`, called with the constants
  DINOCell's own `DINOFlowsSlidingWindowPipeline.run` uses (250 iterations,
  flow-error check off, minimum size 15, maximum size fraction 0.4). Its
  probability is a sigmoid output, so spaCR's `<object>_cellprob_threshold`
  -- a Cellpose logit -- is applied through the logistic function: the
  default 0 is DINOCell's own 0.5.
* SAMCell is SAM ViT-B fine-tuned to predict a cell distance map. Masks come
  from SAMCell's own `SlidingWindowPipeline.cells_from_dist_map` (contour
  centroids as seeds, watershed inside the fill threshold), with its default
  thresholds.

DINOCell and SAMCell are single-channel 2-D models: each reads the object's
own channel (the first in the batch, as `_get_cellpose_channels` orders
them), stretched to 8 bits because both packages quantise their input to
uint8 before CLAHE. z-stack and t-stack runs are refused for every backend
here rather than flattened.

A DINOCell or SAMCell that an older spaCR pip-installed INTO spaCR's own
environment still works, in process, exactly as before; a backend
environment wins when both exist.

Nothing here imports torch, cellpose, transformers or either package at
module scope; tests/test_perf_guard.py holds the launch path to
that.
"""
from __future__ import annotations

import atexit
import collections
import inspect
import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

_CELLPOSE = "cellpose"
_CELLPOSE3 = "cellpose3"
_DINOCELL = "dinocell"
_SAMCELL = "samcell"
_PAPERS = "papers"

#: DeepCell's SpotNet, for fluorescent spots rather than cells.
_SPOTNET = "spotnet"

#: Every value ``segmentation_backend`` accepts, the default first.
_BACKEND_NAMES = (_CELLPOSE, _CELLPOSE3, _DINOCELL, _SAMCELL)

#: The models the Cellpose 3 backend names, as Cellpose 3 names them.
_CELLPOSE3_MODELS = ("cyto3", "cyto2", "cyto", "nuclei")

_RESTORATION_MODELS = tuple(
    f"{operation}_{structure}"
    for operation in ("denoise", "deblur", "oneclick")
    for structure in ("cyto3", "cyto2", "nuclei")
)

#: The request/response protocol between spaCR and a backend worker.
_PROTOCOL = 1

#: Written into a backend environment LAST; an environment without it is an
#: install that did not finish.
_MARKER = "spacr-backend.json"

#: Environment variable naming the folder backend environments live in.
_ROOT_ENV = "SPACR_BACKENDS_DIR"

#: Environment variable naming the PyTorch wheel index for backend installs;
#: ``pypi`` means PyPI's own torch.
_TORCH_INDEX_ENV = "SPACR_BACKEND_TORCH_INDEX"

#: PyTorch's wheel indexes, one folder per build (``cpu``, ``cu124``, ...).
_TORCH_WHEELS = "https://download.pytorch.org/whl/"

#: spaCR's device override, honoured by the workers too.
_DEVICE_ENV = "SPACR_DEVICE"

_INSTALLED = "installed"
_INSTALLABLE = "installable"
_INSTALLING = "installing"
_UNAVAILABLE = "not installable here"

#: Every state a backend row can be in.
_STATES = (_INSTALLED, _INSTALLABLE, _INSTALLING, _UNAVAILABLE)

#: How long a failed network or interpreter probe keeps a row unavailable.
_PROBE_SECONDS = 600.0

#: A worker nobody has asked anything for this long is shut down; the next
#: request starts it again.
_IDLE_SECONDS = 600.0

#: Variables that would send pip, or the backend's Python, somewhere other
#: than the backend's own environment.
_STRIPPED_VARIABLES = (
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
    "PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT",
    "PIP_REQUIRE_VIRTUALENV", "__PYVENV_LAUNCHER__",
    "DEEPCELL_ACCESS_TOKEN",
)

#: The variable DeepCell reads its access token from. It is stripped from
#: every backend command above and handed back only to SpotNet's worker, so
#: pip, the self-test and the other backends never see it.
_DEEPCELL_TOKEN_ENV = "DEEPCELL_ACCESS_TOKEN"

#: The archive deepcell-spots 0.4.2 fetches its SpotNet weights as.
_SPOTNET_ARCHIVE = "SpotDetection-8.tar.gz"

#: Variables that win over ``HF_HOME``, so pointing ``HF_HOME`` inside a
#: backend's environment is not enough on its own. ``huggingface_hub``
#: derives its caches from ``HF_HOME`` only when none of these is set:
#: ``HF_HUB_CACHE`` falls back to ``HUGGINGFACE_HUB_CACHE``, which falls back
#: to ``$HF_HOME/hub``, and the assets and xet caches are the same shape.
#: Anyone who has moved their Hugging Face cache off their home disk has one
#: of these exported, and would get a backend's weights outside the
#: environment that is supposed to own them.
_HF_CACHE_VARIABLES = (
    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
    "HF_ASSETS_CACHE", "HUGGINGFACE_ASSETS_CACHE",
    "HF_XET_CACHE",
)

#: What a candidate interpreter must be able to do to build an environment.
_INTERPRETER_CHECK = (
    "import sys, venv, ensurepip; print('%d.%d' % sys.version_info[:2])")

#: DINOCell's inference constants, copied from ``dinocell.main.segment`` and
#: ``DINOFlowsSlidingWindowPipeline.run`` (dinocell 0.74).
_DINOCELL_CROP = 512
_DINOCELL_OVERLAP = _DINOCELL_CROP // 4
_DINOCELL_NITER = 250
_DINOCELL_FLOW_THRESHOLD = 0
_DINOCELL_MIN_SIZE = 15
_DINOCELL_MAX_SIZE_FRACTION = 0.4

#: SAMCell's base model, crop and published checkpoints (samcell 1.2.0 README).
_SAMCELL_BASE_MODEL = "facebook/sam-vit-base"
_SAMCELL_CROP = 256
_SAMCELL_RELEASE = (
    "https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/")
_SAMCELL_WEIGHTS = {
    "generalist": "samcell-generalist.pt",
    "cyto": "samcell-cyto.pt",
}


@dataclass(frozen=True)
class _BackendSpec:
    """What one optional backend installs, and what installing it needs.

    :param name: the ``segmentation_backend`` value.
    :param label: what a person reads.
    :param module: the package's import name.
    :param probe: the modules the self-test imports -- the ones the adapter
        imports, so an environment missing a dependency fails during the
        install rather than on the first field.
    :param distribution: the pip distribution whose version is recorded.
    :param requirements: what pip installs, pinned to the release this
        adapter was written against.
    :param torch: PyTorch requirements installed first, from the wheel index
        that matches spaCR's own PyTorch build.
    :param python: the lowest and highest ``(major, minor)`` the pins install
        on.
    :param platforms: ``sys.platform`` prefixes the backend runs on.
    :param licence: the SPDX identifier of the package's licence.
    :param licence_note: the licence in a sentence, with where the weights
        come from and under what terms.
    :param homepage: the project's page.
    :param size_gb: roughly the disk a CPU install takes; a CUDA build of
        PyTorch takes several gigabytes more.
    :param in_process: whether a copy an older spaCR installed into spaCR's
        own environment is still used.
    :param models: the named models the backend offers.
    :param blurb: one sentence for the zoo row.
    :param segments: whether this is a segmentation backend. The figure
        reader (``papers``) is installed and run the same way but is never
        offered as one.
    :param published: the project's own reported results, quoted with their
        source, for the zoo row's scorecard note. spaCR has not measured
        these backends on its own data, and the note says so.
    """

    name: str
    label: str
    module: str
    probe: tuple
    distribution: str
    requirements: tuple
    torch: tuple
    python: tuple
    platforms: tuple = ("linux", "darwin", "win32")
    licence: str = ""
    licence_note: str = ""
    homepage: str = ""
    size_gb: float = 2.0
    in_process: bool = False
    models: tuple = ()
    blurb: str = ""
    published: str = ""
    segments: bool = True


#: Every optional backend. The versions are the ones each adapter was
#: written and tested against; the licences were read from each release's
#: own LICENSE file and PyPI record on 2026-09-19. ``packaging`` is in
#: Cellpose 3's list because fastremap 1.20, which Cellpose 3 needs, imports
#: it without declaring it: measured on the first real install, 2026-09-19.
_SPECS = {
    _CELLPOSE3: _BackendSpec(
        name=_CELLPOSE3, label="Cellpose 3", module="cellpose",
        probe=("cellpose.models",), distribution="cellpose",
        requirements=("cellpose==3.1.1.3", "packaging"),
        torch=("torch",), python=((3, 9), (3, 12)),
        licence="BSD-3-Clause",
        licence_note=(
            "Cellpose 3.1.1.3 is BSD-3-Clause (Copyright 2020 Howard Hughes "
            "Medical Institute). Cellpose downloads its cyto, cyto2, cyto3 "
            "and nuclei weights from cellpose.org itself, into the "
            "backend's own folder."),
        homepage="https://github.com/MouseLand/cellpose", size_gb=2.0,
        models=_CELLPOSE3_MODELS,
        blurb=(
            "Cellpose 3 with its cyto3, cyto2, cyto and nuclei models, and "
            "any Cellpose-format model added from bioimage.io. It runs in "
            "an environment of its own, so spaCR's Cellpose 4 is "
            "untouched."),
        published=(
            "Published results: Stringer and Pachitariu, 'Cellpose3: one-click "
            "image restoration for improved cellular segmentation', Nature "
            "Methods 2025 (doi:10.1038/s41592-025-02595-5). The project's "
            "README publishes no results table. spaCR has not scored this "
            "backend on its own data.")),
    _DINOCELL: _BackendSpec(
        name=_DINOCELL, label="DINOCell", module="dinocell",
        probe=("dinocell.main", "dinocell.model", "dinocell.pipeline",
               "cellpose.dynamics", "cellpose.plot", "cv2"),
        distribution="dinocell", requirements=("dinocell==0.74",),
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        python=((3, 11), (3, 14)), licence="MIT",
        licence_note=(
            "DINOCell 0.74 is MIT (Copyright 2026 Kaden Stillwagon); its "
            "weights, KadenStillwagon/DINOCell on Hugging Face, are MIT "
            "too."),
        homepage="https://github.com/kadenstillwagon/DINOCell", size_gb=3.0,
        in_process=True,
        blurb=(
            "DINOCell, a DINOv2 model that predicts Cellpose-style flows, "
            "for live-cell and label-free images. It pins its own torch, "
            "numpy and Cellpose, so it runs in an environment of its own."),
        published=(
            "Published results (project README, read 2026-09-21; Stillwagon et "
            "al. 2026, arXiv:2604.10609): LIVECell test set SEG 0.784, DET "
            "0.926, MMA 0.876, against Cellpose-SAM 0.710 / 0.852 / 0.807. "
            "spaCR has not scored this backend on its own data.")),
    _SAMCELL: _BackendSpec(
        name=_SAMCELL, label="SAMCell", module="samcell",
        probe=("samcell.model", "samcell.pipeline"),
        distribution="samcell",
        requirements=("samcell==1.2.0", "matplotlib>=3.3.0"),
        torch=("torch",), python=((3, 9), (3, 14)), licence="MIT",
        licence_note=(
            "SAMCell 1.2.0 is MIT (Copyright 2025 Saahil Sanganeriya). It "
            "fine-tunes facebook/sam-vit-base, which is Apache-2.0, and its "
            "checkpoints come from the project's GitHub release."),
        homepage="https://github.com/saahilsanganeriya/SAMCell", size_gb=3.0,
        in_process=True,
        blurb=(
            "SAMCell, SAM ViT-B fine-tuned to predict a cell distance map, "
            "trained partly on LIVECell. It runs in an environment of its "
            "own."),
        published=(
            "Published results (project README, read 2026-09-21; "
            "Sanganeriya et al., PLOS ONE 2025, doi:10.1371/journal.pone."
            "0319532): LIVECell test set SEG 0.652, DET 0.893, OP_CSB 0.772, "
            "against Cellpose 0.589 / 0.779 / 0.684. spaCR has not scored "
            "this backend on its own data.")),
    _SPOTNET: _BackendSpec(
        name=_SPOTNET, label="SpotNet (DeepCell)", module="deepcell_spots",
        probe=("deepcell_spots", "deepcell_spots.applications", "tensorflow"),
        distribution="deepcell-spots",
        requirements=("trackpy==0.6.1", "deepcell==0.12.10",
                      "deepcell-spots==0.4.2"),
        torch=("torch", "torchvision"), python=((3, 7), (3, 10)),
        licence="Modified Apache-2.0, NON-COMMERCIAL ACADEMIC USE ONLY",
        licence_note=(
            "DeepCell's models and training data are licensed for "
            "non-commercial academic use only (a modified Apache licence), "
            "which is NOT the licence spaCR itself carries. Its weights are "
            "not public either: they are fetched from users.deepcell.org "
            "with a free account's access token, which spaCR reads from "
            "DEEPCELL_ACCESS_TOKEN. Read the licence before using SpotNet "
            "for anything commercial."),
        homepage="https://github.com/vanvalenlab/deepcell-spots",
        size_gb=3.0, segments=False,
        blurb=(
            "SpotNet finds fluorescent SPOTS -- single molecules, FISH "
            "puncta, sequencing-by-synthesis signals -- and returns their "
            "coordinates, not masks. Its environment contains TensorFlow "
            "and PyTorch. deepcell-spots 0.4.2 needs Python 3.7 to 3.10 "
            "available to create that environment; spaCR itself may use "
            "a newer Python. The weights need a free DeepCell token. trackpy and "
            "deepcell are pinned with it: deepcell-spots pins neither, and "
            "pip walked back to trackpy 0.2.3 (2014), whose setup.py cannot "
            "build (reported 2026-09-22)."),
        published=(
            "Published results: Laubscher et al., 'Accurate single-molecule "
            "spot detection for image-based spatial transcriptomics with "
            "weakly supervised deep learning', Cell Systems 2024 "
            "(doi:10.1016/j.cels.2023.12.008). spaCR has not scored it on "
            "its own data.")),
    _PAPERS: _BackendSpec(
        name=_PAPERS, label="Plaque figure reader", module="ultralytics",
        probe=("ultralytics", "rapidocr_onnxruntime"),
        distribution="ultralytics",
        requirements=("ultralytics==8.4.157", "rapidocr-onnxruntime==1.4.4",
                      "pdfplumber==0.11.10"),
        torch=("torch", "torchvision"), python=((3, 9), (3, 12)),
        licence="AGPL-3.0 (ultralytics) / Apache-2.0 (RapidOCR) / MIT (pdfplumber)",
        licence_note=(
            "ultralytics is AGPL-3.0, RapidOCR Apache-2.0 and pdfplumber "
            "MIT. They are "
            "installed into this environment of their own and run in a "
            "separate process, so spaCR's own environment and licence are "
            "untouched."),
        homepage="https://github.com/ultralytics/ultralytics", size_gb=3.0,
        segments=False,
        blurb=(
            "What Plaque Assay's Figure mode needs: the YOLO detector that "
            "finds plaque images in a figure, RapidOCR, which reads the "
            "panel letters and labels around them, and pdfplumber, which "
            "renders a PDF's pages and reads its text layer. Installed apart from "
            "spaCR so its torch and opencv cannot change spaCR's."),
        published=(
            "Published results: none for this use. The detector's own "
            "measurements on published figures are in item 424 "
            "(toxoplasma_well_detector_v2)."))
}


class _InstallFailed(RuntimeError):
    """An install step failed; the message carries its output verbatim."""


class _InstallBlocked(_InstallFailed):
    """This computer cannot install the backend; the message says why."""


class _InstallCancelled(RuntimeError):
    """The person pressed Cancel. Nothing was left behind."""


class _BackendError(RuntimeError):
    """A backend's own failure, raised in spaCR with its message verbatim.

    :param message: what happened, beginning with the backend's name.
    :param remote_type: the exception's class name inside the worker.
    :param remote_traceback: its traceback there, for a bug report.
    """

    def __init__(self, message, remote_type="", remote_traceback=""):
        """Keep the worker's exception type and traceback beside the message."""
        super().__init__(message)
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback


class _BackendCancelled(RuntimeError):
    """A request was abandoned mid-flight; its worker was stopped, or, for a
    request sent with ``keep_on_cancel``, left to finish unheard."""


def _final_lines(lines):
    """What a terminal would show after ``lines``: a carriage return
    overwrites.

    A progress bar (tqdm, a download) redraws itself on one line by
    printing ``\\r`` and the new state. Read as text, every redraw became a
    line of its own, and an error that quoted the worker's last output
    quoted forty stacked copies of one bar (item 507). Each line keeps the
    text after its last carriage return that has any; a line that ENDS in a
    carriage return is replaced by the next line, the way the bar replaced
    it on screen.

    :param lines: raw lines, with their line endings, as a stream reader
        opened with ``newline=''`` returns them; a plain string is split.
    :returns: the lines as they would stand, without their endings.
    """
    if isinstance(lines, str):
        lines = lines.splitlines(keepends=True)
    shown = []
    overwrite = False
    for raw in lines:
        text = raw[:-2] if raw.endswith("\r\n") else raw.rstrip("\n")
        pending = text.endswith("\r") and not raw.endswith("\r\n")
        parts = [part for part in text.split("\r") if part.strip()]
        text = parts[-1] if parts else ""
        if overwrite and shown:
            shown[-1] = text if text else shown[-1]
        elif text or not pending:
            shown.append(text)
        overwrite = pending
    return shown


@dataclass(frozen=True)
class _BackendState:
    """Where one optional backend stands on this computer.

    :param name: the backend.
    :param state: one of :data:`_STATES`.
    :param reason: why, in a sentence -- where it is installed, what is
        installing it, or what stops it being installed here.
    :param env: its environment's folder, whether or not it exists.
    :param record: what the install recorded -- versions, device, interpreter.
    :param in_process: installed inside spaCR's own environment by an older
        spaCR, which this spaCR still uses but never removes.
    """

    name: str
    state: str
    reason: str = ""
    env: str = ""
    record: dict = field(default_factory=dict)
    in_process: bool = False

    @property
    def ready(self):
        """Whether it can segment now."""
        return self.state == _INSTALLED


def _spec(name):
    """The spec for an optional backend.

    A backend that does not segment -- the plaque figure reader, SpotNet --
    is not a ``segmentation_backend`` value, so it is found by its own name
    before the segmentation names are checked.

    :raises ValueError: for Cellpose 4 or a name spaCR has no backend for.
    """
    asked = str(name).strip().lower()
    if asked in _SPECS and not _SPECS[asked].segments:
        return _SPECS[asked]
    backend = _backend_name(name)
    if backend not in _SPECS:
        raise ValueError(
            f"{backend!r} is spaCR's own Cellpose, not an optional backend")
    return _SPECS[backend]


def _backend_name(value):
    """Canonical backend name for a ``segmentation_backend`` value.

    :param value: the setting's value; ``None`` or blank means Cellpose.
    :returns: one of :data:`_BACKEND_NAMES`.
    :raises ValueError: for a name spaCR has no backend for.
    """
    if value is None:
        return _CELLPOSE
    name = str(value).strip().lower()
    if not name:
        return _CELLPOSE
    if name not in _BACKEND_NAMES:
        choices = ", ".join(repr(n) for n in _BACKEND_NAMES)
        raise ValueError(
            f"segmentation_backend={value!r} is not a segmentation backend "
            f"spaCR has. Choose one of {choices}.")
    return name


def _cellpose3_model(model_name=None, object_type=None):
    """The Cellpose 3 model an object's model setting selects.

    A Cellpose 3 name, or a checkpoint file, is used as it is. A blank, or
    the NAME of a model of another Cellpose -- spaCR's default ``cpsam`` --
    means the Cellpose 3 model for the object: ``nuclei`` for nuclei,
    ``cyto3`` for everything else. A PATH that is not there is refused:
    Cellpose 3 itself would quietly run cyto3 in its place.

    :param model_name: the object's ``<object>_model_name`` setting.
    :param object_type: ``'cell'``, ``'nucleus'``, ``'pathogen'``, ...
    :returns: a model name or an absolute path.
    :raises FileNotFoundError: for a path that names no file.
    """
    name = str(model_name or "").strip()
    if name in _CELLPOSE3_MODELS:
        return name
    path = os.path.expanduser(name)
    if name and os.path.isfile(path):
        return os.path.abspath(path)
    if name and (os.sep in name or "/" in name or os.path.splitext(name)[1]):
        raise FileNotFoundError(
            f"no Cellpose 3 model at {name!r}: the file is not there. Name "
            f"one of {', '.join(_CELLPOSE3_MODELS)}, or the path of a "
            f"Cellpose 3 checkpoint.")
    return "nuclei" if object_type == "nucleus" else "cyto3"


def _backends_root(root=None):
    """The folder backend environments live in.

    :param root: an explicit folder, which wins.
    :returns: ``$SPACR_BACKENDS_DIR`` when set, else ``~/.spacr/backends``.
    """
    if root:
        return os.path.abspath(os.path.expanduser(str(root)))
    configured = os.environ.get(_ROOT_ENV, "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(os.path.expanduser("~"), ".spacr", "backends")


def _env_python(env, windows=None):
    """The Python inside a backend environment.

    :param env: the environment's folder.
    :param windows: lay it out for Windows; the running system when None.
    """
    windows = (os.name == "nt") if windows is None else windows
    if windows:
        return os.path.join(env, "Scripts", "python.exe")
    return os.path.join(env, "bin", "python")


def _worker_path():
    """This file, which is what a backend environment runs; None when the
    running spaCR has no source file to hand it (a frozen build)."""
    path = os.path.abspath(__file__)
    if path.endswith(".py") and os.path.isfile(path):
        return path
    return None


def _importable(module):
    """Whether ``module`` imports in spaCR's own environment, without
    importing it."""
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _pid_alive(pid):
    """Whether a process id is running on this computer."""
    if pid == os.getpid():
        return True
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        return bool(psutil.pid_exists(pid))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _lock_path(root, name):
    """The file that says an install is running."""
    return os.path.join(root, f"{name}.lock")


def _read_lock(root, name):
    """The running install's lock, or None when there is none or it is stale.

    A lock written on another computer (a home folder shared over the
    network) cannot be checked, so it is believed.
    """
    import socket

    try:
        with open(_lock_path(root, name), encoding="utf-8") as handle:
            lock = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(lock, dict):
        return None
    if lock.get("host") and lock.get("host") != socket.gethostname():
        return lock
    try:
        pid = int(lock.get("pid"))
    except (TypeError, ValueError):
        return None
    return lock if _pid_alive(pid) else None


def _acquire_lock(root, name):
    """Claim the install of ``name``, or refuse because one is running.

    :raises _InstallBlocked: when the folder cannot be written.
    :raises _InstallFailed: when another install holds the lock.
    """
    import socket

    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        raise _InstallBlocked(
            f"spaCR cannot create {root}, where backend environments are "
            f"kept: {exc}") from exc
    path = _lock_path(root, name)
    if os.path.exists(path):
        if _read_lock(root, name) is not None:
            raise _InstallFailed(
                f"{_spec(name).label} is already being installed; its lock "
                f"is {path}.")
        os.remove(path)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise _InstallFailed(
            f"{_spec(name).label} is already being installed; its lock is "
            f"{path}.") from exc
    except OSError as exc:
        raise _InstallBlocked(
            f"spaCR cannot write to {root}, where backend environments are "
            f"kept: {exc}") from exc
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, stream)


def _release_lock(root, name):
    """Drop the install lock; a lock already gone is fine."""
    try:
        os.remove(_lock_path(root, name))
    except OSError:
        pass


def _read_marker(env):
    """The finished install's record, or None."""
    try:
        with open(os.path.join(env, _MARKER), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _write_marker(env, record):
    """Write the install record atomically, so a half-written one is never
    read as a finished install."""
    path = os.path.join(env, _MARKER)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


#: ``name -> interpreters`` found for it, once per process.
_CANDIDATES = {}

#: ``name -> (reason, time)`` from the last failed probe.
_PROBED = {}


def _interpreter_candidates(spec, *, executable=None, version=None,
                            frozen=None, which=None, windows=None):
    """Interpreters that could build ``spec``'s environment, best first.

    spaCR's own Python when it is in range, then ``python3.X`` on PATH (or
    the ``py -3.X`` launcher on Windows), newest first. Nothing is RUN here:
    the install's preflight checks each candidate off the GUI thread.

    :returns: a list of argv tuples.
    """
    lo, hi = spec.python
    version = tuple(sys.version_info[:2]) if version is None else tuple(version)
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    which = shutil.which if which is None else which
    windows = (os.name == "nt") if windows is None else windows
    found = []
    if not frozen and lo <= version <= hi:
        found.append((executable or sys.executable,))
    for minor in range(hi[1], lo[1] - 1, -1):
        if windows:
            launcher = which("py")
            if launcher:
                found.append((launcher, f"-3.{minor}"))
        else:
            path = which(f"python3.{minor}")
            if path:
                found.append((path,))
    unique = []
    for candidate in found:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _candidates(spec):
    """:func:`_interpreter_candidates`, looked up once per process."""
    if spec.name not in _CANDIDATES:
        _CANDIDATES[spec.name] = _interpreter_candidates(spec)
    return _CANDIDATES[spec.name]


def _versions(pair):
    """``(3, 12)`` as ``3.12``."""
    return f"{pair[0]}.{pair[1]}"


def _nearest_existing(path):
    """``path``, or the closest folder above it that exists."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _static_blocker(spec, root, candidates):
    """What stops ``spec`` being installed here, from facts that need no
    network and no subprocess; ``''`` when nothing does."""
    if not any(sys.platform.startswith(p) for p in spec.platforms):
        return f"{spec.label} does not run on {sys.platform}."
    if _worker_path() is None:
        return ("this spaCR build ships no Python source for a backend "
                "worker to run.")
    if not candidates:
        lo, hi = (_versions(p) for p in spec.python)
        return (f"{spec.label} needs Python {lo} to {hi}. spaCR runs "
                f"Python {_versions(sys.version_info[:2])}, and no Python "
                f"{lo} to {hi} was found on this computer to build its "
                f"environment with.")
    where = _nearest_existing(root)
    if not os.access(where, os.W_OK | os.X_OK):
        return (f"spaCR cannot write to {where}, where backend environments "
                f"are kept. Set {_ROOT_ENV} to a folder it can write.")
    return ""


def _backend_state(name, root=None):
    """Where ``name`` stands: installed, installable, installing or not
    installable here, and why.

    Cheap enough for the GUI thread: a few file checks under the backends
    folder, and no network and no subprocess. A network or interpreter
    problem found by an earlier probe (:func:`_probe_blockers`, or an install
    that stopped in its preflight) is reported for :data:`_PROBE_SECONDS`.

    :param name: an optional backend.
    :param root: the backends folder; see :func:`_backends_root`.
    :returns: a :class:`_BackendState`.
    """
    spec = _spec(name)
    root = _backends_root(root)
    env = os.path.join(root, spec.name)
    lock = _read_lock(root, spec.name)
    if lock is not None:
        return _BackendState(
            spec.name, _INSTALLING,
            f"being installed since {lock.get('started', '?')} by process "
            f"{lock.get('pid', '?')}.", env)
    record = _read_marker(env)
    if record is not None and os.path.exists(_env_python(env)):
        return _BackendState(
            spec.name, _INSTALLED, f"in its own environment, {env}.", env,
            record)
    if spec.in_process and _importable(spec.module):
        return _BackendState(
            spec.name, _INSTALLED,
            f"inside spaCR's own environment, where an older spaCR "
            f"installed it; it runs inside spaCR. `pip uninstall "
            f"{spec.distribution}` removes it from there.", env,
            in_process=True)
    blocker = _static_blocker(spec, root, _candidates(spec))
    if blocker:
        return _BackendState(spec.name, _UNAVAILABLE, blocker, env)
    probed = _PROBED.get(spec.name)
    if probed is not None and time.time() - probed[1] < _PROBE_SECONDS:
        return _BackendState(spec.name, _UNAVAILABLE, probed[0], env)
    if record is not None:
        reason = (f"its environment at {env} lost its Python; installing "
                  f"builds it again.")
    elif os.path.isdir(env):
        reason = ("an earlier install did not finish; installing starts "
                  "again from the beginning.")
    else:
        reason = (f"installs into an environment of its own, {env}, and "
                  f"leaves spaCR's own environment alone.")
    return _BackendState(spec.name, _INSTALLABLE, reason, env)


def _not_installed_message(name, state=None):
    """The ImportError text for a backend that cannot segment yet."""
    spec = _spec(name)
    state = state or _backend_state(spec.name)
    return (
        f"{spec.label} is not installed ({state.state}: {state.reason}) "
        f"Install it from the Model Zoo, which builds it an environment of "
        f"its own under {state.env} and leaves spaCR's own environment "
        f"alone, or segment with segmentation_backend='cellpose'.")


def _probe_network(timeout=5.0, opener=None):
    """``''`` when pip's index answers, else the reason it did not.

    Any HTTP answer counts, a 403 or 404 included: the question is whether
    the index is reachable, and proxies set in the environment are honoured
    the way pip honours them.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    url = (os.environ.get("PIP_INDEX_URL", "").strip()
           or "https://pypi.org/simple/pip/")
    host = urllib.parse.urlsplit(url).hostname or url
    opener = urllib.request.urlopen if opener is None else opener
    try:
        with opener(urllib.request.Request(url, method="HEAD"),
                    timeout=timeout):
            return ""
    except urllib.error.HTTPError:
        return ""
    except Exception as exc:                                 # noqa: BLE001
        return f"no network: pip's index at {host} could not be reached ({exc})."


def _probe_blockers(names=None, root=None, probe=None):
    """Check the network for every backend that could be installed.

    For a background thread: rows then say "not installable here: no
    network" BEFORE the click rather than after it.

    :param names: the backends to check; every optional one when None.
    :param root: the backends folder.
    :param probe: the network check, for tests.
    :returns: ``{name: reason}`` for each backend the network blocks.
    """
    names = list(_SPECS) if names is None else [_spec(n).name for n in names]
    waiting = [n for n in names
               if _backend_state(n, root).state in (_INSTALLABLE, _UNAVAILABLE)]
    if not waiting:
        return {}
    problem = (probe or _probe_network)()
    blocked = {}
    for name in waiting:
        if problem:
            _PROBED[name] = (problem, time.time())
            blocked[name] = problem
        elif name in _PROBED and _PROBED[name][0].startswith("no network"):
            del _PROBED[name]
    return blocked


def _torch_index_url(version=None):
    """The PyTorch wheel index that matches spaCR's own PyTorch build.

    A backend on a CUDA machine wants a CUDA torch, and one on a CPU-only
    install should not download three gigabytes of CUDA libraries, so the
    backend gets the same kind of PyTorch spaCR has: ``2.5.1+cu124`` means
    the ``cu124`` index, ``+cpu`` the ``cpu`` one, and a plain version (PyPI's
    own build) means PyPI. ``$SPACR_BACKEND_TORCH_INDEX`` overrides it, and
    ``pypi`` there means PyPI.

    :param version: spaCR's torch version; read from its metadata when None.
    :returns: an index URL, or None for PyPI.
    """
    configured = os.environ.get(_TORCH_INDEX_ENV, "").strip()
    if configured:
        return None if configured.lower() in ("pypi", "default") else configured
    if version is None:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _version

        try:
            version = _version("torch")
        except PackageNotFoundError:
            return None
    local = str(version).partition("+")[2].strip().lower()
    if re.fullmatch(r"cpu|cu\d+|rocm[\d.]+|xpu", local):
        return _TORCH_WHEELS + local
    return None


@dataclass(frozen=True)
class _Step:
    """One command of an install.

    :param label: what the progress line says.
    :param argv: the command.
    :param selftest: whether its last line is the worker's hello.
    """

    label: str
    argv: tuple
    selftest: bool = False


def _install_plan(spec, env, interpreter, torch_index=None, worker=None):
    """The commands that build ``spec``'s environment, in order.

    Every pip here is the ENVIRONMENT'S pip, run as ``<env python> -m pip``;
    nothing is ever run against spaCR's own interpreter except ``-m venv``,
    which only reads it.

    :param spec: the backend.
    :param env: the environment's folder.
    :param interpreter: the argv of the Python that builds it.
    :param torch_index: a PyTorch wheel index, or None for PyPI.
    :param worker: this file's path, for the self-test.
    :returns: a list of :class:`_Step`.
    """
    python = _env_python(env)
    pip = (python, "-m", "pip", "install", "--disable-pip-version-check",
           "--no-input", "--progress-bar", "off")
    steps = [_Step("Create the environment",
                   tuple(interpreter) + ("-m", "venv", env)),
             _Step("Update pip", pip + ("--upgrade", "pip"))]
    if spec.torch:
        index = ("--index-url", torch_index) if torch_index else ()
        steps.append(_Step("Install PyTorch", pip + tuple(spec.torch) + index))
    steps.append(_Step(f"Install {spec.label}",
                       pip + tuple(spec.requirements)))
    steps.append(_Step(
        "Check it loads",
        (python, "-I", worker or _worker_path(), "--selftest", spec.name),
        selftest=True))
    return steps


def _clean_env(env):
    """The environment variables a backend's commands run with.

    Anything that would point pip or Python somewhere else is removed, the
    user's own site-packages is switched off, and the environment's scripts
    come first on PATH.
    """
    environ = {k: v for k, v in os.environ.items()
               if k not in _STRIPPED_VARIABLES}
    environ.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
                   PYTHONIOENCODING="utf-8", VIRTUAL_ENV=env,
                   PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    scripts = os.path.dirname(_env_python(env))
    environ["PATH"] = scripts + os.pathsep + environ.get("PATH", "")
    return environ


def _worker_env(name, env):
    """:func:`_clean_env`, and each backend's weights kept inside its own
    environment, so uninstalling removes them too.

    Cellpose 3 downloads its models where ``CELLPOSE_LOCAL_MODELS_PATH``
    says; DINOCell fetches its 383 MB checkpoint with ``hf_hub_download``,
    which reads ``HF_HOME`` and otherwise writes to the person's own
    ``~/.cache/huggingface``. Pointing it inside the environment is what
    makes :func:`_uninstall_backend` give the disk back, and what makes the
    preflight's free-space check -- which measures the backends folder --
    the check that matters.

    SAMCell has two downloads: its fine-tuned checkpoint uses Torch's hub
    cache, and its SAM backbone uses Transformers and Hugging Face. Both
    are scoped to the environment; legacy Transformers cache overrides
    must be removed alongside the Hugging Face overrides.

    Setting ``HF_HOME`` is necessary and not sufficient. :func:`_clean_env`
    forwards the rest of the inherited environment, and every variable in
    :data:`_HF_CACHE_VARIABLES` overrides the path ``HF_HOME`` would give,
    so they are dropped here as well. Without that, the one person the fix
    is for -- someone whose Hugging Face cache is already too big for their
    home disk, and who has moved it -- is the one person it would miss.
    """
    environ = _clean_env(env)
    if name == _CELLPOSE3:
        environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.join(env, "models")
    elif name in (_DINOCELL, _SAMCELL):
        environ["HF_HOME"] = os.path.join(env, "huggingface")
        for variable in _HF_CACHE_VARIABLES:
            environ.pop(variable, None)
        if name == _SAMCELL:
            environ["TORCH_HOME"] = os.path.join(env, "torch")
            for variable in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                             "PYTORCH_PRETRAINED_BERT_CACHE", "HF_MODULES_CACHE"):
                environ.pop(variable, None)
    return environ


def _deepcell_token_path():
    """Where spaCR looks for a DeepCell access token on disk.

    :returns: ``~/.spacr/deepcell_token``.
    """
    return os.path.join(os.path.expanduser("~"), ".spacr", "deepcell_token")


def _deepcell_token(environ=None, path=None):
    """The DeepCell access token SpotNet's weights are fetched with.

    ``DEEPCELL_ACCESS_TOKEN`` wins when it is set; otherwise the first line
    of ``~/.spacr/deepcell_token``, stripped. A token file other users can
    read is still used, with a warning naming the file and the fix; the
    token itself is never logged, printed or returned anywhere but here.

    :param environ: the variables to read, :data:`os.environ` when None.
    :param path: the token file, :func:`_deepcell_token_path` when None.
    :returns: ``(token, source)``; ``(None, None)`` when there is none.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get(_DEEPCELL_TOKEN_ENV, "") or "").strip()
    if value:
        return value, _DEEPCELL_TOKEN_ENV
    path = path or _deepcell_token_path()
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
        mode = os.stat(path).st_mode
    except (OSError, UnicodeDecodeError):
        return None, None
    if not value:
        return None, None
    if os.name != "nt" and mode & 0o077:
        LOG.warning("%s can be read by other users; `chmod 600 %s` keeps "
                    "the DeepCell token yours.", path, path)
    return value, path


def _serve_env(name, env):
    """:func:`_worker_env` for a running worker: SpotNet's also gets the
    DeepCell token, and no other process spaCR starts does, and a home
    inside its environment (:func:`_spotnet_home`) so the weights it fetches
    are removed with it. Installs keep the real home and its pip cache.
    """
    environ = _worker_env(name, env)
    if name == _SPOTNET:
        environ["HOME"] = environ["USERPROFILE"] = _spotnet_home(env)
        token, _source = _deepcell_token()
        if token:
            environ[_DEEPCELL_TOKEN_ENV] = token
    return environ


def _spotnet_home(env):
    """The home folder SpotNet's worker is given, inside its environment.

    DeepCell caches its weights under ``Path.home() / ".deepcell"`` and
    reads no variable that would move them, so the worker's home is this
    folder: the weights then live and die with the environment.
    """
    return os.path.join(env, "home")


def _spotnet_weights_cached(env):
    """Whether SpotNet's weights are already inside its environment."""
    return os.path.isfile(os.path.join(
        _spotnet_home(env), ".deepcell", "models", _SPOTNET_ARCHIVE))


def _credential_note(name, environ=None, token_path=None):
    """What a backend's Model Zoo row says about its credentials, or ''.

    Only SpotNet has any: where its DeepCell token goes, and whether spaCR
    found one. The token itself is never part of the note.
    """
    if name != _SPOTNET:
        return ""
    _token, source = _deepcell_token(environ, token_path)
    found = (f"A token was found in {source}." if source else
             "No token was found.")
    return (f"DeepCell access token: get a free one at users.deepcell.org, "
            f"then set {_DEEPCELL_TOKEN_ENV} or put it alone in "
            f"{token_path or _deepcell_token_path()} (chmod 600). Only "
            f"SpotNet's own worker is given it. {found}")


def _spotnet_readiness(root=None, environ=None, token_path=None):
    """Whether SpotNet can detect spots now, and why not when it cannot.

    It needs its environment installed and either its weights already
    fetched into that environment or a DeepCell token to fetch them with.

    :returns: ``(ready, reason)``; the reason says what to do.
    """
    state = _backend_state(_SPOTNET, root)
    if not state.ready or state.in_process:
        return False, (
            f"SpotNet is not installed ({state.state}: {state.reason}) "
            f"Install it from the Model Zoo.")
    token, _source = _deepcell_token(environ, token_path)
    if token or _spotnet_weights_cached(state.env):
        return True, f"SpotNet is installed in {state.env}."
    return False, (
        f"SpotNet is installed but has no DeepCell access token to fetch "
        f"its weights with. Get a free token at users.deepcell.org, then "
        f"set {_DEEPCELL_TOKEN_ENV} or put it alone in "
        f"{_deepcell_token_path()} (chmod 600).")


def _detect_spots(image, threshold=0.95, root=None, worker_for=None):
    """SpotNet's spots in one 2-D image, from its own environment.

    :param image: an ``H x W`` array.
    :param threshold: SpotNet's detection probability, 0 to 1.
    :param root: the backends folder.
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    :returns: an ``N x 2`` float array of ``(y, x)`` pixel coordinates.
    :raises ImportError: when SpotNet cannot run here, with the reason.
    """
    ready, reason = _spotnet_readiness(root)
    if not ready:
        raise ImportError(reason)
    env = _backend_state(_SPOTNET, root).env
    with tempfile.TemporaryDirectory(prefix="spacr_spotnet_") as folder:
        path = os.path.join(folder, "image.npy")
        np.save(path, np.ascontiguousarray(image, dtype=np.float32),
                allow_pickle=False)
        reply = (worker_for or _worker_for)(_SPOTNET, env).request(
            "detect_spots", image=path, threshold=float(threshold))
    spots = np.asarray(reply.get("spots") or [], dtype=float)
    return spots.reshape(-1, 2)


def _detached(windows=None):
    """Popen arguments that give a child its own process group, so Cancel
    can stop it and everything it started.

    :param windows: for Windows; the running system when None.
    """
    windows = (os.name == "nt") if windows is None else windows
    if windows:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP",
                                         0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def _kill_tree(proc, grace=5.0, windows=None):
    """Stop a child and everything it started; gently, then not.

    :param windows: for Windows; the running system when None.
    """
    windows = (os.name == "nt") if windows is None else windows
    if proc.poll() is not None:
        return
    try:
        if windows:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if windows:
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    proc.wait(timeout=grace)


_MAX_ABANDONED = 2


def _raw_lines(stream):
    """``stream``'s lines with their carriage returns kept.

    A pipe opened as text translates every ``\\r`` into a line break, and
    a progress bar's redraws then read as separate lines. Reading the
    underlying bytes with ``newline=''`` keeps each line's own ending, so
    :func:`_final_lines` can tell a redraw from a new line. A stand-in
    stream without bytes underneath is returned as it is.
    """
    import io

    raw = getattr(stream, "buffer", None)
    if raw is None:
        return stream
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace",
                            newline="")


def _pump(stream, sink, done=None):
    """Copy ``stream``'s lines into ``sink`` until it ends, then ``done``."""
    try:
        for line in stream:
            sink(line)
    except (OSError, ValueError):
        pass
    if done is not None:
        done()


def _run_step(argv, *, env=None, cwd=None, on_line=None, cancel=None,
              popen=None, poll=0.1):
    """Run one install command, streaming its output, until it ends or
    ``cancel`` is set.

    :param argv: the command.
    :param env: its environment variables.
    :param cwd: its working folder.
    :param on_line: called with every line it prints, stdout and stderr
        together, as it prints it.
    :param cancel: a :class:`threading.Event`; setting it stops the command
        and everything it started.
    :param popen: :class:`subprocess.Popen`, or a stand-in for tests.
    :param poll: seconds between checks of ``cancel``.
    :returns: ``(exit code, the last 400 lines)``.
    :raises _InstallCancelled: when cancelled.
    """
    proc = (popen or subprocess.Popen)(
        list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True,
        encoding="utf-8", errors="replace", bufsize=1, **_detached())
    lines = queue.Queue()
    finished = threading.Event()
    reader = threading.Thread(
        target=_pump, args=(proc.stdout, lines.put, finished.set), daemon=True)
    reader.start()
    tail = collections.deque(maxlen=400)

    def _drain():
        """Pass every line the reader has queued to ``tail`` and ``on_line``."""
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            text = line.rstrip("\r\n")
            tail.append(text)
            if on_line is not None:
                on_line(text)

    while True:
        if cancel is not None and cancel.is_set():
            _kill_tree(proc)
            reader.join(timeout=5)
            raise _InstallCancelled("the install was cancelled")
        _drain()
        if finished.is_set():
            try:
                code = proc.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                continue
            break
        finished.wait(timeout=poll)
    reader.join(timeout=5)
    _drain()
    return code, list(tail)


def _quote(argv):
    """A command as a person would type it."""
    import shlex

    return " ".join(shlex.quote(str(a)) for a in argv)


def _last_reply(lines):
    """The last JSON object among a command's output lines, or None."""
    for line in reversed(list(lines)):
        text = line.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except ValueError:
                continue
    return None


def _remove_tree(path, root):
    """Delete a backend environment, and refuse to delete anything else.

    :raises RuntimeError: for a path that is not directly inside ``root``, or
        that is the environment spaCR itself runs in.
    """
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    running = {os.path.abspath(p) for p in (sys.prefix, sys.base_prefix,
                                            sys.exec_prefix)}
    if os.path.dirname(path) != root or path in running:
        raise RuntimeError(
            f"refusing to delete {path}: it is not a backend environment "
            f"under {root}")
    if not os.path.lexists(path):
        return

    def _retry(function, target, *_error):
        """Windows marks some files read-only; make them writable and retry."""
        import stat

        os.chmod(target, stat.S_IWRITE)
        function(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry)
    else:
        shutil.rmtree(path, onerror=_retry)


def _preflight(spec, root, *, run=None, probe=None):
    """Check this computer can build ``spec``'s environment; pick its Python.

    Writable folder, free disk, a reachable package index, and a Python in
    range that has ``venv`` and ``ensurepip`` -- in that order, so the first
    thing wrong is the thing reported. Runs off the GUI thread.

    :returns: the argv of the Python to build it with.
    :raises _InstallBlocked: saying what is wrong.
    """
    probe_file = os.path.join(root, f".{spec.name}.write-test")
    try:
        with open(probe_file, "w", encoding="utf-8") as handle:
            handle.write("spaCR")
        os.remove(probe_file)
    except OSError as exc:
        raise _InstallBlocked(
            f"spaCR cannot write to {root}, where backend environments are "
            f"kept: {exc}") from exc
    free = shutil.disk_usage(root).free
    if free < spec.size_gb * 2 ** 30:
        raise _InstallBlocked(
            f"{spec.label} needs about {spec.size_gb:.0f} GB free in {root}; "
            f"{free / 2 ** 30:.1f} GB is.")
    problem = (probe or _probe_network)()
    if problem:
        _PROBED[spec.name] = (problem, time.time())
        raise _InstallBlocked(problem)
    run = subprocess.run if run is None else run
    lo, hi = spec.python
    tried = []
    for candidate in _candidates(spec):
        shown = " ".join(candidate)
        try:
            done = run(list(candidate) + ["-c", _INTERPRETER_CHECK],
                       capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            tried.append(f"{shown}: {exc}")
            continue
        if done.returncode != 0:
            said = (done.stderr or done.stdout or "").strip().splitlines()
            last = said[-1] if said else f"exit code {done.returncode}"
            if "ensurepip" in last or "venv" in last:
                last += (" (on Debian and Ubuntu the python3-venv package "
                         "provides it)")
            tried.append(f"{shown}: {last}")
            continue
        try:
            got = tuple(int(p) for p in done.stdout.split()[-1].split("."))
        except (IndexError, ValueError):
            tried.append(f"{shown}: did not say its version")
            continue
        if lo <= got <= hi:
            return tuple(candidate)
        tried.append(f"{shown} is Python {_versions(got)}")
    reason = (f"no Python {_versions(lo)} to {_versions(hi)} that can build "
              f"an environment was found")
    if tried:
        reason += ": " + "; ".join(tried)
    reason += "."
    _PROBED[spec.name] = (reason, time.time())
    raise _InstallBlocked(reason)


def _install_backend(name, *, root=None, progress=None, cancel=None,
                     torch_index=None, runner=None, preflight=None,
                     worker=None):
    """Build ``name``'s environment and install it there. Off the GUI thread.

    The environment is marked finished only after its self-test has loaded
    the package, so a failed or cancelled install leaves no environment that
    looks usable -- the folder is removed, the row goes back to
    "installable", and spaCR's own environment is never touched either way.
    The whole output goes to ``<root>/<name>.log``, which stays behind for a
    bug report.

    :param name: the backend.
    :param root: the backends folder.
    :param progress: ``progress(step, steps, text)``, called as it goes.
    :param cancel: a :class:`threading.Event` that stops it.
    :param torch_index: the PyTorch wheel index; :func:`_torch_index_url`
        when None, PyPI when ``''``.
    :param runner: :func:`_run_step`, or a stand-in for tests.
    :param preflight: :func:`_preflight`, or a stand-in for tests.
    :param worker: the worker's path, for tests.
    :returns: the :class:`_BackendState` afterwards.
    :raises _InstallBlocked: when this computer cannot install it.
    :raises _InstallFailed: when a step failed, with its output.
    :raises _InstallCancelled: when cancelled.
    """
    spec = _spec(name)
    root = _backends_root(root)
    env = os.path.join(root, spec.name)
    state = _backend_state(spec.name, root)
    if state.state == _INSTALLED:
        return state
    report = progress or (lambda step, steps, text: None)
    report(0, 1, "Checking this computer can install it")
    _acquire_lock(root, spec.name)
    log_path = os.path.join(root, f"{spec.name}.log")
    try:
        interpreter = (preflight or _preflight)(spec, root)
        _PROBED.pop(spec.name, None)
        if os.path.lexists(env):
            _remove_tree(env, root)
        index = _torch_index_url() if torch_index is None else (torch_index or None)
        steps = _install_plan(spec, env, interpreter, torch_index=index,
                              worker=worker)
        hello = None
        with open(log_path, "w", encoding="utf-8") as log:
            for number, step in enumerate(steps):
                report(number, len(steps), step.label)
                log.write(f"$ {_quote(step.argv)}\n")
                log.flush()

                def _line(text, _number=number, _label=step.label):
                    """Log one output line of this step and report it as progress."""
                    log.write(text + "\n")
                    report(_number, len(steps), f"{_label}: {text}")

                code, tail = (runner or _run_step)(
                    step.argv, env=_worker_env(spec.name, env), cwd=root,
                    on_line=_line, cancel=cancel)
                if code != 0:
                    shown = "\n".join(tail[-40:]) or "(it printed nothing)"
                    raise _InstallFailed(
                        f"{step.label} failed: `{_quote(step.argv)}` exited "
                        f"with code {code}.\n\n{shown}\n\nThe whole log is "
                        f"{log_path}.")
                if step.selftest:
                    hello = _last_reply(tail)
        if not hello or not hello.get("ok"):
            error = (hello or {}).get("error") or {}
            raise _InstallFailed(
                f"The environment was built, but {spec.label} does not load "
                f"in it: {error.get('type', '')} {error.get('message', '')}"
                f"\n\n{error.get('traceback', '')}\nThe whole log is "
                f"{log_path}.")
        _write_marker(env, {
            "backend": spec.name, "protocol": _PROTOCOL,
            "requirements": list(spec.requirements),
            "torch": list(spec.torch), "torch_index": index or "",
            "interpreter": list(interpreter),
            "python": hello.get("python", ""),
            "packages": hello.get("packages", {}),
            "device": hello.get("device", ""),
            "licence": spec.licence,
            "installed": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        report(len(steps), len(steps), f"{spec.label} is installed")
    except BaseException:
        if os.path.lexists(env):
            try:
                _remove_tree(env, root)
            except (OSError, RuntimeError):
                LOG.warning("could not remove the unfinished %s", env,
                            exc_info=True)
        raise
    finally:
        _release_lock(root, spec.name)
    return _backend_state(spec.name, root)


def _uninstall_backend(name, root=None):
    """Remove ``name``'s environment, and everything it downloaded into it.

    :returns: the :class:`_BackendState` afterwards.
    :raises RuntimeError: while it is being installed, or when it lives in
        spaCR's own environment, which spaCR never changes.
    """
    spec = _spec(name)
    root = _backends_root(root)
    state = _backend_state(spec.name, root)
    if state.state == _INSTALLING:
        raise RuntimeError(
            f"{spec.label} is {state.reason} Cancel that install first.")
    if state.in_process and not os.path.isdir(state.env):
        raise RuntimeError(
            f"{spec.label} is installed {state.reason}")
    _shutdown_workers(spec.name)
    _remove_tree(state.env, root)
    _PROBED.pop(spec.name, None)
    return _backend_state(spec.name, root)


def _plain(value):
    """A request parameter as a JSON value."""
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float)):
        return value
    return float(value)


class _WorkerProcess:
    """One backend worker: a Python in the backend's environment, serving
    requests over a pipe.

    :param name: the backend.
    :param env: its environment.
    :param popen: :class:`subprocess.Popen`, or a stand-in for tests.
    :param worker: the worker script; this file when None.
    """

    _abandoned = frozenset()
    _overwrite = False

    def __init__(self, name, env, *, popen=None, worker=None):
        """Start the worker and ask it hello, which loads the package."""
        spec = _spec(name)
        self.name = spec.name
        self.label = spec.label
        self.env = env
        self.last_used = time.monotonic()
        self._replies = queue.Queue()
        self._stderr = collections.deque(maxlen=200)
        self._overwrite = False
        self._abandoned = set()
        self._lock = threading.Lock()
        self._next_id = 0
        self._proc = (popen or subprocess.Popen)(
            [_env_python(env), "-I", worker or _worker_path(), "--serve",
             spec.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=env, env=_serve_env(spec.name, env),
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            **_detached())
        threading.Thread(
            target=_pump, args=(self._proc.stdout, self._took,
                                lambda: self._replies.put(None)),
            daemon=True).start()
        threading.Thread(
            target=_pump, args=(_raw_lines(self._proc.stderr), self._said),
            daemon=True).start()
        try:
            self.hello = self.request("hello")
        except BaseException:
            self.kill()
            raise

    def _said(self, line):
        """Keep the worker's own output for an error message.

        A line that ended in a carriage return is a progress bar about to
        redraw itself, and the next line takes its place (see
        :func:`_final_lines`), so the tail an error quotes holds each bar
        once, in its last state.
        """
        shown = _final_lines([line])
        text = shown[-1] if shown else ""
        if self._overwrite and self._stderr:
            self._stderr[-1] = text or self._stderr[-1]
        elif text or not line.endswith("\r"):
            self._stderr.append(text)
        self._overwrite = line.endswith("\r") and not line.endswith("\r\n")
        if not self._overwrite:
            LOG.debug("%s: %s", self.name, text)

    def _took(self, line):
        """Queue one reply, dropping those to requests nobody waits for."""
        try:
            ident = json.loads(line).get("id")
        except (ValueError, AttributeError):
            ident = None
        if ident is not None and ident in self._abandoned:
            self._abandoned.discard(ident)
            return
        self._replies.put(line)

    @property
    def alive(self):
        """Whether the worker is still running."""
        return self._proc.poll() is None

    @property
    def busy(self):
        """Whether a request is in flight, heard or abandoned."""
        return self._lock.locked() or bool(self._abandoned)

    def _abandon(self, ident):
        """Stop waiting for request ``ident`` without stopping the worker.

        The worker is told, so a request still in its queue is skipped
        rather than run; one already running finishes and its reply is
        dropped. The worker keeps its loaded models, which is the point:
        restarting it costs the environment's Python, torch, and every
        model again (item 507 measured about 17 seconds for Cellpose 3's
        restoration on this machine). Past :data:`_MAX_ABANDONED` requests
        still owed, or when the worker cannot be told, it is stopped as
        before.
        """
        self._abandoned.add(ident)
        if len(self._abandoned) > _MAX_ABANDONED:
            self.kill()
            return
        try:
            self._proc.stdin.write(json.dumps(
                {"protocol": _PROTOCOL, "id": 0, "op": "cancel",
                 "target": ident}) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            self.kill()

    def _stopped(self):
        """The error for a worker that went away, with its last words."""
        try:
            code = self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
        tail = "\n".join(_final_lines(
            "\n".join(list(self._stderr)[-40:]))) or "(it printed nothing)"
        return _BackendError(
            f"The {self.label} backend stopped (exit code {code}). Its last "
            f"output:\n{tail}")

    def request(self, op, *, should_cancel=None, keep_on_cancel=False,
                **payload):
        """Send one request and wait for its reply.

        :param op: ``hello``, ``segment`` or ``shutdown``.
        :param should_cancel: polled while waiting; True stops the worker.
        :param keep_on_cancel: on a cancel, leave the worker running and
            abandon the request instead (:meth:`_abandon`). For short
            requests on a worker whose loaded models are worth keeping; a
            long batch should stop, which is the default.
        :param payload: the request's other fields.
        :returns: the reply.
        :raises _BackendError: with the backend's own message, verbatim.
        :raises _BackendCancelled: when ``should_cancel`` said so.
        """
        with self._lock:
            self._next_id += 1
            ident = self._next_id
            message = dict(payload, protocol=_PROTOCOL, id=ident, op=op)
            try:
                self._proc.stdin.write(json.dumps(message) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError):
                raise self._stopped() from None
            while True:
                try:
                    line = self._replies.get(timeout=0.2)
                except queue.Empty:
                    if should_cancel is not None and should_cancel():
                        if keep_on_cancel:
                            self._abandon(ident)
                        else:
                            self.kill()
                        raise _BackendCancelled(
                            f"the {self.label} request was cancelled") from None
                    continue
                if line is None:
                    raise self._stopped()
                try:
                    reply = json.loads(line)
                except ValueError:
                    raise _BackendError(
                        f"The {self.label} backend answered with something "
                        f"that is not a reply: {line.strip()[:500]}") from None
                if not isinstance(reply, dict) or reply.get("id") != ident:
                    if (isinstance(reply, dict)
                            and reply.get("id") in self._abandoned):
                        self._abandoned.discard(reply.get("id"))
                    continue
                self.last_used = time.monotonic()
                if reply.get("protocol") != _PROTOCOL:
                    raise _BackendError(
                        f"The {self.label} backend speaks protocol "
                        f"{reply.get('protocol')!r} and spaCR speaks "
                        f"{_PROTOCOL}. Reinstall it from the Model Zoo.")
                if not reply.get("ok"):
                    error = reply.get("error") or {}
                    kind = error.get("type") or "an error"
                    raise _BackendError(
                        f"{self.label} raised {kind}: "
                        f"{error.get('message', '')}", error.get("type", ""),
                        error.get("traceback", ""))
                return reply

    def kill(self):
        """Stop the worker now, mid-request if need be."""
        _kill_tree(self._proc)

    def close(self, timeout=5.0):
        """Ask the worker to finish, and stop it if it does not."""
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps(
                    {"protocol": _PROTOCOL, "id": 0, "op": "shutdown"}) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.kill()
        try:
            self._proc.stdin.close()
        except (OSError, ValueError):
            pass


#: ``name -> _WorkerProcess``: one worker per backend, shared by everything
#: in this process that segments with it.
_WORKERS = {}
_WORKERS_LOCK = threading.Lock()
_REAPER = []


def _worker_for(name, env, factory=None):
    """The running worker for ``name``, started (or restarted) on demand.

    HANDING A WORKER OUT COUNTS AS USING IT. ``last_used`` used to move only
    when a reply arrived, and ``busy`` only while a request is in flight, so
    between this function releasing the lock and the caller's first write
    there was a moment where a worker idle for :data:`_IDLE_SECONDS` was
    neither busy nor recently used -- and :func:`_reap_idle`, which runs on
    its own thread, could shut it down in that moment. The write then failed
    and was reported as "the backend stopped (exit code 0)". A mask run with
    more than ten minutes between batches is exactly the run that hits it.
    Stamping it here, under the lock, means the worker this call returns
    cannot be reaped out from under its caller.
    """
    with _WORKERS_LOCK:
        worker = _WORKERS.get(name)
        if worker is not None and (not worker.alive or worker.env != env):
            worker.close()
            worker = None
        if worker is None:
            worker = (factory or _WorkerProcess)(name, env)
            _WORKERS[name] = worker
        worker.last_used = time.monotonic()
        if not _REAPER:
            reaper = threading.Thread(target=_reap_forever, daemon=True)
            _REAPER.append(reaper)
            reaper.start()
        return worker


def _reap_idle(now=None, idle=_IDLE_SECONDS):
    """Shut down workers that are dead or have been idle for ``idle``
    seconds; the memory a loaded model holds is given back."""
    now = time.monotonic() if now is None else now
    with _WORKERS_LOCK:
        for name, worker in list(_WORKERS.items()):
            if worker.alive and (worker.busy or now - worker.last_used < idle):
                continue
            worker.close()
            del _WORKERS[name]


def _reap_forever(interval=30.0, rounds=None):
    """:func:`_reap_idle` every ``interval`` seconds, on a daemon thread."""
    count = 0
    while rounds is None or count < rounds:
        time.sleep(interval)
        _reap_idle()
        count += 1


def _shutdown_workers(name=None):
    """Close ``name``'s worker, or every worker when None."""
    with _WORKERS_LOCK:
        for key in [k for k in _WORKERS if name is None or k == name]:
            _WORKERS.pop(key).close()


atexit.register(_shutdown_workers)


@dataclass(frozen=True)
class _RestorationPlan:
    """Immutable model identity captured off the GUI thread before Apply.

    Including checkpoint and environment identity in a request key prevents
    enhanced image caches surviving a model change or backend reinstall.
    """

    env: str
    model: str
    diameter: float
    device: str
    weights_sha256: str
    cellpose_version: str

    def _identity(self):
        """The identity the worker must still have when inference starts."""
        return {"backend": _CELLPOSE3, "model": self.model,
                "device": self.device, "weights_sha256": self.weights_sha256,
                "cellpose_version": self.cellpose_version}


def _restoration_plan(model, diameter, *, root=None, device="cpu",
                      should_cancel=None, worker_for=None):
    """Load an isolated restoration model and capture its identity.

    Call from a background worker: first use may download weights. Backend
    installation remains an explicit Model Zoo action. CPU is the default;
    no application-wide automatic accelerator selection is used here.
    """
    if model not in _RESTORATION_MODELS:
        raise ValueError(f"unsupported same-grid restoration model: {model!r}")
    diameter = float(diameter)
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("restoration diameter must be finite and positive")
    _check_restoration_cancel(should_cancel)
    state = _backend_state(_CELLPOSE3, root)
    if state.state != _INSTALLED or state.in_process:
        raise ImportError(_not_installed_message(_CELLPOSE3, state))
    worker = (worker_for or _worker_for)(_CELLPOSE3, state.env)
    reply = worker.request("restoration_model", should_cancel=should_cancel,
                           keep_on_cancel=True, model=model,
                           device=device or "cpu")
    _check_restoration_cancel(should_cancel)
    identity = reply["identity"]
    return _RestorationPlan(
        env=state.env, model=model, diameter=diameter,
        device=identity["device"], weights_sha256=identity["weights_sha256"],
        cellpose_version=identity["cellpose_version"])


def _check_restoration_cancel(should_cancel):
    """Discard cancelled work even when its reply has already arrived."""
    if should_cancel is not None and should_cancel():
        raise _BackendCancelled("the restoration request was cancelled")


_KEEP_PIXELS = 1 << 20


def _keep_restoring(source, plan):
    """Whether a cancelled restoration should finish in its worker rather
    than stop it.

    Stopping the worker throws away its Python, torch and loaded models,
    about 5 to 10 seconds to rebuild on this machine's CPU (item 507), and
    the next request pays that. Letting the cancelled request finish costs
    whatever it had left. On a GPU, or on a CPU for a plane of at most
    :data:`_KEEP_PIXELS` (a magnifier box: about half a second), finishing
    is the cheaper; a whole field on a CPU (18 seconds for 1994 x 1994) is
    cheaper to stop.
    """
    on_cpu = str(getattr(plan, "device", "cpu") or "cpu").startswith("cpu")
    return not on_cpu or int(np.asarray(source).size) <= _KEEP_PIXELS


def _restore_plane(image, plan, *, should_cancel=None, worker_for=None):
    """Return a restored copy and provenance for one captured model plan.

    This blocking operation belongs on a background thread. Scratch files
    are removed after success, failure or cancellation. Output values retain
    normalized model units and are never cast back to the source's dtype.
    """
    _check_restoration_cancel(should_cancel)
    source = np.asarray(image)
    if (source.ndim != 2 or min(source.shape) < 2
            or source.dtype.kind not in "uif" or not np.isfinite(source).all()):
        raise ValueError("restoration needs one finite real intensity plane")
    worker = (worker_for or _worker_for)(_CELLPOSE3, plan.env)
    with tempfile.TemporaryDirectory(prefix="spacr-restoration-") as scratch:
        input_path = os.path.join(scratch, "input.npy")
        output_path = os.path.join(scratch, "output.npy")
        np.save(input_path, source, allow_pickle=False)
        reply = worker.request(
            "restore", should_cancel=should_cancel,
            keep_on_cancel=_keep_restoring(source, plan), model=plan.model,
            diameter=plan.diameter, device=plan.device,
            expected_identity=plan._identity(), input=input_path,
            output=output_path)
        _check_restoration_cancel(should_cancel)
        restored = np.load(output_path, allow_pickle=False)
        record = reply["provenance"]
        if any(record.get(key) != value
               for key, value in plan._identity().items()):
            raise _BackendError("restoration model changed; select the model again")
        if (restored.shape != source.shape or restored.dtype != np.float32
                or not np.isfinite(restored).all()):
            raise _BackendError("restoration returned an invalid intensity plane")
        _check_restoration_cancel(should_cancel)
        return restored, dict(record)


class _RemoteBackend:
    """A backend in its own environment, answering ``CellposeModel.eval``.

    The images go to the worker as ``.npy`` files; the masks, and whatever
    flows the backend has, come back the same way.

    :param name: the backend.
    :param model: the model it runs -- a Cellpose 3 name or a checkpoint
        path; ``''`` for backends with one model.
    :param device: ``'cpu'``, ``'cuda'``, ... or None for ``$SPACR_DEVICE``,
        and failing that the worker's own best guess.
    :param root: the backends folder.
    :param options: passed to the backend (``weights_path``, ``variant``).
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    :raises ImportError: when its environment is not installed.
    """

    def __init__(self, name, *, model="", device=None, root=None,
                 options=None, worker_for=None):
        """Check the environment is there and remember what to run."""
        spec = _spec(name)
        state = _backend_state(spec.name, root)
        if state.state != _INSTALLED or state.in_process:
            raise ImportError(_not_installed_message(spec.name, state))
        self.name = spec.name
        self.label = spec.label
        self.model = str(model or "")
        self.env = state.env
        self.device = (str(device) if device is not None
                       else os.environ.get(_DEVICE_ENV, "").strip() or "auto")
        self.options = dict(options or {})
        self._worker_for = worker_for or _worker_for
        self.note = (f"in its own environment, {self.env}"
                     + (f"; model {self.model}" if self.model else ""))
        self._said = set()

    def eval(self, x, batch_size=None, channel_axis=-1, normalize=True,
             diameter=None, flow_threshold=None, cellprob_threshold=0.0,
             min_size=None, resample=None, progress=None, should_cancel=None,
             **cellpose_only):
        """Segment each image of a batch in the backend's worker.

        :param x: a 2-D image, or a list of ``(H, W)`` / ``(H, W, C)``
            images.
        :param should_cancel: polled while the worker runs; True stops it.
        :param cellpose_only: other Cellpose arguments, accepted so the call
            site is the same as Cellpose's.
        :returns: ``(masks, flows, None)`` with one entry per image.
        :raises _BackendError: with the backend's own message.
        """
        images = ([x] if isinstance(x, np.ndarray) and x.ndim == 2
                  else list(x))
        params = {"channel_axis": channel_axis, "normalize": normalize,
                  "diameter": diameter, "flow_threshold": flow_threshold,
                  "cellprob_threshold": cellprob_threshold,
                  "min_size": min_size, "resample": resample,
                  "batch_size": batch_size}
        params = {k: _plain(v) for k, v in params.items() if v is not None}
        worker = self._worker_for(self.name, self.env)
        scratch = tempfile.mkdtemp(prefix="spacr-backend-")
        try:
            inputs = []
            for index, image in enumerate(images):
                path = os.path.join(scratch, f"image_{index}.npy")
                np.save(path, np.asarray(image), allow_pickle=False)
                inputs.append(path)
            reply = worker.request(
                "segment", should_cancel=should_cancel, model=self.model,
                device=self.device, inputs=inputs, outputs=scratch,
                params=params, options=self.options)
            self._report(reply)
            masks, flows = [], []
            for item in reply.get("outputs") or ():
                masks.append(np.load(item["mask"], allow_pickle=False))
                flows.append([None if p is None
                              else np.load(p, allow_pickle=False)
                              for p in item.get("flows") or ()])
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if len(masks) != len(images):
            raise _BackendError(
                f"The {self.label} backend returned {len(masks)} masks for "
                f"{len(images)} images.")
        return masks, flows, None

    def _report(self, reply):
        """Say once, by name, what the backend could not honour.

        A setting the user set and the run ignored is a result nobody can
        account for afterwards, which is why this is printed rather than
        dropped. Once per backend instance: a run is thousands of batches
        and the same line thousands of times is the same as no line.
        """
        for key, lead in (("translated", "translated"),
                          ("ignored", "cannot honour")):
            values = list(reply.get(key) or ())
            if not values:
                continue
            fresh = [v for v in values if (key, v) not in self._said]
            if not fresh:
                continue
            self._said.update((key, v) for v in fresh)
            if key == "ignored":
                print(f"{self.label} {lead}: {', '.join(fresh)} -- "
                      f"these settings did not reach the model",
                      file=sys.stderr)
            else:
                for value in fresh:
                    print(f"{self.label} {lead} {value}", file=sys.stderr)


def _import_dinocell():
    """Import DINOCell's model, pipeline factory and weight resolver.

    :returns: ``(DINOCell, get_pipeline, get_weights_path)``.
    :raises ImportError: saying where DINOCell is installed from.
    """
    try:
        from dinocell.main import get_weights_path
        from dinocell.model import DINOCell
        from dinocell.pipeline import get_pipeline
    except (ImportError, OSError) as exc:
        raise ImportError(
            "DINOCell is not installed. Install it from the Model Zoo, which "
            "builds it an environment of its own and leaves spaCR's own "
            "environment alone -- DINOCell pins exact versions of torch, "
            "numpy and Cellpose that conflict with spaCR's -- or segment "
            "with segmentation_backend='cellpose'. Cellpose segmentation is "
            f"unaffected. The import failed with: {exc}"
        ) from exc
    return DINOCell, get_pipeline, get_weights_path


def _import_samcell():
    """Import SAMCell's model wrapper and sliding-window pipeline.

    :returns: ``(FinetunedSAM, SlidingWindowPipeline)``.
    :raises ImportError: saying where SAMCell is installed from.
    """
    try:
        from samcell.model import FinetunedSAM
        from samcell.pipeline import SlidingWindowPipeline
    except (ImportError, OSError) as exc:
        raise ImportError(
            "SAMCell is not installed. Install it from the Model Zoo, which "
            "builds it an environment of its own and leaves spaCR's own "
            "environment alone, or segment with "
            "segmentation_backend='cellpose'. Cellpose segmentation is "
            f"unaffected. The import failed with: {exc}"
        ) from exc
    return FinetunedSAM, SlidingWindowPipeline


def _object_plane(image, channel_axis=-1):
    """The object's own channel of one batch image, as a 2-D array.

    :param image: ``(H, W)`` or ``(H, W, C)`` array.
    :param channel_axis: the axis ``model.eval`` was told holds channels.
    :returns: ``(H, W)`` array.
    :raises ValueError: for anything that is not one 2-D plane.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(
            f"segmentation backends other than Cellpose take 2-D images; got "
            f"an array of shape {arr.shape}")
    axis = -1 if channel_axis is None else channel_axis
    return np.take(arr, 0, axis=axis)


def _to_uint8(plane):
    """Min-max stretch a plane into ``uint8``; a flat plane becomes zeros.

    :param plane: 2-D numeric array; non-finite pixels take the minimum.
    :returns: ``uint8`` array of the same shape.
    """
    arr = np.asarray(plane, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    scaled = (np.where(finite, arr, lo) - lo) / (hi - lo)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def _tile_starts(length, crop, overlap):
    """Distinct tile origins along one axis, as DINOCell's sliding window.

    ``SlidingWindowHelper.seperate_into_crops_v2`` steps by
    ``crop - 2 * overlap`` and clamps the last tile to the edge, which repeats
    that tile whenever the clamp lands on an earlier origin. Averaging a
    repeat changes nothing, so running it once gives the same prediction
    for less compute.

    :returns: sorted list of origins.
    """
    if length <= crop:
        return [0]
    stride = max(1, crop - 2 * overlap)
    return sorted({min(start, length - crop)
                   for start in range(0, length, stride)})


def _resize_nearest(array, shape):
    """Nearest-neighbour resample of the last two axes to ``shape``.

    :param array: ``(..., h, w)`` array (labels stay labels).
    :param shape: target ``(H, W)``.
    :returns: ``(..., H, W)`` array of the same dtype.
    """
    arr = np.asarray(array)
    h, w = arr.shape[-2:]
    height, width = int(shape[0]), int(shape[1])
    if (h, w) == (height, width):
        return arr
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(np.intp),
                      h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(np.intp),
                      w - 1)
    return arr[..., rows[:, None], cols[None, :]]


def _as_label_image(labels):
    """Sequential labels in the dtype Cellpose returns (``uint16``, or
    ``uint32`` past 65,535 objects).

    Relabelled with numpy alone, in the order of the original ids -- what
    ``skimage.segmentation.relabel_sequential`` does -- because the Cellpose
    3 environment has no scikit-image.

    :param labels: 2-D integer label image, background 0.
    :returns: relabelled array.
    """
    arr = np.asarray(labels)
    if arr.size and arr.max() > 0:
        values, inverse = np.unique(arr.astype(np.int64, copy=False),
                                    return_inverse=True)
        arr = inverse.reshape(arr.shape) + (0 if values[0] == 0 else 1)
    dtype = np.uint16 if arr.max(initial=0) < 2 ** 16 else np.uint32
    return arr.astype(dtype, copy=False)


def _probability_threshold(cellprob_threshold):
    """A Cellpose cell-probability logit threshold as a probability.

    :param cellprob_threshold: Cellpose's threshold, or ``None`` for 0.
    :returns: ``1 / (1 + exp(-threshold))``; 0 maps to 0.5.
    """
    if cellprob_threshold is None:
        return 0.5
    value = min(50.0, max(-50.0, float(cellprob_threshold)))
    return 1.0 / (1.0 + math.exp(-value))


class _PlaneBackend:
    """A single-channel 2-D segmenter that answers ``CellposeModel.eval``.

    Subclasses implement :meth:`_segment_plane`; this class turns a batch into
    the ``(masks, flows, styles)`` triple ``parse_cellpose4_output`` reads.
    """

    name = "plane"
    #: What the backend does with the Cellpose settings the eval call passes.
    note = ""

    def __init__(self, device=None):
        """Record the device, resolving spaCR's accelerator when None."""
        if device is None:
            from . import accelerator

            device = accelerator.torch_device()
        self.device = device

    def eval(self, x, batch_size=None, channel_axis=-1,
             cellprob_threshold=0.0, flow_threshold=None, progress=None,
             **cellpose_only):
        """Segment each image of a batch.

        :param x: list of ``(H, W, C)`` (or ``(H, W)``) images.
        :param channel_axis: axis holding channels; the first channel is used.
        :param cellprob_threshold: Cellpose logit threshold, used by backends
            that predict a cell probability.
        :param cellpose_only: diameter, min_size, resample and the other
            Cellpose arguments; accepted so the call site is unchanged.
        :returns: ``(masks, flows, None)`` with one entry per image.
        """
        images = ([x] if isinstance(x, np.ndarray) and x.ndim == 2
                  else list(x))
        masks, flows = [], []
        for image in images:
            plane = _object_plane(image, channel_axis)
            labels, flow = self._segment_plane(
                _to_uint8(plane), cellprob_threshold=cellprob_threshold)
            labels = np.asarray(labels)
            if labels.shape != plane.shape:
                raise ValueError(
                    f"the {self.name} backend returned labels of shape "
                    f"{labels.shape} for an image of shape {plane.shape}")
            masks.append(_as_label_image(labels))
            flows.append(flow)
        return masks, flows, None

    def _segment_plane(self, image, cellprob_threshold=None):
        """Label one ``uint8`` plane.

        :returns: ``(labels, flow)`` where ``flow`` is the per-image list
            ``[display image, dP or None, probability or None, None]``.
        """
        raise NotImplementedError


class _DinoCellBackend(_PlaneBackend):
    """DINOCell: DINOv2 ViT-B predicting Cellpose-style flows."""

    name = _DINOCELL
    note = ("cellprob_threshold is applied to DINOCell's cell probability "
            "through the logistic function (0 -> 0.5); diameter, "
            "flow_threshold, min_size and resample are Cellpose settings and "
            "are not used")

    def __init__(self, device=None, weights_path=None):
        """Load DINOCell's checkpoint (from the Hugging Face cache unless
        ``weights_path`` names one) and build its flows pipeline."""
        super().__init__(device)
        DINOCell, get_pipeline, get_weights_path = _import_dinocell()
        import torch

        self.device = torch.device(self.device)
        weights = weights_path or get_weights_path()
        model = DINOCell(
            dino_model=None, decoder_type="upsample", objective_type="flows",
            use_dino_weights=False, patch_size=8, feat_size=64,
            crop_size=_DINOCELL_CROP, drop_rate=0.05, dropout_in_encoder=True,
            finetune_vision=True, finetune_decoder=True,
            finetune_prediction_head=True)
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.to(self.device).eval()
        self._pipeline = get_pipeline(
            "flows", model, self.device, crop_size=_DINOCELL_CROP,
            use_advanced_augmentations=False, overlap_size=_DINOCELL_OVERLAP)

    def _predict(self, image):
        """``(dx, dy, cell probability)`` for a plane at least one crop wide.

        :param image: ``uint8`` plane with both sides >= the crop.
        :returns: ``(3, H, W)`` float32 array.
        """
        crop = _DINOCELL_CROP
        height, width = image.shape
        total = np.zeros((3, height, width), np.float32)
        count = np.zeros((height, width), np.float32)
        for y in _tile_starts(height, crop, _DINOCELL_OVERLAP):
            for x in _tile_starts(width, crop, _DINOCELL_OVERLAP):
                tile = np.ascontiguousarray(image[y:y + crop, x:x + crop])
                preds = self._pipeline.get_model_prediction(tile)
                for channel, pred in enumerate(preds[:3]):
                    total[channel, y:y + crop, x:x + crop] += (
                        pred[0].detach().float().cpu().numpy())
                count[y:y + crop, x:x + crop] += 1.0
        return total / count

    def _segment_plane(self, image, cellprob_threshold=None):
        """Predict flows, then label them with Cellpose's dynamics.

        A plane narrower than one tile is upscaled (aspect ratio kept, where
        DINOCell's own ``_resize`` would make it square) and the labels are
        resampled back to the plane's own shape.
        """
        import cv2
        from cellpose.dynamics import compute_masks
        from cellpose.plot import dx_to_circ

        height, width = image.shape
        scale = _DINOCELL_CROP / min(height, width)
        if scale > 1:
            size = (max(_DINOCELL_CROP, math.ceil(width * scale)),
                    max(_DINOCELL_CROP, math.ceil(height * scale)))
            work = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        else:
            work = image
        dx, dy, probability = self._predict(work)
        d_p = np.stack([dy, dx])
        labels = compute_masks(
            dP=d_p, cellprob=probability, niter=_DINOCELL_NITER,
            cellprob_threshold=_probability_threshold(cellprob_threshold),
            flow_threshold=_DINOCELL_FLOW_THRESHOLD, do_3D=False,
            min_size=_DINOCELL_MIN_SIZE,
            max_size_fraction=_DINOCELL_MAX_SIZE_FRACTION,
            device=self.device)
        shape = (height, width)
        display = np.moveaxis(
            _resize_nearest(np.moveaxis(dx_to_circ(d_p), -1, 0), shape), 0, -1)
        flow = [display, _resize_nearest(d_p, shape),
                _resize_nearest(probability, shape), None]
        return _resize_nearest(labels, shape), flow


def _samcell_weights_path(variant="generalist", download=False):
    """Where a SAMCell checkpoint lives in torch's hub cache.

    :param variant: ``'generalist'`` or ``'cyto'``.
    :param download: fetch the GitHub release asset when it is not cached.
    :returns: the local path (which may not exist when ``download`` is False).
    """
    import torch

    filename = _SAMCELL_WEIGHTS[variant]
    path = os.path.join(torch.hub.get_dir(), "checkpoints", filename)
    if download and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = _SAMCELL_RELEASE + filename
        print(f"Downloading SAMCell weights {url} -> {path}")
        torch.hub.download_url_to_file(url, path, progress=True)
    return path


class _SamCellBackend(_PlaneBackend):
    """SAMCell: SAM ViT-B fine-tuned to predict a cell distance map."""

    name = _SAMCELL
    note = ("SAMCell's own peak (0.47) and fill (0.09) thresholds apply; "
            "cellprob_threshold, flow_threshold, diameter, min_size and "
            "resample are Cellpose settings and are not used")

    def __init__(self, device=None, weights_path=None, variant="generalist"):
        """Load SAM ViT-B, apply a SAMCell checkpoint (downloaded once into
        torch's hub cache unless ``weights_path`` names one) and build the
        sliding-window pipeline."""
        super().__init__(device)
        FinetunedSAM, SlidingWindowPipeline = _import_samcell()
        import torch

        self.device = torch.device(self.device)
        weights = weights_path or _samcell_weights_path(variant, download=True)
        model = FinetunedSAM(_SAMCELL_BASE_MODEL)
        model.load_weights(weights, map_location=self.device)
        self._pipeline = SlidingWindowPipeline(
            model, self.device, crop_size=_SAMCELL_CROP)

    def _segment_plane(self, image, cellprob_threshold=None):
        """Predict SAMCell's distance map and label it with its own watershed.

        ``predict_on_full_img`` raises on failure; ``run`` would log and
        return an all-zero label image, which would be saved as a field with
        no cells.
        """
        dist_map = self._pipeline.predict_on_full_img(image)
        labels = self._pipeline.cells_from_dist_map(dist_map)
        return labels, [dist_map, None, dist_map, None]


class _Cellpose3Adapter:
    """Cellpose 3, inside its own environment, answering the same ``eval``.

    :param model: a Cellpose 3 model name, or a Cellpose-format checkpoint.
    :param device: a torch device name.
    :raises FileNotFoundError: for a model that is neither. Measured on
        cellpose 3.1.1.3, 2026-09-19: handed a path that does not exist,
        ``CellposeModel`` logs a warning and segments with cyto3.
    """

    name = _CELLPOSE3

    def __init__(self, model="cyto3", device="cpu"):
        """Load the model; a named one brings Cellpose's size model with it."""
        import torch
        from cellpose import models

        self.model = model
        self.ignored = set()
        self.translated = set()
        where = torch.device(device)
        gpu = where.type != "cpu"
        if model in _CELLPOSE3_MODELS:
            self._model = models.Cellpose(gpu=gpu, model_type=model,
                                          device=where)
            self._sized = True
        elif os.path.isfile(model):
            self._model = models.CellposeModel(gpu=gpu, pretrained_model=model,
                                               device=where)
            self._sized = False
        else:
            raise FileNotFoundError(
                f"no Cellpose 3 model called {model!r}: it is not one of "
                f"{', '.join(_CELLPOSE3_MODELS)}, and no file is there. "
                f"Cellpose 3 would have run cyto3 in its place without a "
                f"word.")

    def eval(self, x, channel_axis=-1, diameter=None, normalize=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
             resample=True, batch_size=None, **other):
        """Segment each image; a second channel is the nucleus channel.

        A diameter of 0 or None lets a named model's size model estimate it,
        and leaves a checkpoint at the diameter it was trained at.

        THE NORMALIZATION IS TRANSLATED, and this is the one place where
        Cellpose 3 and Cellpose-SAM genuinely disagree about a setting.
        Mask generation scales every image by its own maximum
        (``prepare_batch_for_segmentation``) and then calls ``eval`` with
        ``normalize=False``. That pair is spaCR's settled behaviour for
        Cellpose-SAM. Cellpose 3's cyto, cyto2, cyto3 and nuclei weights
        were fitted on Cellpose's own per-channel 1st-to-99th-percentile
        normalization, so handing them a max-scaled image with
        normalization off is a different input distribution from the one
        they were trained on: one bright speck crushes the rest of the
        field toward zero and the masks quietly get worse.

        Turning it back on costs nothing, because dividing by the maximum
        is a pure linear scale with no offset and percentiles scale with
        it -- Cellpose's normalization of ``x / max(x)`` is its
        normalization of ``x``. So spaCR's preparation is left alone and
        Cellpose 3 normalizes as it was trained to.

        :param batch_size: forwarded to Cellpose 3, which has its own
            default of 8, so the user's value has to be passed on.
        :param other: anything else the call site passes. What this
            Cellpose cannot take is recorded in :attr:`ignored` and
            reported by name, rather than disappearing.
        :returns: ``(masks, flows, None)``; each flows entry is Cellpose's
            ``[RGB flow, dP, cell probability, None]``.
        """
        if normalize is False:
            normalize = True
            self.translated.add(
                "normalize=False became normalize=True: spaCR's scaling is "
                "linear, and these weights were trained on Cellpose's "
                "percentile normalization")
        extra = {"batch_size": batch_size} if batch_size else {}
        extra.update({k: v for k, v in other.items() if v is not None})
        extra = self._accepted(extra)
        masks, flows = [], []
        for image in x:
            image = np.asarray(image)
            channels, axis = [0, 0], None
            if image.ndim == 3:
                axis = -1 if channel_axis is None else channel_axis
                if image.shape[axis] >= 2:
                    channels = [1, 2]
                else:
                    image, axis = np.take(image, 0, axis=axis), None
            elif image.ndim != 2:
                raise ValueError(
                    f"the Cellpose 3 backend segments 2-D images; got an "
                    f"array of shape {image.shape}")
            size = float(diameter) if diameter else (0.0 if self._sized
                                                     else None)
            output = self._model.eval(
                image, channels=channels, channel_axis=axis, diameter=size,
                normalize=normalize, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, min_size=min_size,
                resample=resample, **extra)
            parts = list(output[1])[:3]
            masks.append(_as_label_image(output[0]))
            flows.append(parts + [None] * (4 - len(parts)))
        return masks, flows, None

    def _accepted(self, extra):
        """The keywords this Cellpose's ``eval`` will take, of those asked
        for; the rest are remembered by name in :attr:`ignored`.

        The signature is read rather than listed, because the answer
        differs between ``Cellpose`` and ``CellposeModel`` and between
        Cellpose 3 releases, and a list written here would go stale
        silently -- which is the failure this method exists to stop.
        """
        try:
            signature = inspect.signature(self._model.eval)
        except (TypeError, ValueError):
            return dict(extra)
        parameters = signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in parameters.values()):
            return dict(extra)
        taken = {k: v for k, v in extra.items() if k in parameters}
        self.ignored.update(set(extra) - set(taken))
        return taken


#: Backend name -> in-process class. Tests replace entries with stubs.
_BACKEND_CLASSES = {_DINOCELL: _DinoCellBackend, _SAMCELL: _SamCellBackend}


def _worker_device(requested=None):
    """The device a worker runs on: the one asked for, else CUDA, else
    Apple's Metal, else the CPU."""
    import torch

    wanted = str(requested or "auto").strip().lower()
    if wanted not in ("", "auto"):
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    metal = getattr(getattr(torch, "backends", None), "mps", None)
    if metal is not None and metal.is_available():
        return "mps"
    return "cpu"


def _worker_hello(name):
    """What a worker says about itself: versions, device, models.

    Importing the package is the point -- an environment whose package does
    not load fails here, during the install's self-test, rather than on the
    first field.
    """
    from importlib import import_module
    from importlib.metadata import PackageNotFoundError, version

    spec = _spec(name)
    for module in spec.probe:
        import_module(module)
    packages = {}
    for distribution in (spec.distribution, "torch", "numpy"):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = ""
    return {"backend": spec.name,
            "python": "%d.%d.%d" % tuple(sys.version_info[:3]),
            "packages": packages, "device": _worker_device(),
            "models": list(spec.models)}


def _worker_adapter(name, model, device, options):
    """The object a worker segments with."""
    if name == _CELLPOSE3:
        return _Cellpose3Adapter(model or "cyto3", device)
    return _BACKEND_CLASSES[name](device=device, **options)


def _worker_segment(name, request, adapters):
    """Segment the request's images and write the masks beside them.

    Adapters are kept by ``(model, device, options)``, so the model loads
    once per worker and not once per request.
    """
    device = _worker_device(request.get("device"))
    model = str(request.get("model") or "")
    options = dict(request.get("options") or {})
    key = (model, device, json.dumps(options, sort_keys=True))
    adapter = adapters.get(key)
    if adapter is None:
        adapter = _worker_adapter(name, model, device, options)
        adapters[key] = adapter
    images = [np.load(path, allow_pickle=False)
              for path in request.get("inputs") or ()]
    started = time.monotonic()
    masks, flows, _styles = adapter.eval(images,
                                         **dict(request.get("params") or {}))
    folder = request["outputs"]
    outputs = []
    for index, mask in enumerate(masks):
        mask_path = os.path.join(folder, f"mask_{index}.npy")
        np.save(mask_path, np.asarray(mask), allow_pickle=False)
        entry = list(flows[index] if flows and index < len(flows) else ())
        saved = []
        for part in range(4):
            value = entry[part] if part < len(entry) else None
            if isinstance(value, np.ndarray):
                path = os.path.join(folder, f"flow_{index}_{part}.npy")
                np.save(path, value, allow_pickle=False)
                saved.append(path)
            else:
                saved.append(None)
        outputs.append({"mask": mask_path, "flows": saved})
    reply = {"outputs": outputs, "device": device,
             "seconds": round(time.monotonic() - started, 3)}
    ignored = sorted(getattr(adapter, "ignored", ()) or ())
    translated = sorted(getattr(adapter, "translated", ()) or ())
    if ignored:
        reply["ignored"] = ignored
    if translated:
        reply["translated"] = translated
    return reply


def _worker_restoration_model(name, request, adapters):
    """Load or reuse a model and return the identity of its loaded weights."""
    if name != _CELLPOSE3:
        raise ValueError("image restoration requires the Cellpose 3 backend")
    model_name = request.get("model")
    if model_name not in _RESTORATION_MODELS:
        raise ValueError(f"unsupported same-grid restoration model: {model_name!r}")
    device = _worker_device(request.get("device") or "cpu")
    key = ("restore", model_name, device)
    cached = adapters.get(key)
    if cached is None:
        import hashlib
        from importlib.metadata import version

        import torch
        from cellpose import denoise

        where = torch.device(device)
        model = denoise.DenoiseModel(
            model_type=model_name, device=where, gpu=where.type != "cpu")
        digest = hashlib.sha256()
        with open(model.pretrained_model, "rb") as weights:
            for block in iter(lambda: weights.read(1024 * 1024), b""):
                digest.update(block)
        identity = {"backend": name, "model": model_name,
                    "cellpose_version": version("cellpose"),
                    "weights_sha256": digest.hexdigest(), "device": str(model.device)}
        cached = (model, identity)
        adapters[key] = cached
    return cached


def _worker_restore(name, request, adapters):
    """Restore a finite intensity plane on its original coordinate grid.

    Output remains float32 in normalized model units, including negative
    values; callers must not interpret it as calibrated fluorescence. The
    existing worker protocol supplies cancellation by terminating the isolated
    process. A restarted worker refuses weights differing from a captured plan.
    """
    diameter = float(request.get("diameter", 30.0))
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("restoration diameter must be finite and positive")
    image = np.load(request["input"], allow_pickle=False)
    if (image.ndim != 2 or min(image.shape) < 2
            or image.dtype.kind not in "uif"
            or not np.isfinite(image).all()):
        raise ValueError("restoration needs one finite real intensity plane")
    image = np.array(image, dtype=np.float32, copy=True)
    if not np.isfinite(image).all():
        raise ValueError("restoration intensity exceeds the float32 range")
    model, identity = _worker_restoration_model(name, request, adapters)
    expected = request.get("expected_identity")
    if expected is not None and expected != identity:
        raise ValueError("restoration model changed; select the model again")
    restored = np.asarray(model.eval(
        image, channels=None, channel_axis=None, diameter=diameter,
        normalize=True, batch_size=1), dtype=np.float32)
    if restored.shape == (*image.shape, 1):
        restored = restored[..., 0]
    if restored.shape != image.shape or not np.isfinite(restored).all():
        raise ValueError("restoration returned invalid values or changed image dimensions")
    np.save(request["output"], restored, allow_pickle=False)
    return {"output": request["output"], "provenance": {
        **identity, "diameter_px": diameter,
        "normalization": "Cellpose 1st/99th percentile",
        "intensity_units": "normalized model output", "dtype": "float32",
        "shape": list(restored.shape)}}


def _worker_detect(request, adapters):
    """Find plaque images in one figure with the YOLO detector, per size.

    The figure arrives as an ``H x W x 3`` RGB ``.npy`` and is handed to
    ultralytics as BGR, the order it reads an array in (see
    ``spacr.plaque._to_detector_channel_order``; this file cannot import
    spaCR, so the one line is repeated).

    :param request: ``image`` (a .npy path), ``weights``, ``imgsz`` (a list
        of sizes) and ``confidence``.
    :param adapters: the worker's cache; the detector is loaded once per
        checkpoint.
    :returns: ``{"boxes": [[x0, y0, x1, y1, confidence, size], ...]}``.
    """
    import numpy as np

    key = ("yolo", str(request["weights"]))
    if key not in adapters:
        from ultralytics import YOLO

        adapters[key] = YOLO(str(request["weights"]))
    model = adapters[key]
    image = np.load(request["image"], allow_pickle=False)
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.ascontiguousarray(image[:, :, ::-1])
    boxes = []
    for size in request.get("imgsz") or [640]:
        for result in model.predict(source=image,
                                    conf=float(request.get("confidence", 0.25)),
                                    imgsz=int(size), verbose=False):
            found = getattr(result, "boxes", None)
            if found is None:
                continue
            for box in found:
                xyxy, conf = box.xyxy, box.conf
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                if conf is not None and hasattr(conf, "cpu"):
                    conf = conf.cpu().numpy()
                x0, y0, x1, y1 = (float(v) for v in
                                  np.asarray(xyxy).ravel()[:4])
                score = (float(np.asarray(conf).ravel()[0])
                         if conf is not None else 1.0)
                boxes.append([x0, y0, x1, y1, score, int(size)])
    return {"boxes": boxes}


def _worker_detect_spots(request, adapters):
    """Find fluorescent spots in one image with SpotNet.

    :param request: ``image`` (a ``.npy`` path, ``H x W`` or ``H x W x 1``
        with finite values) and ``threshold`` (a finite detection probability
        from 0 to 1). Invalid inputs are rejected before loading weights.
    :param adapters: the worker's cache; the application loads once.
    :returns: ``{"spots": [[y, x], ...]}`` in image pixels.
    """
    import numpy as np

    image = np.load(str(request["image"]), allow_pickle=False)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[-1] != 1 or not all(image.shape):
        raise ValueError("SpotNet needs one nonempty single-channel image.")
    batch = image[None].astype("float32")
    if not np.isfinite(batch).all():
        raise ValueError("SpotNet image values must be finite.")
    threshold = float(request.get("threshold", 0.95))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("SpotNet threshold must be between 0 and 1.")
    if "spotnet" not in adapters:
        from deepcell_spots.applications import SpotDetection

        adapters["spotnet"] = SpotDetection()
    found = adapters["spotnet"].predict(batch, threshold=threshold)
    if (not isinstance(found, (list, tuple, np.ndarray))
            or (isinstance(found, np.ndarray) and found.ndim == 0)
            or len(found) != 1):
        raise ValueError("SpotNet must return coordinates for exactly one image.")
    spots = np.asarray(found[0], dtype=float)
    if spots.shape in ((0,), (0, 2)):
        return {"spots": []}
    if spots.ndim != 2 or spots.shape[1] != 2 or not np.isfinite(spots).all():
        raise ValueError("SpotNet coordinates must be finite (y, x) pairs.")
    return {"spots": spots.tolist()}


def _worker_read_text(request, adapters):
    """Read the words in one image with RapidOCR.

    :param request: ``image``, a .npy path.
    :param adapters: the worker's cache; the reader is built once.
    :returns: ``{"words": [[[[x, y], ...4], text, confidence], ...]}``.
    """
    import numpy as np

    if "rapidocr" not in adapters:
        from rapidocr_onnxruntime import RapidOCR

        adapters["rapidocr"] = RapidOCR()
    image = np.load(request["image"], allow_pickle=False)
    results, _elapsed = adapters["rapidocr"](image)
    words = [[[[float(x), float(y)] for x, y in box], str(text), float(conf)]
             for box, text, conf in (results or [])]
    return {"words": words}


def _worker_read_pdf(request, adapters):
    """Render a PDF's pages and read their text layer with pdfplumber.

    :param request: ``pdf`` (a path), ``dest`` (a folder for the page
        images), ``dpi`` and ``x_tolerance`` (pdfplumber's, in points).
    :param adapters: the worker's cache; unused.
    :returns: ``{"pages": [{"path", "text", "words": [[text, x0, y0, x1,
        y1], ...]}, ...]}`` with the words in the rendered image's pixels.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "This figure reader was installed before it read PDFs. "
            "Reinstall the plaque figure reader from the Model Zoo to add "
            "pdfplumber.") from exc
    dest = request["dest"]
    os.makedirs(dest, exist_ok=True)
    dpi = int(request.get("dpi", 200))
    tolerance = float(request.get("x_tolerance", 1.5))
    scale = dpi / 72.0
    pages = []
    with pdfplumber.open(request["pdf"]) as document:
        for number, page in enumerate(document.pages, start=1):
            target = os.path.join(dest, f"page_{number:03d}.png")
            page.to_image(resolution=dpi).save(target)
            pages.append({
                "path": target,
                "text": page.extract_text(x_tolerance=tolerance) or "",
                "words": [[w["text"], w["x0"] * scale, w["top"] * scale,
                           w["x1"] * scale, w["bottom"] * scale]
                          for w in page.extract_words(x_tolerance=tolerance)]})
    return {"pages": pages}


def _handle(name, request, adapters):
    """Answer one request; every failure becomes an error reply, never a
    dead worker."""
    ident = request.get("id") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict):
            raise ValueError("a request is one JSON object per line")
        if request.get("protocol") != _PROTOCOL:
            raise ValueError(
                f"spaCR sent protocol {request.get('protocol')!r}; this "
                f"worker speaks {_PROTOCOL}. Reinstall the backend from the "
                f"Model Zoo.")
        op = request.get("op")
        if op == "hello":
            body = _worker_hello(name)
        elif op == "segment":
            body = _worker_segment(name, request, adapters)
        elif op == "restore":
            body = _worker_restore(name, request, adapters)
        elif op == "restoration_model":
            _, identity = _worker_restoration_model(name, request, adapters)
            body = {"identity": dict(identity)}
        elif op == "detect":
            body = _worker_detect(request, adapters)
        elif op == "detect_spots":
            body = _worker_detect_spots(request, adapters)
        elif op == "read_text":
            body = _worker_read_text(request, adapters)
        elif op == "read_pdf":
            body = _worker_read_pdf(request, adapters)
        elif op == "shutdown":
            body = {}
        else:
            raise ValueError(f"unknown request {op!r}")
    except Exception as exc:                                 # noqa: BLE001
        import traceback

        return {"protocol": _PROTOCOL, "id": ident, "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc),
                          "traceback": traceback.format_exc()}}
    body.update(protocol=_PROTOCOL, id=ident, ok=True)
    return body


def _serve(name, stdin, stdout):
    """Answer requests from ``stdin`` on ``stdout`` until shutdown or EOF.

    ``stdin`` is read on a thread of its own, so a ``cancel`` naming a
    request that is still queued is heard while another one runs: the
    cancelled one is answered with a ``Cancelled`` error and never started.
    spaCR has already stopped waiting for it (item 507).
    """
    adapters = {}
    pending = queue.Queue()
    cancelled = set()
    guard = threading.Lock()
    finished = object()

    def read():
        """Queue each request; note each cancel at once."""
        try:
            for line in stdin:
                text = line.strip()
                if not text:
                    continue
                try:
                    request = json.loads(text)
                except ValueError:
                    request = text
                if isinstance(request, dict) and request.get("op") == "cancel":
                    with guard:
                        cancelled.add(request.get("target"))
                    continue
                pending.put(request)
        except (OSError, ValueError):
            pass
        pending.put(finished)

    threading.Thread(target=read, daemon=True).start()
    while True:
        request = pending.get()
        if request is finished:
            break
        ident = request.get("id") if isinstance(request, dict) else None
        with guard:
            skip = ident is not None and ident in cancelled
            cancelled.discard(ident)
        if skip:
            reply = {"protocol": _PROTOCOL, "id": ident, "ok": False,
                     "error": {"type": "Cancelled", "traceback": "",
                               "message": "cancelled before it started"}}
        else:
            reply = _handle(name, request, adapters)
        stdout.write(json.dumps(reply) + "\n")
        stdout.flush()
        if isinstance(request, dict) and request.get("op") == "shutdown":
            break
    return 0


def _protocol_channel():
    """A private copy of stdout for replies; stdout itself then goes to
    stderr, so a library that prints cannot corrupt a reply."""
    sys.stdout.flush()
    channel = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(channel, "w", encoding="utf-8", buffering=1)


def _worker_main(argv=None, stdin=None, stdout=None):
    """The worker's command line: ``--serve <name>`` or ``--selftest <name>``.

    :returns: the exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if (len(args) != 2 or args[0] not in ("--serve", "--selftest")
            or args[1] not in _SPECS):
        sys.stderr.write("usage: python -I _segmentation_backends.py "
                         "--serve|--selftest <backend>\n")
        return 2
    mode, name = args
    if mode == "--selftest":
        reply = _handle(name, {"protocol": _PROTOCOL, "id": 0, "op": "hello"},
                        {})
        out = sys.stdout if stdout is None else stdout
        out.write(json.dumps(reply) + "\n")
        out.flush()
        return 0 if reply["ok"] else 1
    return _serve(name, sys.stdin if stdin is None else stdin,
                  _protocol_channel() if stdout is None else stdout)


def _load_backend(name, *, device=None, z_plan=None, t_plan=None,
                  model_name=None, object_type=None, root=None, **options):
    """Build the model object ``generate_cellpose_masks_sam`` calls ``eval`` on.

    A backend installed in its own environment is used there, through
    :class:`_RemoteBackend`. Otherwise DINOCell and SAMCell are built in
    process, which works only when an older spaCR installed them into
    spaCR's own environment and raises an ImportError naming the Model Zoo
    when it did not; Cellpose 3 never runs in process.

    :param name: a non-Cellpose value of ``segmentation_backend``.
    :param device: torch device; the resolved accelerator when None.
    :param z_plan: the run's z-stack plan; must be None.
    :param t_plan: the run's t-stack plan; must be None.
    :param model_name: the object's model setting; see
        :func:`_cellpose3_model`. Used by Cellpose 3 only.
    :param object_type: the object being segmented.
    :param root: the backends folder.
    :param options: passed to the backend (``weights_path``, ``variant``).
    :returns: an object with Cellpose's ``eval``.
    :raises ValueError: for Cellpose, an unknown name, or a 3-D/4-D run.
    :raises ImportError: when the backend is not installed.
    """
    backend = _backend_name(name)
    if backend == _CELLPOSE:
        raise ValueError(
            "segmentation_backend='cellpose' is built by the mask generator "
            "itself, not by _load_backend")
    if z_plan is not None or t_plan is not None:
        raise ValueError(
            f"segmentation_backend={backend!r} segments single 2-D planes, "
            f"and this run has z_stack or t_stack on. Use "
            f"segmentation_backend='cellpose' for 3-D and 4-D runs, or turn "
            f"z_stack and t_stack off.")
    state = _backend_state(backend, root)
    if state.ready and not state.in_process:
        model = _RemoteBackend(
            backend, device=device, root=root, options=options,
            model=(_cellpose3_model(model_name, object_type)
                   if backend == _CELLPOSE3 else ""))
    elif backend == _CELLPOSE3:
        raise ImportError(_not_installed_message(backend, state))
    else:
        model = _BACKEND_CLASSES[backend](device=device, **options)
    note = getattr(model, "note", "")
    print(f"Segmentation backend: {backend}"
          + (f" -- {note}." if note else "."))
    return model


if __name__ == "__main__":
    sys.exit(_worker_main())
