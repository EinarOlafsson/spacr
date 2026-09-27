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

PROMPTS. micro-SAM is installed and run the same way but never segments a
whole run: Make Masks sends it points and a box on one object
(:class:`_PromptClient`), and it answers with that object's mask. The
worker embeds each field once (``sam_embed``) and answers every later
prompt on it from that embedding (``sam_prompt``).

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

#: CAREamics, which trains a Noise2Void denoiser on a run's own noisy fields
#: and applies it before segmentation; it denoises, it does not segment.
_CAREAMICS = "careamics"

#: Noise2Void's training patch side in pixels and patches per batch, the
#: sizes CAREamics' own 2-D examples use.
_N2V_PATCH = 64
_N2V_BATCH = 16

#: CellProfiler, which runs a lab's own ``.cppipe`` pipeline headlessly on
#: spaCR's fields; it measures, it does not segment for spaCR.
_CELLPROFILER = "cellprofiler"

#: Where a backend that needs Java keeps the development kit it fetched.
_JDK_FOLDER = "jdk"

#: Every value ``segmentation_backend`` accepts, the default first.
_BACKEND_NAMES = (_CELLPOSE, _CELLPOSE3, _DINOCELL, _SAMCELL)

#: The models the Cellpose 3 backend names, as Cellpose 3 names them.
_CELLPOSE3_MODELS = ("cyto3", "cyto2", "cyto", "nuclei")

#: What an object's model setting starts with when it names a Cellpose 3
#: model: ``cellpose3:cyto3``, or ``cellpose3:/path/to/weights`` for any
#: Cellpose 3 checkpoint. The same spelling as Make Masks' Mode box keys.
#: It lets one object run Cellpose 3 while the others stay on Cellpose-SAM,
#: and it is what the model zoo writes when a Cellpose 3 row is chosen.
_CELLPOSE3_PREFIX = "cellpose3:"

#: Cellpose 4 with Meta's DINOv3 backbone, which runs Cellpose-DINO
#: checkpoints (bioimage.io's CellposeDINO ViT-L and ViT-B, item 525). It is
#: not a ``segmentation_backend`` value: an object chooses it through its
#: model setting, ``cellpose_dino:<checkpoint path>``, the way
#: ``cellpose3:`` chooses Cellpose 3, and every other object stays on
#: spaCR's own Cellpose-SAM.
_CELLPOSE_DINO = "cellpose_dino"

#: What an object's model setting starts with when it names a Cellpose-DINO
#: checkpoint; what the model zoo writes when a Cellpose-DINO row is chosen.
_CELLPOSE_DINO_PREFIX = "cellpose_dino:"

#: The DINOv3 commit the Cellpose-DINO backend installs. DINOv3 is not on
#: PyPI (https://pypi.org/pypi/dinov3/json answered 404 on 2026-09-25);
#: Cellpose itself says to install it from GitHub. The archive of one pinned
#: commit is what pip installs, so the install needs no git and builds the
#: same package every time. It was facebookresearch/dinov3's main on
#: 2026-09-25.
_DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"

#: The pip requirement for that commit.
_DINOV3_REQUIREMENT = (
    "dinov3 @ https://github.com/facebookresearch/dinov3/archive/"
    f"{_DINOV3_COMMIT}.zip")

#: StarDist (item 551): star-convex polygons, for nuclei. It runs on
#: TensorFlow, so its environment has no PyTorch at all. Like
#: Cellpose-DINO it is not a ``segmentation_backend`` value: an object
#: chooses it through its model setting, ``stardist:<model or folder>``.
_STARDIST = "stardist"

#: StarDist's own pretrained 2-D models, the fluorescence one first.
_STARDIST_MODELS = ("2D_versatile_fluo", "2D_versatile_he",
                    "2D_paper_dsb2018")

#: The object diameter, in pixels, StarDist's models are run at when the
#: object's diameter is set: the plane is rescaled by ``30 / diameter``
#: (StarDist's own ``scale``). StarDist publishes no training size, so this
#: was measured (2026-09-26, 2D_versatile_fluo, toxo_mito plate1_E01_1_1,
#: 40x nuclei of median 86 px, 49 in spaCR's Cellpose-SAM reference): at
#: native scale StarDist cut them into 111 objects of median 14 px; scaled
#: to about 50, 30 and 22 px it found 58, 44 and 39 objects of median 73,
#: 85 and 91 px. A blank diameter runs the plane at its own scale.
_STARDIST_DIAMETER = 30.0

#: InstanSeg (item 552): embedding-based instance segmentation of nuclei
#: and cells, channel-agnostic. Chosen through an object's model setting,
#: ``instanseg:<model or file>``, like StarDist.
_INSTANSEG = "instanseg"

#: InstanSeg's own published models (its model-index.json, 0.1.1).
_INSTANSEG_MODELS = ("fluorescence_nuclei_and_cells", "brightfield_nuclei")

#: The object diameter, in pixels at the model's own pixel size, InstanSeg
#: is run at when the object's diameter is set (see
#: :class:`_InstanSegAdapter`). InstanSeg publishes no object size, so this
#: was measured (2026-09-26, fluorescence_nuclei_and_cells, nuclei output,
#: toxo_mito plate1_E01_1_1, 40x nuclei of median 86 px, 49 in spaCR's
#: Cellpose-SAM reference): at the model's own scale it found 35 objects of
#: median 20 px; given pixel sizes that bring the nuclei to about 43, 26
#: and 17 px it found 34, 39 and 40 objects of median 51, 81 and 77 px.
_INSTANSEG_DIAMETER = 26.0

#: Omnipose (item 553): Cellpose-style flows on a distance field, for
#: bacteria and other elongated or filamentous cells. Chosen through an
#: object's model setting, ``omnipose:<model or file>``.
_OMNIPOSE = "omnipose"

#: Omnipose's own 2-D models (``omnipose.core``'s boundary-field lists),
#: the phase-contrast bacteria model first.
_OMNIPOSE_MODELS = ("bact_phase_omni", "bact_fluor_omni", "worm_omni",
                    "worm_bact_omni", "worm_high_res_omni", "cyto2_omni")

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

#: micro-SAM, Segment Anything fine-tuned for microscopy. It is not a
#: ``segmentation_backend`` value: it answers prompts -- points and a box on
#: one object -- in Make Masks, and returns that one object's mask.
_MICROSAM = "microsam"

#: The micro-SAM model prompts are answered with: its light-microscopy
#: generalist on a ViT-B encoder, which micro-SAM itself defaults to.
_MICROSAM_MODEL = "vit_b_lm"

#: A field whose longer side is above this is embedded in tiles of
#: :data:`_MICROSAM_TILE` with :data:`_MICROSAM_HALO` of overlap, the way
#: micro-SAM's own annotator does for large images. SAM looks at 1024
#: pixels on the longer side, so a larger field embedded whole is shrunk
#: first and small objects lose their edges.
_MICROSAM_TILE_ABOVE = 1536
_MICROSAM_TILE = 1024
_MICROSAM_HALO = 256

#: How many fields' embeddings one micro-SAM worker keeps. Going back to the
#: field before costs nothing; a ViT-B embedding of one tile is about 4 MB.
_MICROSAM_KEEP = 4

#: At most this many CPU threads answer one prompt. The mask decoder is
#: small, and on a CPU other work is also using, every thread torch starts
#: waits on the others: on a 16-thread CPU under load, a prompt took 1.9 to
#: 3.0 s on all 16 threads and 0.25 to 0.85 s on four (2026-09-26). The
#: embedding, which is large, keeps every thread.
_MICROSAM_PROMPT_THREADS = 4


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
    :param without_dependencies: requirements pip installs with
        ``--no-deps``, after everything else. For a package whose declared
        dependencies reach far past the part spaCR runs -- micro-SAM's
        napari viewer, its tracker and that tracker's commercial solver --
        the part spaCR runs is listed in ``requirements`` instead, and the
        self-test proves the list is enough.
    :param prefix: what an object's model setting starts with to choose
        this backend, e.g. ``'stardist:'``, for a backend that is chosen
        that way and is no ``segmentation_backend`` value. Its models are
        :attr:`models` by name, or a model file or folder by path; see
        :func:`_prefixed_model`.
    :param default_model: the model a bare prefix runs.
    :param alpha: built from the future-features list and shown only with
        Preferences' "Show alpha features" on (item 569).
    :param built_here: requirements pip builds from source against the
        environment's own numpy, with ``--no-build-isolation``, after
        :attr:`requirements` and before :attr:`without_dependencies`.
    :param java: the Java major version a development kit is fetched for,
        into the environment's own ``jdk`` folder, before
        :attr:`built_here` is built; ``''`` for a backend without Java.
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
    without_dependencies: tuple = ()
    prefix: str = ""
    default_model: str = ""
    alpha: bool = False
    built_here: tuple = ()
    java: str = ""


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
    _CELLPOSE_DINO: _BackendSpec(
        name=_CELLPOSE_DINO, label="Cellpose-DINO", module="cellpose",
        probe=("cellpose.models", "cellpose.vit", "dinov3.hub.backbones"),
        distribution="cellpose",
        requirements=("cellpose==4.2.1.1", _DINOV3_REQUIREMENT),
        torch=("torch", "torchvision"), python=((3, 11), (3, 14)),
        licence="BSD-3-Clause (Cellpose) / DINOv3 License (dinov3)",
        licence_note=(
            "Cellpose 4.2.1.1 is BSD-3-Clause (Copyright 2025 Howard Hughes "
            "Medical Institute). DINOv3 is NOT open source: its code and "
            "weights are Meta's DINO Materials under the DINOv3 License "
            "(github.com/facebookresearch/dinov3, LICENSE.md, 19 August "
            "2025), which binds whoever uses them, so installing this "
            "backend and running a Cellpose-DINO model is agreeing to it. "
            "It allows use, copying and modification; it requires that "
            "anything passed on carries the licence, that a publication "
            "using the results acknowledges DINOv3, and it forbids "
            "military, weapons, nuclear and espionage uses and anyone under "
            "trade sanctions. Cellpose-DINO checkpoints are DINOv3 fine-"
            "tuned by the Cellpose authors, so the same terms reach them "
            "whatever licence their page names. spaCR ships none of it: "
            "dinov3 is fetched from Meta's GitHub at commit "
            f"{_DINOV3_COMMIT[:12]}, and each checkpoint from the page it "
            "is listed on."),
        homepage="https://github.com/facebookresearch/dinov3", size_gb=3.0,
        blurb=(
            "Cellpose 4 with Meta's DINOv3 backbone, which runs the "
            "Cellpose-DINO models (ViT-L and ViT-B) listed from "
            "bioimage.io. It runs in an environment of its own, so "
            "spaCR's own Cellpose is untouched; its masks, flows and cell "
            "probability come back in Cellpose-SAM's shapes."),
        published=(
            "Published results: none for the DINO backbone yet; Cellpose-SAM "
            "is Pachitariu, Rariden and Stringer, bioRxiv 2025 (doi:10.1101/"
            "2025.04.28.651001), and DINOv3 is Simeoni et al., "
            "arXiv:2508.10104. spaCR has not scored this backend on a "
            "benchmark of its own data; one Toxoplasma PV field is measured "
            "in item 525.")),
    _STARDIST: _BackendSpec(
        name=_STARDIST, label="StarDist", module="stardist",
        probe=("stardist.models", "csbdeep.utils", "tensorflow"),
        distribution="stardist",
        requirements=("stardist==0.9.2", "csbdeep==0.8.2",
                      "tensorflow==2.21.0"),
        torch=(), python=((3, 10), (3, 13)),
        licence="BSD-3-Clause (StarDist, CSBDeep) / Apache-2.0 (TensorFlow)",
        licence_note=(
            "StarDist 0.9.2 is BSD-3-Clause (Copyright 2018-2025 Uwe "
            "Schmidt, Martin Weigert), CSBDeep 0.8.2 BSD-3-Clause and "
            "TensorFlow 2.21 Apache-2.0. The pretrained models are "
            "downloaded by StarDist itself from the stardist/stardist-models "
            "GitHub release (BSD-3-Clause), checked against the digest "
            "StarDist pins, into the backend's own folder. spaCR ships "
            "none of it."),
        homepage="https://github.com/stardist/stardist", size_gb=2.0,
        models=_STARDIST_MODELS,
        blurb=(
            "StarDist, star-convex polygons for nuclei, with its "
            "2D_versatile_fluo, 2D_versatile_he and 2D_paper_dsb2018 "
            "models or a StarDist model folder of your own. It runs on "
            "TensorFlow in an environment of its own; its masks and "
            "object probability come back in Cellpose-SAM's shapes."),
        published=(
            "Published results: Schmidt, Weigert, Broaddus and Myers, 'Cell "
            "Detection with Star-convex Polygons', MICCAI 2018 "
            "(arXiv:1806.03535). The README publishes no results table. "
            "spaCR has not scored it against those results; item 532 "
            "scores it on spaCR's own fields."),
        prefix="stardist:", default_model="2D_versatile_fluo", alpha=True),
    _INSTANSEG: _BackendSpec(
        name=_INSTANSEG, label="InstanSeg", module="instanseg",
        probe=("instanseg", "instanseg.utils.utils"),
        distribution="instanseg-torch",
        requirements=("instanseg-torch==0.1.1",),
        torch=("torch",), python=((3, 9), (3, 13)),
        licence="Apache-2.0",
        licence_note=(
            "InstanSeg (instanseg-torch 0.1.1) is Apache-2.0, and so are its "
            "fluorescence_nuclei_and_cells and brightfield_nuclei models, "
            "which InstanSeg downloads itself from the instanseg/instanseg "
            "GitHub release (instanseg_models_v0.1.1) into the backend's "
            "own folder. spaCR ships none of it."),
        homepage="https://github.com/instanseg/instanseg", size_gb=3.0,
        models=_INSTANSEG_MODELS,
        blurb=(
            "InstanSeg, embedding-based segmentation of nuclei and cells "
            "that reads any number of fluorescence channels, with its "
            "fluorescence_nuclei_and_cells and brightfield_nuclei models or "
            "an InstanSeg TorchScript file. A nucleus object keeps its "
            "nuclei, every other object its cells. It runs in an "
            "environment of its own."),
        published=(
            "Published results: Goldsborough et al., 'InstanSeg: an "
            "embedding-based instance segmentation algorithm optimized for "
            "accurate, efficient and portable cell segmentation', arXiv 2024 "
            "(arXiv:2408.15954). spaCR has not scored it against those "
            "results; item 532 scores it on spaCR's own fields."),
        prefix="instanseg:", default_model="fluorescence_nuclei_and_cells",
        alpha=True),
    _OMNIPOSE: _BackendSpec(
        name=_OMNIPOSE, label="Omnipose", module="omnipose",
        probe=("omnipose.core", "cellpose_omni.models"),
        distribution="omnipose",
        requirements=("omnipose==1.1.4", "ncolor==1.5.3"),
        torch=("torch", "torchvision"), python=((3, 11), (3, 13)),
        licence="Omnipose NonCommercial License (University of Washington)",
        licence_note=(
            "Omnipose is NOT open source: omnipose 1.1.4 carries the "
            "Omnipose NonCommercial License (Copyright 2021 University of "
            "Washington), which permits use, modification and "
            "redistribution for noncommercial purposes only; commercial use "
            "needs a licence from UW CoMotion (license@uw.edu). That is not "
            "the licence spaCR carries. Its models are downloaded by "
            "Omnipose itself from the kevinjohncutler/omnipose-models GitHub "
            "repository, which names no licence of its own, into the "
            "backend's own folder. spaCR ships none of it. Read the licence "
            "before using Omnipose for anything commercial."),
        homepage="https://github.com/kevinjohncutler/omnipose", size_gb=4.0,
        models=_OMNIPOSE_MODELS,
        blurb=(
            "Omnipose, for bacteria and other elongated or filamentous "
            "cells, with its bact_phase_omni, bact_fluor_omni, worm and "
            "cyto2_omni models or an Omnipose checkpoint. It runs in an "
            "environment of its own; its masks, flows and distance field "
            "come back in Cellpose-SAM's shapes. Noncommercial use only."),
        published=(
            "Published results: Cutler et al., 'Omnipose: a high-precision "
            "morphology-independent solution for bacterial cell "
            "segmentation', Nature Methods 2022 (doi:10.1038/s41592-022-"
            "01639-4). spaCR has not scored it against those results; "
            "item 553 scores it on Omnipose's own bacteria test images."),
        prefix="omnipose:", default_model="bact_phase_omni", alpha=True),
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
    _MICROSAM: _BackendSpec(
        name=_MICROSAM, label="micro-SAM", module="micro_sam",
        probe=("micro_sam.util", "micro_sam.prompt_based_segmentation"),
        distribution="micro_sam",
        requirements=("segment-anything-py==1.0.1", "python-elf==0.9.2",
                      "bioimage-cpp==0.9.0", "xxhash", "zarr", "pooch",
                      "imageio", "scikit-image", "tqdm"),
        without_dependencies=("micro-sam==1.8.14",),
        torch=("torch", "torchvision"), python=((3, 11), (3, 13)),
        licence="MIT (micro-SAM) / Apache-2.0 (segment-anything) / "
                "CC-BY-4.0 (vit_b_lm weights)",
        licence_note=(
            "micro-SAM 1.8.14 is MIT (computational-cell-analytics). It "
            "builds on Meta's Segment Anything, Apache-2.0. Its "
            "light-microscopy model, vit_b_lm ('SAM LM Generalist (ViT-B)', "
            "375 MB), is CC-BY-4.0 on bioimage.io and is downloaded into "
            "the backend's own folder the first time a field is prompted. "
            "micro-SAM itself is installed without its declared "
            "dependencies -- napari, PyQt6, bioimageio.core and trackastra, "
            "whose tracking solver pulls in the proprietary gurobipy -- "
            "because the prompt path spaCR runs imports none of them."),
        homepage="https://github.com/computational-cell-analytics/micro-sam",
        size_gb=3.0, segments=False, models=(_MICROSAM_MODEL,), alpha=True,
        blurb=(
            "micro-SAM, Segment Anything fine-tuned for microscopy, for "
            "prompt-based segmentation in Make Masks: click points on one "
            "object, or drag a box round it, and it returns that object's "
            "mask. Each field is embedded once and every later click on "
            "it is answered from the cached embedding."),
        published=(
            "Published results: Archit et al., 'Segment Anything for "
            "Microscopy', Nature Methods 2025 (doi:10.1038/s41592-024-"
            "02580-4). spaCR has not scored this backend on its own "
            "data.")),
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
    _CAREAMICS: _BackendSpec(
        name=_CAREAMICS, label="CAREamics Noise2Void", module="careamics",
        probe=("careamics", "careamics.config", "lightning",
               "lightning.pytorch.callbacks", "torch"),
        distribution="careamics",
        requirements=("careamics==0.3.4", "lightning==2.6.6"),
        torch=("torch>=2.6,<2.12", "torchvision<=0.26.0"),
        python=((3, 11), (3, 13)),
        licence="BSD-3-Clause",
        licence_note=(
            "CAREamics 0.3.4 is BSD-3-Clause (Copyright 2023, CAREamics "
            "contributors). It downloads no weights: every Noise2Void model "
            "is trained here, on the run's own images, and stays with the "
            "experiment."),
        homepage="https://careamics.github.io", size_gb=2.5, segments=False,
        alpha=True,
        blurb=(
            "Self-supervised denoising: CAREamics trains a Noise2Void (N2V2) "
            "network on a run's own noisy fields, with no clean images, and "
            "Make Masks denoises every segmentation channel with it before "
            "the enhancement chain. Training wants a GPU; a CPU trains a "
            "small model slowly. PyTorch is held below 2.12 and torchvision "
            "at 0.26 or older, the newest pair CAREamics 0.3.4 accepts."),
        published=(
            "Published results: Krull, Buchholz and Jug, 'Noise2Void - "
            "learning denoising from single noisy images', CVPR 2019; Hock "
            "et al., 'N2V2 - fixing Noise2Void checkerboard artifacts with "
            "modified sampling strategies and a tweaked network "
            "architecture', ECCV 2022 workshops. spaCR has not scored it on "
            "its own data.")),
    _CELLPROFILER: _BackendSpec(
        name=_CELLPROFILER, label="CellProfiler", module="cellprofiler",
        probe=("javabridge", "bioformats", "cellprofiler_core.preferences",
               "cellprofiler_core.pipeline",
               "cellprofiler_core.utilities.java", "cellprofiler.modules"),
        distribution="cellprofiler",
        requirements=("numpy==1.23.5", "scipy==1.9.0", "scikit-image==0.18.3",
                      "centrosome==1.2.3", "h5py==3.7.0", "matplotlib<3.8",
                      "psutil", "pyzmq~=22.3", "docutils==0.15.2", "boto3",
                      "imageio", "inflect<7", "Jinja2", "joblib", "mahotas",
                      "Pillow", "scikit-learn<1", "six", "future",
                      "tifffile<2022.4.22", "requests", "prokaryote==2.4.4",
                      "cython<3", "wheel", "install-jdk==1.1.0"),
        built_here=("python-javabridge==4.0.3", "python-bioformats==4.0.7"),
        without_dependencies=("cellprofiler-core==4.2.8.1",
                              "cellprofiler==4.2.8.1"),
        torch=(), python=((3, 8), (3, 9)), java="11",
        licence="BSD-3-Clause (CellProfiler) / GPL-2.0 (Bio-Formats, "
                "OpenJDK with Classpath Exception)",
        licence_note=(
            "CellProfiler 4.2.8.1 and cellprofiler-core are BSD-3-Clause "
            "(Broad Institute). They read images through Bio-Formats "
            "(GPL-2.0, shipped inside prokaryote) on a Java 11 development "
            "kit that the install fetches from Eclipse Adoptium (GPL-2.0 "
            "with the Classpath Exception) into the backend's own folder. "
            "CellProfiler's desktop interface (wxPython) and its MySQL "
            "export are not installed: pipelines run headless. spaCR ships "
            "none of it."),
        homepage="https://cellprofiler.org", size_gb=1.5, segments=False,
        alpha=True,
        blurb=(
            "CellProfiler 4.2, run headless on a lab's own .cppipe "
            "pipeline from Measure: spaCR hands it each field's channels "
            "and masks as TIFFs and brings its per-object measurements back "
            "into measurements.db keyed by spaCR's object ids. It needs a "
            "Python 3.8 or 3.9 on this computer to build its environment."),
        published=(
            "Published results: Stirling et al., 'CellProfiler 4: "
            "improvements in speed, utility and usability', BMC "
            "Bioinformatics 2021 (doi:10.1186/s12859-021-04344-9). It "
            "measures what the pipeline says; spaCR scores nothing here.")),
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
    before the segmentation names are checked. So is Cellpose-DINO, which
    segments but is chosen through an object's model setting.

    :raises ValueError: for Cellpose 4 or a name spaCR has no backend for.
    """
    asked = str(name).strip().lower()
    if asked in _SPECS:
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
    chosen = _cellpose3_choice(model_name)
    name = chosen if chosen is not None else str(model_name or "").strip()
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


def _cellpose3_choice(model_name):
    """The Cellpose 3 model a model setting names with ``cellpose3:``.

    :param model_name: an object's model setting, e.g. ``'cellpose3:cyto2'``
        or ``'cellpose3:/models/cp3_weights.pth'``.
    :returns: what follows the prefix -- a Cellpose 3 name or a checkpoint
        path, ``''`` when nothing does -- or None for a setting that does not
        name a Cellpose 3 model, which is every Cellpose-SAM setting.
    """
    text = str(model_name or "").strip()
    if text[:len(_CELLPOSE3_PREFIX)].lower() != _CELLPOSE3_PREFIX:
        return None
    return text[len(_CELLPOSE3_PREFIX):].strip()


def _cellpose3_value(model):
    """The model setting that chooses a Cellpose 3 model or checkpoint.

    :param model: a Cellpose 3 name, a checkpoint path, or a value that
        already carries the prefix.
    :returns: ``'cellpose3:<model>'``.
    """
    chosen = _cellpose3_choice(model)
    return _CELLPOSE3_PREFIX + (chosen if chosen is not None
                                else str(model or "").strip())


def _cellpose3_is_chosen(settings):
    """Whether a run's settings send any object to Cellpose 3.

    True when ``segmentation_backend`` is ``'cellpose3'`` or when an
    object's model setting names a Cellpose 3 model. The legacy Cellpose 3
    settings apply exactly then, and are shown exactly then.

    :param settings: a settings mapping; absent keys count as not chosen.
    :returns: a bool.
    """
    settings = settings or {}
    backend = str(settings.get("segmentation_backend") or "").strip().lower()
    if backend == _CELLPOSE3:
        return True
    return any(_cellpose3_choice(value) is not None
               for key, value in settings.items()
               if str(key).endswith("_model_name")
               or key == "pathogen_model")


def _cellpose_dino_choice(model_name):
    """The checkpoint a model setting names with ``cellpose_dino:``.

    :param model_name: an object's model setting, e.g.
        ``'cellpose_dino:/models/cellposedino_vit_b'``.
    :returns: what follows the prefix, ``''`` when nothing does, or None
        for a setting that does not name a Cellpose-DINO checkpoint.
    """
    text = str(model_name or "").strip()
    if text[:len(_CELLPOSE_DINO_PREFIX)].lower() != _CELLPOSE_DINO_PREFIX:
        return None
    return text[len(_CELLPOSE_DINO_PREFIX):].strip()


def _cellpose_dino_value(path):
    """The model setting that runs a Cellpose-DINO checkpoint.

    :param path: the checkpoint's path, or a value that already carries the
        prefix.
    :returns: ``'cellpose_dino:<path>'``.
    """
    chosen = _cellpose_dino_choice(path)
    return _CELLPOSE_DINO_PREFIX + (chosen if chosen is not None
                                    else str(path or "").strip())


def _cellpose_dino_model(model_name):
    """The checkpoint file a ``cellpose_dino:`` model setting names.

    Cellpose 4 handed a path that is not there logs a warning and runs
    cpsam_v2 in its place, so a missing file is refused here instead.

    :param model_name: the setting, with or without the prefix.
    :returns: the checkpoint's absolute path.
    :raises FileNotFoundError: when no file is there, or none is named.
    """
    chosen = _cellpose_dino_choice(model_name)
    name = chosen if chosen is not None else str(model_name or "").strip()
    path = os.path.expanduser(name)
    if not name or not os.path.isfile(path):
        raise FileNotFoundError(
            f"no Cellpose-DINO checkpoint at {name!r}: the file is not "
            f"there. Download a Cellpose-DINO model from the model zoo and "
            f"press Use this model, which writes "
            f"{_CELLPOSE_DINO_PREFIX}<its path>.")
    return os.path.abspath(path)


def _cellpose_dino_is_chosen(settings):
    """Whether a run's settings send any object to Cellpose-DINO.

    :param settings: a settings mapping.
    :returns: a bool.
    """
    return any(_cellpose_dino_choice(value) is not None
               for key, value in (settings or {}).items()
               if str(key).endswith("_model_name")
               or key == "pathogen_model")


def _prefixed_names():
    """The backends an object chooses by a model-setting prefix of their
    spec's own (StarDist, InstanSeg, Omnipose), in :data:`_SPECS` order."""
    return tuple(name for name, spec in _SPECS.items() if spec.prefix)


def _prefixed_choice(name, model_name):
    """What a model setting names after backend ``name``'s prefix.

    :param name: a backend with a :attr:`_BackendSpec.prefix`.
    :param model_name: an object's model setting, e.g.
        ``'stardist:2D_versatile_fluo'``.
    :returns: what follows the prefix, ``''`` when nothing does, or None
        for a setting without it.
    """
    prefix = _SPECS[name].prefix
    text = str(model_name or "").strip()
    if not prefix or text[:len(prefix)].lower() != prefix:
        return None
    return text[len(prefix):].strip()


def _prefixed_value(name, model):
    """The model setting that runs ``model`` in backend ``name``.

    :param model: a model name or path, or a value already carrying the
        prefix.
    :returns: ``'<prefix><model>'``.
    """
    chosen = _prefixed_choice(name, model)
    return _SPECS[name].prefix + (chosen if chosen is not None
                                  else str(model or "").strip())


def _prefixed_backend(model_name):
    """The prefixed backend a model setting names, or None.

    :param model_name: an object's model setting.
    :returns: a name from :func:`_prefixed_names`, or None.
    """
    for name in _prefixed_names():
        if _prefixed_choice(name, model_name) is not None:
            return name
    return None


def _prefixed_split(name, model_name):
    """``(model, target)`` of a prefixed setting: a ``#<target>`` suffix
    names which of a model's outputs to keep (InstanSeg's ``nuclei`` or
    ``cells``) and is not part of the model."""
    chosen = _prefixed_choice(name, model_name)
    text = chosen if chosen is not None else str(model_name or "").strip()
    model, _hash, target = text.partition("#")
    return model.strip(), target.strip().lower()


def _prefixed_model(name, model_name):
    """The model a prefixed backend's setting selects.

    One of the backend's own model names is used as it is, a blank one is
    its :attr:`_BackendSpec.default_model`, and anything else is a path to a
    model file or folder, which must be there: a missing path is refused
    rather than quietly run as the default.

    :param name: a backend from :func:`_prefixed_names`.
    :param model_name: the setting, with or without the prefix.
    :returns: a model name or an absolute path.
    :raises FileNotFoundError: for a path that names nothing.
    """
    spec = _SPECS[name]
    model, _target = _prefixed_split(name, model_name)
    if not model:
        return spec.default_model
    if model in spec.models:
        return model
    path = os.path.expanduser(model)
    if os.path.exists(path):
        return os.path.abspath(path)
    raise FileNotFoundError(
        f"no {spec.label} model called {model!r}: it is not one of "
        f"{', '.join(spec.models)}, and no file or folder is there.")


def _prefixed_model_ok(value):
    """Whether a prefixed setting names a model its backend can load.

    :returns: True for a model of the backend's own, a bare prefix and a
        path that exists; False for anything else; None when ``value`` has
        no backend prefix.
    """
    name = _prefixed_backend(value)
    if name is None:
        return None
    try:
        _prefixed_model(name, value)
    except FileNotFoundError:
        return False
    return True


def _prefixed_is_chosen(settings):
    """Whether a run's settings send any object to a prefixed backend.

    :param settings: a settings mapping.
    :returns: a bool.
    """
    return any(_prefixed_backend(value) is not None
               for key, value in (settings or {}).items()
               if str(key).endswith("_model_name")
               or key == "pathogen_model")


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


def _requirement_name(requirement):
    """The distribution a requirement names, normalised for comparison.

    :param requirement: e.g. ``'pdfplumber==0.11.10'``.
    :returns: e.g. ``'pdfplumber'``; ``'rapidocr-onnxruntime'`` for either
        spelling of that name.
    """
    name = re.split(r"[<>=!~\[;@ ]", str(requirement).strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _stale_requirements(name, record=None, root=None):
    """The pins ``name`` needs now that its install record does not list.

    An environment holds what spaCR pinned on the day it was built. A pin
    added later -- the figure reader's pdfplumber, added by item 424 after
    the reader first installed (item 469) -- is in no environment built
    before it, and only building the environment again adds it (item 518).
    The record's own list is the evidence; nothing is imported and no
    process is started, so this is cheap enough for the GUI thread.

    :param name: the backend.
    :param record: its install record; read from disk when None.
    :param root: the backends folder, when ``record`` is read.
    :returns: the missing requirements in the spec's order; empty when the
        record lists every one or keeps no list to compare with.
    """
    spec = _spec(name)
    if record is None:
        record = _read_marker(os.path.join(_backends_root(root), spec.name))
    listed = (record or {}).get("requirements")
    if not isinstance(listed, list):
        return []
    have = {_requirement_name(item) for item in listed}
    return [item for item in (spec.requirements + spec.built_here
                              + spec.without_dependencies)
            if _requirement_name(item) not in have]


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
    if spec.java:
        steps.append(_Step(
            "Fetch a Java development kit",
            (python, "-c", "import sys, jdk; jdk.install(sys.argv[1], "
             "path=sys.argv[2])", spec.java,
             os.path.join(env, _JDK_FOLDER))))
    if spec.built_here:
        steps.append(_Step(f"Build {spec.label}'s extensions",
                           pip + ("--no-build-isolation",)
                           + tuple(spec.built_here)))
    if spec.without_dependencies:
        steps.append(_Step(f"Install {spec.label}",
                           pip + ("--no-deps",)
                           + tuple(spec.without_dependencies)))
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

    StarDist fetches its pretrained models with Keras' ``get_file``, which
    keeps them under ``KERAS_HOME`` (``~/.keras`` otherwise).

    InstanSeg downloads its models to ``INSTANSEG_BIOIMAGEIO_PATH``, and
    otherwise into its own package folder.

    Omnipose, like Cellpose, downloads its models to
    ``CELLPOSE_LOCAL_MODELS_PATH``, and otherwise to ``~/.cellpose``.

    SAMCell has two downloads: its fine-tuned checkpoint uses Torch's hub
    cache, and its SAM backbone uses Transformers and Hugging Face. Both
    are scoped to the environment; legacy Transformers cache overrides
    must be removed alongside the Hugging Face overrides.

    micro-SAM fetches its model with pooch into ``MICROSAM_CACHEDIR``,
    which otherwise defaults to the person's own cache folder; it is
    pointed inside the environment for the same reason.

    Setting ``HF_HOME`` is necessary and not sufficient. :func:`_clean_env`
    forwards the rest of the inherited environment, and every variable in
    :data:`_HF_CACHE_VARIABLES` overrides the path ``HF_HOME`` would give,
    so they are dropped here as well. Without that, the one person the fix
    is for -- someone whose Hugging Face cache is already too big for their
    home disk, and who has moved it -- is the one person it would miss.
    """
    environ = _clean_env(env)
    if name in (_CELLPOSE3, _CELLPOSE_DINO):
        environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.join(env, "models")
    elif name == _STARDIST:
        environ["KERAS_HOME"] = os.path.join(env, "keras")
    elif name == _INSTANSEG:
        environ["INSTANSEG_BIOIMAGEIO_PATH"] = os.path.join(
            env, "instanseg_models")
    elif name == _OMNIPOSE:
        environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.join(
            env, "models")
    elif name in (_DINOCELL, _SAMCELL):
        environ["HF_HOME"] = os.path.join(env, "huggingface")
        for variable in _HF_CACHE_VARIABLES:
            environ.pop(variable, None)
        if name == _SAMCELL:
            environ["TORCH_HOME"] = os.path.join(env, "torch")
            for variable in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                             "PYTORCH_PRETRAINED_BERT_CACHE", "HF_MODULES_CACHE"):
                environ.pop(variable, None)
    elif name == _MICROSAM:
        environ["MICROSAM_CACHEDIR"] = os.path.join(env, "micro_sam")
        environ["TORCH_HOME"] = os.path.join(env, "torch")
    spec = _SPECS.get(name)
    if spec is not None and spec.java:
        home = _java_home(env)
        if home:
            environ["JAVA_HOME"] = home
            environ["PATH"] = (os.path.join(home, "bin") + os.pathsep
                               + environ.get("PATH", ""))
        else:
            environ.pop("JAVA_HOME", None)
    return environ


def _java_home(env):
    """The Java development kit inside a backend environment, or ``''``.

    The install fetches it into ``<env>/jdk/<release folder>``; the folder
    whose ``bin`` holds ``javac`` is the one ``JAVA_HOME`` names, so the
    build and the worker never use a Java found elsewhere on the computer.
    """
    folder = os.path.join(env, _JDK_FOLDER)
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return ""
    for name in names:
        home = os.path.join(folder, name)
        if os.path.isfile(os.path.join(home, "bin", "javac")) or \
                os.path.isfile(os.path.join(home, "bin", "javac.exe")):
            return home
    return ""


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


def _run_cellprofiler(pipeline, files, output, *, root=None,
                      worker_for=None, should_cancel=None):
    """Run a CellProfiler pipeline headlessly on ``files``, in its own
    environment.

    The pipeline's own input modules choose among ``files`` exactly as they
    would in CellProfiler; nothing about the pipeline is changed.

    :param pipeline: a ``.cppipe`` or ``.cpproj`` path.
    :param files: the image paths handed to the pipeline's file list.
    :param output: a folder for the per-object tables the worker writes;
        also CellProfiler's default output folder, so a pipeline that
        exports or saves files puts them there.
    :param root: the backends folder.
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    :param should_cancel: polled while it runs; True stops the worker.
    :returns: the worker's reply: ``images`` maps each image number to the
        file names it read, and ``objects`` maps each object name to its
        ``columns`` and the ``path`` of a float ``.npy`` table whose first
        two columns are ``ImageNumber`` and ``ObjectNumber``.
    :raises ImportError: when CellProfiler is not installed here.
    """
    state = _backend_state(_CELLPROFILER, root)
    if not state.ready or state.in_process:
        raise ImportError(_not_installed_message(_CELLPROFILER, state))
    os.makedirs(output, exist_ok=True)
    return (worker_for or _worker_for)(_CELLPROFILER, state.env).request(
        "run_cellprofiler", should_cancel=should_cancel,
        pipeline=os.path.abspath(str(pipeline)),
        files=[os.path.abspath(str(f)) for f in files],
        output=os.path.abspath(str(output)))


class _PromptClient:
    """micro-SAM answering prompts on a field, from its own environment.

    The GUI's half of prompt-based segmentation. Each call sends the prompt
    for a field the worker has already embedded; when it has not -- the
    field is new, the worker was restarted after sitting idle, or it
    dropped the field to keep newer ones -- the worker says so, the field
    is sent and embedded, and the prompt is asked again. The image is asked
    for only then, so a click on a field already embedded sends no image.

    Nothing here imports torch or micro-SAM; the worker does.

    :param model: the micro-SAM model, :data:`_MICROSAM_MODEL` by default.
    :param device: ``'cpu'``, ``'cuda'``, ... or None for ``$SPACR_DEVICE``,
        and failing that the worker's own best guess.
    :param root: the backends folder.
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    """

    def __init__(self, *, model=_MICROSAM_MODEL, device=None, root=None,
                 worker_for=None):
        """Remember what to run; nothing is started until the first prompt."""
        self.model = str(model or _MICROSAM_MODEL)
        self.device = (str(device) if device is not None
                       else os.environ.get(_DEVICE_ENV, "").strip() or "auto")
        self.root = root
        self._worker_for = worker_for or _worker_for

    def readiness(self):
        """Whether micro-SAM can answer a prompt now, and why not.

        File checks only, so the GUI thread may ask.

        :returns: ``(ready, reason)``.
        """
        state = _backend_state(_MICROSAM, self.root)
        if state.ready and not state.in_process:
            return True, f"micro-SAM is installed in {state.env}."
        return False, _not_installed_message(_MICROSAM, state)

    def segment(self, key, image, points=(), labels=(), box=None, *,
                should_cancel=None, on_start=None, on_embed=None):
        """The mask of the one object the prompt points at.

        :param key: names the field and how it was prepared; the worker
            keeps its embedding under it.
        :param image: the field as micro-SAM should see it, or a function
            returning it, called only when the field must be embedded.
        :param points: ``(y, x)`` image pixels.
        :param labels: one per point: 1 on the object, 0 off it.
        :param box: ``(y0, x0, y1, x1)`` image pixels, or None.
        :param should_cancel: polled while waiting; True abandons the
            request and leaves the worker, and its loaded model, running.
        :param on_start: called with no arguments when micro-SAM's worker
            is not running and is about to be started, which loads torch and
            micro-SAM in its environment.
        :param on_embed: called with no arguments just before a field is
            embedded, which is the slow part; the first field embedded also
            downloads the model.
        :returns: a dict: ``mask`` (a boolean ``H x W`` array), ``seconds``
            (the prompt), ``embed_seconds`` (None when the embedding was
            already there), ``score``, ``tiled``, ``device``, ``model`` and
            ``versions`` (the environment's recorded packages).
        :raises ImportError: when micro-SAM is not installed.
        :raises _BackendError: with micro-SAM's own message.
        :raises _BackendCancelled: when ``should_cancel`` said so.
        """
        state = _backend_state(_MICROSAM, self.root)
        if not state.ready or state.in_process:
            raise ImportError(_not_installed_message(_MICROSAM, state))
        running = _WORKERS.get(_MICROSAM)
        if on_start is not None and (running is None or not running.alive):
            on_start()
        worker = self._worker_for(_MICROSAM, state.env)
        payload = {
            "key": str(key),
            "points": [[float(y), float(x)] for y, x in points],
            "labels": [int(bool(v)) for v in labels],
            "box": None if box is None else [float(v) for v in box]}
        embedded = None
        with tempfile.TemporaryDirectory(prefix="spacr_microsam_") as folder:
            output = os.path.join(folder, "mask.npy")
            try:
                reply = worker.request(
                    "sam_prompt", should_cancel=should_cancel,
                    keep_on_cancel=True, output=output, **payload)
            except _BackendError as exc:
                if exc.remote_type != "LookupError":
                    raise
                if on_embed is not None:
                    on_embed()
                field = image() if callable(image) else image
                path = os.path.join(folder, "image.npy")
                np.save(path, np.ascontiguousarray(field), allow_pickle=False)
                embedded = worker.request(
                    "sam_embed", should_cancel=should_cancel,
                    keep_on_cancel=True, image=path, key=str(key),
                    model=self.model, device=self.device)
                reply = worker.request(
                    "sam_prompt", should_cancel=should_cancel,
                    keep_on_cancel=True, output=output, **payload)
            mask = np.load(output, allow_pickle=False).astype(bool)
        packages = dict((state.record or {}).get("packages") or {})
        return {"mask": mask, "seconds": reply.get("seconds"),
                "score": reply.get("score"),
                "embed_seconds": (embedded or {}).get("seconds"),
                "tiled": (embedded or {}).get("tiled"),
                "device": (embedded or {}).get("device")
                or (state.record or {}).get("device", ""),
                "model": self.model, "versions": packages}


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


_OUTPUT_LISTENERS = []
_OUTPUT_LOCK = threading.Lock()


def _listen_to_workers(listener):
    """Hear every line a backend worker prints to stderr, as it prints it.

    Item 507. A worker's progress bars (Cellpose 3 restoration and
    segmentation, a model download) were kept only for an error message; a
    screen that wants them as live progress registers here. ``listener`` is
    called as ``listener(label, line)`` on the worker's reader thread, with
    the line's own ending kept (a bar's redraw ends in ``\\r``), so it must
    return at once and hand the line to its own thread. A bound method is
    held weakly and goes away with its object.

    :param listener: ``listener(label, line)``.
    :returns: a function that stops the listening.
    """
    import weakref

    try:
        held = weakref.WeakMethod(listener)
    except TypeError:
        held = (lambda: listener)
    with _OUTPUT_LOCK:
        _OUTPUT_LISTENERS.append(held)

    def _stop():
        """Stop hearing the workers."""
        with _OUTPUT_LOCK:
            if held in _OUTPUT_LISTENERS:
                _OUTPUT_LISTENERS.remove(held)

    return _stop


def _tell_listeners(label, line):
    """Pass one worker line to everyone listening; a listener that fails or
    has gone is dropped, and the worker never waits on one."""
    with _OUTPUT_LOCK:
        held = list(_OUTPUT_LISTENERS)
    for ref in held:
        listener = ref()
        try:
            if listener is None:
                raise RuntimeError("gone")
            listener(label, line)
        except Exception:
            with _OUTPUT_LOCK:
                if ref in _OUTPUT_LISTENERS:
                    _OUTPUT_LISTENERS.remove(ref)


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
                     worker=None, reinstall=False):
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
    :param reinstall: build it again even though it is installed, the way a
        backend whose record lacks a pin spaCR now needs is brought up to
        date (item 518; see :func:`_stale_requirements`). Its running worker
        is stopped first and its environment removed. Without it an
        installed backend is returned as it is.
    :returns: the :class:`_BackendState` afterwards.
    :raises _InstallBlocked: when this computer cannot install it.
    :raises _InstallFailed: when a step failed, with its output.
    :raises _InstallCancelled: when cancelled.
    """
    spec = _spec(name)
    root = _backends_root(root)
    env = os.path.join(root, spec.name)
    state = _backend_state(spec.name, root)
    if state.state == _INSTALLED and (not reinstall or state.in_process):
        return state
    report = progress or (lambda step, steps, text: None)
    report(0, 1, "Checking this computer can install it")
    _acquire_lock(root, spec.name)
    log_path = os.path.join(root, f"{spec.name}.log")
    try:
        interpreter = (preflight or _preflight)(spec, root)
        _PROBED.pop(spec.name, None)
        if os.path.lexists(env):
            _shutdown_workers(spec.name)
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
            "requirements": list(spec.requirements + spec.built_here
                                 + spec.without_dependencies),
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
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
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
        once, in its last state. Every raw line, redraws included, also goes
        to the screens listening (:func:`_listen_to_workers`).
        """
        _tell_listeners(getattr(self, "label", "") or self.name, line)
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


def _n2v_worker(root=None, worker_for=None):
    """CAREamics' running worker, or ImportError saying how to install it."""
    state = _backend_state(_CAREAMICS, root)
    if state.state != _INSTALLED or state.in_process:
        raise ImportError(_not_installed_message(_CAREAMICS, state))
    return (worker_for or _worker_for)(_CAREAMICS, state.env)


def _n2v_train(images, output, *, epochs=20, seed=0, device=None, root=None,
               worker_for=None, should_cancel=None):
    """Train a Noise2Void (N2V2) denoiser on noisy planes, in CAREamics' own
    environment, with no clean targets.

    Blocking; call it off the GUI thread. The planes go to the worker as
    ``.npy`` files in a scratch folder removed afterwards.

    :param images: 2-D intensity planes, at least :data:`_N2V_PATCH` on a
        side, the noisy images the model learns from.
    :param output: the ``.ckpt`` file the trained model is written to.
    :param epochs: passes over the training patches.
    :param seed: the random seed patches, masking and weights start from.
    :param device: ``'cpu'``, ``'cuda'``, ... or None for the worker's own
        best guess.
    :param root: the backends folder.
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    :param should_cancel: polled while it trains; True stops the worker.
    :returns: the training record: the checkpoint, the losses per epoch, the patch and batch sizes, the device, the seconds and
        CAREamics' version.
    :raises ImportError: when CAREamics is not installed.
    :raises ValueError: for no planes or a plane smaller than a patch.
    """
    planes = [np.asarray(image, dtype=np.float32) for image in images]
    if not planes:
        raise ValueError("Noise2Void needs at least one noisy image to train on")
    for plane in planes:
        if plane.ndim != 2 or min(plane.shape) < _N2V_PATCH:
            raise ValueError(
                f"Noise2Void trains on 2-D planes at least {_N2V_PATCH} "
                f"pixels on a side; got {plane.shape}")
        if not np.isfinite(plane).all():
            raise ValueError("Noise2Void needs finite intensities")
    worker = _n2v_worker(root, worker_for)
    output = os.path.abspath(str(output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spacr_n2v_") as folder:
        inputs = []
        for index, plane in enumerate(planes):
            path = os.path.join(folder, f"plane_{index:04d}.npy")
            np.save(path, plane, allow_pickle=False)
            inputs.append(path)
        reply = worker.request(
            "n2v_train", should_cancel=should_cancel, inputs=inputs,
            output=output, epochs=int(epochs), seed=int(seed),
            patch=_N2V_PATCH, batch=_N2V_BATCH, device=device or "auto",
            work=os.path.join(folder, "work"))
    return {key: value for key, value in reply.items()
            if key not in ("protocol", "id", "ok")}


def _n2v_denoise(image, checkpoint, *, device=None, root=None,
                 worker_for=None, should_cancel=None):
    """One plane denoised by a trained Noise2Void checkpoint.

    :param image: a 2-D intensity plane.
    :param checkpoint: the ``.ckpt`` :func:`_n2v_train` wrote.
    :param device: as for :func:`_n2v_train`.
    :returns: a float32 plane of the same shape, in the input's intensity
        units: CAREamics normalises with the training data's statistics and
        undoes it on the way out.
    :raises ImportError: when CAREamics is not installed.
    """
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("Noise2Void denoises one finite 2-D plane")
    worker = _n2v_worker(root, worker_for)
    with tempfile.TemporaryDirectory(prefix="spacr_n2v_") as folder:
        input_path = os.path.join(folder, "input.npy")
        output_path = os.path.join(folder, "output.npy")
        np.save(input_path, source, allow_pickle=False)
        worker.request("n2v_denoise", should_cancel=should_cancel,
                       keep_on_cancel=source.size <= _KEEP_PIXELS,
                       checkpoint=os.path.abspath(str(checkpoint)),
                       input=input_path, output=output_path,
                       device=device or "auto")
        out = np.load(output_path, allow_pickle=False)
    if out.shape != source.shape or not np.isfinite(out).all():
        raise _BackendError("Noise2Void returned an invalid plane")
    return out.astype(np.float32, copy=False)


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
             augment=None, **cellpose_only):
        """Segment each image of a batch in the backend's worker.

        :param x: a 2-D image, or a list of ``(H, W)`` / ``(H, W, C)``
            images.
        :param normalize: a bool, or Cellpose 3's normalization dict such
            as ``{"normalize": True, "percentile": [1, 99]}``.
        :param augment: Cellpose 3's test-time augmentation; sent only when
            given, so a backend that has none is not asked for it.
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
                  "batch_size": batch_size, "augment": augment}
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

        A NAMED model is a ``Cellpose``, whose ``eval`` ends in ``**kwargs``
        and hands them to the ``CellposeModel`` it holds as ``.cp``. Reading
        the wrapper would accept everything, and a keyword the inner model
        lacks would then fail inside Cellpose rather than be named here, so
        the inner signature is the one read (cellpose 3.1.1.3, 2026-09-25).
        """
        target = getattr(self._model, "cp", None) or self._model
        try:
            signature = inspect.signature(target.eval)
        except (TypeError, ValueError):
            return dict(extra)
        parameters = signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in parameters.values()):
            return dict(extra)
        taken = {k: v for k, v in extra.items() if k in parameters}
        self.ignored.update(set(extra) - set(taken))
        return taken


class _CellposeDinoAdapter:
    """Cellpose 4 with DINOv3, inside its own environment, answering the
    ``eval`` spaCR's Cellpose-SAM path calls.

    The checkpoint is built exactly as Cellpose 4 builds any checkpoint --
    ``CellposeModel(pretrained_model=<path>)``, which reads the backbone
    from the weights (``encoder.cls_token`` means DINO, the width says ViT-L
    or ViT-B) -- and ``eval`` gets the same keywords the Cellpose-SAM path
    passes in spaCR, so the masks, flows and cell probability come back in
    the shapes that path already handles and nothing downstream changes.

    :param model: a Cellpose-DINO checkpoint's path.
    :param device: a torch device name.
    :param models_module: ``cellpose.models``, or a stand-in for tests.
    :raises FileNotFoundError: for a path that names no file.
    :raises ValueError: for a checkpoint that is not a DINO one; Cellpose-SAM
        runs in spaCR itself.
    """

    name = _CELLPOSE_DINO

    def __init__(self, model, device="cpu", models_module=None):
        """Load the checkpoint, in float32 unless the device is CUDA."""
        import torch

        if models_module is None:
            from cellpose import models as models_module
        if not model or not os.path.isfile(model):
            raise FileNotFoundError(
                f"no Cellpose-DINO checkpoint at {model!r}; Cellpose 4 "
                f"would have run cpsam_v2 in its place without a word.")
        self.model = model
        self.ignored = set()
        self.translated = set()
        where = torch.device(device)
        self._model = models_module.CellposeModel(
            gpu=where.type != "cpu", pretrained_model=model, device=where,
            use_bfloat16=where.type == "cuda")
        backbone = str(getattr(self._model, "backbone", "") or "")
        if not backbone.startswith("dino"):
            raise ValueError(
                f"{os.path.basename(model)} is a {backbone or 'non-DINO'} "
                f"Cellpose checkpoint, not a Cellpose-DINO one. Put its path "
                f"in the model setting without {_CELLPOSE_DINO_PREFIX} and "
                f"spaCR's own Cellpose runs it.")

    def eval(self, x, **params):
        """Segment each image as spaCR's Cellpose-SAM path would.

        :param x: a list of ``(H, W)`` or ``(H, W, C)`` images.
        :param params: Cellpose 4 ``eval`` keywords. Those this Cellpose's
            ``eval`` does not take are named in :attr:`ignored`.
        :returns: ``(masks, flows, None)``: one label image per image, and
            per image Cellpose's ``[RGB flow, dP, cell probability, None]``.
        """
        images = [np.asarray(image) for image in x]
        taken = self._accepted({k: v for k, v in params.items()
                                if v is not None})
        output = self._model.eval(images, **taken)
        masks, flows = output[0], output[1]
        masks = [_as_label_image(mask) for mask in masks]
        per_image = []
        for index in range(len(masks)):
            entry = flows[index] if index < len(flows) else ()
            parts = [None if part is None else np.asarray(part)
                     for part in list(entry)[:3]]
            per_image.append(parts + [None] * (4 - len(parts)))
        return masks, per_image, None

    def _accepted(self, extra):
        """The keywords ``CellposeModel.eval`` takes, of those asked for."""
        try:
            parameters = inspect.signature(self._model.eval).parameters
        except (TypeError, ValueError):
            return dict(extra)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in parameters.values()):
            return dict(extra)
        taken = {k: v for k, v in extra.items() if k in parameters}
        self.ignored.update(set(extra) - set(taken))
        return taken


def _drop_small(labels, min_size):
    """``labels`` without the objects smaller than ``min_size`` pixels,
    relabelled as :func:`_as_label_image` does."""
    arr = np.asarray(labels)
    size = int(min_size or 0)
    if size > 1 and arr.max(initial=0) > 0:
        counts = np.bincount(arr.ravel().astype(np.int64))
        small = counts < size
        small[0] = False
        if small.any():
            arr = np.where(small[arr.astype(np.int64)], 0, arr)
    return _as_label_image(arr)


class _PrefixedAdapter:
    """What the StarDist, InstanSeg and Omnipose workers share.

    Each answers the ``eval`` call spaCR's Cellpose-SAM path makes and
    returns ``(masks, flows, None)`` with Cellpose's per-image flows layout
    ``[RGB flow, dP, cell probability, None]``, filling what its model has.
    A setting that changes what Cellpose-SAM segments and that this model
    has no counterpart for is named in :attr:`ignored`; one whose meaning
    had to be changed on the way is said in :attr:`translated`. Settings
    that only arrange the work -- batch size, progress, channel axis -- are
    used where they apply and not reported.
    """

    name = ""
    #: The Cellpose settings this model has no counterpart for.
    unsupported = ()

    def __init__(self):
        """Start with nothing ignored or translated."""
        self.ignored = set()
        self.translated = set()

    def _note(self, params):
        """Record each unsupported setting the call gave a value."""
        for key in self.unsupported:
            if params.get(key) is not None:
                self.ignored.add(key)

    def eval(self, x, channel_axis=-1, min_size=None, **params):
        """Segment each image's object channel.

        :param x: a list of ``(H, W)`` or ``(H, W, C)`` images; the first
            channel is the object's own, as ``_get_cellpose_channels``
            orders them.
        :param min_size: objects smaller than this many pixels are removed.
        :param params: the rest of the Cellpose-SAM call.
        :returns: ``(masks, flows, None)``, one entry per image.
        """
        self._note(params)
        masks, flows = [], []
        for image in x:
            plane = _object_plane(image, channel_axis)
            labels, parts = self._segment(plane, **params)
            labels = np.asarray(labels)
            if labels.shape != plane.shape:
                raise ValueError(
                    f"the {self.name} backend returned labels of shape "
                    f"{labels.shape} for an image of shape {plane.shape}")
            rgb, d_p, probability = (list(parts or ()) + [None] * 3)[:3]
            masks.append(_drop_small(labels, min_size))
            flows.append([
                None if rgb is None else np.asarray(rgb),
                None if d_p is None else np.asarray(d_p, np.float32),
                None if probability is None else _resize_nearest(
                    np.asarray(probability, np.float32), plane.shape),
                None])
        return masks, flows, None

    def _segment(self, plane, **params):
        """``(labels, [RGB flow, dP, probability])`` for one 2-D plane;
        any of the three may be None, and the list may be shorter."""
        raise NotImplementedError


class _StarDistAdapter(_PrefixedAdapter):
    """StarDist 2-D, inside its own TensorFlow environment (item 551).

    A named model is StarDist's own pretrained one, which StarDist
    downloads and checks against its pinned digest; a path is a StarDist
    model folder (``config.json`` beside ``weights_best.h5``), loaded the
    way StarDist loads a model it trained.

    THE SETTINGS. The plane is normalised to its 1st and 99.8th
    percentiles, as StarDist's models were trained and as its own examples
    do; spaCR's own scaling is linear, so this is the normalisation of the
    raw plane (the same argument as Cellpose 3's). The cell-probability
    threshold is a Cellpose logit: its default 0 keeps the probability
    threshold StarDist tuned for the model, any other value becomes a
    probability through the logistic function. A diameter rescales the
    plane so its objects are :data:`_STARDIST_DIAMETER` pixels across
    (StarDist's ``scale``); labels and probability come back at the
    plane's own size. StarDist has no flow threshold or resampling, and
    those are named as not honoured. The minimum size is applied to its
    objects. Large planes are tiled as StarDist itself guesses.

    :param model: a name from :data:`_STARDIST_MODELS` or a model folder.
    :param device: accepted for the shared signature; TensorFlow places
        the network itself.
    :raises FileNotFoundError: for a folder that is not there.
    """

    name = _STARDIST
    unsupported = ("flow_threshold", "resample")

    def __init__(self, model="2D_versatile_fluo", device="cpu",
                 models_module=None):
        """Load the model."""
        super().__init__()
        if models_module is None:
            from stardist import models as models_module
        self.model = model
        if model in _STARDIST_MODELS:
            self._model = models_module.StarDist2D.from_pretrained(model)
        elif os.path.isdir(model):
            folder = os.path.abspath(model)
            self._model = models_module.StarDist2D(
                None, name=os.path.basename(folder),
                basedir=os.path.dirname(folder))
        else:
            raise FileNotFoundError(
                f"no StarDist model called {model!r}: it is not one of "
                f"{', '.join(_STARDIST_MODELS)}, and no model folder is "
                f"there.")

    def _segment(self, plane, normalize=True, cellprob_threshold=None,
                 diameter=None, **other):
        """StarDist's instances and object probability for one plane."""
        from csbdeep.utils import normalize as percentile_normalize

        if normalize is False:
            self.translated.add(
                "normalize=False became StarDist's 1-99.8 percentile "
                "normalisation, which its models were trained on")
        image = percentile_normalize(np.asarray(plane, np.float32), 1, 99.8,
                                     axis=(0, 1))
        threshold = None
        if cellprob_threshold not in (None, 0, 0.0):
            threshold = _probability_threshold(cellprob_threshold)
            self.translated.add(
                f"cellprob_threshold={cellprob_threshold} became StarDist's "
                f"prob_thresh={threshold:.3f}")
        scale = (_STARDIST_DIAMETER / float(diameter)
                 if diameter and float(diameter) > 0 else None)
        guess = getattr(self._model, "_guess_n_tiles", None)
        tiles = guess(image) if callable(guess) else None
        (labels, _details), (probability, _distances) = (
            self._model.predict_instances(
                image, prob_thresh=threshold, n_tiles=tiles, scale=scale,
                show_tile_progress=False, return_predict=True, verbose=False))
        return labels, [None, None, probability]


class _InstanSegAdapter(_PrefixedAdapter):
    """InstanSeg, inside its own environment (item 552).

    A named model is InstanSeg's own, which InstanSeg downloads from its
    GitHub release into ``INSTANSEG_BIOIMAGEIO_PATH`` (inside the backend's
    folder); a path is an InstanSeg TorchScript file (``instanseg.pt``, or
    the folder holding one).

    THE OUTPUT. ``fluorescence_nuclei_and_cells`` segments both; ``target``
    keeps one: ``nuclei`` for a nucleus object, ``cells`` for every other,
    unless the model setting ends in ``#nuclei`` or ``#cells``. A model
    with one output ignores it.

    THE SETTINGS. InstanSeg normalises each plane to its own percentiles,
    as it was trained, whatever ``normalize`` says (spaCR's scaling is
    linear, so this is the normalisation of the raw plane). InstanSeg
    rescales by pixel size, and spaCR's mask settings carry none, so a
    diameter is given to it as the pixel size that makes the objects
    :data:`_INSTANSEG_DIAMETER` pixels across at the model's own pixel
    size; a blank diameter runs the plane at the model's pixel size. A
    plane InstanSeg calls small is segmented whole, a larger one in its own
    512-pixel tiles. The flow threshold, cell-probability threshold and
    resampling have no InstanSeg counterpart and are named as not honoured.
    The minimum size is applied to its objects. InstanSeg gives no
    probability map.

    :param model: a name from :data:`_INSTANSEG_MODELS`, a TorchScript file,
        or a folder holding ``instanseg.pt``.
    :param device: a torch device name.
    :param target: ``'nuclei'``, ``'cells'`` or ``''`` (``'cells'``).
    :param instanseg_class: ``instanseg.InstanSeg``, or a stand-in for tests.
    :raises FileNotFoundError: for a path with no model.
    """

    name = _INSTANSEG
    unsupported = ("flow_threshold", "cellprob_threshold", "resample")

    def __init__(self, model="fluorescence_nuclei_and_cells", device="cpu",
                 target="", instanseg_class=None):
        """Load the model on ``device``."""
        super().__init__()
        if instanseg_class is None:
            from instanseg import InstanSeg as instanseg_class
        self.model = model
        self.target = target if target in ("nuclei", "cells") else "cells"
        if model in _INSTANSEG_MODELS:
            network = model
        else:
            path = (os.path.join(model, "instanseg.pt")
                    if os.path.isdir(model) else model)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"no InstanSeg model called {model!r}: it is not one of "
                    f"{', '.join(_INSTANSEG_MODELS)}, and no TorchScript "
                    f"file is there.")
            import torch

            network = torch.jit.load(path, map_location="cpu")
        self._model = instanseg_class(network, device=device, verbosity=0)

    def _pixel_size(self, diameter):
        """The pixel size that brings ``diameter`` to the model's scale."""
        if not diameter or float(diameter) <= 0:
            return None
        native = getattr(getattr(self._model, "instanseg", None),
                         "pixel_size", None)
        if not native:
            self.ignored.add("diameter")
            return None
        return float(native) * _INSTANSEG_DIAMETER / float(diameter)

    def _segment(self, plane, normalize=True, diameter=None, **other):
        """InstanSeg's instances of the chosen target for one plane."""
        if normalize is False:
            self.translated.add(
                "normalize=False became InstanSeg's own percentile "
                "normalisation, which its models were trained on")
        image = np.asarray(plane, np.float32)[np.newaxis]
        pixel_size = self._pixel_size(diameter)
        choose = getattr(self._model, "_get_eval_function_to_use", None)
        size = choose(plane.size) if callable(choose) else "small"
        if size == "small":
            labels = self._model.eval_small_image(
                image, pixel_size=pixel_size, normalise=True,
                return_image_tensor=False, target=self.target)
        else:
            labels = self._model.eval_medium_image(
                image, pixel_size=pixel_size, normalise=True, tile_size=512,
                batch_size=1, return_image_tensor=False, target=self.target)
        labels = np.asarray(labels.numpy() if hasattr(labels, "numpy")
                            else labels)
        return labels.reshape((-1,) + labels.shape[-2:])[0], []


def _instanseg_options(model_name, object_type=None):
    """The output InstanSeg keeps: the setting's ``#nuclei`` / ``#cells``,
    else ``nuclei`` for a nucleus object and ``cells`` for any other."""
    _model, target = _prefixed_split(_INSTANSEG, model_name)
    if target not in ("nuclei", "cells"):
        target = "nuclei" if object_type == "nucleus" else "cells"
    return {"target": target}


class _OmniposeAdapter(_PrefixedAdapter):
    """Omnipose, inside its own environment (item 553).

    A named model is one of Omnipose's own, built as Omnipose builds it
    (``cellpose_omni.models.CellposeModel(model_type=...)``), which fetches
    its weights into ``CELLPOSE_LOCAL_MODELS_PATH`` inside the backend's
    folder. A path is an Omnipose checkpoint; its input channels and output
    classes are read from the weights, because Omnipose would otherwise
    build a network of its default shape and fail to load them.

    THE SETTINGS. Each plane is segmented as one grey channel
    (``channels=[0, 0]``) with ``omni=True``, Omnipose's own percentile
    normalisation (spaCR's scaling is linear, so this is the normalisation
    of the raw plane) and no rescaling: Omnipose's bacterial models are
    used at the image's own scale, which is how Omnipose runs them, so the
    diameter is named as not honoured. The flow threshold is Omnipose's
    flow threshold and the cell-probability threshold is its
    ``mask_threshold`` on the distance field, both logits of the same kind
    Cellpose's are; resampling is Omnipose's own ``resample``. Its flows
    come back as Cellpose's do: the RGB flow, ``dP`` and, where Cellpose
    has the cell probability, Omnipose's distance field.

    :param model: a name from :data:`_OMNIPOSE_MODELS` or a checkpoint path.
    :param device: a torch device name.
    :param models_module: ``cellpose_omni.models``, or a stand-in for tests.
    :raises FileNotFoundError: for a path that names no file.
    """

    name = _OMNIPOSE
    unsupported = ("diameter",)

    def __init__(self, model="bact_phase_omni", device="cpu",
                 models_module=None):
        """Build the network and load its weights on ``device``."""
        super().__init__()
        import torch

        if models_module is None:
            from cellpose_omni import models as models_module
        self.model = model
        where = torch.device(device)
        gpu = where.type != "cpu"
        if model in _OMNIPOSE_MODELS:
            self._model = models_module.CellposeModel(
                gpu=gpu, model_type=model, device=where)
        elif os.path.isfile(model):
            nchan, nclasses = _omnipose_shape(model)
            self._model = models_module.CellposeModel(
                gpu=gpu, pretrained_model=model, device=where, nchan=nchan,
                nclasses=nclasses, dim=2, omni=True)
        else:
            raise FileNotFoundError(
                f"no Omnipose model called {model!r}: it is not one of "
                f"{', '.join(_OMNIPOSE_MODELS)}, and no file is there. "
                f"Omnipose would have run cyto in its place without a "
                f"word.")

    def _segment(self, plane, normalize=True, flow_threshold=None,
                 cellprob_threshold=None, resample=None, **other):
        """Omnipose's masks and distance field for one plane."""
        if normalize is False:
            self.translated.add(
                "normalize=False became Omnipose's percentile "
                "normalisation, which its models were trained on")
        output = self._model.eval(
            np.asarray(plane, np.float32), channels=[0, 0], rescale=None,
            omni=True, normalize=True,
            flow_threshold=0.4 if flow_threshold is None
            else float(flow_threshold),
            mask_threshold=0.0 if cellprob_threshold is None
            else float(cellprob_threshold),
            resample=True if resample is None else bool(resample),
            tile=False, augment=False, verbose=False)
        return output[0], list(output[1])[:3]


def _omnipose_shape(path):
    """``(input channels, output classes)`` of an Omnipose checkpoint.

    Read from the weights: the first convolution's input channels, and the
    output layer's channels less the one extra flow component a 2-D model
    has, which is how ``CellposeModel`` counts classes.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    convs = [value for value in state.values()
             if hasattr(value, "ndim") and value.ndim == 4]
    if not convs:
        raise ValueError(f"{os.path.basename(path)} holds no Omnipose network")
    return int(convs[0].shape[1]), int(convs[-1].shape[0]) - 1


#: Backend name -> in-process class. Tests replace entries with stubs.
_BACKEND_CLASSES = {_DINOCELL: _DinoCellBackend, _SAMCELL: _SamCellBackend}

#: Backend name -> the worker adapter of a prefixed backend.
_PREFIXED_ADAPTERS = {_STARDIST: _StarDistAdapter,
                      _INSTANSEG: _InstanSegAdapter,
                      _OMNIPOSE: _OmniposeAdapter}


def _worker_device(requested=None):
    """The device a worker runs on: the one asked for, else CUDA, else
    Apple's Metal, else the CPU.

    StarDist's environment has TensorFlow and no PyTorch; there a GPU
    TensorFlow can see is ``'gpu'``.
    """
    wanted = str(requested or "auto").strip().lower()
    if wanted not in ("", "auto"):
        return wanted
    try:
        import torch
    except ImportError:
        return _tensorflow_device()
    if torch.cuda.is_available():
        return "cuda"
    metal = getattr(getattr(torch, "backends", None), "mps", None)
    if metal is not None and metal.is_available():
        return "mps"
    return "cpu"


def _tensorflow_device():
    """``'gpu'`` when TensorFlow sees one, else ``'cpu'``."""
    try:
        import tensorflow as tf
    except ImportError:
        return "cpu"
    return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"


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
    for distribution in (spec.distribution, "torch", "numpy") + (
            () if spec.torch else ("tensorflow",)):
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
    if name == _CELLPOSE_DINO:
        return _CellposeDinoAdapter(model, device)
    if name in _PREFIXED_ADAPTERS:
        return _PREFIXED_ADAPTERS[name](model or _SPECS[name].default_model,
                                        device, **options)
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


def _cellprofiler_started(adapters):
    """Start CellProfiler headless and its Java machine, once per worker.

    A Java machine cannot be started twice in one process, so the worker
    keeps it for its lifetime and stops it as the process exits.
    """
    if "cellprofiler" not in adapters:
        import cellprofiler_core.preferences as preferences
        from cellprofiler_core.utilities.java import start_java, stop_java

        preferences.set_headless()
        preferences.set_allow_schema_write(False)
        start_java()
        atexit.register(stop_java)
        adapters["cellprofiler"] = preferences
    return adapters["cellprofiler"]


def _worker_run_cellprofiler(request, adapters):
    """Run one CellProfiler pipeline on a list of images.

    :param request: ``pipeline`` (a ``.cppipe`` or ``.cpproj`` path),
        ``files`` (the images the pipeline's input modules choose from) and
        ``output`` (where the object tables are written; also CellProfiler's
        default output folder).
    :param adapters: the worker's cache; the Java machine starts once.
    :returns: ``{"image_sets": n, "images": {number: [file names]},
        "objects": {name: {"columns": [...], "path": ".npy"}}}``. Only
        numeric per-object features are kept, each object table's first
        two columns being ``ImageNumber`` and ``ObjectNumber``.
    :raises RuntimeError: when the pipeline does not complete.
    """
    import numpy as np

    preferences = _cellprofiler_started(adapters)
    from cellprofiler_core.pipeline import Pipeline

    pipeline_path = str(request["pipeline"])
    files = [str(f) for f in request.get("files") or []]
    output = str(request["output"])
    if not os.path.isfile(pipeline_path):
        raise FileNotFoundError(f"no CellProfiler pipeline at {pipeline_path}")
    if not files:
        raise ValueError("no images were given to the CellProfiler pipeline")
    os.makedirs(output, exist_ok=True)
    preferences.set_default_output_directory(output)
    preferences.set_default_image_directory(os.path.dirname(files[0]))
    pipeline = Pipeline()
    pipeline.load(pipeline_path)
    pipeline.add_pathnames_to_file_list(files)
    measurements = pipeline.run()
    if measurements is None:
        raise RuntimeError("the CellProfiler pipeline produced no "
                           "measurements; its input modules matched no "
                           "image set among the files spaCR gave it")
    status = ""
    if measurements.has_feature("Experiment", "Exit_Status"):
        status = str(measurements.get_experiment_measurement("Exit_Status"))
    if status and status != "Complete":
        raise RuntimeError(f"the CellProfiler pipeline stopped: {status}")
    numbers = [int(n) for n in measurements.get_image_numbers()]
    names = [f for f in measurements.get_feature_names("Image")
             if f.startswith(("FileName_", "ObjectsFileName_"))]
    images = {}
    for number in numbers:
        images[str(number)] = [
            str(measurements.get_measurement("Image", f, number))
            for f in names]
    objects = {}
    for index, name in enumerate(measurements.get_object_names()):
        if name in ("Image", "Experiment"):
            continue
        features = [f for f in measurements.get_feature_names(name)
                    if f != "Number_Object_Number"]
        rows, keep = [], None
        for number in numbers:
            ids = np.asarray(measurements.get_measurement(
                name, "Number_Object_Number", number), dtype=float).ravel()
            if not ids.size:
                continue
            values = []
            for feature in features:
                column = np.asarray(measurements.get_measurement(
                    name, feature, number)).ravel()
                if column.size != ids.size:
                    column = np.full(ids.size, np.nan)
                try:
                    values.append(column.astype(float))
                except (TypeError, ValueError):
                    values.append(None)
            numeric = [v is not None for v in values]
            keep = numeric if keep is None else [
                a and b for a, b in zip(keep, numeric)]
            rows.append((number, ids, values))
        if keep is None:
            continue
        columns = [f for f, k in zip(features, keep) if k]
        blocks = []
        for number, ids, values in rows:
            kept = [v for v, k in zip(values, keep) if k]
            blocks.append(np.column_stack(
                [np.full(ids.size, float(number)), ids] + kept))
        path = os.path.join(output, f"objects_{index}.npy")
        np.save(path, np.vstack(blocks), allow_pickle=False)
        objects[name] = {"columns": ["ImageNumber", "ObjectNumber"] + columns,
                         "path": path}
    return {"image_sets": len(numbers), "images": images, "objects": objects}


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
    except ModuleNotFoundError as exc:
        if exc.name != "pdfplumber":
            raise ImportError(
                f"pdfplumber is in the figure reader's environment but does "
                f"not load: {exc}") from exc
        raise ModuleNotFoundError(
            "This figure reader's environment has no pdfplumber: it was "
            "installed before the reader read PDFs. Plaque Assay offers to "
            "reinstall it in Figure mode, which adds pdfplumber.",
            name="pdfplumber") from exc
    except ImportError as exc:
        raise ImportError(
            f"pdfplumber is in the figure reader's environment but does not "
            f"load: {exc}") from exc
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


def _sam_predictor(adapters, model, device):
    """micro-SAM's predictor for ``model`` on ``device``, loaded once.

    The first call downloads the model into ``MICROSAM_CACHEDIR`` -- inside
    the backend's environment (:func:`_worker_env`) -- and every later one
    returns the predictor already in memory.
    """
    from micro_sam import util

    key = ("sam", model, device)
    if key not in adapters:
        adapters[key] = util.get_sam_model(model_type=model, device=device)
    return adapters[key]


def _worker_sam_embed(request, adapters):
    """Compute and keep micro-SAM's embedding of one field.

    The embedding is the expensive half of a prompt -- the image encoder
    over the whole field -- and it depends on the field alone, so it is
    computed once per field and kept under the caller's ``key``; the last
    :data:`_MICROSAM_KEEP` fields are kept. A field whose longer side is
    above :data:`_MICROSAM_TILE_ABOVE` is embedded in tiles.

    :param request: ``image`` (a ``.npy`` path, ``H x W`` or ``H x W x C``),
        ``key``, ``model`` and ``device``.
    :param adapters: the worker's cache.
    :returns: ``{"key", "seconds", "tiled", "shape", "device", "model"}``.
    """
    from micro_sam import util

    device = _worker_device(request.get("device"))
    model = str(request.get("model") or _MICROSAM_MODEL)
    started = time.monotonic()
    predictor = _sam_predictor(adapters, model, device)
    loaded = time.monotonic() - started
    image = np.load(str(request["image"]), allow_pickle=False)
    if image.ndim not in (2, 3) or not all(image.shape[:2]):
        raise ValueError("micro-SAM needs one nonempty 2-D field, with or "
                         "without channels.")
    tiled = max(image.shape[:2]) > _MICROSAM_TILE_ABOVE
    started = time.monotonic()
    embeddings = util.precompute_image_embeddings(
        predictor, image, ndim=2,
        tile_shape=(_MICROSAM_TILE, _MICROSAM_TILE) if tiled else None,
        halo=(_MICROSAM_HALO, _MICROSAM_HALO) if tiled else None,
        verbose=tiled)
    seconds = time.monotonic() - started
    kept = adapters.setdefault("sam_embeddings", collections.OrderedDict())
    key = str(request["key"])
    kept.pop(key, None)
    kept[key] = (model, device, embeddings, tuple(image.shape[:2]))
    while len(kept) > _MICROSAM_KEEP:
        kept.popitem(last=False)
    return {"key": key, "seconds": seconds, "load_seconds": loaded,
            "tiled": bool(tiled), "shape": list(image.shape[:2]),
            "device": str(device), "model": model}


def _worker_sam_prompt(request, adapters):
    """Answer one prompt on a field already embedded, with that object's mask.

    Points are ``(y, x)`` in image pixels, each with label 1 (on the
    object) or 0 (not on it); the box is ``(y0, x0, y1, x1)``. Points, a
    box, or both, the way micro-SAM's own annotator takes them.

    :param request: ``key``, ``points``, ``labels``, ``box`` and ``output``
        (the ``.npy`` path the mask is written to).
    :param adapters: the worker's cache.
    :returns: ``{"seconds", "score", "pixels"}``.
    :raises LookupError: when the field has no embedding here -- the worker
        was restarted, or the field was dropped to keep others -- so spaCR
        embeds it again and asks once more.

    On the CPU the prompt runs on at most :data:`_MICROSAM_PROMPT_THREADS`
    threads, and the thread count is put back afterwards.
    """
    import torch
    from micro_sam import prompt_based_segmentation as prompts

    key = str(request["key"])
    entry = adapters.get("sam_embeddings", {}).get(key)
    if entry is None:
        raise LookupError(f"micro-SAM has no embedding for field {key!r}.")
    model, device, embeddings, shape = entry
    adapters["sam_embeddings"].move_to_end(key)
    predictor = _sam_predictor(adapters, model, device)
    points = np.asarray(request.get("points") or [], dtype=float).reshape(-1, 2)
    labels = np.asarray(request.get("labels") or [], dtype=int).reshape(-1)
    if len(labels) != len(points):
        raise ValueError("micro-SAM needs one label for every point.")
    box = request.get("box")
    box = None if box is None else np.asarray(box, dtype=float).reshape(4)
    threads = torch.get_num_threads()
    if str(device) == "cpu":
        torch.set_num_threads(max(1, min(threads, _MICROSAM_PROMPT_THREADS)))
    try:
        started = time.monotonic()
        mask, scores = _sam_answer(prompts, predictor, embeddings, points,
                                   labels, box)
        seconds = time.monotonic() - started
    finally:
        torch.set_num_threads(threads)
    mask = np.asarray(mask).astype(bool)
    while mask.ndim > 2:
        mask = mask[0]
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"micro-SAM returned a {mask.shape} mask for a "
                         f"{shape} field.")
    np.save(str(request["output"]), mask, allow_pickle=False)
    score = np.asarray(scores, dtype=float).ravel()
    return {"seconds": seconds,
            "score": float(score.max()) if score.size else None,
            "pixels": int(mask.sum())}


def _sam_answer(prompts, predictor, embeddings, points, labels, box):
    """micro-SAM's answer to points, a box or both: ``(mask, scores)``.

    :param prompts: ``micro_sam.prompt_based_segmentation``.
    :raises ValueError: when there is neither a point nor a box.
    """
    if box is not None and len(points):
        mask, scores, _logits = prompts.segment_from_box_and_points(
            predictor, box, points, labels, image_embeddings=embeddings,
            return_all=True)
    elif box is not None:
        mask, scores, _logits = prompts.segment_from_box(
            predictor, box, image_embeddings=embeddings, return_all=True)
    elif len(points):
        mask, scores, _logits = prompts.segment_from_points(
            predictor, points, labels, image_embeddings=embeddings,
            return_all=True)
    else:
        raise ValueError("micro-SAM needs a point or a box.")
    return mask, scores


def _n2v_accelerator(device):
    """Lightning's accelerator for a worker device name."""
    return {"cuda": "gpu", "mps": "mps"}.get(str(device).split(":")[0], "cpu")


def _worker_n2v_train(request, adapters):
    """Train CAREamics' N2V2 on the request's planes and save a checkpoint.

    A tenth of the patches, one to eight, are set aside from the training
    planes for validation, as CAREamics does; with no clean targets there
    is nothing else to validate against. The patches are loaded in the
    worker's own process: a data-loader process forked from a worker that
    is reading its requests on a thread never starts.
    """
    import careamics
    import lightning
    import torch
    from careamics import CAREamist
    from careamics.config import create_n2v_config

    planes = [np.load(path, allow_pickle=False).astype(np.float32)
              for path in request["inputs"]]
    device = _worker_device(request.get("device"))
    lightning.seed_everything(int(request.get("seed", 0)), workers=True)
    patch = int(request.get("patch", _N2V_PATCH))
    patches = sum((p.shape[0] // patch) * (p.shape[1] // patch) for p in planes)
    if patches < 2:
        raise ValueError("Noise2Void needs at least two training patches")
    validation = max(1, min(8, patches // 10))
    config = create_n2v_config(
        experiment_name="spacr_n2v", data_type="array", axes="YX",
        patch_size=[patch, patch], batch_size=int(request.get("batch", _N2V_BATCH)),
        num_epochs=int(request.get("epochs", 20)), use_n2v2=True,
        n_val_patches=validation)
    data = config.data_config
    for loader in (data.train_dataloader_params, data.val_dataloader_params,
                   data.pred_dataloader_params):
        loader["num_workers"] = 0
        loader.pop("persistent_workers", None)
    params = dict(config.training_config.trainer_params or {})
    params.update(accelerator=_n2v_accelerator(device), devices=1,
                  enable_progress_bar=False)
    config.training_config.trainer_params = params
    losses = _N2VLosses()
    started = time.monotonic()
    careamist = CAREamist(config, work_dir=request["work"],
                          callbacks=[losses.callback()],
                          enable_progress_bar=False)
    careamist.train(train_data=planes)
    output = str(request["output"])
    careamist.trainer.save_checkpoint(output)
    adapters.pop(("n2v", output), None)
    return {"checkpoint": output,
            "method": "N2V2 (CAREamics)", "epochs": int(params.get(
                "max_epochs", request.get("epochs", 20))),
            "patch": patch, "batch": int(request.get("batch", _N2V_BATCH)),
            "planes": len(planes), "shapes": [list(p.shape) for p in planes],
            "patches": patches, "validation_patches": validation,
            "seed": int(request.get("seed", 0)),
            "train_loss": losses.train, "val_loss": losses.val,
            "device": device, "seconds": round(time.monotonic() - started, 3),
            "careamics": careamics.__version__, "torch": torch.__version__}


class _N2VLosses:
    """The training and validation losses Lightning logs, epoch by epoch."""

    def __init__(self):
        """Start with no epochs."""
        self.train = []
        self.val = []

    def callback(self):
        """A Lightning callback that appends each epoch's losses here."""
        from lightning.pytorch.callbacks import Callback

        record = self

        class _Record(Callback):
            """Copy the logged epoch losses into the record."""

            def on_train_epoch_end(self, trainer, module):
                """Keep the epoch's training loss."""
                value = trainer.callback_metrics.get("train_loss_epoch",
                                                     trainer.callback_metrics.get("train_loss"))
                if value is not None:
                    record.train.append(round(float(value), 6))

            def on_validation_epoch_end(self, trainer, module):
                """Keep the epoch's validation loss, sanity checks aside."""
                value = trainer.callback_metrics.get("val_loss")
                if value is not None and not trainer.sanity_checking:
                    record.val.append(round(float(value), 6))

        return _Record()


def _worker_n2v_denoise(request, adapters):
    """Denoise one plane with a trained checkpoint, loaded once per worker."""
    from careamics import CAREamist

    checkpoint = str(request["checkpoint"])
    key = ("n2v", checkpoint)
    careamist = adapters.get(key)
    if careamist is None:
        careamist = CAREamist(checkpoint_path=checkpoint,
                              work_dir=os.path.dirname(checkpoint),
                              enable_progress_bar=False)
        adapters[key] = careamist
    image = np.load(request["input"], allow_pickle=False).astype(np.float32)
    tile = _N2V_PATCH * 4
    tiled = max(image.shape) > tile
    predictions = careamist.predict(
        image, axes="YX", data_type="array",
        tile_size=(tile, tile) if tiled else None,
        tile_overlap=(_N2V_PATCH // 2, _N2V_PATCH // 2) if tiled else None,
        num_workers=0)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, list):
        predictions = predictions[0]
    out = np.asarray(predictions, dtype=np.float32).reshape(image.shape)
    np.save(request["output"], out, allow_pickle=False)
    return {"output": request["output"]}


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
        elif op == "n2v_train":
            body = _worker_n2v_train(request, adapters)
        elif op == "n2v_denoise":
            body = _worker_n2v_denoise(request, adapters)
        elif op == "run_cellprofiler":
            body = _worker_run_cellprofiler(request, adapters)
        elif op == "read_text":
            body = _worker_read_text(request, adapters)
        elif op == "read_pdf":
            body = _worker_read_pdf(request, adapters)
        elif op == "sam_embed":
            body = _worker_sam_embed(request, adapters)
        elif op == "sam_prompt":
            body = _worker_sam_prompt(request, adapters)
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
    A ``cellpose_dino`` name, or a model setting reading
    ``cellpose_dino:<path>``, builds the Cellpose-DINO backend on that
    checkpoint; it runs only in its own environment.
    A StarDist, InstanSeg or Omnipose name, or a model setting carrying
    one of their prefixes (``stardist:``, ...), builds that backend on the
    model the setting names (:func:`_load_prefixed`).

    :raises ValueError: for Cellpose, an unknown name, or a 3-D/4-D run.
    :raises ImportError: when the backend is not installed.
    :raises FileNotFoundError: for a Cellpose-DINO checkpoint not there.
    """
    if (str(name or "").strip().lower() == _CELLPOSE_DINO
            or _cellpose_dino_choice(model_name) is not None):
        return _load_cellpose_dino(model_name, device=device, z_plan=z_plan,
                                   t_plan=t_plan, root=root)
    prefixed = _prefixed_backend(model_name)
    if prefixed is None and str(name or "").strip().lower() in _prefixed_names():
        prefixed = str(name).strip().lower()
    if prefixed is not None:
        return _load_prefixed(prefixed, model_name, device=device,
                              z_plan=z_plan, t_plan=t_plan,
                              object_type=object_type, root=root)
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


def _load_cellpose_dino(model_name, *, device=None, z_plan=None, t_plan=None,
                        root=None, worker_for=None):
    """The Cellpose-DINO backend on one checkpoint, as a model with ``eval``.

    :param model_name: ``cellpose_dino:<path>`` or the bare path.
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    :raises ValueError: for a z_stack or t_stack run.
    :raises FileNotFoundError: when the checkpoint is not there.
    :raises ImportError: when the backend is not installed.
    """
    if z_plan is not None or t_plan is not None:
        raise ValueError(
            "Cellpose-DINO segments single 2-D planes, and this run has "
            "z_stack or t_stack on. Turn them off, or segment this object "
            "with a Cellpose-SAM model.")
    checkpoint = _cellpose_dino_model(model_name)
    model = _RemoteBackend(_CELLPOSE_DINO, model=checkpoint, device=device,
                           root=root, worker_for=worker_for)
    print(f"Segmentation backend: {_CELLPOSE_DINO} -- {model.note}.")
    return model


def _load_prefixed(name, model_name, *, device=None, z_plan=None,
                   t_plan=None, object_type=None, root=None,
                   worker_for=None):
    """A prefixed backend (StarDist, InstanSeg, Omnipose) on one model.

    :param name: a backend from :func:`_prefixed_names`.
    :param model_name: ``<prefix><model>``, or the bare model; blank runs
        the backend's default model. A ``#nuclei`` / ``#cells`` suffix
        chooses InstanSeg's output; without it a nucleus object takes the
        nuclei and every other object the cells.
    :param object_type: the object being segmented.
    :param worker_for: :func:`_worker_for`, or a stand-in for tests.
    :raises ValueError: for a z_stack or t_stack run.
    :raises FileNotFoundError: when a model path is not there.
    :raises ImportError: when the backend is not installed.
    """
    spec = _SPECS[name]
    if z_plan is not None or t_plan is not None:
        raise ValueError(
            f"{spec.label} segments single 2-D planes, and this run has "
            f"z_stack or t_stack on. Turn them off, or segment this object "
            f"with a Cellpose-SAM model.")
    model = _prefixed_model(name, model_name)
    options = _prefixed_options(name, model_name, object_type)
    backend = _RemoteBackend(name, model=model, device=device, root=root,
                             options=options, worker_for=worker_for)
    print(f"Segmentation backend: {name} -- {backend.note}.")
    return backend


def _prefixed_options(name, model_name, object_type=None):
    """What a prefixed backend's worker is told beside the model: the
    output InstanSeg keeps (:func:`_instanseg_options`). Empty for the
    others."""
    if name == _INSTANSEG:
        return _instanseg_options(model_name, object_type)
    return {}


if __name__ == "__main__":
    sys.exit(_worker_main())
