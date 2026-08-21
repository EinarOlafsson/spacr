"""Report which regression backends are available in this environment.

:mod:`spacr.regression_spec` says what the backends ARE -- pure data, no
imports. This module is the only place that asks the environment about them:
is the package installed, is there a CUDA device, and can this backend fit the
selected ``regression_type``? Each disabled entry includes its reason; an
inapplicable control is never hidden or silently substituted.

The availability probe does not import torch while a settings panel is being
built. It inspects driver device nodes unless torch is already loaded, then
:func:`spacr.mixed_gpu.resolve_device` performs the definitive fit-time check.

Installing optional backends
----------------------------

Three optional backends cannot be enabled by pressing Install, so this section
explains why and what
to do instead. It is written here rather than only in
:data:`INSTALL_RECIPES` because ``docs/source/api`` is built by sphinx-autoapi
from this source, and autoapi publishes a module DOCSTRING verbatim while it
renders a dict as a truncated repr. The two copies cannot drift:
``test_the_module_docstring_carries_every_command_the_gui_shows`` asserts that
every command in :data:`INSTALL_RECIPES` appears below.

**pymer4 / lme4 needs R.** It is an interface to R's lme4, not a
reimplementation. The pymer4 0.9.2 wheel does not declare its runtime
dependencies, although its modules import ``polars`` and ``rpy2``. Install the
Python dependencies and R packages explicitly::

    conda install -c conda-forge r-base
    R -e 'install.packages(c("lme4","lmerTest"), repos="https://cloud.r-project.org")'
    pip install rpy2 polars
    pip install pymer4

This is a heavier ask than the other backends -- a second language runtime, its
own package library and a compiled bridge between them -- which is said plainly
so a reader can decide before starting rather than halfway through.

**cuML needs a different environment.** ``cuml-cu12`` 26.8.0 declares
``requires_python >= 3.11`` and its recent wheels are cp311 ONLY (25.10, 25.12
and 26.2 shipped cp310-cp313; 26.4, 26.6 and 26.8 ship cp311 alone -- the
window is narrowing)::

    conda create -n spacr-gpu python=3.11
    conda activate spacr-gpu
    pip install spacr
    pip install cuml-cu12

Use a separate environment because cuML has narrower Python and CUDA wheel
compatibility than spaCR's core installation. Its coordinate-descent results
may also differ from another implementation at the same alpha; treat it as a
backend choice, not only a speed switch.

**numpyro and gpytorch install cleanly and answer a different question.**
numpyro samples a posterior with NUTS and gpytorch fits a Gaussian process;
neither produces the point estimates and standard errors statsmodels does, so
a reader installing them to go faster has misunderstood what they are for::

    pip install numpyro
    pip install --upgrade 'jax[cuda12]'
    pip install gpytorch
"""

from __future__ import annotations

import importlib.util
import os
import sys
from html import escape

from .regression_spec import (ALL_REGRESSION_TYPES, DEFAULT_REGRESSION_BACKEND,
                              REGRESSION_BACKENDS, REGRESSION_BACKEND_LABELS,
                              REGRESSION_BACKEND_ORDER, REGRESSION_TYPES)

__all__ = [
    "resolve_backend_name", "backend_label", "backend_supports",
    "backend_status", "backend_menu", "backend_choices",
    "describe_backends", "cuda_present_without_importing_torch",
    "package_installed", "INSTALL_RECIPES", "BACKEND_REQUIREMENTS",
    "install_recipe", "describe_install_recipes", "backend_install_offer",
    "availability_entry", "availability_entries",
]


def resolve_backend_name(value) -> str:
    """The canonical backend name for whatever a panel or a CSV posted.

    Accepts the canonical name (``'torch'``), the label the combo shows
    (``'torch (GPU)'``), any casing, and ``None``/``''`` for "not chosen".

    THE LABEL IS AN ACCEPTED SPELLING ON PURPOSE. The design requires
    every entry to read ``(CPU)`` or ``(GPU)``, and both GUIs render a combo's
    options verbatim -- so the option strings ARE the labels and the value
    posted back is a label. Normalising here is what keeps the settings CSV
    written with the short canonical name.

    :param value: what was posted.
    :returns: a key of :data:`REGRESSION_BACKENDS`.
    :raises ValueError: on a name no backend answers to. Naming the valid
        ones in the message, because a settings CSV with a typo in it is
        exactly when a bare "invalid" costs the most.
    """
    if value is None:
        return DEFAULT_REGRESSION_BACKEND
    text = str(value).strip()
    if not text:
        return DEFAULT_REGRESSION_BACKEND
    if text in REGRESSION_BACKENDS:
        return text
    if text in REGRESSION_BACKEND_LABELS:
        return REGRESSION_BACKEND_LABELS[text]
    lowered = text.lower()
    for name, spec in REGRESSION_BACKENDS.items():
        if lowered in (name.lower(), str(spec['label']).lower()):
            return name
    # 'statsmodels (cpu)' with the suffix mangled, 'lme4', 'rapids' -- the
    # spellings a person actually types.
    aliases = {'lme4': 'pymer4', 'rapids': 'cuml', 'pytorch': 'torch',
               'sm': 'statsmodels', 'default': 'statsmodels'}
    if lowered in aliases:
        return aliases[lowered]
    raise ValueError(
        f"regression_backend={value!r} is not a backend spaCR knows. Choose "
        f"one of: {', '.join(REGRESSION_BACKEND_ORDER)}. The default, "
        f"{DEFAULT_REGRESSION_BACKEND!r}, is what every existing result was "
        f"produced with.")


def backend_label(name) -> str:
    """The combo entry for a backend -- always suffixed ``(CPU)``/``(GPU)``."""
    return str(REGRESSION_BACKENDS[resolve_backend_name(name)]['label'])


def backend_supports(name, regression_type) -> bool:
    """Can ``name`` fit ``regression_type``?

    ``regression_type=None`` means "auto-selected from the response", which
    only the default backend can answer, since the choice is made after the
    data is read.
    """
    spec = REGRESSION_BACKENDS[resolve_backend_name(name)]
    types = spec['types']
    if types == ALL_REGRESSION_TYPES:
        return True
    if regression_type is None:
        return False
    return str(regression_type).lower() in types


def package_installed(name) -> bool:
    """Is a backend's package importable? Does NOT import it.

    ``find_spec`` on a package that is not there raises ``ModuleNotFoundError``
    for a missing PARENT, which is a different question from the one being
    asked, so it is caught.
    """
    if not name:
        return True
    try:
        return importlib.util.find_spec(str(name)) is not None
    except (ImportError, ValueError):
        return False


def cuda_present_without_importing_torch() -> bool:
    """Is there a CUDA device, decided without paying for ``import torch``?

    See the module docstring for why this is a driver probe rather than
    ``torch.cuda.is_available()``.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return any(os.path.exists(node) for node in
               ("/dev/nvidiactl", "/dev/nvidia0", "/proc/driver/nvidia/gpus"))


def backend_status(name, regression_type=None) -> dict:
    """Whether a backend is choosable right now, and the reason if not.

    :param name: backend name or label.
    :param regression_type: what the user is asking it to fit. ``None``
        skips the compatibility half of the question.
    :returns: ``{name, label, device, enabled, reason, short_reason, url,
        summary, cost, differs, pip}``. ``reason`` is ``''`` when enabled -- a
        disabled entry always carries the sentence that says why, and it is
        written to be shown ON the control, not logged.

    ``short_reason`` IS THE SAME REFUSAL AT COMBO-ENTRY LENGTH, and it exists
    because a reason nobody can read is a reason nobody has. A greyed-out
    row in a dropdown says only "not this one"; Qt shows an item tooltip
    lazily and only while the popup is open, so the sentence was reachable
    but missable. The short form goes IN the entry's own text and in the box
    under it, where it cannot be hovered past.
    """
    key = resolve_backend_name(name)
    spec = REGRESSION_BACKENDS[key]
    status = {
        'name': key,
        'label': spec['label'],
        'device': spec['device'],
        'url': spec['url'],
        'summary': spec['summary'],
        'cost': spec['cost'],
        'differs': spec['differs'],
        'pip': spec['pip'],
        'enabled': True,
        'reason': '',
        'short_reason': '',
    }

    def refuse(reason, short):
        status['enabled'] = False
        status['reason'] = reason
        status['short_reason'] = short
        return status

    # THE TYPE FIRST, because it is about the choice the user just made
    # rather than about the machine, and it is the one they can fix from the
    # same panel. "cuML has no mixed model" is instruction 141 C's own
    # example.
    if regression_type is not None and not backend_supports(key,
                                                            regression_type):
        types = spec['types']
        listed = ', '.join(types) if types != ALL_REGRESSION_TYPES else 'any'
        return refuse(
            f"{spec['label']} cannot fit regression_type="
            f"{regression_type!r}; it fits {listed}.",
            f"no {regression_type} model; fits {listed}")
    if regression_type is None and spec['types'] != ALL_REGRESSION_TYPES:
        # SAY WHICH TYPES, AND WHETHER IT IS INSTALLED. Reported 2026-08-21:
        # with the family left to be chosen from the response, every optional
        # backend read "unavailable: needs an explicit regression type" --
        # seven identical lines saying what was MISSING and nothing about
        # what any of them does or whether it is even on the machine.
        #
        # "write the explisit regression type and what needs to be done if it
        # is not installed. if it is intalled write installed."
        #
        # Both facts belong here because they are answered differently: the
        # types tell the user which choice would make this row selectable,
        # and the install state tells them whether making that choice would
        # be enough.
        listed = ', '.join(spec['types'])
        installed = package_installed(spec['package'])
        if not spec['implemented']:
            state = "not wired up yet"
            if not installed and spec['pip']:
                state += f", not installed -- {spec['pip']}"
        elif installed:
            state = "installed"
        else:
            state = (f"not installed -- {spec['pip']}" if spec['pip']
                     else "not installed")
        return refuse(
            f"{spec['label']} fits {listed}. Choose one of those as "
            f"regression_type to select it -- with the family left to be "
            f"decided from the response it is only known after the data is "
            f"read, and only {DEFAULT_REGRESSION_BACKEND} can fit whichever "
            f"it turns out to be. This backend is {state}.",
            f"fits {listed}; {state}")

    # NOT INSTALLED IS SAID ALONGSIDE NOT WIRED UP, not instead of it.
    #
    # The unimplemented test used to return first, so on a machine with none
    # of the six optional packages -- which is every machine, since they are
    # extras -- the pip command instruction 141 C asks for was never shown by
    # anything. Both facts are true of the same entry and both are what a
    # reader needs: installing the package alone would not make it choosable,
    # and neither would wiring it up alone.
    missing = not package_installed(spec['package'])
    if not spec['implemented']:
        reason = (
            f"{spec['label']} is described here but spaCR does not route any "
            f"fit through it yet, so choosing it would change nothing. "
            f"Listed rather than hidden so the plan is visible.")
        short = "not wired up yet"
        if missing and spec['pip']:
            reason += (f" Its package is not installed here either: "
                       f"{spec['pip']}")
            short = f"not wired up yet, not installed -- {spec['pip']}"
        return refuse(reason, short)

    if missing:
        return refuse(
            f"{spec['label']} needs a package that is not installed: "
            f"{spec['pip']}",
            f"not installed -- {spec['pip']}")

    if spec['device'] == 'gpu' and not cuda_present_without_importing_torch():
        return refuse(
            f"{spec['label']} needs a CUDA device and none was found. spaCR "
            f"will not quietly run it on the CPU instead -- a fit you asked "
            f"to run on the GPU and that silently did not is the slow run "
            f"you were avoiding, reported as the fast one.",
            "needs a CUDA device; none found")

    return status


def backend_menu(regression_type=None) -> list:
    """Every backend in panel order, each with its status.

    This is what a combo box is built from: the entries are all present, in
    one order, and the disabled ones carry their own reason.
    """
    return [backend_status(name, regression_type)
            for name in REGRESSION_BACKEND_ORDER]


def backend_choices() -> list:
    """The combo's option strings -- the labels, in panel order."""
    return [REGRESSION_BACKENDS[name]['label']
            for name in REGRESSION_BACKEND_ORDER]


def _first_sentence(text) -> str:
    """Return the opening sentence of a backend summary.

    Backend summaries place the identifying sentence first and qualifications
    afterward. Splitting at the first sentence therefore preserves a complete
    description without truncating by character count.
    """
    body = " ".join(str(text or "").split())
    head, sep, _rest = body.partition(". ")
    return head + "." if sep else body


def describe_backends(regression_type=None, html: bool = True,
                      selected=None, compact: bool = False) -> str:
    """The text the model box shows: every backend, briefly, with its link.

    The design: "the text box should describe all of the packages that
    are available and what they do, briefly, and link the API for each", and
    141 D: where a backend cannot agree with statsmodels by construction, the
    box says what differs.

    :param regression_type: greys the entries that cannot fit it, and their
        reason is included.
    :param html: emit ``<a href=...>`` links (the Qt backend box renders rich
        text). ``False`` gives plain text with the URL inline, for a log.
    :param selected: the backend the panel currently holds. In ``compact``
        mode it is the one written out in full.
    :param compact: ONE LINE PER BACKEND instead of one paragraph.

        THE SETTINGS PANEL IS ONE PAGE and the full text is
        3,101 characters -- about ninety wrapped lines in a settings field,
        which is not a description, it is a document that happens to be in a
        combo box's neighbourhood. Compact keeps every backend and every API
        link and drops what a reader does not need about the seven they did
        NOT pick: the measured cost of each, and the second and later
        sentences of each summary. The one they DID pick is written out in
        full, cost and caveat included, because that is the run they are
        about to start.
    :returns: one paragraph per backend, or one line per backend under
        ``compact`` with the selected one expanded above them.
    """
    entries = backend_menu(regression_type)
    if not compact:
        lines = []
        for status in entries:
            head = status['label']
            if html:
                head = (f"<b>{head}</b> "
                        f"<a href=\"{status['url']}\">API</a>")
            else:
                head = f"{head} -- {status['url']}"
            body = [status['summary'], status['cost']]
            if status['differs']:
                body.append("DIFFERENT ANSWER: " + status['differs'])
            if not status['enabled']:
                body.append("Unavailable: " + status['reason'])
            text = f"{head}: " + " ".join(body)
            lines.append(text)
        joiner = "<br><br>" if html else "\n\n"
        return joiner.join(lines)

    chosen = resolve_backend_name(selected)
    blocks = []
    for status in entries:
        if status['name'] != chosen:
            continue
        body = [status['summary'], status['cost']]
        if status['differs']:
            body.append("DIFFERENT ANSWER: " + status['differs'])
        if not status['enabled']:
            body.append("UNAVAILABLE: " + status['reason'])
        if html:
            blocks.append(
                f"<p><b>{escape(status['label'])}</b> &middot; "
                f"<a href=\"{escape(status['url'], quote=True)}\">API</a>"
                f"<br>{escape(' '.join(body))}</p>")
        else:
            blocks.append(f"{status['label']} -- {status['url']}\n"
                          + " ".join(body))

    others = []
    for status in entries:
        if status['name'] == chosen:
            continue
        tail = (_first_sentence(status['summary']) if status['enabled']
                else f"unavailable: {status['short_reason']}")
        if html:
            others.append(
                f"<b>{escape(status['label'])}</b> "
                f"<a href=\"{escape(status['url'], quote=True)}\">API</a> "
                f"&mdash; {escape(tail)}")
        else:
            others.append(f"{status['label']} -- {status['url']} -- {tail}")
    if html:
        blocks.append("<p><i>The other backends spaCR knows</i><br>"
                      + "<br>".join(others) + "</p>")
        return "".join(blocks)
    blocks.append("The other backends spaCR knows\n" + "\n".join(others))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# What would make an unavailable backend available, and can it be done HERE
# ---------------------------------------------------------------------------
#
# Instruction 158. `backend_status` already says WHY an entry is greyed out;
# this half says what would ungrey it, and whether that is honestly possible
# in this environment. The three answers are install-here, possible-elsewhere
# and not-possible, and the shapes are `spacr.updater.InstallOffer` so the
# Image UMAP's GPU acceleration (`spacr.gpu_reduce.install_offer`) answers in
# the same vocabulary and one shared panel serves both.

#: Installation recipes shared by the generated API reference and the GUI.
#: Keeping one source ensures both surfaces present the same requirements.
INSTALL_RECIPES = {
    'pymer4': (
        "pymer4 / lme4 NEEDS R. It is an interface to R's lme4, not a "
        "reimplementation of it, so no amount of pip can supply it on its "
        "own.\n"
        "Measured 2026-08-18: pymer4 0.9.2's wheel declares NO dependencies "
        "at all, so `pip install pymer4` truthfully reports one additive "
        "package -- and the install then fails at `import polars` in "
        "pymer4/io.py, and every module under pymer4/models/ opens with "
        "`from rpy2.robjects.packages import importr`.\n"
        "\n"
        "    1. an R installation (4.2 or newer):\n"
        "           conda install -c conda-forge r-base\n"
        "    2. lme4 and lmerTest inside R:\n"
        "           R -e 'install.packages(c(\"lme4\",\"lmerTest\"), "
        "repos=\"https://cloud.r-project.org\")'\n"
        "    3. the Python-to-R bridge:\n"
        "           pip install rpy2 polars\n"
        "    4. and only then:\n"
        "           pip install pymer4\n"
        "\n"
        "THIS IS A HEAVIER ASK THAN THE OTHER BACKENDS -- a second language "
        "runtime, its own package library and a compiled bridge between them "
        "-- which is said here so a reader can decide before starting rather "
        "than halfway through."),
    'cuml': (
        "cuML is RAPIDS' GPU ridge / lasso / elastic-net. It needs an "
        "interpreter its wheels are built for and a CUDA 12 device, and on "
        "the interpreter spaCR is usually run on it CANNOT be installed: "
        "measured 2026-08-18, cuml-cu12 26.8.0 declares requires_python "
        ">= 3.11 and the recent wheels are cp311 ONLY (25.10/25.12/26.2 "
        "shipped cp310-cp313; 26.4/26.6/26.8 ship cp311 alone -- the window "
        "is narrowing).\n"
        "\n"
        "    conda create -n spacr-gpu python=3.11\n"
        "    conda activate spacr-gpu\n"
        "    pip install spacr\n"
        "    pip install cuml-cu12\n"
        "\n"
        "WHAT IT COSTS, measured rather than guessed. On the default spaCR "
        "environment (Python 3.10, numpy 1.26.4) resolving it anyway moves "
        "numpy 1.26.4 -> 2.2.6 and downgrades numba and llvmlite. On a 3.12 "
        "environment numpy is untouched, but it still downgrades numba "
        "0.66 -> 0.64 and llvmlite 0.48 -> 0.46, pins the CUDA runtime back "
        "to 12 (cuda-bindings 13.3.1 -> 12.9.7, cuda-toolkit 13.0.3 -> "
        "12.9.2), and moves pandas 2.3.3 -> 3.0.3.\n"
        "\n"
        "AND IT CAN RETURN A DIFFERENT ANSWER, not merely a faster one: two "
        "coordinate-descent implementations of a penalised path, solved to "
        "different tolerances, can SELECT DIFFERENT VARIABLES at the same "
        "alpha. That is a different result and not a tolerance. cuML's UMAP "
        "is likewise not bit-identical to umap-learn's."),
    'numpyro': (
        "numpyro ANSWERS A DIFFERENT QUESTION. It samples a posterior with "
        "NUTS; it does not produce the point estimates and standard errors "
        "statsmodels does, so a reader installing it to go faster has "
        "misunderstood what it is for. Sampling is slower per fit and "
        "parallel across chains.\n"
        "\n"
        "    pip install numpyro\n"
        "    # for the GPU, install a CUDA build of jaxlib as well:\n"
        "    pip install --upgrade 'jax[cuda12]'\n"),
    'gpytorch': (
        "gpytorch ANSWERS A DIFFERENT QUESTION. It fits a Gaussian process: "
        "a posterior over functions with a predictive variance, not a "
        "coefficient table with p-values. Install it because you want that, "
        "not because you want the same answer sooner.\n"
        "\n"
        "    pip install gpytorch    # torch is already a spaCR dependency\n"),
}

#: Which pip requirement actually installs a backend. Separate from
#: ``REGRESSION_BACKENDS[...]['pip']``, which is PROSE meant for a combo entry
#: ("pip install pymer4  (plus R, rpy2, lme4)") and would be handed to the
#: resolver verbatim if it were reused here.
BACKEND_REQUIREMENTS = {
    'torch': 'torch',
    'pymer4': 'pymer4',
    'cuml': 'cuml-cu12',
    'pyfixest': 'pyfixest',
    'glum': 'glum',
    'numpyro': 'numpyro',
    'gpytorch': 'gpytorch',
}


def install_recipe(name) -> str:
    """The written recipe for making ``name`` available, or ``''``.

    :param name: a backend name or label.
    :returns: the entry in :data:`INSTALL_RECIPES`, or an empty string for a
        backend that needs no recipe because ``pip install`` is the whole
        story.
    """
    try:
        key = resolve_backend_name(name)
    except (ValueError, KeyError):
        return ""
    return INSTALL_RECIPES.get(key, "")


def describe_install_recipes(html: bool = False) -> str:
    """Every recipe in :data:`INSTALL_RECIPES`, in panel order.

    Packages that cannot be installed directly need recipes on the API page.
    This renders them in one block so a caller --
    the documentation build, a log, or the box under the backend combo -- can
    show all of them without knowing which backends have one.
    """
    blocks = []
    for name in REGRESSION_BACKEND_ORDER:
        recipe = INSTALL_RECIPES.get(name)
        if not recipe:
            continue
        label = str(REGRESSION_BACKENDS[name]['label'])
        if html:
            blocks.append(
                f"<p><b>{escape(label)}</b><br>"
                + escape(recipe).replace("\n", "<br>") + "</p>")
        else:
            blocks.append(f"{label}\n{'-' * len(label)}\n{recipe}")
    return ("".join(blocks) if html else "\n\n".join(blocks))


def _cuml_python_supported() -> bool:
    """Can ``cuml-cu12`` be installed into the running interpreter?

    Delegates to :data:`spacr.gpu_reduce.SUPPORTED_PYTHON` rather than
    repeating the version window, because the Image UMAP asks the same
    question about the same wheel and two copies of it would drift.
    """
    from .gpu_reduce import python_supported
    return bool(python_supported())


def backend_install_offer(name, regression_type=None):
    """What pressing **Install** on a greyed-out backend entry should do.

    THE ENVIRONMENT IS ASKED FIRST, before spaCR's own wiring, because the
    answer a user can act on is about their machine. "cuML needs Python 3.11
    and this is 3.10" is a thing they can go and do; "spaCR routes no fit
    through it yet" is a thing only spaCR can fix, and saying that instead
    would hide the part they could have acted on.

    :param name: a backend name or label.
    :param regression_type: what the panel currently asks to fit. Used only
        for the message, never to decide installability -- a family mismatch
        is fixed by choosing another family, not by installing anything.
    :returns: a :class:`spacr.updater.InstallOffer`.
    """
    from .updater import (offer_elsewhere, offer_impossible, offer_install,
                          offer_ready)

    key = resolve_backend_name(name)
    spec = REGRESSION_BACKENDS[key]
    label = str(spec['label'])
    recipe = INSTALL_RECIPES.get(key, "")

    if regression_type is not None and not backend_supports(key,
                                                            regression_type):
        types = spec['types']
        listed = ', '.join(types) if types != ALL_REGRESSION_TYPES else 'any'
        return offer_impossible(
            label,
            f"{label} is not greyed out because anything is missing -- it "
            f"cannot fit regression_type={regression_type!r} at all. It fits "
            f"{listed}. Nothing to install; change the regression type "
            f"instead.", recipe)

    if regression_type is None and spec['types'] != ALL_REGRESSION_TYPES:
        return offer_impossible(
            label,
            f"{label} is greyed out because the regression type is still "
            f"'auto'. With the family chosen from the response after the "
            f"data is read, only {DEFAULT_REGRESSION_BACKEND} can promise to "
            f"fit whichever one it turns out to be. Nothing to install; name "
            f"a regression type instead.", recipe)

    package = spec['package']
    if not package:
        return offer_ready(label, f"{label} is part of spaCR. Nothing to "
                                  f"install.")

    installed = package_installed(package)

    # 1. NOT POSSIBLE BY INSTALLING -- said first when it is true of the
    #    package itself rather than of this machine.
    if key == 'pymer4' and not installed:
        return offer_impossible(
            label,
            "pymer4 cannot be made available by installing a Python package: "
            "it is a bridge to R's lme4 and needs R itself. spaCR will not "
            "run a pip command that reports success and leaves you with an "
            "import error.", recipe)

    # 2. POSSIBLE, BUT NOT HERE.
    if key == 'cuml' and not installed and not _cuml_python_supported():
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return offer_elsewhere(
            label,
            f"cuML needs Python 3.11; this spaCR is on {version}. Installing "
            f"it here would either fail or succeed at breaking the install -- "
            f"resolving cuml-cu12 against this environment moves numpy and "
            f"the CUDA runtime torch is built against. Nothing has been run.",
            recipe)

    # 3. INSTALLABLE HERE.
    if not installed:
        requirement = BACKEND_REQUIREMENTS.get(key, package)
        message = (f"{label} needs {requirement}, which is not installed in "
                   f"this environment. What it would change is shown before "
                   f"anything is installed.")
        if not spec['implemented']:
            message += (" NOTE: spaCR routes no fit through this backend yet, "
                        "so installing the package will not make the entry "
                        "choosable -- it makes the package available to you.")
        return offer_install(label, message, requirement, recipe)

    # The package is here. Whatever is left is not an install problem.
    if spec['device'] == 'gpu' and not cuda_present_without_importing_torch():
        return offer_impossible(
            label,
            f"{label} is installed; what is missing is a CUDA device, and "
            f"installing more cannot supply one. Check the driver with "
            f"nvidia-smi.", recipe)
    if not spec['implemented']:
        return offer_impossible(
            label,
            f"{label} is installed, but spaCR routes no fit through it yet, "
            f"so choosing it would change nothing. Listed rather than hidden "
            f"so the plan is visible.", recipe)
    return offer_ready(label, f"{label} is available.")


def availability_entry(name, regression_type=None) -> dict:
    """One backend as the shared hover panel wants it.

    The panel (:mod:`spacr.qt.widgets.availability_panel`) takes a mapping so
    that neither of its two callers has to import Qt to build one, and so
    that this module keeps its promise not to import torch or PySide6.

    :returns: ``{key, title, reason, url, offer, enabled}``.
    """
    status = backend_status(name, regression_type)
    return {
        'key': status['name'],
        'title': str(status['label']),
        'reason': str(status['reason'] or status['summary']),
        'url': str(status['url']),
        'enabled': bool(status['enabled']),
        'offer': backend_install_offer(name, regression_type),
    }


def availability_entries(regression_type=None) -> list:
    """Every backend as a panel entry, in panel order."""
    return [availability_entry(name, regression_type)
            for name in REGRESSION_BACKEND_ORDER]
