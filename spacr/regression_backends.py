"""Which backend can fit which model HERE, and why the rest are greyed out.

:mod:`spacr.regression_spec` says what the backends ARE -- pure data, no
imports. This module is the only place that asks the environment about them:
is the package installed, is there a CUDA device, can this backend fit the
``regression_type`` the user picked. Instruction 141 C wants each answer
written ON the disabled entry, which is instruction 106's rule -- an
inapplicable control is disabled WITH ITS REASON, never absent and never
silently substituted.

IT MUST NOT IMPORT TORCH. Deciding whether to grey out the GPU entry happens
while a settings panel is being built, on the GUI thread, and
``tests/test_a_settings_panel_does_not_import_torch.py`` exists because that
kind of lookup once cost 2.2 seconds and 900 MB. So the CUDA question is
answered by stat()-ing the driver's device nodes unless torch is ALREADY
loaded, in which case asking it is free and exact. The probe can only be
wrong in the permissive direction -- driver present, no usable device -- and
:func:`spacr.mixed_gpu.resolve_device` corrects that at fit time by refusing
with the real reason. A panel that offers an entry which later refuses is
recoverable; a panel that greys out a GPU which works is not, because the
user has no way to argue with it.
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
    "package_installed",
]


def resolve_backend_name(value) -> str:
    """The canonical backend name for whatever a panel or a CSV posted.

    Accepts the canonical name (``'torch'``), the label the combo shows
    (``'torch (GPU)'``), any casing, and ``None``/``''`` for "not chosen".

    THE LABEL IS AN ACCEPTED SPELLING ON PURPOSE. Instruction 141 C requires
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
        return refuse(
            f"{spec['label']} needs an explicit regression_type. With "
            f"regression_type left to be chosen from the response, the "
            f"family is only known after the data is read, and only "
            f"{DEFAULT_REGRESSION_BACKEND} can fit whichever one it turns "
            f"out to be.",
            "needs an explicit regression type")

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
    """The opening sentence of a summary, for the one-line entries.

    "briefly" is the maintainer's own word for what the box should say about
    each package. Every ``summary`` in :data:`REGRESSION_BACKENDS` opens with
    the sentence that answers "what is this", and the ones that follow
    qualify it, so the split is where the brief version already is rather
    than a truncation at a character count.
    """
    body = " ".join(str(text or "").split())
    head, sep, _rest = body.partition(". ")
    return head + "." if sep else body


def describe_backends(regression_type=None, html: bool = True,
                      selected=None, compact: bool = False) -> str:
    """The text the model box shows: every backend, briefly, with its link.

    Instruction 141 B: "the text box should describe all of the packages that
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

        THE SETTINGS PANEL IS ONE PAGE (instruction 135) and the full text is
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
