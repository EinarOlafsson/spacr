"""The installed Cellpose API, written down, so a mock cannot drift from it.

Companion to the ``check_cellpose_eval_call`` contract in ``tests/conftest.py``.
That one polices *one* argument (``channel_axis``) by running the real
``transforms.convert_image``. This module polices the *whole signature*.

Why a second layer is needed
----------------------------
A double spelled ``def eval(self, x, **kwargs)`` accepts every keyword that has
ever existed. It therefore cannot fail when spaCR passes ``interp=``,
``tile=``, ``net_avg=`` or ``model_loaded=`` — all of which Cellpose 4 removed
outright — nor when spaCR stops passing something Cellpose now requires. Fifteen
green tests is exactly what a ``**kwargs`` double produces against a library
that has moved underneath it, which is how ``channel_axis=3`` shipped.

The cure is for every double to declare the REAL parameter list, verbatim, with
the REAL defaults and no ``**kwargs``. Then Python's own argument binding is the
test: an argument Cellpose 4 does not accept raises ``TypeError`` at the call
site, naming it.

Repeating a 24-parameter signature across a dozen files is only safe if
something checks the copies against the original. That is
:func:`assert_declares_installed_eval_signature`, and
``tests/test_cellpose_api_contract.py`` runs it over every double in the suite
plus over the constants below.

The one sanctioned deviation
----------------------------
Doubles default ``channel_axis`` to ``MISSING_CHANNEL_AXIS`` rather than to the
library's ``None``. ``None`` is a *legal* value meaning "auto-detect", so a
double that defaults to it cannot distinguish "the caller passed nothing" from
"the caller passed None" — the exact hole ``MISSING_CHANNEL_AXIS`` exists to
close. :data:`SENTINEL_DEFAULTS` records the substitution so the signature
check can allow it and nothing else.

Deprecated-but-accepted arguments
---------------------------------
Cellpose 4 kept several Cellpose 3 parameters in the signature, logs a warning
and then ignores them. Binding cannot catch those — they are spelled correctly,
they simply do nothing. :data:`DEPRECATED_EVAL_ARGUMENTS` and
:data:`DEPRECATED_INIT_ARGUMENTS` name them so a double can emulate the real
library's *behaviour* (drop the value) instead of its *signature* alone, which
is what turns "spaCR still passes ``channels=[0, 0]``" from invisible into a
failing test.
"""
from __future__ import annotations

import inspect

import cellpose
from cellpose import models as _cp_models

from tests.conftest import MISSING_CHANNEL_AXIS

#: Version of the cellpose the suite is running against. Every constant in this
#: module was read off this install with ``inspect.signature``.
INSTALLED_CELLPOSE_VERSION = cellpose.version

#: ``cellpose.models.CellposeModel.__init__`` parameters, name -> default.
#:
#: Cellpose 3's ``models.Cellpose`` wrapper and ``models.SizeModel`` do not
#: exist in 4.x at all; ``CellposeModel`` is the only entry point, and
#: ``pretrained_model`` (not ``model_type``) is how you choose weights.
CELLPOSE_INIT_PARAMETERS = {
    "gpu": False,
    "pretrained_model": "cpsam",
    "model_type": None,
    "diam_mean": None,
    "device": None,
    "nchan": None,
    "use_bfloat16": True,
}

#: ``cellpose.models.CellposeModel.eval`` parameters, name -> default, in
#: declaration order. ``x`` is positional and has no default.
#:
#: Read straight off the installed 4.0.7. NOTE that the ``CellposeModel`` class
#: docstring in that same file still advertises the Cellpose 3 signature
#: (``interp=True``, no ``max_size_fraction``, no ``flow3D_smooth``) — the
#: docstring is stale and this dict follows ``inspect.signature`` instead.
CELLPOSE_EVAL_PARAMETERS = {
    "batch_size": 8,
    "resample": True,
    "channels": None,
    "channel_axis": None,
    "z_axis": None,
    "normalize": True,
    "invert": False,
    "rescale": None,
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "do_3D": False,
    "anisotropy": None,
    "flow3D_smooth": 0,
    "stitch_threshold": 0.0,
    "min_size": 15,
    "max_size_fraction": 0.4,
    "niter": None,
    "augment": False,
    "tile_overlap": 0.1,
    "bsize": 256,
    "compute_masks": True,
    "progress": None,
}

#: Defaults a compliant double is allowed to substitute, ``name -> default``.
#: See "The one sanctioned deviation" above.
SENTINEL_DEFAULTS = {"channel_axis": MISSING_CHANNEL_AXIS}

#: How many values ``CellposeModel.eval`` returns on the installed cellpose.
#:
#: THREE — ``(masks, flows, styles)`` — on both of its return paths: the
#: list/5-D recursion returns ``masks, flows, styles`` and the leaf returns
#: ``masks, [dx_to_circ(dP), dP, cellprob], styles``. Cellpose 3 returned a
#: fourth value, ``diams``; its ``eval`` docstring in 4.0.7 STILL promises
#: ``(masks, flows, styles, diams)`` and is wrong. A double that returns four
#: values is emulating Cellpose 3 and will hide a ``a, b, c = model.eval(...)``
#: unpack that is in fact correct — or bless one that is not.
CELLPOSE_EVAL_RETURN_ARITY = 3

#: ``eval`` keywords Cellpose 4 accepts, warns about, and then does not act on.
#: Values are the warning the real library logs.
DEPRECATED_EVAL_ARGUMENTS = {
    "channels": ("channels deprecated in v4.0.1+. If data contain more than 3 "
                 "channels, only the first 3 channels will be used"),
    "rescale": "rescaling deprecated in v4.0.1+",
}

#: ``__init__`` keywords Cellpose 4 accepts, warns about, and then does not act
#: on. ``model_type`` is the dangerous one: it is how Cellpose 3 chose weights,
#: so a call site that still passes it silently runs ``cpsam`` instead of the
#: model the user asked for.
DEPRECATED_INIT_ARGUMENTS = {
    "model_type": "model_type argument is not used in v4.0.1+. Ignoring this argument...",
    "diam_mean": "diam_mean argument are not used in v4.0.1+. Ignoring this argument...",
    "nchan": "nchan argument is deprecated in v4.0.1+. Ignoring this argument",
}

#: ``eval`` keywords that are in the signature, are not warned about, and are
#: still never read. ``progress`` is forwarded through the list recursion and
#: consumed by nothing; Cellpose 3 drove a QProgressBar with it.
INERT_EVAL_ARGUMENTS = frozenset({"progress"})


def installed_eval_parameters():
    """``{name: default}`` for the installed ``CellposeModel.eval``, minus self/x."""
    return _parameters_of(_cp_models.CellposeModel.eval, skip=("self", "x"))


def installed_init_parameters():
    """``{name: default}`` for the installed ``CellposeModel.__init__``."""
    return _parameters_of(_cp_models.CellposeModel.__init__, skip=("self",))


def _parameters_of(func, skip=()):
    """``{name: default}`` in declaration order, excluding ``skip``."""
    return {
        name: param.default
        for name, param in inspect.signature(func).parameters.items()
        if name not in skip
    }


def assert_declares_installed_eval_signature(func, *, where=""):
    """Fail unless ``func`` is a faithful stand-in for ``CellposeModel.eval``.

    Checks, in order:

    * no ``**kwargs`` — that is the whole point; a double that keeps it can
      still swallow an argument the real library would reject;
    * no ``*args`` — same reason for positional arguments;
    * exactly the installed parameter names, in the installed order, after the
      leading image argument (whatever the double calls it);
    * the installed defaults, except where :data:`SENTINEL_DEFAULTS` sanctions
      a substitution.

    :param func: the double's ``eval`` (an unbound function or a bound method).
    :param where: human-readable location, used only in the failure message.
    :raises AssertionError: with the precise divergence.
    """
    label = where or getattr(func, "__qualname__", repr(func))
    sig = inspect.signature(func)
    params = [p for name, p in sig.parameters.items() if name != "self"]

    var_kw = [p.name for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
    assert not var_kw, (
        f"{label}: eval() still takes **{var_kw[0]}. A double with **kwargs "
        f"accepts every argument Cellpose has ever had, including the ones "
        f"{INSTALLED_CELLPOSE_VERSION} removed, so it cannot fail when spaCR "
        f"passes one. Declare the real parameters instead — see "
        f"CELLPOSE_EVAL_PARAMETERS in tests/cellpose_api_contract.py."
    )
    var_pos = [p.name for p in params if p.kind is inspect.Parameter.VAR_POSITIONAL]
    assert not var_pos, (
        f"{label}: eval() still takes *{var_pos[0]}, which accepts any number "
        f"of positional arguments. Declare the real parameters instead."
    )

    assert params, f"{label}: eval() takes no image argument at all."
    declared = {p.name: p.default for p in params[1:]}
    expected = dict(CELLPOSE_EVAL_PARAMETERS)

    missing = [n for n in expected if n not in declared]
    extra = [n for n in declared if n not in expected]
    assert not missing and not extra, (
        f"{label}: eval() does not declare the installed cellpose "
        f"{INSTALLED_CELLPOSE_VERSION} signature.\n"
        f"  missing: {missing}\n"
        f"  unknown: {extra}\n"
        f"A missing parameter means a call passing it raises TypeError even "
        f"though Cellpose accepts it; an unknown one means the double accepts "
        f"an argument Cellpose would reject."
    )
    assert list(declared) == list(expected), (
        f"{label}: eval() declares the right parameters in the wrong order.\n"
        f"  declared: {list(declared)}\n"
        f"  installed: {list(expected)}\n"
        f"Order matters: spaCR calls eval() positionally in places, so a "
        f"reordered double binds different values than the real library."
    )

    wrong = {
        name: (declared[name], SENTINEL_DEFAULTS.get(name, expected[name]))
        for name in expected
        if declared[name] is not SENTINEL_DEFAULTS.get(name, object())
        and declared[name] != expected[name]
    }
    assert not wrong, (
        f"{label}: eval() declares defaults the installed cellpose does not "
        f"use — {{name: (double, cellpose)}} = {wrong}. A wrong default makes "
        f"'spaCR omitted this' behave differently in the test than in a run."
    )


def eval_arguments(local_vars, image_parameter="x"):
    """``{name: value}`` for every declared ``eval`` parameter.

    Call as ``eval_arguments(locals())`` on the FIRST line of a double's
    ``eval``: at that point ``locals()`` holds exactly the bound parameters and
    nothing else. This replaces the ``**kwargs`` dict the doubles used to
    record, with one behavioural difference that is the whole point — every
    parameter is present, holding either what the caller passed or cellpose's
    own default, so ``record['diameter']`` no longer raises ``KeyError``
    depending on the call.

    Use :func:`configured_eval_arguments` where the question is "did spaCR set
    this at all" rather than "what value did Cellpose see".

    :param local_vars: the result of ``locals()``.
    :param image_parameter: name of the leading image argument, excluded.
    :returns: dict in declaration order.
    """
    return {k: v for k, v in local_vars.items()
            if k != "self" and k != image_parameter}


def configured_eval_arguments(local_vars, image_parameter="x"):
    """The ``eval`` parameters spaCR actually configured, ``{name: value}``.

    A parameter left at cellpose's own default is dropped: passing
    ``channels=None`` and passing nothing are the same call as far as the
    library is concerned (``if channels is not None:`` is the only place it is
    read), so treating them alike is accurate rather than lax. This is the
    dict to make ``'channels' not in ...`` style assertions against.

    :param local_vars: the result of ``locals()``.
    :param image_parameter: name of the leading image argument, excluded.
    :returns: dict holding only the parameters set to a non-default value.
    """
    out = {}
    for name, value in eval_arguments(local_vars, image_parameter).items():
        default = CELLPOSE_EVAL_PARAMETERS.get(name, _UNSET)
        if default is _UNSET or not _same_as_default(value, default):
            out[name] = value
    return out


def init_arguments(local_vars):
    """``{name: value}`` for every declared ``__init__`` parameter.

    Call as ``init_arguments(locals())`` on the first line of a double's
    ``__init__``. See :func:`eval_arguments`.
    """
    return {k: v for k, v in local_vars.items() if k != "self"}


class _Unset:
    """Sentinel for 'cellpose has no such parameter'."""

    __slots__ = ()

    def __repr__(self):  # pragma: no cover - debugging aid only
        return "<unset>"


_UNSET = _Unset()


def _same_as_default(value, default):
    """``value`` is cellpose's default, without tripping over numpy arrays.

    ``==`` on an ndarray returns an array, so a plain ``value == default`` in a
    boolean context raises. Identity first, then a guarded equality.
    """
    if value is default:
        return True
    try:
        return bool(value == default)
    except Exception:      # arrays, and anything else with an exotic __eq__
        return False


def emulate_deprecated_eval_arguments(**kwargs):
    """Return the ``eval`` kwargs Cellpose 4 will actually act on.

    Mirrors the real library: the deprecated names are accepted, logged and
    then dropped. Handing a double's recorded call through this makes the
    difference between "spaCR configured Cellpose" and "spaCR configured
    nothing" visible to an assertion.

    :param kwargs: the keywords the caller passed to ``eval``.
    :returns: ``(honoured, dropped)`` — two dicts.
    """
    dropped = {k: v for k, v in kwargs.items()
               if k in DEPRECATED_EVAL_ARGUMENTS or k in INERT_EVAL_ARGUMENTS}
    honoured = {k: v for k, v in kwargs.items() if k not in dropped}
    return honoured, dropped


def emulate_pretrained_model(pretrained_model="cpsam", model_type=None):
    """What weights Cellpose 4 will actually load, given both arguments.

    ``model_type`` is warned about and thrown away, so the answer is always
    ``pretrained_model`` — which defaults to ``cpsam``. A caller that selects
    its model through ``model_type=`` gets ``cpsam`` no matter what it asked
    for, silently, and this function is how a double reproduces that.

    :param pretrained_model: the ``pretrained_model=`` the caller passed.
    :param model_type: the ``model_type=`` the caller passed, if any.
    :returns: the checkpoint the real library would load.
    """
    return pretrained_model
