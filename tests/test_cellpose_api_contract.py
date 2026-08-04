"""The installed Cellpose API is what the doubles say it is — enforced.

``tests/cellpose_api_contract.py`` writes down the signatures of the installed
``cellpose.models.CellposeModel``. Writing them down is only worth something if
something checks the copy against the original and checks that the suite's
doubles match, which is what this module does:

1. :func:`test_the_recorded_init_signature_is_the_installed_one` and its
   ``eval`` twin diff the constants against ``inspect.signature``. If cellpose
   is upgraded and a parameter moves, these fail first and name it.
2. :func:`test_cellpose_eval_returns_three_values` pins the return arity, read
   off the library's own source rather than its docstring — which still
   promises the Cellpose 3 four-tuple and is wrong.
3. :func:`test_every_converted_double_declares_the_installed_signature` sweeps
   ``tests/`` and holds every converted double to the real parameter list. It
   is the successor to ``CELLPOSE_MOCK_RATCHET`` in
   ``test_test_suite_hygiene.py``, which is now empty.
4. The ``xfail(strict=True)`` block at the bottom pins the call sites where
   spaCR hands Cellpose 4 an argument it accepts and silently discards. Each
   one names the product file, the argument and the fix.

Everything here is offline and CPU-only: no weights are loaded, and the only
thing imported from cellpose is its signature metadata.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.cellpose_api_contract import (
    CELLPOSE_EVAL_PARAMETERS,
    CELLPOSE_EVAL_RETURN_ARITY,
    CELLPOSE_INIT_PARAMETERS,
    DEPRECATED_EVAL_ARGUMENTS,
    DEPRECATED_INIT_ARGUMENTS,
    INSTALLED_CELLPOSE_VERSION,
    MISSING_CHANNEL_AXIS,
    SENTINEL_DEFAULTS,
    assert_declares_installed_eval_signature,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
    installed_eval_parameters,
    installed_init_parameters,
)
from tests.conftest import check_cellpose_eval_call

TESTS_DIR = Path(__file__).resolve().parent


# ===========================================================================
# 1. the constants are the installed library's
# ===========================================================================

def test_the_recorded_init_signature_is_the_installed_one():
    """``CELLPOSE_INIT_PARAMETERS`` == ``CellposeModel.__init__``.

    Names, order and defaults. If a cellpose upgrade renames or reorders one,
    this is the test that says so, and every double that copied the list is
    caught by the sweep below rather than silently accepting a stale argument.
    """
    installed = installed_init_parameters()
    assert installed == CELLPOSE_INIT_PARAMETERS, (
        f"cellpose {INSTALLED_CELLPOSE_VERSION} CellposeModel.__init__ has "
        f"moved: installed={installed}, recorded={CELLPOSE_INIT_PARAMETERS}"
    )
    assert list(installed) == list(CELLPOSE_INIT_PARAMETERS)


def test_the_recorded_eval_signature_is_the_installed_one():
    """``CELLPOSE_EVAL_PARAMETERS`` == ``CellposeModel.eval``, minus self/x."""
    installed = installed_eval_parameters()
    assert installed == CELLPOSE_EVAL_PARAMETERS, (
        f"cellpose {INSTALLED_CELLPOSE_VERSION} CellposeModel.eval has moved: "
        f"installed={installed}, recorded={CELLPOSE_EVAL_PARAMETERS}"
    )
    assert list(installed) == list(CELLPOSE_EVAL_PARAMETERS)


def test_cellpose_4_ships_no_Cellpose_wrapper_and_no_SizeModel():
    """The Cellpose 3 entry points are gone, not merely discouraged.

    ``models.Cellpose`` was the wrapper that ran a ``SizeModel`` to estimate
    diameters and then delegated to ``CellposeModel``. Cellpose 4 has one
    architecture and neither class exists, so any code still reaching for them
    raises ``AttributeError`` — which is what ``spacr.spacrops`` did on every
    real run while its mock happily provided a fake ``models.Cellpose``.
    """
    from cellpose import models

    assert hasattr(models, "CellposeModel")
    assert not hasattr(models, "Cellpose"), (
        "cellpose.models.Cellpose is back; the 3.x wrapper is not what spaCR "
        "targets and spacr/doctor.py flags its presence as a broken install"
    )
    assert not hasattr(models, "SizeModel")


def test_weights_are_chosen_by_pretrained_model_not_model_type():
    """``model_type=`` is accepted, warned about and dropped in v4.0.1+.

    This is the single most dangerous survivor of the Cellpose 3 API: it is
    spelled correctly, it binds, and it does nothing, so a caller that selects
    weights with it loads ``cpsam`` and is never told.
    """
    assert "model_type" in CELLPOSE_INIT_PARAMETERS       # still binds
    assert "model_type" in DEPRECATED_INIT_ARGUMENTS      # and still a no-op
    assert emulate_pretrained_model(model_type="cyto2") == "cpsam"
    assert emulate_pretrained_model("my_ckpt.pth", "cyto2") == "my_ckpt.pth"

    source = inspect.getsource(_installed_init())
    assert "model_type argument is not used in v4.0.1+" in source


@pytest.mark.parametrize("name", sorted(DEPRECATED_EVAL_ARGUMENTS))
def test_the_deprecated_eval_arguments_are_read_only_to_be_warned_about(name):
    """``channels`` and ``rescale`` bind, warn, and reach nothing else.

    Proved over the AST rather than the text: every load of the parameter
    inside ``eval`` must sit in the ``if <name> is not None:`` guard that
    raises the deprecation warning. A load anywhere else would mean cellpose
    started acting on it again, and the notes in
    ``spacr.model_compare.IGNORED_ARGUMENTS`` — and every xfail in this suite
    that cites them — would need revisiting.
    """
    assert name in CELLPOSE_EVAL_PARAMETERS, f"{name} is not an eval parameter"

    tree = ast.parse(textwrap.dedent(inspect.getsource(_installed_eval())))
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == name
    ]
    assert len(guards) == 1, (
        f"expected exactly one `if {name} is not None:` deprecation guard in "
        f"cellpose {INSTALLED_CELLPOSE_VERSION} eval, found {len(guards)}"
    )
    guarded = {id(n) for n in ast.walk(guards[0])}

    stray = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == name
        and isinstance(node.ctx, ast.Load) and id(node) not in guarded
    ]
    assert not stray, (
        f"cellpose {INSTALLED_CELLPOSE_VERSION} reads `{name}` outside its "
        f"deprecation guard (eval source lines {stray}) -- it is no longer a "
        f"no-op, so DEPRECATED_EVAL_ARGUMENTS and the xfail pins citing it "
        f"are wrong"
    )

    # ...and the guard really is a warning, quoting the message we recorded.
    warned = ast.dump(guards[0])
    assert "models_logger" in warned and "warning" in warned
    assert DEPRECATED_EVAL_ARGUMENTS[name].split(".")[0][:30] in \
        inspect.getsource(_installed_eval())


def _installed_init():
    from cellpose import models
    return models.CellposeModel.__init__


def _installed_eval():
    from cellpose import models
    return models.CellposeModel.eval


# ===========================================================================
# 2. the return arity
# ===========================================================================

def test_cellpose_eval_returns_three_values():
    """Both return paths of the installed ``eval`` yield a 3-tuple.

    Read off the source, because the docstring of this very method still says
    ``(masks, flows, styles, diams)`` — the Cellpose 3 shape. A double that
    trusted the docstring and returned four values would keep a four-value
    unpack green against a library that raises ``ValueError`` on one.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(_installed_eval())))
    arities = [
        len(node.value.elts)
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    ]
    assert arities, "no tuple return found in CellposeModel.eval"
    assert set(arities) == {CELLPOSE_EVAL_RETURN_ARITY}, (
        f"cellpose {INSTALLED_CELLPOSE_VERSION} CellposeModel.eval returns "
        f"{sorted(set(arities))} values, not {CELLPOSE_EVAL_RETURN_ARITY}"
    )
    assert "(masks, flows, styles, diams)" in (_installed_eval().__doc__ or ""), (
        "the stale four-value promise has been corrected upstream -- drop this "
        "assertion and the warning it justifies in tests/cellpose_api_contract.py"
    )


# ===========================================================================
# 3. every double matches
# ===========================================================================

#: Cellpose doubles that still take ``**kwargs`` and declare only part of the
#: signature. They satisfy the older ``channel_axis`` contract (they name it,
#: default it to a sentinel and read it), so they are not blind — but they can
#: still absorb an argument cellpose 4 removed.
#:
#: Entries come off this list, never on. Converting one means writing out the
#: parameter list from :data:`CELLPOSE_EVAL_PARAMETERS`, exactly as the
#: fourteen doubles above it were.
PARTIAL_SIGNATURE_RATCHET = {
    ("test_cellpose4_model_story.py", "_M"): 3,
    ("test_cellpose4_spacrops_submodules.py", "_FakeCellposeModel"): 1,
    ("test_cellpose_channel_axis_contract.py", "_AxisRecordingModel"): 1,
    ("test_cellpose_channel_axis_contract.py", "_ThreeTupleModel"): 1,
    ("test_cov_object_organelle_sam.py", "_FakeCellposeModel"): 1,
    ("test_coverage_fill_cellpose_gpu_funcs.py", "_FakeModel"): 1,
    ("test_coverage_fill_cellpose_gpu_funcs.py", "_BadEvalModel"): 1,
    ("test_coverage_fill_pipeline_v2.py", "_FakeModel"): 1,
    ("test_coverage_fill_pipeline_v2.py", "_CaptureModel"): 1,
    ("test_coverage_fill_pipeline_v2.py", "_ListModel"): 1,
    ("test_zstack.py", "_M"): 1,
}

#: Total partial ``eval`` methods above, not keys.
PARTIAL_SIGNATURE_CEILING = 13


def _eval_doubles():
    """``(relpath, class, lineno, ast.FunctionDef)`` per Cellpose ``eval`` double.

    A double is any method named ``eval`` that takes an argument besides
    ``self``. Nothing else in this suite has that shape: ``torch.nn.Module``
    stand-ins spell it ``def eval(self)``, and ``ModelConfig.eval_kwargs`` is a
    different name. Matching on shape rather than on the class's name is
    deliberate — ``_M``, ``_FakeCP`` and ``_RecordingCP`` are CellposeModel
    stand-ins and nothing about their names says so.
    """
    out = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):      # pragma: no cover
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                if fn.name != "eval":
                    continue
                args = fn.args
                named = [a.arg for a in args.posonlyargs + args.args
                         + args.kwonlyargs if a.arg != "self"]
                if not named:
                    continue      # torch's eval(self), not cellpose's
                out.append((str(path.relative_to(TESTS_DIR)), cls.name,
                            fn.lineno, fn))
    return out


def _declares_full_signature(fn):
    """``fn`` declares the installed eval parameter list and no ``**kwargs``."""
    args = fn.args
    if args.kwarg is not None or args.vararg is not None:
        return False
    named = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs
             if a.arg != "self"]
    return named[1:] == list(CELLPOSE_EVAL_PARAMETERS)


def test_the_double_sweep_finds_the_doubles_it_is_supposed_to():
    """Guard the guard: the sweep must actually be looking at something."""
    doubles = _eval_doubles()
    assert len(doubles) >= 20, (
        f"only {len(doubles)} Cellpose eval doubles found; the candidate "
        f"filter has stopped matching"
    )
    names = {cls for _, cls, _, _ in doubles}
    assert {"_M", "_FakeCP", "_RecordingCP"} <= names, (
        "the name-invisible doubles are not being found, so the filter has "
        "gone back to matching on class names"
    )


def test_every_converted_double_declares_the_installed_signature():
    """No double may hold a parameter list the installed cellpose does not.

    A double outside :data:`PARTIAL_SIGNATURE_RATCHET` has been converted, and
    conversion means the whole signature: same names, same order, same
    defaults, no ``**kwargs``. Anything else and the double is once again able
    to accept an argument cellpose 4 would reject.
    """
    wrong = []
    for rel, cls, lineno, fn in _eval_doubles():
        if PARTIAL_SIGNATURE_RATCHET.get((rel, cls)):
            continue
        if not _declares_full_signature(fn):
            wrong.append(f"{rel}:{lineno} {cls}.eval")
    assert not wrong, (
        "these Cellpose doubles do not declare the installed cellpose "
        f"{INSTALLED_CELLPOSE_VERSION} eval signature:\n  "
        + "\n  ".join(wrong)
        + "\n\nWrite the parameters out with their real defaults and drop "
          "**kwargs -- see CELLPOSE_EVAL_PARAMETERS in "
          "tests/cellpose_api_contract.py. A double that accepts everything "
          "cannot fail when spaCR passes an argument cellpose removed."
    )


def test_the_partial_signature_ratchet_only_shrinks():
    total = sum(PARTIAL_SIGNATURE_RATCHET.values())
    assert total <= PARTIAL_SIGNATURE_CEILING, (
        f"PARTIAL_SIGNATURE_RATCHET grew to {total} eval methods (ceiling "
        f"{PARTIAL_SIGNATURE_CEILING}). Entries come off this list, never on."
    )
    missing = sorted({f for f, _ in PARTIAL_SIGNATURE_RATCHET
                      if not (TESTS_DIR / f).is_file()})
    assert not missing, (
        f"PARTIAL_SIGNATURE_RATCHET names modules that no longer exist: "
        f"{missing}"
    )
    # And the fourteen that were converted stay converted.
    converted = [(rel, cls) for rel, cls, _, fn in _eval_doubles()
                 if _declares_full_signature(fn)]
    assert len(converted) >= 14, (
        f"only {len(converted)} doubles still declare the full signature; "
        f"fourteen were converted and none may regress"
    )


def test_the_signature_checker_rejects_the_shapes_it_must():
    """The checker's own teeth, against synthesised doubles.

    Written as source and compiled here rather than as suite-level classes: a
    deliberately-wrong CellposeModel double defined at module scope is a mock
    another test could pick up by accident.
    """
    real = ", ".join(
        f"{name}={SENTINEL_DEFAULTS[name]!r}" if name in SENTINEL_DEFAULTS
        else f"{name}={default!r}"
        for name, default in CELLPOSE_EVAL_PARAMETERS.items()
    )

    def compile_eval(params):
        namespace = {"MISSING_CHANNEL_AXIS": MISSING_CHANNEL_AXIS}
        exec(f"def eval(self, x, {params}):\n    return None, None, None",
             namespace)
        return namespace["eval"]

    # The compliant shape passes.
    good = compile_eval(real.replace(repr(MISSING_CHANNEL_AXIS),
                                     "MISSING_CHANNEL_AXIS"))
    assert_declares_installed_eval_signature(good, where="synthetic")

    # **kwargs is rejected outright, whatever else it declares.
    with pytest.raises(AssertionError, match=r"still takes \*\*"):
        assert_declares_installed_eval_signature(
            compile_eval(real.replace(repr(MISSING_CHANNEL_AXIS),
                                      "MISSING_CHANNEL_AXIS") + ", **kwargs"),
            where="synthetic")

    # A missing parameter is named.
    with pytest.raises(AssertionError, match="missing"):
        assert_declares_installed_eval_signature(
            compile_eval("diameter=None"), where="synthetic")

    # So is one cellpose does not have -- e.g. cellpose 3's `interp`.
    trailing = real.replace(repr(MISSING_CHANNEL_AXIS),
                            "MISSING_CHANNEL_AXIS") + ", interp=True"
    with pytest.raises(AssertionError, match=r"unknown: \['interp'\]"):
        assert_declares_installed_eval_signature(compile_eval(trailing),
                                                 where="synthetic")


def test_the_argument_recorders_separate_configured_from_defaulted():
    """``configured_eval_arguments`` is what "spaCR passed this" means now.

    The doubles bind every parameter, so ``'channels' in record`` is always
    true and can no longer express "spaCR did not set it". The configured view
    restores that question, and answers it the way cellpose does: a parameter
    left at the library default reaches nothing.
    """
    def probe(**kwargs):
        defaults = dict(CELLPOSE_EVAL_PARAMETERS)
        defaults.update(kwargs)
        bound = {"self": object(), "x": np.zeros((4, 4)), **defaults}
        return eval_arguments(bound), configured_eval_arguments(bound)

    full, configured = probe()
    assert set(full) == set(CELLPOSE_EVAL_PARAMETERS)
    assert configured == {}, "nothing set means nothing configured"

    full, configured = probe(channels=[0, 0], diameter=30)
    assert configured == {"channels": [0, 0], "diameter": 30}
    assert full["channels"] == [0, 0]
    # Passing the library's own default is the same call as passing nothing.
    _, configured = probe(channels=None, flow_threshold=0.4)
    assert configured == {}
    # An ndarray value must not raise on the equality check.
    _, configured = probe(diameter=np.array([30.0, 40.0]))
    assert "diameter" in configured

    assert init_arguments({"self": object(), "gpu": True}) == {"gpu": True}


# ===========================================================================
# 4. the bugs the tightening exposed
# ===========================================================================

class _ContractModel:
    """A ``CellposeModel`` double built strictly from the installed signature.

    Used by the pins below, which need a double whose *behaviour* matches
    cellpose 4's — deprecated arguments accepted and then thrown away — rather
    than one that merely accepts them.
    """

    last = None

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True):
        self.init_kwargs = init_arguments(locals())
        self.loaded_model = emulate_pretrained_model(pretrained_model,
                                                     model_type)
        self.calls = []
        self.configured = []
        type(self).last = self

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
        check_cellpose_eval_call(x, channel_axis, z_axis=z_axis, do_3D=do_3D,
                                 stitch_threshold=stitch_threshold,
                                 require_channel_axis=False)
        bound = locals()
        self.calls.append(eval_arguments(bound))
        self.configured.append(configured_eval_arguments(bound))
        images = x if isinstance(x, list) else [x]
        masks = [np.zeros(np.asarray(i).shape[:2], np.uint16) for i in images]
        for mask in masks:
            mask[1:4, 1:4] = 1
        if not isinstance(x, list):
            return masks[0], [np.zeros(masks[0].shape + (3,), np.float32),
                              None, None], None
        return masks, [[np.zeros(m.shape + (3,), np.float32), None, None]
                       for m in masks], None


@pytest.fixture
def contract_cellpose(monkeypatch):
    """Install :class:`_ContractModel` as ``cellpose.models.CellposeModel``."""
    from cellpose import models as cp_models
    _ContractModel.last = None
    monkeypatch.setattr(cp_models, "CellposeModel", _ContractModel)
    return _ContractModel


@pytest.mark.xfail(strict=True, reason=(
    "spacr/qt/widgets/timelapse_preview.py:336 selects a non-cpsam model with "
    "CellposeModel(model_type=model_name). cellpose 4.0.7 logs 'model_type "
    "argument is not used in v4.0.1+. Ignoring this argument...' and drops it, "
    "leaving pretrained_model at its 'cpsam' default, so the Timelapse "
    "preview silently segments with cpsam whatever the user picked. This is "
    "the exact twin of spacr/qt/widgets/live_preview.py:637. Fix: resolve the "
    "name through spacr.utils._resolve_cellpose_pretrained and pass it as "
    "pretrained_model=, dropping model_type= entirely."))
def test_timelapse_preview_model_selection_reaches_the_weights(
        contract_cellpose, monkeypatch):
    """Picking a model in the Timelapse preview must change the checkpoint."""
    from spacr.qt.widgets import timelapse_preview as TP

    TP.segment_frame(np.zeros((16, 16), np.float32),
                     {"model": "cyto2", "channel": 0, "normalise": False})

    model = contract_cellpose.last
    assert model.loaded_model != "cpsam", (
        "cellpose 4 discarded the requested model and ran cpsam instead"
    )


def test_the_preview_panels_do_pass_the_arguments_cellpose_still_reads():
    """Not everything the previews send is dead — diameter genuinely is not.

    Stated here so the two ``model_type`` pins above are not read as "the
    preview path is broken": its eval call is correct, and ``diameter`` is
    still the one size argument cellpose 4 acts on (it rescales by
    ``30 / diameter``).
    """
    for name in ("diameter", "flow_threshold", "cellprob_threshold"):
        assert name in CELLPOSE_EVAL_PARAMETERS
        assert name not in DEPRECATED_EVAL_ARGUMENTS
    source = inspect.getsource(_installed_eval())
    assert "image_scaling = 30. / diameter" in source
