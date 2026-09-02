"""measure.py: the scikit-image capability probe, and one dead guard.

`expand_labels` grew a `spacing` argument after scikit-image 0.22, and
setup.py's floor is >=0.22.0 -- so spaCR PROBES for it rather than
assuming. The comment says why in the sharpest possible terms: without
the argument a 3-D run silently measures an unscaled radius, measured
wrong by 2.000x on a (2.0, 0.2, 0.2) voxel.

A silently wrong measurement is the worst failure this package can have,
so both halves of that probe are worth holding: the answer it gives on a
normal install, and what it does when the signature cannot be read at
all.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import pytest


@pytest.fixture(autouse=True)
def _only_one_spacr_measure_survives_this_file():
    """Put the ORIGINAL ``spacr.measure`` back in ``sys.modules`` afterwards.

    THIS FILE POISONED EVERY LATER TEST IN THE PROCESS, and the mechanism is
    the nastiest kind: nothing fails where the mistake is.

    `_reimport_measure` deletes `spacr.measure` from `sys.modules` so the
    module-level capability probe runs again. The next `import spacr.measure`
    then builds a SECOND, DIFFERENT module object -- and anything still
    holding the first one is now talking to a module the pipeline does not
    use.

    `tests/test_measure_hooks.py` is exactly that: it registers hooks through
    its own reference and then runs a measurement, which imports the other
    copy and finds no hooks registered at all. Ten tests failed in company and
    passed alone (instruction 346), and the bisect that found it took 112
    files down to this one.

    Restoring the original objects is enough, and is better than forbidding
    the reload: the reload is the point of this file, and the probe genuinely
    has to run twice to be tested.
    """
    # THE SAME PREDICATE `_reimport_measure` DELETES BY, and getting this
    # wrong is how the first attempt at this fixture failed. That function
    # says `startswith("spacr.measure")` -- no dot -- so it also removes
    # `spacr.measure_hooks`, WHICH IS WHERE THE HOOK REGISTRY LIVES. A
    # restore that matched only `spacr.measure` and `spacr.measure.*` put the
    # module back and left the registry module rebuilt, which is exactly the
    # state that made `tests/test_measure_hooks.py` register its hooks into a
    # copy nobody reads.
    def _ours(name):
        return name.startswith("spacr.measure")

    saved = {name: module for name, module in sys.modules.items()
             if _ours(name)}
    try:
        yield
    finally:
        for name in [n for n in list(sys.modules) if _ours(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


def _reimport_measure():
    for name in [n for n in list(sys.modules) if n.startswith("spacr.measure")]:
        del sys.modules[name]
    return importlib.import_module("spacr.measure")


class TestTheExpandLabelsProbe:

    def test_the_probe_matches_the_installed_signature(self):
        """Whatever it decides must agree with the real function."""
        from skimage.segmentation import expand_labels

        import spacr.measure as measure

        expected = "spacing" in inspect.signature(expand_labels).parameters
        assert measure._EXPAND_LABELS_TAKES_SPACING is expected

    def test_a_signature_that_cannot_be_read_answers_no(self, monkeypatch):
        """THE UNCOVERED PAIR.

        A C-implemented `expand_labels` -- which is what a differently
        built scikit-image can present -- makes `inspect.signature`
        raise. Answering "no spacing argument" is the safe direction: the
        caller then does its own scaling instead of passing an argument
        that would be rejected.
        """
        import skimage.segmentation as seg

        # `dict.update` is C-implemented and has no readable signature,
        # which is exactly the shape being guarded against.
        with pytest.raises((TypeError, ValueError)):
            inspect.signature(dict.update)

        monkeypatch.setattr(seg, "expand_labels", dict.update)
        measure = _reimport_measure()
        try:
            assert measure._EXPAND_LABELS_TAKES_SPACING is False
        finally:
            monkeypatch.undo()
            _reimport_measure()

    def test_the_probe_is_restored_afterwards(self):
        """The re-import above must not leave a poisoned module behind."""
        from skimage.segmentation import expand_labels

        import spacr.measure as measure

        expected = "spacing" in inspect.signature(expand_labels).parameters
        assert measure._EXPAND_LABELS_TAKES_SPACING is expected


class TestTheMaskSetGuardThatCannotFire:
    """`if name not in masks: masks = dict(masks, **{name: mask})`.

    Unreachable, and pinned from the producing side rather than forced.

    `_all_masks()` includes an object type exactly when its mask is not
    None and has a non-zero size. `_with_distances` is called with that
    same mask, and it has already returned for an empty props frame --
    and a non-empty props frame can only have come from a non-empty
    mask. So by the time the guard runs, the name is always present.
    """

    def test_the_mask_set_rule_is_still_size_based(self):
        import spacr.measure as measure

        source = inspect.getsource(measure.measure_crop) if hasattr(
            measure, "measure_crop") else ""
        # the helpers are nested; read the enclosing function's source
        import re

        module_source = inspect.getsource(measure)
        assert "def _all_masks():" in module_source
        block = module_source.split("def _all_masks():", 1)[1]
        assert "if mask is not None and getattr(mask, 'size', 0):" in block, (
            "the mask set no longer admits every non-empty mask, so the "
            "`name not in masks` guard below it may be reachable")

    def test_the_distance_merge_returns_early_on_an_empty_frame(self):
        """The other half of the argument: an empty frame never reaches it."""
        import spacr.measure as measure

        module_source = inspect.getsource(measure)
        block = module_source.split("def _with_distances(", 1)[1]
        assert "if not distances_on or len(frame) == 0:" in block
        early = block.index("if not distances_on or len(frame) == 0:")
        guard = block.index("if name not in masks:")
        assert early < guard, (
            "the empty-frame return no longer precedes the mask guard")

    def test_every_caller_passes_the_mask_the_name_stands_for(self):
        """`_with_distances(cell_props, cell_mask, 'cell')` and its two twins.

        A caller that passed a different mask under a name would make the
        guard live -- and would also be measuring the wrong object.
        """
        import spacr.measure as measure

        module_source = inspect.getsource(measure)
        for name in ("cell", "nucleus", "pathogen"):
            assert (f"_with_distances({name}_props, {name}_mask, '{name}')"
                    in module_source), (
                f"the {name} distance merge no longer passes its own mask")
