"""The macro recorder when the disk, not the run, is what goes wrong.

The recorder's whole promise is that it never costs a run: a script that
cannot be written is a log line and nothing more. These tests drive the
three ways writing actually fails on a real install -- a macro folder that
is not a folder, a run with no journal folder to keep a second copy in,
and a caller that hands ``Macro.write`` a bare filename -- and pin the
part users depend on: the pipeline still finishes, the step is still
recorded in the chain, and whatever copy *could* be written is on disk.

The rendering half is covered where the settings dict holds something
Python cannot round-trip (a fitted model object, a numpy array), which the
script must admit to rather than silently mangle.
"""
from __future__ import annotations

import pytest

from spacr import macro
from spacr.macro import Macro, MacroStep


@pytest.fixture
def isolated_macros(tmp_path, monkeypatch):
    """Point the macro folder, the home dir and the run logs at tmp_path.

    ``macros_dir()`` defaults to ``~/.spacr/macros``; without this a test
    would write scripts into the developer's real home and inherit chains
    recorded by whatever ran before it.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    macro.reset()
    yield tmp_path
    macro.reset()


def _demo_macro(module: str = "mask", **settings) -> Macro:
    """A one-step chain that renders, without going through a run."""
    chain = Macro()
    chain.steps.append(MacroStep(module=module,
                                 entry_module="spacr.core",
                                 entry_func="preprocess_generate_masks",
                                 settings=dict(settings),
                                 user_set=tuple(settings),
                                 status="success"))
    return chain


# ---------------------------------------------------------------------------
# Where the script is written


# ---------------------------------------------------------------------------
# Finishing a recording that never started a log capture


# ---------------------------------------------------------------------------
# When the script cannot be written


# ---------------------------------------------------------------------------
# Rendering a value Python cannot round-trip


class _Model:
    """Stands in for a fitted estimator that reached a settings dict."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __str__(self) -> str:
        return f"<fitted model {self.tag}>"


def test_a_key_holding_several_unrenderable_values_is_named_once(
        isolated_macros):
    """The lossy-conversion list must name a key once, not once per item.

    ``coerced`` is what tells a reader "this line is a string of the thing,
    not the thing" -- the methods exporter prints it verbatim. A list of
    two fitted models under one key is one lossy key; naming it twice
    would make the exported methods section claim two settings were
    approximated when only one was, and a consumer de-duplicating the list
    would be papering over a recorder bug.
    """
    chain = _demo_macro("classify",
                        src=str(isolated_macros),
                        models=[_Model("a"), _Model("b")],
                        threshold=0.5)

    source = macro.render(chain)
    step = chain.steps[0]

    assert step.coerced == ("models",)
    assert "<fitted model a>" in source
    assert "<fitted model b>" in source
    # A genuine literal in the same dict is rendered as itself and is not
    # named as coerced, which is what the distinction is for.
    assert "0.5" in source
    assert "threshold" not in step.coerced
    compile(source, "<macro>", "exec")


