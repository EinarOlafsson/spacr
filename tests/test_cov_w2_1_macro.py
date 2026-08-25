"""The macro recorder when the process it is recording in is not whole.

Recording must never fail a run, so nearly every lookup in this module is
wrapped: no Qt, no plugins module, no artifacts registry, a version file
that will not import. Those branches are what a headless cluster install
actually takes, and they are reached here by removing the thing the lookup
imports rather than by asserting a mock was called. The reader half --
:func:`spacr.macro.read_macro` -- is driven with real scripts on disk,
including ones spaCR did not write.
"""
from __future__ import annotations

import logging
import os
import sys
import types

import pytest

from spacr import macro
from spacr.macro import MacroError, MacroStep


@pytest.fixture
def no_module(monkeypatch):
    """Make one dotted import fail, the way a partial install does."""
    def _remove(*names):
        for name in names:
            monkeypatch.setitem(sys.modules, name, None)

    return _remove


# ---------------------------------------------------------------------------
# Finding the entry point to call


def test_a_module_key_that_is_blank_has_no_entry_point():
    """An unset module is not an error; it simply calls nothing."""
    assert macro.entry_for("") == ("", "")
    assert macro.entry_for(None) == ("", "")


@pytest.mark.parametrize("dotted", ["", "   ", "justoneword", "pkg.mod:3bad",
                                    ":func", "pkg.mod:"])
def test_an_entry_that_is_not_two_halves_is_no_entry(dotted):
    """A half-written `entry=` must not become an import line in a script."""
    assert macro._split_entry(dotted) == ("", "")


def test_an_unknown_app_has_no_entry_point():
    """Annotate and Make Masks are interactive; there is nothing to call."""
    assert macro.entry_for("no-such-app-anywhere") == ("", "")


def test_the_shipped_table_being_unreadable_falls_through(no_module):
    """Without `spacr.validate` the registration seams still answer."""
    no_module("spacr.validate")

    assert macro.entry_for("mask") == ("", "")


def test_a_process_without_qt_registers_no_entry(no_module):
    """A headless recorder is a supported install."""
    no_module("spacr.qt.app")

    assert macro._registered_entry_text("mask") == ""


def test_a_broken_app_registry_registers_no_entry(monkeypatch):
    """A registry that raises on lookup costs the script its entry, not the run."""
    import spacr.qt.app as qt_app

    class _Hostile(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("registry is mid-rebuild")

    monkeypatch.setattr(qt_app, "APP_META", _Hostile())

    assert macro._registered_entry_text("mask") == ""


def test_a_process_without_plugins_has_no_plugin_entry(no_module):
    """The plugin system is optional; its absence is not a failure."""
    no_module("spacr.plugins")

    assert macro._plugin_entry_text("anything") == ""


def test_a_plugin_entrypoint_is_used_when_nothing_else_answers(monkeypatch):
    """A plugin app's `entrypoint` is the third and last source."""
    import spacr.plugins as plugins

    monkeypatch.setattr(plugins, "get_app", lambda key: types.SimpleNamespace(
        entrypoint="my_plugin.pipeline:run"))

    assert macro.entry_for("my-plugin") == ("my_plugin.pipeline", "run")


# ---------------------------------------------------------------------------
# Filling in the defaults the script has to pin


def test_a_blank_module_key_has_no_defaults():
    """Nothing to look up, and nothing claimed to have answered."""
    assert macro.module_defaults("") == ({}, "none")


def test_a_broken_defaults_registry_falls_through(monkeypatch):
    """A registry that raises must not stop the next source answering."""
    import spacr.plugins as plugins
    import spacr.settings as settings_module

    def _explode(key):
        raise RuntimeError("registry is mid-rebuild")

    monkeypatch.setattr(settings_module, "has_registered_defaults", _explode)
    monkeypatch.setattr(plugins, "get_app", lambda key: types.SimpleNamespace(
        defaults="ignored"))
    monkeypatch.setattr(plugins, "load_object",
                        lambda ref: (lambda: {"scale": 3}))

    assert macro.module_defaults("plug") == ({"scale": 3}, "plugin")


def test_nothing_answers_when_no_source_is_importable(monkeypatch, no_module):
    """`defaults_source` says 'none' rather than pretending the keys existed."""
    no_module("spacr.settings", "spacr.plugins",
              "spacr.qt.screens.settings_model")

    assert macro.module_defaults("mask") == ({}, "none")


def test_a_plugin_supplies_the_defaults_when_nothing_else_does(monkeypatch,
                                                               no_module):
    """A plugin app's `defaults` factory is consulted before the built-ins."""
    import spacr.plugins as plugins

    no_module("spacr.settings")
    monkeypatch.setattr(plugins, "get_app", lambda key: types.SimpleNamespace(
        defaults="ignored", entrypoint=""))
    monkeypatch.setattr(plugins, "load_object",
                        lambda ref: (lambda settings: {"scale": 3}))

    assert macro.module_defaults("plug") == ({"scale": 3}, "plugin")


def test_a_plugin_whose_defaults_factory_raises_supplies_none(monkeypatch,
                                                              no_module):
    """A broken plugin falls through to the built-in source."""
    import spacr.plugins as plugins

    def _explode(ref):
        raise RuntimeError("plugin is not installed properly")

    monkeypatch.setattr(plugins, "get_app", lambda key: types.SimpleNamespace(
        defaults="ignored"))
    monkeypatch.setattr(plugins, "load_object", _explode)

    assert macro._plugin_defaults("plug") is None


def test_a_defaults_factory_is_called_the_way_its_signature_asks():
    """Calling and retrying on TypeError cannot tell whose TypeError it is."""
    assert macro._takes_an_argument(lambda settings: {}) is True
    assert macro._takes_an_argument(lambda: {}) is False
    assert macro._takes_an_argument(lambda *args: {}) is True
    assert macro._takes_an_argument(lambda *, only_kw=None: {}) is False


def test_something_with_no_readable_signature_is_offered_the_dict():
    """An unreadable signature is assumed to want the settings dict."""
    assert macro._takes_an_argument(object()) is True


def test_run_control_defaults_missing_costs_only_the_seed(monkeypatch,
                                                          no_module):
    """Without `runctx` the caller's settings are still recorded in full."""
    no_module("spacr.runctx")

    resolved, defaulted, source = macro.explicit_settings(
        "mask", {"src": "/data/plate01"})

    assert resolved["src"] == "/data/plate01"
    assert "src" not in defaulted
    assert source in {"registered", "settings_model", "none"}


# ---------------------------------------------------------------------------
# The pieces a step records, when the thing that knows is missing


def test_a_settings_hash_that_cannot_be_computed_is_blank(no_module):
    """A record without a digest is honest; a wrong digest is not."""
    no_module("spacr.artifacts")

    assert macro._settings_hash({"src": "/data"}) == ""


def test_the_project_falls_back_to_src_when_ports_cannot_answer(no_module):
    """`src` is what the user typed, and it is better than nothing."""
    no_module("spacr.ports")

    assert macro._project_root("mask", {"src": "/data/plate01"}) == \
        "/data/plate01"
    assert macro._project_root("mask", {"src": ["/data/a", "/data/b"]}) == \
        "/data/a"
    assert macro._project_root("mask", {"src": []}) == ""
    assert macro._project_root("mask", {}) == ""


def test_declared_outputs_that_cannot_be_resolved_are_none(no_module):
    """A record with no outputs is preferable to a record that guesses."""
    no_module("spacr.ports")

    assert macro._outputs("mask", {"src": "/data"}) == ()


def test_an_unreadable_version_is_recorded_as_unknown(no_module):
    """The header still renders; it just cannot claim a version."""
    no_module("spacr.version")

    assert macro._version() == "unknown"


# ---------------------------------------------------------------------------
# Recording, which may not fail a run


class _AwkwardRecord:
    """A log record whose `run_id` cannot be read."""

    @property
    def run_id(self):
        raise RuntimeError("a filter upstream replaced this attribute")


def test_a_log_record_that_cannot_be_read_is_still_passed_on():
    """The capture sits on the root logger; it may not swallow a run's logs."""
    capture = macro._RunIdCapture(1)

    assert capture.handle(_AwkwardRecord()) is True
    assert capture.observed == ()


def test_a_recorder_that_cannot_start_is_a_no_op(monkeypatch):
    """Recording is optional; the run continues either way."""
    def _explode(thread_id):
        raise RuntimeError("no logging in this process")

    monkeypatch.setattr(macro, "_RunIdCapture", _explode)

    assert macro.begin_recording("mask", {"src": "/data"}) is None
    assert macro.finish_recording(None) is None


def test_a_capture_that_will_not_close_does_not_stop_the_record(monkeypatch,
                                                                tmp_path):
    """Detaching the handler is best-effort; the step is still appended."""
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    macro.reset()
    recording = macro.begin_recording("mask", {"src": str(tmp_path)})
    assert recording is not None

    def _refuse():
        raise RuntimeError("already closed")

    monkeypatch.setattr(recording.capture, "close", _refuse)
    step = macro.finish_recording(recording, status="success")

    assert step is not None
    assert step.module == "mask"
    assert recording.capture not in logging.getLogger().handlers
    macro.reset()


# ---------------------------------------------------------------------------
# Chaining and writing


def test_an_empty_chain_continues_nothing():
    """The first step of a chain links to no predecessor."""
    assert macro._continues(macro.Macro(), MacroStep(module="mask")) == ""


def test_a_run_folder_that_will_not_take_the_script_is_logged(monkeypatch,
                                                              tmp_path,
                                                              caplog):
    """The stable copy is still written; only the run-folder copy is lost."""
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    step = MacroStep(module="mask", run_dir=str(run_dir))
    chain = macro.Macro(steps=[step])
    real_write = macro.Macro.write

    def _refuse_the_run_folder(self, path):
        if str(path).startswith(str(run_dir)):
            raise OSError("read-only run folder")
        return real_write(self, path)

    monkeypatch.setattr(macro.Macro, "write", _refuse_the_run_folder)

    with caplog.at_level(logging.ERROR, logger="spacr.macro"):
        macro._write_everywhere(chain, step)

    assert os.path.isfile(chain.path)
    assert "could not write the macro into" in caplog.text


def test_a_chain_says_what_it_is_and_how_long_it_is():
    """The one-line form a log message uses."""
    chain = macro.Macro(steps=[MacroStep(module="mask"),
                               MacroStep(module="measure")])

    assert len(chain) == 2
    assert chain.macro_id in str(chain)
    assert "mask -> measure" in str(chain)


@pytest.mark.parametrize("given, expected", [
    ("", "STEP"),
    ("!!!", "STEP"),
    ("3d viewer", "STEP_3D_VIEWER"),
])
def test_a_module_name_that_is_not_an_identifier_still_names_a_constant(
        given, expected):
    """The settings constant has to be legal Python whatever the app is called."""
    assert macro._identifier(given) == expected


# ---------------------------------------------------------------------------
# Rendering values a settings dict really contains


def _render(value):
    return macro._render_value(value, macro._Threader([]), "")


def test_a_one_element_tuple_keeps_its_trailing_comma():
    """`(3)` is the number 3; the script has to say `(3,)`."""
    assert _render((3,)) == "(3,)"
    assert _render((3, 4)) == "(3, 4)"


def test_an_empty_set_is_rendered_as_a_call_not_as_braces():
    """`{}` is an empty dict, which is a different value."""
    assert _render(set()) == "set()"
    assert _render({7}) == "{7}"


def test_a_value_that_is_not_a_literal_is_recorded_as_coerced():
    """The script admits the lossy conversion instead of looking exact."""
    coerced = []

    rendered = macro._render_value(object(), macro._Threader([]), "",
                                   coerced, "model")

    assert rendered.startswith("'<object object at")
    assert coerced == ["model"]


# ---------------------------------------------------------------------------
# Reading a macro back without executing it


def _script(tmp_path, body, name="macro.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_a_file_with_no_record_is_not_a_macro(tmp_path):
    """The exporter's input may be any Python file; say so plainly."""
    path = _script(tmp_path, "X = 1\n")

    with pytest.raises(MacroError, match="carries no MACRO record"):
        macro.read_macro(path)


def test_a_newer_schema_asks_for_a_newer_spacr(tmp_path):
    """Reading a record this build does not understand would misreport it."""
    path = _script(tmp_path, f"MACRO = {{'schema': {macro.MACRO_SCHEMA + 5}}}\n")

    with pytest.raises(MacroError, match="Upgrade spaCR"):
        macro.read_macro(path)


def test_assignments_the_reader_cannot_evaluate_are_skipped(tmp_path):
    """An edited macro keeps its record even with hand-written code above it."""
    path = _script(tmp_path, (
        "import os\n"
        "TOTALS = {}\n"
        "TOTALS['a'] = 1\n"
        "SIZE = len('abcd')\n"
        "PROJECT_1 = os.path.join('/data', 'plate01')\n"
        "SPARSE = {1, 2}\n"
        "PAIR = (1, -2)\n"
        "OFFSET = +5\n"
        f"MACRO = {{'schema': {macro.MACRO_SCHEMA}, 'root': PROJECT_1,\n"
        "          'sparse': SPARSE, 'pair': PAIR, 'offset': OFFSET}\n"
    ))

    record = macro.read_macro(path)

    assert record["root"] == os.path.join("/data", "plate01")
    assert record["sparse"] == {1, 2}
    assert record["pair"] == (1, -2)
    assert record["offset"] == 5


def test_a_name_the_file_never_bound_is_refused():
    """Resolving it would mean running the file, which is the whole point."""
    import ast

    with pytest.raises(MacroError, match="unbound name 'MYSTERY'"):
        macro._evaluate(ast.parse("MYSTERY", mode="eval").body, {})


def test_an_expression_the_reader_will_not_run_is_refused():
    """Not importing the file is worthless if arbitrary calls are evaluated."""
    import ast

    with pytest.raises(MacroError, match="will not\n?\\s*evaluate"):
        macro._evaluate(ast.parse("open('/etc/passwd')", mode="eval").body, {})


# ---------------------------------------------------------------------------
# Summarising a record


def test_the_summary_lists_what_a_step_produced():
    """The methods section starts from what is on disk."""
    record = {
        "spacr_version": "1.2.3",
        "macro_id": "abc123",
        "generated_utc": "2026-01-01T00:00:00Z",
        "steps": [{
            "index": 1, "module": "mask", "entry": "spacr.core.masks",
            "run_id": "r1", "settings_hash": "0123456789abcdef",
            "settings": {"src": "/data", "scale": 2},
            "user_set": ["src"], "status": "success",
            "outputs": ["/data/merged", "/data/masks", "/data/stack",
                        "/data/measurements"],
        }],
    }

    text = macro.summarise(record)

    assert "spaCR 1.2.3 — mask" in text
    assert "1 of 2 settings chosen" in text
    assert "produced 4: /data/merged, /data/masks, /data/stack …" in text
