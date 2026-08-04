"""The macro recorder, verified by execution rather than by inspection.

The test that matters here is :class:`TestRoundTrip`: a module is run
through the same seam the GUI runs it through, the recorder emits its
script, the script is executed in a *fresh interpreter* on the same
inputs, and the two outputs are compared byte for byte. A recorder that
emits plausible-looking code which does not run is worse than none, and
nothing short of running it proves otherwise.

The rest pins the four promises the emitted script makes:

* a chain (mask → measure → classify) is one script, in dependency order,
  with the intermediate paths threaded through a named constant;
* a setting changed by the caller appears changed in the script, and the
  settings that were merely defaults appear anyway, explicitly;
* every script parses and compiles;
* the header carries the spaCR version, the run id and the settings hash,
  and the run id is *the same id the run used*.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import macro


REPO_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# A real, importable pipeline module — the thing the emitted script imports
# ---------------------------------------------------------------------------

#: A genuine spaCR-shaped entry point: one positional settings dict, real
#: file input, real file output, deterministic. Written to a temp folder and
#: imported by name, so the emitted `from spacr_macro_demo import ...` is a
#: real import that a fresh interpreter can satisfy — which is what makes
#: the round trip a round trip rather than a string comparison.
DEMO_MODULE = '''\
"""A miniature spaCR module, for the macro recorder's round-trip test."""
import json
import os


def run_demo(settings):
    """Sum the numbers in every input file and write the totals.

    Deliberately sensitive to more than `src`: `scale` multiplies, `label`
    names the output and `skip_empty` changes which files count. A macro
    that dropped or defaulted any of them would produce a different file,
    which is exactly what the round-trip assertion detects.
    """
    src = settings["src"]
    if isinstance(src, (list, tuple)):
        src = src[0]
    scale = settings.get("scale", 1)
    label = settings.get("label", "demo")
    skip_empty = settings.get("skip_empty", False)
    totals = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(src, name)) as handle:
            numbers = [int(line) for line in handle.read().split() if line]
        if skip_empty and not numbers:
            continue
        totals[name] = sum(numbers) * scale
    out = os.path.join(src, f"{label}.json")
    with open(out, "w") as handle:
        json.dump({"label": label, "scale": scale, "totals": totals},
                  handle, indent=2, sort_keys=True)
    print(f"demo wrote {out}")
    return out


def demo_defaults(settings=None):
    """The module's defaults, through the `register_defaults` seam."""
    values = dict(settings or {})
    values.setdefault("src", "")
    values.setdefault("scale", 1)
    values.setdefault("label", "demo")
    values.setdefault("skip_empty", False)
    return values
'''

DEMO_KEY = "macro_demo"


@pytest.fixture
def demo_package(tmp_path, monkeypatch):
    """Write the demo pipeline module and put it on the import path."""
    folder = tmp_path / "pkg"
    folder.mkdir()
    (folder / "spacr_macro_demo.py").write_text(DEMO_MODULE)
    monkeypatch.syspath_prepend(str(folder))
    return str(folder)


@pytest.fixture
def isolated_journal(tmp_path, monkeypatch):
    """Point the run journal, the macro folder and the run logs at tmp_path.

    ``run_journal.runs_root`` is ``Path.home()/'.spacr'/'runs'``, so a test
    that did not do this would leave real runs in the developer's home.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    macro.reset()
    yield home
    macro.reset()


@pytest.fixture
def registered_demo(demo_package):
    """Register the demo module the way a new pipeline app registers.

    ``register_defaults`` + ``register_app(..., entry=...)`` — the two
    seams a module is supposed to join through — rather than an edit to
    ``APP_FUNCTIONS`` or the settings tables. Both are undone afterwards,
    because the app registry is process-global and a leaked row shows up
    in another test file as a mysterious extra tile.
    """
    from spacr.settings import register_defaults, unregister_defaults
    import spacr_macro_demo

    register_defaults(DEMO_KEY, spacr_macro_demo.demo_defaults, replace=True)
    registered_app = False
    try:
        from spacr.qt.app import register_app, unregister_app, SECTION_ORDER
        register_app(DEMO_KEY, "Macro Demo", "round-trip fixture",
                     SECTION_ORDER[0],
                     entry="spacr_macro_demo:run_demo")
        registered_app = True
    except Exception:
        # No Qt in this environment, or the registry refused the row. The
        # recorder still resolves the entry point below; this only decides
        # *which* of its three sources answers.
        unregister_app = None
    yield DEMO_KEY
    unregister_defaults(DEMO_KEY)
    if registered_app and unregister_app is not None:
        unregister_app(DEMO_KEY)


def make_inputs(root: Path) -> Path:
    """Create the demo module's inputs: three files of numbers."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("1\n2\n3\n")
    (root / "b.txt").write_text("10\n20\n")
    (root / "c.txt").write_text("")
    return root


def run_through_the_gui_seam(app_key, settings, entry):
    """Run one module exactly as the GUI worker runs it.

    Two statements, and they are the two statements
    ``spacr.qt.bridge.PipelineWorker.run`` executes around every pipeline:
    open the journal, call the entry point with the settings dict. The
    recorder hangs off the first of those, which is why this is the seam
    a test drives rather than a QThread.
    """
    from spacr.run_journal import open_run
    with open_run(app_key, settings) as run:
        entry(settings)
        run.set_status("success")
    return run


# ---------------------------------------------------------------------------
# The round trip — run it, emit it, run the emission, compare
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Execute the emitted script and compare its outputs to the run's."""

    def test_emitted_script_reproduces_the_run(
            self, tmp_path, isolated_journal, registered_demo):
        """The whole promise, verified by running the thing.

        Run the module through the GUI seam, keep what it produced, delete
        it, execute the emitted script in a fresh interpreter, and require
        the bytes to match.
        """
        from spacr.qt.bridge import resolve_pipeline_entry

        src = make_inputs(tmp_path / "plate")
        settings = {"src": str(src), "scale": 3, "label": "run"}
        entry = resolve_pipeline_entry(registered_demo)
        assert entry is not None, "the demo app did not register an entry"

        run = run_through_the_gui_seam(registered_demo, settings, entry)

        produced = src / "run.json"
        assert produced.is_file(), "the run itself produced nothing"
        expected = produced.read_bytes()

        script = Path(macro.macro_path(run.dir))
        assert script.is_file(), f"no macro written into {run.dir}"

        produced.unlink()
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                [str(tmp_path / "pkg"), REPO_ROOT])},
        )
        assert result.returncode == 0, (
            f"the emitted script did not run:\n{result.stdout}\n{result.stderr}")
        assert produced.is_file(), "the emitted script produced no output"
        assert produced.read_bytes() == expected, (
            "the emitted script produced a different result than the run it "
            "was recorded from")

    def test_the_script_is_not_a_replay_harness(
            self, tmp_path, isolated_journal, registered_demo):
        """It imports and calls the entry point — the code a user would write.

        Guards the difference between this and ``spacr-repro``: a script
        that shelled out to spaCR, or loaded ``settings.json`` off disk,
        would pass the round trip above and be worthless as an on-ramp to
        the API.
        """
        from spacr.qt.bridge import resolve_pipeline_entry
        src = make_inputs(tmp_path / "plate")
        run = run_through_the_gui_seam(
            registered_demo, {"src": str(src), "label": "x"},
            resolve_pipeline_entry(registered_demo))
        source = Path(macro.macro_path(run.dir)).read_text()

        assert "from spacr_macro_demo import run_demo" in source
        assert "run_demo(MACRO_DEMO_SETTINGS)" in source
        assert "settings.json" not in source
        assert "subprocess" not in source
        assert "spacr-repro" not in source

    def test_a_failed_run_still_leaves_its_script(
            self, tmp_path, isolated_journal, registered_demo):
        """The script exists even when the run raised.

        A failed run is the one you most want to repeat with one setting
        changed, so the recorder writes from the journal's `finally`, not
        from its success path.
        """
        from spacr.run_journal import open_run

        def _boom(settings):
            raise RuntimeError("no")

        src = make_inputs(tmp_path / "plate")
        with pytest.raises(RuntimeError):
            with open_run(registered_demo, {"src": str(src)}) as run:
                _boom({})
        record = macro.read_macro(macro.macro_path(run.dir))
        assert record["steps"][0]["status"] == "failed"


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

class TestChains:
    """mask → measure → classify is one script, in dependency order."""

    def _chain(self, root):
        for module in ("mask", "measure", "classify"):
            recording = macro.begin_recording(module, {"src": str(root)})
            macro.finish_recording(recording, status="success")
        return macro.current_macro()

    def test_three_runs_become_one_script_in_order(self, tmp_path,
                                                   isolated_journal):
        chain = self._chain(tmp_path / "plate")
        assert chain.modules == ("mask", "measure", "classify")
        source = chain.source()

        positions = [source.index(f"# Step {n} — ") for n in (1, 2, 3)]
        assert positions == sorted(positions), "steps are out of order"

        calls = source.split("def main():")[1]
        assert (calls.index("preprocess_generate_masks(")
                < calls.index("measure_crop(")
                < calls.index("deep_spacr("))

    def test_the_edge_is_the_one_ports_declares(self, tmp_path,
                                                isolated_journal):
        """Not "these ran near each other" — the pipeline contract says so."""
        from spacr.ports import next_modules
        chain = self._chain(tmp_path / "plate")
        assert "measure" in next_modules("mask")
        assert [step.link for step in chain.steps] == ["", "ports", "ports"]

    def test_intermediate_paths_are_threaded_not_repeated(
            self, tmp_path, isolated_journal):
        """Step 2 reads step 1's project through the same named constant."""
        root = tmp_path / "plate"
        chain = self._chain(root)
        source = chain.source()

        assert f"PROJECT_1 = {str(root)!r}" in source
        for block in ("MASK_SETTINGS", "MEASURE_SETTINGS", "CLASSIFY_SETTINGS"):
            body = source.split(f"{block} = {{")[1].split("\n}")[0]
            assert "'src': PROJECT_1," in body, f"{block} repeats the path"
        # The literal appears exactly once: in the constant.
        assert source.count(f"{str(root)!r}") == 1

    def test_a_path_below_the_project_is_rebuilt_from_the_constant(
            self, tmp_path, isolated_journal):
        root = tmp_path / "plate"
        root.mkdir()
        first = macro.begin_recording("mask", {"src": str(root)})
        macro.finish_recording(first, status="success")
        second = macro.begin_recording(
            "measure", {"src": str(root),
                        "tar_path": str(root / "datasets" / "training.tar")})
        macro.finish_recording(second, status="success")
        source = macro.current_macro().source()
        assert "os.path.join(PROJECT_1, 'datasets', 'training.tar')" in source
        assert "\nimport os" in source, "os.path.join used without importing os"

    def test_an_unrelated_project_starts_a_new_chain(self, tmp_path,
                                                     isolated_journal):
        """Two plates are two analyses, and welding them together lies."""
        for name in ("plate_a", "plate_b"):
            recording = macro.begin_recording("mask",
                                              {"src": str(tmp_path / name)})
            macro.finish_recording(recording, status="success")
        assert len(macro.macros()) == 2
        assert all(len(chain.steps) == 1 for chain in macro.macros())

    def test_a_chain_stops_growing_at_the_bound(self, tmp_path,
                                                isolated_journal):
        """A window left open for a week must not accumulate one script."""
        root = tmp_path / "plate"
        for _ in range(macro.MAX_CHAIN_STEPS + 3):
            recording = macro.begin_recording("mask", {"src": str(root)})
            macro.finish_recording(recording, status="success")
        assert len(macro.macros()) == 2
        assert len(macro.macros()[0].steps) == macro.MAX_CHAIN_STEPS
        assert len(macro.current_macro().steps) == 3

    def test_a_long_idle_gap_starts_a_new_chain(self, tmp_path,
                                                isolated_journal):
        """Yesterday's plate is not step one of today's analysis."""
        root = tmp_path / "plate"
        first = macro.begin_recording("mask", {"src": str(root)})
        macro.finish_recording(first, status="success")
        chain = macro.current_macro()
        chain.touched -= macro.CHAIN_IDLE_SECONDS + 60

        second = macro.begin_recording("measure", {"src": str(root)})
        macro.finish_recording(second, status="success")
        assert len(macro.macros()) == 2
        assert macro.current_macro().modules == ("measure",)

    def test_only_so_many_chains_are_kept_in_memory(self, tmp_path,
                                                   isolated_journal):
        """Bounded memory: the older scripts are already on disk."""
        for index in range(macro.MAX_RETAINED_MACROS + 5):
            recording = macro.begin_recording(
                "mask", {"src": str(tmp_path / f"plate{index}")})
            macro.finish_recording(recording, status="success")
        assert len(macro.macros()) == macro.MAX_RETAINED_MACROS
        written = list(Path(os.environ[macro.MACRO_DIR_ENV]).glob("*.py"))
        assert len(written) == macro.MAX_RETAINED_MACROS + 5

    def test_the_chain_script_is_written_to_every_run_folder(
            self, tmp_path, isolated_journal, registered_demo):
        """The last run's folder holds the whole chain, not just its own step."""
        from spacr.qt.bridge import resolve_pipeline_entry
        entry = resolve_pipeline_entry(registered_demo)
        src = make_inputs(tmp_path / "plate")
        first = run_through_the_gui_seam(
            registered_demo, {"src": str(src), "label": "one"}, entry)
        second = run_through_the_gui_seam(
            registered_demo, {"src": str(src), "label": "two"}, entry)

        early = macro.read_macro(macro.macro_path(first.dir))
        late = macro.read_macro(macro.macro_path(second.dir))
        assert len(early["steps"]) == 1
        assert len(late["steps"]) == 2
        assert [step["settings"]["label"] for step in late["steps"]] == [
            "one", "two"]


# ---------------------------------------------------------------------------
# Settings: what the user chose, and what was merely a default
# ---------------------------------------------------------------------------

class TestSettings:
    """A change made in the GUI shows up in the script; defaults are explicit."""

    def test_a_changed_setting_appears_changed(self, tmp_path,
                                               isolated_journal,
                                               registered_demo):
        from spacr.qt.bridge import resolve_pipeline_entry
        src = make_inputs(tmp_path / "plate")
        run = run_through_the_gui_seam(
            registered_demo,
            {"src": str(src), "scale": 7, "label": "changed"},
            resolve_pipeline_entry(registered_demo))

        source = Path(macro.macro_path(run.dir)).read_text()
        assert "'scale': 7," in source
        assert "'scale': 1," not in source, "the default leaked in"

        record = macro.read_macro(macro.macro_path(run.dir))
        step = record["steps"][0]
        assert step["settings"]["scale"] == 7
        assert set(step["user_set"]) == {"src", "scale", "label"}

    def test_settings_that_happen_to_be_defaults_are_written_out(
            self, tmp_path, isolated_journal, registered_demo):
        """The unset keys are in the script with their values, and marked."""
        from spacr.qt.bridge import resolve_pipeline_entry
        src = make_inputs(tmp_path / "plate")
        run = run_through_the_gui_seam(
            registered_demo, {"src": str(src)},
            resolve_pipeline_entry(registered_demo))

        source = Path(macro.macro_path(run.dir)).read_text()
        assert "'scale': 1,  # spaCR default" in source
        assert "'skip_empty': False,  # spaCR default" in source

        record = macro.read_macro(macro.macro_path(run.dir))
        step = record["steps"][0]
        assert set(step["defaulted"]) >= {"scale", "label", "skip_empty"}
        assert step["user_set"] == ["src"]

    def test_the_run_seed_is_pinned_even_when_nobody_set_it(
            self, tmp_path, isolated_journal):
        """The default that matters most: an unpinned seed is not reproducible."""
        from spacr.runctx import DEFAULT_SEED
        resolved, defaulted, _ = macro.explicit_settings(
            "mask", {"src": str(tmp_path)})
        assert resolved["random_seed"] == DEFAULT_SEED
        assert resolved["on_error"] == "stop"
        assert "random_seed" in defaulted and "on_error" in defaulted

    def test_a_value_that_is_not_a_literal_is_admitted_not_hidden(
            self, tmp_path, isolated_journal):
        """A non-literal is rendered as text AND named in `coerced`.

        Settings dicts are supposed to hold literals, but a caller can put
        anything in one. Silently emitting `repr(obj)` would produce a
        script that runs and is wrong; the record says which key lost
        fidelity.
        """
        class Odd:
            def __str__(self):
                return "odd-value"

        recording = macro.begin_recording(
            "mask", {"src": str(tmp_path), "cell_diameter": Odd()})
        step = macro.finish_recording(recording, status="success")
        source = macro.current_macro().source()
        assert "'cell_diameter': 'odd-value'," in source
        assert "cell_diameter" in step.coerced


# ---------------------------------------------------------------------------
# The script is Python
# ---------------------------------------------------------------------------

class TestTheScriptIsPython:
    """Everything render() produces parses and compiles."""

    @pytest.mark.parametrize("modules", [
        (),
        ("mask",),
        ("mask", "measure"),
        ("mask", "measure", "classify"),
        ("mask", "mask"),
        ("annotate",),
        ("map_barcodes",),
    ])
    def test_every_shape_of_macro_compiles(self, modules, tmp_path,
                                           isolated_journal):
        chain = macro.Macro()
        for module in modules:
            recording = macro.begin_recording(module,
                                              {"src": str(tmp_path / "p")})
            step = macro.finish_recording(recording, status="success")
            if step is not None:
                chain.steps.append(step)
        source = macro.render(chain)
        ast.parse(source)
        compile(source, "<macro>", "exec")

    def test_an_interactive_module_is_recorded_but_not_called(
            self, tmp_path, isolated_journal):
        """Annotate has no API entry point; the script says so and still parses."""
        recording = macro.begin_recording("annotate", {"src": str(tmp_path)})
        macro.finish_recording(recording, status="success")
        source = macro.current_macro().source()
        compile(source, "<macro>", "exec")
        assert "no API entry point" in source
        assert macro.read_macro.__doc__  # sanity: reading it back still works

    def test_the_settings_dict_survives_literal_eval(self, tmp_path,
                                                     isolated_journal):
        """The settings block on its own is a literal — no hidden machinery."""
        recording = macro.begin_recording(
            "measure", {"src": str(tmp_path), "channels": [0, 1],
                        "experiment": "exp1"})
        macro.finish_recording(recording, status="success")
        source = macro.current_macro().source()
        block = "{" + source.split("MEASURE_SETTINGS = {")[1].split("\n}")[0]
        # PROJECT_1 is the one name in it; substitute and it must literal_eval.
        parsed = ast.literal_eval(
            block.replace("PROJECT_1", repr(str(tmp_path))) + "\n}")
        assert parsed["channels"] == [0, 1]
        assert parsed["experiment"] == "exp1"


# ---------------------------------------------------------------------------
# Provenance: the header, and the id that joins everything
# ---------------------------------------------------------------------------

class TestProvenance:
    """The header carries the version, the run id and the settings hash."""

    def test_the_run_id_is_the_id_the_run_used(self, tmp_path,
                                               isolated_journal):
        """Not a new id, not the journal's tag — the runctx id.

        This is the whole join: the script, ``read_run_log(run_id)`` and
        the artifact rows have to name the same run or none of the three
        can be connected to the others.
        """
        from spacr.run_journal import open_run
        from spacr.runctx import run_context

        seen = {}
        with open_run("mask", {"src": str(tmp_path)}) as run:
            with run_context("mask", {"src": str(tmp_path)}) as context:
                seen["id"] = context.run_id
            run.set_status("success")

        record = macro.read_macro(macro.macro_path(run.dir))
        step = record["steps"][0]
        assert step["run_id"] == seen["id"]
        assert step["run_id_source"] == "runctx"

    def test_the_id_still_joins_the_run_log(self, tmp_path, isolated_journal):
        """`read_run_log(macro run id)` returns that run's lines."""
        from spacr.run_journal import open_run
        from spacr.runctx import read_run_log, run_context

        with open_run("mask", {"src": str(tmp_path)}) as run:
            with run_context("mask", {"src": str(tmp_path)}) as context:
                context.log.info("a line only this run wrote")
            run.set_status("success")

        record = macro.read_macro(macro.macro_path(run.dir))
        lines = read_run_log(record["steps"][0]["run_id"])
        assert any("a line only this run wrote" in entry["message"]
                   for entry in lines)

    def test_a_run_with_no_run_context_falls_back_and_says_so(
            self, tmp_path, isolated_journal):
        """Honest about a join it cannot make, rather than inventing an id."""
        from spacr.run_journal import open_run
        with open_run("mask", {"src": str(tmp_path)}) as run:
            run.set_status("success")
        record = macro.read_macro(macro.macro_path(run.dir))
        assert record["steps"][0]["run_id_source"] == "journal"
        assert record["steps"][0]["run_id"]

    def test_the_header_carries_version_id_and_hash(self, tmp_path,
                                                    isolated_journal):
        from spacr.artifacts import settings_hash
        from spacr.version import get_version

        recording = macro.begin_recording("mask", {"src": str(tmp_path)})
        step = macro.finish_recording(recording, status="success")
        header = macro.current_macro().source().split('"""')[0]

        assert get_version() in header
        assert step.settings_hash in header
        assert step.run_id in header
        assert step.settings_hash == settings_hash(step.settings)

    def test_the_settings_hash_is_the_one_the_artifacts_carry(
            self, tmp_path, isolated_journal):
        """Same digest, so a macro and an artifact row can be matched."""
        from spacr.artifacts import settings_hash
        resolved, _, _ = macro.explicit_settings("mask", {"src": str(tmp_path)})
        recording = macro.begin_recording("mask", {"src": str(tmp_path)})
        step = macro.finish_recording(recording, status="success")
        assert step.settings_hash == settings_hash(resolved)


# ---------------------------------------------------------------------------
# The machine-readable half
# ---------------------------------------------------------------------------

class TestMachineReadable:
    """`read_macro` parses a script it does not trust, and never runs it."""

    def test_reading_a_macro_does_not_execute_it(self, tmp_path,
                                                 isolated_journal):
        recording = macro.begin_recording("mask", {"src": str(tmp_path)})
        macro.finish_recording(recording, status="success")
        path = tmp_path / "hostile.py"
        source = macro.current_macro().source()
        path.write_text(
            source + "\n\nraise SystemExit('read_macro executed the file')\n"
            "import os as _os; _os.remove(__file__)\n")
        record = macro.read_macro(path)
        assert record["modules"] == ["mask"]
        assert path.is_file()

    def test_a_file_that_is_not_a_macro_says_so(self, tmp_path):
        path = tmp_path / "plain.py"
        path.write_text("x = 1\n")
        with pytest.raises(macro.MacroError):
            macro.read_macro(path)

    def test_a_newer_schema_is_refused_not_guessed(self, tmp_path,
                                                   isolated_journal):
        recording = macro.begin_recording("mask", {"src": str(tmp_path)})
        macro.finish_recording(recording, status="success")
        path = tmp_path / "future.py"
        path.write_text(macro.current_macro().source().replace(
            f"'schema': {macro.MACRO_SCHEMA},",
            f"'schema': {macro.MACRO_SCHEMA + 99},"))
        with pytest.raises(macro.MacroError, match="schema"):
            macro.read_macro(path)

    def test_the_record_has_what_a_methods_section_needs(self, tmp_path,
                                                         isolated_journal):
        """The exporter's input contract, asserted key by key."""
        recording = macro.begin_recording(
            "mask", {"src": str(tmp_path), "cell_diameter": 17})
        macro.finish_recording(recording, status="success")
        record = macro.read_macro(macro.current_macro().path)

        assert record["schema"] == macro.MACRO_SCHEMA
        assert record["spacr_version"]
        step = record["steps"][0]
        for key in ("index", "module", "entry", "run_id", "settings_hash",
                    "settings", "defaulted", "user_set", "project_root",
                    "status", "elapsed_s", "outputs", "spacr_version"):
            assert key in step, f"the record has no {key!r}"
        assert step["settings"]["cell_diameter"] == 17
        assert json.loads(macro.to_json(record))["macro_id"] == \
            record["macro_id"]
        assert "mask" in macro.summarise(record)

    def test_outputs_are_the_ports_the_module_declares(self, tmp_path,
                                                       isolated_journal):
        """What the step produced, taken from spacr.ports rather than guessed."""
        root = tmp_path / "plate"
        (root / "merged").mkdir(parents=True)
        (root / "merged" / "field.npy").write_bytes(b"x")
        recording = macro.begin_recording("mask", {"src": str(root)})
        step = macro.finish_recording(recording, status="success")
        assert any(path.endswith("merged") for path in step.outputs)

    def test_the_cheap_output_lookup_agrees_with_the_globbing_one(
            self, tmp_path, isolated_journal):
        """The one line borrowed from `resolve_port` still means the same.

        `_outputs` computes a port's location itself instead of resolving
        every glob, because a mask project's merged/*.npy port matches ten
        thousand files and the record only wants the folder. This is the
        guard against that shortcut drifting away from what
        `spacr.ports.declared_outputs` would have said.
        """
        from spacr.ports import declared_outputs, known_modules
        root = tmp_path / "plate"
        for folder in ("merged", "masks", "measurements", "datasets"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "measurements" / "measurements.db").write_bytes(b"")
        settings = {"src": str(root)}
        for module in known_modules():
            expected = tuple(dict.fromkeys(
                resolved.location
                for resolved in declared_outputs(module, root=str(root))
                if os.path.exists(resolved.location)))
            assert macro._outputs(module, settings) == expected, module


# ---------------------------------------------------------------------------
# Contracts the renderer depends on
# ---------------------------------------------------------------------------

class TestContracts:
    """What the emitted call assumes about the modules it calls."""

    def test_every_entry_point_takes_settings_first(self):
        """`func(SETTINGS)` is correct for every shipped module.

        The recorder emits a single positional argument, which is also how
        ``spacr.qt.bridge.PipelineWorker`` calls a pipeline. Read the real
        signatures out of the source rather than importing 25 modules, and
        fail here — loudly, in one place — if one ever stops taking the
        settings dict first.
        """
        from spacr.validate import APP_FUNCTIONS
        offenders = []
        for key, dotted in sorted(APP_FUNCTIONS.items()):
            module_path, _, func = dotted.rpartition(".")
            path = Path(REPO_ROOT, *module_path.split(".")).with_suffix(".py")
            if not path.is_file():
                offenders.append(f"{key}: no module at {path}")
                continue
            tree = ast.parse(path.read_text())
            found = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == func]
            if not found:
                offenders.append(f"{key}: {dotted} not found")
                continue
            args = [argument.arg for argument in found[0].args.args]
            if not args or args[0] != "settings":
                offenders.append(f"{key}: {dotted} takes {args}")
        assert not offenders, "\n".join(offenders)

    def test_entry_resolution_matches_the_gui(self):
        """The script imports what the Run button ran."""
        from spacr.validate import APP_FUNCTIONS
        for key in ("mask", "measure", "classify", "regression"):
            module_path, func = macro.entry_for(key)
            assert f"{module_path}.{func}" == APP_FUNCTIONS[key]

    def test_recording_never_takes_a_run_down(self, tmp_path, monkeypatch,
                                              isolated_journal):
        """Every failure mode inside the recorder costs a log line, not the run."""
        monkeypatch.setattr(macro, "_build_step",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("recorder is broken")))
        recording = macro.begin_recording("mask", {"src": str(tmp_path)})
        assert macro.finish_recording(recording, status="success") is None
        assert macro.finish_recording(None) is None

    def test_the_journal_hook_is_the_only_wiring(self):
        """One hook, in one place, and this is where it is.

        The recorder is reached from ``spacr.run_journal.open_run`` and
        nowhere else — the seam the Qt GUI, the Tk GUI and the CLI all
        launch runs through. If a second call site appears, either this
        assertion or the "one script per chain" behaviour is wrong.
        """
        import spacr
        root = Path(spacr.__file__).parent
        callers = set()
        for path in root.rglob("*.py"):
            if path.name in ("macro.py",):
                continue
            text = path.read_text(errors="ignore")
            if "begin_recording" in text or "finish_recording" in text:
                callers.add(path.relative_to(root).as_posix())
        assert callers == {"run_journal.py"}, callers
