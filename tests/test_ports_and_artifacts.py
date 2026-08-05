"""Real tests for the typed ports (:mod:`spacr.ports`) and the artifact
registry (:mod:`spacr.artifacts`).

Not smoke tests. Every assertion here is about behaviour a downstream feature
depends on: a registration that round-trips through SQLite, staleness that
flips when an upstream setting changes, a ``downstream_of`` that is
transitive, and a missing input that comes back with a reason a user can act
on.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import artifacts, ports  # noqa: E402
from spacr.database_concurrency import DatabaseConfigurationError  # noqa: E402
from spacr.validate import ERROR, WARNING  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """No test may inherit a shared-registry override from the environment."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


@pytest.fixture
def restore_ports():
    """Snapshot and restore the global port declarations around a test."""
    saved = dict(ports.PORTS)
    yield ports.PORTS
    ports.PORTS.clear()
    ports.PORTS.update(saved)


def make_project(root: Path, *, merged: int = 2, planes: int = 3,
                 db_tables=("png_list", "cell"), crops: bool = False) -> str:
    """Build a plate folder shaped the way the mask pipeline leaves one."""
    root.mkdir(parents=True, exist_ok=True)
    if merged:
        (root / "merged").mkdir(exist_ok=True)
        for index in range(merged):
            np.save(root / "merged" / f"plate1_A01_{index}.npy",
                    np.zeros((6, 6, planes), dtype=np.uint16))
    if db_tables is not None:
        (root / "measurements").mkdir(exist_ok=True)
        connection = sqlite3.connect(root / "measurements" / "measurements.db")
        for table in db_tables:
            connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')
            connection.execute(f'INSERT INTO "{table}" VALUES (1)')
        connection.commit()
        connection.close()
    if crops:
        crop_dir = root / "data" / "A01" / "cell_png"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / "object_1.png").write_bytes(b"\x89PNG")
    return str(root)


def put(registry, name, kind, *, module="mask", role="", inputs=(),
        settings=None, content=None):
    """Register a one-file artifact under the registry's project."""
    path = Path(registry.project) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content is not None else name, encoding="utf-8")
    return registry.register(module=module, kind=kind, role=role,
                             path=str(path), inputs=inputs, settings=settings)


# ===========================================================================
# spacr.ports — declarations
# ===========================================================================

def test_shape_contract_describes_itself():
    full = ports.ShapeContract(ndim=3, min_planes=2, dtype="uint16")
    assert full.describe() == "3-D, at least 2 planes, dtype uint16"
    assert ports.ShapeContract().describe() == "any array"


def test_port_relative_covers_every_shape_of_declaration():
    assert ports.Port("k", "r", "merged", "*.npy").relative() == "merged/*.npy"
    assert ports.Port("k", "r", "measurements/db").relative() == "measurements/db"
    assert ports.Port("k", "r").relative() == "."


def test_module_ports_port_lookup_finds_inputs_and_outputs():
    spec = ports.module_ports("measure")
    assert spec.port("merged").kind == ports.MERGED_ARRAYS
    assert spec.port("db").kind == ports.MEASUREMENTS_DB
    with pytest.raises(KeyError, match="no port with role"):
        spec.port("nonexistent")


def test_register_module_ports_refuses_bad_declarations(restore_ports):
    with pytest.raises(ValueError, match="non-empty key"):
        ports.register_module_ports(ports.ModulePorts(key="  "))
    with pytest.raises(ValueError, match="already declared"):
        ports.register_module_ports(ports.ModulePorts(key="measure"))
    with pytest.raises(ValueError, match="roles must be unique"):
        ports.register_module_ports(ports.ModulePorts(
            key="dupe",
            consumes=(ports.Port("a", "same"),),
            produces=(ports.Port("b", "same"),)))
    assert "dupe" not in ports.PORTS


def test_register_module_ports_lowercases_and_overwrites(restore_ports):
    stored = ports.register_module_ports(ports.ModulePorts(
        key="MyPlugin", summary="from a plugin",
        produces=(ports.Port(ports.CROPS, "out", "out"),)))
    assert stored.key == "myplugin"
    assert ports.module_ports("MYPLUGIN") is stored
    assert "myplugin" in ports.known_modules()

    replacement = ports.register_module_ports(ports.ModulePorts(
        key="myplugin", summary="second version"), overwrite=True)
    assert ports.module_ports("myplugin").summary == "second version"
    assert replacement.produces == ()


def test_module_ports_accepts_aliases_and_names_the_alternatives():
    assert ports.module_ports("measure_crop").key == "measure"
    assert ports.module_ports("generate_masks").key == "mask"
    with pytest.raises(ports.UnknownModule) as excinfo:
        ports.module_ports("does_not_exist")
    assert "measure" in str(excinfo.value)


def test_the_declared_graph_matches_the_real_pipeline():
    assert "mask" in ports.producers_of(ports.MERGED_ARRAYS)
    assert ports.producers_of(ports.RAW_IMAGES) == ()
    assert "measure" in ports.consumers_of(ports.MERGED_ARRAYS)
    # classify consumes crops optionally, measure produces them.
    assert "classify" in ports.consumers_of(ports.CROPS)
    assert "classify" not in ports.consumers_of(ports.CROPS, required_only=True)
    assert "measure" in ports.next_modules("mask")
    assert "mask" in ports.upstream_modules("measure")
    assert "measure" not in ports.next_modules("measure")


# ===========================================================================
# spacr.ports — path resolution
# ===========================================================================

def test_project_root_reuses_the_existing_conventions(tmp_path):
    plate = tmp_path / "plate1"
    plate.mkdir()
    assert ports.project_root(None) == ""
    assert ports.project_root({}) == ""
    assert ports.project_root({"src": ""}) == ""
    assert ports.project_root({"src": None}) == ""
    assert ports.project_root({"src": []}) == ""
    assert ports.project_root(str(plate)) == str(plate)
    # a list of plates: the first names the first project
    assert ports.project_root({"src": [str(plate), "/other"]}) == str(plate)
    # src already pointing at merged/ means the project is its parent
    assert ports.project_root({"src": str(plate / "merged")}) == str(plate)
    # module-specific source keys
    assert ports.project_root({"dst": str(plate)}, "foreign") == str(plate)
    assert ports.project_root({"inputs": str(plate)}, "external_masks") == str(plate)


def test_resolve_port_globs_alternatives_and_filters_extensions(tmp_path):
    root = tmp_path / "plate"
    (root / "orig").mkdir(parents=True)
    (root / "a.tif").write_bytes(b"x")
    (root / "notes.txt").write_text("not an image")
    (root / "orig" / "b.TIF").write_bytes(b"x")

    images = ports.module_ports("mask").port("images")
    resolved = ports.resolve_port(images, str(root))
    assert resolved.exists and resolved.count == 2
    assert {os.path.basename(p) for p in resolved.paths} == {"a.tif", "b.TIF"}
    assert resolved.kind == ports.RAW_IMAGES
    assert resolved.role == "images"
    assert resolved.location == str(root)

    # a pattern-less port is present or absent, nothing else
    db_port = ports.module_ports("measure").port("db")
    absent = ports.resolve_port(db_port, str(root))
    assert not absent.exists and absent.count == 0 and absent.paths == ()
    make_project(root)
    present = ports.resolve_port(db_port, str(root))
    assert present.exists and present.count == 1


def test_declared_inputs_and_outputs_resolve_against_the_project(tmp_path):
    root = make_project(tmp_path / "plate", crops=True)
    outputs = {r.role: r for r in ports.declared_outputs("measure", root=root)}
    assert outputs["db"].exists
    assert outputs["crops"].exists and outputs["crops"].count == 1
    inputs = {r.role: r for r in ports.declared_inputs("measure", {"src": root})}
    assert inputs["merged"].count == 2


# ===========================================================================
# spacr.ports — readiness
# ===========================================================================

def test_a_missing_input_is_reported_with_a_reason_and_a_fix(tmp_path):
    empty = tmp_path / "fresh"
    empty.mkdir()
    readiness = ports.check_ready("measure", {"src": str(empty)})

    assert not readiness
    assert readiness.ok is False
    assert readiness.satisfied == ()
    problem = readiness.errors[0]
    assert problem.setting == "merged"
    assert "no merged-arrays at" in problem.message
    assert str(empty / "merged" / "*.npy") in problem.message
    # the fix names the module that would produce it, not a generic hint
    assert "Run mask" in problem.fix
    assert "measure cannot run" in readiness.reason
    assert "NOT READY" in str(readiness)


def test_readiness_reason_counts_the_other_errors(tmp_path, restore_ports):
    root = tmp_path / "plate"
    root.mkdir()
    ports.register_module_ports(ports.ModulePorts(
        key="twoinputs",
        consumes=(ports.Port(ports.MERGED_ARRAYS, "a", "merged", "*.npy"),
                  ports.Port(ports.MEASUREMENTS_DB, "b",
                             "measurements/measurements.db"))))
    readiness = ports.check_ready("twoinputs", str(root))
    assert len(readiness.errors) == 2
    assert readiness.reason.endswith("(+1 more)")


def test_an_optional_input_downgrades_to_a_warning(tmp_path):
    root = make_project(tmp_path / "plate", crops=False)
    readiness = ports.check_ready("classify", {"src": root})
    assert readiness.ok is True
    assert readiness.satisfied == ("db",)
    assert [p.setting for p in readiness.warnings] == ["crops"]
    assert readiness.warnings[0].severity == WARNING
    assert "READY" in format_first_line(readiness)


def format_first_line(readiness):
    """First line of the rendered readiness report."""
    return ports.format_readiness(readiness).splitlines()[0]


def test_no_project_and_a_nonexistent_project_are_told_apart(tmp_path):
    nothing = ports.check_ready("measure", {})
    assert not nothing
    assert "no project folder" in nothing.errors[0].message
    assert "'src'" in nothing.errors[0].fix

    foreign_like = ports.check_ready("mask", {"src": str(tmp_path / "gone")})
    assert not foreign_like
    assert "does not exist" in foreign_like.errors[0].message


def test_the_source_key_named_in_the_fix_follows_the_module(restore_ports):
    ports.register_module_ports(ports.ModulePorts(key="foreign"))
    readiness = ports.check_ready("foreign", {})
    assert "'dst'" in readiness.errors[0].fix


def test_a_ready_project_says_what_it_found(tmp_path):
    root = make_project(tmp_path / "plate", crops=True)
    readiness = ports.check_ready("measure", {"src": root})
    assert readiness
    assert readiness.satisfied == ("merged",)
    assert readiness.problems == ()
    assert readiness.reason == f"measure can run in {root}: merged"


def test_a_module_with_no_declared_inputs_is_ready(tmp_path, restore_ports):
    root = tmp_path / "plate"
    root.mkdir()
    ports.register_module_ports(ports.ModulePorts(key="noinputs"))
    readiness = ports.check_ready("noinputs", str(root))
    assert readiness
    assert "nothing required" in readiness.reason


def test_too_few_matches_is_its_own_message(tmp_path, restore_ports):
    root = make_project(tmp_path / "plate", merged=1)
    ports.register_module_ports(ports.ModulePorts(
        key="needstwo",
        consumes=(ports.Port(ports.MERGED_ARRAYS, "merged", "merged", "*.npy",
                             min_count=2),)))
    readiness = ports.check_ready("needstwo", str(root))
    assert not readiness
    assert "only 1 merged-arrays" in readiness.errors[0].message
    assert "2 required" in readiness.errors[0].message
    assert "Re-run mask" in readiness.errors[0].fix


def test_a_port_with_no_producer_gets_a_generic_fix(tmp_path, restore_ports):
    root = tmp_path / "plate"
    root.mkdir()
    ports.register_module_ports(ports.ModulePorts(
        key="orphan", consumes=(ports.Port("nobody-makes-this", "x", "sub"),)))
    readiness = ports.check_ready("orphan", str(root))
    assert "Put the nobody-makes-this there" in readiness.errors[0].fix


# ---- shape contract -------------------------------------------------------

def _readiness_for_merged(root):
    return ports.check_ready("measure", {"src": str(root)})


def test_a_truncated_merged_array_is_caught_from_its_header(tmp_path):
    root = tmp_path / "plate"
    make_project(root, merged=1)
    victim = root / "merged" / "plate1_A01_0.npy"
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])
    readiness = _readiness_for_merged(root)
    assert not readiness
    assert "is truncated" in readiness.errors[0].message


def test_a_file_that_is_not_an_npy_at_all_is_caught(tmp_path):
    root = tmp_path / "plate"
    make_project(root, merged=0)
    (root / "merged").mkdir(parents=True, exist_ok=True)
    (root / "merged" / "broken.npy").write_bytes(b"this is not numpy")
    readiness = _readiness_for_merged(root)
    assert not readiness
    assert "not a readable .npy" in readiness.errors[0].message


def test_a_two_dimensional_array_fails_the_rank_contract(tmp_path):
    root = tmp_path / "plate"
    make_project(root, merged=0)
    (root / "merged").mkdir(parents=True, exist_ok=True)
    np.save(root / "merged" / "flat.npy", np.zeros((4, 4), dtype=np.uint16))
    readiness = _readiness_for_merged(root)
    assert not readiness
    assert "expected 3 axes" in readiness.errors[0].message


def test_a_merged_array_with_one_plane_has_no_mask(tmp_path):
    root = tmp_path / "plate"
    make_project(root, merged=1, planes=1)
    readiness = _readiness_for_merged(root)
    assert not readiness
    assert "1 plane(s), at least 2 are needed" in readiness.errors[0].message


def test_the_dtype_contract_accepts_and_rejects(tmp_path, restore_ports):
    root = tmp_path / "plate"
    make_project(root, merged=1)
    ports.register_module_ports(ports.ModulePorts(
        key="wants_float",
        consumes=(ports.Port(ports.MERGED_ARRAYS, "merged", "merged", "*.npy",
                             shape=ports.ShapeContract(dtype="f8")),)))
    readiness = ports.check_ready("wants_float", str(root))
    assert not readiness
    assert "is u2, expected f8" in readiness.errors[0].message

    ports.register_module_ports(ports.ModulePorts(
        key="wants_float",
        consumes=(ports.Port(ports.MERGED_ARRAYS, "merged", "merged", "*.npy",
                             shape=ports.ShapeContract(dtype="u2")),)),
        overwrite=True)
    assert ports.check_ready("wants_float", str(root))


def test_a_zero_dimensional_array_does_not_trip_the_plane_check(
        tmp_path, restore_ports):
    root = tmp_path / "plate"
    (root / "merged").mkdir(parents=True)
    np.save(root / "merged" / "scalar.npy", np.array(7, dtype=np.uint16))
    ports.register_module_ports(ports.ModulePorts(
        key="planes_only",
        consumes=(ports.Port(ports.MERGED_ARRAYS, "merged", "merged", "*.npy",
                             shape=ports.ShapeContract(min_planes=2)),)))
    assert ports.check_ready("planes_only", str(root))


def test_only_the_sampled_arrays_are_read(tmp_path):
    root = tmp_path / "plate"
    make_project(root, merged=4)
    # break the last array alphabetically; sample=1 must not look at it
    victim = root / "merged" / "plate1_A01_3.npy"
    victim.write_bytes(b"not numpy")
    assert ports.check_ready("measure", {"src": str(root)}, sample=1)
    assert not ports.check_ready("measure", {"src": str(root)}, sample=4)


# ---- database table contract ---------------------------------------------

def test_a_database_without_png_list_blocks_classify(tmp_path):
    root = make_project(tmp_path / "plate", db_tables=("cell",))
    readiness = ports.check_ready("classify", {"src": root})
    assert not readiness
    assert "has no 'png_list' table" in readiness.errors[0].message
    assert "save_png" in readiness.errors[0].fix


def test_an_empty_png_list_blocks_classify_too(tmp_path):
    root = Path(make_project(tmp_path / "plate", db_tables=()))
    connection = sqlite3.connect(root / "measurements" / "measurements.db")
    connection.execute("CREATE TABLE png_list (value INTEGER)")
    connection.commit()
    connection.close()
    readiness = ports.check_ready("classify", {"src": str(root)})
    assert not readiness
    assert "exists but is empty" in readiness.errors[0].message


def test_something_that_is_not_a_database_is_named_as_such(tmp_path):
    root = tmp_path / "plate"
    make_project(root, db_tables=None)
    (root / "measurements").mkdir(parents=True, exist_ok=True)
    (root / "measurements" / "measurements.db").write_text("plain text")
    readiness = ports.check_ready("classify", {"src": str(root)})
    assert not readiness
    assert "is not a SQLite database" in readiness.errors[0].message


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_an_unreadable_database_is_reported_not_crashed(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    database = root / "measurements" / "measurements.db"
    os.chmod(database, 0o000)
    try:
        readiness = ports.check_ready("classify", {"src": str(root)})
    finally:
        os.chmod(database, 0o644)
    assert not readiness
    assert "cannot be opened" in readiness.errors[0].message


# ---- rendering ------------------------------------------------------------

def test_format_readiness_shows_errors_warnings_and_artifacts(tmp_path):
    root = make_project(tmp_path / "plate", merged=0)
    registry = artifacts.open_registry(root)
    artifacts.register_run_outputs("measure", {"src": root}, roots=[root])
    readiness = ports.check_ready("classify", {"src": root}, registry=registry)
    text = ports.format_readiness(readiness)
    assert text.startswith("READY: classify in ")
    assert "inputs found: db" in text
    assert "warning [crops]" in text
    assert "artifacts: " in text

    broken = ports.check_ready("measure", {"src": root})
    rendered = ports.format_readiness(broken)
    assert "error   [merged]" in rendered
    assert "fix: " in rendered


def test_describe_ports_renders_every_facet_of_a_declaration():
    text = ports.describe_ports("mask")
    assert text.startswith("mask — ")
    assert "merged: merged-arrays at merged/*.npy" in text
    assert "[optional]" in text
    assert "shape: 3-D, at least 2 planes" in text
    assert "tables: png_list" in ports.describe_ports("classify")
    assert "(nothing declared)" in ports.describe_ports("recruitment")


def test_describe_ports_without_a_summary_or_descriptions(restore_ports):
    ports.register_module_ports(ports.ModulePorts(
        key="bare", produces=(ports.Port(ports.CROPS, "out", "out"),)))
    lines = ports.describe_ports("bare").splitlines()
    assert lines[0] == "bare"
    assert lines[-1] == "    out: crops at out"


# ===========================================================================
# spacr.artifacts — provenance primitives
# ===========================================================================

def test_material_settings_drops_only_what_cannot_change_a_number():
    settings = {"src": "/data/plate1", "n_jobs": 8, "verbose": True,
                "cell_diameter": 30, "torch": "2.4.0"}
    material = artifacts.material_settings(settings)
    assert material == {"cell_diameter": 30}
    assert artifacts.material_settings(None) == {}
    assert artifacts.material_settings({}) == {}


def test_settings_hash_ignores_cosmetics_and_notices_science():
    base = {"src": "/a", "cell_diameter": 30}
    assert artifacts.settings_hash(base) == artifacts.settings_hash(
        {"src": "/somewhere/else", "n_jobs": 4, "cell_diameter": 30})
    assert artifacts.settings_hash(base) != artifacts.settings_hash(
        {"src": "/a", "cell_diameter": 60})


def test_content_fingerprint_covers_files_folders_and_absence(tmp_path):
    missing = artifacts.content_fingerprint(tmp_path / "nope")
    assert missing.method == "missing" and not missing

    target = tmp_path / "one.txt"
    target.write_text("hello")
    whole = artifacts.content_fingerprint(target)
    assert whole.method == "sha256" and whole.n_files == 1
    assert whole.size_bytes == 5 and bool(whole)

    sampled = artifacts.content_fingerprint(target, full_hash_limit=2)
    assert sampled.method == "sampled"
    assert sampled.digest != whole.digest

    folder = tmp_path / "tree"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.txt").write_text("aaa")
    (folder / "b.txt").write_text("bb")
    os.symlink(folder / "b.txt", folder / "link.txt")
    os.symlink(folder / "sub", folder / "linkdir")
    tree = artifacts.content_fingerprint(folder)
    assert tree.method == "tree"
    assert tree.n_files == 2          # symlinks are not counted
    assert tree.size_bytes == 5

    (folder / "b.txt").write_text("changed")
    assert artifacts.content_fingerprint(folder).digest != tree.digest


def test_registry_path_honours_the_shared_override(tmp_path, monkeypatch):
    assert artifacts.registry_path(tmp_path) == str(
        tmp_path / artifacts.ARTIFACTS_DB_NAME)
    with pytest.raises(ValueError, match="nowhere to keep"):
        artifacts.registry_path(None)
    monkeypatch.setenv(artifacts.ARTIFACTS_DB_ENV, str(tmp_path / "shared.db"))
    assert artifacts.registry_path(None) == str(tmp_path / "shared.db")
    assert artifacts.registry_path("/other/project") == str(tmp_path / "shared.db")


# ===========================================================================
# spacr.artifacts — the registry
# ===========================================================================

def test_a_registration_round_trips_through_sqlite(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    settings = {"src": str(root), "cell_diameter": 30, "n_jobs": 8}

    artifact = registry.register(
        module="mask", kind=ports.MERGED_ARRAYS, role="merged",
        path=root / "merged", settings=settings, run_id="run-1",
        extra={"fields": 2})

    fetched = registry.get(artifact.artifact_id)
    assert fetched == artifact
    assert fetched.project == str(root)
    assert fetched.module == "mask"
    assert fetched.kind == ports.MERGED_ARRAYS
    assert fetched.role == "merged"
    assert fetched.path == str(root / "merged")
    assert fetched.run_id == "run-1"
    assert fetched.settings == {"cell_diameter": 30}
    assert fetched.settings_hash == artifacts.settings_hash(settings)
    assert fetched.spacr_version
    assert fetched.fingerprint_method == "tree"
    assert fetched.n_files == 2
    assert fetched.created_ns > 0 and fetched.created_utc.endswith("+00:00")
    assert fetched.status == artifacts.STATUS_COMPLETE
    assert fetched.extra == {"fields": 2}
    assert fetched.exists
    assert fetched.schema_version == artifacts.SCHEMA_VERSION

    payload = fetched.to_dict()
    assert payload["artifact_id"] == artifact.artifact_id
    assert payload["inputs"] == []
    assert artifact.artifact_id in str(artifact)

    # a second, independent Registry object over the same file sees it
    assert artifacts.open_registry(root).get(artifact.artifact_id) == artifact
    assert (root / artifacts.ARTIFACTS_DB_NAME).is_file()


def test_the_registry_table_has_the_declared_schema(tmp_path):
    root = tmp_path / "plate"
    root.mkdir()
    artifacts.open_registry(root)
    connection = sqlite3.connect(root / artifacts.ARTIFACTS_DB_NAME)
    try:
        columns = [row[1] for row in connection.execute(
            "PRAGMA table_info(artifacts)")]
        edges = [row[1] for row in connection.execute(
            "PRAGMA table_info(artifact_inputs)")]
    finally:
        connection.close()
    assert columns == list(artifacts._COLUMNS)
    assert edges == ["artifact_id", "input_id", "position"]
    for required in ("module", "settings_hash", "spacr_version", "created_utc",
                     "path", "fingerprint"):
        assert required in columns


def test_registering_the_same_content_twice_updates_one_row(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    first = put(registry, "out.txt", ports.CROPS, settings={"a": 1})
    second = registry.register(module="mask", kind=ports.CROPS,
                               path=root / "out.txt", settings={"a": 1},
                               run_id="second-run")
    assert second.artifact_id == first.artifact_id
    assert second.created_ns > first.created_ns
    assert len(registry.all()) == 1
    assert registry.get(first.artifact_id).run_id == "second-run"


def test_register_refuses_an_incomplete_record(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    with pytest.raises(ValueError, match="module that produced it"):
        registry.register(module=" ", kind="k", path="/x")
    with pytest.raises(ValueError, match="needs a kind"):
        registry.register(module="mask", kind="", path="/x")
    with pytest.raises(ValueError, match="needs a path"):
        registry.register(module="mask", kind="k", path="")


def test_register_accepts_a_precomputed_digest_and_fingerprint(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    artifact = registry.register(
        module="mask", kind=ports.MASKS, path=root / "merged",
        settings_digest="deadbeef",
        fingerprint=artifacts.Fingerprint("cafe", "precomputed", 12, 3))
    assert artifact.settings_hash == "deadbeef"
    assert artifact.fingerprint == "cafe"
    assert artifact.fingerprint_method == "precomputed"
    assert artifact.n_files == 3


def test_inputs_are_deduplicated_and_accept_artifacts_or_ids(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    upstream = put(registry, "up.txt", ports.MERGED_ARRAYS)
    child = put(registry, "down.txt", ports.MEASUREMENTS_DB,
                module="measure",
                inputs=[upstream, upstream.artifact_id, upstream])
    assert child.inputs == (upstream.artifact_id,)
    assert registry.get(child.artifact_id).inputs == (upstream.artifact_id,)


def test_queries_filter_by_kind_project_and_module(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    merged = put(registry, "m.txt", ports.MERGED_ARRAYS)
    database = put(registry, "d.txt", ports.MEASUREMENTS_DB, module="measure")
    other = registry.register(module="measure", kind=ports.MEASUREMENTS_DB,
                              path=root / "d.txt", project=tmp_path / "second",
                              settings={"x": 1})

    assert [a.artifact_id for a in registry.by_kind(ports.MERGED_ARRAYS)] == [
        merged.artifact_id]
    assert {a.artifact_id for a in registry.by_kind(ports.MEASUREMENTS_DB)} == {
        database.artifact_id, other.artifact_id}
    assert [a.artifact_id for a in registry.by_kind(
        ports.MEASUREMENTS_DB, project=str(root))] == [database.artifact_id]
    assert registry.by_kind(ports.MERGED_ARRAYS, module="measure") == []
    assert len(registry.by_kind(ports.MEASUREMENTS_DB, limit=1)) == 1

    assert {a.artifact_id for a in registry.by_project()} == {
        merged.artifact_id, database.artifact_id}
    assert {a.artifact_id for a in registry.by_project("")} == {
        merged.artifact_id, database.artifact_id, other.artifact_id}
    assert [a.artifact_id for a in registry.by_project(
        str(tmp_path / "second"))] == [other.artifact_id]
    assert [a.artifact_id for a in registry.by_project(
        kind=ports.MERGED_ARRAYS)] == [merged.artifact_id]
    assert len(registry.by_project("", limit=2)) == 2

    assert registry.latest(ports.MERGED_ARRAYS).artifact_id == merged.artifact_id
    assert registry.latest(ports.MERGED_ARRAYS, role="nope") is None
    assert registry.latest(ports.MEASUREMENTS_DB,
                           path=str(root / "d.txt")) is not None
    assert registry.get("nosuchid") is None
    assert len(registry.all(limit=1)) == 1


def test_forget_removes_the_row_and_its_outgoing_edges(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    upstream = put(registry, "u.txt", ports.MERGED_ARRAYS)
    child = put(registry, "c.txt", ports.MEASUREMENTS_DB, module="measure",
                inputs=[upstream])
    assert registry.forget(child) == 1
    assert registry.get(child.artifact_id) is None
    assert registry.downstream_of(upstream) == []
    assert registry.forget("nosuchid") == 0


# ---- the DAG --------------------------------------------------------------

def _chain(registry):
    """Register raw -> merged -> db -> model, each naming the last as input."""
    raw = put(registry, "raw.txt", ports.RAW_IMAGES)
    merged = put(registry, "merged.txt", ports.MERGED_ARRAYS, inputs=[raw])
    database = put(registry, "db.txt", ports.MEASUREMENTS_DB, module="measure",
                   inputs=[merged])
    model = put(registry, "model.txt", ports.MODEL_WEIGHTS, module="classify",
                inputs=[database])
    return raw, merged, database, model


def test_downstream_of_is_transitive(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    raw, merged, database, model = _chain(registry)

    everything = {a.artifact_id for a in registry.downstream_of(raw)}
    assert everything == {merged.artifact_id, database.artifact_id,
                          model.artifact_id}

    immediate = registry.downstream_of(raw, transitive=False)
    assert [a.artifact_id for a in immediate] == [merged.artifact_id]

    assert {a.artifact_id for a in registry.downstream_of(database)} == {
        model.artifact_id}
    assert registry.downstream_of(model) == []


def test_upstream_of_is_transitive_the_other_way(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    raw, merged, database, model = _chain(registry)

    assert [a.artifact_id for a in registry.upstream_of(model)] == [
        database.artifact_id]
    assert {a.artifact_id for a in registry.upstream_of(model, transitive=True)} == {
        database.artifact_id, merged.artifact_id, raw.artifact_id}
    assert registry.upstream_of(raw, transitive=True) == []


def test_a_diamond_reports_each_descendant_once(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    top = put(registry, "top.txt", ports.RAW_IMAGES)
    left = put(registry, "left.txt", ports.MERGED_ARRAYS, inputs=[top])
    right = put(registry, "right.txt", ports.CHANNEL_STACKS, inputs=[top])
    bottom = put(registry, "bottom.txt", ports.MEASUREMENTS_DB,
                 module="measure", inputs=[left, right])
    found = registry.downstream_of(top)
    assert len(found) == 3
    assert {a.artifact_id for a in found} == {
        left.artifact_id, right.artifact_id, bottom.artifact_id}
    # newest first
    assert found[0].artifact_id == bottom.artifact_id


def test_an_unregistered_input_is_skipped_by_the_walk(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    child = put(registry, "c.txt", ports.MEASUREMENTS_DB, module="measure",
                inputs=["ghost0000000000"])
    assert registry.upstream_of(child) == []


# ---- staleness ------------------------------------------------------------

def test_staleness_flips_when_an_upstream_setting_changes(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)

    merged = registry.register(
        module="mask", kind=ports.MERGED_ARRAYS, role="merged",
        path=root / "merged", settings={"cell_diameter": 30})
    database = registry.register(
        module="measure", kind=ports.MEASUREMENTS_DB, role="db",
        path=root / "measurements" / "measurements.db",
        settings={"cell_min_size": 10}, inputs=[merged])

    fresh = registry.is_stale(database)
    assert fresh.stale is False
    assert not fresh
    assert fresh.reasons == ()
    assert str(fresh).endswith("current")

    # Mask is re-run with a different diameter: same path, new settings hash,
    # so a NEW artifact now sits where the measurement's input used to.
    (root / "merged" / "plate1_A02_0.npy").write_bytes(b"x")
    reproduced = registry.register(
        module="mask", kind=ports.MERGED_ARRAYS, role="merged",
        path=root / "merged", settings={"cell_diameter": 60})
    assert reproduced.artifact_id != merged.artifact_id

    now = registry.is_stale(database)
    assert now.stale is True and bool(now)
    assert artifacts.CAUSE_UPSTREAM_SUPERSEDED in now.causes
    assert "was re-produced by mask after this" in now.reasons[0]
    assert "stale" in str(now)


def test_re_registering_the_same_upstream_also_makes_a_child_stale(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    upstream = put(registry, "u.txt", ports.MERGED_ARRAYS, settings={"d": 1})
    child = put(registry, "c.txt", ports.MEASUREMENTS_DB, module="measure",
                inputs=[upstream])
    assert not registry.is_stale(child)

    again = registry.register(module="mask", kind=ports.MERGED_ARRAYS,
                              path=root / "u.txt", settings={"d": 1})
    assert again.artifact_id == upstream.artifact_id
    staleness = registry.is_stale(child)
    assert staleness.stale
    assert staleness.causes == (artifacts.CAUSE_UPSTREAM_NEWER,)


def test_a_forgotten_input_makes_its_child_stale_with_a_reason(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    upstream = put(registry, "u.txt", ports.MERGED_ARRAYS)
    child = put(registry, "c.txt", ports.MEASUREMENTS_DB, module="measure",
                inputs=[upstream])
    registry.forget(upstream)
    staleness = registry.is_stale(child)
    assert staleness.stale
    assert staleness.causes == (artifacts.CAUSE_UPSTREAM_MISSING,)
    assert upstream.artifact_id in staleness.reasons[0]


def test_staleness_is_inherited_transitively(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    merged = put(registry, "m.txt", ports.MERGED_ARRAYS, settings={"d": 30})
    database = put(registry, "d.txt", ports.MEASUREMENTS_DB, module="measure",
                   inputs=[merged])
    model = put(registry, "w.txt", ports.MODEL_WEIGHTS, module="classify",
                inputs=[database])
    assert not registry.is_stale(model)

    # only the top of the chain is re-produced
    registry.register(module="mask", kind=ports.MERGED_ARRAYS,
                      path=root / "m.txt", settings={"d": 60})

    assert registry.is_stale(database).causes == (
        artifacts.CAUSE_UPSTREAM_SUPERSEDED,)
    inherited = registry.is_stale(model)
    assert inherited.stale
    assert inherited.causes == (artifacts.CAUSE_UPSTREAM_STALE,)
    assert "is itself stale" in inherited.reasons[0]


def test_changed_settings_alone_make_an_artifact_stale(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    artifact = put(registry, "m.txt", ports.MERGED_ARRAYS,
                   settings={"cell_diameter": 30})
    assert not registry.is_stale(artifact, settings={"cell_diameter": 30,
                                                     "n_jobs": 12})
    changed = registry.is_stale(artifact, settings={"cell_diameter": 60})
    assert changed.stale
    assert changed.causes == (artifacts.CAUSE_SETTINGS_CHANGED,)
    assert "the settings differ" in changed.reasons[0]


def test_an_unregistered_artifact_is_stale_and_missing(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    staleness = registry.is_stale("nosuchartifact")
    assert staleness.stale and staleness.missing
    assert staleness.causes == (artifacts.CAUSE_UNKNOWN,)
    assert "not in the registry" in str(staleness)


def test_a_deleted_file_is_missing_but_not_by_itself_stale(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    artifact = put(registry, "gone.txt", ports.CROPS)
    (root / "gone.txt").unlink()
    staleness = registry.is_stale(artifact)
    assert staleness.missing is True
    assert staleness.stale is False


def test_a_provenance_cycle_terminates(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    artifact = put(registry, "self.txt", ports.CROPS)
    # a caller that names the artifact as its own input
    looped = registry.register(module="mask", kind=ports.CROPS,
                               path=root / "self.txt",
                               inputs=[artifact.artifact_id])
    assert looped.artifact_id == artifact.artifact_id
    staleness = registry.is_stale(artifact)
    assert staleness.stale is False
    assert staleness.causes == ()


def test_the_cycle_guard_reports_itself_when_reached(tmp_path):
    registry = artifacts.open_registry(make_project(tmp_path / "plate"))
    artifact = put(registry, "x.txt", ports.CROPS)
    with registry._open() as connection:
        guarded = registry._staleness(connection, artifact.artifact_id, None,
                                      {artifact.artifact_id})
    assert guarded.causes == (artifacts.CAUSE_CYCLE,)
    assert guarded.stale is False


# ---- storage plumbing -----------------------------------------------------

def test_the_registry_uses_wal_on_a_local_filesystem(tmp_path):
    root = tmp_path / "plate"
    root.mkdir()
    artifacts.open_registry(root)
    connection = sqlite3.connect(root / artifacts.ARTIFACTS_DB_NAME)
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()
    assert str(mode).upper() == "WAL"


def test_a_network_filesystem_keeps_the_default_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "filesystem_type", lambda path: "nfs")
    root = tmp_path / "plate"
    root.mkdir()
    artifacts.open_registry(root)
    connection = sqlite3.connect(root / artifacts.ARTIFACTS_DB_NAME)
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()
    assert str(mode).upper() != "WAL"


def test_a_refused_journal_mode_falls_back_instead_of_failing(tmp_path,
                                                              monkeypatch):
    real_connect = artifacts.connect

    def refuse_wal(path, **kwargs):
        if kwargs.pop("journal_mode", None) is not None:
            raise DatabaseConfigurationError("SQLite kept journal_mode=DELETE")
        return real_connect(path, **kwargs)

    monkeypatch.setattr(artifacts, "connect", refuse_wal)
    root = tmp_path / "plate"
    root.mkdir()
    registry = artifacts.open_registry(root)
    artifact = put(registry, "a.txt", ports.CROPS)
    assert registry.get(artifact.artifact_id) is not None


def test_a_read_only_consumer_refuses_to_conjure_a_registry(tmp_path):
    root = tmp_path / "plate"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no artifact registry"):
        artifacts.open_registry(root, create=False)
    artifacts.open_registry(root)
    assert artifacts.open_registry(root, create=False) is not None


def test_a_registry_needs_somewhere_to_live():
    with pytest.raises(ValueError, match="nowhere to keep"):
        artifacts.Registry()


def test_the_shared_override_puts_every_project_in_one_file(tmp_path,
                                                            monkeypatch):
    shared = tmp_path / "campaign.db"
    monkeypatch.setenv(artifacts.ARTIFACTS_DB_ENV, str(shared))
    first = Path(make_project(tmp_path / "plate1"))
    second = Path(make_project(tmp_path / "plate2"))
    one = artifacts.open_registry(first)
    two = artifacts.open_registry(second)
    put(one, "a.txt", ports.CROPS)
    put(two, "b.txt", ports.CROPS)
    assert shared.is_file()
    assert not (first / artifacts.ARTIFACTS_DB_NAME).exists()
    assert len(one.by_project("")) == 2
    assert len(one.by_project()) == 1


def test_two_writers_do_not_lose_a_registration(tmp_path):
    """Concurrent registration is the case the SQLite storage exists for."""
    import threading

    root = Path(make_project(tmp_path / "plate"))
    artifacts.open_registry(root)
    errors: list = []
    barrier = threading.Barrier(4)

    def worker(index: int) -> None:
        try:
            registry = artifacts.open_registry(root)
            barrier.wait(timeout=20)
            for step in range(5):
                put(registry, f"w{index}_{step}.txt", ports.CROPS,
                    content=f"{index}:{step}")
        except BaseException as exc:                       # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert len(artifacts.open_registry(root).by_project()) == 20


# ---- module-level convenience --------------------------------------------

def test_the_module_level_functions_resolve_a_registry(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    upstream = artifacts.register(
        project=root, module="mask", kind=ports.MERGED_ARRAYS,
        path=root / "merged", settings={"cell_diameter": 30})
    child = artifacts.register(
        project=root, module="measure", kind=ports.MEASUREMENTS_DB,
        path=root / "measurements" / "measurements.db", inputs=[upstream])

    assert [a.artifact_id for a in artifacts.by_kind(
        ports.MERGED_ARRAYS, project=root)] == [upstream.artifact_id]
    assert len(artifacts.by_project(project=root)) == 2
    assert artifacts.latest(ports.MEASUREMENTS_DB,
                            project=root).artifact_id == child.artifact_id
    assert [a.artifact_id for a in artifacts.downstream_of(
        upstream, project=root)] == [child.artifact_id]
    assert artifacts.is_stale(child, project=root).stale is False

    # an already-open registry is used as given
    registry = artifacts.open_registry(root)
    assert artifacts.by_kind(ports.MERGED_ARRAYS, registry=registry)
    assert artifacts.by_project(registry=registry)
    assert artifacts.latest(ports.MERGED_ARRAYS, registry=registry)
    assert artifacts.downstream_of(upstream, registry=registry)
    assert artifacts.is_stale(child, registry=registry) is not None
    assert artifacts.register(registry=registry, project=root, module="mask",
                              kind=ports.MASKS, path=root / "merged")


# ---- the run-completion hook ---------------------------------------------

def test_register_run_outputs_records_the_whole_chain(tmp_path):
    root = Path(make_project(tmp_path / "plate", crops=True))
    mask_settings = {"src": str(root), "cell_diameter": 30, "n_jobs": 8}

    produced = artifacts.register_run_outputs("mask", mask_settings,
                                              roots=[str(root)], run_id="r1")
    kinds = {a.kind for a in produced}
    assert ports.MERGED_ARRAYS in kinds
    # Mask writes object counts into the database; only Measure produces the
    # measurement tables an analysis module means by "measurements-db".
    assert ports.OBJECT_COUNTS in kinds
    assert ports.MEASUREMENTS_DB not in kinds
    assert all(a.module == "mask" and a.run_id == "r1" for a in produced)
    assert all(a.settings_hash == artifacts.settings_hash(mask_settings)
               for a in produced)

    # Measure's src is the merged folder; the project is still the plate.
    measure_settings = {"src": str(root / "merged"), "cell_min_size": 10}
    measured = artifacts.register_run_outputs("measure", measure_settings)
    assert {a.kind for a in measured} == {ports.MEASUREMENTS_DB, ports.CROPS}
    assert all(a.project == str(root) for a in measured)

    registry = artifacts.open_registry(root)
    merged = registry.latest(ports.MERGED_ARRAYS)
    database = [a for a in measured if a.kind == ports.MEASUREMENTS_DB][0]
    assert database.inputs == (merged.artifact_id,)
    assert merged.artifact_id in {
        a.artifact_id for a in registry.upstream_of(database)}


def test_register_run_outputs_skips_roots_that_are_not_there(tmp_path):
    root = make_project(tmp_path / "plate")
    produced = artifacts.register_run_outputs(
        "mask", {"src": root}, roots=[root, str(tmp_path / "absent"), ""])
    assert produced
    assert all(a.project == root for a in produced)


def test_register_run_outputs_takes_explicit_inputs_and_a_registry(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    seed = put(registry, "seed.txt", ports.RAW_IMAGES)
    produced = artifacts.register_run_outputs(
        "mask", {"src": str(root)}, roots=[str(root)], registry=registry,
        inputs=[seed.artifact_id], status=artifacts.STATUS_PARTIAL)
    assert produced
    assert all(a.inputs == (seed.artifact_id,) for a in produced)
    assert all(a.status == artifacts.STATUS_PARTIAL for a in produced)


def test_register_run_outputs_records_nothing_for_an_empty_project(tmp_path):
    root = tmp_path / "plate"
    root.mkdir()
    assert artifacts.register_run_outputs("mask", {"src": str(root)}) == ()


def test_register_run_outputs_raises_or_reports_as_asked(tmp_path, capsys):
    with pytest.raises(ports.UnknownModule):
        artifacts.register_run_outputs("not_a_module", {"src": str(tmp_path)})
    assert artifacts.register_run_outputs(
        "not_a_module", {"src": str(tmp_path)}, strict=False) == ()
    printed = capsys.readouterr().out
    assert "could not record not_a_module outputs" in printed


# ---- the two modules together --------------------------------------------

def test_check_ready_warns_about_a_stale_input_it_finds_in_the_registry(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    artifacts.register_run_outputs("mask", {"src": str(root), "d": 30},
                                   roots=[str(root)], registry=registry)
    artifacts.register_run_outputs("measure", {"src": str(root / "merged")},
                                   roots=[str(root)], registry=registry)

    clean = ports.check_ready("classify", {"src": str(root)}, registry=registry)
    assert clean.ok
    assert clean.inputs                       # provenance came back
    assert not [p for p in clean.warnings if "stale" in p.message]

    # Mask is re-run with a different diameter after Measure: the database
    # Classify is about to read no longer matches its inputs.
    (root / "merged" / "plate1_A03_0.npy").write_bytes(b"x")
    artifacts.register_run_outputs("mask", {"src": str(root), "d": 60},
                                   roots=[str(root)], registry=registry)
    # ... and re-point the newest database artifact at the measure run
    stale_check = ports.check_ready("classify", {"src": str(root)},
                                    registry=registry)
    warnings = [p for p in stale_check.warnings if "is stale" in p.message]
    assert warnings, ports.format_readiness(stale_check)
    assert warnings[0].severity == WARNING
    assert "Re-run " in warnings[0].fix


def test_check_ready_without_a_registered_input_adds_no_provenance(tmp_path):
    root = Path(make_project(tmp_path / "plate"))
    registry = artifacts.open_registry(root)
    readiness = ports.check_ready("measure", {"src": str(root)},
                                  registry=registry)
    assert readiness.ok
    assert readiness.inputs == ()


@pytest.mark.integration
def test_a_finished_mask_run_registers_its_outputs(tmp_path, monkeypatch):
    """The hook, exercised through the real entry point.

    Cellpose, the array merge and the overlay renderer are true externalities
    and are stubbed; everything the hook depends on — the run reaching its
    end, the project root, the declared outputs — is real.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    import spacr.core as core
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.plot as splot
    import spacr.utils as su

    noop = lambda *args, **kwargs: None                       # noqa: E731
    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam", noop)
    monkeypatch.setattr(sio, "_load_and_concatenate_arrays", noop)
    monkeypatch.setattr(splot, "plot_arrays", noop)
    monkeypatch.setattr(su, "_pivot_counts_table", noop)
    monkeypatch.setattr(su, "cleanup_pipeline_folders",
                        lambda *args, **kwargs: [])

    root = Path(make_project(tmp_path / "plate1", merged=1))
    (root / "stack").mkdir()
    np.save(root / "stack" / "f0.npy", np.zeros((3, 8, 8), np.uint16))

    core.preprocess_generate_masks({
        "src": str(root), "metadata_type": "cellvoyager",
        "channels": [0, 1, 2], "cell_channel": 1, "nucleus_channel": 0,
        "pathogen_channel": None, "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False,
        "batch_size": 10, "save": True, "custom_regex": None,
        "randomize": True, "examples_to_plot": 2, "cell_diameter": 30,
    })

    assert (root / artifacts.ARTIFACTS_DB_NAME).is_file()
    registry = artifacts.open_registry(root, create=False)
    recorded = registry.by_project()
    # the declared output paths are the ones the run really wrote, including
    # settings/gen_mask_settings.csv
    assert {a.kind for a in recorded} == {ports.MERGED_ARRAYS,
                                          ports.OBJECT_COUNTS,
                                          ports.SETTINGS_CSV}
    assert registry.latest(ports.SETTINGS_CSV).path == str(
        root / "settings" / "gen_mask_settings.csv")
    assert all(a.module == "mask" for a in recorded)
    assert all(a.spacr_version for a in recorded)
    # and the next step can now be planned off what was registered
    assert ports.check_ready("measure", {"src": str(root)}, registry=registry)


def test_the_core_hook_is_wired_where_the_run_finishes():
    """The mask pipeline must call the registry, and only in one place."""
    source = (_REPO_ROOT / "spacr" / "core.py").read_text(encoding="utf-8")
    # exactly one call site, and one local import beside it
    assert source.count("register_run_outputs(") == 1
    assert source.count("from .artifacts import register_run_outputs") == 1
    hook = source.index("register_run_outputs(")
    assert "ledger.stamp(db_path)" in source[:hook]
    assert "strict=False" in source[hook:hook + 400]


def test_problem_severities_come_from_validate():
    """Ports speak the settings pre-flight's language, not a private one."""
    assert ERROR == "error" and WARNING == "warning"
    readiness = ports.check_ready("measure", {})
    assert all(p.severity in (ERROR, WARNING) for p in readiness.problems)
    assert readiness.errors[0].is_error
