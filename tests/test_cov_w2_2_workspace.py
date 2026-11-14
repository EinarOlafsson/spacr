"""A workspace is mostly promises about what happens when something is wrong.

Saving and restoring the interactive state around a run is best-effort by
design: a panel that is not open, a figure whose file has been deleted, a
provider that raises, a state document from another machine. Every one of
those has to end as a line in the report rather than as an exception, because
the run itself has already finished and its results must not be lost to a
bookkeeping failure.

Nothing here is mocked away. The documents are written to and read back from
real directories, the digests are real SHA-256 over real bytes, and the
failure paths are produced by real broken inputs -- a file replaced by a
directory, a home directory that does not exist, a document that is not JSON.
"""

import pytest

from spacr.workspace import (CARRY_KEY, DEFAULT_COPY_LIMIT_MB, DEFAULT_MODE,
                             _carried, _file_record, _state_of, _walk_strings,
                             check_files, clear_providers, collect,
                             copy_limit_from_settings, default_copy_limit_mb,
                             default_mode, has_workspace, hash_file,
                             inventory_text, mode_from_settings, providers,
                             register, report_text, resolve_mode,
                             set_default_mode, unregister)


@pytest.fixture(autouse=True)
def process_defaults_restored():
    """The module keeps process-wide state; put all of it back afterwards."""
    before = (default_mode(), default_copy_limit_mb())
    registered = providers()
    yield
    clear_providers()
    for name, provider in registered.items():
        register(name, provider)
    set_default_mode(before[0], before[1])


# ---------------------------------------------------------------------------
# the process-wide defaults
# ---------------------------------------------------------------------------

def test_a_nonsense_copy_limit_leaves_the_old_one_standing():
    """An unparseable limit is ignored rather than becoming zero.

    Zero would silently stop every file being carried, which looks exactly
    like the feature being off.
    """
    set_default_mode("copy", 64)
    assert default_copy_limit_mb() == 64.0

    assert set_default_mode("copy", "not a number") == "copy"
    assert default_copy_limit_mb() == 64.0

    assert set_default_mode("copy", object()) == "copy"
    assert default_copy_limit_mb() == 64.0

    # a negative one is equally ignored
    set_default_mode("copy", -5)
    assert default_copy_limit_mb() == 64.0


def test_the_mode_aliases_a_user_actually_types_are_understood():
    """`save_workspace` is a settings field, so it arrives as text."""
    assert resolve_mode(None) == DEFAULT_MODE
    assert resolve_mode(True) == "reference"
    assert resolve_mode(False) == "off"
    assert resolve_mode("  COPY ") == "copy"
    assert resolve_mode("no") == "off"
    assert resolve_mode("yes") == "copy"
    assert resolve_mode("sometimes") == DEFAULT_MODE


# ---------------------------------------------------------------------------
# what a contributor offers
# ---------------------------------------------------------------------------

def test_a_contributor_can_offer_state_four_ways():
    """A mapping, the new method, the legacy pair, or nothing at all."""
    class New:
        def workspace_state(self):
            return {"a": 1}

    class Legacy:
        def plot_state(self):
            return {"b": 2}

    class Silent:
        pass

    assert _state_of(None) is None
    assert _state_of({"a": 1}) == {"a": 1}
    assert _state_of(New()) == {"a": 1}
    assert _state_of(Legacy()) == {"b": 2}
    assert _state_of(Silent()) is None


def test_a_contributor_that_answers_with_the_wrong_shape_offers_nothing():
    """A state getter returning a list is not a state document."""
    class Wrong:
        def workspace_state(self):
            return ["not", "a", "mapping"]

    class WrongLegacy:
        def plot_state(self):
            return 7

    assert _state_of(Wrong()) is None
    assert _state_of(WrongLegacy()) is None


# ---------------------------------------------------------------------------
# the file inventory
# ---------------------------------------------------------------------------

def test_a_file_that_cannot_be_read_has_no_digest(tmp_path):
    """`hash_file` answers None rather than raising, and the record says so."""
    assert hash_file(tmp_path / "not_there") is None
    assert hash_file(tmp_path) is None          # a directory is not readable

    record = _file_record("figure", tmp_path / "not_there")
    assert record == {"role": "figure", "path": str(tmp_path / "not_there"),
                      "exists": False}


def test_a_directory_is_recorded_as_one_and_not_hashed(tmp_path):
    """A referenced folder is inventoried without a size or a digest."""
    record = _file_record("outdir", tmp_path)
    assert record["exists"] is True
    assert record["kind"] == "directory"
    assert "sha256" not in record
    assert "size" not in record


def test_a_section_that_is_not_a_mapping_carries_nothing():
    """The carry key only exists inside a state document."""
    assert _carried(None) == []
    assert _carried(["a", "b"]) == []
    assert _carried("a string") == []


# ---------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------

def test_asking_whether_nonsense_has_a_workspace_is_false():
    """`has_workspace` answers rather than raising on a bad argument."""
    assert has_workspace(None) is False
    assert has_workspace(object()) is False


# ---------------------------------------------------------------------------
# the state of the files, later
# ---------------------------------------------------------------------------

def test_a_directory_is_present_without_a_digest_to_compare(tmp_path):
    """Nothing was hashed, so nothing can have changed."""
    outdir = tmp_path / "results"
    outdir.mkdir()
    doc = collect({"run": {"outdir": str(outdir)}}, saved="t")
    assert [e["state"] for e in check_files(doc)] == ["present"]


# ---------------------------------------------------------------------------
# saying what is in one
# ---------------------------------------------------------------------------

def test_there_is_no_inventory_for_something_that_is_not_a_document():
    """`inventory_text` says so rather than raising on a None."""
    assert inventory_text(None) == "no workspace document"
    assert inventory_text([1, 2]) == "no workspace document"


def test_a_junk_file_record_does_not_stop_the_inventory_printing():
    """A document from elsewhere may hold anything in its file list."""
    text = inventory_text({"version": 1, "sections": {}, "files": [
        "not a record", {"path": "/tmp/x", "size": 10}]})
    assert "/tmp/x" in text
    assert "not a record" not in text


def test_a_restore_report_names_what_did_not_come_back():
    """Nothing restored still prints a line; trouble is listed under it."""
    assert report_text({}) == "restored: nothing"

    text = report_text({
        "restored": ["montage"],
        "skipped": [{"section": "volcano", "why": "nothing on screen owns it"},
                    {"why": "no sections in the document"}],
        "files": [{"state": "missing", "role": "figure", "path": "/gone.png"},
                  {"state": "present", "role": "table", "path": "/here.csv"}],
    })
    assert "restored: montage" in text
    assert "not restored — volcano: nothing on screen owns it" in text
    assert "not restored — (document): no sections in the document" in text
    assert "missing — figure: /gone.png" in text
    assert "/here.csv" not in text


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_a_contributor_needs_a_name_and_a_callable():
    """Both are refused loudly: a nameless section cannot be written back."""
    with pytest.raises(ValueError):
        register("", lambda: {})
    with pytest.raises(TypeError):
        register("volcano", {"page": 1})


def test_a_registered_provider_is_resolved_at_collection_time():
    """The callable is held, not the widget it returns.

    Retaining the panel itself would keep a rebuilt or closed screen alive.
    """
    clear_providers()
    built = []

    def provider():
        built.append(1)
        return {"page": len(built)}

    register("volcano", provider)
    assert providers()["volcano"] is provider
    assert built == [], "registering already built the panel"

    doc = collect(providers(), saved="t")
    assert doc["sections"]["volcano"] == {"page": 1}

    assert unregister("volcano") is True
    assert unregister("volcano") is False
    assert "volcano" not in providers()


def test_a_snapshot_of_the_registry_is_a_copy():
    """Mutating what `providers()` returned does not touch the registry."""
    register("volcano", lambda: {"page": 1})
    snapshot = providers()
    snapshot.clear()
    assert "volcano" in providers()


def test_clearing_the_registry_also_restores_the_defaults():
    """Test teardown and application shutdown both want one call for this."""
    register("volcano", lambda: {"page": 1})
    set_default_mode("copy", 1)

    clear_providers()

    assert providers() == {}
    assert default_mode() == DEFAULT_MODE
    assert default_copy_limit_mb() == float(DEFAULT_COPY_LIMIT_MB)


# ---------------------------------------------------------------------------
# what the run settings ask for
# ---------------------------------------------------------------------------

def test_settings_that_say_nothing_get_the_process_default():
    """`save_workspace` unset means "whatever this process was told"."""
    set_default_mode("copy", 32)
    assert mode_from_settings({}) == "copy"
    assert mode_from_settings({"save_workspace": None}) == "copy"
    assert mode_from_settings(None) == "copy"
    assert copy_limit_from_settings({}) == 32.0


def test_an_explicit_setting_overrides_the_process_default():
    """The run's own value wins, in every spelling `resolve_mode` accepts."""
    set_default_mode("copy", 32)
    assert mode_from_settings({"save_workspace": "off"}) == "off"
    assert mode_from_settings({"save_workspace": False}) == "off"
    assert copy_limit_from_settings({"workspace_copy_limit_mb": 8}) == 8.0
    assert copy_limit_from_settings({"workspace_copy_limit_mb": "16"}) == 16.0


def test_a_nonsense_limit_in_the_settings_falls_back_to_the_default():
    """A bad value in a settings CSV does not silently mean zero."""
    set_default_mode("copy", 32)
    assert copy_limit_from_settings({"workspace_copy_limit_mb": "big"}) == 32.0
    assert copy_limit_from_settings({"workspace_copy_limit_mb": -1}) == 32.0
    assert copy_limit_from_settings({"workspace_copy_limit_mb": None}) == 32.0


# ---------------------------------------------------------------------------
# bounded walks
# ---------------------------------------------------------------------------

def test_the_string_walk_is_bounded_and_skips_the_reserved_keys():
    """Reserved keys are carried separately, so the walk must not re-add them.

    Depth is bounded for the same reason `_jsonable`'s is: a state that
    contains itself must cost a truncated section, not the process.
    """
    nested = {"a": "/x/y"}
    for _ in range(20):
        nested = {"deeper": nested}
    nested[CARRY_KEY] = [{"path": "/should/not/be/walked"}]

    found = dict(_walk_strings(nested))
    assert all("should/not" not in v for v in found.values())
    assert len(found) <= 1
