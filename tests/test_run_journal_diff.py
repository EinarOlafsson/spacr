"""Tests for spacr.run_journal's provenance diff (diff_runs / format_run_diff).

Every test runs against a tmp runs-root: ``runs_root`` is monkeypatched so
the user's real ``~/.spacr/runs`` is never read or written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacr import run_journal as rj
from spacr.run_journal import (
    diff_runs, format_run_diff, resolve_run_dir, values_equal,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_runs(monkeypatch, tmp_path):
    """Point runs_root at tmp so tests never touch ~/.spacr/runs."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: root)
    return root


ENV_A = {"spacr": "1.4.3.7", "spacr_git": "2683638+dirty", "python": "3.10.19",
         "torch": "2.9.1", "cellpose": "4.0.7", "numpy": "1.26.4"}
ENV_B = dict(ENV_A, spacr="1.4.8.7", spacr_git="3e83647+dirty")


def make_run(root: Path, run_id: str, settings=None, *, app_key="mask",
             status="success", env=None, start_utc="2026-07-23T21:47:37+00:00",
             elapsed_s=19.061, write_settings=True, write_manifest=True,
             settings_text=None, manifest_text=None) -> Path:
    """Write a synthetic run folder and return its path."""
    d = root / run_id
    (d / "outputs").mkdir(parents=True, exist_ok=True)
    if settings_text is not None:
        (d / "settings.json").write_text(settings_text)
    elif write_settings:
        (d / "settings.json").write_text(json.dumps(settings or {}, indent=2))
    if manifest_text is not None:
        (d / "manifest.json").write_text(manifest_text)
    elif write_manifest:
        (d / "manifest.json").write_text(json.dumps({
            "app_key": app_key, "start_utc": start_utc,
            "end_utc": start_utc, "elapsed_s": elapsed_s, "status": status,
            "env": ENV_A if env is None else env,
            "model_hashes": {}, "n_settings": len(settings or {}),
            "traceback": None,
        }, indent=2))
    return d


# ---------------------------------------------------------------------------
# Bucketing — the core requirement
# ---------------------------------------------------------------------------

def test_buckets_changed_only_in_a_only_in_b_and_same(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {
        "src": "/data/plate1", "cell_channel": 0, "diameter": 30,
        "dropped_knob": True, "legacy_flag": "x",
    })
    b = make_run(root, "r_b__mask", {
        "src": "/data/plate2", "cell_channel": 0, "diameter": 60,
        "new_knob": 7,
    }, env=ENV_B)

    d = diff_runs(a, b)

    assert [c["key"] for c in d["changed"]] == ["diameter", "src"]
    assert d["changed"][0] == {"key": "diameter", "a": 30, "b": 60}
    assert d["changed"][1]["a"] == "/data/plate1"
    assert d["changed"][1]["b"] == "/data/plate2"
    assert d["only_in_a"] == ["dropped_knob", "legacy_flag"]
    assert d["only_in_b"] == ["new_knob"]
    # `same` is a count, never a list — the payload has to stay small.
    assert d["same"] == 1
    assert isinstance(d["same"], int)


def test_changed_is_sorted_and_carries_both_values(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"z": 1, "a": 1, "m": 1})
    b = make_run(root, "r_b__mask", {"z": 2, "a": 2, "m": 2})
    d = diff_runs(a, b)
    assert [c["key"] for c in d["changed"]] == ["a", "m", "z"]
    assert all(set(c) == {"key", "a", "b"} for c in d["changed"])


def test_identical_runs_report_no_changes(_isolated_runs):
    root = _isolated_runs
    settings = {"src": "/x", "n": 3}
    a = make_run(root, "r_a__mask", settings)
    b = make_run(root, "r_b__mask", dict(settings))
    d = diff_runs(a, b)
    assert d["changed"] == []
    assert d["only_in_a"] == d["only_in_b"] == []
    assert d["same"] == 2
    assert d["env"] == []


# ---------------------------------------------------------------------------
# Environment diff (from manifest.json)
# ---------------------------------------------------------------------------

def test_env_differences_extracted_from_manifest(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1}, env=ENV_A)
    b = make_run(root, "r_b__mask", {"n": 1}, env=ENV_B)
    d = diff_runs(a, b)
    env = {e["key"]: (e["a"], e["b"]) for e in d["env"]}
    assert env == {
        "spacr": ("1.4.3.7", "1.4.8.7"),
        "spacr_git": ("2683638+dirty", "3e83647+dirty"),
    }
    assert [e["key"] for e in d["env"]] == sorted(env)


def test_env_key_present_on_one_side_only_is_a_difference(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {}, env={"spacr": "1.0"})
    b = make_run(root, "r_b__mask", {}, env={"spacr": "1.0", "cellpose": "4.0.7"})
    d = diff_runs(a, b)
    assert d["env"] == [{"key": "cellpose", "a": None, "b": "4.0.7"}]


def test_env_not_fabricated_when_one_side_has_no_snapshot(_isolated_runs):
    """One unreadable manifest must not turn every package on the other
    side into a phantom 'changed to None' row."""
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {}, env=ENV_A)
    b = make_run(root, "r_b__mask", {}, write_manifest=False)
    d = diff_runs(a, b)
    assert d["env"] == []
    assert d["meta"]["b"]["errors"]


def test_env_missing_or_wrongly_typed_is_tolerated(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {}, manifest_text=json.dumps(
        {"app_key": "mask", "status": "success", "env": "not-a-dict"}))
    b = make_run(root, "r_b__mask", {}, manifest_text=json.dumps(
        {"app_key": "mask", "status": "success"}))
    d = diff_runs(a, b)
    assert d["env"] == []
    assert d["meta"]["a"]["spacr_version"] is None


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------

def test_structurally_equal_values_are_not_changes(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {
        "channels": [1, 2],
        "unset": None,
        "diameter": 30,
        "flag": True,
        "empty": None,
    })
    b = make_run(root, "r_b__mask", {
        "channels": [1, 2],       # same list
        "unset": "None",          # CSV round-trip of None
        "diameter": "30",         # CSV round-trip of an int
        "flag": "True",           # CSV round-trip of a bool
        "empty": "",              # CSV writes None as an empty cell
    })
    d = diff_runs(a, b)
    assert d["changed"] == []
    assert d["same"] == 5


def test_stringified_list_matches_real_list(_isolated_runs):
    """Seen in real journals: one run stored ``[0, 1, 2]``, the next
    stored ``"[0, 1, 2]"`` because settings went through str()."""
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"channels": "[0, 1, 2]"})
    b = make_run(root, "r_b__mask", {"channels": [0, 1, 2]})
    assert diff_runs(a, b)["changed"] == []


def test_real_value_changes_still_detected(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"channels": [1, 2], "d": None, "f": True})
    b = make_run(root, "r_b__mask", {"channels": [1, 3], "d": 60, "f": False})
    d = diff_runs(a, b)
    assert [c["key"] for c in d["changed"]] == ["channels", "d", "f"]


@pytest.mark.parametrize("a,b", [
    ([1, 2], [1, 2]),
    ([1, 2], (1, 2)),
    (None, "None"),
    (None, "null"),
    (None, ""),
    ("  spam ", "spam"),
    (True, "true"),
    (False, "False"),
    (3, "3"),
    (3, 3.0),
    ("[0, 1, 2]", [0, 1, 2]),
    ({"a": 1}, "{'a': 1}"),
    (float("nan"), float("nan")),
    (Path("/tmp/x"), "/tmp/x"),
    ({1, 2}, {2, 1}),
])
def test_values_equal_true(a, b):
    assert values_equal(a, b) is True
    assert values_equal(b, a) is True


@pytest.mark.parametrize("a,b", [
    ([1, 2], [2, 1]),
    (None, 0),
    ("spam", "eggs"),
    (3, 4),
    ("[0, 1]", [0, 2]),
    ({"a": 1}, {"a": 2}),
])
def test_values_equal_false(a, b):
    assert values_equal(a, b) is False


def test_values_equal_handles_numpy_arrays():
    np = pytest.importorskip("numpy")
    assert values_equal(np.array([1, 2]), [1, 2]) is True
    assert values_equal(np.array([1, 2]), [1, 3]) is False
    assert values_equal(np.int64(5), 5) is True


def test_values_equal_falls_back_to_repr_when_eq_raises():
    class Weird:
        def __eq__(self, other):
            raise RuntimeError("no comparison for you")
        def __repr__(self):
            return "Weird()"
        __hash__ = None
    assert values_equal(Weird(), Weird()) is True


def test_values_equal_returns_false_when_even_repr_raises():
    class Cursed:
        def __eq__(self, other):
            raise RuntimeError("nope")
        def __repr__(self):
            raise RuntimeError("nope either")
        __hash__ = None
    assert values_equal(Cursed(), Cursed()) is False


def test_normalisation_is_depth_capped():
    v = "leaf"
    for _ in range(14):
        v = [v]
    # Does not recurse forever, and identical deep structures still match.
    assert rj._normalize_value(v) is not None
    assert values_equal(v, json.loads(json.dumps(v))) is True
    assert rj._normalize_value("x", _depth=99) == repr("x")


def test_normalisation_of_exotic_containers():
    class BadList:
        def tolist(self):
            raise RuntimeError("boom")
        def __repr__(self):
            return "BadList()"
    assert rj._normalize_value(BadList()) == "BadList()"

    class BadSet(frozenset):
        def __iter__(self):
            raise RuntimeError("boom")
    assert rj._normalize_value(BadSet()) == repr(BadSet())

    # dict whose str()-ed keys collide, with unorderable values: the
    # sorted() fast path raises, the unsorted fallback still returns.
    out = rj._normalize_value({1: None, "1": 5})
    assert isinstance(out, tuple) and len(out) == 2


def test_long_strings_are_not_literal_evaled():
    long_literal = "[" + ", ".join("1" for _ in range(3000)) + "]"
    assert len(long_literal) > 4096
    assert rj._normalize_value(long_literal) == long_literal
    # An unparseable string comes back stripped, not mangled.
    assert rj._normalize_value(" [1, ") == "[1,"
    assert rj._normalize_value("nucleus") == "nucleus"


# ---------------------------------------------------------------------------
# Malformed / partial run folders
# ---------------------------------------------------------------------------

def test_missing_settings_json_is_reported_not_raised(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", write_settings=False)
    d = diff_runs(a, b)
    assert d["changed"] == []
    assert d["only_in_a"] == ["n"]
    assert d["meta"]["b"]["n_settings"] == 0
    assert any("no settings" in e for e in d["meta"]["b"]["errors"])
    # And it renders.
    assert "no settings" in format_run_diff(d)


def test_corrupt_manifest_json_is_reported_not_raised(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", {"n": 2}, manifest_text="{not json at all,,,")
    d = diff_runs(a, b)
    assert [c["key"] for c in d["changed"]] == ["n"]
    assert d["meta"]["b"]["app_key"] is None
    assert d["meta"]["b"]["status"] is None
    assert any("manifest.json unreadable" in e for e in d["meta"]["b"]["errors"])
    assert d["env"] == []
    assert "manifest.json unreadable" in format_run_diff(d)


def test_missing_manifest_run_still_in_flight(_isolated_runs):
    """A crashed / still-running pipeline leaves settings but no manifest."""
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", {"n": 2}, write_manifest=False)
    d = diff_runs(a, b)
    assert [c["key"] for c in d["changed"]] == ["n"]
    assert any("no manifest.json" in e for e in d["meta"]["b"]["errors"])


def test_manifest_that_is_json_but_not_an_object(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", {"n": 1}, manifest_text="[1, 2, 3]")
    d = diff_runs(a, b)
    assert any("not an object" in e for e in d["meta"]["b"]["errors"])


def test_corrupt_settings_json_falls_back_to_settings_csv(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1, "src": "/x"})
    b = make_run(root, "r_b__mask", settings_text="{oops")
    (b / "settings.csv").write_text("Key,Value\nn,2\nsrc,/x\n")
    d = diff_runs(a, b)
    assert [c["key"] for c in d["changed"]] == ["n"]
    assert d["same"] == 1
    errs = d["meta"]["b"]["errors"]
    assert any("settings.json unreadable" in e for e in errs)
    assert any("settings.csv" in e for e in errs)


def test_settings_csv_only_run_is_read(_isolated_runs):
    """Older / hand-edited runs may have only the CSV twin. Values come
    back as strings there, which normalisation must see through."""
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1, "src": "/x", "flag": True})
    b = make_run(root, "r_b__mask", write_settings=False)
    (b / "settings.csv").write_text("Key,Value\nn,2\nsrc,/x\nflag,True\n")
    d = diff_runs(a, b)
    assert [c["key"] for c in d["changed"]] == ["n"]
    assert d["same"] == 2          # src and flag survive the CSV round-trip
    assert d["meta"]["b"]["errors"] == []


def test_corrupt_settings_json_and_corrupt_csv(_isolated_runs, monkeypatch):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", settings_text="{oops")
    (b / "settings.csv").write_text("Key,Value\nn,2\n")

    def _boom(path):
        raise RuntimeError("csv is toast")
    monkeypatch.setattr(rj, "_read_settings_csv", _boom)

    d = diff_runs(a, b)
    assert d["meta"]["b"]["n_settings"] == 0
    assert any("settings.csv unreadable" in e for e in d["meta"]["b"]["errors"])


def test_settings_json_that_is_not_a_dict(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", settings_text="[1, 2, 3]")
    d = diff_runs(a, b)
    assert d["meta"]["b"]["n_settings"] == 0
    assert any("not a dict" in e for e in d["meta"]["b"]["errors"])


def test_two_empty_run_folders_do_not_raise(_isolated_runs):
    root = _isolated_runs
    a = (root / "empty_a"); a.mkdir()
    b = (root / "empty_b"); b.mkdir()
    d = diff_runs(a, b)
    assert d["changed"] == [] and d["same"] == 0 and d["env"] == []
    assert format_run_diff(d)  # renders


# ---------------------------------------------------------------------------
# Run reference resolution
# ---------------------------------------------------------------------------

def test_accepts_run_id_string_as_well_as_path(_isolated_runs):
    root = _isolated_runs
    make_run(root, "2026-07-23_214737_b66bae6b__mask", {"n": 1})
    b = make_run(root, "2026-07-25_121815_521f4b3b__mask", {"n": 2})
    # A: bare run-id string. B: Path object.
    d = diff_runs("2026-07-23_214737_b66bae6b__mask", b)
    assert [c["key"] for c in d["changed"]] == ["n"]
    assert d["meta"]["a"]["run_id"] == "2026-07-23_214737_b66bae6b__mask"
    assert d["meta"]["b"]["run_id"] == "2026-07-25_121815_521f4b3b__mask"
    # …and as a plain path string.
    assert diff_runs(str(root / "2026-07-23_214737_b66bae6b__mask"),
                     str(b))["same"] == 0


def test_resolve_run_dir_accepts_unique_prefix(_isolated_runs):
    root = _isolated_runs
    d = make_run(root, "2026-07-23_214737_b66bae6b__mask", {})
    make_run(root, "2026-07-25_121815_521f4b3b__mask", {})
    assert resolve_run_dir("2026-07-23") == d


def test_resolve_run_dir_rejects_ambiguous_prefix(_isolated_runs):
    root = _isolated_runs
    make_run(root, "2026-07-23_a__mask", {})
    make_run(root, "2026-07-23_b__mask", {})
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        resolve_run_dir("2026-07-23")


def test_resolve_run_dir_accepts_run_object(_isolated_runs):
    root = _isolated_runs
    d = make_run(root, "r_a__mask", {})
    run = rj.Run(app_key="mask", settings={}, dir=d)
    assert resolve_run_dir(run) == d


def test_resolve_run_dir_raises_for_unknown_run(_isolated_runs):
    with pytest.raises(FileNotFoundError, match="no such run"):
        resolve_run_dir("nope-not-a-run")
    with pytest.raises(FileNotFoundError):
        resolve_run_dir("/definitely/not/here")
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(None)
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(17)  # not path-like at all


def test_resolve_run_dir_survives_unreadable_runs_root(monkeypatch, tmp_path):
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError, match="no such run"):
        resolve_run_dir("whatever")


def test_diff_runs_integrates_with_open_run(_isolated_runs):
    """End-to-end against folders written by the journal itself."""
    with rj.open_run("mask", {"src": "/x", "diameter": 30}) as run_a:
        pass
    with rj.open_run("mask", {"src": "/x", "diameter": 60}) as run_b:
        pass
    d = diff_runs(run_a.dir.name, run_b.dir)
    assert d["changed"] == [{"key": "diameter", "a": 30, "b": 60}]
    assert d["same"] == 1
    assert d["meta"]["a"]["status"] == "success"
    assert d["meta"]["a"]["spacr_version"]
    assert d["env"] == []  # same process, same environment


# ---------------------------------------------------------------------------
# Cross-app comparison
# ---------------------------------------------------------------------------

def test_different_app_keys_allowed_and_flagged(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"src": "/x", "cell_channel": 0},
                 app_key="mask")
    b = make_run(root, "r_b__measure", {"src": "/y", "table_name": "cell"},
                 app_key="measure")
    d = diff_runs(a, b)
    assert d["meta"]["app_key_differs"] is True
    assert d["meta"]["a"]["app_key"] == "mask"
    assert d["meta"]["b"]["app_key"] == "measure"
    assert [c["key"] for c in d["changed"]] == ["src"]      # still diffed
    assert "different pipelines" in format_run_diff(d)


def test_same_app_key_not_flagged(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {}, app_key="mask")
    b = make_run(root, "r_b__mask", {}, app_key="mask")
    assert diff_runs(a, b)["meta"]["app_key_differs"] is False


def test_unknown_app_key_is_not_reported_as_a_difference(_isolated_runs):
    """A run with no readable manifest has app_key None — that is
    'unknown', not 'a different pipeline'."""
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {}, app_key="mask")
    b = make_run(root, "r_b__mask", {}, write_manifest=False)
    d = diff_runs(a, b)
    assert d["meta"]["app_key_differs"] is False


# ---------------------------------------------------------------------------
# meta block
# ---------------------------------------------------------------------------

def test_meta_carries_required_fields_for_both_sides(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1}, status="failed", elapsed_s=1.5,
                 start_utc="2026-07-23T21:47:37+00:00")
    b = make_run(root, "r_b__mask", {"n": 2}, status="success", elapsed_s=10.9,
                 start_utc="2026-07-25T12:18:15+00:00", env=ENV_B)
    m = diff_runs(a, b)["meta"]
    for side, run_id, status in (("a", "r_a__mask", "failed"),
                                 ("b", "r_b__mask", "success")):
        assert {"run_id", "app_key", "status", "start_utc",
                "elapsed_s"} <= set(m[side])
        assert m[side]["run_id"] == run_id
        assert m[side]["status"] == status
    assert m["a"]["elapsed_s"] == 1.5
    assert m["b"]["start_utc"] == "2026-07-25T12:18:15+00:00"
    assert m["a"]["spacr_version"] == "1.4.3.7"
    assert m["b"]["spacr_version"] == "1.4.8.7"
    assert Path(m["a"]["dir"]).name == "r_a__mask"


def test_diff_result_is_json_serialisable(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1, "ch": [0, 1]})
    b = make_run(root, "r_b__mask", {"n": 2, "extra": None})
    text = json.dumps(diff_runs(a, b))
    assert "\"changed\"" in text


# ---------------------------------------------------------------------------
# format_run_diff
# ---------------------------------------------------------------------------

def _drifty_pair(root):
    """Mimic the real cross-release case: 200 keys of schema drift
    hiding three settings the user actually changed."""
    old = {f"legacy_key_{i:03d}": i for i in range(200)}
    old.update({"src": "/data/plate1", "cell_channel": 0, "diameter": 30,
                "verbose": True})
    new = {f"new_key_{i:02d}": i for i in range(12)}
    new.update({"src": "/data/plate2", "cell_channel": 1, "diameter": 30,
                "verbose": True})
    a = make_run(root, "2026-07-23_old__mask", old, env=ENV_A)
    b = make_run(root, "2026-07-25_new__mask", new, env=ENV_B)
    return a, b


def test_format_shows_changed_keys_first_and_summarises_drift(_isolated_runs):
    root = _isolated_runs
    a, b = _drifty_pair(root)
    d = diff_runs(a, b)
    out = format_run_diff(d)

    # The signal is present…
    assert "cell_channel" in out
    assert "src" in out
    assert "/data/plate1" in out and "/data/plate2" in out
    assert "Settings changed (2 of 4 shared keys)" in out
    # …and comes before the environment and the drift summary.
    assert out.index("Settings changed") < out.index("Environment changed")
    assert out.index("Environment changed") < out.index("Schema drift")

    # Drift is a one-line summary, not a dump.
    assert "Schema drift: +12 keys added, -200 removed" in out
    assert "1.4.3.7" in out and "1.4.8.7" in out
    assert "(+194 more)" in out
    assert "(+6 more)" in out
    # Only a handful of drifted keys are named — never all of them.
    named = sum(1 for k in list(d["only_in_a"]) + list(d["only_in_b"])
                if k in out)
    assert named <= 12, f"format dumped {named} drifted keys"
    assert "legacy_key_199" not in out
    assert "new_key_11" not in out
    # And the report stays short enough to read in a terminal.
    assert len(out.splitlines()) < 30


def test_format_includes_run_identity_and_env(_isolated_runs):
    root = _isolated_runs
    a, b = _drifty_pair(root)
    out = format_run_diff(diff_runs(a, b))
    assert "2026-07-23_old__mask" in out
    assert "2026-07-25_new__mask" in out
    assert "mask" in out and "success" in out
    assert "19.1s" in out
    assert "spacr_git" in out
    assert "2683638+dirty" in out


def test_format_max_drift_names_is_configurable(_isolated_runs):
    root = _isolated_runs
    a, b = _drifty_pair(root)
    d = diff_runs(a, b)
    out = format_run_diff(d, max_drift_names=1)
    assert "(+199 more)" in out
    assert "legacy_key_001" not in out
    # 0 means "just the counts"
    bare = format_run_diff(d, max_drift_names=0)
    assert "(200 keys, not shown)" in bare
    assert "legacy_key_000" not in bare
    assert rj._drift_names([], 6) == ""
    assert rj._drift_names(["a", "b"], 6) == "a, b"


def test_format_of_identical_runs_says_so(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1})
    b = make_run(root, "r_b__mask", {"n": 1})
    out = format_run_diff(diff_runs(a, b))
    assert "identical" in out
    assert "same versions on both runs" in out
    assert "Schema drift: none" in out


def test_format_truncates_very_long_values(_isolated_runs):
    root = _isolated_runs
    long_a = "/very/long/path/" + "x" * 300
    a = make_run(root, "r_a__mask", {"src": long_a})
    b = make_run(root, "r_b__mask", {"src": "/short"})
    out = format_run_diff(diff_runs(a, b))
    assert "…" in out
    assert long_a not in out
    assert max(len(line) for line in out.splitlines()) < 160


def test_format_elides_the_common_prefix_of_long_values(_isolated_runs):
    """Two long paths sharing a head must not truncate to the same text
    — the reader has to be able to see what actually differs."""
    root = _isolated_runs
    head = "/tmp/pytest-of-olafsson/pytest-"
    tail = "/test_module_preprocess_generate_masks0/data/plate1"
    a = make_run(root, "r_a__mask", {"src": head + "562" + tail})
    b = make_run(root, "r_b__mask", {"src": head + "671" + tail})
    out = format_run_diff(diff_runs(a, b))
    assert "…562" in out and "…671" in out
    assert head not in out                    # common head elided
    src_line = [ln for ln in out.splitlines() if ln.strip().startswith("src")][0]
    left, right = src_line.split("  →  ")
    assert left.strip() != right.strip()      # the two sides differ on screen

    # A short vs long pair with no meaningful common head still truncates
    # from the right, as before.
    assert rj._render_change_pair("/x", "/very/long/" + "y" * 90) == (
        "/x", "/very/long/" + "y" * 34 + "…")


def test_format_renders_missing_elapsed_and_unknown_fields(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1}, elapsed_s=None)
    b = make_run(root, "r_b__mask", {"n": 2}, write_manifest=False)
    out = format_run_diff(diff_runs(a, b))
    assert "—" in out          # elapsed placeholder
    assert "?" in out          # unknown app_key / status / start
    assert "no manifest.json" in out


def test_format_handles_empty_diff_dict():
    """Defensive: a caller handing us {} must still get a report."""
    out = format_run_diff({})
    assert "Run diff" in out
    assert "identical" in out
    assert "Schema drift: none" in out


def test_format_renders_only_added_or_only_removed(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1}, env=ENV_A)
    b = make_run(root, "r_b__mask", {"n": 1, "extra": 2}, env=ENV_A)
    out = format_run_diff(diff_runs(a, b))
    assert "+1 keys added, -0 removed" in out
    assert "added:   extra" in out
    assert "removed:" not in out

    c = make_run(root, "r_c__mask", {}, env=ENV_A)
    out2 = format_run_diff(diff_runs(a, c))
    assert "removed: n" in out2
    assert "added:" not in out2


def test_format_drift_version_label_when_versions_match(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1}, env=ENV_A)
    b = make_run(root, "r_b__mask", {"m": 1}, env=ENV_A)
    out = format_run_diff(diff_runs(a, b))
    assert "since 1.4.3.7" in out


def test_format_drift_without_any_version_info(_isolated_runs):
    root = _isolated_runs
    a = make_run(root, "r_a__mask", {"n": 1}, env={})
    b = make_run(root, "r_b__mask", {"m": 1}, env={})
    out = format_run_diff(diff_runs(a, b))
    assert "Schema drift: +1 keys added, -1 removed" in out
    assert "since" not in out
