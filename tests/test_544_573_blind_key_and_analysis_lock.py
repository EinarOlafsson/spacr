"""Blind scoring keys (item 544) and the preregistered analysis lock (item 573).

Both live in :mod:`spacr.run_journal`, beside the runs they are about:

* a blinding key shuffles items under codes, is kept outside the data
  folder, and every unblinding is logged with who and when;
* a lock freezes settings, plan and files with a hash and a timestamp; an
  unchanged re-run verifies, a change is a deviation, a change after an
  unblinding on the same folder is post-hoc, an edited lock is caught, and
  the verdict reaches the manifest, the report and the methods text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacr import run_journal as rj


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A run journal of this test's own, and a plate folder beside it."""
    runs = tmp_path / "home" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    plate = tmp_path / "plate1"
    plate.mkdir()
    return plate


def test_a_key_shuffles_codes_and_lives_outside_the_data(journal):
    items = [str(journal / f"r1c{i}_f1_o{i}.png") for i in range(40)]
    key = rj.start_blinding(items + items[:3], scope="annotate", src=journal,
                            seed=7)
    assert sorted(key["order"]) == sorted(items)
    assert key["order"] != items
    assert sorted(key["codes"].values()) == [f"B{i:04d}" for i in range(1, 41)]
    assert [key["codes"][item] for item in key["order"]][:3] == [
        "B0001", "B0002", "B0003"]
    assert all(str(journal) not in code for code in key["codes"].values())
    again = rj.start_blinding(items, scope="annotate", src=journal, seed=7)
    assert again["order"] == key["order"]
    assert again["key_id"] != key["key_id"]

    stored = rj._read_blinding_key(key["key_id"])
    assert stored["seed"] == 7 and stored["order"] == key["order"]
    key_file = rj._blinding_root() / f"{key['key_id']}.json"
    assert key_file.is_file()
    assert journal not in key_file.parents
    assert not list(journal.rglob("*.json"))


def test_unblinding_returns_the_key_and_records_who_and_when(journal):
    key = rj.start_blinding(["a", "b", "c"], scope="make_masks", src=journal)
    events = rj._blinding_events(key["key_id"])
    assert [e["event"] for e in events] == ["blinded"]
    assert rj._unblinding_times(journal) == []

    opened = rj.unblind(key["key_id"], reason="scoring finished")
    assert opened == {code: item for item, code in key["codes"].items()}
    events = rj._blinding_events(key["key_id"])
    assert [e["event"] for e in events] == ["blinded", "unblinded"]
    record = events[-1]
    assert record["reason"] == "scoring finished"
    assert record["who"] == rj._who() and record["utc"] > events[0]["utc"]
    assert rj._unblinding_times(journal) == [record["utc"]]
    assert rj._unblinding_times(journal / "images") == [record["utc"]]
    assert rj._unblinding_times(journal.parent / "plate2") == []

    rj._close_blinding(key["key_id"], reason="screen closed")
    assert rj._blinding_events(key["key_id"])[-1]["event"] == "closed"
    with pytest.raises(FileNotFoundError):
        rj.unblind("no-such-key")


def _settings(plate, **changes):
    settings = {"src": str(plate), "model_type": "xgboost",
                "prediction_threshold": 0.5, "channels": [0, 1, 2],
                "hash_inputs": False, "_plot_theme": {"x": 1}}
    settings.update(changes)
    return settings


def test_an_unchanged_rerun_verifies_against_the_lock(journal):
    lock = rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                            hypotheses="Knockouts reduce infection.",
                            thresholds={"prediction_threshold": 0.5},
                            note="before unblinding")
    assert len(lock["sha256"]) == 64 and lock["locked_utc"]
    assert lock["sha256"] == rj._lock_digest(lock)
    assert "_plot_theme" not in lock["settings"]
    assert "hash_inputs" not in lock["settings"]
    stored = json.loads((rj._locks_root() / f"{lock['lock_id']}.json")
                        .read_text())
    assert stored == lock

    rerun = _settings(journal, channels="[0, 1, 2]", hash_inputs=True)
    result = rj.check_analysis_lock(rerun, app_key="ml_analyze")
    assert result["status"] == "verified", result
    assert result["deviations"] == [] and result["sha256"] == lock["sha256"]
    assert "verified" in result["summary"]

    other = rj.check_analysis_lock(_settings(journal), app_key="classify")
    assert other["status"] == "unlocked"
    elsewhere = rj.check_analysis_lock(
        _settings(journal.parent), app_key="ml_analyze")
    assert elsewhere["status"] == "unlocked"


def test_a_change_is_a_deviation_and_post_hoc_after_unblinding(journal):
    key = rj.start_blinding(["x", "y"], scope="annotate", src=journal)
    rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    changed = _settings(journal, prediction_threshold=0.4)

    before = rj.check_analysis_lock(changed, app_key="ml_analyze")
    assert before["status"] == "deviation"
    assert [d["key"] for d in before["deviations"]] == ["prediction_threshold"]
    assert before["deviations"][0]["locked"] == 0.5
    assert before["deviations"][0]["now"] == 0.4

    rj.unblind(key["key_id"])
    # 2026-09-30: an edit first seen while still blind stays a deviation
    # after the key is opened; only an edit first seen after it is post-hoc.
    kept = rj.check_analysis_lock(changed, app_key="ml_analyze")
    assert kept["status"] == "deviation"
    assert kept["deviations"][0]["first_seen_utc"] < kept["unblinded_utc"]
    assert kept["deviations"][0]["post_hoc"] is False
    late = _settings(journal, prediction_threshold=0.3)
    after = rj.check_analysis_lock(late, app_key="ml_analyze")
    assert after["status"] == "post_hoc"
    assert after["unblinded_utc"] > after["locked_utc"]
    assert "POST-HOC" in after["summary"]
    assert "prediction_threshold" in after["summary"]
    same = rj.check_analysis_lock(_settings(journal), app_key="ml_analyze")
    assert same["status"] == "verified"


def test_a_lock_made_after_unblinding_is_not_preregistered(journal):
    key = rj.start_blinding(["x"], scope="annotate", src=journal)
    rj.unblind(key["key_id"])
    lock = rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    assert lock["unblinded_before_lock"]
    result = rj.check_analysis_lock(_settings(journal), app_key="ml_analyze")
    assert result["status"] == "not_preregistered"


def test_an_edited_lock_and_a_changed_model_file_are_caught(journal):
    model = journal / "model.pth"
    model.write_bytes(b"weights v1")
    lock = rj.lock_analysis(_settings(journal, model_path=str(model)),
                            app_key="ml_analyze")
    assert list(lock["files"].values()) == [rj.hash_file(model, full=True)]

    model.write_bytes(b"weights v2")
    result = rj.check_analysis_lock(_settings(journal, model_path=str(model)),
                                    app_key="ml_analyze")
    assert result["status"] == "deviation"
    assert [d["key"] for d in result["deviations"]] == [
        f"file:{model.resolve()}"]

    path = rj._locks_root() / f"{lock['lock_id']}.json"
    edited = json.loads(path.read_text())
    edited["settings"]["prediction_threshold"] = 0.1
    path.write_text(json.dumps(edited))
    tampered = rj.check_analysis_lock(_settings(journal, model_path=str(model)),
                                      app_key="ml_analyze")
    assert tampered["status"] == "tampered"


def test_a_journalled_run_carries_the_verdict_in_its_manifest(journal):
    with rj.open_run("ml_analyze", _settings(journal)) as run:
        pass
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert "analysis_lock" not in manifest

    rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    with rj.open_run("ml_analyze", _settings(journal)) as same:
        pass
    manifest = json.loads((same.dir / "manifest.json").read_text())
    assert manifest["analysis_lock"]["status"] == "verified"
    assert not any("Analysis lock" in w
                   for w in manifest["provenance_warnings"])

    with rj.open_run("ml_analyze",
                     _settings(journal, model_type="random_forest")) as moved:
        pass
    manifest = json.loads((moved.dir / "manifest.json").read_text())
    assert manifest["analysis_lock"]["status"] == "deviation"
    assert manifest["analysis_lock"]["deviations"][0]["key"] == "model_type"
    assert any("DEVIATION" in w and "model_type" in w
               for w in manifest["provenance_warnings"])


def test_the_report_and_the_methods_text_flag_a_post_hoc_change(journal):
    from spacr.methods_export import build_digest, caveats_for
    from spacr.report import collect_report, render_text

    key = rj.start_blinding(["x"], scope="annotate", src=journal)
    rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    with rj.open_run("ml_analyze", _settings(journal)) as clean:
        pass
    rj.unblind(key["key_id"])
    with rj.open_run("ml_analyze",
                     _settings(journal, prediction_threshold=0.9)) as late:
        pass

    report = collect_report(journal, run_dirs=[clean.dir, late.dir])
    text = render_text(report)
    assert "preregistered analysis lock" in text
    assert "verified" in text and "POST-HOC" in text
    top = report.sections[0].notes
    assert any(late.dir.name in note and "POST-HOC" in note for note in top)
    assert not any(clean.dir.name in note for note in top)

    digest = build_digest(run_dir=late.dir)
    assert digest["run"]["analysis_lock"]["status"] == "post_hoc"
    caveats = caveats_for(digest)
    assert any("post-hoc" in c and "prediction_threshold" in c
               for c in caveats)
    clean_caveats = caveats_for(build_digest(run_dir=clean.dir))
    assert any("exactly as preregistered" in c for c in clean_caveats)
    unlocked = rj.open_run("measure", {"src": str(journal / "other")})
    with unlocked as plain:
        pass
    assert "analysis_lock" not in build_digest(run_dir=plain.dir)["run"]
    assert not any("preregistered" in c
                   for c in caveats_for(build_digest(run_dir=plain.dir)))
