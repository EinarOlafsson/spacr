"""The preregistered analysis lock beyond settings (item 573, 2026-09-30).

What the first alpha left out, each checked here:

* Gate Editor gating strategies -- a saved gate file (named by any setting
  or handed over) is locked by its gates, not its bytes, and gates held as
  objects are compared when handed to the check;
* models recorded while the run is under way (``Run.record_model``), after
  the settings were checked;
* one lock spanning several pipelines;
* an edit is post-hoc only when it was first seen after unblinding;
* the HTML report is well formed and carries the verdicts where a reader
  looks for them.
"""
from __future__ import annotations

import json
from html.parser import HTMLParser

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


def _settings(plate, **changes):
    """Classifier settings on ``plate``, with ``changes`` applied."""
    settings = {"src": str(plate), "model_type": "xgboost",
                "prediction_threshold": 0.5, "channels": [0, 1, 2]}
    settings.update(changes)
    return settings


def _gate(name, low, parent=None, column="cell_area"):
    """One threshold gate as the Gate Editor saves it."""
    return {"kind": "threshold", "name": name, "parent": parent,
            "column": column, "low": low, "high": None}


def test_a_gate_editor_file_is_locked_by_its_gates_not_its_bytes(journal):
    gates = journal / "strategy.json"
    gates.write_text(json.dumps({"gates": [_gate("big", 100.0),
                                           _gate("bright", 5.0, "big")]}))
    lock = rj.lock_analysis(_settings(journal, gating=str(gates)),
                            app_key="ml_analyze")
    label = str(gates.resolve())
    assert set(lock["gates"]) == {label} and not lock["files"]
    assert set(lock["gates"][label]["gates"]) == {"big", "bright"}

    gates.write_text(json.dumps(json.loads(gates.read_text()), indent=4,
                                sort_keys=True))
    same = rj.check_analysis_lock(_settings(journal, gating=str(gates)),
                                  app_key="ml_analyze")
    assert same["status"] == "verified", same["deviations"]
    assert rj._gate_file_lock_notes(gates) == [
        f"these gates match analysis lock {lock['sha256'][:16]}"]

    gates.write_text(json.dumps({"gates": [_gate("big", 150.0),
                                           _gate("bright", 5.0, "big")]}))
    moved = rj.check_analysis_lock(_settings(journal, gating=str(gates)),
                                   app_key="ml_analyze")
    assert moved["status"] == "deviation"
    assert [(d["key"], d["detail"]) for d in moved["deviations"]] == [
        (f"gates:{label}", "changed big")]
    note, = rj._gate_file_lock_notes(gates)
    assert "differ" in note and "changed big" in note and "deviation" in note
    gates.unlink()
    gone = rj.check_analysis_lock(_settings(journal, gating=str(gates)),
                                  app_key="ml_analyze")
    assert "gone" in [d for d in gone["deviations"]
                      if d["key"].startswith("gates:")][0]["detail"]


def test_gates_held_by_the_gate_editor_are_checked_when_handed_over(journal):
    from spacr.qt.widgets.gate_spec import GateSet, gate_from_dict

    held = GateSet([gate_from_dict(_gate("infected", 0.2,
                                         column="pathogen_area"))])
    lock = rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                            gates={"infection": held})
    assert lock["gates"]["infection"]["source"] == "memory"
    settings = _settings(journal)
    assert rj.check_analysis_lock(settings, app_key="ml_analyze")[
        "status"] == "verified"
    assert rj.check_analysis_lock(settings, app_key="ml_analyze",
                                  gates={"infection": held})[
        "status"] == "verified"
    loose = GateSet([gate_from_dict(_gate("infected", 0.1,
                                          column="pathogen_area"))])
    result = rj.check_analysis_lock(settings, app_key="ml_analyze",
                                    gates={"infection": loose})
    assert result["status"] == "deviation"
    assert result["deviations"][0]["detail"] == "changed infected"
    extra = rj.check_analysis_lock(settings, app_key="ml_analyze",
                                   gates={"other": loose})
    assert extra["deviations"][0]["detail"] == "not in the lock"
    with pytest.raises(ValueError):
        rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                         gates={"bad": {"not": "gates"}})


def test_a_model_recorded_during_the_run_is_checked_against_the_lock(journal):
    model = journal / "cyto.pt"
    model.write_bytes(b"weights v1")
    rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                     models={"cellpose_cyto": str(model)})
    with rj.open_run("ml_analyze", _settings(journal)) as same:
        same.record_model("cellpose_cyto", model)
    manifest = json.loads((same.dir / "manifest.json").read_text())
    assert manifest["analysis_lock"]["status"] == "verified"

    other = journal / "nuclei.pt"
    other.write_bytes(b"other")
    model.write_bytes(b"weights v2")
    with rj.open_run("ml_analyze", _settings(journal)) as run:
        assert run._analysis_lock["status"] == "verified"
        run.record_model("cellpose_cyto", model)
        run.record_model("nucleus", other)
    manifest = json.loads((run.dir / "manifest.json").read_text())
    lock = manifest["analysis_lock"]
    assert lock["status"] == "deviation"
    assert [d["key"] for d in lock["deviations"]] == [
        "model:cellpose_cyto", "model:nucleus"]
    assert lock["deviations"][1]["locked"] is None
    warnings = [w for w in manifest["provenance_warnings"]
                if "Analysis lock" in w]
    assert warnings == [lock["summary"]]
    with pytest.raises(FileNotFoundError):
        rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                         models={"gone": str(journal / "missing.pt")})


def test_a_model_the_lock_does_not_name_is_listed_not_judged(journal):
    from spacr.methods_export import build_digest, caveats_for

    rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    model = journal / "cyto.pt"
    model.write_bytes(b"weights")
    with rj.open_run("ml_analyze", _settings(journal)) as run:
        run.record_model("cellpose_cyto", model)
    lock = json.loads((run.dir / "manifest.json").read_text())["analysis_lock"]
    assert lock["status"] == "verified"
    assert lock["uncovered_models"] == ["cellpose_cyto"]
    assert "not covered by the lock: cellpose_cyto" in lock["summary"]
    assert any("does not cover the models cellpose_cyto" in c
               for c in caveats_for(build_digest(run_dir=run.dir)))
    assert rj._recorded_models("ml_analyze", journal) == {
        "cellpose_cyto": str(model.resolve())}
    assert rj._recorded_models("classify", journal) == {}


def test_one_lock_spans_several_pipelines(journal, tmp_path):
    second = tmp_path / "plate2"
    second.mkdir()
    measure = {"src": str(second), "cell_min_size": 100}
    lock = rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                            pipelines={"measure": measure})
    assert set(lock["pipelines"]) == {"ml_analyze", "measure"}
    assert rj._find_lock("measure", second)["lock_id"] == lock["lock_id"]
    assert rj.check_analysis_lock(measure, app_key="measure")[
        "status"] == "verified"
    assert rj.check_analysis_lock(_settings(journal), app_key="ml_analyze")[
        "pipelines"] == ["measure", "ml_analyze"]
    assert rj.check_analysis_lock(measure, app_key="classify")[
        "status"] == "unlocked"
    assert rj.check_analysis_lock(dict(measure, src=str(journal)),
                                  app_key="measure")["status"] == "unlocked"

    key = rj.start_blinding(["x"], scope="annotate", src=journal)
    rj.unblind(key["key_id"])
    late = rj.check_analysis_lock(dict(measure, cell_min_size=50),
                                  app_key="measure")
    assert late["status"] == "post_hoc"
    assert late["deviations"][0]["key"] == "cell_min_size"
    with pytest.raises(ValueError):
        rj.lock_analysis(_settings(journal), app_key="ml_analyze",
                         pipelines={"ml_analyze": {}})


def test_post_hoc_is_decided_by_when_the_edit_was_first_seen(journal):
    from spacr.methods_export import build_digest, caveats_for

    key = rj.start_blinding(["x"], scope="annotate", src=journal)
    rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    early = _settings(journal, model_type="random_forest")
    with rj.open_run("ml_analyze", early) as blind:
        pass
    assert blind._analysis_lock["status"] == "deviation"
    rj.unblind(key["key_id"])
    with rj.open_run("ml_analyze", early) as again:
        pass
    assert again._analysis_lock["status"] == "deviation"
    both = dict(early, prediction_threshold=0.8)
    with rj.open_run("ml_analyze", both) as mixed:
        pass
    verdict = mixed._analysis_lock
    assert verdict["status"] == "post_hoc"
    assert {d["key"]: d["post_hoc"] for d in verdict["deviations"]} == {
        "model_type": False, "prediction_threshold": True}
    assert "model_type (before unblinding)" in verdict["summary"]
    assert "prediction_threshold (after unblinding)" in verdict["summary"]
    caveat, = [c for c in caveats_for(build_digest(run_dir=mixed.dir))
               if "preregistered" in c]
    assert "changes to prediction_threshold are post-hoc" in caveat
    assert "model_type were made before the key was opened" in caveat


def test_a_lock_of_the_first_schema_still_checks(journal):
    lock = rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    old = {k: v for k, v in lock.items()
           if k not in ("pipelines", "gates", "models", "sha256")}
    old["schema"] = 1
    old["sha256"] = rj._lock_digest(old)
    path = rj._locks_root() / f"{lock['lock_id']}.json"
    path.write_text(json.dumps(old))
    result = rj.check_analysis_lock(_settings(journal), app_key="ml_analyze")
    assert result["status"] == "verified" and result["sha256"] == old["sha256"]
    assert result["pipelines"] == ["ml_analyze"]


_VOID = frozenset({"meta", "img", "br", "hr", "link", "input", "col", "wbr"})


class _Tree(HTMLParser):
    """Parse a page into nested nodes, recording any mis-nested end tag."""

    def __init__(self):
        """Start from an empty root."""
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "#root", "attrs": {}, "kids": [], "text": ""}
        self.stack = [self.root]
        self.errors = []
        self.decl = []

    def handle_decl(self, decl):
        """Keep the doctype."""
        self.decl.append(decl)

    def handle_starttag(self, tag, attrs):
        """Open a node; a void element is closed at once."""
        node = {"tag": tag, "attrs": dict(attrs), "kids": [], "text": ""}
        self.stack[-1]["kids"].append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        """A self-closed element."""
        self.stack[-1]["kids"].append(
            {"tag": tag, "attrs": dict(attrs), "kids": [], "text": ""})

    def handle_endtag(self, tag):
        """Close the open node, which must be the one this tag names."""
        if tag in _VOID:
            return
        if self.stack[-1]["tag"] != tag:
            self.errors.append((tag, self.stack[-1]["tag"]))
            return
        self.stack.pop()

    def handle_data(self, data):
        """Text counts for every open node."""
        for node in self.stack[1:]:
            node["text"] += data


def _find(node, tag):
    """Every node below ``node`` with ``tag``, in document order."""
    found = []
    for kid in node["kids"]:
        if kid["tag"] == tag:
            found.append(kid)
        found.extend(_find(kid, tag))
    return found


def test_the_html_report_is_well_formed_and_carries_the_verdicts(journal):
    from spacr.report import collect_report, render_html

    key = rj.start_blinding(["x"], scope="annotate", src=journal)
    rj.lock_analysis(_settings(journal), app_key="ml_analyze")
    with rj.open_run("ml_analyze", _settings(journal)) as clean:
        pass
    rj.unblind(key["key_id"])
    with rj.open_run("ml_analyze",
                     _settings(journal, prediction_threshold=0.9)) as late:
        pass
    page = render_html(collect_report(journal,
                                      run_dirs=[clean.dir, late.dir]))

    tree = _Tree()
    tree.feed(page)
    tree.close()
    assert tree.errors == [] and len(tree.stack) == 1
    assert [d.lower() for d in tree.decl] == ["doctype html"]
    html, = tree.root["kids"]
    assert html["tag"] == "html"
    assert [k["tag"] for k in html["kids"]] == ["head", "body"]
    assert len(_find(html, "title")) == 1 and not _find(html, "script")
    sections = _find(html, "section")
    ids = [s["attrs"].get("id") for s in sections]
    assert all(ids) and len(ids) == len(set(ids))
    toc = [a["attrs"]["href"] for a in _find(_find(html, "nav")[0], "a")]
    assert toc == [f"#{i}" for i in ids]

    top = [li["text"] for li in _find(sections[0], "li")]
    assert any(late.dir.name in t and "POST-HOC" in t for t in top)
    assert not any(clean.dir.name in t for t in top)
    locked = [s for s in sections if "Preregistered analysis lock" in s["text"]]
    assert len(locked) == 1
    listed = [li["text"] for li in _find(locked[0], "li")]
    assert any(clean.dir.name in t and "verified" in t for t in listed)
    assert any(late.dir.name in t and "POST-HOC" in t for t in listed)
