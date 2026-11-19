"""Item 973: Voyager3 annotates OPS clusters from fetched UniProt facts through an injected AI provider.

No real AI and no network: UniProt is answered by a fake ``spacr.uniprot._get``
and the AI by stub providers.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from spacr import uniprot, voyager3 as V

ROOT = Path(__file__).resolve().parents[1]

HEADER = ("Entry\tGene Names (primary)\tGene Names\tReviewed\tAnnotation\tProtein names\tFunction [CC]\t"
          "Gene Ontology (biological process)\tGene Ontology (cellular component)\t"
          "Gene Ontology (molecular function)\tPathway\tSubcellular location [CC]")

ENTRIES = {
    "THOC1": ("Q96FV9", "5.0", "THO complex subunit 1", "FUNCTION: Part of the THO complex (PubMed:123). {ECO:1}",
              "mRNA export from nucleus [GO:0006406]; RNA splicing [GO:0008380]"),
    "THOC2": ("Q8NI27", "5.0", "THO complex subunit 2", "FUNCTION: Part of the THO complex.",
              "mRNA export from nucleus [GO:0006406]"),
    "ALYREF": ("Q86V81", "5.0", "THO complex subunit 4", "FUNCTION: Export adapter.",
               "mRNA export from nucleus [GO:0006406]"),
    "LENG8": ("Q96PV6", "2.0", "Leukocyte receptor cluster member 8", "", ""),
    "RPL3": ("P39023", "5.0", "60S ribosomal protein L3", "FUNCTION: Ribosome.", "cytoplasmic translation [GO:0002181]"),
    "RPL4": ("P36578", "5.0", "60S ribosomal protein L4", "FUNCTION: Ribosome.", "cytoplasmic translation [GO:0002181]"),
    "RPL5": ("P46777", "5.0", "60S ribosomal protein L5", "FUNCTION: Ribosome.", "cytoplasmic translation [GO:0002181]"),
}


def _tsv(genes):
    """The fake UniProt TSV for the asked genes (one reviewed, one unreviewed row for THOC1)."""
    lines = [HEADER]
    for gene in genes:
        if gene not in ENTRIES:
            continue
        acc, score, name, function, go_p = ENTRIES[gene]
        lines.append("\t".join([acc, gene, gene, "reviewed", score, name, function, go_p, "nucleus [GO:1]",
                                "RNA binding [GO:2]", "", "SUBCELLULAR LOCATION: Nucleus {ECO:2}."]))
        if gene == "THOC1":
            lines.append("\t".join(["X0X0X0", gene, gene, "unreviewed", "1.0", "fragment", "", "", "", "", "", ""]))
    return "\n".join(lines) + "\n"


@pytest.fixture
def fake_uniprot(monkeypatch):
    """Answer UniProt from :data:`ENTRIES`, counting the requests and stating a release."""
    calls = []

    def fake_get(url):
        """Return the TSV for the genes named in ``url`` and no next page."""
        calls.append(url)
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)["query"][0]
        uniprot._RELEASE.update(release="2026_04", release_date="04-November-2026")
        return _tsv(re.findall(r"gene:([A-Za-z0-9-]+)", query)), ""

    monkeypatch.setattr(uniprot, "_get", fake_get)
    monkeypatch.setattr(V, "REQUEST_GAP_S", 0.0)
    return calls


class StubProvider:
    """Answers with the members that share a GO process term, as JSON in prose."""

    name = "stub"
    label = "Stub"

    def __init__(self, answer=None):
        """Use ``answer(prompt)`` when given, else the GO rule."""
        self.answer = answer
        self.prompts = []

    def stream_chat(self, messages, system="", model=None):
        """Yield one answer for the single user message."""
        prompt = messages[-1]["content"]
        self.prompts.append((prompt, system, model))
        if self.answer is not None:
            yield self.answer(prompt)
            return
        genes = re.findall(r"^### (\S+)", prompt, flags=re.M)
        support = [g for g in genes if g.startswith(("THOC", "ALYREF"))]
        body = {"cluster_id": "x", "biological_role": "mRNA export", "confidence": "high",
                "supporting_genes": [{"gene": g, "evidence": "GO process"} for g in support],
                "misfit_genes": [g for g in genes if g not in support],
                "uncharacterized_candidates": [{"gene": "LENG8", "suggested_role": "mRNA export", "priority": 7,
                                                "rationale": "low annotation score"}] if "LENG8" in genes else [],
                "summary": "TREX."}
        yield "Answer:\n```json\n" + json.dumps(body) + "\n```"


def _clusters(tmp_path, extra=()):
    """A cluster table: an mRNA-export cluster, a ribosome cluster, a control and a singleton."""
    rows = [("THOC1", 1), ("THOC2", 1), ("ALYREF", 1), ("LENG8", 1), ("RPL3", 2), ("RPL4", 2), ("RPL5", 2),
            ("nontargeting_1", 2), ("SOLO", 3)] + list(extra)
    path = tmp_path / "clusters.csv"
    pd.DataFrame(rows, columns=["gene", "leiden_r10"]).to_csv(path, index=False)
    return str(path)


def _settings(tmp_path, **more):
    """Voyager3 settings on the test cluster table."""
    return {"dst_root": str(tmp_path / "run"), "ops_voyager3_clusters": _clusters(tmp_path),
            "ops_voyager3_cluster_column": "leiden_r10", **more}


def test_facts_are_cleaned_matched_to_the_reviewed_entry_and_cached(fake_uniprot, tmp_path):
    cache = str(tmp_path / "cache")
    facts, meta = V.fetch_gene_facts(["THOC1", "LENG8", "MISSING"], "human", cache_dir=cache)
    assert facts["THOC1"]["accession"] == "Q96FV9" and facts["THOC1"]["function"] == "Part of the THO complex."
    assert facts["THOC1"]["go_process"] == "mRNA export from nucleus; RNA splicing"
    assert facts["THOC1"]["location"] == "Nucleus."
    assert facts["LENG8"]["annotation_score"] == 2.0 and facts["MISSING"]["found"] is False
    assert meta["releases"] == ["2026_04"] and meta["requests"] == 1 and meta["taxon"] == 9606
    again, meta2 = V.fetch_gene_facts(["THOC1", "LENG8", "MISSING"], "human", cache_dir=cache)
    assert len(fake_uniprot) == 1 and meta2["requests"] == 0 and "reviewed%3Atrue" in fake_uniprot[0]
    assert again == facts and meta2["releases"] == ["2026_04"]


def test_the_default_uniprot_query_and_cache_key_are_unchanged(fake_uniprot, tmp_path):
    resolution = uniprot.resolve("human")
    frame, busy, meta = uniprot._fetch_gene_rows(resolution, ["THOC1"], cache_dir=str(tmp_path))
    assert meta == {} and not busy and "reviewed" not in fake_uniprot[0] and "cc_pathway" not in fake_uniprot[0]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["uniprot_tax_9606_genes_" + __import__("hashlib").sha1(b"THOC1").hexdigest()[:16] + ".json"]


def test_an_unknown_organism_is_refused():
    with pytest.raises(ValueError, match="organism"):
        V.fetch_gene_facts(["A"], "not an organism at all")


def test_the_prompt_lists_every_gene_once_with_the_schema_and_is_stable(fake_uniprot, tmp_path):
    facts, meta = V.fetch_gene_facts(["THOC1", "LENG8"], cache_dir=str(tmp_path))
    text = V.build_cluster_prompt(7, ["LENG8", "THOC1", "THOC1"], facts, {"mean_stability": 0.8}, "HeLa", meta)
    assert re.findall(r"^### (\S+)", text, flags=re.M) == ["LENG8", "THOC1"]
    assert V.TEMPLATE_VERSION in text and '"uncharacterized_candidates"' in text and "release 2026_04" in text
    assert "mean stability 0.8" in text and "Screen context: HeLa" in text and '"cluster_id": "7"' in text
    assert text == V.build_cluster_prompt(7, ["THOC1", "LENG8"], facts, {"mean_stability": 0.8}, "HeLa", meta)


GENES = ["THOC1", "THOC2", "ALYREF", "LENG8"]
GOOD = {"cluster_id": "1", "biological_role": "mRNA export", "confidence": "High",
        "supporting_genes": [{"gene": "thoc1", "evidence": "GO"}, "THOC2", {"gene": "ALYREF"}],
        "misfit_genes": [], "uncharacterized_candidates": [
            {"gene": "LENG8", "suggested_role": "export", "priority": "7", "rationale": "r"}], "summary": "s"}


@pytest.mark.parametrize("answer, status", [
    ("```json\n" + json.dumps(GOOD) + "\n```", "ok"),
    ("The answer is " + json.dumps(GOOD) + " and that is all.", "ok"),
    ("no json here", "unparsed"),
    ('```json\n{"biological_role": "x", "confidence": \n```', "unparsed"),
    ("", "no_answer"),
    (None, "no_answer"),
    ("[1, 2, 3]", "unparsed"),
    (json.dumps(dict(GOOD, supporting_genes=["THOC1", "NOTAMEMBER"])), "partial"),
    (json.dumps(dict(GOOD, uncharacterized_candidates=[{"gene": "LENG8", "priority": 11}])), "partial"),
    (json.dumps(dict(GOOD, uncharacterized_candidates=[{"gene": "LENG8", "priority": "high"}])), "partial"),
    (json.dumps({k: v for k, v in GOOD.items() if k != "misfit_genes"}), "partial"),
    (json.dumps(dict(GOOD, confidence="certain")), "partial"),
    (json.dumps(dict(GOOD, supporting_genes="THOC1")), "partial"),
    (json.dumps(GOOD)[:120], "unparsed"),
])
def test_answers_are_parsed_without_raising(answer, status):
    parsed = V.parse_cluster_answer(answer, GENES)
    assert parsed["status"] == status
    if status == "ok":
        assert [e["gene"] for e in parsed["supporting"]] == ["THOC1", "THOC2", "ALYREF"]
        assert parsed["confidence_model"] == "high" and parsed["confidence_rule"] == "high"
        assert parsed["support_fraction"] == 0.75 and parsed["candidates"][0]["priority"] == 7
    if status != "ok":
        assert parsed["reasons"]


def test_an_unknown_gene_is_dropped_and_counted():
    parsed = V.parse_cluster_answer(json.dumps(dict(GOOD, supporting_genes=["THOC1", "NOTAMEMBER"])), GENES)
    assert parsed["unknown_genes"] == ["NOTAMEMBER"] and parsed["support_fraction"] == 0.25
    assert parsed["confidence_rule"] == "none" and parsed["confidence_agrees"] is False


@pytest.mark.parametrize("fraction, word", [(0.71, "high"), (0.7, "medium"), (0.5, "medium"), (0.49, "low"),
                                            (0.3, "low"), (0.29, "none")])
def test_the_confidence_rule(fraction, word):
    assert V._confidence_from_fraction(fraction) == word


def test_a_run_writes_tables_report_and_provenance_and_the_replay_is_identical(fake_uniprot, tmp_path):
    stub = StubProvider()
    out = V.run_voyager3(_settings(tmp_path, ops_voyager3_context="HeLa", ops_voyager3_model="m1",
                                   ops_voyager3_shuffle_control=True), report=lambda text: None, provider=stub)
    folder = Path(out["folder"])
    assert folder == tmp_path / "run" / "results" / "ops_voyager3" and out["clusters"] == 2
    clusters = pd.read_csv(folder / "voyager3_clusters.csv", dtype=str, keep_default_na=False)
    assert list(clusters["cluster"]) == ["1", "2", "S1", "S2"]
    assert list(clusters["kind"]) == ["real", "real", "shuffled", "shuffled"]
    first = clusters.iloc[0]
    assert first["supporting_genes"] == "ALYREF;THOC1;THOC2" and first["misfit_genes"] == "LENG8"
    assert first["support_fraction"] == "0.7500" and first["confidence_rule"] == "high"
    cands = pd.read_csv(folder / "voyager3_candidates.csv", dtype=str, keep_default_na=False)
    assert cands.iloc[0]["gene"] == "LENG8" and cands.iloc[0]["annotation_score"] == "2"
    assert all(model == "m1" and system == V.SYSTEM_PROMPT for _p, system, model in stub.prompts)
    assert not any("nontargeting" in prompt or "SOLO" in prompt for prompt, _s, _m in stub.prompts)
    record = json.loads((folder / "voyager3_provenance.json").read_text())
    assert record["voyager3"] == V.TEMPLATE_VERSION and record["provider"] == "stub" and record["model"] == "m1"
    assert record["uniprot"]["releases"] == ["2026_04"] and record["clusters"][0]["prompt"] == stub.prompts[0][0]
    assert record["facts"]["THOC1"]["accession"] == "Q96FV9" and record["shuffle_seed"] == V.SHUFFLE_SEED
    report = (folder / "voyager3_report.md").read_text()
    assert "## Cluster 1 (real, 4 genes)" in report and "## Glossary" in report
    before = {name: (folder / name).read_bytes() for name in
              ("voyager3_clusters.csv", "voyager3_candidates.csv", "voyager3_gene_facts.csv", "voyager3_report.md")}
    replay = tmp_path / "replay"
    V.voyager3_replay(folder / "voyager3_provenance.json", str(replay))
    assert {name: (replay / name).read_bytes() for name in before} == before
    calls = len(fake_uniprot)
    V.voyager3_replay(folder / "voyager3_provenance.json")
    assert {name: (folder / name).read_bytes() for name in before} == before and len(fake_uniprot) == calls


def test_a_failing_provider_is_asked_twice_and_the_cluster_kept(fake_uniprot, tmp_path):
    class Broken(StubProvider):
        """Raises on every call."""

        def stream_chat(self, messages, system="", model=None):
            """Fail like a CLI that exited."""
            self.prompts.append(messages)
            raise RuntimeError("cli exited 1")
            yield ""

    broken = Broken()
    out = V.run_voyager3(_settings(tmp_path, ops_voyager3_cluster_ids="1"), report=lambda t: None, provider=broken)
    row = pd.read_csv(out["outputs"]["clusters"], dtype=str, keep_default_na=False).iloc[0]
    assert len(broken.prompts) == 2 and row["status"] == "no_answer" and "cli exited 1" in row["reasons"]


def test_a_slow_provider_times_out(fake_uniprot, tmp_path):
    class Slow(StubProvider):
        """Sleeps past the timeout."""

        cancelled = 0

        def stream_chat(self, messages, system="", model=None):
            """Wait, then answer too late."""
            time.sleep(1.0)
            yield "late"

        def cancel_stream(self):
            """Count the cancellations."""
            Slow.cancelled += 1

    out = V.run_voyager3(_settings(tmp_path, ops_voyager3_cluster_ids=[2], ops_voyager3_timeout=0.05),
                         report=lambda t: None, provider=Slow())
    row = pd.read_csv(out["outputs"]["clusters"], dtype=str, keep_default_na=False).iloc[0]
    assert row["status"] == "no_answer" and "timed out" in row["reasons"] and Slow.cancelled == 2


def test_progress_lines_are_ones_the_module_bar_reads(fake_uniprot, tmp_path):
    from spacr.qt.run_progress import parse_progress

    lines = []
    V.run_voyager3(_settings(tmp_path), report=lines.append, provider=StubProvider())
    samples = [s for line in lines for s in parse_progress(line)]
    assert [(s.done, s.total, s.label) for s in samples] == [(1, 2, "Voyager3"), (2, 2, "Voyager3")]
    assert any("0 replayed, 2 asked" in line for line in lines)


def test_missing_clusters_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no cluster table"):
        V.run_voyager3({"dst_root": str(tmp_path)}, provider=StubProvider())
    with pytest.raises(ValueError, match="no 'cluster' column"):
        V.run_voyager3({"ops_voyager3_clusters": _clusters(tmp_path)}, provider=StubProvider())


def test_none_is_refused(monkeypatch):
    import spacr.qt.preferences as prefs

    monkeypatch.setattr(prefs, "get_preferred_provider", lambda: "none")
    with pytest.raises(ValueError, match="says None"):
        V._resolve_provider("")


def test_the_free_provider_is_accepted(monkeypatch):
    import spacr.qt.preferences as prefs
    from spacr.qt.ai import providers

    hosted = providers.hosted_provider()
    monkeypatch.setattr(prefs, "get_preferred_provider", lambda: "spacr_hosted")
    monkeypatch.setattr(type(hosted), "is_configured", lambda self: True)
    assert V._resolve_provider("") is hosted
    assert V._resolve_provider("spacr_hosted") is hosted
    monkeypatch.setattr(type(hosted), "is_configured", lambda self: False)
    with pytest.raises(ValueError, match="not usable now"):
        V._resolve_provider("spacr_hosted")


def test_the_report_names_the_free_provider_and_its_light_model(fake_uniprot, tmp_path):
    class Free(StubProvider):
        """Answers like spaCR AI (free) on its light tier."""

        name = "spacr_hosted"
        last_tier = "light"

    out = V.run_voyager3(_settings(tmp_path), report=lambda t: None, provider=Free())
    report = Path(out["outputs"]["report"]).read_text()
    record = json.loads(Path(out["provenance"]).read_text())
    assert "AI provider: spacr_hosted" in report and "Quality note" in report and "2 of 2 answers" in report
    assert record["provider"] == "spacr_hosted" and record["clusters"][0]["attempts"][0]["tier"] == "light"
    (tmp_path / "b").mkdir()
    plain = V.run_voyager3(_settings(tmp_path / "b"), report=lambda t: None, provider=StubProvider())
    assert "Quality note" not in Path(plain["outputs"]["report"]).read_text()


def test_a_named_provider_must_be_signed_in(monkeypatch):
    from spacr.qt.ai import providers

    class Offline:
        """A provider that is not configured."""

        name = "codex"

        def is_configured(self):
            """Not installed."""
            return False

    monkeypatch.setattr(providers, "get_provider", lambda name: Offline())
    with pytest.raises(ValueError, match="not usable now"):
        V._resolve_provider("codex")


def test_voyager3_never_uses_a_vendor_api():
    for path in (ROOT / "spacr" / "voyager3.py", ROOT / "spacr" / "qt" / "voyager3_panel.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "import anthropic" not in source and "from anthropic" not in source
        assert "api_key" not in source and "x-api-key" not in source and "message_batches" not in source and "batches.create" not in source
        assert "openai" not in source and "generativeai" not in source


def test_the_settings_are_off_by_default_and_alpha():
    from spacr.ops_settings import OPS_CATEGORIES, ops_defaults
    from spacr.settings import ALPHA_FEATURES

    defaults = ops_defaults()
    keys = [k for k in defaults if k.startswith("ops_voyager3_")]
    assert defaults["ops_voyager3_enabled"] is False and defaults["ops_voyager3_shuffle_control"] is False
    assert set(ALPHA_FEATURES[973]["settings"]) | set(ALPHA_FEATURES[980]["settings"]) == set(keys) and len(keys) == 16
    assert defaults["ops_voyager3_resume"] is True and defaults["ops_voyager3_answers_dir"] is None
    assert all(k in OPS_CATEGORIES["Voyager3 cluster annotation"] for k in keys)


def test_the_profile_path_runs_voyager3_only_when_asked(monkeypatch, tmp_path):
    from spacr import ops_clusters, ops_tvn as T

    ran = []
    monkeypatch.setattr(T, "_profile_state", lambda *args: {})
    monkeypatch.setattr(ops_clusters, "cluster_sweep_phase", lambda *a, **k: {"outputs": {}})
    monkeypatch.setattr(V, "run_voyager3", lambda settings, report=None: ran.append(1) or {
        "outputs": {"clusters": "c.csv"}})
    T.run_profile_phases({"dst_root": str(tmp_path)}, phases=("cluster_sweep",))
    record = json.loads((tmp_path / "ops_profile" / "A" / "runtime.json").read_text())
    assert not ran and not any(k.startswith("ops_voyager3_") for k in record["settings"])
    assert "voyager3" not in record
    T.run_profile_phases({"dst_root": str(tmp_path), "ops_voyager3_enabled": True}, phases=("cluster_sweep",))
    assert ran == [1]
    out = T.run_profile_phases({"dst_root": str(tmp_path)}, phases=("voyager3",))
    record = json.loads((tmp_path / "ops_profile" / "A" / "runtime.json").read_text())
    assert ran == [1, 1] and out["outputs"]["voyager3_clusters"] == "c.csv"
    assert record["settings"]["ops_voyager3_enabled"] is False


def test_run_ops_accepts_the_phase(monkeypatch, tmp_path):
    from spacr import ops_engine, ops_tvn as T

    seen = []
    monkeypatch.setattr(T, "run_profile_phases", lambda settings, phases, report=None: seen.append(phases) or {})
    ops_engine.run_ops({"dst_root": str(tmp_path)}, phases=("voyager3",))
    assert seen == [["voyager3"]] and ops_engine._PROFILE_PHASES[-1] == "voyager3"


def test_a_long_prompt_is_written_with_shorter_facts(monkeypatch):
    fact = {"gene": "G", "found": True, "accession": "P1", "reviewed": True, "annotation_score": 5.0,
            "protein_name": "p", "function": "word " * 200, "go_process": "; ".join(f"t{i}" for i in range(8)),
            "go_component": "c", "go_function": "f", "pathway": "", "location": "Nucleus."}
    genes = [f"G{i}" for i in range(40)]
    facts = {g: dict(fact, gene=g) for g in genes}
    full = V.build_cluster_prompt(1, genes, facts)
    monkeypatch.setattr(V, "PROMPT_BUDGET", len(full) - 1)
    short = V.build_cluster_prompt(1, genes, facts)
    assert len(short) < len(full) and "shortened to fit" in short and "GO component" not in short
    assert "GO process: t0; t1; t2; t3\n" in short and re.findall(r"^### (\S+)", short, flags=re.M) == sorted(genes)


class Counting(StubProvider):
    """The GO-rule stub, counting its calls and reporting a model and a cost like the Claude CLI."""

    calls = 0

    def stream_chat(self, messages, system="", model=None):
        """Count, then answer like :class:`StubProvider`."""
        Counting.calls += 1
        self.last_result = {"modelUsage": {"stub-model-1": {}}, "total_cost_usd": 0.25}
        yield from StubProvider.stream_chat(self, messages, system, model)


TABLES = ("voyager3_clusters.csv", "voyager3_candidates.csv", "voyager3_gene_facts.csv")


def test_an_interrupted_run_resumes_to_the_same_tables(fake_uniprot, tmp_path):
    (tmp_path / "whole").mkdir()
    whole = V.run_voyager3(_settings(tmp_path / "whole"), report=lambda t: None, provider=Counting())
    Counting.calls = 0
    (tmp_path / "cut").mkdir()
    first = _settings(tmp_path / "cut", ops_voyager3_cluster_ids="1")
    V.run_voyager3(first, report=lambda t: None, provider=Counting())
    assert Counting.calls == 1
    answers = sorted((tmp_path / "cut" / "run" / "results" / "ops_voyager3" / "answers").glob("*.json"))
    saved = json.loads(answers[0].read_text())
    assert len(answers) == 1 and saved["models"] == ["stub-model-1"] and saved["cost_usd"] == 0.25
    assert saved["response"] and saved["seconds"] is not None and not list(answers[0].parent.glob("*.tmp"))
    Counting.calls = 0
    lines = []
    again = V.run_voyager3(dict(first, ops_voyager3_cluster_ids=None), report=lines.append, provider=Counting())
    assert Counting.calls == 1 and any("1 replayed, 1 asked" in line for line in lines)
    for name in TABLES:
        assert (Path(again["folder"]) / name).read_bytes() == (Path(whole["folder"]) / name).read_bytes(), name
    record = json.loads(Path(again["provenance"]).read_text())
    assert [c["answer_source"] for c in record["clusters"]] == ["replayed", "asked"]
    assert record["answers"]["replayed"] == 1 and record["answers"]["asked"] == 1


def test_a_changed_model_or_resume_off_asks_again(fake_uniprot, tmp_path):
    settings = _settings(tmp_path, ops_voyager3_answers_dir=str(tmp_path / "saved"))
    Counting.calls = 0
    V.run_voyager3(settings, report=lambda t: None, provider=Counting())
    V.run_voyager3(settings, report=lambda t: None, provider=Counting())
    assert Counting.calls == 2 and len(list((tmp_path / "saved").glob("*.json"))) == 2
    V.run_voyager3(dict(settings, ops_voyager3_model="other"), report=lambda t: None, provider=Counting())
    assert Counting.calls == 4
    V.run_voyager3(dict(settings, ops_voyager3_resume=False), report=lambda t: None, provider=Counting())
    assert Counting.calls == 6


def test_a_changed_template_asks_again(fake_uniprot, tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    Counting.calls = 0
    V.run_voyager3(settings, report=lambda t: None, provider=Counting())
    monkeypatch.setattr(V, "TEMPLATE_VERSION", "voyager3-test")
    V.run_voyager3(settings, report=lambda t: None, provider=Counting())
    assert Counting.calls == 4


def test_a_failed_answer_is_not_saved(fake_uniprot, tmp_path):
    class Broken(StubProvider):
        """Always fails."""

        def stream_chat(self, messages, system="", model=None):
            """Raise like a CLI that exited."""
            raise RuntimeError("exit 1")
            yield ""

    V.run_voyager3(_settings(tmp_path), report=lambda t: None, provider=Broken())
    assert not list((tmp_path / "run" / "results" / "ops_voyager3" / "answers").glob("*"))
