"""Item 980: Voyager3's model-free enrichment method (hypergeometric + BH over UniProt GO/keywords, CORUM, Reactome)."""
from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import uniprot, voyager3 as V

HEADER = "Entry\tGene Names (primary)\tGene Names\tReviewed\tGene Ontology (biological process)\t" \
         "Gene Ontology (cellular component)\tGene Ontology (molecular function)\tKeywords"


def _screen(n_genes=60):
    """Gene names; THO* genes share a GO process and a keyword, RPL* genes a component."""
    genes = [f"THO{i}" for i in range(6)] + [f"RPL{i}" for i in range(6)] + [f"G{i:02d}" for i in range(n_genes - 12)]
    return genes


def _row(gene):
    """One fake reviewed UniProt row for ``gene``."""
    go_p = "mRNA export from nucleus [GO:0006406]" if gene.startswith("THO") else ""
    go_c = "cytosolic large ribosomal subunit [GO:0022625]" if gene.startswith("RPL") else "nucleus [GO:0005634]"
    kw = "mRNA transport; Nucleus" if gene.startswith("THO") else "Nucleus"
    return "\t".join([f"P{abs(hash(gene)) % 99999:05d}", gene, gene, "reviewed", go_p, go_c, "", kw])


@pytest.fixture
def fake_uniprot(monkeypatch):
    """Answer UniProt for any gene query, enrichment fields or fact fields alike."""
    calls = []

    def fake_get(url):
        """Return reviewed rows for the genes named in ``url``."""
        calls.append(url)
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)["query"][0]
        genes = re.findall(r"gene:([A-Za-z0-9-]+)", query)
        if "keyword" in url:
            return HEADER + "\n" + "\n".join(_row(g) for g in genes) + "\n", ""
        head = "Entry\tGene Names (primary)\tGene Names\tReviewed\tAnnotation\tProtein names\tFunction [CC]"
        return head + "\n" + "\n".join(f"P1\t{g}\t{g}\treviewed\t3.0\t{g} protein\t" for g in genes) + "\n", ""

    monkeypatch.setattr(uniprot, "_get", fake_get)
    monkeypatch.setattr(V, "REQUEST_GAP_S", 0.0)
    return calls


def _table(tmp_path):
    """Clusters: 1 = five THO genes + one other, 2 = five RPL genes + three others, 3 = random genes."""
    genes = _screen()
    rows = [(g, 1) for g in genes[:5]] + [("G00", 1)] + [(g, 2) for g in genes[6:11]] + [("G01", 2), ("G02", 2), ("G03", 2)]
    rows += [(g, 3) for g in genes[20:28]] + [(g, 4) for g in genes[28:]] + [("nontargeting_1", 4)]
    path = tmp_path / "clusters.csv"
    pd.DataFrame(rows, columns=["gene", "cluster"]).to_csv(path, index=False)
    return str(path)


def _references(tmp_path):
    """A Reactome GMT and a zipped CORUM table naming the RPL genes."""
    gmt = tmp_path / "ReactomePathways.gmt"
    gmt.write_text("Peptide chain elongation\tR-HSA-156902\tRPL0\tRPL1\tRPL2\tRPL3\tRPL4\n"
                   "A mouse pathway\tR-MMU-1\tRPL0\tRPL1\n")
    corum = tmp_path / "allComplexes.txt"
    corum.write_text("ComplexID\tComplexName\tOrganism\tsubunits(Gene name)\n"
                     "1\tTHO complex\tHuman\tTHO0;THO1;THO2;THO3\n2\tMouse thing\tMouse\tTHO0;THO1;THO2\n")
    zipped = tmp_path / "corum.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.write(corum, "allComplexes.txt")
    (tmp_path / "PROVENANCE.txt").write_text("test reference set, 2026-11-18")
    return str(gmt), str(zipped)


def _settings(tmp_path, **more):
    """Enrichment-mode settings on the test screen."""
    gmt, corum = _references(tmp_path)
    return {"dst_root": str(tmp_path / "run"), "ops_voyager3_clusters": _table(tmp_path),
            "ops_voyager3_method": "enrichment", "ops_voyager3_reactome": gmt, "ops_voyager3_corum": corum, **more}


def test_bh_matches_the_textbook_adjustment():
    assert V._bh([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])


def test_the_enrichment_run_names_the_planted_roles_and_replays(fake_uniprot, tmp_path):
    out = V.run_voyager3(_settings(tmp_path, ops_voyager3_shuffle_control=True), report=lambda t: None)
    folder = Path(out["folder"])
    table = pd.read_csv(folder / "voyager3_clusters.csv", dtype=str, keep_default_na=False).set_index("cluster")
    assert table.loc["1", "confidence_rule"] == "high" and table.loc["1", "support_fraction"] == "0.8333"
    assert table.loc["1", "supporting_genes"] == "THO0;THO1;THO2;THO3;THO4"
    assert table.loc["2", "confidence_rule"] == "medium"
    assert table.loc["3", "confidence_rule"] == "none" and table.loc["3", "biological_role"] == "no enriched term"
    assert set(table["kind"]) == {"real", "shuffled"} and table.loc["1", "confidence_model"] == ""
    terms = pd.read_csv(folder / "voyager3_enrichment_terms.csv", dtype=str, keep_default_na=False)
    assert {"GO process", "UniProt keyword", "Reactome", "CORUM"} <= set(terms["source"])
    assert "R-MMU-1" not in set(terms["term_id"]) and "CORUM:2" not in set(terms["term_id"])
    tho = terms[(terms["cluster"] == "1") & (terms["term_id"] == "GO:0006406")].iloc[0]
    assert (tho["k"], tho["n"], tho["K"], tho["N"]) == ("5", "6", "5", "54")
    cands = pd.read_csv(folder / "voyager3_candidates.csv", dtype=str, keep_default_na=False)
    assert "G00" in set(cands[cands["cluster"] == "1"]["gene"])
    record = json.loads((folder / "voyager3_provenance.json").read_text())
    refs = record["enrichment"]["references"]
    assert record["method"] == "enrichment" and record["enrichment"]["background_size"] == 54
    assert refs["corum"]["sha256"] and refs["reactome"]["provenance_note"].startswith("test reference set")
    before = {p.name: p.read_bytes() for p in folder.glob("voyager3_*") if p.suffix in (".csv", ".md")}
    V.voyager3_replay(folder / "voyager3_provenance.json", str(tmp_path / "replay"))
    assert {n: (tmp_path / "replay" / n).read_bytes() for n in before} == before


def test_a_cached_reference_is_used_when_the_source_is_gone(fake_uniprot, tmp_path):
    settings = _settings(tmp_path)
    V.run_voyager3(settings, report=lambda t: None)
    Path(settings["ops_voyager3_reactome"]).unlink()
    Path(settings["ops_voyager3_corum"]).unlink()
    out = V.run_voyager3(settings, report=lambda t: None)
    refs = json.loads(Path(out["provenance"]).read_text())["enrichment"]["references"]
    assert refs["reactome"]["offline"] and refs["corum"]["offline"]
    with pytest.raises(ValueError, match="no cached copy"):
        V._reference_copy(str(tmp_path / "missing.gmt"), "other", str(tmp_path / "cache"))


def test_planted_truth_is_recovered_and_random_clusters_stay_quiet():
    rng = np.random.default_rng(980)
    genes = [f"g{i}" for i in range(2000)]
    terms = {f"T{t}": set(rng.choice(genes, int(rng.integers(5, 60)), replace=False)) for t in range(200)}
    gene_terms = {g: set() for g in genes}
    for term, members in terms.items():
        for g in members:
            gene_terms[g].add(term)
    planted_ok, random_hits, sizes = 0, 0, rng.integers(10, 41, 80)
    for c in range(40):
        size = int(sizes[c]); share = float(rng.uniform(0.4, 0.9))
        term = f"P{c}"
        members = list(rng.choice(genes, size, replace=False))
        for g in members[: int(round(share * size))]:
            gene_terms[g].add(term)
        terms[term] = {g for g in genes if term in gene_terms[g]}
    term_size = {t: sum(t in gene_terms[g] for g in genes) for t in terms}
    names = {t: ("test", t) for t in terms}
    for c in range(40):
        members = sorted(g for g in genes if f"P{c}" in gene_terms[g])
        others = [g for g in rng.permutation(genes) if g not in members][: int(sizes[c]) - len(members)]
        parsed, _t = V._enrich_cluster(members + list(others), gene_terms, term_size, names, len(genes), 0.05)
        planted_ok += parsed["biological_role"] == f"P{c}"
    for c in range(40, 80):
        members = list(rng.choice(genes, int(sizes[c]), replace=False))
        parsed, _t = V._enrich_cluster(members, gene_terms, term_size, names, len(genes), 0.05)
        random_hits += parsed["confidence_rule"] in ("high", "medium")
    assert planted_ok >= 38 and random_hits <= 2


def test_ai_stays_the_default_and_unknown_methods_are_refused(tmp_path):
    from spacr.ops_settings import ops_defaults

    assert ops_defaults()["ops_voyager3_method"] == "ai"
    with pytest.raises(ValueError, match="ops_voyager3_method"):
        V.run_voyager3({"ops_voyager3_clusters": _table(tmp_path), "ops_voyager3_method": "magic"})


def test_the_enrichment_method_makes_no_ai_call(fake_uniprot, tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_resolve_provider", lambda name: (_ for _ in ()).throw(AssertionError("AI asked")))
    started = time.monotonic()
    V.run_voyager3(_settings(tmp_path), report=lambda t: None)
    assert time.monotonic() - started < 30
