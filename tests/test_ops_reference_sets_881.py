"""Item 881: versioned gene-gene reference sets, SELECT / REPORT halves and the 2022 time split.

Small fixtures in the published file formats stand in for CORUM, KEGG,
STRING and BioGRID; nothing is downloaded.
"""
import gzip
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from spacr import ops_reference as R
from spacr.tabular import read_table, table_format


@pytest.fixture(autouse=True)
def _fresh_cache():
    R.clear_cache()
    yield
    R.clear_cache()


def _brieflow(folder):
    corum = pd.DataFrame({"gene_name": ["A", "B", "C", "B", "D", "E", "F", "G", None],
                          "group": ["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"]})
    corum.to_csv(folder / "corum_group_benchmark.tsv", sep="\t", index=False)
    kegg = pd.DataFrame({"gene_name": ["A", "D", "E"], "group": ["k1", "k1", "k1"]})
    kegg.to_csv(folder / "kegg_group_benchmark.tsv", sep="\t", index=False)
    string = pd.DataFrame({"gene_name": ["A", "C", "B", "B", "E"], "pair": [1, 1, 2, 2, 3]})
    string.to_csv(folder / "string_pair_benchmark.tsv", sep="\t", index=False)


def _prepared(folder):
    pd.DataFrame({"a": ["A", "A", "B"], "b": ["B", "C", "D"], "combined_score": [900, 500, 700]}
                 ).to_parquet(folder / "string_v12_ge400.parquet", index=False)
    pd.DataFrame({"a": ["A", "C"], "b": ["E", "F"]}).to_parquet(folder / "biogrid_physical.parquet", index=False)


def test_pairs_of_every_kind_of_reference(tmp_path):
    _brieflow(tmp_path)
    _prepared(tmp_path)
    assert R.reference_pairs("CORUM", tmp_path) == {("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("E", "F")}
    assert R.reference_pairs("KEGG", str(tmp_path)) == {("A", "D"), ("A", "E"), ("D", "E")}
    assert R.reference_pairs("STRING_brieflow", [tmp_path]) == {("A", "C")}
    assert R.reference_pairs("STRING>=0.7", tmp_path) == {("A", "B"), ("B", "D")}
    assert R.reference_pairs("STRING>=0.4", tmp_path) == {("A", "B"), ("A", "C"), ("B", "D")}
    assert R.reference_pairs("BioGRID", tmp_path) == {("A", "E"), ("C", "F")}
    groups = R.reference_groups("CORUM", tmp_path)
    assert list(groups) == ["c1", "c2", "c3", "c4"] and groups["c4"] == ["G"]
    with pytest.raises(KeyError):
        R.reference_groups("STRING>=0.7", tmp_path)


def test_halves_follow_the_seeded_complex_split_and_the_md5_parity(tmp_path):
    _brieflow(tmp_path)
    _prepared(tmp_path)
    halves = R.group_halves(["c1", "c2", "c3", "c4"])
    perm = np.random.default_rng(20261107).permutation(4)
    assert halves == {["c1", "c2", "c3", "c4"][i]: k % 2 for k, i in enumerate(perm)}
    select = R.reference_pairs("CORUM@SELECT", tmp_path)
    report = R.reference_pairs("CORUM@REPORT", tmp_path)
    assert not select & report
    assert select | report <= R.reference_pairs("CORUM", tmp_path)
    for p in R.reference_pairs("STRING>=0.4", tmp_path):
        h = int(hashlib.md5(f"{p[0]}|{p[1]}".encode()).hexdigest(), 16) % 2
        assert R.pair_half(*p) == h
        assert p in R.reference_pairs("STRING>=0.4@" + ("SELECT", "REPORT")[h], tmp_path)
    with pytest.raises(KeyError):
        R.reference_pairs("CORUM@LATER", tmp_path)
    assert R.split_references(["CORUM", "KEGG"], "select") == ["CORUM@SELECT", "KEGG@SELECT"]
    assert R.split_references(["CORUM@SELECT"], "REPORT") == ["CORUM@REPORT"]
    assert R.split_references(["CORUM"], "all") == ["CORUM"]
    with pytest.raises(ValueError):
        R.split_references(["CORUM"], "both")


def test_a_shared_report_pair_is_dropped_from_the_report_half(tmp_path):
    groups = {f"c{k}": ["X", "Y"] for k in range(6)}
    rows = [(g, c) for c, members in groups.items() for g in members]
    pd.DataFrame(rows, columns=["gene_name", "group"]).to_csv(
        tmp_path / "corum_group_benchmark.tsv", sep="\t", index=False)
    assert R.reference_pairs("CORUM@SELECT", tmp_path) == {("X", "Y")}
    assert R.reference_pairs("CORUM@REPORT", tmp_path) == frozenset()


def test_files_are_found_in_the_first_folder_that_has_them(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    _prepared(second)
    assert R.reference_file("BioGRID@SELECT", [first, second]) == str(second / "biogrid_physical.parquet")
    with pytest.raises(FileNotFoundError, match="corum_group_benchmark.tsv"):
        R.reference_file("CORUM", [first, second])
    with pytest.raises(KeyError):
        R.reference_file("GO", first)
    with pytest.raises(ValueError):
        R.reference_dirs("  ")
    with pytest.raises(ValueError):
        R.reference_dirs(None)


def test_positive_matrix_controls_and_symbols():
    genes = ["A", "B&X", "C", "nontargeting_1", "A"]
    m = R.positive_matrix(genes, {("A", "B"), ("A", "A"), ("C", "Q")})
    assert m[0, 1] and m[1, 0] and m[4, 1] and m[0, 4] and m[4, 0]
    assert not m.diagonal().any() and not m[2].any()
    assert R.gene_symbol("B&X") == "B"
    assert R.is_control(["NonTargeting_3", "NTpseudo_2", "A"]).tolist() == [True, True, False]


def test_record_names_file_md5_release_and_pairs(tmp_path):
    _prepared(tmp_path)
    record = R.reference_record(["STRING>=0.7@REPORT"], tmp_path)["STRING>=0.7@REPORT"]
    path = tmp_path / "string_v12_ge400.parquet"
    assert record["file"] == str(path)
    assert record["md5"] == hashlib.md5(path.read_bytes()).hexdigest()
    assert record["version"].startswith("STRING v12.0")
    assert record["pairs"] == len(R.reference_pairs("STRING>=0.7@REPORT", tmp_path))


def _gz(path, text):
    with gzip.open(path, "wt") as handle:
        handle.write(text)


_BIOGRID_COLUMNS = ["Entrez Gene Interactor A", "Entrez Gene Interactor B", "Official Symbol Interactor A",
                    "Official Symbol Interactor B", "Experimental System", "Experimental System Type",
                    "Organism ID Interactor A", "Organism ID Interactor B"]


def _biogrid_zip(path, rows):
    frame = pd.DataFrame(rows, columns=_BIOGRID_COLUMNS)
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("BIOGRID-ORGANISM-Mus_musculus.tab3.txt", "x\n")
        handle.writestr("BIOGRID-ORGANISM-Homo_sapiens-9.9.tab3.txt", frame.to_csv(sep="\t", index=False))


def _downloads(tmp_path):
    raw, archive, brieflow = tmp_path / "raw", tmp_path / "archive", tmp_path / "bf"
    for d in (raw / "string_v12", raw / "biogrid", archive / "string_v11.5", archive / "biogrid_2022",
              archive / "corum", brieflow):
        d.mkdir(parents=True)
    _brieflow(brieflow)
    _gz(raw / "string_v12" / "9606.protein.info.v12.0.txt.gz",
        "#string_protein_id\tpreferred_name\tsize\np1\tA\t1\np2\tB\t1\np3\tC\t1\np4\tD\t1\np5\tE\t1\n")
    _gz(raw / "string_v12" / "9606.protein.links.full.v12.0.txt.gz",
        "protein1 protein2 neighborhood combined_score\n"
        "p1 p2 0 900\np2 p1 0 900\np1 p3 0 450\np2 p4 0 800\np4 p2 0 800\np3 p5 0 750\np1 p9 0 990\n")
    _gz(archive / "string_v11.5" / "9606.protein.info.v11.5.txt.gz",
        "protein_external_id\tpreferred_name\tsize\np1\tA_old\t1\np2\tB\t1\np3\tC\t1\np4\tD\t1\n")
    _gz(archive / "string_v11.5" / "9606.protein.links.v11.5.txt.gz",
        "protein1 protein2 combined_score\np1 p2 950\np2 p1 950\np1 p3 500\np3 p4 300\n")
    new = [(1, 2, "A", "B", "Two-hybrid", "physical", 9606, 9606),
           (2, 1, "B", "A", "Affinity", "physical", 9606, 9606),
           (3, 4, "C", "D", "Affinity", "physical", 9606, 9606),
           (1, 3, "A", "C", "Synthetic", "genetic", 9606, 9606),
           (5, 6, "E", "F", "Affinity", "physical", 9606, 10090),
           (1, 1, "A", "A", "Affinity", "physical", 9606, 9606),
           (3, 9, "C", "Z", "Affinity", "physical", 9606, 9606)]
    old = [(1, 2, "A1", "B", "Two-hybrid", "physical", 9606, 9606),
           (3, 1, "C", "A1", "Affinity", "physical", 9606, 9606),
           (4, 3, "D", "C", "Synthetic", "genetic", 9606, 9606)]
    _biogrid_zip(raw / "biogrid" / "BIOGRID-ORGANISM-LATEST.tab3.zip", new)
    _biogrid_zip(archive / "biogrid_2022" / "BIOGRID-ORGANISM-4.4.214.tab3.zip", old)
    complexes = pd.DataFrame({"Organism": ["Human", "Human", "Mouse"],
                              "subunits(Gene name)": ["A;B;None", "D;E", "A;C"]})
    with zipfile.ZipFile(archive / "corum" / "corum3_allComplexes_wayback20221016.txt.zip", "w") as handle:
        handle.writestr("allComplexes.txt", complexes.to_csv(sep="\t", index=False))
    return raw, archive, brieflow


def test_current_references_from_string_and_biogrid_downloads(tmp_path):
    raw, _, _ = _downloads(tmp_path)
    out = tmp_path / "prepared"
    counts = R.prepare_current_references(str(raw), str(out))
    string = pd.read_parquet(out / "string_v12_ge400.parquet")
    assert set(zip(string.a, string.b)) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")}
    assert "neighborhood" in string and "protein1" not in string
    biogrid = pd.read_parquet(out / "biogrid_physical.parquet")
    assert biogrid.to_dict("records") == [
        {"a": "A", "b": "B", "n_evidence": 2, "systems": "Affinity;Two-hybrid"},
        {"a": "C", "b": "D", "n_evidence": 1, "systems": "Affinity"},
        {"a": "C", "b": "Z", "n_evidence": 1, "systems": "Affinity"}]
    assert counts == {"string_v12_ge400_pairs": 4, "string_v12_ge700_pairs": 3, "biogrid_physical_pairs": 3,
                      "biogrid_file": "BIOGRID-ORGANISM-Homo_sapiens-9.9.tab3.txt"}
    assert sorted(p.name for p in out.iterdir()) == ["biogrid_physical.parquet", "string_v12_ge400.parquet"]


def test_time_split_diffs_on_stable_ids(tmp_path):
    raw, archive, brieflow = _downloads(tmp_path)
    out = tmp_path / "refs"
    versions = R.prepare_time_split_references(str(archive), str(raw), str(brieflow), str(out))
    pairs = {f: set(zip(*pd.read_parquet(out / f"{f}.parquet")[["a", "b"]].T.values))
             for f in ("string_v11.5_ge400", "string_v11.5_ge700", "string_new_v12_ge700",
                       "biogrid_4.4.214_physical", "biogrid_new_after_4.4.214", "corum3_pairs",
                       "corum_new_after_3.0")}
    assert pairs["string_v11.5_ge400"] == {("A", "B"), ("A", "C")}
    assert pairs["string_v11.5_ge700"] == {("A", "B")}
    assert pairs["string_new_v12_ge700"] == {("B", "D")}
    assert pairs["biogrid_4.4.214_physical"] == {("A", "B"), ("A", "C")}
    assert pairs["biogrid_new_after_4.4.214"] == set()
    assert pairs["corum3_pairs"] == {("A", "B"), ("D", "E")}
    assert pairs["corum_new_after_3.0"] == {("B", "D")}
    assert versions["string"] == {"v11.5_ge400_pairs": 2, "v11.5_ge700_pairs": 1, "new_v12_ge700_pairs": 1}
    assert versions["biogrid"]["added_pairs"] == 0 and versions["biogrid"]["old_pairs"] == 2
    assert versions["corum"]["corum3_human_complexes"] == 2
    archive_md5 = hashlib.md5((archive / "corum" / "corum3_allComplexes_wayback20221016.txt.zip")
                              .read_bytes()).hexdigest()
    assert versions["CORUM_2022"].endswith(archive_md5)
    assert R.reference_versions(str(out)) == json.loads((out / "versions.json").read_text())
    assert R.reference_pairs("STRINGnew>=0.7", out) == {("B", "D")}
    again = R.prepare_time_split_references(str(archive), str(raw), str(brieflow), str(out), parts=("corum",))
    assert again["string"] == versions["string"]
    assert R.reference_versions(str(tmp_path)) is None


def test_compressed_text_reads_through_the_table_funnel(tmp_path):
    path = tmp_path / "x.tsv.gz"
    _gz(path, "a\tb\n1\t2\n")
    assert table_format(path) == "csv"
    assert read_table(path, report=None).to_dict("records") == [{"a": 1, "b": 2}]
    with pytest.raises(Exception):
        table_format(tmp_path / "x.gz")
