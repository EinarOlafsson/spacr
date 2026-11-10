"""Item 885: protein-knowledge blocks (spacr.gene_knowledge) on synthetic proteins, mocked embedders, CPU only."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import spacr.gene_knowledge as gk

AA = "ACDEFGHIKLMNPQRSTVWY"
N_LIB, N_NT = 30, 3


def _seq(rng, n):
    return "".join(rng.choice(list(AA), n))


def esm_fake(batch):
    """Per-residue vectors from residue identity and position in the window."""
    return [np.array([[AA.index(c) if c in AA else 0, k, 1.0, (AA.index(c) if c in AA else 0) * 0.5 + 0.01 * k]
                      for k, c in enumerate(s)], float) for s in batch]


def text_fake(batch):
    """A deterministic sentence vector from the sentence's letters."""
    out = []
    for s in batch:
        v = np.zeros(12)
        for i, ch in enumerate(s.lower()):
            v[ord(ch) % 12] += 1.0 / (1 + i % 5)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def encode_fake(texts, lens):
    """Encoder states (L x 1024) from the window's characters."""
    out = []
    for t, L in zip(texts, lens):
        body = t.split(" ", 1)[1].replace(" ", "")[:L]
        h = torch.zeros(L, 1024)
        for k, c in enumerate(body):
            h[k, ord(c.upper()) % 1024] = 1.0
            h[k, 1023] = k / 100
        out.append(h)
    return out


def head_fake(h):
    """3Di logits: the state of a residue follows its letter."""
    idx = h[0, :, :1023].argmax(1)
    logits = torch.zeros(1, 20, h.shape[1])
    logits[0, idx % 20, torch.arange(h.shape[1])] = 1.0
    return logits


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("gk")
    rng = np.random.default_rng(0)
    genes = [f"G{i:02d}" for i in range(N_LIB)]
    nodes = pd.DataFrame({"gene": genes + [f"ntpseudo_{k}" for k in range(N_NT)],
                          "sym": genes + [f"ntpseudo_{k}" for k in range(N_NT)],
                          "is_nt": [False] * N_LIB + [True] * N_NT,
                          "fold": [i % 5 for i in range(N_LIB)] + [0, 1, 2]})
    nodes.to_parquet(root / "nodes.parquet")
    seqs = [_seq(rng, int(rng.integers(20, 60))) for _ in genes]
    seqs[5] = _seq(rng, 130)
    seqs[7] = seqs[6]
    fams = ["PF00001", "PF00002", "PF00003", "IPR000001", "IPR000002"]
    cur = []
    for i, g in enumerate(genes):
        pf = ";".join(sorted({fams[i % 3], fams[(i + 1) % 3]})) + ";"
        cur.append({"gene": g, "accession": f"P{i:05d}", "source": "library", "length": len(seqs[i]), "pfam": pf,
                    "interpro": fams[3 + i % 2] + ";", "sequence": seqs[i], "function_current": None})
    for k in range(5):
        s = None if k == 1 else _seq(rng, 40)
        cur.append({"gene": f"DEC{k}", "accession": f"Q{k:05d}", "source": "decoy_pool", "length": 40,
                    "pfam": "PF00002;", "interpro": "IPR000001;", "sequence": s,
                    "function_current": f"Decoy protein {k} binds membranes. Acts in transport."})
    cur = pd.DataFrame(cur)
    cur.to_parquet(root / "current.parquet")
    fn = []
    for i, g in enumerate(genes):
        other = genes[(i + 1) % N_LIB]
        fn.append({"gene": g, "match": "primary", "accession": f"P{i:05d}", "protein_names": g,
                   "function": f"Required for the activity of {other} (PubMed:1). Binds DNA {{ECO:0000269}}. "
                               f"Interacts with P{(i + 2) % N_LIB:05d} in cells."})
    pd.DataFrame(fn).to_csv(root / "cache.tsv", sep="\t", index=False)
    with gzip.open(root / "sprot.dat.gz", "wt") as fh:
        for i, g in enumerate(genes[:25]):
            fh.write(f"ID   {g}_HUMAN\nAC   P{i:05d}; OLD{i};\nOX   NCBI_TaxID=9606;\nGN   Name={g}; Synonyms=X;\n")
            fh.write(f"CC   -!- FUNCTION: Archived function sentinel ARCHIVE text of {g} {'and its partner ' * (i % 4)}in\nCC       {i} cells.\n")
            fh.write("CC   -!- SUBCELLULAR LOCATION: Nucleus.\n")
            fh.write(f"DR   Pfam; PF09{i % 2:03d}; X.\nDR   InterPro; IPR0990{i % 2:02d}; Y.\nDR   GO; GO:1; Z.\n")
            fh.write(f"SQ   SEQUENCE   10 AA;\n     {'W' * 10 + 'C' * (i % 3)} \n//\n")
        fh.write("ID   MOUSE\nAC   M1;\nOX   NCBI_TaxID=10090;\nGN   Name=G00;\n//\n")
        fh.write("ID   DEC0_HUMAN\nAC   Q00000;\nOX   NCBI_TaxID=9606;\nGN   Name=DEC0;\nSQ   SEQUENCE\n     WWWWW\n//\n")
    s = {"gene_knowledge_src": str(root / "out"), "gene_knowledge_nodes": str(root / "nodes.parquet"),
         "gene_knowledge_uniprot_current_table": str(root / "current.parquet"),
         "gene_knowledge_uniprot_archive": str(root / "sprot.dat.gz"),
         "gene_knowledge_uniprot_function_cache": str(root / "cache.tsv"),
         "gene_knowledge_esm_pca": 3, "gene_knowledge_esm_window": 50, "gene_knowledge_esm_stride": 25,
         "gene_knowledge_esm_tokens_per_batch": 200, "gene_knowledge_text_pca": 4, "gene_knowledge_domain_min_genes": 2,
         "gene_knowledge_domain_svd": 2, "gene_knowledge_prostt5_window": 50, "gene_knowledge_prostt5_stride": 30,
         "gene_knowledge_prostt5_residues_per_batch": 120, "gene_knowledge_s3di_pca": 3, "gene_knowledge_kmer_svd": 3,
         "gene_knowledge_struct_spec_dims": 3, "gene_knowledge_device": "cpu"}
    gk.fetch_protein_annotations(s)
    gk.embed_sequences(s, embedder=esm_fake)
    gk.embed_text(s, embedder=text_fake, batch_size=7)
    gk.predict_3di(s, encode=encode_fake, head=head_fake)
    out = Path(s["gene_knowledge_src"]) / "gene_knowledge"
    aa = gk.read_fasta(out / "prostt5" / "aa.fasta")
    keys = list(aa)
    m8 = [(keys[0], keys[0], 1.0, 1e-9, 300), (keys[0], keys[1], 0.9, 1e-8, 200), (keys[1], keys[2], 0.2, 1e-4, 50),
          (keys[2], keys[1], 0.2, 1e-4, 50), (keys[3], keys[-1], 0.5, 1e-5, 80)]
    (root / "aln.m8").write_text("".join("\t".join(map(str, r)) + "\n" for r in m8))
    s["gene_knowledge_foldseek_alignments"] = str(root / "aln.m8")
    img = pd.DataFrame(rng.normal(size=(N_LIB + 2, 5)), columns=[f"emb_{k}" for k in range(5)])
    img.insert(0, "gene", genes + ["NTpseudo_0", "NTpseudo_1"])
    img.to_parquet(root / "img.parquet")
    s["gene_knowledge_image_blocks"] = {"img_dino": str(root / "img.parquet")}
    s["gene_knowledge_image_pca"] = 3
    ref = root / "refs"
    ref.mkdir()
    pd.DataFrame({"gene_name": genes[:10] + genes[10:20], "group": ["c1"] * 10 + ["c2"] * 10}).to_csv(
        ref / "corum_group_benchmark.tsv", sep="\t", index=False)
    pairs = [(genes[i], genes[j]) for i in range(N_LIB) for j in range(i + 1, N_LIB) if (i + j) % 4 == 0]
    pd.DataFrame({"a": [a for a, b in pairs], "b": [b for a, b in pairs], "combined_score": 800}).to_parquet(
        ref / "string_v12_ge400.parquet")
    pd.DataFrame({"a": [a for a, b in pairs[::2]], "b": [b for a, b in pairs[::2]]}).to_parquet(ref / "biogrid_physical.parquet")
    s["gene_knowledge_profile_reference_dir"] = str(ref)
    res = gk.build_knowledge_blocks(s)
    return {"root": root, "s": s, "out": out, "res": res, "nodes": nodes, "cur": cur, "seqs": seqs}


def test_defaults_and_options():
    from spacr.settings import tooltips

    d = gk.get_gene_knowledge_default_settings()
    assert d["gene_knowledge_seed"] == 20261109 and not d["gene_knowledge_allow_download"]
    assert all(k in tooltips for k in d)
    assert gk.options({"gene_knowledge_image_blocks": "{'a': 'b'}"})["image_blocks"] == {"a": "b"}
    with pytest.raises(ValueError):
        gk.out_dir(gk.options({}))


def test_windowed_mean_equals_the_mean_of_window_averaged_residues():
    seqs = ["ACDEFGHIKLMNPQ" * 9, "WY", "ACD" * 20]
    win, stride = 50, 25
    got = gk.windowed_mean(seqs, esm_fake, win, stride, 120, 4)
    for i, s in enumerate(seqs):
        acc = np.zeros((len(s), 4))
        cnt = np.zeros(len(s))
        for a in gk.window_spans(len(s), win, stride):
            acc[a:a + win] += esm_fake([s[a:a + win]])[0]
            cnt[a:a + win] += 1
        assert np.allclose(got[i], (acc / cnt[:, None]).mean(0), atol=1e-5)
    assert gk.window_spans(10, 50, 25) == [0] and gk.window_spans(120, 50, 25) == [0, 25, 50, 70]


def test_swissprot_parser_and_archive_matching(world):
    old = pd.read_parquet(world["out"] / "data" / "uniprot_2022.parquet")
    lib = old[old.source == "library"].set_index("gene")
    assert lib.loc["G03", "match2022"] == "accession" and lib.loc["G29", "match2022"] == "none"
    assert lib.loc["G03", "function_2022"] == "Archived function sentinel ARCHIVE text of G03 and its partner and its partner and its partner in 3 cells."
    assert lib.loc["G03", "pfam_2022"] == "PF09001" and lib.loc["G02", "interpro_2022"] == "IPR099000"
    assert lib.loc["G04", "sequence_2022"] == "W" * 10 + "C"
    assert lib.loc["G00", "acc_all"] == "P00000;OLD0"
    assert old.set_index("gene").loc["DEC0", "match2022"] == "accession"
    e = gk.parse_swissprot_entry(["AC   X1;", "GN   Name=ABC {ECO:1}; Synonyms=Q;", "GN   Name=DEF;"])
    assert e["gene"] == "ABC" and not e["human"] and e["acc"] == ["X1"]


def test_fetch_current_from_a_mocked_rest_service(tmp_path):
    fields = "Entry\tGene Names (primary)\tLength\tPfam\tInterPro\tFunction [CC]\tSequence\n"

    def get(url):
        """Answer the four kinds of UniProt queries."""
        if "accessions=" in url and "Q9" in url:
            return fields + "".join(f"Q9{k}\tD{k}\t5\tPF1;\tIPR1;\tFUNCTION: d.\tMKVLA\n" for k in range(4))
        if "accessions=" in url:
            return fields + "P1\tA\t5\tPF1;\tIPR1;\tFUNCTION: a.\tMKVLA\n"
        if "gene_exact" in url:
            return fields + "P2\tB\t6\t\t\t\tMKVLAA\n"
        return "Entry\tGene Names (primary)\n" + "".join(f"Q9{k}\tD{k}\n" for k in range(4)) + "P1\tA\n"

    cache = pd.DataFrame({"gene": ["A", "B"], "accession": ["P1", None]})
    o = gk.options({"gene_knowledge_decoy_draw": 3})
    out = gk.fetch_current(["A", "B"], cache, o, tmp_path, get=get)
    assert list(out.columns) == ["gene", "accession", "source", "length", "pfam", "interpro", "sequence",
                                 "function_current"]
    assert set(out[out.source == "library"].gene) == {"A", "B"} and (out.source == "decoy_pool").sum() == 3
    assert not set(out[out.source == "decoy_pool"].gene) & {"A", "B"}
    assert (tmp_path / "human_reviewed_index.tsv").exists()


def test_text_masking_hides_fold_genes_and_every_sentence_naming_them(world):
    out = world["out"]
    S = pd.read_parquet(out / "text" / "sentences.parquet")
    assert set(S.version) == {"current", "2022"}
    assert S[S.gene == "G03"].mentions.str.contains("G04").any()
    assert S[(S.gene == "G03") & S.sentence.str.contains("P00005")].mentions.str.contains("G05").all()
    assert not S.sentence.str.contains("PubMed|ECO:").any()
    z = np.load(out / "blocks.npz")
    nodes = world["nodes"]
    summ = json.loads((out / "blocks_summary.json").read_text())
    for k in range(5):
        test = nodes.fold.to_numpy() == k
        assert not z[f"H_txt_f{k}"][test].any()
        assert summ["blocks"][f"txt_f{k}"]["sentences_dropped_naming_fold_genes"] > 0
    SE = np.load(out / "text" / "sent_emb.npy")
    node_of = {g: i for i, g in enumerate(pd.read_parquet(out / "node_protein.parquet").protein)}
    test = nodes.fold.to_numpy() == 2
    names = list(nodes.sym[:N_LIB]) + summ["decoy_proteins"]
    hidden = set(np.array(names)[test]) | set(nodes.gene[test])
    X, h, dropped, _ = gk.text_block(S, SE, "current", node_of, names, list(nodes.gene), test,
                                     ~nodes.is_nt.to_numpy(), 4)
    kept = S[(S.version == "current") & ~S.gene.map(lambda g: test[node_of[g]] if g in node_of else True)]
    bad = kept.mentions.map(lambda m: bool(set(str(m).split(";")) & hidden))
    assert dropped == int(bad.sum()) and dropped > 0
    G = "G00"
    i = node_of[G]
    want = SE[kept[(kept.gene == G) & ~bad].sid.to_numpy()].mean(0)
    M = np.zeros((len(names), SE.shape[1]))
    M[i] = want
    assert h[i] and not h[test].any()


def test_per_fold_pca_never_sees_a_test_row(world, monkeypatch):
    import sklearn.decomposition as skd

    seen = []
    real_fit = skd.PCA.fit

    def fit(self, X, y=None):
        """Record the fit rows."""
        seen.append(np.asarray(X).copy())
        return real_fit(self, X, y)

    monkeypatch.setattr(skd.PCA, "fit", fit)
    M = np.arange(40, dtype=float).reshape(10, 4) + np.eye(10, 4)
    have = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
    lib = np.array([1] * 8 + [0] * 2, bool)
    M[3] = 1e9
    M[8] = 1e9
    X, _ = gk.pca_block(M, have, lib, 2)
    assert len(seen[-1]) == (have & lib).sum() and not (np.abs(seen[-1]) > 1e8).any()
    assert np.all(X[~have] == 0)


def test_time_blocks_read_only_the_archive(world):
    out = world["out"]
    z = np.load(out / "blocks.npz")
    old = pd.read_parquet(out / "data" / "uniprot_2022.parquet").set_index("gene")
    nodes = world["nodes"]
    has_old = np.array([isinstance(old.sequence_2022.get(g), str) and bool(old.sequence_2022.get(g))
                        for g in nodes.gene[:N_LIB]])
    assert np.array_equal(z["H_esm_time"][:N_LIB], has_old)
    assert not np.array_equal(z["X_esm_time"][:N_LIB][has_old], z["X_esm"][:N_LIB][has_old])
    summ = json.loads((out / "blocks_summary.json").read_text())
    assert summ["blocks"]["dom_time"]["vocab"] == 4
    S = pd.read_parquet(out / "text" / "sentences.parquet")
    assert S[S.version == "2022"].sentence.str.contains("ARCHIVE").all()
    assert not S[S.version == "current"].sentence.str.contains("ARCHIVE").any()
    assert z["H_txt_time"][:25].all() and not z["H_txt_time"][25:N_LIB].any()


def test_decoys_are_out_of_library_and_carry_knowledge(world):
    out = world["out"]
    summ = json.loads((out / "blocks_summary.json").read_text())
    assert summ["decoy_proteins"] == ["DEC0", "DEC2", "DEC3"]
    assert not set(summ["decoy_proteins"]) & set(world["nodes"].gene)
    z = np.load(out / "blocks.npz")
    assert abs(z["H_esm"][N_LIB:].mean() - z["H_esm"][:N_LIB].mean()) <= 0.02
    assert abs(z["H_dom"][N_LIB:].mean() - z["H_dom"][:N_LIB].mean()) <= 0.02
    assert z["H_img_dino"][N_LIB:].tolist() == [True, True, False]
    with pytest.raises(ValueError):
        gk.node_proteins(world["nodes"], world["cur"][world["cur"].source == "library"])


def test_foldseek_edges_and_paralogs(world):
    out = world["out"]
    st = np.load(out / "structure.npz")
    E, EP = st["E_struct"], st["E_paralog_fident30"]
    assert (E[:, 0] < E[:, 1]).all() and len({tuple(e) for e in E}) == len(E)
    assert {tuple(e) for e in EP} <= {tuple(e) for e in E}
    lib = ~world["nodes"].is_nt.to_numpy()
    assert lib[EP].all()
    assert (6, 7) in {tuple(e) for e in EP}
    summ = json.loads((out / "structure_summary.json").read_text())
    assert summ["foldseek_hits_E1e-3"] == 4 and summ["paralog_pairs_fident30"] == len(EP)
    m8 = pd.DataFrame({"q": ["a", "b", "a"], "t": ["b", "a", "c"], "fident": [0.5, 0.5, 0.1]})
    E2, P2 = gk.structure_pairs(m8, ["a", "b", "c", "a"], np.array([1, 1, 1, 0], bool))
    assert {tuple(e) for e in E2} == {(0, 1), (0, 2), (1, 3), (2, 3)} and {tuple(e) for e in P2} == {(0, 1)}
    assert st["X_s3di"].shape[1] == 3 and st["X_s3di_kmer"].shape[1] == 3 and st["X_struct_spec"].shape[1] == 3


def test_paralog_mask_is_exactly_identity_union_shared_pfam():
    import scipy.sparse as sp

    B = sp.csr_matrix(np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float))
    vocab = ["PF1", "PF2", "IPR3"]
    M = gk.paralog_mask(4, np.array([[2, 3]]), B, vocab)
    assert {tuple(x) for x in np.argwhere(M)} == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert not gk.paralog_mask(4, np.array([[2, 3]]), B, vocab, shared_pfam=False)[0, 1]
    assert not gk.paralog_mask(4).any()


def test_profile_metrics_select_report_and_paralog_excluded(world):
    T = gk.profile_metrics(world["s"])
    assert {"profile", "blocks", "dims", "mAP_CORUM@SELECT", "mAP_BioGRID@REPORT", "SELECT_score", "REPORT_score",
            "mAP_CORUM@SELECT_noparalog", "SELECT_score_noparalog"} <= set(T.columns)
    assert "esm" in set(T.profile) and T.profile.iloc[-1].startswith("control")
    assert ((T.SELECT_score >= 0) & (T.SELECT_score <= 1)).all()
    assert (world["out"] / "profile_metrics.csv").exists()


def test_three_di_windows_take_the_first_window_and_embed(world):
    tdi, E = gk.three_di(["ACDEFGHIKL" * 7], encode_fake, head_fake, 50, 30, 60)
    assert len(tdi[0]) == 70 and set(tdi[0]) <= set(gk.SS) and E.shape == (1, 1024)
    stored = gk.read_fasta(world["out"] / "prostt5" / "3di.fasta")
    assert len(stored) == len({s for s in world["cur"].sequence.dropna()})
    cnn = gk.CNN()
    assert cnn(torch.zeros(1, 9, 1024)).shape == (1, 20, 9)


def test_rerun_from_saved_settings_is_byte_identical(world, tmp_path):
    from spacr.cli import load_settings_file

    out = world["out"]
    s = dict(world["s"])
    s.update({"gene_knowledge_esm_embeddings": str(out / "esm" / "esm_by_md5.npz"),
              "gene_knowledge_text_embeddings": str(out / "text"),
              "gene_knowledge_prostt5_outputs": str(out / "prostt5"),
              "gene_knowledge_uniprot_archive_table": str(out / "data" / "uniprot_2022.parquet")})
    first = dict(s, gene_knowledge_src=str(tmp_path / "a"))
    gk.run_gene_knowledge(first)
    saved = load_settings_file(str(tmp_path / "a" / "gene_knowledge" / "settings" / "gene_knowledge_settings.csv"))
    saved["gene_knowledge_src"] = str(tmp_path / "b")
    assert gk.main(["--settings", str(_write_settings(saved, tmp_path / "b.csv"))]) == 0
    a, b = tmp_path / "a" / "gene_knowledge", tmp_path / "b" / "gene_knowledge"
    for name in ("blocks_summary.json", "structure_summary.json", "profile_metrics.csv", "data/dom_vocab_current.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    for name in ("blocks.npz", "structure.npz"):
        za, zb = np.load(a / name), np.load(b / name)
        assert sorted(za.files) == sorted(zb.files), (name, sorted(set(za.files) ^ set(zb.files)))
        assert [k for k in za.files if k != "X_struct_spec" and not np.array_equal(za[k], zb[k])] == [], name
    ref = np.load(out / "blocks.npz")
    mine = np.load(a / "blocks.npz")
    assert [k for k in ref.files if not np.array_equal(ref[k], mine[k])] == []


def _write_settings(s, path):
    from spacr.cli import write_settings_file

    return write_settings_file(s, path)[0]


class _Enc(dict):
    """A tokenizer output that moves to a device and exposes attributes."""

    def to(self, dev):
        return self

    def __getattr__(self, k):
        return self[k]


class _Tok:
    """A fake tokenizer: one token per character plus a start token."""

    @classmethod
    def from_pretrained(cls, *a, **k):
        assert k.get("local_files_only") is True
        return cls()

    def __call__(self, batch, **k):
        L = max(len(s) for s in batch) + 2
        ids = torch.zeros(len(batch), L, dtype=torch.long)
        for i, s in enumerate(batch):
            ids[i, 1:1 + len(s)] = torch.tensor([ord(c) % 50 for c in s])
        return _Enc(input_ids=ids, attention_mask=(ids > 0).long() | (torch.arange(L) == 0).long())

    batch_encode_plus = __call__


class _Model(torch.nn.Module):
    """A fake transformer: embeds token ids."""

    def __init__(self, dim=6):
        super().__init__()
        self.emb = torch.nn.Embedding(64, dim)
        self.config = type("C", (), {"hidden_size": dim})

    @classmethod
    def from_pretrained(cls, *a, **k):
        torch.manual_seed(0)
        return cls(1024 if "T5" in cls.__name__ else 6)

    def forward(self, input_ids=None, attention_mask=None, **k):
        return type("O", (), {"last_hidden_state": self.emb(input_ids)})


class _T5(_Model):
    """Fake ProstT5 encoder."""


def test_local_model_loaders_with_fake_transformers(world, monkeypatch, tmp_path):
    import sys
    import types

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = fake.T5Tokenizer = _Tok
    fake.EsmModel = fake.AutoModel = _Model
    fake.T5EncoderModel = _T5
    monkeypatch.setitem(sys.modules, "transformers", fake)
    o = gk.options({"gene_knowledge_device": "cpu", "gene_knowledge_prostt5_cnn": str(tmp_path / "cnn.pt")})
    dev = torch.device("cpu")
    out = gk.esm_embedder(o, dev)(["ACD", "WY"])
    assert out[0].shape == (3, 6) and out[1].shape == (2, 6)
    v = gk.text_embedder(o, dev)(["a b c", "dd"])
    assert v.shape == (2, 6) and np.allclose(np.linalg.norm(v, axis=1), 1)
    torch.save({"state_dict": gk.CNN().state_dict()}, tmp_path / "cnn.pt")
    enc, head = gk._prostt5(o, dev)
    tdi, E = gk.three_di(["ACDEFG"], enc, head, 50, 30, 100)
    assert len(tdi[0]) == 6 and E.shape == (1, 1024)
    s = dict(world["s"], gene_knowledge_src=str(tmp_path / "m"), gene_knowledge_prostt5_cnn=str(tmp_path / "cnn.pt"))
    gk.fetch_protein_annotations(s)
    assert gk.embed_sequences(s).exists()
    assert gk.predict_3di(s)["emb3di_by_md5.npz"].exists()
    assert gk.embed_text(s, batch_size=50)["embeddings"].exists()
    monkeypatch.setitem(sys.modules, "transformers", None)
    for fn in (gk.esm_embedder, gk.text_embedder, gk._prostt5):
        with pytest.raises(ImportError):
            fn(o, dev)


def test_foldseek_commands_http_and_tar_archive(world, monkeypatch, tmp_path):
    import io
    import subprocess
    import tarfile

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))
    o = gk.options({"gene_knowledge_foldseek_bin": "fs"})
    assert gk.run_foldseek(o, tmp_path, tmp_path / "p").name == "aln.m8"
    assert [c[1] for c in calls] == ["base:createdb", "base:createdb", "search", "convertalis"]
    calls.clear()
    gk.run_foldseek({**o, "structure_source": "afdb", "afdb_dir": "/x"}, tmp_path, tmp_path)
    assert calls[0][1] == "createdb" and "--mask-bfactor-threshold" in calls[0]

    class R(io.BytesIO):
        """A urlopen response."""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seq = iter([OSError("down"), R(b"ok")])

    def urlopen(url, timeout):
        """Fail once, then answer."""
        x = next(seq)
        if isinstance(x, Exception):
            raise x
        return x

    monkeypatch.setattr(gk.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gk.time, "sleep", lambda s: None)
    assert gk.http_get("http://x") == "ok"
    monkeypatch.setattr(gk.urllib.request, "urlopen", lambda u, timeout: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(RuntimeError):
        gk.http_get("http://x", tries=2)
    tar = tmp_path / "rel.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(world["root"] / "sprot.dat.gz", arcname="uniprot_sprot.dat.gz")
    cur = pd.read_parquet(world["out"] / "data" / "uniprot_current.parquet")
    assert gk.parse_archive(tar, cur).match2022.value_counts()["accession"] >= 25
    with pytest.raises(RuntimeError):
        gk.fetch_protein_annotations({"gene_knowledge_src": str(tmp_path / "n"),
                                      "gene_knowledge_nodes": world["s"]["gene_knowledge_nodes"]})
