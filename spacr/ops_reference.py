"""Gene-gene reference sets for scoring optical pooled screen gene profiles.

A gene profile is judged by how well it finds the genes a curated resource
links to it. This module reads those resources by name and version, turns
each into a set of unordered gene pairs ``(a, b)`` with ``a < b``, and splits
them into the halves of a selection protocol, so that a choice made on one
half is reported on the other.

The names, and the file each one is read from inside ``reference_dir``:

* ``CORUM`` and ``KEGG``: Brieflow's group benchmarks
  (``corum_group_benchmark.tsv``, ``kegg_group_benchmark.tsv``; columns
  ``gene_name`` and ``group``); every two members of a group are a pair.
* ``STRING_brieflow``: Brieflow's STRING pair benchmark
  (``string_pair_benchmark.tsv``; ``gene_name`` and ``pair``).
* ``STRING>=0.7`` and ``STRING>=0.4``: STRING v12 links with a combined
  score of at least 700 or 400 (``string_v12_ge400.parquet``, made by
  :func:`prepare_current_references`).
* ``BioGRID``: BioGRID physical human pairs, latest release
  (``biogrid_physical.parquet``).
* the references known when Funk et al. 2022 was published and the evidence
  added later, made by :func:`prepare_time_split_references`:
  ``STRING11.5>=0.7``, ``STRING11.5>=0.4``, ``BioGRID2022`` (4.4.214),
  ``CORUM3`` (CORUM 3.0), ``STRINGnew>=0.7``, ``BioGRIDnew`` and
  ``CORUMnew``.

``NAME@SELECT`` and ``NAME@REPORT`` are the two halves. CORUM is split by
complex: the complex names, sorted, are permuted with seed
:data:`SPLIT_SEED` and alternate between the halves, and a REPORT pair that
also lies in a SELECT complex is dropped. Every other reference is split by
pair, on the parity of the md5 of ``"a|b"``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The seed of the CORUM complex split, fixed before any result was seen.
SPLIT_SEED = 20261107

#: The five references every evaluation scores.
MAIN_REFERENCES: Tuple[str, ...] = ("CORUM", "STRING>=0.7", "STRING>=0.4", "KEGG", "BioGRID")

#: The halves of the selection protocol that an evaluation scores.
HALF_REFERENCES: Tuple[str, ...] = tuple(
    f"{name}@{half}" for name in ("CORUM", "STRING>=0.7", "BioGRID") for half in ("SELECT", "REPORT"))

#: References as they stood in 2022, then the evidence added since.
TIME_SPLIT_OLD: Tuple[str, ...] = ("STRING11.5>=0.4", "BioGRID2022", "CORUM3")
TIME_SPLIT_NEW: Tuple[str, ...] = ("STRINGnew>=0.7", "BioGRIDnew", "CORUMnew")
TIME_REFERENCES: Tuple[str, ...] = ("STRING11.5>=0.7", "BioGRID2022", "CORUM3",
                                    "STRINGnew>=0.7", "BioGRIDnew", "CORUMnew")

#: The file each reference is read from.
REFERENCE_FILES: Dict[str, str] = {
    "CORUM": "corum_group_benchmark.tsv",
    "KEGG": "kegg_group_benchmark.tsv",
    "STRING_brieflow": "string_pair_benchmark.tsv",
    "STRING>=0.7": "string_v12_ge400.parquet",
    "STRING>=0.4": "string_v12_ge400.parquet",
    "BioGRID": "biogrid_physical.parquet",
    "STRING11.5>=0.7": "string_v11.5_ge700.parquet",
    "STRING11.5>=0.4": "string_v11.5_ge400.parquet",
    "BioGRID2022": "biogrid_4.4.214_physical.parquet",
    "STRINGnew>=0.7": "string_new_v12_ge700.parquet",
    "BioGRIDnew": "biogrid_new_after_4.4.214.parquet",
    "CORUM3": "corum3_pairs.parquet",
    "CORUMnew": "corum_new_after_3.0.parquet",
}

#: The release each reference stands for.
REFERENCE_VERSIONS: Dict[str, str] = {
    "CORUM": "Brieflow corum_group_benchmark.tsv (aconcagua-analysis)",
    "KEGG": "Brieflow kegg_group_benchmark.tsv (aconcagua-analysis)",
    "STRING_brieflow": "Brieflow string_pair_benchmark.tsv (aconcagua-analysis)",
    "STRING>=0.7": "STRING v12.0 full links, combined score >= 700",
    "STRING>=0.4": "STRING v12.0 full links, combined score >= 400",
    "BioGRID": "BioGRID organism release, latest at download, physical, human",
    "STRING11.5>=0.7": "STRING v11.5, combined score >= 700",
    "STRING11.5>=0.4": "STRING v11.5, combined score >= 400",
    "BioGRID2022": "BioGRID 4.4.214 (2022-09-25), physical, human",
    "STRINGnew>=0.7": "STRING v12.0 >= 700 pairs below 400 or absent in v11.5",
    "BioGRIDnew": "BioGRID latest physical pairs absent from 4.4.214",
    "CORUM3": "CORUM 3.0 allComplexes.txt (2018-09-03), human",
    "CORUMnew": "Brieflow CORUM pairs absent from CORUM 3.0",
}

#: Prefixes of control rows: non-targeting guides and the pseudo-genes built
#: from them. They are never queries or partners.
CONTROL_PREFIXES: Tuple[str, ...] = ("nontargeting", "ntpseudo")

_GROUP_REFERENCES = ("CORUM", "KEGG")


def gene_symbol(name: Any) -> str:
    """The gene symbol of a profile row; ``"A&B"`` (a shared guide) gives ``"A"``.

    :param name: a row name.
    :returns: the symbol.
    """
    return str(name).split("&")[0]


def is_control(names: Iterable[Any], prefixes: Sequence[str] = CONTROL_PREFIXES) -> np.ndarray:
    """True for every name that starts with a control prefix, in any case.

    :param names: row names.
    :param prefixes: the control prefixes.
    :returns: a boolean array.
    """
    return pd.Series(list(names), dtype=object).astype(str).str.lower().str.startswith(
        tuple(p.lower() for p in prefixes)).to_numpy(dtype=bool)


def reference_dirs(reference_dir: Any) -> Tuple[str, ...]:
    """The folders a reference is looked for in, in order.

    :param reference_dir: one folder, or a list of folders.
    :returns: the folders as a tuple of strings.
    :raises ValueError: when no folder is given.
    """
    if reference_dir is None or (isinstance(reference_dir, str) and not reference_dir.strip()):
        raise ValueError("ops_reference_dir is empty: give the folder (or folders) holding the reference files")
    folders = [reference_dir] if isinstance(reference_dir, (str, os.PathLike)) else list(reference_dir)
    return tuple(os.path.expanduser(os.fspath(f)) for f in folders)


def reference_file(name: str, reference_dir: Any) -> str:
    """The path of a reference's file, from the first folder that has it.

    :param name: a reference name, with or without its ``@`` half.
    :param reference_dir: see :func:`reference_dirs`.
    :returns: the path.
    :raises KeyError: for an unknown name.
    :raises FileNotFoundError: when no folder holds the file.
    """
    base = name.split("@")[0]
    if base not in REFERENCE_FILES:
        raise KeyError(f"unknown reference {name!r}; choose from {sorted(REFERENCE_FILES)}")
    folders = reference_dirs(reference_dir)
    for folder in folders:
        path = os.path.join(folder, REFERENCE_FILES[base])
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"reference {name!r} needs {REFERENCE_FILES[base]} in one of {list(folders)}")


def _read(path: str, **kwargs) -> pd.DataFrame:
    """Read a reference table exactly as written."""
    from .tabular import read_table

    return read_table(path, canonicalise=False, report=None, **kwargs)


def _group_pairs(groups: Mapping[str, Sequence[str]]) -> set:
    """Every unordered pair of members of every group."""
    pairs = set()
    for members in groups.values():
        u = sorted(set(members))
        pairs |= {(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))}
    return pairs


@lru_cache(maxsize=None)
def _groups(name: str, folders: Tuple[str, ...]) -> Dict[str, List[str]]:
    """Cached :func:`reference_groups`."""
    table = _read(reference_file(name, folders), sep="\t").dropna(subset=["gene_name", "group"])
    return {g: sorted(set(d.gene_name)) for g, d in table.groupby("group")}


def reference_groups(name: str, reference_dir: Any) -> Dict[str, List[str]]:
    """The groups of a group reference (``CORUM`` or ``KEGG``).

    :param name: ``CORUM`` or ``KEGG``.
    :param reference_dir: see :func:`reference_dirs`.
    :returns: ``{group: sorted member symbols}``, groups in sorted order.
    :raises KeyError: for a name that is not a group reference.
    """
    if name not in _GROUP_REFERENCES:
        raise KeyError(f"{name!r} is not a group reference; choose from {list(_GROUP_REFERENCES)}")
    return dict(_groups(name, reference_dirs(reference_dir)))


def pair_half(a: str, b: str) -> int:
    """Which half a pair falls in: 0 for SELECT, 1 for REPORT.

    :param a: the first symbol (the smaller).
    :param b: the second symbol.
    :returns: the parity of the md5 of ``"a|b"``.
    """
    return int(hashlib.md5(f"{a}|{b}".encode()).hexdigest(), 16) % 2


def group_halves(groups: Iterable[str], seed: int = SPLIT_SEED) -> Dict[str, int]:
    """Split groups between the halves: 0 for SELECT, 1 for REPORT.

    :param groups: the group names.
    :param seed: the permutation seed.
    :returns: ``{group: half}``; the sorted names are permuted and alternate.
    """
    names = sorted(groups)
    perm = np.random.default_rng(seed).permutation(len(names))
    return {names[i]: int(k % 2) for k, i in enumerate(perm)}


@lru_cache(maxsize=None)
def _pairs(name: str, folders: Tuple[str, ...]) -> FrozenSet[Tuple[str, str]]:
    """Cached :func:`reference_pairs`."""
    if "@" in name:
        base, half = name.split("@")
        if half not in ("SELECT", "REPORT"):
            raise KeyError(f"{name!r}: the half is SELECT or REPORT")
        h = {"SELECT": 0, "REPORT": 1}[half]
        if base == "CORUM":
            groups = _groups("CORUM", folders)
            halves = group_halves(groups)
            select = _group_pairs({g: m for g, m in groups.items() if halves[g] == 0})
            report = _group_pairs({g: m for g, m in groups.items() if halves[g] == 1})
            return frozenset(select if h == 0 else report - select)
        return frozenset(p for p in _pairs(base, folders) if pair_half(*p) == h)
    path = reference_file(name, folders)
    if name in _GROUP_REFERENCES:
        return frozenset(_group_pairs(_groups(name, folders)))
    if name == "STRING_brieflow":
        table = _read(path, sep="\t").dropna(subset=["gene_name"])
        return frozenset(_group_pairs({k: d.gene_name.unique() for k, d in table.groupby("pair")}))
    if name in ("STRING>=0.7", "STRING>=0.4"):
        table = _read(path, columns=["a", "b", "combined_score"])
        table = table[table.combined_score >= (700 if name == "STRING>=0.7" else 400)]
        return frozenset(zip(table.a, table.b))
    table = _read(path, columns=["a", "b"])
    return frozenset(zip(table.a, table.b))


def reference_pairs(name: str, reference_dir: Any) -> FrozenSet[Tuple[str, str]]:
    """The unordered gene pairs ``(a, b)``, ``a < b``, of a reference.

    :param name: a name of :data:`REFERENCE_FILES`, optionally with
        ``@SELECT`` or ``@REPORT``.
    :param reference_dir: see :func:`reference_dirs`.
    :returns: the pairs.
    :raises KeyError: for an unknown name or half.
    :raises FileNotFoundError: when the reference's file is missing.
    """
    return _pairs(name, reference_dirs(reference_dir))


def split_references(names: Sequence[str], split: str) -> List[str]:
    """The reference names an evaluation split scores.

    :param names: base reference names.
    :param split: ``'select'``, ``'report'`` or ``'all'``.
    :returns: the names with the half appended, or unchanged for ``'all'``.
    :raises ValueError: for an unknown split.
    """
    split = str(split).lower()
    if split == "all":
        return list(names)
    if split not in ("select", "report"):
        raise ValueError(f"ops_eval_split is 'select', 'report' or 'all'; got {split!r}")
    return [f"{n.split('@')[0]}@{split.upper()}" for n in names]


def positive_matrix(genes: Sequence[Any], pairs: Iterable[Tuple[str, str]]) -> np.ndarray:
    """Which rows are reference partners of which.

    :param genes: row names; ``"A&B"`` is read as ``A``. Two rows with one
        symbol share its partners.
    :param pairs: the reference pairs.
    :returns: a symmetric boolean ``n x n`` matrix with a false diagonal.
    """
    symbols = [gene_symbol(g) for g in genes]
    index: Dict[str, List[int]] = {}
    for i, g in enumerate(symbols):
        index.setdefault(g, []).append(i)
    out = np.zeros((len(symbols), len(symbols)), bool)
    for a, b in pairs:
        ia, ib = index.get(a), index.get(b)
        if ia and ib:
            for i in ia:
                for j in ib:
                    out[i, j] = out[j, i] = True
    np.fill_diagonal(out, False)
    return out


def file_md5(path: str) -> str:
    """The md5 of a file, read in blocks.

    :param path: the file.
    :returns: the hex digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_record(names: Sequence[str], reference_dir: Any) -> Dict[str, Dict[str, Any]]:
    """What each reference of a run was: its file, md5, release and pair count.

    :param names: reference names, halves allowed.
    :param reference_dir: see :func:`reference_dirs`.
    :returns: ``{name: {"file", "md5", "version", "pairs"}}``.
    """
    out = {}
    for name in names:
        path = reference_file(name, reference_dir)
        out[name] = {"file": path, "md5": file_md5(path),
                     "version": REFERENCE_VERSIONS[name.split("@")[0]],
                     "pairs": len(reference_pairs(name, reference_dir))}
    return out


def clear_cache() -> None:
    """Forget every reference read so far, so changed files are read again."""
    _pairs.cache_clear()
    _groups.cache_clear()


def _ordered(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """``frame`` with columns ``a`` and ``b`` holding each pair smaller first."""
    x, y = frame[a], frame[b]
    return frame.assign(a=x.where(x < y, y), b=y.where(x < y, x))


def _zip_member(archive: str, contains: str, folder: str) -> Tuple[str, str]:
    """Extract the member of ``archive`` whose name contains ``contains``.

    :returns: ``(extracted path, member name)``.
    """
    with zipfile.ZipFile(archive) as handle:
        name = [n for n in handle.namelist() if contains in n][0]
        return handle.extract(name, folder), name


def _biogrid_physical(archive: str, folder: str, columns: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    """The human-human physical interactions of a BioGRID organism archive."""
    path, name = _zip_member(archive, "Homo_sapiens", folder)
    table = _read(path, sep="\t", low_memory=False, usecols=list(columns) + [
        "Experimental System Type", "Organism ID Interactor A", "Organism ID Interactor B"])
    os.remove(path)
    keep = ((table["Experimental System Type"] == "physical")
            & (table["Organism ID Interactor A"] == 9606) & (table["Organism ID Interactor B"] == 9606))
    return table[keep], name


def prepare_current_references(raw_dir: str, out_dir: str) -> Dict[str, Any]:
    """Write the current STRING and BioGRID pair tables.

    Reads, inside ``raw_dir``, ``string_v12/9606.protein.info.v12.0.txt.gz``
    and ``string_v12/9606.protein.links.full.v12.0.txt.gz`` (links with a
    combined score of at least 400, named by STRING's preferred names) and
    ``biogrid/BIOGRID-ORGANISM-LATEST.tab3.zip`` (human-human physical
    pairs by official symbol, with the number of records and the
    experimental systems behind each).

    :param raw_dir: the folder of downloads.
    :param out_dir: where ``string_v12_ge400.parquet`` and
        ``biogrid_physical.parquet`` are written.
    :returns: the pair counts and the BioGRID release file.
    """
    from .tabular import write_table

    os.makedirs(out_dir, exist_ok=True)
    info = _read(os.path.join(raw_dir, "string_v12", "9606.protein.info.v12.0.txt.gz"),
                 sep="\t", usecols=[0, 1])
    names = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
    keep = []
    for chunk in _read(os.path.join(raw_dir, "string_v12", "9606.protein.links.full.v12.0.txt.gz"),
                       sep=" ", chunksize=2_000_000):
        chunk = chunk[chunk.combined_score >= 400]
        chunk = chunk.assign(a=chunk.protein1.map(names), b=chunk.protein2.map(names)).dropna(subset=["a", "b"])
        keep.append(chunk[chunk.a < chunk.b].drop(columns=["protein1", "protein2"]))
    string = pd.concat(keep, ignore_index=True)
    write_table(string, os.path.join(out_dir, "string_v12_ge400.parquet"), canonicalise=False)
    with tempfile.TemporaryDirectory(dir=out_dir) as folder:
        table, release = _biogrid_physical(
            os.path.join(raw_dir, "biogrid", "BIOGRID-ORGANISM-LATEST.tab3.zip"), folder,
            ["Official Symbol Interactor A", "Official Symbol Interactor B", "Experimental System"])
    a1, b1 = table["Official Symbol Interactor A"], table["Official Symbol Interactor B"]
    table = table.assign(a=a1.where(a1 < b1, b1), b=b1.where(a1 < b1, a1))
    table = table[table.a != table.b]
    grouped = table.groupby(["a", "b"]).agg(
        n_evidence=("Experimental System", "size"),
        systems=("Experimental System", lambda x: ";".join(sorted(set(x)))))
    write_table(grouped.reset_index(), os.path.join(out_dir, "biogrid_physical.parquet"), canonicalise=False)
    return {"string_v12_ge400_pairs": int(len(string)),
            "string_v12_ge700_pairs": int((string.combined_score >= 700).sum()),
            "biogrid_physical_pairs": int(len(grouped)), "biogrid_file": release}


def _time_split_string(archive_dir: str, raw_dir: str, out_dir: str) -> Dict[str, int]:
    """STRING v11.5 pairs and the v12 pairs of at least 700 added since."""
    from .tabular import write_table

    i12 = _read(os.path.join(raw_dir, "string_v12", "9606.protein.info.v12.0.txt.gz"), sep="\t", usecols=[0, 1])
    i11 = _read(os.path.join(archive_dir, "string_v11.5", "9606.protein.info.v11.5.txt.gz"),
                sep="\t", usecols=[0, 1])
    i12.columns = i11.columns = ["id", "sym"]
    sym = dict(zip(i11.id, i11.sym))
    sym.update(dict(zip(i12.id, i12.sym)))
    old = {}
    for chunk in _read(os.path.join(archive_dir, "string_v11.5", "9606.protein.links.v11.5.txt.gz"),
                       sep=" ", chunksize=2_000_000):
        chunk = chunk[(chunk.combined_score >= 400) & (chunk.protein1 < chunk.protein2)]
        old.update(dict(zip(zip(chunk.protein1, chunk.protein2), chunk.combined_score)))
    o = pd.DataFrame([(p, q, s) for (p, q), s in old.items()], columns=["p", "q", "score"])
    o["sa"], o["sb"] = o.p.map(sym), o.q.map(sym)
    o = _ordered(o.dropna(subset=["sa", "sb"]), "sa", "sb")
    o = o[o.a != o.b]
    write_table(o[o.score >= 700][["a", "b", "score"]].drop_duplicates(["a", "b"]),
                os.path.join(out_dir, "string_v11.5_ge700.parquet"), canonicalise=False)
    write_table(o[["a", "b", "score"]].drop_duplicates(["a", "b"]),
                os.path.join(out_dir, "string_v11.5_ge400.parquet"), canonicalise=False)
    present = set(i11.id)
    new = []
    for chunk in _read(os.path.join(raw_dir, "string_v12", "9606.protein.links.full.v12.0.txt.gz"), sep=" ",
                       chunksize=2_000_000, usecols=["protein1", "protein2", "combined_score"]):
        chunk = chunk[(chunk.combined_score >= 700) & (chunk.protein1 < chunk.protein2)]
        keep = [(p, q) not in old and p in present and q in present
                for p, q in zip(chunk.protein1, chunk.protein2)]
        new.append(chunk[keep])
    n = pd.concat(new)
    n["sa"], n["sb"] = n.protein1.map(sym), n.protein2.map(sym)
    n = _ordered(n.dropna(subset=["sa", "sb"]), "sa", "sb")
    n = n[n.a != n.b].drop_duplicates(["a", "b"])
    write_table(n[["a", "b", "combined_score"]], os.path.join(out_dir, "string_new_v12_ge700.parquet"),
                canonicalise=False)
    return {"v11.5_ge400_pairs": int(len(o)), "v11.5_ge700_pairs": int((o.score >= 700).sum()),
            "new_v12_ge700_pairs": int(len(n))}


def _time_split_biogrid(archive_dir: str, raw_dir: str, out_dir: str) -> Dict[str, Any]:
    """BioGRID 4.4.214 physical pairs and the pairs added since, diffed on Entrez ids."""
    from .tabular import write_table

    columns = ["Entrez Gene Interactor A", "Entrez Gene Interactor B",
               "Official Symbol Interactor A", "Official Symbol Interactor B"]
    with tempfile.TemporaryDirectory(dir=out_dir) as folder:
        old, n_old = _biogrid_physical(
            os.path.join(archive_dir, "biogrid_2022", "BIOGRID-ORGANISM-4.4.214.tab3.zip"), folder, columns)
        new, n_new = _biogrid_physical(
            os.path.join(raw_dir, "biogrid", "BIOGRID-ORGANISM-LATEST.tab3.zip"), folder, columns)
    sym = dict(zip(old["Entrez Gene Interactor A"].astype(str), old["Official Symbol Interactor A"]))
    sym.update(dict(zip(old["Entrez Gene Interactor B"].astype(str), old["Official Symbol Interactor B"])))
    for c in "AB":
        sym.update(dict(zip(new[f"Entrez Gene Interactor {c}"].astype(str), new[f"Official Symbol Interactor {c}"])))

    def entrez_pairs(table):
        """Unordered Entrez pairs, self pairs dropped."""
        e = _ordered(table.assign(x=table["Entrez Gene Interactor A"].astype(str),
                                  y=table["Entrez Gene Interactor B"].astype(str)), "x", "y")
        return set(zip(e.a, e.b)) - {(u, u) for u in e.a}

    def by_symbol(pairs):
        """Entrez pairs named by symbol, ordered, self pairs dropped."""
        d = pd.DataFrame([(sym.get(u), sym.get(v)) for u, v in pairs], columns=["sa", "sb"]).dropna()
        d = _ordered(d, "sa", "sb")
        return d[d.a != d.b][["a", "b"]].drop_duplicates()

    po, pn = entrez_pairs(old), entrez_pairs(new)
    write_table(by_symbol(po), os.path.join(out_dir, "biogrid_4.4.214_physical.parquet"), canonicalise=False)
    genes_old = {u for p in po for u in p}
    added = {p for p in pn - po if p[0] in genes_old and p[1] in genes_old}
    write_table(by_symbol(added), os.path.join(out_dir, "biogrid_new_after_4.4.214.parquet"), canonicalise=False)
    return {"old_file": n_old, "new_file": n_new, "old_pairs": len(po), "new_pairs": len(pn),
            "added_pairs": len(added)}


def _time_split_corum(archive_dir: str, brieflow_dir: str, out_dir: str) -> Dict[str, int]:
    """CORUM 3.0 pairs and the current Brieflow CORUM pairs absent from it."""
    from .tabular import write_table

    with tempfile.TemporaryDirectory(dir=out_dir) as folder:
        path, _ = _zip_member(os.path.join(archive_dir, "corum", "corum3_allComplexes_wayback20221016.txt.zip"),
                              "allComplexes.txt", folder)
        c = _read(path, sep="\t", low_memory=False)
    c = c[c.Organism == "Human"]
    old, genes = set(), set()
    for subunits in c["subunits(Gene name)"].dropna():
        u = sorted({x.strip() for x in subunits.split(";") if x.strip() and x.strip() != "None"})
        genes |= set(u)
        old |= {(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))}
    write_table(pd.DataFrame(sorted(old), columns=["a", "b"]), os.path.join(out_dir, "corum3_pairs.parquet"),
                canonicalise=False)
    now = set(_group_pairs(_groups("CORUM", (brieflow_dir,))))
    new = sorted(p for p in now - old if p[0] in genes and p[1] in genes)
    write_table(pd.DataFrame(new, columns=["a", "b"]), os.path.join(out_dir, "corum_new_after_3.0.parquet"),
                canonicalise=False)
    return {"corum3_human_complexes": int(len(c)), "corum3_pairs": len(old), "current_brieflow_pairs": len(now),
            "new_pairs_both_genes_in_corum3": len(new), "current_pairs_in_corum3": len(now & old)}


def prepare_time_split_references(archive_dir: str, raw_dir: str, brieflow_dir: str, out_dir: str,
                                  *, parts: Sequence[str] = ("string", "biogrid", "corum")) -> Dict[str, Any]:
    """Write the references known in 2022 and the evidence added since.

    STRING v11.5 is diffed against v12 on protein ids and BioGRID 4.4.214
    against the latest release on Entrez ids, so a renamed symbol never makes
    a new pair; a new pair needs both genes present in the old release. CORUM
    3.0 (the Wayback snapshot of ``allComplexes.txt``) is diffed against
    Brieflow's current CORUM table on symbols named in CORUM 3.0.

    :param archive_dir: holds ``string_v11.5/``, ``biogrid_2022/`` and
        ``corum/`` with the archived releases.
    :param raw_dir: holds the current ``string_v12/`` and ``biogrid/``
        downloads.
    :param brieflow_dir: holds ``corum_group_benchmark.tsv``.
    :param out_dir: where the seven parquet tables and ``versions.json``
        are written.
    :param parts: any of ``"string"``, ``"biogrid"`` and ``"corum"``; a
        part left out keeps what ``versions.json`` already says of it.
    :returns: the contents of ``versions.json``.
    """
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "versions.json")
    versions: Dict[str, Any] = {}
    if os.path.isfile(target):
        with open(target, encoding="utf-8") as handle:
            versions = json.load(handle)
    versions.update({
        "STRING_2022": "v11.5 (stringdb-downloads.org, 9606.protein.links.v11.5 + info)",
        "STRING_now": "v12.0 full links", "BioGRID_2022": "4.4.214 (compiled 2022-09-25)",
        "CORUM_now": "Brieflow corum_group_benchmark.tsv (aconcagua-analysis)"})
    if "string" in parts:
        versions["string"] = _time_split_string(archive_dir, raw_dir, out_dir)
    if "biogrid" in parts:
        versions["biogrid"] = _time_split_biogrid(archive_dir, raw_dir, out_dir)
        versions["BioGRID_now"] = f"LATEST at preparation ({versions['biogrid']['new_file']})"
    if "corum" in parts:
        archive = os.path.join(archive_dir, "corum", "corum3_allComplexes_wayback20221016.txt.zip")
        versions["CORUM_2022"] = ("CORUM 3.0 allComplexes.txt (2018-09-03), Wayback snapshot 2022-10-16, md5 "
                                  + file_md5(archive))
        versions["corum"] = _time_split_corum(archive_dir, brieflow_dir, out_dir)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(versions, handle, indent=1)
    clear_cache()
    return versions


def reference_versions(out_dir: str) -> Optional[Dict[str, Any]]:
    """The ``versions.json`` written by :func:`prepare_time_split_references`.

    :param out_dir: the folder it was written to.
    :returns: its contents, or None when there is none.
    """
    target = os.path.join(out_dir, "versions.json")
    if not os.path.isfile(target):
        return None
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


#: References whose shuffled-label and fitness-matched controls are scored.
CONTROL_REFERENCES: Tuple[str, ...] = ("CORUM", "STRING>=0.7", "BioGRID")

#: The evaluation settings and their defaults.
EVALUATION_DEFAULTS: Dict[str, Any] = {
    "ops_eval_references": list(MAIN_REFERENCES),
    "ops_eval_split": "all",
    "ops_eval_null_size": 5000,
    "ops_eval_heldout_folds": 5,
    "ops_eval_decoys": 10,
    "ops_eval_time_split": False,
}


def embed_gene_profiles(values, variance: float = 0.99, seed: int = 0) -> np.ndarray:
    """The components gene profiles are compared in, as Brieflow embeds them.

    Columns with 5 % or more missing values are dropped, other gaps take the
    column median, constant columns are dropped, the rest are standardised,
    and a PCA keeping ``variance`` of the variance is fitted and applied in
    one step (``fit_transform``), as ``funk_benchmark.py`` does.

    :param values: rows by features.
    :param variance: the share of variance kept.
    :param seed: the PCA seed.
    :returns: rows by components, double precision.
    """
    from sklearn.decomposition import PCA

    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    x = x[:, np.isnan(x).mean(axis=0) < 0.05]
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    spread = x.std(axis=0)
    x = (x[:, spread > 0] - x[:, spread > 0].mean(axis=0)) / spread[spread > 0]
    return PCA(n_components=variance, random_state=seed).fit_transform(x)


def evaluation_options(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The evaluation settings with their defaults filled in and checked.

    :param settings: any mapping holding ``ops_eval_*`` keys.
    :returns: ``{key: value}`` for every key of :data:`EVALUATION_DEFAULTS`.
    :raises ValueError: for an unknown split or reference, or a count below 1.
    """
    settings = dict(settings or {})
    out = {k: (settings[k] if settings.get(k) is not None else v) for k, v in EVALUATION_DEFAULTS.items()}
    references = out["ops_eval_references"]
    out["ops_eval_references"] = [references] if isinstance(references, str) else list(references)
    unknown = [r for r in out["ops_eval_references"] if r.split("@")[0] not in REFERENCE_FILES]
    if unknown:
        raise ValueError(f"ops_eval_references names unknown references {unknown}; choose from "
                         f"{sorted(REFERENCE_FILES)}")
    split_references(out["ops_eval_references"], out["ops_eval_split"])
    for key in ("ops_eval_null_size", "ops_eval_heldout_folds", "ops_eval_decoys"):
        if int(out[key]) < 1:
            raise ValueError(f"{key} must be at least 1; got {out[key]}")
        out[key] = int(out[key])
    out["ops_eval_split"] = str(out["ops_eval_split"]).lower()
    out["ops_eval_time_split"] = bool(out["ops_eval_time_split"])
    return out


def _scored_references(options: Mapping[str, Any]) -> List[Tuple[str, bool]]:
    """``(reference, with p values)`` in the order an evaluation scores them."""
    base = [r for r in options["ops_eval_references"] if "@" not in r]
    split = options["ops_eval_split"]
    if split == "all":
        scored = [(r, True) for r in base]
        scored += [(r, False) for r in HALF_REFERENCES if r.split("@")[0] in base]
    else:
        scored = [(r, True) for r in split_references(base, split)]
    if options["ops_eval_time_split"]:
        scored += [(r, False) for r in TIME_REFERENCES]
    return scored


def _fitness(names: np.ndarray, cells_per_guide: Mapping[str, float], *, fill: bool) -> np.ndarray:
    """Cells per guide of every row; single-guide control rows take the median."""
    values = pd.Series(cells_per_guide, dtype=float)
    median = values.median()
    cpg = np.array([values.get(n, np.nan) if not str(n).startswith("nontargeting_") else median for n in names])
    return np.where(np.isfinite(cpg), cpg, np.nanmedian(cpg)) if fill else cpg


def _shuffled(names: np.ndarray, targeting: np.ndarray, rng) -> np.ndarray:
    """``names`` with the targeting rows' labels permuted."""
    out = names.copy()
    out[targeting] = names[targeting][rng.permutation(targeting.sum())]
    return out


def _check_profiles(names: np.ndarray, pcs: np.ndarray, sims: np.ndarray) -> None:
    """Refuse profiles the tie rule would score as perfect."""
    targeting = np.flatnonzero(~is_control(names))
    if (np.linalg.norm(pcs[targeting], axis=1) < 1e-8).mean() > 0.01:
        raise ValueError("more than 1% of the gene profiles are zero vectors; refusing to score them "
                         "(a tie ranks the partner first, so they would score AP 1)")
    sample = sims[np.ix_(targeting[:300], targeting[300:600])]
    if np.unique(sample.astype(np.float32)).size < 0.5 * sample.size:
        raise ValueError("more than half of the sampled similarities are tied; refusing to score them")


def evaluate_gene_profiles(names: Sequence[Any], pcs, cells_per_guide: Mapping[str, float],
                           reference_dir: Any, settings: Optional[Mapping[str, Any]] = None,
                           *, arm: str = "", null_cache: Optional[str] = None
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every retrieval metric and control of one set of gene profiles.

    Cosine similarity of the components ranks the genes. Rows, in order:
    the per-gene retrieval mAP of every scored reference
    (:func:`spacr.sp_stats.pair_map`, :func:`spacr.sp_stats.summarize_pair_map`;
    p values for the whole references only), each followed for CORUM,
    STRING >= 0.7 and BioGRID by two controls: gene labels shuffled among
    the targeting rows (seed 7) and fitness-matched decoy partners (seed 3);
    the share of control rows among the ten nearest neighbours of every
    targeting gene; the pair AUPRC of Brieflow's STRING pairs (30 gene
    resamples) and of STRING >= 0.7, and the first with shuffled labels
    (seed 8); held-out CORUM members, and the same with the shuffled labels;
    the time split.

    :param names: the row names; control rows start with ``nontargeting``
        or ``NTpseudo``.
    :param pcs: rows by components (:func:`embed_gene_profiles`).
    :param cells_per_guide: ``{gene: cells per guide}``, the fitness used to
        match decoys; single-guide control rows take the median.
    :param reference_dir: see :func:`reference_dirs`.
    :param settings: the ``ops_eval_*`` settings (:func:`evaluation_options`).
    :param arm: the label of the ``arm`` column.
    :param null_cache: see :func:`spacr.sp_stats._pair_null`.
    :returns: ``(metrics, per_gene)``: one row per metric, and the per-gene
        table of every reference with a ``reference`` column.
    :raises ValueError: for degenerate profiles (see the module notes) or
        bad settings.
    """
    from . import sp_stats

    options = evaluation_options(settings)
    names = np.asarray(names)
    pcs = np.asarray(pcs, dtype=float)
    sims = sp_stats.profile_cosine(pcs)
    _check_profiles(names, pcs, sims)
    np.fill_diagonal(sims, -np.inf)
    controls = is_control(names)
    targeting = ~controls
    bins = sp_stats.fitness_bins(_fitness(names, cells_per_guide, fill=False), controls)
    null_size = options["ops_eval_null_size"]
    rows, per_gene = [], []
    for reference, with_p in _scored_references(options):
        timed = reference in TIME_REFERENCES
        positives = positive_matrix(names, reference_pairs(reference, reference_dir))
        table = sp_stats.pair_map(sims, names, positives, controls=controls, null_size=null_size,
                                  pvalues=with_p, null_cache=null_cache)
        summary = sp_stats.summarize_pair_map(table, 300 if "@" in reference or timed else 1000)
        rows.append({"arm": arm, "metric_set": "retrieval", "reference": reference, "control": "", **summary})
        per_gene.append(table.assign(reference=reference))
        if reference in CONTROL_REFERENCES:
            rng = np.random.default_rng(7)
            rng.permutation(len(names))
            shuffled = _shuffled(names, targeting, rng)
            control = sp_stats.pair_map(sims, shuffled, positive_matrix(shuffled, reference_pairs(
                reference, reference_dir)), controls=controls, null_size=null_size, null_cache=null_cache)
            rows.append({"arm": arm, "metric_set": "retrieval", "reference": reference,
                         "control": "shuffled gene labels", **sp_stats.summarize_pair_map(control, 200)})
            decoys = sp_stats.fitness_matched_positives(positives, bins, seed=3)
            control = sp_stats.pair_map(sims, names, decoys, controls=controls, null_size=null_size,
                                        null_cache=null_cache)
            rows.append({"arm": arm, "metric_set": "retrieval", "reference": reference,
                         "control": "fitness-matched random partners", **sp_stats.summarize_pair_map(control, 200)})
    top = np.argsort(-sims[targeting], axis=1)[:, :10]
    rows.append({"arm": arm, "metric_set": "NT control", "reference": "", "control": "NT pseudo-genes",
                 "nt_in_top10": float(controls[top].mean()), "nt_share_of_rows": float(controls.mean())})
    finite = np.where(np.isfinite(sims), sims, -1e9)
    for reference, boot in (("STRING_brieflow", 30), ("STRING>=0.7", 0)):
        positives = positive_matrix(names, reference_pairs(reference, reference_dir))
        rows.append({"arm": arm, "metric_set": "pair AUPRC", "reference": reference, "control": "",
                     **sp_stats.pair_auprc(finite, positives, controls=controls, n_boot=boot)})
    shuffled = _shuffled(names, targeting, np.random.default_rng(8))
    positives = positive_matrix(shuffled, reference_pairs("STRING_brieflow", reference_dir))
    rows.append({"arm": arm, "metric_set": "pair AUPRC", "reference": "STRING_brieflow",
                 "control": "shuffled gene labels", **sp_stats.pair_auprc(finite, positives, controls=controls)})
    zeroed = np.nan_to_num(np.where(np.isfinite(sims), sims, np.nan), nan=0.0)
    groups = reference_groups("CORUM", reference_dir)
    for label, rows_names, seed in (("", names, 0), ("shuffled gene labels", shuffled, 1)):
        rows.append({"arm": arm, "metric_set": "held-out CORUM", "reference": "CORUM", "control": label,
                     **sp_stats.heldout_complex_members(
                         zeroed, rows_names, bins, groups, controls=controls,
                         folds=options["ops_eval_heldout_folds"], decoys=options["ops_eval_decoys"], seed=seed)})
    if options["ops_eval_time_split"]:
        rows += [dict(arm=arm, **r) for r in _time_split(finite, names, bins, controls, reference_dir)]
    metrics = pd.DataFrame(rows)
    metrics["genes"] = int(targeting.sum())
    return metrics, pd.concat(per_gene, ignore_index=True)


def _time_split(sims, names, bins, controls, reference_dir) -> List[Dict[str, Any]]:
    """The time-split rows (:func:`spacr.sp_stats.time_split_pairs`)."""
    from . import sp_stats

    old = set().union(*(reference_pairs(r, reference_dir) for r in TIME_SPLIT_OLD))
    new = set().union(*(reference_pairs(r, reference_dir) for r in TIME_SPLIT_NEW))
    return sp_stats.time_split_pairs(sims, names, bins, old, new, controls=controls)


def specificity_metrics(names: Sequence[Any], pcs, cells_per_guide: Mapping[str, float],
                        reference_dir: Any, settings: Optional[Mapping[str, Any]] = None,
                        *, arm: str = "") -> pd.DataFrame:
    """Specificity and new-biology rows of one set of gene profiles.

    Rows: retrieval of CORUM, STRING >= 0.7 and BioGRID partners against
    fitness-matched non-partners (:func:`spacr.sp_stats.fitness_matched_retrieval`);
    CORUM complex identification, and the same with gene labels shuffled
    among the targeting rows (seed 9); the time split with its binomial p
    values. The components are compared at single precision, as the stored
    arm files hold them; missing fitness takes the median.

    :param names: the row names.
    :param pcs: rows by components.
    :param cells_per_guide: ``{gene: cells per guide}``.
    :param reference_dir: see :func:`reference_dirs`.
    :param settings: the ``ops_eval_*`` settings.
    :param arm: the label of the ``arm`` column.
    :returns: one row per metric.
    """
    from . import sp_stats

    options = evaluation_options(settings)
    names = np.asarray(names)
    sims = sp_stats.profile_cosine(np.asarray(pcs, dtype=np.float32))
    np.fill_diagonal(sims, -np.inf)
    controls = is_control(names)
    bins = sp_stats.fitness_bins(_fitness(names, cells_per_guide, fill=True), controls)
    rows = []
    for reference in CONTROL_REFERENCES:
        positives = positive_matrix(names, reference_pairs(reference, reference_dir))
        rows.append({"arm": arm, "metric_set": "specificity: fitness-matched non-partners",
                     "reference": reference, "control": "",
                     **sp_stats.fitness_matched_retrieval(sims, positives, bins, controls=controls,
                                                          decoys=options["ops_eval_decoys"])})
    zeroed = np.where(np.isfinite(sims), sims, 0.0)
    np.fill_diagonal(zeroed, 0.0)
    groups = reference_groups("CORUM", reference_dir)
    rows.append({"arm": arm, "metric_set": "complex identification", "reference": "CORUM", "control": "",
                 **sp_stats.complex_identification(zeroed, names, groups, controls=controls)})
    shuffled = _shuffled(names, ~controls, np.random.default_rng(9))
    rows.append({"arm": arm, "metric_set": "complex identification", "reference": "CORUM",
                 "control": "shuffled gene labels",
                 **sp_stats.complex_identification(zeroed, shuffled, groups, controls=controls)})
    if options["ops_eval_time_split"]:
        rows += [dict(arm=arm, **r) for r in _time_split(np.where(np.isfinite(sims), sims, -1e9), names, bins,
                                                          controls, reference_dir)]
    return pd.DataFrame(rows)
