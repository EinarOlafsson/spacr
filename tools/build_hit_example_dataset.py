"""Stage the Investigate Hit example set (a cut of the published TSG101 screen).

Source: the maintainer's GFP-TSG101 recruitment screen in HeLa cells, plate 1
(einarolafsson/spacr-example-screen ``plate1-measurements.tar``), with its
per-cell cross-validated predictions ``plate1_dv.csv`` and per-well guide
read counts ``plate_1_unique_combinations.csv`` from the spaCR example-data
release. Columns c1, c2 and c3 hold the screen's controls: c1 is enriched for
the negative control (SAG1, TGGT1_233460), c2 for the positive control (GRA14,
TGGT1_239740, whose loss removes TSG101 recruitment) and c3 a mix; the other
columns carry the whole guide library.

What is cut, by fixed rules and without resampling:

* wells: columns c1, c2 and c3 of rows r1-r8, plus every imaged well whose
  reads hold no GRA14 guide (the target-free wells Investigate Hit compares
  against);
* fields: f1-f4 of the sixteen imaged in each of those wells;
* every row of every table of ``measurements.db`` belonging to those fields,
  the matching rows of ``plate1_dv.csv``, and the read counts of those wells
  (counts are per well, so a well's guide fractions are unchanged).

Then spaCR itself is run on the cut: :func:`spacr.ml.perform_regression`
(OLS, one row per well, control columns kept) writes ``results/``, which
names GRA14 (``239740``) and its guides; its ``regression_data.csv`` is the
per-well guide-fraction table Investigate Hit reads. Finally
:func:`spacr.hit_investigation.investigate_hit` is run once on the cut as a
check, into a scratch folder that is not shipped.

Output (``dest``): ``hit_example/`` (the unpacked set), ``README.md`` (the
dataset card), ``guide_fractions.csv`` (loose main table) and
``spacr-example-hit.tar``.

Usage: python tools/build_hit_example_dataset.py DEST --source PLATE1_DB
           --scores plate1_dv.csv --counts plate_1_unique_combinations.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

ARCHIVE = "spacr-example-hit.tar"
FOLDER = "hit_example"
SCORES = "plate1_dv.csv"
COUNTS = "plate_1_unique_combinations.csv"
FRACTIONS = "guide_fractions.csv"
TARGET_GENE = "239740"
TARGET_PREFIX = "TGGT1_239740_"
CONTROL_COLUMNS = ("c1", "c2", "c3")
CONTROL_ROWS = tuple(f"r{n}" for n in range(1, 9))
FIELDS = ("f1", "f2", "f3", "f4")
TABLES = ("cell", "cytoplasm", "nucleus", "pathogen", "png_list")


def _wells(counts: pd.DataFrame, scored: pd.DataFrame) -> list:
    """The kept wells: control columns of r1-r8 and every GRA14-free well."""
    imaged = set(zip(scored["row"], scored["col"]))
    target = counts[counts["grna_name"].astype(str).str.startswith(TARGET_PREFIX)]
    with_target = set(zip(target["row_name"], target["column_name"]))
    sequenced = set(zip(counts["row_name"], counts["column_name"]))
    free = {well for well in sequenced - with_target if well in imaged}
    controls = {(row, col) for row in CONTROL_ROWS for col in CONTROL_COLUMNS}
    return sorted(controls | free,
                  key=lambda w: (int(w[1][1:]), int(w[0][1:])))


def _cut_database(source: Path, dest: Path, keep: set) -> dict:
    """Copy every table's rows for the kept fields into a new database."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    out = sqlite3.connect(dest)
    counts = {}
    try:
        for name, sql in src.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table'"):
            out.execute(sql)
            frame = pd.read_sql_query(f'SELECT * FROM "{name}"', src)
            if name in TABLES:
                key = list(zip(frame["rowID"], frame["columnID"],
                               frame["fieldID"]))
                frame = frame[[k in keep for k in key]]
            frame.to_sql(name, out, if_exists="append", index=False)
            counts[name] = len(frame)
        for (sql,) in src.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND sql IS NOT NULL"):
            out.execute(sql)
        out.commit()
        out.execute("VACUUM")
    finally:
        src.close()
        out.close()
    return counts


def _regression(folder: Path) -> Path:
    """Run spaCR's regression on the cut; return its results folder."""
    from spacr.ml import perform_regression
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({
        "src": str(folder),
        "paired_data": [{"score": str(folder / SCORES),
                         "count": str(folder / COUNTS), "plate": "plate1"}],
        "dependent_variable": "pred",
        "analysis_mode": "regression",
        "analysis_unit": "well",
        "regression_type": "ols",
        "inference": "parametric",
        "transform": None,
        "agg_type": "mean",
        "min_cells_per_well": 10,
        "fraction_threshold": 0.02,
        "filter_value": [],
        "filter_column": "columnID",
        "outlier_detection": False,
        "batch_correction": None,
        "model_plate_position": False,
        "level": "both",
        "verbose": False,
    })
    outcome = perform_regression(settings)
    return Path(outcome["res_folder"])


def _guides(results: Path) -> list:
    """GRA14's guides as the regression's per-well table names them."""
    data = pd.read_csv(results / "regression_data.csv")
    guides = sorted({g for g in data["grna"].astype(str)
                     if g.startswith(TARGET_PREFIX)
                     or g.startswith(TARGET_GENE + "_")})
    if not guides:
        raise SystemExit("the regression table names no GRA14 guide")
    return guides


def build(dest: Path, source: Path, scores: Path, counts: Path) -> dict:
    """Cut the screen, run spaCR on the cut, write the card and the tar."""
    dest = Path(dest)
    folder = dest / FOLDER
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    count_frame = pd.read_csv(counts)
    score_frame = pd.read_csv(scores)
    wells = _wells(count_frame, score_frame)
    keep = {(row, col, field) for row, col in wells for field in FIELDS}
    tables = _cut_database(source, folder / "measurements" / "measurements.db",
                           keep)
    key = list(zip(score_frame["row"], score_frame["col"], score_frame["field"]))
    score_cut = score_frame[[k in keep for k in key]]
    score_cut.to_csv(folder / SCORES, index=False)
    well_set = set(wells)
    count_cut = count_frame[[w in well_set for w in zip(
        count_frame["row_name"], count_frame["column_name"])]]
    count_cut.to_csv(folder / COUNTS, index=False)
    final = _regression(folder)
    run = final.resolve().relative_to(folder.resolve()).as_posix()
    guides = _guides(final)
    shutil.copyfile(final / "regression_data.csv", folder / FRACTIONS)
    _scrub_paths(folder)
    hit = {"target_gene": TARGET_GENE, "target_guides": guides,
           "score_column": "pred", "hit_direction": "positive",
           "results_folder": run,
           "db_path": "measurements/measurements.db",
           "predictions_file": SCORES, "guide_fractions_file": FRACTIONS}
    hit.update(_hit_row(final))
    data = pd.read_csv(final / "regression_data.csv")
    support = data[data["grna"].astype(str).isin(guides)
                   & (data["fraction"] > 0)]
    hit["hit_n_guides"] = len(guides)
    hit["hit_well_support"] = int(support["prc"].nunique())
    (folder / "hit.json").write_text(json.dumps(hit, indent=2) + "\n",
                                     encoding="utf-8")
    check = _check(folder, hit)
    (dest / FRACTIONS).write_bytes((folder / FRACTIONS).read_bytes())
    summary = {"wells": [f"{r}{c}" for r, c in wells], "tables": tables,
               "scores": len(score_cut), "count_rows": len(count_cut),
               "hit": hit, "check": check,
               "fraction_wells": {"target": check["target_wells"],
                                  "control": check["control_wells"]}}
    _write_card(dest, folder, summary)
    _write_tar(dest, folder)
    summary["tar_bytes"] = (dest / ARCHIVE).stat().st_size
    summary["tar_sha256"] = hashlib.sha256(
        (dest / ARCHIVE).read_bytes()).hexdigest()
    return summary


def _scrub_paths(folder: Path) -> None:
    """Replace the build folder in text outputs by the dataset placeholder.

    spaCR fills ``<dataset>`` in ``settings/*.csv`` with the unpack location
    (:func:`spacr.example_archives.make_the_example_paths_absolute`); the
    other text files keep it as a plain marker instead of a build path.
    """
    from spacr.example_archives import DATASET_PLACEHOLDER

    prefix = str(folder.resolve()).rstrip("/")
    for path in folder.rglob("*"):
        if path.suffix not in {".csv", ".json", ".txt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if prefix in text:
            path.write_text(text.replace(prefix, DATASET_PLACEHOLDER),
                            encoding="utf-8")


def _hit_row(results: Path) -> dict:
    """GRA14's gene-level coefficient from the regression's results."""
    for name in ("results.csv", "regression_results.csv"):
        path = results / name
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        label = next((c for c in ("feature", "gene", "term", "Unnamed: 0")
                      if c in table.columns), table.columns[0])
        rows = table[table[label].astype(str).str.contains(TARGET_GENE)]
        gene_rows = rows[rows[label].astype(str).str.startswith("gene_")]
        rows = gene_rows if not gene_rows.empty else rows
        if rows.empty:
            continue
        row = rows.iloc[0]
        found = {"hit_term": str(row[label])}
        for out, names in (("hit_effect", ("coefficient", "coef", "beta")),
                           ("hit_p_value", ("p_value", "P>|t|", "pvalue")),
                           ("hit_fdr", ("p_adjusted", "q_value", "fdr"))):
            for column in names:
                if column in table.columns:
                    found[out] = float(row[column])
                    break
        return found
    raise SystemExit(f"no results table in {results} names {TARGET_GENE}")


def _check(folder: Path, hit: dict) -> dict:
    """Run Investigate Hit once on the cut, outside the shipped folder."""
    from spacr.hit_investigation import investigate_hit

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / FOLDER
        shutil.copytree(folder, work)
        outcome = investigate_hit({
            "db_path": str(work / hit["db_path"]),
            "predictions_file": str(work / SCORES),
            "guide_fractions_file": str(work / FRACTIONS),
            "results_folder": str(work / hit["results_folder"]),
            "target_gene": hit["target_gene"],
            "target_guides": hit["target_guides"],
            "score_column": "pred", "hit_direction": "positive",
            "dst": str(Path(scratch) / "out"),
            "hit_bootstrap": 500, "hit_permutations": 1000,
            "hit_pipeline_permutations": 0, "verbose": False,
        })
        result = outcome["result"]
        return {"cells": int(len(result.cells)),
                "wells": int(len(result.wells)),
                "split_level": result.split_level,
                "target_wells": int(
                    (result.wells["target_guide_fraction"] > 0).sum()),
                "control_wells": int(
                    (result.wells["target_guide_fraction"] <= 0).sum()),
                "prevalence_difference": float(
                    result.validation.get("prevalence_difference", float("nan")))}


CARD = """\
---
license: mit
pretty_name: spaCR Investigate Hit example (TSG101 recruitment screen, plate 1 cut)
tags:
- spacr
- microscopy
- toxoplasma
- crispr-screen
size_categories:
- 1K<n<10K
---

# spaCR — Investigate Hit example

A small cut of the published spaCR GFP-TSG101 recruitment screen
([einarolafsson/spacr-example-screen](https://huggingface.co/datasets/einarolafsson/spacr-example-screen)),
with a Regression run made from it by spaCR. It is the data behind
**Load test data…** on spaCR's Investigate Hit screen (an alpha feature):
the screen opens the hit GRA14 (`{gene}`, the screen's positive control,
whose loss removes TSG101 recruitment to the vacuole) with the guides the
Regression names ({guides}), and traces it to candidate cells.

{size_line}`{fractions}` is the guide-fraction table, also loose here.

## Source

HeLa GFP-TSG101 cells infected with a pooled *Toxoplasma gondii* CRISPR-Cas9
library, plate 1 of the four-plate screen:

* `plate1-measurements.tar` from
  [einarolafsson/spacr-example-screen](https://huggingface.co/datasets/einarolafsson/spacr-example-screen)
  (spaCR Measure database: cell, cytoplasm, nucleus, pathogen and png_list
  tables);
* `plate1_dv.csv` (per-cell cross-validated predictions of spaCR's
  classifier: `pred` is the probability of the GRA14-knockout-like
  phenotype) and `plate_1_unique_combinations.csv` (guide read counts per
  well) from the spaCR example-data release.

Columns c1, c2 and c3 hold the controls: c1 is enriched for the negative
control SAG1 (`TGGT1_233460`), c2 for the positive control GRA14
(`TGGT1_239740`), c3 a mix; the other columns carry the whole library.

## What was cut

Fixed rules, no resampling, no edited values:

* {n_wells} wells: columns c1, c2 and c3 of rows r1-r8, plus the {n_free}
  imaged wells whose reads hold no GRA14 guide at all ({free});
* fields f1-f4 of the sixteen imaged per well;
* every row of every table of `measurements.db` for those fields
  ({tables}), the matching {n_scores:,} rows of `plate1_dv.csv`, and the
  {n_counts:,} read-count rows of those wells (counts are per well, so the
  guide fractions are those of the whole well).

## Made with spaCR on the cut

* `{run}/`: `spacr.ml.perform_regression` (OLS, one row per well, mean
  `pred`, guide fraction threshold 0.02, control columns kept). It names
  `{gene}` ({term}: effect {effect}, p {p_value}, adjusted {fdr}).
* `{fractions}`: that run's `regression_data.csv`, one fraction per well and
  guide as the regression used them (a guide under 2 % of a well's reads
  counts as absent, so {n_target} wells carry GRA14 and {n_control} do not).
* `hit.json`: the hit and the file names, as the Load test data button
  fills them in.

A check run of `spacr.hit_investigation.investigate_hit` on this set
cross-fitted {check_cells:,} cells over {check_wells} wells (held out by
{split}). Hit-like prevalence in GRA14 wells minus GRA14-free wells was
{difference}: on this cut the morphology alone does not separate the two,
which the screen reports honestly. Its outputs are not shipped.

This is a demonstration cut: {n_wells} wells of one plate cannot stand for
the screen, and the hit is the screen's built-in positive control. Use the
full plates in spacr-example-screen for analysis.

## Licence and citation

MIT, as the maintainer's other spaCR example sets
(spacr-example-annotate, spacr-example-recruitment). Please cite spaCR:

* Olafsson EB, Arnold C-S, Kellermeier JA, Rimple PA, Kaur H, Wang Y,
  Sexton JZ, Svärd S, Carruthers VB, O'Meara MJ. spaCR: spatial phenotype
  analysis of CRISPR-Cas9 screens. Zenodo. doi:10.5281/zenodo.21343316

Built by `tools/build_hit_example_dataset.py` in
[spaCR](https://github.com/EinarOlafsson/spacr).
"""


def _fmt(value) -> str:
    """Three significant figures, or a dash."""
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return "—"


def _card(summary: dict, size_line: str) -> str:
    """The dataset card for ``summary``, with ``size_line`` about the tar."""
    hit = summary["hit"]
    controls = {f"{row}{col}" for row in CONTROL_ROWS for col in CONTROL_COLUMNS}
    free = [w for w in summary["wells"] if w not in controls]
    return CARD.format(
        gene=hit["target_gene"],
        guides=", ".join(f"`{g}`" for g in hit["target_guides"]),
        size_line=size_line, fractions=FRACTIONS,
        n_wells=len(summary["wells"]), n_free=len(free), free=", ".join(free),
        tables=", ".join(f"{k} {v:,}" for k, v in summary["tables"].items()
                         if k in TABLES),
        n_scores=summary["scores"], n_counts=summary["count_rows"],
        run=hit["results_folder"],
        term=hit.get("hit_term", hit["target_gene"]),
        effect=_fmt(hit.get("hit_effect")), p_value=_fmt(hit.get("hit_p_value")),
        fdr=_fmt(hit.get("hit_fdr")),
        n_target=summary["fraction_wells"]["target"],
        n_control=summary["fraction_wells"]["control"],
        check_cells=summary["check"]["cells"],
        check_wells=summary["check"]["wells"],
        split=summary["check"]["split_level"],
        difference=_fmt(summary["check"]["prevalence_difference"]))


def _write_card(dest: Path, folder: Path, summary: dict) -> None:
    """Write the card that ships inside the tar."""
    (folder / "README.md").write_text(_card(summary, (
        f"Downloaded as `{ARCHIVE}` from the dataset page. ")),
        encoding="utf-8")


def _write_tar(dest: Path, folder: Path) -> None:
    """Pack ``folder`` as ``hit_example/...`` with fixed metadata."""
    with tarfile.open(dest / ARCHIVE, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(folder.rglob("*")):
            if path.is_dir() or path.suffix in {".db-shm", ".db-wal"}:
                continue
            info = tar.gettarinfo(str(path),
                                  arcname=str(path.relative_to(folder)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                tar.addfile(info, handle)


def finish_card(dest: Path, summary: dict) -> None:
    """Write the dataset page's card, with the archive's size and digest."""
    data = (dest / ARCHIVE).read_bytes()
    line = (f"`{ARCHIVE}`: {len(data):,} bytes ({len(data) / 1e6:.1f} MB), "
            f"one uncompressed tar, SHA-256 "
            f"`{hashlib.sha256(data).hexdigest()}`. ")
    (dest / "README.md").write_text(_card(summary, line), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", type=Path)
    parser.add_argument("--source", required=True, type=Path,
                        help="plate 1 measurements.db (spacr-example-screen)")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--counts", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.dest, args.source, args.scores, args.counts)
    finish_card(args.dest, result)
    print(json.dumps(result, indent=2, default=str))
