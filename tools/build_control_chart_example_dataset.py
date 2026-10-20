"""Stage the Control Chart example set (CPJUMP1 compound plates, CC0 1.0).

Source: Cell Painting Gallery cpg0000-jump-pilot, source_4, batch
2020_11_04_CPJUMP1 (Chandrasekaran et al. 2024, "Three million images and
morphological profiles of cells treated with matched chemical and genetic
perturbations", Nature Methods). The 24 compound plates of that batch share
one plate map (JUMP-Target-1 compound plate): 64 DMSO negative-control wells
(60 on some plates), 26 Cell Painting positive-control wells, 28 diverse
positive controls, and 306 compounds at one dose. A549 and U2OS cells, 24 and
48 hours, with +/-20 % seeding-density and Cas9 A549 variants.

Control charts need at least eight plates to estimate limits from, so the
four plates staged on 2026-10-03 (BR00117010-13) were not enough; this set
keeps every compound plate of the batch.

Well-level "augmented" profiles are cut to plate/well/control metadata and
the same eleven CellProfiler features as the dose example. Plate conditions
come from the CPJUMP1 paper's experiment metadata (benchmark/input/
2020_11_04_CPJUMP1.csv in jump-cellpainting/2024_Chandrasekaran_NatureMethods_
CPJUMP1).

Output (``dest``): ``control_chart_wells.csv``, ``plates.csv``, ``README.md``
(the dataset card) and ``spacr-example-control-chart.tar`` holding the three.

Usage: python tools/build_control_chart_example_dataset.py [dest] [--cache DIR]
"""
from __future__ import annotations

import argparse
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

BASE = ("https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/"
        "source_4/workspace/profiles/2020_11_04_CPJUMP1/{p}/{p}_augmented.csv.gz")
EXPERIMENTS = ("https://raw.githubusercontent.com/jump-cellpainting/"
               "2024_Chandrasekaran_NatureMethods_CPJUMP1/main/benchmark/"
               "input/2020_11_04_CPJUMP1.csv")
ARCHIVE = "spacr-example-control-chart.tar"
TABLE = "control_chart_wells.csv"
FEATURES = (
    "Cells_Number_Object_Number", "Cells_AreaShape_Area",
    "Nuclei_AreaShape_Area", "Cells_AreaShape_FormFactor",
    "Cells_Intensity_MeanIntensity_DNA", "Cells_Intensity_MeanIntensity_Mito",
    "Cells_Intensity_MeanIntensity_ER", "Cells_Intensity_MeanIntensity_RNA",
    "Cells_Intensity_MeanIntensity_AGP",
    "Nuclei_Intensity_IntegratedIntensity_DNA",
    "Cytoplasm_Granularity_1_Mito",
)
CARD = """\
---
license: cc0-1.0
pretty_name: spaCR Control Chart example (CPJUMP1 compound plates)
tags:
- spacr
- cell-painting
- high-content-screening
- quality-control
size_categories:
- 1K<n<10K
---

# spaCR — Control Chart example (CPJUMP1)

Well-level Cell Painting profiles of the {n_plates} compound plates of the
CPJUMP1 pilot, cut to plate, well and control metadata and eleven
CellProfiler features. It is the data behind **Load test data…** on spaCR's
Control Chart screen (an alpha feature): the screen opens
`control_chart_wells.csv`, charts the DMSO negative-control wells plate by
plate, and marks the plates whose controls drift outside the limits.

{n_wells:,} wells, {size_kb:,} kB as `{archive}` (one uncompressed tar, what
spaCR downloads) and the same table loose as `control_chart_wells.csv`.

## Source

* Cell Painting Gallery `cpg0000-jump-pilot`, `source_4`, batch
  `2020_11_04_CPJUMP1`, file
  `workspace/profiles/2020_11_04_CPJUMP1/<plate>/<plate>_augmented.csv.gz`
  (median-aggregated, well-level CellProfiler profiles).
* Plates: {plates}.
* Plate conditions (`plates.csv`) from the CPJUMP1 paper's experiment
  metadata, `benchmark/input/2020_11_04_CPJUMP1.csv` in
  [jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1).

## Columns

`control_chart_wells.csv`: `plate`, `run_order` (rank of the plate barcode,
1 = lowest; the barcode order, not a recorded acquisition time),
`plate_condition`, `cell_line`, `timepoint_h`, `well`, `well_type`
(`negcon` = DMSO, `poscon_cp`, `poscon_diverse`, `poscon_orf`, `trt`),
`pert_iname`, `broad_sample`, then the features
{features}.
`Cells_Number_Object_Number` is CellProfiler's median object number per
well, which rises with the number of cells in a field.

The plates mix two cell lines, two time points and three seeding densities,
so a chart of every plate shows those design changes as shifts. Filter on
`plate_condition` to chart one condition.

## Licence and citation

CC0 1.0, as the Cell Painting Gallery publishes it. Please cite:

* Chandrasekaran SN, Cimini BA, Goodale A, et al. Three million images and
  morphological profiles of cells treated with matched chemical and genetic
  perturbations. *Nature Methods* 21, 1114–1121 (2024).
  doi:10.1038/s41592-024-02241-6
* Weisbart E, Kumar A, Arevalo J, et al. Cell Painting Gallery: an open
  resource for image-based profiling. *Nature Methods* 21, 1775–1777 (2024).
  doi:10.1038/s41592-024-02399-z

Built by `tools/build_control_chart_example_dataset.py` in
[spaCR](https://github.com/EinarOlafsson/spacr).
"""


def _download(url: str, path: Path) -> Path:
    """Fetch ``url`` to ``path`` unless it is already there."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _conditions(cache: Path) -> pd.DataFrame:
    """One row per compound plate: barcode, condition, cell line, hours."""
    table = pd.read_csv(_download(EXPERIMENTS, cache / "2020_11_04_CPJUMP1.csv"))
    table = table[table["Description"].str.contains("Compound")]
    hours = table["Description"].str.extract(r"(\d+)-hour")[0].astype(int)
    return pd.DataFrame({
        "plate": table["Assay_Plate_Barcode"].to_numpy(),
        "plate_condition": table["Description"].map(
            lambda text: re.sub(r"\s+Plate \d+$", "", text)).to_numpy(),
        "cell_line": table["Description"].str.split().str[0].to_numpy(),
        "timepoint_h": hours.to_numpy(),
    }).sort_values("plate", ignore_index=True)


def _wells(plates: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Every well of every compound plate, cut to metadata and features."""
    frames = []
    for plate in plates["plate"]:
        path = _download(BASE.format(p=plate),
                         cache / f"{plate}_augmented.csv.gz")
        raw = pd.read_csv(path, usecols=[
            "Metadata_Plate", "Metadata_Well", "Metadata_pert_type",
            "Metadata_control_type", "Metadata_pert_iname",
            "Metadata_broad_sample", *FEATURES])
        well_type = raw["Metadata_control_type"].where(
            raw["Metadata_pert_type"] == "control", "trt").fillna("trt")
        frames.append(pd.DataFrame({
            "plate": raw["Metadata_Plate"], "well": raw["Metadata_Well"],
            "well_type": well_type,
            "pert_iname": raw["Metadata_pert_iname"].fillna("DMSO"),
            "broad_sample": raw["Metadata_broad_sample"].fillna("DMSO"),
            **{feature: raw[feature] for feature in FEATURES}}))
    wells = pd.concat(frames, ignore_index=True)
    plates = plates.assign(run_order=range(1, len(plates) + 1))
    wells = wells.merge(plates, on="plate", how="left", validate="many_to_one")
    head = ["plate", "run_order", "plate_condition", "cell_line",
            "timepoint_h", "well", "well_type", "pert_iname", "broad_sample"]
    return wells[head + list(FEATURES)].sort_values(
        ["run_order", "well"], ignore_index=True)


def build(dest: Path, cache: Path) -> Path:
    """Write the table, the plate list, the card and the archive into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    plates = _conditions(cache)
    wells = _wells(plates, cache)
    wells.to_csv(dest / TABLE, index=False)
    plates.to_csv(dest / "plates.csv", index=False)
    size_kb = round((dest / TABLE).stat().st_size / 1000)
    (dest / "README.md").write_text(CARD.format(
        n_plates=len(plates), n_wells=len(wells), size_kb=size_kb,
        archive=ARCHIVE, plates=", ".join(plates["plate"]),
        features=", ".join(f"`{f}`" for f in FEATURES)), encoding="utf-8")
    with tarfile.open(dest / ARCHIVE, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in (TABLE, "plates.csv", "README.md"):
            info = tar.gettarinfo(str(dest / name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with (dest / name).open("rb") as handle:
                tar.addfile(info, handle)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", nargs="?", default=str(
        Path.home() / ".cache" / "spacr" / "example_data" / "control_chart_build"))
    parser.add_argument("--cache", default=str(
        Path.home() / ".cache" / "spacr" / "cpjump1_raw"))
    args = parser.parse_args()
    print(build(Path(args.dest), Path(args.cache)))
