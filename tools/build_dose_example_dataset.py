"""Stage the dose-response example set (LINCS Cell Painting, CC0 1.0).

Source: Cell Painting Gallery cpg0004-lincs (Way et al. 2022, "Morphology and
gene expression profiling provide complementary information for mapping cell
state", Cell Systems), batch 2016_04_01_a549_48hr_batch1, plates
SQ00014812-SQ00014815: four replicates of plate map C-7161-01-LM6-022, A549
cells, 56 compounds at six doses (0.04-10 uM) plus DMSO wells. Licence CC0 1.0.

Well-level "augmented" profiles are cut to plate/well/compound/dose metadata
and eleven CellProfiler features. From that one table the script derives:

* ``dose_plate.csv``                       Dose-Response
* ``runs/<run>/results.csv`` (two runs)   Prediction Profiler, Run Compare,
                                           Run History (registered at load)
* ``training/model/...`` + ``settings/``   Training Runs
* ``README.md`` (dataset card) and ``spacr-example-dose.tar`` (all of it)

Usage: python tools/build_dose_example_dataset.py [dest] [--cache DIR]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ("https://cellpainting-gallery.s3.amazonaws.com/cpg0004-lincs/broad/"
        "workspace/profiles/2016_04_01_a549_48hr_batch1/{p}/{p}_augmented.csv.gz")
PLATES = ("SQ00014812", "SQ00014813", "SQ00014814", "SQ00014815")
FEATURES = (
    "Cells_Number_Object_Number", "Cells_AreaShape_Area",
    "Nuclei_AreaShape_Area", "Cells_AreaShape_FormFactor",
    "Cells_Intensity_MeanIntensity_DNA", "Cells_Intensity_MeanIntensity_Mito",
    "Cells_Intensity_MeanIntensity_ER", "Cells_Intensity_MeanIntensity_RNA",
    "Cells_Intensity_MeanIntensity_AGP",
    "Nuclei_Intensity_IntegratedIntensity_DNA",
    "Cytoplasm_Granularity_1_Mito",
)
RESPONSE = "Cells_Number_Object_Number"
CARD = """\
---
license: cc0-1.0
pretty_name: spaCR Dose-Response example (LINCS Cell Painting, A549)
tags:
- spacr
- cell-painting
- dose-response
- high-content-screening
size_categories:
- 1K<n<10K
---

# spaCR — Dose-Response example (LINCS Cell Painting)

Well-level Cell Painting profiles of one LINCS plate map, four replicate
plates, cut to plate, well, compound, dose and eleven CellProfiler features,
with two regression runs and two training runs made from them. It is the data
behind **Load test data…** on spaCR's Dose-Response, Prediction Profiler, Run
Compare, Run History and Training Runs screens (alpha features).

{n_wells:,} wells: {n_curves} compounds at six doses (55 from 0.041 to
10 µM, one from 0.033 to 8.1 µM), MG-132 and bortezomib at one dose (20 µM)
as positive controls, and DMSO vehicle wells (dose 0). `{archive}` (one
uncompressed tar, what spaCR downloads) holds everything; `dose_plate.csv` is
also published loose.

## Source

* Cell Painting Gallery `cpg0004-lincs`, batch `2016_04_01_a549_48hr_batch1`,
  file `broad/workspace/profiles/2016_04_01_a549_48hr_batch1/<plate>/<plate>_augmented.csv.gz`
  (median-aggregated, well-level CellProfiler profiles).
* Plates **SQ00014812, SQ00014813, SQ00014814, SQ00014815**: four replicates
  of plate map **C-7161-01-LM6-022**. A549 cells, 48 h.

## Files

* `dose_plate.csv`: `plate`, `well`, `compound`, `moa`, `dose_uM`, then
  {features}. `Cells_Number_Object_Number` is CellProfiler's median object
  number per well, which rises with the number of cells in a field; it is the
  response the Dose-Response button fits.
* `runs/<run>/results.csv` and `settings.json`: ordinary least squares of
  that response on each compound's scaled log10 dose (0 = vehicle or another
  compound, 0.1–1 = 0.041–10 µM), raw (`regression_raw`) and divided by each
  plate's DMSO median (`regression_dmso_normalised`). Derived here, not part
  of the source.
* `training/`: logistic regression (SGD) telling DMSO from ≥3.3 µM wells,
  trained on SQ00014812–14 and validated on SQ00014815, 20 epochs at learning
  rate 0.01 and 40 epochs at 0.001, in spaCR's training layout. Derived here.

## Licence and citation

CC0 1.0, as the Cell Painting Gallery publishes it. Please cite:

* Way GP, Natoli T, Adeboye A, et al. Morphology and gene expression profiling
  provide complementary information for mapping cell state. *Cell Systems*
  13(11), 911–923 (2022). doi:10.1016/j.cels.2022.10.001
* Weisbart E, Kumar A, Arevalo J, et al. Cell Painting Gallery: an open
  resource for image-based profiling. *Nature Methods* 21, 1775–1777 (2024).
  doi:10.1038/s41592-024-02399-z

Built by `tools/build_dose_example_dataset.py` in
[spaCR](https://github.com/EinarOlafsson/spacr).
"""
ARCHIVE = "spacr-example-dose.tar"


def _plates(cache: Path) -> pd.DataFrame:
    """Read the four plates, downloading any not yet in ``cache``."""
    frames = []
    cache.mkdir(parents=True, exist_ok=True)
    for plate in PLATES:
        path = cache / f"{plate}_augmented.csv.gz"
        if not path.is_file():
            urllib.request.urlretrieve(BASE.format(p=plate), path)
        frames.append(pd.read_csv(path, usecols=[
            "Metadata_Plate", "Metadata_Well", "Metadata_pert_iname",
            "Metadata_moa", "Metadata_mmoles_per_liter", *FEATURES]))
    raw = pd.concat(frames, ignore_index=True)
    plate = pd.DataFrame({
        "plate": raw["Metadata_Plate"],
        "well": raw["Metadata_Well"],
        "compound": raw["Metadata_pert_iname"].fillna("DMSO"),
        "moa": raw["Metadata_moa"].fillna(""),
        "dose_uM": raw["Metadata_mmoles_per_liter"].round(4),
    })
    for feature in FEATURES:
        plate[feature] = raw[feature]
    plate.loc[plate["dose_uM"] == 0, "compound"] = "DMSO"
    return plate


def _design(plate: pd.DataFrame) -> pd.DataFrame:
    """One column per compound: its scaled log10 dose, else 0."""
    dosed = plate[(plate["dose_uM"] > 0) & (plate["dose_uM"] <= 10)]
    low, high = np.log10(dosed["dose_uM"].min()), np.log10(10.0)
    columns = {}
    for compound in sorted(dosed["compound"].unique()):
        mask = (plate["compound"] == compound) & (plate["dose_uM"] > 0)
        scaled = (np.log10(plate["dose_uM"].clip(lower=1e-9)) - low) / (high - low)
        columns[compound] = np.where(mask, 0.1 + 0.9 * scaled.clip(0, 1), 0.0)
    return pd.DataFrame(columns, index=plate.index)


def _regression(plate: pd.DataFrame, normalise: bool, out: Path) -> None:
    """Fit and write one OLS run with its settings."""
    import statsmodels.api as sm

    y = plate[RESPONSE].astype(float)
    if normalise:
        vehicle = plate[plate["compound"] == "DMSO"].groupby("plate")[RESPONSE]
        y = y / plate["plate"].map(vehicle.median())
    design = sm.add_constant(_design(plate)).rename(columns={"const": "Intercept"})
    fit = sm.OLS(y, design.loc[:, (design != 0).any()]).fit()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": fit.params.index, "coefficient": fit.params.values,
                  "std_err": fit.bse.values, "p_value": fit.pvalues.values}
                 ).to_csv(out / "results.csv", index=False)
    settings = {"regression_type": "ols", "dependent_variable": RESPONSE,
                "plate_normalisation": "dmso_median" if normalise else "none",
                "dose_scale": "log10, 0.1-1 for 0.041-10 uM",
                "plates": list(PLATES)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2))


def _training(plate: pd.DataFrame, root: Path) -> None:
    """Train two SGD logistic runs and write spaCR's training layout."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import average_precision_score, log_loss
    from sklearn.preprocessing import StandardScaler

    features = [f for f in FEATURES if f != RESPONSE]
    rows = plate[(plate["compound"] == "DMSO") | (plate["dose_uM"] >= 3.3)]
    label = (rows["compound"] != "DMSO").astype(int).to_numpy()
    held = (rows["plate"] == PLATES[-1]).to_numpy()
    scaler = StandardScaler().fit(rows.loc[~held, features])
    x = scaler.transform(rows[features])
    for epochs, rate in ((20, 0.01), (40, 0.001)):
        model = SGDClassifier(loss="log_loss", learning_rate="constant",
                              eta0=rate, random_state=0)
        run = root / "model" / "logistic_sgd" / "profiles" / f"epochs_{epochs}"
        run.mkdir(parents=True, exist_ok=True)
        curves = {"train": [], "validation": []}
        for epoch in range(1, epochs + 1):
            model.partial_fit(x[~held], label[~held], classes=[0, 1])
            for split, mask in (("train", ~held), ("validation", held)):
                p = model.predict_proba(x[mask])[:, 1]
                truth = label[mask]
                call = (p >= 0.5).astype(int)
                curves[split].append({
                    "epoch": epoch, "loss": log_loss(truth, p, labels=[0, 1]),
                    "accuracy": float((call == truth).mean()),
                    "neg_accuracy": float((call[truth == 0] == 0).mean()),
                    "pos_accuracy": float((call[truth == 1] == 1).mean()),
                    "prauc": average_precision_score(truth, p),
                    "optimal_threshold": 0.5})
        for split, values in curves.items():
            pd.DataFrame(values).to_csv(run / f"{split}.csv", index=False)
        settings = root / "settings"
        settings.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Key": ["model_type", "epochs", "learning_rate",
                              "classes", "train_plates", "val_plate",
                              "features"],
                      "Value": ["logistic_sgd", epochs, rate,
                                "DMSO vs >=3.3 uM", ",".join(PLATES[:-1]),
                                PLATES[-1], ",".join(features)]}
                     ).to_csv(settings / f"train_test_logistic_sgd_{epochs}.csv",
                              index=False)


def build(dest: Path, cache: Path) -> Path:
    """Write the whole set into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    plate = _plates(cache)
    plate.to_csv(dest / "dose_plate.csv", index=False)
    _regression(plate, False, dest / "runs" / "regression_raw")
    _regression(plate, True, dest / "runs" / "regression_dmso_normalised")
    _training(plate, dest / "training")
    dosed = plate[plate["dose_uM"] > 0].groupby("compound")["dose_uM"]
    (dest / "README.md").write_text(CARD.format(
        n_wells=len(plate), n_compounds=dosed.ngroups,
        n_curves=int((dosed.nunique() >= 6).sum()), archive=ARCHIVE,
        features=", ".join(f"`{f}`" for f in FEATURES)), encoding="utf-8")
    _archive(dest)
    return dest


def _archive(dest: Path) -> Path:
    """Tar every file of the set, relative to ``dest``, with fixed owners."""
    import tarfile

    path = dest / ARCHIVE
    names = sorted(p for p in dest.rglob("*")
                   if p.is_file() and p.name != ARCHIVE
                   and not p.name.startswith("."))
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in names:
            info = tar.gettarinfo(str(name), arcname=str(name.relative_to(dest)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with name.open("rb") as handle:
                tar.addfile(info, handle)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", nargs="?", default=str(
        Path.home() / ".cache" / "spacr" / "example_data" / "dose_response_lincs"))
    parser.add_argument("--cache", default=str(
        Path.home() / ".cache" / "spacr" / "lincs_raw"))
    args = parser.parse_args()
    print(build(Path(args.dest), Path(args.cache)))
