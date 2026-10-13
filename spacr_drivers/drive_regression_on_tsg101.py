"""Drive regression on the real four-plate tsg101 screen.

The screen keeps a score table and a guide-count table per plate and the
settings that were run against them, so this is regression on its own data
rather than on a fixture: OLS at guide and gene level, hits called with a
per-gene explanation, the bundled Toxoplasma annotation joined onto the gene
table, and the figures.

THE FIT TYPE IS THE RECORDED ONE unless a third argument names another.
``mixed`` is what the screen was analysed with and it is the long path -- a
mixed model at both guide and gene level over four plates is a run to leave
going, not one to watch. ``ols`` on the same tables is minutes.

TWO THINGS ARE CHANGED FROM THE RECORDED SETTINGS, and both are printed:

* the permutation count is lowered, because the recorded 200,000 is hours of
  work and the shape of the result is settled long before that;
* the measurements databases named in ``paired_data`` are not staged. The fit
  reads scores and counts only -- the databases survive the settings round
  trip so a reloaded run still names them, and copying two gigabytes to prove
  they are unread is not worth it.

THE SCREEN CARRIES ITS OWN RIGHT ANSWER, and it is the controls. A fixture
that plants a coefficient is the usual way to ask a regression whether it
recovered anything; a real screen does it with a positive control, and this
one names ``positive_control=239740`` in its own settings file. A fit that
writes every table and every figure and leaves that gene un-called, or calls
it on the wrong side of zero, has produced a result nobody can use -- and
counting the output files does not notice. So the driver reads the gene
table back and checks where the controls landed.

A BLANK IS NOT AN ANSWER. ``hinge_threshold,`` on a line is what a saved
settings file looks like for a box nobody filled in, and a settings loader
that reads that blank as a value refuses the run over a setting the chosen
regression does not even use. That is why this driver loads settings through
spaCR's own loader rather than through pandas.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (check, dataset_root, preflight, read_settings, require,
                      run, scratch, settings_file, stage)

DEFAULT_ROOT = "/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen"

PLATES = (1, 2, 3, 4)

REQUIRED = tuple(
    [f"plate{n}_dv.csv" for n in PLATES]
    + [f"plate_{n}_unique_combinations.csv" for n in PLATES]
    + ["settings/regression.csv"])

SETTINGS_CANDIDATES = ("settings/regression.csv",)

#: Enough permutations for the guide-level p-values to be stable, few enough
#: to finish. The recorded value is printed beside it.
PERMUTATIONS = 1000

#: Pre-flight errors that are wrong about regression, with the reason. The
#: run is fine; the check is not.
KNOWN_FALSE_POSITIVES = {
    "src": "regression fits score and count tables. It does not open a "
           "measurements database -- even the one paired_data carries is "
           "unread -- so requiring src/measurements/measurements.db refuses "
           "a run that would have worked.",
}

#: Control identifiers are gene names that happen to be all digits. Loading a
#: settings CSV parses them back as ints, which is not what spaCR declares
#: them to be and not what the score tables hold.
STRING_VALUED = ("positive_control", "negative_control")


def paired_data(work):
    """The score/count pairs, pointed at the scratch copies."""
    return [{"score": str(work / f"plate{n}_dv.csv"),
             "count": str(work / f"plate_{n}_unique_combinations.csv"),
             "plate": f"plate{n}",
             "database": None} for n in PLATES]


def summarise(work):
    """Report what the run wrote, so a silent no-op cannot look like a result."""
    written = sorted(p for p in Path(work).rglob("*")
                     if p.is_file() and p.suffix in (".csv", ".pdf", ".png"))
    print(f"\n{len(written)} result files written under {work}")
    for path in written[:20]:
        print(f"  {path.relative_to(work)}")
    return bool(written)


def gene_table(work):
    """The fitted gene-level table and the features the run called, or None.

    :returns: ``(gene_rows, called_features)``, or ``None`` when the run did
        not write a gene table at all.
    """
    import pandas as pd

    tables = sorted(Path(work).rglob("results_gene.csv"))
    if not tables:
        return None
    genes = pd.read_csv(tables[0])
    called = set()
    for path in sorted(Path(work).rglob("results_significant.csv")):
        called |= set(pd.read_csv(path)["feature"].astype(str))
    return genes, called


def control_row(genes, control):
    """The gene-level row for one control identifier, or None.

    Features are spelled ``gene_fraction:gene[239740]``, so the control is
    matched inside the feature name rather than against a bare column.
    """
    if control is None:
        return None
    hit = genes[genes["feature"].astype(str).str.contains(str(control),
                                                          regex=False)]
    return None if hit.empty else hit.iloc[0]


def main(argv):
    """Stage the screen's score and count tables and fit its own settings."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the tsg101 screen")
    print(f"dataset root: {root}")
    recorded = (Path(argv[2]).expanduser() if len(argv) > 2 and argv[2]
                else settings_file(root, SETTINGS_CANDIDATES,
                                   what="the regression run"))
    print(f"settings:     {recorded}")

    work = scratch("regression_on_tsg101")
    stage(root, [name for name in REQUIRED if name.endswith(".csv")
                 and "/" not in name], work)

    settings = read_settings(recorded)
    print(f"permutations: {PERMUTATIONS} (the recorded run used "
          f"{settings.get('guide_permutations')})")
    settings["src"] = str(work)
    settings["paired_data"] = paired_data(work)
    settings["guide_permutations"] = PERMUTATIONS
    if len(argv) > 3:
        settings["regression_type"] = argv[3]
    print(f"regression_type: {settings.get('regression_type')}"
          + ("  (the long path; name 'ols' as the third argument for the "
             "quick one)" if settings.get("regression_type") == "mixed" else ""))
    for key in STRING_VALUED:
        if not isinstance(settings.get(key), str) and settings.get(key) is not None:
            print(f"{key} came back from the settings file as "
                  f"{type(settings[key]).__name__} {settings[key]!r}; the "
                  f"score tables hold it as text, so it is restored to a "
                  f"string here")
            settings[key] = str(settings[key])
    preflight(settings, "regression", KNOWN_FALSE_POSITIVES)

    import matplotlib

    matplotlib.use("Agg")
    from spacr.ml import perform_regression

    perform_regression(settings)
    summarise(work)
    check_the_controls(work, settings)


def check_the_controls(work, settings):
    """Read the gene table back and say where the screen's controls landed.

    :raises WrongAnswer: when the positive control never reached the fit, or
        reached it and came out un-called or on the wrong side of zero.
    """
    tables = gene_table(work)
    check(tables is not None,
          "the run wrote no results_gene.csv, so there is no fitted gene "
          "table to check the controls against")
    genes, called = tables
    check(len(genes) > 0, "results_gene.csv is empty: the fit produced no "
                          "gene-level coefficients at all")
    print(f"\ngene-level rows fitted: {len(genes)}; features called: "
          f"{len(called)}")

    positive = settings.get("positive_control")
    row = control_row(genes, positive)
    check(row is not None,
          f"the positive control {positive!r} is named in the screen's own "
          f"settings but no row of the {len(genes)}-row gene table carries "
          f"it. A control the fit never saw is a join that lost it, not a "
          f"result.")
    print(f"positive control {positive}: coefficient {row['coefficient']:+.3f}"
          f", q={row['q_value']:.3g}, "
          f"{'called' if row['feature'] in called else 'NOT called'}")
    check(row["coefficient"] > 0,
          f"the positive control {positive} came out at "
          f"{row['coefficient']:+.3f}. A positive control on the negative "
          f"side of zero is the screen read backwards, which is the one "
          f"failure a table of coefficients cannot show by existing.")
    check(row["feature"] in called,
          f"the positive control {positive} was fitted at "
          f"{row['coefficient']:+.3f} (q={row['q_value']:.3g}) and was not "
          f"called. If the strongest thing in the screen is not a hit, "
          f"nothing else in the hit list means anything.")

    negative = settings.get("negative_control")
    quiet = control_row(genes, negative)
    if quiet is None:
        print(f"negative control {negative}: not a row of the gene table for "
              f"this screen, so there is nothing to check it against")
    else:
        print(f"negative control {negative}: coefficient "
              f"{quiet['coefficient']:+.3f}, "
              f"{'CALLED' if quiet['feature'] in called else 'not called'}")
        check(quiet["feature"] not in called,
              f"the negative control {negative} was called a hit at "
              f"{quiet['coefficient']:+.3f}; a screen that calls its own "
              f"negative control is calling noise")


if __name__ == "__main__":
    run(main)
