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

from _support import (dataset_root, preflight, read_settings, require, run,
                      scratch, settings_file, stage)

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


if __name__ == "__main__":
    run(main)
