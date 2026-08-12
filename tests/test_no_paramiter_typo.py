"""'paramiter' is not a word (issue #21).

The reported spelling was `hyperparamiters`, which is gone. `paramiter`
survived it in three places, two of them printed to the user during a run:

    core.py   print('Testing paramiters:', ...)
    core.py   print('Testing clustering paramiters:', ...)
    sim.py    def generate_paramiters(settings)
    cli.py    a --help string naming that function

The function could not simply be renamed -- it is public and imported by
tests and by anything a user wrote against it -- so `generate_paramiters`
remains as an alias for `generate_parameters`. That alias is the ONLY
occurrence of the misspelling this test allows.
"""

import pathlib
import re

import spacr


PACKAGE = pathlib.Path(spacr.__file__).parent

#: The alias line and the comment explaining it. Nothing else may match.
ALLOWED = re.compile(r"generate_paramiters = generate_parameters|"
                     r"^\s*#[:#]?.*generate_paramiters")


def _offenders():
    found = []
    for path in PACKAGE.rglob("*.py"):
        # The i18n catalogs are GENERATED from the tooltip tables; fixing a
        # spelling there means fixing its source, not the catalog.
        if "i18n_catalogs" in path.parts:
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "paramiter" in line.lower() and not ALLOWED.search(line):
                found.append(f"{path.relative_to(PACKAGE)}:{number}: {line.strip()}")
    return found


def test_the_misspelling_survives_only_as_the_alias():
    offenders = _offenders()
    assert not offenders, (
        "'paramiter' is not a word; the only permitted occurrence is the "
        "back-compat alias:\n  " + "\n  ".join(offenders))


def test_the_alias_still_points_at_the_real_function():
    """Removing it would break anything that imported the old spelling."""
    from spacr.sim import generate_parameters, generate_paramiters

    assert generate_paramiters is generate_parameters
    assert callable(generate_parameters)


def test_the_scan_actually_reads_the_package():
    """A scan that walked nothing would pass the test above."""
    files = [p for p in PACKAGE.rglob("*.py")
             if "i18n_catalogs" not in p.parts]
    assert len(files) > 50, f"only found {len(files)} modules to scan"
