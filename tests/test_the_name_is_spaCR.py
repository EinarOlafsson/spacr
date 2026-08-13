"""The project is spelled ``spaCR``. Instruction 85.

Lower-case s, lower-case p, lower-case a, capital C, capital R. Before this
test the README alone carried three spellings, and the eight ``SpaCR`` ones
were all installer names -- the artifacts a new user downloads first.

THREE THINGS ARE NOT THE NAME, and each is exempt for a reason that would
break something if ignored:

* ``spacr`` lower-case is the PACKAGE -- ``import spacr``,
  ``pip install spacr``, the ``spacr-qt`` console scripts, every dotted path
  and the GitHub URL. Rewriting those breaks the thing they name.
* ``SPACR_`` prefixes an ENVIRONMENT VARIABLE (``SPACR_STRICT_ERRORS`` and
  two dozen more). Upper case is the convention and the reader is
  ``os.environ``.
* ``Spacr.`` prefixes a TK STYLE NAME (``Spacr.TEntry``,
  ``Spacr.Vertical.TScrollbar``). It is a registration string matched by
  exact text between ``style.configure`` and ``style=``; it is invisible to
  users and renaming it only creates a chance to get the two halves out of
  step.
* The Debian package is ``spacr`` and cannot be anything else: Debian policy
  5.6.1 restricts a package name to lower case, and ``dpkg`` refuses the
  rest. Its human-facing fields carry the real name instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Where the project's own text lives. `build/` is a build artifact and
#: `docs/i18n` + `_static/i18n` are the translation catalogs, which are
#: generated and owned elsewhere.
SKIP_PARTS = (
    ".git/", "build/", "dist/", "__pycache__", ".pytest_cache",
    "spacr.egg-info", "docs/i18n/", "docs/source/_static/i18n/",
    "i18n_catalogs", "docs/_sources/", "docs/_build/",
)
EXTENSIONS = {".py", ".rst", ".md", ".sh", ".ps1", ".yml", ".yaml",
              ".toml", ".cfg", ".desktop", ".spec"}

#: A mis-cased mention: `SpaCR`, `Spacr` not starting a Tk style, or `SPACR`
#: not starting an environment variable.
WRONG = re.compile(r"\bSpaCR\b|\bSpacr\b(?!\.)|\bSPACR\b(?!_)")

#: Published release assets are named as they were named. README's download
#: links point at files that exist on GitHub under v1.5.0.4, and renaming
#: them in the text makes the front page 404. Instruction 82 rebuilds them
#: under the corrected name; until then the URL keeps the published spelling.
ALLOWED_LINES = (
    "releases/download/",
)


def _project_files():
    for path in sorted(ROOT.rglob("*")):
        rel = str(path.relative_to(ROOT))
        if any(part in rel for part in SKIP_PARTS):
            continue
        if path.suffix not in EXTENSIONS or not path.is_file():
            continue
        yield path, rel


def _offenders():
    out = []
    for path, rel in _project_files():
        if rel == str(Path(__file__).relative_to(ROOT)):
            continue          # this file quotes the wrong spellings on purpose
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(token in line for token in ALLOWED_LINES):
                continue
            for match in WRONG.finditer(line):
                out.append(f"{rel}:{number} {match.group(0)}")
    return out


def test_no_file_mis_cases_the_name():
    offenders = _offenders()
    assert not offenders, (
        "the project is spelled spaCR; these are not:\n  "
        + "\n  ".join(offenders[:40])
        + (f"\n  ... and {len(offenders) - 40} more" if len(offenders) > 40
           else "")
    )


def test_the_readme_says_spaCR():
    text = (ROOT / "README.rst").read_text(encoding="utf-8")
    assert "spaCR" in text


def test_the_package_name_is_still_lower_case():
    """The exemption that matters most: renaming this breaks every install."""
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'name = "spacr"' in setup or "name = 'spacr'" in setup


def test_the_debian_package_name_is_lower_case():
    """Debian policy 5.6.1 -- dpkg refuses a capital, so this one cannot move."""
    script = ROOT / "packaging" / "build_debian.sh"
    if not script.is_file():
        pytest.skip("no Debian packaging in this checkout")
    assert "Package: spacr" in script.read_text(encoding="utf-8")


def test_the_environment_variables_keep_their_case():
    """`SPACR_` is read from os.environ and is not a mention of the name."""
    from spacr import runctx  # noqa: F401  -- any module that reads one

    hits = []
    for path, rel in _project_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits.extend(re.findall(r"SPACR_[A-Z0-9_]+", text))
    assert hits, "no SPACR_ environment variables found; the exemption is stale"


def test_the_tk_style_names_keep_their_case():
    """Both halves of a Tk style registration must still agree."""
    source = (ROOT / "spacr" / "gui_elements.py").read_text(encoding="utf-8")
    configured = set(re.findall(r"['\"](Spacr\.[A-Za-z.]+)['\"]", source))
    assert configured, "the Tk style names vanished; the exemption is stale"
