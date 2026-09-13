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
* The Debian package is ``spacr`` and cannot be anything else: Debian policy
  5.6.1 restricts a package name to lower case, and ``dpkg`` refuses the
  rest. Its human-facing fields carry the real name instead.

A FOURTH EXEMPTION IS GONE, and with it the test that guarded it. Tk style
names were registration strings of the form ``Spacr.TEntry`` --
capital-S-lower-pacr, matched by exact text between ``style.configure`` and
``style=`` -- so the pattern below had to let a capitalised ``Spacr``
through whenever a dot followed it. Tk is deleted, no such string exists
anywhere in the tree, and the pattern is correspondingly stricter: a
capitalised ``Spacr`` is now a mis-spelling wherever it appears.
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

#: A mis-cased mention: `SpaCR`, `Spacr`, or `SPACR` not starting an
#: environment variable.
WRONG = re.compile(r"\bSpaCR\b|\bSpacr\b|\bSPACR\b(?!_)")

#: Published release assets are named as they were named. README's download
#: links point at files that exist on GitHub under v1.5.0.4, and renaming
#: them in the text makes the front page 404. The URL keeps the published
#: spelling until a release actually builds artifacts under the corrected
#: name -- the packaging scripts already emit `spaCR-<version>-...`, so the
#: next tag closes this on its own. Do not "fix" these lines ahead of that
#: release; the front page 404s the moment they stop matching GitHub.
ALLOWED_LINES = (
    "releases/download/",
)

#: The same published assets, named WITHOUT their URL. The installer guide
#: tells a user to run `SpaCR-<version>-Windows-Online-Setup.exe` and to
#: `chmod +x SpaCR-*-Linux-x86_64-Online.run` -- those are the filenames
#: GitHub actually serves, and correcting the spelling in the instructions
#: would tell people to run a file that does not exist. It is the same
#: exception as the download links above, reached from the other side.
#:
#: DELIBERATELY NARROW: it matches an installer ASSET name, not the word.
#: `SpaCR is a tool for...` in the same file is still a failure.
#: A PUBLISHED RELEASE TITLE, which is somebody else's text. GitHub serves
#: v1.5.0.4 under the title "SpaCR 1.5.0.4" -- mis-cased at the time and
#: unchangeable now -- while v1.5.0.5 and v1.5.0.6 are correctly "spaCR".
#: `capture_install_web.py` waits for the release page to CONTAIN its title,
#: so the spelling there has to be the spelling GitHub renders; correcting it
#: makes the wait time out on a page that is fine.
#:
#: Same reasoning as the asset URLs above, and it expires the same way: the
#: pin is `RELEASE_VERSION = "1.5.0.4"`, and the moment it moves to a release
#: titled correctly the exemption stops matching anything.
RELEASE_TITLE = re.compile(
    r"\bSpaCR (?:\{RELEASE_VERSION\}|\d+(?:\.\d+)+)")

ASSET = re.compile(
    r"\bSpaCR-[\w.*<>-]+-(?:Windows|macOS|Linux)[\w.*-]*"
    r"\.(?:exe|pkg|run|dmg|deb|zip|tar\.gz)")

#: The first 1.5.0.x installers created these case-sensitive directories and
#: one old QSettings namespace. Their recovery and migration code must keep
#: those bytes even though ordinary prose must spell the project ``spaCR``.
#: Match only the literal identifiers so ``SpaCR`` elsewhere on the same line
#: still fails. ``\\+`` also recognizes doubled backslashes inside Python
#: string literals without widening the Windows path exemption.
LEGACY_IDENTIFIERS = re.compile(
    r"Application Support/SpaCR/"
    r"|\$env:LOCALAPPDATA\\+SpaCR\\+"
    r'|"Olafsson Lab", "SpaCR"'
)

#: THE TRANSLATION PIPELINE'S PROTECTED PLACEHOLDERS, which are all-caps by
#: construction and must stay that way.
#:
#: `tools/tutorials/authoring/tools/translate_pre_app.py` wraps every term
#: that must survive a sentence model as an upper-case token, and then has to
#: put the real word back afterwards:
#:
#:     # NLLB occasionally rewrites the middle of the spaCR placeholder as a
#:     # natural-language conjunction (for example SPACRANDTOKEN in German).
#:     # The stable SPACR[A-Z]*TOKEN shape still identifies it uniquely.
#:     text = re.sub(r"SPACR[A-Z]*TOKEN", "spaCR", text)
#:
#: THE `SPACR` THERE IS A PATTERN, NOT A MENTION, and the replacement string
#: on the same line already spells the project correctly. "Fixing" the
#: spelling breaks the match, the placeholder is never restored, and every
#: affected translation ships with a raw token in it -- so this rule would
#: have caused the defect it exists to prevent.
#:
#: THE COMMENT ABOVE THE REGEX HAS TO QUOTE THE SHAPE, not describe it in
#: words. An earlier wording said "the stable SPACR prefix and TOKEN suffix",
#: which splits the token across two English words; nothing spans them, so the
#: bare `SPACR` was a mention again and this rule failed on a comment that was
#: right about everything except how it spelled itself. That wording was
#: replaced by the quote above on 2026-09-11, came back on 2026-09-12 when the
#: tutorial narration commit rewrote the file from an older copy, and is the
#: only thing this rule has ever caught in that file. If it fails there again,
#: read the comment before the code -- the regex is almost certainly fine.
#:
#: Matched only in the shape those placeholders actually take: all capitals,
#: optionally carrying a regex class or group, ending in TOKEN. Ordinary
#: prose cannot collide with it.
PLACEHOLDER_TOKEN = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:\[[^\]]*\]\*?|\(\?:[^)]*\))?[A-Z0-9]*TOKEN\b")


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
        # DOES THIS FILE DEFINE `SPACR` AS A CONSTANT? Computed once per
        # file, because the exemption belongs to the file that assigns it and
        # not to every file that happens to write the word.
        defines_a_constant = bool(
            path.suffix == ".py"
            and re.search(r"^SPACR\s*[:=]", text, re.M))
        for number, line in enumerate(text.splitlines(), 1):
            # The asset name is removed rather than the LINE being skipped,
            # so a line that names an installer AND mis-cases the project
            # still fails on the second one.
            line = ASSET.sub("", line)
            line = RELEASE_TITLE.sub("", line)
            line = LEGACY_IDENTIFIERS.sub("", line)
            line = PLACEHOLDER_TOKEN.sub("", line)
            if defines_a_constant and "SPACR" in line:
                # `SPACR` IS AN IDENTIFIER IN THIS FILE, not a mention of the
                # project. Several tests do `SPACR = <path to the package>`
                # and then walk it -- an ALL-CAPS module constant, which is
                # what Python spells a constant with. The rule already allows
                # `SPACR_` for an environment variable; this is the same
                # thing one line further on.
                #
                # PER FILE, and only when the file ASSIGNS it: a file that
                # merely writes SPACR in prose gets no exemption from one
                # that does.
                line = line.replace("SPACR", "")
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
