"""The README's lightweight-install numbers describe the tree it has now.

Instruction 328. The README offers a full clone and a light one, and the
only reason to read the light one is the number beside it. A number in
prose is a claim about a tree that moves, and this one had already moved:
the README said

    Full clone: 427 MB. Core clone: 76 MB.

when the checkout had grown to ~957 MB, and it was quoting the size of the
CHECKOUT in a place where a reader choosing between two commands wants the
size of the DOWNLOAD. Both halves were wrong in a way nothing could notice,
because nothing read them.

Re-measured on 2026-09-14 against ``main`` on the real remote, with
``packaging/measure_clone_forms.sh``:

    form             seconds    downloaded      .git    worktree
    full                 131       5.78 GiB  5927 MiB    941 MiB
    depth1                31        595 MiB   596 MiB    941 MiB
    depth1-filter         59        575 MiB*  575 MiB    941 MiB
    light                  5             --    31 MiB     55 MiB

    * two fetches: 263 KiB of trees, then the checkout goes back for
      every blob it needs. The filter is SLOWER than plain --depth 1 and
      saves nothing, which is the opposite of what instruction 328
      predicted and is why the README now says so.

So these tests pin the parts a reader acts on: that the size quoted for the
checkout still matches the checkout, that anything large the light install
drops is named rather than discovered later, that the limitations of a
shallow clone are stated, and that the helper which re-measures all of it
still prints the commands it claims to run.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
README = REPO / "README.rst"
EXCLUDES = REPO / "packaging" / "source_install_excludes.txt"
HELPER = REPO / "packaging" / "measure_clone_forms.sh"

SECTION = "Install from source (light)"

# A single exclusion worth more than this has to be named in the README.
# docs/ (419 MiB) and tools/ (376 MiB) are over it today; tests/ (35 MiB)
# and the translation catalogs (17 MiB) are not, and are named anyway.
LARGE = 50 * 1024 * 1024


def _section(title: str = SECTION) -> str:
    """The body of one README section, without the next section's heading.

    Slicing matters. `docs`, `tests` and `git pull` all appear elsewhere in
    the README, so asserting them against the whole file would pass no
    matter what this section said.
    """
    text = README.read_text(encoding="utf-8")
    start = text.index(title + "\n")
    body = text[start + len(title) + 1:]
    body = body[body.index("\n") + 1:]          # drop the ~~~ underline
    nxt = re.search(r"(?m)^[~\-=]{3,}$", body)
    if nxt:
        body = body[:body.rfind("\n", 0, nxt.start() - 1)]
    return body


def _tracked_bytes() -> int:
    """Apparent size of every tracked file, which is what a checkout costs.

    `du` would answer in disk blocks and inflate an 8556-file tree by ~2%;
    the number in the README came from a fresh clone, so this has to be the
    same kind of number.
    """
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                         capture_output=True)
    total = 0
    for name in out.stdout.split(b"\0"):
        if not name:
            continue
        try:
            total += (REPO / os.fsdecode(name)).stat().st_size
        except OSError:
            pass
    return total


def _exclusions():
    """Each excluded path from the list, with what it weighs today."""
    lines = [ln.strip() for ln in EXCLUDES.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                         capture_output=True)
    names = [os.fsdecode(n) for n in out.stdout.split(b"\0") if n]
    sizes = {}
    for n in names:
        try:
            sizes[n] = (REPO / n).stat().st_size
        except OSError:
            pass
    for line in lines[1:]:                      # lines[0] is the `/*`
        prefix = line.lstrip("!").lstrip("/")
        yield line, sum(v for k, v in sizes.items() if k.startswith(prefix))


def _git_checkout_or_skip():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")


def test_the_checkout_size_in_the_readme_matches_the_tree():
    """THE REGRESSION: 427 MB against a 957 MB tree, for a fortnight.

    Nothing reads a number in prose, so nothing noticed. This does. The
    figure is measured against `main` and the tests run on whatever branch
    is checked out, so the tolerance is wide enough for that drift (1.8%
    on 2026-09-14) and nowhere near wide enough for the failure it exists
    to catch: 427 against 957 is 55% out.
    """
    _git_checkout_or_skip()
    section = _section()
    stated = re.search(r"([\d.]+)\s*MB checkout", section)
    assert stated, (
        "the README no longer says how big the checkout is; that number is "
        "the only reason to read the light install at all")
    claimed = float(stated.group(1)) * 1024 * 1024
    actual = _tracked_bytes()
    assert actual > 0, "no tracked files; the measurement is meaningless"
    drift = abs(claimed - actual) / actual
    assert drift < 0.15, (
        f"the README says the checkout is {claimed / 1048576:.0f} MB; the "
        f"tracked tree is {actual / 1048576:.0f} MB ({drift:.0%} out). "
        f"Re-measure with packaging/measure_clone_forms.sh and update the "
        f"README, including the date beside it.")


def test_the_readme_names_every_large_thing_the_light_install_drops():
    """A user finds out what is missing HERE, or later and the hard way.

    `tools/` is the case that made this worth a test: it holds the tutorial
    masters, it grew to 376 MiB, the exclusion list drops it and the README
    described the light install without ever mentioning it.
    """
    _git_checkout_or_skip()
    section = _section().lower()
    for line, size in _exclusions():
        if size < LARGE:
            continue
        segments = line.lstrip("!").strip("/").split("/")
        names = {segments[0].lower(), segments[-1].lower()}
        assert any(name in section for name in names), (
            f"the light install drops {line} ({size / 1048576:.0f} MiB) and "
            f"the README's light section never says so")


def test_the_readme_offers_a_git_command_and_not_only_a_script():
    """Two lines of git, for a reader who will not pipe a script into sh.

    Instruction 328 asked for exactly this form, and until now the section
    had only the curl'd installer.
    """
    section = _section()
    before_script = section.split("curl")[0]
    assert "git clone --depth 1 " in before_script, (
        "the light section no longer shows a plain shallow clone")
    assert "github.com/EinarOlafsson/spacr.git" in before_script, (
        "the shallow clone does not name the repository to clone")
    assert "pip install -e ." in before_script, (
        "the shallow clone stops short of installing anything")


def test_the_readme_says_what_a_shallow_clone_cannot_do():
    """The part a bare command leaves out and a user discovers later."""
    section = _section()
    for missing in ("git log", "git blame", "git bisect"):
        assert missing in section, (
            f"a shallow clone has no {missing} and the README does not say so")
    assert "git pull" in section, "a shallow clone's pull stays shallow"
    assert "--unshallow" in section, (
        "the README does not say how to convert a shallow clone back")


def test_the_readme_says_which_form_a_contributor_takes():
    """Two commands and no way to choose is not a choice."""
    section = _section()
    assert re.search(r"[Cc]ontributor", section), (
        "the README does not say who should take the full clone")
    assert re.search(r"histor(y|ies)", section), (
        "the README does not say what the full clone buys: the history")


def test_the_readme_does_not_sell_a_filter_that_saved_nothing():
    """Measured, and it contradicts the instruction that asked for it.

    `--depth 1 --filter=blob:none` took 59 s against 31 s for `--depth 1`
    alone and downloaded the same bytes: a non-bare clone checks out HEAD,
    and checking out HEAD fetches every blob the filter declined. Offering
    it as the lightweight form would cost a reader time and save nothing.
    """
    section = _section()
    for line in section.splitlines():
        if line.strip().lstrip("# ").startswith("git clone"):
            assert "--filter=blob:none" not in line, (
                "the README recommends --filter=blob:none; measured, it is "
                "slower than --depth 1 alone and downloads the same bytes")
    assert "--filter=blob:none" in section, (
        "the README should say why the obvious flag is not used")
    assert re.search(r"saves nothing|does not help|no help", section), (
        "--filter=blob:none is named without saying it does not help")


def test_the_readme_dates_its_numbers_and_names_what_remeasures_them():
    """A number without a date cannot be told from a number that is stale."""
    section = _section()
    assert re.search(r"20\d\d-\d\d-\d\d", section), (
        "the light install's numbers carry no date")
    assert "measure_clone_forms.sh" in section, (
        "nothing in the README says how to re-measure the numbers")
    assert HELPER.is_file(), "packaging/measure_clone_forms.sh is missing"
    assert os.access(HELPER, os.X_OK), "the helper is not executable"


def test_the_helper_prints_the_command_each_form_would_run():
    """--dry-run is the part of the helper a test can run: no network.

    It prints the very string the real run executes, so a form whose flags
    drifted -- a `depth1` that quietly grew a filter, a `full` that quietly
    became shallow and stopped measuring the thing it is named for -- shows
    up here.
    """
    run = subprocess.run(["sh", str(HELPER), "--dry-run"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    printed = {}
    for line in run.stdout.splitlines():
        if line.strip():
            printed[line.split()[0]] = line
    assert set(printed) == {"full", "depth1", "depth1-filter", "light"}, printed

    assert "--depth" not in printed["full"], (
        "the `full` form is shallow; it measures nothing")
    assert "--filter" not in printed["full"]

    assert "--depth 1" in printed["depth1"]
    assert "--filter" not in printed["depth1"], (
        "`depth1` and `depth1-filter` must differ by the filter alone, or "
        "the comparison between them says nothing")

    assert "--depth 1" in printed["depth1-filter"]
    assert "--filter=blob:none" in printed["depth1-filter"]

    assert "install_from_source.sh" in printed["light"], (
        "the `light` form must measure the installer the README points at")
    assert "--no-install" in printed["light"], (
        "the measurement must not pip install into the caller's environment")


def test_the_helper_refuses_to_call_a_local_clone_a_download():
    """Git hardlinks a clone of a PATH, which is not a transfer at all.

    Measured while writing this: `--depth 1` against a local path produced
    a 6.5 GB .git and downloaded nothing. A number from that run would be
    nonsense, so the helper says so before printing one.
    """
    local = subprocess.run(["sh", str(HELPER), "--dry-run",
                            "--repo", "/some/local/path"],
                           capture_output=True, text=True, timeout=60)
    assert local.returncode == 0, local.stderr
    assert "local path" in local.stderr, (
        "the helper measured a local clone without warning that git "
        "hardlinks one")

    remote = subprocess.run(["sh", str(HELPER), "--dry-run",
                             "--repo", "https://example.com/spacr.git"],
                            capture_output=True, text=True, timeout=60)
    assert remote.returncode == 0, remote.stderr
    assert "local path" not in remote.stderr, (
        "the helper warns about a URL, so the warning means nothing")


@pytest.mark.slow
def test_the_light_checkout_launches_spacr_and_not_merely_imports_it():
    """Instruction 328's own condition: LAUNCH the result, do not watch pip.

    An install missing a resource does not fail at install time. It fails
    the first time somebody opens the window, which is the worst place for
    it to fail, so the thing that has to be proved is that the window
    opens -- offscreen, out of the fetched tree, with nothing installed.

    `tests/test_the_lightweight_source_install_still_runs.py` already
    proves the tree IMPORTS. This builds and shows the main window out of
    it. Measured: the 57 MB checkout draws its window in 1.1 s.

    MEASURED AND WORTH SAYING, because it bounds what a launch proves: the
    window still comes up with `spacr/resources/` excluded ENTIRELY. The
    resources load later, or not at all, so launching is not evidence that
    the install kept them -- which is why the manifest check below is here
    as well, and why it is not folded into the launch.
    """
    _git_checkout_or_skip()
    pytest.importorskip("PySide6")
    import sys
    import tempfile
    from tests.child_env import child_env

    script = REPO / "packaging" / "install_from_source.sh"
    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if not branch or branch == "HEAD":
        pytest.skip("detached HEAD; no branch to fetch")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out")
        fetched = subprocess.run(
            ["sh", str(script), "--repo", str(REPO), "--branch", branch,
             "--dir", out, "--no-install"],
            capture_output=True, text=True, timeout=900)
        if fetched.returncode != 0:
            pytest.skip(f"fetch failed here: {fetched.stderr[-300:]}")

        # Every resource directory the WHEEL ships has to survive the
        # sparse checkout, because a source install is supposed to be at
        # least as complete as a pip install. MANIFEST.in is the list of
        # what ships; its own `exclude`/`prune` lines take things back out
        # (the Cellpose checkpoints under models/cp are excluded there too,
        # which is what makes dropping them from the source install safe).
        manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
        ships, dropped = [], []
        for line in manifest.splitlines():
            found = re.match(r"recursive-include\s+(spacr/resources/\S+)", line)
            if found:
                ships.append(found.group(1))
            gone = re.match(r"(?:exclude|prune)\s+(spacr/resources/\S+)", line)
            if gone:
                dropped.append(gone.group(1).rstrip("*").rstrip("/"))
        assert ships, "MANIFEST.in ships no resources; this check is dead"
        kept = [d for d in ships
                if not any(d == x or d.startswith(x + "/") for x in dropped)]
        assert kept, "every shipped resource directory is excluded elsewhere"
        for directory in kept:
            assert (pathlib.Path(out) / directory).is_dir(), (
                f"the wheel ships {directory} and the light install does "
                f"not have it; spaCR will fail somewhere later instead")

        # A PEP 660 editable install answers `import spacr` through a
        # MetaPathFinder, which `import` consults BEFORE sys.path -- so a
        # bare sys.path.insert would launch the DEVELOPMENT checkout and
        # call it proof. The finder has to go, and the window's own module
        # has to be shown to have come out of the fetched tree.
        probe = (
            "import os, sys\n"
            "def _mod(f):\n"
            "    return getattr(f, '__module__', None) or type(f).__module__\n"
            "sys.meta_path = [f for f in sys.meta_path\n"
            "                 if '__editable__' not in _mod(f)]\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import spacr\n"
            "assert spacr.__file__.startswith(sys.argv[1]), spacr.__file__\n"
            "from PySide6.QtWidgets import QApplication\n"
            "from spacr.qt.app import MainWindow\n"
            "app = QApplication.instance() or QApplication([])\n"
            "window = MainWindow()\n"
            "window.show()\n"
            "app.processEvents()\n"
            "assert window.isVisible()\n"
            "print('LAUNCHED', sys.modules['spacr.qt.app'].__file__)\n"
            "window.close()\n"
        )
        launched = subprocess.run(
            [sys.executable, "-c", probe, out],
            capture_output=True, text=True, timeout=900,
            env=child_env(home=os.path.join(tmp, "home"), qt=True),
        )
        assert "LAUNCHED" in launched.stdout, (
            "the lightweight checkout does not get as far as a window:\n"
            + launched.stderr[-3000:])
        assert out in launched.stdout, (
            "the window came from outside the fetched tree, so the light "
            f"install proved nothing: {launched.stdout}")


INSTALLER = REPO / "packaging" / "install_from_source.sh"


def _listed_exclusions():
    """The exclusion file as git reads it: no comments, no blank lines."""
    return [line.strip()
            for line in EXCLUDES.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _show_exclusions(argv0, cwd=None):
    """Ask the installer which list it would use, and for what."""
    run = subprocess.run(["sh", str(argv0), "--show-exclusions"],
                         capture_output=True, text=True, timeout=60,
                         cwd=str(cwd) if cwd else None)
    assert run.returncode == 0, run.stderr
    first, _, rest = run.stdout.partition("\n")
    assert first.startswith("exclusions read from: "), run.stdout
    return first[len("exclusions read from: "):], rest.split()


@pytest.mark.slow
def test_an_edited_exclusion_list_reaches_the_checkout(tmp_path):
    """THE REGRESSION, and it made editing the exclusion list a no-op.

    `read_the_exclusions` resolved `dirname "$0"` AFTER the script had
    `cd`-ed into the new checkout. For the ordinary invocation --

        sh packaging/install_from_source.sh

    -- `$0` is relative, so the dirname resolved against the CHECKOUT,
    `packaging/` was not there, and the script silently fell back to its
    embedded copy of the list. Editing
    packaging/source_install_excludes.txt and running the script the
    obvious way changed nothing, and nothing said so. An absolute `$0`
    happened to work, which is why the end-to-end test in
    test_the_lightweight_source_install_still_runs.py -- which builds its
    path with `str(SCRIPT)` -- never saw it.

    Checking this needs the file and the embedded copy to DISAGREE, and
    they are pinned identical in the repository (rightly: a standalone
    installer carries its own list). So the pair is copied out to a
    temporary directory, one extra rule is added to the copy of the file,
    and the fetch is run from there by a relative path. The rule has to
    reach the clone's sparse-checkout, and the directory it names has to
    be missing from the tree.
    """
    _git_checkout_or_skip()
    marker = "!/spacr/resources/themes/"
    assert (REPO / "spacr/resources/themes").is_dir(), (
        "the marker directory is gone; pick another one")
    assert marker not in _listed_exclusions(), (
        "the marker is already excluded, so its absence proves nothing")

    (tmp_path / "install_from_source.sh").write_bytes(INSTALLER.read_bytes())
    (tmp_path / "source_install_excludes.txt").write_text(
        EXCLUDES.read_text(encoding="utf-8") + f"\n{marker}\n",
        encoding="utf-8")

    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if not branch or branch == "HEAD":
        pytest.skip("detached HEAD; no branch to fetch")

    out = tmp_path / "out"
    run = subprocess.run(
        ["sh", "./install_from_source.sh", "--repo", str(REPO),
         "--branch", branch, "--dir", str(out), "--no-install"],
        capture_output=True, text=True, timeout=900, cwd=str(tmp_path))
    if run.returncode != 0:
        pytest.skip(f"fetch failed here: {run.stderr[-300:]}")

    written = (out / ".git" / "info" / "sparse-checkout").read_text()
    assert marker in written.split(), (
        "the edited exclusion list never reached the checkout; the script "
        "used its embedded copy instead:\n" + written)
    assert not (out / "spacr" / "resources" / "themes").exists(), (
        "the rule was written and git ignored it, which is a different "
        "bug but the same broken promise")


def test_the_installer_on_its_own_still_knows_the_list(tmp_path):
    """The curl'd case: one file, no checkout around it, no second fetch."""
    alone = tmp_path / "install_spacr.sh"
    alone.write_bytes(INSTALLER.read_bytes())
    source, rules = _show_exclusions(alone)
    assert "embedded" in source, (
        f"a standalone copy claims to read a file that is not there: {source}")
    assert rules == _listed_exclusions(), (
        "the embedded copy and the file disagree, so the standalone "
        "installer would exclude different paths from the checkout one")


def test_asking_what_would_be_excluded_creates_nothing(tmp_path):
    """A diagnostic that leaves a ./spacr behind is not a diagnostic."""
    before = set(os.listdir(tmp_path))
    run = subprocess.run(["sh", str(INSTALLER), "--show-exclusions"],
                         capture_output=True, text=True, timeout=60,
                         cwd=str(tmp_path))
    assert run.returncode == 0, run.stderr
    assert set(os.listdir(tmp_path)) == before, (
        "--show-exclusions created something in the working directory")


@pytest.mark.parametrize("script,must_have", [
    pytest.param(INSTALLER,
                 {"--dir", "--branch", "--no-install", "--show-exclusions"},
                 id="install_from_source"),
    pytest.param(HELPER,
                 {"--repo", "--branch", "--forms", "--dry-run"},
                 id="measure_clone_forms"),
])
def test_the_scripts_help_lists_every_flag_and_no_shell(script, must_have):
    """`usage` prints a HAND-COUNTED line range, and both ends of it bite.

    Too short and a working flag is undocumented: install_from_source.sh
    ended one line above `--no-install` and `--help`, and
    measure_clone_forms.sh shipped with a range that stopped at `--keep`,
    so `--dry-run` -- the only flag a test can exercise -- was invisible.
    Too long and the range runs off the end of the comment header into the
    script: install_from_source.sh printed `set -eu` as though it were help.

    So this checks both ends. The flags come from the `case` arms, which is
    the only place that decides what the script really accepts.
    """
    body = script.read_text(encoding="utf-8")
    flags = set(re.findall(r"^\s+(--[a-z-]+)\)", body, re.M))
    flags |= set(re.findall(r"^\s+-h\|(--[a-z-]+)\)", body, re.M))
    assert must_have <= flags, flags
    run = subprocess.run(["sh", str(script), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    for flag in sorted(flags):
        assert flag in run.stdout, (
            f"{flag} works but {script.name} --help never mentions it")

    code = {line for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")}
    leaked = [line for line in run.stdout.splitlines() if line in code]
    assert not leaked, (
        f"{script.name} --help ran off the end of its comment header and "
        f"printed the script itself: {leaked}")


INSTALLER = REPO / "packaging" / "install_from_source.sh"
CONTRIBUTING = REPO / "CONTRIBUTING.md"
LOCALIZED = REPO / "docs" / "i18n" / "readme"
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


def test_the_readme_names_the_default_branch_the_installer_uses():
    """A plain clone gets the default branch, so the README must name it.

    Until 2026-09-19 the README said "The default branch is ``nightly``"
    beside a plain ``git clone``. GitHub's default branch was ``main``
    (``gh repo view --json defaultBranchRef``), ``install_from_source.sh``
    defaulted to ``main``, and CONTRIBUTING.md called ``main`` "the release
    branch and the default branch". A reader who followed the README got the
    latest release while being told they had the development branch.

    The GitHub setting cannot be read offline, so this holds the three
    in-tree statements of it to one answer.
    """
    readme = re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))
    claim = re.search(r"This clones ``([\w.-]+)``, the default branch", readme)
    assert claim, "the README no longer says which branch a plain clone gets"
    installer = re.search(r'(?m)^BRANCH="([\w.-]+)"', INSTALLER.read_text(
        encoding="utf-8"))
    assert installer, "install_from_source.sh no longer sets a default BRANCH"
    contributing = re.search(
        r"(?m)^\| `([\w.-]+)` \|[^\n]*\bthe default branch\b",
        CONTRIBUTING.read_text(encoding="utf-8"))
    assert contributing, "CONTRIBUTING.md no longer names the default branch"
    assert claim.group(1) == installer.group(1) == contributing.group(1), (
        f"README says {claim.group(1)!r}, install_from_source.sh defaults "
        f"to {installer.group(1)!r} and CONTRIBUTING.md names "
        f"{contributing.group(1)!r} as the default branch")


def test_both_source_install_sections_read_as_reviewed_prose_in_every_language():
    """The nine READMEs carry these sections as reviewed text, not a draft.

    Until 2026-09-19 they carried the translation model's first draft: the
    German heading read "(Licht)" for "(light)" and called the checkout a
    "Kasse" -- a till -- and the French a "caisse". 328's DONE MEANS asked
    for reviewed prose here, so every paragraph and heading of the two
    sections must have a source-bound record in every language, and that
    record must be what the localized README actually shows.

    A new English sentence in either section fails here until it has been
    translated, rather than reaching the nine READMEs as model output on
    the next rebuild.
    """
    import sys

    tools = REPO / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from build_documentation_i18n import (
        REVIEWED_README_BLOCKS,
        translatable_blocks,
    )

    blocks, _layout = translatable_blocks(README.read_text(encoding="utf-8"))
    first = blocks.index("Install from source")
    last = blocks.index("Command-line entry points")
    section = blocks[first:last]
    assert "Install from source (light)" in section
    assert len(section) >= 8, section
    for source in section:
        reviewed = REVIEWED_README_BLOCKS.get(source, {})
        assert set(reviewed) == set(LANGUAGES), (
            f"no reviewed translation in "
            f"{sorted(set(LANGUAGES) - set(reviewed))} for {source[:60]!r}")
        for language, target in reviewed.items():
            localized = (LOCALIZED / f"README.{language}.rst").read_text(
                encoding="utf-8").replace("<../../source/", "<docs/source/")
            assert target in localized, (language, source[:60])

    drafts = {
        "de": ("(Licht)", "Kasse", "Standard-Zweig ist ``nightly``"),
        "fr": ("(lumière)", "caisse"),
        "es": ("(luz)", "compra de 1186"),
        "pt": ("(luz)",),
        "sv": ("(ljus)", "kassan"),
        "hi": ("(प्रकाश)",),
        "ko": ("(빛)",),
        "zh_CN": ("(光)", "支票"),
        "is": ("(Light)",),
    }
    for language, fragments in drafts.items():
        localized = (LOCALIZED / f"README.{language}.rst").read_text(
            encoding="utf-8")
        left = [fragment for fragment in fragments if fragment in localized]
        assert not left, f"{language} still carries the model draft: {left}"
