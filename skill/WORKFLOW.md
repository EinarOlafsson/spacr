# Working in this repository

Discipline that is specific to spaCR. Some of it exists because several
agents have shared this working tree at once, and the failure modes of
that are unusual.

## Git

**Several agents may be editing this tree at the same time.** That single
fact drives most of what follows.

**Commit with an explicit pathspec. Never `git add -A` + bare
`git commit`.**

```bash
git commit -F /tmp/msg.txt -- path/one.py path/two.py tests/qt/test_x.py
```

Anything else sweeps another agent's staged work into your commit. This
has happened; the repair was an `-s ours` merge in a detached worktree.

* `git add <path>` is only for a **new** file, since `git commit -- <path>`
  cannot see an untracked one. Add exactly that file.
* `git commit -- <paths> -m "..."` **does not work** — `-m` after `--` is
  read as a pathspec. Use `-F <file>`.
* Run `git show --stat HEAD` after every commit and confirm the file list
  is the one you meant.

**Never rebase, amend or reset `nightly`.** It is shared and it is pushed.
Diverged? Merge:

```bash
git pull --no-rebase --no-edit origin nightly && git push origin nightly
```

**No `Co-Authored-By` trailer.** The maintainer is the sole author.

**Never push without being asked.** Commit freely; push when told.

## Commit messages

Prose, not a changelog line. Say what was wrong, what the evidence was,
and why the fix is the one chosen. The messages here are long on purpose —
they are the record of *why*, which the diff cannot carry.

Include measurements when there are any: "44.4% of the settings column was
painted by nothing at all; 0.0% after" is worth more than "fixed the black
box".

State what you left undone.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/qt/test_x.py -q -p no:randomly
```

* `QT_QPA_PLATFORM=offscreen` for anything touching Qt.
* `-p no:randomly` while iterating, so a failure is reproducible.
* Full qt suite is ~7,400 tests and hours. Run the files you touched, then
  the files that touch what you touched.
* Long runs: `nohup ... &` and poll the log. A foreground run will hit the
  two-minute command timeout and you will lose the result.

**Before changing a failing test, find out whether it is wrong.** Both
happen here. Several tests have been wrong since the commit that
introduced them; several "stale" tests were right and the code had
regressed. The docstrings usually say what the test is defending.

**When you do change a test, say why in its docstring** — what it used to
assert, and why that is no longer the right claim. A test whose history is
invisible gets "fixed" again by the next person.

Coverage needs the `sitecustomize` torch pre-import shim, or it dies with
a `_has_torch_function` crash.

## Never touch the user's machine

* Do not write real preferences (INVARIANTS §8).
* Do not leave background processes running. Stale pytest runs held 9.2 GB
  for over a day here.
* Do not install or upgrade anything without being asked.

## Version

`python packaging/release.py bump`. Bump the patch after a batch, never
per commit, smallest honest increment. Verify from a clean worktree —
this tree has had several agents in it:

```bash
git worktree add --detach /tmp/spacr-verify HEAD
```

## Leaving work unfinished

A file in `instructions/`, not a TODO comment. Say what the state is, why
it matters, what to do, and how to know it worked.

Include the things you decided **not** to do, with the reason. A decision
not to act is a result; without the reason the next person either
rediscovers it or "fixes" something that was intentional.
