# Measuring spaCR's coverage honestly

Three tools and four ways the number lies. Everything here was learned by
being wrong first; instruction 288 carries the long version.

## batched_coverage.sh

Runs the whole `tests/` tree in batches of 60 files with `--cov-append`,
against a git worktree rather than the working tree.

**Batched, because a single `-n N` run loses a worker.** xdist discards a dead
worker's coverage data while its tests still count as passed, so the run
reports a number too low by whatever that worker measured and says nothing
about it. Batching costs one batch instead of the run, and the loss is visible
in the log.

**A worktree, because line numbers belong to a commit.** A coverage artifact is
only valid for the exact tree it measured. Edits made afterwards -- yours, or
another lane's -- shift the lines, and the report then names different code.
Coverage.py never reports a comment as missing, so a missing-line run that
lands on comments is proof the artifact and the tree disagree.

**It refuses to start if HEAD does not import.** A run measured on a HEAD that
cannot load reports hundreds of failures and a coverage figure for a package
that never ran. That happened here on 2026-08-30 -- 211 failures in batch one,
all a single missing name split across a commit boundary -- and cost forty
minutes to learn what one import would have said.

## check_arcs.py

    check_arcs.py <coverage.json> "spacr/x.py:105,129-131,88--80"

Prints REACHED or MISSING per target. A bare number is a line; `a-b` is a
branch arc; `a--b` is an exit arc (coverage.py writes a return as a negative
target -- line `a` leaving the function that starts at line `b`).

**Use it on every test written for a named branch, before writing the commit
message.** Six tests in this repository have passed while exercising nothing
at all, and every one of them asserted an ABSENCE -- no exception, an empty
list, a key not present -- which is exactly what a function returns when it
did not run. Only this catches that.

Cheaper habit, for tests: when you assert something is absent, drive one input
in the same test that produces it. Then the test fails on a mistyped
identifier without needing coverage at all.

## nopragma.rc

`exclude_lines = a^` -- a pattern matching nothing, so `# pragma: no cover`
stops hiding lines. Instruction 288 forbids the pragma; this is how you see
what it was covering.

## Scoping work from a report

Hand out arcs, not modules. Measured on the same 20 modules with the same
agents: "cover this module" closed 0 of 83 targeted arcs, and "reach line 818
and arc 816->822" closed 79. The scope was the only variable.

## Housekeeping

`/tmp/pytest-of-olafsson` reached 19 GB in 101 abandoned per-run trees and
took `/` to 405 MB mid-run. pytest keeps the last three runs' `tmp_path` per
run, and nothing removes them. Clear it before a long run -- and note that
`fuser -m <dir>` answers for the filesystem, not the directory, so it calls
every one of them busy.
