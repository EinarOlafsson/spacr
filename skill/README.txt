================================================================================
THE spaCR SOFTWARE ENGINEER SKILL
================================================================================

What this is
------------
Everything an engineer -- human or agent -- needs to work on spaCR that is
NOT visible from reading the code: the invariants, the working discipline,
the map, and the open work.

It lives in the repository rather than in any one tool's config directory,
so it travels with the software, survives a change of machine, and is
visible to every agent working in this tree.

How to load it
--------------
Claude Code picks it up automatically. .claude/skills/spacr-engineer/
holds a short SKILL.md whose only job is to send the reader here, so the
content lives in one place and the tool-specific wiring stays thin.

Any other agent, or a person: read skill/SKILL.md and follow it. Nothing
here depends on a particular tool.

FIRST, EVERY SESSION
--------------------
    python skill/refresh.py

That regenerates FACTS.md and checks the invariants. It exits non-zero if
one no longer holds.

A FAIL is not noise. It means either the code regressed or this skill is
describing software that has moved -- and the whole value of INVARIANTS.md
is that its contents are true. Decide which it is, fix it, and say which
in the commit message.

The files
---------
    SKILL.md          the skill; how to work here, and the duty to keep
                      this current
    FACTS.md          GENERATED. Version, counts, module sizes, the
                      invariant check results. Never hand-edit.
    INVARIANTS.md     the rules. Each cost real debugging and each says
                      how it was found, so a reader can tell a rule that
                      still applies from one that quietly stopped.
    WORKFLOW.md       git, tests and commits in THIS repo. Several agents
                      have shared this working tree at once and the
                      failure modes of that are unusual.
    ARCHITECTURE.md   the map: where things live and why.
    refresh.py        regenerates FACTS.md, checks the invariants.

The open work is in ../instructions/, not here. This folder is about how
to work; that one is about what is left.

    ../instructions/open/    every task not yet finished
    ../instructions/done/    every task that is

A task gets a file in open/ BEFORE the work starts and moves to done/ when
it finishes. A session that runs out of context, or a machine that goes
down, loses everything that was only ever in the conversation -- this is
what stops that. refresh.py checks the ledger with everything else.

Why it self-checks
------------------
A skill file is a claim about a repository, and a repository moves. A
skill that is only ever READ decays into a confident description of a
program that no longer exists, and the reader has no way to tell which
half is stale. refresh.py is what stops that: every rule a machine can
verify is verified on load.
