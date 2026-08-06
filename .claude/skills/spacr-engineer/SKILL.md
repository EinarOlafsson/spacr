---
name: spacr-engineer
description: Software engineer for the spaCR codebase — a PySide6 desktop application for CRISPR screen image analysis. Use when reading, changing, testing, debugging or reviewing anything under this repository. Carries the invariants and the working discipline that are not visible from the code.
---

# spaCR software engineer

The skill lives in `skill/` at the repository root, so that it is visible
to any agent working in this tree — Claude Code, Codex, or a human — and
not only to the one whose tool directory happens to hold it.

**Start here, every session:**

```bash
python skill/refresh.py
```

It regenerates `skill/FACTS.md` and checks every invariant that a machine
can verify. A FAIL is the session's first job: either the code regressed
or the skill is stale. Decide which, fix it, say which in the commit.

Then read `skill/SKILL.md` and follow it. In short:

| File | What it is |
|---|---|
| `skill/SKILL.md` | How to work here, and the duty to keep this current |
| `skill/FACTS.md` | Generated. Never hand-edit. |
| `skill/INVARIANTS.md` | The rules, each with the evidence that produced it |
| `skill/WORKFLOW.md` | Git, tests, commits — several agents share this tree |
| `skill/ARCHITECTURE.md` | Where things live and why |
| `instructions/00_INDEX.txt` | The open work |

Updating this skill when you learn something is part of the job, not
housekeeping. See the "Keep this skill current" section of
`skill/SKILL.md`.
