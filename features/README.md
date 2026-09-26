# spaCR features

Two lists, and **neither one blocks a release**.

| | what it holds |
|---|---|
| `new/` | what spaCR has gained. Every feature that has landed, with the measurements and the reasoning that produced it. |
| `future/` | what spaCR might gain next. Nothing here is promised and nothing here is scheduled. |

`00_INDEX.txt` is generated from the two folders by
`tools/build_instruction_index.py`. Do not hand-edit it — an index that
disagrees with the folder is worse than no index, because it is believed.

## Why the files are long

A file here is not a ticket. It records what was measured, what was tried and
failed, and what was decided not to do and why. That last part is the one
worth keeping: a decision not to act is a result, and re-deriving it later
costs the same as deriving it the first time.

Several files contain **retracted claims left in place**, marked as retracted.
They stay because a theory that was tested and refuted is cheaper to read than
to re-run — four separate theories about why a block would not translate were
each argued convincingly and each measured false.

## What changed on 2026-09-12

This was one `instructions/` list whose items carried release-blocking status,
which made a plan into a gate. Splitting it was the maintainer's decision, and
so was retiring two items outright and folding a third into the item it
duplicated.

The word "instruction" survives in `tools/build_instruction_index.py` and in
prose inside older files. The tool works on the new folders; the name is the
only thing left over, and renaming it was not worth the churn while CI was
being brought green.

## Alpha features (the maintainer's rule, 2026-09-26)

Everything built from `features/future/` ships as an ALPHA feature: its
settings, buttons, menu entries, screens and model-zoo rows are hidden unless
the user turns on **Preferences -> Show alpha features** (off by default).
Items in `features/new/` ship visible as before. When a future item is built,
register everything it adds with the alpha gate (see item 569) and say so in
the item's note; promoting a feature out of alpha is the maintainer's call.
