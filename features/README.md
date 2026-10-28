# spaCR features

Agent rules and dated handoffs live in [spacr/resources/AI](../spacr/resources/AI/);
read the [current handoff](../spacr/resources/AI/HANDOFF.md) before working.

Dated development snapshots are mapped in the [private archive index](../spacr/resources/AI/archive-index.md).

The canonical [Feature List](FEATURE_LIST.md) links every numbered item in
[`feature_list/`](feature_list/), including distinct files that share a historic ID.
Files retain their complete histories and original `Lifecycle: new` or
`Lifecycle: future` metadata. Lifecycle is a category, not completion status.
The generated list quotes recorded status; dated trailing notes remain authoritative.

Run `python tools/build_instruction_index.py` to regenerate the Feature List and
compatible [`00_INDEX.txt`](00_INDEX.txt), or add `--check` to verify both.
The old New/Future section names remain as index aliases; their files now share
one physical folder. Do not create replacement primary records in `new/` or `future/`.
Unnumbered operational and historical documents retain their existing locations.

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
prose inside older files. The tool now reads the consolidated folder; the name is the
only thing left over, and renaming it was not worth the churn while CI was
being brought green.

## Alpha features (the maintainer's rule, 2026-09-26)

Everything marked `Lifecycle: future` in `features/feature_list/` ships as an ALPHA feature: its
settings, buttons, menu entries, screens and model-zoo rows are hidden unless
the user turns on **Preferences -> Show alpha features** (off by default).
Items marked `Lifecycle: new` ship visible as before. When a future item is built,
register everything it adds with the alpha gate (see item 569) and say so in
the item's note; promoting a feature out of alpha is the maintainer's call.
Registering is one entry in `ALPHA_FEATURES` in `spacr/settings.py`, keyed by
the item number: `settings`, `choices`, `widgets` (Qt object names), `apps`
and `models`. `tests/qt/test_alpha_features_gate.py` fails when a future
item's Status headline says built, implemented or done and nothing is
registered.
