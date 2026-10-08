# N47 existing hidden-menu stylesheet work, 2026-10-08

This checkpoint changes only three executable lines in existing
`theme.apply_stylesheet_per_window`. Closed menus now use its already-existing
`aboutToShow`/`Show` stylesheet delivery. Open menus still change immediately;
actual popup and direct show, hide/reopen and changing themes are checked.
No API, signature, prose or UI captions changed. Complete normal before/after
maps are equal: API 13,232; UI 7,288; other five runtime buckets unchanged.

The historical 0b serial file took 1,824.273 seconds. A fresh exact current
30-case field-fade profile passes in 6.54 seconds, with no ambient ticks or
shade paths captured. It spends 0.814 seconds in 29 preference applications,
0.508 seconds applying per-window sheets, and 0.175 seconds forgetting them.
The unconditional QApplication stylesheet restore costs only 0.1 ms total
in this fresh profile, so no fixture change was justified. Historical cached
widget counts cannot establish the currently surviving native population.

The distinct bounded ABBA probe keeps 150 actual hidden menus, each with 12
actions, and alternates full current dark/light sheets. Its six baseline
applications take 225.5–366.3 ms; six candidate applications take 0.150–1.097
ms, while the current sheet reaches each sampled menu on opening. Cold first
applications are included. This is avoided hidden-menu work, not an
explanation or cure of the multi-minute hosted file. Menus are normally
deleted after the probe. No rendering density/detail, test order, assertion,
timing limit, cleanup policy, ratchet or failure policy changed.

The capped CPU Qt 6.12.0/pytest 8.4.2 owning cohort passes all 59 tests in
24.39 seconds under branch coverage. A replay of the exact baseline helper
fails both new closed-menu cases; its visible-menu control passes. These
phases overlap in test identity and are not summed as unique acceptance.
Both inserted guard outcomes execute directly. Existing uncovered source
paths in the focused report are not waived or treated as whole-module
coverage. Full original-order six-hour Qt acceptance remains open.

`menu_probe.py` is the exact original ABBA harness used before the source
commit. `reproduce_menus.py` additionally accepts the installed candidate by
restoring its frozen baseline helper in memory. The initial inventory setup
errors are preserved; final before/after logs and complete source maps are
accepted. Frozen full sources, patch, profile, coverage, commands embodied
by the scripts and original raw logs are hashed. Run
`python features/data/47_hidden_menu_stylesheet_cpu_2026-10-08/verify.py --git --current`
after integration; no agent-only commit object is required for verification.
