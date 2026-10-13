N656 per-screen shortcut rebinding CPU checkpoint, 2026-10-08

Source checkpoint ccf9ce7a4ac8f11456b4245eeb35295eb70cf7ff, baseline fb846a1d992.
Four source files, two test files, no new spaCR module, generated edit or GPU work.
53 scoped editable rows: Annotate34, Make Masks16, Field browser3. Global rows
remain editable. Stable default-key identities preserve aliases independently.
Saved maps update real shortcuts and event handlers on existing/new screens.
Real typing, annotation/undo, tool changes, zoom dismissal, navigation and disk
quarantine/restore are checked. Empty bindings, defaults, sorting, corrupt maps,
modifier/text handling and conflicts retain meaningful positive/negative guards.

Authoritative final-source coverage is source-final-coverage.json.gz: 81 checks
followed by two strengthened existing nodes, source bytes unchanged. Every changed
executable statement and outgoing arc in all four files is directly covered.
The earlier 222-check owning cohort is useful regression evidence, but predates
final literal-caption helper completion. Counts overlap and are not summed.
Iteration failure logs retain intermediate implementation mistakes honestly.
No full-suite, global numerical-ratchet, native FPS, GPU or crash-cure claim.

Normal tools produced API13215 before/after: zero arrivals/removals, changed module
body spacr.qt.shortcuts only. Six normal runtime buckets: ui7245 ->7267 (+22,
zero removals); other five value mappings unchanged. Actual old shortcut module
was imported from a baseline tree with all other production Python bytes unchanged.
That tree needed existing tracked resources/packaging, reflected in probe scripts.
The source caches are complete normal-tool snapshots, not hand-generated catalogs.
All22 new tr captions occur in normal generated/compact ownership.

Owner next steps: regenerate API/runtime reviewed catalogs normally, document
F1 / Change shortcuts / Scope rows, then cross-platform hosted checks. No
ratchet changes are needed; public action/default names are retained.

Verification from repository: python features/data/656_screen_shortcut_rebinding_cpu_2026-10-08/verify.py --git
Frozen only: python .../verify.py --frozen
Current source comparison: python .../verify.py --git --current
Original inventory scripts retain exact original absolute paths as provenance.
Run corresponding normal public_docstrings/canonical_sources on actual baseline
and candidate worktrees to reproduce inventory comparison; do not mutate catalogs.

Tests used CUDA_VISIBLE_DEVICES='', SPACR_DEVICE=cpu, QT_QPA_PLATFORM=offscreen,
MPLBACKEND=Agg, tools/run_capped.sh4G. Hosted-version overlay PYTHONPATH:
.:/mnt/wd4tb/scratch/ci-7a-root-20261008/qt612-overlay:
/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay
Interpreter /home/olafsson/anaconda3/envs/spacr/bin/python; pytest -q -p no:randomly.
Final-source files: test_screen_shortcut_rebinding.py,
test_rebindable_shortcuts_and_settings_profiles.py, test_shortcut_overlay.py,
test_every_callable_in_the_package_is_documented.py. Extra two named nodes are
recorded in remaining-branches.log. Full source/test diff is frozen separately.
