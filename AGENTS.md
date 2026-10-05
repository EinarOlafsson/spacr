# spaCR agent rules (Codex, Claude and others)

Read `features/HANDOFF.md`, section "CURRENT HANDOFF FOR CODEX AND OTHER AGENTS", first.

1. Origin has only `main` and `nightly`. Work from `nightly`; push only with `git push origin HEAD:nightly`.
2. Branch work in a worktree under `/mnt/wd4tb/spacr-worktrees/`; scratch in `/mnt/wd4tb/scratch/`.
3. Never `git stash`. Never reset, check out or clean a shared checkout.
4. Commits carry no `Co-Authored-By` or AI trailer; the hook refuses them.
5. Items from `features/future/` ship alpha: register in `ALPHA_FEATURES` with a literal `setObjectName`.
6. Alpha organisms (all but Toxoplasma) sit behind "Show alpha species" (`ALPHA_SPECIES`).
7. Tutorials show only the current layout and no alpha content except lessons 86/87; record with `--fresh`. Lane owner: spacr-d7.
8. User strings through `tr()`. No `#` comments in `spacr/`. No new modules under `spacr/`.
9. Tables via `spacr.tabular`; figures via `spacr.plot.save_figure`, styled by `spacr.figures.style._apply_user_style`.
10. Regenerate generated artefacts with their tools; never hand-merge them or raise a ratchet ceiling.
11. Tests: `CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 8G python -m pytest`, `-n 4` at most, never the whole suite.
12. Keep RAM under 110 GB (`tools/run_capped.sh`). GPU only via `tools/gpu_turn.sh`.
13. Never touch livecell or cellposeTIME jobs; they belong to another session.
14. `einarolafsson/models` on Hugging Face is a DATASET repo.
15. Item state is in each item file's dated trailing notes and `features/00_INDEX.txt` (generated).
