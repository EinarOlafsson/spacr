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
12. Keep RAM under 110 GB (`tools/run_capped.sh`). GPU only via `tools/gpu_turn.sh`. Chrome and its graphics processes must NEVER count as compute jobs or block any GPU queue, even when `nvidia-smi` lists them. Verify this exclusion in the actual scheduler used by every new run, including older worktrees.
13. Never touch livecell or cellposeTIME jobs; they belong to another session.
14. `einarolafsson/models` on Hugging Face is a DATASET repo.
15. Item state is in each item file's dated trailing notes and `features/00_INDEX.txt` (generated).
16. Stay responsive: never run anything that can take over ~1 minute in the foreground (pytest, sphinx, builds, renders, training, sleep/until/wait loops). Use Bash `run_in_background: true` (you are notified when it ends) or hand it to a subagent; the main session must always be free to answer. A local hook blocks foreground pytest/sphinx/long sleeps; mark a known-short command with `# fg-ok`.
17. Tutorials ship one lesson at a time, end to end, before the next starts: record it on the GNOME capture host (real app, real pointer, every clip through the smoothness gate; re-record failures once), synthesize audio only for changed sentences, translate only changed sentences into every lesson language, stage and render, publish (HF, Pages, verify `--published`, record), then commit and push to nightly. Never encode or upload while a recording runs; recordings run alone on an idle machine.
