Unchanged motion settings now preserve the existing ready frame instead of
re-rendering it during Preferences Save. Six controls compare both widget
and engine values, so a caller that changed the exposed engine still gets
the requested setting and immediate pixels. Changed values, custom-palette
refreshes, pending input, normal animation and native detail retain their
existing behavior. No renderer or scientific input changes.

The source is frozen in ambient.before.py.gz and ambient.after.py.gz; hashes,
the exact diff, complete logs and branch evidence are included. The five-file
affected cohort passes 321 cases. A subsequent test-only correction makes the
direction case change from up to down rather than reapply its default; the final
17 focused cases reach all 22 added executable statements and 13 touching arcs.
These phases overlap; no unique-case sum is claimed.

On an actual 3840x2160 display at maximum detail/density/size, paired ABBA
comparisons restore the exact six old setters from 711 in an otherwise identical
process. Unchanged apply_ambient_preferences medians 599/587 ms become 0.145/
0.096 ms; redundant GUI-thread shaders fall from 5 to 0. This is the background-update
step, not the full dialog's Save time, and not a universal timing guarantee.

The final actual MainWindow probe visits Home/Mask/Measure/Annotate and
performs three modal Saves at 200->100->200%, Random->Spacr->Random and radius
0->65%->0. Checked producer frames retain alpha FF; nine natural GC callbacks
run on the GUI thread. Deferred close leaves 0 widgets/0 ambient/0 workers.
Peak 1331836 KiB and retained anonymous 1032636 KiB are not lower-memory claims.
Reported installed Save SIGSEGV, hard 24 FPS and final hosted Qt remain OPEN.

Replay from this source worktree using its installed spaCR Python environment:

```bash
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=xcb PYTHONPATH=. \
SPACR_SOURCE_ROOT="$PWD" PROBE_STORE=/mnt/wd4tb/scratch/unique-probe.ini \
XDG_CONFIG_HOME=/mnt/wd4tb/scratch/unique-config \
tools/run_capped.sh 4G xvfb-run -a -s '-screen 0 3840x2160x24' \
python features/data/663_unchanged_preferences_cpu_2026-10-06/probe_source_unchanged_ambient_controls.py
```

The probe loads the 711 baseline from git; its complete source is also frozen
here. The separate real MainWindow probe takes a unique receipt suffix as its
single argument. Use fresh configuration and receipt paths for each replay.
