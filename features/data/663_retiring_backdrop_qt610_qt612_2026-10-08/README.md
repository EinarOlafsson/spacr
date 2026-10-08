Source-bound CPU/Xvfb evidence for N663 backdrop retirement, 2026-10-08.

The before receipt uses commit `75c72e858a15c67768b503470773cb5db19e2e51` and Qt 6.10.0. At the first real Preferences Apply/Keep, `app.allWidgets()` still includes an old parentless, visible `AmbientWidget` whose timer is active alongside the new parented widget. A focused Qt 6.10 regression against that source also failed at `assert old.isHidden()` after `_retire_the_dock_backdrop` and `apply_ambient_preferences`; that negative test output was observed directly but was not retained as a raw log.

The after receipts use commit `8f245a658c9791debe9fe9848c5bf2c691bfb77e` with the identical app probe on Qt 6.10.0 and Qt 6.12.0. Three real modal Preferences Save/Keep cycles switch the main Spaceout backdrop, exercise independent popup backdrops, hide/reopen, visit Measure and Make Masks, and close the window. Both processes exited zero with no Python exceptions or captured slot/wrapper/painter messages. At the first Save, parentless retiring backdrops are hidden with stopped timers while the parented backdrop is visible and animating. At shutdown, all eight observed producers are retired, no `AmbientWidget` remains, and the native MainWindow has been destroyed.

The regression exercises the real dock and screen retirement methods, plus field-to-fractal replacement. The new nodes passed 3/3 on Qt 6.10 and 2/2 on Qt 6.12; the owning Spaceout file passed 27/27 on Qt 6.10, and two pre-existing parentless Preferences tests passed 2/2. These are bounded CPU checks. They demonstrate and repair orphan revival before deferred deletion. They do not reproduce or explain the user's GPU/ultra native crash, and they do not establish crash acceptance on the installed application.

The portable probe is `probe.py`. It takes `SPACR_PROBE_OUT`, creates private settings/home/log paths, and records imported module paths and SHA-256 values. From the source worktree, the Qt 6.10 invocation was:

```bash
SPACR_PROBE_OUT=/path/to/output CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. xvfb-run -a -s '-screen 0 1920x1080x24' ./tools/run_capped.sh 4G /home/olafsson/anaconda3/bin/python probe.py
```

Qt 6.12 used the same invocation with `PYTHONPATH=/mnt/wd4tb/scratch/ci-7a-root-20261008/qt612-overlay:.`. The probe's checkpoint field is intentionally fixed to its source commit; change that field when replaying another source. Qt emitted only OpenType font warnings to the captured Qt handler. The separate GDK monitor warning appears in the raw logs after the successful receipt.

SHA-256 payloads:

```text
9f7fefab5dab2e11edf4d1dd564b68e5a36d839a61f4c31863625ef5546d7f79  after-qt610.json
e956ec395b62d99f3442e2005344c60efb7274dee65387012287b5d1a0e4b287  after-qt610.log.gz
4a1607a810830a0ba03f9f59c5c2db8715b9425f9edf2cefac588c6bfef14d98  after-qt612.json
19a100021edc16ec3d37168239ec6f18fbbc55e0c9fa1d333f371ae6a19e1e67  after-qt612.log.gz
0de4309566b35061f2c030520ebbd3382e9fc2d07c25ee91b4c15ac568e2705d  before-qt610.json
35d0b5edd9d1326b4cc8898e8af2cacc3e597e84012d83a87959408155780c7d  probe.py
```
