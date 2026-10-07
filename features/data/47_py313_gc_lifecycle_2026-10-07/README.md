# Python 3.13 Qt GC and Preferences Save probe, 2026-10-07

The installed crash archive at `52aa7a5af9:features/data/663_local_native_crash_trace_2026-10-06/pid-3063654-faulthandler.log` records Python 3.13 garbage collection at `gc_policy.py:96` while `app.py:7331` is inside `app.exec()`. The ambient producer is waiting on its stop event and four path-probe workers are waiting on queues. The trace identifies neither the native destructor nor a Save action. It is not the separate hosted Python 3.12 puncta crash.

| Probe | Source and environment | Result |
|---|---|---|
| Actual `AmbientWidget` impulse lens, Detail 200%, radius .65, 3840×2160 Xvfb screen, cyclic `QWidget`/`QTimer` objects | source `b87a3c38aa09936c157c4bded6627c6e6342f713`; Python 3.13.14, PySide6 6.11.2, NumPy 2.5.2; 4 GiB cgroup, CUDA hidden | exit 0; 257 shaded frames; 6 natural GC callbacks and 2,625 destroyed QWidget cycles, all on the GUI thread; renderer worker stopped after close; VmHWM 357,392 KiB |
| Actual `MainWindow`, one modal Preferences Save changing Detail 200%→100%, 3840×2160 Xvfb screen | same source/interpreter/Qt/NumPy/cap; scratch overlay additionally contained minimal CPU UI dependencies | exit 0; stored detail 1.0; 2 natural GC callbacks on GUI thread; 0 ambient widgets/workers after close; VmHWM 590,604 KiB |

The actual source byte SHA-256 values are in the raw logs. Both probes used the repository's `tools/run_capped.sh 4G` with `CUDA_VISIBLE_DEVICES=''`, `QT_QPA_PLATFORM=xcb`, and a private `XDG_CONFIG_HOME`. The `spacr13` Conda environment itself was not modified; PySide6, NumPy and UI dependencies were installed under `/mnt/wd4tb/scratch/qt-py313-gc-20261007/overlay`. The archived scripts retain that scratch path and are the exact measured scripts. `ambient_gc_probe.py` SHA-256 is `8c96b8879eb7e861cb33d5805c8c49753fd2b25d082165f981d74fd5e6c43120`; `mainwindow_save_probe.py` SHA-256 is `02d8e41f093bd1601d8e47b81a8741d8df07f557923fa1e49794b1e0d3a544eb`.

This is a bounded negative reproduction. The MainWindow probe uses only Home and Preferences, not Mask/Annotate screens or the workstation's complete installed dependency set. It does not establish the cause of the user's SIGSEGV, nor does it pass N47's uninterrupted full serial acceptance. No production code or memory limit changed.
