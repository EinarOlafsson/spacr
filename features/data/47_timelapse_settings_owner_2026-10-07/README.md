# N47 Timelapse settings test ownership, 2026-10-07

`tests/qt/test_timelapse_always_on.py` built temporary Timelapse, Measure and Mask `SettingsWidgets` panels without Qt parents in its fixture and two direct paths. The panel is now constructed by one private test helper that registers a QWidget owner with `qtbot` and passes it as parent. The assertions about hidden, present and unchanged controls remain the same.

The entire file followed by one unrelated `test_field_fade.py` sentinel passed 9/9 both before and after under the 4 GiB cgroup, hidden CUDA and Qt offscreen. The unchanged passive serial journal measured the following boundaries; no forced collection, app source or ceiling changed.

| Boundary | Before | After |
|---|---:|---:|
| After Timelapse file teardown | 1,369 widgets / 581 top levels; RSS 551.0 MiB | 445 widgets / 15 top levels; RSS 545.3 MiB |
| After field-fade sentinel teardown | 928 widgets / 379 top levels; RSS 551.0 MiB | 2 widgets / 1 top level; RSS 545.4 MiB |

The transient 445-widget snapshot is deferred Qt cleanup, not a persisting tree at the next ordinary boundary. This bounded pair shows a 5.6 MiB RSS difference at sentinel end, without a full-suite saving claim. Both runs emitted the same `SpawnProcess-1` session-exit warning; this test file does not create that process and the ownership change did not resolve it. N47 uninterrupted serial acceptance remains open. `manifest.json` pins source and journal hashes.
