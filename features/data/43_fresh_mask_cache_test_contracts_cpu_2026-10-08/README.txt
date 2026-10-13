Fresh Mask cache contracts, 2026-10-08

Two existing tests require Mask to be unbuilt: main-window navigation must add one stack page, and rebuild-before-built must have no previous Mask screen. A normal constructor can restore a persisted Mask first. The archived plugin writes and verifies a real saved session immediately before each constructor; both original nodes fail before and pass after explicitly requesting Home. App, original assertions, timing limits and deliberate session-restore tests remain unchanged.

Actual pytest8.4.2; CUDA-hidden/offscreen CPU;4 GiB cap: before2 failed7.33s; after2 passed5.53s. Both processes retired. Each after record selects actual Home with empty module cache despite persisted Mask. Both runs use normal strict pytest.ini. Existing full-Ruff findings match exactly, including two baseline F401 unused imports in test_main_window.py; no incidental import cleanup performed. git diff check passed. Durations are correctness-run observations, not performance acceptance.

Verify with verify_manifest.py --git HEAD --bind-current. reproduce_after.sh replays both real persisted-session assertions in an isolated checkout using the retained pytest8.4.2 overlay.
