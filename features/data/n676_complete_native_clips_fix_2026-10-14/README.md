# Held review patch: complete native tutorial actions

Commit `e257c1f3c3e04408968f656eade6a452dbe0bfc0`, parent `fb281b62f2085c9b86db1363eb3029c08a3cb305`. This is a review artifact; application/tool source remains held during release CI. No source push or integration is authorized by this packet. Root owns later docs/features publication and Home owns source integration.

The rejected 223-second lesson 43 master trimmed every native clip from its beginning to the longest narration duration. Source lost its completed path (27.067-second capture reduced to 14.8 seconds); Apply lost the search, enable and model configuration (58.1 seconds reduced to 26.933 seconds). The old player independently skipped native scene tails and jumped to video EOF when shorter audio ended. Complete raw captures remained valid and unchanged.

The patch derives full native duration from verified contiguous decoded frames at the exact visual frame rate, avoiding a phantom frame from rounded container milliseconds. Newly verified timelines opt in with `complete_native_clips: true`. Their video plays at speed 1; speech plays naturally once, then pauses while the remaining real footage continues. Video EOF owns completion. Explicit seeks, chapter/transcript times, voice/language reload and pause/resume retain coherent clocks. Flag-absent historical playback keeps its previous behavior. The staged verifier independently checks full native tails and final EOF, and rejects injected old skip behavior. No resynthesis, fabricated holds, looping, speedup, gate changes or recapture.

Exactly seven intended tools/test files are present in the format patch:

- `tools/tutorials/authoring/web/app_v2.js`
- `tools/tutorials/native_live_timing.py`
- `tools/tutorials/tests/test_native_live_timing.py`
- `tools/tutorials/tests/test_player_complete_native_clips.py`
- `tools/tutorials/tests/test_player_native_live_video.py`
- `tools/tutorials/tests/test_staged_complete_native_boundary.py`
- `tools/tutorials/verify_staged_lesson.py`

`spacr/` and `tools/tutorials/human_input.py` are byte-identical to accepted capture source `a5dc7353bc1da39795c565dd4d4dd27f3f2ba5af`; the seven-file commit also changes no canonical lesson, translation or audio.

| Focused validation | Actual terminal result |
| --- | --- |
| Native timing, visual frame alignment and live clip renderer | 25 passed, 0.38s; observed capped pytest result (no separate retained log) |
| Player source/contracts including new complete-mode browser cases | 18 passed, 46.44s |
| Final real Chrome rerun, complete-mode and historical native cases | 4 passed, 46.05s; GPU and accelerated video decode disabled |
| Complete-mode staged verifier and owning checks | 23 passed, 20.90s; includes old-skip negative mutation |

These overlapping focused suites are separate outcomes, not 70 unique tests. Earlier failed fixture/test observations remain preserved; the table binds the final applicable passes.

Actual regenerated master: 329.000 seconds, 9870 frames, 3840×2160, 30fps. Complete Source uses 812/30 seconds and Apply uses 1743/30 seconds. Existing 14 scene references/12 clips are preserved. Full decode, frame count and presentation-time checks passed; four actual decoded samples show the completed Source, exported QC, restored Illumination and final Measure model configuration. All 50 original audio and 50 timing-sidecar hashes remain unchanged. The sample review is bounded visual evidence, not full-film or publication acceptance.

Evidence bindings:

- `/media/carruthers/mnt3/claude/n676_stages/work/lesson43-r34/complete-native-player-reviewed-tests.log` — SHA256 `df95289a8122e8a3d381c459b722ae45cef50e4d3b8678f4000828802a3517a7`
- `/media/carruthers/mnt3/claude/n676_stages/work/lesson43-r34/complete-native-player-terminal-tests.log` — SHA256 `7ebeae24b6bed860076f7a751c2eb871c593a7921eaf084e8c99303b823ea106`
- `/media/carruthers/mnt3/claude/n676_stages/work/lesson43-r34/native-verifier-tests-r2.log` — SHA256 `138d681c33d247c606c3822603fc69c903d702e5168838ab61e9c5860862f816`
- `/media/carruthers/mnt3/claude/n676_stages/work/lesson43-r34/complete-master-review.json` — SHA256 `8e79c5fb0ea4adf7457882589586a30a06e8de9a4918f1a7c2a719ad2cdc6003`
- `/media/carruthers/mnt3/claude/n676_stages/work/lesson43-r34/failed-223-render/snapshot.json` — SHA256 `e767dfcd6841505d38f4174c0eb9f0d4f65cfa5f3699cf75a5b1ff6f3b59049a`
- `/media/carruthers/mnt3/claude/n676_stages/host/stage-43-r34-20261014/production/43_illumination/current-video-acceptance.json` — SHA256 `3ba7e89887f5666864e4a73e309a93a95942da5912792bc66210e5ff0abc7564`

The failed 223-second master/timing/web assets remain under `../failed-223-render/`. Actual 71-route lesson 43 browser verification and candidate construction were still running when this packet was prepared. Their outcome, hosted readback, published browser verification and final publication are deliberately not claimed here.
