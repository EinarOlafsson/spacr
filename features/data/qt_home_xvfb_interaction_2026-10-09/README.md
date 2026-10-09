# Home interaction checks on a software display (2026-10-09)

Source baseline: `e6004557024ec29a25d8a378beef05bd91ab9be2`. The test-only correction is `6a9939c9059c16c3738c0d1fba19ecf496db824a`. The application source was unchanged. Python was `/home/olafsson/anaconda3/envs/spacr/bin/python` (3.12.13), with PySide6 6.11.2 and pytest 9.1.1. All tests used `tools/run_capped.sh 4G`, hidden CUDA, single-threaded BLAS, and `-p no:randomly`.

The five exact nodes from the cancelled c4 serial failure passed under Qt's `offscreen` platform on current e600. Under `xvfb-run -a -s '-screen 0 1920x1080x24'` and `QT_QPA_PLATFORM=xcb`, the same five produced two passing Preferences nodes and three failing Home nodes. The Home hover fixture had moved its 1400×900 page to (4000,2400), outside that virtual display. Moving it inside the available screen made the actual stage-colour rim pixels pass. The keyboard test invoked the dock shortcut without activating its window. Activating the window before that action made focus land in the sidebar. Neither pixel-colour nor focus assertions were relaxed.

On the final test blobs, `final-five.log.gz` records 5 passed in 8.98 s. `home-twofiles.log.gz` records 219 passed in 85.81 s on the intermediate candidate with equivalent on-screen placement and window activation, before a local import was moved and activation was changed to `qtbot.waitUntil`. `offscreen-fourfiles.log.gz` records 501 passed in 114.54 s before this correction; offscreen success alone did not reproduce the Xvfb boundary. This bounded evidence does not prove the entire original serial order is clean.

Source Git blobs, in order: `app.py` `cd24d01f37ec795f420bcb814c011f2964ff627c`; `preferences.py` `810e4e096585149aaa23b9921d053a046d6bf5af`; old Home stage test `fec72c27530d36629efb9c73b68eb2a2a22c542e`; old Home v2 test `f0a3e844978fde2c8b87e4beca3ef3dca1ddd0a7`; final Home stage test `3b59b07289cc8cc0dd310e4b58fadde6c252a6af`; final Home v2 test `3c3a4c8e313b213b313c78455248bc73110d1375`.

Compressed raw log SHA-256 digests:

| Payload | SHA-256 |
| --- | --- |
| `before-five.log.gz` | `5f1018be31f65d0f153136f4d4d11aa65e1f130518f9e3a62fcb9204f4647b7e` |
| `after-five.log.gz` | `6e341fb13f36748e085450e2e0d86cbe4bf6d5090e72ca747dcfeb838ec07acd` |
| `final-five.log.gz` | `b3e3980f1e8f8766afd1f165117cb7d30feca84352564c7a6a80ea00ef92adb3` |
| `home-twofiles.log.gz` | `9bc99ad57a422f79be79b7400ff6d11253cb6ca420a07d27c35757fe745a9c5f` |
| `offscreen-fourfiles.log.gz` | `6080153f40f3caf5902c1e0d6172ac58d6ca78a713ea7146bf08903b4b9772e1` |

The unrelated old Preferences `Field ripples` failure was already fixed between c4 and e600 by displaying the row label expected by its help table. Its sizing node passed both current offscreen and Xvfb replays.
