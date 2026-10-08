# Spaceout Preferences popup backdrop hotfix

The published `891c9db2a17d7b1eda56f3a5b8d1f681790c2209` source installed a second Spaceout fractal when Preferences requested its independently selected `drift` backdrop. In a real MainWindow with the dialog filters installed, the CPU-backed live-fractal count changed from one to two while Preferences opened. The popup's actual widget was `CpuFractalWidget`, despite the saved `drift` choice.

The isolated hotfix `310b604e0218a1cb0caf527dd7f4bad0445dd576` marks the dialog backdrop as independent before installation. The normal Spaceout backdrop remains one fractal; Preferences receives an `AmbientWidget` painting `drift`. A real software-X Qt 6.11.2 route opened and closed Preferences with one fractal throughout. The focused Qt 6.12 cohort passed 24 tests, including the popup, original Spaceout installation, and ordinary-launch controls. The new test uses a lightweight fractal constructor double to check routing and lifetime counts; the separate Xvfb runs exercised the actual CPU fractal widget.

The original Qt 6.11.2 scratch probe intentionally omitted the normal event-loop shutdown and aborted while tearing down a running QThread **after** recording the two-fractal observation. That teardown abort is not evidence of the user's opening-time crash. The reported GPU/ultra crash was not reproduced, and no GPU acceptance is claimed.

The normal API extractor kept 13,215 symbols and changed only the existing `AmbientWidget.set_theme` and `AmbientWidget.set_palette` prose. No UI string or public signature was added. Source snapshots, patch, raw logs, and their checksums are bound in `receipt.json`; run `python verify.py` here, or `python verify.py --git` after committing the archive.
