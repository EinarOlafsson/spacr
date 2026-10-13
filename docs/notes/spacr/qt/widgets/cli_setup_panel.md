# Notes from `spacr/qt/widgets/cli_setup_panel.py`

Written with the module on 2026-09-19 (item 420). The module carries no comments, so its reasons live here and in its docstrings.

## CliSetupPanel

THE INSTALL RUNS ON A `JobRunner` WORKER, NEVER ON THE GUI THREAD. An installer takes minutes; the setup screen is modal, and a modal frozen for minutes looks crashed. Each output line crosses to the GUI thread through the `_output_arrived` signal, which is the only thing `job_runner`'s rules allow a worker to do. Measured on 2026-09-19 with a stand-in `npm` that took 1.6 s: choosing the provider and pressing Install returned to the event loop after 0.33 s, 0.3 s of which was the probe itself holding the prompt on screen to photograph it; the panel showed the installer's `\r`-redrawn progress line while it ran, and the GPT mark turned READY without a restart once the stand-in `codex` answered `login status` with 0.

THE COMMAND STAYS ON SCREEN FROM START TO FINISH, with Copy beside it, because the item asks for it for the user who would rather run it themselves, and because it is the one thing every failure message can point at.

EVERY FAILURE NAMES A NEXT STEP (`OUTCOME_TEXT`). npm's permission failure gets its own sentence: on this machine `npm` is `/usr/bin/npm`, its global folder belongs to root, and `npm install -g` fails with EACCES; `npm config set prefix ~/.local` is npm's documented fix and puts the CLI in a folder `cli_install.locate` searches.

CLOSING THE SCREEN STOPS THE INSTALLER (`shutdown`, called by `SetupSlides.accept` and `reject`). An installer still running behind a closed screen has no Cancel button anywhere.
