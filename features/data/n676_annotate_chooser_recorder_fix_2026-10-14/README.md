# Annotate recorder chooser race — held source review patch

Actual public09 turn39 stopped at21:50:33 EDT with rc143 after about19min
stuck. Only opening01 finalized; no accepted whole lesson or scientific result,
and no loaded example data. Root preserved an all-thread SIGUSR1 snapshot,
then SIGTERM'd only its exact owned app PID222586. Cleanup reports exit143
and no survivors. Original raw stage remains untouched at
`/media/carruthers/mnt3/claude/n676_stages/host/lesson09-r39`.

The main thread was inside `capture_refresh.py`'s opening mouse click →
Annotate `_choose_the_test_data` → `dialog.exec()`. The recorder scheduled a
single chooser callback1200ms before real pointer travel/button activation.
That callback could run with no modal, return with an error, and leave the
subsequently opened chooser without a surviving selection callback.

Held local commit `b1b6d07f107fa53059d093015735966fcbc5a499` changes only
`tools/tutorials/capture_refresh.py` and its focused test. It uses the existing
`capture_ui.when_modal` host wait, requires exactly the real active chooser,
and retains genuine route hover/capture/clicks. A total180s watchdog spans
discovery plus selection; the actual chooser clips alone total36.867s
(16.167+20.7), so a30s global deadline would incorrectly abort healthy work.
Absent, wrong or late modals fail closed; click-finally cancels the watchdog,
and canceled callbacks cannot reject a later unrelated modal.

Final16 focused CPU tests PASS9.89s, private profile/4GB lease, CUDA empty and
software Mesa environment; minimal real QWidgets/chooser/organism only, no
MainWindow, native host or GL context. Tests cover both routes after1600ms
opening delay, wrong modal, expired wait then late modal, canceled callback,
and timer-pumping simulated actual36.867s capture duration. Independent HIGH
source review PASS. All test children exited and own RAM lease was removed.

App, human_input, capture adapter, alpha, data, gates and canonical content
remain unchanged. This is a source review artifact, not native acceptance.
Source stays HELD locally; Work publishes docs/features only. Root owns any
Home integration decision and fresh whole09 turn40 queue; turn40 remains
pending at this handoff. No retry can turn this stuck turn39 into acceptance.

Exact files copied from sealed evidence (no images/videos):

- `annotate-chooser-recorder-fix.patch` — 11177B, SHA256 `c07c25417d29c81fc1da6be7197b0420045023a8e344f31f68b6b7ccb55d6913`.
- `chooser-tests.txt` — 99B, SHA256 `306201810061ff1cd0f7b3407c7d77604741e48143ad635285cde110b81b4e70`.
- `cleanup.json` — 56B, SHA256 `bfdf0d63395674875000989e0bd7231b5e778e72dc7302317234434e273aa749`.
- `root-stuck-run-stop.json` — 528B, SHA256 `9037b12ee3a375cbc516cd050fce7845a18cd9b1edf0bf4721e5d2cb97cac6ee`.
- `stacks.txt` — 4114B, SHA256 `ce0731f979f98d73b75c5eca081e98568e698250e753091a7a7017cffb370915`.
