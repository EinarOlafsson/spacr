# Real historical Linux GUI upgrade

Run: https://github.com/EinarOlafsson/spacr/actions/runs/36350873895
Witness source: `187637a160efb6b567d687ade4c4a425fec22b1c`.
Host: Ubuntu 24.04, x86_64, glibc 2.39.

The original checksum-pinned released 1.5.0.1 online installer was executed.
Its installed GUI Help action launched its real pip subprocess, which failed
with `No module named pip`. The documented private-uv external repair installed
1.5.0.5. That version's real GUI Help action then upgraded to public 1.5.1.0
through its own uv command. A fresh installed Python process verified the new
version; the private prefix stayed the same, pip remained absent, and all
three states retained Torch 2.14.0+cpu with no CUDA build.

Raw source-bound state/command receipts and the actual terminal dialogs are
included. The two result screenshots were inspected. Full input provenance,
installer, logs and prompt screenshots remain in the CI artifact and scratch
copy. No updater, child process, transport or update-dialog function was mocked.

This proves the Linux historical online recovery chain, including the explicit
external repair. It does not claim that broken 1.5.0.1 updates without repair,
that a frozen bundle updates, or that macOS/Windows passed this chain. The
first Mac witness failed because Cocoa reports a blank QMessageBox title;
the driver correction still needs actual execution.
