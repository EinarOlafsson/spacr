# Actual installed Windows/macOS Measure results

Run: https://github.com/EinarOlafsson/spacr/actions/runs/36348757658
Source: `2c9497c42ac84c19b9e1c5d8affbffd915a3ff15`.

Windows 2025 Server (10.0.26100, AMD64) installed the actual NSIS artifact and
passed its native Qt/packaged-resource/real Measure Run checks. Its synthetic
field produced 16 cell rows, one successful field and no failures. Torch was
2.14.0+cpu. The subsequent Claude command tests are separate, failed jobs.

macOS 15.7.9 arm64 installed the DMG's actual application and completed the
same real Measure run with 16 cell rows. Its job then FAILED the added menu
witness because that witness required a system menu, whereas the later
maintainer request deliberately keeps the unified bar inside the window
(commit1ce9f8d3f, 2026-08-31). The failed raw receipt is preserved as failed.
Correction must test the requested real visible menu and window controls on
Cocoa, then observe Preferences and Quit; no completed menu acceptance is
claimed here.

Both downloaded databases independently passed SQLite integrity and cell-count
checks (ok; 16). Raw receipts/screenshots and source/database hashes are retained.
These are synthetic-fixture application checks, not scientific model validation,
frozen replacement, account/consent acceptance or full feature53 closure.
