# Installed Linux acceptance, 2026-09-27

Run: https://github.com/EinarOlafsson/spacr/actions/runs/36348757658
Source: `2c9497c42ac84c19b9e1c5d8affbffd915a3ff15`.

Both fresh Debian 12 and Ubuntu 22.04 jobs installed the actual frozen Debian
artifact, launched native XCB Qt from an empty working directory, and clicked
Measure's real Run action on one synthetic fixture field. Each completed with
16 cell rows, one successful field and no failed fields. Imports originate in
the installed `/opt/spacr` bundle; Torch is `2.14.0+cpu`. Raw receipts and
screenshots are retained without alteration.

Both jobs removed the package and asserted that its application, launcher and
desktop entry were gone while the completed analysis database remained. The
downloaded databases independently passed SQLite `PRAGMA integrity_check` and
`SELECT count(*) FROM cell` (ok; 16). Their hashes appear in SHA256SUMS.txt;
full databases and logs remain in the downloaded CI artifact and scratch copy.

Visual review found clipped Actions controls at 1280x720 in both screenshots,
and a very narrow settings pane in Debian. These are functional small-run
receipts, not layout acceptance; that defect remains under investigation.

The fixture is synthetic, not scientific model validation. These results do
not cover frozen self-update, GPU selection, account or consent behavior,
macOS/Windows acceptance, or changes after the recorded source commit.
