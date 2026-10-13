The e41 serial run failed while building module settings: `Section._on_toggle`
imported `field_ripple_for_widget` from an absent ambient module. The same
original test also found a stale stand-in ambient interface. The production
change catches `ImportError` only at the two optional ripple-import sites;
the section's visibility, signal, scrollbar and layout work still runs.

The original missing-module screen assertion and a direct scroll-hosted
section assertion pass. Two further cases observe that a present helper is
scheduled in both host modes. The complete owning file passed 57 tests under
Qt 6.12 offscreen after the Section and stand-in changes. After the separate
optional Preferences theme-choice guard, the 29-case Section, field-ripple,
edge, layout and folding cohort passed with branch coverage. Those results
bind to distinct source revisions; the receipt lists each without combining
their counts.

An attempted Xvfb run crashed in pytest-qt's `qapp` fixture before its first
test. Its raw log is retained as a native setup failure, not counted as a
successful application test or as evidence about the hosted fault.

Run `python verify.py --current` for filesystem payload and source checks, or
`python verify.py --git --current` after committing the archive. The hosted
artifact ZIP includes its original source provenance and is checked for ZIP
integrity and exact Section source identity.
