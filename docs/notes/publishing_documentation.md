# Automatic main and nightly documentation

Pushing to either `main` or `nightly` starts the **docs** workflow. It pins both
branch heads, builds each branch's API, guides and committed tutorial player,
and publishes them together without mixing their content:

- Main: https://einarolafsson.github.io/spacr/
- Nightly preview: https://einarolafsson.github.io/spacr/nightly/

A local commit becomes public after it is pushed and the build/deployment
succeeds. No manual documentation dispatch is needed. Include this workflow
change when merging nightly into main so pushes to main use the same policy.
A version bump is not required to update documentation; a version increase
followed by pushing main also triggers the separate package release workflow.
Three- and four-component numeric versions, including `1.5.1.0`, are supported.
The documentation version is read from matching `setup.py` and
`spacr/_version.py` in each checkout, never an older installed distribution.

## Translation incompatibilities are report-only

Every build refreshes and checks its English manifests. Missing, stale and
incompatible localized entries are registered in a JSON artifact and in the
published `translation-compatibility.json`, with the source commit. A locale
whose API source contracts do not match is omitted from the built payload;
the browser falls back to English. The source translation is retained in Git
for repair. Translation completion is never a publication prerequisite.

Named acceptance tests of the real translation corpus are advisory. They still
run and retain full failure diagnostics in `.translation-reports/*.json`,
uploaded by CI. Pytest displays these as report-only expected failures. English
parametrizations and tests of extraction, formatting, escaping and fallback
implementation remain required. The allowlist is explicit in
`tools/pytest_translation_compatibility.py`; it does not waive unrelated tests.
Strict `--audit` commands remain available as diagnostics for translation work.
`tools/report_translation_compatibility.py --full-audit --output report.json`
runs both auditors as report-only checks. An audit error is recorded as an
error, never labeled compatible.

Sphinx warnings/errors, broken English links and malformed publication inputs
still fail a build. A failed build leaves the previously deployed site intact.

## Tutorials follow their branch's committed catalog

Each channel uses `docs/source/_extra/tutorials` from its pinned checkout.
Pushing a ready lesson's catalog, player and media references publishes it on
nightly; merging that commit into main publishes it on main. Changed narration
scripts do not automatically create new recordings. Private authoring output
becomes available only after its verified candidate and media references are
committed. Missing voices must not be advertised as available.

Narration and 4K recordings keep their pinned external media revisions. The
publisher stores identical local video/poster bytes once under content hashes;
each channel retains its own lesson mapping. A changed nightly recording cannot
replace the main recording. Both sites are checked together against the Pages
size budget. `channels.json` records their exact source commits.

Use `tools/tutorials/publish_release_candidate.py` to verify and promote a ready
media subset. Finishing every translation and voice is not required before
publishing independently verified lessons.
