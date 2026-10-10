================================================================================
CODECOV FOR spaCR, WITH A README BADGE
================================================================================

Status:    DONE 2026-10-10 (nightly); main gets its first upload on the
           next push to main. See the 2026-10-10 note.
Requested: 2026-10-10 — "configure https://app.codecov.io/github/EinarOlafsson/
           spacr/new for spacr and add a badge when you are done"
Owner:     Home (CI); WS (README badge in the localized READMEs)
Related:   N43/F288 (green CI and 90% per-module coverage gate); N681 (README
           test-count badges)

--------------------------------------------------------------------------------
WHAT THE STATE IS
--------------------------------------------------------------------------------

spaCR measures coverage in the tests workflow (12 coverage shards, a per-
module 90% gate via tools/verify_module_coverage.py, and a
spacr-module-coverage-report-<run> artifact), but nothing is uploaded to
Codecov and the repository is not set up there (the requested URL is
Codecov's onboarding page for the repo). The README has no coverage badge.

--------------------------------------------------------------------------------
WHY IT MATTERS
--------------------------------------------------------------------------------

Coverage is only visible inside workflow artifacts. Codecov gives a public
coverage history, per-file views, PR comments and a badge, so users and
contributors can see test coverage at a glance.

--------------------------------------------------------------------------------
WHAT TO DO
--------------------------------------------------------------------------------

1. Combine the 12 shard coverage data files into one coverage.xml (Cobertura)
   in the existing combine step of .github/workflows/tests.yml (or the
   coverage job in _pytest-suite.yml), for nightly and main pushes.
2. Upload with codecov/codecov-action (pinned major version), flags per
   branch, fail_ci_if_error false so a Codecov outage never turns CI red.
   Authentication: CODECOV_TOKEN repository secret (the maintainer copies the
   upload token from the Codecov repo settings page; never print or commit
   it). Use tokenless upload only if Codecov accepts it for this public repo.
3. Add codecov.yml: project and patch status informational (the per-module
   90% gate stays the enforcing check), ignore tests/, docs/, tools/ and
   generated i18n catalogs, matching the coverage .coveragerc scope.
4. After the first successful upload on main and nightly, add the badge to
   README.rst next to the existing badges, linking to
   https://app.codecov.io/github/EinarOlafsson/spacr; WS adds it to the 9
   localized READMEs and regenerates them with their tool.

--------------------------------------------------------------------------------
HOW TO KNOW IT WORKED
--------------------------------------------------------------------------------

- A tests run on nightly shows a successful Codecov upload step.
- https://app.codecov.io/github/EinarOlafsson/spacr shows the commit with a
  coverage percentage matching coverage.py's combined total (within 0.5 pp).
- The README badge renders a percentage, not "unknown".

--------------------------------------------------------------------------------
DELIBERATELY NOT DONE
--------------------------------------------------------------------------------

- Codecov status checks do not block merges; the existing per-module gate
  remains the single enforcing coverage check.

--------------------------------------------------------------------------------
2026-10-10 NOTE (Home, Claude): BUILT AND VERIFIED ON NIGHTLY
--------------------------------------------------------------------------------

- .github/workflows/tests.yml, job coverage-combine: the combine step now has
  id `combine`; after the ratchet gate, "Write Cobertura coverage.xml for
  Codecov" runs `coverage xml` on the same combined .coverage for pushes to
  nightly and main (if: always() and the combine step succeeded, so a red
  ratchet still uploads), then "Upload combined coverage to Codecov" runs
  codecov/codecov-action@v7 with disable_search, flags = branch name,
  fail_ci_if_error false. Authentication: secrets.CODECOV_TOKEN when that
  secret exists, otherwise GitHub OIDC (use_oidc), for which the job now
  has permissions contents: read, id-token: write. No token is needed today.
- codecov.yml: project and patch statuses informational, comment off,
  require_ci_to_pass false. .coveragerc has no omit list (source = spacr),
  so the ignore list only names paths outside that source (tests, docs,
  tools, features, packaging, Notebooks). The i18n catalogs are NOT
  ignored, because coverage.py and the ratchet count them; ignoring them
  would break the "within 0.5 pp of coverage.py" check.
- First upload: tests run 38059494502 (commit 36579507d, includes 735d467fd),
  upload step success (HTTP 200, OIDC). Codecov activated the repository on
  that upload. Codecov reports 98.36% (635 files, 284711 hits, 3046 misses,
  1682 partials); coverage.py's combined total in the same run is 98.77%
  (0.41 pp apart: Codecov counts partial branches as not hit).
- README.rst: |Coverage| badge beside |Test counts|, image
  https://codecov.io/gh/EinarOlafsson/spacr/branch/nightly/graph/badge.svg,
  target https://app.codecov.io/github/EinarOlafsson/spacr. It uses the
  NIGHTLY branch because main has no upload yet (main's badge reads
  "unknown"). Switch to branch/main after the first push to main uploads.
- Left for WS: add the same badge to the 9 localized READMEs and regenerate
  them with their tool.
