# Automatic main and nightly documentation

Pushing to either `main` or `nightly` starts the **docs** workflow for that
branch. It builds the branch's API, guides and committed tutorial player and
publishes it to that channel's own host, so the two channels never share a
size budget:

- Main: https://einarolafsson.github.io/spacr/ (GitHub Pages, main only)
- Nightly preview: https://einarolafsson-spacr-docs-nightly.static.hf.space/
  (Hugging Face static Space `einarolafsson/spacr-docs-nightly`)

The publisher generates a `/nightly/` stub: `/spacr/nightly/<page>` redirects to
the same page on the Space. This requires a successful main deployment with
the split-host workflow; committing the workflow on nightly alone does not
replace the existing public Pages site. A main deploy fails before upload if the Pages
site exceeds 900 MB (`PAGES_LIMIT` in `tools/publish_docs_channels.py`); the
step summary reports the size. The nightly upload replaces the Space's files
in one `create_commit` operation pinned to the inspected parent revision, so
removed pages disappear and a concurrent update refuses instead of overwriting
an unseen revision. The host's `.gitattributes` is retained when it is absent
from the assembled site. It needs
the repository secret `HF_DOCS_TOKEN`: a fine-grained Hugging Face token with
write access to that one Space.

To select another static Space, set both repository variables `HF_DOCS_SPACE`
(`owner/name`) and `HF_DOCS_URL` (its matching HTTPS `.static.hf.space/` URL).
The workflow rejects a partial or mismatched pair before publishing and resolves
one target for both assemblers, the uploader and both verification steps. With
neither variable set, the address above remains the default. Configure the new
target only after its hosted verification and token permissions are confirmed;
changing these variables does not migrate existing sites, bookmarks or tokens.

Avoid replacing this with a batched `upload_folder`: a quota failure after an
earlier batch can leave files from different documentation builds. Successful
homepage and receipt checks establish those checked files, not the identity of
every file in a site left by a historical partial upload. A quota refusal is a
failed publication; it is not fixed by a successful build or an older live page.

After uploading, `verify-space-tree --site <assembled-site> --space <owner/name>
--revision <full-commit>` checks the complete immutable repository snapshot:
expected paths, file sizes, Git blob identities and LFS content identities.
Only a remote-only `.gitattributes` is exempt. Keep the disposable local artifact
stable during this check. It does not establish public-host freshness or fetch
every public file; `verify-nightly` separately checks the deployed receipts and
homepage, and migration acceptance also checks representative deep links.

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
the browser falls back to English. Editable source review records are retained in the pinned public translation
repository for repair. Translation completion is never a publication prerequisite.

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

## Check the existing CI token without rebuilding documentation

The nightly workflow can verify `HF_DOCS_TOKEN` against the accepted static
Space independently of a documentation build:

```bash
gh workflow run docs.yml --ref nightly \
  -f mode=verify-host-access \
  -f access_space=einarolafsson/spacr-docs-nightly-v2
```

This mode skips the normal resolve/build/deploy chain. It reads the current
immutable Space tree and its existing README, verifies the README's Git content
identity, then makes an authenticated commit-endpoint request with only those
same README bytes and that exact parent commit. The returned immutable tree
must match every original file identity and size. A no-content-change history
commit may be recorded; no site file is removed, old history rewritten or paid
hardware requested. The normal SDK upload shortcut is deliberately bypassed
because it can skip unchanged files without reaching the commit endpoint.

The job uses only the existing repository secret, serializes with nightly
publication, and records public repository/commit identities rather than token
or exception contents. A denied write or changed tree fails the job. Success
proves this token could access that target at the checked time; it does not
change the canonical host variables, deploy main Pages or verify CDN contents.
This check does not require copying a broader local token into GitHub.

## Tutorials follow their branch's committed catalog

Each channel uses `docs/source/_extra/tutorials` from its pinned checkout.
Pushing a ready lesson's catalog, player and media references publishes it on
nightly; merging that commit into main publishes it on main. Changed narration
scripts do not automatically create new recordings. Private authoring output
becomes available only after its verified candidate and media references are
committed. Missing voices must not be advertised as available.

Narration and 4K recordings keep their pinned external media revisions. The
publisher stores local video/poster bytes under content hashes in each site's
`_media/`; each channel retains its own lesson mapping, so a changed nightly
recording cannot replace the main recording. Each site's `channels.json`
records its exact source commit.

Use `tools/tutorials/publish_release_candidate.py` to verify and promote a ready
media subset. Finishing every translation and voice is not required before
publishing independently verified lessons.

The isolated access check reports a fixed, nonsecret ``operation`` on failure:
``metadata``, ``tree``, ``readme``, ``commit`` or ``readback``. The original
exception class and HTTP status are retained; exception messages, credentials
and request URLs are omitted. A 403 at an early read boundary does not prove
that the commit endpoint was reached. Check the reported boundary before
adjusting the selected Space permissions. The original run38743415875 predates
this field and cannot identify its failing boundary retrospectively.

## Translation review source inputs

Before local source checks or documentation generation, run
`python tools/bootstrap_translation_reviews.py fetch`. The Docs and source-test
workflows restore the same immutable public Git inputs before their readers.
No private development-archive access is required. Offline cache use, preserved
local edits and external review authoring are described in
[the translation input guide](../i18n/TRANSLATION_INPUTS.md).
