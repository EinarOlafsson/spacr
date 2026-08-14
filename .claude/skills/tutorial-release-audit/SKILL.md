---
name: tutorial-release-audit
description: Audit, repair, publish, or accept spaCR tutorial releases. Use for lesson inventory/content reviews, scene and spotlight inspection, narration pronunciation/cadence/freshness, 4K and hosted-media validation, Pages publication, phone-sized live playback, or any request to rerun and preserve tutorial quality controls.
---

# Tutorial Release Audit

Treat the editable workspace as source and the spaCR Pages bundle as derived:

- source: `/mnt/firecuda2/Claude/toxoplasma_projects/tutorials`
- derived: `docs/source/_extra/tutorials`

Never repair a derived video, poster, caption, or catalog directly. Update the
source, render there, run its release gates, then publish with
`tools/publish_tutorials.py` from the editable workspace.

## Audit sequence

1. Read `references/release-contract.md` for the exact release matrix and
   acceptance commands.
2. Reconcile the current spaCR module registry with the lesson catalog. Refuse
   missing, duplicate, unknown, or stale module keys.
3. Review every lesson as a bounded input -> action -> result workflow. Check
   current UI, useful Preview/Search demonstrations, concise narration, and
   semantic spotlight geometry.
4. Recreate scene stills from committed videos with
   `python tools/sample_tutorial_frames.py --all --output <audit-dir>`. Review
   the generated manifest and images; do not commit the large audit directory.
5. Run the editable workspace's visual, player, pronunciation, narration, and
   strict audio-release tests before publishing.
6. Publish locally first, validate the spaCR media budget/tests, then upload
   hosted artifacts. Compare source and derived hashes.
7. After the main-only Pages deployment, run
   `python tools/verify_tutorial_live.py --browser --json <report.json>`.
   Preserve the JSON report and screenshot with the release evidence when an
   instruction or release record requires durable proof.

## Fail closed

- Do not accept file existence as audio validation; decode every supported
  track and verify timing, dead air, freshness, and render provenance.
- Do not accept a screenshot because it looks polished; verify that the scene
  teaches a real current control or result and that its narration names it.
- Do not accept a local player test as live publication proof. Match the live
  index/player/catalog hashes to the committed bundle and exercise hosted
  narration in a phone-sized browser.
- Do not bypass the localization/docs audit to deploy Pages. Record an
  unrelated gate as a blocker and rerun after its owner repairs it.
- Keep instruction evidence truthful: report exact command counts, Sphinx exit
  status and warnings, hosted commit identity, repository commits, and live
  URLs.

## Preserve evidence efficiently

The committed 1440p silent masters and hosted timing sidecars are the durable
screen source. `sample_tutorial_frames.py` deterministically recreates compact
scene stills and hashes them, avoiding a second 700+ MiB keyframe archive.
Commit small manifests/reports when needed; store bulk audit stills outside the
repository.
