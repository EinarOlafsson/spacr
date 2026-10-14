# Held capture integration with the current Home CI repair

Local commit **fcf4055999a6b5a40fd1875496189bc03709389a** merges held capture
source f7b88b7ce with pinned Home repair eb72a1a85. Its worktree is
`/media/carruthers/mnt3/claude/spacr-capture-integration-20261013`.
The one conflict was release-audit test comments/format around identical
native lesson sets. Resolution matches upstream; test semantics stay intact.
No app, tool or generated-artifact conflicts occurred. The app tree matches
eb72 exactly; the held capture/staging/queue/acceptance tools match f7 exactly.

All **148** canonical/export/original source hashes and lengths match the
existing generated snapshot, including all **145** checkpoint records.
The existing checked-copy tool was used; no metadata was reconstructed.
Historical README/source-summary extras are preserved outside snapshot claims.
The report binds the immutable 6b04 publication through its existing receipt,
manifest and browser evidence. No capture, render or upload occurred.

Capped CPU/offscreen validation passed **165 capture cases**. Three affected
publication files passed **49 cases**, initially skipping the route check
because Node was absent from that process PATH. The skipped check then passed
using the **already installed Playwright Node v24.21.0** from the render
Python environment, with its directory added only to that test process PATH.
No package or installed environment changed. Thus all **215 unique selected
cases** now have passing evidence; original logs retain the initial skip.
Raw logs, JUnit reports, terminal receipts and Node identity are included.
Both tested Python children pinned their import root and asserted the spaCR
source origin before pytest. This is focused compatibility proof, not full CI.

This integration is committed locally and held until Home says release done.
It still lacks the separate Workbench opening repair, so it does not establish
final-app readiness or authorize the failed r28 footage for publication.
CH remains frozen/clean at 29552bae7 and the original ready source remains
f7b88b7ce. After final app source is available, integrate it, validate affected
behavior, then make a fresh complete lesson 20 recording with corrected
telemetry before audio, translations, staging, rendering and publication.

Home has received the checkpoint and quality reports. Its fresh full nightly
CI is 38488240863 on eb72a1a85, confirmed with active jobs. Release is still
pending. The goal remains all 84 existing routes; lesson 43 waits for 20 end to end.
This handoff push contains features only.
