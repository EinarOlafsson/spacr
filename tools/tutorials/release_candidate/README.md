# Tutorial release candidate — updated 12 September 2026

This is the **unpublished, maintainer-approved candidate**, not the current
live tutorial site. `checkpoint.json` identifies the complete on-disk package;
`release-manifest.json` records every web/media file and its SHA-256 hash.

| Contents | Count |
| --- | ---: |
| Fully produced tutorial packages | 71 |
| Coming soon screens | 6 |
| Total navigable entries | 77 |
| Catalog languages | 14 |
| Narration languages / voices per ready lesson | 8 / 50 |
| Verified narration tracks, including retained tracks | 3,550 |

The six unavailable workflows are Map Barcodes, Model Compare, Model Zoo,
Investigate Hit, OPS and Embeddings. They remain reachable through the correct Main modules
or parent-grouped Submodules navigation, but do not load old media, claim a
successful run, offer completion, or inflate the available count. The underlying
application/data issues remain for their owners; the user approved these screens
instead of waiting for those fixes.

All original held recordings and catalogs are preserved. The candidate only
adds availability and current-parent metadata; ready narration and visual
masters are not regenerated for this change. Complete text sources, scripts,
catalogs and evidence are in Git. At the maintainer's request the web videos,
posters, fonts and example downloads are also committed; the 6.5-GiB narration
and 4K package remains in the original workspace and separately copied candidate,
indexed by the committed hashes, pending media-host upload approval.

## Verification and preview

Use the existing isolated tutorial environment, not a new application install:

```bash
/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09/.venv/bin/python \
  tools/tutorials/verify_release_candidate.py \
  /mnt/firecuda2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09/release-candidate-6ttmkjq2
```

For an interactive local preview, add `--serve`. It binds only to localhost,
prints the URL, supports media seeking and serves the complete candidate.
Stop it with Ctrl-C. Git contains the web media, but the complete narration and
4K files are still required from the local candidate or a verified media host.

This rebuilt package includes the narration-paired caption fix and native
caption-track reload fix. Its exact files passed 71 English playback cases
(audio hashes, requested seek positions, video synchronization, paired
transcript and two native caption reloads each), plus 84 Coming soon cases
across all fourteen languages. Both deliberate placeholder-player mutations
failed, with the unchanged player passing before and after. These are browser
checks, not a new native-GUI or listening review. The refreshed Platform
Installers Heart track additionally passed native-caption checks at all
36 sentence midpoints, including both CUDA mentions.

`source-verification-summary.json` records the frozen preservation baseline:
catalogs extracted from Git commit `d2d4c189b`, before the public Coming soon
conversion. Rebuilding now needs `build_release_candidate.py --baseline` pointed
at that extracted catalog directory. Byte-exact retained-lesson checks remain
in force; comparing against today's already-converted public catalogs would
test the wrong baseline. The earlier private candidate is preserved unchanged.

The separately published Heart pronunciation repair belongs to the older live
Platform Installers recording, not this refreshed recording's different script.
This candidate now has its own first-CUDA correction, produced through the
normal renderer and checked in `audio_repairs/platform-heart-refreshed-20260912`.
All 49 other installer voice pairs and the shared movie remain byte-identical.
Do not overwrite the older live repair with this different narration. Technical
phoneme and playback checks are not human pronunciation/listening acceptance.

## Release hold

**Do not publish yet.** The candidate index deliberately uses local relative
media roots. After the maintainer clears the hold:

1. Recheck the manifest and confirm the intended app version still matches
   the recordings. Check the current GUI route inventory again.
2. Upload the candidate's `media_host/` to a new versioned media location,
   preserving the currently live files. Verify all uploaded bytes and pin the
   deployed player to that exact media revision; do not use stale legacy audio.
3. Stage this candidate's `web/` as the tutorial docs payload, replace its two
   local media roots with the verified hosted roots, and update the deployment
   catalog/cache/inventory pins and bundled tutorial index together. Retain
   unrelated tracked historical assets unless separately authorized to remove
   them; measure the actual documentation build, not only this payload.
4. Run the exact deployment's tutorial tests and live verifier before declaring
   it published. Keep the previous media revision available for rollback.

The public-source route test also includes the six Coming soon screens, but
that does not prove Pages deployed them. Passing candidate checks is not a claim
that the live site or all GitHub CI is green. Technical validation also does not constitute native-speaker listening
approval or repair the application defects disclosed in the tutorials.
