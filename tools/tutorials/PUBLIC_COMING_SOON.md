# Coming soon in the interactive suite

The website source at `docs/source/_extra/tutorials/` now contains 77
entries: 71 playable lessons and six text-only Coming soon screens.
They are part of the existing player, not a separate preview page:

| Entry | Location |
| --- | --- |
| Map Barcodes | Main modules |
| Embeddings | Main modules |
| Model Compare | Submodules → Make Masks |
| Model Zoo | Submodules → Make Masks |
| Investigate Hit | Submodules → Regression |
| OPS | Submodules → Mask |

The screens have localized copy in all 14 catalog languages, request no
video or narration, and cannot count as completed lessons. Language and
caption selectors remain accessible. Existing media for held lessons is
retained on disk, not presented as a successful demonstration.

This is a **website-source update, not proof of deployment**. The public
site remains unchanged until the normal docs workflow successfully deploys
it. A nightly push builds but does not automatically publish Pages. Do not
bypass the localization audit or publish all nightly docs incidentally.

This change deliberately does not install the refreshed candidate narration:
that media remains private. All 71 existing playable lesson bodies, voice
choices, media URLs and their narration/caption pairings are preserved.
Only current GUI navigation metadata is normalized. The preservation hashes
for all 14 catalogs are in `public-coming-soon-integration.json`.

Reproduction: preserve a baseline copy of the website's pre-integration
`catalog/` and `index.html`, then run `integrate_public_coming_soon.py`
with `--baseline`, `--destination`, `--player` pointing to the candidate's
`web/`, and `--report`. Do not use the modified destination as its baseline.

Browser verification (Playwright and Chromium required):

```sh
python tools/tutorials/verify_public_coming_soon.py --output /tmp/tutorial-public-checks
```

This tests 84 screen/language cases, widths 320/390/768/1440, and real
existing English playback followed by cancellation on entering a placeholder.
`--mutate-playability` deliberately breaks the served player and must fail;
it never edits repository source. The successful local run's evidence is in
`/tmp/spacr-public-coming-soon-checks/`; it is not a live-site verification.
After deployment, run `tools/verify_tutorial_live.py` for the exact public
inventory, cache keys, and source-byte comparison.
