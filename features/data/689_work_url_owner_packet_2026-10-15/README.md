# Work-owned N689 URL replacements

This is an **unapplied owner proposal**. Cherry-picking this commit adds only this
portable packet; it does not edit either tutorial tool. Work retains integration
ownership and must preserve any newer source.

Current maintainer reporting rule: use one row per whole numbered item, combine
all subtasks, show ✅ only when the entire item is complete, and express remaining
time in estimated hours. A component delivery is evidence, not a whole-item tick.
Reprint the full grouped table when a whole item closes.

## Exact changes

| Work-owned path | Exact URL suffix retained |
| --- | --- |
| `tools/tutorials/audit_user_walkthroughs.py` | `/tutorials/catalog/lessons_en.json` |
| `tools/tutorials/build_cellpose_training_images.py` | `/cellpose_training.html` |

Replace only the hostname `einarolafsson-spacr-docs-nightly.static.hf.space`
with `einarolafsson-spacr-docs-nightly-v2.static.hf.space` at the two locations
shown in `proposal.patch`. The immutable source pin and each full-file preimage
and proposed SHA-256 are in `owner-receipt.json`; `receipt.json` records the
portable packet validation. The original proposal bytes are retained.

## Owner integration

1. Use your own clean private worktree and read the current versions of both
   files. Compare their SHA-256 with `owner-receipt.json`.
2. If both preimages match, run from that worktree:

   ```sh
   git apply --check features/data/689_work_url_owner_packet_2026-10-15/proposal.patch
   git apply features/data/689_work_url_owner_packet_2026-10-15/proposal.patch
   ```

3. If newer source differs, preserve it. Review each exact old line recorded in
   `owner-receipt.json` and replace only the old hostname at that location. Do not
   restore either file from the old snapshot. If already updated, treat it as a
   no-op; if absent or ambiguous, report the mismatch to Home rather than guess.
4. Inspect the diff: exactly these two hostname replacements, with every path
   suffix and all other content unchanged. Use Work's normal owning validation
   and generation where required; this packet itself does not regenerate media.
5. Commit the source change separately and relay its exact pin to Home. Follow
   the current Work source-publication authorization; this packet does not alter
   that authorization or permit shared-checkout changes.

## Accepted host and limits

The free v2 Space was checked at immutable revision
`732766ad32dd47784cb5c2a15c8c57516aae8252`; representative public acceptance is
committed in `e7e6b4fc4`. That accepted assembled artifact comes from older source
`8e0a1fd`, not the current application head. The project website points to the v2
host. This proposal does not establish current-head docs, tutorial playback,
new recording/media acceptance, main Pages cutover or whole-item N689 completion.
The old Space and its history remain intact.
