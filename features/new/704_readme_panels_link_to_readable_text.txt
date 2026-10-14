================================================================================
README PANELS LINK TO READABLE TEXT
================================================================================

Status:    specified and source-audited; implementation not started
Requested: maintainer's Home takeover queue, 2026-10-13: every README text
           panel image links to its readable text version. Where a panel
           already links to a page, its text starts the linked page, so
           clicking continues where the panel leaves off.
Owner:     Home (generators and publication); Work (translation review)
Related:   tools/build_readme_images.py, packaging/generate_readme_visuals.py,
           tools/build_model_cards.py, tools/build_score_cards.py

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

1. Give every prose, table and list-row text panel a clickable, readable text
   destination. Cover English and all nine README translations. Preserve the
   text's structure, facts, table values and useful continuation links.
2. Existing linked owned pages begin their substantive content with the panel
   text. Shared or external destinations need an owned text-first continuation
   that retains the original detail URL; one shared page cannot begin with
   several different panel texts. Do not replace installer/download buttons
   with an informational detour.
3. Keep the current visual design: 13.6 px body text, 16 px titles, phone and
   desktop picture variants, teal-rim dark-grey score cards with blue bars,
   and two cards per row. Keep model score pictures clickable, with complete
   readable score text/values first at their owned destinations. Phone and
   desktop label differences remain an explicit responsive-text constraint.
4. Generate readable outputs from the same authoring data as their images.
   Regenerate README, model pages and derived outputs with their normal tools;
   do not hand-edit generated artifacts or maintain another factual copy.
5. Publish and verify actual GitHub/PyPI links and destination leading content,
   including all locale routes. Copy the resulting approved README/docs
   outputs to main using the established documentation-only procedure.

--------------------------------------------------------------------------------
ACCEPTANCE
--------------------------------------------------------------------------------

- Every generated text panel is inside an appropriate working link; no text
  panel is left unlinked or given an empty/invented destination.
- Each destination's first substantive content contains the associated panel
  text and preserves its structured values. Verify rendered behavior, not
  merely an anchor count or a lexical source match.
- Existing detail and installer links remain usable; shared destinations have
  distinct faithful text-first continuations where required.
- Desktop/phone layout and typography remain visually consistent; accessible
  text and link behavior work on actual GitHub/PyPI and published pages.
- Normal generation and owning checks agree on all ten locale output sets;
  modified tests verify real per-panel leading-content and route behavior.

2026-10-14 00:59 EDT — Home registered the existing authorized queue item.
Read-only preflight binds 83 source/output files to candidate a744a9fda2.
English inventory: 65 prose/table/list-row panels, 46 unlinked; all ten locales
have 46 unlinked text pictures. All 93 English pictures are mapped separately
from navigation buttons and cover art. Exact generator seams and acceptance
plan: features/data/704_panel_text_link_preflight_2026-10-14/. Home independently
verified all 83 source bytes and four artifact identities. This is preparation,
not generated, rendered, hosted or whole-item acceptance. Implementation stays
after release and the held branches in the maintainer's queue order.
