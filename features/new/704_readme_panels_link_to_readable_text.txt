================================================================================
README PANELS LINK TO READABLE TEXT
================================================================================

Status:    implemented and locally validated; hosted publication pending
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

2026-10-14 15:20 EDT — Home-B implemented the assigned generator scope.
All 65 prose/table/list-row panels now open their own faithful readable text
page in each of the ten README languages (650 generated Markdown pages).
The same authoring blocks supply image and text; ordinary generation owns and
checks both, including missing/stale/obsolete text pages and generated blocks.
Original detail URLs remain inline with rebased local paths/fragments, while
installer/download/navigation buttons retain their original direct targets.
The 17 model pages now begin with complete score text and baseline values from
the plotting data; their index begins with the score-grid heading. Full desktop
labels are retained in readable text; phone labels remain their existing shorter
visual forms. No literal responsive-wording equivalence is claimed.

Normal README/model generation completed. The 1,660 existing README panel SVGs,
icon assets and model-score SVGs are byte-identical to the branch base; body/title
sizes, responsive pictures, card colours/bars and two-card rows are unchanged.
The focused six-file cohort passed all 118 cases (zero failures/skips, 20.14s),
including rendered Markdown structure/word order across all ten languages,
local continuation routes, PyPI link conversion, figure currency and existing
presentation contracts. Initial cohort was 115 passed/3 failed: it exposed
Korean emphasis delimiters rendered literally next to Hangul, plus a temporary
fixture-root mistake and model-image ordering contract; all three were fixed.
Text-only correction used the owning generator's compose(render=False) and
page writer, followed by normal model generation; no generated files hand-edited.

Evidence: data/704_readable_panel_implementation_2026-10-14/receipt.json and
the retained initial/final test reports. The dev extra now explicitly declares
markdown-it-py for rendering checks; runtime dependencies are unchanged.
This closes local implementation, NOT whole-item hosted acceptance. Original
Home still owns publication: verify actual GitHub and PyPI links/rendering and
leading content on the published source, including all locale routes, and
promote approved README/docs outputs to main using the established procedure.

2026-10-14 — Home-B verified PyPI sanitization and prepared exact promotion.
The four owning PyPI packaging checks passed in 1.26s. The explicit panel-link
check now uses readme_renderer.rst (PyPI's actual sanitizing renderer), verifying
each of the 65 English panel images retains its own exact readable destination
after rendering setup.py's real long_description. The existing image/alt and
external-link checks also pass. No runtime packaging defect was demonstrated.
setup.py intentionally resolves these links against main, not nightly.

A documentation-only promotion patch contains exactly 678 files: ten READMEs,
650 readable pages, seventeen model pages and their index. All 1,707 image assets
referenced by those READMEs already match main e0bedd1534d71708f65b60a2d9e24b6328d4ce9d
byte-for-byte. The patch applies to an alternate index loaded from that exact
main commit and produces tree 189ef70712fa327c66896502100ad0c681a7ddac without
changing any branch, working tree or remote. Patch, exact file list and source
manifest: /mnt/wd4tb/scratch/home-b-n704-20261014/promotion/.
Receipt: data/704_pypi_sanitizer_acceptance_2026-10-14/receipt.json.
This is renderer verification and promotion preparation, not a PyPI upload,
actual public route acceptance or whole-item completion. Original Home owns
the remaining publication and hosted checks.
