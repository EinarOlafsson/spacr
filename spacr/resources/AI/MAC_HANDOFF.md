# Mac session handoff — 2026-10-21

The maintainer has started this session to complete current Mac-related work. Read `AGENTS.md` first, then the latest requirements and trailing notes in the owning items below. Use a fresh fetch of `origin/nightly` in an owned Mac worktree; do not reset or clean the maintainer's checkout. Commit and publish finished work to nightly with descriptive messages. Linux `/mnt/...` paths in historical notes refer to the Home host; use the Mac paths permitted in rule 37.

## Priority and ownership

1. **703 and 646: install, update, restart and uninstall.** Follow their detailed Mac sections. Item 703 is now complete at the maintainer-accepted Mac scope (3a45ba05c053); do not reopen its explicitly waived newer-candidate rerun or held Windows/Linux lanes. Continue with outstanding 646 checks. Required update behavior: one Install/Update action runs in the background, then an explicit Restart opens the new version through the startup window and main window. No uninstall/reinstall terminal interrogation. Verify the actual old/new versions, app bundle, launcher/interpreter, preserved user data and Preferences > Performance uninstall scope/options. Protect real user data: use disposable installations and explicit test-owned paths for destructive checks.
2. **673: True 3D.** Verify the currently available Mac backends, visible rendering, picking, dragging, fallback and responsiveness on a real display. The current viewer remains untested by the maintainer. Older Intel-Mac Metal/OpenGL measurements in the item are historical evidence, not acceptance of later changes. Do not claim Vulkan or Direct3D validation on a Mac.
3. **696, 697 and 707: Make Masks interaction.** Read each item's current native UI checklist: zoom/diameter, disconnected strokes, layout/footer/tabs at both required window sizes. Fix actual failures and preserve segmentation/undo and existing label identities. Accessibility item 655 is outside this handoff; do not implement or reopen it. Preserve separately recorded acceptance.
4. **444: sound and Preferences.** Test repeated Preferences opens and normal quit with sound enabled on the real audio stack. Home has just repaired repeated shutdown return values; retain that fix. Report any actual native hang or thread teardown fault before broad changes.
5. **686: complete at the accepted website scope.** Laptop performance is accepted qualitatively; the retained published/mobile/browser checks and current ten-locale release links complete this item. No new Mac browser matrix, measured FPS or future-release dispatch is required. Future release synchronization belongs to the release owner; report a concrete regression if encountered.

Detailed acceptance steps belong in these existing item files under `features/new/`: 703_updates_replace_old_application_versions_and_keep_user_data.txt; 646_clean_uninstall_purge.txt; 673_make_masks_3d_editable_anchors_and_surfaces.txt; 696_magnification_default_diameter_and_auto_diameter.txt; 697_each_disconnected_brush_stroke_is_a_new_object.txt; 707_make_masks_titled_button_groups_and_collapsible_bars.txt; 444_sound_on_wedges_the_preferences_dialog.txt; 686_a_free_project_website_with_rich_effects.txt.

## Other existing Mac follow-ups

These are the previously recorded Mac obligations, not a new broad acceptance matrix. Read each item's new Mac section for prerequisites, exact workflow and minimal result reporting. Preserve accepted N703, F655 and N686 scope.

- [305 — installed wheel/sdist and native trace follow-up](../../../features/new/305_the_installed_package_loads_and_runs_optimally.txt): optional follow-up only; the maintainer-accepted DONE decision stands. Identify the actual frozen artifacts and installed module paths; reuse its native trace with 641 when compatible.
- [420 — real provider install and authentication](../../../features/new/420_the_setup_screen_installs_the_ai_provider_it_offers.txt): test the offered Mac installer, embedded sign-in/browser return and READY state. Accounts, credentials, network and permissions are genuine prerequisites; do not infer vendor authentication from local stand-ins.
- [F423 — real isolated segmenter install](../../../features/future/423_third_party_segmenters_in_the_model_zoo.txt): preserve alpha gating and accepted Linux work; verify install and a small actual adapter inference on supported Mac backends without changing spaCR's environment. Ordinary tutorials remain excluded.
- [429 — real external-volume confirmation](../../../features/new/429_a_stack_file_spacr_wrote_is_one_it_can_read.txt): existing code/docs are complete; check actual metadata sidecars on a real external filesystem. Distinguish an independent Mac check from the reporter's confirmation.
- [641 — native per-module timing](../../../features/new/641_performance_audit_2026_10_04.txt): CPU audit acceptance stands. Reuse existing tools on a quiet real display, report actual registry keys and cold/warm first-paint timings, and retain unmet targets honestly.

## Current Home work to preserve

Home owns general GitHub CI (43), documentation correctness (112), the measured README checkout-size correction, and the shared thumbnail-worker close/park fix. The thumbnail fix and its normal API/Help translations are published on nightly, with nine passing lifecycle tests and fourteen API/Help checks. Fetch that completed publication rather than implementing it twice. Mac-specific failures discovered during native testing belong to this Mac session; record ownership before editing overlapping files.

The latest published version is 1.5.1.5; 1.5.1.6 is not published yet. Distinguish an explicitly identified nightly/candidate test from an actual published-version upgrade. A successful no-op on an already-current version does not establish an actual upgrade, but 703's newer-candidate rerun was explicitly waived; do not create a new acceptance gate. Item 708 has only its existing Work-owned tutorial follow-up and is not reopened by this handoff. Home is completing corrected-source CI before release preparation.

## Acceptance and reporting

For each test record the tested commit/version, macOS version, Intel/Apple Silicon, relevant backend and concise pass/fail/remaining result in its owning item (at most five dated trailing lines). Reuse existing tests and normal generators; no evidence packets or broad inventories. Close a whole item only when all its required acceptance is met; a successful Mac check does not establish Windows/Linux or cluster acceptance. Whole-item completion triggers the Progress List table as specified in AGENTS.md.

The maintainer's standing approval covers routine current-goal decisions. Hold new questions until 2026-10-21 21:10 UTC (17:10 Detroit), unless the maintainer explicitly invites them sooner. Missing hardware, credentials or scientific truth remain unresolved facts. Great Lakes tasks 699/702 are still untested and remain with their cluster owner; they are not made complete by Mac desktop checks.

Shared catalog ownership: Home owns item 717 runtime-caption and public-API catalog publication through final pinned acceptance. Coordinate changes to that item's public docstrings and captions with Home until closure; Mac native/private work and work on other items can continue.
