# Handoff notes for Home

Items the WS tutorial lane left for Home while Home was unreachable.

## 2026-10-25: for Home (Home unreachable since the account switch)

1. **OPS example section is never re-shown (blocks tutorial 76).** Reproduced on nightly 7883b5ee11.
   - **Symptom:** on the OPS page (Mask → OPS), the example section with "Load test data…" stays hidden. isVisibleTo(parent) is False even at level All, with Modified off and an empty query.
   - **Cause:** settings_search._apply_section_state hides sections with 0 setting rows while narrowing (Essentials). This section holds only prose and a button, and nothing re-shows it once narrowing ends.
   - **Suggested fix:** never hide sections without setting rows, or restore all section visibility when the filter clears. Add a test with a prose-only section across Essentials → All.
2. **WS pushed one publication just before the hold** (12: 3a187ec48a, 8ed61bd68e), docs and pin only.
