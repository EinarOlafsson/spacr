# Qt recipe and Make Masks branch receipt, 2026-10-07

This is a bounded source-bound check on commit
`b87a3c38aa09936c157c4bded6627c6e6342f713`, under a 4 GiB cgroup,
Qt offscreen and hidden CUDA. `affected-125.log` is the unedited combined
stdout/stderr from the four affected test files and five existing right-sweep
neighbors: 125 passed. `focused-9-coverage.json.gz` is the complete raw
pytest-cov branch JSON for the nine newly added tests; all nine passed.
`manifest.json` hashes both raw files and each changed source/test input.

The provided older 392-shard coverage JSON had two missing `recipes.py`
branches against a zero-branch baseline. Both were executed by the new valid
unsaved-recipe preview and externally deleted save/reload tests. The same
older report had 35 missing `make_masks.py` branches against a 28-branch
baseline. Seven of those old arcs were executed by focused user-behavior
tests: competing buttons during Divide/Merge, a right press in the image
letterbox, restoring an unlisted saved ensemble model, cancelled Browse,
zero-object uncertainty, and an unreadable ranking batch that must not
overwrite valid scores. The old-to-current arc mapping in `manifest.json`
accounts for the five source lines inserted before old line 2598; negative
function-exit targets shift by five as well.

The only product edit prevents a competing Right press from sweeping a mask
object while a Draw outline is in progress. Before the edit, the real-event
regression removed 427 existing-object pixels and emitted two edits; after,
the existing object remains unchanged and the outline is one edit. The
focused coverage JSON executes both outcomes of this new guard. Existing
standalone right-sweep neighbors pass.

This focused coverage overlap is evidence for the old missing branches; it
is not a full numerical ratchet run or a hosted CI verdict. It changes no
threshold, exclusion or serial-memory guard.
