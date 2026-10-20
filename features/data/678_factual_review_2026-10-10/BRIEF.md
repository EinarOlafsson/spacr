# N678 factual-language review — shared brief for every agent

Repository checkout (git worktree, do NOT commit, push, stash, rebase or switch branches):
  /media/carruthers/mnt3/claude/spacr-564
Python env: /home/carruthers/anaconda3/envs/spacr/bin/python

## The request (from the maintainer, item features/new/678_objective_factual_user_facing_language_review.txt)
Replace wording that is vague, promotional, formulaic, metaphorical or nonsensical with objective
descriptions of actual functionality. Style references: the official NumPy, pandas, SciPy,
scikit-image, scikit-learn and Numba documentation — a short factual summary followed by concrete
behaviour; inputs, parameter meanings, defaults, units, outputs and limitations where they help.
The maintainer explicitly rejected: "Image analysis in spaCR follows your pictures from pixels to
answers." Describe operations, inputs and outputs instead of metaphors or claims that the software
"supplies answers".

## What to change and what to leave alone
- CHANGE: metaphors ("tells its story", "the road", "journey", "think of this as a map"),
  promotional words (powerful, seamless, effortless, intuitive, easy, simply, just, robust when it
  means "good" rather than the statistical term), filler, slogans, vague claims ("handles
  everything", "makes analysis easy"), anthropomorphism ("spaCR learns/knows/decides"), and any
  claim you cannot verify in the code.
- VERIFY every factual claim you write against the implementation (grep/read spacr/ code, settings
  in spacr/settings.py, spacr/resources/module_workflows.json). If a sentence is factually WRONG,
  fix it and record the evidence (file:line). Never invent behaviour, defaults or numbers.
- KEEP: technical terms used in their technical sense ("robust z-score", "robust standard
  deviation", "robust rank aggregation", "easy examples" in focal loss), all instructions and
  technical distinctions, warnings, accessibility notes.
- NEVER change: code blocks, commands, literals in ``double backticks``, file names, URLs, roles
  (:doc:, :ref:, :class:...), setting names, API names, module names, and app UI names (button,
  menu, screen labels must stay exactly as the app shows them), headings' anchors/labels, tables'
  structure, image/figure directives, generated blocks (anything between markers such as
  "BEGIN GENERATED"/"END GENERATED", or the installer download tables maintained by
  packaging/release.py). Do not reflow paragraphs you do not change (keeps translation catalogs
  stable). Do not rewrite a sentence merely to make it different.
- OUT OF SCOPE (other agents/sessions are changing these areas — do not edit text about them):
  themes, animations and Preferences > Performance; Make Masks; Gate Editor; Edit > Undo/Redo and
  keyboard shortcuts; storage backends/settings (DuckDB/Parquet/PostgreSQL); figure export and
  figure integrity. If a paragraph in your files is about one of these, leave it and note it.
- Edit ONLY the files assigned to you. Do not run the full test suite or build the docs.
  You may run quick greps and read code freely.
- Keep RAM use small (no heavy jobs).

## Output (required)
Write /media/carruthers/mnt3/claude/n678/findings/<your-agent-id>.json:
{
  "agent": "<id>", "files": {"<path>": {"paragraphs_read": N, "paragraphs_changed": M}},
  "changes": [{"file": ..., "before": "<exact old sentence(s)>", "after": "<new>",
               "kind": "promotional|metaphor|vague|filler|factual-error|precision",
               "evidence": "<file:line or reason>"}],
  "left_as_is_notable": [{"file": ..., "text": ..., "reason": ...}],
  "out_of_scope_skipped": [{"file": ..., "text": ..., "area": ...}]
}
"paragraphs_read" must be the real number of prose paragraphs you read in full (not skimmed).
Your final message: totals (read/changed per file), the 5 most important changes, and anything
you were unsure about. Honest counts matter more than volume of edits.
