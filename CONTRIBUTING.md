# Contributing to spaCR

Thank you for considering a contribution. spaCR is research software: people
use it to decide what their experiments mean, so the bar here is less "does it
work" than "can the result be trusted, and does the code say honestly what it
did". Everything below follows from that.

- **Questions and bug reports** → [open an issue](https://github.com/EinarOlafsson/spacr/issues/new/choose)
- **Security problems** → do *not* open an issue; see [SECURITY.md](SECURITY.md)
- **Behaviour in the community** → [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Branches

| branch | what it is |
|---|---|
| `nightly` | where development happens. Branch from here, open pull requests against here. |
| `main` | the release branch and the default branch. It advances at a release, not per change. |

A push to `main` that changes `setup.py` starts the release workflow and
publishes to PyPI. Please never target `main` directly.

## Setting up

spaCR supports **Python 3.9 through 3.15** (`>=3.9,<3.16,!=3.14.1` — 3.14.1 is
excluded for a specific upstream defect).

```bash
git clone https://github.com/EinarOlafsson/spacr.git
cd spacr
conda env create -f environment.yaml     # or your own venv
conda activate spacr
pip install -e . --no-build-isolation
```

Then verify which tree you are actually running, because an editable install
can point somewhere else entirely:

```bash
python -c "import spacr; print(spacr.__file__)"
```

If that prints a path other than your checkout, fix it before you measure
anything. A check run as `python /some/other/dir/script.py` puts *that
script's* directory on `sys.path`, never your working directory, and will
happily verify a different copy of spaCR than the one you edited.

## Running the tests

```bash
python -m pytest tests/                       # the non-Qt suite
xvfb-run -a python -m pytest tests/qt/        # Qt needs a display
python -m pytest -p no:randomly tests/...     # for anything order-sensitive
```

Notes that will save you time:

* **Do not add `-q` to `pytest.ini`.** It already explains why in a comment:
  verbosity is cumulative, and the `-q` you type on top of a configured one
  becomes `-qq`, which suppresses the pass/fail summary entirely. A run then
  prints a row of dots, exits 1, and looks fine.
* **Headless Qt refuses static modals** by design — `QMessageBox.information`
  and `QInputDialog.getText` raise in tests, because a modal runs its event
  loop in C++ and hangs the run. Patch them with `monkeypatch.setattr`.
* **A green test suite is not evidence a feature works.** If your change
  touches something a user clicks, open the application and press it.

## Commits and pull requests

* One logical change per commit, and **always pass explicit paths**:
  `git commit -F - -- path/one.py path/two.py`. A `git commit` with no paths
  commits whatever is staged, which has silently shipped test-only commits here
  before. `git show --stat HEAD` is the check that catches it.
* Write commit messages that say *why*, not just what. The repository's history
  is used as documentation.
* If your change fixes something a user reported, say so in the issue too, with
  the commit SHA.

## Files you must not edit by hand

Several files in this repository are **generated**. Editing them works until
the next build overwrites it, and in the meantime puts unreviewed content into
the tree:

| generated file | what regenerates it |
|---|---|
| `features/00_INDEX.txt` | `tools/build_instruction_index.py` |
| `spacr/qt/i18n_catalogs/*.py` | `tools/build_i18n_catalogs.py` |
| `docs/source/_static/i18n/api/*.json` | `tools/build_documentation_i18n.py` |
| `docs/i18n/readme/*` | `tools/build_documentation_i18n.py` |

To change a translated string, change its **English source** and the reviewed
override, then rebuild. Never type a translation straight into a catalog.

## Captions passed through a helper

`tools/build_i18n_catalogs.py` finds user-facing text by reading the calls it
knows — `setText("…")`, `QLabel("…")`, `tr("…")`. A local helper such as
`self._say("Choose a folder first.")` hides that literal: the extractor sees
the helper forwarding a parameter, records nothing, and the caption stays in
English in every language.

**Route a user-facing caption through a helper only if an extractor rule
exists for it.** The rules are `_HELPER_CAPTION_RULES`, keyed by the calling
module and the helper name, with the argument position that holds the
caption. They are keyed rather than matched by name because the same helper
name means different things in different modules: one `_say(text, state)` takes
a style key as its second argument, and another shows data verbatim and must
never be translated. Add the rule in the same change as the helper, then
check that the parameter-position test in
`tests/test_a_helper_does_not_hide_a_caption_from_the_catalog.py` passes.

## Adding a public module

Two steps that are easy to miss and both turn CI red:

1. Add it to `spacr/__init__.py::_SUBMODULES`, or every compatibility-matrix
   cell fails on `test_smoke.py::test_lazy_loader_matches_files`.
2. Run `python tools/build_documentation_i18n.py --sources-only`, which writes
   only `en.json`, or the docs job goes red.

## Docstrings and settings

Public classes document their `__init__`, because that is where settings are
accepted and where a user looks for them. A parameter whose wrong value
produces a *plausible wrong answer* rather than an error deserves a sentence
saying so. Tooltips for settings live in `spacr.settings.tooltips`, and note
that it is **not complete on import** — six pipelines register their own keys
when their module is imported.

## Prose that reaches a translation catalog

Public docstrings, setting tooltips and UI captions are machine-translated
into nine languages. The build rejects a block that comes back as English or
loses its meaning. You only get that verdict after about an hour of building
and auditing, so write catalog text in a shape the model can translate. Code
comments and the notes under `docs/notes/` are not translated, so these rules
do not apply to them.

* **One idea per sentence.** A sentence that holds a claim, its reason and an
  exception becomes three sentences. Dense, precise prose is the house style,
  and the rule does not ask you to cut content. It asks you to split it.
* **Put a verb around every identifier.** A run of bare names, such as
  `ops_geometry, ops_objects, ops_reads, ops_barcodes`, gives the model
  nothing to translate, and the audit reads it as untranslated English. Write
  a sentence that says what the names are: "The storage contract has two
  halves. In `measurements.db`, which is authoritative: …".
* **Do not make an instruction number or a phase label the subject.** Avoid
  forms like "372 states the contract:", "B2: run a segmenter over each
  window" and "C4 samples phenotype channels". The label is an opaque token in
  the place where a translator expects a name, so the model renumbers it or
  drops it. Name the thing the label stands for, such as "The storage contract
  is …" or "Phenotype channels are sampled …". Put the reference in the commit
  message or the ledger file, because neither reaches a catalog. Labels mean
  nothing to a reader of the API reference, so a public docstring is better
  without them in any position.

`python tools/check_translatable_prose.py` checks the last rule in about
fifteen seconds, on the text the model actually receives.
`tests/test_prose_that_reaches_a_catalog_can_be_translated.py` runs it with
the other docstring tests. The first two rules cannot be checked from the
English. Sentence length and identifier density were both measured, and
neither can flag the blocks that failed without also flagging thousands of
blocks that pass. **A
green check is not a green build.** Some failures come from word combinations
or vocabulary that a model declines in one language. Only the build finds
those. When a block fails there, read the model's output for that language
before you rewrite the English.

## The instruction ledger

Work in this repository is tracked in an instruction ledger, not only in
issues. Each item is one file saying what the state is, why it matters, what to
do, how to know it worked, and what was deliberately *not* done.

**It lives on the `nightly` branch, under `features/`** — it is working
material rather than product, so it is deliberately not published on `main`.
`features/00_INDEX.txt` is generated from the folder and
`features/TEMPLATE.txt` is the shape.

You do not need to file one to contribute a fix. If you are picking up
something substantial, reading the relevant item first will usually save you
from re-deriving a decision that has already been made — and several of those
files record measurements that took hours to produce.

## Reporting a result

If your contribution includes a benchmark, an accuracy claim, or a "this is
faster" — include the command, the machine, and the numbers. Two measurements
that disagree are worth more than one that looks right.
