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
| `instructions/00_INDEX.txt` | `tools/build_instruction_index.py` |
| `spacr/qt/i18n_catalogs/*.py` | `tools/build_i18n_catalogs.py` |
| `docs/source/_static/i18n/api/*.json` | `tools/build_documentation_i18n.py` |
| `docs/i18n/readme/*` | `tools/build_documentation_i18n.py` |

To change a translated string, change its **English source** and the reviewed
override, then rebuild. Never type a translation straight into a catalog.

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

## The instruction ledger

Work in this repository is tracked in an instruction ledger, not only in
issues. Each item is one file saying what the state is, why it matters, what to
do, how to know it worked, and what was deliberately *not* done.

**It lives on the `nightly` branch, under `instructions/`** — it is working
material rather than product, so it is deliberately not published on `main`.
`instructions/00_INDEX.txt` is generated from the folder and
`instructions/TEMPLATE.txt` is the shape.

You do not need to file one to contribute a fix. If you are picking up
something substantial, reading the relevant item first will usually save you
from re-deriving a decision that has already been made — and several of those
files record measurements that took hours to produce.

## Reporting a result

If your contribution includes a benchmark, an accuracy claim, or a "this is
faster" — include the command, the machine, and the numbers. Two measurements
that disagree are worth more than one that looks right.
