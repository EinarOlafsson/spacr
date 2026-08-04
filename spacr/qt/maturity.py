"""What "alpha" is allowed to mean, and which modules still qualify.

Twenty-six of the shell's forty-three modules shipped labelled **alpha**.
Twenty-six is not a shelf of experiments, it is most of the application, and
a label most of the application wears stops being information: a user who
sees it on Database Browser, Report, Align & Stitch and Plaque Assay alike
learns that it predicts nothing and starts ignoring it — including on the
one module where it would have been worth heeding.

So each of the twenty-six was assessed against evidence that exists in the
repository rather than against how new it feels:

* a dedicated test file with real assertions — the strongest signal, because
  it is the only one that says somebody has pinned the behaviour down;
* a headless path (a ``spacr-run`` module, or an explicit ``cli_note``
  saying why there is deliberately none);
* documentation or a shipped tutorial lesson;
* whether anything else in the codebase depends on it.

Nothing was found that qualified for removal. Every one of the twenty-six
has an implementation of three hundred to three and a half thousand lines, a
dedicated test file, and either a CLI module or a written reason it is
GUI-only; none of the twenty-six screen modules contains a single
``NotImplementedError``, ``TODO``, ``FIXME`` or "coming soon". The alpha
shelf was not holding unfinished work. It was holding finished work nobody
had gone back to relabel.

The two levels below are therefore what the evidence supports:

``stable``
    A real pipeline or library behind it, hundreds of assertions across
    several test files, and documentation a user can be pointed at.

``beta``
    Real and wired and tested, but missing exactly one of those — usually
    documentation, or time in use. Beta is an honest statement, and it is
    the one that makes the remaining labels worth reading.

Applied through :data:`spacr.qt.app.APP_STAGE` rather than by editing the
table in ``app.py``, so the assessment and its reasons live together in one
file that can be re-read and argued with, and so a module that registers
itself from its own file is corrected the same way as a built-in one.

What an *unassessed* module reads as
------------------------------------

``stable`` is the ABSENCE of a line in ``APP_STAGE``. That is a fine way to
record a sign-off and a bad default: an app registered without a ``stage=``
argument — a new screen, a plugin, a module whose author simply did not
think about it — inherits the highest label in the system by saying nothing
at all, and ``app_stage()`` answers ``"stable"`` for a key nobody has ever
looked at.

Twenty modules landed after the assessment above. Each of them happens to
pass ``stage=STAGE_ALPHA`` at its own registration, so each is labelled
correctly — but only because twenty separate authors remembered, and the one
who forgets is silently promoted rather than silently demoted.

So :func:`apply` has a second phase. Every registered app that appears in
none of the three assessment tables below and carries no explicit stage is
written into ``APP_STAGE`` as :data:`UNASSESSED_STAGE` — alpha — which is
what "nobody has checked this one yet" means. It is a *default*, not a
demotion: a module that declares beta keeps beta, and an assessment recorded
here always wins over both.

Eight apps had been inheriting stable in exactly that way — the seven
original core-pipeline modules and Recruitment — so they were assessed too.
:data:`AFFIRMED` records "looked at, already in the right place" the way
:data:`PROMOTIONS` records "looked at, moved". Absence from ``APP_STAGE``
now means one of those two things, and both are written down in this file.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

LOG = logging.getLogger("spacr.qt.maturity")


#: app key -> (new stage, why). Ordered by section, the way the sidebar is,
#: so a reader can check a whole section at once.
#:
#: Every reason names the evidence, not an opinion. "Well tested" is not a
#: reason; "769 assertions across four files, a spacr-run module and
#: tutorial lesson 32" is one, because the next person to disagree can go
#: and look.
PROMOTIONS: Dict[str, Tuple[str, str]] = {
    # -- Data --------------------------------------------------------------
    "align": ("stable",
              "769 assertions across four test files, a `spacr-run align` "
              "module with its own pre-flight validation rules, and tutorial "
              "lesson 32."),
    "convert": ("stable",
                "518 assertions across four test files, `convert_folder` "
                "behind `spacr-run convert` with a registered defaults "
                "entry, and tutorial lesson 35."),
    "foreign": ("stable",
                "A 3,539-line importer with 750 assertions, pre-flight rules "
                "and input-key checks, `spacr-run` aliases and tutorial "
                "lesson 36."),
    "batch": ("stable",
              "The documented headless path for half the other modules is "
              "`batch.run_queue`; 279 assertions and six documentation pages "
              "rest on it."),
    "db_browser": ("stable",
                   "The largest screen in the shell (2,696 lines) with the "
                   "largest dedicated test file in the suite — 159 tests, "
                   "543 assertions — and tutorial lesson 34."),
    "external_masks": ("beta",
                       "A real `spacr-run external_masks` pipeline with a "
                       "defaults entry and tutorial lesson 31, but only 49 "
                       "assertions across two small files — the thinnest "
                       "evidence of any CLI-backed module here."),
    "queue": ("beta",
              "Fully wired into MainWindow, but the tests cover the 319-line "
              "engine rather than the screen, which appears only "
              "incidentally in four others."),
    "distributed_jobs": ("beta",
                         "1,018 lines behind a single 8-test file — the "
                         "weakest tests-per-line ratio in the batch; the "
                         "`spacr-remote` CLI carries the real contract."),
    "illumination": ("beta",
                     "Wired end to end — CLI module, validation entry, a "
                     "measure preprocessing hook and registered defaults, "
                     "148 assertions — but it landed in one commit and has "
                     "no documentation or lesson at all."),
    "data_manager": ("beta",
                     "222 assertions and a deliberate no-headless-delete "
                     "design, but no documentation, and a launch crash "
                     "already in its short history."),
    # -- Segmentation models ----------------------------------------------
    "model_compare": ("stable",
                      "363 assertions and a live cross-screen signal — Model "
                      "Zoo's compare button drives it, so shelving it breaks "
                      "another module."),
    "model_zoo": ("stable",
                  "280 assertions, tutorial lesson 22, an integration in the "
                  "plugin SDK documentation, its own drop handler, and it "
                  "feeds Model Compare."),
    # -- Results & QC ------------------------------------------------------
    "plate_view": ("stable",
                   "165 assertions across three focused files (1536-well and "
                   "linked-filter among them), `plate_qc.detect_edge_effect` "
                   "behind it, and tutorial lesson 33."),
    "agreement": ("stable",
                  "130 tests and 431 assertions including a dedicated "
                  "behavioural regression test, tutorial lesson 23, and a "
                  "README section."),
    "train_compare": ("stable",
                      "The most-edited screen in this group, 354 assertions "
                      "across three files including a settings-climb "
                      "regression test, and tutorial lesson 28."),
    "report": ("stable",
               "454 assertions and the widest reach of anything here — ten "
               "documentation pages and twenty cross-module files refer to "
               "it."),
    "classifier_evaluation": ("beta",
                              "The best documentation of the group — its own "
                              "page, the leakage audit and lesson 39 — but "
                              "the screen itself has 5 tests and 21 "
                              "assertions."),
    "run_history": ("beta",
                    "The run_journal layer underneath is settled (226 "
                    "assertions, ten commits); the screen on top of it is 6 "
                    "tests old."),
    "barcode_qc": ("beta",
                   "187 assertions, a CLI module, and automatic invocation "
                   "from the sequencing pipeline — but it landed in a single "
                   "commit and has no documentation."),
    "run_compare": ("beta",
                    "122 tests and 344 assertions, but one commit old, "
                    "undocumented, and nothing else in the codebase depends "
                    "on it yet."),
    # -- Explore -----------------------------------------------------------
    "layer_viewer": ("beta",
                     "112 assertions, translations in nine languages and a "
                     "slot on the Home shelf, but no documentation and a "
                     "library underneath it that is days old."),
    "graph_builder": ("beta",
                      "48 tests and 146 assertions for a 358-line screen is "
                      "unusually heavy coverage, but it has no "
                      "documentation and no dependants."),
    "anndata_export": ("beta",
                       "A real `spacr-run anndata_export` module with "
                       "registered defaults, a validation entry and 239 "
                       "assertions, but one commit old and undocumented."),
    "feature_dict": ("beta",
                     "The 3,434-line library under it is load-bearing — "
                     "report, schema, foreign and anndata_export all import "
                     "it — with 429 assertions; the panel is newer, and its "
                     "app row still escapes the registry parity check."),
    # -- Toxoplasma --------------------------------------------------------
    "invasion": ("stable",
                 "A real `spacr-run invasion` pipeline with pre-flight "
                 "rules, curated settings categories, 238 assertions and "
                 "tutorial lesson 26."),
    # -- Design ------------------------------------------------------------
    "power": ("beta",
              "197 tests and 465 assertions across five files, and the only "
              "module in the Design section — but one commit old with no "
              "documentation."),
}

#: app key -> (stage it already has, why that is right). The apps that were
#: reading ``stable`` through the *absence* of an ``APP_STAGE`` line rather
#: than through anybody's decision.
#:
#: These eight are the oldest thing in spaCR: the seven core-pipeline
#: modules the library was written to run, plus Recruitment. Nothing here
#: moves an app; the table exists so that the second phase of :func:`apply`
#: can tell "signed off" apart from "never looked at", which is the whole
#: difference an empty entry could not express. Same evidence rule as
#: :data:`PROMOTIONS`: countable, and checkable by the next reader.
AFFIRMED: Dict[str, Tuple[str, str]] = {
    "mask": ("stable",
             "The module the library exists for: 20 test files and 1,526 "
             "assertions, three `spacr-run` entry points onto "
             "`preprocess_generate_masks`, a standalone `spacr.app_mask` "
             "window and 17 tutorial files."),
    "measure": ("stable",
                "19 test files and 796 assertions, `spacr-run measure` over "
                "`measure_crop`, a standalone `spacr.app_measure` window, "
                "5 tutorial files — and the seg-QC banner and diameter "
                "panel were both built onto this screen."),
    "annotate": ("stable",
                 "GUI-only by design and documented as such — annotation is "
                 "the one step with no headless meaning — with 12 test "
                 "files, 865 assertions, a standalone `spacr.app_annotate` "
                 "window and 7 tutorial files."),
    "classify": ("stable",
                 "12 test files and 697 assertions over `spacr.deep_spacr`, "
                 "three `spacr-run` entry points, a standalone "
                 "`spacr.app_classify` window and 5 tutorial files."),
    "ml_analyze": ("stable",
                   "8 test files and 637 assertions over `spacr.ml`, with "
                   "2 `spacr-run` entry points; Regression, the hit list "
                   "and the classifier-evaluation screen all read what it "
                   "writes."),
    "map_barcodes": ("stable",
                     "8 test files and 530 assertions over "
                     "`spacr.sequencing`, 2 `spacr-run` entry points, and "
                     "Barcode QC is invoked automatically from the end of "
                     "this pipeline."),
    "regression": ("stable",
                   "8 test files and 780 assertions across `spacr.ml` and "
                   "`spacr.models`, 2 `spacr-run` entry points, and the "
                   "hit list and Report both consume its output."),
    "recruitment": ("stable",
                    "3 test files and 294 assertions over "
                    "`spacr.submodules.analyze_recruitment`, 2 `spacr-run` "
                    "entry points, and a documented parasite-recruitment "
                    "readout older than the Qt shell itself."),
}

#: Modules assessed and deliberately left where they are. Empty, and that
#: emptiness is the finding: see the module docstring. Kept as a named,
#: iterated-over structure rather than a sentence in a comment so that
#: retiring something later is a one-line change with a reason attached,
#: not a rediscovery of this whole exercise.
RETIREMENTS: Dict[str, str] = {}

#: What an app nobody has assessed reads as. Alpha, because that is the
#: label that means "built and reachable, not yet trusted end to end", and
#: an app whose maturity nobody has stated is not one anybody has trusted.
#:
#: Deliberately NOT the empty string and deliberately not ``stable``: the
#: whole point of the second phase of :func:`apply` is that this default is
#: written down in one place and applied, rather than being whatever falls
#: out of ``APP_STAGE.get(key, STAGE_STABLE)``.
UNASSESSED_STAGE = "alpha"


def assessed_keys() -> frozenset:
    """Every app key somebody has actually looked at, in any of the tables."""
    return frozenset(PROMOTIONS) | frozenset(AFFIRMED) | frozenset(RETIREMENTS)


def _registered_keys():
    """The app keys currently in the registry, or ``()`` if it is absent.

    Imported here rather than at module scope for the same reason
    :func:`apply` does it: this module is imported by the launch sequence
    and by tests that never build a registry at all.
    """
    try:
        from .app import APPS
    except Exception:
        LOG.debug("the app registry is not importable", exc_info=True)
        return ()
    return tuple(row[0] for row in APPS)


def unassessed_apps(stages: Dict[str, str] = None, keys=None) -> List[str]:
    """Registered apps nobody has assessed, in registry order.

    An app is unassessed when it appears in none of :data:`PROMOTIONS`,
    :data:`AFFIRMED` or :data:`RETIREMENTS`. Whether it *carries* a stage is
    a different question: a module that declared alpha for itself is
    unassessed and correctly labelled, which is exactly the state most of
    the shelf is in.

    :param stages: unused for the decision; accepted so callers can pass the
        same two arguments they pass :func:`apply`.
    :param keys: the app keys to consider; defaults to the live registry.
    """
    known = assessed_keys()
    return [key for key in (_registered_keys() if keys is None else keys)
            if key not in known]


def apply(stages: Dict[str, str] = None, keys=None) -> List[str]:
    """Write the assessment into the shell's stage table, then fill the gaps.

    Two phases, and they answer two different questions:

    1. :data:`PROMOTIONS` — the apps somebody assessed and moved. Idempotent,
       and it never *demotes*: a module some other code has already promoted
       further than this table says stays where it is. That matters because
       the table is a snapshot of one assessment, and the next assessment
       should not be silently undone by re-importing this module.
    2. Every registered app that is in none of the assessment tables and has
       no line of its own is written in as :data:`UNASSESSED_STAGE`. This is
       what stops a new module inheriting ``stable`` from the absence of an
       entry — see the module docstring. It only ever writes where there is
       nothing, so it cannot overrule an author, a plugin, or phase 1.

    :param stages: the table to write into. Defaults to
        :data:`spacr.qt.app.APP_STAGE`; injectable so a test does not have
        to mutate the live registry.
    :param keys: the app keys phase 2 considers. Defaults to the live
        registry; pass ``()`` to run phase 1 alone.
    :returns: the app keys whose stage this call changed.
    """
    if stages is None:
        try:
            from .app import APP_STAGE
        except Exception:
            LOG.debug("the app registry is not importable", exc_info=True)
            return []
        stages = APP_STAGE

    order = {"alpha": 0, "beta": 1, "stable": 2}
    changed: List[str] = []
    for app_key, (stage, _reason) in PROMOTIONS.items():
        current = str(stages.get(app_key, "stable"))
        if order.get(current, 2) >= order.get(stage, 0):
            continue
        if stage == "stable":
            # Stable is the ABSENCE of a line, not a line reading "stable".
            # ``APP_STAGE`` exists to record what is *not* signed off, and
            # signing an app off is deleting its entry — writing the word in
            # would give the table a second way to say the same thing, which
            # `test_every_app_has_a_stage_and_it_is_written_down_once` exists
            # to prevent.
            stages.pop(app_key, None)
        else:
            stages[app_key] = stage
        changed.append(app_key)

    for app_key in RETIREMENTS:
        try:
            from .app import unregister_app
            unregister_app(app_key)
        except Exception:
            LOG.debug("could not retire %r", app_key, exc_info=True)

    # Phase 2 — the default, made explicit. `stages.get(key)` and not
    # `key in stages` because a table that somehow holds an empty string or
    # a None for a key has not said anything about it either.
    for app_key in unassessed_apps(stages, keys):
        if str(stages.get(app_key) or ""):
            continue
        stages[app_key] = UNASSESSED_STAGE
        changed.append(app_key)
    return changed


def reason_for(app_key: str) -> str:
    """Why ``app_key`` is where it is, or ``""`` when it was not assessed.

    An empty string is a real answer here and the UI is entitled to say it
    plainly: this module is alpha because nobody has checked it, not because
    somebody checked it and concluded alpha.
    """
    entry = PROMOTIONS.get(str(app_key)) or AFFIRMED.get(str(app_key))
    if entry is not None:
        return entry[1]
    return RETIREMENTS.get(str(app_key), "")


def register() -> bool:
    """Entry point for :data:`spacr.qt.SELF_REGISTERING_MODULES`.

    Idempotent — :func:`apply` never demotes, so a second launch in one
    process (the test suite does this) is a no-op rather than a conflict.
    """
    apply()
    return True
