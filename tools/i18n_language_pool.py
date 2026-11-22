"""Run one independent piece of i18n work per locale, optionally in parallel.

The landing pipeline validates the same review evidence once per locale in
several steps (catalog build and audit, caption arrivals, review scope), and
the locales do not share state. :func:`map_languages` runs such a function
for every locale either in this process or in spawned worker processes, and
always returns the outcomes in the order of the locales it was given, so a
result never depends on which worker finished first.

Library callers get one worker unless they ask for more; only a command-line
entry point reads :func:`language_workers`, which takes the count from
``SCHED_THREADS`` (the thread grant of the shared job scheduler) and defaults
to one. With one worker everything runs in the calling process, exactly as
before this module existed, so tests that patch a builder's module globals
keep seeing their patches.

The worker functions live here rather than in the tools themselves because a
tool runs as ``__main__`` and a spawned worker cannot import functions from a
``__main__`` module by name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class Outcome:
    """What one locale's work produced: a value, or the exception it raised.

    :param value: the function's return value when it succeeded.
    :param error: the exception it raised, or ``None``.
    """

    value: object = None
    error: BaseException | None = None

    def unwrap(self) -> object:
        """Return the value, or raise the exception the work raised.

        :returns: the value of a successful outcome.
        """
        if self.error is not None:
            raise self.error
        return self.value


def language_workers(default: int = 1) -> int:
    """Return the worker count a command-line run may use.

    :param default: the count when ``SCHED_THREADS`` is unset or invalid.
    :returns: a positive worker count.
    """
    try:
        return max(1, int(os.environ.get("SCHED_THREADS", "")))
    except ValueError:
        return max(1, default)


def _outcome(function: Callable[[str], object], language: str) -> Outcome:
    """Run ``function(language)`` and capture its result or exception.

    :param function: the per-locale work.
    :param language: the locale.
    :returns: the outcome.
    """
    try:
        return Outcome(value=function(language))
    except Exception as error:
        return Outcome(error=error)


def map_languages(
    function: Callable[[str], object],
    languages: Iterable[str],
    *,
    workers: int = 1,
) -> list[Outcome]:
    """Run ``function`` once per locale and return outcomes in locale order.

    Exceptions are captured rather than raised so a caller can raise them at
    exactly the point where a serial loop would have met them.

    :param function: a picklable, module-level callable taking a locale
        (``functools.partial`` of one is fine).
    :param languages: the locales, in the order the results are wanted.
    :param workers: the number of processes; one or fewer runs in-process.
    :returns: one :class:`Outcome` per locale, in the given order.
    """
    from functools import partial

    languages = list(languages)
    workers = min(int(workers), len(languages))
    if workers <= 1:
        return [_outcome(function, language) for language in languages]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(partial(_outcome, function), languages))


def review_scope_counts(docs: Mapping[str, str], language: str) -> tuple[int, int]:
    """Count one locale's reviewed runtime records and reviewed API blocks.

    :param docs: the current English API docstrings.
    :param language: the locale.
    :returns: ``(runtime records, API blocks)``.
    """
    from build_documentation_i18n import reviewed_api_block_translations
    from build_i18n_catalogs import reviewed_runtime_translations

    runtime_count = len(reviewed_runtime_translations(language))
    return runtime_count, len(reviewed_api_block_translations(docs, language))


def runtime_review_translations(
    sources: Mapping[str, object] | None, language: str,
) -> dict[str, str]:
    """Load one locale's reviewed runtime translations.

    :param sources: a caller-owned source snapshot, or ``None`` for the
        builder's own (cached) loader.
    :param language: the locale.
    :returns: the reviewed source-to-translation map.
    """
    import build_i18n_catalogs as builder

    if sources is None:
        return builder.reviewed_runtime_translations(language)
    return builder._reviewed_runtime_translations(language, sources=dict(sources))


def catalog_audit_failures(sources: Mapping[str, object], language: str) -> list[str]:
    """Return one locale's runtime catalog audit failures.

    :param sources: the canonical English sources.
    :param language: the locale.
    :returns: the failure lines, in the order the audit finds them.
    """
    import build_i18n_catalogs as builder

    return builder._audit_language_failures(sources, language)


def source_bound_review_snapshot(language: str) -> tuple[dict[str, object], dict[str, str]]:
    """Load one locale's reviewed runtime translations from fresh sources.

    This is what ``reviewed_runtime_translations`` computes, returned together
    with the source snapshot it was validated against, so a caller can reuse
    it wherever it would load the same locale against equal sources.

    :param language: the locale.
    :returns: ``(sources, reviewed source-to-translation map)``.
    """
    import build_i18n_catalogs as builder

    sources = builder.canonical_sources()
    return sources, builder._reviewed_runtime_translations(language, sources=sources)
