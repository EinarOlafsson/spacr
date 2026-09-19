"""Everything in spaCR that is addressable by name, and how to rank it.

Instruction 422. The search field beside the Help menu is only as good as
this: a user who knows what a thing is CALLED should reach it without knowing
which module owns it or which collapsed heading it was filed under.

THE INDEX IS NEVER A HAND LIST. Every row here is derived from something that
already decides the answer:

======================  ====================================================
kind                    where the rows come from
======================  ====================================================
``module``              :data:`spacr.qt.app.APPS`, the live app registry
``setting``             :func:`~spacr.qt.screens.settings_model.resolve_default_settings`
                        and :func:`~spacr.qt.screens.settings_model.categories_for_app`
                        per module, described by
                        :func:`~spacr.qt.screens.settings_model.get_tooltips`
``api``                 ``API_ENTRIES`` in :mod:`spacr.qt.help_api_index`,
                        generated from the published API manifest
``preference``          ``PREFERENCE_ENTRIES``, walked out of the source of
                        the preferences dialog
======================  ====================================================

A setting that cannot be found is then a fact about a generator rather than
about whether somebody remembered to add a row, which is the whole argument of
the instruction.

ADDING A KIND IS REGISTERING A PROVIDER, not editing a switch. A provider is
``() -> Iterable[HelpEntry]``; :func:`register_provider` adds one and
:func:`build_index` calls each of them behind its own guard, so a provider
that raises costs its own rows and nothing else. What a result DOES when it is
opened is registered the same way, in :mod:`spacr.qt.help_search`.

IMPORTING THIS COSTS NOTHING; BUILDING THE INDEX COSTS A SECOND. Every
registry is imported inside the provider that reads it, so the module itself
pulls in neither Qt nor ``spacr.settings``. Building it measured 1.1 s and
13,000 rows on this tree, which is far too much for the GUI thread --
:mod:`spacr.qt.help_search` runs :func:`build_index` on a worker and this
module stays testable without a display.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.qt.help_index")

#: The kinds this module ships providers for, in the order a tie between two
#: equally good matches is broken. A module is the coarsest thing a user can
#: ask for and the cheapest to be wrong about; an API entry is the finest, and
#: there are ten thousand of them, so it goes last.
KIND_ORDER: Tuple[str, ...] = ("module", "setting", "preference", "api")

#: How much a kind is worth when two entries score the same. Small enough that
#: it never outranks a better textual match.
_KIND_BONUS = {kind: (len(KIND_ORDER) - i) * 0.01
               for i, kind in enumerate(KIND_ORDER)}


@dataclass(frozen=True)
class HelpEntry:
    """One addressable thing, and what it takes to get back to it.

    :ivar kind: which provider produced it; also which opener knows what to
        do with it. One of :data:`KIND_ORDER`, or a kind a caller registered.
    :ivar title: the identifier a user would type -- a setting key, a module
        name, a dotted API symbol, a preference label.
    :ivar subtitle: where it lives. "in Mask ▸ Cell Segmentation" for a
        setting, "Preferences ▸ Animation" for a preference.
    :ivar description: what it does, in the language the user thinks in. It
        is matched on, which is what makes "touching" find
        ``merge_edge_pathogen_cells``.
    :ivar payload: what the opener needs and nothing more -- an app key, a
        setting key, a dotted symbol, a tab object name.
    """

    kind: str
    title: str
    subtitle: str = ""
    description: str = ""
    payload: Dict[str, str] = field(default_factory=dict)

    @property
    def haystack(self) -> str:
        """Title, subtitle and description, lowercased, for matching."""
        return f"{self.title}\n{self.subtitle}\n{self.description}".lower()


Provider = Callable[[], Iterable[HelpEntry]]

_PROVIDERS: Dict[str, Provider] = {}


def register_provider(name: str, provider: Provider) -> None:
    """Add or replace a source of entries.

    :param name: the provider's name; registering the same name twice
        replaces the first, which is what lets a test swap a cheap provider
        in for an expensive one.
    :param provider: ``() -> Iterable[HelpEntry]``.
    """
    _PROVIDERS[str(name)] = provider


def provider_names() -> Tuple[str, ...]:
    """The registered providers, in registration order."""
    return tuple(_PROVIDERS)


def build_index(names: Optional[Sequence[str]] = None) -> List[HelpEntry]:
    """Run every provider and collect what they know.

    Each provider is guarded on its own: a search field that loses its
    settings because the API manifest could not be read is worse than one
    that quietly offers fewer kinds.

    :param names: providers to run; all of them when ``None``.
    :returns: every entry, in provider order.
    """
    wanted = list(_PROVIDERS) if names is None else [
        n for n in names if n in _PROVIDERS]
    out: List[HelpEntry] = []
    for name in wanted:
        try:
            rows = list(_PROVIDERS[name]())
        except Exception:
            LOG.debug("the %s provider could not be read", name, exc_info=True)
            continue
        out.extend(rows)
    return out


#: How far apart the letters of a fuzzy match may be spread before it stops
#: being a match. MEASURED, not chosen: without a ceiling, "clldiam" matched
#: `spacr.qt.folder_metadata.save_filename_map` -- seven letters scattered
#: over forty -- and ranked it above `cell_diameter`, because a long enough
#: name contains almost any short sequence of letters. Three times the term
#: keeps `clldiam` -> `cell_diameter` (span 9 for 7 letters) and drops that.
_FUZZY_SPAN_LIMIT = 3


def _subsequence_span(term: str, text: str) -> Optional[int]:
    """How wide a window of ``text`` the letters of ``term`` fit in, in order.

    The fuzzy half of the match: "clldiam" finds ``cell_diameter`` and
    "prprcs" finds ``preprocess``. A tight span is a better match than a
    loose one, so the width is what the score is built from rather than a
    bare yes -- and a span wider than :data:`_FUZZY_SPAN_LIMIT` times the
    term is not reported at all.

    :param term: the query term, lowercased.
    :param text: the haystack, lowercased.
    :returns: the span in characters, or ``None`` when the letters are not
        all there in order and close enough together.
    """
    if not term:
        return 0
    start = -1
    position = 0
    for letter in term:
        found = text.find(letter, position)
        if found < 0:
            return None
        if start < 0:
            start = found
        position = found + 1
    span = position - start
    if span > _FUZZY_SPAN_LIMIT * len(term):
        return None
    return span


@lru_cache(maxsize=None)
def _name_parts(title: str) -> Tuple[str, str, Tuple[str, ...]]:
    """A title split the three ways a query can address it.

    ``spacr.qt.screens.mask`` is addressed by all of it, by ``mask``, and by
    any of its words; ``cell_diameter`` by all of it and by ``cell`` or
    ``diameter``; ``Image UMAP`` by ``umap``. Splitting on dots, underscores
    and spaces alike is what makes one rule serve a dotted symbol, a snake
    case setting key and an English module name.

    Cached because the split is the same for every keystroke and there are
    eleven thousand titles.

    :param title: the entry's title.
    :returns: ``(whole, tail, words)``, all lowercased.
    """
    whole = title.lower()
    tail = whole.rpartition(".")[2]
    words = tuple(w for w in re.split(r"[^a-z0-9]+", whole) if w)
    return whole, tail, words


def _term_score(term: str, entry: HelpEntry) -> Optional[float]:
    """How well one query term matches one entry, or ``None`` for no match.

    Five bands, and the ordering between them is the whole ranking: the name
    a user typed beats a word of it, which beats the name they half-typed,
    which beats a word from a description. Matching the description at all is
    what the instruction asks for -- a user who does not know the exact name
    is the user who needs this most -- but a description hit must never
    outrank a name hit, or typing a setting's own key buries it under the
    twenty settings whose help text mentions it.

    :param term: one whitespace-separated query term, lowercased.
    :param entry: the candidate.
    :returns: a score, higher being better, or ``None``.
    """
    whole, tail, words = _name_parts(entry.title)
    if whole == term:
        return 100.0
    if tail == term or term in words:
        return 80.0
    if whole.startswith(term):
        return 60.0 + 10.0 * len(term) / max(len(whole), 1)
    if tail.startswith(term) or any(w.startswith(term) for w in words):
        return 50.0 + 10.0 * len(term) / max(len(tail), 1)
    if term in whole:
        return 40.0
    span = _subsequence_span(term, tail) if len(term) >= 3 else None
    if span is not None:
        return 20.0 + 10.0 * len(term) / max(span, 1)
    described = f"{entry.subtitle}\n{entry.description}".lower()
    if term in described:
        return 10.0
    return None


def score(entry: HelpEntry, terms: Sequence[str]) -> Optional[float]:
    """The entry's score for a whole query, or ``None`` when a term misses.

    Terms are ANDed, the same rule the per-module settings search uses:
    typing a second word must narrow, because otherwise typing more is a way
    of getting more results, which is the opposite of what typing means.

    :param entry: the candidate.
    :param terms: lowercased query terms.
    :returns: the summed score plus the kind tiebreak, or ``None``.
    """
    total = 0.0
    for term in terms:
        one = _term_score(term, entry)
        if one is None:
            return None
        total += one
    return total + _KIND_BONUS.get(entry.kind, 0.0)


#: How many rows of any one kind a result list may hold. THE LIST IS MIXED
#: AND HAS TO STAY MIXED: there are 10,326 API symbols against 39 modules, so
#: a plain top-40 by score is an API list with the occasional module in it,
#: and the instruction's whole point is that one query returns several kinds
#: at once. A cap per kind is how the small kinds keep their seat.
PER_KIND_LIMIT = 8


def search(index: Sequence[HelpEntry], query: str,
           limit: int = 40,
           per_kind: Optional[int] = PER_KIND_LIMIT) -> List[HelpEntry]:
    """The best matches for ``query``, best first.

    :param index: what :func:`build_index` returned.
    :param query: raw text from the search box.
    :param limit: how many rows to return in total.
    :param per_kind: how many rows of one kind may be returned; ``None``
        lifts the cap, which is what a test asking "is it in there at all"
        wants.
    :returns: matching entries, highest score first; ties are broken by kind
        and then by title, so the same query always lists the same way.
    """
    terms = str(query or "").lower().split()
    if not terms:
        return []
    scored: List[Tuple[float, int, str, HelpEntry]] = []
    for entry in index:
        value = score(entry, terms)
        if value is None:
            continue
        rank = KIND_ORDER.index(entry.kind) if entry.kind in KIND_ORDER \
            else len(KIND_ORDER)
        scored.append((-value, rank, entry.title.lower(), entry))
    scored.sort(key=lambda row: row[:3])
    out: List[HelpEntry] = []
    per: Dict[str, int] = {}
    for _value, _rank, _title, entry in scored:
        if per_kind is not None and per.get(entry.kind, 0) >= per_kind:
            continue
        per[entry.kind] = per.get(entry.kind, 0) + 1
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def _visible_apps() -> List[Tuple[str, str, str]]:
    """``(key, name, description)`` for every module a user can open.

    Read through :func:`spacr.qt.app.app_is_visible` so a module the
    maturity preference hides is not offered by search either -- a result
    that opens nothing is the dead link of the module kinds.
    """
    from .app import APPS, app_is_visible

    out: List[Tuple[str, str, str]] = []
    for key, name, desc, _section in APPS:
        try:
            if not app_is_visible(key):
                continue
        except Exception:
            LOG.debug("could not ask whether %r is visible", key, exc_info=True)
        out.append((str(key), str(name), str(desc)))
    return out


def module_entries() -> List[HelpEntry]:
    """One entry per module in the live registry.

    :returns: ``kind="module"`` entries carrying ``{"app": key}``.
    """
    out: List[HelpEntry] = []
    for key, name, desc in _visible_apps():
        out.append(HelpEntry(
            kind="module",
            title=name,
            subtitle="Module",
            description=f"{desc} {key}".strip(),
            payload={"app": key},
        ))
    return out


def setting_entries() -> List[HelpEntry]:
    """One entry per module a setting appears in, plus its API consumers.

    ONE ROW PER MODULE, which is the maintainer's answer to the open
    question the instruction left: a setting in four modules is four rows,
    each naming its module and the category it sits under, rather than one
    row that then asks which. The user already knows which module they meant;
    a row that asks is a second click for information the list could have
    shown.

    THE API HALF OF A SETTING IS NOT EMITTED HERE. ``SETTING_CONSUMERS``
    names every addressable function that reads a key, and
    :func:`api_entries` folds that into the API row for each of those
    functions -- one row per symbol, saying which settings it reads. Emitting
    them from both providers listed ``spacr.io.preprocess_img_data`` once for
    itself and once for every setting it reads, which was eleven identical
    rows in one result list.

    :returns: ``kind="setting"`` entries carrying ``{"app", "key",
        "category"}``.
    """
    from .screens.settings_model import (
        categories_for_app, get_categories, get_tooltips,
        resolve_default_settings,
    )

    tips = {}
    try:
        tips = get_tooltips()
    except Exception:
        LOG.debug("no setting tooltips could be read", exc_info=True)
    shared = {}
    try:
        shared = get_categories()
    except Exception:
        LOG.debug("no setting categories could be read", exc_info=True)

    out: List[HelpEntry] = []
    for key, name, _desc in _visible_apps():
        try:
            defaults = resolve_default_settings(key)
        except Exception:
            LOG.debug("no defaults for %r", key, exc_info=True)
            continue
        try:
            categories = categories_for_app(key, shared)
        except Exception:
            LOG.debug("no categories for %r", key, exc_info=True)
            categories = {}
        placed: Dict[str, str] = {}
        for title, keys in categories.items():
            for setting in keys:
                if setting in defaults:
                    placed.setdefault(setting, str(title))
        for setting in defaults:
            category = placed.get(setting, "")
            where = f"{name} ▸ {category}" if category else name
            out.append(HelpEntry(
                kind="setting",
                title=str(setting),
                subtitle=where,
                description=str(tips.get(setting, "")),
                payload={"app": key, "key": str(setting),
                         "category": category},
            ))
    return out


def _settings_each_symbol_reads() -> Dict[str, List[str]]:
    """Invert ``SETTING_CONSUMERS`` into ``symbol -> [setting, ...]``.

    :returns: dotted symbol -> the settings it reads, sorted.
    """
    try:
        from .help_api_index import SETTING_CONSUMERS
    except Exception:
        LOG.debug("the generated setting consumers are missing", exc_info=True)
        return {}
    out: Dict[str, List[str]] = {}
    for key in sorted(SETTING_CONSUMERS):
        for module, qualname in SETTING_CONSUMERS[key]:
            out.setdefault(f"{module}.{qualname}", []).append(key)
    return out


def api_entries() -> List[HelpEntry]:
    """One entry per public API symbol, saying which settings it reads.

    The settings half is what makes an API row findable by the name of a
    setting: searching ``cell_diameter`` offers the setting's own row in each
    module that exposes it AND the entry of each function that takes it,
    which is what the instruction asks a setting query to return.

    :returns: ``kind="api"`` entries carrying ``{"symbol": dotted name}``,
        and ``"reads"`` when the symbol consumes settings.
    """
    try:
        from .help_api_index import API_ENTRIES
    except Exception:
        LOG.debug("the generated API index is missing", exc_info=True)
        return []
    reads = _settings_each_symbol_reads()
    out: List[HelpEntry] = []
    known = set()
    for symbol, summary in API_ENTRIES:
        known.add(symbol)
        settings = reads.get(symbol, ())
        subtitle = "API reference"
        description = summary
        if settings:
            subtitle = "API reference — reads " + ", ".join(settings[:4])
            description = f"{summary} Reads {', '.join(settings)}."
        payload = {"symbol": symbol}
        if settings:
            payload["reads"] = " ".join(settings)
        out.append(HelpEntry(kind="api", title=symbol, subtitle=subtitle,
                             description=description, payload=payload))
    for symbol in sorted(set(reads) - known):
        out.append(HelpEntry(
            kind="api", title=symbol,
            subtitle="API reference — reads " + ", ".join(reads[symbol][:4]),
            description=f"Reads {', '.join(reads[symbol])}.",
            payload={"symbol": symbol, "reads": " ".join(reads[symbol])},
        ))
    return out


#: The one page of Preferences that is not always built. It exists only while
#: the fractal backdrop is on, so its rows are offered only then -- a result
#: that opened Preferences and then could not find its own tab would be the
#: preference kind's version of a dead link. The generator turns the backdrop
#: on so that these 13 rows are IN the generated table; this is what decides
#: whether they are in the INDEX on a given machine.
_CONDITIONAL_TABS = {"PreferencesTabFractal": "spaceout_enabled"}


def _tab_exists(object_name: str) -> bool:
    """Whether the preferences page ``object_name`` is built in this process.

    :param object_name: the page's object name.
    :returns: True unless a conditional page's condition is off.
    """
    condition = _CONDITIONAL_TABS.get(object_name)
    if condition is None:
        return True
    try:
        from . import theme

        return bool(getattr(theme, condition)())
    except Exception:
        LOG.debug("could not ask whether %s is built", object_name,
                  exc_info=True)
        return False


def preference_entries() -> List[HelpEntry]:
    """One entry per labelled row of the preferences dialog.

    :returns: ``kind="preference"`` entries carrying ``{"tab", "label"}``,
        where ``tab`` is the page's object name.
    """
    try:
        from .help_api_index import PREFERENCE_ENTRIES
    except Exception:
        LOG.debug("the generated preference index is missing", exc_info=True)
        return []
    return [
        HelpEntry(kind="preference", title=label,
                  subtitle=f"Preferences ▸ {tab_title}",
                  description=tip,
                  payload={"tab": object_name, "label": label})
        for label, tab_title, object_name, tip in PREFERENCE_ENTRIES
        if _tab_exists(object_name)
    ]


register_provider("module", module_entries)
register_provider("setting", setting_entries)
register_provider("preference", preference_entries)
register_provider("api", api_entries)
