"""Methods and Results sections, written from a run digest the model cannot leave.

The deliverable is two paragraphs of a paper: a **methods** section that says
what was actually done, and a **results** section that says what came out. The
hard part is not the prose. The hard part is that a language model asked to
write about an experiment will produce numbers, and the numbers will be
plausible, and nobody reading the paragraph can tell which of them came from
the run.

So the model never sees the data. It sees a **run digest**: a structured
record of the modules that ran, the parameters they ran with, the counts, the
versions, the QC verdicts and the statistics *already computed* by the rest of
spaCR. It writes prose around those numbers, and then:

* :func:`verify_numbers` extracts every number from what came back and checks
  each one against the digest. A number that is not in the digest — not as a
  value, not as a correct rounding of one, not as a verbatim quote of a
  digest string — is **unsupported**, and
* :func:`check_draft` refuses a draft that carries one. The refusal is the
  guarantee: the model cannot introduce a figure, because a draft carrying an
  invented figure does not get returned as a draft.

Two consequences worth stating plainly. First, the digest is the contract:
anything the prose may quote has to be *in* it, which is why the digest
carries the alpha, the confidence level and the seed as numbers rather than
leaving the model to supply the obvious ones. Second, there is a deterministic
renderer — :func:`render_methods` and :func:`render_results` — that writes the
same sections from the same digest with no model at all. It is what runs when
no AI provider is configured, it is what a rejected draft falls back to, and
it is what the number-provenance tests assert against, because a test that
plants a number in the digest and requires it in the output must not be
testing a stub.

**The caveats are not optional.** A methods section that omits a QC failure,
or the fact that illumination correction did not run, or the ``on_error=skip``
that dropped eleven fields, is not a shorter methods section — it is a wrong
one. :func:`build_digest` collects those into :data:`DIGEST_CAVEATS` and both
renderers, and the prompt, are required to state every one.

Public API::

    from spacr.methods_export import build_digest, render_methods

    digest = build_digest(project="/data/plate7", run_dir=run.dir,
                          results_folder=".../results/pred/ols")
    print(render_methods(digest))
    print(render_results(digest))

The AI half lives in :mod:`spacr.qt.ai.manuscript`, which reuses the console's
existing provider plumbing rather than opening a second client.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple, Union)

__all__ = [
    "ALWAYS_ALLOWED",
    "DIGEST_VERSION",
    "Verification",
    "build_digest",
    "check_draft",
    "digest_numbers",
    "digest_strings",
    "extract_numbers",
    "render_methods",
    "render_prompt",
    "render_results",
    "system_prompt",
    "verify_numbers",
]

LOG = logging.getLogger("spacr.methods_export")

#: Bumped when the digest layout changes incompatibly.
DIGEST_VERSION = 1

#: Structural numbers prose cannot avoid: a count of one, a pair, a
#: percentage's base. They are allowed unconditionally because refusing them
#: would reject "each of the two channels" while catching nothing — every
#: number that carries a CLAIM is larger or more precise than these. The set
#: is deliberately tiny, and everything else a paragraph may quote has to be
#: in the digest.
ALWAYS_ALLOWED: frozenset = frozenset({0.0, 1.0, 2.0, 100.0})

#: Tokens stripped before numbers are extracted, because they are structure
#: rather than claims: a numbered list marker, a dotted version, an ISO date
#: or timestamp, and a long hexadecimal id.
_NOISE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^\s{0,8}\(?\d{1,3}[.)]\s", re.MULTILINE),
    re.compile(r"\b\d+(?:\.\d+){2,}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?"),
    re.compile(r"\b[0-9a-f]{8,}\b"),
    re.compile(r"\bref\s*\d+\b", re.IGNORECASE),
)

#: What a number looks like in prose.
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: Digest keys whose sentences a methods section must state. Collected by
#: :func:`build_digest` and asserted by the tests: a caveat the pipeline knows
#: about and the paragraph omits is a wrong methods section, not a short one.
DIGEST_CAVEATS = "caveats"


def _utcnow() -> str:
    """Current UTC time, ISO-8601 to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Walking a digest
# ---------------------------------------------------------------------------

def _walk(value: Any) -> Iterable[Any]:
    """Yield every leaf of a nested mapping/sequence, containers included."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


#: Digest keys whose values are prose SPACR generated from the numbers
#: elsewhere in the digest, rather than text the run recorded. They are
#: excluded from :func:`digest_strings` on purpose: if quoting a caveat
#: verbatim were enough to license the figures inside it, the number check
#: would pass on a sentence spaCR wrote and prove nothing. Every number in a
#: caveat comes from a numeric field, so excluding them here costs nothing
#: and makes the check real.
_GENERATED_PROSE_KEYS: frozenset = frozenset({DIGEST_CAVEATS})


def _walk_keyed(value: Any, key: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield ``(nearest mapping key, leaf)`` for every leaf."""
    if isinstance(value, Mapping):
        for name, item in value.items():
            yield from _walk_keyed(item, str(name))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk_keyed(item, key)
    else:
        yield key, value


def digest_numbers(digest: Mapping[str, Any]) -> Set[float]:
    """Every number the digest actually asserts.

    Numeric leaves, plus strings that are *entirely* a number (a gene id such
    as ``"233460"`` is a string in the tables and a number in prose). Digits
    embedded in a longer string — a path, a run id, a version — are
    deliberately NOT harvested here: those are handled by
    :func:`digest_strings`, which lets the prose quote them verbatim without
    also licensing every digit inside them as a free-standing figure.

    :param digest: a digest as :func:`build_digest` returns.
    :returns: the numbers, as floats.
    """
    found: Set[float] = set()
    for leaf in _walk(digest):
        if isinstance(leaf, bool):
            continue
        if isinstance(leaf, (int, float)):
            if isinstance(leaf, float) and not math.isfinite(leaf):
                continue
            found.add(float(leaf))
        elif isinstance(leaf, str):
            token = leaf.strip()
            if _NUMBER.fullmatch(token):
                try:
                    found.add(float(token))
                except ValueError:                    # pragma: no cover
                    continue
    return found


def digest_strings(digest: Mapping[str, Any]) -> List[str]:
    """Every non-trivial string the digest carries, longest first.

    Quoting one of these verbatim — a path, a run id, a settings digest, a
    package version, a gene name — is not inventing a number, so
    :func:`verify_numbers` removes them from the text before it looks for
    figures. Longest first so a version is removed whole rather than having
    its prefix eaten by a shorter match.

    :param digest: a digest.
    :returns: the strings, longest first.
    """
    found: Set[str] = set()
    for key, leaf in _walk_keyed(digest):
        if key in _GENERATED_PROSE_KEYS:
            continue
        if isinstance(leaf, str):
            token = leaf.strip()
            if len(token) >= 2 and any(ch.isdigit() for ch in token):
                found.add(token)
    return sorted(found, key=len, reverse=True)


def extract_numbers(text: str,
                    strings: Sequence[str] = ()) -> List[str]:
    """Every number a paragraph asserts, as it was written.

    ``strings`` — usually :func:`digest_strings` — are removed first, so a
    sentence that quotes ``run 8f21c0a3`` or ``/data/plate7`` is not accused
    of asserting ``8`` and ``7``. Structural tokens (list markers, dotted
    versions, ISO dates, long hex ids) go next. What is left is a claim.

    :param text: the prose to check.
    :param strings: substrings that may be quoted verbatim.
    :returns: the number tokens, in the order they appear.
    """
    cleaned = str(text or "")
    for token in sorted(strings, key=len, reverse=True):
        if token:
            cleaned = cleaned.replace(token, " ")
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return _NUMBER.findall(cleaned)


def _supported(token: str, allowed: Set[float]) -> bool:
    """True when ``token`` is one of ``allowed`` or a correct rounding of one.

    Two rules, and the split between them is the whole judgement:

    * **A number written with decimals, or in scientific notation, is a
      measurement quoted to some precision.** ``0.043`` for a q-value of
      ``0.04312`` invents nothing, so the token is accepted when some digest
      number lies within half a unit in its last place — computed from the
      token as written, so ``1.23e+04`` gets a tolerance of 50 and ``0.043``
      gets one of 0.0005.
    * **A bare integer is a count, and a count is exact.** ``517 genes``
      either is the number in the digest or is a different claim. Allowing an
      integer to be a rounding would let ``3`` stand for a coefficient of
      ``2.9013``, which is how a hit count and an effect size end up as the
      same sentence.
    """
    try:
        value = float(token)
    except ValueError:                                # pragma: no cover
        return False
    if not math.isfinite(value):                      # pragma: no cover
        return False
    if value in ALWAYS_ALLOWED or -value in ALWAYS_ALLOWED:
        return True

    body = token.lower()
    mantissa, _, exponent_text = body.partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    try:
        exponent = int(exponent_text) if exponent_text else 0
    except ValueError:                                # pragma: no cover
        exponent = 0
    counted = decimals == 0 and not exponent_text
    tolerance = 0.5 * (10.0 ** (exponent - decimals))

    for candidate in allowed:
        if candidate == value:
            return True
        if counted:
            continue
        if abs(candidate - value) <= tolerance:
            return True
    return False


@dataclass(frozen=True)
class Verification:
    """The verdict on one generated section's numbers.

    :param ok: no unsupported number was found.
    :param checked: how many number tokens were examined.
    :param supported: the tokens that trace to the digest.
    :param unsupported: the tokens that do not. These are the inventions.
    :param missing_caveats: caveats the digest carries that the text omits.
    """

    ok: bool
    checked: int = 0
    supported: Tuple[str, ...] = ()
    unsupported: Tuple[str, ...] = ()
    missing_caveats: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """True when the text is clean."""
        return self.ok

    def problem(self) -> str:
        """One sentence naming what is wrong, or ``""``."""
        parts: List[str] = []
        if self.unsupported:
            listed = ", ".join(sorted(set(self.unsupported))[:8])
            parts.append(
                f"{len(set(self.unsupported))} number(s) in the draft are not "
                f"in the run digest: {listed}")
        if self.missing_caveats:
            parts.append(
                f"{len(self.missing_caveats)} caveat(s) the run recorded are "
                f"not stated")
        return "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy."""
        return {"ok": self.ok, "checked": self.checked,
                "supported": list(self.supported),
                "unsupported": list(self.unsupported),
                "missing_caveats": list(self.missing_caveats)}


def verify_numbers(text: str, digest: Mapping[str, Any], *,
                   require_caveats: bool = False) -> Verification:
    """Check that every number in ``text`` came from ``digest``.

    This is the assertion the whole module exists to make. It is applied to
    what a model returns, and a draft that fails it is not returned as a
    draft — see :func:`check_draft`.

    :param text: the generated prose.
    :param digest: the digest it was generated from.
    :param require_caveats: also require every sentence in the digest's
        ``caveats`` to be represented. Applied to the methods section, which
        is where the caveats belong.
    :returns: a :class:`Verification`.
    """
    allowed = digest_numbers(digest)
    tokens = extract_numbers(text, digest_strings(digest))
    supported: List[str] = []
    unsupported: List[str] = []
    for token in tokens:
        (supported if _supported(token, allowed) else unsupported).append(token)

    missing: List[str] = []
    if require_caveats:
        folded = str(text or "").casefold()
        for caveat in digest.get(DIGEST_CAVEATS, ()) or ():
            key = _caveat_key(str(caveat))
            if key and key not in folded:
                missing.append(str(caveat))

    return Verification(
        ok=not unsupported and not missing, checked=len(tokens),
        supported=tuple(supported), unsupported=tuple(unsupported),
        missing_caveats=tuple(missing))


def _caveat_key(caveat: str) -> str:
    """The distinctive phrase of a caveat, folded, for a containment test.

    The first clause up to the first comma or full stop. Matching the whole
    sentence would require the model to copy it verbatim, which is not what
    is being asked — the requirement is that the FACT is stated, and the fact
    is in the opening clause.
    """
    head = re.split(r"[,.;]", str(caveat).strip(), maxsplit=1)[0]
    return head.strip().casefold()


def check_draft(methods: str, results: str,
                digest: Mapping[str, Any]) -> Tuple[Verification, Verification]:
    """Verify both sections. ``(methods verdict, results verdict)``.

    The methods section additionally has to state the run's caveats.
    """
    return (verify_numbers(methods, digest, require_caveats=True),
            verify_numbers(results, digest))


# ---------------------------------------------------------------------------
# Building the digest
# ---------------------------------------------------------------------------

def _safe(step: str, notes: List[str], fn, *args, **kwargs):
    """Run one collector; record its failure rather than losing the digest.

    A digest assembled from six independent subsystems must not be all-or-
    nothing: a project whose model card is unreadable still has a methods
    section to write, and the missing piece is a note the paragraph can
    honestly carry.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                          # noqa: BLE001
        LOG.info("digest: %s unavailable (%s)", step, exc)
        notes.append(f"{step} could not be read: {exc}")
        return None


def build_digest(*,
                 project: Union[str, os.PathLike, None] = None,
                 run_dir: Union[str, os.PathLike, None] = None,
                 macro_path: Union[str, os.PathLike, None] = None,
                 settings: Optional[Mapping[str, Any]] = None,
                 results_folder: Union[str, os.PathLike, None] = None,
                 metadata_files: Sequence[Union[str, os.PathLike]] = (),
                 regression_type: str = "",
                 hits: Any = None,
                 model_path: Union[str, os.PathLike, None] = None,
                 title: str = "",
                 top_hits: int = 10,
                 extra: Optional[Mapping[str, Any]] = None,
                 ) -> Dict[str, Any]:
    """Assemble everything a methods and results section may quote.

    Every source is optional and every one of them is read defensively: a
    subsystem that cannot answer contributes a note rather than an exception,
    because a digest is written at the END of a long run and losing it to a
    missing model card would be absurd.

    :param project: the project root. Supplies the provenance summary from
        :mod:`spacr.pipeline_graph` and the segmentation QC from
        :mod:`spacr.seg_qc`.
    :param run_dir: a :mod:`spacr.run_journal` run folder. Supplies the
        manifest: versions, timings, seeds, warnings, status.
    :param macro_path: the emitted ``macro.py``; defaults to the one inside
        ``run_dir``. Supplies the per-module steps, the settings each ran
        with and which of them the user actually chose.
    :param settings: settings to read directly, when there is no journal.
    :param results_folder: a regression results folder, for the statistics.
    :param metadata_files: annotation CSVs to join into the hit list.
    :param regression_type: the backend, for how the hit list is ranked.
    :param hits: an already-built :class:`spacr.hits.HitList`, instead of
        reading ``results_folder``.
    :param model_path: a classifier checkpoint whose model card carries the
        held-out metrics.
    :param title: what to call the experiment in the prose.
    :param top_hits: how many hits to carry into the digest.
    :param extra: anything else to record, under ``"extra"``.
    :returns: the digest, JSON-serializable throughout.
    """
    notes: List[str] = []
    digest: Dict[str, Any] = {
        "digest_version": DIGEST_VERSION,
        "generated_utc": _utcnow(),
        "title": str(title or ""),
        "project": (os.path.abspath(os.path.expanduser(os.fspath(project)))
                    if project else ""),
        "spacr_version": _safe("the spaCR version", notes, _version) or "",
        "run": {},
        "environment": {},
        "modules": [],
        "parameters": {},
        "qc": {},
        "classifier": {},
        "statistics": {},
        "hits": [],
        "provenance": {},
        "constants": {"confidence_level_percent": 95},
        "caveats": [],
        "notes": [],
    }

    run_settings: Dict[str, Any] = dict(settings or {})

    if run_dir is not None:
        manifest = _safe("the run journal", notes, _read_manifest, run_dir)
        if manifest:
            digest["run"].update(manifest["run"])
            digest["environment"] = manifest["environment"]
            if not run_settings:
                run_settings = manifest["settings"]

    macro_file = macro_path or (os.path.join(str(run_dir), "macro.py")
                                if run_dir else None)
    if macro_file and os.path.isfile(str(macro_file)):
        steps = _safe("the emitted macro", notes, _read_macro_steps,
                      macro_file)
        if steps:
            digest["modules"] = steps["modules"]
            digest["parameters"] = steps["parameters"]
            digest["run"].setdefault("run_id", steps["run_id"])
            if not run_settings and steps["settings"]:
                run_settings = steps["settings"]

    live = _safe("the active run context", notes, _live_run)
    if live:
        for key, value in live.items():
            digest["run"].setdefault(key, value)

    if run_settings:
        digest["run"].setdefault("n_settings", len(run_settings))
        digest["qc"].update(_illumination(run_settings))
        digest["run"].setdefault("seed", _resolve_seed(run_settings, notes))
        policy = _error_policy(run_settings)
        for key, value in policy.items():
            digest["run"].setdefault(key, value)

    if project:
        segmentation = _safe("the segmentation QC", notes, _segmentation_qc,
                             project)
        if segmentation:
            digest["qc"]["segmentation"] = segmentation
        provenance = _safe("the artifact registry", notes, _provenance,
                           project)
        if provenance:
            digest["provenance"] = provenance

    if model_path:
        card = _safe("the model card", notes, _model_card, model_path)
        if card:
            digest["classifier"] = card

    hit_list = hits
    if hit_list is None and results_folder:
        hit_list = _safe("the regression results", notes, _build_hits,
                         results_folder, metadata_files, regression_type)
    if hit_list is not None:
        digest["statistics"] = _safe("the hit list summary", notes,
                                     lambda: hit_list.summary()) or {}
        digest["hits"] = _top_hits(hit_list, top_hits)

    if extra:
        digest["extra"] = json.loads(json.dumps(dict(extra), default=str))

    digest["notes"] = notes
    digest["caveats"] = caveats_for(digest)
    return json.loads(json.dumps(digest, default=str))


def _version() -> str:
    """The running spaCR version."""
    from .version import get_version

    return str(get_version())


def _read_manifest(run_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The journal manifest of one run, as digest fragments."""
    from . import run_journal

    folder = run_journal.resolve_run_dir(run_dir)
    manifest_path = os.path.join(str(folder), "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        settings = run_journal.load_run_settings(folder)
    except FileNotFoundError:
        settings = {}
    performance = manifest.get("performance") or {}
    seeds = manifest.get("seeds") or {}
    return {
        "run": {
            "journal_run_id": os.path.basename(str(folder)),
            "app_key": manifest.get("app_key", ""),
            "status": manifest.get("status", ""),
            "started_utc": manifest.get("start_utc", ""),
            "ended_utc": manifest.get("end_utc", ""),
            "elapsed_s": manifest.get("elapsed_s"),
            "wall_s": performance.get("wall_s"),
            "process_cpu_s": performance.get("process_cpu_s"),
            "n_input_files": performance.get("input_files"),
            "n_output_files": performance.get("output_files"),
            "seed_declared": seeds.get("declared") or {},
            "warnings": list(manifest.get("warnings") or ()),
            "n_settings": manifest.get("n_settings"),
        },
        "environment": dict(manifest.get("env") or {}),
        "settings": dict(settings),
    }


def _read_macro_steps(macro_file: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The modules that ran, from the script the run emitted.

    :mod:`spacr.macro` parses it rather than importing it, so a digest can be
    built from a macro this process did not write and does not trust.
    """
    from .macro import read_macro

    record = read_macro(macro_file)
    modules: List[Dict[str, Any]] = []
    parameters: Dict[str, Any] = {}
    last_settings: Dict[str, Any] = {}
    for step in record.get("steps", ()):
        step_settings = dict(step.get("settings") or {})
        chosen = {key: step_settings.get(key)
                  for key in step.get("user_set", ())
                  if key in step_settings}
        modules.append({
            "index": step.get("index"),
            "module": step.get("module", ""),
            "entry": step.get("entry", ""),
            "run_id": step.get("run_id", ""),
            "settings_hash": step.get("settings_hash", ""),
            "status": step.get("status", ""),
            "elapsed_s": step.get("elapsed_s"),
            "spacr_version": step.get("spacr_version", ""),
            "n_settings": len(step_settings),
            "n_user_set": len(step.get("user_set", ())),
            "n_defaulted": len(step.get("defaulted", ())),
            "n_outputs": len(step.get("outputs", ())),
            "link": step.get("link", ""),
        })
        parameters[str(step.get("module", f"step{step.get('index')}"))] = chosen
        last_settings = step_settings
    return {"modules": modules, "parameters": parameters,
            "run_id": record.get("steps", [{}])[-1].get("run_id", "")
            if record.get("steps") else "",
            "settings": last_settings}


def _live_run() -> Dict[str, Any]:
    """Whatever the active :mod:`spacr.runctx` context knows, or ``{}``."""
    from . import runctx

    context = runctx.current_run_context()
    if context is None:
        return {}
    payload = context.to_dict()
    report = payload.get("seed_report") or {}
    return {
        "run_id": payload.get("run_id", ""),
        "seed": payload.get("seed"),
        "on_error": payload.get("on_error", ""),
        "on_error_attempts": payload.get("on_error_attempts"),
        "n_skipped": len(payload.get("skipped") or ()),
        "skipped": [
            {key: item.get(key) for key in ("unit", "stage", "reason",
                                            "exc_type")}
            for item in (payload.get("skipped") or ())
        ],
        "seeded": list(report.get("seeded") or ()),
        "seed_caveats": list(report.get("caveats") or ()),
        "deterministic": bool(report.get("deterministic", False)),
        "started_utc": payload.get("started_utc", ""),
    }


def _resolve_seed(settings: Mapping[str, Any],
                  notes: List[str]) -> Optional[int]:
    """The seed the run used, from its settings."""
    from . import runctx

    try:
        return runctx.resolve_seed(settings)
    except Exception as exc:                          # noqa: BLE001
        notes.append(f"the seed could not be resolved: {exc}")
        return None


def _error_policy(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``on_error`` policy the run was configured with."""
    from .runctx import DEFAULT_ON_ERROR, DEFAULT_RETRIES

    return {
        "on_error": str(settings.get("on_error", DEFAULT_ON_ERROR)),
        "on_error_attempts": settings.get("on_error_attempts",
                                          DEFAULT_RETRIES),
    }


def _illumination(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether illumination correction ran, and with what."""
    enabled = bool(settings.get("illumination_correction", False))
    return {
        "illumination_correction": enabled,
        "illumination_model": str(settings.get("illumination_model", "") or ""),
        "illumination_estimator": str(
            settings.get("illumination_estimator", "") or ""),
    }


def _segmentation_qc(project: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The recorded segmentation QC verdict, without re-scoring anything."""
    from . import seg_qc

    found = seg_qc.read_digest(project)
    flags: Dict[str, int] = {}
    n_ok = n_warn = n_fail = 0
    for card in found.scorecards:
        summary = card.summary or {}
        n_ok += int(summary.get("n_ok", 0) or 0)
        n_warn += int(summary.get("n_warn", 0) or 0)
        n_fail += int(summary.get("n_fail", 0) or 0)
        for flag, count in (summary.get("flag_counts") or {}).items():
            flags[str(flag)] = flags.get(str(flag), 0) + int(count)
    return {
        "verdict": found.verdict,
        "headline": found.headline,
        "n_fields": found.n_fields,
        "n_ok": n_ok, "n_warn": n_warn, "n_fail": n_fail,
        "object_types": list(found.object_types),
        "flags_fired": dict(sorted(flags.items())),
        "stale": bool(found.stale),
    }


def _provenance(project: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The whole-project provenance summary, from the artifact registry."""
    from .pipeline_graph import build_graph, stale_summary

    graph = build_graph(project)
    summary = stale_summary(graph)
    return {
        "n_artifacts": summary["n_nodes"],
        "n_current": summary["n_current"],
        "n_stale": summary["n_stale"],
        "n_missing": summary["n_missing"],
        "modules": summary["modules"],
        "verdict": summary["verdict"],
    }


def _model_card(model_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The classifier's card: classes, split rule and held-out metrics."""
    from .deep_spacr import read_model_card

    card = read_model_card(model_path)
    if not card:
        raise FileNotFoundError(
            f"no model card beside {os.path.basename(str(model_path))}")
    held = dict(card.get("held_out") or {})
    return {
        "model_file": card.get("model_file", ""),
        "module": card.get("module", ""),
        "classes": list(card.get("classes") or ()),
        "epochs": card.get("epochs"),
        "split_rule": card.get("split_rule", ""),
        "settings_hash": card.get("settings_hash", ""),
        "spacr_version": card.get("spacr_version", ""),
        "training_set": {
            "n_train": (card.get("training_set") or {}).get("n_train"),
            "class_balance": (card.get("training_set") or {}).get(
                "class_balance") or {},
        },
        "held_out": {
            "n": held.get("n"),
            "accuracy": held.get("accuracy"),
            "f1_macro": held.get("f1_macro"),
            "per_class_accuracy": list(held.get("per_class_accuracy") or ()),
            "class_support": list(held.get("class_support") or ()),
        },
        "warnings": list(card.get("warnings") or ()),
    }


def _build_hits(results_folder: Union[str, os.PathLike],
                metadata_files: Sequence[Union[str, os.PathLike]],
                regression_type: str):
    """The hit list of a regression results folder."""
    from .hits import build_hit_list

    return build_hit_list(results_folder, metadata_files=list(metadata_files),
                          regression_type=regression_type)


def _top_hits(hit_list: Any, limit: int) -> List[Dict[str, Any]]:
    """The top rows of a hit list, as digest rows."""
    rows: List[Dict[str, Any]] = []
    for hit in list(hit_list)[:max(0, int(limit))]:
        rows.append({
            "rank": hit.rank, "gene": hit.gene, "name": hit.name,
            "effect": _finite(hit.effect), "p_value": _finite(hit.p_value),
            "q_value": _finite(hit.q_value), "n_guides": hit.n_guides,
            "n_agree": hit.n_agree, "agreement": _finite(hit.agreement),
            "direction": hit.direction, "flags": list(hit.flags),
        })
    return rows


def _finite(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is not a real number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------

def caveats_for(digest: Mapping[str, Any]) -> List[str]:
    """The sentences a methods section is not allowed to leave out.

    Everything the pipeline KNOWS about the run's limitations, stated as
    prose the paragraph has to carry: which QC flags fired, whether
    illumination correction ran, the seed, the ``on_error`` policy and what
    it dropped, the classifier's held-out metrics, and whether any artifact
    the numbers rest on is stale.

    :param digest: a digest under construction.
    :returns: one sentence per caveat, in the order a methods section would
        state them.
    """
    run = digest.get("run") or {}
    qc = digest.get("qc") or {}
    classifier = digest.get("classifier") or {}
    provenance = digest.get("provenance") or {}
    caveats: List[str] = []

    seed = run.get("seed")
    if seed is None:
        caveats.append(
            "No random seed was set, so the run is not bit-for-bit "
            "reproducible.")
    else:
        caveats.append(f"The random seed was {seed}.")

    policy = str(run.get("on_error") or "")
    skipped = int(run.get("n_skipped") or 0)
    if policy == "skip":
        caveats.append(
            f"The error policy was on_error=skip, so failing units were "
            f"dropped rather than stopping the run; {skipped} were skipped.")
    elif policy == "retry":
        caveats.append(
            f"The error policy was on_error=retry with "
            f"{run.get('on_error_attempts')} attempts per unit.")
    elif policy:
        caveats.append(
            "The error policy was on_error=stop, so any failure ended the "
            "run rather than being skipped.")

    if "illumination_correction" in qc:
        if qc.get("illumination_correction"):
            caveats.append("Illumination correction was applied.")
        else:
            caveats.append(
                "Illumination correction was not applied, so uneven "
                "field illumination is not corrected for.")

    segmentation = qc.get("segmentation") or {}
    verdict = str(segmentation.get("verdict") or "")
    if verdict:
        flags = segmentation.get("flags_fired") or {}
        if flags:
            named = ", ".join(f"{flag} ({count})"
                              for flag, count in list(flags.items())[:5])
            caveats.append(
                f"Segmentation QC returned {verdict}: {named}.")
        else:
            caveats.append(f"Segmentation QC returned {verdict}.")
    if segmentation.get("stale"):
        caveats.append(
            "The segmentation QC scorecard is older than the masks it "
            "describes.")

    held = classifier.get("held_out") or {}
    if held.get("accuracy") is not None:
        caveats.append(
            f"The classifier's held-out accuracy was "
            f"{held['accuracy']:.4g} on {held.get('n')} objects "
            f"(macro F1 {held.get('f1_macro')}).")
    if classifier.get("split_rule"):
        caveats.append(
            f"The held-out split was made by {classifier['split_rule']}.")
    for warning in classifier.get("warnings") or ():
        caveats.append(f"The model card warns: {warning}")

    stale = int(provenance.get("n_stale") or 0)
    missing = int(provenance.get("n_missing") or 0)
    if stale:
        caveats.append(
            f"{stale} registered artifact(s) are stale — an input was "
            f"produced again after them.")
    if missing:
        caveats.append(
            f"{missing} registered artifact(s) are no longer on disk.")

    for warning in (run.get("warnings") or ())[:5]:
        caveats.append(f"The run recorded a warning: {warning}")

    return caveats


# ---------------------------------------------------------------------------
# The deterministic renderers
# ---------------------------------------------------------------------------

def render_methods(digest: Mapping[str, Any]) -> str:
    """Write the methods section from the digest, with no model involved.

    What runs when no AI provider is configured, what a rejected draft falls
    back to, and what the number-provenance tests assert against. Every
    number in the output comes from ``digest``; every caveat in the digest is
    stated. No trailing newline.
    """
    run = digest.get("run") or {}
    environment = digest.get("environment") or {}
    modules = digest.get("modules") or []
    statistics = digest.get("statistics") or {}
    classifier = digest.get("classifier") or {}

    lines: List[str] = ["## Methods", ""]

    version = digest.get("spacr_version") or environment.get("spacr") or ""
    opening = "Image analysis was performed with spaCR"
    if version:
        opening += f" {version}"
    if environment.get("python"):
        opening += f" on Python {environment['python']}"
    tools = [f"{name} {environment[name]}"
             for name in ("torch", "cellpose", "numpy", "scipy", "pandas",
                          "scikit_image", "scikit_learn")
             if environment.get(name)]
    if tools:
        opening += f", using {', '.join(tools)}"
    lines.append(opening + ".")

    if modules:
        named = " → ".join(step.get("module", "?") for step in modules)
        lines.append(
            f"The pipeline ran {len(modules)} module(s) in the order "
            f"{named}.")
        for step in modules:
            chosen = (digest.get("parameters") or {}).get(
                step.get("module", ""), {})
            if chosen:
                spelled = ", ".join(
                    f"{key} = {value}" for key, value in
                    list(chosen.items())[:12])
                lines.append(
                    f"For {step.get('module')}, the parameters set explicitly "
                    f"were: {spelled}.")
            else:
                lines.append(
                    f"{step.get('module')} ran entirely at its spaCR "
                    f"defaults.")
    elif run.get("app_key"):
        lines.append(f"The {run['app_key']} module was run.")

    if classifier.get("classes"):
        lines.append(
            f"A classifier over {len(classifier['classes'])} class(es) "
            f"({', '.join(str(c) for c in classifier['classes'])}) was "
            f"trained for {classifier.get('epochs')} epoch(s).")

    if statistics.get("regression_type"):
        lines.append(
            f"Hits were called from a {statistics['regression_type']} "
            f"regression over {statistics.get('n_genes_tested')} gene(s), "
            f"ranked by {statistics.get('ranking')} at a threshold of "
            f"{statistics.get('alpha')}, with Benjamini-Hochberg control of "
            f"the false discovery rate.")

    if run.get("run_id") or run.get("journal_run_id"):
        identifier = run.get("run_id") or run.get("journal_run_id")
        lines.append(
            f"The run is recorded under id {identifier}; its settings, "
            f"package versions and outputs are in the run journal.")

    lines.append("")
    lines.append("### Caveats")
    for caveat in digest.get(DIGEST_CAVEATS, ()) or ():
        lines.append(f"- {caveat}")
    if not digest.get(DIGEST_CAVEATS):
        lines.append("- None recorded.")
    for note in digest.get("notes") or ():
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip("\n")


def render_results(digest: Mapping[str, Any]) -> str:
    """Write the results section from the digest, with no model involved.

    Every number comes from ``digest``. No trailing newline.
    """
    statistics = digest.get("statistics") or {}
    qc = digest.get("qc") or {}
    hits = digest.get("hits") or []
    classifier = digest.get("classifier") or {}
    provenance = digest.get("provenance") or {}

    lines: List[str] = ["## Results", ""]

    segmentation = qc.get("segmentation") or {}
    if segmentation.get("n_fields"):
        lines.append(
            f"Segmentation QC scored {segmentation['n_fields']} field(s): "
            f"{segmentation.get('n_ok')} passed, "
            f"{segmentation.get('n_warn')} raised a warning and "
            f"{segmentation.get('n_fail')} failed "
            f"(overall verdict: {segmentation.get('verdict')}).")

    held = classifier.get("held_out") or {}
    if held.get("accuracy") is not None:
        lines.append(
            f"On {held.get('n')} held-out objects the classifier reached an "
            f"accuracy of {held['accuracy']:.4g} and a macro F1 of "
            f"{held.get('f1_macro')}.")

    if statistics:
        lines.append(
            f"Of {statistics.get('n_genes_tested')} gene(s) tested, "
            f"{statistics.get('n_significant')} cleared the threshold of "
            f"{statistics.get('alpha')} — {statistics.get('n_up')} with a "
            f"positive effect and {statistics.get('n_down')} with a negative "
            f"one. {statistics.get('n_corroborated')} of them were "
            f"corroborated by two or more gRNAs agreeing in sign.")
        if statistics.get("max_abs_effect") is not None:
            lines.append(
                f"The largest absolute effect among them was "
                f"{statistics['max_abs_effect']}, with a median of "
                f"{statistics.get('median_abs_effect')}.")

    if hits:
        lines.append("")
        lines.append("The strongest hits were:")
        for row in hits:
            lines.append(
                f"- {row['name']} ({row['gene']}): effect "
                f"{row['effect']}, p = {row['p_value']}, q = "
                f"{row['q_value']}, {row['n_agree']} of {row['n_guides']} "
                f"gRNAs agreeing.")

    if provenance.get("n_artifacts"):
        lines.append("")
        lines.append(
            f"The figures above rest on {provenance['n_artifacts']} "
            f"registered artifact(s), of which "
            f"{provenance.get('n_current')} are current, "
            f"{provenance.get('n_stale')} stale and "
            f"{provenance.get('n_missing')} missing.")
    return "\n".join(lines).rstrip("\n")


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

def system_prompt() -> str:
    """The instruction the model is held to. Names the rule it must not break."""
    return (
        "You are writing two sections of a methods-and-results manuscript for "
        "a CRISPR imaging screen analysed with spaCR.\n"
        "\n"
        "You are given a RUN DIGEST: a structured record of what was run, "
        "with what parameters, on how much data, with which software "
        "versions, and with every statistic already computed. You do not have "
        "the data and you must not act as though you do.\n"
        "\n"
        "Rules, in order of importance:\n"
        "1. EVERY NUMBER you write must appear in the digest. Do not compute, "
        "convert, round to a precision the digest does not support, estimate, "
        "or recall a number from elsewhere. If a number you want is not in "
        "the digest, write the sentence without it. Your output is checked "
        "against the digest automatically and a draft containing a number "
        "that is not there is rejected.\n"
        "2. State every sentence listed under `caveats` in the Methods "
        "section. They are the limitations the pipeline recorded; omitting "
        "one produces a wrong methods section, not a shorter one.\n"
        "3. Write in the past tense, third person, no first person plural "
        "beyond normal scientific usage, no marketing language, no claims "
        "about biological significance that the digest does not support.\n"
        "4. Return exactly two sections, in this order and with these "
        "headings:\n"
        "## Methods\n"
        "## Results\n"
        "Nothing before, between or after them except the prose itself."
    )


def render_prompt(digest: Mapping[str, Any]) -> Tuple[str, str]:
    """``(system prompt, user message)`` for one digest.

    A pure function of the digest — which is the point. The model receives
    this and nothing else, so there is no path by which raw data could reach
    it, and a test can assert that a number planted in the digest is the only
    place a number in the prompt can have come from.

    :param digest: the digest to write about.
    :returns: the two prompt halves.
    """
    body = json.dumps(digest, indent=2, sort_keys=False, default=str)
    caveats = digest.get(DIGEST_CAVEATS) or []
    listed = "\n".join(f"- {caveat}" for caveat in caveats) or "- None."
    title = digest.get("title") or "this screen"
    user = (
        f"Write the Methods and Results sections for {title}.\n\n"
        f"These caveats must each be stated in the Methods section:\n"
        f"{listed}\n\n"
        f"RUN DIGEST (the only source of numbers you may use):\n"
        f"```json\n{body}\n```\n"
    )
    return system_prompt(), user
