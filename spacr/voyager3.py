"""Voyager3: give each OPS gene cluster a biological role, from fetched UniProt facts, through the user's own AI.

The last step of the OPS profile path (item 973, alpha). For every gene of
every cluster, :func:`fetch_gene_facts` fetches the reviewed UniProt entry
(protein names, function, GO process / component / function, pathway,
subcellular location and the annotation score) through :mod:`spacr.uniprot`,
whose cache makes a rerun offline. :func:`build_cluster_prompt` writes one
English prompt per cluster with those facts, the cluster's statistics and the
screen context; the template is not translated. The prompt goes to the AI
provider the user chose under Preferences, spaCR AI (free) or a vendor
command-line tool signed in with the user's own account; nothing here uses a
vendor API, an API key or a batch service. The report and the provenance name
the provider and model, and say when the free provider answered with its
smaller model. :func:`parse_cluster_answer`
reads the structured answer (role, confidence, supporting genes, genes that do
not fit, uncharacterised candidates with suggested roles) and never raises on a
malformed one.

The confidence is also computed by spaCR from the share of the cluster's
genes the answer cites as support (above 70 % high, 50-70 % medium, 30-50 %
low, below that none, the thresholds of Brieflow's MozzareLLM), so a stated
confidence that the cited genes do not bear out is visible. Gene names that are
not in the cluster are dropped and counted.

Every run writes ``voyager3_provenance.json``: the system text, every prompt,
the template version, the provider and model, the UniProt release and fetch
date, the facts each prompt used and every raw answer. :func:`voyager3_replay`
writes the tables again from that file alone, and :func:`run_voyager3` writes
its own tables through the same code, so the two are byte-identical.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["TEMPLATE_VERSION", "SYSTEM_PROMPT", "build_cluster_prompt", "fetch_gene_facts",
           "parse_cluster_answer", "run_voyager3", "voyager3_replay"]

#: The prompt template's version, saved with every run.
TEMPLATE_VERSION = "voyager3-1"

#: The UniProt return fields Voyager3 asks for.
UNIPROT_FIELDS: Tuple[str, ...] = (
    "accession", "gene_primary", "gene_names", "reviewed", "annotation_score",
    "protein_name", "cc_function", "go_p", "go_c", "go_f", "cc_pathway",
    "cc_subcellular_location",
)

#: Genes per UniProt request.
BATCH = 100

#: Seconds between two fresh UniProt requests.
REQUEST_GAP_S = 0.5

#: Characters kept of a gene's function text, and of its pathway and location.
FUNCTION_CHARS = 600
SHORT_CHARS = 300

#: GO terms kept per aspect.
GO_TERMS = 8

#: Characters a prompt may hold before its facts are written compactly.
PROMPT_BUDGET = 60000

#: Function characters and GO process terms kept per gene in a compact prompt.
COMPACT_CHARS = 200
COMPACT_GO_TERMS = 4

#: Seed of the size-preserving shuffle control.
SHUFFLE_SEED = 20261117

#: Smallest cluster that is annotated.
MIN_GENES = 2

#: The confidence words, best first.
CONFIDENCE = ("high", "medium", "low", "none")

#: Name of spaCR's free hosted provider.
_HOSTED = "spacr_hosted"

#: The system text sent with every prompt.
SYSTEM_PROMPT = (
    "You are Voyager3, spaCR's annotator for gene clusters from optical pooled "
    "screens. The genes of one cluster gave similar cell phenotypes when knocked "
    "out. Assign the biological role they share, using the UniProt facts given; "
    "you may add well-established knowledge, but say so in the evidence. Answer "
    "with one JSON object only.")

_TASKS = """Tasks:
1. Name the single biological role (pathway, protein complex or process) shared by most of these genes. If no role is shared by at least 30% of the genes, give "no coherent role".
2. supporting_genes: the genes whose facts support that role, each with a short evidence phrase that names the UniProt field it comes from (function, GO, pathway or location), or "general knowledge".
3. misfit_genes: the genes whose facts do not fit the role.
4. uncharacterized_candidates: genes with annotation score 1 or 2, no function text, or no reviewed entry, for which the cluster suggests a role. Give suggested_role, priority 1-10 (10 = strongest case for follow-up) and a one-sentence rationale. Do not prioritise a gene only because it is in the cluster.
5. confidence: "high" if more than 70% of the genes support the role, "medium" for 50-70%, "low" for 30-50%, "none" below 30%.
Use only gene symbols from the list above."""

_SCHEMA = """Answer with one JSON object in a ```json block, with exactly these fields:
{"cluster_id": "<id>", "biological_role": "<role or no coherent role>", "confidence": "high|medium|low|none", "supporting_genes": [{"gene": "<symbol>", "evidence": "<field: phrase>"}], "misfit_genes": ["<symbol>"], "uncharacterized_candidates": [{"gene": "<symbol>", "suggested_role": "<role>", "priority": 1, "rationale": "<sentence>"}], "summary": "<two sentences>"}"""

_ANSWER_KEYS = frozenset({"biological_role", "dominant_process", "role", "confidence", "pathway_confidence",
                          "supporting_genes", "misfit_genes", "uncharacterized_candidates"})

_FACT_COLUMNS = ("gene", "found", "accession", "reviewed", "annotation_score", "protein_name",
                 "function", "go_process", "go_component", "go_function", "pathway", "location")

_CLUSTER_COLUMNS = ("cluster", "kind", "n_genes", "status", "biological_role", "confidence_model",
                    "confidence_rule", "confidence_agrees", "support_fraction", "n_supporting",
                    "supporting_genes", "misfit_genes", "n_candidates", "candidates", "unknown_genes",
                    "reasons", "summary")

_CANDIDATE_COLUMNS = ("cluster", "kind", "gene", "suggested_role", "priority", "annotation_score",
                      "found", "rationale")

_HEADERS = {
    "accession": ("Entry",), "gene_primary": ("Gene Names (primary)",), "gene_names": ("Gene Names",),
    "reviewed": ("Reviewed",), "annotation_score": ("Annotation",), "protein_name": ("Protein names",),
    "function": ("Function [CC]",), "go_process": ("Gene Ontology (biological process)",),
    "go_component": ("Gene Ontology (cellular component)",),
    "go_function": ("Gene Ontology (molecular function)",), "pathway": ("Pathway",),
    "location": ("Subcellular location [CC]",),
}


def _text(value: Any) -> str:
    """A table cell as stripped text; missing values give ``""``.

    :param value: the cell.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _cut(text: str, limit: int) -> str:
    """``text`` cut at the last space before ``limit`` characters, marked ``...``.

    :param text: cleaned text.
    :param limit: the most characters kept before the mark.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip(" ,;.") + " ..."


def _clean(text: Any, prefix: str = "") -> str:
    """UniProt comment text without evidence tags, PubMed citations or its ``PREFIX:`` labels.

    :param text: the raw cell.
    :param prefix: the comment label to drop, such as ``FUNCTION``.
    """
    out = _text(text)
    if prefix:
        out = re.sub(rf"\b{prefix}:\s*", " ", out)
    out = re.sub(r"\{ECO:[^}]*\}", "", out)
    out = re.sub(r"\((?:PubMed|PMID):[^)]*\)", "", out)
    out = re.sub(r";?\s*(?:Single|Multi)-pass[^.;]*protein", "", out)
    out = re.sub(r";?\s*(?:Peripheral|Lipid-anchor)[^.;]*(?:protein|side)", "", out)
    out = re.sub(r"\s+([.,;])", r"\1", out)
    out = re.sub(r"([.;])(?:\s*[.;])+", r"\1", out)
    return re.sub(r"\s+", " ", out).strip(" ;")


def _go_terms(text: Any) -> str:
    """The first :data:`GO_TERMS` term names of a UniProt GO cell, joined by semicolons, without ids.

    :param text: the raw cell (``name [GO:id]; ...``).
    """
    names = [re.sub(r"\s*\[GO:\d+\]", "", part).strip() for part in _text(text).split(";")]
    return "; ".join([name for name in names if name][:GO_TERMS])


def _cell(row: Mapping[str, Any], key: str) -> str:
    """The raw value of fact ``key`` in a UniProt TSV row, by header.

    :param row: one row as a mapping of TSV headers.
    :param key: a key of :data:`_HEADERS`.
    """
    for header in _HEADERS[key]:
        if header in row:
            return _text(row[header])
    return ""


def _empty_fact(gene: str) -> Dict[str, Any]:
    """The fact record of a gene with no reviewed UniProt entry.

    :param gene: the gene symbol.
    """
    record: Dict[str, Any] = {column: "" for column in _FACT_COLUMNS}
    record.update(gene=gene, found=False, annotation_score=None)
    return record


def _fact(gene: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    """The cleaned fact record of ``gene`` from its UniProt row.

    :param gene: the gene symbol asked for.
    :param row: its matched UniProt TSV row.
    """
    score = _cell(row, "annotation_score")
    try:
        score_value: Optional[float] = float(score) if score else None
    except ValueError:
        score_value = None
    return {
        "gene": gene, "found": True, "accession": _cell(row, "accession"),
        "reviewed": _cell(row, "reviewed").lower() == "reviewed", "annotation_score": score_value,
        "protein_name": _cut(_cell(row, "protein_name"), 150),
        "function": _cut(_clean(_cell(row, "function"), "FUNCTION"), FUNCTION_CHARS),
        "go_process": _go_terms(_cell(row, "go_process")),
        "go_component": _go_terms(_cell(row, "go_component")),
        "go_function": _go_terms(_cell(row, "go_function")),
        "pathway": _cut(_clean(_cell(row, "pathway"), "PATHWAY"), SHORT_CHARS),
        "location": _cut(_clean(re.sub(r"Note=.*", "", _cell(row, "location")),
                                "SUBCELLULAR LOCATION"), SHORT_CHARS),
    }


def _match(gene: str, rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The UniProt row that stands for ``gene``.

    A reviewed entry whose primary gene name is the symbol, then any entry
    whose primary name is, then a reviewed entry naming it as a synonym;
    case is ignored and ties keep UniProt's order.

    :param gene: the gene symbol.
    :param rows: the UniProt rows of its batch.
    """
    want = gene.upper()
    primary = [r for r in rows if _cell(r, "gene_primary").upper() == want]
    for row in primary:
        if _cell(row, "reviewed").lower() == "reviewed":
            return row
    if primary:
        return primary[0]
    for row in rows:
        names = _cell(row, "gene_names").upper().split()
        if want in names and _cell(row, "reviewed").lower() == "reviewed":
            return row
    return None


def fetch_gene_facts(genes: Sequence[str], organism: str = "human", cache_dir: Optional[str] = None,
                     report: Optional[Callable[[str], None]] = None) -> Tuple[Dict[str, Dict[str, Any]],
                                                                               Dict[str, Any]]:
    """Fetch each gene's reviewed UniProt facts, through :mod:`spacr.uniprot` and its cache.

    Genes are asked for in sorted batches of :data:`BATCH`, Swiss-Prot only,
    with :data:`UNIPROT_FIELDS`; fresh requests are :data:`REQUEST_GAP_S`
    seconds apart, and a cached batch makes no request. Evidence tags and
    PubMed citations are removed and long text is cut at a word.

    :param genes: gene symbols.
    :param organism: anything :func:`spacr.uniprot.resolve` reads as an
        organism (a name such as ``human`` or a taxon id).
    :param cache_dir: the UniProt cache folder; None caches nothing.
    :param report: called with a line per batch fetched.
    :returns: ``(facts, meta)``: per gene, ``gene``, ``found``,
        ``accession``, ``reviewed``, ``annotation_score`` (1-5 or None),
        ``protein_name``, ``function``, ``go_process``, ``go_component``,
        ``go_function``, ``pathway`` and ``location``; and the run's
        ``organism``, ``taxon``, ``fields``, the UniProt ``releases`` and
        ``release_dates``, the ``fetched`` dates, ``requests`` made and
        ``busy`` batches.
    :raises ValueError: when ``organism`` is not an organism.
    """
    from . import uniprot

    resolution = uniprot.resolve(organism)
    if resolution.kind != "organism":
        raise ValueError(f"ops_voyager3_organism must name an organism or a taxon id; got {organism!r}")
    say = report or (lambda text: None)
    names = sorted({str(g).strip() for g in genes if str(g).strip()})
    facts: Dict[str, Dict[str, Any]] = {}
    releases, dates, fetched, busy, requests = set(), set(), set(), 0, 0
    for start in range(0, len(names), BATCH):
        batch = names[start:start + BATCH]
        frame, was_busy, meta = uniprot._fetch_gene_rows(resolution, batch, cache_dir=cache_dir,
                                                         fields=UNIPROT_FIELDS, reviewed=True)
        busy += int(bool(was_busy))
        if meta:
            releases.add(meta.get("release", ""))
            dates.add(meta.get("release_date", ""))
            fetched.add(meta.get("fetched", ""))
        if was_busy or not meta.get("cached", False):
            requests += 1
            time.sleep(REQUEST_GAP_S)
        rows = frame.to_dict("records") if frame is not None else []
        for gene in batch:
            row = _match(gene, rows)
            facts[gene] = _fact(gene, row) if row is not None else _empty_fact(gene)
        say(f"Voyager3: UniProt batch {start // BATCH + 1}: {sum(f['found'] for f in facts.values())} "
            f"of {len(facts)} genes found")
    meta_out = {"organism": str(organism), "taxon": resolution.taxon, "fields": list(UNIPROT_FIELDS),
                "reviewed_only": True, "releases": sorted(r for r in releases if r),
                "release_dates": sorted(d for d in dates if d), "fetched": sorted(d for d in fetched if d),
                "requests": requests, "busy_batches": busy}
    return facts, meta_out


def _fact_lines(fact: Mapping[str, Any], compact: bool = False) -> List[str]:
    """The prompt lines of one gene's facts.

    :param fact: a record from :func:`fetch_gene_facts`.
    :param compact: keep the function to :data:`COMPACT_CHARS` characters
        and the GO process terms to :data:`COMPACT_GO_TERMS`, and leave out
        the GO component and function terms.
    """
    lines = [f"### {fact['gene']}"]
    if not fact.get("found"):
        lines.append("not found in reviewed UniProtKB (Swiss-Prot) for this organism")
        return lines
    score = fact.get("annotation_score")
    score_text = "unknown" if score is None else f"{float(score):g}"
    lines.append(f"accession: {fact.get('accession') or 'n/a'}; annotation score: {score_text}")
    fields = (("protein_name", "protein"), ("function", "function"), ("pathway", "pathway"),
              ("location", "location"), ("go_process", "GO process"))
    if not compact:
        fields += (("go_component", "GO component"), ("go_function", "GO function"))
    for key, label in fields:
        value = _text(fact.get(key))
        if compact and key == "function":
            value = _cut(value, COMPACT_CHARS)
        if compact and key == "go_process":
            value = "; ".join(value.split("; ")[:COMPACT_GO_TERMS]) if value else ""
        lines.append(f"{label}: {value if value else 'none given'}")
    return lines


def build_cluster_prompt(cluster: Any, genes: Sequence[str], facts: Mapping[str, Mapping[str, Any]],
                         stats: Optional[Mapping[str, Any]] = None, context: str = "",
                         uniprot_meta: Optional[Mapping[str, Any]] = None) -> str:
    """Write the English Voyager3 prompt of one cluster (template :data:`TEMPLATE_VERSION`).

    The prompt names the template version, the cluster and its size, its
    statistics, the screen context and the UniProt release, then one block of
    facts per gene in sorted order, the tasks (with MozzareLLM's 70 / 50 /
    30 % confidence rule) and the JSON answer schema. A prompt longer than
    :data:`PROMPT_BUDGET` characters is written again with shorter facts
    (function cut to 200 characters, 4 GO process terms, no GO component or
    function terms), and says so. The text is the same for the same inputs;
    it is not translated.

    :param cluster: the cluster id.
    :param genes: the cluster's gene symbols.
    :param facts: per gene, a record from :func:`fetch_gene_facts`; a gene
        missing here is written as not found.
    :param stats: cluster statistics to state, such as ``mean_stability``;
        None or empty states none.
    :param context: the screen context (cell type, markers, population).
    :param uniprot_meta: the ``meta`` of :func:`fetch_gene_facts`, for the
        release and fetch date.
    :returns: the prompt text.
    """
    members = sorted({str(g) for g in genes})
    meta = dict(uniprot_meta or {})
    release = ", ".join(meta.get("releases") or []) or "unknown"
    fetched = ", ".join(meta.get("fetched") or []) or "unknown"
    stat_text = "; ".join(f"{key.replace('_', ' ')} {value}" for key, value in sorted((stats or {}).items()))
    text = ""
    for compact in (False, True):
        lines = [f"Voyager3 template {TEMPLATE_VERSION}", f"Cluster: {cluster}", f"Genes in cluster: {len(members)}",
                 f"Cluster statistics: {stat_text or 'none recorded'}",
                 f"Screen context: {str(context or '').strip() or 'not given'}", "",
                 f"Gene facts from UniProtKB release {release} (fetched {fetched}). "
                 "Annotation score runs from 1 (least) to 5 (best characterised)."
                 + (" Facts are shortened to fit the prompt." if compact else ""), ""]
        for gene in members:
            lines.extend(_fact_lines(facts.get(gene) or _empty_fact(gene), compact))
        lines.extend(["", _TASKS, "", _SCHEMA.replace("<id>", str(cluster))])
        text = "\n".join(lines) + "\n"
        if len(text) <= PROMPT_BUDGET:
            break
    return text


def _json_object(text: str) -> Tuple[Optional[Any], str]:
    """The JSON object of an answer: the first ``json`` fence, else the first balanced ``{...}``.

    :param text: the raw answer.
    :returns: ``(object or None, reason when None)``.
    """
    for block in re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL):
        try:
            return json.loads(block), ""
        except ValueError:
            continue
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for at in range(start, len(text)):
            char = text[at]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:at + 1]), ""
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None, "no JSON object in the answer"


def _confidence_word(value: Any) -> str:
    """A confidence word in :data:`CONFIDENCE`, or ``""`` when ``value`` is not one.

    :param value: the stated confidence.
    """
    word = _text(value).lower()
    if word in CONFIDENCE:
        return word
    if word in ("no", "n/a", "na", "no coherent", "no coherent role", "no coherent pathway", "very low"):
        return "none"
    return ""


def _confidence_from_fraction(fraction: float) -> str:
    """The confidence word for the share of a cluster's genes that support its role.

    Above 0.7 is ``high``, 0.5 to 0.7 ``medium``, 0.3 to 0.5 ``low`` and
    below 0.3 ``none`` (MozzareLLM's thresholds).

    :param fraction: supporting genes over cluster genes.
    """
    if fraction > 0.7:
        return "high"
    if fraction >= 0.5:
        return "medium"
    if fraction >= 0.3:
        return "low"
    return "none"


def _gene_list(value: Any, members: Mapping[str, str], unknown: List[str]) -> List[Dict[str, Any]]:
    """The entries of a gene list whose gene is a cluster member, first mention kept.

    :param value: a list of symbols or of objects with a ``gene`` field.
    :param members: upper-case symbol to the cluster's spelling.
    :param unknown: collects the names that are not members.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in value if isinstance(value, list) else []:
        entry = dict(item) if isinstance(item, Mapping) else {"gene": item}
        name = _text(entry.get("gene") or entry.get("symbol"))
        gene = members.get(name.upper())
        if gene is None:
            if name and name not in unknown:
                unknown.append(name)
            continue
        if gene in seen:
            continue
        seen.add(gene)
        entry["gene"] = gene
        out.append(entry)
    return out


def _priority(value: Any) -> Optional[int]:
    """A priority from 1 to 10, or None when ``value`` is not one.

    :param value: the stated priority (number or numeric text).
    """
    try:
        number = float(_text(value))
    except ValueError:
        return None
    if number != number or not 1 <= number <= 10 or number != int(number):
        return None
    return int(number)


def parse_cluster_answer(text: Optional[str], genes: Sequence[str]) -> Dict[str, Any]:
    """Read one Voyager3 answer into checked fields; never raises.

    The JSON is the first ``json`` fence, else the first balanced object; an
    object with none of the answer fields (such as one gene entry of a cut-off
    answer) counts as unparsed. Gene names are matched to the cluster without regard to case; names that
    are not members are dropped into ``unknown_genes``. A priority that is not
    a whole number from 1 to 10 becomes None. ``support_fraction`` is the
    share of the cluster's genes given as support, and ``confidence_rule``
    its confidence word: above 0.7 ``high``, 0.5 to 0.7 ``medium``, 0.3 to
    0.5 ``low``, below 0.3 ``none``.

    :param text: the raw answer; None or blank gives status ``no_answer``.
    :param genes: the cluster's gene symbols.
    :returns: ``status`` (``ok``, ``partial``, ``unparsed`` or
        ``no_answer``), ``reasons``, ``biological_role``,
        ``confidence_model``, ``confidence_rule``, ``confidence_agrees``,
        ``support_fraction``, ``supporting`` (``gene``, ``evidence``),
        ``misfit``, ``candidates`` (``gene``, ``suggested_role``,
        ``priority``, ``rationale``), ``unknown_genes`` and ``summary``.
    """
    members = {str(g).upper(): str(g) for g in genes}
    result: Dict[str, Any] = {
        "status": "ok", "reasons": [], "biological_role": "", "confidence_model": "", "confidence_rule": "none",
        "confidence_agrees": None, "support_fraction": 0.0, "supporting": [], "misfit": [], "candidates": [],
        "unknown_genes": [], "summary": ""}
    raw = "" if text is None else str(text)
    if not raw.strip():
        result.update(status="no_answer", reasons=["empty answer"])
        return result
    obj, reason = _json_object(raw)
    if not isinstance(obj, Mapping):
        result.update(status="unparsed", reasons=[reason or "the JSON is not an object"])
        return result
    if not set(obj) & _ANSWER_KEYS:
        result.update(status="unparsed", reasons=["the JSON object has none of the answer fields"])
        return result
    reasons: List[str] = []
    role = _text(obj.get("biological_role") or obj.get("dominant_process") or obj.get("role"))
    if not role:
        reasons.append("missing biological_role")
    stated = obj.get("confidence", obj.get("pathway_confidence"))
    confidence = _confidence_word(stated)
    if not confidence:
        reasons.append(f"confidence not one of high/medium/low/none: {_text(stated)[:40]!r}")
    unknown: List[str] = []
    for field in ("supporting_genes", "misfit_genes", "uncharacterized_candidates"):
        if field not in obj:
            reasons.append(f"missing {field}")
        elif not isinstance(obj[field], list):
            reasons.append(f"{field} is not a list")
    supporting = [{"gene": e["gene"], "evidence": _text(e.get("evidence"))}
                  for e in _gene_list(obj.get("supporting_genes"), members, unknown)]
    misfit = [e["gene"] for e in _gene_list(obj.get("misfit_genes"), members, unknown)]
    candidates = []
    for entry in _gene_list(obj.get("uncharacterized_candidates"), members, unknown):
        priority = _priority(entry.get("priority"))
        if priority is None and entry.get("priority") is not None:
            reasons.append(f"priority of {entry['gene']} is not 1-10: {_text(entry.get('priority'))[:20]!r}")
        candidates.append({"gene": entry["gene"], "suggested_role": _text(entry.get("suggested_role")),
                           "priority": priority, "rationale": _text(entry.get("rationale"))})
    if unknown:
        reasons.append(f"{len(unknown)} gene(s) not in the cluster")
    fraction = len(supporting) / len(members) if members else 0.0
    rule = _confidence_from_fraction(fraction)
    result.update(status="partial" if reasons else "ok", reasons=reasons, biological_role=role,
                  confidence_model=confidence, confidence_rule=rule,
                  confidence_agrees=(confidence == rule) if confidence else None,
                  support_fraction=round(fraction, 4), supporting=supporting, misfit=misfit,
                  candidates=candidates, unknown_genes=unknown, summary=_text(obj.get("summary")))
    return result


def _collect(provider: Any, messages: List[Dict[str, str]], system: str, model: Optional[str],
             chunks: List[str], failed: List[BaseException]) -> None:
    """Gather a provider's streamed answer on a worker thread.

    :param provider: the chat provider.
    :param messages: the conversation (one user message).
    :param system: the system text.
    :param model: the model name, or None for the provider's default.
    :param chunks: receives the answer's pieces.
    :param failed: receives the exception, if one is raised.
    """
    try:
        for chunk in provider.stream_chat(messages, system=system, model=model):
            chunks.append(str(chunk))
    except BaseException as exc:                            # noqa: BLE001
        failed.append(exc)


def _ask(provider: Any, prompt: str, model: Optional[str], timeout: float) -> Dict[str, Any]:
    """One attempt: the answer text, the error, the seconds taken and, for spaCR AI (free), the model tier.

    :param provider: the chat provider.
    :param prompt: the user message.
    :param model: the model name, or None.
    :param timeout: seconds before the stream is cancelled.
    """
    chunks: List[str] = []
    failed: List[BaseException] = []
    if hasattr(provider, "last_result"):
        try:
            provider.last_result = {}
        except Exception:                                   # noqa: BLE001
            pass
    started = time.monotonic()
    worker = threading.Thread(target=_collect, name="spacr-voyager3", daemon=True,
                              args=(provider, [{"role": "user", "content": prompt}], SYSTEM_PROMPT, model,
                                    chunks, failed))
    worker.start()
    worker.join(timeout)
    error = ""
    if worker.is_alive():
        cancel = getattr(provider, "cancel_stream", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:                               # noqa: BLE001
                pass
        error = f"timed out after {timeout:g} s"
    elif failed:
        error = f"{type(failed[0]).__name__}: {failed[0]}"[:500]
    tier = str(getattr(provider, "last_tier", "") or "") if getattr(provider, "name", "") == _HOSTED else ""
    result = getattr(provider, "last_result", None)
    result = result if isinstance(result, Mapping) else {}
    usage = result.get("modelUsage")
    return {"response": "" if error else "".join(chunks), "error": error,
            "seconds": round(time.monotonic() - started, 3), "tier": tier,
            "models": sorted(usage) if isinstance(usage, Mapping) else [],
            "cost_usd": result.get("total_cost_usd")}


def _resolve_provider(name: str) -> Any:
    """The chat provider Voyager3 uses: ``name``, else the one chosen under Preferences.

    Any provider Preferences offers is accepted: spaCR AI (free) or a vendor
    command-line tool signed in with the user's own account; ``none`` is
    refused, and so is a provider that is not usable now.

    :param name: a provider name, or ``""`` for the Preferences choice.
    :raises ValueError: when no usable provider is chosen or configured.
    """
    from .qt.ai.providers import configured_providers, get_provider, hosted_provider

    chosen = str(name or "").strip()
    if not chosen:
        try:
            from .qt.preferences import get_preferred_provider

            chosen = get_preferred_provider()
        except Exception:                                   # noqa: BLE001
            chosen = ""
    if chosen == "none":
        raise ValueError("Voyager3 needs an AI provider, and Preferences → AI says None")
    if chosen:
        provider = hosted_provider() if chosen == _HOSTED else get_provider(chosen)
        if provider is None or not provider.is_configured():
            raise ValueError(f"the AI provider {chosen!r} is not usable now (not installed, not signed in, "
                             "or the network is not allowed)")
        return provider
    ready = configured_providers()
    if not ready:
        raise ValueError("Voyager3 found no usable AI provider; choose one under Preferences → AI")
    return ready[0]


def _cluster_id(value: Any) -> str:
    """A cluster id as text; whole-number floats lose their ``.0``.

    :param value: the cluster cell.
    """
    if isinstance(value, float) and value == value and value == int(value):
        return str(int(value))
    return _text(value)


def _sort_key(cluster: str) -> Tuple[int, Any]:
    """Numeric ids in numeric order, then the others as text.

    :param cluster: a cluster id.
    """
    return (0, int(cluster)) if re.fullmatch(r"-?\d+", cluster) else (1, cluster)


def _read_clusters(settings: Mapping[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]], str]:
    """The clusters to annotate, their statistics and the table they came from.

    :param settings: read for ``ops_voyager3_clusters`` (default
        ``<dst_root>/results/ops_clusters/consensus_partition.csv``),
        ``ops_voyager3_cluster_column``, ``ops_voyager3_cluster_ids`` and
        ``ops_control_prefix``.
    :raises ValueError: for a missing table or column.
    """
    from .tabular import read_table

    path = settings.get("ops_voyager3_clusters")
    if not path:
        if not settings.get("dst_root"):
            raise ValueError("Voyager3 needs ops_voyager3_clusters or dst_root")
        path = os.path.join(str(settings["dst_root"]), "results", "ops_clusters", "consensus_partition.csv")
    if not os.path.isfile(str(path)):
        raise ValueError(f"Voyager3 found no cluster table at {path}; run the clustering phases first or set "
                         "ops_voyager3_clusters")
    table = read_table(str(path), canonicalise=False, report=None)
    column = str(settings.get("ops_voyager3_cluster_column") or "cluster")
    for needed in ("gene", column):
        if needed not in table.columns:
            raise ValueError(f"the cluster table {path} has no {needed!r} column")
    prefix = str(settings.get("ops_control_prefix") or "nontargeting").lower()
    keep = ~table["gene"].astype(str).str.lower().str.startswith(prefix)
    if "targeting" in table.columns:
        keep &= table["targeting"].astype(bool)
    if "unassigned" in table.columns:
        keep &= ~table["unassigned"].astype(bool)
    table = table[keep & table[column].notna()]
    wanted = settings.get("ops_voyager3_cluster_ids")
    if isinstance(wanted, str):
        wanted = [part for part in re.split(r"[,\s]+", wanted) if part]
    wanted_ids = {_cluster_id(w) for w in wanted} if wanted else None
    clusters: Dict[str, List[str]] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    for cluster, rows in table.groupby(table[column].map(_cluster_id), sort=False):
        if wanted_ids is not None and cluster not in wanted_ids:
            continue
        genes = sorted({str(g) for g in rows["gene"]})
        if len(genes) < MIN_GENES:
            continue
        clusters[cluster] = genes
        record: Dict[str, Any] = {}
        if "stability" in rows.columns:
            values = rows["stability"].astype(float)
            if values.notna().any():
                record["mean_stability"] = round(float(values.mean()), 3)
        stats[cluster] = record
    order = sorted(clusters, key=_sort_key)
    return {c: clusters[c] for c in order}, {c: stats[c] for c in order}, str(path)


def _shuffled(clusters: Mapping[str, List[str]], seed: int = SHUFFLE_SEED) -> Dict[str, List[str]]:
    """The same cluster sizes with the genes permuted across them (seeded).

    :param clusters: cluster id to genes.
    :param seed: the permutation seed.
    """
    import numpy as np

    pool = [gene for genes in clusters.values() for gene in genes]
    order = np.random.default_rng(seed).permutation(len(pool))
    out: Dict[str, List[str]] = {}
    at = 0
    for cluster, genes in clusters.items():
        out[f"S{cluster}"] = sorted(pool[i] for i in order[at:at + len(genes)])
        at += len(genes)
    return out


def _settings_record(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``ops_voyager3_*`` settings, JSON-safe, for the provenance file.

    :param settings: the run's settings.
    """
    return {key: (value if isinstance(value, (str, int, float, bool, type(None), list)) else str(value))
            for key, value in sorted(settings.items()) if key.startswith("ops_voyager3_")}


def _answer_key(prompt: str, model: Optional[str]) -> str:
    """The SHA-256 naming a saved answer: template version, model, system text and prompt.

    :param prompt: the user message.
    :param model: the model name, or None.
    """
    import hashlib

    text = "\n".join((TEMPLATE_VERSION, model or "", SYSTEM_PROMPT, prompt))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _saved_answer(folder: str, key: str) -> Optional[Dict[str, Any]]:
    """The saved answer named ``key``, or None when there is none or it is unreadable or empty.

    :param folder: the answers folder.
    :param key: from :func:`_answer_key`.
    """
    path = os.path.join(folder, f"{key}.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(record, Mapping) or not str(record.get("response") or "").strip():
        return None
    return dict(record)


def _save_answer(folder: str, key: str, attempt: Mapping[str, Any], model: Optional[str]) -> str:
    """Save one answer atomically (temporary file, then rename) and return its path.

    :param folder: the answers folder, created when missing.
    :param key: from :func:`_answer_key`.
    :param attempt: the successful attempt from :func:`_ask`.
    :param model: the model name asked for, or None.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{key}.json")
    record = {"response": attempt["response"], "seconds": attempt.get("seconds"), "tier": attempt.get("tier", ""),
              "models": list(attempt.get("models") or []), "cost_usd": attempt.get("cost_usd"),
              "model_asked": model or "", "template": TEMPLATE_VERSION,
              "saved": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    os.replace(temporary, path)
    return path


def _provider_args(provider: Any) -> List[str]:
    """The command-line arguments the provider adds for its model and speed, or ``[]``.

    :param provider: the chat provider.
    """
    try:
        from .qt.ai.providers import _provider_arguments

        return [str(a) for a in _provider_arguments(provider)]
    except Exception:                                       # noqa: BLE001
        return []


def _quality_note(provenance: Mapping[str, Any]) -> str:
    """A note on answers that came from spaCR AI (free), or ``""``.

    :param provenance: the run's provenance record.
    """
    if provenance.get("provider") != _HOSTED:
        return ""
    finals = [(record.get("attempts") or [{}])[-1].get("tier", "") for record in provenance.get("clusters") or []]
    light = sum(tier == "light" for tier in finals)
    note = ("Quality note: the answers came from spaCR AI (free), a free model within a daily allowance, which can "
            "be weaker at this task than a signed-in Claude, GPT (Codex) or Gemini.")
    if light:
        note += f" {light} of {len(finals)} answers came from its smaller (light) model."
    return note


def _version() -> str:
    """spaCR's version, or ``""`` when it cannot be read."""
    try:
        from importlib.metadata import version

        return version("spacr")
    except Exception:                                       # noqa: BLE001
        return ""


def run_voyager3(settings: Mapping[str, Any], report: Optional[Callable[[str], None]] = None,
                 provider: Any = None, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Annotate every cluster with Voyager3 and write its tables, report and provenance.

    Reads the clusters (``ops_voyager3_clusters``), fetches their genes' UniProt
    facts (:func:`fetch_gene_facts`, cached in ``uniprot_cache`` beside the
    outputs), asks the provider once per cluster (one retry after an error or
    an empty answer) and writes the outputs through :func:`voyager3_replay`'s
    code. With ``ops_voyager3_shuffle_control`` the same is done for a seeded
    shuffle of the genes across the clusters, sizes kept. Prints a
    ``Progress: n/total, operation_type: Voyager3`` line per cluster, with the
    counts of answers replayed and asked.

    Each answer is saved as it arrives, in ``ops_voyager3_answers_dir``
    (``answers`` beside the outputs when empty), as one JSON file named by the
    SHA-256 of the template version, model, system text and prompt, holding
    the raw text, the model ids and cost the provider reported, and the
    seconds taken; the file is written to a temporary name and renamed. With
    ``ops_voyager3_resume`` on (the default) a rerun replays the saved answer
    of every prompt it finds and asks only the others, so an interrupted run
    picks up where it stopped and writes the same tables; a changed model,
    template or prompt asks again. Failed and timed-out calls are not saved.
    The provenance records which answers were replayed and which were asked.

    :param settings: the OPS settings; read for ``dst_root``, the
        ``ops_voyager3_*`` settings and ``ops_control_prefix``.
    :param report: called with progress lines; ``print`` when None.
    :param provider: a chat provider object (``name``, ``stream_chat``);
        None uses ``ops_voyager3_provider`` or the Preferences choice,
        spaCR AI (free) included.
    :param out_dir: the output folder; ``<dst_root>/results/ops_voyager3``
        when None.
    :returns: ``{"folder", "outputs", "clusters", "provenance"}``.
    :raises ValueError: for missing clusters, an unknown organism or no
        usable provider.
    """
    from .ops_settings import ops_defaults

    settings = ops_defaults(dict(settings))
    say = report or (lambda text: print(text, flush=True))
    clusters, stats, source = _read_clusters(settings)
    if not clusters:
        raise ValueError(f"Voyager3 found no cluster with at least {MIN_GENES} genes in {source}")
    if out_dir is None:
        out_dir = os.path.join(str(settings.get("dst_root") or os.path.dirname(source)), "results", "ops_voyager3")
    os.makedirs(out_dir, exist_ok=True)
    chat = provider if provider is not None else _resolve_provider(str(settings.get("ops_voyager3_provider") or ""))
    model = str(settings.get("ops_voyager3_model") or "").strip() or None
    timeout = float(settings.get("ops_voyager3_timeout") or 300)
    context = str(settings.get("ops_voyager3_context") or "")
    jobs = [(cluster, "real", genes) for cluster, genes in clusters.items()]
    if settings.get("ops_voyager3_shuffle_control"):
        jobs += [(cluster, "shuffled", genes) for cluster, genes in _shuffled(clusters).items()]
    genes_all = sorted({gene for _c, _k, genes in jobs for gene in genes})
    say(f"Voyager3: {len(clusters)} clusters, {len(genes_all)} genes; fetching UniProt facts")
    facts, uniprot_meta = fetch_gene_facts(genes_all, str(settings.get("ops_voyager3_organism") or "human"),
                                           cache_dir=os.path.join(out_dir, "uniprot_cache"), report=say)
    resume = bool(settings.get("ops_voyager3_resume", True))
    answers_dir = str(settings.get("ops_voyager3_answers_dir") or os.path.join(out_dir, "answers"))
    records = []
    counts = {"replayed": 0, "asked": 0}
    for done, (cluster, kind, genes) in enumerate(jobs, start=1):
        prompt = build_cluster_prompt(cluster, genes, facts, stats.get(cluster) if kind == "real" else {},
                                      context, uniprot_meta)
        key = _answer_key(prompt, model)
        saved = _saved_answer(answers_dir, key) if resume else None
        if saved is not None:
            attempts = [{"response": saved["response"], "error": "", "seconds": saved.get("seconds", 0.0),
                         "tier": saved.get("tier", ""), "models": saved.get("models", []),
                         "cost_usd": saved.get("cost_usd")}]
            counts["replayed"] += 1
            source_kind = "replayed"
        else:
            attempts = [_ask(chat, prompt, model, timeout)]
            if attempts[0]["error"] or not attempts[0]["response"].strip():
                attempts.append(_ask(chat, prompt, model, timeout))
            final = attempts[-1]
            if not final["error"] and final["response"].strip():
                _save_answer(answers_dir, key, final, model)
            counts["asked"] += 1
            source_kind = "asked"
        records.append({"cluster": cluster, "kind": kind, "genes": list(genes),
                        "stats": dict(stats.get(cluster) or {}) if kind == "real" else {},
                        "prompt": prompt, "attempts": attempts, "response": attempts[-1]["response"],
                        "answer_key": key, "answer_source": source_kind})
        say(f"Progress: {done}/{len(jobs)}, operation_type: Voyager3, cluster {cluster} ({kind}), "
            f"{counts['replayed']} replayed, {counts['asked']} asked")
    provenance = {
        "voyager3": TEMPLATE_VERSION, "spacr_version": _version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": str(getattr(chat, "name", type(chat).__name__)),
        "provider_label": str(getattr(chat, "label", "")), "model": model or "",
        "provider_args": [] if model else _provider_args(chat), "timeout_s": timeout,
        "cluster_table": source, "context": context, "system": SYSTEM_PROMPT, "uniprot": uniprot_meta,
        "shuffle_seed": SHUFFLE_SEED if settings.get("ops_voyager3_shuffle_control") else None,
        "settings": _settings_record(settings), "facts": {gene: facts[gene] for gene in genes_all},
        "answers": {"dir": answers_dir, "resume": resume, **counts}, "clusters": records}
    path = os.path.join(out_dir, "voyager3_provenance.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1, sort_keys=True)
    outputs = dict(_write_outputs(provenance, out_dir), provenance=path)
    say(f"Voyager3: wrote {outputs['clusters']}")
    return {"folder": out_dir, "outputs": outputs, "clusters": len(clusters), "provenance": path}


def voyager3_replay(provenance: Any, out_dir: Optional[str] = None) -> Dict[str, str]:
    """Write a Voyager3 run's tables and report again from its provenance file alone.

    Nothing is fetched and no provider is asked: the saved raw answers are
    parsed again with the saved facts, through the same code the run wrote
    them with, so the files are byte-identical to the run's.

    :param provenance: the path of ``voyager3_provenance.json``, or its
        loaded content.
    :param out_dir: where to write; the provenance file's folder when None.
    :returns: the paths of ``clusters``, ``candidates``, ``facts`` and
        ``report``.
    :raises ValueError: when no folder can be told for loaded content.
    """
    if isinstance(provenance, Mapping):
        record = dict(provenance)
        if out_dir is None:
            raise ValueError("voyager3_replay needs out_dir when given loaded provenance")
    else:
        with open(str(provenance), "r", encoding="utf-8") as handle:
            record = json.load(handle)
        out_dir = out_dir or os.path.dirname(os.path.abspath(str(provenance)))
    os.makedirs(str(out_dir), exist_ok=True)
    return _write_outputs(record, str(out_dir))


def _joined(values: Sequence[Any]) -> str:
    """Values as one ``;``-separated cell.

    :param values: the values.
    """
    return ";".join(str(v) for v in values)


def _tables(provenance: Mapping[str, Any]) -> Tuple[Any, Any, Any, List[Dict[str, Any]]]:
    """The cluster, candidate and fact tables of a provenance record, and the parsed answers.

    :param provenance: the run's provenance record.
    """
    import pandas as pd

    facts = provenance.get("facts") or {}
    rows, candidates, parsed_all = [], [], []
    for record in provenance.get("clusters") or []:
        genes = list(record.get("genes") or [])
        parsed = parse_cluster_answer(record.get("response"), genes)
        parsed_all.append(parsed)
        errors = [a.get("error") for a in record.get("attempts") or [] if a.get("error")]
        reasons = list(parsed["reasons"]) + [f"provider: {e}" for e in errors]
        rows.append({
            "cluster": record["cluster"], "kind": record.get("kind", "real"), "n_genes": len(genes),
            "status": parsed["status"], "biological_role": parsed["biological_role"],
            "confidence_model": parsed["confidence_model"], "confidence_rule": parsed["confidence_rule"],
            "confidence_agrees": "" if parsed["confidence_agrees"] is None else str(parsed["confidence_agrees"]),
            "support_fraction": f"{parsed['support_fraction']:.4f}", "n_supporting": len(parsed["supporting"]),
            "supporting_genes": _joined(e["gene"] for e in parsed["supporting"]),
            "misfit_genes": _joined(parsed["misfit"]), "n_candidates": len(parsed["candidates"]),
            "candidates": _joined(f"{c['gene']} ({c['suggested_role']})" for c in parsed["candidates"]),
            "unknown_genes": _joined(parsed["unknown_genes"]), "reasons": " | ".join(reasons),
            "summary": parsed["summary"]})
        for entry in parsed["candidates"]:
            fact = facts.get(entry["gene"]) or {}
            score = fact.get("annotation_score")
            candidates.append({
                "cluster": record["cluster"], "kind": record.get("kind", "real"), "gene": entry["gene"],
                "suggested_role": entry["suggested_role"],
                "priority": "" if entry["priority"] is None else str(entry["priority"]),
                "annotation_score": "" if score is None else f"{float(score):g}",
                "found": str(bool(fact.get("found"))), "rationale": entry["rationale"]})
    fact_rows = []
    for gene in sorted(facts):
        fact = dict(facts[gene])
        score = fact.get("annotation_score")
        fact["annotation_score"] = "" if score is None else f"{float(score):g}"
        fact["found"] = str(bool(fact.get("found")))
        fact["reviewed"] = str(bool(fact.get("reviewed"))) if fact.get("found") == "True" else ""
        fact_rows.append({column: _text(fact.get(column)) for column in _FACT_COLUMNS})
    frames = (pd.DataFrame(rows, columns=list(_CLUSTER_COLUMNS)),
              pd.DataFrame(candidates, columns=list(_CANDIDATE_COLUMNS)),
              pd.DataFrame(fact_rows, columns=list(_FACT_COLUMNS)))
    return frames[0], frames[1], frames[2], parsed_all


def _write_outputs(provenance: Mapping[str, Any], out_dir: str) -> Dict[str, str]:
    """Write the tables and the report of a provenance record into ``out_dir``.

    Every cell is text, so the files depend only on the record.

    :param provenance: the run's provenance record.
    :param out_dir: the output folder.
    """
    from .tabular import write_table

    clusters, candidates, facts, _parsed = _tables(provenance)
    outputs = {}
    for name, frame in (("clusters", clusters), ("candidates", candidates), ("facts", facts)):
        target = os.path.join(out_dir, f"voyager3_{'gene_facts' if name == 'facts' else name}.csv")
        write_table(frame.astype(str), target, canonicalise=False, lineterminator="\n")
        outputs[name] = target
    target = os.path.join(out_dir, "voyager3_report.md")
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_report_text(provenance, clusters, candidates))
    outputs["report"] = target
    return outputs


def _cell_md(value: Any) -> str:
    """A value safe inside a Markdown table cell.

    :param value: the value.
    """
    return _text(value).replace("|", "/").replace("\n", " ")


def _report_text(provenance: Mapping[str, Any], clusters: Any, candidates: Any) -> str:
    """The readable Voyager3 report: provenance, summary counts, one section per cluster, glossary.

    :param provenance: the run's provenance record.
    :param clusters: the cluster table.
    :param candidates: the candidate table.
    """
    meta = provenance.get("uniprot") or {}
    lines = ["# Voyager3 cluster annotation", "",
             f"- Template: {provenance.get('voyager3', '')}; spaCR {provenance.get('spacr_version', '')}",
             f"- AI provider: {provenance.get('provider', '')}; model: {provenance.get('model') or 'provider default'}"
             + (f" ({' '.join(provenance.get('provider_args') or [])})" if provenance.get("provider_args") else ""),
             f"- UniProtKB release: {', '.join(meta.get('releases') or []) or 'unknown'} "
             f"({', '.join(meta.get('release_dates') or []) or 'date unknown'}); fetched "
             f"{', '.join(meta.get('fetched') or []) or 'unknown'}; reviewed entries only; organism "
             f"{meta.get('organism', '')} (taxon {meta.get('taxon', '')})",
             f"- Clusters from: {provenance.get('cluster_table', '')}",
             f"- Run started: {provenance.get('created', '')}", "",
             "The roles below are generated by an AI model from the UniProt facts in the prompt. They can be "
             "wrong; check them before you act on them.", ""] + ([_quality_note(provenance), ""]
                                                                  if _quality_note(provenance) else []) + [
             "## Summary", "", "| Clusters | Kind | High | Medium | Low | None | Unparsed or no answer |",
             "|---|---|---|---|---|---|---|"]
    for kind in ("real", "shuffled"):
        part = clusters[clusters["kind"] == kind]
        if not len(part):
            continue
        good = part[part["status"].isin(["ok", "partial"])]
        counts = [int((good["confidence_rule"] == word).sum()) for word in CONFIDENCE]
        label = kind if kind == "real" else "shuffled (control)"
        lines.append(f"| {len(part)} | {label} | " + " | ".join(map(str, counts))
                     + f" | {len(part) - len(good)} |")
    lines.extend(["", "Confidence here is spaCR's rule on the share of genes cited as support; the model's own "
                  "word is in the cluster sections.", ""])
    for _index, row in clusters.iterrows():
        lines.extend([f"## Cluster {row['cluster']} ({row['kind']}, {row['n_genes']} genes)", "",
                      f"- Role: {row['biological_role'] or 'none given'}",
                      f"- Confidence: {row['confidence_rule']} by rule (support {row['support_fraction']}); "
                      f"model said {row['confidence_model'] or 'nothing usable'}",
                      f"- Status: {row['status']}" + (f" ({row['reasons']})" if row["reasons"] else ""),
                      f"- Supporting genes: {row['supporting_genes'] or 'none'}",
                      f"- Genes that do not fit: {row['misfit_genes'] or 'none'}",
                      f"- Not in the cluster (dropped): {row['unknown_genes'] or 'none'}"])
        mine = candidates[(candidates["cluster"] == row["cluster"]) & (candidates["kind"] == row["kind"])]
        if len(mine):
            lines.extend(["", "| Candidate | Suggested role | Priority | UniProt annotation score | Rationale |",
                          "|---|---|---|---|---|"])
            for _i, cand in mine.iterrows():
                lines.append(f"| {_cell_md(cand['gene'])} | {_cell_md(cand['suggested_role'])} | "
                             f"{cand['priority'] or 'n/a'} | {cand['annotation_score'] or 'not found'} | "
                             f"{_cell_md(cand['rationale'])} |")
        if row["summary"]:
            lines.extend(["", row["summary"]])
        lines.append("")
    lines.extend([
        "## Glossary", "",
        "- **Annotation score:** UniProt's 1-5 score of how well an entry is annotated (5 = best characterised).",
        "- **Confidence by rule:** high above 70 % of the cluster's genes cited as support, medium 50-70 %, "
        "low 30-50 %, none below 30 % (MozzareLLM's thresholds).",
        "- **GO:** Gene Ontology (process, component, function).",
        "- **Misfit genes:** members whose facts do not fit the role.",
        "- **Shuffled (control):** the same cluster sizes with the genes permuted across clusters (seeded); "
        "coherent roles should be rare there.",
        "- **Status:** ok; partial (some fields missing or invalid); unparsed (no JSON object); no answer.",
        "- **Swiss-Prot:** the reviewed section of UniProtKB.",
        "- **Uncharacterised candidate:** a member with annotation score 1-2, no function text or no reviewed "
        "entry, given a suggested role by the model.", ""])
    return "\n".join(lines)
