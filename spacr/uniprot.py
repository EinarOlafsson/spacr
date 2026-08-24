"""Resolve an organism or an accession against UniProt, and cache what comes back.

Regression's gene annotation used to be a BOOLEAN. ``Toxoplasma=True``
joined a bundled table of *Toxoplasma gondii* annotations onto the
coefficients; ``False`` left them as bare accessions. So the one thing the
module knew about biology was welded to one parasite, and a *Plasmodium*
screen, a *Neospora* screen or a host-gene screen got nothing at all.

The setting is a FIELD instead:

* empty, or any spelling of *Toxoplasma gondii* -- exactly what it did
  before, from the bundled CSVs, with no network at all. That path is the
  default and it must never depend on this module;
* an organism name or a taxon id -- that organism's reviewed proteome from
  UniProt;
* a single accession -- that one entry.

WHAT MUST RESOLVE, and why :data:`ORGANISMS` is a table rather than a call
to UniProt's own search: the organisms a user of this software actually
images should resolve without a round trip, and offline. Hosts people
culture cells from; every studied apicomplexan; the other parasites that
come up beside them.

NOTHING HERE IS REQUIRED TO OPEN THE MODULE. Every network call is behind a
cache, every failure is a warning that names the near-misses UniProt
offered, and an unresolvable name leaves the results unannotated rather than
stopping a run that has already done its fitting.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

#: UniProt's REST root. One place, so a test can point it somewhere else.
REST = "https://rest.uniprot.org"

#: Seconds any single request may take.
TIMEOUT = 30

#: What a UniProt accession looks like. The two shapes UniProt documents:
#: the six-character form (P04637) and the ten-character one (A0A0G2K3N4).
ACCESSION = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$",
    re.IGNORECASE)

#: A bare NCBI taxonomy id.
TAXON = re.compile(r"^(?:taxid:)?([0-9]{2,7})$", re.IGNORECASE)

#: The fields asked for, in the order the joined table carries them.
FIELDS: Tuple[str, ...] = (
    "accession", "id", "protein_name", "gene_names", "organism_name",
    "length", "cc_subcellular_location", "go_p", "go_c", "go_f",
    "ft_signal", "ft_transmem", "cc_function", "xref_alphafolddb",
)

#: Rows per request. UniProt's paging cap for the search endpoint.
PAGE = 500

#: The name spaCR uses for the bundled Toxoplasma path, and every spelling
#: of it that means "the default, offline, exactly as before".
BUNDLED = "toxoplasma"
BUNDLED_NAMES = frozenset({
    "", "toxoplasma", "toxo", "toxoplasma gondii", "t. gondii", "tgondii",
    "tgme49", "tggt1", "true", "yes", "1", "on", "default",
})


#: Organism name -> NCBI taxonomy id.
#:
#: Curated rather than searched, for two reasons. It resolves offline, which
#: is what lets the module open on a machine with no network; and it fixes
#: WHICH strain a bare genus name means, so "Plasmodium falciparum" is 3D7
#: every time rather than whichever proteome UniProt ranks first today.
#:
#: Three groups: the hosts whose cells people image, every studied
#: apicomplexan, and the other parasites that come up beside them.
#:
#: SPECIES-LEVEL IDS UNLESS THE STRAIN IS WHAT UNIPROT INDEXES. Every id
#: here was checked against UniProt by asking for one entry, and four of the
#: sixty-two returned nothing: Babesia bovis T2Bo, Leishmania major
#: Friedlin, Cryptosporidium hominis TU502 and Hammondia hammondi were
#: written as strain taxa that carry no proteins of their own. Toxoplasma
#: ME49 and P. falciparum 3D7 ARE indexed that way, which is why they stay
#: strain-level -- there is no rule here, only what UniProt actually holds.
ORGANISMS: Dict[str, int] = {
    # -- hosts people culture and image -----------------------------------
    "homo sapiens": 9606,
    "human": 9606,
    "mus musculus": 10090,
    "mouse": 10090,
    "rattus norvegicus": 10116,
    "rat": 10116,
    "mesocricetus auratus": 10036,
    "syrian hamster": 10036,
    "hamster": 10036,
    "cricetulus griseus": 10029,
    "chinese hamster": 10029,
    "macaca mulatta": 9544,
    "rhesus macaque": 9544,
    "rhesus": 9544,
    "chlorocebus sabaeus": 60711,
    "vero": 60711,
    "green monkey": 60711,
    "bos taurus": 9913,
    "bovine": 9913,
    "cow": 9913,
    "sus scrofa": 9823,
    "pig": 9823,
    "porcine": 9823,
    "ovis aries": 9940,
    "sheep": 9940,
    "canis lupus familiaris": 9615,
    "dog": 9615,
    "canine": 9615,
    "felis catus": 9685,
    "cat": 9685,
    "feline": 9685,
    "oryctolagus cuniculus": 9986,
    "rabbit": 9986,
    "gallus gallus": 9031,
    "chicken": 9031,
    "danio rerio": 7955,
    "zebrafish": 7955,
    "drosophila melanogaster": 7227,
    "drosophila": 7227,
    "fruit fly": 7227,
    "caenorhabditis elegans": 6239,
    "c. elegans": 6239,
    "xenopus laevis": 8355,
    "xenopus": 8355,
    "saccharomyces cerevisiae": 559292,
    "yeast": 559292,
    "arabidopsis thaliana": 3702,
    "arabidopsis": 3702,

    # -- apicomplexa ------------------------------------------------------
    "toxoplasma gondii": 508771,          # ME49
    "toxoplasma gondii me49": 508771,
    "toxoplasma gondii gt1": 507601,
    "toxoplasma gondii rh": 383379,
    "neospora caninum": 572307,
    "neospora": 572307,
    "hammondia hammondi": 99158,
    "besnoitia besnoiti": 94643,
    "cystoisospora suis": 483139,
    "sarcocystis neurona": 42890,
    "cyclospora cayetanensis": 88456,
    "plasmodium falciparum": 36329,       # 3D7
    "plasmodium falciparum 3d7": 36329,
    "p. falciparum": 36329,
    "plasmodium vivax": 126793,
    "plasmodium knowlesi": 5851,
    "plasmodium malariae": 5858,
    "plasmodium ovale": 36330,
    "plasmodium berghei": 5823,
    "plasmodium yoelii": 73239,
    "plasmodium chabaudi": 31271,
    "cryptosporidium parvum": 353152,
    "cryptosporidium hominis": 237895,
    "cryptosporidium": 353152,
    "eimeria tenella": 5802,
    "eimeria": 5802,
    "eimeria acervulina": 5801,
    "eimeria maxima": 5804,
    "babesia bovis": 5865,
    "babesia microti": 1133968,
    "babesia": 5865,
    "theileria parva": 5875,
    "theileria annulata": 5874,
    "theileria": 5875,
    "cytauxzoon felis": 88764,
    # The three tissue-cyst and intestinal apicomplexans the genus name
    # alone was missing. Each points at the species a screen would be run
    # on rather than at the genus taxon, which carries no proteome.
    "sarcocystis neurona": 42890,
    "sarcocystis": 42890,
    "cyclospora cayetanensis": 88456,
    "cyclospora": 88456,
    "cystoisospora suis": 483139,
    "cystoisospora": 483139,

    # -- other parasites --------------------------------------------------
    "trypanosoma brucei": 185431,
    "t. brucei": 185431,
    "trypanosoma cruzi": 353153,
    "t. cruzi": 353153,
    "leishmania major": 5664,
    "leishmania donovani": 5661,
    "leishmania infantum": 5671,
    "leishmania": 5664,
    "giardia intestinalis": 184922,
    "giardia lamblia": 184922,
    "giardia": 184922,
    "entamoeba histolytica": 294381,
    "entamoeba": 294381,
    "trichomonas vaginalis": 412133,
    "trichomonas": 412133,
    "schistosoma mansoni": 6183,
    "schistosoma japonicum": 6182,
    "schistosoma": 6183,
    "acanthamoeba castellanii": 1257118,
    "naegleria fowleri": 5763,
    "anopheles gambiae": 180454,
    "aedes aegypti": 7159,
    "ixodes scapularis": 6945,
}


@dataclass(frozen=True)
class Resolution:
    """What a piece of text in the annotation field turned out to mean.

    :param kind: ``'bundled'``, ``'accession'``, ``'organism'`` or
        ``'unknown'``.
    :param text: what the user typed, stripped.
    :param taxon: the NCBI taxonomy id, for an organism.
    :param accession: the accession, for a single entry.
    :param near: names close to what was typed, for an unknown one.
    """

    kind: str
    text: str = ""
    taxon: Optional[int] = None
    accession: str = ""
    near: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.kind != "unknown"


def canonical(text) -> str:
    """The lookup form of an organism name: lower case, single spaces."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def resolve(text) -> Resolution:
    """What ``text`` names, without touching the network.

    Order matters. The bundled names win, because that path must work with
    no network and must not be changed by anything here. An accession is
    recognised by shape. Everything else is looked up as an organism, and a
    name that is not in the table comes back ``unknown`` WITH the near
    misses, because "did you mean" is the whole difference between a
    typo the user can fix and a silent absence of annotation.

    :param text: whatever is in the annotation field.
    :returns: a :class:`Resolution`.
    """
    raw = str(text or "").strip()
    name = canonical(raw)
    if name in BUNDLED_NAMES:
        return Resolution("bundled", raw)
    if ACCESSION.match(raw):
        return Resolution("accession", raw, accession=raw.upper())
    taxon = TAXON.match(raw)
    if taxon:
        return Resolution("organism", raw, taxon=int(taxon.group(1)))
    if name in ORGANISMS:
        return Resolution("organism", raw, taxon=ORGANISMS[name])
    return Resolution("unknown", raw, near=near_misses(name))


def near_misses(name, limit: int = 5) -> Tuple[str, ...]:
    """Organism names close to ``name``, for a message that helps."""
    import difflib

    name = canonical(name)
    if not name:
        return ()
    close = difflib.get_close_matches(name, sorted(ORGANISMS), n=limit,
                                      cutoff=0.6)
    if close:
        return tuple(close)
    # A GENUS ON ITS OWN is the common miss -- "plasmodium", "leishmania" --
    # and difflib does not rate a prefix highly against a two-word name.
    head = name.split()[0]
    return tuple(sorted(n for n in ORGANISMS if n.startswith(head))[:limit])


def organisms_for(group: str = "") -> Tuple[str, ...]:
    """Every name that resolves, or those starting with ``group``."""
    group = canonical(group)
    return tuple(sorted(n for n in ORGANISMS if not group
                        or n.startswith(group)))


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _cache_path(cache_dir, key: str) -> str:
    return os.path.join(str(cache_dir), f"uniprot_{key}.json")


def _read_cache(cache_dir, key: str):
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, key)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:                                        # noqa: BLE001
        return None


def _write_cache(cache_dir, key: str, payload) -> None:
    if not cache_dir:
        return
    try:
        os.makedirs(str(cache_dir), exist_ok=True)
        with open(_cache_path(cache_dir, key), "w", encoding="utf-8") as out:
            json.dump(payload, out)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not cache the UniProt answer", exc_info=True)


def _get(url: str) -> Tuple[str, str]:
    """``(body, next_url)``. The second is UniProt's cursor for the next page."""
    request = urllib.request.Request(
        url, headers={"Accept": "text/plain",
                      "User-Agent": "spaCR (https://github.com/EinarOlafsson/spacr)"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = response.read().decode("utf-8", "replace")
        link = response.headers.get("Link", "") or ""
    return body, _next_page(link)


def _next_page(link_header: str) -> str:
    """The ``rel="next"`` URL out of a Link header, or "".

    UniProt caps a page at 500 rows and hands the next cursor back in this
    header. Without following it a "proteome" was the first 500 entries of
    one: a human screen looking for TP53 found nothing, because reviewed
    human is twenty thousand entries and TP53 is not in the first five
    hundred.
    """
    # NOT split(","). A Link header is comma-separated, but the URL inside
    # it carries `fields=accession,id,protein_name,...` with literal commas,
    # so splitting cut the URL into fragments and the one holding
    # `rel="next"` had lost its opening bracket. The header parsed as having
    # no next page, every fetch stopped at 500 rows, and a "proteome" was
    # the first five hundred entries of one -- which is why a human screen
    # looking for TP53 matched nothing.
    found = re.search(r'<([^>]+)>\s*;\s*rel="next"', str(link_header or ""))
    return found.group(1) if found else ""


def _search_url(query: str, size: int) -> str:
    return (f"{REST}/uniprotkb/search?"
            + urllib.parse.urlencode({
                "query": query,
                "fields": ",".join(FIELDS),
                "format": "tsv",
                "size": int(size),
            }))


def fetch(resolution: Resolution, *, cache_dir=None, reviewed: bool = True,
          limit: int = 20000):
    """Everything UniProt has for ``resolution``, as a DataFrame.

    :param resolution: from :func:`resolve`. ``bundled`` and ``unknown``
        return None -- neither is a UniProt question.
    :param cache_dir: where the answer is kept so a rerun is offline.
    :param reviewed: Swiss-Prot only. For an organism with no reviewed
        entries at all the caller gets an empty frame and a warning, which
        is truer than silently including unreviewed predictions.
    :param limit: stop after this many rows.
    :returns: a DataFrame, or None when there is nothing to ask.
    """
    import pandas as pd

    if resolution.kind == "accession":
        key = f"acc_{resolution.accession}"
        query = f"accession:{resolution.accession}"
        size = 1
    elif resolution.kind == "organism":
        key = f"tax_{resolution.taxon}_{'sp' if reviewed else 'all'}"
        query = f"organism_id:{resolution.taxon}"
        if reviewed:
            query += " AND reviewed:true"
        size = min(PAGE, limit)
    else:
        return None

    cached = _read_cache(cache_dir, key)
    if cached is not None:
        return pd.DataFrame(cached)

    frames, url, fetched = [], _search_url(query, size), 0
    try:
        while url and fetched < int(limit):
            text, url = _get(url)
            page = _parse_tsv(text)
            if page is None or not len(page):
                break
            frames.append(page)
            fetched += len(page)
    except Exception as error:                               # noqa: BLE001
        LOG.warning("UniProt could not be reached for %s: %s",
                    resolution.text or resolution.accession, error)
        if not frames:
            return None
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 \
        else frames[0]
    _write_cache(cache_dir, key, frame.to_dict("records"))
    return frame


def _parse_tsv(text: str):
    """UniProt's TSV as a DataFrame, or None when it is not one."""
    import io

    import pandas as pd

    if not str(text or "").strip():
        return None
    try:
        frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    except Exception:                                        # noqa: BLE001
        LOG.debug("UniProt did not return a table", exc_info=True)
        return None
    return frame


#: Most genes a targeted query will ask for by name rather than pulling the
#: whole proteome.
#:
#: A full proteome is the honest answer to "what does UniProt have for this
#: organism", and for human it is 20,431 entries over 41 pages -- 259
#: seconds the first time, and a 40 MB cache. A coefficient table naming
#: three genes does not need that. Past this many the proteome is fetched
#: once and cached, which is cheaper than a query URL with a thousand
#: clauses in it.
TARGETED_MAX = 300


def fetch_genes(resolution: Resolution, genes, *, cache_dir=None):
    """The entries for named ``genes`` in ``resolution``'s organism.

    :param genes: gene names or accessions from the table being annotated.
    :returns: a DataFrame, or None when the query could not be made.
    """
    import pandas as pd

    names = [str(g).strip() for g in (genes or []) if str(g).strip()]
    names = sorted({n for n in names})
    if not names or resolution.kind != "organism":
        return None
    if len(names) > TARGETED_MAX:
        return None

    import hashlib

    digest = hashlib.sha1("|".join(names).encode()).hexdigest()[:16]
    key = f"tax_{resolution.taxon}_genes_{digest}"
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        return pd.DataFrame(cached)

    # gene: AND accession, because a screen library names its targets one
    # way or the other and neither spelling is wrong.
    clauses = " OR ".join(
        f"gene:{n}" if not ACCESSION.match(n) else f"accession:{n}"
        for n in names)
    query = f"({clauses}) AND organism_id:{resolution.taxon}"
    frames, url, fetched = [], _search_url(query, PAGE), 0
    try:
        while url and fetched < 20000:
            text, url = _get(url)
            page = _parse_tsv(text)
            if page is None or not len(page):
                break
            frames.append(page)
            fetched += len(page)
    except Exception as error:                               # noqa: BLE001
        LOG.debug("targeted UniProt query failed: %s", error)
        if not frames:
            return None
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 \
        else frames[0]
    _write_cache(cache_dir, key, frame.to_dict("records"))
    return frame


def annotation_for(text, *, cache_dir=None, genes=None):
    """``(frame, note)`` for whatever is in the annotation field.

    The one call a pipeline needs. It never raises: an unreachable UniProt,
    an unknown organism and an empty proteome all come back as ``(None,
    note)`` so the run carries on with unannotated results, which is what
    the field is for.
    """
    resolution = resolve(text)
    if resolution.kind == "bundled":
        return None, ""
    if resolution.kind == "unknown":
        near = ", ".join(resolution.near)
        return None, (
            f"{resolution.text!r} is not an organism spaCR knows and is not "
            f"an accession"
            + (f" -- did you mean {near}?" if near else "")
            + ". Results are not annotated.")
    # ASK FOR WHAT IS NEEDED FIRST. A table naming a few genes gets a query
    # naming those genes, which is seconds; only a table naming more than
    # `TARGETED_MAX` of them pulls the whole proteome.
    frame = fetch_genes(resolution, genes, cache_dir=cache_dir)
    if frame is None or not len(frame):
        frame = fetch(resolution, cache_dir=cache_dir)
    note = ""
    if (frame is None or not len(frame)) and resolution.kind == "organism":
        # NO REVIEWED ENTRIES IS NORMAL for most parasites. Swiss-Prot has
        # a few hundred proteins for Plasmodium and none at all for some
        # strain-level taxa -- Babesia bovis and Leishmania major both came
        # back empty on the reviewed query while having thousands of
        # unreviewed entries. Refusing to annotate those organisms because
        # nobody has curated them by hand would make the field useless for
        # exactly the organisms it was asked for.
        #
        # So the fallback is taken and SAID, because a TrEMBL annotation is
        # a prediction and the reader is entitled to know which they have.
        frame = fetch(resolution, cache_dir=cache_dir, reviewed=False)
        if frame is not None and len(frame):
            note = (f"{resolution.text} has no reviewed (Swiss-Prot) "
                    f"entries; these {len(frame)} are unreviewed TrEMBL "
                    f"predictions.")
    if frame is None or not len(frame):
        return None, (f"UniProt returned nothing for {resolution.text!r}. "
                      f"Results are not annotated.")
    return frame, note
