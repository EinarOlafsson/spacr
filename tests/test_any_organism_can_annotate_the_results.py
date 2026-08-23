"""The gene annotation was a boolean, and so it was Toxoplasma or nothing.

``Toxoplasma=True`` joined a bundled table of *Toxoplasma gondii*
annotations onto the coefficients and ``False`` left bare accessions. The
one thing the regression module knew about biology was welded to one
parasite: a *Plasmodium* screen, a *Neospora* screen or a host-gene screen
got nothing.

Asked for on 2026-08-23 -- "it would be cool if that was a field that you
could fill with any uniprot id or organism name and it would pull what
uniprot has. Defaults to Toxo and Toxo data hardcoded as before" -- with the
organisms named: the hosts people image, every studied apicomplexan, and
the other parasites that come up beside them.

THE DEFAULT MUST NOT REACH THE NETWORK. Everything about the bundled path
is unchanged, and the tests that need UniProt say so and are skipped
without it. The two things that had to be right are checked offline:
resolution, and the promise that the default is still the bundled tables.
"""
from __future__ import annotations

import os
import socket

import pandas as pd
import pytest

from spacr.uniprot import (ACCESSION, BUNDLED_NAMES, ORGANISMS, Resolution,
                           near_misses, resolve)


def _has_network() -> bool:
    try:
        socket.create_connection(("rest.uniprot.org", 443), timeout=4).close()
        return True
    except Exception:                                        # noqa: BLE001
        return False


needs_uniprot = pytest.mark.skipif(
    not _has_network(), reason="UniProt is not reachable from here")


# ---------------------------------------------------------------------------
# resolution, offline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "toxoplasma", "Toxoplasma", "toxo",
                                  "Toxoplasma gondii", "True"])
def test_the_default_is_the_bundled_toxoplasma_path(text):
    """Whatever spelling arrives, it must not become a UniProt question."""
    assert resolve(text).kind == "bundled"


#: Every organism named in the request, by the name a user would type.
ASKED_FOR = [
    # hosts
    "human", "mouse", "rat", "hamster", "rhesus", "bovine", "pig", "dog",
    "chicken", "zebrafish", "drosophila", "c. elegans", "xenopus",
    # apicomplexa
    "Plasmodium falciparum", "Plasmodium vivax", "Plasmodium knowlesi",
    "Neospora caninum", "Eimeria tenella", "Cryptosporidium parvum",
    "Babesia bovis", "Theileria parva", "Sarcocystis neurona",
    "Cyclospora cayetanensis", "Cystoisospora suis", "Hammondia hammondi",
    # other parasites
    "Trypanosoma brucei", "Trypanosoma cruzi", "Leishmania major",
    "Giardia", "Entamoeba histolytica", "Trichomonas vaginalis",
    "Schistosoma mansoni",
]


@pytest.mark.parametrize("name", ASKED_FOR)
def test_every_organism_asked_for_resolves(name):
    resolved = resolve(name)
    assert resolved.kind == "organism", f"{name} did not resolve"
    assert isinstance(resolved.taxon, int) and resolved.taxon > 0


def test_a_taxon_id_resolves():
    assert resolve("9606").taxon == 9606
    assert resolve("taxid:5664").taxon == 5664


@pytest.mark.parametrize("accession", ["P04637", "Q9NRR4", "A0A0G2K3N4"])
def test_an_accession_is_recognised_by_shape(accession):
    resolved = resolve(accession)
    assert resolved.kind == "accession"
    assert resolved.accession == accession.upper()


def test_an_unknown_name_says_what_was_close():
    resolved = resolve("Plasmodum falciprum")

    assert resolved.kind == "unknown"
    assert resolved.near, "an unknown name with no near miss is a dead end"
    assert any("plasmodium" in n for n in resolved.near)


def test_a_bare_genus_offers_its_species():
    """difflib rates a prefix poorly against a two-word name."""
    near = near_misses("sarcocystis")
    assert any(n.startswith("sarcocystis") for n in near)


def test_the_table_has_no_duplicate_strain_confusion():
    """Every name maps to an int, and none of them is zero or negative."""
    assert len(ORGANISMS) > 90
    assert all(isinstance(t, int) and t > 0 for t in ORGANISMS.values())


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------

def test_the_setting_defaults_to_toxoplasma():
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({})
    assert settings["annotation_source"] == "toxoplasma"


def test_the_old_boolean_still_turns_it_off():
    """`Toxoplasma=False` is the one thing a NAME cannot say."""
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({"Toxoplasma": False})
    assert settings["annotation_source"] == ""


def test_the_field_wins_over_the_boolean():
    from spacr.ml import _annotation_source

    assert _annotation_source({"Toxoplasma": False,
                               "annotation_source": "human"}) == "human"
    assert _annotation_source({"Toxoplasma": True}) == "toxoplasma"
    assert _annotation_source({"Toxoplasma": False}) == ""


def test_the_cache_sits_beside_the_run():
    from spacr.ml import _annotation_cache

    assert _annotation_cache({"src": "/tmp/run"}).startswith("/tmp/run")
    assert _annotation_cache({}) is None


def test_resolving_the_default_does_not_import_the_network_path(monkeypatch):
    """The bundled path must work on a machine with no network at all."""
    import spacr.uniprot as U

    def refuse(*_args, **_kwargs):
        raise AssertionError("the bundled path reached the network")

    monkeypatch.setattr(U, "_get", refuse)
    assert resolve("toxoplasma").kind == "bundled"
    assert U.annotation_for("toxoplasma") == (None, "")


# ---------------------------------------------------------------------------
# against the real UniProt
# ---------------------------------------------------------------------------

@needs_uniprot
@pytest.mark.parametrize("name", ["human", "Plasmodium falciparum",
                                  "Neospora caninum", "Theileria parva"])
def test_a_real_organism_comes_back_with_entries(tmp_path, name):
    from spacr.uniprot import annotation_for

    frame, note = annotation_for(name, cache_dir=str(tmp_path),
                                 genes=["TP53", "ISN1", "FEN1"])
    if frame is None:                       # an organism with nothing at all
        pytest.fail(f"{name} returned nothing: {note}")
    assert len(frame) > 0
    assert "Entry" in frame.columns


@needs_uniprot
def test_a_real_accession_comes_back_as_one_entry(tmp_path):
    from spacr.uniprot import annotation_for

    frame, _note = annotation_for("P04637", cache_dir=str(tmp_path))

    assert frame is not None and len(frame) == 1
    assert frame.iloc[0]["Entry"] == "P04637"
    assert "p53" in str(frame.iloc[0]["Protein names"]).lower()


@needs_uniprot
def test_a_human_coefficient_table_is_annotated(tmp_path):
    """The whole point, end to end, on data UniProt really holds."""
    from spacr.annotation import annotate_with

    coefficients = pd.DataFrame({
        "gene": ["TP53", "EGFR", "MYC", "NOTAGENE"],
        "coefficient": [1.2, -0.4, 0.8, 0.1],
    })

    annotated, note = annotate_with(coefficients, "human",
                                    cache_dir=str(tmp_path))

    assert len(annotated) == len(coefficients), "the join multiplied rows"
    assert "product" in annotated.columns
    named = annotated.set_index("gene")["gene_name"]
    assert "TP53" in str(named["TP53"])
    assert "EGFR" in str(named["EGFR"])
    # The one that is not a gene stays empty rather than matching something.
    assert pd.isna(named["NOTAGENE"])
    assert not note


@needs_uniprot
def test_the_answer_is_cached_so_a_rerun_is_offline(tmp_path, monkeypatch):
    from spacr.annotation import annotate_with
    import spacr.uniprot as U

    coefficients = pd.DataFrame({"gene": ["TP53"], "coefficient": [1.0]})
    annotate_with(coefficients, "human", cache_dir=str(tmp_path))

    def refuse(*_args, **_kwargs):
        raise AssertionError("a cached answer reached the network")

    monkeypatch.setattr(U, "_get", refuse)
    again, _note = annotate_with(coefficients, "human",
                                 cache_dir=str(tmp_path))
    assert "product" in again.columns
