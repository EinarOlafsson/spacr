"""Two narrow gaps: the bundled annotation and a rebuilt model.

Each is the last branch of something that works every other day: the
optional Toxoplasma annotation join of a hit list, and rebuilding a
classifier architecture from its recorded configuration without downloading
weights.
"""
from __future__ import annotations

import pandas as pd

from spacr import torch_artifacts as TA
from spacr.hits import build_hit_list


# ---------------------------------------------------------------------------
# The bundled Toxoplasma annotation
# ---------------------------------------------------------------------------

def _gene_table(genes):
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{gene}]" for gene in genes],
        "coefficient": [0.4, -0.3, 0.2][: len(genes)],
        "p_value": [0.001, 0.02, 0.3][: len(genes)],
    })


def test_the_bundled_annotation_names_the_genes_it_recognises():
    """A hit list of gene numbers is unreadable; the join is what makes it a list."""
    hits = build_hit_list({"gene": _gene_table(["233460", "254470", "227280"])},
                          toxoplasma=True)

    assert hits.gene("233460").annotation["gene_name"] == "SAG1"
    assert hits.gene("254470").annotation["gene_name"] == "MYR1"
    assert hits.gene("227280").annotation["hyperlopit"] == "dense granules"
    assert any("Bundled Toxoplasma annotation joined by gene number" in note
               for note in hits.notes)


def test_the_annotation_join_leaves_the_row_count_alone():
    """One gene must stay one hit: a fan-out here doubles a screen's hits."""
    table = _gene_table(["233460", "254470", "227280"])

    hits = build_hit_list({"gene": table}, toxoplasma=True)

    assert len(hits) == len(table)
    assert len({hit.gene for hit in hits}) == len(table)


def test_a_column_the_user_supplied_wins_over_the_bundles(tmp_path):
    """The user passed that file on purpose; the bundle must not overwrite it."""
    path = tmp_path / "my_names.csv"
    pd.DataFrame({"Gene ID": ["TGME49_233460"],
                  "gene_name": ["my own name for SAG1"]}).to_csv(path,
                                                                 index=False)

    hits = build_hit_list({"gene": _gene_table(["233460"])},
                          metadata_files=[str(path)], toxoplasma=True)

    assert hits.gene("233460").annotation["gene_name"] == "my own name for SAG1"


def test_leaving_the_annotation_off_leaves_the_list_unannotated():
    """The control: nothing is joined unless it was asked for."""
    hits = build_hit_list({"gene": _gene_table(["233460"])})

    assert "gene_name" not in hits.gene("233460").annotation
    assert not any("Bundled Toxoplasma" in note for note in hits.notes)


# ---------------------------------------------------------------------------
# A checkpoint that has to rebuild its own architecture
# ---------------------------------------------------------------------------

def test_a_configuration_rebuilds_the_architecture_without_downloading(
        monkeypatch):
    """``pretrained=False`` is the point: loading must work offline."""
    import torchvision

    def refuse(*_args, **_kwargs):
        raise AssertionError("rebuilding a model must not fetch weights")

    monkeypatch.setattr(torchvision.models._api, "get_model_weights", refuse,
                        raising=False)

    model = TA.build_model_from_configuration({
        "model_name": "resnet18", "num_classes": 5, "image_size": 64,
        "dropout_rate": None, "use_checkpoint": False, "multilabel": False})

    assert model.model_name == "resnet18"
    assert model.num_classes == 5
    assert model.pretrained is False
