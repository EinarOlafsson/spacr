"""Scientific-copy contracts for the Home module registry."""

from __future__ import annotations


def _summaries() -> dict[str, str]:
    from spacr.qt.app import APPS

    return {
        key: description
        for key, _label, description, _section in APPS
    }


def test_mask_summary_names_every_supported_segmentation_compartment():
    """The Home tile must describe the complete consolidated Mask route."""
    summary = _summaries()["mask"]
    folded = summary.casefold()
    for term in ("cell", "nuclei", "pathogen", "organelle"):
        assert term in folded
    assert "Cellpose" in summary


def test_home_summaries_exclude_reviewed_colloquial_and_marketing_copy():
    """Keep module summaries concise, literal, and scientifically phrased."""
    banned = (
        "someone else's",
        "turn images",
        "run them overnight",
        "costs in disk",
        "one-click",
        "what produced what",
        "watch the prediction move",
        "one world",
        "the n behind",
    )
    failures = []
    for key, summary in _summaries().items():
        folded = summary.casefold()
        for phrase in banned:
            if phrase in folded:
                failures.append(f"{key}: {phrase!r}")
    assert not failures, failures
