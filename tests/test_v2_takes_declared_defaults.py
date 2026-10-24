"""The v2 mask pipeline answered from its own inline fallbacks.

`preprocess_generate_masks` dispatches to `run_v2` and RETURNS, while
`set_default_settings_preprocess_generate_masks` is only called further down
inside the per-source loop that the v2 branch never reaches. So every
``settings.get(key, fallback)`` in that branch answered with the fallback
written beside it rather than with the module's declared default.

One of them changes segmentation. ``cell_flow_threshold`` was declared 1.0
when this was fixed and the inline fallback was 0.4, and it is forwarded
verbatim to ``model.eval(flow_threshold=...)``. Cellpose's
``remove_bad_flow_masks`` discards a mask whose flow error exceeds the
threshold, so on a field with per-object flow errors {0.00, 0.12, 0.30, 0.75}
the v1 pipeline kept four cells and the v2 branch kept three -- same plate,
same settings dict, same weights, different answer depending only on
``pipeline_style``. Since 2026-09-19 (#123, ledger 428) the declared default
is 0.4, the same number as the fallback, so the two agree by value today; the
ordering tests below are what keep them agreeing if either moves again.
"""

import ast
import inspect
import pathlib

import pytest

import spacr.core as core
import spacr.settings as S


def _v2_branch_source():
    """The body of the `pipeline_style == 'v2'` branch, as text."""
    source = inspect.getsource(core.preprocess_generate_masks)
    tree = ast.parse(source.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "pipeline_style" in ast.unparse(node.test):
            return "\n".join(ast.unparse(stmt) for stmt in node.body)
    raise AssertionError("the v2 branch is gone; update this test")


def test_the_branch_applies_the_defaults_before_reading_the_dict(tmp_path, monkeypatch):
    """Apply real declared defaults before reading processing arguments for dispatch."""
    from spacr import pipeline_v2

    processing_keys = {
        "cell_model_name", "cell_diameter", "batch_fields", "metadata_type",
        "custom_regex", "keep_npz", "cell_cellprob_threshold", "cell_flow_threshold",
    }
    real_defaults = S.set_default_settings_preprocess_generate_masks
    real_organelle_defaults = S._set_organelle_defaults
    declared = real_defaults(dict(
        src=str(tmp_path), pipeline_style="v2", cell_channel=0,
        nucleus_channel=None, pathogen_channel=None, seg_qc="off"))
    events = []
    reads = []
    dispatched = []

    class ObservedSettings(dict):
        def get(self, key, default=None):
            if key in processing_keys:
                assert "defaults" in events, f"processing read before defaults: {key}"
                reads.append(key)
            return super().get(key, default)

    def defaults(settings):
        events.append("defaults")
        return real_defaults(settings)

    def organelle_defaults(settings):
        assert "defaults" in events
        events.append("organelles")
        return real_organelle_defaults(settings)

    def run(src, **kwargs):
        assert events == ["defaults", "organelles", "organelles"]
        dispatched.append(kwargs)
        return {"stacks": []}

    monkeypatch.setattr(S, "set_default_settings_preprocess_generate_masks", defaults)
    monkeypatch.setattr(S, "_set_organelle_defaults", organelle_defaults)
    monkeypatch.setattr(pipeline_v2, "run_v2", run)
    for explicit in (None, 0.25):
        events.clear()
        reads.clear()
        source = tmp_path / ("declared" if explicit is None else "explicit")
        source.mkdir()
        settings = ObservedSettings(
            src=str(source), pipeline_style="v2", cell_channel=0,
            nucleus_channel=None, pathogen_channel=None, seg_qc="off")
        if explicit is not None:
            settings["cell_flow_threshold"] = explicit
        core.preprocess_generate_masks(settings)
        actual = dispatched[-1]
        assert processing_keys <= set(reads)
        assert actual["flow_threshold"] == (
            declared["cell_flow_threshold"] if explicit is None else explicit)
        assert actual["cellprob_threshold"] == declared["cell_cellprob_threshold"]
        assert actual["batch_fields"] == declared["batch_fields"]
        assert actual["model_name"] == declared["cell_model_name"]
        assert actual["postprocess_settings"] is settings
        assert actual["illumination_settings"] is settings
        if explicit is not None:
            assert actual["flow_threshold"] == explicit
    assert len(dispatched) == 2


def test_the_organelle_defaults_are_applied_too():
    assert "_set_organelle_defaults" in _v2_branch_source()


def test_cell_ft_is_the_declared_value_not_the_inline_fallback():
    """The one fallback that changes segmentation output."""
    declared = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/does-not-matter"})["cell_flow_threshold"]
    # 1.0 until 2026-09-02, when ab656821b raised it to 100 ("accepts every
    # candidate"). 2026-09-19 (#123, ledger 428): 0.4, Cellpose's own
    # default, by the maintainer's decision. The point of this test is not
    # the number: whatever is DECLARED is what the v2 branch has to read.
    # Declared and inline fallback are now both 0.4, so a v2 branch that
    # read the fallback would no longer change the answer -- until one of
    # the two moves, which is why the ordering test above stays.
    assert declared == 0.4, (
        "the declared cell_flow_threshold moved; this test pins the value the v2 branch "
        "must now agree with")

    body = _v2_branch_source()
    assert "settings.get('cell_flow_threshold', 0.4)" in body or \
           'settings.get("cell_flow_threshold", 0.4)' in body, (
        "the inline fallback is gone entirely -- fine, but then this test "
        "should assert the new spelling instead")


def test_the_defaults_helper_is_idempotent():
    """It is now called twice on the v1 path, so it must be setdefault-only."""
    once = S.set_default_settings_preprocess_generate_masks({"src": "/tmp/x"})
    twice = S.set_default_settings_preprocess_generate_masks(dict(once))
    assert once == twice


def test_a_value_the_caller_set_still_wins():
    """setdefault, not overwrite: an explicit cell_flow_threshold must survive."""
    given = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/x", "cell_flow_threshold": 0.25})
    assert given["cell_flow_threshold"] == 0.25
