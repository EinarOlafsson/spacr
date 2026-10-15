"""Operator command argument contracts and actual local header description."""
import json

import numpy as np
import pytest
import tifffile

from tools.web_mask_generator import run


@pytest.fixture
def arguments(tmp_path):
    return [str(tmp_path / "input.tif"), "--output", str(tmp_path / "output.zip"),
            "--checkpoint", str(tmp_path / "trusted.CP_model"), "--sha256", "ab" * 32,
            "--model-id", "toxoplasma_plaque_v4", "--model-version", "v4"]


def test_describe_reads_real_headers_without_model_or_inference(tmp_path, monkeypatch, capsys):
    image = tmp_path / "cells.tif"
    tifffile.imwrite(image, np.arange(12, dtype=np.uint16).reshape(3, 4), metadata={"axes": "YX"})
    monkeypatch.setattr(run, "run_plaque_download", lambda *a, **kw: pytest.fail("description ran inference"))
    assert run.main([str(image), "--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == {"format": "TIFF", "series": [{"axes": "YX", "dtype": "uint16", "shape": [3, 4]}]}


@pytest.mark.parametrize("scale,diameter", [(["--diameter", "52.5"], 52.5), (["--native-scale"], None)])
def test_operator_choices_reach_adapter_without_rewriting(arguments, monkeypatch, capsys, scale, diameter):
    calls = []
    def adapter(output, **kwargs):
        calls.append((output, kwargs))
        return output
    monkeypatch.setattr(run, "run_plaque_download", adapter)
    assert run.main(arguments + scale + ["--index", "T=2", "--index", "Z=1", "--channel", "1", "--channel", "0", "--series", "2", "--flow-threshold", "0.6", "--cellprob-threshold", "-1"]) == 0
    output, call = calls[0]
    assert call["effective_settings"] == {"diameter": diameter, "flow_threshold": 0.6, "CP_prob": -1}
    assert call["selection"] == {"series": 2, "indices": {"T": 2, "Z": 1}, "channels": [1, 0], "page": 0, "pdf_dpi": 150}
    assert call["expected_sha256"] == "ab" * 32
    assert json.loads(capsys.readouterr().out) == {"result_zip": str(output)}


@pytest.mark.parametrize("extra", [[], ["--diameter", "52", "--native-scale"], ["--native-scale", "--index", "T=0", "--index", "T=1"], ["--native-scale", "--index", "t=0"], ["--native-scale", "--index", "T=-1"], ["--native-scale", "--index", "TZ=1"], ["--native-scale", "--index", "T=word"], ["--native-scale", "--index", "Z"]])
def test_ambiguous_or_invalid_requests_cannot_start_inference(arguments, extra, monkeypatch):
    monkeypatch.setattr(run, "run_plaque_download", lambda *a, **kw: pytest.fail("bad command ran inference"))
    with pytest.raises(SystemExit) as error:
        run.main(arguments + extra)
    assert error.value.code == 2


def test_missing_model_provenance_is_refused_before_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "run_plaque_download", lambda *a, **kw: pytest.fail("bad command ran inference"))
    with pytest.raises(SystemExit) as error:
        run.main([str(tmp_path / "input.tif"), "--native-scale"])
    assert error.value.code == 2
