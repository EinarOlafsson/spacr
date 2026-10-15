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


@pytest.mark.parametrize('family', ['toxoplasma_well_detector', *run.OBJECT_ROLES])
def test_remaining_operator_routes_have_family_specific_settings(arguments, monkeypatch, family):
    called = []
    def adapter(output, **kw):
        called.append(kw)
        return output
    monkeypatch.setattr(run, 'run_mask_download', adapter)
    monkeypatch.setattr(run, 'run_well_download', adapter)
    extra = [] if family == 'toxoplasma_well_detector' else ['--settings-json', '{"background": 123}']
    assert run.main(arguments + ['--family', family, '--magnification', '40'] + extra) == 0
    settings = called[0]['effective_settings']
    if family == 'toxoplasma_well_detector':
        assert settings == {'confidence': 0.25, 'min_axis_ratio': 0.7, 'imgsz': 640}
        assert 'family' not in called[0]
    else:
        assert called[0]['family'] == family
        assert settings['diameter'] is None and settings['magnification'] == 40 and settings['background'] == 123
        assert settings['CP_prob'] == (-1 if run.OBJECT_ROLES[family] == 'pathogen' else 0)


@pytest.mark.parametrize('extra', ['bad', '[]', '{"checkpoint": "/client"}'])
def test_mask_operator_rejects_invalid_advanced_settings(arguments, monkeypatch, extra):
    monkeypatch.setattr(run, 'run_mask_download', lambda *a, **kw: pytest.fail('invalid settings ran model'))
    with pytest.raises(SystemExit):
        run.main(arguments + ['--family', 'live_cell', '--settings-json', extra])


def test_generic_cli_auto_and_nonmask_setting_refusal(arguments, monkeypatch):
    called = []
    monkeypatch.setattr(run, 'run_mask_download', lambda output, **kw: called.append(kw) or output)
    assert run.main(arguments + ['--family', 'live_cell', '--native-scale']) == 0
    assert called[0]['effective_settings']['diameter'] == 'auto'
    with pytest.raises(SystemExit):
        run.main(arguments + ['--native-scale', '--settings-json', '{"background": 10}'])
