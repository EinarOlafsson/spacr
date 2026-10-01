"""Item 18: the attribution methods the maintainer chose to build for Classify.

SHAP (GradientSHAP / DeepSHAP on crops, TreeSHAP / KernelSHAP on features),
HiRes-CAM, Ablation-CAM, Chefer transformer relevance, saliency with
SmoothGrad, native feature importance and permutation importance. Every model
here is tiny and hand-wired so the right answer is known: the pixel models
read only the top-left corner of channel 0, and the tabular data has one
planted informative feature.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from spacr import attribution as att  # noqa: E402

IMG = 16
CORNER = 6


class CornerCNN(nn.Module):
    """Score depends only on channel 0 inside the top-left corner."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.act = nn.ReLU()
        self.head = nn.Linear(4, 2)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.conv.weight[0, 0, 1, 1] = 1.0
            self.conv.weight[1, 1, 1, 1] = 1.0
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[1, 0] = 8.0
            self.head.weight[0, 0] = -8.0
        mask = torch.zeros(1, 1, IMG, IMG)
        mask[..., :CORNER, :CORNER] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        feat = self.act(self.conv(x)) * self.mask
        return self.head(feat.mean(dim=(2, 3)))


class TinyViT(nn.Module):
    """One attention block over 4x4 patches with a class token.

    Hand-wired: the patch embedding's first dimension is the patch mean of
    channel 0, attention is uniform (zero queries and keys), values and the
    output projection are the identity, and the head reads dimension 0 of the
    class token. Class 1's evidence therefore sits in the corner patch alone.
    """

    def __init__(self, dim=4, patch=4):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.head = nn.Linear(dim, 2)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.bias.zero_()
            self.embed.weight[0, 0] = 1.0 / (patch * patch)
            self.attn.in_proj_weight.zero_()
            self.attn.in_proj_bias.zero_()
            self.attn.in_proj_weight[2 * dim:] = torch.eye(dim)
            self.attn.out_proj.weight.copy_(torch.eye(dim))
            self.attn.out_proj.bias.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[1, 0] = 8.0
            self.head.weight[0, 0] = -8.0

    def forward(self, x):
        tokens = self.embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], 1)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.head(out[:, 0])


@pytest.fixture
def image():
    """Signal in channel 0's corner; distractor texture in channels 1-2 only."""
    rng = np.random.default_rng(0)
    img = np.zeros((3, IMG, IMG), dtype="float32")
    img[0, :CORNER, :CORNER] = 1.0
    img[1:] = rng.random((2, IMG, IMG)).astype("float32")
    return img


def _corner_contrast(amap):
    inside = amap[:CORNER, :CORNER].mean()
    outside = np.concatenate([amap[CORNER:, :].ravel(),
                              amap[:CORNER, CORNER:].ravel()]).mean()
    return float(inside), float(outside)


PIXEL_METHODS = ["hirescam", "ablation_cam", "gradient_shap",
                 "deeplift_shap", "saliency"]


class TestTheRegistry:
    def test_the_new_methods_are_registered_in_their_families(self):
        fams = att.methods_by_family()
        assert {"hirescam", "ablation_cam"} <= set(fams["cam"])
        assert set(fams["shap"]) == {"gradient_shap", "deeplift_shap"}
        assert {"chefer", "attention_rollout"} == set(fams["attention"])
        assert {"gradcam", "gradcam_pp"} <= set(fams["cam"])

    def test_the_torchcam_gradcam_is_reachable_as_a_cam_type(self):
        assert att.resolve_cam_type("torchcam_gradcam") == "gradcam"
        assert att.resolve_cam_type("torchcam_gradcam_pp") == "gradcam_pp"
        assert att.resolve_cam_type("gradcam") is None
        assert att.resolve_cam_type("hirescam") == "hirescam"
        with pytest.raises(att.UnknownMethodError):
            att.resolve_cam_type("grad_cam")

    def test_the_settings_menu_offers_every_cam_type(self):
        from spacr.settings_spec import _CAM_TYPE_CHOICES, _cam_type_choices

        assert tuple(_CAM_TYPE_CHOICES) == att.cam_type_choices()
        assert _cam_type_choices() == list(att.cam_type_choices())
        assert len(set(_CAM_TYPE_CHOICES)) == len(_CAM_TYPE_CHOICES)


class TestPixelMethodsFindThePlantedSignal:
    @pytest.mark.parametrize("method", PIXEL_METHODS)
    def test_map_shape_and_peak_in_the_corner(self, method, image):
        result = att.attribute(CornerCNN(), image, method, target=1)
        assert result.map.shape == (IMG, IMG)
        assert np.isfinite(result.map).all()
        assert not result.is_flat()
        inside, outside = _corner_contrast(result.map)
        assert inside > 5 * max(outside, 1e-9)
        row, col = result.peak()
        assert row < CORNER and col < CORNER

    def test_smoothgrad_saliency_keeps_the_signal(self, image):
        result = att.smoothgrad(CornerCNN(), image, "saliency", n_samples=4,
                                sigma=0.05, target=1, seed=0)
        inside, outside = _corner_contrast(result.map)
        assert inside > outside

    @pytest.mark.parametrize("method", ["hirescam", "ablation_cam",
                                        "gradient_shap", "deeplift_shap"])
    @pytest.mark.parametrize("n_out", [1, 3])
    def test_both_head_shapes(self, method, n_out, image):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(4, n_out))
        result = att.attribute(model, image, method)
        assert result.map.shape == (IMG, IMG)
        assert result.n_classes == (2 if n_out == 1 else n_out)

    def test_hirescam_on_a_real_resnet(self):
        from spacr.utils import choose_model

        model = choose_model("resnet18", torch.device("cpu"),
                             init_weights=False, num_classes=2).eval()
        img = np.random.default_rng(1).random((3, 64, 64)).astype("float32")
        result = att.attribute(model, img, "hirescam")
        assert result.map.shape == (64, 64)
        assert np.isfinite(result.map).all()


class TestChefer:
    def test_class_specific_relevance_lands_on_the_signal_patch(self, image):
        result = att.attribute(TinyViT(), image, "chefer", target=1)
        assert result.map.shape == (IMG, IMG)
        assert result.family == "attention"
        row, col = result.peak()
        assert row < 4 and col < 4
        other = att.attribute(TinyViT(), image, "chefer", target=0)
        assert not np.allclose(other.map, result.map)

    def test_it_refuses_a_convnet_with_the_reason(self, image):
        with pytest.raises(att.NoSpatialLayerError) as excinfo:
            att.attribute(CornerCNN(), image, "chefer")
        assert "not applicable" in str(excinfo.value)

    def test_it_runs_on_torchvision_vit(self):
        from spacr.utils import choose_model

        model = choose_model("vit_b_16", torch.device("cpu"),
                             init_weights=False, num_classes=2).eval()
        img = np.random.default_rng(2).random((3, 224, 224)).astype("float32")
        result = att.attribute(model, img, "chefer")
        assert result.map.shape == (224, 224)
        assert np.isfinite(result.map).all()
        assert not result.is_flat()


class TestNotApplicableIsGreyedWithAReason:
    def test_by_architecture_name(self):
        ok, reason = att.method_applicability("chefer", model_type="resnet50")
        assert not ok and "attention" in reason
        ok, reason = att.method_applicability("chefer", model_type="maxvit_t")
        assert not ok and "MaxViT" in reason
        ok, reason = att.method_applicability("hirescam",
                                              model_type="vit_b_16")
        assert not ok and "patch embedding" in reason
        ok, reason = att.method_applicability("deeplift_shap",
                                              model_type="resnet50")
        assert not ok and "ReLU" in reason
        assert att.method_applicability("chefer", model_type="vit_b_16")[0]
        assert att.method_applicability("hirescam", model_type="maxvit_t")[0]
        assert att.method_applicability("gradient_shap",
                                        model_type="vit_b_16")[0]

    def test_by_loaded_model(self):
        assert att.architecture_kind(TinyViT()) == "vit"
        assert att.architecture_kind(CornerCNN()) == "cnn"
        table = att.applicable_methods(model=TinyViT())
        assert table["chefer"][0] and not table["ablation_cam"][0]
        table = att.applicable_methods(model=CornerCNN())
        assert table["hirescam"][0] and not table["chefer"][0]

    def test_a_missing_backend_says_what_to_install(self, monkeypatch):
        real = importlib.util.find_spec

        def fake(name, *a, **k):
            if name in ("torchcam", "captum"):
                return None
            return real(name, *a, **k)

        monkeypatch.setattr(importlib.util, "find_spec", fake)
        ok, reason = att.method_applicability("gradcam", model_type="resnet50")
        assert not ok and "pip install" in reason
        ok, reason = att.method_applicability("gradient_shap")
        assert not ok and "pip install captum" in reason

    def test_legacy_cam_types(self):
        assert att.cam_type_applicability("saliency_image",
                                          model_type="vit_b_16")[0]
        assert not att.cam_type_applicability("gradcam",
                                              model_type="vit_b_16")[0]


class TestTheActivationPipeline:
    def _project(self, tmp_path):
        import tarfile
        from PIL import Image

        ds = tmp_path / "proj" / "datasets"
        ds.mkdir(parents=True)
        (tmp_path / "proj" / "measurements").mkdir()
        tar_path = ds / "ds.tar"
        with tarfile.open(tar_path, "w") as tar:
            for i in range(3):
                path = tmp_path / f"plate1_A01_1_{i}.png"
                arr = np.random.default_rng(i).integers(
                    0, 255, (32, 32, 3)).astype("uint8")
                Image.fromarray(arr).save(path)
                tar.add(path, arcname=path.name)
        return str(tar_path)

    def _settings(self, tmp_path, cam_type):
        from spacr.utils import TorchModel

        model_path = tmp_path / "m.pth"
        torch.save(TorchModel(model_name="resnet18", pretrained=False,
                              num_classes=1, image_size=32), str(model_path))
        return {
            "dataset": self._project(tmp_path), "model_path": str(model_path),
            "model_type": "resnet18", "cam_type": cam_type,
            "target_layer": None, "image_size": 32, "batch_size": 3,
            "channels": [1, 2, 3], "normalize": False,
            "normalize_input": True, "save": True, "plot": True,
            "correlation": False, "overlay": True, "shuffle": False,
            "n_jobs": 0, "manders_thresholds": [15, 50, 75],
        }

    def test_hirescam_writes_maps_and_the_overlay_grid(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from spacr.deep_spacr import generate_activation_map

        settings = self._settings(tmp_path, "hirescam")
        generate_activation_map(settings)
        out = os.path.join(os.path.dirname(settings["dataset"]), "ds",
                           "hirescam")
        pngs = [f for _r, _d, fs in os.walk(out) for f in fs
                if f.endswith(".png")]
        grids = os.listdir(os.path.join(out, "batch_grids"))
        assert len(pngs) == 3
        assert any(g.endswith(".pdf") for g in grids)

    def test_a_method_that_does_not_apply_is_refused_before_the_run(
            self, tmp_path):
        from spacr.deep_spacr import generate_activation_map

        with pytest.raises(ValueError, match="does not apply"):
            generate_activation_map(self._settings(tmp_path, "chefer"))


def _planted(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 5)),
                     columns=["noise_a", "noise_b", "signal", "noise_c",
                              "noise_d"])
    y = (x["signal"] + 0.1 * rng.normal(size=n) > 0).astype(int)
    return x, y


class TestTabularImportance:
    def test_every_measure_ranks_the_planted_feature_first(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        from spacr.surrogate import rank_feature_importance

        x, y = _planted()
        model = RandomForestClassifier(n_estimators=40, random_state=0)
        model.fit(x[:160], y[:160])
        methods = ["gain", "permutation"]
        if importlib.util.find_spec("shap") is not None:
            methods.append("shap")
        table, paths = rank_feature_importance(
            model, x[160:], y[160:], methods=methods, destination=str(tmp_path))
        assert table.loc[0, "feature"] == "signal"
        assert table.loc[0, "rank"] == 1
        for column in methods:
            assert table.sort_values(column, ascending=False).iloc[0][
                "feature"] == "signal"
        assert os.path.isfile(paths["importance"])
        assert os.path.isfile(paths["importance_png"])
        written = pd.read_csv(paths["importance"])
        assert list(written["feature"])[0] == "signal"

    def test_kernel_shap_for_a_model_without_trees(self):
        pytest.importorskip("shap")
        from sklearn.linear_model import LogisticRegression
        from spacr.surrogate import rank_feature_importance

        x, y = _planted(n=120)
        model = LogisticRegression().fit(x, y)
        table, _paths = rank_feature_importance(
            model, x[:40], y[:40], methods=["gain", "shap"],
            shap_explainer="kernel", shap_max_samples=20)
        assert "gain" not in table
        assert any("gain omitted" in w for w in table.attrs["warnings"])
        assert table.sort_values("shap", ascending=False).iloc[0][
            "feature"] == "signal"

    def test_availability_greys_what_does_not_apply(self, monkeypatch):
        from spacr.surrogate import importance_method_availability

        table = importance_method_availability(
            model_family="hist_gradient_boosting")
        assert not table["gain"]["available"]
        assert "no native" in table["gain"]["reason"]
        assert table["permutation"]["available"]
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name, *a, **k: None if name == "shap" else real(name, *a, **k))
        table = importance_method_availability(model_family="random_forest")
        assert not table["tree_shap"]["available"]
        assert "pip install shap" in table["kernel_shap"]["reason"]

    def test_the_surrogate_computes_only_the_chosen_measures(self):
        from spacr.surrogate import fit_surrogate

        x, y = _planted(n=200)
        frame = x.copy()
        frame["cv_prediction"] = y
        frame["plateID"] = "plate1"
        frame["rowID"] = [f"r{i % 4}" for i in range(len(frame))]
        frame["columnID"] = [f"c{(i // 4) % 5}" for i in range(len(frame))]
        result = fit_surrogate(frame, importance_methods=["permutation"],
                               n_estimators=30, verbose=False)
        assert "permutation" in result.importance
        assert "gain" not in result.importance
        assert "shap" not in result.importance
        assert result.importance.iloc[0]["feature"] == "signal"


class TestTheExplanationScreens:
    def test_the_activation_menu_greys_methods_for_the_model_type(self, qtbot):
        from PySide6.QtCore import Qt
        from spacr.qt.screens.settings_model import SettingsWidgets

        builder = SettingsWidgets("activation")
        builder.build_sections()
        combo = builder._built_control("cam_type")
        names = [combo.itemData(i) or combo.itemText(i)
                 for i in range(combo.count())]
        assert {"hirescam", "chefer", "gradient_shap"} <= set(names)

        def enabled(name):
            index = names.index(name)
            return combo.model().item(index).isEnabled(), combo.itemData(
                index, Qt.ToolTipRole)

        builder.set_value_for_key("model_type", "resnet50")
        ok, tip = enabled("chefer")
        assert not ok and "Not applicable" in tip
        assert enabled("hirescam")[0]
        builder.set_value_for_key("model_type", "vit_b_16")
        assert enabled("chefer")[0]
        assert not enabled("hirescam")[0]

    def test_the_explain_panel_greys_gain_for_hist_gradient_boosting(
            self, qtbot):
        from spacr.qt.screens.model_explanation import ExplainCvPanel

        panel = ExplainCvPanel()
        qtbot.addWidget(panel)
        assert panel.importance_boxes["gain"].isEnabled()
        panel.backend.setCurrentIndex(
            panel.backend.findData("hist_gradient_boosting"))
        box = panel.importance_boxes["gain"]
        assert not box.isEnabled() and not box.isChecked()
        assert "Not applicable" in box.toolTip()
        assert "gain" not in panel.importance_methods()
        panel.backend.setCurrentIndex(panel.backend.findData("random_forest"))
        assert panel.importance_boxes["gain"].isChecked()


class TestTheActivationSweepOffersTheNewMethods:
    """Item 18 follow-up: the hyperparameter sweep's cam_type candidates."""

    NEW = ("hirescam", "ablation_cam", "gradient_shap", "deeplift_shap",
           "chefer")

    def test_the_default_grid_names_every_new_method(self):
        from spacr.hyperparam import DEFAULT_SPACES

        methods = DEFAULT_SPACES["activation"]["cam_type"]
        for name in self.NEW:
            assert name in methods
        assert all(name in att.ATTRIBUTION_METHODS for name in methods)
        families = {att.ATTRIBUTION_METHODS[m].family for m in methods}
        assert families == {"cam", "gradient", "shap", "perturbation",
                            "attention"}

    def test_the_form_aliases_reach_the_registry_names(self):
        from spacr.hyperparam import _activation_params

        assert _activation_params({"cam_type": "torchcam_gradcam"})[0] \
            == "gradcam"
        assert _activation_params({"cam_type": "torchcam_gradcam_pp"})[0] \
            == "gradcam_pp"
        assert _activation_params({"cam_type": "saliency_image"})[0] \
            == "saliency"
        assert _activation_params({"cam_type": "hirescam"})[0] == "hirescam"

    def _space(self, model, model_type=None):
        from spacr.hyperparam import (DEFAULT_SPACES, ActivationSearchData,
                                      SearchSpace,
                                      _applicable_activation_space)

        data = ActivationSearchData(model=model, images=[],
                                    model_type=model_type)
        space = SearchSpace(DEFAULT_SPACES["activation"])
        return _applicable_activation_space(space, data)

    def test_a_cnn_keeps_the_cams_and_shap_and_drops_chefer(self):
        space, notes = self._space(CornerCNN())
        kept = space.params["cam_type"]
        for name in ("hirescam", "ablation_cam", "gradient_shap",
                     "deeplift_shap", "saliency"):
            assert name in kept
        assert "chefer" not in kept
        assert any("'chefer'" in note and "Vision Transformers" in note
                   for note in notes)

    def test_a_resnet_also_drops_deeplift_shap_with_the_reason(self):
        from torchvision.models import resnet18

        space, notes = self._space(resnet18(weights=None), "resnet18")
        kept = space.params["cam_type"]
        assert "deeplift_shap" not in kept and "chefer" not in kept
        assert "hirescam" in kept and "gradient_shap" in kept
        assert any("'deeplift_shap'" in note and "ReLU" in note
                   for note in notes)

    def test_a_vit_keeps_chefer_and_drops_every_cam(self):
        space, notes = self._space(TinyViT(), "vit_b_16")
        kept = space.params["cam_type"]
        assert "chefer" in kept and "gradient_shap" in kept
        for name in ("gradcam", "gradcam_pp", "layercam", "hirescam",
                     "ablation_cam"):
            assert name not in kept
            assert any(f"'{name}'" in note for note in notes)

    def test_nothing_applicable_is_refused_with_the_reasons(self):
        from spacr.hyperparam import (ActivationSearchData, SearchSpace,
                                      _applicable_activation_space)

        data = ActivationSearchData(model=CornerCNN(), images=[])
        with pytest.raises(ValueError, match="chefer"):
            _applicable_activation_space(
                SearchSpace({"cam_type": ["chefer"]}), data)

    def test_the_sweep_runs_the_filtered_grid_and_says_what_it_left_out(
            self, monkeypatch):
        from spacr import hyperparam as hp

        captured = {}

        def grid(fit, space, **kwargs):
            captured.update(space=space, notes=kwargs["notes"])
            return hp.SearchResult(metric="deletion_auc")

        monkeypatch.setattr(hp, "grid_search", grid)
        data = hp.ActivationSearchData(model=CornerCNN(), images=[object()])
        hp.activation_search(
            data, hp.SearchSpace({"cam_type": ["hirescam", "chefer"]}))
        assert tuple(captured["space"].params["cam_type"]) == ("hirescam",)
        assert any("'chefer' was left out" in n for n in captured["notes"])

    def test_a_replaced_attribution_call_is_not_filtered(self, monkeypatch):
        from spacr import hyperparam as hp

        captured = {}

        def grid(fit, space, **kwargs):
            captured.update(space=space)
            return hp.SearchResult(metric="deletion_auc")

        monkeypatch.setattr(hp, "grid_search", grid)
        data = hp.ActivationSearchData(model=CornerCNN(), images=[object()])
        space = hp.SearchSpace({"cam_type": ["chefer"]})
        hp.activation_search(data, space,
                             attribute_fn=lambda *_a: np.zeros((2, 2)))
        assert captured["space"] is space
