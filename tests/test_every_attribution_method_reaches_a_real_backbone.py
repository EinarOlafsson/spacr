"""Grad-CAM, saliency, and the rest, driven on backbones spaCR builds.

Instruction 236 B6: "GRAD-CAM, SALIENCY, AND THE SURROGATE MODELS."

WHAT WAS FOUND. `attention_rollout` could not run on ANY architecture
spaCR offers. Its docstring says it "reads spaCR's
torch.nn.MultiheadAttention blocks, which return their attention weights
from forward" -- and torchvision's ViT calls them with
`need_weights=False`, so they return nothing and a forward hook has
nothing to capture. The method raised "has MultiheadAttention blocks but
none returned attention weights" on vit_b_16, which is the only backbone
family it exists for. The docstring described what the blocks would do if
somebody asked them, and nobody was asking.

The blocks are asked now. `need_weights=True` also takes PyTorch off its
fused kernel -- which is the point, since the fused path computes no
explicit attention matrix at all -- and it runs once, under no_grad, for
one image.

The two refusals that remain are correct and say so: swin_t's shifted-
window attention is not `nn.MultiheadAttention`, and a convnet has no
attention at all. Both messages point at the CAM family instead.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacr.attribution import (Attribution, NoSpatialLayerError,   # noqa: E402
                               attribute, methods_by_family)


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(0).random((3, 224, 224)).astype("float32")


def _backbone(name, num_classes=2):
    from spacr.utils import choose_model

    model = choose_model(name, torch.device("cpu"), init_weights=False,
                         num_classes=num_classes)
    model.eval()
    return model


@pytest.fixture(scope="module")
def convnet():
    return _backbone("resnet18")


@pytest.fixture(scope="module")
def transformer():
    return _backbone("vit_b_16")


#: Methods that run on a plain convnet, and what each one costs. Excluded:
#: attention_rollout (needs attention) and deeplift (captum refuses a model
#: that reuses one ReLU module, which resnet does -- and says so).
ON_A_CONVNET = ["saliency", "input_x_gradient", "integrated_gradients",
                "guided_backprop", "eigencam", "gradcam", "gradcam_pp",
                "layercam", "xgradcam"]


class TestTheMapsAreReal:
    @pytest.mark.parametrize("method", ON_A_CONVNET)
    def test_it_produces_a_finite_map_at_the_input_resolution(
            self, convnet, image, method):
        """A map that is the wrong size, or holds a NaN, is a picture that
        cannot be laid over the cell it is explaining."""
        if method in ("gradcam", "gradcam_pp", "layercam", "xgradcam"):
            pytest.importorskip("torchcam")
        answer = attribute(convnet, image, method=method)
        assert isinstance(answer, Attribution)
        assert answer.map.shape == (224, 224)
        assert np.isfinite(answer.map).all()

    @pytest.mark.parametrize("method", ON_A_CONVNET)
    def test_it_names_the_backend_that_produced_it(self, convnet, image,
                                                   method):
        """Three libraries draw these maps and they do not agree on scale
        or sign; a reader comparing two panels needs to know which drew
        which."""
        if method in ("gradcam", "gradcam_pp", "layercam", "xgradcam"):
            pytest.importorskip("torchcam")
        answer = attribute(convnet, image, method=method)
        assert answer.backend in ("captum", "torchcam", "spacr")

    def test_the_families_are_the_four_it_documents(self):
        assert set(methods_by_family()) == {"attention", "gradient", "cam",
                                            "perturbation"}


class TestAttentionRollout:
    def test_it_runs_on_the_transformer_it_exists_for(self, transformer,
                                                      image):
        """THE DEFECT. It raised on vit_b_16 -- one of the ten backbones
        spaCR offers, and the only family this method applies to."""
        answer = attribute(transformer, image, method="attention_rollout")
        assert answer.map.shape == (224, 224)
        assert np.isfinite(answer.map).all()

    def test_it_rolls_every_block(self, transformer, image):
        """Twelve for vit_b_16. One captured block would be a map of the
        first layer wearing the name of the whole rollout."""
        answer = attribute(transformer, image, method="attention_rollout")
        assert "12 attention blocks" in " ".join(answer.notes)

    def test_it_says_it_is_not_class_conditional(self, transformer, image):
        """The map is identical for every target -- it describes where
        information flowed, not what the model concluded. A reader who
        thought otherwise would read it as evidence the classes were
        separated."""
        answer = attribute(transformer, image, method="attention_rollout")
        assert any("not class-conditional" in note for note in answer.notes)

    def test_the_model_is_left_exactly_as_it_was_found(self, transformer,
                                                       image):
        """`forward` is replaced for the duration of one pass. A model left
        with the override in place would carry it into training, where
        need_weights=True disables the fused kernel on every step."""
        blocks = [m for m in transformer.modules()
                  if isinstance(m, torch.nn.MultiheadAttention)]
        before = [block.forward for block in blocks]
        attribute(transformer, image, method="attention_rollout")
        assert [block.forward for block in blocks] == before

    def test_it_is_restored_even_when_the_pass_fails(self, transformer):
        """The undo is in a finally, or one bad image poisons the model."""
        blocks = [m for m in transformer.modules()
                  if isinstance(m, torch.nn.MultiheadAttention)]
        before = [block.forward for block in blocks]
        with pytest.raises(Exception):
            attribute(transformer, np.zeros((3, 7, 7), dtype="float32"),
                      method="attention_rollout")
        assert [block.forward for block in blocks] == before

    def test_a_convnet_is_refused_and_pointed_at_a_cam(self, convnet, image):
        """A refusal that names the alternative is the difference between
        'no' and 'no, do this instead'."""
        with pytest.raises(NoSpatialLayerError) as raised:
            attribute(convnet, image, method="attention_rollout")
        assert "CAM" in str(raised.value)

    def test_a_window_attention_backbone_is_refused_by_name(self, image):
        """swin_t's shifted-window attention is not
        `nn.MultiheadAttention`, so there is nothing to roll."""
        with pytest.raises(NoSpatialLayerError):
            attribute(_backbone("swin_t"), image,
                      method="attention_rollout")


class TestTheRefusalsThatShouldStay:
    def test_a_cam_on_a_transformer_says_why_not(self, transformer, image):
        """The only convolution in a ViT is the patch projection, which
        runs before every attention block -- so a CAM hooked there explains
        the patch embedding, not the model."""
        pytest.importorskip("torchcam")
        with pytest.raises(NoSpatialLayerError) as raised:
            attribute(transformer, image, method="gradcam")
        assert "attention" in str(raised.value)

    def test_deeplift_refuses_a_model_that_reuses_an_activation(
            self, convnet, image):
        """A captum limitation, not a spaCR one -- and the message says
        which module is shared rather than producing a wrong map."""
        from spacr.attribution import AttributionError as Refusal

        with pytest.raises(Refusal, match="reuses one activation module"):
            attribute(convnet, image, method="deeplift")

    def test_deeplift_runs_where_it_can(self, image):
        """The refusal above must be about the model, not about deeplift."""
        answer = attribute(_backbone("efficientnet_b0"), image,
                           method="deeplift")
        assert answer.map.shape == (224, 224)


class TestTheOptionalBackend:
    def test_a_missing_backend_names_the_install(self):
        """Five of the six CAM variants come from torchcam, which is an
        extra rather than a core dependency -- deliberately, because both
        of its releases declare a spurious numpy<2 pin. The refusal has to
        name the extra, since `gradcam` is the method most people reach
        for first."""
        import inspect

        from spacr import attribution

        source = inspect.getsource(attribution._torchcam_cam)
        assert "pip install" in source
        assert "torchcam" in source
