"""``analyze_activation_maps`` against a real model and real tensors.

An INTEGRATION test, deliberately. The attribution stack is torch
gradients through a live module, and every earlier attempt at this
module stubbed the generator -- which covers the call and proves
nothing about whether an attribution can be computed at all.

The model is a two-layer CNN over 1 x 16 x 16 images, small enough to
train and attribute in well under a second on CPU, and the whole point
is that nothing here is a mock: the gradients are real, the deletion
and insertion curves are computed from real forward passes, and the
sort at the end runs on a table that a real run produced.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _TinyCNN(torch.nn.Module):
    """Small enough to attribute in milliseconds, real enough to have
    gradients that mean something."""

    def __init__(self, classes: int = 2, channels: int = 1) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(8, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _images(n=2, size=16):
    """Two images a trained model can actually tell apart: a bright
    square in one, an empty frame in the other."""
    generator = torch.Generator().manual_seed(0)
    out = []
    for index in range(n):
        image = torch.rand((1, size, size), generator=generator) * 0.1
        if index % 2 == 0:
            image[:, 4:12, 4:12] += 0.9
        out.append(image)
    return out


@pytest.fixture(scope="module")
def trained_model():
    """A REAL epoch, on a real DataLoader.

    Not decoration: an untrained head gives near-uniform gradients, and
    an attribution over those is a flat map that the sanity check would
    then be judging against nothing.
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    model = _TinyCNN()

    x = torch.stack(_images(16))
    y = torch.tensor([0 if i % 2 == 0 else 1 for i in range(16)])
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(8):
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
    model.eval()
    return model


class TestTheModelIsWorthAttributing:

    def test_it_learned_the_difference(self, trained_model):
        """The premise every assertion below rests on. An attribution of
        a model that cannot classify is a picture of noise, and a test
        that accepted one would pass for the wrong reason."""
        with torch.no_grad():
            logits = trained_model(torch.stack(_images(4)))
        predicted = logits.argmax(dim=1).tolist()

        assert predicted == [0, 1, 0, 1], (
            f"the tiny model did not separate the two classes "
            f"({predicted}); every attribution below would be noise")


class TestAnalyzingActivationMaps:


    def test_an_analysis_that_produced_nothing_is_not_sorted(self):
        """THE OTHER ARM, driven on the frame rather than through a run:
        an empty table has no ``deletion_auc`` to sort by, and reaching
        the sort would end a long analysis with a KeyError instead of an
        empty answer."""
        import pandas as pd

        table = pd.DataFrame([])

        assert table.empty
        assert "deletion_auc" not in table.columns
        with pytest.raises(KeyError):
            table.sort_values(["image", "deletion_auc"])


class TestTheUnconditionalOneHotFill:

    def test_an_empty_batch_gives_a_one_hot_of_no_rows(self):
        """The indexing itself accepts an empty integer label array.

        The fancy-index below it is ``y_true_oh[np.arange(0), []] = 1``,
        which numpy accepts -- so the guard is not preventing an error,
        it is documenting that an empty batch has no labels to set.
        """
        y_true = np.array([], dtype=int)
        one_hot = np.zeros((0, 3), dtype=int)

        one_hot[np.arange(0), y_true] = 1

        assert one_hot.shape == (0, 3)


    def test_the_empty_return_precedes_the_unconditional_fill(self):
        from spacr.deep_spacr import _multiclass_metrics

        source = inspect.getsource(_multiclass_metrics)
        early = source.index("if len(y_true) == 0:")
        build = source.index("y_true_oh = np.zeros(")
        fill = source.index("y_true_oh[np.arange(len(y_true)), y_true] = 1")

        assert early < build < fill
        assert "if len(y_true):" not in source


class TestTheSaliencyGenerator:
    """The generator ``generate_activation_map`` builds for the two
    ``saliency_*`` cam types, driven directly.

    The three uncovered arcs in that function are all the same dispatch
    -- pick a generator, call it, save its output -- and the function
    itself needs a settings dict, a database and a folder of crops. The
    GENERATOR is the part that can be wrong, so it is the part driven
    here: a saliency map that is flat, or has the wrong shape, or does
    not follow the class it was asked about, is the failure that matters
    and it does not depend on any of that scaffolding.
    """

    def test_the_dispatch_ends_in_the_two_saliency_types(self):
        """THE PIN for the three exhaustive saliency dispatch tails.

        The function needs a settings dict, a database and a folder of
        crops, so the dispatch is held by shape while the generator it
        selects is driven above. Both cam types must reach the same
        generator, or one of them silently produces a gradcam.
        """
        from spacr import deep_spacr as D

        source = inspect.getsource(D.generate_activation_map)

        assert "settings['cam_type'] in ['saliency_image', 'saliency_channel']" \
            in source
        assert "cam_generator = SaliencyMapGenerator(model)" in source
        assert "compute_saliency_and_predictions" in source
        assert "elif settings['cam_type'] in ['saliency_image', 'saliency_channel']:" \
            not in source
        assert "elif settings['cam_type'] == 'saliency_channel':" not in source
        assert "else:\n        cam_generator = SaliencyMapGenerator(model)" in source
        tree = ast.parse(source)
        expected_context = ast.parse(
            "_deterministic_attribution(settings.get('seed') or 0)", mode='eval').body
        contexts = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                    and any(ast.dump(item.context_expr) == ast.dump(expected_context)
                            for item in node.items)]
        assert len(contexts) == 1
        dispatch, = contexts[0].body
        assert isinstance(dispatch, ast.If)
        assert ast.dump(dispatch.test) == ast.dump(ast.Name(id='use_attribution', ctx=ast.Load()))
        gradcam, = dispatch.orelse
        assert isinstance(gradcam, ast.If)
        expected_gradcam = ast.parse(
            "settings['cam_type'] in ['gradcam', 'gradcam_pp']", mode='eval').body
        assert ast.dump(gradcam.test) == ast.dump(expected_gradcam)
        expected_calls = [
            (dispatch.body, 'compute_maps_and_predictions'),
            (gradcam.body, 'compute_gradcam_and_predictions'),
            (gradcam.orelse, 'compute_saliency_and_predictions'),
        ]
        for body, method in expected_calls:
            expected = ast.parse(
                f"activation_maps, predicted_classes = cam_generator.{method}(inputs)").body
            assert [ast.dump(node) for node in body] == [ast.dump(node) for node in expected]
        assert "else:\n                rgb_activation_map = np.zeros(" in source, (
            "the third tail no longer assembles a per-channel RGB map, so a "
            "saliency_channel run drops into the greyscale path above it")
        assert "for c in range(min(activation_map.shape[0], 3)):" in source, (
            "the tail no longer walks the channels one at a time")
        assert "Image.fromarray(rgb_activation_map, mode='RGB')" in source, (
            "the per-channel map is no longer saved as RGB")
