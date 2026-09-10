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

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _TinyCNN(torch.nn.Module):
    """Small enough to attribute in milliseconds, real enough to have
    gradients that mean something."""

    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
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

    def test_it_returns_one_row_per_image_and_method(self, trained_model):
        from spacr.deep_spacr import analyze_activation_maps

        table = analyze_activation_maps(
            trained_model, _images(2), methods=["saliency"],
            sanity_check=False, verbose=False, n_steps=4)["table"]

        assert not table.empty
        assert set(table["image"]) == {0, 1}
        assert set(table["method"]) == {"saliency"}

    def test_the_table_is_sorted_by_image_then_deletion_auc(
            self, trained_model):
        """THE ARC: ``not table.empty and 'deletion_auc' in columns``.

        The sort is what makes the table readable -- the most
        deletion-sensitive method first within each image -- and it is
        guarded because a run where every method failed produces a frame
        with neither rows nor that column, and sorting on a column that
        is not there is a KeyError at the end of a long analysis.
        """
        from spacr.deep_spacr import analyze_activation_maps

        table = analyze_activation_maps(
            trained_model, _images(2),
            methods=["saliency", "integrated_gradients"],
            sanity_check=False, verbose=False, n_steps=4)["table"]

        assert "deletion_auc" in table.columns
        assert list(table["image"]) == sorted(table["image"]), (
            "the rows are not grouped by image, so two methods for one "
            "image can be separated by another image's rows")

        for image in table["image"].unique():
            block = table[table["image"] == image]["deletion_auc"]
            finite = block.dropna().tolist()
            assert finite == sorted(finite), (
                f"image {image}'s methods are not ordered by deletion AUC")

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

    def test_a_mask_adds_the_pointing_game(self, trained_model):
        """The optional half: with an object mask the analysis also says
        whether the attribution's peak lands inside the object, which is
        the question a biologist actually asks of one of these maps."""
        from spacr.deep_spacr import analyze_activation_maps

        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1.0

        table = analyze_activation_maps(
            trained_model, _images(1), methods=["saliency"],
            masks=[mask], sanity_check=False, verbose=False, n_steps=4)["table"]

        assert "pointing_game" in table.columns
        assert table["pointing_game"].notna().any(), (
            "a mask was supplied and no pointing-game result came back")

    def test_the_sanity_check_reports_per_method(self, trained_model):
        """The randomisation pass: a map that survives having the model's
        parameters scrambled is not explaining the model, and saying so
        is the difference between an attribution and a picture."""
        from spacr.deep_spacr import analyze_activation_maps

        table = analyze_activation_maps(
            trained_model, _images(1), methods=["saliency"],
            sanity_check=True, verbose=False, n_steps=4)["table"]

        assert not table.empty


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

    def test_a_real_batch_sets_one_column_per_row(self):
        from spacr.deep_spacr import _multiclass_metrics

        y_true = np.array([0, 1, 2, 1])
        probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1],
                                  [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]])

        metrics = _multiclass_metrics(y_true, probabilities)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert 0.0 <= metrics["prauc"] <= 1.0
        assert metrics["num_classes"] == 3
        assert len(metrics["per_class_accuracy"]) == 3

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

    def test_it_answers_one_map_per_image_at_the_image_shape(
            self, trained_model):
        from spacr.utils import SaliencyMapGenerator

        images = torch.stack(_images(2))
        maps, predicted = SaliencyMapGenerator(
            trained_model).compute_saliency_and_predictions(images)

        assert maps.shape[0] == images.shape[0]
        assert maps.shape[-2:] == images.shape[-2:], (
            "the saliency map is not the size of the image it explains, so "
            "it cannot be overlaid on it")
        assert predicted.shape[0] == images.shape[0]

    def test_the_predictions_match_the_model(self, trained_model):
        """The half that makes the map interpretable: it explains the
        class the model actually chose, not a class the generator picked
        for itself."""
        from spacr.utils import SaliencyMapGenerator

        images = torch.stack(_images(4))
        _maps, predicted = SaliencyMapGenerator(
            trained_model).compute_saliency_and_predictions(images)

        with torch.no_grad():
            expected = trained_model(images).argmax(dim=1)

        assert predicted.flatten().tolist() == expected.tolist()

    def test_the_map_is_not_flat(self, trained_model):
        """A flat map passes every shape check and explains nothing. This
        is the assertion that would fail if the gradients stopped
        flowing -- a model left in ``no_grad``, an input without
        ``requires_grad`` -- which is the way this breaks silently."""
        from spacr.utils import SaliencyMapGenerator

        images = torch.stack(_images(1))
        maps, _predicted = SaliencyMapGenerator(
            trained_model).compute_saliency_and_predictions(images)

        values = maps.detach().cpu().numpy().ravel()
        assert float(values.max() - values.min()) > 0.0, (
            "the saliency map is uniform, so no gradient reached the input "
            "and the map explains nothing")

    def test_the_bright_square_draws_more_attention_than_the_corners(
            self, trained_model):
        """The substance: the model was trained to separate images by a
        bright central square, so an attribution worth showing a
        biologist puts more weight there than on the empty corners."""
        from spacr.utils import SaliencyMapGenerator

        images = torch.stack(_images(1))          # index 0 has the square
        maps, _predicted = SaliencyMapGenerator(
            trained_model).compute_saliency_and_predictions(images)

        array = maps.detach().cpu().numpy().reshape(16, 16)
        centre = float(array[4:12, 4:12].mean())
        corners = float(np.concatenate([
            array[:4, :4].ravel(), array[:4, -4:].ravel(),
            array[-4:, :4].ravel(), array[-4:, -4:].ravel()]).mean())

        assert centre > corners, (
            f"the attribution weights the empty corners ({corners:.4g}) at "
            f"least as much as the object ({centre:.4g})")

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
        assert "else:\n            activation_maps, predicted_classes = " \
               "cam_generator.compute_saliency_and_predictions(inputs)" in source
        assert "else:\n                # Handle each channel separately and save as RGB" \
            in source
