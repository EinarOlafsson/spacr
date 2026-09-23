"""CPU evidence for opt-in orientation scoring and both inference entry points."""
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from spacr.inference_augmentation import DEFAULTS, predict_augmented, transforms_for


class CornerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return (images[:, 0, 0, 0] * 8 - 4)[:, None]


@pytest.mark.parametrize('rotations,horizontal,vertical,count', [
    (False, False, False, 1), (True, False, False, 4),
    (False, True, False, 2), (False, False, True, 2),
    (False, True, True, 4), (True, True, True, 8)])
def test_unique_views_preserve_pixels(rotations, horizontal, vertical, count):
    views = transforms_for(dict(tta_rotations=rotations, tta_horizontal_flip=horizontal,
                               tta_vertical_flip=vertical))
    image = torch.arange(9).reshape(1, 1, 3, 3)
    actual = []
    for rotation, flip in views:
        result = torch.rot90(image, rotation, (-2, -1))
        result = result.flip(-1) if flip else result
        actual.append(tuple(result.flatten().tolist()))
    assert len(set(actual)) == count == len(views)
    assert views[0] == (0, False)
    assert all(sorted(view) == list(range(9)) for view in actual)


def test_disabled_is_one_forward_pass_and_unchanged_schema():
    from spacr.deep_spacr import _inference_predictions

    model = CornerModel().eval()
    images = torch.zeros(2, 1, 3, 3)
    scores, labels, extras = _inference_predictions(model, images, dict(tta_rotations=True))
    assert model.calls == 1
    assert extras == {}
    assert scores.tolist() == pytest.approx([torch.sigmoid(torch.tensor(-4.)).item()] * 2)
    assert labels.tolist() == [0, 0]


@pytest.mark.parametrize('aggregation,expected_label,agreement', [
    ('probability_mean', 1, .25), ('majority_vote', 0, .75)])
def test_mean_and_vote_differ_and_original_is_retained(aggregation, expected_label, agreement):
    class Sequence(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index = 0

        def forward(self, images):
            probability = [.99, .49, .49, .49][self.index]
            self.index += 1
            return torch.logit(torch.full((len(images), 1), probability))

    probabilities = torch.tensor([.99, .49, .49, .49])
    scores, labels, extra = predict_augmented(Sequence(), torch.zeros(2, 1, 3, 3),
                                             dict(tta_rotations=True, tta_aggregation=aggregation))
    assert scores.tolist() == pytest.approx([.615, .615])
    assert labels.tolist() == [expected_label] * 2
    assert extra['original_pred'].tolist() == pytest.approx([.99] * 2)
    assert extra['prediction_std'].tolist() == pytest.approx([probabilities.std(unbiased=False).item()] * 2)
    assert extra['transform_agreement'].tolist() == [agreement] * 2
    assert extra['review_flag'].tolist() == [True, True]


def test_multiclass_retains_all_probabilities_and_input():
    class ThreeClasses(torch.nn.Module):
        def forward(self, images):
            return torch.tensor([[1., 2., 3.]]).expand(len(images), -1)

    images = torch.arange(18.).reshape(2, 1, 3, 3)
    original = images.clone()
    scores, labels, extra = predict_augmented(ThreeClasses(), images, dict(tta_rotations=True))
    assert torch.equal(original, images)
    assert labels.tolist() == [2, 2]
    assert torch.equal(scores, extra['prob_class_2'])
    assert extra['prediction_std'].tolist() == [0., 0.]
    assert extra['review_flag'].tolist() == [False, False]
    assert sum(extra[f'prob_class_{i}'] for i in range(3)).tolist() == pytest.approx([1., 1.])


def test_folder_and_tar_use_same_augmentation_and_save_diagnostics(tmp_path, monkeypatch):
    from spacr import deep_spacr as deep

    images = tmp_path / 'images'
    images.mkdir()
    pixels = np.zeros((8, 8, 3), np.uint8)
    pixels[:3, :3] = 255
    path = images / 'plate1_A01_f1_o1.png'
    Image.fromarray(pixels).save(path)
    tar_path = tmp_path / 'crops.tar'
    with tarfile.open(tar_path, 'w') as archive:
        archive.add(path, arcname=path.name)
    monkeypatch.setattr(deep, '_load_inference_model', lambda *args: (CornerModel().eval(), {}))
    monkeypatch.setattr(deep, 'pick_device', lambda **kwargs: (torch.device('cpu'), ''))
    model_path = str(tmp_path / 'model.pth')
    options = dict(tta_enabled=True, tta_rotations=True, tta_aggregation='majority_vote')
    folder = deep.apply_model(str(images), model_path, image_size=8, batch_size=1,
                               n_jobs=0, normalize=False, **options)
    archive = deep.apply_model_to_tar(dict(tar_path=str(tar_path), model_path=model_path,
                                          image_size=8, batch_size=1, n_jobs=0,
                                          normalize=False, verbose=False,
                                          score_threshold=.5, **options))
    for column in ('pred', 'original_pred', 'prediction_mean', 'prediction_std',
                   'transform_agreement', 'review_flag', 'tta_views', 'predicted_label'):
        assert archive[column].tolist() == folder[column].tolist()
    assert archive['cv_predictions'].tolist() == archive['predicted_label'].tolist()
    assert archive['tta_views'].tolist() == [4]
    assert any('transform_agreement' in file.read_text() for file in tmp_path.rglob('*.csv'))


def test_defaults_types_and_classify_category_are_complete():
    from spacr import settings
    from spacr.qt.screens.settings_model import categories_for_app

    defaults = settings.deep_spacr_defaults({})
    assert all(defaults[key] == value for key, value in DEFAULTS.items())
    assert all(defaults[key] is False for key in DEFAULTS if settings.expected_types[key] is bool)
    for key in DEFAULTS:
        assert key in settings.tooltips
    for app in ('classify', 'classify_merged'):
        categories = categories_for_app(app, settings.categories)
        matches = [keys for title, keys in categories.items() if 'Test-time augmentation' in title]
        assert len(matches) == 1 and set(matches[0]) == set(DEFAULTS)
