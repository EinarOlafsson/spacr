"""Optional deterministic multi-view phenotype classification."""
from __future__ import annotations

import math


DEFAULTS = {
    'tta_enabled': False,
    'tta_rotations': False,
    'tta_horizontal_flip': False,
    'tta_vertical_flip': False,
    'tta_aggregation': 'probability_mean',
    'tta_min_agreement': .75,
    'tta_max_std': .15,
}


def transforms_for(settings):
    """Return distinct right-angle rotations/reflections, with identity first.

    :param settings: mapping containing the optional tta_* switches.
    :returns: (quarter-turns, horizontal-reflection) pairs; at most eight.
        Both flips together also include their composed 180-degree rotation.
    """
    rotations = range(4) if settings.get('tta_rotations', False) else (0,)
    views = []
    for rotation in rotations:
        flips = [(rotation, False)]
        if settings.get('tta_horizontal_flip', False):
            flips.append((rotation, True))
        if settings.get('tta_vertical_flip', False):
            flips.append(((rotation + 2) % 4, True))
        if settings.get('tta_horizontal_flip', False) and settings.get('tta_vertical_flip', False):
            flips.append(((rotation + 2) % 4, False))
        for view in flips:
            if view not in views:
                views.append(view)
    return views


def predict_augmented(model, images, settings):
    """Average probabilities or vote across selected image orientations.

    :param model: evaluation-mode callable returning binary or multiclass logits.
    :param images: NCHW tensor; input pixels are never modified in place.
    :param settings: tta_* options and optional binary score_threshold.
    :returns: score tensor, selected class tensor and named diagnostic tensors.
        Binary scores remain positive-class probabilities; multiclass scores
        refer to the selected class. prediction_std is the population standard
        deviation of that same class's probability across views. Agreement is
        the fraction of view labels matching the selected label, not calibrated
        confidence. Majority ties prefer mean probability, then lowest index.
    :raises ValueError: invalid aggregation, thresholds or model output shape.
    """
    import torch

    aggregation = settings.get('tta_aggregation', DEFAULTS['tta_aggregation'])
    if aggregation not in ('probability_mean', 'majority_vote'):
        raise ValueError('tta_aggregation must be probability_mean or majority_vote')
    thresholds = {key: float(settings.get(key, default)) for key, default in (
        ('score_threshold', .5), ('tta_min_agreement', .75), ('tta_max_std', .15))}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError('TTA agreement, standard deviation and score thresholds must be between 0 and 1')
    probabilities = []
    with torch.inference_mode():
        for rotation, flip in transforms_for(settings):
            view = torch.rot90(images, rotation, dims=(-2, -1))
            if flip:
                view = view.flip(-1)
            logits = model(view)
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                positive = torch.sigmoid(logits.reshape(-1))
                probability = torch.stack((1 - positive, positive), dim=1)
            elif logits.ndim == 2 and logits.shape[1] >= 2:
                probability = logits.softmax(dim=1)
            else:
                raise ValueError('Classification model must return N, Nx1 or NxC logits')
            if probability.shape[0] != images.shape[0]:
                raise ValueError('Classification output batch does not match the input images')
            if not torch.isfinite(probability).all():
                raise ValueError('Classification produced non-finite probabilities')
            probabilities.append(probability)
    stack = torch.stack(probabilities)
    mean = stack.mean(dim=0)
    std = stack.std(dim=0, unbiased=False)
    binary = mean.shape[1] == 2
    threshold = thresholds['score_threshold']
    view_labels = (stack[..., 1] >= threshold).long() if binary else stack.argmax(dim=-1)
    predicted = (mean[:, 1] >= threshold).long() if binary else mean.argmax(dim=1)
    if aggregation == 'majority_vote':
        counts = torch.nn.functional.one_hot(view_labels, num_classes=mean.shape[1]).sum(dim=0)
        candidates = counts == counts.max(dim=1, keepdim=True).values
        predicted = mean.masked_fill(~candidates, -1).argmax(dim=1)
    selected = torch.ones_like(predicted) if binary else predicted
    scores = mean.gather(1, selected[:, None]).squeeze(1)
    dispersion = std.gather(1, selected[:, None]).squeeze(1)
    agreement = (view_labels == predicted[None, :]).float().mean(dim=0)
    original_label = view_labels[0]
    original_score = stack[0, :, 1] if binary else stack[0].max(dim=1).values
    extras = {
        'original_pred': original_score, 'original_predicted_label': original_label,
        'prediction_mean': scores, 'prediction_std': dispersion,
        'transform_agreement': agreement,
        'review_flag': ((agreement < thresholds['tta_min_agreement']) |
                        (dispersion > thresholds['tta_max_std'])),
        'tta_views': torch.full_like(predicted, len(probabilities)),
    }
    for index in range(mean.shape[1]):
        extras[f'prob_class_{index}'] = mean[:, index]
        extras[f'original_prob_class_{index}'] = stack[0, :, index]
        extras[f'prob_class_{index}_std'] = std[:, index]
    return scores, predicted, extras
