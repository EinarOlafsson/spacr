"""The nine CellProfiler Analyst 3.0.5 classifiers, built as CellProfiler Analyst builds them.

CellProfiler Analyst (CPA; github.com/CellProfiler/CellProfiler-Analyst,
commit f5e1b5e2b2c35bb1aff4516ae5a49fe420df0350, version 3.0.5, BSD
3-Clause licence, Copyright 2003-2024 Broad Institute) offers nine
classifiers in its Classifier window. Eight are scikit-learn estimators that
CPA's ``GeneralClassifier`` constructs from a fixed string, with an optional
:class:`~sklearn.preprocessing.StandardScaler` fitted on the training data;
:data:`CPA_GENERAL` holds the strings and scaler flags verbatim from
``cpa/classifier.py``. The ninth, FastGentleBoosting, is CPA's own boosted
stumps (Torralba et al. 2007, "Sharing visual features for multiclass and
multiview object detection", IEEE PAMI), ported here from
``cpa/fastgentleboosting.py`` with the same arithmetic in ``float32``, the
same unstable ``argsort`` and the same tie rule, and scored as CPA scores
cells (sum over rules of ``a`` if ``x > threshold`` else ``b``, argmax).

:func:`make_cpa_model` returns an estimator with ``fit``, ``predict`` and
``predict_proba``; :func:`spacr.moa_benchmark.loco_supervised` fits it as
``cpa_<name>``, such as ``cpa_RandomForestClassifier``. A classifier that
takes a ``random_state`` gets the fold's seed, as the BBBC021 reanalysis
(``cpa_models.py``) set it. FastGentleBoosting has no probabilities: its
additive scores go through a softmax so that averaging over a treatment's
wells is defined.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: The CPA release and commit the classifiers are taken from.
CPA_VERSION = '3.0.5'
CPA_COMMIT = 'f5e1b5e2b2c35bb1aff4516ae5a49fe420df0350'

#: Classifier name -> (CPA's constructor string, StandardScaler on), from ``cpa/classifier.py``.
CPA_GENERAL: Dict[str, Tuple[str, bool]] = {
    'RandomForestClassifier': ('ensemble.RandomForestClassifier(n_estimators=100)', False),
    'AdaBoostClassifier': ('ensemble.AdaBoostClassifier()', False),
    'SVC': ('svm.SVC(probability=True)', True),
    'GradientBoostingClassifier': ('ensemble.GradientBoostingClassifier()', False),
    'LogisticRegression': ('linear_model.LogisticRegression()', False),
    'LDA': ('discriminant_analysis.LinearDiscriminantAnalysis()', False),
    'KNeighborsClassifier': ('neighbors.KNeighborsClassifier()', True),
    'NeuralNetwork': ("neural_network.MLPClassifier(hidden_layer_sizes=(12,12), "
                      "solver='lbfgs', max_iter=500)", True),
}

#: Every CPA classifier name, FastGentleBoosting first as in CPA's menu.
CPA_NAMES: Tuple[str, ...] = ('FastGentleBoosting',) + tuple(CPA_GENERAL)

#: FastGentleBoosting's rule count: CPA's "max rules" box default.
FGB_RULES = 5


def _estimator(name: str) -> Any:
    """A fresh scikit-learn estimator for one CPA name, with CPA's parameters."""
    from sklearn import discriminant_analysis, ensemble, linear_model, neighbors, neural_network, svm

    if name == 'RandomForestClassifier':
        return ensemble.RandomForestClassifier(n_estimators=100)
    if name == 'AdaBoostClassifier':
        return ensemble.AdaBoostClassifier()
    if name == 'SVC':
        return svm.SVC(probability=True)
    if name == 'GradientBoostingClassifier':
        return ensemble.GradientBoostingClassifier()
    if name == 'LogisticRegression':
        return linear_model.LogisticRegression()
    if name == 'LDA':
        return discriminant_analysis.LinearDiscriminantAnalysis()
    if name == 'KNeighborsClassifier':
        return neighbors.KNeighborsClassifier()
    return neural_network.MLPClassifier(hidden_layer_sizes=(12, 12), solver='lbfgs', max_iter=500)


def _tilesum(array: np.ndarray, n: int) -> np.ndarray:
    """The column sums of ``array`` repeated on ``n`` rows."""
    return np.tile(np.sum(array, axis=0), (n, 1))


def _weak_learner(labels: np.ndarray, weights: np.ndarray,
                  values: np.ndarray) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    """The best stump on one feature: CPA's ``TrainWeakLearner``.

    :param labels: examples by classes, +1 or -1.
    :param weights: examples by classes.
    :param values: the feature, one value per example.
    :returns: ``(threshold, error, a, b)``; ``a`` is added where
        ``value > threshold`` and ``b`` elsewhere.
    """
    order = np.argsort(values)
    s_values = values[order]
    s_labels = labels[order, :]
    s_weights = weights[order, :]
    n = labels.shape[0]
    weighted = s_weights * s_labels
    num_a = _tilesum(weighted, n) - np.cumsum(weighted, axis=0)
    den_a = _tilesum(s_weights, n) - np.cumsum(s_weights, axis=0)
    den_a[den_a <= 0.0] = 1.0
    a = num_a / den_a
    b = np.cumsum(weighted, axis=0) / np.cumsum(s_weights, axis=0)
    negative = s_labels < 0
    positive = s_labels > 0
    below_neg = np.cumsum(s_weights * negative, axis=0)
    below_pos = np.cumsum(s_weights * positive, axis=0)
    above_neg = _tilesum(s_weights * negative, n) - below_neg
    above_pos = _tilesum(s_weights * positive, n) - below_pos
    error = (below_neg * ((-1 - b) ** 2) + below_pos * ((1 - b) ** 2)
             + above_neg * ((-1 - a) ** 2) + above_pos * ((1 - a) ** 2)).sum(axis=1)
    index = int(np.argmin(error))
    while index + 1 < len(s_values) and s_values[index] == s_values[index + 1]:
        index += 1
    return s_values[index], error[index], a[index, :].copy(), b[index, :].copy()


def fast_gentle_boosting(label_matrix: np.ndarray, values: np.ndarray,
                         rules: int = FGB_RULES) -> List[Tuple[int, Any, np.ndarray, np.ndarray, float]]:
    """Train CPA's FastGentleBoosting: ``rules`` boosted stumps over every class at once.

    The port of CPA 3.0.5's ``FastGentleBoosting.Train``: class-balanced
    weights, one stump per round chosen over every feature by
    :func:`_weak_learner`, weights renewed as
    ``balancing * exp(-score * label)`` normalised per class, and an early
    stop when a stump makes no error.

    :param label_matrix: examples by classes, +1 for the example's class and
        -1 elsewhere, ``float32``.
    :param values: examples by features, ``float32``.
    :param rules: the number of stumps.
    :returns: one ``(feature, threshold, a, b, expected_worst_margin)`` per
        stump; an empty list when there is nothing to train on.
    """
    if 0 in values.shape:
        return []
    computed = np.zeros(label_matrix.shape, np.float32)
    n, k = label_matrix.shape
    weights = np.ones(label_matrix.shape, np.float32)
    margin_correct = np.zeros((n, k - 1), np.float32)
    margin_incorrect = np.zeros((n, k - 1), np.float32)
    for column in range(k):
        mask = (label_matrix[:, column] == 1).reshape((n, 1))
        weights[np.tile(mask, (1, k))] /= sum(mask)
    balancing = weights.copy()
    model = []
    for _ in range(int(rules)):
        best = float(np.inf)
        chosen = None
        for feature in range(values.shape[1]):
            threshold, error, a, b = _weak_learner(label_matrix, weights, values[:, feature])
            if error < best:
                best = error
                chosen = (error, feature, threshold, a, b)
        error, feature, threshold, a, b = chosen
        above = np.tile(np.reshape(values[:, feature] > threshold, (n, 1)), (1, k))
        adjustment = above * np.tile(a, (n, 1)) + (1 - above) * np.tile(b, (n, 1))
        recomputed = computed + adjustment
        reweights = balancing * np.exp(-recomputed * label_matrix)
        reweights = reweights / sum(reweights)
        step_correct = adjustment[label_matrix > 0].reshape((n, 1))
        relative = step_correct - adjustment[label_matrix < 0].reshape((n, k - 1))
        gain = relative > 0
        margin_correct += relative * gain
        margin_incorrect += (-relative) * (~gain)
        with np.errstate(invalid='ignore', divide='ignore'):
            share = margin_correct / (margin_correct + margin_incorrect)
        worst = sum(balancing[:, 0] * share.min(axis=1)) / sum(balancing[:, 0])
        computed = recomputed
        model.append((int(feature), threshold, a, b, worst))
        if error == 0.0:
            break
        weights = reweights
    return model


class CPAClassifier:
    """One CellProfiler Analyst 3.0.5 classifier with a scikit-learn interface.

    :param name: one of :data:`CPA_NAMES`.
    :param seed: ``random_state`` of an estimator that takes one.
    :param rules: FastGentleBoosting's stump count.
    :raises ValueError: for a name CPA does not offer.
    """

    def __init__(self, name: str, seed: int = 0, rules: int = FGB_RULES) -> None:
        """Choose the classifier; nothing is fitted yet."""
        if name not in CPA_NAMES:
            raise ValueError(f'CPA offers {CPA_NAMES}, not {name!r}')
        self.name, self.seed, self.rules = name, int(seed), int(rules)
        self.model: List[Tuple[int, Any, np.ndarray, np.ndarray, float]] = []
        self.estimator: Any = None
        self.scaler: Any = None

    def fit(self, X: Any, y: Any) -> 'CPAClassifier':
        """Train on ``X`` (examples by features) and labels ``y``.

        :param X: the training values; used as ``float32``, as CPA does.
        :param y: one label per example.
        :returns: this classifier.
        """
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.name == 'FastGentleBoosting':
            labels = -np.ones((len(y), len(self.classes_)), np.float32)
            labels[np.arange(len(y)), np.searchsorted(self.classes_, y)] = 1
            self.model = fast_gentle_boosting(labels, X, self.rules)
            return self
        from sklearn.preprocessing import StandardScaler

        self.estimator = _estimator(self.name)
        if 'random_state' in self.estimator.get_params():
            self.estimator.set_params(random_state=self.seed)
        self.scaler = StandardScaler() if CPA_GENERAL[self.name][1] else None
        values = self.scaler.fit_transform(X) if self.scaler is not None else X
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            self.estimator.fit(values, y)
        return self

    def _scaled(self, X: Any) -> np.ndarray:
        """``X`` as ``float32``, through the training scaler when there is one."""
        X = np.asarray(X, np.float32)
        return self.scaler.transform(X) if self.scaler is not None else X

    def scores(self, X: Any) -> np.ndarray:
        """FastGentleBoosting's additive class scores, as CPA scores cells.

        :param X: examples by features.
        :returns: examples by classes.
        """
        X = np.asarray(X, np.float32)
        total = np.zeros((len(X), len(self.classes_)), np.float64)
        for feature, threshold, a, b, _ in self.model:
            total += np.where((X[:, int(feature)] > threshold)[:, None], a[None, :], b[None, :])
        return total

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities; for FastGentleBoosting a softmax of :meth:`scores`.

        :param X: examples by features.
        :returns: examples by classes, in the order of ``classes_``.
        """
        if self.name == 'FastGentleBoosting':
            scores = self.scores(X)
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        return self.estimator.predict_proba(self._scaled(X))

    def predict(self, X: Any) -> np.ndarray:
        """The predicted class of each example.

        :param X: examples by features.
        :returns: one label per example.
        """
        if self.name == 'FastGentleBoosting':
            return self.classes_[self.scores(X).argmax(axis=1)]
        return np.asarray(self.estimator.predict(self._scaled(X)))


def make_cpa_model(name: str, seed: int = 0, rules: Optional[int] = None) -> CPAClassifier:
    """A fresh CPA classifier for one fold.

    :param name: one of :data:`CPA_NAMES`, with or without a ``cpa_`` prefix.
    :param seed: ``random_state`` of an estimator that takes one.
    :param rules: FastGentleBoosting's stump count; ``None`` takes
        :data:`FGB_RULES`.
    :returns: an unfitted :class:`CPAClassifier`.
    :raises ValueError: for a name CPA does not offer.
    """
    name = str(name)
    if name.startswith('cpa_'):
        name = name[len('cpa_'):]
    return CPAClassifier(name, seed=seed, rules=FGB_RULES if rules is None else rules)
