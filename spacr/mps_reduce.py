"""Opt-in Metal reducers with sklearn-shaped results and bounded working memory.

Cluster assignment, centroid updates and well-conditioned PCA projections run
on MPS. PCA fitting and numerically sensitive projections use CPU float64:
forming a float32 covariance matrix can erase valid scientific components.
"""

from __future__ import annotations

import numpy as np


def _matrix(data):
    import torch

    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite two-dimensional matrix")
    return torch.as_tensor(np.ascontiguousarray(array), device="mps")


def _centered(data, mean):
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(mean):
        raise ValueError("Feature count differs from fitted reducer")
    return array - mean


class MetalPCA:
    """Stable CPU PCA fitting with Metal projection where precision permits."""

    def __init__(
        self, n_components=None, *, whiten=False, svd_solver="auto", random_state=None, copy=True
    ):
        if svd_solver not in ("auto", "full", "covariance_eigh"):
            raise ValueError("This PCA solver remains on the CPU")
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, data, y=None):
        from sklearn.decomposition import PCA

        self._reference = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        self._reference.fit(np.asarray(data, dtype=np.float64))
        for name in (
            "n_features_in_",
            "mean_",
            "components_",
            "explained_variance_",
            "explained_variance_ratio_",
            "singular_values_",
            "n_components_",
            "n_samples_",
        ):
            setattr(self, name, getattr(self._reference, name))
        self.decomposition_backend_ = "cpu"
        values = self.explained_variance_
        self.projection_backend_ = (
            "cpu" if len(values) and values[-1] <= values[0] * 1e-6 else "mps"
        )
        return self

    def transform(self, data):
        import torch

        if self.projection_backend_ == "cpu":
            return self._reference.transform(np.asarray(data, dtype=np.float64))
        # Subtract offsets before the float32 transfer, so large absolute
        # intensities do not erase small, meaningful differences.
        centered = _centered(data, self.mean_)
        x = _matrix(centered)
        out = x @ torch.as_tensor(self.components_.T.copy(), dtype=torch.float32, device="mps")
        if self.whiten:
            # Match the float64 fitted reference. A float32 epsilon floor
            # suppresses valid whitening when all feature values are small.
            scale = np.sqrt(self.explained_variance_).clip(min=np.finfo(np.float64).eps)
            out /= torch.as_tensor(scale.astype(np.float32), device="mps")
        return out.cpu().numpy()

    def fit_transform(self, data, y=None):
        return self.fit(data, y).transform(data)

    def inverse_transform(self, data):
        values = np.asarray(data)
        if self.whiten:
            values = values * np.sqrt(self.explained_variance_)
        return values @ self.components_ + self.mean_


class MetalKMeans:
    """Lloyd KMeans with chunked Metal assignment and k-means++ initialization."""

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        algorithm="lloyd",
        copy_x=True,
        verbose=0,
    ):
        if algorithm != "lloyd" or not isinstance(init, str) or init not in ("k-means++", "random"):
            raise ValueError("This KMeans initialization/algorithm remains on the CPU")
        self.n_clusters = int(n_clusters)
        self.init, self.n_init = init, n_init
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.random_state = random_state

    @staticmethod
    def _assign(x, centers):
        import torch

        labels, distances = [], []
        for batch in x.split(1024):
            # Direct differences avoid catastrophic cancellation in
            # ||x||² + ||c||² - 2*x*c, even for translated feature tables.
            squared = torch.stack([(batch - center).square().sum(1) for center in centers], dim=1)
            distance, label = squared.min(dim=1)
            labels.append(label)
            distances.append(distance)
        return torch.cat(labels), torch.cat(distances)

    def fit(self, data, y=None):
        import torch

        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
            raise ValueError("Expected a nonempty finite two-dimensional matrix")
        self._offset = array.mean(axis=0)
        x = _matrix(array - self._offset)
        n, self.n_features_in_ = x.shape
        if not 1 <= self.n_clusters <= n or self.max_iter < 1:
            raise ValueError("Invalid cluster or iteration count")
        rng = np.random.default_rng(self.random_state)
        runs = (
            (1 if self.init == "k-means++" else 10) if self.n_init == "auto" else int(self.n_init)
        )
        if runs < 1:
            raise ValueError("n_init must be positive")
        best = None
        for _ in range(runs):
            selected = [int(rng.integers(n))]
            if self.init == "random":
                selected = rng.choice(n, self.n_clusters, replace=False).tolist()
            while len(selected) < self.n_clusters:
                _, distances = self._assign(x, x[selected])
                probabilities = distances.cpu().numpy().astype(float)
                total = probabilities.sum()
                selected.append(
                    int(rng.choice(n, p=probabilities / total))
                    if total > 0
                    else int(rng.integers(n))
                )
            centers = x[selected].clone()
            for _iteration in range(self.max_iter):
                labels, _ = self._assign(x, centers)
                updated = []
                for k in range(self.n_clusters):
                    members = x[labels == k]
                    updated.append(members.mean(0) if len(members) else x[int(rng.integers(n))])
                new = torch.stack(updated)
                shift = (new - centers).square().sum().item()
                centers = new
                if shift <= self.tol * self.tol:
                    break
            labels, distances = self._assign(x, centers)
            inertia = float(distances.sum().item())
            if best is None or inertia < best[0]:
                best = inertia, centers.cpu().numpy(), labels.cpu().numpy(), _iteration + 1
        self.inertia_, self._centered_centers, self.labels_, self.n_iter_ = best
        self.cluster_centers_ = self._centered_centers.astype(np.float64) + self._offset
        return self

    def predict(self, data):
        import torch

        x = _matrix(_centered(data, self._offset))
        return (
            self._assign(x, torch.as_tensor(self._centered_centers, device="mps"))[0].cpu().numpy()
        )

    def fit_predict(self, data, y=None):
        return self.fit(data, y).labels_

    def fit_transform(self, data, y=None):
        import torch

        self.fit(data, y)
        x = _matrix(np.asarray(data, dtype=np.float64) - self._offset)
        centers = torch.as_tensor(self._centered_centers, device="mps")
        return (
            torch.cat(
                [
                    torch.stack(
                        [(batch - center).square().sum(1).sqrt() for center in centers], dim=1
                    )
                    for batch in x.split(1024)
                ]
            )
            .cpu()
            .numpy()
        )
