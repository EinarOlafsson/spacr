"""Opt-in Metal reducers with sklearn-shaped results and bounded working memory.

Cluster assignment, centroid updates and well-conditioned PCA projections run
on MPS. PCA fitting and numerically sensitive projections use CPU float64:
forming a float32 covariance matrix can erase valid scientific components.
"""

from __future__ import annotations

import numpy as np


def _matrix(data):
    """Validate finite two-dimensional data with at least one row and return a contiguous float32 Metal tensor."""
    import torch

    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite two-dimensional matrix")
    return torch.as_tensor(np.ascontiguousarray(array), device="mps")


def _centered(data, mean):
    """Validate the fitted feature count and return float64 mean-centered values."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(mean):
        raise ValueError("Feature count differs from fitted reducer")
    return array - mean


class MetalPCA:
    """Stable CPU PCA fitting with Metal projection where precision permits."""

    def __init__(
        self, n_components=None, *, whiten=False, svd_solver="auto", random_state=None, copy=True
    ):
        """Store PCA options, refusing solvers outside auto, full and covariance_eigh. The copy argument is accepted for compatibility and is not used."""
        if svd_solver not in ("auto", "full", "covariance_eigh"):
            raise ValueError("This PCA solver remains on the CPU")
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, data, y=None):
        """Fit PCA on CPU float64 data and select the projection backend.

        :param data: sample-by-feature matrix used to fit PCA.
        :param y: ignored compatibility argument.
        :returns: this fitted reducer."""
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
        """Project samples using the fitted CPU or Metal backend.

        :param data: sample-by-feature matrix with the fitted feature count.
        :returns: NumPy array of projected samples, whitened when configured."""
        import torch

        if self.projection_backend_ == "cpu":
            return self._reference.transform(np.asarray(data, dtype=np.float64))
        centered = _centered(data, self.mean_)
        x = _matrix(centered)
        out = x @ torch.as_tensor(self.components_.T.copy(), dtype=torch.float32, device="mps")
        if self.whiten:
            scale = np.sqrt(self.explained_variance_).clip(min=np.finfo(np.float64).eps)
            out /= torch.as_tensor(scale.astype(np.float32), device="mps")
        return out.cpu().numpy()

    def fit_transform(self, data, y=None):
        """Fit PCA and project the same samples.

        :param data: sample-by-feature matrix used for fitting and projection.
        :param y: ignored compatibility argument.
        :returns: NumPy array of projected samples, whitened when configured."""
        return self.fit(data, y).transform(data)

    def inverse_transform(self, data):
        """Reconstruct feature values from fitted PCA coordinates.

        :param data: sample-by-component matrix of projected coordinates.
        :returns: NumPy array in the original feature space, undoing whitening when configured."""
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
        """Store Lloyd KMeans options, accepting random or k-means++ initialization. The copy_x and verbose arguments are accepted for compatibility and are not used."""
        if algorithm != "lloyd" or not isinstance(init, str) or init not in ("k-means++", "random"):
            raise ValueError("This KMeans initialization/algorithm remains on the CPU")
        self.n_clusters = int(n_clusters)
        self.init, self.n_init = init, n_init
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.random_state = random_state

    #: Elements of the ``rows x clusters x features`` difference tensor one
    #: assignment chunk may hold (64 MiB of float32).
    _CHUNK_ELEMENTS = 1 << 24

    @staticmethod
    def _assign(x, centers):
        """Return nearest-center labels and squared distances in memory-bounded chunks.

        One broadcast kernel per chunk, not one per center per 1024 rows: on an
        M2 the per-center loop launched ~500 kernels an iteration and made a
        20000 x 50, k=8 fit 36x slower than sklearn's. The arithmetic is the
        same per-element difference, so distances and first-index ties match.
        """
        import torch

        rows = max(1, MetalKMeans._CHUNK_ELEMENTS // max(1, centers.numel()))
        labels, distances = [], []
        for batch in x.split(rows):
            squared = (batch[:, None, :] - centers[None, :, :]).square().sum(2)
            distance, label = squared.min(dim=1)
            labels.append(label)
            distances.append(distance)
        return torch.cat(labels), torch.cat(distances)

    def _update(self, x, labels, rng):
        """Return the member mean of every cluster, reseeding empty ones from ``rng``.

        Sums come from a one-hot matmul rather than ``index_add_``, whose
        atomic adds on Metal make the result order-dependent, and the counts
        cross to the CPU in one transfer instead of one per cluster.
        """
        import torch

        sums = torch.zeros(self.n_clusters, x.shape[1], dtype=x.dtype, device=x.device)
        counts = torch.zeros(self.n_clusters, dtype=x.dtype, device=x.device)
        rows = max(1, self._CHUNK_ELEMENTS // self.n_clusters)
        for batch, members in zip(x.split(rows), labels.split(rows)):
            hot = torch.nn.functional.one_hot(members, self.n_clusters).to(x.dtype)
            sums += hot.T @ batch
            counts += hot.sum(0)
        updated = sums / counts.clamp(min=1)[:, None]
        for k in np.flatnonzero(counts.cpu().numpy() == 0):
            updated[k] = x[int(rng.integers(len(x)))]
        return updated

    def fit(self, data, y=None):
        """Fit cluster centers with chunked Metal assignment and updates.

        :param data: nonempty finite sample-by-feature matrix.
        :param y: ignored compatibility argument.
        :returns: this fitted clusterer."""
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
                new = self._update(x, labels, rng)
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
        """Assign samples to the nearest fitted cluster centers on Metal.

        :param data: sample-by-feature matrix with the fitted feature count.
        :returns: NumPy array of cluster indices."""
        import torch

        x = _matrix(_centered(data, self._offset))
        return (
            self._assign(x, torch.as_tensor(self._centered_centers, device="mps"))[0].cpu().numpy()
        )

    def fit_predict(self, data, y=None):
        """Fit cluster centers and return the training sample assignments.

        :param data: nonempty finite sample-by-feature matrix.
        :param y: ignored compatibility argument.
        :returns: NumPy array of training sample cluster indices."""
        return self.fit(data, y).labels_

    def fit_transform(self, data, y=None):
        """Fit cluster centers and compute distances to every center on Metal.

        :param data: nonempty finite sample-by-feature matrix.
        :param y: ignored compatibility argument.
        :returns: sample-by-cluster NumPy array of Euclidean distances."""
        import torch

        self.fit(data, y)
        x = _matrix(np.asarray(data, dtype=np.float64) - self._offset)
        centers = torch.as_tensor(self._centered_centers, device="mps")
        rows = max(1, self._CHUNK_ELEMENTS // max(1, centers.numel()))
        return (
            torch.cat(
                [
                    (batch[:, None, :] - centers[None, :, :]).square().sum(2).sqrt()
                    for batch in x.split(rows)
                ]
            )
            .cpu()
            .numpy()
        )
