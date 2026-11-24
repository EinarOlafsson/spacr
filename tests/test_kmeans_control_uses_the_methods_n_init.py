"""The shuffled k-means control is fitted exactly like the real k-means."""
import inspect
import re

from spacr import ops_clusters


def test_real_and_shuffled_kmeans_share_n_init():
    source = inspect.getsource(ops_clusters.benchmark_clusters)
    values = re.findall(r"KMeans\([^)]*n_init=(\d+)", source)
    assert len(values) == 2 and len(set(values)) == 1
