import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spacr.umap import generate_image_umap

class TestGenerateImageUmap(unittest.TestCase):
    def test_generate_image_umap(self):
        # Mock necessary dependencies
        db_paths = ['/path/to/db1.db', '/path/to/db2.db']
        tables = ['cell']
        visualize = 'cell'
        image_nr = 100
        dot_size = 50
        n_neighbors = 30
        min_dist = 0.1
        metric = 'cosine'
        eps = 0.5
        min_samples = 5
        filter_by = None
        img_zoom = 0.3
        plot_by_cluster = False
        plot_cluster_grids = False
        remove_cluster_noise = False
        figuresize = 20
        remove_highly_correlated = True
        log_data = False
        black_background = False
        remove_image_canvas = False
        plot_outlines = False
        plot_points = True
        smooth_lines = False
        row_limit = None
        verbose = False
        
        # Call the function
        fig = generate_image_umap(db_paths, tables, visualize, image_nr, dot_size, n_neighbors, min_dist, metric, eps, min_samples, filter_by, img_zoom, plot_by_cluster, plot_cluster_grids, remove_cluster_noise, figuresize, remove_highly_correlated, log_data, black_background, remove_image_canvas, plot_outlines, plot_points, smooth_lines, row_limit, verbose)

        self.assertIsInstance(fig, plt.Figure)

if __name__ == '__main__':
    unittest.main()