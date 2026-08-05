Resumable multi-objective UMAP search
=====================================

Image UMAP can search ``n_neighbors`` and ``min_dist`` using one criterion or
a multi-objective mode. Open **UMAP settings…** from the Hyperparameter Search
panel and choose ``multi_objective`` as the criterion.

Why several objectives?
-----------------------

No score proves that a UMAP contains biologically meaningful structure. The
multi-objective mode therefore retains three separate measurements:

``neighborhood_preservation``
   The geometric mean of trustworthiness and continuity. Trustworthiness
   penalizes neighbours invented by the two-dimensional embedding; continuity
   penalizes true feature-space neighbours that the embedding tears apart.
   Requiring both prevents either failure from being hidden.

``stability``
   The mean fraction of nearest neighbours shared by embeddings fitted with
   different reproducible seeds. It is invariant to rotations and reflections.
   A visually striking feature that disappears between repeats is not stable
   structure.

``cluster_structure``
   Positive silhouette structure on a scale from zero to one. When labels are
   supplied through the Python API they define the partition. Otherwise spaCR
   fits reproducible K-means partitions from 2 through 8 clusters and reports
   the strongest silhouette together with the selected cluster counts. This is
   evidence of geometric structure, not proof that a cluster is biological.

The three weights are normalized to sum to one. Their weighted geometric mean
is called ``multi_objective`` and guides the grid or adaptive search. The
geometric mean penalizes a collapsed objective more strongly than an arithmetic
average.

Pareto front
------------

The composite score is not presented as the only answer. spaCR also marks the
Pareto front in the result table. A configuration is Pareto-optimal when no
other tested configuration improves one objective without making another
worse. Inspect the embedding panels and objective tooltips for every
non-dominated row before propagating a configuration.

The neighborhood, stability and cluster objectives, normalized weights,
repeat count, raw silhouette, cluster source and discovered cluster counts are
stored in each trial's ``extra_metrics``.

Repeated fits and runtime
-------------------------

``stability repeats`` controls how many seeded embeddings are fitted per
configuration. The minimum is 2 and the default is 3. Runtime scales roughly
as:

``configurations × stability repeats × one UMAP fit``

Use three repeats for routine searches and increase the count when the leading
Pareto configurations have similar stability or when a final analysis must be
especially reproducible.

Adaptive search and stopping
----------------------------

Adaptive 2×2 mode evaluates the four diagonal corners around the current
``n_neighbors``/``min_dist`` centre. The weighted multi-objective score chooses
the direction of the next move. Search stops at the maximum round count, when a
round fails to exceed ``minimum improvement``, or after **Stop** is requested.
A stopped result is explicitly marked partial.

Resume behavior
---------------

Enable **Resume checkpoint** to continue from
``results/.spacr_checkpoints/umap_search.json`` under the current project (or
from an explicit ``checkpoint_path`` in the Python API). Each completed trial
is written atomically and its primary embedding is stored beside the JSON.
An interrupted adaptive round evaluates only missing corners before choosing a
direction.

Resume refuses to combine incompatible work. Feature and label hashes, search
space, criterion, seed, neighborhood size, adaptive increments, stopping
threshold, stability repeat count, objective weights and embedder identity must
match the checkpoint.

Python API
----------

Use :func:`spacr.hyperparam.umap_search` for a full search,
:func:`spacr.hyperparam.umap_objective_scores` to score repeated embeddings,
:func:`spacr.hyperparam.embedding_stability` for stability alone, and
:meth:`spacr.hyperparam.SearchResult.pareto_front` to retrieve non-dominated
trials.

.. code-block:: python

   from spacr.hyperparam import SearchSpace, umap_search

   result = umap_search(
       features,
       SearchSpace({"n_neighbors": [5, 15, 50], "min_dist": [0.05, 0.2]}),
       metric="multi_objective",
       stability_repeats=3,
       objective_weights={
           "neighborhood_preservation": 0.4,
           "stability": 0.3,
           "cluster_structure": 0.3,
       },
       checkpoint_path="results/.spacr_checkpoints/umap_search.json",
       resume=True,
   )
   for trial in result.pareto_front():
       print(trial.params, trial.extra_metrics)
