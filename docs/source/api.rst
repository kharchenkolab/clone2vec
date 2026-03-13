.. currentmodule:: sl

API
===

Datasets
--------

.. autosummary::
   :toctree: API
   :nosignatures:

   datasets.Weinreb_in_vitro
   datasets.Erickson_murine_development

Preprocessing
-------------

.. autosummary::
   :toctree: API
   :nosignatures:

   pp.clones_adata
   pp.make_unique_clones
   pp.recalc_composition
   pp.transfer_annotation
   pp.transfer_expression

Tools
-----

.. autosummary::
   :toctree: API
   :nosignatures:

   tl.clonal_nn
   tl.clonocluster
   tl.smooth
   tl.group_connectivity

Embeddings
----------

.. autosummary::
   :toctree: API
   :nosignatures:

   tl.clone2vec
   tl.clone2vec_Poi
   tl.project_clone2vec
   tl.project_clone2vec_Poi
   tl.find_mnn
   tl.align

Associations
------------

.. autosummary::
   :toctree: API
   :nosignatures:

   tl.associations
   tl.graph_associations
   tl.eigenvalue_test
   tl.catboost

Utils
-----

.. autosummary::
   :toctree: API
   :nosignatures:

   utils.stack_layers
   utils.correct_shap
   utils.connect_clones
   utils.get_connectivity_matrix
   utils.regress_categories
   utils.impute
   utils.gs

Plotting
--------

.. autosummary::
   :toctree: API
   :nosignatures:

   pl.group_scatter
   pl.group_kde
   pl.loss_history
   pl.clone_size
   pl.nesting_clones
   pl.volcano
   pl.shap_volcano
   pl.barplot
   pl.heatmap
   pl.predictors_comparison
   pl.catboost_perfomance
   pl.clones2cells
   pl.graph
   pl.group_connectivity
   pl.pca_loadings
   pl.scaled_dotplot
   pl.scatter2vars
   pl.embedding_axis
   pl.small_cbar
   pl.fancy_legend

Seurat
------

.. autosummary::
   :toctree: API
   :nosignatures:

   seurat.read
   seurat.integrate_data