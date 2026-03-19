clone2vec — Python package for single cell lineage tracing analysis
================================================================

**clone2vec** is a Python package for analysis of lineage tracing coupled with single-cell RNA-Seq.

The main key of the package are *clonal embeddings* — vector representations of the whole clones in low dimensional space (*clone2vec*). These
representations is a dropout-robust and cluster-free way of representation of heterogeneity within clonal behaviour for cell type tree-free
hypothesis generation regarding cells' multipotency.

*clone2vec* builds representation of clones in exact same way with popular word embedding algorithm — *word2vec* — via construction two-layers
fully connected neural network (specifically it uses Skip-Gram architecture) that aims to predict neighbour cells clonal labellings by clonal label
of cells. As a result, clones that exist in similar context in gene expression space will have similar weights in this neural network, and these
weights will be used as embedding for further analysis.

Source code for the package could be found on `GitHub <https://github.com/kharchenkolab/clone2vec>`__.

Installation
------------

clone2vec package might be installed via pip:

.. code-block:: console

   pip install clone2vec

If you want to install the package with all optional dependencies use:

.. code-block:: console

   pip install clone2vec[full]

The latest development version can be installed from GitHub using:

.. code-block:: console

   pip install git+https://github.com/kharchenkolab/clone2vec

.. toctree::
   :caption: Main
   :maxdepth: 2
   :hidden:

   basics
   tutorial
   api

.. toctree::
   :caption: Examples
   :maxdepth: 0
   :hidden:

   Clonal_embeddings
   Fast_clonal_embeddings
   Spatial_classification_mRNA
   