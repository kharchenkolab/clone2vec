scLiTr — Python package for single cell lineage tracing analysis
================================================================

**scLiTr** is a Python package for analysis of lineage tracing coupled with single-cell RNA-Seq.

The main key of the package are *clonal embeddings* — vector representations of the whole clones in low dimensional space (*clone2vec*). These
representations is a dropout-robust and cluster-free way of representation of heterogeneity within clonal behaviour for cell type tree-free
hypothesis generation regarding cells' multipotency.

*clone2vec* builds representation of clones in exact same way with popular word embedding algorithm — *word2vec* — via construction two-layers
fully connected neural network (specifically it uses Skip-Gram architecture) that aims to predict neighbour cells clonal labellings by clonal label
of cells. As a result, clones that exist in similar context in gene expression space will have similar weights in this neural network, and these
weights will be used as embedding for further analysis.

Source code for the package could be found on `GitHub <https://github.com/kharchenkolab/scLiTr>`__.

Installation
------------

scLiTr package might be installed via pip:

.. code-block:: console

   pip install sclitr

or the latest development version can be installed from GitHub using:

.. code-block:: console

   pip install git+https://github.com/kharchenkolab/scLiTr

clones2cells
------------
For interactive exploration of clonal and gene expression embeddings together we recommend using our simple tool `clones2cells <https://github.com/serjisa/clones2cells_app>`__. You can install all necessary dependencies via pip:

.. code-block:: console

   pip install streamlit plotly streamlit_plotly_events pandas

and after launch the tool from the command line:

.. code-block:: console

   streamlit run https://raw.githubusercontent.com/serjisa/clones2cells_app/main/clones2cells_viewer.py --theme.base light

.. toctree::
   :caption: Main
   :maxdepth: 2
   :hidden:

   basics
   tutorial
   api

.. toctree::
   :caption: Exampels
   :maxdepth: 0
   :hidden:

   Basic_usage
   Multiple_injections
   Cluster_based_SHAP
   Cluster_free_SHAP
