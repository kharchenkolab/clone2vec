Tutorial
========

Step 1: Clonal nearest neighbours graph construction
****************************************************

Firstly, we have to identify *k* nearest clonally labelled cells for each cell. It will create "bag of clones"
(similar to "bag of words") that will be used for *clone2vec* training further.

.. code-block:: python

    import scanpy as sc
    import sclitr as sl

    sl.tl.clonal_nn(
        adata,
        obs_name="clone",  # Column with clonal labels
        use_rep="X_pca",   # Which dimred to use for graph construction
        min_size=5,         # Minimal clone size
    )

Minimal clone size parameter is used to exclude small clones from embedding construction.

Step 2: clone2vec
*****************

Now, we have to train our neural network to predict clonal labels of nearest neighbours for each
clonally labelled cell.

.. code-block:: python

    clones = sl.pp.clones_adata(
        adata,
        obs_name="clone",
        min_size=5,
        fill_obs="cell_type",  # Optional: composition column to fill clones layers
    )

    # build clone graph in clones.obsp["gex_adjacency"] using cell-level embedding
    # e.g., cosine similarity between clone centroids in `adata.obsm["X_pca"]` and kNN
    # (see your pipeline; not shown here for brevity)

    sl.tl.clone2vec(
        clones,
        z_dim=10,
        obsp_key="gex_adjacency",  # graph between clones
        obsm_key="clone2vec",
        uns_key="clone2vec",
    )

After execution of this function we have AnnData-object :code:`clones` with clonal vector representation
stored in :code:`clones.obsm["clone2vec"]`. Now we can work with it like with regular scRNA-Seq dataset.

Step 3: clone2vec analysis
**************************

.. code-block:: python

    sc.pp.neighbors(clones, use_rep="clone2vec")
    sc.tl.umap(clones)
    sc.tl.leiden(clones)

And after perform all other additional steps of analysis.

Step 4: Identify predictors of clonal behaviour
***********************************************

In the simplest case, the model can be built to identify gene expression predictors of (a) position on a
clonal embedding and (b) cell type composition of clones based on the expression in progenitor cells (if they exist).
More broadly, we don't have to limit the prediction by the progenitor cells, and in this case the algorithm will
identify general gene expression predictors of the distribution of the clone on an embedding.

.. code-block:: python

    shapdata_c2v = sl.tl.catboost(
        adata,
        obsm_key="X_umap",  # predict clone position; replace with your embedding
        gs_key="gs",         # optional: use gs split info if available
        model="regressor",
        num_trees=1000,
    )

    shapdata_ct = sl.tl.associations(
        adata,
        response_key="proportions",    # or a specific layer/metric
        response_field="obsm",         # depends on how you store targets
        method="gam",                  # pearson/spearman/gam
    )

For a more detailed walkthrough see the Examples section.