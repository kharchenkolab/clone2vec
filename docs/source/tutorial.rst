Tutorial
========

Step 1: Clonal nearest neighbours graph construction
****************************************************

Firstly, we will create AnnData-object with clones.

.. code-block:: python

    import scanpy as sc
    import clone2vec as c2v

    clones = c2v.pp.clones_adata(
        adata,
        obs_name="clone", # Column with clonal labels
        min_size=2,       # Minimal clone size
        na_value="NA",    # Value for non-labelled cells
    )


Minimal clone size parameter is used to exclude small clones from embedding construction.

Step 2: clone2vec
*****************

Now, we have to train our neural network to predict clonal labels of nearest neighbours for each
clonally labelled cell.

.. code-block:: python

    c2v.tl.clonal_nn(
        adata,
        clones,
        use_rep="X_pca", # Which dimred to use for graph construction
    )

    c2v.tl.clone2vec(clones)


After execution of this function we have AnnData-object :code:`clones` with clonal vector representation
stored in :code:`clones.obsm["clone2vec"]`. Now we can work with it like with regular scRNA-Seq dataset.

Step 3: clone2vec analysis
**************************

.. code-block:: python

    sc.pp.neighbors(clones, use_rep="clone2vec")
    sc.tl.umap(clones)
    sc.tl.leiden(clones)

And after perform all other additional steps of analysis.

For a more detailed walkthrough see the Examples section.