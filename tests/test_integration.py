import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import pytest
import clone2vec as c2v


# ===== Fixtures =====

@pytest.fixture(scope="session")
def adata():
    return sc.read_h5ad("tests/Weinreb_subsampled.h5ad")

@pytest.fixture(scope="session")
def clones_base(adata):
    clones = c2v.pp.clones_adata(adata, obs_name="clone", min_size=30, fill_obs=None)
    return clones

@pytest.fixture(scope="session")
def clones_with_pca(clones_base):
    cl = clones_base.copy()
    rng = np.random.RandomState(42)
    cl.obsm["X_pca"] = rng.randn(cl.n_obs, 5).astype(np.float32)
    return cl

@pytest.fixture(scope="session")
def clones_dense(clones_with_pca):
    cl = clones_with_pca.copy()
    Xp = cl.obsm["X_pca"]
    cl.obsm["X_pca"] = np.array(Xp)
    n = cl.n_obs
    split = int(0.6 * n)
    cl.obs["batch"] = pd.Categorical(["A"] * split + ["B"] * (n - split))
    return cl

@pytest.fixture(scope="session")
def clones_dataframe(clones_with_pca):
    cl = clones_with_pca.copy()
    Xp = cl.obsm["X_pca"]
    cols = [f"PC{i+1}" for i in range(Xp.shape[1])]
    cl.obsm["X_pca"] = pd.DataFrame(Xp, index=cl.obs_names, columns=cols)
    labels = np.where(np.arange(cl.n_obs) % 2 == 0, "A", "B")
    cl.obs["batch"] = pd.Categorical(labels)
    return cl

# ===== find_mnn tests =====

@pytest.mark.parametrize("rep_kind", ["dense", "dataframe"]) 
def test_find_mnn_builds_graph_and_uns(rep_kind, clones_dense, clones_dataframe):
    clones = clones_dense if rep_kind == "dense" else clones_dataframe
    c2v.tl.find_mnn(clones, batch_key="batch", use_rep="X_pca", k=5, metric="euclidean", uns_key="integration_anchors", graph_key="mnn")

    assert "mnn" in clones.obsp
    G = clones.obsp["mnn"]
    assert sp.issparse(G)
    assert G.shape == (clones.n_obs, clones.n_obs)

    diff = (G - G.T).tocoo()
    assert diff.nnz == 0

    assert "integration_anchors" in clones.uns
    params = clones.uns["integration_anchors"]
    assert params["batch_key"] == "batch"
    assert params["use_rep"] == "X_pca"
    assert params["graph_key"] == "mnn"
    assert isinstance(params["order"], str)
    assert len(params["order"]) > 0

# ===== align tests =====

def test_align_produces_adjusted_representation(clones_dense):
    clones = clones_dense.copy()
    c2v.tl.find_mnn(clones, batch_key="batch", use_rep="X_pca", k=5, metric="euclidean", uns_key="integration_anchors", graph_key="mnn")

    clones.obsm["clone2vec"] = np.array(clones.obsm["X_pca"]) + 1e-6

    c2v.tl.align(clones, use_rep="clone2vec", adj_rep="clone2vec_pa", uns_key="integration_anchors")

    assert "clone2vec_pa" in clones.obsm
    Z_orig = clones.obsm["clone2vec"]
    Z_adj = clones.obsm["clone2vec_pa"]
    assert Z_adj.shape == Z_orig.shape

    counts = clones.obs["batch"].value_counts()
    minority = counts.idxmin()
    mask_query = clones.obs["batch"].values == minority

    changed = np.any(np.abs(Z_adj[mask_query] - Z_orig[mask_query]) > 1e-8)
    assert changed

# ===== internal helpers round-trip =====

def test_order_write_read_round_trip():
    sim = pd.DataFrame(
        [[0, 5, 2], [5, 0, 3], [2, 3, 0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    order = c2v.integration._hierarchical_concatenation(sim)
    s = c2v.integration._write_order(order)
    order_back = c2v.integration._read_order(s)

    assert str(order) == str(order_back)