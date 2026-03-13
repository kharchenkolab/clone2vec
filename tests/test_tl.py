import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pytest
import sclitr as sl


# ===== Fixtures =====

@pytest.fixture(scope="session")
def adata():
    return sc.read_h5ad("tests/Weinreb_subsampled.h5ad")

def _prep_neighbors(ad):
    ad = ad.copy()
    if "X_pca" not in ad.obsm:
        sc.pp.pca(ad)
    if "connectivities" not in ad.obsp:
        sc.pp.neighbors(ad)
    return ad

@pytest.fixture(scope="session")
def adata_with_graph_dense(adata):
    ad = adata.copy()
    X = ad.X
    ad.X = np.array(X.toarray() if sp.issparse(X) else X)
    ad = _prep_neighbors(ad)
    return ad

@pytest.fixture(scope="session")
def adata_with_graph_sparse(adata):
    ad = adata.copy()
    X = ad.X
    ad.X = sp.csr_matrix(X if sp.issparse(X) else np.array(X))
    ad = _prep_neighbors(ad)
    return ad

# ===== tl.smooth tests =====

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_smooth_X(matrix_kind, adata_with_graph_dense, adata_with_graph_sparse):
    ad = adata_with_graph_dense if matrix_kind == "dense" else adata_with_graph_sparse
    sl.tl.smooth(ad, field="X", key_added="smoothed", graph_key="connectivities", n_steps=2)
    assert "X_smoothed" in ad.layers
    X_orig = ad.X
    X_sm = ad.layers["X_smoothed"]
    if sp.issparse(X_orig):
        assert X_sm.shape == X_orig.shape
    else:
        assert np.array(X_sm).shape == np.array(X_orig).shape
    a = X_orig.toarray() if sp.issparse(X_orig) else np.array(X_orig)
    b = X_sm.toarray() if sp.issparse(X_sm) else np.array(X_sm)
    assert np.mean(np.abs(a - b)) > 0

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_smooth_layer(matrix_kind, adata_with_graph_dense, adata_with_graph_sparse):
    ad = adata_with_graph_dense if matrix_kind == "dense" else adata_with_graph_sparse
    ad.layers["custom"] = ad.X.copy()
    sl.tl.smooth(ad, field="layer", names="custom", key_added="sm", graph_key="connectivities", n_steps=1)
    assert "custom_sm" in ad.layers
    L_orig = ad.layers["custom"]
    L_sm = ad.layers["custom_sm"]
    a = L_orig.toarray() if sp.issparse(L_orig) else np.array(L_orig)
    b = L_sm.toarray() if sp.issparse(L_sm) else np.array(L_sm)
    assert a.shape == b.shape
    assert np.mean(np.abs(a - b)) > 0

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_smooth_obs(matrix_kind, adata_with_graph_dense, adata_with_graph_sparse):
    ad = adata_with_graph_dense if matrix_kind == "dense" else adata_with_graph_sparse
    rng = np.random.RandomState(0)
    ad.obs["rand_obs"] = rng.rand(ad.n_obs).astype(float)
    sl.tl.smooth(ad, field="obs", names="rand_obs", key_added="sm", graph_key="connectivities", n_steps=2)
    assert "rand_obs_sm" in ad.obs
    v_orig = ad.obs["rand_obs"].values
    v_sm = ad.obs["rand_obs_sm"].values
    assert v_orig.shape == v_sm.shape
    assert np.mean(np.abs(v_orig - v_sm)) > 0

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_smooth_obsm(matrix_kind, adata_with_graph_dense, adata_with_graph_sparse):
    ad = adata_with_graph_dense if matrix_kind == "dense" else adata_with_graph_sparse
    if "X_pca" not in ad.obsm:
        sc.pp.pca(ad)
    Z_orig = ad.obsm["X_pca"].copy()
    sl.tl.smooth(ad, field="obsm", names="X_pca", key_added="sm", graph_key="connectivities", n_steps=1)
    assert "X_pca_sm" in ad.obsm
    Z_sm = ad.obsm["X_pca_sm"]
    assert Z_sm.shape == Z_orig.shape
    assert np.mean(np.abs(np.array(Z_orig) - np.array(Z_sm))) > 0

# ===== utils.gs tests =====

def test_gs_without_batch(adata_with_graph_dense):
    pytest.importorskip("geosketch", reason="geosketch not installed; skipping gs tests")
    ad = adata_with_graph_dense.copy()
    sl.utils.gs(ad, use_rep="X_pca", n=0.2, obs_key="gs", random_state=1)
    assert "gs" in ad.obs
    vals = ad.obs["gs"].astype(str).values
    assert set(np.unique(vals)) <= {"full", "sketch"}
    assert np.sum(vals == "sketch") >= 1
    assert "gs" in ad.uns
    params = ad.uns["gs"]
    assert params["use_rep"] == "X_pca"
    assert params["obs_key"] == "gs"

def test_gs_with_batch(adata_with_graph_dense):
    pytest.importorskip("geosketch", reason="geosketch not installed; skipping gs tests")
    ad = adata_with_graph_dense.copy()
    n = ad.n_obs
    split = int(0.5 * n)
    ad.obs["batch"] = pd.Categorical(["A"] * split + ["B"] * (n - split))
    sl.utils.gs(ad, use_rep="X_pca", batch_key="batch", n=0.2, obs_key="gs_b", random_state=2)
    assert "gs_b" in ad.obs
    vals = ad.obs["gs_b"].astype(str).values
    assert set(np.unique(vals)) <= {"full", "sketch"}
    for b in ["A", "B"]:
        mask = ad.obs["batch"].astype(str).values == b
        assert np.sum(vals[mask] == "sketch") >= 1
    assert "gs" in ad.uns or "gs_b" in ad.uns

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_gs_use_X_dense_sparse(matrix_kind, adata_with_graph_dense, adata_with_graph_sparse):
    pytest.importorskip("geosketch", reason="geosketch not installed; skipping gs tests")
    ad = adata_with_graph_dense if matrix_kind == "dense" else adata_with_graph_sparse
    sl.utils.gs(ad, use_rep="X", n=0.1, obs_key="gs_x", random_state=3)
    assert "gs_x" in ad.obs
    vals = ad.obs["gs_x"].astype(str).values
    assert set(np.unique(vals)) <= {"full", "sketch"}


# ===== Fixtures for clonal_nn / clonocluster / group_connectivity =====

@pytest.fixture(scope="session")
def clones_with_pca(adata):
    cl = sl.pp.clones_adata(adata, obs_name="clone", min_size=30, fill_obs="cell_type")
    cl_expr = sl.pp.transfer_expression(adata, cl, obs_name="clone")
    sc.pp.pca(cl_expr, n_comps=min(10, cl_expr.n_vars - 1, cl_expr.n_obs - 1))
    return cl_expr


# ===== tl.clonal_nn tests =====

def test_clonal_nn_basic(adata, clones_with_pca):
    ad = adata.copy()
    if "X_pca" not in ad.obsm:
        sc.pp.pca(ad)
    cl = clones_with_pca.copy()
    sl.tl.clonal_nn(ad, cl, obs_name="clone", k=5, use_rep="X_pca", obsp_name="gex_adj")
    assert "gex_adj" in cl.obsp
    assert cl.obsp["gex_adj"].shape == (cl.n_obs, cl.n_obs)
    assert sp.issparse(cl.obsp["gex_adj"])
    assert "clonal_nn" in cl.uns


def test_clonal_nn_split_by(adata, clones_with_pca):
    ad = adata.copy()
    if "X_pca" not in ad.obsm:
        sc.pp.pca(ad)
    cl = clones_with_pca.copy()
    sl.tl.clonal_nn(
        ad, cl, obs_name="clone", k=5, use_rep="X_pca",
        obsp_name="gex_adj_split", split_by="cell_type",
    )
    assert "gex_adj_split" in cl.obsp
    assert cl.obsp["gex_adj_split"].shape == (cl.n_obs, cl.n_obs)


# ===== tl.clonocluster tests =====

def test_clonocluster(adata_with_graph_dense):
    ad = adata_with_graph_dense.copy()
    # Build a synthetic lineage graph in obsp
    n = ad.n_obs
    rng = np.random.RandomState(42)
    rows = rng.randint(0, n, size=50)
    cols = rng.randint(0, n, size=50)
    vals = np.ones(50)
    lineage = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    ad.obsp["connected"] = lineage
    # Ensure use_rep is in neighbors params (not always set by sc.pp.neighbors)
    if "use_rep" not in ad.uns.get("neighbors", {}).get("params", {}):
        ad.uns["neighbors"]["params"]["use_rep"] = "X_pca"
    sl.tl.clonocluster(ad, alpha=0.2, beta=0.1, key_added="cc", lineage_graph="connected")
    assert "cc_connectivities" in ad.obsp
    assert "cc" in ad.uns
    assert ad.uns["cc"]["connectivities_key"] == "cc_connectivities"


# ===== tl.group_connectivity tests =====

def test_group_connectivity(adata_with_graph_dense):
    ad = adata_with_graph_dense.copy()
    ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
    sl.tl.group_connectivity(ad, groupby="cell_type", graph_key="connectivities")
    assert "group_connectivity" in ad.uns
    gc = ad.uns["group_connectivity"]
    assert "connectivity" in gc
    assert "label_names" in gc
    k = len(ad.obs["cell_type"].cat.categories)
    assert gc["connectivity"].shape == (k, k)