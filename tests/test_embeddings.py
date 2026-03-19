import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pytest
import clone2vec as c2v


# ===== Fixtures =====

@pytest.fixture(scope="session")
def adata():
    return sc.read_h5ad("tests/Weinreb_subsampled.h5ad")

@pytest.fixture(scope="session")
def clones_basic(adata):
    clones = c2v.pp.clones_adata(adata, obs_name="clone", min_size=3, fill_obs="cell_type")
    c2v.tl.clonal_nn(adata, clones, obs_name="clone", k=10, use_rep="X_pca", obsp_name="gex_adjacency")
    return clones

@pytest.fixture(scope="session")
def clones_dense(clones_basic):
    cl = clones_basic.copy()
    G = cl.obsp["gex_adjacency"]
    cl.obsp["gex_adjacency"] = G.toarray() if sp.issparse(G) else np.array(G)
    return cl

@pytest.fixture(scope="session")
def clones_sparse(clones_basic):
    cl = clones_basic.copy()
    G = cl.obsp["gex_adjacency"]
    cl.obsp["gex_adjacency"] = sp.csr_matrix(G) if not sp.issparse(G) else G.tocsr()
    return cl

# ===== clone2vec (Skip-Gram) =====

def test_clone2vec_sparse(clones_sparse):
    c2v.embeddings.clone2vec(
        clones_sparse,
        z_dim=4,
        obsp_key="gex_adjacency",
        max_iter=5,
        progress_bar=True,
        device="cpu",
    )
    assert "clone2vec" in clones_sparse.obsm
    Z = clones_sparse.obsm["clone2vec"]
    assert Z.shape == (clones_sparse.n_obs, 4)
    assert "clone2vec" in clones_sparse.uns
    params = clones_sparse.uns["clone2vec"]
    assert params["type"] == "fit"
    assert "loss_history" in params

def test_clone2vec_dense(clones_dense):
    c2v.embeddings.clone2vec(
        clones_dense,
        z_dim=3,
        obsp_key="gex_adjacency",
        max_iter=5,
        progress_bar=True,
        device="cpu",
    )
    Z = clones_dense.obsm["clone2vec"]
    assert Z.shape == (clones_dense.n_obs, 3)

# ===== clone2vec (fastglmpca) =====

def test_clone2vec_poi_sparse(clones_sparse):
    c2v.embeddings.clone2vec_Poi(
        clones_sparse,
        z_dim=5,
        obsp_key="gex_adjacency",
        max_iter=5,
        learning_rate=0.5,
        progress_bar=False,
    )
    assert "clone2vec_Poi" in clones_sparse.obsm
    Z = clones_sparse.obsm["clone2vec_Poi"]
    assert Z.shape == (clones_sparse.n_obs, 5)
    assert "clone2vec_Poi" in clones_sparse.uns
    params = clones_sparse.uns["clone2vec_Poi"]
    assert params["type"] == "fit"

def test_clone2vec_poi_dense(clones_dense):
    c2v.embeddings.clone2vec_Poi(
        clones_dense,
        z_dim=2,
        obsp_key="gex_adjacency",
        max_iter=5,
        learning_rate=0.5,
        progress_bar=False,
    )
    Z = clones_dense.obsm["clone2vec_Poi"]
    assert Z.shape == (clones_dense.n_obs, 2)

# ===== Projection =====

@pytest.fixture(scope="session")
def projection_setup(adata):
    clones_all = c2v.pp.clones_adata(adata, obs_name="clone", min_size=3, fill_obs="cell_type")
    c2v.tl.clonal_nn(adata, clones_all, obs_name="clone", k=10, use_rep="X_pca", obsp_name="gex_adjacency")

    clones_ref = clones_all.copy()
    c2v.embeddings.clone2vec(
        clones_ref,
        z_dim=3,
        obsp_key="gex_adjacency",
        max_iter=5,
        progress_bar=True,
        device="cpu",
    )
    c2v.embeddings.clone2vec_Poi(
        clones_ref,
        z_dim=3,
        obsp_key="gex_adjacency",
        max_iter=5,
        progress_bar=False,
    )

    clones_query = clones_all.copy()
    nn = clones_query.obsp["gex_adjacency"]
    if sp.issparse(nn):
        nn = nn.toarray()
    clones_query.obsm["ref_gex_adjacency"] = nn

    return clones_query, clones_ref

def test_project_clone2vec(projection_setup):
    clones_query, clones_ref = projection_setup
    c2v.embeddings.project_clone2vec(
        clones_query,
        clones_ref,
        obsm_key_query="ref_gex_adjacency",
        uns_key_query="clone2vec",
        uns_key_ref="clone2vec",
        obsm_key="clone2vec",
        progress_bar=True,
        max_iter=5,
        device="cpu",
    )
    assert "clone2vec" in clones_query.obsm
    Zq = clones_query.obsm["clone2vec"]
    assert Zq.shape[0] == clones_query.n_obs
    assert "clone2vec" in clones_query.uns
    assert clones_query.uns["clone2vec"]["type"] == "project"

def test_project_clone2vec_poi(projection_setup):
    clones_query, clones_ref = projection_setup
    c2v.embeddings.project_clone2vec_Poi(
        clones_query,
        clones_ref,
        obsm_key_query="ref_gex_adjacency",
        uns_key_query="clone2vec_Poi",
        uns_key_ref="clone2vec_Poi",
        obsm_key="clone2vec_Poi",
        progress_bar=True,
        max_iter=5,
        device="cpu",
    )
    assert "clone2vec_Poi" in clones_query.obsm
    Zq = clones_query.obsm["clone2vec_Poi"]
    assert Zq.shape[0] == clones_query.n_obs
    assert "clone2vec_Poi" in clones_query.uns
    assert clones_query.uns["clone2vec_Poi"]["type"] == "project"