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
# ===== mask_key and custom initialization =====

def _add_mask(clones, n_cols=2, seed=0):
    clones.obsm["mask"] = np.random.RandomState(seed).poisson(2, (clones.n_obs, n_cols)).astype(float)

def test_clone2vec_mask_key(clones_sparse):
    cl = clones_sparse.copy()
    _add_mask(cl)
    c2v.tl.clone2vec(cl, z_dim=3, mask_key="mask", max_iter=3, progress_bar=False, device="cpu")
    params = cl.uns["clone2vec"]
    assert cl.obsm["clone2vec"].shape == (cl.n_obs, 3)
    assert list(params["var_names"][-2:]) == ["mask1", "mask2"]
    assert params["output"]["weight"].shape[0] == cl.n_obs + 2

def test_clone2vec_missing_mask_key(clones_sparse):
    with pytest.raises(KeyError):
        c2v.tl.clone2vec(clones_sparse.copy(), z_dim=3, mask_key="absent", max_iter=1, progress_bar=False)

def test_clone2vec_poi_mask_key(clones_sparse):
    pytest.importorskip("fastglmpca")
    cl = clones_sparse.copy()
    _add_mask(cl)
    c2v.tl.clone2vec_Poi(cl, z_dim=3, mask_key="mask", max_iter=3, progress_bar=False)
    params = cl.uns["clone2vec_Poi"]
    assert cl.obsm["clone2vec_Poi"].shape == (cl.n_obs, 3)
    assert params["weights"]["V"].shape[0] == cl.n_obs + 2
    assert list(params["var_names"][-2:]) == ["mask1", "mask2"]

def test_clone2vec_custom_init(clones_sparse):
    cl = clones_sparse.copy()
    cl.obsm["init"] = np.random.RandomState(0).randn(cl.n_obs, 3).astype(np.float32)
    c2v.tl.clone2vec(cl, z_dim=3, init="custom", init_obsm="init", max_iter=0, progress_bar=False, device="cpu")
    assert np.allclose(cl.obsm["clone2vec"], cl.obsm["init"], atol=1e-5)

def test_project_clone2vec_mask_key(projection_setup):
    clones_query, clones_ref = projection_setup
    cq = clones_query.copy()
    _add_mask(cq, seed=1)
    c2v.tl.project_clone2vec(cq, clones_ref, mask_key="mask", obsm_key="proj", uns_key_query="proj",
                             max_iter=3, progress_bar=False, device="cpu")
    assert cq.obsm["proj"].shape == (cq.n_obs, clones_ref.obsm["clone2vec"].shape[1])
    assert len(cq.uns["proj"]["var_names"]) == clones_ref.n_obs + 2

def test_project_clone2vec_poi_mask_key(projection_setup):
    pytest.importorskip("fastglmpca")
    clones_query, clones_ref = projection_setup
    cq = clones_query.copy()
    _add_mask(cq, seed=1)
    c2v.tl.project_clone2vec_Poi(cq, clones_ref, mask_key="mask", max_iter=3, progress_bar=False)
    assert cq.obsm["clone2vec_Poi"].shape[0] == cq.n_obs
    assert cq.uns["clone2vec_Poi_project"]["type"] == "project"
    assert len(cq.uns["clone2vec_Poi_project"]["var_names"]) == clones_ref.n_obs + 2

def test_project_clone2vec_requires_reference(projection_setup):
    clones_query, clones_ref = projection_setup
    with pytest.raises(KeyError):
        c2v.tl.project_clone2vec(clones_query.copy(), clones_ref, uns_key_ref="absent")

# ===== orthogonalize =====

def _probabilities(U, W, b):
    logits = U.astype(np.float64) @ W.T.astype(np.float64) + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(axis=1, keepdims=True)

def _fit_pair(clones, **kwargs):
    plain, orth = clones.copy(), clones.copy()
    args = dict(z_dim=4, max_iter=5, progress_bar=False, device="cpu", **kwargs)
    c2v.tl.clone2vec(plain, **args)
    c2v.tl.clone2vec(orth, orthogonalize=True, **args)
    return plain, orth

def test_clone2vec_default_unchanged(clones_sparse):
    cl = clones_sparse.copy()
    c2v.tl.clone2vec(cl, z_dim=4, max_iter=5, progress_bar=False, device="cpu")
    model = c2v.embeddings.SkipGram(z_dim=4, device="cpu")
    model.fit(cl.obsp["gex_adjacency"], random_state=4, max_iter=5, init="svd", progress_bar=False, device="cpu")
    params = cl.uns["clone2vec"]
    assert np.array_equal(cl.obsm["clone2vec"], model.U)
    assert np.array_equal(params["output"]["weight"], model.output.weight.detach().numpy())
    assert np.array_equal(params["output"]["bias"], model.output.bias.detach().numpy())
    assert params["init"]["orthogonalize"] is False
    assert "singular_values" not in params

def test_orthogonalize_preserves_probabilities(clones_sparse):
    plain, orth = _fit_pair(clones_sparse)
    p, o = plain.uns["clone2vec"]["output"], orth.uns["clone2vec"]["output"]
    assert np.allclose(
        _probabilities(plain.obsm["clone2vec"], p["weight"], p["bias"]),
        _probabilities(orth.obsm["clone2vec"], o["weight"], o["bias"]),
        atol=1e-6,
    )
    assert orth.uns["clone2vec"]["init"]["orthogonalize"] is True

def test_orthogonalize_centered_orthogonal(clones_sparse):
    _, orth = _fit_pair(clones_sparse)
    U = orth.obsm["clone2vec"].astype(np.float64)
    W = orth.uns["clone2vec"]["output"]["weight"].astype(np.float64)
    S = orth.uns["clone2vec"]["singular_values"]
    assert np.allclose(U.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(U.T @ U, np.diag(S ** 2), rtol=1e-4, atol=1e-4 * S[0] ** 2)
    assert np.allclose(W.T @ W, np.eye(4), atol=1e-5)
    assert np.all(np.diff(S) <= 0)

def test_orthogonalize_equivalent_inputs():
    rng = np.random.RandomState(0)
    U, V, b = rng.randn(50, 4), rng.randn(60, 4), rng.randn(60)
    A, s = rng.randn(4, 4), rng.randn(4)
    A_inv_T = np.linalg.inv(A).T
    U1, V1, b1, S1 = c2v.embeddings._orthogonalize(U, V, b)
    U2, V2, b2, S2 = c2v.embeddings._orthogonalize(U @ A + s, V @ A_inv_T, b - V @ A_inv_T @ s)
    for x, y in [(U1, U2), (V1, V2), (b1, b2), (S1, S2)]:
        assert np.allclose(x, y, atol=1e-8)

def test_orthogonalize_rank_warning(monkeypatch):
    messages = []
    monkeypatch.setattr(c2v.embeddings.logg, "warning", messages.append)
    rng = np.random.RandomState(0)
    U = rng.randn(50, 4)
    U[:, 3] = U[:, 0] + U[:, 1]
    c2v.embeddings._orthogonalize(U, rng.randn(60, 4), rng.randn(60))
    assert any("rank 3" in m for m in messages)

def test_orthogonalize_mask_key(clones_sparse):
    cl = clones_sparse.copy()
    _add_mask(cl)
    plain, orth = _fit_pair(cl, mask_key="mask")
    p, o = plain.uns["clone2vec"]["output"], orth.uns["clone2vec"]["output"]
    assert o["weight"].shape == (cl.n_obs + 2, 4)
    assert np.allclose(o["weight"].T @ o["weight"], np.eye(4), atol=1e-5)
    assert np.allclose(
        _probabilities(plain.obsm["clone2vec"], p["weight"], p["bias"]),
        _probabilities(orth.obsm["clone2vec"], o["weight"], o["bias"]),
        atol=1e-6,
    )

def test_orthogonalize_propagates(adata, clones_basic, monkeypatch):
    cl = clones_basic.copy()
    c2v.tl.clone2vec(cl, z_dim=3, max_iter=3, progress_bar=False, device="cpu", orthogonalize=True)

    fit = c2v.tools.clone2vec
    monkeypatch.setattr(c2v.tools, "clone2vec", lambda *args, **kwargs: fit(*args, max_iter=3, device="cpu", **kwargs))
    c2v.tl.mask_heterogeneity(adata, cl, mask=(adata.obs["cell_type"] == "Undiff").values)
    assert cl.uns["mask"]["clone2vec"]["init"]["orthogonalize"] is True
    assert "singular_values" in cl.uns["mask"]["clone2vec"]

    c2v.tl.permute_embeddings(adata, cl, n_permutations=1, progress_bar=False,
                              clone2vec_kwargs={"max_iter": 3, "device": "cpu"})
    perm = cl.obsm["clonal_permutations"][..., 0]
    assert cl.uns["clonal_permutations"]["init"]["orthogonalize"] is True
    assert np.allclose(perm.mean(axis=0), 0, atol=1e-5)

    cq = cl.copy()
    nn = cq.obsp["gex_adjacency"]
    cq.obsm["ref_gex_adjacency"] = nn.toarray() if sp.issparse(nn) else nn
    c2v.tl.project_clone2vec(cq, cl, obsm_key="proj", uns_key_query="proj", max_iter=3, progress_bar=False, device="cpu")
    assert cq.uns["proj"]["init"]["orthogonalize"] is True
