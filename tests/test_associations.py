import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pytest
import clone2vec as c2v
import pytest

# Skip CatBoost tests if library is unavailable
catboost = pytest.importorskip("catboost", reason="CatBoost not installed; skipping catboost tests")

from scipy.stats import pearsonr, spearmanr


# ===== Fixtures =====

@pytest.fixture(scope="session")
def adata():
    return sc.read_h5ad("tests/Weinreb_subsampled.h5ad")

@pytest.fixture(scope="session")
def adata_dense(adata):
    ad = adata.copy()
    X = ad.X
    ad.X = X.toarray() if sp.issparse(X) else np.array(X)
    return ad

@pytest.fixture(scope="session")
def adata_sparse(adata):
    ad = adata.copy()
    X = ad.X
    ad.X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    return ad

@pytest.fixture(scope="session")
def adata_with_umap(adata_dense):
    ad = adata_dense.copy()
    rng = np.random.RandomState(0)
    ad.obsm["X_umap"] = rng.rand(ad.n_obs, 2)
    return ad

@pytest.fixture(scope="session")
def adata_small_dense(adata_with_umap):
    ad = adata_with_umap.copy()
    n_cells = min(300, ad.n_obs)
    n_genes = min(200, ad.n_vars)
    ad = ad[:n_cells, :n_genes].copy()
    return ad

@pytest.fixture(scope="session")
def adata_small_sparse(adata_small_dense):
    ad = adata_small_dense.copy()
    ad.X = sp.csr_matrix(ad.X) if not sp.issparse(ad.X) else ad.X.tocsr()
    return ad

# ===== associations(): Pearson/Spearman on X =====

@pytest.mark.parametrize("method", ["pearson", "spearman"]) 
@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_associations_x_pearson_spearman(method, matrix_kind, adata_small_dense, adata_small_sparse):
    ad = adata_small_dense if matrix_kind == "dense" else adata_small_sparse
    ad = ad.copy()
    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="X",
        method=method,
        random_state=0,
    )
    assert f"X_umap:X:{method}:r" in ad.varm
    assert f"X_umap:X:{method}:pvalue" in ad.varm
    assert f"X_umap:X:{method}:p_adj" in ad.varm
    r = ad.varm[f"X_umap:X:{method}:r"]
    assert r.shape[0] == ad.n_vars
    assert r.shape[1] == ad.obsm["X_umap"].shape[1]

# ===== associations(): layers and raw paths =====

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_associations_layers(matrix_kind, adata_small_dense, adata_small_sparse):
    ad = adata_small_dense if matrix_kind == "dense" else adata_small_sparse
    ad = ad.copy()
    custom = ad.X.toarray() if sp.issparse(ad.X) else ad.X.copy()
    ad.layers["l1"] = custom if matrix_kind == "dense" else sp.csr_matrix(custom)
    ad.layers["l2"] = custom if matrix_kind == "dense" else sp.csr_matrix(custom)
    ad.obsm["X_umap"] = np.random.RandomState(1).rand(ad.n_obs, 2)
    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="layers",
        layers=["l1", "l2"],
        method="pearson",
        random_state=0,
    )
    assert "X_umap:l1:pearson:r" in ad.varm
    assert "X_umap:l1:pearson:pvalue" in ad.varm
    assert "X_umap:l1:pearson:p_adj" in ad.varm
    assert "X_umap:l2:pearson:r" in ad.varm
    assert "X_umap:l2:pearson:pvalue" in ad.varm
    assert "X_umap:l2:pearson:p_adj" in ad.varm

def test_associations_layers_concat(adata_small_dense):
    ad = adata_small_dense.copy()
    ad.layers["l1"] = ad.X.copy()
    ad.layers["l2"] = ad.X.copy()
    ad.obsm["X_umap"] = np.random.RandomState(2).rand(ad.n_obs, 1)
    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="layers",
        layers=["l1", "l2"],
        concat_layers=True,
        method="pearson",
        random_state=0,
    )
    assert "X_umap:multi_layer:pearson:r" in ad.varm
    assert "X_umap:multi_layer:pearson:pvalue" in ad.varm
    assert "X_umap:multi_layer:pearson:p_adj" in ad.varm

def test_associations_use_raw(adata_small_dense):
    ad = adata_small_dense.copy()
    ad.raw = ad
    ad.obsm["X_umap"] = np.random.RandomState(3).rand(ad.n_obs, 2)
    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="X",
        use_raw=True,
        method="pearson",
        random_state=0,
    )
    assert "X_umap:X_raw:pearson:r" in ad.raw.varm
    assert "X_umap:X_raw:pearson:pvalue" in ad.raw.varm
    assert "X_umap:X_raw:pearson:p_adj" in ad.raw.varm

# ===== associations(): GAM small slice =====

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_associations_gam_small(matrix_kind, adata_small_dense, adata_small_sparse):
    ad = adata_small_dense if matrix_kind == "dense" else adata_small_sparse
    ad = ad.copy()
    ad = ad[:, :min(50, ad.n_vars)].copy()
    ad.obsm["X_umap"] = np.random.RandomState(4).rand(ad.n_obs, 1)
    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="X",
        method="gam",
        random_state=0,
        progress_bar=False,
        n_jobs=1,
        spline_df=4,
    )
    assert "X_umap:X:gam:r2" in ad.varm
    assert "X_umap:X:gam:amplitude" in ad.varm
    assert "X_umap:X:gam:p_adj" in ad.varm

# ===== catboost(): regression on dense and sparse =====

def _add_validation_split(ad, frac=0.3):
    n = ad.n_obs
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    sel = rng.choice(idx, size=max(1, int(frac * n)), replace=False)
    s = pd.Series("full", index=ad.obs_names)
    s.iloc[sel] = "sketch"
    ad.obs["valid_split"] = pd.Categorical(s)

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_catboost_regressor_basic(matrix_kind, adata_small_dense, adata_small_sparse):
    ad = adata_small_dense if matrix_kind == "dense" else adata_small_sparse
    ad = ad.copy()
    ad = ad[:, :min(50, ad.n_vars)].copy()
    ad.obsm["X_umap"] = np.random.RandomState(5).rand(ad.n_obs, 2)
    _add_validation_split(ad, frac=0.3)
    shapdata = c2v.tl.catboost(
        ad,
        obsm_key="X_umap",
        validation_key="valid_split",
        validation_value="sketch",
        use_gpu=False,
        num_trees=20,
        early_stopping_rounds=5,
        verbose=False,
        return_model=False,
        random_state=0,
    )
    assert isinstance(shapdata, sc.AnnData)
    assert "X_umap" in shapdata.obsm
    assert "X_umap:predicted" in shapdata.obsm
    assert "mean_shap" in shapdata.varm
    assert "gex_r" in shapdata.varm
    assert "validation" in shapdata.obs
    assert "catboost_info" in shapdata.uns
    pred = shapdata.obsm["X_umap:predicted"]
    if sp.issparse(pred):
        assert pred.shape[0] == shapdata.n_obs
    else:
        assert pred.shape[0] == shapdata.n_obs


# ===== SciPy vs manual correlations match =====

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_pearson_matches_scipy(matrix_kind, adata_small_dense, adata_small_sparse):
    ad = adata_small_dense if matrix_kind == "dense" else adata_small_sparse
    ad = ad.copy()
    ad = ad[:, :min(50, ad.n_vars)].copy()
    rng = np.random.RandomState(11)
    ad.obsm["X_umap"] = rng.rand(ad.n_obs, 1)

    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="X",
        method="pearson",
        random_state=0,
    )

    r_df = ad.varm["X_umap:X:pearson:r"]
    p_df = ad.varm["X_umap:X:pearson:pvalue"]
    y = ad.obsm["X_umap"].ravel()

    for j in range(r_df.shape[0]):
        x = ad.X[:, j].toarray().ravel() if sp.issparse(ad.X) else ad.X[:, j].ravel()
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r_scipy, p_scipy = pearsonr(x, y)
        r_pkg = r_df.iloc[j, 0]
        p_pkg = p_df.iloc[j, 0]
        assert np.isfinite(r_pkg)
        assert np.isfinite(p_pkg)
        assert np.isclose(r_pkg, r_scipy, atol=1e-8)
        assert np.isclose(p_pkg, p_scipy, atol=1e-8)


@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_spearman_matches_scipy(matrix_kind, adata_small_dense, adata_small_sparse):
    ad = adata_small_dense if matrix_kind == "dense" else adata_small_sparse
    ad = ad.copy()
    ad = ad[:, :min(50, ad.n_vars)].copy()
    rng = np.random.RandomState(12)
    ad.obsm["X_umap"] = rng.rand(ad.n_obs, 1)

    c2v.tl.associations(
        ad,
        response_key="X_umap",
        response_field="obsm",
        use_rep="X",
        method="spearman",
        random_state=0,
    )

    r_df = ad.varm["X_umap:X:spearman:r"]
    p_df = ad.varm["X_umap:X:spearman:pvalue"]
    y = ad.obsm["X_umap"].ravel()

    for j in range(r_df.shape[0]):
        x = ad.X[:, j].toarray().ravel() if sp.issparse(ad.X) else ad.X[:, j].ravel()
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r_scipy, p_scipy = spearmanr(x, y)
        r_pkg = r_df.iloc[j, 0]
        p_pkg = p_df.iloc[j, 0]
        assert np.isfinite(r_pkg)
        assert np.isfinite(p_pkg)
        assert np.isclose(r_pkg, r_scipy, atol=1e-8)
        assert np.isclose(p_pkg, p_scipy, atol=1e-8)


# ===== binomial=True tests =====

@pytest.fixture(scope="session")
def adata_binomial(adata_small_dense):
    """Create an adata with count-based response in obsm for binomial testing."""
    ad = adata_small_dense.copy()
    rng = np.random.RandomState(99)
    # Simulate count data: 3 categories, the response columns are counts
    n = ad.n_obs
    counts = rng.multinomial(50, [0.5, 0.3, 0.2], size=n).astype(float)
    ad.obsm["clone_counts"] = counts
    return ad


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_associations_binomial_correlation(method, adata_binomial):
    ad = adata_binomial.copy()
    c2v.tl.associations(
        ad,
        response_key="clone_counts",
        response_field="obsm",
        use_rep="X",
        method=method,
        binomial=True,
        random_state=0,
    )
    key_prefix = f"clone_counts:X:{method}"
    assert f"{key_prefix}:r" in ad.varm
    assert f"{key_prefix}:pvalue" in ad.varm
    assert f"{key_prefix}:p_adj" in ad.varm

    r = ad.varm[f"{key_prefix}:r"]
    assert r.shape[0] == ad.n_vars
    assert r.shape[1] == 3
    # Correlations should be in [-1, 1] where finite
    finite_r = r.values[np.isfinite(r.values)]
    assert np.all(finite_r >= -1.0 - 1e-10)
    assert np.all(finite_r <= 1.0 + 1e-10)


def test_associations_binomial_gam(adata_binomial):
    ad = adata_binomial.copy()
    ad = ad[:, :min(20, ad.n_vars)].copy()
    # Use only 1 response column for speed
    ad.obsm["clone_counts_1col"] = ad.obsm["clone_counts"][:, :2]
    c2v.tl.associations(
        ad,
        response_key="clone_counts_1col",
        response_field="obsm",
        use_rep="X",
        method="gam",
        binomial=True,
        random_state=0,
        progress_bar=False,
        n_jobs=1,
        spline_df=4,
    )
    key_prefix = "clone_counts_1col:X:gam"
    assert f"{key_prefix}:r2" in ad.varm
    assert f"{key_prefix}:amplitude" in ad.varm
    assert f"{key_prefix}:pvalue" in ad.varm
    assert f"{key_prefix}:p_adj" in ad.varm


def test_weighted_corr_vs_manual():
    """Verify that _fast_corr with weights matches a manual weighted Pearson."""
    from clone2vec.associations import _fast_corr

    rng = np.random.RandomState(42)
    n, p, q = 100, 5, 2
    X = rng.randn(n, p)
    Y = rng.randn(n, q)
    weights = rng.rand(n) * 10 + 1  # positive weights

    res = _fast_corr(X, Y, method="pearson", significance=True, slope=False, weights=weights)

    # Manual weighted Pearson
    w = weights / weights.sum()
    for j in range(p):
        for k in range(q):
            x = X[:, j]
            y = Y[:, k]
            mx = np.sum(w * x)
            my = np.sum(w * y)
            cov_xy = np.sum(w * (x - mx) * (y - my))
            sx = np.sqrt(np.sum(w * (x - mx) ** 2))
            sy = np.sqrt(np.sum(w * (y - my) ** 2))
            r_manual = cov_xy / (sx * sy)
            assert np.isclose(res["r"][j, k], r_manual, atol=1e-10), \
                f"Weighted Pearson mismatch at ({j},{k}): {res['r'][j, k]} vs {r_manual}"

    # Check p-values are finite
    assert np.all(np.isfinite(res["pvalue"]))