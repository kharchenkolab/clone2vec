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

# ===== Tests for stack_layers =====

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_stack_layers_basic(matrix_kind, adata_dense, adata_sparse):
    ad = adata_dense if matrix_kind == "dense" else adata_sparse
    ad = ad.copy()

    X_base = ad.X.toarray() if sp.issparse(ad.X) else ad.X.copy()
    X_l1 = X_base.copy()
    X_l2 = X_base.copy()

    if X_l1.shape[0] >= 3:
        X_l1[0, :] = np.nan
        X_l1[2, :] = np.nan
    if X_l2.shape[0] >= 2:
        X_l2[1, :] = np.nan

    ad.layers["l1"] = X_l1 if matrix_kind == "dense" else sp.csr_matrix(X_l1)
    ad.layers["l2"] = X_l2 if matrix_kind == "dense" else sp.csr_matrix(X_l2)

    rng = np.random.RandomState(0)
    ad.obsm["X_umap"] = rng.rand(ad.n_obs, 2)
    ad.obsm["DF_embed"] = pd.DataFrame(rng.rand(ad.n_obs, 3), columns=["a","b","c"], index=ad.obs_names)

    stacked = sl.utils.stack_layers(ad, layers=["l1", "l2"], layer_col_added="layer")

    mask_l1 = ~sl.utils._nan_mask(ad.layers["l1"]) if matrix_kind == "sparse" else ~np.isnan(X_l1).any(axis=1)
    mask_l2 = ~sl.utils._nan_mask(ad.layers["l2"]) if matrix_kind == "sparse" else ~np.isnan(X_l2).any(axis=1)
    expected_rows = int(mask_l1.sum() + mask_l2.sum())

    assert stacked.n_obs == expected_rows
    assert stacked.n_vars == ad.n_vars

    if matrix_kind == "sparse":
        assert sp.issparse(stacked.X)
    else:
        assert isinstance(stacked.X, np.ndarray)

    assert "layer" in stacked.obs.columns
    assert set(stacked.obs["layer"].unique().tolist()) <= {"l1", "l2"}

    assert all(name.startswith("l1:") or name.startswith("l2:") for name in stacked.obs_names)

    assert "X_umap" in stacked.obsm
    assert stacked.obsm["X_umap"].shape[0] == expected_rows
    assert "DF_embed" in stacked.obsm
    df_embed = stacked.obsm["DF_embed"]
    assert isinstance(df_embed, pd.DataFrame)
    assert df_embed.shape[0] == expected_rows
    assert list(df_embed.columns) == ["a","b","c"]

# ===== Tests for _nan_mask =====

def test_nan_mask_dense():
    X = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, 0.5, 1.2],
        [4.0, 5.0, np.nan],
        [7.0, 8.0, 9.0],
    ])
    mask = sl.utils._nan_mask(X)
    assert mask.dtype == bool
    assert mask.tolist() == [False, True, True, False]


def test_nan_mask_sparse():
    data = np.array([1.0, np.nan, 2.0, 3.0])
    rows = np.array([0, 1, 2, 3])
    cols = np.array([0, 0, 1, 2])
    X = sp.csr_matrix((data, (rows, cols)), shape=(5, 4))
    mask = sl.utils._nan_mask(X)
    # Row 1 has NaN, others do not
    assert mask.tolist() == [False, True, False, False, False]


def test_nan_mask_sparse_empty():
    X = sp.csr_matrix((3, 4))
    mask = sl.utils._nan_mask(X)
    assert mask.tolist() == [False, False, False]


# ===== Tests for regress_categories =====

def test_regress_categories(adata_dense):
    ad = adata_dense.copy()
    ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
    sl.utils.regress_categories(ad, obs_key="cell_type", layer=None, key_added="regressed")
    assert "X_regressed" in ad.layers
    orig = ad.X
    reg = ad.layers["X_regressed"]
    assert reg.shape == orig.shape
    assert np.mean(np.abs(orig - reg)) > 0


# ===== Tests for get_connectivity_matrix =====

def test_get_connectivity_matrix():
    n_groups = 3
    conn = np.random.RandomState(0).rand(n_groups, n_groups)
    labels = ["A", "B", "C"]
    ad = sc.AnnData(np.zeros((5, 2)))
    ad.uns["group_connectivity"] = {
        "connectivity": conn,
        "label_names": labels,
    }
    df = sl.utils.get_connectivity_matrix(ad, uns_key="group_connectivity")
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == labels
    assert list(df.columns) == labels
    assert np.allclose(df.values, conn)


# ===== Tests for connect_clones =====

def test_connect_clones():
    n_clones = 10
    rng = np.random.RandomState(1)
    clones = sc.AnnData(
        X=np.zeros((n_clones, 2)),
        obs=pd.DataFrame({
            "group": pd.Categorical(["g1"] * 4 + ["g2"] * 3 + ["g3"] * 3),
        }),
    )
    clones.obs_names = [f"clone_{i}" for i in range(n_clones)]
    sl.utils.connect_clones(clones, groupby="group", graph_key_added="test_graph")
    assert "test_graph" in clones.obsp
    assert clones.obsp["test_graph"].shape == (n_clones, n_clones)
    assert sp.issparse(clones.obsp["test_graph"])
    # clones in same group should have nonzero entries
    assert clones.obsp["test_graph"].nnz > 0


def test_connect_clones_oriented():
    n_clones = 6
    clones = sc.AnnData(
        X=np.zeros((n_clones, 2)),
        obs=pd.DataFrame({
            "group": pd.Categorical(["g1"] * 3 + ["g2"] * 3),
            "time": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }),
    )
    clones.obs_names = [f"c{i}" for i in range(n_clones)]
    sl.utils.connect_clones(
        clones, groupby="group", graph_key_added="oriented",
        orient_col="time", orient_rule="increase",
    )
    assert "oriented" in clones.obsp
    assert clones.obsp["oriented"].nnz > 0


# ===== Tests for correct_shap =====

def test_correct_shap():
    n_obs, n_vars, n_preds = 20, 10, 3
    rng = np.random.RandomState(2)
    shapdata = sc.AnnData(
        X=rng.rand(n_obs, n_vars),
        obs=pd.DataFrame(index=[f"o{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    pred_cols = [f"p{i}" for i in range(n_preds)]
    shapdata.varm["mean_shap"] = pd.DataFrame(
        rng.rand(n_vars, n_preds), index=shapdata.var_names, columns=pred_cols,
    )
    shapdata.varm["gex_r"] = pd.DataFrame(
        rng.randn(n_vars, n_preds), index=shapdata.var_names, columns=pred_cols,
    )
    sl.utils.correct_shap(shapdata, shap_key="mean_shap", corr_key="gex_r",
                           correct_sign=True, normalize=False)
    assert "signed_mean_shap" in shapdata.varm

    sl.utils.correct_shap(shapdata, shap_key="mean_shap", corr_key="gex_r",
                           correct_sign=True, normalize=True)
    assert "signed_norm_mean_shap" in shapdata.varm


# ===== Tests for impute =====

def test_impute_numeric(adata_dense):
    ad = adata_dense.copy()
    if "X_pca" not in ad.obsm:
        sc.pp.pca(ad)
    rng = np.random.RandomState(3)
    vals = rng.rand(ad.n_obs)
    # mark some as NA
    na_idx = rng.choice(ad.n_obs, size=max(5, ad.n_obs // 10), replace=False)
    vals[na_idx] = np.nan
    ad.obs["score"] = vals
    ad.obs["score"] = ad.obs["score"].astype(float)
    sl.utils.impute(ad, obs_name="score", value_to_impute="nan",
                     use_rep="X_pca", key_added="imp", k=5)
    assert "score_imp" in ad.obs
    # imputed values should not be NaN for any cell
    assert ad.obs["score_imp"].notna().all()


def test_impute_categorical(adata_dense):
    ad = adata_dense.copy()
    if "X_pca" not in ad.obsm:
        sc.pp.pca(ad)
    labels = ad.obs["cell_type"].astype(str).copy()
    rng = np.random.RandomState(4)
    na_idx = rng.choice(ad.n_obs, size=max(5, ad.n_obs // 10), replace=False)
    labels.iloc[na_idx] = "NA"
    ad.obs["ct_with_na"] = labels
    sl.utils.impute(ad, obs_name="ct_with_na", value_to_impute="NA",
                     use_rep="X_pca", key_added="imp", k=5)
    assert "ct_with_na_imp" in ad.obs
    imputed_vals = ad.obs["ct_with_na_imp"].astype(str).values
    assert np.sum(imputed_vals == "NA") == 0
    assert "impute_prob" in ad.obsm