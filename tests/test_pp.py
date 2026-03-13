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

@pytest.fixture(scope="session")
def adata_with_umap(adata_dense):
    ad = adata_dense.copy()
    ad.obsm["X_umap"] = np.random.RandomState(0).rand(ad.n_obs, 2)
    return ad

@pytest.fixture(scope="session")
def clones_basic(adata):
    return sl.pp.clones_adata(adata, obs_name="clone", min_size=30, fill_obs=None)

@pytest.fixture(scope="session")
def clones_with_fill(adata):
    return sl.pp.clones_adata(adata, obs_name="clone", min_size=30, fill_obs="cell_type")

@pytest.fixture(scope="session")
def clones_with_embed(clones_with_fill):
    cl = clones_with_fill.copy()
    cl.obsm["clone2vec"] = np.random.RandomState(1).rand(cl.n_obs, 2)
    return cl

# ===== clones_adata =====

def test_clones_adata_no_fill(clones_basic):
    cl = clones_basic
    assert cl.n_obs > 0
    assert "n_cells" in cl.obs
    assert "n_fates" in cl.obs
    assert "counts" in cl.layers
    assert "proportions" in cl.layers
    assert np.allclose(cl.X, cl.layers["proportions"]) 
    sums = np.asarray(cl.layers["counts"].sum(axis=1)).ravel()
    props_sum = np.asarray(cl.layers["proportions"].sum(axis=1)).ravel()
    assert np.all(props_sum[(sums > 0)] - 1 < 1e-6)

def test_clones_adata_with_fill(clones_with_fill):
    cl = clones_with_fill
    assert cl.uns.get("fill_obs") == "cell_type"
    assert len(cl.var_names) >= 1
    calculated_n_fates = (cl.layers["proportions"] > 0).sum(axis=1)
    if sp.issparse(calculated_n_fates):
        calculated_n_fates = np.array(calculated_n_fates).ravel()
    else:
        calculated_n_fates = np.array(calculated_n_fates).ravel()
    assert np.array_equal(cl.obs["n_fates"].values, calculated_n_fates.astype(int))

# ===== make_unique_clones =====

def test_make_unique_clones(adata):
    ad = adata.copy()
    ad.obs["clone_1"] = ad.obs["clone"].copy()
    ad.obs["clone_2"] = ad.obs["clone"].copy()
    adu = sl.pp.make_unique_clones(ad, ["clone_1", "clone_2"], final_obs_name="clone_combined")
    n_non_na = int(np.sum(ad.obs["clone"] != "NA"))
    n_na = int(np.sum(ad.obs["clone"] == "NA"))
    assert adu.n_obs == 2 * n_non_na + n_na
    sample = ad[ad.obs["clone"] != "NA"].obs.iloc[0]
    expected_label = f"clone_1_{sample['clone']}"
    assert expected_label in set(adu.obs["clone_combined"])

# ===== recalc_composition =====

def test_recalc_composition(adata, clones_with_fill):
    cl_new = sl.pp.recalc_composition(adata, clones_with_fill, fill_obs="cell_type", obs_name="clone")
    assert cl_new.uns.get("fill_obs") == "cell_type"
    counts = cl_new.layers["counts"]
    props = cl_new.layers["proportions"]
    sums = np.asarray(counts.sum(axis=1)).ravel()
    props_sum = np.asarray(props.sum(axis=1)).ravel()
    assert np.all(props_sum[(sums > 0)] - 1 < 1e-6)
    if sp.issparse(props):
        assert np.allclose(cl_new.X.toarray(), props.toarray())
    else:
        assert np.allclose(cl_new.X, props)

# ===== transfer_annotation =====

def test_transfer_annotation_clones_to_adata(adata, clones_with_fill):
    cl = clones_with_fill.copy()
    cl.obs["size_bin"] = pd.Categorical(np.where(cl.obs["n_cells"] >= cl.obs["n_cells"].median(), "large", "small"))
    sl.pp.transfer_annotation(adata, cl, annotation_obs_clones=["size_bin"], created_obs_name=["c2v_size"], obs_name="clone")
    assert "c2v_size" in adata.obs
    has_na = np.any(adata.obs["c2v_size"].astype(str).values == "NA")
    assert has_na

def test_transfer_annotation_adata_to_clones(adata, clones_with_fill):
    cl = clones_with_fill.copy()
    cl.obs["dummy"] = pd.Categorical(np.where(cl.obs["n_cells"] >= cl.obs["n_cells"].median(), "A", "B"))
    mapping = dict(zip(cl.obs_names, cl.obs["dummy"].astype(str)))
    ad = adata.copy()
    ad.obs["dummy_from_clones"] = ad.obs["clone"].map(lambda c: mapping.get(c, "NA"))
    sl.pp.transfer_annotation(ad, cl, annotation_obs_adata=["dummy_from_clones"], created_obs_name=["gex_dummy"], obs_name="clone")
    assert "gex_dummy" in cl.obs
    vals = cl.obs["gex_dummy"].astype(str).values
    assert set(np.unique(vals)) <= {"A", "B", "NA"}

# ===== transfer_expression =====

@pytest.mark.parametrize("agg_strategy", ["average", "sum"])
@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_transfer_expression_basic(agg_strategy, matrix_kind, adata_dense, adata_sparse, clones_with_fill):
    ad = adata_dense if matrix_kind == "dense" else adata_sparse
    cl = clones_with_fill
    cl_expr = sl.pp.transfer_expression(ad, cl, obs_name="clone", strategy=agg_strategy, layers_to_obsm=["counts", "proportions", "X"])
    assert cl_expr.n_obs == cl.n_obs
    assert len(cl_expr.var_names) == (ad.raw.var_names.size if (ad.raw is not None and cl_expr.var_names.equals(ad.raw.var_names)) else ad.var_names.size)
    assert "counts" in cl_expr.obsm
    assert "proportions" in cl_expr.obsm
    assert "X" in cl_expr.obsm

def test_transfer_expression_split_obs_dense(adata_dense, clones_with_fill):
    ad = adata_dense
    cl = clones_with_fill
    cl_expr = sl.pp.transfer_expression(ad, cl, obs_name="clone", split_obs="cell_type", strategy="average")
    groups = [g for g in pd.unique(ad.obs["cell_type"]) if pd.notna(g)]
    for g in groups:
        assert g in cl_expr.layers
    group = groups[0]
    ct_mask = (ad.obs["cell_type"].values == group)
    clone_counts = pd.Series(ad.obs.loc[ct_mask, "clone"]).value_counts()
    zero_clones = [c for c in cl.obs_names if clone_counts.get(c, 0) == 0]
    if len(zero_clones) > 0:
        idx = cl.obs_names.get_loc(zero_clones[0])
        layer = cl_expr.layers[group]
        row = layer[idx]
        row_vals = row.toarray().ravel() if sp.issparse(row) else np.array(row).ravel()
        assert np.all(np.isnan(row_vals))

def test_transfer_expression_use_raw_and_layer(adata_dense, clones_with_fill):
    ad = adata_dense.copy()
    ad.raw = ad
    ad.layers["custom"] = ad.X.copy()
    cl_expr_raw = sl.pp.transfer_expression(ad, clones_with_fill, obs_name="clone", use_raw=True)
    assert len(cl_expr_raw.var_names) == ad.raw.var_names.size
    cl_expr_layer = sl.pp.transfer_expression(ad, clones_with_fill, obs_name="clone", layer="custom")
    assert len(cl_expr_layer.var_names) == ad.var_names.size