import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from scipy import stats
import clone2vec as c2v
from clone2vec.associations import _masked_score

N_PERM = 10
Z_DIM = 4
N_GENES = 40


# ===== Fixtures =====

@pytest.fixture(scope="session")
def adata():
    ad = sc.read_h5ad("tests/Weinreb_subsampled.h5ad")
    return sc.pp.subsample(ad, fraction=0.4, random_state=0, copy=True)

@pytest.fixture(scope="session")
def mask(adata):
    return (adata.obs["cell_type"] == "Undiff").values

@pytest.fixture(scope="session")
def clones(adata, mask):
    cl = c2v.pp.clones_adata(adata, obs_name="clone", min_size=10)
    c2v.tl.clonal_nn(adata, cl, k=10, use_rep="X_pca")
    c2v.tl.clone2vec(cl, z_dim=Z_DIM, max_iter=5, device="cpu", progress_bar=False)
    c2v.tl.permute_embeddings(
        adata, cl, n_permutations=N_PERM, progress_bar=False,
        clone2vec_kwargs={"max_iter": 5, "device": "cpu"},
    )
    fit = c2v.tools.clone2vec
    with pytest.MonkeyPatch.context() as mp:
        # mask_heterogeneity doesn't expose clone2vec arguments, the default 500 epochs are too slow for tests
        mp.setattr(c2v.tools, "clone2vec", lambda *args, **kwargs: fit(*args, max_iter=5, device="cpu", **kwargs))
        c2v.tl.mask_heterogeneity(adata, cl, mask=mask)
    return cl

@pytest.fixture(scope="session")
def clones_gex(adata, clones):
    cl = c2v.pp.transfer_expression(adata, clones)
    genes = cl.var_names[np.argsort(-cl.X.toarray().var(axis=0))[:N_GENES]]
    cl = cl[:, genes].copy()
    cl.obs["group"] = np.where(cl.obsm["clone2vec_mask"][:, 0] > np.median(cl.obsm["clone2vec_mask"][:, 0]), "A", "B")
    c2v.tl.graph_associations(cl, n_neighbors=5, n_pcs=5)
    c2v.tl.permuted_graph_associations(adata, cl, progress_bar=False)
    return cl


# ===== tl.permute_embeddings =====

def test_permute_embeddings_outputs(adata, clones):
    assert clones.obsm["clonal_permutations"].shape == (clones.n_obs, Z_DIM, N_PERM)
    params = clones.uns["clonal_permutations"]
    assert params["use_rep"] == "clone2vec"
    assert params["method"] == "multinomial"
    assert params["codes"].shape == (adata.n_obs, N_PERM)


def test_permute_embeddings_preserves_clone_sizes(adata, clones):
    params = clones.uns["clonal_permutations"]
    categories = np.asarray(params["categories"])
    observed = adata.obs["clone"].astype(str).value_counts().sort_index()
    for i in range(N_PERM):
        permuted = pd.Series(categories[params["codes"][:, i]]).value_counts().sort_index()
        pd.testing.assert_series_equal(permuted, observed, check_names=False)
    assert not (params["codes"][:, 0] == params["codes"][:, 1]).all()


def test_permute_embeddings_requires_embedding(adata):
    cl = c2v.pp.clones_adata(adata, obs_name="clone", min_size=10)
    c2v.tl.clonal_nn(adata, cl, k=10, use_rep="X_pca")
    with pytest.raises(ValueError):
        c2v.tl.permute_embeddings(adata, cl, n_permutations=2, progress_bar=False)


# ===== utils.unpack_permutations =====

def test_unpack_permutations(clones):
    unpacked = c2v.utils.unpack_permutations(clones)
    assert len(unpacked) == N_PERM
    for i, ad in enumerate(unpacked):
        assert list(ad.obs_names) == list(clones.obs_names)
        np.testing.assert_array_equal(ad.obsm["clone2vec"], clones.obsm["clonal_permutations"][..., i])
        assert ad.uns["permutation"] == i


# ===== tl.spectral_test =====

def test_spectral_test_permutations(clones):
    cl = clones.copy()
    c2v.tl.spectral_test(cl, progress_bar=False)
    res = cl.uns["spectral_test"]
    assert res["flavor"] == "permutations"
    assert res["use_rep"] == "clone2vec"
    null = res["null_distribution"]
    assert null.shape == (N_PERM,)
    expected_p = (1 + np.sum(null[np.isfinite(null)] >= res["stat"])) / (np.isfinite(null).sum() + 1)
    assert res["p_value"] == pytest.approx(expected_p)


def test_spectral_test_array_flavors():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((200, 10))
    spiked = noise + 3 * np.outer(rng.standard_normal(200), rng.standard_normal(10))
    for flavor in ("asymptotic", "synthetic"):
        res_noise = c2v.tl.spectral_test(noise, flavor=flavor, n_simulations=200, progress_bar=False)
        res_spiked = c2v.tl.spectral_test(spiked, flavor=flavor, n_simulations=200, progress_bar=False)
        assert 0 <= res_spiked["p_value"] < 0.05 < res_noise["p_value"] <= 1
    with pytest.raises(ValueError):
        c2v.tl.spectral_test(noise, flavor="permutations")


def test_spectral_test_requires_permutations(adata):
    cl = c2v.pp.clones_adata(adata, obs_name="clone", min_size=10)
    with pytest.raises(KeyError):
        c2v.tl.spectral_test(cl)


# ===== tl.mask_heterogeneity =====

def test_mask_heterogeneity(clones):
    assert clones.obsm["clone2vec_mask"].shape == (clones.n_obs, Z_DIM)
    assert not np.allclose(clones.obsm["clone2vec_mask"], clones.obsm["clone2vec"])
    assert {"clonal_nn", "clone2vec"} <= set(clones.uns["mask"])


def test_mask_heterogeneity_requires_embedding(adata, mask):
    cl = c2v.pp.clones_adata(adata, obs_name="clone", min_size=10)
    with pytest.raises(ValueError):
        c2v.tl.mask_heterogeneity(adata, cl, mask=mask)


# ===== tl.graph_associations / tl.permuted_graph_associations =====

def test_graph_associations(clones_gex):
    for col in ("morans_i", "morans_i_pval", "morans_i_p_adj"):
        assert col in clones_gex.var
    assert clones_gex.obsm["X_sPCA"].shape == (clones_gex.n_obs, 5)
    assert clones_gex.varm["sPCs"].shape == (N_GENES, 5)
    assert clones_gex.uns["graph_associations"]["use_rep"] == "clone2vec"
    evals = clones_gex.uns["spca"]["evals"]
    assert np.all(np.diff(evals) <= 1e-8)


def test_graph_associations_requires_transfer(clones):
    with pytest.raises(ValueError):
        c2v.tl.graph_associations(clones.copy())


def test_permuted_graph_associations(clones_gex):
    var = clones_gex.var
    for col in ("null_mean", "null_std", "perm_pval", "perm_p_adj", "z_score", "z_pval"):
        assert col in var
    pvals = var["perm_pval"].dropna()
    assert ((pvals >= 1 / (N_PERM + 1)) & (pvals <= 1)).all()
    res = clones_gex.uns["permuted_graph_associations"]
    null = res["null_distribution"]
    assert null.shape == (N_GENES, N_PERM)
    expected = (1 + (null.values >= var["morans_i"].values[:, None]).sum(1)) / (N_PERM + 1)
    np.testing.assert_allclose(var["perm_pval"].values, expected)
    assert res["spca"].shape[0] == 5


def test_permuted_graph_associations_requires_graph_associations(adata, clones):
    cl = c2v.pp.transfer_expression(adata, clones)
    with pytest.raises(ValueError):
        c2v.tl.permuted_graph_associations(adata, cl, progress_bar=False)


# ===== tl.masked_markers =====

@pytest.mark.parametrize("method", ["regression", "spearman"])
def test_masked_markers(clones_gex, method):
    res = c2v.tl.masked_markers(clones_gex, groupby="group", reference="A", test="B", method=method)
    assert list(res.columns) == ["scores", "logfoldchanges", "pvals", "pvals_adj"]
    assert res["scores"].is_monotonic_decreasing
    assert res["pvals"].between(0, 1).all()
    assert (res["pvals_adj"] >= res["pvals"] - 1e-12).all()

    d, keep, _, _ = _masked_score(clones_gex, "group", "A", "B", "clone2vec_mask", 10, "exact")
    gene = res.index[0]
    y = clones_gex[keep, gene].X.toarray().ravel()
    if method == "regression":
        ref = stats.linregress(d, y)
        assert res.loc[gene, "pvals"] == pytest.approx(ref.pvalue)
        assert res.loc[gene, "logfoldchanges"] == pytest.approx(ref.slope / np.log(2))
    else:
        ref = stats.spearmanr(d, y)
        assert res.loc[gene, "scores"] == pytest.approx(ref.statistic)
        assert res.loc[gene, "pvals"] == pytest.approx(ref.pvalue)


def test_masked_markers_overlapping_groups(clones_gex):
    with pytest.raises(ValueError):
        c2v.tl.masked_markers(clones_gex, groupby="group", reference="A", test=["A", "B"])


# ===== plotting =====

def test_pl_spectral_test(clones):
    cl = clones.copy()
    c2v.tl.spectral_test(cl, progress_bar=False)
    fig = c2v.pl.spectral_test(cl, return_fig=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_pl_moran_test(clones_gex):
    fig = c2v.pl.moran_test(clones_gex, gene=clones_gex.var_names[0], return_fig=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


@pytest.mark.parametrize("against", ["same", "first"])
def test_pl_spca_test(clones_gex, against):
    fig = c2v.pl.spca_test(clones_gex, component=2, against=against, return_fig=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_pl_de_trend(clones_gex):
    fig = c2v.pl.de_trend(
        clones_gex, genes=list(clones_gex.var_names[:2]), groupby="group",
        reference="A", test="B", return_fig=True, show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")
