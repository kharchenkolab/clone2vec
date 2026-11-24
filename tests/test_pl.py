import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pytest
import matplotlib.pyplot as plt
import clone2vec as c2v


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
def adata_with_umap_dense(adata_dense):
    ad = adata_dense.copy()
    rng = np.random.RandomState(0)
    ad.obsm["X_umap"] = rng.rand(ad.n_obs, 2)
    return ad

@pytest.fixture(scope="session")
def adata_with_umap_sparse(adata_sparse):
    ad = adata_sparse.copy()
    rng = np.random.RandomState(1)
    ad.obsm["X_umap"] = rng.rand(ad.n_obs, 2)
    return ad

@pytest.fixture(scope="session")
def clones(adata):
    # Build clones AnnData with n_cells populated for clone_size tests
    cl = c2v.pp.clones_adata(adata, obs_name="clone", min_size=3, fill_obs=None)
    return cl

# ===== group_scatter and group_kde =====

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_group_scatter_returns_figure(matrix_kind, adata_with_umap_dense, adata_with_umap_sparse):
    ad = adata_with_umap_dense if matrix_kind == "dense" else adata_with_umap_sparse
    assert "cell_type" in ad.obs, "Expected 'cell_type' in obs for plotting"
    group = pd.unique(ad.obs["cell_type"].astype(str))[0]
    fig = c2v.pl.group_scatter(ad, groupby="cell_type", groups=group, return_fig=True, basis="X_umap")
    assert isinstance(fig, plt.Figure)

@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"]) 
def test_group_kde_returns_figure(matrix_kind, adata_with_umap_dense, adata_with_umap_sparse):
    from collections import Counter

    ad = adata_with_umap_dense if matrix_kind == "dense" else adata_with_umap_sparse
    assert "cell_type" in ad.obs, "Expected 'cell_type' in obs for plotting"
    counts = Counter(ad.obs["cell_type"].astype(str))
    group = None
    for g, c in counts.items():
        if c >= 5:
            group = g
            break
    if group is None:
        group = list(counts.keys())[0]
    fig = c2v.pl.group_kde(ad, groupby="cell_type", groups=group, basis="X_umap", return_fig=True)
    assert isinstance(fig, plt.Figure)

# ===== embedding =====

def test_embedding_with_obsm(adata_with_umap_dense):
    ad = adata_with_umap_dense.copy()
    rng = np.random.RandomState(42)
    ad.obsm["X_test_emb"] = rng.rand(ad.n_obs, 3)
    ax = c2v.pl.embedding(ad, obsm_name="X_test_emb", obsm_component=1, basis="X_umap", show=False)
    # temporary column should be removed after plotting
    assert "X_test_emb_1" not in ad.obs.columns
    plt.close("all")


# ===== loss_history and clone_size =====

def test_loss_history_plots_line(clones):
    cl = clones.copy()
    cl.uns["clone2vec"] = {"loss_history": [1.0, 0.8, 0.6, 0.5]}
    fig = c2v.pl.loss_history(cl, uns_key="clone2vec", return_fig=True)
    assert isinstance(fig, plt.Figure)

def test_clone_size_returns_figure(clones):
    cl = clones.copy()
    assert "n_cells" in cl.obs
    fig = c2v.pl.clone_size(cl, return_fig=True)
    assert isinstance(fig, plt.Figure)

# ===== nesting_clones =====

def test_nesting_clones_runs_with_synthetic_labels(adata_dense):
    ad = adata_dense.copy()
    n = ad.n_obs
    rng = np.random.RandomState(2)
    early = np.where(rng.rand(n) > 0.5, "E1", "E2")
    late = np.where(rng.rand(n) > 0.6, "L1", "L2")
    ad.obs["early_barcodes"] = pd.Categorical(early)
    ad.obs["late_barcodes"] = pd.Categorical(late)
    c2v.pl.nesting_clones(ad, early_injection="early_barcodes", late_injection="late_barcodes", min_clone_size=5)
    assert True

# ===== volcano and shap_volcano =====

def test_volcano_basic():
    pvals = np.random.RandomState(3).rand(50)
    logfcs = np.random.RandomState(4).rand(50) * 2 - 1
    names = pd.Series([f"g{i}" for i in range(50)])
    fig = c2v.pl.volcano(pvals=pvals, logfcs=logfcs, names=names, return_fig=True)
    assert isinstance(fig, plt.Figure)

def test_shap_volcano_sparse_layer(adata_dense):
    ad = adata_dense.copy()
    ad = ad[:, :min(60, ad.n_vars)].copy()
    rng = np.random.RandomState(5)
    shap_layer = sp.csr_matrix(rng.rand(ad.n_obs, ad.n_vars))
    ad.layers["SHAP"] = shap_layer
    ad.varm["gex_r"] = pd.DataFrame({"SHAP": pd.Series(rng.rand(ad.n_vars), index=ad.var_names)})
    fig = c2v.pl.shap_volcano(ad, layer="SHAP", min_corr=0.2, return_fig=True)
    assert isinstance(fig, plt.Figure)

# ===== barplot and heatmap =====

@pytest.mark.parametrize("kind", ["h", "v"]) 
def test_barplot_basic(kind):
    s = pd.Series(np.linspace(0, 1, 10), index=[f"c{i}" for i in range(10)])
    fig = c2v.pl.barplot(s, kind=kind, return_fig=True, title="bar")
    assert isinstance(fig, plt.Figure)

def test_heatmap_basic():
    df = pd.DataFrame(np.random.RandomState(6).rand(8, 5), index=[f"r{i}" for i in range(8)], columns=[f"c{j}" for j in range(5)])
    fig = c2v.pl.heatmap(df, return_fig=True)
    assert isinstance(fig, plt.Figure)


# ===== pca_loadings =====

def test_pca_loadings(adata_dense):
    ad = adata_dense.copy()
    rng = np.random.RandomState(7)
    n_comps = 3
    ad.varm["sPCs"] = rng.randn(ad.n_vars, n_comps)
    fig = c2v.pl.pca_loadings(ad, key="sPCs", comp=0, n_genes=3, return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ===== pl.group_connectivity =====

def test_pl_group_connectivity(adata_with_umap_dense):
    ad = adata_with_umap_dense.copy()
    ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
    # Need neighbors for tl.group_connectivity
    if "connectivities" not in ad.obsp:
        sc.pp.pca(ad)
        sc.pp.neighbors(ad)
    c2v.tl.group_connectivity(ad, groupby="cell_type", graph_key="connectivities")
    fig = c2v.pl.group_connectivity(ad, basis="X_umap", return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ===== graph =====

def test_graph(adata_with_umap_dense):
    ad = adata_with_umap_dense.copy()
    if "connectivities" not in ad.obsp:
        sc.pp.pca(ad)
        sc.pp.neighbors(ad)
    fig = c2v.pl.graph(ad, graph_key="connectivities", basis="X_umap", return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ===== predictors_comparison =====

def test_predictors_comparison():
    rng = np.random.RandomState(8)
    n_vars = 30
    shapdata = sc.AnnData(
        X=rng.rand(10, n_vars),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    shapdata.varm["signed_mean_shap"] = pd.DataFrame(
        rng.randn(n_vars, 2), index=shapdata.var_names, columns=["groupA", "groupB"],
    )
    fig = c2v.pl.predictors_comparison(
        shapdata, group1="groupA", group2="groupB",
        n_genes=5, return_fig=True,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ===== catboost_performance =====

def test_catboost_performance():
    rng = np.random.RandomState(9)
    n_obs = 20
    fates = ["fateA", "fateB"]
    shapdata = sc.AnnData(
        X=rng.rand(n_obs, 5),
        obs=pd.DataFrame({"validation": [True] * 10 + [False] * 10}),
    )
    shapdata.obsm["X_comp"] = pd.DataFrame(
        rng.rand(n_obs, 2), columns=fates, index=shapdata.obs_names,
    )
    shapdata.obsm["X_comp:predicted"] = pd.DataFrame(
        rng.rand(n_obs, 2), columns=fates, index=shapdata.obs_names,
    )
    shapdata.uns["catboost_info"] = {"obsm_key": "X_comp", "fates_used": fates}
    fig = c2v.pl.catboost_performance(shapdata, var_names=["fateA"], return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ===== embedding_axis =====

def test_embedding_axis():
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2], [0, 1, 2])
    c2v.pl.embedding_axis(ax, label="UMAP")
    # No error means success
    plt.close("all")


# ===== small_cbar =====

def test_small_cbar():
    fig, ax = plt.subplots()
    sc_plot = ax.scatter([0, 1, 2], [0, 1, 2], c=[0.1, 0.5, 0.9], cmap="viridis")
    fig.colorbar(sc_plot, ax=ax)
    c2v.pl.small_cbar(ax)
    # No error means success
    plt.close("all")


# ===== scaled_dotplot =====

def test_scaled_dotplot(adata_dense):
    ad = adata_dense.copy()
    ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
    genes = list(ad.var_names[:3])
    result = c2v.pl.scaled_dotplot(ad, groupby="cell_type", var_names=genes)
    assert result is not None
    plt.close("all")


# ===== scatter2vars =====

def test_scatter2vars(adata_with_umap_dense):
    ad = adata_with_umap_dense.copy()
    genes = list(ad.var_names[:2])
    fig = c2v.pl.scatter2vars(
        ad, var1=genes[0], var2=genes[1],
        basis="X_umap", return_fig=True, use_raw=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ===== fancy_legend =====

def test_fancy_legend():
    textalloc = pytest.importorskip("textalloc", reason="textalloc not installed; skipping fancy_legend test")
    fig, ax = plt.subplots()
    for i, (label, color) in enumerate([("A", "red"), ("B", "blue"), ("C", "green")]):
        ax.scatter(np.random.rand(10), np.random.rand(10), c=color, label=label)
    ax.legend()
    c2v.pl.fancy_legend(ax, center_loc=True)
    plt.close("all")


# ===== clones2cells =====

def _run_clones2cells(ad, cl, monkeypatch, **kwargs):
    pytest.importorskip("ipywidgets", reason="ipywidgets not installed; skipping clones2cells test")
    js = pytest.importorskip("jscatter", reason="jscatter not installed; skipping clones2cells test")
    composed = []
    compose = js.compose

    def capture(items, **kw):
        composed.extend(items)
        return compose(items, **kw)

    monkeypatch.setattr(js, "compose", capture)
    widget = c2v.pl.clones2cells(ad, cl, obs_name="clone", show=False, **kwargs)
    (scatter_cells, _), (scatter_clones, _) = composed
    return widget, scatter_cells, scatter_clones

@pytest.fixture
def c2c_data(adata_with_umap_dense, clones):
    cl = clones.copy()
    cl.obsm["X_umap"] = np.random.RandomState(2).rand(cl.n_obs, 2)
    return adata_with_umap_dense.copy(), cl

@pytest.mark.parametrize("mode", ["highlight", "filter"])
@pytest.mark.parametrize("cells_color", [None, "cell_type", "gene"])
def test_clones2cells(mode, cells_color, c2c_data, monkeypatch):
    ad, cl = c2c_data
    if cells_color == "gene":
        cells_color = ad.var_names[0]
    widget, scatter_cells, scatter_clones = _run_clones2cells(
        ad, cl, monkeypatch, cells_color=cells_color, mode=mode, norm_cells=(0, "p99"),
    )
    assert widget is not None

    scatter_clones.widget.selection = np.array([0, 1])
    assert set(cl.uns["clones2cells"]["last_selection"]) == set(cl.obs_names[:2])
    scatter_clones.widget.selection = np.array([], dtype=int)

    # the first n_obs points are all cells in both modes
    scatter_cells.widget.selection = np.arange(ad.n_obs)
    stat = ad.uns["clones2cells"]["clone_stat"]
    assert len(ad.uns["clones2cells"]["last_selection"]) == ad.n_obs
    assert np.allclose(stat.loc[cl.obs_names].values, 100)
    scatter_cells.widget.selection = np.array([], dtype=int)

def test_clones2cells_count_and_clone_color(c2c_data, monkeypatch):
    ad, cl = c2c_data
    _, scatter_cells, _ = _run_clones2cells(ad, cl, monkeypatch, clones_color="n_cells", selection_stat="count")
    scatter_cells.widget.selection = np.arange(ad.n_obs)
    stat = ad.uns["clones2cells"]["clone_stat"]
    sizes = ad.obs["clone"].astype(str).value_counts()
    assert np.allclose(stat.loc[cl.obs_names].values, sizes.loc[cl.obs_names].values)

def test_clones2cells_errors(c2c_data):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("jscatter")
    ad, cl = c2c_data
    with pytest.raises(ValueError):
        c2v.pl.clones2cells(ad, cl, obs_name="clone", mode="other", show=False)
    with pytest.raises(KeyError):
        c2v.pl.clones2cells(ad, cl, obs_name="absent", show=False)
    with pytest.raises(KeyError):
        c2v.pl.clones2cells(ad, cl, obs_name="clone", cells_color="absent", show=False)

def test_clones2cells_helpers():
    from clone2vec import clones2cells as c2c

    x = np.arange(101, dtype=float)
    assert c2c._resolve_norm(None, x) == (0, 100)
    assert c2c._resolve_norm((None, "p50"), x) == (0, 50)
    assert c2c._resolve_norm(None, np.ones(3)) == (1, 2)
    with pytest.raises(ValueError):
        c2c._resolve_norm((10, 0), x)

    assert c2c._bin(np.array([-1, 0, 0.5, 1, 2]), 0, 1, n=4).tolist() == [0, 0, 2, 3, 3]

    labels = pd.Series(pd.Categorical(["a", "b", None, "c"]))
    assert c2c._clone_codes(labels, pd.Index(["b", "a"])).tolist() == [1, 0, -1, -1]
    assert c2c._clone_codes(pd.Series(["a", None, "b"]), pd.Index(["b", "a"])).tolist() == [1, -1, 0]

    ad = sc.AnnData(np.zeros((3, 1)))
    ad.obs["g"] = pd.Categorical(["x", "y", "x"])
    assert c2c._palette_colors(ad, "g", ["x", "y"], {"x": "red", "y": "blue"}) == ["#ff0000", "#0000ff"]
    assert c2c._palette_colors(ad, "g", ["x", "y"], "black") == ["#000000", "#000000"]
