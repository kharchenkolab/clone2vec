import numpy as np
import pandas as pd
import scanpy as sc
import pytest

pytest.importorskip("rpy2", reason="rpy2 not installed; skipping Seurat tests")
from rpy2.robjects.packages import isinstalled
from clone2vec import seurat

needs_matrix = pytest.mark.skipif(not isinstalled("Matrix"), reason="R package Matrix not installed")
needs_seurat = pytest.mark.skipif(
    not (isinstalled("Seurat") and isinstalled("SeuratObject")),
    reason="R packages Seurat/SeuratObject not installed",
)


def _small_adata(n_cells=6, n_genes=4, seed=0):
    rng = np.random.RandomState(seed)
    ad = sc.AnnData(rng.poisson(1, (n_cells, n_genes)).astype(float))
    ad.obs_names = [f"cell{i}" for i in range(n_cells)]
    ad.var_names = [f"gene{i}" for i in range(n_genes)]
    return ad


@needs_matrix
@pytest.mark.parametrize("sparse", [True, False])
def test_rmatrix_round_trip(sparse):
    ad = _small_adata()
    r_mat = seurat._get_RMatrix(ad, sparse=sparse)
    back = seurat._dgCMatrix_to_sp(r_mat).T.toarray()
    assert np.allclose(back, ad.X)


@needs_matrix
def test_rmatrix_dummy_and_raw():
    ad = _small_adata()
    ad.raw = ad.copy()
    ad.X = np.zeros_like(ad.X)
    assert seurat._dgCMatrix_to_sp(seurat._get_RMatrix(ad, sparse=True, dummy=True)).nnz == 0
    raw = seurat._dgCMatrix_to_np(seurat._get_RMatrix(ad, sparse=True, use_raw=True)).T
    assert np.allclose(raw, ad.raw.X)


@needs_seurat
def test_read_rds(tmp_path):
    import rpy2.robjects as ro
    pytest.importorskip("mudata")

    ad = _small_adata(n_cells=30, n_genes=20)
    ro.globalenv["counts"] = seurat._get_RMatrix(ad, sparse=True)
    path = str(tmp_path / "obj.rds")
    ro.r(f"""
        suppressMessages(library(Seurat))
        obj <- CreateSeuratObject(counts = counts)
        obj <- NormalizeData(obj, verbose = FALSE)
        saveRDS(obj, "{path}")
    """)

    mdata = seurat.read(path)
    assert "RNA" in mdata.mod
    rna = mdata.mod["RNA"]
    assert rna.shape == (30, 20)
    assert rna.raw is not None
    assert np.allclose(rna.raw[:, ad.var_names].X.toarray(), ad[rna.obs_names].X)
    assert "nCount_RNA" in mdata.obs


@needs_seurat
def test_integrate_data_only_anchors():
    rng = np.random.RandomState(0)
    n_cells, n_genes = 120, 40
    ad = sc.AnnData(rng.poisson(3, (n_cells, n_genes)).astype(float))
    ad.obs_names = [f"cell{i}" for i in range(n_cells)]
    ad.var_names = [f"gene_{i}" for i in range(n_genes)]
    ad.obs["batch"] = pd.Categorical(["a"] * (n_cells // 2) + ["b"] * (n_cells // 2))
    sc.pp.log1p(ad)
    ad.raw = ad.copy()
    sc.pp.scale(ad)

    seurat.integrate_data(
        ad, batch_key="batch", n_comps=10, k_anchor=5, k_filter=30, k_score=10,
        max_features=40, only_anchors=True, progress_bar=False,
    )

    G = ad.obsp["seurat_mnn"]
    assert G.shape == (n_cells, n_cells)
    assert G.nnz > 0
    assert abs(G - G.T).sum() == 0
    assert ad.uns["seurat_integration"]["mode"] == "anchors"
    assert ad.uns["seurat_integration"]["batch_key"] == "batch"
    assert list(ad.var_names) == [f"gene_{i}" for i in range(n_genes)]
