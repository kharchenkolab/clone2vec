from __future__ import annotations

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd

import sys
import os

from typing import Literal
from tqdm import tqdm

logg = sc.logging

__all__ = [
    "integrate_data",
    "read",
]

def __dir__():
    return sorted(__all__)

def _dgCMatrix_to_sp(
    mat,
    output_type: Literal["csc", "csr"] = "csc"
) -> sp.csr_matrix:
    try:
        p = np.array(mat.slots["p"])
        i = np.array(mat.slots["i"])
        x = np.array(mat.slots["x"])
        dims = np.array(mat.slots["Dim"])
        scipy_csc = sp.csc_matrix((x, i, p), shape=(dims[0], dims[1]))
        if output_type == "csr":
            return scipy_csc.tocsr()
        else:
            return scipy_csc
    except Exception:
        if output_type == "csr":
            return sp.csr_matrix(np.array(mat))
        else:
            return sp.csc_matrix(np.array(mat))

def _dgCMatrix_to_np(Matrix):
    data = np.array(Matrix.do_slot("x"))
    indices = np.array(Matrix.do_slot("i"))
    indptr = np.array(Matrix.do_slot("p"))
    
    shape = (Matrix.do_slot("Dim")[0], Matrix.do_slot("Dim")[1])
    csc_mat = sp.csc_matrix((data, indices, indptr), shape=shape)
    
    return csc_mat.toarray()

def _get_RMatrix(
    adata: sc.AnnData,
    dummy: bool = False,
    sparse: bool | None = None,
    use_raw: bool = False,
) -> ro.rinterface.Sexp:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    Matrix = importr("Matrix")

    if use_raw:
        X = adata.raw[:, adata.var_names].X
    else:
        X = adata.X

    X = X.T
    n_genes, n_cells = X.shape
    dim = ro.IntVector([n_genes, n_cells])
    dimnames = ro.ListVector({
        "rownames": ro.StrVector(adata.var_names),
        "colnames": ro.StrVector(adata.obs_names),
    })

    if sparse is None:
        sparse = sp.issparse(X)

    if dummy:
        if sparse:
            return Matrix.sparseMatrix(
                i = ro.IntVector([]),
                p = ro.IntVector([0] * (n_cells + 1)),
                x = ro.FloatVector([]),
                dim = dim,
                dimnames = dimnames,
            )
        else:
            return ro.r.matrix(
                ro.FloatVector(np.zeros(n_genes * n_cells)),
                nrow=n_genes, ncol=n_cells,
                byrow=False, dimnames=dimnames,
            )

    if sparse:
        X = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
        return Matrix.sparseMatrix(
            i = ro.IntVector((X.indices + 1).tolist()),
            p = ro.IntVector(X.indptr),
            x = ro.FloatVector(X.data),
            dim = dim,
            dimnames = dimnames,
        )

    data_vec = ro.FloatVector(np.asarray(X, dtype=float).flatten(order="F"))
    return ro.r.matrix(
        data_vec,
        nrow=n_genes,
        ncol=n_cells,
        byrow=False,
        dimnames=dimnames,
    )

def read(
    path: str | os.PathLike,
    qs: bool = False,
    assays: str | list[str] | None = None,
    use_slots: Literal["counts", "data"] | list[Literal["counts", "data"]] = ["counts", "data"],
    graphs: bool = True,
    dimreds: bool = True,
) -> "mudata.MuData":
    """
    Reads .rds-file or .qsave-file with Seurat object and returns AnnData or MuData object.

    Parameters
    ----------
    path: str | os.PathLike
        Path to the file.
    qs: bool, optional
        Whether to use qs package to read the file. Default is False.
    assays: str | list[str] | None, optional
        Assays to process. If None, all assays are processed.
    use_slots: Literal["counts", "data"] | list[Literal["counts", "data"]], optional
        Slots to use for counts. Can be "counts" or "data". Default is ["counts", "data"].
    graphs: bool, optional
        Whether to process graphs. Default is True.
    dimreds: bool, optional
        Whether to process dimensionality reductions. Default is True.

    Returns
    -------
    mudata.MuData
        MuData object.
    """
    logg.warning("This is an rpy2-wrapper around Seurat. This function is very RAM intensive and has a poor garbage collection.")

    import gc
    import rpy2
    import rpy2.robjects as ro
    import mudata

    mudata.set_options(pull_on_update=False)

    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    import rpy2.rinterface_lib.callbacks as callbacks
    from rpy2.robjects import pandas2ri

    class _RCallbacks():
        def __init__(self, callback):
            self.callback = callback
        def __enter__(self):
            self._print = callbacks.consolewrite_print
            self._warnerror = callbacks.consolewrite_warnerror
            callbacks.consolewrite_print = (lambda x: None)
            callbacks.consolewrite_warnerror = self.callback
            return self
        def __exit__(self, exc_type, exc, tb):
            callbacks.consolewrite_print = self._print
            callbacks.consolewrite_warnerror = self._warnerror

    start = logg.info(f"creating MuData object from Seurat object")
    if isinstance(use_slots, str):
        use_slots = [use_slots]
    
    try:
        with ro.local_context():
            with _RCallbacks(lambda x: None):
                Seurat = importr("Seurat")
                base = importr("base")

                logg.info("    reading Seurat object")
                if qs:
                    qs = importr("qs")
                    seurat_object = qs.qread(path)
                else:
                    seurat_object = base.readRDS(path)
                
                logg.info("    getting .obs")
                with localconverter(ro.default_converter + pandas2ri.converter):
                    obs = seurat_object.slots["meta.data"]
                    obs = pd.DataFrame(obs)
                obs_names = obs.index

                available_assays = list(base.names(seurat_object.slots["assays"]))
                if assays is None:
                    assays_to_process = available_assays
                elif isinstance(assays, str):
                    assays_to_process = [assays]
                else:
                    assays_to_process = assays
                valid_assays = [a for a in assays_to_process if a in available_assays]
                if len(valid_assays) == 0:
                    raise ValueError(f"no valid assays found, available: {available_assays}")
                if len(valid_assays) < len(assays_to_process):
                    absent_assays = [a for a in assays_to_process if not (a in available_assays)]
                    logg.warning(f"following assays are not available: {absent_assays}")

                adatas = {}
                for assay_name in valid_assays:
                    logg.info(f"    processing assay {assay_name}")

                    assay_obj = seurat_object.slots["assays"].rx2(assay_name)
                    assay_class = tuple(assay_obj.rclass)[0]

                    collected_matrices = {}
                    for slot_type in use_slots:
                        mat = None
                        try:
                            if assay_class == "Assay5":
                                layers = assay_obj.slots["layers"]
                                layer_names = list(base.names(layers))
                                if slot_type in layer_names:
                                    mat = layers.rx2(slot_type)
                            else:
                                if slot_type in assay_obj.slots.keys():
                                    mat = assay_obj.slots[slot_type]
                        except Exception:
                            pass

                        if mat is not None and not isinstance(mat, rpy2.rinterface_lib.sexp.NULLType):
                            collected_matrices[slot_type] = mat

                    if len(collected_matrices) == 0:
                        logg.warning(f"no matrices found for assay {assay_name}")
                        continue

                    adata_structure = {
                        "X": None,
                        "raw.X": None,
                    }

                    if len(collected_matrices) == 1:
                        if len(use_slots) == 2:
                            logg.warning(f"only one slot found for assay {assay_name}, using {slot_type} as .X")

                        slot_type = list(collected_matrices.keys())[0]
                        r_X = collected_matrices[slot_type]
                        X = _dgCMatrix_to_sp(r_X).T
                        adata_structure["X"] = slot_type

                        logg.hint(f"using slot '{slot_type}' as .X")
                    else:
                        r_X = collected_matrices["data"]
                        X = _dgCMatrix_to_sp(r_X).T
                        adata_structure["X"] = "data"

                        r_Xraw = collected_matrices["counts"]
                        Xraw = _dgCMatrix_to_sp(r_Xraw).T
                        adata_structure["raw.X"] = "counts"

                        logg.hint("using slot 'data' as .X and slot 'counts' as .raw.X")

                    gene_names = {}
                    if assay_class == "Assay5":
                        features = pd.DataFrame(
                            np.array(assay_obj.slots["features"]),
                            index=np.array(base.rownames(assay_obj.slots["features"])),
                            columns=np.array(base.colnames(assay_obj.slots["features"])),
                        ).astype(bool)

                        gene_names["X"] = features.index[features[adata_structure["X"]]]
                        if adata_structure["raw.X"] is not None:
                            gene_names["raw.X"] = features.index[features[adata_structure["raw.X"]]]
                    else:
                        gene_names["X"] = base.rownames(assay_obj.slots[adata_structure["X"]])
                        if adata_structure["raw.X"] is not None:
                            gene_names["raw.X"] = base.rownames(assay_obj.slots[adata_structure["raw.X"]])

                    adatas[assay_name] = sc.AnnData(
                        X=X,
                        obs=pd.DataFrame(index=obs_names),
                        var=pd.DataFrame(index=gene_names["X"]),
                    )

                    if adata_structure["raw.X"] is not None:
                        adatas[assay_name].raw = sc.AnnData(
                            X=Xraw,
                            obs=pd.DataFrame(index=obs_names),
                            var=pd.DataFrame(index=gene_names["raw.X"]),
                        )
                    
                mdata = mudata.MuData(adatas)
                mdata.obs = obs

                if dimreds:
                    logg.info("    collecting dimensionality reductions")
                    if "reductions" in seurat_object.slots.keys():
                        reductions = seurat_object.slots["reductions"]
                        if len(reductions) > 0:
                            for red_name in base.names(reductions):
                                red_obj = reductions.rx2(red_name)
                                emb = np.array(red_obj.slots["cell.embeddings"])
                                mdata.obsm[red_name] = emb
                                mdata.uns[red_name] = {
                                    "assay.used": np.array(red_obj.slots["assay.used"]),
                                }

                if graphs:
                    logg.info("    collecting graphs")
                    if "graphs" in seurat_object.slots.keys():
                        graphs = seurat_object.slots["graphs"]
                        if len(graphs) > 0:
                            for graph_name in base.names(graphs):
                                graph_obj = graphs.rx2(graph_name)
                                mdata.obsp[graph_name] = _dgCMatrix_to_sp(graph_obj, output_type="csr")

                if "commands" in seurat_object.slots.keys():
                    mdata.uns["seurat_commands"] = []
                    for command in seurat_object.slots["commands"]:
                        mdata.uns["seurat_commands"].append("\n".join(np.array(command.slots["call.string"])))

                # Garbage collection, maybe it will help
                gc.collect()
                del seurat_object
                ro.r("gc(verbose = FALSE)")
                gc.collect()

                lines = [
                    "created MuData with",
                    f"     .mod with different assays",
                ]
                if len(mdata.obsp.keys()) > 0:
                    lines.append(f"     .obsp with different graphs")
                if len(mdata.obsm.keys()) > 0:
                    lines.append(f"     .obsm with different dimensionality reductions")

                logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
                return mdata
    finally:
        gc.collect()
        with _RCallbacks(lambda x: None):
            ro.r("gc(verbose = FALSE)")
        gc.collect()


def integrate_data(
    adata: sc.AnnData,
    batch_key: str,
    method: Literal["cca", "rpca"] = "cca",
    layer_name: str = "integrated",
    scale_integrated: bool = True,
    scale_max_value: float = 10.,
    l2_norm: bool = True,
    n_comps: int = 30,
    k_anchor: int = 5,
    k_filter: int = 200,
    k_score: int = 30,
    max_features: int = 200,
    n_trees: int = 50,
    eps: float = 0,
    k_weight: int = 100,
    progress_bar: bool = True,
    only_anchors: bool = False,
    mnn_graph_key: str = "seurat_mnn",
    uns_key: str = "seurat_integration",
) -> ro.rinterface.Sexp:
    """
    Performs single-cell data integration via Seurat's CCA or RPCA approach.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Key in `adata.obs` that contains batch information.
    method
        Integration method to use. Either "cca" or "rpca".
    layer_name
        Name of the layer to store the integrated expression matrix.
    scale_integrated
        Whether to scale the integrated expression matrix.
    scale_max_value
        Maximum value to scale the integrated expression matrix.
    l2_norm
        Perform L2 normalization on the CCA cell embeddings after dimensional reduction.
    n_comps
        Which dimensions to use from the CCA to specify the neighbor search space.
    k_anchor
        How many neighbors (k) to use when picking anchors.
    k_filter
        How many neighbors (k) to use when filtering anchors.
    k_score
        How many neighbors (k) to use when scoring anchors.
    max_features
        The maximum number of features to use when specifying the neighborhood search space in the anchor filtering.
    n_trees
        More trees gives higher precision when using annoy approximate nearest neighbor search.
    eps
        Error bound on the neighbor finding algorithm.
    k_weight
        Number of neighbors to consider when weighting anchors.
    progress_bar
        Whether to show a progress bar.
    return_anchors
        Whether to return the anchors.
        
    Returns
    -------
    anchors
        Seurat anchors object.
    """
    logg.warning("This is an rpy2-wrapper around Seurat. This function is very RAM intensive and has a poor garbage collection.")
    
    import gc
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as callbacks
    from rpy2.robjects.packages import importr
    from contextlib import nullcontext

    class _RCallbacks():
        def __init__(self, callback):
            self.callback = callback
        def __enter__(self):
            self._print = callbacks.consolewrite_print
            self._warnerror = callbacks.consolewrite_warnerror
            callbacks.consolewrite_print = (lambda x: None)
            callbacks.consolewrite_warnerror = self.callback
            return self
        def __exit__(self, exc_type, exc, tb):
            callbacks.consolewrite_print = self._print
            callbacks.consolewrite_warnerror = self._warnerror

    try:
        with ro.local_context():
            with _RCallbacks(lambda x: None):
                Seurat = importr("Seurat")
                SeuratObject = importr("SeuratObject")

            try:
                batch = ro.StrVector(adata.obs[batch_key].astype(str).values)
                batch.names = ro.StrVector(adata.obs_names)
            except KeyError:
                raise ValueError(f"Batch key {batch_key} not found in adata.obs.")

            if sp.issparse(adata.X):
                raise ValueError("adata.X must be dense scaled matrix.")

            if adata.raw is None:
                raise ValueError("adata.raw must contain log1p-transformed counts.")

            start = logg.info(f"integrating data with Seurat {method.upper()} approach")

            # Avoiding conflicts with Seurat's convention of using "-" instead of "_"
            var_names = adata.var_names
            underscores = sum(["_" in i for i in var_names]) > 0
            if underscores:
                raw_var_names = adata.raw.var_names
                var_names_mod = [i.replace("_", "-") for i in var_names]
                raw_var_names_mod = [i.replace("_", "-") for i in adata.raw.var_names]
                adata.var_names = var_names_mod
                adata.raw.var.index = raw_var_names_mod
                
            seurat_obj = SeuratObject.CreateSeuratObject(
                counts=_get_RMatrix(adata, sparse=True, dummy=True),
                project="AnnData",
            )

            seurat_obj = SeuratObject.SetAssayData(
                object = seurat_obj,
                slot = "data", 
                new_data = _get_RMatrix(adata, sparse=True, use_raw=True, dummy=False),
            )

            seurat_obj = SeuratObject.SetAssayData(
                object = seurat_obj,
                slot = "scale.data", 
                new_data = _get_RMatrix(adata, sparse=False, use_raw=False, dummy=False),
            )

            seurat_obj = SeuratObject.AddMetaData(
                seurat_obj,
                metadata=batch,
                col_name="batch",
            )

            seurat_obj_list = Seurat.SplitObject(seurat_obj, split_by="batch")
            features = ro.StrVector(adata.var_names.values)

            ro.globalenv["seurat.obj.list"] = seurat_obj_list
            ro.globalenv["features"] = features

            if method == "rpca":
                with _RCallbacks(lambda x: None):
                    ro.r("""
                        seurat.obj.list <- lapply(X = seurat.obj.list, FUN = function(x) {
                            x <- RunPCA(x, features = features, verbose = FALSE, npcs = n_comps)
                        })
                    """.replace("n_comps", str(n_comps)))
            
            if sc.settings.verbosity >= 2:
                prefix = "    "
                progress_bar = True
            else:
                prefix = ""

            n_batches = len(set(adata.obs[batch_key].values))
            total_pairs = int(n_batches * (n_batches - 1) / 2)

            if progress_bar:
                cm = tqdm(
                    total=total_pairs,
                    desc=f"{prefix}finding anchors",
                    unit="pair",
                    file=sys.stdout,
                )
            else:
                cm = nullcontext()

            with cm as pbar:
                if progress_bar:
                    pbar.first_iteration = True
                    pbar.prev_message = ""

                def FindAnchorsCallback(s):
                    if progress_bar:
                        s_clean = s.strip().replace("\n", "")
                        if ("Running CCA" in s_clean) or ("Projecting new data onto SVD" in s_clean):
                            if not ("Projecting new data onto SVD" in pbar.prev_message):
                                if pbar.first_iteration:
                                    pbar.first_iteration = False
                                else:
                                    pbar.update(1)
                        if "is deprecated" not in s_clean:
                            pbar.set_postfix_str(s_clean)
                        pbar.prev_message = s_clean

                with _RCallbacks(FindAnchorsCallback):
                    ro.r(f"""
                        integration.anchors <- FindIntegrationAnchors(
                            object.list = seurat.obj.list,
                            normalization.method = 'LogNormalize',
                            anchor.features = features,
                            scale = FALSE,
                            reduction = '{method}',
                            l2.norm = {str(l2_norm).upper()},
                            dims = 1:{n_comps},
                            k.anchor = {k_anchor},
                            k.filter = {k_filter},
                            k.score = {k_score},
                            max.features = {max_features},
                            n.trees = {n_trees},
                            eps = {eps}
                        )
                    """)

                if progress_bar:
                    pbar.update(1)
                    pbar.set_postfix_str("")

            # Reconstructing MNN matrix
            with _RCallbacks(lambda x: None):
                cell_names = {}
                for i, seurat_obj_i in enumerate(seurat_obj_list):
                    cell_names[i + 1] = np.array(ro.r["colnames"](seurat_obj_i))
                anchors = pd.DataFrame(
                    ro.r("integration.anchors@anchors"),
                    index=["cell1", "cell2", "weight", "dataset1", "dataset2"],
                ).T
                anchors["cell1"] = anchors.apply(lambda x: cell_names[int(x.dataset1)][int(x.cell1) - 1], axis=1)
                anchors["cell2"] = anchors.apply(lambda x: cell_names[int(x.dataset2)][int(x.cell2) - 1], axis=1)
                del anchors["dataset1"], anchors["dataset2"]

                obs_index = pd.Index(adata.obs_names)
                idx1 = obs_index.get_indexer(anchors["cell1"].values).astype(np.int64)
                idx2 = obs_index.get_indexer(anchors["cell2"].values).astype(np.int64)

                rows = np.concatenate([idx1, idx2])
                cols = np.concatenate([idx2, idx1])
                values = np.concatenate([anchors["weight"].values, anchors["weight"].values])
                adata.obsp[mnn_graph_key] = sp.csr_matrix((values, (rows, cols)), shape=(adata.n_obs, adata.n_obs))

                adata.uns[uns_key] = {
                    "method": method,
                    "l2_norm": l2_norm,
                    "n_comps": n_comps,
                    "k.anchor": k_anchor,
                    "k.filter": k_filter,
                    "k.score": k_score,
                    "max.features": max_features,
                    "n.trees": n_trees,
                    "eps": eps,
                    "batch_key": batch_key,
                    "graph_key": mnn_graph_key,
                }

                if only_anchors:
                    adata.uns[uns_key]["mode"] = "anchors"

                    # Additional garbage collection
                    try:
                        del seurat_obj, seurat_obj_list, features
                    except NameError:
                        pass

                    gc.collect()
                    for i in ["seurat.obj.list", "features", "integration.anchors"]:
                        if i in ro.globalenv:
                            del ro.globalenv[i]

                    ro.r("gc(verbose = FALSE)")
                    gc.collect()

                    lines = [
                        "added",
                        f"     .obsp['{mnn_graph_key}'] MNN graph",
                        f"     .uns['{uns_key}'] MNN search parameters"
                    ]
                    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
                    return None

            total_integrations = n_batches - 1
            if progress_bar:
                cm = tqdm(
                    total=total_integrations,
                    desc=f"{prefix}integrating data",
                    unit="int",
                    file=sys.stdout,
                )
            else:
                cm = nullcontext()

            with cm as pbar:
                if progress_bar:
                    pbar.first_iteration = True
                    
                def IntegrateDataCallback(s):
                    if progress_bar:
                        s_clean = s.strip()
                        if "Merging" in s_clean:
                            if pbar.first_iteration:
                                pbar.first_iteration = False
                            else:
                                pbar.update(1)
                        pbar.set_postfix_str(s_clean)

                with _RCallbacks(IntegrateDataCallback):
                    ro.r(f"""
                        seurat.integrated <- IntegrateData(
                            anchorset = integration.anchors,
                            normalization.method = "LogNormalize",
                            features.to.integrate = features,
                            dims = 1:{n_comps},
                            k.weight = {k_weight}
                        )
                    """)

                if progress_bar:
                    pbar.update(1)
                    pbar.set_postfix_str("")
                
            seurat_integrated = ro.r("seurat.integrated")
            integrated_matrix = SeuratObject.GetAssayData(
                seurat_integrated, 
                assay="integrated", 
                slot="data",
            )
            var_names_integrated = list(ro.r["rownames"](integrated_matrix))
            obs_names_integrated = list(ro.r["colnames"](integrated_matrix))
            integrated_matrix = _dgCMatrix_to_np(integrated_matrix).T
            integrated_matrix = pd.DataFrame(
                integrated_matrix,
                columns=var_names_integrated,
                index=obs_names_integrated,
            )
            adata.layers[layer_name] = integrated_matrix.loc[
                adata.obs_names, adata.var_names
            ].values.astype("float32")
            if scale_integrated:
                sc.pp.scale(adata, layer=layer_name, max_value=scale_max_value)

            adata.uns[uns_key]["mode"] = "anchors + integration"
            adata.uns[uns_key]["k.weight"] = k_weight

            if underscores:
                adata.var_names = var_names
                adata.raw.var.index = raw_var_names

            lines = [
                "added",
                f"     .layers['{layer_name}'] {'scaled ' if scale_integrated else ''}integrated expression matrix",
                f"     .obsp['{mnn_graph_key}'] MNN graph",
                f"     .uns['{uns_key}'] integration parameters"
            ]
            logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

            # Additional garbage collection #1
            try:
                del (
                    seurat_integrated, integrated_matrix, var_names_integrated,
                    obs_names_integrated, seurat_obj, seurat_obj_list, features,
                )
            except NameError:
                pass

            with _RCallbacks(lambda x: None):
                gc.collect()
                ro.r("gc(verbose = FALSE)")
                for i in ["seurat.obj.list", "features", "integration.anchors", "seurat.integrated"]:
                    if i in ro.globalenv:
                        del ro.globalenv[i]
                gc.collect()
    finally:
        # Additional garbage collection #2
        gc.collect()
        with _RCallbacks(lambda x: None):
            ro.r("gc(verbose = FALSE)")
        gc.collect()