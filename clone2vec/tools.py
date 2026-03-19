from __future__ import annotations

import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sys
from contextlib import nullcontext
from typing import Any, Literal
from tqdm import tqdm
from .embeddings import (
    clone2vec, clone2vec_Poi, project_clone2vec, project_clone2vec_Poi
)
from .integration import find_mnn, align
from .associations import associations, graph_associations
from .eigenvalues import eigenvalue_test
from .catboost import catboost

logg = sc.logging

__all__ = [
    "clonal_nn",
    "smooth",
    "clone2vec",
    "clone2vec_Poi",
    "project_clone2vec",
    "project_clone2vec_Poi",
    "find_mnn",
    "align",
    "associations",
    "group_connectivity",
    "clonocluster",
    "graph_associations",
    "eigenvalue_test",
    "catboost",
]

def __dir__():
    return sorted(__all__)
    

def clonal_nn(
    adata: sc.AnnData,
    clones: sc.AnnData,
    obs_name: str | None = None,
    k: int = 15,
    use_rep: str = "X_pca",
    obsp_name: str = "gex_adjacency",
    adata_ref: sc.AnnData | None = None,
    clones_ref: sc.AnnData | None = None,
    nn_transfer_name: str = "ref_gex_adjacency",
    random_state: None | int = 4,
    split_by: str | None = None,
    progress_bar: bool = False,
    mask_obs: list[np.ndarray | pd.Series] | np.ndarray | pd.Series | None = None,
    **kwargs,
) -> None:
    """
    Computes and adds a clone-to-clone adjacency graph to a clones object.

    The graph is built by finding nearest neighbors in a given representation (e.g., PCA)
    and aggregating these relationships at the clone level.

    Parameters
    ----------
    adata : AnnData
        The original, cell-level annotated data matrix.
    clones : AnnData
        The clone-level AnnData object, typically from `create_clone_adata`.
    obs_name : str | None, optional
        Column in `adata.obs` containing clonal information. If None, use `clones.uns["obs_name"]`, 
        by default None.
    k : int, optional
        Number of nearest neighbors to retrieve for each cell, by default 15.
    use_rep : str, optional
        Representation in `adata.obsm` to build the kNN graph, by default "X_pca".
    obsp_name : str, optional
        Name of the new graph slot in `clones.obsp`, by default "gex_adjacency".
    adata_ref : sc.AnnData | None, optional
        Reference cell-level annotated data matrix for transfer, by default None.
    clones_ref : sc.AnnData | None, optional
        Reference clone-level AnnData object for transfer, by default None.
    nn_transfer_name : str, optional
        Name of the new graph slot in `clones_ref.obsp` for transfer, by default "ref_gex_adjacency".
    random_state : None | int, optional
        Random state for reproducibility, by default 4.
    split_by : str | None, optional
        Column in `clones.obs` to split the independent kNN-graph constriction, by default None.
    progress_bar : bool, optional
        Whether to display a progress bar in the case of split_by, by default False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to `pynndescent.NNDescent`.

    Returns
    -------
    None
        Updates `clones` in-place with the computed adjacency graph in `.obsp[obsp_name]`.
    """
    start = logg.info("computing clone-to-clone adjacency graph")
    import pynndescent

    if obs_name is None:
        try:
            obs_name = clones.uns["obs_name"]
        except KeyError:
            raise KeyError("obs_name not found in clones.uns. Please provide obs_name.")
    try:
        train_adata = adata[adata.obs[obs_name].isin(clones.obs_names)]
    except KeyError:
        raise KeyError(f"obs_name '{obs_name}' not found in adata.obs. Please provide a valid obs_name.")
    
    if use_rep not in train_adata.obsm:
        raise KeyError(f"Representation '{use_rep}' not found in adata.obsm.")

    # Explicit mapping from clone labels to indices to avoid out-of-range codes
    n_clones = len(clones.obs_names)
    label_to_idx = pd.Series(np.arange(n_clones, dtype=np.int32), index=pd.Index(clones.obs_names))

    train_labels_encoded = label_to_idx.loc[train_adata.obs[obs_name]].to_numpy()

    n_train_cells = train_adata.n_obs

    if mask_obs is None:
        mask_series_list: list[pd.Series] = []
        mask_names: list[str] = []
    else:
        if isinstance(mask_obs, (np.ndarray, pd.Series)):
            mask_obs = [mask_obs]
        mask_series_list = []
        mask_names = []
        for i, m in enumerate(mask_obs):
            if isinstance(m, pd.Series):
                if m.index.is_unique and m.index.equals(adata.obs_names):
                    s = m.astype(bool)
                else:
                    s = pd.Series(np.asarray(m).astype(bool), index=adata.obs_names)
            else:
                arr = np.asarray(m).astype(bool)
                if arr.shape[0] != adata.n_obs:
                    raise ValueError("mask_obs must have length equal to adata.n_obs")
                s = pd.Series(arr, index=adata.obs_names)
            name = s.name if s.name is not None else f"mask_{i+1}"
            mask_series_list.append(s)
            mask_names.append(name)
        mask_union_train = np.logical_or.reduce([ms.loc[train_adata.obs_names].to_numpy() for ms in mask_series_list]) if len(mask_series_list) > 0 else np.zeros(n_train_cells, dtype=bool)
        masks_train_list = [ms.loc[train_adata.obs_names].to_numpy() for ms in mask_series_list]
    if mask_obs is None:
        mask_union_train = np.zeros(n_train_cells, dtype=bool)
        masks_train_list = []
    masked_sum_cols = None

    if adata_ref is not None and clones_ref is not None:
        if split_by is not None:
            raise ValueError("split_by is not supported when using clones_ref.")
        if use_rep not in adata_ref.obsm:
            raise KeyError(f"Representation '{use_rep}' not found in adata_ref.obsm.")

        index = pynndescent.NNDescent(adata_ref.obsm[use_rep], random_state=random_state, **kwargs)
        neighbors_indices, _ = index.query(train_adata.obsm[use_rep], k=k)

        ref_label_to_idx = pd.Series(np.arange(len(clones_ref.obs_names), dtype=np.int32), index=pd.Index(clones_ref.obs_names))
        ref_labels = adata_ref.obs[obs_name].values
        ref_neighbor_labels = ref_labels[neighbors_indices]
        ref_neighbor_flat = pd.Series(ref_neighbor_labels.ravel()).map(ref_label_to_idx).fillna(-1).astype(np.int32).to_numpy()

        row_idx = np.repeat(np.arange(n_train_cells, dtype=np.int32), k)
        col_idx = ref_neighbor_flat
        valid_mask = (col_idx >= 0) & (col_idx < len(clones_ref.obs_names))

        nonmasked_edges = valid_mask & (~mask_union_train[row_idx])
        B_sparse = sp.csr_matrix(
            (np.ones(nonmasked_edges.sum(), dtype=np.int32), (row_idx[nonmasked_edges], col_idx[nonmasked_edges])),
            shape=(n_train_cells, len(clones_ref.obs_names))
        )

        I_sparse = sp.csr_matrix(
            (np.ones(n_train_cells, dtype=np.int32),
             (train_labels_encoded.astype(np.int32), np.arange(n_train_cells, dtype=np.int32))),
            shape=(n_clones, n_train_cells)
        )

        nn_transfer = I_sparse @ B_sparse

        if nn_transfer_name in clones.obsm:
            logg.warning(f"Overwriting existing matrix '{nn_transfer_name}' in clones.obsm")

        clones.obsm[nn_transfer_name] = nn_transfer
        if len(masks_train_list) > 0:
            masked_cols = []
            for m in masks_train_list:
                masked_edges_m = valid_mask & (m[row_idx])
                B_masked_m = sp.csr_matrix(
                    (np.ones(masked_edges_m.sum(), dtype=np.int32), (row_idx[masked_edges_m], col_idx[masked_edges_m])),
                    shape=(n_train_cells, len(clones_ref.obs_names))
                )
                masked_adj_m = I_sparse @ B_masked_m
                counts_m = np.asarray(masked_adj_m.sum(axis=1)).ravel()
                masked_cols.append(counts_m)
            masked_nn = np.column_stack(masked_cols)
            clones.obsm["masked_nn"] = np.asarray(masked_nn)
        clones.uns["clonal_nn"] = {
            "k": k, "use_rep": use_rep, "obsm_name": nn_transfer_name,
        }
        if mask_series_list:
            clones.uns["clonal_nn"]["mask"] = {n: ms.loc[adata.obs_names].to_numpy() for n, ms in zip(mask_names, mask_series_list)}
        lines = [
            "added",
            f"     .obsm['{nn_transfer_name}'] clone-to-reference adjacency.",
            ("     .obsm['masked_nn'] masked neighbour counts." if len(mask_series_list) > 0 else None),
            "     .uns['clonal_nn'] parameters.",
        ]
        logg.info("    finished ({time_passed})", deep="\n".join([l for l in lines if l is not None]), time=start)
    else:
        if split_by is None:
            index = pynndescent.NNDescent(train_adata.obsm[use_rep], random_state=random_state, **kwargs)
            neighbors_indices, _ = index.query(train_adata.obsm[use_rep], k=k)

            neighbor_labels_encoded = train_labels_encoded[neighbors_indices]

            row_idx = np.repeat(np.arange(n_train_cells, dtype=np.int32), k)
            col_idx = neighbor_labels_encoded.flatten().astype(np.int32)
            valid_mask = (col_idx >= 0) & (col_idx < n_clones)

            nonmasked_edges = valid_mask & (~mask_union_train[row_idx])
            B_sparse = sp.csr_matrix(
                (np.ones(nonmasked_edges.sum(), dtype=np.int32), (row_idx[nonmasked_edges], col_idx[nonmasked_edges])),
                shape=(n_train_cells, n_clones)
            )

            I_sparse = sp.csr_matrix(
                (np.ones(n_train_cells, dtype=np.int32), 
                 (train_labels_encoded.astype(np.int32), np.arange(n_train_cells, dtype=np.int32))),
                shape=(n_clones, n_train_cells)
            )

            gex_adjacency = I_sparse @ B_sparse
            if len(masks_train_list) > 0:
                masked_cols = []
                for m in masks_train_list:
                    masked_edges_m = valid_mask & (m[row_idx])
                    B_masked_m = sp.csr_matrix(
                        (np.ones(masked_edges_m.sum(), dtype=np.int32), (row_idx[masked_edges_m], col_idx[masked_edges_m])),
                        shape=(n_train_cells, n_clones)
                    )
                    masked_adj_m = I_sparse @ B_masked_m
                    counts_m = np.asarray(masked_adj_m.sum(axis=1)).ravel()
                    masked_cols.append(counts_m)
                masked_nn = np.column_stack(masked_cols)
                clones.obsm["masked_nn"] = np.asarray(masked_nn)
        else:
            if split_by not in train_adata.obs:
                raise KeyError(f"Column '{split_by}' not found in adata.obs.")

            train_index_map = pd.Series(np.arange(n_train_cells, dtype=np.int32), index=train_adata.obs_names)
            I_sparse = sp.csr_matrix(
                (np.ones(n_train_cells, dtype=np.int32),
                 (train_labels_encoded.astype(np.int32), np.arange(n_train_cells, dtype=np.int32))),
                shape=(n_clones, n_train_cells)
            )

            gex_adjacency = sp.csr_matrix((n_clones, n_clones), dtype=np.int32)
            masked_sum_cols = None

            categories = pd.unique(train_adata.obs[split_by])
            categories = [c for c in categories if pd.notna(c)]

            if sc.settings.verbosity.value >= 2:
                prefix = "    "
                progress_bar = True
            else:
                prefix = ""

            cm = tqdm(
                categories,
                desc=prefix + f"split-by '{split_by}'",
                file=sys.stdout,
            ) if progress_bar else nullcontext(categories)
            with cm as iterator:
                for cat in iterator:
                    subset = train_adata[train_adata.obs[split_by] == cat]
                    n_subset = subset.n_obs
                    if n_subset == 0:
                        continue
                    k_subset = int(min(k, n_subset))
                    index = pynndescent.NNDescent(subset.obsm[use_rep], random_state=random_state, **kwargs)
                    neighbors_indices, _ = index.query(subset.obsm[use_rep], k=k_subset)

                    subset_rows = train_index_map.loc[subset.obs_names].to_numpy()
                    subset_labels_encoded = train_labels_encoded[subset_rows]
                    neighbor_labels_encoded = subset_labels_encoded[neighbors_indices]

                    row_idx = np.repeat(subset_rows.astype(np.int32), k_subset)
                    col_idx = neighbor_labels_encoded.flatten().astype(np.int32)
                    valid_mask = (col_idx >= 0) & (col_idx < n_clones)

                    nonmasked_edges = valid_mask & (~mask_union_train[row_idx])
                    B_sparse = sp.csr_matrix(
                        (np.ones(nonmasked_edges.sum(), dtype=np.int32), (row_idx[nonmasked_edges], col_idx[nonmasked_edges])),
                        shape=(n_train_cells, n_clones)
                    )

                    gex_adjacency += I_sparse @ B_sparse
                    if len(masks_train_list) > 0:
                        cols_this = []
                        for m in masks_train_list:
                            masked_edges_m = valid_mask & (m[row_idx])
                            B_masked_m = sp.csr_matrix(
                                (np.ones(masked_edges_m.sum(), dtype=np.int32), (row_idx[masked_edges_m], col_idx[masked_edges_m])),
                                shape=(n_train_cells, n_clones)
                            )
                            masked_adj_m = I_sparse @ B_masked_m
                            counts_m = np.asarray(masked_adj_m.sum(axis=1)).ravel()
                            cols_this.append(counts_m)
                        if masked_sum_cols is None:
                            masked_sum_cols = np.column_stack(cols_this)
                        else:
                            masked_sum_cols += np.column_stack(cols_this)

        if obsp_name in clones.obsp:
            logg.warning(f"Overwriting existing graph '{obsp_name}' in clones.obsp")

        clones.obsp[obsp_name] = gex_adjacency
        clones.uns["clonal_nn"] = {
            "k": k, "use_rep": use_rep, "obsp_name": obsp_name, "split_by": split_by,
        }
        if mask_series_list:
            clones.uns["clonal_nn"]["mask"] = {n: ms.loc[adata.obs_names].to_numpy() for n, ms in zip(mask_names, mask_series_list)}
            if masked_sum_cols is None and len(masks_train_list) > 0 and split_by is None:
                masked_sum_cols = clones.obsm.get("masked_nn", None)
                if isinstance(masked_sum_cols, pd.DataFrame):
                    masked_sum_cols = masked_sum_cols.to_numpy()
            if masked_sum_cols is not None:
                clones.obsm["masked_nn"] = np.asarray(masked_sum_cols)
        lines = [
            "added",
            f"     .obsp['{obsp_name}'] clone adjacency graph.",
            ("     .obsm['masked_nn'] masked neighbour counts." if len(mask_series_list) > 0 else None),
            "     .uns['clonal_nn'] parameters.",
        ]
        logg.info("    finished ({time_passed})", deep="\n".join([l for l in lines if l is not None]), time=start)

def smooth(
    adata: sc.AnnData,
    field: Literal["X", "layer", "obs", "obsm"] = "X",
    names: str | list[str] | None = None,
    key_added: str = "smoothed",
    graph_key: str = "connectivities",
    progress_bar: bool = False,
    n_steps: int = 3,
) -> None:
    """
    Performing kNN smoothing on `adata.X`, `adata.layers[layer]`, `adata.obs[name]`, or `adata.obsm[name]` for `n_steps` iterations using
    `adata.obsp[graph_key]`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the clone level.
    field : Literal["X", "layer", "obs", "obsm"], optional
        Field to smooth, by default "X".
    names : str | list[str] | None, optional
        Key(s) in `adata.layers`, `adata.obs`, or `adata.obsm` containing field to smooth, by default None.
    key_added : str, optional
        Key for `adata.layers`, `adata.obs`, or `adata.obsm` to store smoothed field.
    graph_key : str, optional
        Key in `adata.obsp` containing kNN graph, by default "connectivities".
    progress_bar : bool, optional
        Whether to display a progress bar, by default False.
    n_steps : int, optional
        Number of smoothing iterations, by default 3.

    Returns
    -------
    None
        Smoothed field is stored in `adata.layers["X_" + key_added]`, `adata.layers[name + "_" + key_added]`,
        `adata.obs[name + "_" + key_added]`, or `adata.obsm[name + "_" + key_added]`.
    """
    start = logg.info("smoothing fields using kNN graph")

    try:
        A = (adata.obsp[graph_key] > 0).astype(float)
    except KeyError:
        raise KeyError(f"Key {graph_key} not found in adata.obsp.")

    A.setdiag(1.0)
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_normalized = sp.diags(1.0 / row_sums) @ A

    Xs = []
    if field == "X":
        Xs.append(adata.X.copy())
        names = ["X"]
    elif field == "layer":
        if names is None:
            raise ValueError("Layer name must be provided when field is 'layer'.")
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in adata.layers:
                logg.warning(f"Layer '{name}' not found in adata.layers. Skipping.")
                names.remove(name)
            else:
                Xs.append(adata.layers[name].copy())
    elif field == "obs":
        if names is None:
            raise ValueError("Observation key must be provided when field is 'obs'.")
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in adata.obs:
                logg.warning(f"Observation key '{name}' not found in adata.obs. Skipping.")
                names.remove(name)
            else:
                Xs.append(adata.obs[name].values.reshape(-1, 1).copy())
    elif field == "obsm":
        if names is None:
            raise ValueError("Observation matrix key must be provided when field is 'obsm'.")
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in adata.obsm:
                logg.warning(f"Observation matrix key '{name}' not found in adata.obsm. Skipping.")
                names.remove(name)
            else:
                Xs.append(adata.obsm[name].copy())
    
    if len(Xs) == 0:
        raise ValueError("No valid data to smooth. Check your input.")

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
    else:
        prefix = ""
    for X, name in zip(Xs, names):
        cm = tqdm(
            range(n_steps),
            desc=prefix + f"smoothing '{name}'",
            file=sys.stdout,
        ) if progress_bar else nullcontext(range(n_steps))
        with cm as iterator:
            for _ in iterator:
                X = row_normalized @ X
        if field == "X":
            adata.layers[f"X_{key_added}"] = X
        elif field == "layer":
            adata.layers[f"{name}_{key_added}"] = X
        elif field == "obs":
            adata.obs[f"{name}_{key_added}"] = X.flatten()
        elif field == "obsm":
            if isinstance(adata.obsm[name], pd.DataFrame):
                adata.obsm[f"{name}_{key_added}"] = pd.DataFrame(
                    X,
                    index=adata.obs_names,
                    columns=adata.obsm[name].columns,
                )
            else:
                adata.obsm[f"{name}_{key_added}"] = X

    lines = ["added"]
    if field == "X":
        lines.append(f"     .layers['X_{key_added}'] smoothed matrix.")
    elif field == "layer":
        for name in names:
            lines.append(f"     .layers['{name}_{key_added}'] smoothed matrix.")
    elif field == "obs":
        for name in names:
            lines.append(f"     .obs['{name}_{key_added}'] smoothed values.")
    elif field == "obsm":
        for name in names:
            lines.append(f"     .obsm['{name}_{key_added}'] smoothed matrix.")
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def clonocluster(
    adata: sc.AnnData,
    alpha: float = 0.2,
    beta: float = 0.1,
    key_added: str = "clonocluster",
    graph_key: str | None = None,
    lineage_graph: str = "connected",
):
    """
    Performing ClonoCluster [PMID: 36819662] algorithm for lineage-informed gene expression clustering.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    alpha : float, optional
        Parameter for clonocluster algorithm, should be between 0 and 1. 0 means no inclusion of lineage information,
        1 means that only lineage information is used. Default is 0.2.
    beta : float, optional
        Parameter for clonocluster algorithm, should be between 0 and 1. Default is 0.1.
    key_added : str, optional
        Key in obsp to store clonocluster connectivities. Default is "ClonoCluster".
    graph_key : str | None, optional
        Key in uns to store graph. If None, looks for "neighbors". Default is None.
    lineage_graph : str, optional
        Key in obsp to store lineage graph. Default is "connected".
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    if not 0 <= beta <= 1:
        raise ValueError("alpha must be between 0 and 1")

    if graph_key is None:
        graph_key = "neighbors"

    connectivities_key = adata.uns[graph_key]["connectivities_key"]
    connectivities = adata.obsp[connectivities_key]
    use_rep = adata.uns[graph_key]["params"]["use_rep"]

    line_before = f"    using obsp['{lineage_graph}'] and obsp['{connectivities_key}']"
    start = logg.info("computing clonocluster connectivities", deep="\n" + line_before)

    adata.obsp[f"{key_added}_connectivities"] = (
        alpha ** beta * (adata.obsp[lineage_graph] - connectivities) +
        connectivities
    )

    adata.uns[key_added] = {
        "connectivities_key": f"{key_added}_connectivities",
        "distances_key": None,
        "params": {
            "method": "umap",
            "metric": "custom",
            "n_neighbors": None,
            "random_state": 0,
            "use_rep": use_rep,
        },
    }

    lines = [
        "added",
        f"     .obsp['{key_added}_connectivities'] clonocluster connectivities",
        f"     .uns['{key_added}'] with graph information",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def group_connectivity(
    adata: sc.AnnData,
    groupby: str,
    graph_key: str = "connectivities",
    directed: bool = True,
    uns_key_added: str = "group_connectivity",
):
    """
    Computes connectivities (number of observed over the number of expected edges) of groups based on the graph.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix at the cell level.
    groupby : str
        Key in `adata.obs` to use for grouping cells.
    graph_key : str, optional
        Key in `adata.obsp` to use for the graph, by default "connectivities".
    directed : bool, optional
        Whether to assume a directed graph, by default True.
    uns_key_added : str, optional
        Key in `adata.uns` to store the group connectivity matrix, by default "group_connectivity".

    Returns
    -------
    None
        The `adata` object is modified in place with the group connectivity matrix stored in `.uns[uns_key_added]`.
    """
    start = logg.info("computing groups connectivity matrix")
    n_obs = adata.n_obs
    
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        adata.obs[groupby] = adata.obs[groupby].astype("category")
    
    labels = adata.obs[groupby].cat.codes.values
    label_names = adata.obs[groupby].cat.categories
    k = len(label_names)
    
    A = adata.obsp[graph_key]
    B = sp.csc_matrix(
        (np.ones_like(labels, dtype=int), (np.arange(n_obs), labels)),
        shape=(n_obs, k)
    )
    
    if directed:
        observed_edges = (B.T @ (A > 0) @ B).toarray()

        cluster_out_degree = observed_edges.sum(axis=1)
        cluster_in_degree = observed_edges.sum(axis=0)
        total_edges = cluster_out_degree.sum()

        if total_edges == 0:
            return np.zeros((k, k), dtype=float)

        expected_edges = np.outer(cluster_out_degree, cluster_in_degree) / total_edges

    else:
        A_undirected = (A + A.T) > 0
        
        observed_edges = (B.T @ A_undirected @ B).toarray()
        cluster_sizes = np.bincount(labels, minlength=k)
        es = observed_edges.sum(axis=1)

        if n_obs <= 1:
            return np.zeros((k, k), dtype=float)

        expected_edges = (
            np.outer(es, cluster_sizes) + np.outer(cluster_sizes, es)
        ) / (n_obs - 1)

    connectivity = np.divide(
        observed_edges, 
        expected_edges, 
        out=np.zeros_like(observed_edges, dtype=float), 
        where=expected_edges > 0
    )
    
    adata.uns[uns_key_added] = {
        "groupby": groupby,
        "graph_key": graph_key,
        "label_names": list(label_names),
        "connectivity": connectivity,
    }

    lines = [
        "added",
        f"     .uns['{uns_key_added}']['connectivity'] groups connectivity matrix",
        f"     .uns['{uns_key_added}']['label_names'] label names",
    ]
    logg.info(
        "    finished ({time_passed})",
        deep="\n".join(lines),
        time=start,
    )
