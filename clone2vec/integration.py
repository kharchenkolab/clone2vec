from __future__ import annotations

import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sys
from contextlib import nullcontext

from tqdm import tqdm

logg = sc.logging

def _mnn(
    A: pd.DataFrame,
    B: pd.DataFrame,
    k: int = 10,
    metric: str = "euclidean",
    k_trim: int = 3,
) -> np.ndarray:
    """
    Find mutual nearest neighbors (MNN) between two datasets A and B.

    Parameters
    ----------
    A : pd.DataFrame
        First dataset with shape (n_samples_A, n_features).
    B : pd.DataFrame
        Second dataset with shape (n_samples_B, n_features).
    k : int, optional
        Number of nearest neighbors to find, by default 10.
    metric : str, optional
        Distance metric to use, by default "euclidean".

    Returns
    -------
    np.ndarray
        Array of MNN pairs, where each row is [index_A, index_B] of a MNN pair.
    """
    import pynndescent

    index1 = pynndescent.NNDescent(A.values, n_neighbors=k, metric=metric)
    neighbors_2to1, _ = index1.query(B.values, k=k)

    index2 = pynndescent.NNDescent(B.values, n_neighbors=k, metric=metric)
    neighbors_1to2, _ = index2.query(A.values, k=k)

    rev_pos_2to1 = [{a: p for p, a in enumerate(arr)} for arr in neighbors_2to1]
    rev_pos_1to2 = [{a: p for p, a in enumerate(arr)} for arr in neighbors_1to2]

    pairs = []
    weights = []

    for i in range(len(A)):
        js = neighbors_1to2[i]
        cand = []
        scores = []
        wts = []
        for pos, j in enumerate(js):
            rp = rev_pos_2to1[j].get(i)
            if rp is None:
                continue
            rank_sum = pos + rp
            w = 1.0 if k <= 1 else 1.0 - (rank_sum / (2.0 * (k - 1)))
            cand.append(j)
            scores.append(rank_sum)
            wts.append(w)
        if cand:
            order = np.argsort(scores)[: min(k_trim, len(cand))]
            for idx in order:
                pairs.append((A.index[i], B.index[cand[idx]]))
                weights.append(wts[idx])

    for j in range(len(B)):
        is_ = neighbors_2to1[j]
        cand = []
        scores = []
        wts = []
        for pos, i in enumerate(is_):
            rp = rev_pos_1to2[i].get(j)
            if rp is None:
                continue
            rank_sum = pos + rp
            w = 1.0 if k <= 1 else 1.0 - (rank_sum / (2.0 * (k - 1)))
            cand.append(i)
            scores.append(rank_sum)
            wts.append(w)
        if cand:
            order = np.argsort(scores)[: min(k_trim, len(cand))]
            for idx in order:
                pairs.append((A.index[cand[idx]], B.index[j]))
                weights.append(wts[idx])

    return np.array(pairs), np.asarray(weights, dtype=float)

def _hierarchical_concatenation(
    similarity_matrix: pd.DataFrame,
    method: str = "ward",
) -> list[tuple[tuple[str, str], str]]:
    """
    Perform hierarchical concatenation on a similarity matrix.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        DataFrame where rows and columns represent items, and values are similarities.
    method : str, optional
        Linkage method to use for clustering, by default "ward".

    Returns
    -------
    list[tuple[tuple[str, str], str]]
        List of concatenation steps, each a pair of merged clusters and their new name.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    labels = similarity_matrix.columns.tolist()
    
    sim_array = similarity_matrix.to_numpy()
    max_similarity = np.max(sim_array)
    dist_array = max_similarity - sim_array
    
    condensed_dist = squareform(dist_array, checks=False)
    Z = linkage(condensed_dist, method=method)
    
    n = len(labels)
    cluster_names = labels.copy()
    concatenation_steps = []

    for i, row in enumerate(Z):
        idx1, idx2 = int(row[0]), int(row[1])
        name1 = cluster_names[idx1]
        name2 = cluster_names[idx2]
        new_cluster_name = f"({name1}+{name2})"
        concatenation_steps.append([[name1, name2], new_cluster_name])
        cluster_names.append(new_cluster_name)

    return concatenation_steps

def _affine_transform(
    A: np.ndarray,
    B: np.ndarray,
    order_A: np.ndarray,
    order_B: np.ndarray,
    weights: np.ndarray = None,
) -> tuple[np.ndarray, dict]:
    """
    Perform weighted affine least-squares alignment between datasets A and B.

    Parameters
    ----------
    A : np.ndarray
        First dataset with shape (n_samples_A, n_features) — query.
    B : np.ndarray
        Second dataset with shape (n_samples_B, n_features) — reference.
    order_A : np.ndarray
        Ordering of samples in A to match with B.
    order_B : np.ndarray
        Ordering of samples in B to match with A.
    weights : np.ndarray, optional
        Weights for each sample pair, by default None.

    Returns
    -------
    tuple[np.ndarray, dict]
        Transformed A and dictionary with rotation matrix, scale, and translation.
    """
    A_matched = A[order_A, :]
    B_matched = B[order_B, :]

    if weights is not None:
        w_sqrt = np.sqrt(weights)[:, np.newaxis]
    else:
        w_sqrt = np.ones((A_matched.shape[0], 1))

    ones_col = np.ones((A_matched.shape[0], 1))
    A_aug = np.hstack([A_matched, ones_col])
    A_w = A_aug * w_sqrt
    B_w = B_matched * w_sqrt

    solution, _, _, _ = np.linalg.lstsq(A_w, B_w, rcond=None)
    M = solution[:-1, :]
    t = solution[-1, :]

    A_transformed = (A @ M) + t
    transform_info = {
        "matrix": M,
        "translation": t,
    }

    return A_transformed, transform_info

def _write_order(order: list) -> str:
    """
    Function transforms list with order of alignment to string.

    Parameters
    ----------
    order : list
        List of concatenation steps, each a pair of merged clusters and their new name.

    Returns
    -------
    str
        String representation of the order of concatenation.
    """
    order_str = ""
    for operation in order:
        order_str += f"[{operation[0][0]}] + [{operation[0][1]}] = [{operation[1]}];"
    return order_str
    
def _read_order(order_str: str) -> list:
    """
    Function transforms string with order of alignment to list.

    Parameters
    ----------
    order_str : str
        String representation of the order of concatenation.

    Returns
    -------
    list
        List of concatenation steps, each a pair of merged clusters and their new name.
    """
    order = []
    for operation in order_str.split(";"):
        if operation == "":
            continue
        clusters, new_cluster = operation.split(" = ")
        clusters = clusters.strip("[]").split("] + [")
        order.append([clusters, new_cluster.strip("[]")])
    return order

def find_mnn(
    clones: sc.AnnData,
    batch_key: str,
    use_rep: str = "X_pca",
    k: int = 10,
    k_trim: int = 3,
    metric: str = "euclidean",
    uns_key: str = "simple_mnn",
    graph_key: str = "mnn",
    progress_bar: bool = True,
) -> None:
    """
    Find anchors between batches in clones.obs[batch_key] using MNNs using clones.obsm[use_rep].

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix at the clone level.
    batch_key : str
        Column in `clones.obs` containing batch information.
    use_rep : str, optional
        Key in `clones.obsm` containing representation to use for anchor finding, by default "X_pca".
    k : int, optional
        Number of nearest neighbors to use for anchor finding, by default 10.
    metric : str, optional
        Distance metric to use for anchor finding, by default "euclidean".
    uns_key : str, optional
        Key in `clones.uns` to store anchor information, by default "simple_mnn".
    graph_key : str, optional
        Key in `clones.obsp` to store MNN graph, by default "mnn".
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.

    Returns
    -------
    None
        MNN graph is stored in `clones.obsp[graph_key]`. 
        Anchor information is stored in `clones.uns[uns_key]`.
    """
    from itertools import combinations
    
    mnn_cols = []
    mnn_rows = []
    mnn_data = []

    if not pd.api.types.is_string_dtype(clones.obs[batch_key]):
        logg.warning(f"clones.obs['{batch_key}'] is not string. Converting to string.")
        clones.obs[batch_key] = clones.obs[batch_key].astype(str)

    if not isinstance(clones.obs[batch_key], pd.CategoricalDtype):
        logg.warning(f"clones.obs['{batch_key}'] is not categorical. Converting to categorical.")
        clones.obs[batch_key] = clones.obs[batch_key].astype("category")

    if not clones.obs_names.is_unique:
        logg.warning("clones.obs_names must be unique. Performing obs_names_make_unique().")
        clones.obs_names_make_unique()

    try:
        matrix = clones.obsm[use_rep]
    except KeyError:
        raise KeyError(f"clones.obsm['{use_rep}'] is not found. Please provide a valid use_rep.")

    # Ensure array-like numeric matrix to avoid pandas reindexing introducing NaNs
    # DataFrame inputs are converted to their underlying values for neighbor search
    matrix_values = matrix.values if isinstance(matrix, pd.DataFrame) else matrix

    line_before = f"    using `clones.obsm['{use_rep}']` for neighbor search"
    start = logg.info("finding MNN anchors across batches", deep="\n" + line_before)

    batches = list(set(clones.obs[batch_key]))
    batches.sort()

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    similarities = {}
    pair_indices = list(combinations(range(len(batches)), 2))
    cm = tqdm(
        pair_indices,
        desc=prefix + "pairwise MNNs",
        file=sys.stdout,
    ) if progress_bar else nullcontext(pair_indices)
    with cm as iterator:
        for i, j in iterator:
            pairs, wts = _mnn(
                pd.DataFrame(
                    matrix_values[clones.obs[batch_key] == batches[i]],
                    index=np.argwhere(clones.obs[batch_key] == batches[i]).flatten(),
                ),
                pd.DataFrame(
                    matrix_values[clones.obs[batch_key] == batches[j]],
                    index=np.argwhere(clones.obs[batch_key] == batches[j]).flatten(),
                ),
                k=k,
                k_trim=k_trim,
                metric=metric,
            )

            if pairs.size == 0:
                similarities.setdefault(batches[i], {})
                similarities.setdefault(batches[j], {})
                similarities[batches[i]][batches[j]] = 0
                similarities[batches[j]][batches[i]] = 0
                continue

            uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
            uniq_wts = np.zeros(len(uniq_pairs), dtype=float)
            for idx_u, w in zip(inv, wts):
                if w > uniq_wts[idx_u]:
                    uniq_wts[idx_u] = w

            similarities.setdefault(batches[i], {})
            similarities.setdefault(batches[j], {})

            mnn_cols.extend(uniq_pairs[:, 0])
            mnn_cols.extend(uniq_pairs[:, 1])

            mnn_rows.extend(uniq_pairs[:, 1])
            mnn_rows.extend(uniq_pairs[:, 0])

            mnn_data.extend(uniq_wts)
            mnn_data.extend(uniq_wts)

            similarities[batches[i]][batches[j]] = len(uniq_pairs)
            similarities[batches[j]][batches[i]] = similarities[batches[i]][batches[j]]

    if graph_key in clones.obsp:
        logg.warning(f"clones.obsp['{graph_key}'] is already found. Overwriting.")

    # Build a symmetric MNN graph over global indices; ensure full shape matches clones.n_obs
    clones.obsp[graph_key] = sp.csr_matrix(
        (np.asarray(mnn_data, dtype=np.float32), (mnn_cols, mnn_rows)),
        shape=(clones.n_obs, clones.n_obs),
    )
    
    similarities = pd.DataFrame(similarities).fillna(0).astype(int)
    integration_order = _hierarchical_concatenation(similarities)

    # Making sure that there are at least 3 MNNs in each integration step
    for integration_step in integration_order:
        (batch1, batch2), result = integration_step

        batch1_names = batch1.replace("(", "").replace(")", "").split("+")
        batch2_names = batch2.replace("(", "").replace(")", "").split("+")

        if similarities.loc[batch1_names, batch2_names].sum().sum() < 3:
            raise ValueError(f"Not enough MNNs between {batch1} and {batch2}. Please increase k.")

    clones.uns[uns_key] = {
        "batch_key": batch_key,
        "use_rep": use_rep,
        "k": k,
        "k_trim": k_trim,
        "metric": metric,
        "graph_key": graph_key,
        "order": _write_order(integration_order),
    }
    lines = [
        "added",
        f"     .obsp['{graph_key}'] symmetric sparse matrix.",
        f"     .uns['{uns_key}'] integration order and parameters.",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def _find_order(batch_vector: pd.Series | np.ndarray | list, graph: sp.csr_matrix | np.ndarray):
    """
    Giving the MNN graph, calculating the integration order.

    Parameters
    ----------
    batch_vector : pd.Series | np.ndarray | list
        Vector of batch labels for each clone.
    graph : sp.csr_matrix | np.ndarray
        Symmetric MNN graph over global indices.

    Returns
    -------
    list
        Integration order.
    """
    similarities = {}
    batch_categories = list(set(batch_vector))
    for i, batch1 in enumerate(batch_categories):
        if batch1 not in similarities:
            similarities[batch1] = {}
        for j in range(i+1, len(batch_categories)):
            batch2 = batch_categories[j]
            if batch2 not in similarities:
                similarities[batch2] = {}
            similarities[batch1][batch2] = graph[batch_vector == batch1][:, batch_vector == batch2].sum()
            similarities[batch2][batch1] = similarities[batch1][batch2]

    similarities = pd.DataFrame(similarities).fillna(0)
    integration_order = _hierarchical_concatenation(similarities)
    return _write_order(integration_order)
    
def align(
    clones: sc.AnnData,
    use_rep: str = "clone2vec",
    adj_rep: str = "clone2vec_pa",
    uns_key: str = "simple_mnn",
    weighted: bool = True,
) -> None:
    """
    Align clones.obsm[use_rep] using MNNs found in `clones.uns[uns_key]` via procrustes alignment (PA).

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix at the clone level.
    use_rep : str, optional
        Key in `clones.obsm` containing representation to use for alignment, by default "clone2vec".
    adj_rep : str, optional
        Key in `clones.obsm` to store aligned representation, by default "clone2vec_pa".
    uns_key : str, optional
        Key in `clones.uns` containing anchor information, by default "simple_mnn".

    Returns
    -------
    None
        Aligned representation is stored in `clones.obsm[adj_rep]`.
    """
    start = logg.info("aligning representation using weighted affine least-squares and MNN anchors")
    try:
        batch_key = clones.uns[uns_key]["batch_key"]
        graph = clones.obsp[clones.uns[uns_key]["graph_key"]]
        if "order" not in clones.uns[uns_key]:
            integration_order = _read_order(_find_order(clones.obs[batch_key].values, graph))
        else:
            integration_order = _read_order(clones.uns[uns_key]["order"])
    except KeyError:
        raise KeyError(f"clones.uns['{uns_key}'] is not full. Please run `find_anchors` first.")

    try:
        emb = clones.obsm[use_rep].copy()
    except KeyError:
        raise KeyError(f"clones.obsm['{use_rep}'] is not found. Please provide a valid use_rep.")

    for integration_step in integration_order:
        (batch1, batch2), result = integration_step

        batch1_samples = batch1.strip("()").split("+")
        batch2_samples = batch2.strip("()").split("+")

        ref = clones.obs[batch_key].isin(batch1_samples).values
        query = clones.obs[batch_key].isin(batch2_samples).values
        
        if sum(ref) < sum(query):
            ref, query = query, ref
        
        sub = graph[query][:, ref].tocoo()
        if weighted:
            weights = sub.data
        else:
            weights = None

        A_aligned, _ = _affine_transform(
            A=emb[query],
            B=emb[ref],
            order_A=sub.row,
            order_B=sub.col,
            weights=weights,
        )

        emb[query] = A_aligned

    clones.obsm[adj_rep] = emb
    lines = [
        "added",
        f"     .obsm['{adj_rep}'] aligned representation.",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
