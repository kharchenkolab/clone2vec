from __future__ import annotations
from pickle import ADDITEMS

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
 
logg = sc.logging

__all__ = [
    "stack_layers",
    "correct_shap",
    "connect_clones",
    "get_connectivity_matrix",
    "gs",
    "regress_categories",
]

def __dir__():
    return sorted(__all__)

def regress_categories(
    adata: sc.AnnData,
    obs_key: str | list[str],
    layer: str | None = None,
    key_added: str = "regressed",
) -> None:
    """
    Performs expression regression on categorical variables in obs_key. Instead of
    calculation of full regression model, it calculates mean centering. Can be effectively used
    to exclude effect of the cell type signature from the expression matrix.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with expression data.
    obs_key : str | list[str]
        Categorical variables in obs to regress.
    layer : str | None, optional
        Layer in adata.layers to regress. If None, use adata.X. Default is None.
    key_added : str, optional
        Key in adata.layers to add regressed expression. Default is "regressed".
    """
    if layer is None:
        layer = "X"
        use_X = True
    else:
        use_X = False

    start = logg.info(f"regressing categories {obs_key} on layer {layer}")
    if isinstance(obs_key, str):
        obs_key = [obs_key]
    for key in obs_key:
        if key not in adata.obs.columns:
            raise ValueError(f"obs_key {key} not found in adata.obs")
        elif adata.obs[key].dtype.kind in ["i", "f"]:
            raise ValueError(f"obs_key {key} is numerical")
    if use_X:
        X = adata.X
    else:
        X = adata.layers[layer]

    regressed = X.copy()
    if sp.issparse(X):
        logg.warning("expression matrix is sparse, converting to dense")
        regressed = regressed.todense()

    for key in obs_key:
        categories = list(set(adata.obs[key]))
        for category in categories:
            mask = (adata.obs[key] == category).values
            group_mean = regressed[mask, :].mean(axis=0)
            regressed[mask, :] -= group_mean

    adata.layers[f"{layer}_{key_added}"] = np.asarray(regressed)
    lines = [
        "added",
        f"     .layers['{layer}_{key_added}'] regressed expression",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)


def get_connectivity_matrix(
    adata: sc.AnnData,
    uns_key: str = "group_connectivity",
) -> pd.DataFrame:
    """
    Get group connectivity matrix from adata.uns[uns_key].

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with group connectivity matrix in adata.uns[uns_key].
    uns_key : str, optional
        Key in adata.uns to group connectivity matrix. Default is "group_connectivity".

    Returns
    -------
    pd.DataFrame
        Group connectivity matrix.
    """
    return pd.DataFrame(
        adata.uns[uns_key]["connectivity"],
        index=adata.uns[uns_key]["label_names"],
        columns=adata.uns[uns_key]["label_names"],
    )

def stack_layers(
    adata: sc.AnnData,
    layers: list[str] | None = None,
    layer_col_added: str = "layer",
    mask_key: str | None | Liretal[False] = None,
) -> None:
    """
    Function stacks layers of an AnnData object into a single layer in adata.X.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with layers to stack.
    layers : list[str] | None, optional
        Layers to stack. If None, all layers are stacked. Default is None.
    layer_col_added : str, optional
        Name of the column added to obs to indicate the layer. Default is "layer".
    mask_key : str | None | Literal[False], optional
        Key in obsm to boolean mask of cells to include in the stack. If None, tries to use "mask_key" in adata.uns. Default is None.
    """
    if mask_key is None and "mask_key" in adata.uns.keys():
        mask_key = adata.uns["mask_key"]

    if layer_col_added in adata.obs.columns:
        logg.warning(f"'{layer_col_added}' column already exists in adata.obs. Overwriting.")

    if layers is None:
        layers = list(adata.layers.keys())
    line_before = f"\n    using layers {layers}"
    start = logg.info("stacking layers into a new AnnData object", deep=line_before)

    Xs = [adata.layers[layer] for layer in layers]
    if mask_key:
        masks = [adata.obsm[mask_key][layer].values for layer in layers]
    else:
        masks = [np.ones(adata.shape[0], dtype=bool) for layer in layers]

    masks = [mask & ~_nan_mask(X) for X, mask in zip(Xs, masks)]
    Xs = [X[mask] for X, mask in zip(Xs, masks)]

    if sp.issparse(Xs[0]):
        X = sp.vstack(Xs, format="csr")
    else:
        X = np.vstack(Xs)
    del Xs

    mask = np.concatenate(masks)

    obs = [adata.obs.values] * len(layers)
    obs = [
        np.hstack([obs_i, np.array([layer] * len(obs_i))[:, None]])
        for obs_i, layer in zip(obs, layers)
    ]
    obs = np.vstack(obs)
    obs = pd.DataFrame(
        index=np.concatenate([layer + ":" + adata.obs.index.astype(str) for layer in layers]),
        data=obs,
        columns=list(adata.obs.columns) + [layer_col_added],
    )[mask]

    obsm = {}
    for obsm_key in adata.obsm.keys():
        obsm[obsm_key] = np.vstack([adata.obsm[obsm_key]] * len(layers))[mask]
        if isinstance(adata.obsm[obsm_key], pd.DataFrame):
            obsm[obsm_key] = pd.DataFrame(obsm[obsm_key])
            obsm[obsm_key].index = obs.index
            obsm[obsm_key].columns = adata.obsm[obsm_key].columns
    
    new_adata = sc.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=adata.var_names),
        varm=adata.varm,
        varp=adata.varp,
        obsm=obsm,
        uns=adata.uns,
    )

    if "mask_key" in new_adata.uns.keys():
        del new_adata.uns["mask_key"]

    lines = [
        "created stacked AnnData with",
        "     '.X' stacked matrix (layers × cells)",
        f"     .obs['{layer_col_added}'] layer labels",
        "     .obsm['<existing>'] stacked to match new index",
        "     .var, .varp, and .uns are inherited from the original adata",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
    return new_adata

def connect_clones(
    clones: sc.AnnData,
    groupby: str,
    graph_key_added: str = "connected",
    orient_col: str | None = None,
    orient_rule: Literal["all_increase", "all_decrease", "increase", "decrease"] | "Callable" | None = None,
    ignore_names: list[str] | str | None = None,
    weight_edges: bool = False,
):
    """
    Creates a graph of connected clones based on a grouping variable.

    Parameters
    ----------
    clones : sc.AnnData
        AnnData object with clone information.
    groupby : str
        Name of the column in obs to group by.
    graph_key_added : str, optional
        Name of the key added to obsp to store the graph. Default is "connected".
    orient_col : str | None, optional
        Name of the column in obs to orient the graph. If None, all pairs of clones are connected. Default is None.
    orient_rule : Literal["all_increase", "all_decrease", "increase", "decrease"] | "Callable" | None, optional
        Rule to orient the graph. If None, uses "increase" if `orient_col` is provided. Default is None.
    ignore_names : list[str] | str | None, optional
        Names of clones to ignore. If None, no clones are ignored. Default is None.
    weight_edges : bool, optional
        Whether to weight edges by the inverse of the number of connected clones. Default is False.

    Returns
    -------
    None
        Adds a graph to clones.obsp[graph_key_added].
    """
    line_before = f"\n    using .obs['{groupby}'] to group clones"
    start = logg.info(f"connecting clones", deep=line_before)

    from itertools import combinations

    if orient_col and (orient_rule is None):
        orient_rule = "increase"
    obs_idx = dict(zip(clones.obs_names, np.arange(len(clones))))

    if isinstance(ignore_names, str):
        ignore_names = [ignore_names]
    
    rows = []
    cols = []
    data = []
    
    ignore_set = set(ignore_names) if ignore_names is not None else None
    for name, connected_component in clones.obs.groupby(groupby, observed=True):
        if ignore_set is not None and name in ignore_set:
            continue
        if len(connected_component) > 1:
            w = (1.0 / len(connected_component)) if weight_edges else 1.0
            if orient_col:
                if isinstance(orient_rule, str):
                    if orient_rule.endswith("increase"):
                        ascending = True
                    else:
                        ascending = False
                    connected_component = connected_component.sort_values(orient_col, ascending=ascending)
                    obs_names = connected_component.index
                    if orient_rule.startswith("all"):
                        for i in range(len(obs_names)):
                            for j in range(i + 1, len(obs_names)):
                                rows.append(obs_idx[obs_names[i]])
                                cols.append(obs_idx[obs_names[j]])
                                data.append(w)
                    else:
                        for i in range(len(obs_names) - 1):
                            rows.append(obs_idx[obs_names[i]])
                            cols.append(obs_idx[obs_names[i + 1]])
                            data.append(w)
                else:
                    for i, j in orient_rule(connected_component):
                        rows += [obs_idx[i]]
                        cols += [obs_idx[j]]
                        data += [w]
            else:
                for i, j in combinations(connected_component.index, 2):
                    rows += [obs_idx[i], obs_idx[j]]
                    cols += [obs_idx[j], obs_idx[i]]
                    data += [w, w]
                    
    clones.obsp[graph_key_added] = sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(clones.n_obs, clones.n_obs),
    )
    lines = [
        f"added",
        f"     .obsp['{graph_key_added}'] with graph of connected clones",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def correct_shap(
    shapdata: sc.AnnData,
    shap_key: str = "mean_shap",
    corr_key: str = "gex_r",
    correct_sign: bool = True,
    normalize: bool = False,
    key_added: str | None = None,
) -> None:
    """
    Correct SHAP values by dividing by the maximum absolute value of SHAP values for each feature.

    Parameters
    ----------
    shapdata : sc.AnnData
        AnnData object with SHAP values in varm[shap_key].
    shap_key : str, optional
        Key in varm to store SHAP values. Default is "mean_shap".
    corr_key : str, optional
        Key in varm to store correlation values. Default is "gex_r".
    correct_sign : bool, optional
        Whether to correct the sign of SHAP values based on the correlation sign. Default is True.
    normalize : bool, optional
        Whether to normalize SHAP values by dividing by the maximum absolute value for each prediction column.
        Default is True.
    key_added : str, optional
        Key in varm to store normalized SHAP values. Default is "norm_mean_shap".

    Returns
    -------
    None
        Corrected SHAP values are stored in varm[key_added].
    """
    line_before = f"    using varm['{shap_key}'] and varm['{corr_key}']"
    start = logg.info("correcting SHAP values", deep="\n" + line_before)
    if normalize:
        norm_shap = shapdata.varm[shap_key].copy() / shapdata.varm[shap_key].max(axis=0)
    else:
        norm_shap = shapdata.varm[shap_key].copy()
    if correct_sign:
        norm_shap *= np.sign(shapdata.varm[corr_key])

    if key_added is None:
        key_added = shap_key
        if normalize:
            key_added = "norm_" + key_added
        if correct_sign:
            key_added = "signed_" + key_added

    shapdata.varm[key_added] = norm_shap
    lines = ["added", f"     .varm['{key_added}'] corrected SHAP values"]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def impute(
    adata: sc.AnnData,
    obs_name: str,
    value_to_impute: str = "NA",
    use_rep: str = "X_pca",
    weights: Literal["gaussian", "linear"] | None = None,
    classification_obsm: str = "impute_prob",
    key_added: str = "imputed",
    gaus_sigma: float | None = None,
    k: int = 10,
):
    """
    Impute missing values in an observation column using k-nearest neighbors.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with observation values in obs[obs_name].
    obs_name : str
        Name of the observation column to impute.
    value_to_impute : str, optional
        Value in obs_name to impute. Default is "NA".
    use_rep : str, optional
        Key in obsm to use for neighbor search. Default is "X_pca".
    weights : Literal["gaussian", "linear"] | None, optional
        Weighting scheme for imputation. Default is None.
    classification_obsm : str, optional
        Key in obsm to use for classification probabilities. Default is "impute_prob".
    key_added : str, optional
        Key in obs to store imputed values. Default is "imputed".
    gaus_sigma : float | None, optional
        Sigma for Gaussian weighting. If None, use median distance of positive neighbors. Default is None.
    k : int, optional
        Number of neighbors to use for imputation. Default is 10.

    Returns
    -------
    None
        Imputed values are stored in obs[f"{obs_name}_{key_added}"].
    """
    import pynndescent

    line_before = f"\n    using obsm['{use_rep}'] for neighbor search"
    start = logg.info("imputing observation values", deep=line_before)

    labels = adata.obs[obs_name]
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    Xrep = adata.obsm[use_rep]
    X_values = Xrep.values if isinstance(Xrep, pd.DataFrame) else (Xrep.toarray() if sp.issparse(Xrep) else np.asarray(Xrep))

    mask_target = labels.astype(str).values == str(value_to_impute)
    mask_donor = ~mask_target

    if mask_target.sum() == 0:
        adata.obs[f"{obs_name}_{key_added}"] = labels.copy()
        return None
    if mask_donor.sum() == 0:
        raise ValueError("No donor cells available for imputation")

    donors_idx = np.where(mask_donor)[0]
    targets_idx = np.where(mask_target)[0]

    X_donors = X_values[donors_idx]
    X_targets = X_values[targets_idx]

    index = pynndescent.NNDescent(X_donors, n_neighbors=k, metric="euclidean")
    nn_idx, nn_dist = index.query(X_targets, k=k)

    if weights is None:
        W = np.ones_like(nn_dist, dtype=float)
    elif weights == "linear":
        eps = 1e-12
        W = 1.0 / (nn_dist + eps)
    elif weights == "gaussian":
        dpos = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
        if gaus_sigma is None:
            gaus_sigma = np.median(dpos) if dpos.size > 0 else 1.0
        W = np.exp(-(nn_dist ** 2) / (2.0 * gaus_sigma ** 2))
    else:
        raise ValueError("weights must be one of {None, 'gaussian', 'linear'}")

    W[~np.isfinite(W)] = 0.0
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    W = W / row_sum

    is_numeric = pd.api.types.is_numeric_dtype(labels)
    if is_numeric:
        y_donors = pd.to_numeric(labels.values[mask_donor], errors="coerce")
        y_neighbors = y_donors[nn_idx]
        y_pred = (W * y_neighbors).sum(axis=1)
        imputed = labels.astype(float).copy()
        imputed.values[mask_target] = y_pred
        adata.obs[f"{obs_name}_{key_added}"] = imputed
        lines = [
            "added",
            f"     .obs['{obs_name}_{key_added}'] imputed numeric values",
        ]
        logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
    else:
        donor_classes = labels.astype(str).values[mask_donor]
        classes = pd.Index(sorted(pd.unique(donor_classes)))
        prob = np.zeros((adata.n_obs, len(classes)), dtype=float)
        for i_row in range(nn_idx.shape[0]):
            cls_i = donor_classes[nn_idx[i_row]]
            w_i = W[i_row]
            for c, w in zip(cls_i, w_i):
                j = classes.get_indexer([c])[0]
                if j >= 0:
                    prob[targets_idx[i_row], j] += w
        prob_df = pd.DataFrame(prob, index=adata.obs_names, columns=classes)
        adata.obsm[classification_obsm] = prob_df
        pred_idx = np.argmax(prob[targets_idx], axis=1)
        pred_vals = classes.values[pred_idx]
        imputed = labels.astype(str).copy()
        imputed.values[mask_target] = pred_vals
        adata.obs[f"{obs_name}_{key_added}"] = imputed
        lines = [
            "added",
            f"     .obsm['{classification_obsm}'] class probabilities",
            f"     .obs['{obs_name}_{key_added}'] imputed categorical values",
        ]
        logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def _nan_mask(X: np.ndarray | sp.sparse.csr_matrix) -> np.ndarray:
    """
    Create a mask for NaN-contaminated rows in a sparse or dense matrix.

    Parameters
    ----------
    X : np.ndarray or sp.sparse.csr_matrix
        Input matrix.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates NaN-contaminated rows.
    """

    if sp.issparse(X):
        X_csr = X.tocsr()
        indptr = X_csr.indptr
        data = X_csr.data
        if data.size == 0:
            mask = np.zeros(X_csr.shape[0], dtype=bool)
        else:
            rows_for_data = np.repeat(np.arange(X_csr.shape[0]), np.diff(indptr))
            nan_data = np.isnan(data)
            mask = np.zeros(X_csr.shape[0], dtype=bool)
            if nan_data.any():
                mask[np.unique(rows_for_data[nan_data])] = True
        return mask
    else:
        arr = np.asarray(X)
        if arr.dtype.kind == "O":
            mask = pd.isna(arr).any(axis=1)
        else:
            mask = np.isnan(arr).any(axis=1)
        return mask

def gs(
    adata: sc.AnnData,
    use_rep: str = "X_pca",
    batch_key: str | None = None,
    n: float | int = 0.2,
    obs_key: str = "gs",
    random_state: int = 42,
    progress_bar: bool = False,
) -> None:
    """
    Function to perform geometric sketching on an AnnData object. Might be useful to identify test and validation
    sets for model training. If batch_key is provided, the sketching is performed within each batch.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix at the cell level.
    use_rep : str, optional
        Representation to use for geometric sketching, by default "X_pca".
    batch_key : str | None, optional
        Key in `adata.obs` to use for batch-specific sketching, by default None.
    n : float | int, optional
        Either a fraction between 0 and 1 or an integer between 1 and the number of cells to sketch, by default 0.2.
    obs_key : str, optional
        Name of the new column in `adata.obs` to store sketching labels, by default "gs".
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    progress_bar : bool, optional
        Whether to display a progress bar, by default False.

    Returns
    -------
    None
        The `adata` object is modified in place with the new sketching labels stored in `.obs[obs_key]`.
        The parameters used for the sketching are stored in `adata.uns["gs"]`.
    """
    start = logg.info("performing geometric sketching")
    from geosketch import gs
    from tqdm import tqdm
    from contextlib import nullcontext
    import sys

    if use_rep == "X":
        logg.warning("Using 'X' as representation may not be appropriate for geometric sketching.")
        X = adata.X
        # Ensure array-like for geosketch; convert sparse to dense
        if hasattr(X, "toarray"):
            X = X.toarray()
    else:
        try:
            X = adata.obsm[use_rep]
        except KeyError:
            raise KeyError(f"Representation '{use_rep}' not found in adata.obsm. Please provide a valid representation.")

    if obs_key in adata.obs:
        logg.warning(f"obs_key '{obs_key}' already exists in adata.obs. Overwriting.")

    n_cells = adata.shape[0]
    if n > n_cells:
        raise ValueError("n must be less than or equal to the number of cells in adata.")
    elif 1 < n <= n_cells:
        alpha = n / n_cells
    elif 0 < n <= 1:
        alpha = n
    elif n <= 0:
        raise ValueError("n must be a positive value.")

    gs_res = np.array(["full"] * len(adata), dtype=object)

    if sc.settings.verbosity.value >= 3:
        prefix = "    "
    else:
        prefix = ""

    if batch_key:
        cm = tqdm(
            adata.obs[batch_key].cat.categories,
            desc=prefix + "dataset batches",
            file=sys.stdout,
        ) if progress_bar else nullcontext(adata.obs[batch_key].cat.categories)
        with cm as iterator:
            for batch in iterator:
                mask = adata.obs[batch_key] == batch
                X_batch = X[mask]
                n_rows_b = X_batch.shape[0]
                idx_batch = gs(X=X_batch, N=np.round(alpha * n_rows_b).astype(int), seed=random_state)
                batch_rows = np.flatnonzero(mask.to_numpy())
                sel_rows = batch_rows[idx_batch]
                gs_res[sel_rows] = "sketch"
    else:
        n_rows = X.shape[0]
        idx = gs(X=X, N=np.round(alpha * n_rows).astype(int), seed=random_state)
        gs_res[idx] = "sketch"

    adata.obs[obs_key] = gs_res
    adata.uns["gs"] = {
        "use_rep": use_rep,
        "batch_key": batch_key,
        "n": n,
        "obs_key": obs_key,
        "random_state": random_state,
        "keys": {"validation": "sketch", "train": "full"},
    }
    lines = [
        "added",
        f"     .obs['{obs_key}'] geometric sketching labels",
        "     .uns['gs'] parameters",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)