from __future__ import annotations
from typing import Literal
from scanpy import AnnData

import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

logg = sc.logging

__all__ = [
    "clones_adata",
    "make_unique_clones",
    "recalc_composition",
    "transfer_annotation",
    "transfer_expression",
]

def __dir__():
    return sorted(__all__)


def clones_adata(
    adata: AnnData,
    obs_name: str = "clone",
    min_size: int = 3,
    na_value: str = "NA",
    fill_obs: str | None = None,
) -> AnnData:
    """
    Creates a clone-level AnnData object from a cell-level AnnData object.

    This function identifies clones meeting a minimum size requirement and, if specified,
    calculates their cellular composition (e.g., by cell type).

    Parameters
    ----------
    adata : AnnData
        The cell-level annotated data matrix.
    obs_name : str, optional
        Column in `adata.obs` containing clonal information, by default "clone".
    min_size : int, optional
        Minimum clone size to be considered, by default 3.
    na_value : str, optional
        Value indicating the absence of clonal information, by default "NA".
    fill_obs : str | None, optional
        Column in `adata.obs` (e.g., 'cell_type') to calculate clone composition, by default None.

    Returns
    -------
    AnnData
        A new AnnData object where each observation is a clone. The `.X` matrix and
        `.layers['proportions']` store composition proportions, and `.layers['counts']`
        stores raw counts. The `.obs['n_cells']` column contains the number of
        cells per clone.
    """
    # Scanpy-style logging
    start = logg.info("creating clone-level AnnData")

    # 1. Identify valid clones based on min_size
    clonal_obs = adata.obs[obs_name]
    clones_counts = clonal_obs.value_counts()
    
    valid_clone_names = clones_counts[clones_counts >= min_size].index
    valid_clone_names = valid_clone_names.drop(na_value, errors="ignore")

    n_all = len(clones_counts.drop(na_value, errors="ignore"))
    n_valid = len(valid_clone_names)
    logg.info(f"    selected {n_valid} clones (>= {min_size})")

    if valid_clone_names.empty:
        raise ValueError(f"No clones with min_size={min_size} found in the dataset.")

    train_adata = adata[adata.obs[obs_name].isin(valid_clone_names)].copy()
    n_clones = len(valid_clone_names)

    # 2. Calculate cellular composition if fill_obs is provided
    if fill_obs and fill_obs not in adata.obs.columns:
        logg.warning(f"'{fill_obs}' not found in adata.obs. Skipping composition analysis.")
        fill_obs = None

    if fill_obs:
        cell_counts_df = pd.crosstab(train_adata.obs[fill_obs], train_adata.obs[obs_name])
        cell_counts_df = cell_counts_df.reindex(columns=valid_clone_names, fill_value=0)

        var_names = list(cell_counts_df.index)
        cell_counts = cell_counts_df.values
        col_sums = cell_counts.sum(axis=0, keepdims=True)
        freqs = np.divide(cell_counts, col_sums, out=np.zeros_like(cell_counts, dtype=float), where=col_sums!=0)
    else:
        var_names = ["None"]
        cell_counts = np.zeros((1, n_clones), dtype=np.int32)
        freqs = np.zeros((1, n_clones), dtype=np.float32)

    # 3. Create the new AnnData object at the clone level
    clones = AnnData(
        X=freqs.T,
        obs=pd.DataFrame(index=list(valid_clone_names)),
        var=pd.DataFrame(index=var_names),
        layers={"proportions": freqs.T, "counts": cell_counts.T},
        uns={"fill_obs": fill_obs, "min_size": min_size, "obs_name": obs_name},
    )

    # 4. Add n_cells information from the filtered data
    clones.obs["n_cells"] = train_adata.obs[obs_name].value_counts().loc[clones.obs_names].values
    clones.obs["n_fates"] = (clones.layers["proportions"] > 0).sum(axis=1)
    clones.var["n_clones"] = (clones.X > 0).sum(axis=0)

    clones.obs["n_cells"] = clones.obs["n_cells"].astype(int)
    clones.obs["n_fates"] = clones.obs["n_fates"].astype(int)
    clones.var["n_clones"] = clones.var["n_clones"].astype(int)

    # Final logging of added fields and summary (Scanpy style)
    lines = [
        "created clones AnnData with",
        "     .X float matrix of proportions (clones × categories)",
        "     .layers['proportions'] float matrix with fate proportions",
        "     .layers['counts'] integer matrix with fate counts",
        "     .obs['n_cells'] integer vector with number of cells per clone",
        "     .obs['n_fates'] integer vector with number of fates per clone",
        "     .var['n_clones'] integer vector with number of clones per fate",
        "     .uns['fill_obs'] string label",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

    return clones

def make_unique_clones(
    adata: AnnData,
    injection_cols: list[str],
    na_value: str = "NA",
    final_obs_name: str = "clone",
) -> AnnData:
    """
    Prepares a clone2vec-friendly AnnData object by handling multiple clonal labels per cell.
    Duplicates cells with multiple clonal labels into separate rows for each unique clone label.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the clonal labels across multiple columns.
    injection_cols : list[str]
        List of column names in `adata.obs` that contain the clonal labels.
    na_value : str, optional
        The value used to indicate missing or absent clonal information. Default is "NA".
    final_obs_name : str, optional
        The name of the new column in `adata.obs` to store the merged clone labels. Default is "clone".

    Returns
    -------
    AnnData
        A new AnnData object with cells duplicated where necessary, with updated clonal labeling in the `final_obs_name` column.
    """
    start = logg.info("creating unique per-cell clone labels")

    bc_list = []
    clone_obs = []

    for bc, clonal_labels in adata.obs[injection_cols].iterrows():
        labeled_cells = clonal_labels[clonal_labels != na_value]
        if len(labeled_cells) == 0:
            bc_list.append(bc)
            clone_obs.append(na_value)
        else:
            for label, clone in clonal_labels[clonal_labels != na_value].items():
                bc_list.append(bc)
                clone_obs.append(label + "_" + clone)
                
    adata_demult = adata[bc_list]
    adata_demult.obs[final_obs_name] = clone_obs
    adata_demult.obs_names_make_unique()
    lines = [
        "created AnnData with",
        f"     .obs['{final_obs_name}'] categorical labels.",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
    
    return adata_demult

def recalc_composition(
    adata: AnnData,
    clones: AnnData,
    fill_obs: str,
    obs_name: str | None = None,
) -> AnnData:
    """
    Create a new annotated data matrix at the clone level with updated cell type proportions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level.
    clones : AnnData
        Annotated data matrix at the clone level.
    fill_obs : str
        Column in `adata.obs` containing updated cell type / cluster labels.
    obs_name : str | None, optional
        Column in `adata.obs` containing clonal information, if None, try to get from `clones.uns['obs_name']`,
        by default None.

    Returns
    -------
    AnnData
        New annotated data matrix at the clone level with updated cell type proportions.
    """
    # Scanpy-style logging
    start = logg.info("recalculating clone composition")

    if obs_name is None:
        try:
            obs_name = clones.uns["obs_name"]
        except KeyError:
            raise KeyError("obs_name not found in clones.uns. Please provide obs_name.")

    if obs_name not in adata.obs.columns:
        raise KeyError(f"obs_name '{obs_name}' not found in adata.obs. Please provide a valid obs_name.")

    if fill_obs not in adata.obs.columns:
        raise KeyError(f"fill_obs '{fill_obs}' not found in adata.obs. Please provide a valid fill_obs.")
    
    clones_new = adata.obs.groupby(
        [fill_obs, obs_name], observed=False
    ).size().unstack().T.loc[clones.obs_names].fillna(0)
    clones_new.index = list(clones_new.index)
    clones_new = sc.AnnData(clones_new)
    
    clones_new.uns = clones.uns.copy()
    clones_new.uns["fill_obs"] = fill_obs
    # concise logging

    clones_new.obs = clones.obs.copy()
    clones_new.obsm = clones.obsm.copy()
    clones_new.obsp = clones.obsp.copy()
    # concise logging
    
    clones_new.layers["counts"] = clones_new.X.copy()
    clones_new.layers["proportions"] = (clones_new.X.T / clones_new.X.sum(axis=1)).T
    clones_new.X = clones_new.layers["proportions"].copy()
    # concise logging
    
    clones_new.obs["n_fates"] = (clones_new.layers["proportions"] > 0).sum(axis=1)
    clones_new.obs["n_fates"] = clones_new.obs["n_fates"].astype(int)
    clones_new.var["n_clones"] = (clones_new.X > 0).sum(axis=0)
    lines = [
        "created clones AnnData with",
        "     .X float matrix of proportions (clones × categories)",
        "     .layers['proportions'] float matrix with fate proportions",
        "     .layers['counts'] integer matrix with fate counts",
        "     .obs['n_fates'] integer vector with number of fates per clone",
        "     .var['n_clones'] integer vector with number of cells per clone",
        "     .uns['fill_obs'] string label",
        "     .obs and .obsm are copied from the original AnnData",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
    return clones_new

def transfer_annotation(
    adata: AnnData,
    clones: AnnData,
    annotation_obs_clones: str | list[str] | None = None,
    annotation_obs_adata: str | list[str] | None = None,
    created_obs_name: str | list[str] | None = None,
    obs_name: str | None = None,
    na_value: str | None = "NA",
) -> None:
    """
    Transfer clonal labels from a `clones` AnnData object to a `adata` AnnData object, or otherwise.
    The direction of the transfer is from `clones` to `adata` if `annotation_obs_clones` is provided,
    or from `adata` to `clones` if `annotation_obs_adata` is provided.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level.
    clones : AnnData
        Annotated data matrix at the clone level.
    annotation_obs_clones : str | list[str] | None, optional
        Column(s) in `clones.obs` containing clonal labels, by default None.
    annotation_obs_adata : str | list[str] | None, optional
        Name of the new column(s) in `adata.obs` to store transferred clonal labels, by default None.
    obs_name : str | None, optional
        Column in `adata.obs` with the original clonal information, by default None.
    na_value : str | None, optional
        Value to assign to cells with no matching clone in the `clones` object, by default "NA".

    Returns
    -------
    None
        The `adata` or `clones` object is modified in place with the new labels stored in `.obs[created_obs_name]`.
    """
    # Scanpy-style logging
    start = logg.info("transferring annotations between cell and clone objects")

    if obs_name is None:
        try:
            obs_name = clones.uns["obs_name"]
        except KeyError:
            raise KeyError("obs_name not found in clones.uns. Please provide obs_name.")
    if obs_name not in adata.obs:
        raise KeyError(f"obs_name '{obs_name}' not found in adata.obs. Please provide a valid obs_name.")

    if (annotation_obs_clones is None) and (annotation_obs_adata is None):
        raise ValueError("Either annotation_obs_clones or annotation_obs_adata must be provided.")

    if annotation_obs_clones and annotation_obs_adata:
        raise ValueError("Only one of annotation_obs_clones or annotation_obs_adata must be provided.")

    if annotation_obs_clones:
        if isinstance(annotation_obs_clones, str):
            annotation_obs_clones = [annotation_obs_clones]
        if isinstance(created_obs_name, str):
            created_obs_name = [created_obs_name]
        if isinstance(created_obs_name, list) and len(annotation_obs_clones) != len(created_obs_name):
            logg.warning("annotation_obs_clones and created_obs_name must have the same length. Keeping default naming.")
            created_obs_name = None
        if created_obs_name is None:
            created_obs_name = [f"c2v_{col}" for col in annotation_obs_clones]
    else:
        if isinstance(annotation_obs_adata, str):
            annotation_obs_adata = [annotation_obs_adata]
        if isinstance(created_obs_name, str):
            created_obs_name = [created_obs_name]
        if isinstance(created_obs_name, list) and len(annotation_obs_adata) != len(created_obs_name):
            logg.warning("annotation_obs_adata and created_obs_name must have the same length. Keeping default naming.")
            created_obs_name = None
        if created_obs_name is None:
            created_obs_name = ["gex_" + col for col in annotation_obs_adata]

    if annotation_obs_clones:
        for col, obs_col in zip(annotation_obs_clones, created_obs_name):
            clone_mapping = dict(clones.obs[col])
            adata.obs[obs_col] = [
                clone_mapping[clone] if clone in clone_mapping else na_value
                for clone in adata.obs[obs_name]
            ]
            adata.obs[obs_col] = adata.obs[obs_col].astype("category")
        added_target = "adata"
        added_obs_cols = created_obs_name
    else:
        for col, obs_col in zip(annotation_obs_adata, created_obs_name):
            mapping_df = adata.obs[[obs_name, col]].copy()
            mapping_df = mapping_df[mapping_df[obs_name].isin(clones.obs_names)]
            nunique_per_obs = mapping_df.groupby(obs_name, observed=False)[col].nunique()
            problematic = nunique_per_obs[nunique_per_obs > 1]
            if len(problematic) > 0:
                raise ValueError(
                    (
                        f"Non-unique mapping from '{obs_name}' to '{col}'. "
                        f"{len(problematic)} values map to multiple annotations."
                    )
                )

            clone_to_annot = mapping_df.groupby(obs_name, observed=False)[col].first()
            clones.obs[obs_col] = [
                clone_to_annot[clone] if clone in clone_to_annot.index else na_value
                for clone in clones.obs_names
            ]
            clones.obs[obs_col] = clones.obs[obs_col].astype("category")
        added_target = "clones"
        added_obs_cols = created_obs_name
    lines = ["added to clonal AnnData"]
    for obs_col in added_obs_cols:
        if added_target == "adata":
            lines.append(f"     .obs['{obs_col}'] categorical labels")
        else:
            lines.append(f"     .obs['{obs_col}'] categorical labels")
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def transfer_expression(
    adata: AnnData,
    clones: AnnData,
    obs_name: str | None = None,
    split_obs: str | None = None,
    use_col: str | None = None,
    strategy: Literal["sum", "average"] = "average",
    layer: str | None = None,
    use_raw: bool | None = None,
    layers_to_obsm: list[str] | str | None = None,
    mask_key: str = "mask",
) -> AnnData:
    """
    Summarize gene expression at the clone level using a specified strategy (sum or average).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level with gene expression data.
    clones : AnnData
        Annotated data matrix at the clone level.
    obs_name : str, optional
        Column name in `adata.obs` containing clonal information. If not provided,
        the value from `clones.uns["obs_name"]` is used.
    split_obs : str | None, optional
        Column name in `adata.obs` to split the data for aggregation. If None,
        the entire dataset is used for aggregation, default None.
    strategy : Literal["sum", "average"], optional
        Strategy for aggregating gene expression ("sum" or "average"), by default "average".
    layer : str | None, optional
        If specified, summarizes expression from `adata.layers[layer]`, by default None.
    use_raw : bool | None, optional
        If specified, uses `adata.raw.X` for summarization, by default None.
    layers_to_obsm : list[str] | str | None, optional
        List of layers in `clones.layers` to transfer to `clones_expr.obsm`.
        If None, all layers with `adata.X` are transferred, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix at the clone level with summarized gene expression. 
        The strategy used for aggregation is stored in `clones.uns["transfer_expression"]["strategy"]`.
        If `split_obs` is provided, the aggregation is performed within each group defined by `split_obs` and stored
        in `clones.layers[g]` where `g` is the group name. Clones without a matching group in `split_obs` contain
        NaN values in this layer if `absent_expression` is "nan", otherwise they contain zero expression.
    """
    if obs_name is None:
        try:
            obs_name = clones.uns["obs_name"]
        except KeyError:
            raise KeyError("obs_name not found in clones.uns. Please provide obs_name.")

    if obs_name not in adata.obs:
        raise KeyError(f"obs_name '{obs_name}' not found in adata.obs. Please provide a valid obs_name.")

    if use_col:
        if use_col not in clones.obs.columns:
            raise KeyError(f"use_col '{use_col}' not found in clones.obs. Please provide a valid use_col or set use_col to None.")

    if use_raw is None and layer is None:
        if adata.raw is None:
            use_raw = False
        else:
            use_raw = True
    elif use_raw and not (layer is None):
        logg.warning(f"Can't use both `adata.raw` and `adata.layers['{layer}']`. Using `adata.raw.X` as expression source.")
        use_raw = True

    if strategy not in ("average", "sum"):
        logg.warning(f"Only average and sum methods of expression aggregation are supported, got '{strategy}'")
        strategy = "average"

    if split_obs and (split_obs not in adata.obs.columns):
        raise KeyError(f"split_obs '{split_obs}' not found in adata.obs. Please provide a valid split_obs or set split_obs to None.")

    if not (layer is None):
        X = adata.layers[layer]
        var_names = adata.var_names
        line_before = f"\n    using `adata.layers['{layer}']` as expression source"
    elif use_raw:
        X = adata.raw.X
        var_names = adata.raw.var_names
        line_before = "\n    using `adata.raw.X` as expression source"
    else:
        X = adata.X
        var_names = adata.var_names
        line_before = "\n    using `adata.X` as expression source"

    start = logg.info("summarizing expression at clone level", deep=line_before)

    n_cells = adata.n_obs
    n_clones = clones.n_obs
    if use_col:
        categories = list(pd.unique(clones.obs[use_col]))
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        clone_cat_idx = np.array([cat_to_idx.get(v, -1) for v in clones.obs[use_col].values], dtype=np.int64)
        clone_to_cat_idx = dict(zip(clones.obs_names, clone_cat_idx))
        adata_vals = adata.obs[obs_name].values
        codes_direct = np.array([cat_to_idx.get(v, -1) for v in adata_vals], dtype=np.int64)
        if np.any(codes_direct != -1):
            codes_fallback = np.array([clone_to_cat_idx.get(v, -1) for v in adata_vals], dtype=np.int64)
            use_direct_mask = codes_direct != -1
            codes = np.where(use_direct_mask, codes_direct, codes_fallback)
        else:
            codes = np.array([clone_to_cat_idx.get(v, -1) for v in adata_vals], dtype=np.int64)
    else:
        categories = list(clones.obs_names)
        labels = pd.Categorical(adata.obs[obs_name].values, categories=categories)
        codes = labels.codes
    valid_mask = codes != -1
    rows = np.where(valid_mask)[0]
    cols = codes[valid_mask]

    if use_col:
        n_categories = len(categories)
        if sp.issparse(X):
            data = np.ones(rows.shape[0], dtype=np.uint8)
            Bc = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_categories))
            X_sum_cat = Bc.T @ X
            counts_cat = np.asarray(Bc.sum(axis=0)).ravel().astype(np.float32)
        else:
            counts_cat = np.bincount(cols, minlength=n_categories).astype(np.float32)
            X_sum_cat = np.zeros((n_categories, X.shape[1]), dtype=X.dtype)
            np.add.at(X_sum_cat, cols, X[rows, :])

        if strategy == "average":
            inv_counts_cat = np.divide(1.0, counts_cat, out=np.zeros_like(counts_cat), where=counts_cat != 0)
            if sp.issparse(X_sum_cat):
                X_agg_cat = sp.diags(inv_counts_cat) @ X_sum_cat
            else:
                X_agg_cat = inv_counts_cat[:, None] * X_sum_cat
        else:
            X_agg_cat = X_sum_cat

        valid_clone_mask = clone_cat_idx != -1
        if sp.issparse(X_agg_cat):
            rows_list = [X_agg_cat.getrow(clone_cat_idx[i]) if valid_clone_mask[i] else sp.csr_matrix((1, X_agg_cat.shape[1]), dtype=X_agg_cat.dtype) for i in range(n_clones)]
            X_agg = sp.vstack(rows_list)
        else:
            X_agg = np.zeros((n_clones, X_agg_cat.shape[1]), dtype=X_agg_cat.dtype)
            X_agg[valid_clone_mask, :] = X_agg_cat[clone_cat_idx[valid_clone_mask], :]
        counts = np.zeros(n_clones, dtype=np.float32)
        counts[valid_clone_mask] = counts_cat[clone_cat_idx[valid_clone_mask]]
    else:
        if sp.issparse(X):
            data = np.ones(rows.shape[0], dtype=np.uint8)
            B = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_clones))
            X_sum = B.T @ X
            counts = np.asarray(B.sum(axis=0)).ravel().astype(np.float32)
        else:
            counts = np.bincount(cols, minlength=n_clones).astype(np.float32)
            X_sum = np.zeros((n_clones, X.shape[1]), dtype=X.dtype)
            np.add.at(X_sum, cols, X[rows, :])

        if strategy == "average":
            inv_counts = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts != 0)
            if sp.issparse(X_sum):
                X_agg = sp.diags(inv_counts) @ X_sum
            else:
                X_agg = inv_counts[:, None] * X_sum
        else:
            X_agg = X_sum

    clones_expr = AnnData(
        X=X_agg,
        obs=clones.obs.copy(),
        var=pd.DataFrame(index=var_names),
        obsm=clones.obsm.copy(),
        obsp=clones.obsp.copy(),
        uns=clones.uns.copy(),
    )
    clones_expr.uns["transfer_expression"] = {"strategy": strategy}

    if layers_to_obsm is None:
        layers_to_obsm = ["X"] + list(clones.layers.keys())
    elif isinstance(layers_to_obsm, str):
        layers_to_obsm = [layers_to_obsm]
    layers_to_obsm = np.array(layers_to_obsm)
    if "X" in layers_to_obsm:
        clones_expr.obsm["X"] = clones.to_df()
        clones_expr.obsm["X"].index = list(clones_expr.obs_names)
        clones_expr.obsm["X"].columns = clones.var_names
    layers_to_obsm = layers_to_obsm[layers_to_obsm != "X"]
    for layer in layers_to_obsm:
        layer = str(layer)
        try:
            clones_expr.obsm[layer] = clones.to_df(layer=layer)
            clones_expr.obsm[layer].index = list(clones_expr.obs_names)
            clones_expr.obsm[layer].columns = clones.var_names
        except ValueError:
            logg.warning(f"Layer '{layer}' not found in clones.obs. Skipping.")
    if isinstance(layers_to_obsm, np.ndarray):
        lt = list(layers_to_obsm)
    else:
        lt = layers_to_obsm
    
    if split_obs is not None:
        groups = pd.unique(adata.obs[split_obs])
        groups = [g for g in groups if pd.notna(g)]
        zero_df = {}

        for g in groups:
            group_mask = (adata.obs[split_obs].values == g) & valid_mask
            g_rows = np.where(group_mask)[0]
            codes_g = codes[g_rows]

            if use_col:
                n_categories = len(categories)
                if sp.issparse(X):
                    data_g = np.ones(g_rows.shape[0], dtype=np.uint8)
                    B_gc = sp.csr_matrix((data_g, (np.arange(g_rows.size), codes_g)), shape=(g_rows.size, n_categories))
                    X_g = X[g_rows, :] if hasattr(X, "__getitem__") else X
                    X_sum_g_cat = B_gc.T @ X_g
                    counts_g_cat = np.asarray(B_gc.sum(axis=0)).ravel().astype(np.float32)
                else:
                    X_g = X[g_rows, :]
                    counts_g_cat = np.bincount(codes_g, minlength=n_categories).astype(np.float32)
                    X_sum_g_cat = np.zeros((n_categories, X.shape[1]), dtype=X.dtype)
                    np.add.at(X_sum_g_cat, codes_g, X_g)
                if strategy == "average":
                    inv_counts_g_cat = np.divide(1.0, counts_g_cat, out=np.zeros_like(counts_g_cat), where=counts_g_cat != 0)
                    if sp.issparse(X_sum_g_cat):
                        X_agg_g_cat = sp.diags(inv_counts_g_cat) @ X_sum_g_cat
                    else:
                        X_agg_g_cat = inv_counts_g_cat[:, None] * X_sum_g_cat
                else:
                    X_agg_g_cat = X_sum_g_cat
                valid_clone_mask = clone_cat_idx != -1
                if sp.issparse(X_agg_g_cat):
                    rows_list_g = [X_agg_g_cat.getrow(clone_cat_idx[i]) if valid_clone_mask[i] else sp.csr_matrix((1, X_agg_g_cat.shape[1]), dtype=X_agg_g_cat.dtype) for i in range(n_clones)]
                    X_agg_g = sp.vstack(rows_list_g)
                else:
                    X_agg_g = np.zeros((n_clones, X_agg_g_cat.shape[1]), dtype=X_agg_g_cat.dtype)
                    X_agg_g[valid_clone_mask, :] = X_agg_g_cat[clone_cat_idx[valid_clone_mask], :]
                counts_g = np.zeros(n_clones, dtype=np.float32)
                counts_g[valid_clone_mask] = counts_g_cat[clone_cat_idx[valid_clone_mask]]
            else:
                if sp.issparse(X):
                    data_g = np.ones(g_rows.shape[0], dtype=np.uint8)
                    B_g = sp.csr_matrix((data_g, (np.arange(g_rows.size), codes_g)), shape=(g_rows.size, n_clones))
                    X_g = X[g_rows, :] if hasattr(X, "__getitem__") else X
                    X_sum_g = B_g.T @ X_g
                    counts_g = np.asarray(B_g.sum(axis=0)).ravel().astype(np.float32)
                else:
                    X_g = X[g_rows, :]
                    counts_g = np.bincount(codes_g, minlength=n_clones).astype(np.float32)
                    X_sum_g = np.zeros((n_clones, X.shape[1]), dtype=X.dtype)
                    np.add.at(X_sum_g, codes_g, X_g)
                if strategy == "average":
                    inv_counts_g = np.divide(1.0, counts_g, out=np.zeros_like(counts_g), where=counts_g != 0)
                    if sp.issparse(X_sum_g):
                        X_agg_g = sp.diags(inv_counts_g) @ X_sum_g
                    else:
                        X_agg_g = inv_counts_g[:, None] * X_sum_g
                else:
                    X_agg_g = X_sum_g

            mask_nonzero = (counts_g > 0)
            if sp.issparse(X_agg_g):
                clones_expr.layers[g] = X_agg_g.tocsr()
            else:
                X_agg_g[~mask_nonzero, :] = np.nan
                clones_expr.layers[g] = X_agg_g
            zero_df[g] = mask_nonzero

        clones_expr.obsm[mask_key] = pd.DataFrame(zero_df, index=clones_expr.obs_names)
        clones_expr.uns["mask_key"] = mask_key
        clones_expr.uns["transfer_expression"]["split_obs"] = split_obs

    nonzero_clones = np.asarray(clones_expr.X.sum(axis=1)).ravel() > 0
    if sum(~nonzero_clones) > 0:
        logg.warning(f"Removing {sum(~nonzero_clones)} clones with no overall expression")
        clones_expr = clones_expr[nonzero_clones].copy()

    lines = [
        "created clone-level AnnData with",
        "     .X aggregated expression matrix (clones × genes)",
        "     .uns['transfer_expression'] dict with 'strategy'",
        f"     .uns['mask_key'] string with obsm key for boolean mask of clones with non-zero cells",
    ]
    if 'lt' in locals() and lt is not None:
        if "X" in lt:
            lines.append("     .obsm['X'] dataframe of features")
        for layer_name in [l for l in lt if l != "X"]:
            lines.append(f"     .obsm['{layer_name}'] dataframe of features")
    if split_obs is not None:
        lines.append(f"     .layers['<groups>'] aggregated matrices per '{split_obs}' group")
        lines.append(f"     .obsm['{mask_key}'] boolean mask of clones with zero cells")
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

    return clones_expr
