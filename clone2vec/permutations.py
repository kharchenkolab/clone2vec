from __future__ import annotations

import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys

from scipy import linalg, interpolate
from contextlib import nullcontext
from tqdm import tqdm
from typing import Literal

from .utils import permute_clones, _silenced

logg = sc.logging

tw_cdf_tabular = {
    -9.0: 0.000000e+00,
    -8.0: 0.000000e+00,
    -7.0: 0.000000e+00,
    -6.0: 1.28e-09,
    -5.0: 1.59e-06,
    -4.0: 0.00134,
    -3.5: 0.00603,
    -3.0: 0.01616,
    -2.5: 0.05590,
    -2.0: 0.14932,
    -1.5: 0.30771,
    -1.0: 0.51085,
    -0.5: 0.70612,
    0.0: 0.85172,
    0.5: 0.93722,
    1.0: 0.97720,
    1.5: 0.99284,
    2.0: 0.99802,
    2.5: 0.99951,
    3.0: 0.99989,
}

_TW1_X = np.array(sorted(tw_cdf_tabular))
_TW1_F = np.array([tw_cdf_tabular[x] for x in _TW1_X])

_tw1_interp = interpolate.PchipInterpolator(_TW1_X, _TW1_F, extrapolate=False)
_TW1_SPLICE = float(_TW1_X[-1])

def _tw1_tail_shape(x: float) -> float:
    """
    Leading-order shape of the beta = 1 Tracy-Widom right tail.

    Parameters
    ----------
    x : float
        Value at which to evaluate the tail, must be positive.

    Returns
    -------
    float
        The unnormalised tail shape at x.
    """
    return x ** -0.75 * np.exp(-2.0 / 3.0 * x ** 1.5)

_TW1_TAIL_C = (1.0 - _TW1_F[-1]) / _tw1_tail_shape(_TW1_SPLICE)

def _tracy_widom_cdf(x: float) -> float:
    """
    Computes the CDF of the Tracy-Widom distribution (with beta = 1).

    The result is an approximation: tabulated values are interpolated below
    x = 3 and extended by a calibrated leading-order tail above it. It is
    continuous and monotone, but its absolute accuracy in the far tail is
    limited by the tabulated values it is anchored to.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF.

    Returns
    -------
    float
        The value of the CDF at x.
    """
    if x <= _TW1_X[0]:
        return 0.0
    if x < _TW1_SPLICE:
        return float(np.clip(_tw1_interp(x), 0.0, 1.0))
    return float(1.0 - _TW1_TAIL_C * _tw1_tail_shape(x))

def _calculate_statistics(X: np.ndarray) -> float:
    """
    Calculates the test statistic for the Tracy-Widom distribution.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n, p) containing the data.

    Returns
    -------
    float
        The value of the test statistic.
    """
    n, p = X.shape
    sd = np.std(X, axis=0)
    if np.any(sd == 0):
        raise ValueError(
            "zero-variance embedding dimensions: "
            f"{np.flatnonzero(sd == 0).tolist()}"
        )
    X_centered = (X - np.mean(X, axis=0)) / sd
    n_dof = n - 1

    # centering caps the rank at min(n - 1, p); any further singular value is a
    # structural zero and would deflate the bulk variance estimate
    rank = min(n_dof, p)
    if rank < 2:
        raise ValueError(
            f"at least two non-degenerate dimensions are required, got {rank}"
        )

    s = linalg.svdvals(X_centered)
    eigenvalues = (s ** 2)[:rank]
    lambda_1 = eigenvalues[0]
    sigma_sq_est = np.mean(eigenvalues[1:]) / n_dof

    mu_np = (np.sqrt(n_dof) + np.sqrt(p)) ** 2
    sigma_np = (np.sqrt(n_dof) + np.sqrt(p)) * \
               ((1 / np.sqrt(n_dof)) + (1 / np.sqrt(p))) ** (1/3)
            
    stat = ((lambda_1 / sigma_sq_est) - mu_np) / sigma_np
    return stat

def spectral_test(
    clones: sc.AnnData | np.ndarray,
    flavor: Literal["permutations", "synthetic", "asymptotic"] = "permutations",
    key: str | None = None,
    permutations_key: str = "clonal_permutations",
    key_added: str = "spectral_test",
    n_simulations: int = 10000,
    null_distribution: np.ndarray | None = None,
    progress_bar: bool = True,
    random_state: int | None = 0,
):
    """
    Tests whether a clonal embedding carries more structure than expected under the null
    using the leading eigenvalue statistic of Johnstone's spiked covariance test.

    With the "permutations" flavor, the statistic of the existing embedding (the one
    `permute_embeddings` inherited the parameters from, e.g. `clones.obsm['clone2vec']`) is
    compared with the null distribution of the statistic over the permuted embeddings calculated
    by `permute_embeddings`, that is with the structure the clonal labelling of the gene
    expression space alone would produce. With the "synthetic" flavor, the null
    distribution is simulated from Gaussian matrices of the same shape, and with the
    "asymptotic" flavor the p-value is taken from the Tracy-Widom distribution.

    Parameters
    ----------
    clones : sc.AnnData | np.ndarray
        Annotated data matrix at the clone level (with the results of `permute_embeddings`
        for the "permutations" flavor), or a numpy array with the embedding for the
        "synthetic" and "asymptotic" flavors.
    flavor : Literal["permutations", "synthetic", "asymptotic"], optional
        Source of the null distribution, by default "permutations".
    key : str | None, optional
        Key in `clones.obsm` with the embedding, used for AnnData input with the
        "synthetic" and "asymptotic" flavors, by default None.
    permutations_key : str, optional
        `key_added` used in `permute_embeddings`, used with the "permutations" flavor,
        by default "clonal_permutations".
    key_added : str, optional
        Key in `clones.uns` to store the results, by default "spectral_test".
    n_simulations : int, optional
        Number of simulations for the "synthetic" flavor, by default 10000.
    null_distribution : np.ndarray | None, optional
        Precomputed null distribution for the "synthetic" flavor, by default None.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    random_state : int | None, optional
        Seed used to generate the null distribution for the "synthetic" flavor,
        by default 0.

    Returns
    -------
    dict | None
        For AnnData input, `clones.uns[key_added]` stores the test statistic, the p-value,
        the null distribution (None for the "asymptotic" flavor), and the flavor; for
        numpy array input, they are returned as a dictionary.
    """
    if isinstance(clones, sc.AnnData):
        if flavor == "permutations":
            if permutations_key not in clones.uns:
                raise KeyError(
                    f"permutations_key '{permutations_key}' not found in clones.uns. "
                    "Did you run `c2v.tl.permute_embeddings(adata, clones)` before?"
                )
            X = clones.obsm[clones.uns[permutations_key]["use_rep"]]
        elif key is None:
            raise ValueError("key must be specified for AnnData input")
        else:
            X = clones.obsm[key]
    elif isinstance(clones, np.ndarray):
        if flavor == "permutations":
            raise ValueError("'permutations' flavor requires AnnData input")
        X = clones
        key_added = None
    else:
        raise ValueError("clones must be AnnData or numpy.ndarray")

    start = logg.info(f"computing spectral test via {flavor} approach")

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    stat = _calculate_statistics(X)
    if flavor == "permutations":
        embeddings = clones.obsm[permutations_key]
        n_permutations = embeddings.shape[-1]
        null_distribution = []
        for i in range(n_permutations):
            try:
                null_distribution.append(_calculate_statistics(embeddings[..., i]))
            except ValueError:
                # a degenerate embedding is uninformative rather than fatal here
                null_distribution.append(np.nan)
        null_distribution = np.asarray(null_distribution, dtype=float)
        finite = np.isfinite(null_distribution)
        if not finite.all():
            logg.warning(
                f"{np.sum(~finite)} of {n_permutations} permutations resulted in a degenerate "
                "embedding and are excluded from the null distribution"
            )
        if not finite.any():
            raise ValueError("all permutations resulted in a degenerate embedding")
        p_value = float(
            (1.0 + np.sum(null_distribution[finite] >= stat)) / (finite.sum() + 1.0)
        )
    elif flavor == "asymptotic":
        p_value = 1.0 - _tracy_widom_cdf(stat)
    elif flavor == "synthetic":
        if null_distribution is None:
            rng = np.random.default_rng(random_state)
            null_distribution = []
            n, p = X.shape
            cm = tqdm(
                range(n_simulations),
                desc=prefix + f"generating null distribution ({n_simulations} simulations)",
                file=sys.stdout,
            ) if progress_bar else nullcontext(range(n_simulations))
            with cm as iterator:
                for i in iterator:
                    X_sim = rng.standard_normal((n, p))
                    null_distribution.append(_calculate_statistics(X_sim))
        null_distribution = np.asarray(null_distribution, dtype=float)
        p_value = float(
            (1.0 + np.sum(null_distribution >= stat)) / (null_distribution.size + 1.0)
        )
    else:
        raise ValueError("flavor must be 'permutations', 'synthetic', or 'asymptotic'")

    results = {
        "stat": stat,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "flavor": flavor,
    }
    if not key_added:
        logg.info("    finished ({time_passed})", time=start)
        return results

    if flavor == "permutations":
        results["method"] = clones.uns[permutations_key]["method"]
        results["use_rep"] = clones.uns[permutations_key]["use_rep"]
        if "batch_key" in clones.uns[permutations_key]:
            results["batch_key"] = clones.uns[permutations_key]["batch_key"]
    clones.uns[key_added] = results
    lines = [
        "added",
        f"     .uns['{key_added}'] eigenvalues test statistics, null distribution, and p-value",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)


# `.uns['clonal_nn']` keeps a summary of the graph, not only constructor arguments
_CLONAL_NN_ARGS = ("k", "use_rep", "obsp_name", "split_by", "random_state")

def permute_embeddings(
    adata: sc.AnnData,
    clones: sc.AnnData,
    batch_key: str | None = None,
    obs_name: str | None = None,
    keep_labels: list[str] | str | None = None,
    method: Literal["poisson", "multinomial"] | None = None,
    n_permutations: int = 100,
    random_state : None | int = 4,
    progress_bar: bool = True,
    key_added: str = "clonal_permutations",
    clonal_nn_kwargs: dict | None = None,
    clone2vec_kwargs: dict | None = None,
    mask: str | None = None,
) -> None:
    """
    Calculates clonal embeddings of clonally reshuffled data to be used as a permutation
    null by the downstream statistics (`spectral_test`, `permuted_graph_associations`).

    The pipeline that produced the existing clonal embedding (`clonal_nn` followed by
    `clone2vec` or `clone2vec_Poi`, with the parameters stored in `clones.uns`) is re-run
    on clonally reshuffled data. Labels are reshuffled by `c2v.utils.permute_clones`, that
    is (a) only between the cells carrying clonal information (including the clones not
    present in `clones`, e.g., the ones smaller than `min_size`) and (b) only within one
    batch, therefore clone sizes and, if `batch_key` is provided, batch composition of the
    clones are preserved in every permutation – a batch effect, if it exists, stays in the
    resulting clonal embeddings. The existing embedding (`clones.obsm['clone2vec']` or
    `clones.obsm['clone2vec_Poi']`) is used as the observed one by the downstream statistics.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix at the cell level, containing the representation used by
        `clonal_nn` in `.obsm`.
    clones : sc.AnnData
        Annotated data matrix at the clone level with the results of `clonal_nn` and
        `clone2vec` (or `clone2vec_Poi`).
    batch_key : str | None, optional
        Column in `adata.obs` to shuffle clonal labels within, if None, labels are
        shuffled across the whole dataset, by default None.
    obs_name : str | None, optional
        Column in `adata.obs` containing clonal information, if None, try to get from
        `clones.uns['obs_name']`, by default None.
    keep_labels : list[str] | str | None, optional
        Labels to keep in place, if None, try to get the label of the cells without
        clonal information from `clones.uns['na_value']`, by default None.
    method : Literal["poisson", "multinomial"] | None, optional
        Whether to embed clones with `clone2vec_Poi` ("poisson") or `clone2vec`
        ("multinomial"). If None, the model of the existing embedding is used ("multinomial"
        with a warning if both exist), by default None.
    n_permutations : int, optional
        Number of clonal permutations, by default 100.
    random_state : None | int, optional
        Random state for reproducibility, also passed to the embedding functions, by default 4.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    key_added : str, optional
        Key in `clones.obsm` and `clones.uns` to store the results, by default
        "clonal_permutations".
    clonal_nn_kwargs : dict | None, optional
        Keyword arguments for `clonal_nn` overriding the ones stored in
        `clones.uns['clonal_nn']` (including the masks, which stay bound to the cells),
        by default None.
    clone2vec_kwargs : dict | None, optional
        Keyword arguments for `clone2vec` or `clone2vec_Poi` overriding the ones stored in
        `clones.uns['clone2vec']['init']` or `clones.uns['clone2vec_Poi']['init']`,
        by default None.
    mask : str | None, optional
        `key_added` of `tl.mask_heterogeneity`. If provided, the parameters are taken from
        `clones.uns[mask]` (`clonal_nn` with the masks, and `clone2vec` or `clone2vec_Poi`
        refitted with `mask_key="masked_nn"` and `init="random"`), and
        `clones.obsm['{c2v_key}_{mask}']` is used as the observed embedding, by default None.

    Returns
    -------
    None
        `clones.obsm[key_added]` stores the permuted embeddings (clones × dimensions ×
        permutations), and `clones.uns[key_added]` stores the key of the observed embedding
        (`use_rep`), the permuted labels as integer codes (`codes`, cells × permutations,
        -1 for missing labels) together with the cell names (`cells`), the label
        categories, and the parameters. `adata` is not modified.
    """
    # Imported here, `tools` imports this module itself
    from .tools import clonal_nn
    from .embeddings import clone2vec, clone2vec_Poi

    hint = (
        "`permute_embeddings` inherits the parameters of the existing clonal embedding, so "
        "`clonal_nn` and `clone2vec` (or `clone2vec_Poi`) have to be run on `clones` beforehand."
    )
    if mask is None:
        src, suffix = clones.uns, ""
    else:
        if mask not in clones.uns:
            raise KeyError(f"clones.uns['{mask}'] not found. Run tl.mask_heterogeneity with key_added='{mask}' first.")
        src, suffix = clones.uns[mask], f"_{mask}"
    if "clonal_nn" not in src:
        raise ValueError(f"Clone-to-clone adjacency graph parameters not found in `clones.uns['clonal_nn']`. {hint}")
    found = [k for k in ("clone2vec", "clone2vec_Poi") if k in src and k + suffix in clones.obsm]
    if method is None:
        if not found:
            raise ValueError(
                "Clonal embedding not found (missing `clones.obsm['clone2vec']`/`clones.uns['clone2vec']` "
                f"or their `clone2vec_Poi` counterparts). {hint}"
            )
        if len(found) == 2:
            logg.warning(
                "Both 'clone2vec' and 'clone2vec_Poi' embeddings found, using 'clone2vec' "
                "(multinomial model). Pass `method='poisson'` to use 'clone2vec_Poi'."
            )
        key = found[0]
    elif method in ("multinomial", "poisson"):
        key = "clone2vec" if method == "multinomial" else "clone2vec_Poi"
        if key not in found:
            raise ValueError(f"Embedding '{key}' (method='{method}') not found in clones.obsm/clones.uns. {hint}")
    else:
        raise ValueError("method must be 'poisson', 'multinomial', or None")
    method = "multinomial" if key == "clone2vec" else "poisson"
    embedding = clone2vec if key == "clone2vec" else clone2vec_Poi

    if obs_name is None:
        try:
            obs_name = clones.uns["obs_name"]
        except KeyError:
            raise KeyError("obs_name not found in clones.uns. Please provide obs_name.")
    if obs_name not in adata.obs:
        raise KeyError(f"obs_name '{obs_name}' not found in adata.obs. Please provide a valid obs_name.")
    if (batch_key is not None) and (batch_key not in adata.obs):
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs. Please provide a valid batch_key.")
    if keep_labels is None:
        keep_labels = clones.uns.get("na_value", "NA")
    if isinstance(keep_labels, str):
        keep_labels = [keep_labels]

    stored = src["clonal_nn"]
    nn_kwargs = {k: v for k, v in stored.items() if k in _CLONAL_NN_ARGS}
    if "mask" in stored:
        # masks are bound to the cells, therefore stay in place under permutations
        nn_kwargs["mask_obs"] = [
            pd.Series(m, index=adata.obs_names, name=n) for n, m in stored["mask"].items()
        ]
    nn_kwargs.update(clonal_nn_kwargs or {})
    clonal_nn_kwargs = nn_kwargs
    clonal_nn_kwargs.setdefault("random_state", random_state)
    mask_obs = clonal_nn_kwargs.get("mask_obs")
    if isinstance(mask_obs, (np.ndarray, pd.Series)):
        mask_obs = [mask_obs]
    mask_names = [getattr(m, "name", None) or f"mask_{i+1}" for i, m in enumerate(mask_obs or [])]

    emb_kwargs = dict(src[key].get("init", {}))
    if mask is not None:
        emb_kwargs.update({"mask_key": "masked_nn", "init": "random"})
    emb_kwargs.update(clone2vec_kwargs or {})
    clone2vec_kwargs = emb_kwargs
    clone2vec_kwargs.pop("progress_bar", None)
    if "n_pcs" in clone2vec_kwargs:
        clone2vec_kwargs["z_dim"] = clone2vec_kwargs.pop("n_pcs")
    clone2vec_kwargs.setdefault("obsp_key", clonal_nn_kwargs.get("obsp_name", "gex_adjacency"))
    clone2vec_kwargs.setdefault("random_state", random_state)
    obsm_key = clone2vec_kwargs.get("obsm_key", key)

    use_rep = clonal_nn_kwargs.get("use_rep", "X_pca")
    if use_rep not in adata.obsm:
        raise KeyError(f"Representation '{use_rep}' not found in adata.obsm.")

    lines_before = [
        f"    using `clones.obsm['{key + suffix}']` as observed embedding",
        f"    embedding permuted clones with '{method}' flavor",
        f"    shuffling '{obs_name}' labels" + (f" within '{batch_key}'" if batch_key else ""),
        f"    masking kNN-graph by {mask_names}" if mask_names else None,
    ]
    start = logg.info(
        "computing clonal permutations",
        deep="\n" + "\n".join([l for l in lines_before if l is not None]),
    )

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    # Minimal copies to keep the permutations away from the objects of the user
    obs_keys = list(dict.fromkeys(
        [obs_name] + [k for k in (batch_key, clonal_nn_kwargs.get("split_by")) if k is not None]
    ))
    adata_perm = sc.AnnData(
        obs=adata.obs[obs_keys].copy(),
        obsm={use_rep: adata.obsm[use_rep]},
    )
    clones_perm = sc.AnnData(obs=pd.DataFrame(index=clones.obs_names))

    # Clonal permutations with respect to the batch key, the labels listed in
    # `keep_labels` (by default, of the cells without clonal information) are kept in place
    labels = adata.obs[obs_name]
    batch = adata.obs[batch_key] if batch_key is not None else None
    categories = pd.Index(labels.dropna().unique())
    if batch is not None:
        n_kept = int((~labels.isin(keep_labels) & labels.notna() & batch.isna()).sum())
        if n_kept > 0:
            logg.warning(
                f"{n_kept} labeled cells have no '{batch_key}' value, "
                "their clonal labels are kept in place"
            )

    rng = np.random.default_rng(random_state)
    codes, embeddings = [], []
    cm = tqdm(
        range(n_permutations),
        desc=prefix + f"permuting clonal labels ({n_permutations} permutations)",
        file=sys.stdout,
    ) if progress_bar else nullcontext(range(n_permutations))
    with _silenced(), cm as iterator:
        for i in iterator:
            permuted = permute_clones(
                labels,
                batch=batch,
                keep_labels=keep_labels,
                random_state=rng,
            ).to_numpy()
            adata_perm.obs[obs_name] = permuted

            clonal_nn(adata_perm, clones_perm, obs_name=obs_name, **clonal_nn_kwargs)
            embedding(clones_perm, progress_bar=False, **clone2vec_kwargs)
            codes.append(categories.get_indexer(permuted))
            embeddings.append(clones_perm.obsm[obsm_key].copy())

    clones.obsm[key_added] = np.stack(embeddings, axis=-1)
    clones.uns[key_added] = {
        "use_rep": key + suffix,
        "cells": np.asarray(adata.obs_names.astype(str)),
        "codes": np.stack(codes, axis=1).astype(np.int32),
        "categories": np.asarray(categories.astype(str)),
        "obs_name": obs_name,
        "keep_labels": list(keep_labels),
        "method": method,
        "n_permutations": n_permutations,
        "clonal_nn": {k: v for k, v in clonal_nn_kwargs.items() if k != "mask_obs" and v is not None},
        "init": clone2vec_kwargs,
    }
    if batch_key is not None:
        clones.uns[key_added]["batch_key"] = batch_key
    if random_state is not None:
        clones.uns[key_added]["random_state"] = random_state
    if mask_names:
        clones.uns[key_added]["clonal_nn"]["mask"] = mask_names

    lines = [
        "added",
        f"     .obsm['{key_added}'] permuted clonal embeddings",
        f"     .uns['{key_added}'] permuted clonal labels of cells, label categories, and parameters",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
