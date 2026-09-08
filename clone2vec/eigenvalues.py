from __future__ import annotations

import scanpy as sc
import numpy as np
import pandas as pd
import sys

from scipy import linalg, interpolate
from contextlib import contextmanager, nullcontext
from tqdm import tqdm
from typing import Literal

from .utils import permute_clones

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

def eigenvalue_test(
    adata: sc.AnnData | np.ndarray,
    key: str | None = None,
    key_added: str = "eigenvalues_test",
    flavor: Literal["asymptotic", "synthetic"] = "synthetic",
    n_simulations: int = 10000,
    progress_bar: bool = True,
    null_distribution: np.ndarray | None = None,
    random_state: int | None = 0,
):
    """
    Performs Johnstone’s Spiked Covariance Test to identify if the embedding is random.

    Parameters
    ----------
    adata : sc.AnnData | np.ndarray
        AnnData object or numpy array containing the data.
    key : str | None
        Key to use for AnnData input.
    key_added : str
        Key to use for adding the results to the AnnData object.
    flavor : Literal["asymptotic", "synthetic"]
        Flavor of the test to use.
    n_simulations : int
        Number of simulations to use for the synthetic approach.
    progress_bar : bool
        Whether to show a progress bar.
    null_distribution : np.ndarray | None
        Null distribution to use for the synthetic approach.
    random_state : int | None
        Seed used to generate the null distribution for the synthetic approach.

    Returns
    -------
    dict
        Dictionary containing the test statistic and p-value.
    """
    start = logg.info(f"computing eigenvalues test via {flavor} approach")

    if isinstance(adata, sc.AnnData):
        if key is None:
            raise ValueError("key must be specified for AnnData input")
        X = adata.obsm[key]
    elif isinstance(adata, np.ndarray):
        X = adata
        key_added = None
    else:
        raise ValueError("adata must be AnnData or numpy.ndarray")

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    stat = _calculate_statistics(X)
    if flavor == "asymptotic":
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
                    stat_sim = _calculate_statistics(X_sim)
                    null_distribution.append(stat_sim)
        null_distribution = np.asarray(null_distribution, dtype=float)
        p_value = float(
            (1.0 + np.sum(null_distribution >= stat)) / (null_distribution.size + 1.0)
        )
    else:
        raise ValueError("flavor must be 'asymptotic' or 'synthetic'")

    if key_added:
        lines = [
            "added",
            f"     .uns['{key_added}'] eigenvalues test statistics and approximate p-value",
        ]
        logg.info("    finished ({time_passed})", deep="\n".join([l for l in lines if l is not None]), time=start)
        adata.uns[key_added] = {
            "stat": stat,
            "p_value": p_value,
            "null_distribution": null_distribution,
        }
    else:
        logg.info("    finished ({time_passed})", time=start)
        return {"stat": stat, "p_value": p_value, "null_distribution": null_distribution}


@contextmanager
def _silenced():
    """
    Temporarily sets scanpy verbosity to 0, restoring it on exit.

    Yields
    ------
    None
        Inside the context all scanpy logging (including warnings) is muted.
    """
    verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0
    try:
        yield
    finally:
        sc.settings.verbosity = verbosity

# `.uns['clonal_nn']` keeps a summary of the graph, not only constructor arguments
_CLONAL_NN_ARGS = ("k", "use_rep", "obsp_name", "split_by", "random_state")

def spectral_permutation_test(
    adata: sc.AnnData,
    clones: sc.AnnData,
    batch_key: str | None = None,
    obs_name: str | None = None,
    method: Literal["poisson", "multinomial"] = "poisson",
    n_permutations: int = 100,
    random_state : None | int = 4,
    progress_bar: bool = True,
    key_added: str = "spectral_permutation_test",
    clonal_nn_kwargs: dict | None = None,
    clone2vec_kwargs: dict | None = None,
) -> None:
    """
    Tests whether a clonal embedding carries more structure than the clonal labelling
    of the gene expression space alone would produce.

    The whole embedding pipeline (`clonal_nn` followed by `clone2vec` or `clone2vec_Poi`)
    is re-run on clonally reshuffled data, and the leading eigenvalue statistic of the
    observed embedding (the one used by `eigenvalue_test`) is compared with the
    permutation null. Labels are reshuffled by `c2v.utils.permute_clones`, that is (a) only
    between the cells carrying clonal information and (b) only within one batch, therefore
    clone sizes and, if `batch_key` is provided, batch composition of the clones are
    preserved in every permutation – a batch effect, if it exists, stays in the resulting
    clonal embeddings.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix at the cell level, containing the representation used by
        `clonal_nn` in `.obsm`.
    clones : sc.AnnData
        Annotated data matrix at the clone level, typically from `create_clone_adata`.
    batch_key : str | None, optional
        Column in `adata.obs` to shuffle clonal labels within, if None, labels are
        shuffled across the whole dataset, by default None.
    obs_name : str | None, optional
        Column in `adata.obs` containing clonal information, if None, try to get from
        `clones.uns['obs_name']`, by default None.
    method : Literal["poisson", "multinomial"], optional
        Whether to embed clones with `clone2vec_Poi` ("poisson") or `clone2vec`
        ("multinomial"), by default "poisson".
    n_permutations : int, optional
        Number of clonal permutations to generate the null distribution, by default 100.
    random_state : None | int, optional
        Random state for reproducibility, also passed to the embedding functions, by default 4.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    key_added : str, optional
        Key in `clones.uns` to store the results, by default "spectral_permutation_test".
    clonal_nn_kwargs : dict | None, optional
        Keyword arguments for `clonal_nn`, if None, try to get from
        `clones.uns['clonal_nn']`, by default None.
    clone2vec_kwargs : dict | None, optional
        Keyword arguments for `clone2vec` or `clone2vec_Poi`, if None, try to get from
        `clones.uns['clone2vec']['init']` or `clones.uns['clone2vec_Poi']['init']`,
        by default None.

    Returns
    -------
    None
        The `clones` object is modified in place with the test statistic, the permutation
        null distribution, and the empirical p-value stored in `.uns[key_added]`.
    """
    # Imported here, `tools` imports this module itself
    from .tools import clonal_nn
    from .embeddings import clone2vec, clone2vec_Poi

    if method == "poisson":
        embedding, key = clone2vec_Poi, "clone2vec_Poi"
    elif method == "multinomial":
        embedding, key = clone2vec, "clone2vec"
    else:
        raise ValueError("method must be 'poisson' or 'multinomial'")

    if obs_name is None:
        try:
            obs_name = clones.uns["obs_name"]
        except KeyError:
            raise KeyError("obs_name not found in clones.uns. Please provide obs_name.")
    if obs_name not in adata.obs:
        raise KeyError(f"obs_name '{obs_name}' not found in adata.obs. Please provide a valid obs_name.")
    if (batch_key is not None) and (batch_key not in adata.obs):
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs. Please provide a valid batch_key.")

    if clonal_nn_kwargs is None:
        clonal_nn_kwargs = clones.uns.get("clonal_nn", {})
    clonal_nn_kwargs = {k: v for k, v in clonal_nn_kwargs.items() if k in _CLONAL_NN_ARGS}
    clonal_nn_kwargs.setdefault("random_state", random_state)

    if clone2vec_kwargs is None:
        if key in clones.uns:
            clone2vec_kwargs = clones.uns[key]["init"]
        else:
            clone2vec_kwargs = {}
    clone2vec_kwargs = dict(clone2vec_kwargs)
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
        f"    embedding permuted clones with '{method}' flavor",
        f"    shuffling '{obs_name}' labels" + (f" within '{batch_key}'" if batch_key else ""),
    ]
    start = logg.info(
        "computing spectral permutation test",
        deep="\n" + "\n".join(lines_before),
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

    # Reference embedding calculation
    with _silenced():
        clonal_nn(adata_perm, clones_perm, obs_name=obs_name, **clonal_nn_kwargs)
        embedding(clones_perm, progress_bar=False, **clone2vec_kwargs)
    stat = _calculate_statistics(clones_perm.obsm[obsm_key])

    # Clonal permutations with respect to the batch key, the labels of the cells without
    # clonal information (or of the clones filtered out of `clones`) are kept in place
    labels = adata.obs[obs_name]
    batch = adata.obs[batch_key] if batch_key is not None else None
    keep_labels = pd.Index(labels.unique()).difference(clones.obs_names).tolist()
    if batch is not None:
        n_kept = int((labels.isin(clones.obs_names) & batch.isna()).sum())
        if n_kept > 0:
            logg.warning(
                f"{n_kept} labeled cells have no '{batch_key}' value, "
                "their clonal labels are kept in place"
            )

    rng = np.random.default_rng(random_state)
    null_distribution = []
    cm = tqdm(
        range(n_permutations),
        desc=prefix + f"permuting clonal labels ({n_permutations} permutations)",
        file=sys.stdout,
    ) if progress_bar else nullcontext(range(n_permutations))
    with _silenced(), cm as iterator:
        for i in iterator:
            adata_perm.obs[obs_name] = permute_clones(
                labels,
                batch=batch,
                keep_labels=keep_labels,
                random_state=rng,
            ).to_numpy()

            clonal_nn(adata_perm, clones_perm, obs_name=obs_name, **clonal_nn_kwargs)
            embedding(clones_perm, progress_bar=False, **clone2vec_kwargs)
            try:
                null_distribution.append(_calculate_statistics(clones_perm.obsm[obsm_key]))
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

    clones.uns[key_added] = {
        "stat": stat,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "method": method,
        "batch_key": batch_key,
    }

    lines = [
        "added",
        f"     .uns['{key_added}'] eigenvalues test statistics, permutation null distribution, and empirical p-value",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)