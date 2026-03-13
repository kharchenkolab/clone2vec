from __future__ import annotations

import scanpy as sc
import numpy as np
import sys

from scipy import linalg, interpolate
from contextlib import nullcontext
from tqdm import tqdm
 
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

def _tracy_widom_cdf(x: float) -> float:
    """
    Computes the CDF of the Tracy-Widom distribution (with beta = 1).
    """
    if x > 3:
        tail_prob = np.exp(-2.0/3.0 * x**1.5) / (4.0 * np.sqrt(np.pi) * x**1.5)
        return 1.0 - tail_prob
    elif x < -10:
        return 0
    else:
        tw_x = np.array(list(tw_cdf_tabular.keys()))
        tw_cdf = np.array(list(tw_cdf_tabular.values()))
        interp = interpolate.interp1d(
            tw_x,
            tw_cdf,
            kind="cubic",
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )
        val = interp(x)
        return float(np.clip(val, 0.0, 1.0))

def _calculate_statistics(X: np.ndarray) -> float:
    """
    """
    n, p = X.shape
    X_centered = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    n_dof = n - 1

    s = linalg.svdvals(X_centered)
    eigenvalues = s ** 2
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
):
    """
    Performs Johnstone’s Spiked Covariance Test to identify if the embedding is random.
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
            null_distribution = []
            n, p = X.shape
            cm = tqdm(
                range(n_simulations),
                desc=prefix + f"generating null distribution ({n_simulations} simulations)",
                file=sys.stdout,
            ) if progress_bar else nullcontext(range(n_simulations))
            with cm:
                for i in cm:
                    X_sim = np.random.normal(size=(n, p))
                    stat_sim = _calculate_statistics(X_sim)
                    null_distribution.append(stat_sim)
            null_distribution = np.array(null_distribution)
        p_value = np.mean(null_distribution >= stat)
        if p_value == 0:
            p_value = f"<{1 / n_simulations}"
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