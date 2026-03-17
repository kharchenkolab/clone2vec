from __future__ import annotations

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import pandas as pd

import os
import sys

from contextlib import nullcontext
from joblib import Parallel, delayed
from typing import Literal
from tqdm import tqdm

from .utils import _nan_mask

logg = sc.logging

def _clr(x: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """
    Center Log-Ratio (CLR) transformation.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (n_samples, n_features).
    epsilon : float, optional
        Small constant to avoid log(0) and division by zero, by default 1e-3.

    Returns
    -------
    np.ndarray
        CLR-transformed array with shape (n_samples, n_features).
    """
    if not np.allclose(x, x.astype(int)):
        if epsilon == 1e-3:
            logg.warning("Pseudocount 0.001 is used for integer counts as an input for CLR, it might skew proportions.")

    if isinstance(x, pd.DataFrame):
        x_cols = x.columns
        x_rows = x.index
        x = x.values.copy()

        x_pseudo = x + epsilon
        log_geo_mean = np.mean(np.log(x_pseudo), axis=1, keepdims=True)
        clr_transformed = np.log(x_pseudo) - log_geo_mean
        
        return pd.DataFrame(clr_transformed, columns=x_cols, index=x_rows)
    else:
        x_pseudo = x + epsilon
        log_geo_mean = np.mean(np.log(x_pseudo), axis=1, keepdims=True)
        clr_transformed = np.log(x_pseudo) - log_geo_mean
        
        return clr_transformed

def _fast_corr(
    X: np.ndarray | sp.spmatrix,
    Y: np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
    significance: bool = True,
    slope: bool = True,
    batch_size: int = 1000,
    progress_bar: bool = True,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate the correlation matrix between X and Y, gracefully handling
    columns with zero variance by setting the correlation to NaN.

    Supports sparse X by converting column-wise dense batches for computation.

    Parameters
    ----------
    X : np.ndarray | scipy.sparse.spmatrix
        Dataset with shape (n_samples, n_features). If sparse, columns are
        processed in dense batches to limit memory usage.
    Y : np.ndarray
        Response matrix with shape (n_samples, n_responses). Assumed dense.
    method : Literal["pearson", "spearman"], optional
        Correlation method to use, by default "pearson".
    significance : bool
        If True, also returns a matrix of p-values corresponding to the
        correlation matrix under the assumption of a t-distribution with
        degrees of freedom equal to the number of samples minus two.
    slope : bool
        If True, also returns a matrix of slopes corresponding to the
        correlation matrix.
    batch_size : int, optional
        Number of X columns to process per batch when X is sparse, by default 1000.
    progress_bar : bool, optional
        Whether to display a progress bar for batch processing when X is sparse,
        by default True.
    weights : np.ndarray | None, optional
        Per-sample weights of shape (n_samples,). When provided, compute
        weighted correlation. Typically used with ``binomial=True`` where weights
        are the total counts per cell, by default None.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with correlation matrix, p-values, and slopes.
    """
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X[:, None].copy()
    if Y.ndim == 1:
        Y = Y[:, None].copy()

    if sp.issparse(Y):
        logg.warning("Sparse input for the response matrix Y isn't supported, converting to dense for correlation calculation.")
        Y = Y.toarray()

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights must have the same length as the number of samples.")

    if method == "spearman":
        Y_data = stats.rankdata(Y, axis=0)
    else:
        Y_data = Y

    # ---- weighted vs unweighted centering / norms ----
    if weights is not None:
        w = weights[:, None]  # (n, 1)
        W = w.sum()
        Y_mean = (w * Y_data).sum(axis=0) / W
        Y_centered = Y_data - Y_mean
        norm_Y = np.sqrt((w * Y_centered ** 2).sum(axis=0))
    else:
        Y_centered = Y_data - np.mean(Y_data, axis=0)
        norm_Y = np.linalg.norm(Y_centered, axis=0)

    n_features = X.shape[1]
    n_responses = Y.shape[1]
    R = np.full((n_features, n_responses), np.nan)

    if not sp.issparse(X):
        X_data = X
        if method == "spearman":
            X_data = stats.rankdata(X_data, axis=0)

        if weights is not None:
            X_mean = (w * X_data).sum(axis=0) / W
            X_centered = X_data - X_mean
            norm_X = np.sqrt((w * X_centered ** 2).sum(axis=0))
        else:
            X_centered = X_data - np.mean(X_data, axis=0)
            norm_X = np.linalg.norm(X_centered, axis=0)

        denominator = norm_X[:, None] * norm_Y

        if weights is not None:
            numerator = (w * X_centered).T @ Y_centered
        else:
            numerator = X_centered.T @ Y_centered
        np.divide(numerator, denominator, where=(denominator != 0), out=R)

        results = {"r": R}

        if significance:
            n = X.shape[0]
            if weights is not None:
                n_eff = W ** 2 / (weights ** 2).sum()
            else:
                n_eff = float(n)
            if n_eff <= 2:
                P = np.full_like(R, np.nan)
                results["pvalue"] = P
            else:
                df = n_eff - 2
                T = np.nan_to_num(R) * np.sqrt(df / (1 - np.nan_to_num(R) ** 2))
                P = 2 * stats.t.sf(np.abs(T), df)
                P[np.isnan(R)] = np.nan
                results["pvalue"] = P

        if slope:
            if weights is not None:
                std_X = np.sqrt((w * (X_data - X_mean) ** 2).sum(axis=0) / W)
                std_Y = np.sqrt((w * (Y_data - Y_mean) ** 2).sum(axis=0) / W)
            else:
                std_X = np.std(X_data, axis=0)
                std_Y = np.std(Y_data, axis=0)

            std_ratio = np.full_like(R, np.nan)
            np.divide(std_Y, std_X[:, None], out=std_ratio, where=(std_X[:, None] != 0))

            S = R * std_ratio
            results["slope"] = S

        return results
    else:
        X_sparse = X.tocsc()

        col_indices = range(0, n_features, max(1, int(batch_size)))

        if sc.settings.verbosity.value >= 2:
            prefix = "    "
        else:
            prefix = ""
        cm = tqdm(
            col_indices,
            desc=prefix + "correlation batches",
            leave=True,
            file=sys.stdout,
        ) if progress_bar else nullcontext(col_indices)

        if slope:
            if weights is not None:
                std_Y = np.sqrt((w * (Y_data - Y_mean) ** 2).sum(axis=0) / W)
            else:
                std_Y = np.std(Y_data, axis=0)
            S = np.full_like(R, np.nan)

        if significance:
            n = X.shape[0]
            P = np.full_like(R, np.nan)
            if weights is not None:
                n_eff = W ** 2 / (weights ** 2).sum()
            else:
                n_eff = float(n)
            df = n_eff - 2

        with cm as iterator:
            for start in iterator:
                end = min(start + batch_size, n_features)
                cols = np.arange(start, end)

                X_batch = X_sparse[:, cols].toarray()
                if method == "spearman":
                    X_batch = stats.rankdata(X_batch, axis=0)

                if weights is not None:
                    X_mean_b = (w * X_batch).sum(axis=0) / W
                    X_centered_b = X_batch - X_mean_b
                    norm_X_b = np.sqrt((w * X_centered_b ** 2).sum(axis=0))
                else:
                    X_centered_b = X_batch - np.mean(X_batch, axis=0)
                    norm_X_b = np.linalg.norm(X_centered_b, axis=0)

                denominator_b = norm_X_b[:, None] * norm_Y

                if weights is not None:
                    numerator_b = (w * X_centered_b).T @ Y_centered
                else:
                    numerator_b = X_centered_b.T @ Y_centered
                R_b = np.full(denominator_b.shape, np.nan)
                np.divide(numerator_b, denominator_b, where=(denominator_b != 0), out=R_b)
                R[start:end, :] = R_b

                if slope:
                    if weights is not None:
                        std_X_b = np.sqrt((w * (X_batch - X_mean_b) ** 2).sum(axis=0) / W)
                    else:
                        std_X_b = np.std(X_batch, axis=0)
                    std_ratio_b = np.full_like(R_b, np.nan)
                    np.divide(std_Y, std_X_b[:, None], out=std_ratio_b, where=(std_X_b[:, None] != 0))
                    S_b = R_b * std_ratio_b
                    S[start:end, :] = S_b

                if significance and n_eff > 2:
                    T_b = np.nan_to_num(R_b) * np.sqrt(df / (1 - np.nan_to_num(R_b) ** 2))
                    P_b = 2 * stats.t.sf(np.abs(T_b), df)
                    P_b[np.isnan(R_b)] = np.nan
                    P[start:end, :] = P_b
                
        results = {"r": R}
        if significance:
            results["pvalue"] = P
        if slope:
            results["slope"] = S

        return results

def _fast_corr_cols(
    X: np.ndarray | sp.spmatrix,
    Y: np.ndarray | sp.spmatrix,
    method: Literal["pearson", "spearman"] = "pearson",
    significance: bool = True,
    slope: bool = True,
    batch_size: int = 1000,
    progress_bar: bool = True,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate correlation statistics between corresponding columns of X and Y
    (1<->1, 2<->2, ...), gracefully handling zero-variance columns by setting
    the result to NaN. Supports sparse inputs similarly to `_fast_corr`.

    Parameters
    ----------
    X : np.ndarray | scipy.sparse.spmatrix
        Dataset with shape (n_samples, n_features).
    Y : np.ndarray | scipy.sparse.spmatrix
        Another dataset with the same shape as X.
    method : Literal["pearson", "spearman"], optional
        Statistic to compute, by default "pearson".
    significance : bool, optional
        If True, also returns p-values (for Pearson/Spearman via t-test,
        for GAM via LRT/Wald), by default True.
    slope : bool, optional
        If True, also returns slopes (for Pearson/Spearman as std ratio times r;
        for GAM the endpoints change), by default True.
    batch_size : int, optional
        Number of columns to process per batch when X (or Y) is sparse,
        by default 1000.
    progress_bar : bool, optional
        Whether to display a progress bar for batch processing when inputs are
        sparse or when method=="gam", by default True.
    weights : np.ndarray | None, optional
        Per-sample weights of shape (n_samples,). When provided, compute
        weighted correlation, by default None.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with statistics. For Pearson/Spearman: keys "r",
        optionally "pvalue" and "slope". For GAM: keys "r2",
        "amplitude", optionally "pvalue" and "slope".
    """
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X[:, None].copy()
    if isinstance(Y, np.ndarray) and Y.ndim == 1:
        Y = Y[:, None].copy()

    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape for column-wise correlation.")

    n_samples, n_cols = X.shape

    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape[0] != n_samples:
            raise ValueError("weights must have the same length as the number of samples.")

    if method in ["pearson", "spearman"]:
        results = {}

        if (not sp.issparse(X)) and (not sp.issparse(Y)):
            X_data = X
            Y_data = Y
            if method == "spearman":
                X_data = stats.rankdata(X_data, axis=0)
                Y_data = stats.rankdata(Y_data, axis=0)

            if weights is not None:
                w = weights[:, None]
                W = w.sum()
                X_mean = (w * X_data).sum(axis=0) / W
                Y_mean = (w * Y_data).sum(axis=0) / W
                X_centered = X_data - X_mean
                Y_centered = Y_data - Y_mean
                norm_X = np.sqrt((w * X_centered ** 2).sum(axis=0))
                norm_Y = np.sqrt((w * Y_centered ** 2).sum(axis=0))
            else:
                X_centered = X_data - np.mean(X_data, axis=0)
                Y_centered = Y_data - np.mean(Y_data, axis=0)
                norm_X = np.linalg.norm(X_centered, axis=0)
                norm_Y = np.linalg.norm(Y_centered, axis=0)
            denom = norm_X * norm_Y

            if weights is not None:
                numer = (w * X_centered * Y_centered).sum(axis=0)
            else:
                numer = np.sum(X_centered * Y_centered, axis=0)
            R = np.full(n_cols, np.nan)
            np.divide(numer, denom, where=(denom != 0), out=R)

            results["r"] = R

            if significance:
                n = n_samples
                if weights is not None:
                    n_eff = W ** 2 / (weights ** 2).sum()
                else:
                    n_eff = float(n)
                if n_eff <= 2:
                    P = np.full_like(R, np.nan)
                else:
                    df = n_eff - 2
                    T = np.nan_to_num(R) * np.sqrt(df / (1 - np.nan_to_num(R) ** 2))
                    P = 2 * stats.t.sf(np.abs(T), df)
                    P[np.isnan(R)] = np.nan
                results["pvalue"] = P

            if slope:
                if weights is not None:
                    std_X = np.sqrt((w * (X_data - X_mean) ** 2).sum(axis=0) / W)
                    std_Y = np.sqrt((w * (Y_data - Y_mean) ** 2).sum(axis=0) / W)
                else:
                    std_X = np.std(X_data, axis=0)
                    std_Y = np.std(Y_data, axis=0)
                std_ratio = np.full_like(R, np.nan)
                np.divide(std_Y, std_X, out=std_ratio, where=(std_X != 0))
                S = R * std_ratio
                results["slope"] = S

            return results

        X_sparse = X.tocsc() if sp.issparse(X) else None
        Y_sparse = Y.tocsc() if sp.issparse(Y) else None

        R = np.full(n_cols, np.nan)
        if significance:
            n = n_samples
            P = np.full(n_cols, np.nan)
            if weights is not None:
                W = weights.sum()
                n_eff = W ** 2 / (weights ** 2).sum()
            else:
                n_eff = float(n)
            df = n_eff - 2
        if slope:
            S = np.full(n_cols, np.nan)

        col_indices = range(0, n_cols, max(1, int(batch_size)))
        if sc.settings.verbosity.value >= 2:
            prefix = "    "
        else:
            prefix = ""
        cm = tqdm(
            total=len(col_indices),
            desc=prefix + "correlation batches",
            leave=True,
            file=sys.stdout,
        ) if progress_bar else nullcontext()
        with cm as pbar:
            for start in col_indices:
                end = min(start + batch_size, n_cols)
                cols = np.arange(start, end)

                X_batch = (X_sparse[:, cols].toarray() if X_sparse is not None else X[:, cols])
                Y_batch = (Y_sparse[:, cols].toarray() if Y_sparse is not None else Y[:, cols])

                if method == "spearman":
                    X_batch = stats.rankdata(X_batch, axis=0)
                    Y_batch = stats.rankdata(Y_batch, axis=0)

                if weights is not None:
                    w = weights[:, None]
                    W_val = w.sum()
                    X_mean_b = (w * X_batch).sum(axis=0) / W_val
                    Y_mean_b = (w * Y_batch).sum(axis=0) / W_val
                    X_centered_b = X_batch - X_mean_b
                    Y_centered_b = Y_batch - Y_mean_b
                    norm_X_b = np.sqrt((w * X_centered_b ** 2).sum(axis=0))
                    norm_Y_b = np.sqrt((w * Y_centered_b ** 2).sum(axis=0))
                else:
                    X_centered_b = X_batch - np.mean(X_batch, axis=0)
                    Y_centered_b = Y_batch - np.mean(Y_batch, axis=0)
                    norm_X_b = np.linalg.norm(X_centered_b, axis=0)
                    norm_Y_b = np.linalg.norm(Y_centered_b, axis=0)

                denom_b = norm_X_b * norm_Y_b
                if weights is not None:
                    numer_b = (w * X_centered_b * Y_centered_b).sum(axis=0)
                else:
                    numer_b = np.sum(X_centered_b * Y_centered_b, axis=0)
                R_b = np.full(len(cols), np.nan)
                np.divide(numer_b, denom_b, where=(denom_b != 0), out=R_b)
                R[start:end] = R_b

                if slope:
                    if weights is not None:
                        std_X_b = np.sqrt((w * (X_batch - X_mean_b) ** 2).sum(axis=0) / W_val)
                        std_Y_b = np.sqrt((w * (Y_batch - Y_mean_b) ** 2).sum(axis=0) / W_val)
                    else:
                        std_X_b = np.std(X_batch, axis=0)
                        std_Y_b = np.std(Y_batch, axis=0)
                    std_ratio_b = np.full(len(cols), np.nan)
                    np.divide(std_Y_b, std_X_b, out=std_ratio_b, where=(std_X_b != 0))
                    S_b = R_b * std_ratio_b
                    S[start:end] = S_b

                if significance and n_eff > 2:
                    T_b = np.nan_to_num(R_b) * np.sqrt(df / (1 - np.nan_to_num(R_b) ** 2))
                    P_b = 2 * stats.t.sf(np.abs(T_b), df)
                    P_b[np.isnan(R_b)] = np.nan
                    P[start:end] = P_b
                if progress_bar:
                    pbar.update(1)

        results = {"r": R}
        if significance:
            results["pvalue"] = P
        if slope:
            results["slope"] = S
        return results

    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

def _calculate_gam_stats(
    x: np.array,
    y: np.array,
    spline_df: int = 5,
    spline_function: str = "bs",
    p_method: Literal["lrt", "wald"] = "lrt",
    random_state: int | None = 42,
    binomial: bool = False,
    totals: np.array | None = None,
    **spline_kwargs,
) -> dict[str, np.ndarray]:
    """
    Calculate GAM statistics for a single pair of columns in X and Y.

    Parameters
    ----------
    x : np.array
        Input feature vector.
    y : np.array
        Response vector. When ``binomial=True`` this should be proportions
        (successes / totals) in the range [0, 1].
    spline_df : int, optional
        Degrees of freedom for the spline, by default 5.
    spline_function : str, optional
        Spline function to use, by default "bs".
    p_method : Literal["lrt", "wald"], optional
        Method to use for p-value calculation, by default "lrt".
    random_state : int | None, optional
        Seed for NumPy RNG to ensure reproducible behavior in stochastic routines, by default 42.
    binomial : bool, optional
        If True, use a Binomial GLM instead of Gaussian. ``y`` should be
        proportions and ``totals`` the per-observation trial counts, by default False.
    totals : np.array | None, optional
        Per-observation trial counts (used only when ``binomial=True``), by default None.
    **spline_kwargs : dict, optional
        Additional keyword arguments to pass to the spline formula.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with GAM statistics.
    """
    if random_state is not None:
        try:
            np.random.seed(int(random_state))
        except Exception:
            pass

    if np.allclose(x, x[0]):
        results = {"r2": np.nan, "amplitude": np.nan, "slope": np.nan, "pvalue": np.nan}
        return results

    if len(x) != len(y):
        logg.warning("x and y must have the same number of samples. Return NaN.")
        results = {"r2": np.nan, "amplitude": np.nan, "slope": np.nan, "pvalue": np.nan}
        return results
    
    try:
        results = {}

        kwargs_str = ""
        for k, v in spline_kwargs.items():
            kwargs_str += f", {k}={v}"

        if binomial:
            if totals is None:
                raise ValueError("totals must be provided when binomial=True")
            family = sm.families.Binomial()
            gam_fit = smf.glm(
                formula=f"y ~ {spline_function}(x, df={spline_df}{kwargs_str})",
                data={"x": x, "y": y},
                family=family,
                freq_weights=totals.astype(float),
            ).fit(random_state=random_state)
        else:
            family = sm.families.Gaussian()
            gam_fit = smf.glm(
                formula=f"y ~ {spline_function}(x, df={spline_df}{kwargs_str})",
                data={"x": x, "y": y}, 
                family=family,
            ).fit(random_state=random_state)
        results["r2"] = gam_fit.pseudo_rsquared()

        fit_prediction = gam_fit.predict({"x": x}).values
        effect_size = fit_prediction.max() - fit_prediction.min()
        results["amplitude"] = effect_size

        endpoints_change = gam_fit.predict({"x": [x.max(), x.min()]}).values
        endpoints_change = endpoints_change[0] - endpoints_change[1]
        results["slope"] = endpoints_change

        if p_method not in ["lrt", "wald"]:
            logg.warning(f"p_method must be 'lrt' or 'wald', got {p_method}. Default to 'lrt'.")
            p_method = "lrt"
        
        if p_method == "lrt":
            if binomial:
                const_fit = smf.glm(
                    formula="y ~ 1",
                    data={"y": y},
                    family=family,
                    freq_weights=totals.astype(float),
                ).fit(random_state=random_state)
            else:
                x_const = sm.add_constant(np.zeros_like(x))
                const_fit = sm.OLS(y, x_const).fit(random_state=random_state)
            lrt = 2 * (gam_fit.llf - const_fit.llf)
            df_d = gam_fit.df_model - const_fit.df_model
            p = stats.chi2.sf(lrt, df_d)
            results["pvalue"] = p
        else:
            wald_stat = gam_fit.wald_test_terms().summary_frame()
            wald_p = wald_stat.iloc[wald_stat.index != "Intercept"]["P>chi2"].values[0]
            results["pvalue"] = wald_p
        
        return results
    except Exception as e:
        logg.warning(f"Error fitting GAM: {e}. Return NaN.")
        results = {"r2": np.nan, "amplitude": np.nan, "slope": np.nan, "pvalue": np.nan}
        return results

def _gam(
    X: np.ndarray,
    Y: np.ndarray,
    spline_df: int = 5,
    spline_degree: int = 3,
    n_jobs: int = -1,
    p_method: Literal["lrt", "wald"] = "lrt",
    progress_bar: bool = True,
    random_state: int | None = 42,
    binomial: bool = False,
    totals: np.ndarray | None = None,
    **spline_kwargs,
) -> dict[str, np.ndarray]:
    """
    Calculate GAM (generalized additive model) statistics for each pair of
    columns in X and Y.

    Parameters
    ----------
    X : np.ndarray
        Dataset with shape (n_samples, n_features).
    Y : np.ndarray
        Response matrix with shape (n_samples, n_responses).
        When ``binomial=True`` these should be proportions.
    spline_df : int, optional
        Degrees of freedom for the spline, by default 5.
    spline_degree : int, optional
        Degree of the spline, by default 3.
    n_jobs : int, optional
        Number of parallel jobs to run, by default -1.
    p_method : Literal["lrt", "wald"], optional
        Method to use for p-value calculation, by default "lrt".
    progress_bar : bool, optional
        If True, display a progress bar, by default True.
    random_state : int | None, optional
        Base seed used to derive a deterministic per-pair seed, ensuring reproducibility, by default 42.
    binomial : bool, optional
        If True, use a Binomial GLM instead of Gaussian, by default False.
    totals : np.ndarray | None, optional
        Per-observation trial counts (used only when ``binomial=True``), by default None.
    **spline_kwargs : dict, optional
        Additional keyword arguments to pass to the spline formula.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with GAM statistics for each pair of columns in X and Y.
    """
    from itertools import product

    if sp.issparse(X):
        X_sparse = X.tocsc()
        
    # Consistency with fast_corr function
    if sp.issparse(Y):
        logg.warning("Sparse input for the response matrix Y isn't supported, converting to dense for correlation calculation.")
        Y = Y.toarray()

    X_col_n = X.shape[1]
    Y_col_n = Y.shape[1]

    tasks = product(np.arange(X_col_n), np.arange(Y_col_n))

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    if progress_bar:
        tasks = tqdm(
            tasks,
            desc=prefix + "fitting GAMs",
            total=(X_col_n * Y_col_n),
            file=sys.stdout,
        )

    if not sp.issparse(X):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_gam_stats)(
                X[:, x],
                Y[:, y],
                spline_df=spline_df,
                p_method=p_method,
                random_state=(None if random_state is None else int(random_state) + x * Y_col_n + y),
                degree=spline_degree,
                binomial=binomial,
                totals=totals,
                **spline_kwargs,
            ) for x, y in tasks
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_gam_stats)(
                X_sparse[:, x].toarray().flatten(),
                Y[:, y],
                spline_df=spline_df,
                p_method=p_method,
                random_state=(None if random_state is None else int(random_state) + x * Y_col_n + y),
                degree=spline_degree,
                binomial=binomial,
                totals=totals,
                **spline_kwargs,
            ) for x, y in tasks
        )

    results = {
        key: np.array([r[key] for r in results]).reshape(X_col_n, Y_col_n)
        for key in results[0]
    }
    return results

def _adjust_pvalues(
    pvalues: np.ndarray,
    method: str = "fdr_bh",
) -> np.ndarray:
    """
    Adjust p-values using the specified method. 
    NaN values in `pvalues` will be left as NaN in the output.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values to adjust. Can be any-dimensional.
    method : str, optional
        Method for adjustment, will be passed to `statsmodels.stats.multitest.multipletests`,
        default is "fdr_bh".

    Returns
    -------
    np.ndarray
        Adjusted p-values. Has the same shape as `pvalues`.
    """
    from statsmodels.stats.multitest import multipletests

    dims = pvalues.shape
    pvalues_flat = pvalues.flatten()
    nan_mask = np.isnan(pvalues_flat)
    fdrs_flat = np.full_like(pvalues_flat, np.nan)
    
    fdrs_flat[~nan_mask] = multipletests(pvalues_flat[~nan_mask], method=method)[1]
    
    return fdrs_flat.reshape(dims)


def graph_associations(
    adata: sc.AnnData,
    layer: None | str = None,
    graph_key: str = "connectivities",
    n_pcs: int = 50,
    adj_method: str = "fdr_bh",
    key_added: str = "X_gPCA",
) -> None:
    """
    Computes the supervised principal components analysis (sPCA) of the data
    [10.1016/j.patcog.2010.12.015]. Shortly, it finds the axes that maximize the correlation
    between the gene expression and the graph Laplacian (one can see it as an
    autocorrelation-aware PCA).

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the data.
    layer : None | str
        Layer to use for the analysis.
    graph_key : str
        Key to use for the graph.
    n_pcs : int
        Number of principal components to compute.
    adj_method : str
        Method for adjusting p-values.
    key_added : str
        Key to use for adding the results to the AnnData object.

    Returns
    -------
    None
    """
    from scipy.sparse.linalg import eigsh
    import scipy.stats as stats

    start = logg.info(f"computing graph associations")

    if layer is None:
        layer = "X"
        use_X = True
    else:
        use_X = False

    if use_X:
        X = adata.X
    else:
        X = adata.layers[layer]

    if sp.issparse(X):
        logg.warning("provided matrix is sparse, make sure that you scaled data before passing to the function")
        X = X.todense()
    
    total_variance = np.sum(X ** 2)

    spcov = X.T @ adata.obsp[graph_key] @ X
    evals, evecs = eigsh(spcov, k=n_pcs, which="LA")
    evals, evecs = evals[::-1], evecs[:, ::-1]

    adata.varm["gPCs"] = evecs
    adata.obsm[key_added] = X @ evecs
    adata.uns["gpca"] = {
        "evals": evals,
        "varexpl": evals / total_variance,
        "graph_key": graph_key,
        "n_pcs": n_pcs,
    }

    # Moran's I
    numerators = np.diag(spcov)
    denominators = np.sum(X ** 2, axis=0)
    w = adata.obsp[graph_key]
    W = w.sum()
    N = len(adata)

    morans_i = (N / W) * (numerators / denominators)
    expected_I = -1.0 / (N - 1)
    S1 = 0.5 * (w + w.T).power(2).sum()
    deg_row = np.array(w.sum(axis=1))
    deg_col = np.array(w.sum(axis=0)).T
    S2 = np.sum((deg_row + deg_col) ** 2)
    W_sq = W ** 2
    var_I = (
        (N * ((N**2 - 3*N + 3) * S1 - N * S2 + 3 * W_sq)) / 
        ((N - 1) * (N - 2) * (N - 3) * W_sq)
    ) - (expected_I ** 2)
    with np.errstate(invalid="ignore"):
        z_scores = (morans_i - expected_I) / np.sqrt(var_I)
    p_values = stats.norm.sf(z_scores)

    adata.var["morans_i"] = morans_i
    adata.var["morans_i_pval"] = p_values
    adata.var["morans_i_p_adj"] = _adjust_pvalues(p_values, method=adj_method)

    lines = [
        "added",
        f"     .obsm['{key_added}'] {n_pcs} graph-guided principal components",
        f"     .varm['gPCs'] matrix of loadings",
        f"     .uns['gPCA'] eigenvalues and variance explained",
        f"     .var['morans_i'] Moran's I statistics",
        f"     .var['morans_i_pval'] p-values",
        f"     .var['morans_i_p_adj'] adjusted p-values",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def associations(
    adata,
    response_key: str | list[str],
    response_field: Literal["obs", "obsm"] = "obsm",
    response_transform: None | Literal["logit", "log1p", "sqrt", "clr"] = None,
    use_rep: Literal["X", "layers"] | None = None,
    use_raw: bool | None = None,
    layers: list[str] | str | None = None,
    method: Literal["pearson", "spearman", "gam"] = "pearson",
    p_adjustment_method: str = "fdr_bh",
    concat_layers: bool = False,
    random_state: int = 42,
    progress_bar: bool = False,
    mask_key: str | None | Literal[False] = None,
    clr_pseudocount: float = 1e-3,
    binomial: bool = False,
    **gam_kwargs,
) -> None:
    """
    Calculate correlation between features in `adata` and a response variable.
    Correlation coefficients, p-values, and False Discovery Rates (FDRs) are computed for each feature.
    Results are stored in `adata.varm` with keys "{X_name}:corr", "{X_name}:pvalue", and "{X_name}:FDR",
    where {X_name} is the name of the feature representation (e.g., "X", "X_raw", or layer names).
    If `p_adjustment_method` is None, no correction is performed.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level.
    response_key : str | list[str] | None
        Key in `adata.obs` or `adata.obsm` containing the response variable. Might be a list
        only if `response_field='obs'`, in this case the response variable is a multi-column DataFrame.
    response_field : Literal["obs", "obsm"], optional
        Whether the response variable is in `adata.obs` or `adata.obsm`, by default "obsm".
    response_transform : None | Literal["logit", "log1p", "sqrt", "clr"], optional
        Transform to apply to the response variable before correlation calculation, by default None.
    use_rep : Literal["X", "layers"] | None, optional
        Whether to use `adata.X` or `adata.layers` for feature representations, if None, uses 
        `adata.X` if `layers` is None, otherwise `adata.layers`, by default None.
    use_raw : bool | None, optional
        Whether to use `adata.raw.X` as the feature representation, by default None.
    layers : list[str] | str | None, optional
        Layers in `adata.layers` to use for feature representations if `use_rep='layers'`. If None and
        use_rep="layers", uses all layers, by default None.
    method : Literal["pearson", "spearman", "gam"], optional
        Method for correlation calculation, by default "pearson".
    p_adjustment_method : str, optional
        Method for multiple testing correction, passed to `statsmodels.stats.multitest.multipletests`,
        default is "fdr_bh".
    random_state : int, optional
        Random seed for GAM model training, by default 42.
    progress_bar : bool, optional
        Whether to show a progress bar, by default False.
    mask_key : str | None | Literal[False], optional
        Key in `adata.obs` or `adata.obsm` containing a boolean mask to filter cells, by default None.
    clr_pseudocount : float, optional
        Pseudocount to add to expression values before CLR transformation, by default 1e-3.
    binomial : bool, optional
        If True, treat the response as count data and use binomial modeling. The response
        matrix is interpreted as integer counts; proportions and row totals are computed
        internally. For ``method='gam'``, a Binomial GLM is fitted (with ``freq_weights=totals``).
        For ``method='pearson'`` or ``'spearman'``, weighted correlation is computed on the
        proportions, with weights equal to row totals, by default False.
    **gam_kwargs : dict, optional
        Additional keyword arguments to pass to `gam`.

    Returns
    -------
    None
        The `adata` object is modified in place with correlation results stored in `adata.varm`.
    """
    if mask_key is None and "mask_key" in adata.uns.keys():
        mask_key = adata.uns["mask_key"]

    if use_rep is None:
        if layers is None:
            use_rep = "X"
        else:
            use_rep = "layers"
    if use_raw and use_rep == "layers":
        logg.warning("use_raw=True is not supported when use_rep='layers'. Setting use_raw=False.")
        use_raw = False
    if (use_raw is None) and (use_rep == "X"):
        if adata.raw:
            logg.warning("Using raw data when use_raw is None and adata.raw is not None. Setting use_raw=True.")
            use_raw = True
    if use_rep == "X" and layers:
        logg.warning("Using layers when use_rep='X' is not supported. Setting use_rep='layers'.")
        use_rep = "layers"

    start_lines = "\n"
    if use_raw:
        Xs = [adata.raw.X]
        masks = [~_nan_mask(adata.raw.X)]
        X_names = ["X_raw"]
        var_names = adata.raw.var_names.values
        start_lines += "    using `adata.raw.X` as features\n"
    elif use_rep == "X":
        Xs = [adata.X]
        masks = [~_nan_mask(adata.X)]
        X_names = ["X"]
        var_names = adata.var_names.values
        start_lines += "    using `adata.X` as features\n"
    else:
        if layers is None:
            layers = list(adata.layers.keys())
        elif isinstance(layers, str):
            layers = [layers]
        Xs = [adata.layers[layer] for layer in layers]
        masks = [~_nan_mask(X) for X in Xs]
        if mask_key:
            masks = [masks[i] & adata.obsm[mask_key][layer].values for i, layer in enumerate(layers)]
        if concat_layers:
            if sp.issparse(Xs[0]):
                Xs = [sp.hstack(Xs, format="csr")]
            else:
                Xs = [np.concatenate(Xs, axis=0)]
            masks = [np.concatenate(masks, axis=0)]
            X_names = ["multi_layer"]
        else:
            X_names = [layer for layer in layers]
        var_names = adata.var_names.values
        if concat_layers:
            start_lines += f"    using layers {layers} (concatenated) as features\n"
        else:
            start_lines += f"    using layers {layers} as features\n"
    
    Xs = [X[mask] for X, mask in zip(Xs, masks)]
    
    if response_field == "obs":
        if isinstance(response_key, str):
            response_key_added = response_key
            Y = adata.obs[response_key].values[:, None]
            y_names = [response_key]
            start_lines += f"    using `adata.obs['{response_key}']` as response"
        else:
            response_key_added = "+".join(response_key)
            Y = adata.obs[response_key].values
            y_names = response_key.columns.values.astype(str)
            start_lines += f"    using `adata.obs[{y_names}]` as response"
    elif response_field == "obsm":
        if isinstance(response_key, list):
            raise ValueError("response_key must be a single string when response_field='obsm'")
        try:
            response_key_added = response_key
            y = adata.obsm[response_key]
            if isinstance(y, pd.DataFrame):
                y_names = y.columns.values.astype(str)
                y = y.values
            else:
                y_names = [f"{response_key}:{i}" for i in range(y.shape[1])]
            start_lines += f"    using `adata.obsm['{response_key}']` as response"
        except KeyError:
            raise KeyError(f"Response key '{response_key}' not found in adata.obsm. Please provide a valid response key")
    else:
        raise TypeError("response_field must be 'obs' or 'obsm'.")

    start = logg.info("computing feature–response associations", deep=start_lines)
    if X_names == ["multi_layer"]:
        Y = np.tile(y, (len(layers), 1))
    else:
        Y = y

    if binomial and response_transform is not None:
        logg.warning("response_transform is ignored when binomial=True.")
        response_transform = None

    if response_transform == "logit":
        Y = np.clip(Y, 0.01, 0.99).copy()
        Y = np.log(Y / (1 - Y))
    elif response_transform == "log1p":
        Y = np.log1p(Y).copy()
    elif response_transform == "sqrt":
        Y = np.sqrt(Y).copy()
    elif response_transform == "clr":
        Y = _clr(Y, epsilon=clr_pseudocount).copy()
    elif response_transform:
        logg.warning("response_transform must be None, 'logit', 'log1p', or 'sqrt'. Setting response_transform=None")
        response_transform = None

    row_totals = None
    if binomial:
        Y = np.asarray(Y, dtype=float)
        row_totals = Y.sum(axis=1)
        safe_totals = np.where(row_totals > 0, row_totals, 1.0)
        Y = Y / safe_totals[:, None]
        start_lines += "\n    binomial=True: response treated as counts, converted to proportions"

    results = {}
    
    for i, (X, X_name) in enumerate(list(zip(Xs, X_names))):
        Y_masked = Y[masks[i]]
        if method in ["pearson", "spearman"]:
            corr_kwargs = dict(method=method, significance=True, progress_bar=progress_bar)
            if binomial and row_totals is not None:
                corr_kwargs["weights"] = row_totals[masks[i]]
            X_res = _fast_corr(X, Y_masked, **corr_kwargs)
        else:
            gam_extra = dict(random_state=random_state, progress_bar=progress_bar)
            if binomial and row_totals is not None:
                gam_extra["binomial"] = True
                gam_extra["totals"] = row_totals[masks[i]]
            X_res = _gam(X, Y_masked, **gam_extra, **gam_kwargs)
        results[X_name] = X_res
        logg.info(f"    computed '{method}' statistics for '{X_name}' features")

    Ps = np.array([results[X_name]["pvalue"] for X_name in X_names])
    FDRs = _adjust_pvalues(Ps, method=p_adjustment_method)
    for i, X_name in enumerate(X_names):
        results[X_name]["p_adj"] = FDRs[i]

    for X_name in X_names:
        for key in results[X_name]:
            if use_raw:
                adata.raw.varm[f"{response_key_added}:{X_name}:{method}:{key}"] = pd.DataFrame(
                    results[X_name][key],
                    index=var_names,
                    columns=y_names,
                )
            else:
                adata.varm[f"{response_key_added}:{X_name}:{method}:{key}"] = pd.DataFrame(
                    results[X_name][key],
                    index=var_names,
                    columns=y_names,
                )

    lines = ["added"]
    for key in results[X_name]:
        if key.endswith("pvalue"):
            postfix = " with uncorrected p-values"
        elif key.endswith("p_adj"):
            postfix = " with corrected p-values"
        elif key.endswith("r"):
            postfix = " with correlation coefficients"
        elif key.endswith("slope"):
            postfix = " with regression slopes coefficients"
        elif key.endswith("r2"):
            postfix = " with models R2 coefficients"
        elif key.endswith("amplitude"):
            postfix = " with amplitudes of fitted GAMs"
        else:
            postfix = ""

        if use_raw:
            lines.append(f"     .raw.varm['<response_key>:<feature matrix>:{method}:{key}'] matrix{postfix}")
        else:
            lines.append(f"     .varm['<response_key>:<feature matrix>:{method}:{key}'] matrix{postfix}")
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)