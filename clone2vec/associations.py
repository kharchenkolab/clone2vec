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
import warnings

from contextlib import nullcontext
from joblib import Parallel, delayed
from typing import Literal
from tqdm import tqdm

from .utils import _nan_mask, _silenced, stack_layers
from .preprocessing import transfer_expression

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

def fast_corr(
    X: np.ndarray | sp.spmatrix | pd.DataFrame | pd.Series,
    Y: np.ndarray | pd.DataFrame | pd.Series,
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
    X : np.ndarray | scipy.sparse.spmatrix | pd.DataFrame | pd.Series
        Dataset with shape (n_samples, n_features). If sparse, columns are
        processed in dense batches to limit memory usage.
    Y : np.ndarray | pd.DataFrame | pd.Series
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
    dict[str, np.ndarray | pd.DataFrame]
        Dictionary with correlation matrix, p-values, and slopes of shape (n_features, n_responses).
        If X or Y is a pandas object, values are DataFrames indexed by X columns (or Series name)
        with Y columns (or Series name) as columns.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)) or isinstance(Y, (pd.DataFrame, pd.Series)):
        def labels(A):
            if isinstance(A, pd.DataFrame):
                return A.columns
            if isinstance(A, pd.Series):
                return [A.name]
            return None
        X_arr = X if sp.issparse(X) else np.asarray(X, dtype=float)
        res = fast_corr(X_arr, np.asarray(Y, dtype=float), method=method, significance=significance, slope=slope,
                        batch_size=batch_size, progress_bar=progress_bar, weights=weights)
        return {k: pd.DataFrame(v, index=labels(X), columns=labels(Y)) for k, v in res.items()}

    if not sp.issparse(X):
        X = np.asarray(X, dtype=float)
    if not sp.issparse(Y):
        Y = np.asarray(Y, dtype=float)
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
    the result to NaN. Supports sparse inputs similarly to `fast_corr`.

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


def _morans_i(
    X: np.ndarray,
    w: sp.spmatrix | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Moran's I for every column of an (already centered) matrix.

    Parameters
    ----------
    X : np.ndarray
        Centered (typically scaled) matrix of shape (n_observations, n_features).
    w : sp.spmatrix | np.ndarray
        Spatial weights (graph) of shape (n_observations, n_observations).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Moran's I, z-scores and one-sided p-values (normal approximation under the
        assumption of normality) for every feature. Features with zero variance get NaN.
    """
    X = np.asarray(X, dtype=np.float64)
    w = sp.csr_matrix(w, dtype=np.float64)
    N = X.shape[0]
    W = w.sum()

    # diag(X.T @ w @ X) without computing the full features × features matrix
    numerators = np.einsum("ij,ij->j", X, np.asarray(w @ X))
    denominators = np.sum(X ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
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

    return morans_i, z_scores, p_values


def _dense_X(adata: sc.AnnData, layer: None | str) -> np.ndarray:
    X = adata.X if layer is None else adata.layers[layer]
    if sp.issparse(X):
        logg.warning("provided matrix is sparse, make sure that you scaled data before passing to the function")
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _spca_eigh(X: np.ndarray, W: sp.spmatrix, n_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    n_obs, n_genes = X.shape
    if n_obs >= n_genes:
        evals, evecs = np.linalg.eigh(X.T @ np.asarray(W @ X))
        order = np.argsort(evals)[::-1][:n_pcs]
        return evals[order], evecs[:, order]

    # With X = U @ diag(s) @ V.T (U, s from the Gram matrix X @ X.T), X.T @ W @ X =
    # V @ (diag(s) @ U.T @ W @ U @ diag(s)) @ V.T: the nonzero eigenpairs come from the small
    # N × N matrix, the remaining eigenvalues are zeros with eigenvectors orthogonal to V.
    # ARPACK on the full genes × genes matrix is slow and fails to converge in the degenerate
    # zero eigenspace, and the Gram matrix is cheaper than a QR decomposition of X.T.
    s, U = np.linalg.eigh(X @ X.T)
    keep = s > s.max() * n_genes * np.finfo(np.float64).eps
    U, s = U[:, keep], np.sqrt(s[keep])
    evals, Y = np.linalg.eigh(s[:, None] * np.asarray(U.T @ (W @ U)) * s[None, :])
    order = np.argsort(evals)[::-1]
    pos, neg = order[evals[order] > 0][:n_pcs], order[evals[order] <= 0]
    n_zero = min(n_pcs - len(pos), n_genes - len(s))
    neg = neg[:n_pcs - len(pos) - n_zero]

    vecs = [X.T @ (U @ (Y[:, pos] / s[:, None]))]
    if n_zero > 0:
        V = X.T @ (U / s[None, :])
        Z = np.random.default_rng(0).standard_normal((n_genes, n_zero))
        vecs.append(np.linalg.qr(Z - V @ (V.T @ Z))[0])
    vecs.append(X.T @ (U @ (Y[:, neg] / s[:, None])))
    return np.concatenate([evals[pos], np.zeros(n_zero), evals[neg]]), np.hstack(vecs)


def spca(
    adata: sc.AnnData,
    layer: None | str = None,
    graph_key: str = "connectivities",
    n_pcs: int = 50,
    key_added: str = "X_sPCA",
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
    key_added : str
        Key to use for adding the results to the AnnData object.

    Returns
    -------
    None
    """
    start = logg.info(f"computing sPCA")

    X = _dense_X(adata, layer)
    W = sp.csr_matrix(adata.obsp[graph_key], dtype=np.float64)

    total_variance = np.sum(X ** 2)

    evals, evecs = _spca_eigh(X, W, n_pcs)
    scores = X @ evecs

    adata.varm["sPCs"] = evecs
    adata.obsm[key_added] = scores
    adata.uns["spca"] = {
        "evals": evals,
        "varexpl": evals / total_variance,
        "graph_key": graph_key,
        "n_pcs": n_pcs,
    }

    lines = [
        "added",
        f"     .obsm['{key_added}'] {n_pcs} supervised principal components",
        f"     .varm['sPCs'] matrix of loadings",
        f"     .uns['spca'] eigenvalues and variance explained",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)


def moran(
    adata: sc.AnnData,
    layer: None | str = None,
    graph_key: str = "connectivities",
    adj_method: str = "fdr_bh",
) -> None:
    """
    Computes Moran's I of every gene over the graph and its analytical p-value
    (normal approximation).

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the data.
    layer : None | str
        Layer to use for the analysis.
    graph_key : str
        Key to use for the graph.
    adj_method : str
        Method for adjusting p-values, by default "fdr_bh".

    Returns
    -------
    None
    """
    start = logg.info(f"computing Moran's I")

    morans_i, _, p_values = _morans_i(_dense_X(adata, layer), adata.obsp[graph_key])

    adata.var["morans_i"] = morans_i
    adata.var["morans_i_pval"] = p_values
    adata.var["morans_i_p_adj"] = _adjust_pvalues(p_values, method=adj_method)

    lines = [
        "added",
        "     .var['morans_i'] Moran's I statistics",
        "     .var['morans_i_pval'] analytical p-values",
        "     .var['morans_i_p_adj'] adjusted p-values",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)


def _graph_adata(
    clones: sc.AnnData,
    max_value: float | None,
    n_neighbors: int | None,
    use_rep: str | None,
) -> sc.AnnData:
    """
    Scales clone-level expression (within `split_obs` groups followed by
    `c2v.utils.stack_layers` if `split_obs` was used in `c2v.pp.transfer_expression`) and
    builds the clonal kNN-graph, as `graph_associations` does.

    Parameters
    ----------
    clones : sc.AnnData
        Clone-level expression returned by `c2v.pp.transfer_expression`.
    max_value : float | None
        Clipping value for `sc.pp.scale`.
    n_neighbors : int | None
        Number of neighbors for the clonal graph, if None, the existing graph is kept.
    use_rep : str | None
        Representation for `sc.pp.neighbors`.

    Returns
    -------
    sc.AnnData
        Copy of `clones` with scaled expression in `.X` and the graph in `.obsp`.
    """

    split_obs = clones.uns["transfer_expression"].get("split_obs")
    adata = clones.copy()
    if split_obs is None:
        logg.info("    scaling expression")
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
        with _silenced():
            sc.pp.scale(adata, max_value=max_value)
    else:
        logg.info(f"    scaling expression within '{split_obs}' groups and stacking them")
        mask = adata.obsm[adata.uns["presence_key"]]
        with _silenced():
            for layer in adata.layers.keys():
                if sp.issparse(adata.layers[layer]):
                    adata.layers[layer] = adata.layers[layer].toarray()
                sc.pp.scale(adata, layer=layer, max_value=max_value, mask_obs=mask[layer].values > 0)
            adata = stack_layers(adata)

    if n_neighbors is not None:
        logg.info(f"    building kNN-graph on '{use_rep}' with {n_neighbors} neighbors")
        with _silenced():
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    return adata


def graph_associations(
    clones: sc.AnnData,
    max_value: float | None = 10,
    n_neighbors: int | None = 10,
    use_rep: str | None = None,
    n_pcs: int = 50,
    adj_method: str = "fdr_bh",
) -> None:
    """
    Computes Moran's I and sPCA of clone-level gene expression over the clonal graph.

    Pipeline: `sc.pp.scale` (per `split_obs` group with
    `c2v.utils.stack_layers` if `split_obs` was used in `c2v.pp.transfer_expression`) →
    `sc.pp.neighbors` (if `n_neighbors` is not None) → `c2v.tl.moran` → `c2v.tl.spca`.

    Parameters
    ----------
    clones : sc.AnnData
        Clone-level expression returned by `c2v.pp.transfer_expression`.
    max_value : float | None
        Clipping value for `sc.pp.scale`, by default 10.
    n_neighbors : int | None
        Number of neighbors for the clonal graph. If None, `clones.obsp['connectivities']`
        is used, by default 10.
    use_rep : str | None
        Representation for `sc.pp.neighbors`. If None, "clone2vec" or "clone2vec_Poi"
        is used, by default None.
    n_pcs : int
        Number of supervised principal components, by default 50.
    adj_method : str
        Method for adjusting Moran's I p-values, by default "fdr_bh".

    Returns
    -------
    None
        Updates `clones` with Moran's I and its analytical p-values (`.var['morans_i']`,
        `.var['morans_i_pval']`, `.var['morans_i_p_adj']`), sPCA results (`.obsm['X_sPCA']`,
        `.varm['sPCs']`, `.uns['spca']`), and the parameters (`.uns['graph_associations']`)
        inherited by `c2v.tl.permuted_graph_associations`.
    """
    if "transfer_expression" not in clones.uns:
        raise ValueError("run `c2v.pp.transfer_expression` before `c2v.tl.graph_associations`")
    if n_neighbors is not None and use_rep is None:
        if "clone2vec" in clones.obsm:
            use_rep = "clone2vec"
        elif "clone2vec_Poi" in clones.obsm:
            use_rep = "clone2vec_Poi"
        else:
            raise KeyError("neither 'clone2vec' nor 'clone2vec_Poi' found in clones.obsm, provide `use_rep`")
    split_obs = clones.uns["transfer_expression"].get("split_obs")

    start = logg.info("computing graph associations")
    adata = _graph_adata(clones, max_value=max_value, n_neighbors=n_neighbors, use_rep=use_rep)

    logg.info("    computing Moran's I and sPCA")
    with _silenced():
        moran(adata, adj_method=adj_method)
        spca(adata, n_pcs=n_pcs)

    clones.varm["sPCs"] = adata.varm["sPCs"]
    clones.uns["spca"] = adata.uns["spca"]
    for col in ("morans_i", "morans_i_pval", "morans_i_p_adj"):
        clones.var[col] = adata.var[col].values

    scores = pd.DataFrame(adata.obsm["X_sPCA"], index=adata.obs_names)
    if split_obs is not None:
        # stacked names are "<layer>:<clone>", positions of one clone are averaged across layers
        layers = adata.obs["layer"].astype(str).values
        clone_names = [name[len(l) + 1:] for name, l in zip(adata.obs_names, layers)]
        scores = scores.groupby(clone_names).mean()
    clones.obsm["X_sPCA"] = scores.reindex(clones.obs_names).values
    params = {"max_value": max_value, "n_neighbors": n_neighbors, "use_rep": use_rep, "n_pcs": n_pcs}
    clones.uns["graph_associations"] = {k: v for k, v in params.items() if v is not None}

    lines = [
        "added",
        f"     .obsm['X_sPCA'] {n_pcs} supervised principal components",
        "     .varm['sPCs'] matrix of loadings",
        "     .uns['spca'] eigenvalues and variance explained",
        "     .var['morans_i'] Moran's I statistics",
        "     .var['morans_i_pval'], .var['morans_i_p_adj'] analytical p-values",
        "     .uns['graph_associations'] parameters",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)


def associations(
    clones: sc.AnnData,
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
    presence_key: str | None | Literal[False] = None,
    clr_pseudocount: float = 1e-3,
    binomial: bool = False,
    **gam_kwargs,
) -> None:
    """
    Calculate correlation between features in `clones` and a response variable.
    Correlation coefficients, p-values, and False Discovery Rates (FDRs) are computed for each feature.
    Results are stored in `clones.varm` with keys "{X_name}:corr", "{X_name}:pvalue", and "{X_name}:FDR",
    where {X_name} is the name of the feature representation (e.g., "X", "X_raw", or layer names).
    If `p_adjustment_method` is None, no correction is performed.

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix at the clone level (e.g., returned by `c2v.pp.transfer_expression`).
    response_key : str | list[str] | None
        Key in `clones.obs` or `clones.obsm` containing the response variable. Might be a list
        only if `response_field='obs'`, in this case the response variable is a multi-column DataFrame.
    response_field : Literal["obs", "obsm"], optional
        Whether the response variable is in `clones.obs` or `clones.obsm`, by default "obsm".
    response_transform : None | Literal["logit", "log1p", "sqrt", "clr"], optional
        Transform to apply to the response variable before correlation calculation, by default None.
    use_rep : Literal["X", "layers"] | None, optional
        Whether to use `clones.X` or `clones.layers` for feature representations, if None, uses 
        `clones.X` if `layers` is None, otherwise `clones.layers`, by default None.
    use_raw : bool | None, optional
        Whether to use `clones.raw.X` as the feature representation, by default None.
    layers : list[str] | str | None, optional
        Layers in `clones.layers` to use for feature representations if `use_rep='layers'`. If None and
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
    presence_key : str | None | Literal[False], optional
        Key in `clones.obs` or `clones.obsm` containing a boolean mask to filter clones, by default None.
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
        The `clones` object is modified in place with correlation results stored in `clones.varm`.
    """
    if presence_key is None and "presence_key" in clones.uns.keys():
        presence_key = clones.uns["presence_key"]

    if use_rep is None:
        if layers is None:
            use_rep = "X"
        else:
            use_rep = "layers"
    if use_raw and use_rep == "layers":
        logg.warning("use_raw=True is not supported when use_rep='layers'. Setting use_raw=False.")
        use_raw = False
    if (use_raw is None) and (use_rep == "X"):
        if clones.raw:
            logg.warning("Using raw data when use_raw is None and clones.raw is not None. Setting use_raw=True.")
            use_raw = True
    if use_rep == "X" and layers:
        logg.warning("Using layers when use_rep='X' is not supported. Setting use_rep='layers'.")
        use_rep = "layers"

    start_lines = "\n"
    if use_raw:
        Xs = [clones.raw.X]
        masks = [~_nan_mask(clones.raw.X)]
        X_names = ["X_raw"]
        var_names = clones.raw.var_names.values
        start_lines += "    using `clones.raw.X` as features\n"
    elif use_rep == "X":
        Xs = [clones.X]
        masks = [~_nan_mask(clones.X)]
        X_names = ["X"]
        var_names = clones.var_names.values
        start_lines += "    using `clones.X` as features\n"
    else:
        if layers is None:
            layers = list(clones.layers.keys())
        elif isinstance(layers, str):
            layers = [layers]
        Xs = [clones.layers[layer] for layer in layers]
        masks = [~_nan_mask(X) for X in Xs]
        if presence_key:
            masks = [masks[i] & (clones.obsm[presence_key][layer].values > 0) for i, layer in enumerate(layers)]
        if concat_layers:
            if sp.issparse(Xs[0]):
                Xs = [sp.hstack(Xs, format="csr")]
            else:
                Xs = [np.concatenate(Xs, axis=0)]
            masks = [np.concatenate(masks, axis=0)]
            X_names = ["multi_layer"]
        else:
            X_names = [layer for layer in layers]
        var_names = clones.var_names.values
        if concat_layers:
            start_lines += f"    using layers {layers} (concatenated) as features\n"
        else:
            start_lines += f"    using layers {layers} as features\n"
    
    Xs = [X[mask] for X, mask in zip(Xs, masks)]
    
    if response_field == "obs":
        if isinstance(response_key, str):
            response_key_added = response_key
            Y = clones.obs[response_key].values[:, None]
            y_names = [response_key]
            start_lines += f"    using `clones.obs['{response_key}']` as response"
        else:
            response_key_added = "+".join(response_key)
            Y = clones.obs[response_key].values
            y_names = response_key.columns.values.astype(str)
            start_lines += f"    using `clones.obs[{y_names}]` as response"
    elif response_field == "obsm":
        if isinstance(response_key, list):
            raise ValueError("response_key must be a single string when response_field='obsm'")
        try:
            response_key_added = response_key
            y = clones.obsm[response_key]
            if isinstance(y, pd.DataFrame):
                y_names = y.columns.values.astype(str)
                y = y.values
            else:
                y_names = [f"{response_key}:{i}" for i in range(y.shape[1])]
            start_lines += f"    using `clones.obsm['{response_key}']` as response"
        except KeyError:
            raise KeyError(f"Response key '{response_key}' not found in clones.obsm. Please provide a valid response key")
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
            X_res = fast_corr(X, Y_masked, **corr_kwargs)
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
                clones.raw.varm[f"{response_key_added}:{X_name}:{method}:{key}"] = pd.DataFrame(
                    results[X_name][key],
                    index=var_names,
                    columns=y_names,
                )
            else:
                clones.varm[f"{response_key_added}:{X_name}:{method}:{key}"] = pd.DataFrame(
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


def _masked_score(
    clones: sc.AnnData,
    groupby: str,
    reference: str | list[str],
    test: str | list[str],
    masked_embedding: str,
    n_neighbors: int,
    keep_clones: Literal["exact", "similar"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.neighbors import kneighbors_graph

    if masked_embedding not in clones.obsm:
        raise KeyError(
            f"Masked embedding '{masked_embedding}' not found in clones.obsm. "
            "Run `tl.mask_heterogeneity` first or provide a valid `masked_embedding`."
        )
    if groupby not in clones.obs.columns:
        raise KeyError(f"groupby '{groupby}' not found in clones.obs. Please provide a valid groupby.")
    if keep_clones not in ("exact", "similar"):
        raise ValueError(f"keep_clones must be one of {{'exact', 'similar'}}, got '{keep_clones}'.")
    if not 0 < n_neighbors < clones.n_obs:
        raise ValueError(f"n_neighbors must be between 1 and {clones.n_obs - 1} (number of clones - 1), got {n_neighbors}.")

    reference = [str(reference)] if isinstance(reference, (str, int)) else [str(g) for g in reference]
    test = [str(test)] if isinstance(test, (str, int)) else [str(g) for g in test]
    if set(reference) & set(test):
        raise ValueError(f"`reference` and `test` overlap: {sorted(set(reference) & set(test))}.")
    labels = clones.obs[groupby].astype(str)
    missing = [g for g in reference + test if g not in set(labels)]
    if missing:
        raise KeyError(f"Groups {missing} not found in clones.obs['{groupby}'].")

    G = kneighbors_graph(np.asarray(clones.obsm[masked_embedding]), n_neighbors=n_neighbors, include_self=False)
    is_test = labels.isin(test).values
    is_ref = labels.isin(reference).values
    n_test = G @ is_test.astype(float)
    n_ref = G @ is_ref.astype(float)

    if keep_clones == "exact":
        keep = is_test | is_ref
    else:
        keep = (n_test + n_ref) > 0
    d = (n_test - n_ref)[keep] / (2 * n_neighbors)
    if keep.sum() < 3:
        raise ValueError(f"Only {keep.sum()} clones kept for testing, at least 3 are required.")
    if np.all(d == d[0]):
        raise ValueError("Neighbourhood scores are constant across kept clones, nothing to test against.")
    return d, keep, is_test, is_ref

def masked_markers(
    clones: sc.AnnData,
    groupby: str,
    reference: str | list[str],
    test: str | list[str],
    masked_embedding: str = "clone2vec_mask",
    method: Literal["regression", "spearman", "deseq2"] = "regression",
    n_neighbors: int = 10,
    keep_clones: Literal["exact", "similar"] = "exact",
) -> pd.DataFrame:
    """
    Differential expression between clone groups along the masked clonal embedding.

    Builds an exact kNN graph on `clones.obsm[masked_embedding]` and scores every clone as
    `(n_test - n_reference) / (2 * n_neighbors)`, where `n_*` is the number of neighbours
    from the corresponding groups. Clone-level expression (`clones.X`, e.g. after
    `pp.transfer_expression`) is then tested against this score.

    Parameters
    ----------
    clones : AnnData
        Clone-level AnnData with expression in `.X` and the masked embedding in `.obsm`.
    groupby : str
        Column in `clones.obs` with clone groups.
    reference : str | list[str]
        Reference group(s); must not overlap with `test`.
    test : str | list[str]
        Test group(s); must not overlap with `reference`.
    masked_embedding : str, optional
        Key in `clones.obsm` with the embedding from `tl.mask_heterogeneity`, by default "clone2vec_mask".
    method : Literal["regression", "spearman", "deseq2"], optional
        "regression": t-statistic of the linear regression slope of expression on the score;
        "spearman": Spearman correlation between expression and the score;
        "deseq2": Wald statistic of the score coefficient of a DESeq2 model (`~score`, requires
        `pydeseq2`) fitted on raw counts, i.e. `clones.X` after `pp.transfer_expression` with
        `strategy="sum"` on integer counts. By default "regression".
    n_neighbors : int, optional
        Number of neighbours in the kNN graph, by default 10.
    keep_clones : Literal["exact", "similar"], optional
        "exact": only clones from `reference` and `test` are used; "similar": clones with at
        least one `reference` or `test` clone among neighbours are used. By default "exact".

    Returns
    -------
    pd.DataFrame
        Indexed by gene names, with columns "scores" (t-statistic, Spearman rho, or DESeq2 Wald
        statistic), "logfoldchanges" (slope per unit of the score in log2 units), "pvals",
        "pvals_adj" (Benjamini-Hochberg),
        sorted by "scores" in descending order. Positive scores mean higher expression in `test`.
    """
    if method not in ("regression", "spearman", "deseq2"):
        raise ValueError(f"method must be one of {{'regression', 'spearman', 'deseq2'}}, got '{method}'.")
    if "transfer_expression" not in clones.uns:
        logg.warning(
            "clones.uns['transfer_expression'] not found: `clones.X` is assumed to contain "
            "clone-level expression (see `pp.transfer_expression`)."
        )

    start = logg.info(
        f"testing markers of {test} vs {reference} along masked embedding",
        deep=f"\n    using `clones.obsm['{masked_embedding}']` with {n_neighbors} neighbours and {method} test",
    )
    d, keep, is_test, is_ref = _masked_score(
        clones, groupby, reference, test, masked_embedding, n_neighbors, keep_clones
    )

    Y = clones.X[keep]
    Y = np.asarray(Y.toarray() if sp.issparse(Y) else Y, dtype=float)
    ok = Y.std(0) > 0
    Y, names = Y[:, ok], clones.var_names[ok]
    dof = len(d) - 2

    if method == "deseq2":
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        strategy = clones.uns.get("transfer_expression", {}).get("strategy")
        if strategy != "sum":
            logg.warning(
                f"DESeq2 requires raw counts summed per clone, but `clones.X` was aggregated with "
                f"strategy '{strategy}'. Use `pp.transfer_expression(..., strategy='sum')` on counts."
            )
        if not np.allclose(Y, np.rint(Y)):
            logg.warning("`clones.X` contains non-integer values, they are rounded for DESeq2.")

        clone_ids = clones.obs_names[keep]
        counts = pd.DataFrame(np.rint(Y).astype(int), index=clone_ids, columns=names)
        meta = pd.DataFrame({"score": d}, index=clone_ids)
        dds = DeseqDataSet(
            counts=counts, metadata=meta, design="~score",
            size_factors_fit_type="poscounts", refit_cooks=False, quiet=True,
        )
        dds.deseq2()
        # contrast over [Intercept, score] coefficients
        ds = DeseqStats(
            dds, contrast=np.array([0.0, 1.0]),
            cooks_filter=False, independent_filter=False, quiet=True,
        )
        ds.summary()
        r = ds.results_df.loc[names]
        res = pd.DataFrame({
            "scores": r["stat"].values,
            "logfoldchanges": r["log2FoldChange"].values,
            "pvals": r["pvalue"].values,
            "pvals_adj": r["padj"].values,
        }, index=names).sort_values("scores", ascending=False)
    else:
        dc, Yc = d - d.mean(), Y - Y.mean(0)
        slope = dc @ Yc / (dc @ dc)
        if method == "regression":
            se = np.sqrt(((Yc - np.outer(dc, slope)) ** 2).sum(0) / dof / (dc @ dc))
            scores = slope / se
            t = scores
        else:
            dr = stats.rankdata(d)
            dr -= dr.mean()
            Yr = stats.rankdata(Y, axis=0)
            Yr -= Yr.mean(0)
            scores = dr @ Yr / np.sqrt((dr @ dr) * (Yr ** 2).sum(0))
            t = scores * np.sqrt(dof / (1 - scores ** 2))
        p = 2 * stats.t.sf(np.abs(t), dof)

        res = pd.DataFrame({
            "scores": scores,
            "logfoldchanges": slope / np.log(2),
            "pvals": p,
            "pvals_adj": stats.false_discovery_control(p),
        }, index=names).sort_values("scores", ascending=False)

    n_kept_test, n_kept_ref = int((keep & is_test).sum()), int((keep & is_ref).sum())
    n_signif = res["pvals_adj"] < 0.05
    lines = [
        "returned DataFrame indexed by genes with 'scores', 'logfoldchanges', 'pvals', 'pvals_adj'",
        f"     {int(keep.sum())} clones used ({n_kept_test} test, {n_kept_ref} reference, "
        f"{int(keep.sum()) - n_kept_test - n_kept_ref} neighbouring)",
        f"     {len(names)} genes tested ({int((~ok).sum())} constant genes skipped)",
        f"     {int((n_signif & (res['scores'] > 0)).sum())} genes up and "
        f"{int((n_signif & (res['scores'] < 0)).sum())} down in test (pvals_adj < 0.05)",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
    return res


def _spca_evals(X: np.ndarray, w: sp.spmatrix, n_pcs: int) -> np.ndarray:
    """
    Leading sPCA eigenvalues (maxima of the Rayleigh quotient v.T @ X.T @ w @ X @ v / v.T @ v)
    scaled by N / (sum(w) * sum(X ** 2)), i.e., `varexpl` of `c2v.tl.spca` in Moran's I units.
    The scaling removes the dependence on the number of clones and the total graph weight, which
    differ between permutations, so the eigenvalues of different graphs are comparable; for a
    single gene the scaled eigenvalue equals its Moran's I.

    Parameters
    ----------
    X : np.ndarray
        Scaled expression matrix (observations × genes).
    w : sp.spmatrix
        Graph of shape (observations × observations).
    n_pcs : int
        Number of leading eigenvalues.

    Returns
    -------
    np.ndarray
        Scaled eigenvalues in decreasing order.
    """
    X = np.asarray(X, dtype=np.float64)
    w = sp.csr_matrix(w, dtype=np.float64)
    evals = _spca_eigh(X, w, n_pcs)[0]
    return evals * X.shape[0] / (w.sum() * np.sum(X ** 2))


def permuted_graph_associations(
    adata: sc.AnnData,
    clones: sc.AnnData,
    permutations_key: str = "clonal_permutations",
    clones_full: sc.AnnData | None = None,
    adj_method: str = "fdr_bh",
    progress_bar: bool = True,
    key_added: str = "permuted_graph_associations",
) -> None:
    """
    Tests whether Moran's I of clone-level gene expression over the clonal embedding
    kNN-graph, and the variance explained by sPCA, are higher than the ones obtained with
    clonally reshuffled data.

    The pipeline of `c2v.tl.graph_associations` (which has to be run on `clones`
    beforehand) is repeated for every permuted labelling calculated by
    `permute_embeddings`, inheriting all its parameters. Under a permutation, both the
    clone-level expression and the embedding are derived from the same permuted labels:
    expression is re-aggregated from `adata` by `c2v.pp.transfer_expression` with the
    parameters stored in `clones.uns['transfer_expression']` (including `split_obs`, `mask`,
    and the expression source), and the kNN-graph is built on the permuted embedding with
    the parameters stored in `clones.uns['graph_associations']`. Only the genes present in
    `clones` are used. The permuted labels are matched to the cells of `adata` by name.

    sPCA eigenvalues (maxima of the Rayleigh quotient of `X.T @ W @ X`) are compared after
    scaling by N / (sum(W) * sum(X ** 2)) – `varexpl` in Moran's I units – since the number
    of clones and the total weight of the graph change between permutations; the k-th
    observed eigenvalue is compared both with the k-th eigenvalue of every permutation and
    with the leading one (variance explained by sPC_1 without clonal structure).

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix at the cell level that expression of `clones` was transferred
        from; all its cells have to be used in `permute_embeddings`.
    clones : sc.AnnData
        Clone-level expression returned by `c2v.pp.transfer_expression` with the results of
        `c2v.tl.graph_associations` and `permute_embeddings`.
    permutations_key : str, optional
        `key_added` used in `permute_embeddings`, by default "clonal_permutations".
    clones_full : sc.AnnData | None, optional
        Clone-level object `permute_embeddings` was run on, if `clones` contains only a subset
        of its clones (e.g., clones without expression removed by `transfer_expression`).
        If provided, permuted embeddings are taken from it and every permutation starts from
        all its clones, so the set of clones retained after the transfer can differ from the
        observed one. If None, permutations are taken from `clones`, by default None.
    adj_method : str, optional
        Method for adjusting p-values, by default "fdr_bh".
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    key_added : str, optional
        Key in `clones.uns` to store the results, by default "permuted_graph_associations".

    Returns
    -------
    None
        `clones.var` is updated (next to `morans_i` and `morans_i_pval` of
        `graph_associations`) with the mean and the standard deviation of the null
        (`null_mean`, `null_std`), the number of permutations with a defined Moran's I
        (`n_permutations`), the empirical one-sided p-value (`perm_pval`, `perm_p_adj`), the
        z-score with respect to the null and its normal p-value (`z_score`, `z_pval`,
        `z_p_adj`), and D'Agostino-Pearson normality test of the null (`normaltest_stat`,
        `normaltest_pval`). `clones.uns[key_added]` stores
        `null_distribution` – a data frame (genes × permutations) with Moran's I values
        under permutations; `spca` – a data frame (sPCs × statistics) with the observed
        variance explained (`varexpl`), the scaled eigenvalue (`eval_norm`), its null mean and
        standard deviation (`null_mean`, `null_std`), the empirical one-sided p-value
        (`perm_pval`), and the z-score with its normal p-value (`z_score`, `z_pval`), and the
        same statistics with respect to the null of the leading eigenvalue (`null_mean_first`,
        `null_std_first`, `perm_pval_first`, `z_score_first`, `z_pval_first`);
        `spca_null_distribution` – a data frame (sPCs × permutations) with the scaled
        eigenvalues under permutations; `n_clones` – number of clones (clone-group pairs if
        `split_obs` was used) retained in every permutation; and the parameters.
    """
    if "graph_associations" not in clones.uns:
        raise ValueError(
            "Graph associations parameters not found in `clones.uns['graph_associations']`. "
            "`permuted_graph_associations` inherits them, so `c2v.tl.graph_associations(clones)` "
            "has to be run beforehand."
        )
    perm_clones = clones if clones_full is None else clones_full
    if permutations_key not in perm_clones.uns or permutations_key not in perm_clones.obsm:
        raise KeyError(
            f"permutations_key '{permutations_key}' not found in .uns/.obsm. "
            "Did you run `c2v.tl.permute_embeddings(adata, clones)` before?"
        )
    graph_params = clones.uns["graph_associations"]
    transfer_params = clones.uns["transfer_expression"]
    params = perm_clones.uns[permutations_key]

    n_neighbors = graph_params.get("n_neighbors")
    max_value = graph_params.get("max_value")
    use_rep = graph_params.get("use_rep")
    n_pcs = graph_params.get("n_pcs", 50)
    if n_neighbors is None:
        raise ValueError(
            "`graph_associations` was run with `n_neighbors=None` (existing graph), which can't "
            "be rebuilt on the permuted embeddings. Please re-run it with `n_neighbors`."
        )
    if use_rep != params.get("use_rep"):
        logg.warning(
            f"`graph_associations` used clones.obsm['{use_rep}'], while permutations were "
            f"calculated for clones.obsm['{params.get('use_rep')}']."
        )

    obs_name = params["obs_name"]
    split_obs = transfer_params.get("split_obs")
    strategy = transfer_params.get("strategy", "average")
    use_col = transfer_params.get("use_col")
    layer = transfer_params.get("layer")
    use_raw = bool(transfer_params.get("use_raw", False))
    mask = transfer_params.get("mask", False)
    if mask == "custom":
        logg.warning(
            "Expression of `clones` was transferred with a custom cell mask that can't be "
            "repeated, `adata` is assumed to contain only the selected cells."
        )
        mask = False
    if split_obs is not None and split_obs not in adata.obs:
        raise KeyError(f"split_obs '{split_obs}' (inherited from `transfer_expression`) not found in adata.obs.")

    categories = np.asarray(params["categories"]).astype(str).astype(object)
    cell_idx = pd.Index(np.asarray(params["cells"]).astype(str)).get_indexer(
        adata.obs_names.astype(str)
    )
    n_missing = int((cell_idx < 0).sum())
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} cells of adata were not used in `permute_embeddings`. Please provide "
            "the cell-level object (or its subset) used in `permute_embeddings`."
        )
    perm_codes = np.asarray(params["codes"])[cell_idx]
    embeddings = perm_clones.obsm[permutations_key]
    n_permutations = embeddings.shape[-1]
    if perm_codes.shape[1] != n_permutations:
        raise ValueError(
            f"clones.uns['{permutations_key}']['codes'] has {perm_codes.shape[1]} permutations, "
            f"clones.obsm['{permutations_key}'] has {n_permutations}"
        )

    # Expression source, as in `transfer_expression`, restricted to the genes of `clones`
    if use_raw:
        X, var_names = adata.raw.X, adata.raw.var_names
        source = "adata.raw.X"
    elif layer is not None:
        X, var_names = adata.layers[layer], adata.var_names
        source = f"adata.layers['{layer}']"
    else:
        X, var_names = adata.X, adata.var_names
        source = "adata.X"
    gene_idx = pd.Index(var_names).get_indexer(clones.var_names)
    if (gene_idx < 0).any():
        raise KeyError(
            f"{int((gene_idx < 0).sum())} genes of clones are not found in `{source}`. Please "
            "provide the cell-level object expression was transferred from."
        )
    adata_perm = sc.AnnData(
        X=X[:, gene_idx],
        obs=adata.obs[[split_obs]].copy() if split_obs is not None else pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=clones.var_names),
    )
    clones_obs = perm_clones.obs[[use_col]] if use_col is not None else pd.DataFrame(index=perm_clones.obs_names)
    clones_uns = {mask: clones.uns[mask]} if isinstance(mask, str) else {}
    transfer_kwargs = {
        "obs_name": obs_name, "split_obs": split_obs, "use_col": use_col, "strategy": strategy,
        "use_raw": False, "layers_to_obsm": [], "mask": mask,
    }

    lines_before = [
        f"    using `{source}` as expression source ({clones.n_vars} genes of clones)",
        f"    inheriting `transfer_expression` parameters: '{strategy}' strategy"
        + (f", split by '{split_obs}'" if split_obs else "")
        + (f", mask '{mask}'" if isinstance(mask, str) else ""),
        f"    inheriting `graph_associations` parameters: kNN-graph on '{use_rep}' with "
        f"{n_neighbors} neighbors, max_value={max_value}",
    ]
    start = logg.info("computing permuted graph associations", deep="\n" + "\n".join(lines_before))

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    with _silenced():
        graph = _graph_adata(clones, max_value=max_value, n_neighbors=n_neighbors, use_rep=use_rep)
        morans_i = _morans_i(graph.X, graph.obsp["connectivities"])[0]
        spca_evals = _spca_evals(graph.X, graph.obsp["connectivities"], n_pcs)
    n_clones_observed = graph.n_obs

    null = np.full((clones.n_vars, n_permutations), np.nan)
    spca_null = np.full((n_pcs, n_permutations), np.nan)
    n_clones_perm = np.zeros(n_permutations, dtype=int)
    cm = tqdm(
        range(n_permutations),
        desc=prefix + f"computing Moran's I and sPCA ({n_permutations} permutations)",
        file=sys.stdout,
    ) if progress_bar else nullcontext(range(n_permutations))
    with _silenced(), cm as iterator:
        for i in iterator:
            codes = perm_codes[:, i]
            adata_perm.obs[obs_name] = np.where(codes >= 0, categories[np.maximum(codes, 0)], None)
            clones_perm = sc.AnnData(
                obs=clones_obs.copy(),
                obsm={use_rep: embeddings[..., i]},
                uns=dict(clones_uns),
            )
            clones_perm = transfer_expression(adata_perm, clones_perm, **transfer_kwargs)
            graph = _graph_adata(clones_perm, max_value=max_value, n_neighbors=n_neighbors, use_rep=use_rep)
            null[:, i] = _morans_i(graph.X, graph.obsp["connectivities"])[0]
            spca_null[:, i] = _spca_evals(graph.X, graph.obsp["connectivities"], n_pcs)
            n_clones_perm[i] = graph.n_obs

    # Statistics
    finite = np.isfinite(null)
    n_valid = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        null_mean = np.nanmean(null, axis=1)
        null_std = np.nanstd(null, axis=1, ddof=1)
        z_scores = (morans_i - null_mean) / null_std
    z_pval = stats.norm.sf(z_scores)
    perm_pval = (1.0 + np.sum(finite & (null >= morans_i[:, None]), axis=1)) / (n_valid + 1.0)
    perm_pval[~np.isfinite(morans_i)] = np.nan

    # D'Agostino-Pearson test requires at least 8 values (20 for a reliable kurtosis test)
    normal_stat = np.full(null.shape[0], np.nan)
    normal_pval = np.full(null.shape[0], np.nan)
    complete = (n_valid == n_permutations) & (null_std > 0)
    if n_permutations >= 8 and complete.any():
        res = stats.normaltest(null[complete], axis=1)
        normal_stat[complete], normal_pval[complete] = res.statistic, res.pvalue
    for j in np.flatnonzero(~complete & (n_valid >= 8) & (null_std > 0)):
        res = stats.normaltest(null[j, finite[j]])
        normal_stat[j], normal_pval[j] = res.statistic, res.pvalue

    results = pd.DataFrame({
        "null_mean": null_mean,
        "null_std": null_std,
        "n_permutations": n_valid,
        "perm_pval": perm_pval,
        "perm_p_adj": _adjust_pvalues(perm_pval, method=adj_method),
        "z_score": z_scores,
        "z_pval": z_pval,
        "z_p_adj": _adjust_pvalues(z_pval, method=adj_method),
        "normaltest_stat": normal_stat,
        "normaltest_pval": normal_pval,
    }, index=clones.var_names)
    for col in results.columns:
        clones.var[col] = results[col].values

    # The k-th observed eigenvalue is compared with the k-th eigenvalue of every permutation
    # and with the leading one (variance explained by sPC_1 without clonal structure)
    first_null = np.broadcast_to(spca_null[:1], spca_null.shape)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spca_null_mean = np.nanmean(spca_null, axis=1)
        spca_null_std = np.nanstd(spca_null, axis=1, ddof=1)
        spca_z = (spca_evals - spca_null_mean) / spca_null_std
        first_z = (spca_evals - spca_null_mean[0]) / spca_null_std[0]
    spca_finite = np.isfinite(spca_null)
    first_finite = np.isfinite(first_null)
    pc_names = [f"sPC_{k + 1}" for k in range(n_pcs)]
    spca_results = pd.DataFrame({
        "varexpl": np.asarray(clones.uns["spca"]["varexpl"])[:n_pcs],
        "eval_norm": spca_evals,
        "null_mean": spca_null_mean,
        "null_std": spca_null_std,
        "perm_pval": (1.0 + np.sum(spca_finite & (spca_null >= spca_evals[:, None]), axis=1))
                     / (spca_finite.sum(axis=1) + 1.0),
        "z_score": spca_z,
        "z_pval": stats.norm.sf(spca_z),
        "null_mean_first": spca_null_mean[0],
        "null_std_first": spca_null_std[0],
        "perm_pval_first": (1.0 + np.sum(first_finite & (first_null >= spca_evals[:, None]), axis=1))
                           / (first_finite.sum(axis=1) + 1.0),
        "z_score_first": first_z,
        "z_pval_first": stats.norm.sf(first_z),
    }, index=pc_names)

    run_params = {
        "permutations_key": permutations_key,
        "use_rep": use_rep,
        "n_neighbors": n_neighbors,
        "max_value": max_value,
        "n_pcs": n_pcs,
        "source": source,
        "strategy": strategy,
        "split_obs": split_obs,
    }
    perm_names = [f"perm_{i}" for i in range(n_permutations)]
    clones.uns[key_added] = {
        "null_distribution": pd.DataFrame(null, index=clones.var_names, columns=perm_names),
        "spca": spca_results,
        "spca_null_distribution": pd.DataFrame(spca_null, index=pc_names, columns=perm_names),
        "n_clones": n_clones_perm,
        "n_clones_observed": int(n_clones_observed),
        "params": {k: v for k, v in run_params.items() if v is not None},
    }

    lines = [
        "added",
        "     .var['null_mean'], .var['null_std'], .var['n_permutations'] Moran's I null",
        "     .var['perm_pval'], .var['perm_p_adj'] permutation p-values",
        "     .var['z_score'], .var['z_pval'], .var['z_p_adj'] z-scores with respect to the null",
        "     .var['normaltest_stat'], .var['normaltest_pval'] normality of the null",
        f"     .uns['{key_added}']['null_distribution'] Moran's I under permutations (genes × permutations)",
        f"     .uns['{key_added}']['spca'] significance of sPCA eigenvalues",
        f"     .uns['{key_added}']['spca_null_distribution'] normalized sPCA eigenvalues under permutations",
        f"     .uns['{key_added}']['n_clones'] number of clones retained in every permutation",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
