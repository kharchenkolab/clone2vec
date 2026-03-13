from __future__ import annotations

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd

import os
import sys

from typing import Literal
from tqdm import tqdm

from .utils import _nan_mask
from .associations import _clr, _fast_corr_cols

logg = sc.logging

class _catboost_tqdm:
    def __init__(self, total_iterations, show: bool = True, prefix: str = ""):
        if show:
            self.pbar = tqdm(
                total=total_iterations,
                desc=f"{prefix}training",
                file=sys.stdout,
            )
        else:
            self.pbar = None
    def write(self, message: str):
        if self.pbar:
            message_stripped = message.strip()
            if message_stripped and message_stripped[0].isdigit() and ":" in message_stripped:
                self.pbar.update(1)
                try:
                    parts = message_stripped.split("\t")
                    self.pbar.set_postfix({
                        "train": float(parts[1].split(":")[-1]),
                        "validation": float(parts[2].split(":")[-1]),
                    })
                except Exception:
                    pass
    def flush(self):
        if self.pbar:
            sys.stdout.flush()
    def close(self):
        if self.pbar:
            self.pbar.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
        self.pbar = None

def _prepare_pools(
    adata,
    obsm_key: str,
    gs_key: str | None = None,
    validation_key: str | None = None,
    validation_value: str | None = None,
    features: list[str] | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    response_transform: None | Literal["logit", "log1p", "sqrt", "clr"] = None,
    clr_pseudocount: float = 1e-3,
    split_multi: bool = False,
    min_size: int = 3,
    loss: str = "MultiRMSE",
) -> tuple["Pool", "Pool", "Pool", np.ndarray]:
    """
    Prepares CatBoost Pools with full, train, and validation datasets.

    Parameters
    ----------
    adata : AnnData
        AnnData object with cell metadata and embeddings.
    obsm_key : str
        Key in `adata.obsm` to use as the embedding.
    gs_key : str | None, optional
        Key in `adata.uns` to use for group splitting. Default is None.
    validation_key : str | None, optional
        Column in `adata.obs` to use for validation split. Default is None.
    validation_value : str | None, optional
        Value in `validation_key` to use for validation split. Default is None.
    features : list[str] | None, optional
        List of features to use as predictors. Default is None.
    use_raw : bool | None, optional
        Whether to use `adata.raw` for feature selection. Default is None.
    layer : str | None, optional
        Layer in `adata.layers` to use for feature selection. Default is None.
    response_transform : None | Literal["logit", "log1p", "sqrt"], optional
        Transform to apply to the response variable. Default is None.
    clr_pseudocount : float, optional
        Pseudocount to add to expression values before CLR transformation, by default 1e-3.
    split_multi : bool, optional
        Whether to split multi-dimensional response into multiple pools. Default is False.
    min_size : int, optional
        Minimum amount of non-zero values for each feature in the train and validation datasets. Default is 3.

    Returns
    -------
    tuple[Pool, Pool, Pool, np.ndarray]
        Full, train, validation pools and validation mask.
    """
    try:
        from catboost import Pool
    except ImportError:
        raise ImportError("catboost package is not installed. Please install it using `pip install catboost`.")

    if validation_key is None or validation_value is None:
        if gs_key is None:
            try:
                validation_key = adata.uns["gs"]["obs_key"]
                validation_value = adata.uns["gs"]["keys"]["validation"]
            except:
                raise ValueError("gs_key not found in adata.uns. Please run sl.tl.gs() or set validation_key and validation_value.")
        else:
            validation_key = adata.uns[gs_key]["obs_key"]
            validation_value = adata.uns[gs_key]["keys"]["validation"]
    validation_mask = adata.obs[validation_key] == validation_value
    train_mask = ~validation_mask

    validation_mask = np.array(validation_mask)
    train_mask = np.array(train_mask)

    if obsm_key not in adata.obsm.keys():
        raise ValueError(f"Key {obsm_key} not found in adata_train.obsm. Please provide a valid key.")

    if adata.obsm[obsm_key].shape[1] > 1:
        label = adata.obsm[obsm_key]
    else:
        label = adata.obsm[obsm_key].flatten()

    if isinstance(label, pd.DataFrame):
        label_names = label.columns
        label = label.values
    else:
        label_names = np.arange(label.shape[1]).astype(str)

    if split_multi:
        # Checking if the label matrix is integer values
        if not np.allclose(label, label.astype(int)) and ((np.round(label.sum(axis=1), 5) > 1).sum() > 0):
            logg.warning("When using MultiClass loss, values must be probabilities or integers. Changing to MultiCrossEntropy.")
            split_multi = False
            loss = "MultiCrossEntropy"

    if response_transform == "logit":
        label = np.clip(label, 0.01, 0.99).copy()
        label = np.log(label / (1 - label))
    elif response_transform == "log1p":
        label = np.log1p(label).copy()
    elif response_transform == "sqrt":
        label = np.sqrt(label).copy()
    elif response_transform == "clr":
        label = _clr(label, epsilon=clr_pseudocount).copy()
    elif response_transform:
        logg.warning("response_transform must be None, 'logit', 'log1p', 'sqrt', or 'clr'. Setting response_transform=None.") 
        response_transform = None

    for feature in features:
        if feature not in adata.obs.columns:
            raise ValueError(f"Feature {feature} not found in adata_train.obs. Please provide a valid feature.")

    cat_features = []
    for feature in features:
        if isinstance(adata.obs[feature], pd.CategoricalDtype) or \
        isinstance(adata.obs[feature], pd.StringDtype) or \
        pd.api.types.is_object_dtype(adata.obs[feature]):
            cat_features.append(feature)

    if layer:
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    fate_size_train = (label[train_mask] != 0).sum(axis=0)
    fate_size_validation = (label[validation_mask] != 0).sum(axis=0)
    
    fates_mask_train = fate_size_train >= min_size
    fates_mask_validation = fate_size_validation >= min_size
    fates_mask = fates_mask_train & fates_mask_validation

    if sum(~fates_mask) > 0:
        logg.warning(f"Removing fates with less than {min_size} clones in validation and/or train from the analysis ({', '.join(label_names[~fates_mask])}).")

    if sp.issparse(adata.X):
        train = pd.DataFrame.sparse.from_spmatrix(
            X[train_mask],
            columns=var_names,
            index=adata[train_mask].obs.index,
        )
        validation = pd.DataFrame.sparse.from_spmatrix(
            X[validation_mask],
            columns=var_names,
            index=adata[validation_mask].obs.index,
        )
        full = pd.DataFrame.sparse.from_spmatrix(
            X,
            columns=var_names,
            index=adata.obs.index,
        )
    else:
        train = pd.DataFrame(
            X[train_mask],
            columns=var_names,
            index=adata[train_mask].obs.index,
        )
        validation = pd.DataFrame(
            X[validation_mask],
            columns=var_names,
            index=adata[validation_mask].obs.index,
        )
        full = pd.DataFrame(
            X,
            columns=var_names,
            index=adata.obs.index,
        )

    for feature in features:
        train[feature] = adata[train_mask].obs[feature]
        validation[feature] = adata[validation_mask].obs[feature]
        full[feature] = adata.obs[feature]

    label_train = label[train_mask][:, fates_mask]
    label_validation = label[validation_mask][:, fates_mask]
    n_dims_response = label_train.shape[1]

    if split_multi and n_dims_response > 1:
        validation = pd.concat([validation] * n_dims_response)
        weights_validation = label_validation.T.flatten()
        label_validation = [i for i in range(label_validation.shape[1]) for j in range(label_validation.shape[0])]

        train = pd.concat([train] * n_dims_response)
        weights_train = label_train.T.flatten()
        label_train = [i for i in range(label_train.shape[1]) for j in range(label_train.shape[0])]
    else:
        weights_train = None
        weights_validation = None

    train = Pool(data=train, label=label_train, cat_features=cat_features, weight=weights_train)
    validation = Pool(data=validation, label=label_validation, cat_features=cat_features, weight=weights_validation)
    full = Pool(data=full, label=label[:, fates_mask], cat_features=cat_features, weight=None)

    return full, train, validation, cat_features, validation_mask, fates_mask, loss

def catboost(
    adata: sc.AnnData,
    obsm_key: str,
    gs_key: str | None = None,
    validation_key: str | None = None,
    validation_value: str | None = None,
    features: list[str] | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    min_size: int = 3,
    model: Literal["regressor", "classifier"] = "regressor",
    num_trees: int = 10000,
    early_stopping_rounds: int = 100,
    verbose: bool = True,
    loss: str | None = None,
    eval_metric: str | None = None,
    response_transform: Literal["logit", "log1p", "sqrt", "clr"] | None = None,
    use_gpu: bool | None = None,
    random_state: int | None = 42,
    prediction_key_added: str = "predicted",
    return_model: bool = False,
    save_model: str | os.PathLike | None = None,
    progress_bar: bool = True,
    catboost_dir: os.PathLike | str | None = None,
    mask_key: str | None | Literal[False] = None,
    clr_pseudocount: float = 1e-3,
    **kwargs,
) -> sc.AnnData | tuple[sc.AnnData, CatBoostRegressor | CatBoostClassifier]:
    """
    Performs CatBoost regression or classification on the data aiming to identify
    associations between the features and the response variable.
    
    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    obsm_key : str
        Key in `adata.obsm` to use for the response variable.
    gs_key : str | None, optional
        Key in `adata.obs` with `sl.tl.gs()` parameters. Default is None.
    validation_key : str | None, optional
        Key in `adata.obs` to use for the train/validation split. If None, reconstructed from
        the adata.uns[gs_key]. Default is None.
    validation_value : str | None, optional
        Value in `adata.obs[validation_key]` to use for the validation set. If None,
        reconstructed from the adata.uns[gs_key]. Default is None.
    features : list[str] | None, optional
        List of additional adata.obs columns to use for the model (e.g. batch labels, layer names,
        and so on). Default is None.
    use_raw : bool | None, optional
        Whether to use `adata.raw` for feature selection. Default is None.
    layer : str | None, optional
        Layer in `adata.layers` to use for feature selection. Default is None.
    model : Literal["regressor", "classifier"], optional
        Whether to perform regression or classification. Default is "regressor".
    num_trees : int, optional
        Number of trees to build. Default is 10000.
    early_stopping_rounds : int, optional
        Number of iterations with no improvement on validation set after which training will be stopped.
        Default is 100.
    verbose : bool, optional
        Whether to print verbose output. Default is True.
    loss : str | None, optional
        Loss function to use. If None, set to "MultiRMSE" for multivariable regression,
        "RMSE" for univariable regression, "MultiCrossEntropy" for multivariable classification,
        and "CrossEntropy" for univariable classification. Default is None.
    eval_metric : str | None, optional
        Evaluation metric to use. Default is None.
    response_transform : Literal["logit", "log1p", "sqrt", "clr"] | None, optional
        Transform to apply to the response variable. Default is None.
    use_gpu : bool | None, optional
        Whether to use GPU for training. Default is None.
    random_state : int | None, optional
        Random seed for reproducibility. Default is 42.
    prediction_key_added : str, optional
        Key in `adata.obsm` to add the predicted values. Default is "predicted".
    return_model : bool, optional
        Whether to return the trained model. Default is False.
    save_model : str | os.PathLike | None, optional
        Path to save the trained model. Default is None.
    progress_bar : bool, optional
        Whether to show a progress bar during training. Default is True.
    catboost_dir : os.PathLike | str | None, optional
        Directory to save CatBoost training information. Default is None.
    mask_key : str | None | Literal[False], optional
        Key in `adata.obs` or `adata.obsm` containing a boolean mask to filter cells, by default None.
    clr_pseudocount : float, optional
        Pseudocount to add to expression values before CLR transformation, by default 1e-3.

    **kwargs
        Additional keyword arguments to pass to CatBoostRegressor or CatBoostClassifier.

    Returns
    -------
    adata : sc.AnnData
        Annotated data matrix with the predicted values added to `adata.obsm[prediction_key_added]`.
    model : CatBoostRegressor | CatBoostClassifier
        Trained CatBoost model.
    """
    import tempfile

    try:
        from catboost.utils import get_gpu_device_count
        from catboost import CatBoostRegressor, CatBoostClassifier
    except ImportError:
        raise ImportError("catboost package is not installed. Please install it using `pip install catboost`.")

    # Scanpy-style logging
    start = logg.info("training CatBoost model for associations")

    if use_gpu is None:
        use_gpu = get_gpu_device_count() > 0
    elif use_gpu and get_gpu_device_count() == 0:
        logg.warning("use_gpu=True, but no GPU devices are available. Setting use_gpu=False.")
        use_gpu = False

    if use_gpu:
        task_type = "GPU"
    else:
        task_type = "CPU"

    if adata.obsm[obsm_key].shape[1] > 1:
        multivariable = True
    else:
        multivariable = False

    if model == "regressor":
        cb_model = CatBoostRegressor
        if loss is None:
            if multivariable:
                loss = "MultiRMSE"
                split_multi = False
            else:
                loss = "RMSE"
                split_multi = False

    elif model == "classifier":
        cb_model = CatBoostClassifier

        if loss is None:
            loss = "MultiClass"
            split_multi = True
        else:
            split_multi = False

    if use_raw and layer:
        logg.warning("use_raw and layer are mutually exclusive. Setting use_raw=False.")
        use_raw = False
    if use_raw is None and adata.raw:
        logg.warning("use_raw is None and adata.raw is not None. Setting use_raw=True.")
        use_raw = True

    if layer:
        if mask_key is None and "mask_key" in adata.uns.keys():
            mask_key = adata.uns["mask_key"]
        if mask_key:
            mask = adata.obsm[mask_key][layer]
            mask = mask.values & ~_nan_mask(adata.layers[layer])
        else:
            mask = ~_nan_mask(adata.layers[layer])
    elif use_raw:
        mask = ~_nan_mask(adata.raw.X)
    else:
        mask = ~_nan_mask(adata.X)
    mask = np.array(mask)
    if sum(~mask) > 0:
        logg.warning(f"{int((~mask).sum())} clones won't be included in the analysis because they consist of missing values.")
    
    if features is None:
        features = []

    full, train, validation, cat_features, validation_mask, fates_mask, loss = _prepare_pools(
        adata[mask],
        obsm_key=obsm_key,
        gs_key=gs_key,
        validation_key=validation_key,
        validation_value=validation_value,
        features=features,
        use_raw=use_raw,
        layer=layer,
        response_transform=response_transform,
        clr_pseudocount=clr_pseudocount,
        split_multi=split_multi,
        min_size=min_size,
        loss=loss,
    )

    if eval_metric is None:
        eval_metric = loss

    with tempfile.TemporaryDirectory() as temp_dir_path:
        if catboost_dir is None:
            catboost_dir = temp_dir_path

        cb_model = cb_model(
            loss_function=loss,
            eval_metric=eval_metric,
            num_trees=num_trees,
            early_stopping_rounds=early_stopping_rounds,
            task_type=task_type,
            boosting_type="Plain",
            verbose=verbose,
            random_seed=random_state,
            train_dir=catboost_dir,
            **kwargs,
        )
        
        if sc.settings.verbosity >= 2:
            prefix = "    "
            progress_bar = True
        else:
            prefix = ""

        with _catboost_tqdm(num_trees, show=progress_bar, prefix=prefix) as custom_iterator:
            cb_model.fit(train, eval_set=validation, use_best_model=True, log_cout=custom_iterator)

    logg.info("    model training finished")

    if prediction_key_added in adata.obsm.keys():
        logg.warning(f"prediction_key_added={prediction_key_added} is already in adata.obsm.keys(). Overwriting.")

    if model == "classifier":
        prediction = cb_model.predict(full, prediction_type="Probability")
    else:
        prediction = cb_model.predict(full)

    if isinstance(adata.obsm[obsm_key], pd.DataFrame):
        prediction_cols = adata.obsm[obsm_key].columns[fates_mask]
        prediction = pd.DataFrame(
            prediction,
            columns=prediction_cols,
            index=adata[mask].obs_names,
        )
        real = adata[mask].obsm[obsm_key].iloc[:, fates_mask]
    elif sp.issparse(adata.obsm[obsm_key]):
        prediction = sp.csr_matrix(prediction)
        prediction_cols = np.arange(prediction.shape[1]).astype(str)[fates_mask]
        real = adata[mask].obsm[obsm_key][:, fates_mask]
    else:
        prediction_cols = np.arange(prediction.shape[1]).astype(str)[fates_mask]
        real = adata[mask].obsm[obsm_key][:, fates_mask]

    logg.info("    computing SHAP")
    raw_shap = cb_model.get_feature_importance(full, type="ShapValues")
    non_zero_genes = np.linalg.norm(raw_shap, axis=1)[:, :-1].sum(axis=0) > 0
    raw_shap = raw_shap[:, :, :-1][:, :, non_zero_genes]

    shapdata = sc.AnnData(
        X=sp.csr_matrix(np.linalg.norm(raw_shap, axis=1)),
        obs=pd.DataFrame(index=adata[mask].obs_names),
        var=pd.DataFrame(index=np.array(full.get_feature_names())[non_zero_genes]),
    )

    if response_transform == "logit":
        shapdata.obsm[obsm_key] = np.clip(real, 0.01, 0.99).copy()
        shapdata.obsm[obsm_key] = np.log(shapdata.obsm[obsm_key] / (1 - shapdata.obsm[obsm_key]))
    elif response_transform == "log1p":
        shapdata.obsm[obsm_key] = np.log1p(real).copy()
    elif response_transform == "sqrt":
        shapdata.obsm[obsm_key] = np.sqrt(real).copy()
    elif response_transform == "clr":
        shapdata.obsm[obsm_key] = _clr(real).copy()
    else:
        shapdata.obsm[obsm_key] = real

    shapdata.obsm[f"{obsm_key}:{prediction_key_added}"] = prediction

    # If we have integers, transforming predictions into estimated counts
    if np.allclose(real, real.astype(int)):
        shapdata.obsm[f"{obsm_key}:{prediction_key_added}"] = (
            shapdata.obsm[f"{obsm_key}:{prediction_key_added}"].T *
            shapdata.obsm[obsm_key].sum(axis=1)
        ).T

    correlations = {}
    mean_shap = {}

    num_features = [i for i in features if i not in cat_features]
    X_features = shapdata.var_names[shapdata.var_names.isin(adata.var_names)]

    logg.info(f"    calculating correlation between SHAP and expression")
    for i, pred_col in enumerate(prediction_cols):
        shapdata.layers[pred_col] = sp.csr_matrix(raw_shap[:, i, :])
        mean_shap[pred_col] = np.abs(shapdata.layers[pred_col]).mean(axis=0).A[0]

        if layer:
            X = adata[mask, X_features].layers[layer]
        elif use_raw:
            X = adata.raw[mask, X_features].X
        else:
            X = adata[mask, X_features].X

        X_corr = _fast_corr_cols(
            shapdata[:, X_features].layers[pred_col],
            X,
            significance=False,
            slope=False,
            progress_bar=False,
        )["r"]

        if len(num_features) > 0:
            obs_cor = _fast_corr_cols(
                shapdata[:, num_features].layers[pred_col],
                adata[mask].obs[num_features].values,
                significance=False,
                slope=False,
                progress_bar=False,
            )["r"]
        else:
            obs_cor = np.array([])

        correlations[pred_col] = pd.Series(
            list(X_corr) + list(obs_cor),
            index=list(X_features) + num_features,
        )

    shapdata.varm["mean_shap"] = pd.DataFrame(mean_shap, index=shapdata.var_names)
    shapdata.varm["gex_r"] = pd.DataFrame(correlations).reindex(shapdata.var_names)
    shapdata.obs["validation"] = validation_mask
    shapdata.var["mean_shap"] = shapdata.X.mean(axis=0).A[0]
    shapdata.uns["catboost_info"] = {
        "obsm_key": obsm_key,
        "validation_key": validation_key,
        "validation_value": validation_value,
        "features": features,
        "model": type(cb_model).__name__,
        "num_trees": num_trees,
        "early_stopping_rounds": early_stopping_rounds,
        "verbose": verbose,
        "loss": loss,
        "eval_metric": eval_metric,
        "response_transform": response_transform,
        "use_gpu": use_gpu,
        "random_state": random_state,
        "kwargs": kwargs,
        "layer": layer,
        "use_raw": use_raw,
        "fates_used": list(prediction_cols),
        "loss_history": cb_model.get_evals_result()["learn"][loss],
        "validation_history": cb_model.get_evals_result()["validation"][eval_metric],
    }

    lines = ["created AnnData with",
             "     .X matrix of mean absolute SHAP values",
             f"     .obsm['{obsm_key}'] coordinates",
             f"     .obsm['{obsm_key}:{prediction_key_added}'] predictions",
             "     .layers['<obsm_columns>'] SHAP values",
             "     .varm['mean_shap'] dataframe of absolute SHAP means per prediction column",
             "     .varm['gex_r'] dataframe of correlations between SHAP and expression",
             "     .obs['validation'] boolean vector",
             "     .uns['catboost_info'] dict of parameters"]

    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
    
    if save_model:
        cb_model.save_model(save_model)

    if return_model:
        return shapdata, cb_model
    else:
        return shapdata