from __future__ import annotations
from tqdm import tqdm
from scanpy import AnnData
from .utils import *

import scanpy as sc
import numpy as np
import pandas as pd


def predict_c2v(
    adata: AnnData,
    clones: AnnData,
    use_rep: str = "clone2vec",
    predicted_rep: str = "clone2vec_predicted",
    add_label: str = "eval_set",
    eval_fraction: float = 0.2,
    clone_col: str = "clone",
    use_gpu: bool = False,
    use_raw: bool = False,
    layer: str | None = None,
    ct_col: str | None = None,
    pseudobulk: bool = True,
    limit_ct: str | None = None,
    use_ct: bool = True,
    num_trees: int = 10000,
    early_stopping_rounds: int = 100,
    filename: str = "model_c2v.cbm",
    use_model: str | None = None,
    verbose: bool = True,
    random_state: None | int = 4,
) -> AnnData:
    """
    This function creates a CatBoost regression model that predicts the clone2vec latent embedding positions 
    based on gene expression data from cells or clones (if pseudobulk is enabled). It also computes Shapley 
    values to assess the feature importance for the prediction of clonal behavior.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing gene expression data used to predict the clone2vec embedding.
    clones : AnnData
        Annotated data matrix containing the clone2vec embedding (output of `sl.tl.clone2vec`).
    use_rep : str, optional
        Slot in `clones.obsm` where clone2vec coordinates are stored, default is "clone2vec".
    predicted_rep : str, optional
        Slot in `clones.obsm` or `clones.uns` where predicted clone2vec embeddings will be stored, default is "clone2vec_predicted".
    add_label : str, optional
        Column in `clones.obs` indicating the training/validation label for model training, default is "eval_set".
    eval_fraction : float, optional
        Fraction of clones used for validation during training, default is 0.2.
    clone_col : str, optional
        Column in `adata.obs` containing clonal labels, default is "clone".
    use_gpu : bool, optional
        If True, uses GPU for CatBoost regression, default is False.
    use_raw : bool, optional
        If True, uses raw gene expression data from `adata.raw`, default is False.
    layer : str or None, optional
        Layer in `adata` containing gene expression data, default is None.
    ct_col : str or None, optional
        Column in `adata.obs` containing cell type information, default is None.
    pseudobulk : bool, optional
        If True, uses pseudobulk gene expression data for each clone instead of single-cell expression, default is True.
    limit_ct : str or None, optional
        Cell type value to limit the training set, default is None.
    use_ct : bool, optional
        If True, includes cell type as a categorical feature in the model, default is True.
    num_trees : int, optional
        Number of trees used for the gradient boosting model, default is 10000.
    early_stopping_rounds : int, optional
        Number of rounds with no improvement before stopping the training, default is 100.
    filename : str, optional
        Filepath to save the trained CatBoost model, default is "model_c2v.cbm".
    use_model : str or None, optional
        If provided, loads an existing model from the given file path, default is None.
    verbose : bool, optional
        If True, logs the training process, default is True.
    random_state : int or None, optional
        Seed for random number generators to ensure reproducibility, default is 4.

    Returns
    -------
    AnnData
        Returns an AnnData object with the following layers:
            - "shap": Shapley values representing feature importance.
            - "shap_c2v{i}": Shapley values for each clone2vec coordinate prediction.
    """
    from scipy.sparse import csr_matrix
    import random

    np.random.seed(random_state)
    random.seed(random_state)
    
    expr = prepare_expr(adata, clone_col, use_raw, layer, ct_col)
    expr = prepare_pseudobulk(expr, pseudobulk, ct_col, limit_ct, use_ct)
        
    if not use_model:
        if add_label in clones.obs.columns:
            train = list(clones.obs_names[clones.obs[add_label] == "train"])
            validation = list(clones.obs_names[clones.obs[add_label] == "validation"])
        else:
            # Train-validation split with Geosketch
            train, validation = split_test_eval(
                clones,
                use_rep=use_rep,
                fraction=eval_fraction,
                add_label=add_label,
                random_state=random_state,
            )

        train_expr = expr[expr.clone_column.isin(train)].copy()
        validation_expr = expr[expr.clone_column.isin(validation)].copy()

        train_c2v = clones[train_expr.clone_column].obsm[use_rep].copy()
        validation_c2v = clones[validation_expr.clone_column].obsm[use_rep].copy()

        del train_expr["clone_column"], validation_expr["clone_column"]

        # Running CatBoost
        model = run_CatBoostRegressor(
            train_expr, train_c2v, validation_expr,
            validation_c2v, use_gpu, num_trees,
            early_stopping_rounds, verbose, random_state,
        )
        model.save_model(filename)
    else:
        # Loading the model from the disk
        from catboost import CatBoostRegressor
        
        model = CatBoostRegressor()
        model.load_model(use_model)
    
    # Predicting the embedding
    from catboost import Pool
    
    expr_predict = expr.copy()
    expr_predict = expr_predict[expr_predict.clone_column.isin(clones.obs_names)]
    clone_column_pred = expr_predict["clone_column"].values.copy()
    del expr_predict["clone_column"]
    
    if use_ct and ct_col:
        expr_pool = Pool(data=expr_predict, cat_features=["celltype_column"])
    else:
        expr_pool = Pool(data=expr_predict)
    c2v_predicted = model.predict(expr_pool)
    
    if sum(~clones.obs_names.isin(clone_column_pred)) != 0:
        print("Some of the clones doesn't contain cell type of interest. Putting the result of prediction in the uns.")
        clones_obs_names = clones.obs_names[
            clones.obs_names.isin(clone_column_pred)
        ].copy()
        uns = True
    else:
        clones_obs_names = clones.obs_names.copy()
        uns = False
        
    prediction = (
        pd.DataFrame(c2v_predicted)
        .groupby(clone_column_pred)
        .mean()
        .loc[clones_obs_names]
        .values
    )
    
    if uns:
        clones.uns[predicted_rep] = prediction.copy()
        clones.uns[predicted_rep + "_names"] = list(clones_obs_names.copy())
    else:
        clones.obsm[predicted_rep] = prediction.copy()
        
    # Extracting shapley values
    raw_shap = model.get_feature_importance(expr_pool, type="ShapValues")
    
    if use_ct:
        ct_col_values = expr_predict.celltype_column.copy()
        expr_predict["celltype_column"] = expr_predict["celltype_column"].astype("category").values.codes

    shapdata = sc.AnnData(
        X=csr_matrix(expr_predict.values),
        obs=pd.DataFrame(index=expr_predict.index),
        var=pd.DataFrame(index=expr_predict.columns),
        layers={"shap": csr_matrix(np.linalg.norm(raw_shap, axis=1)[:, :-1])},
    )

    if use_ct:
        shapdata.obs[ct_col] = ct_col_values
    
    for i in range(raw_shap.shape[1]):
        shapdata.layers[f"shap_c2v{i}"] = csr_matrix(raw_shap[:, i, :-1])
    
    return shapdata


def predict_ct(
    adata: AnnData,
    clones: AnnData,
    use_rep: str = "clone2vec",
    predicted_rep: str = "ct_predicted",
    add_label: str = "eval_set",
    eval_fraction: float = 0.2,
    clone_col: str = "clone",
    use_gpu: bool = False,
    use_raw: bool = False,
    layer: str | None = None,
    ct_col: str | None = None,
    pseudobulk: bool = True,
    limit_ct: str | None = None,
    use_ct: bool = True,
    num_trees: int = 10000,
    early_stopping_rounds: int = 100,
    filename: str = "model_ct.cbm",
    use_model: str | None = None,
    verbose: bool = True,
    ct_layer: str | None = None,
    random_state: None | int = 4,
) -> AnnData:
    """
    This function creates a CatBoost classifier model that predicts the cell type composition of each clone 
    based on gene expression data from cells or clones (if pseudobulk is enabled). It computes Shapley values 
    to identify genes important for predicting cell type proportions in clones.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing gene expression data used to predict the cell type composition.
    clones : AnnData
        Annotated data matrix containing the clone2vec embedding (output of `sl.tl.clone2vec`).
    use_rep : str, optional
        Slot in `clones.obsm` where clone2vec coordinates are stored, default is "clone2vec".
    predicted_rep : str, optional
        Slot in `clones.obsm` or `clones.uns` where predicted cell type composition will be stored, default is "ct_predicted".
    add_label : str, optional
        Column in `clones.obs` indicating the training/validation label for model training, default is "eval_set".
    eval_fraction : float, optional
        Fraction of clones used for validation during training, default is 0.2.
    clone_col : str, optional
        Column in `adata.obs` containing clonal labels, default is "clone".
    use_gpu : bool, optional
        If True, uses GPU for CatBoost classifier, default is False.
    use_raw : bool, optional
        If True, uses raw gene expression data from `adata.raw`, default is False.
    layer : str or None, optional
        Layer in `adata` containing gene expression data, default is None.
    ct_col : str or None, optional
        Column in `adata.obs` containing cell type information, default is None.
    pseudobulk : bool, optional
        If True, uses pseudobulk gene expression data for each clone instead of single-cell expression, default is True.
    limit_ct : str or None, optional
        Cell type value to limit the training set, default is None.
    use_ct : bool, optional
        If True, includes cell type as a categorical feature in the model, default is True.
    num_trees : int, optional
        Number of trees used for the gradient boosting model, default is 10000.
    early_stopping_rounds : int, optional
        Number of rounds with no improvement before stopping the training, default is 100.
    filename : str, optional
        Filepath to save the trained CatBoost model, default is "model_ct.cbm".
    use_model : str or None, optional
        If provided, loads an existing model from the given file path, default is None.
    verbose : bool, optional
        If True, logs the training process, default is True.
    ct_layer : str or None, optional
        Name of the layer in `clones` containing cell type proportions, default is None (uses `clones.X`).
    random_state : int or None, optional
        Seed for random number generators to ensure reproducibility, default is 4.

    Returns
    -------
    AnnData
        Returns an AnnData object with the following layers:
            - "shap": Shapley values representing feature importance.
            - "shap_{ct}": Shapley values for each cell type proportion prediction.
    """
    from scipy.sparse import csr_matrix
    import random

    np.random.seed(random_state)
    random.seed(random_state)
    
    expr = prepare_expr(adata, clone_col, use_raw, layer, ct_col)
    expr = prepare_pseudobulk(expr, pseudobulk, ct_col, limit_ct, use_ct)
        
    if not use_model:
        if add_label in clones.obs.columns:
            train = list(clones.obs_names[clones.obs[add_label] == "train"])
            validation = list(clones.obs_names[clones.obs[add_label] == "validation"])
        else:
            # Train-validation split with Geosketch
            train, validation = split_test_eval(
                clones,
                use_rep=use_rep,
                fraction=eval_fraction,
                add_label=add_label,
                random_state=random_state,
            )

        train_expr = expr[expr.clone_column.isin(train)].copy()
        validation_expr = expr[expr.clone_column.isin(validation)].copy()
        
        if ct_layer:
            train_ct = clones[train_expr.clone_column].layers[ct_layer].copy()
            validation_ct = clones[validation_expr.clone_column].layers[ct_layer].copy()
        else:
            train_ct = clones[train_expr.clone_column].X.copy()
            validation_ct = clones[validation_expr.clone_column].X.copy()

        del train_expr["clone_column"], validation_expr["clone_column"]

        # Running CatBoost
        model = run_CatBoostClassifier(
            train_expr, train_ct, validation_expr,
            validation_ct, use_gpu, num_trees,
            early_stopping_rounds, verbose, random_state,
        )
        model.save_model(filename)
    else:
        # Loading the model from the disk
        from catboost import CatBoostClassifier
        
        model = CatBoostClassifier()
        model.load_model(use_model)
    
    # Predicting the embedding
    from catboost import Pool
    
    expr_predict = expr.copy()
    expr_predict = expr_predict[expr_predict.clone_column.isin(clones.obs_names)]
    clone_column_pred = expr_predict["clone_column"].values.copy()
    del expr_predict["clone_column"]
    
    if use_ct and ct_col:
        expr_pool = Pool(data=expr_predict, cat_features=["celltype_column"])
    else:
        expr_pool = Pool(data=expr_predict)
    ct_predicted = model.predict_proba(expr_pool)
    
    if sum(~clones.obs_names.isin(clone_column_pred)) != 0:
        print("Some of the clones doesn't contain cell type of interest. Putting the result of prediction in the uns.")
        clones_obs_names = clones.obs_names[
            clones.obs_names.isin(clone_column_pred)
        ].copy()
        uns = True
    else:
        clones_obs_names = clones.obs_names.copy()
        uns = False
        
    prediction = (
        pd.DataFrame(ct_predicted)
        .groupby(clone_column_pred)
        .mean()
        .loc[clones_obs_names]
        .values
    )
    
    if uns:
        clones.uns[predicted_rep] = prediction.copy()
        clones.uns[predicted_rep + "_names"] = list(clones_obs_names.copy())
    else:
        clones.obsm[predicted_rep] = prediction.copy()
        
    # Extracting shapley values
    raw_shap = model.get_feature_importance(expr_pool, type="ShapValues")
    
    if use_ct:
        ct_col_values = expr_predict.celltype_column.copy()
        expr_predict["celltype_column"] = expr_predict["celltype_column"].astype("category").values.codes

    shapdata = sc.AnnData(
        X=csr_matrix(expr_predict.values),
        obs=pd.DataFrame(index=expr_predict.index),
        var=pd.DataFrame(index=expr_predict.columns),
        layers={"shap": csr_matrix(np.linalg.norm(raw_shap, axis=1)[:, :-1])},
    )

    if use_ct:
        shapdata.obs[ct_col] = ct_col_values
    
    for i, ct in enumerate(clones.var_names):
        shapdata.layers[f"shap_{ct}"] = csr_matrix(raw_shap[:, i, :-1])
    
    return shapdata


def clonal_nn(
    adata: AnnData,
    obs_name: str,
    k: int = 15,
    non_clonal_str: str = "NA",
    use_rep: str = "X_pca",
    tqdm_bar: bool = False,
    min_size: int = 10,
    random_state: None | int = 4,
    copy: bool = False,
    obsm_name: str = "bag-of-clones",
    **kwargs,
) -> None | AnnData:
    """
    Find the top k clonally labeled cells for each cell using k-nearest neighbors.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing cell-level information.
    obs_name : str
        Column in `adata.obs` containing clonal information.
    k : int, optional
        Number of nearest neighbors to retrieve for each cell, by default 15.
    non_clonal_str : str, optional
        Value indicating the absence of clonal information, by default "NA".
    use_rep : str, optional
        Representation in `adata.obsm` to build the kNN graph, by default "X_pca".
    tqdm_bar : bool, optional
        Whether to show a progress bar, by default False.
    min_size : int, optional
        Minimum clone size to be considered in the analysis, by default 10.
    random_state : None | int, optional
        Random state for reproducibility, by default 4.
    copy : bool, optional
        Whether to return a copy of `adata`, by default False.
    obsm_name : str, optional
        Name of the new graph slot in `adata.obsm`, by default "bag-of-clones".

    Returns
    -------
    None or AnnData
        Updates `adata` in place unless `copy=True`, in which case a new AnnData object is returned.
    """
    from scipy.sparse import csr_matrix, coo_matrix
    import pynndescent
    
    adata_to_update = adata.copy() if copy else adata

    # 1. Preprocessing (vectorized)
    clonal_obs = adata.obs[obs_name].copy()
    clones_counts = clonal_obs.value_counts()
    small_clones = clones_counts[clones_counts < min_size].index

    clonal_obs[clonal_obs.isin(small_clones)] = non_clonal_str
    clonal_obs = clonal_obs.astype("category")

    # 2. Prepare data for kNN
    valid_clones = clonal_obs.cat.categories[clonal_obs.cat.categories != non_clonal_str]
    n_clones = len(valid_clones)
    clone_to_int = pd.Series(np.arange(n_clones), index=valid_clones)

    is_clonal = (clonal_obs != non_clonal_str).to_numpy()
    train_data = adata.obsm[use_rep][is_clonal]

    if train_data.shape[0] == 0:
        adata_to_update.obsm[obsm_name] = csr_matrix((adata.shape[0], n_clones))
        adata_to_update.uns[obsm_name + "_names"] = valid_clones.to_list()
        return adata_to_update if copy else None

    # 3. Build and query the kNN index
    index = pynndescent.NNDescent(train_data, random_state=random_state, **kwargs)
    index.prepare()
    
    test_data = adata.obsm[use_rep]
    neighbors_indices, _ = index.query(test_data, k=k)

    # 4. Vectorized aggregation to build the sparse matrix
    train_labels = clonal_obs[is_clonal]
    train_labels_encoded = clone_to_int[train_labels].to_numpy()

    neighbor_labels_encoded = train_labels_encoded[neighbors_indices]

    n_test = test_data.shape[0]
    row_ind = np.repeat(np.arange(n_test), k)
    col_ind = neighbor_labels_encoded.flatten()
    data = np.ones(n_test * k, dtype=np.float32)

    bag_of_clones_matrix = coo_matrix(
        (data, (row_ind, col_ind)), shape=(n_test, n_clones)
    ).tocsr()

    # 5. Update the AnnData object
    adata_to_update.obsm[obsm_name] = bag_of_clones_matrix
    adata_to_update.uns[f"{obsm_name}_names"] = valid_clones.to_list()

    if copy:
        return adata_to_update


def clone2vec(
    adata: AnnData,
    obs_name: str = "clone",
    z_dim: int = 10,
    n_epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
    fill_ct: None | str = None,
    tqdm_bar: bool = True,
    obsm_name: str = "bag-of-clones",
    obsm_key: str = "clone2vec",
    uns_key: str = "clone2vec_mean_loss",
    random_state: None | int = 4,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
) -> AnnData:
    """
    Learn a clonal embedding using a SkipGram model and return the resulting clone embeddings.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level containing clonal annotations.
    obs_name : str
        Column name in `adata.obs` containing clonal information.
    z_dim : int, optional
        The dimensionality of the embedding, by default 10.
    n_epochs : int, optional
        Number of epochs for training, by default 100.
    batch_size : int, optional
        Batch size for training, by default 64.
    device : str, optional
        The device for training the neural network, by default "cpu".
    fill_ct : None | str, optional
        Column in `adata.obs` with cell type labels to fill missing clones, by default None.
    tqdm_bar : bool, optional
        Whether to display a progress bar during training, by default True.
    obsm_name : str, optional
        Slot in `adata.obsm` containing the clonal kNN graph, by default "bag-of-clones".
    obsm_key : str, optional
        Slot in `clones.obsm` to store the clone embeddings, by default "clone2vec".
    uns_key : str, optional
        Key in `clones.uns` to store the mean loss values per epoch, by default "clone2vec_mean_loss".
    random_state : None | int, optional
        Random state for reproducibility, by default 4.

    Returns
    -------
    AnnData
        Annotated data matrix at the clone level, containing clone embeddings and other statistics.
    """
    from .utils import SkipGram
    from tqdm import tqdm

    import random
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    clone2idx = dict(zip(
        adata.uns[f"{obsm_name}_names"],
        range(len(adata.uns[f"{obsm_name}_names"])),
    ))
    idx2clone = dict(zip(clone2idx.values(), clone2idx.keys()))
    indices = np.array(range(len(adata.uns[f"{obsm_name}_names"])))

    k_estimated = list(set(adata.obsm[obsm_name].toarray().sum(axis=1)))
    if len(k_estimated) > 1:
        raise Exception("adata.obsm[obsm_name] should contain the result of kNN graph construction with fixed k.")
    k_estimated = int(k_estimated[0])

    adata_only_clones = adata[adata.obs[obs_name].isin(adata.uns[f"{obsm_name}_names"])]
    pairs = []
    for X, clone in zip(
        adata_only_clones.obsm[obsm_name].toarray(),
        adata_only_clones.obs[obs_name],
    ):
        cl_nn = X > 0
        pairs += list(zip([clone2idx[clone]] * k_estimated, indices[cl_nn]))
    pairs = np.array(pairs)

    train_loader = torch.utils.data.DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SkipGram(
        z_dim=z_dim,
        vocab_size=len(indices),
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.NLLLoss()

    epochs_mean_loss = []
    best_loss = np.inf
    patience_counter = 0

    for epoch in (tqdm(range(n_epochs)) if tqdm_bar else range(n_epochs)):
        losses = []
        for batch_idx, data in enumerate(train_loader):
            model.train()
            
            x_batch = data[:, 0].to(device)
            y_batch = data[:, 1].to(device)
            
            log_ps = model(x_batch)
            loss = criterion(log_ps, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

        current_loss = np.mean(losses)
        epochs_mean_loss.append(current_loss)
        
        if current_loss < best_loss - early_stopping_min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            if tqdm_bar:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    clone2vec = model.embedding.weight.data.cpu().numpy()
    if not (fill_ct is None) and not (fill_ct in adata.obs.columns):
        print(f"{fill_ct} isn't in the `adata.obs.columns`. Keeping `clones.X` empty.")
        fill_ct = None

    if not (fill_ct is None):
        cell_counts = adata_only_clones.obs.groupby(
            [fill_ct, obs_name]
        ).size().unstack(fill_value=0)[adata_only_clones.uns[f"{obsm_name}_names"]]
        
        var_names = list(cell_counts.index)
        obs_names = list(cell_counts.columns)
        
        cell_counts = cell_counts.values
        freqs = cell_counts / cell_counts.sum(axis=0)
    else:
        var_names = ["None"]
        obs_names = adata.uns[f"{obsm_name}_names"]

        cell_counts = np.matrix([0] * len(adata.uns[f"{obsm_name}_names"]))
        freqs = np.matrix([0] * len(adata.uns[f"{obsm_name}_names"]))

    clones = sc.AnnData(
        X=cell_counts.T,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
        layers={
            "frequencies": freqs.T,
            "counts": cell_counts.T,
        },
        obsm={
            obsm_key: clone2vec.copy(),
        },
        uns={
            uns_key: epochs_mean_loss,
        },
    )
    if fill_ct:
        clones.obs["n_cells"] = clones.X.sum(axis=1)
    else:
        clones.obs["n_cells"] = adata.obs[
            obs_name
        ].value_counts()[clones.obs_names].values

    return clones

def transfer_clonal_annotation(
    adata: AnnData,
    clones: AnnData,
    adata_clone_name: str,
    adata_obs_name: str,
    clones_obs_name: str,
    fill_values: str = "NA",
) -> None:
    """
    Transfer clonal labels from a `clones` AnnData object to a `adata` AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level.
    clones : AnnData
        Annotated data matrix at the clone level.
    adata_clone_name : str
        Column in `adata.obs` with the original clonal information.
    adata_obs_name : str
        Name of the new column in `adata.obs` to store transferred clonal labels.
    clones_obs_name : str
        Column in `clones.obs` containing clonal labels.
    fill_values : str, optional
        Value to assign to cells with no matching clone in the `clones` object, by default "NA".

    Returns
    -------
    None
        The `adata` object is modified in place with the new clonal labels.
    """
    
    clone_mapping = dict(clones.obs[clones_obs_name])
    adata.obs[adata_obs_name] = [
        clone_mapping[clone] if clone in clone_mapping else fill_values
        for clone in adata.obs[adata_clone_name]
    ]
    adata.obs[adata_obs_name] = adata.obs[adata_obs_name].astype("category")

def summarize_expression(
    adata: AnnData,
    clones: AnnData,
    obs_name: str = "clone",
    strategy: str = "average",
    layer: str | None = None,
    use_raw: bool | None = None,
    subset_obs: str | None = None,
    target_value: str | None = None,
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
        Column name in `adata.obs` containing clonal information, by default "clone".
    strategy : str, optional
        Strategy for aggregating gene expression ("sum" or "average"), by default "average".
    layer : str | None, optional
        If specified, summarizes expression from `adata.layers[layer]`, by default None.
    use_raw : bool | None, optional
        If specified, uses `adata.raw.X` for summarization, by default None.
    subset_obs : str | None, optional
        Column name in `adata.obs` to subset expression by (e.g., cell type), by default None.
    target_value : str | None, optional
        Value in `adata.obs[subset_obs]` for further subsetting, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix at the clone level with summarized gene expression.
    """
    from scipy.sparse import csr_matrix
    
    if use_raw and not (layer is None):
        raise Exception(f"Can't use both `adata.raw` and `adata.layers['{layer}']`")
    if strategy not in ("average", "sum"):
        raise Exception(f"Only average and sum methods of expression aggregation are supported")
    
    if subset_obs is None:
        mask = np.array([True] * len(adata))
    else:
        mask = np.array(adata.obs[subset_obs] == target_value)

    if not (layer is None):
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw or not (adata.raw is None):
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    clones_expr = {}
    for clone in clones.obs_names:
        if strategy == "average":
            try:
                clones_expr[clone] = X[(adata.obs[obs_name].values == clone) & mask].mean(axis=0).A[0]
            except ZeroDivisionError:
                clones_expr[clone] = np.zeros(len(var_names))
        else:
            clones_expr[clone] = X[(adata.obs[obs_name] == clone) & mask].sum(axis=0).A[0]
    
    clones_expr = sc.AnnData(pd.DataFrame(clones_expr, index=var_names).T)
    
    clones_expr.X = csr_matrix(clones_expr.X)
    clones_expr.uns = clones.uns.copy()
    clones_expr.obs = clones.obs.copy()
    clones_expr.obsm = clones.obsm.copy()
    clones_expr.obsp = clones.obsp.copy()
    
    return clones_expr

def refill_ct(
    adata: AnnData,
    obs_name: str,
    ct_col: str,
    clones: AnnData,
) -> AnnData:
    """
    Create a new annotated data matrix at the clone level with updated cell type proportions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at the cell level.
    obs_name : str
        Column in `adata.obs` containing clonal information.
    ct_col : str
        Column in `adata.obs` containing updated cell type labels.
    clones : AnnData
        Annotated data matrix at the clone level.

    Returns
    -------
    AnnData
        New annotated data matrix at the clone level with updated cell type proportions.
    """
    clones_new = sc.AnnData(adata.obs.groupby(
        [ct_col, obs_name]
    ).size().unstack().T.loc[clones.obs_names].fillna(0))
    
    clones_new.uns = clones.uns.copy()
    clones_new.obs = clones.obs.copy()
    clones_new.obsm = clones.obsm.copy()
    clones_new.obsp = clones.obsp.copy()
    
    clones_new.layers["counts"] = clones_new.X.copy()
    clones_new.layers["frequencies"] = (clones_new.X.T / clones_new.X.sum(axis=1)).T
    
    return clones_new
