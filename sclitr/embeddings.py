from __future__ import annotations

import scanpy as sc
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import pandas as pd

import torch
import random
import time
import sys

from typing import Literal
from tqdm import tqdm

logg = sc.logging

def _create_pairs(X: np.ndarray | sp.spmatrix) -> list[tuple[int, int]]:
    """
    Creates a list of pairs (row, col) from a sparse or dense count matrix X.

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Input matrix.

    Returns
    -------
    list[tuple[int, int]]
        List of pairs (i, j) where X[i, j] is non-zero.
    """

    if sp.issparse(X):
        coo = X.tocoo()
        data = coo.data
    else:
        coo = np.asarray(X)
        rows, cols = np.nonzero(coo)
        data = coo[rows, cols]
        coo = type("obj", (object,), {"row": rows, "col": cols, "data": data})

    rounded = np.round(data)
    if not np.allclose(data, rounded):
        raise ValueError("X contains non-integer counts.")
    
    counts = rounded.astype(int)
    if np.any(counts < 0):
        raise ValueError("X contains negative counts.")

    return [
        (int(r), int(c))
        for r, c, cnt in zip(coo.row, coo.col, counts)
        for _ in range(cnt)
    ]

class SkipGram(nn.Module):
    """
    A SkipGram model that dynamically defines its embedding size during fitting.
    """

    def __init__(self, z_dim: int, device: str | None = None):
        """
        Initialize the SkipGram model structure.

        Parameters
        ----------
        z_dim : int
            Dimensionality of the embedding space.
        device : str or None, optional
            Preferred compute device. If None, it will be auto-detected.
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.embedding = None
        self.output = None
        self.is_fitted_ = False

    def _set_device(self, device: str | None) -> str:
        """
        Set the compute device for the model.

        Parameters
        ----------
        device : str or None, optional
            Device to use for training. If None, it will be auto-detected. Default is None.

        Returns
        -------
        str
            The selected compute device.
        """
        if device is None:
            if torch.cuda.is_available():
                final_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                final_device = "mps"
            else:
                final_device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logg.warning("CUDA not available. Falling back to CPU.")
            final_device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logg.warning("MPS not available. Falling back to CPU.")
            final_device = "cpu"
        else:
            final_device = device
        
        self.device = final_device
        self.to(final_device)
        return final_device

    def forward(self, input_word: torch.Tensor) -> torch.Tensor:
        emb_input = self.embedding(input_word)
        context_scores = self.output(emb_input)
        return self.log_softmax(context_scores)

    def _initialize_or_resize_layers(self, input_vocab_size: int, output_vocab_size: int):
        """
        Initialize or resize the embedding and output layers.

        Parameters
        ----------
        input_vocab_size : int
            Size of the input vocabulary.
        output_vocab_size : int
            Size of the output vocabulary.
        """
        if not self.is_fitted_:
            self.embedding = nn.Embedding(input_vocab_size, self.z_dim)
            self.output = nn.Linear(self.z_dim, output_vocab_size)
            self.is_fitted_ = True
            return

        current_input_size = self.embedding.num_embeddings
        current_output_size = self.output.out_features

        if current_input_size != input_vocab_size:
            new_embedding = nn.Embedding(input_vocab_size, self.z_dim)
            copy_rows = min(current_input_size, input_vocab_size)
            with torch.no_grad():
                new_embedding.weight.data[:copy_rows] = self.embedding.weight.data[:copy_rows]
            self.embedding = new_embedding

        if current_output_size != output_vocab_size:
            new_output = nn.Linear(self.z_dim, output_vocab_size)
            copy_rows = min(current_output_size, output_vocab_size)
            with torch.no_grad():
                new_output.weight.data[:copy_rows] = self.output.weight.data[:copy_rows]
                new_output.bias.data[:copy_rows] = self.output.bias.data[:copy_rows]
            self.output = new_output
            
    def fit(
        self,
        X: np.ndarray | sp.spmatrix,
        random_state: int | None = 42,
        batch_size: int = 128,
        device: str | None = None,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-4,
        progress_bar: bool = True,
        max_iter: int = 500,
        return_model: bool = False,
        init: Literal["svd", "random", "custom"] = "random",
        init_embeddings: np.ndarray | None = None,
        init_output_weight: np.ndarray | None = None,
    ) -> np.ndarray | "SkipGram":
        """
        Fitting the SkipGram model to the input count matrix.

        Parameters
        ----------
        X : np.ndarray or sp.spmatrix
            Input count matrix.
        random_state : int or None, optional
            Random seed for reproducibility. Default is 42.
        batch_size : int, optional
            Batch size for training. Default is 128.
        device : str or None, optional
            Device to use for training. Default is None.
        learning_rate : float, optional
            Learning rate for Adam optimizer. Default is 0.001.
        early_stopping_patience : int, optional
            Patience for early stopping. Default is 5.
        early_stopping_min_delta : float, optional
            Minimum delta for early stopping. Default is 1e-4.
        progress_bar : bool, optional
            Whether to show a progress bar. Default is True.
        max_iter : int, optional
            Maximum number of iterations. Default is 500.
        return_model : bool, optional
            Whether to return the fitted model. Default is False.
        init : Literal["svd", "random", "custom"], optional
            Initialization method for embeddings. Default is "random".
        init_embeddings : np.ndarray or None, optional
            Custom initial embeddings. Default is None.
        init_output_weight : np.ndarray or None, optional
            Custom initial output weight. Default is None.

        Returns
        -------
        np.ndarray or SkipGram
            If return_model is True, returns the fitted model. Otherwise, returns the embeddings.
        """
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)

        input_vocab_size, output_vocab_size = X.shape
        self._initialize_or_resize_layers(input_vocab_size, output_vocab_size)
        self.learning_rate = learning_rate
        
        device = self._set_device(device)
        
        # Normalize init parameter
        if init not in {"svd", "random", "custom"}:
            logg.warning("init must be one of {'random', 'custom', 'svd'}. Falling back to 'svd'.")
            init = "svd"

        # If custom arrays are provided but init is not 'custom', accept them
        if (init_embeddings is not None or init_output_weight is not None) and init != "custom":
            init = "custom"

        if init == "svd":
            U0, V0 = self._compute_svd_init(X, self.z_dim)
            with torch.no_grad():
                self.embedding.weight.data = torch.from_numpy(U0.astype(np.float32)).to(device)
                self.output.weight.data = torch.from_numpy(V0.astype(np.float32)).to(device)
                self.output.bias.data.zero_()
        elif init == "custom":
            if init_embeddings is not None:
                if init_embeddings.shape != (input_vocab_size, self.z_dim):
                    logg.warning(
                        f"init_embeddings shape {init_embeddings.shape} must be (vocab_size={input_vocab_size}, z_dim={self.z_dim}). Falling back to SVD."
                    )
                    U0, V0 = self._compute_svd_init(X, self.z_dim)
                    with torch.no_grad():
                        self.embedding.weight.data = torch.from_numpy(U0.astype(np.float32)).to(device)
                        self.output.weight.data = torch.from_numpy(V0.astype(np.float32)).to(device)
                        self.output.bias.data.zero_()
                else:
                    with torch.no_grad():
                        self.embedding.weight.data = torch.from_numpy(init_embeddings.astype(np.float32)).to(device)
            else:
                U0, V0 = self._compute_svd_init(X, self.z_dim)
                with torch.no_grad():
                    self.embedding.weight.data = torch.from_numpy(U0.astype(np.float32)).to(device)
                    self.output.weight.data = torch.from_numpy(V0.astype(np.float32)).to(device)
                    self.output.bias.data.zero_()

            if init_output_weight is not None:
                if init_output_weight.shape != (output_vocab_size, self.z_dim):
                    logg.warning(
                        f"init_output_weight shape {init_output_weight.shape} must be (vocab_size={output_vocab_size}, z_dim={self.z_dim}). Keeping current initialization."
                    )
                else:
                    with torch.no_grad():
                        self.output.weight.data = torch.from_numpy(init_output_weight.astype(np.float32)).to(device)

            with torch.no_grad():
                if self.output.bias is not None:
                    self.output.bias.data.zero_()
        else:
            with torch.no_grad():
                self.embedding.weight.data.uniform_(-1.0, 1.0)
                self.output.weight.data.uniform_(-1.0, 1.0)
                self.output.bias.data.zero_()

        pairs = _create_pairs(X)
        train_loader = torch.utils.data.DataLoader(
            pairs,
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        if sc.settings.verbosity.value >= 2:
            prefix = "    "
            progress_bar = True
        else:
            prefix = ""

        if progress_bar:
            cm = tqdm(
                range(max_iter),
                desc=prefix + "SG epochs",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )
        else:
            from contextlib import nullcontext
            cm = nullcontext(range(max_iter))

        patience_counter = 0
        prev_loglik = None
        self.loglik_history_ = []

        with cm as iterator:
            for epoch in iterator:
                losses = []
                for x_batch, y_batch in train_loader:
                    self.train()
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    log_ps = self(x_batch)
                    loss = criterion(log_ps, y_batch)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                
                current_loss = np.mean(losses) if losses else float("inf")
                self.loglik_history_.append(-current_loss)
                loglik = -current_loss
                
                if prev_loglik is not None:
                    rel_delta = abs((loglik - prev_loglik) / (abs(prev_loglik) + 1e-9))
                    if rel_delta < early_stopping_min_delta:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    if progress_bar:
                        iterator.set_postfix({"loss": f"{current_loss:.4f}", "Δ": f"{rel_delta:.2e}"})
                else:
                    if progress_bar:
                        iterator.set_postfix({"loss": f"{current_loss:.4f}"})
                
                if patience_counter >= early_stopping_patience:
                    break
                prev_loglik = loglik

        if progress_bar:
            if epoch == max_iter - 1:
                tqdm.write(prefix + f"reached max_iter {max_iter}")
            else:
                tqdm.write(prefix + f"early stopping at epoch {epoch + 1}")

        self.U = self.embedding.weight.data.cpu().numpy()
        self.V = self.output.weight.data.cpu().numpy()
        self.loglik_history_ = np.array(self.loglik_history_)

        return self if return_model else self.U

    def _compute_svd_init(
        self,
        X: np.ndarray | sp.spmatrix,
        z_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute SVD-based initialization for embeddings and output weights.

        Parameters
        ----------
        X : np.ndarray or sp.spmatrix
            Input count matrix.
        z_dim : int
            Target dimensionality.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of `(init_embeddings, init_output_weight)` arrays.
        """
        from scipy.sparse.linalg import svds
        from scipy.linalg import svd

        if sp.issparse(X):
            X = X.tocsr().astype(np.float64)
            X = X.copy()
            X.data = np.log1p(X.data)
            U, S, Vt = svds(X, k=z_dim)
            order = np.argsort(S)[::-1]
            S = S[order]
            U = U[:, order]
            Vt = Vt[order, :]
        else:
            X = np.asarray(X, dtype=np.float64)
            U, S, Vt = svd(np.log1p(X), full_matrices=False)
            U = U[:, :z_dim]
            S = S[:z_dim]
            Vt = Vt[:z_dim, :]

        S_sqrt = np.sqrt(S)
        init_embeddings = U * S_sqrt[np.newaxis, :]
        init_output_weight = (Vt.T * S_sqrt[np.newaxis, :])
        return init_embeddings.astype(np.float32), init_output_weight.astype(np.float32)

    def project(
        self,
        X_new: np.ndarray | sp.spmatrix,
        random_state: int | None = 42,
        batch_size: int = 128,
        device: str | None = None,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-4,
        progress_bar: bool = True,
        max_iter: int = 500,
        return_model: bool = False,
    ) -> np.ndarray | "SkipGram":
        """
        Project new data onto the fitted SkipGram model.

        Parameters
        ----------
        X_new : np.ndarray or sp.spmatrix
            New count matrix to project.
        random_state : int | None, optional
            Random seed for reproducibility. Default is 42.
        batch_size : int, optional
            Batch size for training. Default is 128.
        device : str | None, optional
            Device to use for training. If None, it will be auto-detected. Default is None.
        learning_rate : float, optional
            Learning rate for Adam optimizer. Default is 0.001.
        early_stopping_patience : int, optional
            Patience for early stopping. Default is 5.
        early_stopping_min_delta : float, optional
            Minimum delta for early stopping. Default is 1e-4.
        progress_bar : bool, optional
            Whether to show a progress bar. Default is True.
        max_iter : int, optional
            Maximum number of iterations. Default is 500.
        return_model : bool, optional
            Whether to return the fitted model. Default is False.

        Returns
        -------
        np.ndarray or SkipGram
            Projected embeddings or the fitted model.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before projecting new data. Call fit() first.")

        start = logg.info("projecting embeddings with SkipGram")
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
        
        n_new, n_old_vocab = X_new.shape
        if n_old_vocab != self.output.out_features:
            raise ValueError(f"X_new must have {self.output.out_features} columns, but got {n_old_vocab}.")

        device = self._set_device(device)
        
        self._initialize_or_resize_layers(n_new, self.output.out_features)
    
        with torch.no_grad():
            self.embedding.weight.data = torch.randn(n_new, self.z_dim, device=device) * 1e-4

        self.output.requires_grad_(False)
        
        pairs = _create_pairs(X_new)
        train_loader = torch.utils.data.DataLoader(pairs, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam([self.embedding.weight], lr=learning_rate)
        criterion = nn.NLLLoss()

        if sc.settings.verbosity.value >= 2:
            prefix = "    "
            progress_bar = True
        else:
            prefix = ""
        
        if progress_bar:
            cm = tqdm(
                range(max_iter),
                desc=prefix + "SG project epochs",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )
        else:
            from contextlib import nullcontext
            cm = nullcontext(range(max_iter))
        patience_counter = 0
        prev_loglik = None
        proj_loglik_history = []

        with cm as iterator:
            for epoch in iterator:
                losses = []
                for x_batch, y_batch in train_loader:
                    self.train()
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    log_ps = self(x_batch)
                    loss = criterion(log_ps, y_batch)
                    loss.backward()
                    
                    optimizer.step()
                    losses.append(loss.item())

                current_loss = np.mean(losses) if losses else float("inf")
                loglik = -current_loss
                proj_loglik_history.append(loglik)

                if prev_loglik is not None:
                    rel_delta = abs((loglik - prev_loglik) / (abs(prev_loglik) + 1e-9))
                    if rel_delta < early_stopping_min_delta:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    if progress_bar:
                        iterator.set_postfix({"loss": f"{current_loss:.4f}", "Δ": f"{rel_delta:.2e}"})
                else:
                    if progress_bar:
                        iterator.set_postfix({"loss": f"{current_loss:.4f}"})

                if patience_counter >= early_stopping_patience:
                    break
                prev_loglik = loglik

        if progress_bar:
            if epoch == max_iter - 1:
                tqdm.write(prefix + f"reached max_iter {max_iter}")
            else:
                tqdm.write(prefix + f"early stopping at epoch {epoch + 1}")

        self.output.requires_grad_(True)
        
        self.U = self.embedding.weight.data.cpu().numpy()
        self.project_loglik_history_ = np.array(proj_loglik_history)
        return self if return_model else self.U

def clone2vec(
    clones: sc.AnnData,
    z_dim: int = 10,
    obsp_key: str = "gex_adjacency",
    mask_key: str | None = None,
    max_iter: int = 500,
    learning_rate: float | None = 0.001,
    device: str | None = None,
    progress_bar: bool = True,
    obsm_key: str = "clone2vec",
    uns_key: str = "clone2vec",
    random_state: None | int = 4,
    init: Literal["svd", "random", "custom"] = "svd",
    init_obsm: str | None = None,
    batch_size: int = 128,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
) -> None:
    """
    Learn a clonal embedding using Skip-Gram and store the embeddings.

    Parameters
    ----------
    clones : AnnData
        The clone-level AnnData object, typically from `create_clone_adata`.
    z_dim : int, optional
        Dimensionality of the clonal embedding, by default 10.
    obsp_key : str, optional
        Key in `clones.obsp` for the graph to use, by default "gex_adjacency".
    max_iter : int, optional
        Maximum number of iterations for optimization, by default 500.
    learning_rate : float | None, optional
        Learning rate for optimization, by default 0.001.
    device : str | None, optional
        Device to use for computation, by default None.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    obsm_key : str, optional
        Key in `clones.obsm` to store the embeddings, by default "c2v".
    uns_key: str, optional
        Key in `clones.uns` to store the results, by default "c2v".
    random_state : None | int, optional
        Random state for reproducibility, by default 4.
    init : Literal["svd", "random", "custom"], optional
        Initialization method, if "svd", uses SVD on the log1p-transformed counts, by default "svd".
    init_obsm : str | None, optional
        Key in `clones.obsm` for custom initialization, by default None.
    batch_size : int, optional
        Batch size for optimization, by default 128.
    early_stopping_patience : int, optional
        Number of iterations with no improvement to wait before stopping, by default 5.
    early_stopping_min_delta : float, optional
        Minimum change in loss to qualify as an improvement, by default 1e-4.
    """
    start = logg.info("fitting clone2vec embeddings")
    if obsp_key not in clones.obsp:
        raise KeyError(f"Graph '{obsp_key}' not found in clones.obsp.")

    start_time = time.time()

    init_embeddings = None
    init_output_weight = None
    n_clones = clones.n_obs
    if init not in {"random", "custom", "svd"}:
        logg.warning("init must be one of {'random', 'custom', 'svd'}. Falling back to SVD.")
        init = "svd"

    if init == "custom":
        if init_obsm is None:
            logg.warning("init_obsm must be provided when init='custom'. Falling back to SVD.")
            init = "svd"
        elif init_obsm not in clones.obsm:
            logg.warning(f"'{init_obsm}' not found in clones.obsm. Falling back to SVD.")
            init = "svd"
        else:
            init_embeddings = np.asarray(clones.obsm[init_obsm])
            if init_embeddings.shape[0] != n_clones:
                logg.warning(
                    f"init_obsm embeddings have {init_embeddings.shape[0]} rows, expected {n_clones}. Falling back to SVD."
                )
                init = "svd"
                init_embeddings = None
            elif init_embeddings.shape[1] != z_dim:
                logg.warning(
                    f"init_obsm embeddings have {init_embeddings.shape[1]} dims, expected {z_dim}. Falling back to SVD."
                )
                init = "svd"
                init_embeddings = None

    # Build training matrix (optionally augmented with mask columns)
    X_base = clones.obsp[obsp_key]
    if mask_key is not None:
        try:
            M = clones.obsm[mask_key]
        except KeyError:
            raise KeyError(f"mask_key '{mask_key}' not found in clones.obsm.")
        M_vals = M.values if isinstance(M, pd.DataFrame) else M
        var_names = list(clones.obs_names) + [f"mask{i+1}" for i in range(M_vals.shape[1])]
        if sp.issparse(X_base):
            if sp.issparse(M_vals):
                X_train = sp.hstack([X_base, M_vals]).tocsr()
            else:
                X_train = sp.hstack([X_base, sp.csr_matrix(M_vals)])
        else:
            X_train = np.hstack([np.asarray(X_base), np.asarray(M_vals)])
    else:
        X_train = X_base
        var_names = np.array(clones.obs_names).tolist()

    model = SkipGram(z_dim=z_dim, device=device)
    model.fit(
        X_train,
        learning_rate=learning_rate,
        progress_bar=progress_bar,
        device=device,
        random_state=random_state,
        max_iter=max_iter,
        init=init,
        init_embeddings=init_embeddings,
        init_output_weight=init_output_weight,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    clones.obsm[obsm_key] = model.U
    clones.uns[uns_key] = {
        "init": {"z_dim": model.z_dim},
        "output": {
            "weight": model.output.weight.detach().cpu().numpy(),
            "bias": model.output.bias.detach().cpu().numpy(),
        },
        "var_names": np.array(var_names).copy(),
        "loss_history": -model.loglik_history_,
        "training_time": time.time() - start_time,
        "type": "fit",
    }

    lines = [
        "added",
        f"     .obsm['{obsm_key}'] embedding matrix",
        f"     .uns['{uns_key}'] training details",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def clone2vec_Poi(
    clones: sc.AnnData,
    z_dim: int = 10,
    obsp_key: str = "gex_adjacency",
    mask_key: str | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    learning_rate: float | None = 0.5,
    device: str | None = None,
    progress_bar: bool = True,
    obsm_key: str = "clone2vec_Poi",
    uns_key: str = "clone2vec_Poi",
    random_state: None | int = 4,
    col_size_factor: bool = True,
    row_intercept: bool = True,
    num_ccd_iter: int = 3,
    adaptive_lr: bool = True,
    slowing_loglik: bool = True,
    lr_decay: float = 0.5,
    min_learning_rate: float = 1e-5,
    max_backtracks: int = 3,
    batch_size_rows: int | None = None,
    batch_size_cols: int | None = None,
    init: str = "svd",
) -> None | torch.nn.Module:
    """
    Learn a clonal embedding using FastGLMPCA (Poisson) and store the embeddings.

    Parameters
    ----------
    clones : AnnData
        The clone-level AnnData object, typically from `create_clone_adata`.
    z_dim : int, optional
        Dimensionality of the clonal embedding, by default 10.
    obsp_key : str, optional
        Key in `clones.obsp` for the graph to use, by default "gex_adjacency".
    max_iter : int, optional
        Maximum number of iterations for optimization, by default 500.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    learning_rate : float | None, optional
        Learning rate for optimization, by default 0.5.
    device : str | None, optional
        Device to use for computation, by default None.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    obsm_key : str, optional
        Key in `clones.obsm` to store the embeddings, by default "clone2vec_Poi".
    uns_key: str, optional
        Key in `clones.uns` to store the model parameters, by default "clone2vec_Poi".
    random_state : None | int, optional
        Random state for reproducibility, by default 4.
    col_size_factor : bool, optional
        Whether to use column size factors, by default True.
    row_intercept : bool, optional
        Whether to use row intercepts, by default True.
    num_ccd_iter : int, optional
        Number of CCD iterations, by default 3.
    adaptive_lr : bool, optional
        If True, reduce learning rate on log-likelihood drops. Default is True.
    slowing_loglik : bool, optional
        If True, adaptively reduce learning rate when log-likelihood changing rate increases. Default is True.
    lr_decay : float, optional
        Multiplicative decay factor applied when log-likelihood decreases. Default is 0.5.
    min_learning_rate : float, optional
        Minimum allowed learning rate during adaptation. Default is 1e-5.
    max_backtracks : int, optional
        Maximum number of backtracking retries per iteration when log-likelihood decreases. Default is 3.
    batch_size_rows : int | None, optional
        Batch size for rows, by default None.
    batch_size_cols : int | None, optional
        Batch size for columns, by default None.
    init : str, optional
        Initialization method, by default "svd".

    Returns
    -------
    None | torch.nn.Module
        If `return_model` is True, returns the trained model.
    """
    try:
        import fastglmpca
    except ImportError:
        raise ImportError("py-fastglmpca is not installed. Please install it to use this function via `pip install fastglmpca`")

    start = logg.info("fitting clone2vec_Poi embeddings")
    if obsp_key not in clones.obsp:
        raise KeyError(f"Graph '{obsp_key}' not found in clones.obsp. Did you run `sl.tl.clonal_nn(adata, clones)` before?")

    start_time = time.time()

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    def custom_tqdm(x):
        class _TqdmIter:
            def __init__(self, iterable, desc, file):
                try:
                    self._tqdm = tqdm(iterable, desc=desc, file=file, leave=True)
                except Exception:
                    self._tqdm = iterable
                self._closed = False
            def __iter__(self):
                return iter(self._tqdm)
            def set_postfix(self, *args, **kwargs):
                try:
                    if hasattr(self._tqdm, "set_postfix"):
                        self._tqdm.set_postfix(*args, **kwargs)
                except Exception:
                    pass
            def close(self):
                try:
                    if hasattr(self._tqdm, "close") and not self._closed:
                        self._tqdm.close()
                        self._closed = True
                except Exception:
                    pass
            def __del__(self):
                self.close()
        return _TqdmIter(x, desc=prefix + "GLM-PCA epochs", file=sys.stdout)

    if progress_bar:
        custom_iter = custom_tqdm
    else:
        custom_iter = None

    # Build training matrix (optionally augmented with mask columns)
    X_base = clones.obsp[obsp_key]
    if mask_key is not None:
        try:
            M = clones.obsm[mask_key]
        except KeyError:
            raise KeyError(f"mask_key '{mask_key}' not found in clones.obsm.")
        M_vals = M.values if isinstance(M, pd.DataFrame) else M
        var_names = list(clones.obs_names) + [f"mask{i+1}" for i in range(M_vals.shape[1])]
        if sp.issparse(X_base):
            if sp.issparse(M_vals):
                X_train = sp.hstack([X_base, M_vals]).tocsr()
            else:
                X_train = sp.hstack([X_base, sp.csr_matrix(M_vals)])
        else:
            X_train = np.hstack([np.asarray(X_base), np.asarray(M_vals)])
    else:
        X_train = X_base
        var_names = np.array(clones.obs_names).tolist()

    model = fastglmpca.poisson(
        X_train,
        max_iter=max_iter,
        n_pcs=z_dim,
        tol=tol,
        return_model=True,
        learning_rate=learning_rate,
        progress_bar=progress_bar,
        device=device,
        verbose=False,
        seed=random_state,
        col_size_factor=col_size_factor,
        row_intercept=row_intercept,
        num_ccd_iter=num_ccd_iter,
        adaptive_lr=adaptive_lr,
        slowing_loglik=slowing_loglik,
        lr_decay=lr_decay,
        min_learning_rate=min_learning_rate,
        max_backtracks=max_backtracks,
        batch_size_rows=batch_size_rows,
        batch_size_cols=batch_size_cols,
        init=init,
        custom_iterator=custom_iter,
    )

    if obsm_key in clones.obsm:
        logg.warning(f"obsm_key '{obsm_key}' already exists in clones.obsm. Overwriting.")
    clones.obsm[obsm_key] = model.U

    if uns_key in clones.uns:
        logg.warning(f"uns_key '{uns_key}' already exists in clones.uns. Overwriting.")
    clones.uns[uns_key] = {
        "init": {
            "n_pcs": model.n_pcs,
            "col_size_factor": model.col_size_factor,
            "row_intercept": model.row_intercept,
            "adaptive_lr": model.adaptive_lr,
            "slowing_loglik": model.slowing_loglik,
            "lr_decay": model.lr_decay,
            "min_learning_rate": model.min_learning_rate,
            "max_backtracks": model.max_backtracks,
            "batch_size_rows": model.batch_size_rows,
            "batch_size_cols": model.batch_size_cols,
            "num_ccd_iter": model.num_ccd_iter,
        },
        "weights": {
            "V": model.V.copy(),
            "d": model.d.copy(),
            "col_offset": model.col_offset.cpu().numpy().copy(),
        },
        "loss_history": -np.array(model.loglik_history_),
        "var_names": np.array(var_names).copy(),
        "training_time": time.time() - start_time,
        "type": "fit",
    }
    
    if model.row_intercept:
        clones.uns[uns_key]["weights"]["row_offset"] = model.row_offset.cpu().numpy().copy()

    lines = [
        "added",
        f"     .obsm['{obsm_key}'] embedding matrix",
        f"     .uns['{uns_key}'] training details",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def project_clone2vec(
    clones_query: sc.AnnData,
    clones_ref: sc.AnnData,
    obsm_key_query: str = "ref_gex_adjacency",
    uns_key_query: str = "clone2vec",
    uns_key_ref: str = "clone2vec",
    obsm_key: str = "clone2vec",
    mask_key: str | None = None,
    random_state: int | None = 42,
    batch_size: int = 128,
    device: str | None = None,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    progress_bar: bool = True,
    max_iter: int = 500,
) -> None:
    """
    Fitting of new clones to the reference clone2vec embeddings using kNN between query and reference datasets.
    The function is using output embedding from the reference clone2vec model and optimizes only input embedding matrix,
    therefore has much faster convergence than training the whole model.

    Parameters
    ----------
    clones_query : sc.AnnData
        The query clone-level AnnData object, typically from `create_clone_adata`.
    clones_ref : sc.AnnData
        The reference clone-level AnnData object, typically from `create_clone_adata`.
    obsm_key_query : str, optional
        Key in `clones_query.obsm` for the graph to use, by default "ref_gex_adjacency".
    uns_key_query : str, optional
        Key in `clones_query.uns` to store the model parameters, by default "clone2vec".
    uns_key_ref : str, optional
        Key in `clones_ref.uns` to store the model parameters, by default "clone2vec".
    obsm_key : str, optional
        Key in `clones_query.obsm` to store the embeddings, by default "clone2vec".
    random_state : None | int, optional
        Random state for reproducibility, by default 42.
    batch_size : int, optional
        Batch size for optimization, by default 128.
    device : str | None, optional
        Device to use for optimization, by default None.
    learning_rate : float, optional
        Learning rate for optimization, by default 0.001.
    early_stopping_patience : int, optional
        Number of iterations with no improvement to wait before stopping, by default 5.
    early_stopping_min_delta : float, optional
        Minimum change in the loss to be considered as an improvement, by default 1e-4.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    """
    start = logg.info("projecting clone2vec embeddings onto reference")
    start_time = time.time()

    try:
        params = clones_ref.uns[uns_key_ref]
    except KeyError:
        raise KeyError(f"clone2vec embeddings not found in clones_ref.uns['{uns_key_ref}']. Did you run `clone2vec` before?")
    
    model = SkipGram(**params["init"], device=device)
    base_out = len(clones_ref)
    extra_cols = 0
    if mask_key is not None:
        try:
            M_tmp = clones_query.obsm[mask_key]
            extra_cols = (M_tmp.values if isinstance(M_tmp, pd.DataFrame) else M_tmp).shape[1]
        except KeyError:
            extra_cols = 0
    model._initialize_or_resize_layers(len(clones_query), base_out + extra_cols)
    with torch.no_grad():
        model.output.weight.data[:base_out] = torch.tensor(params["output"]["weight"])
        model.output.bias.data[:base_out] = torch.tensor(params["output"]["bias"])
        if extra_cols > 0:
            model.output.weight.data[base_out:] = 0.0
            model.output.bias.data[base_out:] = 0.0

    try:
        nn = clones_query.obsm[obsm_key_query]
    except KeyError:
        raise KeyError(f"obsm_key_query '{obsm_key_query}' not found in clones_query.obsm. It should contain nearest neighbours of the query clones from the reference dataset.")
    
    if not isinstance(nn, pd.DataFrame):
        if nn.shape[1] != len(clones_ref.obs_names):
            raise ValueError(f"In the case of unnamed columns, clones_query.obsm['{obsm_key_query}'] should have {len(clones_ref.obs_names)} columns, but got {nn.shape[1]}.")
        logg.warning(f"clones_query.obsm['{obsm_key_query}'] contains a matrix with unnamed columns. We're assuming that the order of neighbours is the same with clones_ref.obs_names")
    else:
        nn = nn.reindex(columns=clones_ref.obs_names, fill_value=0).values

    if mask_key is not None:
        try:
            M = clones_query.obsm[mask_key]
        except KeyError:
            raise KeyError(f"mask_key '{mask_key}' not found in clones_query.obsm.")
        M_vals = M.values if isinstance(M, pd.DataFrame) else M
        if sp.issparse(nn):
            nn = sp.hstack([nn, sp.csr_matrix(M_vals)])
        else:
            nn = np.hstack([np.asarray(nn), np.asarray(M_vals)])

    if obsm_key in clones_query.obsm:
        logg.warning(f"obsm_key '{obsm_key}' already exists in clones_query.obsm. Overwriting.")

    clones_query.obsm[obsm_key] = model.project(
        X_new=nn,
        random_state=random_state,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        progress_bar=progress_bar,
        max_iter=max_iter,
    )

    if uns_key_query in clones_query.uns:
        logg.warning(f"uns_key_query '{uns_key_query}' already exists in clones_query.uns. Overwriting.")
    
    var_names = list(clones_ref.obs_names) + ([f"mask{i+1}" for i in range(M_vals.shape[1])] if mask_key is not None else [])
    clones_query.uns[uns_key_query] = {
        "init": {"z_dim": model.z_dim},
        "loss_history": -model.project_loglik_history_,
        "training_time": time.time() - start_time,
        "type": "project",
        "var_names": np.array(var_names).copy(),
    }
    lines = [
        "added",
        f"     .obsm['{obsm_key}'] projected embeddings",
        f"     .uns['{uns_key_query}'] training details",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)

def project_clone2vec_Poi(
    clones_query: sc.AnnData,
    clones_ref: sc.AnnData,
    obsm_key_query: str = "ref_gex_adjacency",
    uns_key_query: str = "clone2vec_Poi_project",
    uns_key_ref: str = "clone2vec_Poi",
    obsm_key: str = "clone2vec_Poi",
    mask_key: str | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    device: str | None = None,
    progress_bar: bool = True,
    random_state: int | None = 42,
    init: str = "svd",
    **kwargs,
):
    """
    Fitting of new clones to the reference clone2vec Poisson embeddings using kNN between query and reference datasets.
    The function is using output embedding from the reference clone2vec model and optimizes only input embedding matrix,
    therefore has much faster convergence than training the whole model.

    Parameters
    ----------
    clones_query : AnnData
        The query clone-level AnnData object, typically from `create_clone_adata`.
    clones_ref : AnnData
        The reference clone-level AnnData object, typically from `create_clone_adata`.
    obsm_key_query : str, optional
        Key in `clones_query.obsm` for the graph to use, by default "ref_gex_adjacency".
    uns_key_query : str, optional
        Key in `clones_query.uns` to store the model parameters, by default "clone2vec".
    uns_key_ref : str, optional
        Key in `clones_ref.uns` to store the model parameters, by default "clone2vec".
    obsm_key : str, optional
        Key in `clones_query.obsm` to store the embeddings, by default "clone2vec".
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    device : str | None, optional
        Device to use for optimization, by default None.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    random_state : None | int, optional
        Random state for reproducibility, by default 42.
    init : str, optional
        Initialization method for the model, by default "svd".
    **kwargs
        Additional keyword arguments for the model.
    """
    try:
        import fastglmpca
    except ImportError:
        raise ImportError("py-fastglmpca is not installed. Please install it to use this function via `pip install fastglmpca`")

    start = logg.info("projecting clone2vec_Poi embeddings onto reference")
    start_time = time.time()

    try:
        params = clones_ref.uns[uns_key_ref]
    except KeyError:
        raise KeyError(f"clone2vec_Poi embeddings not found in clones_ref.uns['{uns_key_ref}']. Did you run `clone2vec_Poi` before?")
    
    model = fastglmpca.PoissonGLMPCA(**params["init"], device=device)
    base_out = params["weights"]["V"].shape[0]
    extra_cols = 0
    model.V = params["weights"]["V"].copy()
    model.d = params["weights"]["d"].copy()
    if mask_key is not None:
        try:
            M_tmp = clones_query.obsm[mask_key]
            extra_cols = (M_tmp.values if isinstance(M_tmp, pd.DataFrame) else M_tmp).shape[1]
        except KeyError:
            extra_cols = 0
    if extra_cols > 0:
        model.V = np.vstack([model.V, np.zeros((extra_cols, model.V.shape[1]), dtype=model.V.dtype)])
        model.d = np.concatenate([model.d, np.zeros(extra_cols, dtype=model.d.dtype)])
    model.col_offset = params["weights"]["col_offset"]
    if "row_offset" in params["weights"]:
        model.row_offset = params["weights"]["row_offset"]

    try:
        nn = clones_query.obsm[obsm_key_query]
    except KeyError:
        raise KeyError(f"obsm_key_query '{obsm_key_query}' not found in clones_query.obsm. It should contain nearest neighbours of the query clones from the reference dataset.")
    
    if not isinstance(nn, pd.DataFrame):
        if nn.shape[1] != len(clones_ref.obs_names):
            raise ValueError(f"In the case of unnamed columns, clones_query.obsm['{obsm_key_query}'] should have {len(clones_ref.obs_names)} columns, but got {nn.shape[1]}.")
        logg.warning(f"clones_query.obsm['{obsm_key_query}'] contains a matrix with unnamed columns. We're assuming that the order of neighbours is the same with clones_ref.obs_names")
    else:
        nn = nn.reindex(columns=clones_ref.obs_names, fill_value=0).values

    if mask_key is not None:
        try:
            M = clones_query.obsm[mask_key]
        except KeyError:
            raise KeyError(f"mask_key '{mask_key}' not found in clones_query.obsm.")
        M_vals = M.values if isinstance(M, pd.DataFrame) else M
        if sp.issparse(nn):
            nn = sp.hstack([nn, sp.csr_matrix(M_vals)])
        else:
            nn = np.hstack([np.asarray(nn), np.asarray(M_vals)])

    if obsm_key in clones_query.obsm:
        logg.warning(f"obsm_key '{obsm_key}' already exists in clones_query.obsm. Overwriting.")

    if sc.settings.verbosity.value >= 2:
        prefix = "    "
        progress_bar = True
    else:
        prefix = ""

    def custom_tqdm(x):
        class _TqdmIter:
            def __init__(self, iterable, desc, file):
                try:
                    self._tqdm = tqdm(iterable, desc=desc, file=file, leave=True)
                except Exception:
                    self._tqdm = iterable
                self._closed = False
            def __iter__(self):
                return iter(self._tqdm)
            def set_postfix(self, *args, **kwargs):
                try:
                    if hasattr(self._tqdm, "set_postfix"):
                        self._tqdm.set_postfix(*args, **kwargs)
                except Exception:
                    pass
            def close(self):
                try:
                    if hasattr(self._tqdm, "close") and not self._closed:
                        self._tqdm.close()
                        self._closed = True
                except Exception:
                    pass
            def __del__(self):
                self.close()
        return _TqdmIter(x, desc=prefix + "GLM-PCA epochs", file=sys.stdout)

    if progress_bar:
        custom_iter = custom_tqdm
    else:
        custom_iter = None

    clones_query.obsm[obsm_key] = model.project(
        nn,
        max_iter=max_iter,
        tol=tol,
        progress_bar=progress_bar,
        init=init,
        seed=random_state,
        custom_iterator=custom_iter,
    )

    if uns_key_query in clones_query.uns:
        logg.warning(f"uns_key_query '{uns_key_query}' already exists in clones_query.uns. Overwriting.")

    var_names = list(clones_ref.obs_names) + ([f"mask{i+1}" for i in range(M_vals.shape[1])] if mask_key is not None else [])
    clones_query.uns[uns_key_query] = {
        "init": {"n_pcs": model.n_pcs},
        "loss_history": -np.array(model.project_loglik_history_),
        "training_time": time.time() - start_time,
        "type": "project",
        "var_names": np.array(var_names).copy(),
    }
    lines = [
        "added",
        f"     .obsm['{obsm_key}'] projected embeddings",
        f"     .uns['{uns_key_query}'] training details",
    ]
    logg.info("    finished ({time_passed})", deep="\n".join(lines), time=start)
