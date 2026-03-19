from __future__ import annotations
from typing import Literal
from collections.abc import Iterable

import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp

import matplotlib
import matplotlib.pyplot as plt

logg = sc.logging

# Curate public API for tab-completion: only plotting functions
__all__ = [
    "group_scatter",
    "group_kde",
    "loss_history",
    "clone_size",
    "nesting_clones",
    "volcano",
    "shap_volcano",
    "barplot",
    "heatmap",
    "catboost_perfomance",
    "clones2cells",
    "predictors_comparison",
    "graph",
    "group_connectivity",
    "embedding_axis",
    "small_cbar",
    "fancy_legend",
    "scaled_dotplot",
    "scatter2vars",
    "pca_loadings",
]

def __dir__():
    return sorted(__all__)

def pca_loadings(
    adata: sc.AnnData,
    key: str = "gPCs",
    comp: int = 0,
    n_genes: int = 5,
    kind: Literal["h", "v"] = "v",
    ax: matplotlib.axes.Axes | None = None,
    show: bool = True,
    return_fig: bool = False,
    title: str | None = None,
    full_border: bool = False,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.axes.Axes | matplotlib.figure.Figure | None:
    """
    Plot PCA loadings for a given component.

    Parameters
    ----------
    adata: sc.AnnData
        Annotated data matrix.
    key: str
        Key for PCA loadings in `adata.varm`.
    comp: int
        Component to plot.
    n_genes: int
        Number of genes to plot.
    kind: Literal["h", "v"]
        Kind of plot, either "h" for horizontal or "v" for vertical.
    ax: matplotlib.axes.Axes | None
        Axes object to plot on.
    show: bool
        Whether to show the plot.
    return_fig: bool
        Whether to return the figure object.
    title: str | None
        Title for the plot.
    full_border: bool
        Whether to draw full border around the plot.
    figsize: tuple[float, float] | None
        Figure size.

    Returns
    -------
    ax: matplotlib.axes.Axes | None
        Axes object with the plot.
    """
    loadings = pd.Series(adata.varm[key][:, comp], index=adata.var_names).sort_values()
    loadings = pd.concat([loadings[:n_genes], loadings[-n_genes:]])

    if figsize is None:
        figsize = (4, 4)

    if ax:
        show = False
        return_ax = False
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    if title is None:
        if key == "gPCs":
            title = f"gPC{comp + 1} loadings"
        elif key == "PCs":
            title = f"PC{comp + 1} loadings"
        else:
            title = f"{key}{comp} loadings"

    barplot(
        loadings,
        cmap="RdBu_r",
        kind=kind,
        vmax=np.abs(loadings).max(),
        vmin=-np.abs(loadings).max(),
        left_spines=True,
        bottom_spines=True,
        top_spines=full_border,
        right_spines=full_border,
        title=title,
        arrow=([] if full_border else None),
        ax=ax,
        show=False,
        return_fig=False,
    )

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()


def group_connectivity(
    adata: sc.AnnData,
    basis: str = "X_umap",
    color_bg: bool = False,
    alpha_bg: float = 0.5,
    s_bg: float | None = None,
    clip: tuple[float, float] = (1, 2),
    ax: matplotlib.axes.Axes | None = None,
    return_fig: bool = False,
    uns_key: str = "group_connectivity",
    title: str | None = None,
    show: bool = True,
    linewidth: float = 1.,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.axes.Axes | matplotlib.figure.Figure | None:
    """
    Plot group connectivity on top of the embedding.

    Parameters
    ----------
    adata: sc.AnnData
        Annotated data matrix.
    basis: str
        Key for embedding coordinates in `adata.obsm`.
    color_bg: bool
        Whether to color the background points.
    alpha_bg: float
        Alpha value for the background points.
    s_bg: float | None
        Size of the background points.
    clip: tuple[float, float]
        Clip values for the connectivity matrix.
    ax: matplotlib.axes.Axes | None
        Axes object to plot on.
    return_fig: bool
        Whether to return the figure object.
    uns_key: str
        Key for group connectivity in `adata.uns`.
    title: str | None
        Title for the plot.
    show: bool
        Whether to show the plot.
    linewidth: float
        Linewidth for the connections.
    figsize: tuple[float, float] | None
        Figure size.

    Returns
    -------
    ax: matplotlib.axes.Axes | matplotlib.figure.Figure | None
        Axes object with the plot.
    """
    from .utils import get_connectivity_matrix

    emb = adata.obsm[basis].copy()
    groupby = adata.uns[uns_key]["groupby"]

    sc.pl._utils.add_colors_for_categorical_sample_annotation(adata, groupby)
    palette = dict(zip(
        adata.obs[groupby].cat.categories,
        adata.uns[f"{groupby}_colors"],
    ))

    if figsize is None:
        figsize = (4, 4)

    if ax:
        show = False
        return_ax = False
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    if not color_bg:
        alpha = 0
    else:
        alpha = alpha_bg

    if title is None:
        title = f"{groupby} connectivity"

    sc.pl.embedding(
        adata,
        basis=basis,
        color=groupby,
        ax=ax,
        alpha=alpha,
        show=False,
        s=s_bg,
        title=title,
    )
    if not color_bg:
        sc.pl.embedding(
            adata,
            basis=basis,
            color=None,
            ax=ax,
            show=False,
            alpha=alpha_bg,
            s=s_bg,
            title=title,
        )

    connectivity = get_connectivity_matrix(adata, uns_key).copy()
    connectivity.values[connectivity.values < clip[0]] = 0
    connectivity.values[connectivity.values > clip[1]] = clip[1]

    emb = pd.DataFrame(
        emb, index=adata.obs_names, columns=["emb1", "emb2"]
    ).groupby(adata.obs[groupby], observed=True).mean()
    for group in emb.index:
        ax.scatter(
            x=emb.loc[group, "emb1"],
            y=emb.loc[group, "emb2"],
            color=palette[group],
            edgecolor="black",
            linewidth=connectivity.loc[group, group] * linewidth,
            s=50,
            zorder=2,
        )

    rows, cols = connectivity.values.nonzero()
    for row, col in zip(rows, cols):
        lw = connectivity.values[row, col] * linewidth
        ax.annotate(
            "",
            xytext=(emb.iloc[row]["emb1"], emb.iloc[row]["emb2"]),
            xy=(emb.iloc[col]["emb1"], emb.iloc[col]["emb2"]),
            xycoords="data",
            arrowprops={
                "arrowstyle": "-|>",
                "lw": lw,
                "color": "black",
                "alpha": 1,
                "mutation_scale": 10,
            },
            zorder=1,
        )

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()
    

def graph(
    adata: sc.AnnData,
    graph_key: str = "connectivities",
    basis: str = "X_umap",
    oriented: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    return_fig: bool = False,
    show: bool = False,
    linewidth: float = 1.,
    linecolor: str = "black",
    linealpha: float = 1.,
    arrowstyle: str | None = None,
    mutation_scale: float = 10.,
    width_condition: "Callable" | None = None,
    color_condition: "Callable" | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> matplotlib.axes.Axes | matplotlib.figure.Figure | None:
    """
    Draws graph on top of the embedding.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with graph in obsp[graph_key].
    graph_key : str, optional
        Key in obsp to store graph. Default is "connectivities".
    basis : str, optional
        Key in obsm to store embedding. Default is "X_umap".
    oriented : bool, optional
        Whether to draw oriented edges. Default is False.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default is None.
    return_fig : bool, optional
        Whether to return figure. Default is False.
    linewidth : float, optional
        Line width. Default is 1.
    linecolor : str, optional
        Line color. Default is "black".
    linealpha : float, optional
        Line alpha. Default is 1.
    arrowstyle : str, optional
        Arrow style. Default is None.
    mutation_scale : float, optional
        Mutation scale. Default is 10.
    width_condition : Callable, optional
        Function to calculate line width. Default is None.
    color_condition : Callable, optional
        Function to calculate line color. Default is None.
    figsize : tuple[float, float], optional
        Figure size. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments for sc.pl.embedding.

    Returns
    -------
    matplotlib.axes.Axes | matplotlib.figure.Figure | None
        Draws with the graph on top.
    """

    if figsize is None:
        figsize = (4, 4)

    if ax:
        show = False
        return_ax = False
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True
        
    sc.pl.embedding(
        adata,
        basis=basis,
        ax=ax,
        show=False,
        **kwargs,
    )
    
    graph = adata.obsp[graph_key]
    if sp.issparse(graph):
        graph = graph.tocoo()
        if sum((graph - graph.T).data >= 1e-10) == 0:
            oriented = False
            
        if not oriented:
            graph = sp.triu(graph + graph.T)
            
        starts = graph.row
        ends = graph.col
    else:
        if ((graph - graph.T) >= 1e-10).sum() == 0:
            oriented = False
        
        if not oriented:
            graph = np.triu(graph + graph.T)
            
        starts, ends = graph.nonzero()
        
    emb = adata.obsm[basis]
    for start, end in zip(starts, ends):
        x_start, y_start = emb[start, :2]
        x_end, y_end = emb[end, :2]
        
        if arrowstyle is None:
            if oriented:
                arrowstyle = "-|>"
            else:
                arrowstyle = "-"
            
        if color_condition:
            linecolor = color_condition(adata[start], adata[end])
            
        if width_condition:
            linewidth = width_condition(adata[start], adata[end])
            
        ax.annotate(
            "",
            xytext=(x_start, y_start),
            xy=(x_end, y_end),
            xycoords="data",
            arrowprops={
                "arrowstyle": arrowstyle,
                "lw": linewidth,
                "color": linecolor,
                "alpha": linealpha,
                "mutation_scale": linewidth * mutation_scale,
            }
        )

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()

def group_scatter(
    adata: sc.AnnData,
    groupby: str,
    groups: list[str] | str | None = None,
    s: list[float] | float = 30,
    ncols: int = 4,
    group_color: list[str] | str = "black",
    frameon: list[bool] | bool = False,
    square: bool = True,
    kwargs_background: list[dict] | dict | None = None,
    kwargs_group: list[dict] | dict | None = None,
    ax: matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None = None,
    title: list[str] | str = None,
    return_fig: bool = False,
    basis: list[str] | str = "X_umap",
    show: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes] | None:
    """
    Plots the spatial distribution of a single clone on a 2D embedding.

    First plots all cells as a background, then overlays the selected clone using a distinct color.

    Parameters
    ----------
    adata : AnnData
        AnnData object with cell metadata and embeddings.
    groupby : str
        Column in `adata.obs` indicating clonal identity.
    groups : list[str] | str | None
        Clones to be highlighted. If None, all groups are shown.
    frameon : list[bool] | bool, optional
        Whether to show axes frame in the plot. Default is False.
    square : bool, optional
        Whether to set the aspect ratio of the plot to be equal. Default is True.
    s : list[float] | float, optional
        Dot size for the highlighted clone. Default is 30.
    kwargs_background : list[dict] | dict | None, optional
        Additional plotting arguments for the background cells.
    kwargs_group : list[dict] | dict | None, optional
        Additional plotting arguments for the highlighted clone.
    group_color : list[str] | str, optional
        Color to use for the highlighted clone. Default is "black".
    ax : matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None, optional
        Axes object to draw the plot on. If None, a new figure is created.
    title : list[str] | str, optional
        Plot title. If None, uses the clone name.
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    basis : list[str] | str, optional
        Embedding key in `adata.obsm` to use for coordinates. Default is "X_umap".
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float], optional
        Figure size. Default is None.

    Returns
    -------
    None | matplotlib.figure.Figure | matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes]
        Returns Figure if `return_fig` is True; otherwise returns None.
    """
    if groups is None:
        try:
            groups = adata.obs[groupby].cat.categories
        except:
            groups = list(set(adata.obs[groupby]))
        if len(groups) > 40:
            raise ValueError("More than 40 groups found in adata.obs[groupby].cat.categories. Please provide a list of groups to plot.")
    if isinstance(groups, str):
        groups = [groups]
    if not isinstance(s, list):
        s = [s] * len(groups)
    elif len(s) != len(groups):
        raise ValueError("s must be a float or a list of floats with length equal to the number of groups")
    if isinstance(group_color, str):
        group_color = [group_color] * len(groups)
    elif len(group_color) != len(groups):
        raise ValueError("group_color must be a str or a list of str with length equal to the number of groups")
    if isinstance(frameon, bool):
        frameon = [frameon] * len(groups)
    elif len(frameon) != len(groups):
        raise ValueError("frameon must be a bool or a list of bool with length equal to the number of groups")
    if kwargs_background is None:
        kwargs_background = [{} for i in range(len(groups))]
    if isinstance(kwargs_background, dict):
        kwargs_background = [kwargs_backgroundcopy() for i in range(len(groups))]
    elif len(kwargs_background) != len(groups):
        raise ValueError("kwargs_background must be a dict or a list of dict with length equal to the number of groups")
    if kwargs_group is None:
        kwargs_group = [{} for i in range(len(groups))]
    if isinstance(kwargs_group, dict):
        kwargs_group = [kwargs_group.copy() for i in range(len(groups))]
    elif len(kwargs_group) != len(groups):
        raise ValueError("kwargs_group must be a dict or a list of dict with length equal to the number of groups")
    if title is None:
        title = [None for i in range(len(groups))]
    elif isinstance(title, str):
        title = [title for i in range(len(groups))]
    elif len(title) != len(groups):
        raise ValueError("title must be a str or a list of str with length equal to the number of groups")
    if isinstance(basis, str):
        basis = [basis] * len(groups)
    elif len(basis) != len(groups):
        raise ValueError("basis must be a str or a list of str with length equal to the number of groups")
    
    ncols = min(len(groups), ncols)
    nrows = len(groups) // ncols
    if len(groups) % ncols != 0:
        nrows += 1

    if figsize is None:
        figsize = (3.5 * ncols, 3.5 * nrows)

    if ax:
        show = False
        return_ax = False
        tight_layout = False
        fig = None
        axes = ax
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        tight_layout = True
        if show:
            return_ax = False
        else:
            return_ax = True

    if isinstance(axes, matplotlib.axes.Axes):
        axes_grid = np.array([[axes]])
    else:
        axes_arr = np.array(axes, dtype=object)
        if axes_arr.ndim == 0:
            axes_grid = np.array([[axes_arr.item()]])
        elif axes_arr.ndim == 1:
            total = nrows * ncols
            if axes_arr.size < total:
                raise ValueError("Provided axes contain fewer panels than required for the number of groups")
            axes_grid = axes_arr.flat[:total].reshape(nrows, ncols)
        elif axes_arr.ndim >= 2:
            total = nrows * ncols
            if axes_arr.size < total:
                raise ValueError("Provided axes contain fewer panels than required for the number of groups")
            axes_grid = axes_arr.flat[:total].reshape(nrows, ncols)
        else:
            raise TypeError("ax must be an Axes, a list/array of Axes, or None")

    for i in range(len(groups)):
        col = i % ncols
        row = i // ncols
        ax_i = axes_grid[row, col]
        
        sc.pl.embedding(
            adata,
            basis=basis[i],
            ax=ax_i,
            show=False,
            frameon=frameon[i],
            **kwargs_background[i]
        )

        if "s" not in kwargs_group[i]:
            kwargs_group[i]["s"] = s[i]

        if title[i] is None:
            title[i] = str(groups[i])

        if "title" not in kwargs_group[i]:
            if "color" not in kwargs_group[i]:
                kwargs_group[i]["title"] = title[i]
            else:
                kwargs_group[i]["title"] = title[i] + "\n(" + kwargs_group[i]["color"] + ")"

        if "color" in kwargs_group[i]:
            if kwargs_group[i]["color"] in adata.obs.columns:
                if not pd.api.types.is_numeric_dtype(adata.obs[kwargs_group[i]["color"]]):
                    adata.obs[kwargs_group[i]["color"]] = adata.obs[kwargs_group[i]["color"]].astype("category")
                    sc.pl._utils.add_colors_for_categorical_sample_annotation(adata, kwargs_group[i]["color"])

        sc.pl.embedding(
            adata[adata.obs[groupby] == groups[i]],
            basis=basis[i],
            ax=ax_i,
            show=False,
            frameon=frameon[i],
            na_color=group_color[i],
            **kwargs_group[i],
        )

        if square:
            try:
                ax_i.set_aspect("equal", "box")
            except Exception:
                ax_i.set_aspect("equal")

    for i in range(len(groups), ncols * nrows):
        axes_grid[i // ncols, i % ncols].axis("off")

    if tight_layout:
        fig.tight_layout()

    if return_fig:
        return fig
    elif return_ax:
        return axes_grid.flat[:len(groups)]
    elif show:
        plt.show()
        

def group_kde(
    adata: sc.AnnData,
    groupby: str,
    groups: list[str] | str | None = None,
    basis: str | list[str] = "X_umap",
    bw_method: list[float | str] | float | str = 0.15,
    ncols: int = 4,
    frameon: list[bool] | bool = False,
    square: bool = True,
    ax: matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None = None,
    return_fig: bool = False,
    cmap: list[matplotlib.colors.Colormap | str] | matplotlib.colors.Colormap | str = "Reds",
    title: list[str] | str | None = None,
    kwargs: list[dict] | dict | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes] | None:
    """
    Plots kernel density estimates (KDE) of cells from one or multiple groups on a 2D embedding.

    Uses `scipy.stats.gaussian_kde` to estimate density of a group and overlays it as a contour plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing embeddings and group annotations.
    groupby : str
        Column in `adata.obs` used to identify cell groups.
    groups : list[str] | str | None
        Specific group(s) to visualize.
    basis : str, optional
        Embedding key from `adata.obsm` to use. Default is "X_umap".
    bw_method : list[float | str] | float | str, optional
        Bandwidth for the KDE. Passed to `scipy.stats.gaussian_kde`. Can be a single value or a list per group. Default is 0.1.
    ncols : int, optional
        Number of columns in the subplot grid when plotting multiple groups. Default is 4.
    square : bool, optional
        Whether to set the aspect ratio of the plot to be equal. Default is True.
    frameon : list[bool] | bool, optional
        Whether to show axes frame for each subplot. Can be a single bool or a list per group. Default is False.
    ax : matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None, optional
        Axes object(s) to draw the plot on. If None, a new figure/grid is created.
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    cmap : list[ColorMap | str] | ColorMap | str, optional
        Colormap(s) to use for KDE overlay. Can be a single value or a list per group. Default is "Reds".
    title : list[str] | str | None, optional
        Plot title(s). If None, uses "{group} KDE".
    kwargs : list[dict] | dict | None, optional
        Additional arguments passed to `sc.pl.embedding` for the background. Can be a single dict or a list per group.
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.

    Returns
    -------
    None | matplotlib.figure.Figure | matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes]
        Returns Figure if `return_fig` is True; otherwise returns None or axes.
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import ListedColormap

    # Resolve groups for backward compatibility
    if groups is None:
        try:
            groups = adata.obs[groupby].cat.categories
        except:
            groups = list(set(adata.obs[groupby]))
        if len(groups) > 40:
            raise ValueError("More than 40 groups found in adata.obs[groupby].cat.categories. Please provide a list of groups to plot.")

    if isinstance(groups, str):
        groups = [groups]
    if groups is None or len(groups) == 0:
        raise ValueError("At least one group must be specified via `group` or `groups`.")

    # Normalize per-group parameters
    if isinstance(frameon, bool):
        frameon = [frameon] * len(groups)
    elif len(frameon) != len(groups):
        raise ValueError("frameon must be a bool or a list[bool] with length equal to number of groups")

    if isinstance(basis, str):
        basis = [basis] * len(groups)
    elif len(basis) != len(groups):
        raise ValueError("basis must be a str or a list[str] with length equal to number of groups")

    if isinstance(bw_method, (float, str)):
        bw_method = [bw_method] * len(groups)
    elif len(bw_method) != len(groups):
        raise ValueError("bw_method must be a scalar or a list with length equal to number of groups")

    if kwargs is None:
        kwargs = [{} for _ in range(len(groups))]
    elif isinstance(kwargs, dict):
        kwargs = [kwargs.copy() for _ in range(len(groups))]
    elif len(kwargs) != len(groups):
        raise ValueError("kwargs must be a dict or a list[dict] with length equal to number of groups")

    if title is None:
        title = [None for _ in range(len(groups))]
    elif isinstance(title, str):
        title = [title for _ in range(len(groups))]
    elif len(title) != len(groups):
        raise ValueError("title must be a str or a list[str] with length equal to number of groups")

    if not isinstance(cmap, list):
        cmap = [cmap for _ in range(len(groups))]
    elif len(cmap) != len(groups):
        raise ValueError("cmap must be a single colormap or a list with length equal to number of groups")

    # Determine subplot grid
    ncols = min(len(groups), ncols)
    nrows = len(groups) // ncols
    if len(groups) % ncols != 0:
        nrows += 1

    # Create axes grid or normalize provided axes to a 2D grid
    if figsize is None:
        figsize = (3.5 * ncols, 3.5 * nrows)

    if ax:
        tight_layout = False
        show = False
        return_ax = False
        fig = None
        axes = ax
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        tight_layout = True
        if show:
            return_ax = False
        else:
            return_ax = True

    # Normalize axes to 2D array shape (nrows, ncols)
    if isinstance(axes, matplotlib.axes.Axes):
        axes_grid = np.array([[axes]])
    else:
        axes_arr = np.array(axes, dtype=object)
        if axes_arr.ndim == 0:
            axes_grid = np.array([[axes_arr.item()]])
        elif axes_arr.ndim == 1:
            total = nrows * ncols
            if axes_arr.size < total:
                raise ValueError("Provided axes contain fewer panels than required for the number of groups")
            axes_grid = axes_arr.flat[:total].reshape(nrows, ncols)
        elif axes_arr.ndim >= 2:
            total = nrows * ncols
            if axes_arr.size < total:
                raise ValueError("Provided axes contain fewer panels than required for the number of groups")
            axes_grid = axes_arr.flat[:total].reshape(nrows, ncols)
        else:
            raise TypeError("ax must be an Axes, a list/array of Axes, or None")

    # For each group, draw background and overlay KDE if enough points
    for i in range(len(groups)):
        grp = groups[i]
        ax_i = axes_grid[i // ncols, i % ncols]

        if title[i] is None:
            title[i] = f"{grp} KDE"

        kwargs_i = dict(kwargs[i])
        if "title" not in kwargs_i:
            kwargs_i["title"] = title[i]

        if "frameon" in kwargs_i:
            pass
        else:
            kwargs_i["frameon"] = frameon[i]
        if "na_color" not in kwargs_i:
            kwargs_i["na_color"] = "black"

        if sum(adata.obs[groupby] == grp) < 5:
            logg.warning(f"Group {grp} has less than 5 points. Skipping KDE plot.")
            sc.pl.embedding(
                adata,
                basis=basis[i],
                ax=ax_i,
                show=False,
                **kwargs_i,
            )
        else:
            kernel = gaussian_kde(
                adata[adata.obs[groupby] == grp].obsm[basis[i]].T,
                bw_method=bw_method[i],
            )

            xmin = min(adata.obsm[basis[i]].T[0])
            xmax = max(adata.obsm[basis[i]].T[0])
            ymin = min(adata.obsm[basis[i]].T[1])
            ymax = max(adata.obsm[basis[i]].T[1])

            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)

            cmap_i = cmap[i]
            if not isinstance(cmap_i, ListedColormap):
                try:
                    cmap_i = matplotlib.colormaps[cmap_i]
                except Exception:
                    logg.warning(f"Invalid colormap name {cmap[i]}. Using default Reds.")
                    cmap_i = matplotlib.colormaps["Reds"]

            my_cmap = cmap_i(np.arange(cmap_i.N))
            alphas = np.linspace(0, 1, cmap_i.N)
            BG = np.asarray([1., 1., 1.,])
            for j in range(cmap_i.N):
                my_cmap[j, :-1] = my_cmap[j, :-1] * alphas[j] + BG * (1. - alphas[j])
            my_cmap = ListedColormap(my_cmap)

            sc.pl.embedding(
                adata,
                basis=basis[i],
                ax=ax_i,
                show=False,
                **kwargs_i,
            )
            ax_i.contour(xx, yy, f, colors="black", linewidths=1)
            ax_i.contourf(xx, yy, f, cmap=my_cmap, alpha=0.8)
        ax_i.set_title(title[i])

        if square:
            try:
                ax_i.set_aspect("equal", "box")
            except Exception:
                ax_i.set_aspect("equal")

    for i in range(len(groups), ncols * nrows):
        axes_grid[i // ncols, i % ncols].axis("off")

    if tight_layout:
        fig.tight_layout()

    if return_fig:
        return fig
    elif return_ax:
        return axes_grid.flat[:len(groups)]
    elif show:
        plt.show()

def loss_history(
    clones: sc.AnnData,
    uns_key: str = "clone2vec",
    return_fig: bool = False,
    loss_key: str = "loss_history",
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: list[matplotlib.axes.Axes] | None = None,
) -> plt.Figure | list[matplotlib.axes.Axes] | None:
    """
    Plot the mean loss per epoch and its change across epochs during clone2vec training.

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix containing training statistics in `clones.uns`.
    uns_key : str, optional
        Key in `clones.uns` that stores a list of mean losses per epoch.
        Default is "c2v_loss_history".
    loss_key : str, optional
        Key in `clones.uns[uns_key]` that stores a list of mean losses per epoch.
        Default is "loss_history".
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    show : bool, optional
        If True, shows the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    ax : list[matplotlib.axes.Axes] | None, optional
        List of axes to plot on. Default is None.

    Returns
    -------
    None or matplotlib.figure.Figure or list[matplotlib.axes.Axes]
        The plot figure if `return_fig` is True, or axes if `return_ax` is True, otherwise None.

    Raises
    -------
    KeyError
        If `uns_key` is not present in `clones.uns`.
    """
    if figsize is None:
        figsize = (8, 3)

    if ax:
        show = False
        return_ax = False
        axes = ax
        if isinstance(axes, (list, np.ndarray)):
            if len(axes) < 2:
                # Fallback or error? Assuming user knows what they are doing if passing axes
                pass
            fig = axes[0].figure
        else:
             # Should be a list
             axes = [ax, ax] # This might be wrong, but better than crashing if user passes one ax
             fig = ax.figure
    else:
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    try:
        loss = np.array(clones.uns[uns_key][loss_key])
    except KeyError:
        try:
            loss = np.array(clones.uns["clone2vec"][loss_key])
            logg.warning(f"uns_key {uns_key} not found in clones.uns. Using clone2vec[{loss_key}] instead.")
        except KeyError:
            try:
                loss = np.array(clones.uns["clone2vec_Poi"][loss_key])
                logg.warning(f"Neither {uns_key} nor clone2vec[{loss_key}] found in clones.uns. Using clone2vec_Poi[{loss_key}] instead.")
            except KeyError:
                raise KeyError(f"uns_key {uns_key}, clone2vec[{loss_key}], and clone2vec_Poi[{loss_key}] not found in clones.uns.")

    sns.lineplot(
        x=range(len(loss)),
        y=loss,
        ax=axes[0],
    )

    axes[0].set_xlabel("Number of epoch")
    axes[0].set_ylabel("Mean loss")
    axes[0].grid(alpha=0.3)

    sns.lineplot(
        x=range(len(loss) - 1),
        y=loss[:-1] - loss[1:],
        ax=axes[1],
    )

    axes[1].set_xlabel("Number of epoch")
    axes[1].set_ylabel("Δ(Mean loss)")
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")

    fig.tight_layout()

    if return_fig:
        return fig
    elif return_ax:
        return axes
    elif show:
        plt.show()

def clone_size(
    clones: sc.AnnData,
    return_fig: bool = False,
    title: str = "Clone size distribution",
    bins: int = 30,
    log: bool = True,
    alpha: float = 1,
    edgecolor: str = "black",
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: list[matplotlib.axes.Axes] | None = None,
) -> plt.Figure | list[matplotlib.axes.Axes] | None:
    """
    Plot basic statistics of clone size distribution.

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix containing clone size stat in clones.obs["n_cells"]
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    title : str, optional
        Title of the plot. Default is "Clone size distribution".
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    ax : list[matplotlib.axes.Axes] | None, optional
        List of axes to plot on. Default is None.

    Returns
    -------
    None or matplotlib.figure.Figure or list[matplotlib.axes.Axes]
        The plot figure if `return_fig` is True, or axes if `return_ax` is True, otherwise None.

    Notes
    -----
    Single-cell clone annotations with only one occurrence (clone size = 1) are excluded
    from both plots.
    """
    if figsize is None:
        figsize = (8, 4)

    if ax:
        show = False
        return_ax = False
        axes = ax
        if isinstance(axes, (list, np.ndarray)):
            if len(axes) < 2:
                pass
            fig = axes[0].figure
        else:
             axes = [ax, ax]
             fig = ax.figure
    else:
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    sns.histplot(
        clones.obs["n_cells"],
        bins=bins,
        log=log,
        alpha=alpha,
        edgecolor=edgecolor,
        ax=axes[0],
    )
    axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Clone size")
    axes[0].set_ylabel("Number of clones")

    size_dist = clones.obs["n_cells"].value_counts().sort_index()

    sns.lineplot(np.cumsum(size_dist[::-1]), ax=axes[1])
    axes[1].grid(alpha=0.3)
    if log:
        axes[1].set_xscale("log")
    axes[1].set_xlabel("Clone size")
    axes[1].set_ylabel("Number of clones bigger than this")

    plt.suptitle(title)
    fig.tight_layout()
    
    if return_fig:
        return fig
    elif return_ax:
        return axes
    elif show:
        plt.show()
    
def nesting_clones(
    adata: sc.AnnData,
    early_injection: str,
    late_injection: str,
    min_clone_size: int = 5,
    non_clonal_str: str = "NA",
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    return_fig: bool = False,
) -> plt.Figure | matplotlib.axes.Axes | None:
    """
    Plot ternary composition of clones labeled by two different injections.

    In ideal clonal reconstructions, smaller clones from a later injection
    should nest within larger clones from an earlier injection — resembling
    a "Russian doll" pattern. This function visualizes the composition of
    shared and unique cells across early and late injections using a ternary plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clonal labels in `.obs`.
    early_injection : str
        Column name in `.obs` containing clone IDs from the early injection.
    late_injection : str
        Column name in `.obs` containing clone IDs from the late injection.
    min_clone_size : int, optional
        Minimum number of cells required for a clone to be included in the plot.
        Default is 5.
    non_clonal_str : str, optional
        Value representing unlabelled or non-clonal cells. Default is "NA".
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    return_fig : bool, optional
        Whether to return the figure. Default is False.

    Returns
    -------
    None or matplotlib.figure.Figure or matplotlib.axes.Axes
        Returns Figure if `return_fig` is True, or ax if `return_ax` is True.
    """
    try:
        import mpltern
    except ImportError:
        raise ImportError("`mpltern` is required for nesting_clones. Please install it via pip: `pip install mpltern`.")

    possible_combinations = adata.obs[[
        early_injection, late_injection,
    ]][(adata.obs[[early_injection, late_injection]] != non_clonal_str).sum(axis=1) == 2]
    possible_combinations = list(set([(i, j) for i, j in possible_combinations.values]))

    early_clone_sizes = []
    late_clone_sizes = []
    di_clone_sizes = []

    for early_clone, late_clone in possible_combinations:
        early = sum(adata.obs[early_injection] == early_clone)
        late = sum(adata.obs[late_injection] == late_clone)
        both = sum((adata.obs[late_injection] == late_clone) & (adata.obs[early_injection] == early_clone))
        if (early >= min_clone_size) and (late >= min_clone_size):
            early_clone_sizes.append(early - both)
            late_clone_sizes.append(late - both)
            di_clone_sizes.append(both)

    if figsize is None:
        figsize = (4, 4)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'ternary', 'ternary_sum': 100})
    if show:
        return_ax = False
    else:
        return_ax = True

    ax.set_tlabel("Both barcodes (%)")
    ax.set_llabel("Early barcode (%)")
    ax.set_rlabel("Late barcode (%)")

    # Order = top, left, right
    ax.scatter(
        di_clone_sizes,
        early_clone_sizes,
        late_clone_sizes,
        s=10,
        color="grey",
        edgecolor="k",
        linewidth=0.5,
    )

    ax.grid(alpha=0.3)

    ax.set_tlim(-3, 103)
    ax.set_llim(-3, 103)
    ax.set_rlim(-3, 103)

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()

def volcano(
    pvals: pd.Series | list = None,
    logfcs: pd.Series | list = None,
    names: pd.Series | list | None = None,
    adj_p: bool = False,
    lg_p: bool = True,
    ax: matplotlib.axes.Axes | None = None,
    return_fig: bool = False,
    draw_p_threshold: bool = True,
    pval_threshold: float | None = 0.05,
    draw_logfc_threshold: bool = True,
    logfc_threshold: float | None = 1.0,
    n_highlite: int = 5,
    highlite_names: list | None = None,
    background_scatter_kws: dict | None = None,
    lines_kws: dict | None = None,
    scatters_kws: dict | None = None,
    pval_line_kws: dict | None = None,
    pval_scatter_kws: dict | None = None,
    logfc_scatter_kws: dict | None = None,
    logfc_scatter_kws_left: dict | None = None,
    logfc_scatter_kws_right: dict | None = None,
    logfc_line_kws: dict | None = None,
    logfc_line_kws_left: dict | None = None,
    logfc_line_kws_right: dict | None = None,
    pval_logfc_scatter_kws: dict | None = None,
    pval_logfc_scatter_kws_left: dict | None = None,
    pval_logfc_scatter_kws_right: dict | None = None,
    highlite_scatter_kws: dict | None = None,
    highlite_scatter_kws_right: dict | None = None,
    highlite_scatter_kws_left: dict | None = None,
    text_kws: dict | None = None,
    text_kws_right: dict | None = None,
    text_kws_left: dict | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = "Volcano plot",
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    rasterize: bool = True,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | matplotlib.axes.Axes | None:
    """
    Draw a volcano using provided p-values and logFCs.

    Parameters
    ----------
    pvals : pd.Series | list
        P-values for each gene.
    logfcs : pd.Series | list
        Log2 fold changes for each gene.
    names : pd.Series | list | None, optional
        Gene names corresponding to p-values and logFCs. If None, indices are used (if provided).
        If not provided, no gene names are shown. Default is None.
    adj_p : bool, optional
        Whether p-values are adjusted (used to label the Y-axis). Default is False.
    lg_p : bool, optional
        Whether to use -log10(p) scale. If False, the function expects -log10(p) in pvals.
        Default is True.
    ax : matplotlib.axes.Axes | None, optional
        Axes to draw the volcano on. Default is None.
    return_fig : bool, optional
        Whether to return the figure. Default is False.
    draw_p_threshold : bool, optional
        Whether to draw the p-value threshold line. Default is True.
    pval_threshold : float | None, optional
        P-value threshold for significance. Default is 0.05.
    draw_logfc_threshold : bool, optional
        Whether to draw the logFC threshold lines. Default is True.
    logfc_threshold : float | None, optional
        Log2 fold change threshold for significance. Default is 1.0.
    n_highlite : int, optional
        Number of top genes to highlight from each side. Default is 5.
    highlite_names : list | None, optional
        List of gene names to highlight in addition to top-DE genes. Default is None.
    background_scatter_kws : dict | None, optional
        Keyword arguments for background (p > pval_threshold and |logFC| < logfc_threshold) scatter plot. Default is None.
    lines_kws : dict | None, optional
        Keyword arguments for all lines. Default is None.
    scatters_kws : dict | None, optional
        Keyword arguments for all scatters. Default is None.
    pval_line_kws : dict | None, optional
        Keyword arguments for p-value threshold line. Default is None.
    pval_scatter_kws : dict | None, optional
        Keyword arguments for p-value threshold scatter. Default is None.
    logfc_scatter_kws : dict | None, optional
        Keyword arguments for logFC threshold scatter. Default is None.
    logfc_scatter_kws_left : dict | None, optional
        Keyword arguments for logFC threshold scatter on the left side. Default is None.
    logfc_scatter_kws_right : dict | None, optional
        Keyword arguments for logFC threshold scatter on the right side. Default is None.
    logfc_line_kws : dict | None, optional
        Keyword arguments for logFC threshold line. Default is None.
    logfc_line_kws_left : dict | None, optional
        Keyword arguments for logFC threshold line on the left side. Default is None.
    logfc_line_kws_right : dict | None, optional
        Keyword arguments for logFC threshold line on the right side. Default is None.
    pval_logfc_scatter_kws : dict | None, optional
        Keyword arguments for p-value and logFC threshold scatter. Default is None.
    pval_logfc_scatter_kws_left : dict | None, optional
        Keyword arguments for p-value and logFC threshold scatter on the left side. Default is None.
    pval_logfc_scatter_kws_right : dict | None, optional
        Keyword arguments for p-value and logFC threshold scatter on the right side. Default is None.
    highlite_scatter_kws : dict | None, optional
        Keyword arguments for highlite scatter plot. Default is None.
    highlite_scatter_kws_right : dict | None, optional
        Keyword arguments for highlite scatter plot on the right side. Default is None.
    highlite_scatter_kws_left : dict | None, optional
        Keyword arguments for highlite scatter plot on the left side. Default is None.
    text_kws : dict | None, optional
        Keyword arguments for text. Default is None.
    text_kws_right : dict | None, optional
        Keyword arguments for text on the right side. Default is None.
    text_kws_left : dict | None, optional
        Keyword arguments for text on the left side. Default is None.
    xlim : tuple[float, float] | None, optional
        x-axis limits. Default is None.
    ylim : tuple[float, float] | None, optional
        y-axis limits. Default is None.
    title : str | None, optional
        Title of the plot. Default is None.
    xlabel : str | None, optional
        x-axis label. Default is None.
    ylabel : str | None, optional
        y-axis label. Default is None.
    grid : bool, optional
        Whether to draw grid lines. Default is True.
    rasterize : bool, optional
        Whether to rasterize the plot. Default is True.
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if return_fig is True.
    """
    try:
        import textalloc as ta
    except ImportError:
        raise ImportError("`textalloc` is required for volcano_plot. Please install it via pip: `pip install textalloc`.")

    if lines_kws is None:
        lines_kws = {}
    if scatters_kws is None:
        scatters_kws = {}

    def _fill_defaults(d, **defaults):
        if d is None:
            d = {}
        for key in defaults:
            if key not in d:
                d[key] = defaults[key]
        return d

    if names is None and not (isinstance(pvals, pd.Series) or isinstance(logfcs, pd.Series)):
        logg.warning("If `names` is None, `pvals` and `logfcs` must be pd.Series. Gene names won't be displayed.")
        highlite_names = None
        n_highlite = 0

        df = pd.DataFrame({"log2fc": logfcs, "p": pvals})
    elif names is None and (isinstance(pvals, pd.Series) or isinstance(logfcs, pd.Series)):
        df = pd.DataFrame({"log2fc": logfcs, "p": pvals})
    else:
        df = pd.DataFrame({"log2fc": logfcs, "p": pvals}, index=names)

    df["p"] = np.where(df.p < 1e-300, 1e-300, df.p)

    if figsize is None:
        figsize = (4, 4)

    if ax:
        show = False
        return_ax = False
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    if lg_p:
        df["-log10p"] = -np.log10(df["p"])
        pval_threshold = -np.log10(pval_threshold)
    else:
        df["-log10p"] = df["p"]
    df.dropna(inplace=True)

    # Drawing < logFC and >p-val
    background_scatter_kws = _fill_defaults(background_scatter_kws, **scatters_kws)
    background_scatter_kws = _fill_defaults(
        background_scatter_kws,
        s=3, alpha=1, color="#AAAAAA", edgecolor="none", rasterized=rasterize,
    )
    df_background = df[
        (df["-log10p"] < pval_threshold) &
        (df["log2fc"].abs() < logfc_threshold)
    ]
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df_background,
        ax=ax,
        zorder=1,
        **background_scatter_kws,
    )

    # Drawing < -logFC and >p-val
    logfc_scatter_kws = _fill_defaults(logfc_scatter_kws, **scatters_kws)
    logfc_scatter_kws = _fill_defaults(
        logfc_scatter_kws, s=3, alpha=1, edgecolor="none", rasterized=rasterize,
    )
    logfc_scatter_kws_left = _fill_defaults(
        logfc_scatter_kws_left, **logfc_scatter_kws,
    )
    logfc_scatter_kws_left = _fill_defaults(
        logfc_scatter_kws_left, color="#A7BACC",
    )
    df_nonsigni_left = df[
        (df["-log10p"] < pval_threshold) &
        (df["log2fc"] <= -logfc_threshold)
    ]
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df_nonsigni_left,
        ax=ax,
        zorder=1,
        **logfc_scatter_kws_left,
    )

    # Drawing > logFC and >p-val
    logfc_scatter_kws_right = _fill_defaults(
        logfc_scatter_kws_right, **logfc_scatter_kws,
    )
    logfc_scatter_kws_right = _fill_defaults(
        logfc_scatter_kws_right, color="#CFA7A7",
    )

    df_nonsigni_right = df[
        (df["-log10p"] < pval_threshold) &
        (df["log2fc"] >= logfc_threshold)
    ]
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df_nonsigni_right,
        ax=ax,
        zorder=1,
        **logfc_scatter_kws_right,
    )

    # Drawing || < logFC and <p-val
    pval_scatter_kws = _fill_defaults(pval_scatter_kws, **scatters_kws)
    pval_scatter_kws = _fill_defaults(
        pval_scatter_kws, s=3, alpha=1, edgecolor="none", color="#afb598", rasterized=rasterize,
    )
    df_signi_center = df[
        (df["-log10p"] >= pval_threshold) &
        (df["log2fc"].abs() < logfc_threshold)
    ]
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df_signi_center,
        ax=ax,
        zorder=1,
        **pval_scatter_kws,
    )

    # Drawing > logFC and < p-val
    pval_logfc_scatter_kws = _fill_defaults(pval_logfc_scatter_kws, **scatters_kws)
    pval_logfc_scatter_kws_right = _fill_defaults(
        pval_logfc_scatter_kws_right,
        **pval_logfc_scatter_kws,
    )
    pval_logfc_scatter_kws_right = _fill_defaults(
        pval_logfc_scatter_kws_right, color="#D64545", s=7, rasterized=rasterize,
        edgecolor=None,
    )
    df_signi_right = df[
        (df["-log10p"] >= pval_threshold) &
        (df["log2fc"] >= logfc_threshold)
    ]
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df_signi_right,
        ax=ax,
        zorder=1,
        **pval_logfc_scatter_kws_right,
    )

    # Drawing < -logFC and < p-val
    pval_logfc_scatter_kws_left = _fill_defaults(
        pval_logfc_scatter_kws_left,
        **pval_logfc_scatter_kws,
    )
    pval_logfc_scatter_kws_left = _fill_defaults(
        pval_logfc_scatter_kws_left, color="#4A90E2", s=7, rasterized=rasterize,
        edgecolor=None,
    )
    df_signi_left = df[
        (df["-log10p"] >= pval_threshold) &
        (df["log2fc"] <= -logfc_threshold)
    ]
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df_signi_left,
        ax=ax,
        zorder=1,
        **pval_logfc_scatter_kws_left,
    )
    
    # Highliting genes
    if highlite_names:
        # Checking type
        if isinstance(highlite_names, str):
            highlite_names = [highlite_names]
        if not isinstance(highlite_names, (list, tuple, pd.Series)):
            logg.warning(
                "highlite_names must be a list, tuple, or pandas Series."
            )
            highlite_names = []
    else:
        highlite_names = []
    gene_not_found = []
    for gene in highlite_names:
        if gene not in df.index:
            gene_not_found.append(gene)
    if len(gene_not_found) > 0:
        logg.warning(
            f"Following genes were not found in the dataframe: {gene_not_found}"
        )
    highlite_names = pd.Series(highlite_names)

    left_highlite_names = list(
        df_signi_left.sort_values("-log10p", ascending=False).index[:n_highlite]
    )
    left_highlite_names += list(
        highlite_names[highlite_names.isin(df[df.log2fc < 0].index)]
    )

    left_highlite_names = np.array(left_highlite_names)
    dropped_genes = []
    if xlim:
        if xlim[0]:
            out_of_field = (df.loc[left_highlite_names, "log2fc"] < xlim[0]).values
            if sum(out_of_field) > 0:
                dropped_genes += list(left_highlite_names[out_of_field])
                left_highlite_names = left_highlite_names[~out_of_field]

    if ylim:
        if ylim[0]:
            out_of_field = (df.loc[left_highlite_names, "-log10p"] < ylim[0]).values
            if sum(out_of_field) > 0:
                dropped_genes += list(left_highlite_names[out_of_field])
                left_highlite_names = left_highlite_names[~out_of_field]
        if ylim[1]:
            out_of_field = (df.loc[left_highlite_names, "-log10p"] > ylim[1]).values
            if sum(out_of_field) > 0:
                dropped_genes += list(left_highlite_names[out_of_field])
                left_highlite_names = left_highlite_names[~out_of_field]

    right_highlite_names = list(
        df_signi_right.sort_values("-log10p", ascending=False).index[:n_highlite]
    )
    right_highlite_names += list(
        highlite_names[highlite_names.isin(df[df.log2fc > 0].index)]
    )

    right_highlite_names = np.array(right_highlite_names)
    if xlim:
        if xlim[1]:
            out_of_field = (df.loc[right_highlite_names, "log2fc"] > xlim[1]).values
            if sum(out_of_field) > 0:
                dropped_genes += list(right_highlite_names[out_of_field])
                right_highlite_names = right_highlite_names[~out_of_field]

    if ylim:
        if ylim[0]:
            out_of_field = (df.loc[right_highlite_names, "-log10p"] < ylim[0]).values
            if sum(out_of_field) > 0:
                dropped_genes += list(right_highlite_names[out_of_field])
                right_highlite_names = right_highlite_names[~out_of_field]
        if ylim[1]:
            out_of_field = (df.loc[right_highlite_names, "-log10p"] > ylim[1]).values   
            if sum(out_of_field) > 0:
                dropped_genes += list(right_highlite_names[out_of_field])
                right_highlite_names = right_highlite_names[~out_of_field]

    if len(dropped_genes) > 0:
        logg.warning(
            f"Following genes are out of the plot area: {dropped_genes}"
        )

    highlite_scatter_kws = _fill_defaults(
        highlite_scatter_kws, **scatters_kws
    )
    highlite_scatter_kws_left = _fill_defaults(
        highlite_scatter_kws_left, **highlite_scatter_kws
    )

    increase_s = False
    add_edgecolor = False
    add_linewidth = False
    if "s" not in highlite_scatter_kws_left:
        increase_s = True
    if "edgecolor" not in highlite_scatter_kws_left:
        add_edgecolor = True
    if "linewidth" not in highlite_scatter_kws_left:
        add_linewidth = True

    highlite_scatter_kws_left = _fill_defaults(
        highlite_scatter_kws_left, **pval_logfc_scatter_kws_left
    )
    if increase_s:
        highlite_scatter_kws_left["s"] += 10
    if add_linewidth:
        highlite_scatter_kws_left["linewidth"] = 0.5
    if add_edgecolor:
        highlite_scatter_kws_left["edgecolor"] = "black"

    left_highlite_names = list(set(left_highlite_names))
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df.loc[left_highlite_names],
        ax=ax,
        zorder=1,
        **highlite_scatter_kws_left,
    )

    highlite_scatter_kws_right = _fill_defaults(
        highlite_scatter_kws_right, **highlite_scatter_kws
    )

    increase_s = False
    add_edgecolor = False
    add_linewidth = False
    if "s" not in highlite_scatter_kws_right:
        increase_s = True
    if "edgecolor" not in highlite_scatter_kws_right:
        add_edgecolor = True
    if "linewidth" not in highlite_scatter_kws_right:
        add_linewidth = True

    highlite_scatter_kws_right = _fill_defaults(
        highlite_scatter_kws_right, **pval_logfc_scatter_kws_right
    )
    if increase_s:
        highlite_scatter_kws_right["s"] += 10
    if add_linewidth:
        highlite_scatter_kws_right["linewidth"] = 0.5
    if add_edgecolor:
        highlite_scatter_kws_right["edgecolor"] = "black"

    right_highlite_names = list(set(right_highlite_names))
    sns.scatterplot(
        x="log2fc",
        y="-log10p",
        data=df.loc[right_highlite_names],
        ax=ax,
        zorder=1,
        **highlite_scatter_kws_right,
    )

    if xlim is None:
        xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    if ylim is None:
        ylim = ax.get_ylim()
    ax.set_ylim(ylim)

    xlines = None
    ylines = None

    pval_line_kws = _fill_defaults(pval_line_kws, **lines_kws)
    pval_line_kws = _fill_defaults(
        pval_line_kws,
        linestyle="--", color="black", linewidth=0.5,
    )
    
    if draw_p_threshold and pval_threshold is not None and pval_line_kws["linewidth"] > 0:
        ax.axhline(pval_threshold, zorder=2, **pval_line_kws)
        xlines = [[xlim[0], xlim[1]]]
        ylines = [[pval_threshold, pval_threshold]]

    logfc_line_kws = _fill_defaults(logfc_line_kws, **lines_kws)
    logfc_line_kws = _fill_defaults(
        logfc_line_kws,
        linestyle="--", color="black", linewidth=0.5,
    )
    logfc_line_kws_left = _fill_defaults(
        logfc_line_kws_left, **logfc_line_kws
    )
    logfc_line_kws_right = _fill_defaults(
        logfc_line_kws_right, **logfc_line_kws
    )
    if draw_logfc_threshold and logfc_threshold is not None and logfc_line_kws["linewidth"] > 0:
        ax.axvline(logfc_threshold, zorder=2, **logfc_line_kws_left)
        ax.axvline(-logfc_threshold, zorder=2, **logfc_line_kws_right)

        if xlines is None:
            xlines = []
        xlines += [[logfc_threshold, logfc_threshold], [-logfc_threshold, -logfc_threshold]]

        if ylines is None:
            ylines = []
        ylines += [[ylim[0], ylim[1]], [ylim[0], ylim[1]]]

    # Parse text_kws consistently
    text_kws = _fill_defaults(text_kws)
    text_kws_left = _fill_defaults(text_kws_left, **text_kws)
    text_kws_right = _fill_defaults(text_kws_right, **text_kws)

    text_kws_left = _fill_defaults(
        text_kws_left,
        textsize=10,
        linecolor="k",
        avoid_crossing_label_lines=True,
        avoid_label_lines_overlap=True,
        textcolor=highlite_scatter_kws_left.get("color", "black"),
        fontstyle="italic",
        max_distance=0.3,
        bbox=dict(
            pad=1.5,
            facecolor="white",
            edgecolor="black",
            alpha=0.5,
        ),
    )

    text_kws_right = _fill_defaults(
        text_kws_right,
        textsize=10,
        linecolor="k",
        linewidth=0.5,
        avoid_crossing_label_lines=True,
        avoid_label_lines_overlap=True,
        textcolor=highlite_scatter_kws_right.get("color", "black"),
        fontstyle="italic",
        max_distance=0.3,
        bbox=dict(
            pad=1.5,
            facecolor="white",
            edgecolor="black",
            alpha=0.5,
        ),
    )

    if len(left_highlite_names) > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Series.__getitem__ treating keys as positions is deprecated.*",
                category=FutureWarning,
            )
            ta.allocate(
                ax,
                x=df.loc[left_highlite_names, "log2fc"],
                y=df.loc[left_highlite_names, "-log10p"],
                text_list=left_highlite_names,
                x_scatter=df.log2fc,
                y_scatter=df["-log10p"],
                x_lines=xlines,
                y_lines=ylines,
                **text_kws_left,
            )

    if len(right_highlite_names) > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Series.__getitem__ treating keys as positions is deprecated.*",
                category=FutureWarning,
            )
            ta.allocate(
                ax,
                x=df.loc[right_highlite_names, "log2fc"],
                y=df.loc[right_highlite_names, "-log10p"],
                text_list=right_highlite_names,
                x_scatter=df.log2fc,
                y_scatter=df["-log10p"],
                x_lines=xlines,
                y_lines=ylines,
                **text_kws_right,
            )

    if grid:
        ax.grid(alpha=0.3, zorder=0)
    if xlabel is None:
        xlabel = "log2(Fold Change)"
    ax.set_xlabel(xlabel)
    if ylabel is None:
        if adj_p:
            ylabel = "–log10(adj. p-value)"
        else:
            ylabel = "–log10(p-value)"
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()

def shap_volcano(
    shapdata: AnnData,
    layer: str,
    min_corr: float = 0.5,
    xlabel: str = "corr(SHAP, Expr)",
    ylabel: str = "mean(|SHAP|)",
    title: str | None = None,
    return_fig: bool = False,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> plt.Figure | matplotlib.axes.Axes | None:
    """
    Draws volcano-like plot showing the result of the associations analysis with CatBoost and SHAP.

    Parameters
    ----------
    shapdata : AnnData
        Annotated data matrix with SHAP values in `layers[layer]` and gene expression correlations in `varm["gex_r"][layer]`.
    layer : str
        Layer in `shapdata.layers` with SHAP values.
    min_corr : float, optional
        Minimum correlation threshold for plotting. Default is 0.5.
    xlabel : str, optional
        Label for the x-axis. Default is "corr(SHAP, Expr)".
    ylabel : str, optional
        Label for the y-axis. Default is "mean(|SHAP|)".
    title : str | None, optional
        Title for the plot. Default is None.
    return_fig : bool, optional
        Whether to return the figure object. Default is False.
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    ax : matplotlib.axes.Axes | None, optional
        Axes object to draw on. Default is None.
    **kwargs
        Additional keyword arguments for `volcano()`.
    """

    if title is None:
        title = f"{layer} proportion predictors"
    
    return volcano(
        pvals=np.abs(shapdata.layers[layer]).mean(axis=0).A[0],
        logfcs=shapdata.varm["gex_r"][layer],
        lg_p=False,
        logfc_threshold=min_corr,
        ylabel=ylabel,
        xlabel=xlabel,
        title=title,
        return_fig=return_fig,
        show=show,
        figsize=figsize,
        ax=ax,
        **kwargs,
    )

def barplot(
    data: pd.Series,
    kind: Literal["h", "v"] = "v",
    ax: plt.Axes | None = None,
    cmap: str = "Reds",
    vmax: float | None = None,
    vmin: float | None = 0.,
    return_fig: bool = False,
    show: bool = True,
    title: str | None = None,
    right_spines: bool | None = None,
    left_spines: bool | None = None,
    top_spines: bool | None = None,
    bottom_spines: bool | None = None,
    width: float = 0.8,
    figsize: tuple[float, float] | None = None,
    edgecolor: str = "black",
    rotation: float | None = None,
    arrow: list[Literal["right-bottom", "top-left"]] | Literal["right-bottom", "top-left"] | None = None,
    **kwargs,
) -> plt.Figure | None:
    """
    Draws barplot colored by the value in each bar.

    Parameters
    ----------
    data : pd.Series
        Data to plot.
    kind : Literal["h", "v"], optional
        Kind of plot. Default is "v".
    ax : plt.Axes | None, optional
        Axes object to draw on. Default is None.
    cmap : str, optional
        Colormap to use. Default is "Reds".
    vmax : float | None, optional
        Maximum value for colormap normalization. Default is None.
    vmin : float | None, optional
        Minimum value for colormap normalization. Default is 0.
    return_fig : bool, optional
        Whether to return the figure object. Default is False.
    title : str | None, optional
        Title for the plot. Default is None.
    right_spines : bool | None, optional
        Whether to show right spines. Default is None.
    left_spines : bool | None, optional
        Whether to show left spines. Default is None.
    top_spines : bool | None, optional
        Whether to show top spines. Default is None.
    bottom_spines : bool | None, optional
        Whether to show bottom spines. Default is None.
    width : float, optional
        Width of the bars. Default is 0.8.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    edgecolor : str, optional
        Edge color of the bars. Default is "black".
    rotation : float | None, optional
        Rotation angle for labels, if None rotation depends on the . Default is None.
    arrow : list[Literal["right-bottom", "top-left"]] | Literal["right-bottom", "top-left"] | None, optional
        Whether to add arrow to the plot. Default is None.
    **kwargs
        Additional keyword arguments for `ax.bar()`.

    Returns
    -------
    fig : plt.Figure | None
        Figure object if `return_fig` is True, otherwise None.
    """

    import matplotlib.colors as mcolors

    if kind == "v":
        if figsize is None:
            figsize = (3, 6)
        if right_spines is None:
            right_spines = False
        if left_spines is None:
            left_spines = False
        if top_spines is None:
            top_spines = False
        if bottom_spines is None:
            bottom_spines = True
        if arrow is None:
            arrow = ["right-bottom"]
        if rotation is None:
            rotation = 0
    else:
        if figsize is None:
            figsize = (6, 3)
        if right_spines is None:
            right_spines = False
        if left_spines is None:
            left_spines = True
        if top_spines is None:
            top_spines = False
        if bottom_spines is None:
            bottom_spines = False
        if arrow is None:
            arrow = ["top-left"]
        if rotation is None:
            rotation = 90
    if isinstance(arrow, str):
        arrow = [arrow]
    if vmax is None:
        vmax = data.max()
    if vmin is None:
        vmin = data.min()

    if ax:
        show = False
        return_ax = False
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    if kind == "h":
        data.plot(ax=ax, kind="bar", width=width, edgecolor=edgecolor, **kwargs)
        ax.tick_params(axis="x", labelrotation=rotation)
    else:
        data = data[::-1]
        data.plot(ax=ax, kind="barh", width=width, edgecolor=edgecolor, **kwargs)
        ax.tick_params(axis="y", labelrotation=rotation)

    ax.spines["right"].set_visible(right_spines)
    ax.spines["top"].set_visible(top_spines)
    ax.spines["left"].set_visible(left_spines)
    ax.spines["bottom"].set_visible(bottom_spines)

    if "right-bottom" in arrow:
        ax.plot(
            1, 0, ">k",
            transform=ax.transAxes,
            clip_on=False,
        )

    if "top-left" in arrow:
        ax.plot(
            0, 1, "^k",
            transform=ax.transAxes,
            clip_on=False,
        )
    
    cmap = plt.get_cmap(cmap)
    vmin = vmin
    vmax = vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for patch, value in zip(ax.patches, data.values):
        color = cmap(norm(value))
        patch.set_facecolor(color)

    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()

def predictors_comparison(
    shapdata: sc.AnnData,
    group1: str,
    group2: str,
    varm_key: str | None = None,
    n_genes: int = 10,
    cmap: str = "RdBu_r",
    title: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    **kwargs,
) -> plt.Figure | plt.Axes | None:
    """
    Draws barplot of SHAP values for two groups.

    Parameters
    ----------
    shapdata : sc.AnnData
        Annotated data matrix with SHAP values in `varm`.
    group1 : str
        Name of the first group.
    group2 : str
        Name of the second group.
    varm_key : str | None, optional
        Key in `varm` to SHAP values. Default is None.
    n_genes : int, optional
        Number of top and bottom genes to plot. Default is 10.
    cmap : str, optional
        Colormap to use. Default is "RdBu_r".
    title: str | None, optional
        Title for the plot. Default is None.
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    ax : plt.Axes | None, optional
        Axes object to draw on. Default is None.
    return_fig : bool, optional
        Whether to return the figure object. Default is False.
    **kwargs
        Additional keyword arguments for `ax.bar()`.

    Returns
    -------
    fig : plt.Figure | plt.Axes | None
        Figure object if `return_fig` is True, Axes if `show` is False, otherwise None.
    """
    if varm_key is None:
        try:
            df = shapdata.varm["signed_mean_shap"]
        except KeyError:
            raise KeyError("varm['signed_mean_shap'] not found. Please run c2v.utils.correct_shap() first with flags `normalize=False` and `correct_sign=True`.")
    else:
        logg.warning(f"We expect sign-corrected and not normalized by maximum value SHAPs in varm['{varm_key}'].")
        df = shapdata.varm[varm_key]
    
    diff = (df[group1] - df[group2]).sort_values(ascending=False)
    diff = pd.concat([diff[:n_genes], diff[-n_genes:]])
    vmax = np.abs(diff).max()

    if title is None:
        title = f"{group1} [+] vs. {group2} [–]"

    return barplot(
        diff[::-1],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        left_spines=True,
        bottom_spines=True,
        title=title,
        show=show,
        figsize=figsize,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


def heatmap(
    df: pd.DataFrame,
    cmap: str = "RdBu_r",
    linecolor: str = "black",
    linewidth: float = 0.5,
    borderwidth_hm: float = 2.,
    square: bool = True,
    cbar_scale: float = 0.6,
    cbar_aspect: float = 20.,
    cbar_pad: float = 0.025,
    cmap_border: float = 1.,
    cbar_title: str = "",
    ax: plt.Axes | None = None,
    x_rotation: float = 0.,
    y_rotation: float = 0.,
    xlabels: bool = True,
    ylabels: bool = True,
    sort_columns: bool = True,
    sort_rows: bool = False,
    linkage_method: str = "average",
    linkage_metric: str = "euclidean",
    return_fig: bool = False,
    title: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> plt.Figure | plt.Axes | None:
    """
    Draws pretty heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be visualized.
    cmap : str, optional
        Colormap to be used. Default is "RdBu_r".
    linecolor : str, optional
        Color of the lines between cells. Default is "black".
    linewidth : float, optional
        Width of the lines between cells. Default is 0.5.
    borderwidth_hm : float, optional
        Width of the border around the heatmap. Default is 2.
    square : bool, optional
        Whether to make cells square. Default is True.
    cbar_scale : float, optional
        Scale of the colorbar. Default is 0.6.
    cbar_aspect : float, optional
        Aspect ratio of the colorbar. Default is 20.
    cbar_pad : float, optional
        Padding between the heatmap and the colorbar. Default is 0.025.
    cmap_border : float, optional
        Width of the border around the colormap. Default is 1.
    cbar_title : str, optional
        Title of the colorbar. Default is "".
    ax : plt.Axes | None, optional
        Axes object to draw the heatmap on. Default is None.
    x_rotation : float, optional
        Rotation angle for x-axis labels. Default is 0.
    y_rotation : float, optional
        Rotation angle for y-axis labels. Default is 0.
    xlabels : bool, optional
        Whether to show x-axis labels. Default is True.
    ylabels : bool, optional
        Whether to show y-axis labels. Default is True.
    sort_columns : bool, optional
        Whether to hierarchically sort columns for display. Default is False.
    sort_rows : bool, optional
        Whether to hierarchically sort rows for display. Default is False.
    linkage_method : str, optional
        Linkage method for hierarchical clustering. Default is "average".
    linkage_metric : str, optional
        Linkage metric for hierarchical clustering. Default is "euclidean".
    return_fig : bool, optional
        Whether to return the figure object. Default is False.
    title : str | None, optional
        Title for the plot. Default is None.
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    **kwargs
        Additional keyword arguments for `sns.heatmap()`.

    Returns
    -------
    fig : plt.Figure | plt.Axes | None
        Figure object if `return_fig` is True, Axes if `show` is False, otherwise None.
    """

    import matplotlib.patches as patches

    if figsize is None:
        figsize = (6, 6)

    if ax:
        show = False
        return_ax = False
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    if sort_columns:
        from scipy.cluster.hierarchy import linkage, leaves_list
        Z = linkage(df.T.values, method=linkage_method, metric=linkage_metric)
        order = leaves_list(Z)
        data = df.iloc[:, order]
    else:
        data = df

    if sort_rows:
        Z = linkage(data.values, method=linkage_method, metric=linkage_metric)
        order = leaves_list(Z)
        data = data.iloc[order, :]

    hm = sns.heatmap(
        data,
        cmap=cmap,
        linecolor=linecolor,
        linewidth=linewidth,
        square=square,
        cbar_kws={
            "shrink": cbar_scale,
            "aspect": cbar_aspect,
            "pad": cbar_pad,
            "label": cbar_title,
        },
        ax=ax,
        **kwargs,
    )
    if xlabels:
        ax.tick_params(axis="x", labelrotation=90 - x_rotation)
        for label in ax.get_xticklabels():
            label.set_rotation_mode("anchor")
            label.set_ha("right")
            label.set_va("center")
    else:
        ax.set_xticks([])
    if ylabels:
        ax.tick_params(axis="y", labelrotation=y_rotation)
        for label in ax.get_yticklabels():
            label.set_rotation_mode("anchor")
    else:
        ax.set_yticks([])

    if title:
        ax.set_title(title)

    rect = patches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor="black",
        linewidth=borderwidth_hm,
        zorder=10,
    )
    ax.add_patch(rect)

    cbar = hm.collections[0].colorbar
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(cmap_border)
    cbar.outline.set_edgecolor("black")

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()

def _cmap(
    initial_cmap: str = "Reds",
    grey_intensity: float = 0.2,
    color_intencity: float = 0.1,
) -> ListedColormap:
    """
    Returns color map for visualization of gene expression on UMAPs. Color
    map will starts from grey, not from white.

    Parameters
    ----------
    initial_cmap : str, optional
        What color map will be the base for novel color map.
    grey_intensity : float, optional
        What intensity of grey should be at the start of color map.
    color_intencity : float, optional
        What intensity of color should be after grey at color map

    Returns
    ----------
    ListedColormap object.
    """
    from matplotlib.colors import ListedColormap

    cm_color = plt.get_cmap(initial_cmap, 128)
    cm_grey = plt.get_cmap("Greys", 128)

    return ListedColormap(np.vstack((
        cm_grey(np.linspace(grey_intensity, 0.2, 1)),
        cm_color(np.linspace(color_intencity, 1, 128)),
    )))

def _prepare_c2c(
    adata: sc.AnnData,
    obsm: str,
    clone_col: str | None = None,
    color: str | None = None,
    cmap: str | None = None,
    palette: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Prepares dataframes for visualization of cells or clones in clones2cells.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    obsm : str
        Name of the obsm matrix to use for visualization.
    clone_col : str, optional
        Name of the column in adata.obs that contains clone information.
        If None, cells will be visualized.
    color : str, optional
        Name of the column in adata.obs or adata.raw.obs that contains
        color information. If None, cells will be visualized in grey.
    cmap : str, optional
        Name of the colormap to use for continuous coloring. If None,
        standard scanpy's procedure for categorical coloring is used.
    palette : str, optional
        Name of the palette to use for categorical coloring. If None,
        standard scanpy's procedure for categorical coloring is used.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with columns "x", "y", "clone" (if clone_col is None) or
        "cell", "clone" (if clone_col is not None).
    cmap : str | None
        Name of the colormap to use for visualization. If None, cells will be
        visualized in grey.
    """

    df = pd.DataFrame({
        "x": adata.obsm[obsm][:, 0],
        "y": adata.obsm[obsm][:, 1],
    })

    if clone_col is None:
        df["clone"] = adata.obs_names.values
    else:
        df["cell"] = adata.obs_names.values
        df["clone"] = adata.obs[clone_col].values

    if color:
        if adata.raw and not (color in adata.obs.columns):
            c = adata.raw.obs_vector(color)
        else:
            c = adata.obs_vector(color)
            
        if not pd.api.types.is_numeric_dtype(c):
            c = pd.core.arrays.categorical.Categorical(c)
            if palette is None:
                sc.pl._utils.add_colors_for_categorical_sample_annotation(adata, color)
                colors = adata.uns[f"{color}_colors"]
            else:
                if type(palette) == dict:
                    colors = []
                    for category in c.categories:
                        colors.append(palette[category])
                else:
                    colors = palette
            cmap = dict(zip(c.categories, colors))
            df["c"] = c
        else:
            if cmap is None:
                cmap = "viridis"
            df["c"] = c
    else:
        cmap = None
        
    return df, cmap

def catboost_perfomance(
    shapdata: sc.AnnData,
    var_names: str | list[str] | None = None,
    set_type: Literal["train", "validation"] = "validation",
    ncols: int = 4,
    ax: matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None = None,
    return_fig: bool = False,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    line_kws: dict | list[dict] | None = None,
    scatter_kws: dict | list[dict] | None = None,
    kwargs: dict | list[dict] | None = None,
    title: str | list[str] | None = None,
) -> matplotlib.figure.Figure | matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes] | None:
    """
    Compares predicted by CatBoost and observed values. It might help to evaluate
    performance of CatBoost model.

    Parameters
    ----------
    shapdata : sc.AnnData
        Annotated data matrix.
    var_names : str | list[str], optional
        Names of the variables to visualize. If None, all fates will be
        visualized.
    set_type : Literal["train", "validation"], optional
        Whether to visualize train or validation set.
    ncols : int, optional
        Number of columns in the plot.
    ax : matplotlib.axes.Axes | list[matplotlib.axes.Axes], optional
        Axes to plot on. If None, new figure will be created.
    return_fig : bool, optional
        Whether to return figure object.
    show : bool, optional
        Whether to show the plot. Default is True.
    figsize : tuple[float, float] | None, optional
        Figure size. Default is None.
    line_kws : dict | list[dict], optional
        Keyword arguments for line plot.
    scatter_kws : dict | list[dict], optional
        Keyword arguments for scatter plot.
    kwargs : dict | list[dict], optional
        Keyword arguments for sns.lineplot.
    title : str | list[str], optional
        Title of the plot.
    
    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes] | None
        Figure object if `return_fig` is True, Axes if `show` is False, otherwise None.
    """

    import scipy.sparse as sp
    from .associations import _fast_corr

    if "catboost_info" not in shapdata.uns.keys():
        raise ValueError("catboost_info not found in shapdata.uns. Please, run `c2v.tl.catboost()` first.")

    obsm_key = shapdata.uns["catboost_info"]["obsm_key"]
    fates_used = shapdata.uns["catboost_info"]["fates_used"]
    if var_names is None:
        var_names = list(fates_used)
    elif isinstance(var_names, str):
        var_names = [var_names]

    if set_type == "validation":
        mask = shapdata.obs["validation"].values
    else:
        mask = (~shapdata.obs["validation"]).values

    if line_kws is None:
        line_kws = [{} for _ in range(len(var_names))]
    elif isinstance(line_kws, dict):
        line_kws = [line_kws.copy() for _ in range(len(var_names))]
    if scatter_kws is None:
        scatter_kws = [{} for _ in range(len(var_names))]
    elif isinstance(scatter_kws, dict):
        scatter_kws = [scatter_kws.copy() for _ in range(len(var_names))]
    if kwargs is None:
        kwargs = [{} for _ in range(len(var_names))]
    elif isinstance(kwargs, dict):
        kwargs = [kwargs.copy() for _ in range(len(var_names))]
    if title is None:
        title = [None for _ in range(len(var_names))]
    elif isinstance(title, str):
        title = [title for _ in range(len(var_names))]

    if len(line_kws) != len(var_names):
        raise ValueError("line_kws must have the same length as var_names")
    if len(scatter_kws) != len(var_names):
        raise ValueError("scatter_kws must have the same length as var_names")
    if len(kwargs) != len(var_names):
        raise ValueError("kwargs must have the same length as var_names")
    if len(title) != len(var_names):
        raise ValueError("title must have the same length as var_names")

    ncols = min(len(var_names), ncols)
    nrows = len(var_names) // ncols + (1 if len(var_names) % ncols != 0 else 0)

    if figsize is None:
        figsize = (3.5 * ncols, 3.5 * nrows)

    if ax:
        tight_layout = False
        show = False
        return_ax = False
        fig = None
        axes = ax
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        tight_layout = True
        if show:
            return_ax = False
        else:
            return_ax = True


    if isinstance(axes, matplotlib.axes.Axes):
        axes_grid = np.array([[axes]])
    else:
        axes_arr = np.array(axes, dtype=object)
        if axes_arr.ndim == 0:
            axes_grid = np.array([[axes_arr.item()]])
        elif axes_arr.ndim == 1:
            total = nrows * ncols
            if axes_arr.size < total:
                raise ValueError("Provided axes contain fewer panels than required for the number of groups")
            axes_grid = axes_arr.flat[:total].reshape(nrows, ncols)
        elif axes_arr.ndim >= 2:
            total = nrows * ncols
            if axes_arr.size < total:
                raise ValueError("Provided axes contain fewer panels than required for the number of groups")
            axes_grid = axes_arr.flat[:total].reshape(nrows, ncols)
        else:
            raise TypeError("ax must be an Axes, a list/array of Axes, or None")

    subset = shapdata[mask]

    for i, v in enumerate(var_names):
        col = i % ncols
        row = i // ncols
        ax_i = axes_grid[row, col]

        if isinstance(subset.obsm[obsm_key], pd.DataFrame):
            x = subset.obsm[obsm_key][v].values
        elif sp.issparse(subset.obsm[obsm_key]):
            idx = fates_used.index(v)
            x = subset.obsm[obsm_key][:, idx].toarray().ravel()
        else:
            idx = fates_used.index(v)
            x = subset.obsm[obsm_key][:, idx]

        pred_key = f"{obsm_key}:predicted"
        if isinstance(subset.obsm[pred_key], pd.DataFrame):
            y = subset.obsm[pred_key][v].values
        elif sp.issparse(subset.obsm[pred_key]):
            idx = fates_used.index(v)
            y = subset.obsm[pred_key][:, idx].toarray().ravel()
        else:
            idx = fates_used.index(v)
            y = subset.obsm[pred_key][:, idx]

        corr = _fast_corr(x, y, slope=True, significance=False, progress_bar=False)["r"][0][0]

        current_line_kws = line_kws[i]
        current_scatter_kws = scatter_kws[i]

        if "color" not in current_line_kws.keys():
            current_line_kws["color"] = sns.color_palette()[3]
        if "color" not in current_scatter_kws.keys():
            current_scatter_kws["color"] = "black"
        if "alpha" not in current_scatter_kws.keys():
            current_scatter_kws["alpha"] = 0.8
        if "s" not in current_scatter_kws.keys():
            current_scatter_kws["s"] = 15

        sns.regplot(
            x=x,
            y=y,
            ax=ax_i,
            scatter_kws=current_scatter_kws,
            line_kws=current_line_kws,
            **kwargs[i],
        )
        line = ax_i.get_lines()[-1]
        line.set_label(f"r = {corr:.3f}")
        ax_i.legend(loc="best")
        if title[i] is None:
            ax_i.set_title(v)
        else:
            ax_i.set_title(title[i])
        ax_i.grid(alpha=0.3)
        ax_i.set_xlabel("Observed")
        ax_i.set_ylabel("Predicted")

    for i in range(len(var_names), ncols * nrows):
        axes_grid[i // ncols, i % ncols].axis("off")

    if tight_layout:
        fig.tight_layout()

    if return_fig:
        return fig
    elif return_ax:
        return axes_grid.flat[:len(var_names)]
    elif show:
        plt.show()


def clones2cells(
    adata: sc.AnnData,
    clones: sc.AnnData,
    obs_name: str | None = None,
    cells_color: str | None = None,
    clones_color: str | None = None,
    mode: Literal["highlite", "filter"] = "highlite",
    cells_basis: str = "X_umap",
    clones_basis: str = "X_umap",
    keep_color: bool = False,
    bg_color: str = "lightgrey",
    selected_color: str = "red",
    cells_cmap: str | None = None,
    clones_cmap: str | None = None,
    cells_palette: str | None = None,
    clones_palette: str | None = None,
    s_cells_bg: int = 1,
    s_cells_selected: int = 2,
    s_clones: int = 6,
    width: float = "auto",
    height: float = 400,
    axes: bool = False,
    norm_clones: tuple[float, float] | None = None,
    norm_cells: tuple[float, float] | None = None,
) -> None:
    """
    Draws two interactive sctterplots with gene expression (left) and clonal (right) embeddings.
    Selection on the right plot leads to the highliting (`mode='highlite'`) or filtering (`mode='filter'`)
    of cells from the corresponding clones. Selection on the left plot leads to coloring the right plot
    by the proportion of selected cells in clones.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix with gene expression.
    clones : sc.AnnData
        Annotated data matrix with clonal embeddings.
    obs_name : str, optional
        Name of column in `clones.obs` with clone labels in gene expression AnnData.
        If not provided, will be taken from `clones.uns["obs_name"]`.
    cells_color : str, optional
        Name of column in `adata.obs` with cells colors. If not provided,
        cells will be colored in `bg_color`.
    clones_color : str, optional
        Name of column in `clones.obs` with clones colors. If not provided,
        clones will be colored in `bg_color`.
    mode : {"highlite", "filter"}, optional
        Whether to highlite (`mode='highlite'`) or filter (`mode='filter'`) cells
        from the selected clones.
    cells_basis : str, optional
        Name of key in `adata.obsm` with cells embeddings.
    clones_basis : str, optional
        Name of key in `clones.obsm` with clonal embeddings.
    keep_color : bool, optional
        Whether to keep color of cells in `cells_color` for `mode='highlite'`.
    bg_color : str, optional
        Color of unselected cells or clones.
    selected_color : str, optional
        Color of selected cells if `mode='highlite'` and `keep_color=False`.
    cells_cmap : str, optional
        Continuous color map for cells.
    clones_cmap : str, optional
        Continuous color map for clones.
    cells_palette : str, optional
        Palette (discrete colormap) for cells. If None, standard scanpy's procedure
        for categorical coloring is used.
    clones_palette : str, optional
        Palette (discrete colormap) for clones. If None, standard scanpy's procedure
        for categorical coloring is used.
    s_cells_bg : int, optional
        Size of unselected cells.
    s_cells_selected : int, optional
        Size of selected cells.
    s_clones : int, optional
        Size of clones.
    width : float, optional
        Width of widget.
    height : float, optional
        Height of widget.
    axes : bool, optional
        Whether to show axes.
    norm_clones : tuple[float, float], optional
        Range of values for clonal embeddings normalization.
    norm_cells : tuple[float, float], optional
        Range of values for gene expression embeddings normalization.

    Returns
    -------
    None
    """
    try:
        import ipywidgets as widgets
        import jscatter as js
    except ImportError:
        raise ImportError("`ipywidgets` and `jscatter` are required for clones2cells. Please install them via pip: `pip install ipywidgets jscatter`.")

    logg.warning("Widget might be displayed incorrectly in Safari browser")
    
    if obs_name is None:
        obs_name = clones.uns["obs_name"]
        
    df_gex, cmap_gex = _prepare_c2c(
        adata=adata,
        obsm=cells_basis,
        clone_col=obs_name,
        color=cells_color,
        cmap=cells_cmap,
        palette=cells_palette,
    )
    
    if cmap_gex:
        df_gex = df_gex.sort_values("c")

    df_clones, cmap_clones = _prepare_c2c(
        adata=clones,
        obsm=clones_basis,
        color=clones_color,
        cmap=clones_cmap,
        palette=clones_palette,
    )
    if cmap_clones:
        df_clones = df_clones.sort_values("c")

    if cmap_gex and (type(cmap_gex) != dict) and (mode == "highlite"):
        if keep_color:
            logg.warning("`keep_color=True` isn't supported in `mode='highlite'`. Switching to `keep_color=False`")
            keep_color = False
        cat_gex = False
    else:
        cat_gex = True

    if (mode == "highlite") and (cmap_gex is None):
        keep_color = False

    clones_whitelist = df_clones["clone"].values
    clone_sizes = df_gex["clone"].value_counts()[clones_whitelist]

    if mode == "highlite":
        in_clones = df_gex.clone.isin(clones_whitelist).values
        
        bg_mask = np.array([True] * len(df_gex) + [False] * sum(in_clones))
        bg_idx = np.argwhere(bg_mask).T[0]

        clone_mask = ~bg_mask
        clone_idx = np.argwhere(clone_mask).T[0]
        
        df_gex = pd.concat([df_gex, df_gex[in_clones]]).reset_index(drop=True)
        
        if keep_color:
            df_gex["c_select"] = df_gex["c"].astype(str).copy()
            df_gex.loc[bg_mask, "c_select"] = "Unselected cells"
            df_gex["c_select"] = df_gex["c_select"].astype("category")
            df_gex["c_select"] = df_gex["c_select"].cat.reorder_categories(
                list(df_gex["c"].cat.categories) + ["Unselected cells"]
            )
            
            cmap_select = cmap_gex.copy()
            cmap_select["Unselected cells"] = bg_color

            df_gex["s_select"] = "Cells in clones"
            df_gex.loc[bg_mask, "s_select"] = "Unselected cells"
            df_gex["s_select"] = df_gex["s_select"].astype("category")

            size_select = {
                "Unselected cells": s_cells_bg,
                "Cells in clones": s_cells_selected,
            }
        else:
            df_gex["c_select"] = "Cells in clones"
            df_gex.loc[bg_mask, "c_select"] = "Unselected cells"
            df_gex["c_select"] = df_gex["c_select"].astype("category")
            
            cmap_select = {
                "Unselected cells": bg_color,
                "Cells in clones": selected_color,
            }
            
            df_gex["s_select"] = "Cells in clones"
            df_gex.loc[bg_mask, "s_select"] = "Unselected cells"
            df_gex["s_select"] = df_gex["s_select"].astype("category")

            size_select = {
                "Unselected cells": s_cells_bg,
                "Cells in clones": s_cells_selected,
            }
    else:
        bg_idx = None

    if cmap_gex is None:
        df_gex["c"] = "Unselected cells"
        cmap_gex = {"Unselected cells": bg_color}
        gex_legend = False
    else:
        gex_legend = True
        
    scatter_gex = js.Scatter(x="x", y="y", width=width, height=height, opacity=1,
                             size=s_cells_bg, data=df_gex, axes=axes)

    scatter_gex.color(by="c", map=cmap_gex, labeling={"variable": None}, norm=norm_cells)
    scatter_gex.filter(bg_idx)
    scatter_gex.size(s_cells_bg)
    scatter_gex.legend(legend=gex_legend)

    if cmap_clones is None:
        df_clones["c"] = "Unselected"
        cmap_clones = {"Unselected": bg_color}
        clones_legend = False
    else:
        clones_legend = True

    if pd.api.types.is_numeric_dtype(df_clones["c"]):
        if norm_clones is None:
            norm_clones = (df_clones["c"].min(), df_clones["c"].max())

    scatter_c2v = js.Scatter(x="x", y="y", width=width, height=height, opacity=1,
                             size=s_clones, data=df_clones, axes=axes)

    scatter_c2v.color(by="c", map=cmap_clones, labeling={"variable": None}, norm=norm_clones)
    scatter_c2v.legend(clones_legend)

    def c2v_selection(change):
        if len(change.new) == 0:
            scatter_gex.color(by="c", map=cmap_gex, labeling={"variable": None}, norm=norm_cells)
            scatter_gex.filter(bg_idx)
            scatter_gex.size(s_cells_bg)
            scatter_gex.legend(gex_legend)
        else:
            clones.uns["clones2cells"] = {"last_selection": df_clones.clone.values[change.new]}
            if mode == "filter":
                selected_clones = df_clones.clone.values[change.new]
                selected_cells = np.argwhere(df_gex.clone.isin(selected_clones)).T[0]
                scatter_gex.filter(selected_cells)
                scatter_gex.size(s_cells_selected)
            else:
                scatter_gex.legend(False)
                selected_clones = df_clones.clone.values[change.new]
                selected_cells = np.argwhere(
                    df_gex.clone.isin(selected_clones) | bg_mask
                ).T[0]
                
                scatter_gex.color(by="c_select", map=cmap_select, norm=norm_cells)
                scatter_gex.size(by="s_select", map=size_select)
                scatter_gex.filter(selected_cells)

    scatter_c2v.widget.observe(c2v_selection, names=["selection"])
    
    if clones_cmap is None:
        clones_cmap = "viridis"

    def gex_selection(change):
        if len(change.new) == 0:
            scatter_c2v.color(by="c", map=cmap_clones, labeling={"variable": None}, norm=norm_clones)
            scatter_c2v.legend(clones_legend)
        else:
            selected_cell_bcs = df_gex.cell.values[change.new]
            selected_cell_bcs = np.array(list(set(selected_cell_bcs)))
            adata.uns["clones2cells"] = {"last_selection": selected_cell_bcs}
            if mode == "filter":
                df_clones["proportions_selected"] = (
                    df_gex.iloc[change.new].clone.value_counts()[clones_whitelist] /
                    clone_sizes
                )[df_clones.clone.values] * 100
                scatter_c2v.color(
                    by="proportions_selected",
                    map=clones_cmap,
                    norm=(0, df_clones["proportions_selected"].max()),
                    labeling={"variable": "% of selected"}
                )
                scatter_c2v.legend(legend=True)
            else:
                selected_mask = np.array([False] * len(df_gex))
                selected_mask[change.new] = True
                
                df_clones["proportions_selected"] = (
                    df_gex
                        .iloc[selected_mask]
                        .clone.value_counts()[clones_whitelist] /
                    clone_sizes
                )[df_clones.clone].values.astype(float) * 100
                
                scatter_c2v.color(
                    by="proportions_selected",
                    map=clones_cmap,
                    norm=(0, df_clones["proportions_selected"].max()),
                    labeling={"variable": "% of selected"}
                )
                scatter_c2v.legend(legend=True)
    
    scatter_gex.widget.observe(gex_selection, names=["selection"])

    display(js.compose([
        (scatter_gex, "Gene expression embedding"),
        (scatter_c2v, "Clonal embedding"),
    ], row_height=height))

def small_cbar(
    axes: matplotlib.axes.Axes | list[matplotlib.axes.Axes],
    width: float = 0.03,
    height: float = 0.20,
    fontsize: int = 10,
    ticks_round: int = 2,
    x_pad: float = 0.02,
    y_pad: float = 0,
    show_numbers: bool = True,
):
    """
    Adds small colorbars to the right of the specified axes.
    
    Parameters:
    -----------
    axes : matplotlib.axes.Axes | list[matplotlib.axes.Axes]
        The axes object(s) to draw on (e.g., fig.axes[0]).
    width : float, optional
        Width of the colorbar. Default 3%.
    height : float, optional
        Height of each colorbar. Default 20%.
    fontsize : int, optional
        Font size for tick labels. Default 10.
    ticks_round : int, optional
        Number of decimal places to round tick labels. Default 2.
    x_pad : float, optional
        Padding from the right edge (0.0 to 1.0). Default 2%.
    y_pad : float, optional
        Padding from the top edge (0.0 to 1.0). Default 0%.
    show_numbers : bool, optional
        Whether to show numeric tick labels. Default True.
    """
    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax in axes:
        mappables = [m for m in ax.collections if m.colorbar is not None]
        if not mappables:
            continue

        for i, m in enumerate(mappables):
            m.colorbar.remove()

            x_pos = 1.0 - width + x_pad
            y_pos = y_pad + (i * (height + 0.05))
            
            cax = ax.inset_axes([x_pos, y_pos, width, height], transform=ax.transAxes)
            cbar = plt.colorbar(m, cax=cax, orientation="vertical")
            
            vmin, vmax = m.get_clim()

            ticks = [vmin, vmax]
            cbar.set_ticks(ticks)
                
            if not show_numbers:
                cbar.ax.set_yticklabels(["Min", "Max"])
            elif ticks_round is not None:
                cbar.ax.set_yticklabels(
                    [f"{np.round(x, ticks_round)}" for x in ticks], 
                    fontsize=fontsize
                )
            
            cbar.outline.set_edgecolor("black")
            cbar.outline.set_linewidth(1)

def embedding_axis(
    axes: matplotlib.axes.Axes | list[matplotlib.axes.Axes],
    x_offset: float = 0.,
    y_offset: float = 0., 
    length: float = 0.15, 
    arrow_width: float = 0.0001,
    head_width: float = 0.02,
    color: str = "black",
    label: str = "UMAP",
    label1: str | None = None,
    label2: str | None = None,
    font_size: int = 10.
):
    """
    Adds two perpendicular arrows (L-shape) to the bottom-left corner of the plot
    to mimic standard single-cell embedding visualizations (Seurat/Scanpy).
    
    Parameters:
    -----------
    axes : matplotlib.axes.Axes | list[matplotlib.axes.Axes]
        The axes object to draw on (e.g., fig.axes[0]).
    x_offset : float, optional
        Distance from left edge (0.0 to 1.0). Default 5%.
    y_offset : float, optional
        Distance from bottom edge (0.0 to 1.0). Default 5%.
    length : float, optional
        Length of the arrows as a fraction of the axes size. Default 15%.
    arrow_width : float, optional
        Width of the arrow shaft. Default 0.0001.
    head_width : float, optional
        Width of the arrow head. Default 0.02.
    color : str, optional
        Color of arrows and text. Default "black".
    label : str, optional
        Label for the horizontal and vertical arrows. Default "UMAP".
    label1 : str, optional
        Label for the horizontal arrow. If None, label + " 1" is used. Default None.
    label2 : str, optional
        Label for the vertical arrow. If None, label + " 2" is used. Default None.
    font_size : int, optional
        Size of the text labels. Default 10.
    """
    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax in axes:
        ax.set_axis_off()
        
        arrow_props = {
            "fc": color,
            "ec": color,
            "width": arrow_width,
            "head_width": head_width,
            "head_length": head_width,
            "transform": ax.transAxes, 
            "length_includes_head": True,
            "overhang": 0.2,
            "clip_on": False,
        }
        
        ax.arrow(x_offset, y_offset, length, 0, **arrow_props)
        ax.arrow(x_offset, y_offset, 0, length, **arrow_props)
        
        text_offset = 0.02
        if label1 is None:
            label1 = label + " 1"
        ax.text(
            x_offset,
            y_offset - (text_offset * 1.5),
            label1,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=font_size,
            color=color,
        )
        
        if label2 is None:
            label2 = label + " 2"
        ax.text(
            x_offset - (text_offset * 1.5),
            y_offset, 
            label2,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            rotation=90,
            fontsize=font_size,
            color=color,
        )

def fancy_legend(
    axes,
    textsize: int = 10,
    fontweight: str = "normal",
    center_loc: bool = False,
    line: bool = False,
    max_distance: float = 0.2,
    **kwargs,
):
    """
    Adds a legend to the plot with cluster labels aligned to the cluster centers.
    
    Parameters:
    -----------
    axes : matplotlib.axes.Axes | list[matplotlib.axes.Axes]
        The axes object to draw on (e.g., fig.axes[0]).
    textsize : int, optional
        Size of the text labels. Default 10.
    fontweight : str, optional
        Font weight of the text labels. Default "normal".
    **kwargs : dict, optional
        Additional keyword arguments passed to textalloc.alloc().
    """

    import textalloc as ta
    import matplotlib.colors as mcolors

    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax in axes:
        legend = ax.get_legend()
        handles = legend.legend_handles
        texts = legend.get_texts()

        legend_map = {}
        for handle, text in zip(handles, texts):
            c = handle.get_facecolor()[0]
            c_hex = mcolors.to_hex(c)
            legend_map[c_hex] = text.get_text()

        cluster_points = {label: [] for label in legend_map.values()}
        cluster_colors = {label: color for color, label in legend_map.items()}

        all_x_points = []
        all_y_points = []

        for collection in ax.collections:
            offsets = collection.get_offsets()
            facecolors = collection.get_facecolors()
            
            if len(offsets) > 0:
                all_x_points.extend(offsets[:, 0])
                all_y_points.extend(offsets[:, 1])
            
            if len(facecolors) == 1:
                c_hex = mcolors.to_hex(facecolors[0])
                if c_hex in legend_map:
                    label = legend_map[c_hex]
                    cluster_points[label].extend(offsets)
            elif len(facecolors) == len(offsets):
                for i, point in enumerate(offsets):
                    c_hex = mcolors.to_hex(facecolors[i])
                    if c_hex in legend_map:
                        label = legend_map[c_hex]
                        cluster_points[label].append(point)

        cluster_centers = {}
        for label, points in cluster_points.items():
            if len(points) > 0:
                pts = np.array(points)
                cluster_centers[label] = np.median(pts, axis=0)

        legend.remove()

        labels = []
        x_pos = []
        y_pos = []
        colors = []
        
        for label, center in cluster_centers.items():
            labels.append(label)
            x_pos.append(center[0])
            y_pos.append(center[1])
            colors.append(cluster_colors[label])

        if not center_loc:
            _, _, texts, _ = ta.allocate(
                ax=ax,
                textsize=textsize,
                fontweight=fontweight,
                x=np.array(x_pos),
                y=np.array(y_pos),
                text_list=labels,
                x_scatter=np.array(all_x_points),
                y_scatter=np.array(all_y_points),
                draw_lines=line,
                linewidth=0.5,
                linecolor=colors,
                draw_all=True,
                max_distance=max_distance,
                textcolor=colors,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "pad": 3,
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.85,
                    "lw": 0.5,
                },
                **kwargs,
            )
        else:
            texts = []
            for lbl, x, y, c in zip(labels, x_pos, y_pos, colors):
                t = ax.text(
                    x,
                    y,
                    lbl,
                    fontsize=textsize,
                    fontweight=fontweight,
                    color=c,
                    ha="center",
                    va="center_baseline",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "pad": 3,
                        "facecolor": "white",
                        "edgecolor": "black",
                        "alpha": 0.85,
                        "lw": 0.5,
                    },
                )
                texts.append(t)

        for text_artist, color in zip(texts, colors):
            bbox = text_artist.get_bbox_patch()
            bbox.set_edgecolor(color)
            text_artist.set_verticalalignment("center_baseline")

def scaled_dotplot(
    adata: sc.AnnData,
    groupby: str,
    var_names: list[str],
    cmap: str = "RdBu_r",
    vmin: float = -2.5,
    vmax: float = 2.5,
    colorbar_title: str = "Mean Z-score",
    max_value: float = 5.,
    **kwargs
):
    """
    Function plots dotplot colored by scaled expression values. The problem with
    sc.pl.dotplot is that dot_size is defined by amount of the expression values
    more or equal than 0, and for scaled expressions it's not the correct case.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    groupby : str
        Key for observations grouping.
    var_names : list[str]
        List of genes to plot.
    cmap : str
        Colormap to use. Default is "RdBu_r".
    vmin : float
        Minimum value for color scaling. Default is -2.5.
    vmax : float
        Maximum value for color scaling. Default is 2.5.
    colorbar_title : str
        Title for the colorbar. Default is "Mean Z-score".
    max_value : float
        Maximum value for scaling expression values. Default is 5.
    **kwargs
        Additional keyword arguments to pass to sc.pl.dotplot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the dotplot.
    """

    if adata.raw:
        adata_scaled = adata.raw[:, var_names].to_adata().copy()
    else:
        adata_scaled = adata[:, var_names].copy()

    sc.pp.scale(adata_scaled, max_value=max_value)
    color_df = sc.pl.DotPlot(
        adata_scaled,
        var_names=var_names,
        groupby=groupby,
    ).dot_color_df

    return sc.pl.dotplot(
        adata,
        groupby=groupby,
        var_names=var_names,
        dot_color_df=color_df,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        colorbar_title=colorbar_title,
        show=False,
        **kwargs,
    )

def scatter2vars(
    adata: sc.AnnData,
    var1: str,
    var2: str,
    clip1: tuple[float, float] | None = None,
    clip2: tuple[float, float] | None = None,
    c1: Literal["red", "green", "blue"] = "red",
    c2: Literal["red", "green", "blue"] = "blue",
    basis: str = "X_umap",
    use_raw: bool | None = None,
    components: str | None = None,
    s: float | None = None,
    alpha: float = 1.,
    frameon: bool = True,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    rasterize: bool = False,
    ticks: bool = False,
    round_ticks: int = 2,
    bright_level: float = 0.8,
    grey_level: float = 0.9,
    show: bool = True,
    return_fig: bool = False,
    cbar: bool = True,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.axes.Axes | matplotlib.figure.Figure | None:
    """
    Plots embedding colored by two continious variables.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    var1 : str
        Key for variable 1.
    var2 : str
        Key for variable 2.
    clip1 : tuple[float, float] | None
        Clip values for variable 1. Default is None.
    clip2 : tuple[float, float] | None
        Clip values for variable 2. Default is None.
    c1 : Literal["red", "green", "blue"]
        Color for variable 1. Default is "red".
    c2 : Literal["red", "green", "blue"]
        Color for variable 2. Default is "blue".
    basis : str
        Key for embedding. Default is "X_umap".
    use_raw : bool | None
        Whether to use raw data. Default is None.
    components : str | None
        Components to use. Default is None.
    s : float | None
        Size of points. Default is None.
    alpha : float 
        Transparency of points. Default is 1.
    frameon : bool
        Whether to draw frame. Default is True.
    title : str | None
        Title for the plot. Default is None.
    ax : matplotlib.axes.Axes | None
        Axes object to draw on. Default is None.
    rasterize : bool
        Whether to rasterize points. Default is False.
    ticks : bool
        Whether to draw ticks. Default is False.
    round_ticks : int
        Number of decimal places to round ticks. Default is 2.
    bright_level : float
        Brightness level for points. Default is 0.8.
    grey_level : float
        Grey level for points. Default is 0.9.
    show : bool
        Whether to show the plot. Default is True.
    return_fig : bool
        Whether to return figure object. Default is False.
    cbar : bool
        Whether to draw colorbar. Default is True.
    figsize : tuple[float, float] | None
        Figure size. Default is None.

    Returns
    -------
    matplotlib.axes.Axes | matplotlib.figure.Figure | None
        Axes object with the plot or figure if return_fig is True. None if show is False.
    """

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    if components is None:
        components = "0,1"
    components = np.array(components.split(",")).astype(int)

    if s is None:
        s = 60000 / len(adata)

    if use_raw is None:
        if adata.raw:
            use_raw = True
        else:
            use_raw = False

    expressions = sc.get.obs_df(adata, keys=[var1, var2], use_raw=use_raw)
    
    if clip1 is None:
        clip1 = (expressions[var1].min(), expressions[var1].max())
    else:
        clip1 = np.array(clip1)
        if clip1[0] is None:
            clip1[0] = expressions[var1].min()
        if clip1[1] is None:
            clip1[1] = expressions[var1].max()
        clip1 = tuple(clip1)
    
    if clip2 is None:
        clip2 = (expressions[var2].min(), expressions[var2].max())
    else:
        clip2 = np.array(clip2)
        if clip2[0] is None:
            clip2[0] = expressions[var2].min()
        if clip2[1] is None:
            clip2[1] = expressions[var2].max()
        clip2 = tuple(clip2)

    expressions[var1] = np.clip(expressions[var1], *clip1)
    expressions[var2] = np.clip(expressions[var2], *clip2)
    expressions[var1] = (expressions[var1] - clip1[0]) / (clip1[1] - clip1[0])
    expressions[var2] = (expressions[var2] - clip2[0]) / (clip2[1] - clip2[0])

    order = expressions.sum(axis=1).sort_values().index
    expressions = expressions.loc[order]
    emb = adata[order].obsm[basis][:, components]

    if c1 not in ["red", "blue", "green"]:
        logg.warning(f"Color {c1} is not supported. Backing up to red.")
        c1 = "red"
    if c2 not in ["red", "blue", "green"]:
        logg.warning(f"Color {c2} is not supported. Backing up to blue.")
        c2 = "blue"
    if c1 == c2:
        logg.warning("Colors for var1 and var2 are the same. Backing up to red and blue.")
        c1 = "red"
        c2 = "blue"

    color_basis = {
        "red": np.array([1, 0, 0]),
        "green": np.array([0, 0.8, 0]),
        "blue": np.array([0, 0, 1]),
    }

    c_00 = np.array([grey_level, grey_level, grey_level])
    c_10 = bright_level * color_basis[c1]
    c_01 = bright_level * color_basis[c2]
    c_11 = bright_level / 2 * (color_basis[c1] + color_basis[c2])

    a = np.asarray(expressions[[var1]])
    b = np.asarray(expressions[[var2]])

    cell_colors = (
        (1 - a) * (1 - b) * c_00 +
        a * (1 - b) * c_10 +
        (1 - a) * b * c_01 +
        a * b * c_11
    )
    cell_colors = np.clip(cell_colors, 0, 1)

    if figsize is None:
        figsize = (4, 4)

    if ax:
        show = False
        return_ax = False
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if show:
            return_ax = False
        else:
            return_ax = True

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=cell_colors,
        s=s,
        alpha=alpha,
        edgecolor="None",
        rasterized=rasterize,
    )

    if basis == "X_umap":
        label = "UMAP"
    elif basis == "X_tsne":
        label = "t-SNE"
    elif basis == "X_pca":
        label = "PCA"
    elif basis == "X_draw_graph_fa":
        label = "ForceAtlas"
    elif basis == "X_pca_harmony":
        label = "Harmony"
    else:
        label = basis

    if frameon:
        ax.set_xlabel(f"{label} {components[0]}")
        ax.set_ylabel(f"{label} {components[1]}")
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ticks = False

    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is None:
        title = f"{var1} vs. {var2}"

    ax.set_title(title)

    # Inserting a legend
    if cbar:
        res = 100
        
        l_a = np.linspace(0, 1, res)
        l_b = np.linspace(0, 1, res)
        L_A, L_B = np.meshgrid(l_a, l_b)
        
        L_A_flat = L_A.flatten()[:, None]
        L_B_flat = L_B.flatten()[:, None]
        
        legend_colors = (
            (1 - L_A_flat) * (1 - L_B_flat) * c_00 +
            L_A_flat * (1 - L_B_flat) * c_10 +
            (1 - L_A_flat) * L_B_flat * c_01 +
            L_A_flat * L_B_flat * c_11
        ).reshape(res, res, 3)
        
        axins = inset_axes(
            ax,
            width="20%", 
            height="20%",
            loc="upper left",
            bbox_to_anchor=(1.08, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        
        axins.imshow(legend_colors, origin="lower", extent=[0, 1, 0, 1])
        
        axins.set_xlabel(var1, fontsize=9)
        axins.set_ylabel(var2, fontsize=9)

        axins.xaxis.tick_top()
        axins.yaxis.tick_right()
        
        axins.set_xticks([0, 0.5, 1])
        axins.set_yticks([0, 0.5, 1])

        ticks1 = []
        for _ in [clip1[0], (clip1[1] - clip1[0]) / 2, clip1[1]]:
            ticks1.append(np.round(_, round_ticks))
        axins.set_xticklabels(ticks1, fontsize=7)
        
        ticks2 = []
        for _ in [clip2[0], (clip2[1] - clip2[0]) / 2, clip2[1]]:
            ticks2.append(np.round(_, round_ticks))
        axins.set_yticklabels(ticks2, fontsize=7)

    if return_fig:
        return fig
    elif return_ax:
        return ax
    elif show:
        plt.show()


Reds = _cmap("Reds")
Blues = _cmap("Blues")
Greens = _cmap("Greens")
Purples = _cmap("Purples")
Oranges = _cmap("Oranges")
