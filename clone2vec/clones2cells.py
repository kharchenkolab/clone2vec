from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import ipywidgets

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import colormaps
from matplotlib import colors as mcolors

logg = sc.logging

PaletteLike = str | Sequence[str] | Mapping[str, str] | None
NormLike = tuple[float | str | None, float | str | None] | None

_N_BINS = 256  # continuous values are drawn with this many colour levels
_NA = "NA"
_FONT = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
    "font-size:12px;color:#333;line-height:1.35;padding:2px 8px 6px 8px;"
)

@dataclass
class _Coloring:
    """Colour of every observation as a small integer class + one colour per class."""

    kind: Literal["none", "categorical", "continuous"]
    codes: np.ndarray  # colour class per observation (AnnData row order)
    colors: list[str]  # hex colour per class
    labels: list[str] | None  # category names (categorical)
    label: str | None  # name of what is coloured
    values: np.ndarray | None  # original values, for tooltips
    norm: tuple[float, float] | None = None
    clipped: tuple[bool, bool] = (False, False)
    order: np.ndarray | None = None  # draw order, bottom -> top; None = data order

    def legend_html(self, extra: list[tuple[str, str]] | None = None, note: str | None = None) -> str:
        parts = []
        if self.kind == "continuous":
            parts.append(_colorbar_html(self.label, self.colors, *self.norm, self.clipped))
        elif self.kind == "categorical":
            parts.append(_title_html(self.label) + _swatches_html(list(zip(self.labels, self.colors))))
        if extra:
            parts.append(_swatches_html(extra))
        if note:
            parts.append(f"<div style='color:#888;font-size:11px;margin-top:2px'>{escape(note)}</div>")
        return f"<div style=\"{_FONT}\">{''.join(parts)}</div>" if parts else ""


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def _title_html(title: str | None) -> str:
    return f"<div style='font-weight:600;margin-bottom:3px'>{escape(str(title))}</div>" if title else ""


def _colorbar_html(title, colors, vmin, vmax, clipped=(False, False), width=220) -> str:
    stops = ", ".join(colors[i] for i in np.linspace(0, len(colors) - 1, 24).round().astype(int))
    lo = ("≤ " if clipped[0] else "") + _fmt(vmin)
    hi = ("≥ " if clipped[1] else "") + _fmt(vmax)
    return (
        _title_html(title)
        + f"<div style='width:{width}px'>"
        f"<div style='height:10px;border-radius:2px;background:linear-gradient(to right, {stops})'></div>"
        "<div style='display:flex;justify-content:space-between;color:#666;font-size:11px;margin-top:2px'>"
        f"<span>{lo}</span><span>{_fmt((vmin + vmax) / 2)}</span><span>{hi}</span></div></div>"
    )


def _swatches_html(items: list[tuple[str, str]], max_items: int = 60) -> str:
    more = len(items) - max_items
    spans = "".join(
        "<span style='display:inline-flex;align-items:center;margin:0 12px 2px 0;white-space:nowrap'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{c};margin-right:5px;"
        f"display:inline-block;flex:none'></span>{escape(str(lab))}</span>"
        for lab, c in items[:max_items]
    )
    if more > 0:
        spans += f"<span style='color:#888'>+{more} more</span>"
    return f"<div style='display:flex;flex-wrap:wrap'>{spans}</div>"


def _cmap_colors(cmap, n: int = _N_BINS) -> list[str]:
    """n hex colours sampled at bin centres of a colormap name, Colormap or list of colours."""
    if cmap is None:
        cmap = "viridis"
    if isinstance(cmap, str):
        cm = colormaps[cmap]
    elif isinstance(cmap, mcolors.Colormap):
        cm = cmap
    else:
        cm = mcolors.LinearSegmentedColormap.from_list("custom", list(cmap))
    return [mcolors.to_hex(c) for c in cm((np.arange(n) + 0.5) / n)]


def _bin(values: np.ndarray, vmin: float, vmax: float, n: int = _N_BINS) -> np.ndarray:
    scaled = (np.clip(values, vmin, vmax) - vmin) / (vmax - vmin) * n
    return np.minimum(scaled.astype(np.int32), n - 1)


def _default_palette(n: int) -> list:
    """scanpy's default categorical palettes; a sampled colormap for very many categories."""
    pal = sc.pl.palettes
    for base in (pal.default_20, pal.default_28, pal.default_102):
        if n <= len(base):
            return list(base[:n])
    return list(colormaps["turbo"](np.linspace(0, 1, n)))


def _palette_colors(adata: sc.AnnData, key: str, categories: list[str], palette: PaletteLike) -> list[str]:
    n = len(categories)
    if palette is None:
        stored = adata.uns.get(f"{key}_colors") if key in adata.obs.columns else None
        colors = list(stored)[:n] if stored is not None and len(stored) >= n else _default_palette(n)
    elif isinstance(palette, Mapping):
        pal = {str(k): v for k, v in palette.items()}
        missing = [c for c in categories if c not in pal]
        if missing:
            logg.warning(f"`palette` has no colour for {missing[:5]}{'...' if len(missing) > 5 else ''}; using defaults.")
        fallback = _default_palette(n)
        colors = [pal.get(c, fallback[i]) for i, c in enumerate(categories)]
    elif isinstance(palette, str):
        if palette in colormaps:
            cm = colormaps[palette]
            if isinstance(cm, mcolors.ListedColormap) and n <= cm.N <= 64:  # qualitative: take colours as they are
                colors = list(cm.colors[:n])
            else:  # continuous colormap: spread categories over it
                colors = list(cm(np.linspace(0, 1, n)))
        elif mcolors.is_color_like(palette):
            colors = [palette] * n
        else:
            raise ValueError(f"`palette='{palette}'` is neither a matplotlib colormap nor a colour.")
    else:
        seq = list(palette)
        if len(seq) < n:
            logg.warning(f"`palette` has {len(seq)} colours for {n} categories; colours will repeat.")
        colors = [seq[i % len(seq)] for i in range(n)]
    return [mcolors.to_hex(c) for c in colors]


def _resolve_norm(norm: NormLike, x: np.ndarray) -> tuple[float, float]:
    """(vmin, vmax); each bound may be a number, None (data min/max) or a percentile like 'p99'."""
    lo_default, hi_default = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)

    def bound(b, default):
        if b is None:
            return default
        if isinstance(b, str):
            if not b.startswith("p"):
                raise ValueError(f"Norm bound '{b}' must be a number, None or a percentile like 'p99'.")
            return float(np.percentile(x, float(b[1:]))) if x.size else default
        return float(b)

    lo, hi = (None, None) if norm is None else norm
    lo, hi = bound(lo, lo_default), bound(hi, hi_default)
    if hi < lo:
        raise ValueError(f"Colour range is inverted: vmin={lo}, vmax={hi}.")
    if hi == lo:  # constant feature
        hi = lo + 1.0
    return lo, hi


def _resolve_basis(adata: sc.AnnData, basis: str) -> np.ndarray:
    """2D float32 coordinates from .obsm; accepts 'umap' as well as 'X_umap'."""
    if basis in adata.obsm:
        key = basis
    elif f"X_{basis}" in adata.obsm:
        key = f"X_{basis}"
    else:
        raise KeyError(f"Basis '{basis}' not found in .obsm (available: {list(adata.obsm.keys())}).")
    emb = np.asarray(adata.obsm[key])
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f".obsm['{key}'] must have at least 2 columns, got shape {emb.shape}.")
    return np.ascontiguousarray(emb[:, :2], dtype=np.float32)


def _get_vector(adata: sc.AnnData, key: str, use_raw: bool | None, layer: str | None):
    """Values of an obs column or a feature (from .X, a layer or .raw), one per observation."""
    if key in adata.obs.columns:
        return adata.obs[key].values
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None and key in adata.raw.var_names
    if use_raw:
        if adata.raw is None:
            raise ValueError("`use_raw=True`, but `.raw` is not set.")
        if key not in adata.raw.var_names:
            raise KeyError(f"'{key}' is neither in .obs nor in .raw.var_names.")
        return adata.raw.obs_vector(key)
    if key in adata.var_names:
        return adata.obs_vector(key, layer=layer)
    raise KeyError(f"'{key}' is neither a column of .obs nor a feature in .var_names / .raw.var_names.")


def _resolve_coloring(
    adata: sc.AnnData,
    key: str | None,
    *,
    cmap,
    palette: PaletteLike,
    norm: NormLike,
    use_raw: bool | None,
    layer: str | None,
    na_color: str,
) -> _Coloring:
    na_color = mcolors.to_hex(na_color)
    if key is None:
        return _Coloring("none", np.zeros(adata.n_obs, np.int32), [na_color], None, None, None)

    vec = _get_vector(adata, key, use_raw, layer)
    is_cat = (
        isinstance(vec, pd.Categorical)
        or isinstance(getattr(vec, "dtype", None), pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(vec)
        or not pd.api.types.is_numeric_dtype(vec)
    )

    if is_cat:
        cat = vec if isinstance(vec, pd.Categorical) else pd.Categorical(vec)
        full = [str(c) for c in cat.categories]
        cmap_full = dict(zip(full, _palette_colors(adata, key, full, palette)))
        cat = cat.remove_unused_categories()
        labels = [str(c) for c in cat.categories]
        codes = np.asarray(cat.codes, dtype=np.int32)
        na = codes < 0
        order = None
        if na.any():
            codes[na] = len(labels)
            labels.append(_NA)
            cmap_full[_NA] = na_color
            order = np.argsort(~na, kind="stable")  # missing values at the bottom
        values = np.asarray(labels, dtype=object)[codes]
        return _Coloring("categorical", codes, [cmap_full[c] for c in labels], labels, key, values, order=order)

    v = np.array(vec, dtype=np.float32).ravel()  # copy: never write into adata
    nan = np.isnan(v)
    finite = v[~nan] if nan.any() else v
    vmin, vmax = _resolve_norm(norm, finite)
    clipped = (bool(finite.size and finite.min() < vmin), bool(finite.size and finite.max() > vmax))
    if nan.any():
        logg.warning(f"'{key}' has {int(nan.sum())} NaN values; they are drawn at the bottom in the lowest colour.")
        v[nan] = vmin
    codes = _bin(v, vmin, vmax)
    order = np.argsort(v, kind="stable")  # high values on top
    return _Coloring("continuous", codes, _cmap_colors(cmap), None, key, v, (vmin, vmax), clipped, order)


def _clone_codes(labels: pd.Series, clone_index: pd.Index) -> np.ndarray:
    """Position of each cell's clone in `clone_index` (-1 if the clone is absent or the label is missing)."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        lut = clone_index.get_indexer(labels.cat.categories.astype(str))
        lut = np.append(lut, -1)  # code -1 (NaN) -> lut[-1] -> -1
        return lut[labels.cat.codes.to_numpy()].astype(np.int32)
    codes = clone_index.get_indexer(labels.astype(str)).astype(np.int32)
    codes[labels.isna().to_numpy()] = -1
    return codes


def _tokens(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _enable_tooltip(scatter, properties: list[str]) -> None:
    try:
        scatter.tooltip(True, properties=properties)
    except (TypeError, ValueError):  # older jscatter: no data columns in tooltips
        scatter.tooltip(True)

def clones2cells(
    adata: sc.AnnData,
    clones: sc.AnnData,
    obs_name: str | None = None,
    cells_color: str | None = None,
    clones_color: str | None = None,
    mode: Literal["highlight", "filter"] = "highlight",
    cells_basis: str = "X_umap",
    clones_basis: str = "X_umap",
    keep_color: bool = True,
    bg_color: str = "lightgrey",
    selected_color: str = "red",
    cells_cmap: str | None = None,
    clones_cmap: str | None = None,
    cells_palette: PaletteLike = None,
    clones_palette: PaletteLike = None,
    s_cells_bg: float = 1,
    s_cells_selected: float = 2,
    s_clones: float = 6,
    width: float | Literal["auto"] = "auto",
    height: float = 400,
    axes: bool = False,
    norm_clones: NormLike = None,
    norm_cells: NormLike = None,
    *,
    use_raw: bool | None = None,
    cells_layer: str | None = None,
    clones_layer: str | None = None,
    selection_stat: Literal["percent", "count"] = "percent",
    selection_cmap: str = "viridis",
    cells_opacity: float = 1.0,
    clones_tooltip: bool = True,
    show: bool = True,
) -> ipywidgets.VBox | None:
    """
    Two linked interactive scatterplots: cells (left) and clones (right).

    Selecting clones on the right highlights (`mode='highlight'`) or filters
    (`mode='filter'`) their cells on the left. Selecting cells on the left colours
    every clone by the share (or number) of its cells that are selected.

    Parameters
    ----------
    adata
        Cells (e.g. gene expression).
    clones
        Clones, one observation per clone, with a clonal embedding in `.obsm`.
    obs_name
        Column of `adata.obs` holding each cell's clone label (labels must match
        `clones.obs_names`). Defaults to `clones.uns["obs_name"]`.
    cells_color, clones_color
        What to colour by: a column of `.obs` or a feature name. Features are taken
        from `.raw` when it contains them (see `use_raw`), otherwise from `.X`
        or from `cells_layer` / `clones_layer`. Numeric values get a colormap
        (shown as a colorbar), everything else (strings, categories, booleans)
        gets a palette.
    mode
        'highlight': all cells stay visible; cells of the selected clones are drawn
        on top and larger, all other cells turn `bg_color`. 'filter': only cells of
        the selected clones are shown.
    cells_basis, clones_basis
        `.obsm` keys of the embeddings ('umap' and 'X_umap' both work).
    keep_color
        In 'highlight' mode, draw highlighted cells in their `cells_color` colour
        (categorical or continuous). If False, or without `cells_color`, they are
        drawn in `selected_color`.
    bg_color
        Colour of non-highlighted cells, uncoloured plots and missing values.
    selected_color
        Colour of highlighted cells when `keep_color=False` or `cells_color=None`.
    cells_cmap, clones_cmap
        Colormap for numeric colouring: a name, a Colormap or a list of colours
        (default 'viridis').
    cells_palette, clones_palette
        Categorical colours: a dict {category: colour}, a list of colours, a
        matplotlib colormap name (e.g. 'tab20') or a single colour. Default:
        `.uns[f"{key}_colors"]` if present, otherwise scanpy's default palettes.
        `.uns` is never modified.
    s_cells_bg, s_cells_selected, s_clones
        Point sizes.
    width, height
        Size of each plot.
    axes
        Show axes.
    norm_cells, norm_clones
        Colour range (vmin, vmax) for numeric colouring. Each bound can be a
        number, None (data min/max) or a percentile string such as 'p99'.
        Values outside the range get the end colours, e.g. (0, 'p99') for genes.
    use_raw
        Take features from `.raw`. Default: when `.raw` exists, contains the
        feature and no layer is requested.
    cells_layer, clones_layer
        Layer to read features from (when not using `.raw`).
    selection_stat
        How clones are coloured after a selection on the cells plot: 'percent'
        of the clone's cells that are selected, or 'count' of selected cells.
    selection_cmap
        Colormap for that colouring.
    cells_opacity
        Opacity of cells; lower it for very dense plots.
    clones_tooltip
        Show clone name, size and colour value on hover over clones.
    show
        Display the widget. If False, the widget is returned instead.

    Returns
    -------
    The widget if `show` is False, otherwise None.

    Notes
    -----
    Numeric colours are drawn with 256 levels. The last selections are stored in
    `clones.uns["clones2cells"]["last_selection"]` (clone names) and
    `adata.uns["clones2cells"]["last_selection"]` (cell names); the per-clone values of
    the last cell selection are in `adata.uns["clones2cells"]["clone_stat"]`.
    """
    try:
        import ipywidgets as widgets
        import jscatter as js
        from IPython.display import display
    except ImportError as e:
        raise ImportError(
            "`ipywidgets` and `jscatter` are required for clones2cells: `pip install ipywidgets jupyter-scatter`."
        ) from e

    if mode not in ("highlight", "filter"):
        raise ValueError(f"`mode` must be 'highlight' or 'filter', got '{mode}'.")
    if selection_stat not in ("percent", "count"):
        raise ValueError(f"`selection_stat` must be 'percent' or 'count', got '{selection_stat}'.")
    if obs_name is None:
        if "obs_name" not in clones.uns:
            raise ValueError("Pass `obs_name` (column of `adata.obs` with clone labels); `clones.uns['obs_name']` is not set.")
        obs_name = clones.uns["obs_name"]
    if obs_name not in adata.obs.columns:
        raise KeyError(f"'{obs_name}' is not a column of `adata.obs`.")

    bg_hex, sel_hex = mcolors.to_hex(bg_color), mcolors.to_hex(selected_color)
    highlight = mode == "highlight"

    # ================= clones =================
    col_cl = _resolve_coloring(
        clones, clones_color, cmap=clones_cmap, palette=clones_palette, norm=norm_clones,
        use_raw=use_raw, layer=clones_layer, na_color=bg_hex,
    )
    n_clones = clones.n_obs
    order_cl = col_cl.order if col_cl.order is not None else np.arange(n_clones)
    clone_names = np.asarray(clones.obs_names[order_cl]).astype(str)
    xy_cl = _resolve_basis(clones, clones_basis)[order_cl]

    # Every cell gets the row of its clone in the clones plot (-1 = not shown there).
    # All selection logic below is integer indexing on these codes.
    cell_code = _clone_codes(adata.obs[obs_name], pd.Index(clone_names))
    clone_size = np.bincount(cell_code[cell_code >= 0], minlength=n_clones)
    if (clone_size == 0).any():
        logg.warning(f"{int((clone_size == 0).sum())} clones in `clones` have no cells in `adata.obs['{obs_name}']`.")

    # Every clone is its own colour category ("k<row>"), and the point data of the clones
    # plot is never changed after setup. Recolouring (by selection, or back to the original
    # colours) only replaces the category -> colour map, which jscatter applies without
    # touching the points. This avoids jscatter's encoding cache entirely.
    clone_tokens = _tokens("k", n_clones)
    sel_colors = _cmap_colors(selection_cmap)
    idle_clone_colors = np.asarray(col_cl.colors, dtype=object)[col_cl.codes[order_cl]]
    stat_label = "% of clone's cells selected" if selection_stat == "percent" else "Selected cells per clone"

    df_clones = pd.DataFrame({
        "x": xy_cl[:, 0],
        "y": xy_cl[:, 1],
        "clone": clone_names,
        "n_cells": clone_size,
        "code": pd.Categorical.from_codes(np.arange(n_clones), categories=clone_tokens),
    })
    tooltip_cols = ["clone", "n_cells"]
    if col_cl.values is not None:
        value_col = clones_color if clones_color not in df_clones.columns else f"{clones_color}_value"
        df_clones[value_col] = col_cl.values[order_cl]
        tooltip_cols.insert(2, value_col)

    # ================= cells =================
    col = _resolve_coloring(
        adata, cells_color, cmap=cells_cmap, palette=cells_palette, norm=norm_cells,
        use_raw=use_raw, layer=cells_layer, na_color=bg_hex,
    )
    keep_color = keep_color and col.kind != "none"
    n_cells = adata.n_obs
    n_cls = len(col.colors)
    order = col.order if col.order is not None else np.arange(n_cells)
    xy = _resolve_basis(adata, cells_basis)

    # Each point gets a fixed colour token; selections only change the filter and the
    # token -> colour map, so millions of points are sent to the browser only once.
    bg_tok = _tokens("b", n_cls)
    if highlight:
        # Background layer = all cells; top layer = a copy of the cells that belong to
        # a clone, drawn above. Highlighting = letting selected top-layer points through
        # the filter and turning the background grey (colour-map swap only).
        fg_rows = order[cell_code[order] >= 0]
        point_rows = np.concatenate([order, fg_rows])
        fg_code = cell_code[fg_rows]
        bg_points = np.arange(n_cells, dtype=np.int64)
        fg_tok = _tokens("f", n_cls)
        tokens = bg_tok + fg_tok
        point_tok = np.concatenate([col.codes[order], n_cls + col.codes[fg_rows]])
        fg_colors = col.colors if keep_color else [sel_hex] * n_cls
        map_idle = dict(zip(tokens, col.colors + col.colors))
        map_active = dict(zip(tokens, [bg_hex] * n_cls + fg_colors))
        size_map = dict(zip(tokens, [s_cells_bg] * n_cls + [s_cells_selected] * n_cls))
    else:
        point_rows = order
        tokens = bg_tok
        point_tok = col.codes[order]
        map_idle = dict(zip(tokens, col.colors))
    point_code = cell_code[point_rows]

    df_cells = pd.DataFrame({
        "x": xy[point_rows, 0],
        "y": xy[point_rows, 1],
        "code": pd.Categorical.from_codes(point_tok, categories=tokens),
    })

    # ---------------- widgets ----------------
    scatter_cells = js.Scatter(
        x="x", y="y", data=df_cells, width=width, height=height,
        opacity=cells_opacity, size=s_cells_bg, axes=axes,
    )
    scatter_cells.color(by="code", map=map_idle)
    if highlight:
        scatter_cells.size(by="code", map=size_map)  # static: background tokens are always small
        scatter_cells.filter(bg_points)
    scatter_cells.legend(False)  # replaced by the HTML legends below the plots

    scatter_clones = js.Scatter(
        x="x", y="y", data=df_clones, width=width, height=height,
        opacity=1, size=s_clones, axes=axes,
    )
    scatter_clones.color(by="code", map=dict(zip(clone_tokens, idle_clone_colors)))
    scatter_clones.legend(False)
    if clones_tooltip:
        _enable_tooltip(scatter_clones, tooltip_cols)

    legend_cells_idle = col.legend_html()
    legend_clones_idle = col_cl.legend_html()
    legend_cells = widgets.HTML(legend_cells_idle)
    legend_clones = widgets.HTML(legend_clones_idle)

    # ---------------- interaction ----------------
    state = {"cells_highlighted": False, "clones_by_selection": False}

    def reset_cells():
        if highlight:
            scatter_cells.color(map=map_idle)
            scatter_cells.filter(bg_points)
        else:
            scatter_cells.filter(None)
            scatter_cells.size(s_cells_bg)
        legend_cells.value = legend_cells_idle
        state["cells_highlighted"] = False

    def on_clones_selected(change):
        idx = np.asarray(change.new, dtype=np.int64)
        if idx.size == 0:
            if state["cells_highlighted"]:
                reset_cells()
            return
        clones.uns["clones2cells"] = {"last_selection": clone_names[idx]}

        is_selected = np.zeros(n_clones + 1, dtype=bool)  # last slot stays False: code -1 maps to it
        is_selected[idx] = True

        if highlight:
            top = np.flatnonzero(is_selected[fg_code])
            scatter_cells.filter(np.concatenate([bg_points, n_cells + top]))
            if not state["cells_highlighted"]:
                scatter_cells.color(map=map_active)
            n_shown = top.size
        else:
            shown = np.flatnonzero(is_selected[point_code])
            scatter_cells.filter(shown)
            if not state["cells_highlighted"]:
                scatter_cells.size(s_cells_selected)
            n_shown = shown.size

        note = f"{n_shown:,} cells from {idx.size:,} selected clone{'s' if idx.size > 1 else ''}"
        if highlight and keep_color:
            legend_cells.value = col.legend_html(extra=[("Other cells", bg_hex)], note=note)
        elif highlight:
            legend_cells.value = _Coloring("none", None, [], None, None, None).legend_html(
                extra=[("Cells in selected clones", sel_hex), ("Other cells", bg_hex)], note=note)
        else:
            legend_cells.value = col.legend_html(note=note) or f"<div style=\"{_FONT}\">{escape(note)}</div>"
        state["cells_highlighted"] = True

    def set_clone_colors(colors_per_clone):
        """Colour of every clone, in clones-plot row order. Map-only update: no points are re-sent."""
        scatter_clones.color(map=dict(zip(clone_tokens, colors_per_clone)))

    def on_cells_selected(change):
        idx = np.asarray(change.new, dtype=np.int64)
        if idx.size == 0:
            if state["clones_by_selection"]:
                set_clone_colors(idle_clone_colors)
                legend_clones.value = legend_clones_idle
                state["clones_by_selection"] = False
            return

        # A cell may be selected twice in highlight mode (background + top copy): dedupe by mask.
        cell_mask = np.zeros(n_cells, dtype=bool)
        cell_mask[point_rows[idx]] = True

        codes = cell_code[cell_mask]
        counts = np.bincount(codes[codes >= 0], minlength=n_clones)
        if selection_stat == "percent":
            stat = np.divide(counts * 100.0, clone_size, out=np.zeros(n_clones), where=clone_size > 0)
        else:
            stat = counts.astype(float)
        vmax = float(stat.max()) or 1.0

        adata.uns["clones2cells"] = {
            "last_selection": np.asarray(adata.obs_names)[cell_mask],
            "clone_stat": pd.Series(stat, index=clone_names, name=selection_stat),
        }
        set_clone_colors(np.asarray(sel_colors, dtype=object)[_bin(stat, 0.0, vmax)])

        n_sel = int(cell_mask.sum())
        legend_clones.value = (
            f"<div style=\"{_FONT}\">{_colorbar_html(stat_label, sel_colors, 0.0, vmax)}"
            f"<div style='color:#888;font-size:11px;margin-top:2px'>{n_sel:,} cells selected, "
            f"{int((counts > 0).sum()):,} clones hit</div></div>"
        )
        state["clones_by_selection"] = True

    scatter_clones.widget.observe(on_clones_selected, names=["selection"])
    scatter_cells.widget.observe(on_cells_selected, names=["selection"])

    plots = js.compose([
        (scatter_cells, "Gene expression embedding"),
        (scatter_clones, "Clonal embedding"),
    ], row_height=height)
    legends = widgets.GridBox(
        [legend_cells, legend_clones],
        layout=widgets.Layout(grid_template_columns="1fr 1fr", width="100%"),
    )
    widget = widgets.VBox([plots, legends])

    if show:
        display(widget)
        return None
    return widget
