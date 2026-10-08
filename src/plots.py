
"""Plotting utilities for Accorpyon/Accordion outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def _accordion_clean_axis_ticks(ax, remove_spines=True):
    """Remove visible tick marks/labels from compact auxiliary axes."""
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
        length=0,
    )
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    if remove_spines:
        for _spine in ax.spines.values():
            _spine.set_visible(False)
    return ax



def _get_detailed_info(adata, annotation_name: str):
    try:
        return adata.uns.get(annotation_name, {}).get("detailed_annotation_info", {})
    except AttributeError:
        return {}


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _scanpy_color_mapping(adata, color_key: Optional[str], labels: Sequence[str]) -> dict:
    """Return label -> color using Scanpy's stored UMAP palette when available."""
    labels = [str(x) for x in labels]
    if color_key is None or not hasattr(adata, "obs"):
        return {}

    colors_key = f"{color_key}_colors"
    if color_key not in adata.obs.columns or colors_key not in adata.uns:
        return {}

    obs_values = adata.obs[color_key]
    if hasattr(obs_values, "cat"):
        categories = [str(x) for x in obs_values.cat.categories]
    else:
        categories = [str(x) for x in pd.Index(obs_values.astype(str)).drop_duplicates()]

    colors = list(adata.uns.get(colors_key, []))
    if not colors:
        return {}

    return {cat: colors[i] for i, cat in enumerate(categories[:len(colors)]) if cat in labels}


def _fallback_palette(labels: Sequence[str]) -> dict:
    """Deterministic, perceptually-spread color palette for any number of labels.

    Used whenever a plot needs colors for categories that Scanpy hasn't
    already assigned (see ``_scanpy_color_mapping``) -- cell types, clusters,
    etc. The first 60 labels get matplotlib's three 20-color qualitative
    colormaps back to back (``tab20`` + ``tab20b`` + ``tab20c``; identical to
    the old plain-``tab20`` behaviour for the first 20). Beyond that, instead
    of silently *recycling* those same 20-60 colors onto unrelated categories
    (which made large legends ambiguous -- two different cell types getting
    the exact same swatch), further colors are generated procedurally by
    stepping around the HSV hue wheel by the golden angle while cycling
    saturation/value. The golden-angle step avoids the hue clustering you'd
    get from naive even spacing when the label count doesn't divide evenly,
    so plots with dozens of cell types/clusters still get visually distinct
    colors instead of duplicates.
    """
    labels = [str(x) for x in labels]
    n = len(labels)

    base_colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        base_colors.extend(cmap(i) for i in range(cmap.N))

    colors = list(base_colors[:n])
    if n > len(base_colors):
        golden_angle = 0.6180339887498949  # 1 / phi -- maximally irrational hue step
        sat_levels = (0.65, 0.85, 0.50)
        val_levels = (0.95, 0.75, 0.55)
        hue = 0.15  # offset so the first extra hues don't echo the base palette
        for i in range(n - len(base_colors)):
            hue = (hue + golden_angle) % 1.0
            sat = sat_levels[i % len(sat_levels)]
            val = val_levels[(i // len(sat_levels)) % len(val_levels)]
            colors.append(mcolors.hsv_to_rgb((hue, sat, val)))

    return {lab: mcolors.to_hex(col) for lab, col in zip(labels, colors)}


def plot_top_markers_per_celltype(
        adata,
        annotation_name: str = "accordion",
        marker_table_key: str = "top_markers_per_cluster",
        color_key: Optional[str] = None,
        celltype_col: Optional[str] = None,
        marker_col: str = "marker",
        score_col: str = "gene_impact_score_per_cluster",
        top_n_markers: Optional[int] = None,
        min_score: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        size_scale: float = 35.0,
        legend_scale: float = 0.55,
        legend_quantiles: Sequence[float] = (0.25, 0.50, 0.75, 0.95),
        rotate_xticks: int = 90,
        title: str = "Top Markers per Cell Type",
        xlabel: str = "Markers",
        ylabel: str = "Cell Types",
        legend_title: str = "Impact Score",
        manual_palette: Optional[Mapping[str, str]] = None,
        ax=None,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
        grid: bool = True,
        invert_yaxis: bool = True,
):
    """Styled dot plot of top markers per cell type.

    This function is intentionally close to the hand-made plot style used in
    Accordion examples: markers on the x-axis, cell types on the y-axis, point
    size proportional to marker impact score, black point edges, bold labels,
    and colors matched to the Scanpy/UMAP palette whenever possible.

    The function reads one of the detailed annotation tables from::

        adata.uns[annotation_name]["detailed_annotation_info"][marker_table_key]

    By default it uses ``top_markers_per_cluster`` because that is the table
    that contains marker impact scores per cluster/cell type. If your run saved
    a different table, pass ``marker_table_key=...``.

    Parameters
    ----------
    adata
        Annotated AnnData returned by :func:`pycellmarkeraccordion.accordion`.
    annotation_name
        Key used in ``adata.uns``. For runs with custom names this may be e.g.
        ``"accordion_res0.4_per_cluster"``.
    marker_table_key
        Detailed annotation table to plot, usually ``"top_markers_per_cluster"``
        or ``"top_markers_per_celltype_cluster"``.
    color_key
        Obs column whose Scanpy colors should be reused. If ``None``, the
        function tries ``celltype_col`` and common Accordion annotation columns.
    celltype_col
        Column in the marker table containing cell type labels. If ``None``, it
        is auto-detected.
    fontscale
        Multiplies every font size in the figure at once (tick labels, title,
        legend, cell counts, entropy row labels and values). ``1.0`` keeps the
        current sizes; ``1.6`` is a good starting point for a figure that has
        to stay readable in a slide or at print size.
    entropy_row_height
        Height of each entropy strip relative to the dot plot (which is 9).
        Default ``0.5``; lower it for thinner bars and more room for the dot
        plot above, raise it if the values inside the strips get cramped.
    show_entropy
        ``True`` (default) adds one heatmap row per entropy metric found in
        ``detailed_annotation_info``. ``False`` draws the dot plot alone --
        handy for producing the with- and without-entropy versions of the same
        figure.
    order_celltypes_by_score
        ``True`` (default) groups the rows by the cluster where each cell type
        scores highest and sorts each group by ``score_col`` descending, so a
        cluster's winning cell type is the top row of its block. ``False``
        keeps the order of the input table.

    manual_palette
        Optional explicit mapping ``{celltype: color}``. Values here override
        Scanpy/fallback colors.
    title, xlabel, ylabel, legend_title
        Text overrides for the plot title, axis labels and the size-legend
        title (default ``"Impact Score"``). Anything not covered here (font
        sizes, bold/weight, etc.) can still be tweaked after the fact on the
        returned ``fig, ax``.
    save
        Optional output path. The figure is saved at ``dpi`` with
        ``bbox_inches='tight'``.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    detailed = _get_detailed_info(adata, annotation_name)
    if marker_table_key not in detailed:
        available = ", ".join(sorted(detailed.keys())) if detailed else "none"
        raise KeyError(
            f"Could not find {marker_table_key!r} in adata.uns[{annotation_name!r}]"
            f"['detailed_annotation_info']. Available keys: {available}. "
            "Run accordion(..., include_detailed_annotation_info=True)."
        )

    df = detailed[marker_table_key].copy()
    if df.empty:
        raise ValueError(f"Marker table {marker_table_key!r} is empty.")

    if celltype_col is None:
        celltype_col = _first_existing_column(df, [
            annotation_name,
            f"{annotation_name}_per_cluster",
            f"{annotation_name}_per_cell",
            "accordion_per_cluster",
            "accordion_per_cell",
            "annotation_per_cell",
            "CL_celltype",
            "celltype",
        ])
    if celltype_col is None or celltype_col not in df.columns:
        raise KeyError(
            "Could not identify the cell type column. Pass celltype_col explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    for col in (marker_col, score_col):
        if col not in df.columns:
            raise KeyError(f"Required column {col!r} not found. Available columns: {list(df.columns)}")

    df = df[[marker_col, celltype_col, score_col]].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[marker_col, celltype_col, score_col])
    df[celltype_col] = df[celltype_col].astype(str)
    df[marker_col] = df[marker_col].astype(str)

    if min_score is not None:
        df = df[df[score_col] >= float(min_score)]

    if top_n_markers is not None:
        df = (
            df.sort_values([celltype_col, score_col], ascending=[True, False])
              .groupby(celltype_col, observed=True, as_index=False)
              .head(int(top_n_markers))
        )

    if df.empty:
        raise ValueError("No markers left to plot after filtering.")

    # Preserve the visible ordering from the detailed table, like the prototype.
    celltypes = df[celltype_col].drop_duplicates().tolist()
    markers = df[marker_col].drop_duplicates().tolist()

    y_map = {c: i for i, c in enumerate(celltypes)}
    x_map = {m: i for i, m in enumerate(markers)}
    df["_x"] = df[marker_col].map(x_map)
    df["_y"] = df[celltype_col].map(y_map)

    # Auto-detect the UMAP/Scanpy palette. Prefer an exact obs column match.
    if color_key is None:
        candidates = [
            celltype_col,
            annotation_name,
            f"{annotation_name}_per_cluster",
            f"{annotation_name}_per_cell",
            "accordion_per_cluster",
            "accordion_per_cell",
            "cell_type",
            "CellHint_harmonised_group",
            "Curated_annotation",
        ]
        for candidate in candidates:
            if hasattr(adata, "obs") and candidate in adata.obs.columns and f"{candidate}_colors" in adata.uns:
                color_key = candidate
                break

    color_map = _scanpy_color_mapping(adata, color_key, celltypes)
    fallback = _fallback_palette(celltypes)
    palette = {ct: color_map.get(ct, fallback[ct]) for ct in celltypes}
    if manual_palette is not None:
        palette.update({str(k): v for k, v in manual_palette.items()})
    df["_color"] = df[celltype_col].map(palette).fillna("#808080")

    # Match the user's visual size mapping: sqrt(score) * scale.
    safe_scores = df[score_col].clip(lower=0)
    sizes = np.sqrt(safe_scores) * float(size_scale)

    if figsize is None:
        figsize = (max(10, 0.45 * len(markers)), max(5, 0.65 * len(celltypes)))

    # Use seaborn for the polished theme while keeping Matplotlib scatter for
    # exact control over edgecolors, numeric categorical positions, and legends.
    sns.set_theme(style="whitegrid", context="talk")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        df["_x"],
        df["_y"],
        s=sizes,
        c=df["_color"],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
    )

    ax.set_yticks(range(len(celltypes)))
    ax.set_yticklabels(celltypes, fontsize=15, weight="bold")

    ax.set_xlabel(xlabel, fontsize=20, weight="bold")
    ax.set_ylabel(ylabel, fontsize=20, weight="bold")
    ax.set_title(title, fontsize=24, weight="bold", pad=16)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=14)

    # Impact-score legend, like the prototype. Use unique rounded quantiles to
    # avoid repeated labels on small/low-variance inputs.
    score_values = df[score_col].dropna().clip(lower=0)
    if len(score_values) > 0:
        legend_scores = np.round(np.quantile(score_values, legend_quantiles)).astype(int)
        legend_scores = [int(s) for s in pd.unique(legend_scores) if s >= 0]
        if not legend_scores:
            legend_scores = [int(round(float(score_values.max())))]

        handles = [
            ax.scatter(
                [],
                [],
                s=np.sqrt(s) * float(size_scale) * float(legend_scale),
                color="black",
                edgecolors="black",
                alpha=0.85,
                label=f"{s}",
            )
            for s in legend_scores
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            title_fontsize=15,
            fontsize=13,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            borderpad=1.2,
            labelspacing=1.2,
        )

    # Pin x/y limits to the actual marker/cell-type counts (with a small pad)
    # rather than leaving them at matplotlib's auto-fit default, so the grid
    # and axes frame line up exactly with the outermost columns/rows.
    ax.set_xticks(range(len(markers)))
    ax.set_xticklabels(markers, rotation=rotate_xticks, fontsize=11, weight="bold")
    if rotate_xticks:
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    ax.set_xlim(-0.6, len(markers) - 0.4)
    ymin, ymax = -0.6, len(celltypes) - 0.4
    if invert_yaxis:
        ax.set_ylim(ymax, ymin)
    else:
        ax.set_ylim(ymin, ymax)
    if grid:
        ax.grid(True, alpha=0.25)
    else:
        ax.grid(False)

    fig.tight_layout()

    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax



def _ordered_categories_from_obs(adata, key: str):
    """Return obs categories in Scanpy color order when possible."""
    if key is None or not hasattr(adata, "obs") or key not in adata.obs.columns:
        return []
    values = adata.obs[key]
    try:
        if pd.api.types.is_categorical_dtype(values):
            return [str(x) for x in values.cat.categories]
    except Exception:
        pass
    vals = pd.Index(values.astype(str)).drop_duplicates().tolist()
    # Cluster labels are often numeric strings: keep natural numeric order when possible.
    try:
        return [str(x) for x in sorted(vals, key=lambda z: int(z))]
    except Exception:
        return [str(x) for x in vals]


def _scanpy_cluster_color_mapping(adata, color_key: Optional[str], cluster_values: Sequence) -> dict:
    """Map cluster ids to Scanpy colors, preserving categorical order."""
    if color_key is None or not hasattr(adata, "uns"):
        return {}
    colors_key = f"{color_key}_colors"
    if colors_key not in adata.uns:
        return {}
    categories = _ordered_categories_from_obs(adata, color_key)
    colors = list(adata.uns.get(colors_key, []))
    if not categories or not colors:
        return {}

    out = {}
    for cat, col in zip(categories, colors):
        out[str(cat)] = col
        try:
            out[int(cat)] = col
        except Exception:
            pass
    return out


def plot_top_celltypes_per_cluster(
        adata,
        annotation_name: str = "accordion",
        table_key: str = "top_celltypes",
        celltype_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        cluster_color_key: Optional[str] = None,
        manual_palette: Optional[Mapping[str, str]] = None,
        score_col: str = "celltype_impact_score",
        ncell_col: str = "ncell_tot_cluster",
        figsize: Optional[Tuple[float, float]] = None,
        size_scale: float = 35.0,
        legend_scale: float = 0.55,
        legend_quantiles: Sequence[float] = (0.25, 0.50, 0.75, 0.95),
        entropy_cmaps: Optional[Mapping[str, str]] = None,
        show_entropy_values: bool = True,
        entropy_vmin: float = 0.0,
        entropy_vmax: float = 1.0,
        title: str = "Top Cell Types per Cluster",
        xlabel: str = "Cluster",
        ylabel: Optional[str] = None,
        legend_title: str = "Impact Score",
        ax_fontsize: int = 9,
        celltype_fontsize: int = 7,
        fontscale: float = 1.0,
        show_entropy: bool = True,
        entropy_row_height: float = 0.5,
        order_celltypes_by_score: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
        show: bool = True,
):
    """Styled dot plot of top cell types per cluster with optional entropy rows.

    The function reads::

        adata.uns[annotation_name]["detailed_annotation_info"]["top_celltypes"]

    and draws clusters on the x-axis and candidate cell types on the y-axis.
    Dot size is proportional to ``celltype_impact_score``; the winning cell type
    per cluster is highlighted with a black border. Cluster colors are reused
    from Scanpy/UMAP palettes such as ``adata.uns["leiden_res0.4_colors"]`` when
    ``cluster_color_key="leiden_res0.4"`` is provided, or auto-detected when
    possible.

    Entropy heatmap rows are added only for entropy tables actually present in
    ``detailed_annotation_info``. Missing entropy metrics are silently skipped,
    so the layout adapts to all/some/no entropies.

    manual_palette
        Optional explicit mapping ``{cluster: color}``. Values here override
        both the reused Scanpy colors and the generated fallback -- clusters
        not listed keep whichever of those two would otherwise apply.
    title, xlabel, ylabel, legend_title
        Text overrides for the plot title, axis labels and the size-legend
        title (default ``"Impact Score"``).

    Returns
    -------
    fig, axes
        Matplotlib figure and a dictionary of axes.
    """
    # One knob for every text size in the figure. The individual fontsize
    # arguments stay the baseline; fontscale multiplies all of them, including
    # the ones that used to be hardcoded (title, legend, cell counts, entropy
    # row labels and values), so the whole figure scales together.
    fs = float(fontscale)
    ax_fontsize = ax_fontsize * fs
    celltype_fontsize = celltype_fontsize * fs
    title_fs = 13 * fs
    small_fs = 8 * fs
    tiny_fs = 7 * fs

    detailed = _get_detailed_info(adata, annotation_name)
    if table_key not in detailed:
        available = ", ".join(sorted(detailed.keys())) if detailed else "none"
        raise KeyError(
            f"Could not find {table_key!r} in adata.uns[{annotation_name!r}]"
            f"['detailed_annotation_info']. Available keys: {available}."
        )

    df = detailed[table_key].copy()
    if df.empty:
        raise ValueError(f"Table {table_key!r} is empty.")

    if celltype_col is None:
        celltype_col = _first_existing_column(df, [
            f"{annotation_name}_per_cluster",
            annotation_name,
            "accordion_per_cluster",
            "annotation_per_cell",
            "CL_celltype",
            "celltype",
        ])
    if celltype_col is None or celltype_col not in df.columns:
        raise KeyError(
            "Could not identify cell type column. Pass celltype_col explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    if cluster_col is None:
        cluster_col = _first_existing_column(df, [
            cluster_color_key if cluster_color_key is not None else "",
            "leiden",
            "cluster",
            "Cluster",
            "clusters",
        ])
    if cluster_col is None or cluster_col not in df.columns:
        raise KeyError(
            "Could not identify cluster column. Pass cluster_col explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    if score_col not in df.columns:
        raise KeyError(f"Required score column {score_col!r} not found. Available columns: {list(df.columns)}")

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[cluster_col, celltype_col, score_col])
    df[celltype_col] = df[celltype_col].astype(str)

    # Cluster order: numeric if possible, otherwise lexical/table order.
    cluster_values = pd.Index(df[cluster_col].dropna().unique()).tolist()
    try:
        clusters = sorted(cluster_values, key=lambda x: int(x))
    except Exception:
        clusters = sorted(cluster_values, key=lambda x: str(x))

    # Row (cell type) order. With order_celltypes_by_score the rows are grouped
    # by the cluster where each cell type scores highest, and inside each group
    # sorted by score descending -- so the winning cell type of a cluster (the
    # black-rimmed dot) is the first row of its block instead of the last. A
    # cell type appearing in several clusters is placed once, in the cluster
    # where it scores highest, since the y axis is shared across clusters.
    if order_celltypes_by_score:
        _best = df.loc[df.groupby(celltype_col, observed=True)[score_col].idxmax()]
        _rank = {c: i for i, c in enumerate(clusters)}
        _best = _best.assign(
            _cx=_best[cluster_col].map(_rank).fillna(
                _best[cluster_col].astype(str).map({str(c): i for i, c in enumerate(clusters)})
            )
        ).sort_values(["_cx", score_col], ascending=[True, False])
        celltypes = _best[celltype_col].tolist()
        y_map = {ct: i for i, ct in enumerate(celltypes)}
        df["_y"] = df[celltype_col].map(y_map)
    elif "y" in df.columns:
        # Preserve a user-provided y ordering when present.
        df["_y"] = pd.to_numeric(df["y"], errors="coerce")
        celltypes = df.sort_values("_y")[celltype_col].drop_duplicates().tolist()
        y_map = {ct: i for i, ct in enumerate(celltypes)}
        df["_y"] = df[celltype_col].map(y_map)
    else:
        celltypes = df[celltype_col].drop_duplicates().tolist()
        y_map = {ct: i for i, ct in enumerate(celltypes)}
        df["_y"] = df[celltype_col].map(y_map)
    cluster_to_x = {c: i for i, c in enumerate(clusters)}
    # Robust mapping for mixed int/string cluster columns.
    cluster_to_x_str = {str(c): i for i, c in enumerate(clusters)}
    df["_x"] = df[cluster_col].map(cluster_to_x)
    if df["_x"].isna().any():
        df["_x"] = df[cluster_col].astype(str).map(cluster_to_x_str)

    winner_idx = df.groupby(cluster_col, observed=True)[score_col].idxmax()
    df["_winner"] = df.index.isin(winner_idx)

    # Auto-detect cluster color key. Prefer explicit key; then exact cluster_col;
    # then common leiden variants with stored Scanpy colors.
    if cluster_color_key is None:
        candidates = [cluster_col]
        if hasattr(adata, "uns"):
            candidates.extend([
                k[:-7] for k in adata.uns.keys()
                if isinstance(k, str) and k.endswith("_colors") and "leiden" in k
            ])
        for candidate in candidates:
            if candidate and hasattr(adata, "uns") and f"{candidate}_colors" in adata.uns:
                cluster_color_key = candidate
                break

    color_map = _scanpy_cluster_color_mapping(adata, cluster_color_key, clusters)
    fallback = _fallback_palette([str(c) for c in clusters])
    df["_color"] = df[cluster_col].map(color_map)
    if df["_color"].isna().any():
        df["_color"] = df[cluster_col].astype(str).map({str(k): v for k, v in color_map.items()})
    if df["_color"].isna().any():
        df["_color"] = df[cluster_col].astype(str).map(fallback)

    df["_color"] = df["_color"].astype(object).fillna("#808080")

    if manual_palette is not None:
        manual = {str(k): v for k, v in manual_palette.items()}
        override = df[cluster_col].astype(str).map(manual)
        df["_color"] = override.combine_first(df["_color"])

    sizes = np.sqrt(df[score_col].fillna(0).clip(lower=0)) * float(size_scale)

    ncell = None
    if ncell_col in df.columns:
        ncell = df.groupby(cluster_col, observed=True)[ncell_col].first()
        # Reindex robustly with string fallback.
        try:
            ncell = ncell.reindex(clusters)
        except Exception:
            ncell.index = ncell.index.astype(str)
            ncell = ncell.reindex([str(c) for c in clusters])

    # Entropy rows: include only metrics present in detailed_annotation_info.
    default_entropy_specs = [
        ("Shannon Entropy", "cluster_entropy_Shannon", "Shannon_entropy", "Greens"),
        # Gini is reported raw (1 - sum(p^2)); it is bounded by 1 by
        # construction, so it shares the [0, 1] scale of the normalized rows.
        ("Gini Entropy", "cluster_entropy_Gini", "Gini_entropy", "Oranges"),
        ("Ontology-sp Rao Entropy", "cluster_entropy_Ontology-sp-Rao", "Ontology-sp-Rao_entropy", "Reds"),
        ("Ontology-Lin Rao Entropy", "cluster_entropy_Ontology-Lin-Rao", "Ontology-Lin-Rao_entropy", "Purples"),
        ("Jaccard Rao Entropy", "cluster_entropy_Jaccard-Rao", "Jaccard-Rao_entropy", "Blues"),
        ("Overlap Rao Entropy", "cluster_entropy_Overlap-Rao", "Overlap-Rao_entropy", "YlOrBr"),
    ]
    if not show_entropy:
        default_entropy_specs = []

    entropy_cmaps = dict(entropy_cmaps or {})
    entropy_rows = []
    cluster_candidates = [cluster_col, cluster_color_key, "leiden", "cluster"]
    for label, key, value_col, cmap_name in default_entropy_specs:
        if key not in detailed:
            continue
        tmp = detailed[key]
        if tmp is None or getattr(tmp, "empty", True) or value_col not in tmp.columns:
            continue
        tmp = tmp.copy()
        tmp_cluster_col = _first_existing_column(tmp, [c for c in cluster_candidates if c])
        if tmp_cluster_col is None:
            continue
        vals = tmp.set_index(tmp_cluster_col)[value_col]
        vals.index = vals.index.astype(str)
        vals = vals.reindex([str(c) for c in clusters])
        entropy_rows.append((label, vals, entropy_cmaps.get(label, cmap_name)))

    n_entropy = len(entropy_rows)
    # Layout: dot plot, a dedicated spacer, the cell-count strip, then one row
    # per entropy metric. The spacer exists so the gap under the dot plot (which
    # must clear the x tick labels and the x label, both growing with fontscale)
    # is set independently of the gap between the entropy rows -- a single
    # hspace cannot do both, and tying them together is what used to either
    # clip "Cluster" or waste a band of white space.
    _row_h = float(entropy_row_height)

    # Explicit spacer rows instead of a single hspace. One hspace has to serve
    # three different needs at once -- clearing the x tick labels and x label
    # under the dot plot, separating the cell-count strip from the first entropy
    # bar, and leaving room for the colorbar tick labels between bars -- and
    # tightening it for one always loosened or broke another. With hspace at 0
    # and a spacer row per gap, each is sized for exactly what it has to clear.
    _gap_label = (2.0 * ax_fontsize * 1.70 + 8.0 * fs) / 72.0   # ticks + label + pad
    _gap_ncells = 0.10 * fs                                     # strip -> first bar
    _gap_cb = (tiny_fs * 2.3) / 72.0                            # colorbar tick labels
    _gaps_in = _gap_label + (_gap_ncells + (n_entropy - 1) * _gap_cb if n_entropy else 0.0)

    _fixed = 9.0 + 0.55 + _row_h * n_entropy
    if figsize is None:
        # The dot plot needs one readable line per cell type, so the figure must
        # be tall enough that the main axes alone can hold them at the current
        # font size: n labels x fontsize x line spacing, divided by the share of
        # the height the main axes gets. 2.0 lines per label, not 1.5: measured
        # on rendered figures, 1.5 leaves consecutive labels almost touching at
        # fontscale 2.2 and overlapping at 2.6; 1.8 keeps a visible gap while
        # costing about two inches less height than 2.0.
        _margins = 1.6
        _need_main_in = len(celltypes) * celltype_fontsize * 1.8 / 72.0
        _need_fig_in = _need_main_in * (_fixed / 9.0) + _gaps_in + _margins
        figsize = (max(12, 0.45 * len(clusters)),
                   max(7, 4.5 + 0.32 * len(celltypes) + (0.75 * _row_h + 0.30) * n_entropy,
                       _need_fig_in))

    # Convert the inch-sized gaps into gridspec units using the height actually
    # available for the rows.
    _usable = max(figsize[1] - 1.6, 1.0)
    _unit_in = max((_usable - _gaps_in) / _fixed, 1e-3)
    _u = lambda inches: max(inches / _unit_in, 1e-3)

    height_ratios = [9.0, _u(_gap_label), 0.55]
    row_of_entropy = []
    for i in range(n_entropy):
        height_ratios.append(_u(_gap_ncells if i == 0 else _gap_cb))
        row_of_entropy.append(len(height_ratios))
        height_ratios.append(_row_h)
    nrows = len(height_ratios)

    sns.set_theme(style="whitegrid", context="notebook")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2,
        width_ratios=[1.4, 8.0],
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.05,
    )

    axes = {}
    ax = fig.add_subplot(gs[0, 1])
    axes["main"] = ax

    mask_winner = df["_winner"]
    ax.scatter(
        df.loc[~mask_winner, "_x"],
        df.loc[~mask_winner, "_y"],
        s=sizes.loc[~mask_winner],
        c=df.loc[~mask_winner, "_color"],
        alpha=0.75,
        edgecolors="none",
    )
    ax.scatter(
        df.loc[mask_winner, "_x"],
        df.loc[mask_winner, "_y"],
        s=sizes.loc[mask_winner],
        c=df.loc[mask_winner, "_color"],
        alpha=0.95,
        edgecolors="black",
        linewidths=1.6,
    )

    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=14 * fs)
    ax.set_yticks(range(len(celltypes)))
    ax.set_yticklabels(celltypes, fontsize=celltype_fontsize)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([str(c) for c in clusters], fontsize=ax_fontsize)
    # Fix the x-limits to match the entropy heatmaps below (extent -0.5..n-0.5)
    # and the "Number of Cells" row, so all the rows line up with the columns.
    ax.set_xlim(-0.5, len(clusters) - 0.5)
    ax.set_xlabel(xlabel, fontsize=ax_fontsize, labelpad=6 * fs)
    ax.set_ylabel(ylabel or celltype_col, fontsize=ax_fontsize)
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()

    score_values = df[score_col].dropna().clip(lower=0)
    if len(score_values) > 0:
        legend_scores = np.quantile(score_values, legend_quantiles)
        legend_scores = np.unique(np.round(legend_scores).astype(int))
        legend_scores = [int(s) for s in legend_scores if s >= 0]
        handles = [
            ax.scatter(
                [], [],
                s=np.sqrt(s) * float(size_scale) * float(legend_scale),
                color="black",
                edgecolors="black",
                alpha=0.85,
                label=f"{s}",
            )
            for s in legend_scores
        ]
        if handles:
            ax.legend(
                handles=handles,
                title=legend_title,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
                fontsize=small_fs,
                title_fontsize=small_fs,
            )

    ax_n = fig.add_subplot(gs[2, 1])
    axes["n_cells"] = ax_n
    # Share the main scatter's x-limits (not its own auto-fit) so each value
    # lines up under the correct cluster column. No x tick labels here: the
    # cluster IDs are already shown on the main scatter above; this row only
    # adds the per-cluster cell count underneath.
    ax_n.set_xlim(ax.get_xlim())
    ax_n.set_xticks([])
    ax_n.set_xticks([], minor=True)
    ax_n.set_ylim(0, 1)
    ax_n.set_yticks([])
    _accordion_clean_axis_ticks(ax_n, remove_spines=False)
    if ncell is not None:
        for i, c in enumerate(clusters):
            try:
                val = ncell.loc[c]
            except Exception:
                val = ncell.loc[str(c)] if str(c) in ncell.index else np.nan
            if pd.notna(val):
                ax_n.text(i, 0.5, str(int(val)), ha="center", va="center", fontsize=small_fs)

        # Row label, placed just left of the axes (in axes-fraction x, data y).
        ax_n.text(
            -0.01, 0.5, "Number of Cells",
            ha="right", va="center", fontsize=small_fs,
            transform=ax_n.get_yaxis_transform(), clip_on=False,
        )
    for spine in ["left", "right"]:
        ax_n.spines[spine].set_visible(False)
    ax_n.spines["top"].set_visible(True)
    ax_n.spines["bottom"].set_visible(False)
    ax_n.tick_params(axis="both", which="both", length=0)

    axes["entropy"] = []
    for i, (label, vals, cmap_name) in enumerate(entropy_rows):
        row = row_of_entropy[i]
        cax = fig.add_subplot(gs[row, 0])
        hax = fig.add_subplot(gs[row, 1])
        _accordion_clean_axis_ticks(hax, remove_spines=True)
        norm = mcolors.Normalize(vmin=entropy_vmin, vmax=entropy_vmax)
        cmap_row = plt.get_cmap(cmap_name)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap_row),
            cax=cax,
            orientation="horizontal",
        )
        cb.set_ticks([entropy_vmin, entropy_vmax])
        cb.ax.tick_params(labelsize=tiny_fs)
        cax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=32, fontsize=small_fs)

        arr = vals.to_numpy(dtype=float).reshape(1, -1)
        hax.imshow(
            arr,
            aspect="auto",
            cmap=cmap_row,
            norm=norm,
            extent=(-0.5, len(clusters) - 0.5, 0, 1),
        )
        if show_entropy_values:
            midpoint = entropy_vmin + 0.45 * (entropy_vmax - entropy_vmin)
            for j, v in enumerate(vals):
                if pd.notna(v):
                    hax.text(
                        j, 0.5, f"{float(v):.2f}",
                        ha="center", va="center", fontsize=tiny_fs,
                        color="white" if float(v) > midpoint else "black",
                    )
        hax.set_yticks([])
        _accordion_clean_axis_ticks(hax, remove_spines=True)
        hax.set_xlim(-0.5, len(clusters) - 0.5)
        hax.set_xticks([])
        hax.set_xticks([], minor=True)
        axes["entropy"].append((cax, hax))

    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig, axes


# ---------------------------------------------------------------------------
# Lollipop plots (detailed per-cluster / per-cell-type marker and cell-type views)
# ---------------------------------------------------------------------------

# Specificity-score (SPs) colour ramps:
#   positive markers -> dark-red ramp, negative markers -> dark-blue ramp.
_SPS_POS_RAMP = ["#8B1A1A", "#a84a45", "#c47b73", "#dcaaa3", "#f1d9d4"]
_SPS_NEG_RAMP = ["#104E8B", "#456fa3", "#7a93bc", "#a8bad5", "#cfdbe7"]
_SPS_LABELS = ["1", "0.50", "0.33", "0.25", "<0.25"]


def _sps_bin(value):
    """Map a specificity score to a (ramp index, label) pair."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 4, "<0.25"
    if v >= 0.75:
        return 0, "1"
    if v >= 0.41:
        return 1, "0.50"
    if v >= 0.29:
        return 2, "0.33"
    if v >= 0.20:
        return 3, "0.25"
    return 4, "<0.25"


def _size_legend_handles(ax, value_to_size, title, color="black"):
    handles = [
        ax.scatter([], [], s=s, color=color, edgecolors="black",
                   alpha=0.9, label=str(v))
        for v, s in value_to_size
    ]
    return handles, title


def plot_markers_lollipop(
        adata,
        group,
        annotation_name: str = "accordion",
        marker_table_key: str = "top_markers_per_celltype_cluster",
        group_col: Optional[str] = None,
        marker_col: str = "marker",
        score_col: str = "gene_impact_score_per_cluster",
        size_col: Optional[str] = None,
        sps_col: str = "SPs",
        marker_type_col: str = "marker_type",
        size_range: Tuple[float, float] = (60.0, 400.0),
        title: Optional[str] = None,
        xlabel: str = "Gene impact score",
        ylabel: str = "",
        legend_title: str = "SPs",
        figsize: Optional[Tuple[float, float]] = None,
        ax=None,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
):
    """Lollipop plot of the top markers for a single cluster / cell type.

    Markers are drawn on the y-axis, the gene impact score on the x-axis (a
    stem from 0 to the score), point **size** encodes the evidence
    consistency score (``ECs``) or marker ``weight``, and point **colour**
    encodes the specificity-score (``SPs``) band, with separate red/blue ramps
    for positive and negative markers.

    Parameters
    ----------
    adata
        AnnData annotated by :func:`pycellmarkeraccordion.accordion` with
        ``include_detailed_annotation_info=True``.
    group
        The cluster id or cell-type label to plot (matched against
        ``group_col``).
    marker_table_key
        Detailed table to read, e.g. ``"top_markers_per_celltype_cluster"``
        (group by cell type) or ``"top_markers_per_cluster"`` (group by cluster).
    group_col
        Column used to select ``group``. If ``None`` it is auto-detected
        (cell-type column for ``*_celltype_*`` tables, otherwise the cluster
        column).
    size_col
        Column mapped to point size. If ``None``, uses ``"ECs"`` when present,
        else ``"weight"``, else a constant size.
    title, xlabel, ylabel, legend_title
        Text overrides for the plot title (default: the group name), the
        x-axis label, y-axis label, and the SPs color-band legend title
        (default ``"SPs"``). The size-legend title still follows ``size_col``.

    Returns
    -------
    fig, ax
    """
    detailed = _get_detailed_info(adata, annotation_name)
    if marker_table_key not in detailed:
        available = ", ".join(sorted(detailed.keys())) if detailed else "none"
        raise KeyError(
            f"Could not find {marker_table_key!r} in adata.uns[{annotation_name!r}]"
            f"['detailed_annotation_info']. Available keys: {available}. "
            "Run accordion(..., include_detailed_annotation_info=True)."
        )

    df = detailed[marker_table_key].copy()
    if df.empty:
        raise ValueError(f"Marker table {marker_table_key!r} is empty.")

    # auto-detect the grouping column
    if group_col is None:
        celltype_candidates = [
            f"{annotation_name}_per_cluster",
            f"{annotation_name}_per_cell",
            annotation_name,
            "accordion_per_cluster",
        ]
        if "celltype" in marker_table_key:
            group_col = _first_existing_column(df, celltype_candidates)
        if group_col is None:
            group_col = _first_existing_column(
                df, ["cluster", "leiden", "Cluster", "clusters"] + celltype_candidates)
    if group_col is None or group_col not in df.columns:
        raise KeyError(
            "Could not identify the grouping column. Pass group_col explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    sub = df[df[group_col].astype(str) == str(group)].copy()
    if sub.empty:
        groups = sorted(df[group_col].astype(str).unique())
        raise ValueError(
            f"No markers for group={group!r} in column {group_col!r}. "
            f"Available groups: {groups}"
        )

    # Auto-detect the score column: the default is the cluster-resolution name,
    # but cell-resolution / cell-type tables use different names.
    if score_col not in sub.columns:
        score_col = _first_existing_column(sub, [
            score_col,
            "gene_impact_score_per_cluster",
            "gene_impact_score_per_celltype_cluster",
            "gene_impact_score_per_celltype_cell",
            "gene_impact_score_per_cell",
            "gene_impact_score",
        ])
    if score_col is None or score_col not in sub.columns:
        raise KeyError(
            "Could not find a gene-impact-score column. Pass score_col explicitly. "
            f"Available columns: {list(sub.columns)}"
        )

    # choose the size column
    if size_col is None:
        size_col = _first_existing_column(sub, ["ECs", "weight"])
    size_title = size_col if size_col else None

    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
    sub = sub.dropna(subset=[marker_col, score_col])
    sub = sub[sub[score_col] > 0]
    if sub.empty:
        raise ValueError(f"No markers with positive impact score for group={group!r}.")

    # sort markers by score for a clean lollipop
    sub = sub.sort_values(score_col, ascending=True)
    markers = sub[marker_col].astype(str).tolist()
    y_pos = np.arange(len(markers))

    # marker type (default all positive)
    if marker_type_col in sub.columns:
        mtype = sub[marker_type_col].astype(str).str.lower().values
    else:
        mtype = np.array(["positive"] * len(sub))

    # SPs -> colour band
    if sps_col in sub.columns:
        bins = [_sps_bin(v) for v in sub[sps_col].values]
    else:
        bins = [(0, "1")] * len(sub)
    colors = []
    band_labels = []
    for (idx, lab), mt in zip(bins, mtype):
        ramp = _SPS_NEG_RAMP if mt.startswith("neg") else _SPS_POS_RAMP
        colors.append(ramp[idx])
        band_labels.append(lab)

    # size mapping from size_col -> point area
    if size_col and size_col in sub.columns:
        sval = pd.to_numeric(sub[size_col], errors="coerce").fillna(1).clip(lower=1, upper=5)
        smin, smax = size_range
        sizes = smin + (sval - 1) / 4.0 * (smax - smin)
        sizes = sizes.values
    else:
        sizes = np.full(len(sub), (size_range[0] + size_range[1]) / 2.0)

    if figsize is None:
        figsize = (8.5, max(3.5, 0.5 * len(markers) + 1.5))
    sns.set_theme(style="whitegrid", context="talk")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.axvline(0, linestyle="--", color="0.6", linewidth=1)
    ax.hlines(y=y_pos, xmin=0, xmax=sub[score_col].values,
              color=colors, linewidth=2.2, zorder=1)
    ax.scatter(sub[score_col].values, y_pos, s=sizes, c=colors,
               edgecolors="black", linewidths=0.6, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(markers, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else str(group),
                 fontsize=16, weight="bold", pad=12)
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)

    # legend 1: SPs colour bands actually present
    present = []
    seen = set()
    for (idx, lab), mt in zip(bins, mtype):
        key = (idx, mt.startswith("neg"))
        if key in seen:
            continue
        seen.add(key)
        ramp = _SPS_NEG_RAMP if mt.startswith("neg") else _SPS_POS_RAMP
        suffix = " (neg)" if mt.startswith("neg") else ""
        present.append((idx, lab + suffix, ramp[idx]))
    present.sort(key=lambda t: t[0])
    sps_handles = [
        ax.scatter([], [], s=120, color=c, edgecolors="black", label=lab)
        for _, lab, c in present
    ]
    leg1 = ax.legend(handles=sps_handles, title=legend_title,
                     bbox_to_anchor=(1.02, 1), loc="upper left",
                     frameon=True, fontsize=11, title_fontsize=12)
    ax.add_artist(leg1)

    # legend 2: size scale
    if size_col and size_col in sub.columns:
        smin, smax = size_range
        size_ticks = [(v, smin + (v - 1) / 4.0 * (smax - smin)) for v in (1, 3, 5)]
        size_handles, size_title2 = _size_legend_handles(ax, size_ticks, size_title)
        ax.legend(handles=size_handles, title=size_title2,
                  bbox_to_anchor=(1.02, 0.45), loc="upper left",
                  frameon=True, fontsize=11, title_fontsize=12)

    fig.tight_layout()
    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


_CONDITION_PALETTE = ["0.5", "#8B1A1A", "#104E8B", "#1b7837", "#762a83", "#e08214"]


def plot_markers_lollipop_by_condition(
        adata,
        group,
        annotation_name: str = "accordion",
        marker_table_key: str = "top_markers_per_celltype_cell",
        condition_col: str = "condition",
        group_col: Optional[str] = None,
        marker_col: str = "marker",
        score_col: Optional[str] = None,
        palette: Optional[Sequence[str]] = None,
        size: float = 70.0,
        title: Optional[str] = None,
        xlabel: str = "Gene impact score",
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
):
    """Markers lollipop for one cell type / signature, **split by condition**.

    Draws one panel per condition (side by side), each showing that condition's
    top markers (markers on the y-axis, gene impact score on the x-axis). Reads
    the cell-resolution table produced by ``accordion_custom(...,
    condition_group_info=...)``, i.e. ``top_markers_per_celltype_cell`` with a
    ``condition`` column.

    title
        Overall figure title (``fig.suptitle``). Defaults to ``str(group)``.
        Each panel is still titled with its own condition name.
    xlabel
        X-axis label repeated on every panel. Default ``"Gene impact score"``.

    Returns
    -------
    fig, axes
    """
    detailed = _get_detailed_info(adata, annotation_name)
    if marker_table_key not in detailed:
        available = ", ".join(sorted(detailed.keys())) if detailed else "none"
        raise KeyError(
            f"Could not find {marker_table_key!r} in adata.uns[{annotation_name!r}]"
            f"['detailed_annotation_info']. Available keys: {available}."
        )
    df = detailed[marker_table_key].copy()
    if condition_col not in df.columns:
        raise KeyError(
            f"No {condition_col!r} column in {marker_table_key!r}. Re-run the "
            "annotation with condition_group_info=<obs column> to split by condition."
        )

    if group_col is None:
        group_col = _first_existing_column(df, [
            f"{annotation_name}_per_cell", f"{annotation_name}_per_cluster",
            annotation_name, "accordion_per_cell",
        ])
    if group_col is None or group_col not in df.columns:
        raise KeyError(
            "Could not identify the cell-type column. Pass group_col explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    if score_col is None or score_col not in df.columns:
        score_col = _first_existing_column(df, [
            "gene_impact_score_per_celltype_cell", "gene_impact_score_per_cell",
            "gene_impact_score_per_celltype_cluster", "gene_impact_score_per_cluster",
            "gene_impact_score",
        ])
    if score_col is None:
        raise KeyError(f"Could not find a score column. Columns: {list(df.columns)}")

    df = df[df[group_col].astype(str) == str(group)].copy()
    if df.empty:
        raise ValueError(f"No rows for group={group!r} in column {group_col!r}.")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[marker_col, score_col])
    df = df[df[score_col] > 0]

    conditions = sorted(df[condition_col].astype(str).unique())
    palette = list(palette) if palette is not None else _CONDITION_PALETTE
    colors = {c: palette[i % len(palette)] for i, c in enumerate(conditions)}

    n = max(1, len(conditions))
    max_markers = max(
        (df[df[condition_col].astype(str) == c][marker_col].nunique() for c in conditions),
        default=1,
    )
    if figsize is None:
        figsize = (max(5, 4.2 * n), max(3.5, 0.45 * max_markers + 1.5))

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, cond in zip(axes, conditions):
        sub = (df[df[condition_col].astype(str) == cond]
               .sort_values(score_col, ascending=True))
        markers = sub[marker_col].astype(str).tolist()
        y_pos = np.arange(len(markers))
        col = colors[cond]
        ax.hlines(y=y_pos, xmin=0, xmax=sub[score_col].values, color=col,
                  linewidth=2.2, zorder=1)
        ax.scatter(sub[score_col].values, y_pos, s=size, color=col,
                   edgecolors="black", linewidths=0.6, zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(markers, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=12, weight="bold")
        ax.set_title(str(cond), fontsize=13, weight="bold")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(title if title is not None else str(group), fontsize=15, weight="bold")
    fig.tight_layout()
    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_celltypes_lollipop(
        adata,
        cluster,
        annotation_name: str = "accordion",
        table_key: str = "top_celltypes",
        celltype_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        score_col: str = "celltype_impact_score",
        percentage_col: str = "percentage",
        color: str = "#8B1A1A",
        size_range: Tuple[float, float] = (60.0, 360.0),
        title: Optional[str] = None,
        xlabel: str = "Cell type impact score",
        ylabel: str = "",
        figsize: Optional[Tuple[float, float]] = None,
        ax=None,
        with_ontology: bool = False,
        ontology_path: Optional[str] = None,
        cl_id_col: Optional[str] = None,
        condition: str = "health",
        ontology_highlight_color: Optional[str] = None,
        ontology_other_color: str = "0.5",
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
):
    """Lollipop plot of the top candidate cell types for a single cluster.

    Candidate cell types are on the y-axis, the cell-type impact score on the
    x-axis, point size encodes the percentage of cells in the cluster assigned
    to that type, and each point is annotated with that percentage.

    Parameters
    ----------
    adata
        AnnData annotated by :func:`pycellmarkeraccordion.accordion` with
        ``include_detailed_annotation_info=True``.
    cluster
        The cluster id to plot.
    percentage_col
        Column with the per-cluster percentage. Defaults to ``"percentage"``.
    with_ontology
        If ``True``, draw the Cell Ontology subtree connecting the candidate
        cell types next to the lollipop, side by side. The candidate cell
        types are highlighted in dark red, their ontology ancestors in grey.
    ontology_path
        Path to a ``.obo`` Cell Ontology file. Defaults to the one packaged with
        pycellmarkeraccordion (``get_default_cell_ontology_path()``).
    cl_id_col
        Column in the table holding Cell Ontology IDs (``CL:xxxxxxx``). If
        ``None`` it is auto-detected, and otherwise candidate names are mapped to
        IDs through the ontology.
    ontology_highlight_color, ontology_other_color
        Colors for the ontology tree's candidate cell types and their
        ancestors (only used when ``with_ontology=True``). ``ontology_highlight_color=None``
        (default) reuses ``color`` so the tree matches the lollipop; pass an
        explicit color to make them independent. ``ontology_other_color``
        defaults to ``"0.5"`` (grey).
    title, xlabel, ylabel
        Text overrides for the plot title (default ``f"Cluster {cluster}"``)
        and axis labels.

    Returns
    -------
    fig, ax
        When ``with_ontology=True``, ``ax`` is a tuple ``(ax_lollipop, ax_tree)``.
    """
    detailed = _get_detailed_info(adata, annotation_name)
    if table_key not in detailed:
        available = ", ".join(sorted(detailed.keys())) if detailed else "none"
        raise KeyError(
            f"Could not find {table_key!r} in adata.uns[{annotation_name!r}]"
            f"['detailed_annotation_info']. Available keys: {available}."
        )

    df = detailed[table_key].copy()
    if df.empty:
        raise ValueError(f"Table {table_key!r} is empty.")

    if celltype_col is None:
        celltype_col = _first_existing_column(df, [
            f"{annotation_name}_per_cluster",
            annotation_name,
            "accordion_per_cluster",
            "CL_celltype",
            "celltype",
        ])
    if cluster_col is None:
        cluster_col = _first_existing_column(df, ["cluster", "leiden", "Cluster", "clusters"])
    if celltype_col is None or cluster_col is None:
        raise KeyError(
            "Could not identify cell-type/cluster columns. Pass celltype_col / "
            f"cluster_col explicitly. Available columns: {list(df.columns)}"
        )

    sub = df[df[cluster_col].astype(str) == str(cluster)].copy()
    if sub.empty:
        clusters = sorted(df[cluster_col].astype(str).unique())
        raise ValueError(
            f"No rows for cluster={cluster!r} in column {cluster_col!r}. "
            f"Available clusters: {clusters}"
        )

    if score_col not in sub.columns:
        raise KeyError(f"Score column {score_col!r} not found. Columns: {list(sub.columns)}")

    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
    sub = sub.dropna(subset=[celltype_col, score_col])
    sub = sub[sub[score_col] > 0]
    if sub.empty:
        raise ValueError(f"No candidate cell types with positive score for cluster={cluster!r}.")

    sub = sub.sort_values(score_col, ascending=True)
    celltypes = sub[celltype_col].astype(str).tolist()
    y_pos = np.arange(len(celltypes))

    has_pct = percentage_col in sub.columns
    if has_pct:
        pct = pd.to_numeric(sub[percentage_col], errors="coerce").fillna(0)
        # percentages may be on a 0-1 or 0-100 scale; normalise to 0-100 for labels
        if pct.max() <= 1.0:
            pct = pct * 100.0
        pmin, pmax = pct.min(), pct.max()
        smin, smax = size_range
        if pmax > pmin:
            sizes = smin + (pct - pmin) / (pmax - pmin) * (smax - smin)
        else:
            sizes = np.full(len(sub), (smin + smax) / 2.0)
        sizes = np.asarray(sizes)
    else:
        sizes = np.full(len(sub), (size_range[0] + size_range[1]) / 2.0)
        pct = None

    sns.set_theme(style="whitegrid", context="talk")
    ax_tree = None
    ontology_layout = None
    if with_ontology:
        # Build the ontology layout *before* sizing the figure: with dynamic,
        # text-width-based spacing (see `_ontology_layout`), the tree's real
        # footprint depends on how many candidates share a level and how long
        # their names are, so the panel/figure size must follow that instead
        # of a fixed guess.
        if cl_id_col is None:
            cl_id_col = _first_existing_column(sub, ["CL_ID", "CL_id", "cl_id"])
        cand_ids = None
        if cl_id_col is not None:
            cand_ids = [str(v) for v in sub[cl_id_col] if isinstance(v, str) and v.strip()]
        ontology_layout = _build_ontology_layout(
            candidate_names=celltypes, candidate_ids=cand_ids,
            ontology_path=ontology_path, condition=condition,
        )
        lollipop_h_in = max(6.5, 0.55 * len(celltypes) + 4.0)
        if figsize is None:
            lollipop_w_in = 5.0
            tree_w_in = max(4.0, min(ontology_layout.get("width_in", 6.0), 14.0))
            tree_h_in = ontology_layout.get("height_in", lollipop_h_in)
            figsize = (lollipop_w_in + tree_w_in + 0.6, max(lollipop_h_in, tree_h_in + 1.0))
            width_ratios = [lollipop_w_in, tree_w_in]
        else:
            width_ratios = [1.1, 1.4]
        if ax is None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_tree = fig.add_subplot(gs[0, 1])
        else:
            ax, ax_tree = ax  # expect a (lollipop, tree) pair
            fig = ax.figure
    else:
        if figsize is None:
            figsize = (8.5, max(3.0, 0.55 * len(celltypes) + 1.5))
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

    ax.axvline(0, linestyle="--", color="0.6", linewidth=1)
    ax.hlines(y=y_pos, xmin=0, xmax=sub[score_col].values, color=color,
              linewidth=2.2, zorder=1)
    ax.scatter(sub[score_col].values, y_pos, s=sizes, color=color,
               edgecolors="black", linewidths=0.6, alpha=0.95, zorder=2)

    if has_pct:
        xmax = float(sub[score_col].max())
        # Offset each % label just past the dot edge using a points offset equal
        # to the dot radius (sqrt(area/pi)), so labels never sit on top of large
        # dots regardless of the data scale.
        for yi, xi, pv, sz in zip(y_pos, sub[score_col].values, pct.values, sizes):
            r_pts = float(np.sqrt(max(float(sz), 1.0) / np.pi))
            ax.annotate(
                f"{pv:.0f}%", xy=(xi, yi),
                xytext=(r_pts + 5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=11, color=color,
            )
        ax.set_xlim(0, xmax * 1.30 + 1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(celltypes, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else f"Cluster {cluster}",
                 fontsize=16, weight="bold", pad=12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)

    if with_ontology:
        _render_ontology_tree(
            ax_tree, ontology_layout,
            highlight_color=ontology_highlight_color if ontology_highlight_color is not None else color,
            other_color=ontology_other_color,
        )

    fig.tight_layout()
    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ((ax, ax_tree) if with_ontology else ax)


# ---------------------------------------------------------------------------
# Cell Ontology tree (standalone panel + the side panel drawn next to the lollipop)
# ---------------------------------------------------------------------------

from functools import lru_cache


@lru_cache(maxsize=4)
def _parse_obo_graph(path: str):
    """Parse a `.obo` ontology into a networkx DiGraph (edges: child -> parent).

    Lightweight parser so we don't depend on the optional ``obonet`` package.
    Each node carries a ``name`` attribute. Cached per file path.
    """
    import networkx as nx

    G = nx.DiGraph()
    id_cur = None
    name_cur = None
    parents: list = []
    in_term = False

    def _flush():
        if id_cur is not None:
            G.add_node(id_cur, name=name_cur or id_cur)
            for parent in parents:
                G.add_edge(id_cur, parent)

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("[Term]"):
                _flush()
                id_cur, name_cur, parents = None, None, []
                in_term = True
                continue
            if line.startswith("["):
                _flush()
                id_cur, name_cur, parents = None, None, []
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id:"):
                id_cur = line[3:].strip()
            elif line.startswith("name:"):
                name_cur = line[5:].strip()
            elif line.startswith("is_a:"):
                # e.g. `is_a: CL:0000084 {is_inferred="true"} ! T cell`
                target = line[5:].split("!")[0].split("{")[0].strip()
                if target:
                    parents.append(target)
        _flush()
    return G


def _ontology_name_to_id(G):
    mapping = {}
    for nid, data in G.nodes(data=True):
        nm = data.get("name")
        if nm:
            mapping.setdefault(str(nm).strip().lower(), nid)
    return mapping


@lru_cache(maxsize=4)
def _marker_db_name_to_id(condition: str = "health"):
    """Map ``CL_celltype`` -> ``CL_ID`` from the packaged marker database.

    Recovers CL IDs directly from the marker database (``accordion_marker``)
    rather than matching cell-type names against the ontology, which is exact.
    Returns an empty dict if the database or the expected columns are missing.
    """
    try:
        from .io import read_marker_database
        df = read_marker_database(condition=condition)
    except Exception:
        return {}
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("cl_celltype")
    id_col = cols.get("cl_id")
    if name_col is None or id_col is None:
        return {}
    mapping = {}
    for nm, cid in zip(df[name_col].astype(str), df[id_col].astype(str)):
        cid = cid.strip()
        if cid and cid.lower() != "nan":
            mapping.setdefault(nm.strip().lower(), cid)
    return mapping


def _minimal_ontology_subgraph(G, ids):
    """Subgraph connecting the candidate terms via their lowest common ancestors.

    Edges in ``G`` point child -> parent. ``nx.descendants(G, n)`` therefore
    returns the ontology *ancestors* (more general terms) of ``n``; the reverse,
    ``nx.ancestors(G, n)``, returns the more *specific* terms below ``n``.

    Returns the induced subgraph over the candidates plus only the ancestors that
    lie between them and their lowest common ancestor(s), pruning everything more
    general than the LCA (e.g. ``leukocyte`` / ``cell``).
    """
    import networkx as nx

    ids = [i for i in ids if i in G]
    if len(ids) <= 1:
        nodes = set(ids)
        for cid in ids:
            nodes |= nx.descendants(G, cid)
        return G.subgraph(nodes).copy()

    anc = {c: (nx.descendants(G, c) | {c}) for c in ids}
    common = set.intersection(*[anc[c] for c in ids])

    if not common:
        # disjoint hierarchies: fall back to the full ancestor closure
        nodes = set(ids)
        for cid in ids:
            nodes |= nx.descendants(G, cid)
        return G.subgraph(nodes).copy()

    # lowest common ancestors: common nodes with no more-specific node also common
    lcas = [c for c in common if not (nx.ancestors(G, c) & common)]

    # everything at the LCA level or more specific than it
    spec_of_lca = set(lcas)
    for lca in lcas:
        spec_of_lca |= nx.ancestors(G, lca)

    # Build the kept node set as the union of the shortest is_a paths from each
    # candidate up to a lowest common ancestor. This guarantees every candidate
    # is actually connected into the subtree (no dangling leaves such as
    # "naive B cell"). Keep all intermediate nodes (no collapse for now).
    keep = set(ids) | set(lcas)
    for c in ids:
        connected = False
        for lca in lcas:
            try:
                keep.update(nx.shortest_path(G, c, lca))
                connected = True
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if not connected:
            keep |= (anc[c] & spec_of_lca)

    return G.subgraph(keep).copy()


def _text_width_in(text, fontsize, weight: str = "normal") -> float:
    """Real rendered width (in inches) of ``text`` at ``fontsize``.

    Measured from actual glyph outlines via :class:`matplotlib.textpath.TextPath`,
    so it needs no live renderer/backend and can be called before any figure
    exists -- unlike ``get_window_extent``, which needs a draw pass.
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    text = str(text)
    if not text:
        return 0.0
    fp = FontProperties(size=fontsize, weight=weight)
    tp = TextPath((0, 0), text, size=fontsize, prop=fp)
    return float(tp.get_extents().width) / 72.0  # points -> inches


def _ontology_layout(H, cand_set=None, name_fontsize=9, id_fontsize=8,
                     min_gap_in=0.35, y_spread_in=1.15):
    """Return a top-down layered layout {node: (x, y)} for a DAG H, in inches.

    Unlike a fixed grid, each node's horizontal footprint is sized from the
    *actual rendered width* of its two label lines (name + CL id), so nodes
    with long names are given enough room and never overlap their neighbours
    in the same layer -- however many nodes end up sharing that layer.

    Edges in H point child -> parent. The most general nodes (out-degree 0)
    sit at the top; depth grows downward.

    Returns
    -------
    pos : dict[node, (x, y)]
    half_width : dict[node, float]
        Half of each node's label footprint (inches), for callers that need
        to compute overall figure/axis extents.
    """
    import networkx as nx

    cand_set = cand_set or set()

    def _half_width(n):
        name = str(H.nodes[n].get("name", n))
        w_name = _text_width_in(name, name_fontsize, "bold" if n in cand_set else "normal")
        w_id = _text_width_in(str(n), id_fontsize, "normal")
        return max(w_name, w_id) / 2.0 + 0.10

    Rv = H.reverse(copy=True)  # parent -> child
    depth = {}
    for n in nx.topological_sort(Rv):
        preds = list(Rv.predecessors(n))  # parents of n in H
        depth[n] = 0 if not preds else max(depth[p] for p in preds) + 1

    layers = {}
    for n, d in depth.items():
        layers.setdefault(d, []).append(n)

    half_width = {n: _half_width(n) for n in H.nodes()}
    pos = {}
    for d in sorted(layers):
        nodes = layers[d]
        if d == 0:
            nodes = sorted(nodes)
        else:
            # order each layer by the mean x of its parents to keep children
            # under their parents and reduce edge crossings (barycenter pass).
            def _bary(n):
                parents = [p for p in H.successors(n) if p in pos]
                return np.mean([pos[p][0] for p in parents]) if parents else 0.0
            nodes = sorted(nodes, key=_bary)

        xs = []
        cur = 0.0
        for i, n in enumerate(nodes):
            if i > 0:
                cur += half_width[nodes[i - 1]] + half_width[n] + min_gap_in
            xs.append(cur)
        if xs:
            offset = (xs[0] + xs[-1]) / 2.0
            xs = [x - offset for x in xs]
        for n, x in zip(nodes, xs):
            pos[n] = (x, -float(d) * y_spread_in)
    return pos, half_width


def _ontology_layout_extent(pos, half_width, y_pad_in=0.6):
    """Overall (width_in, height_in) spanned by an ``_ontology_layout`` result."""
    if not pos:
        return 4.0, 3.0
    min_x = min(x - half_width.get(n, 0.6) for n, (x, _y) in pos.items())
    max_x = max(x + half_width.get(n, 0.6) for n, (x, _y) in pos.items())
    ys = [y for _, y in pos.values()]
    width_in = (max_x - min_x) + 0.6
    height_in = (max(ys) - min(ys)) + 2 * y_pad_in
    return width_in, height_in


def _build_ontology_layout(candidate_names, candidate_ids=None,
                           ontology_path=None, condition="health"):
    """Resolve candidates to CL ids, build the minimal connecting subtree and
    lay it out, all before any figure/axes exist.

    Returns a dict with keys ``H``, ``pos``, ``half_width``, ``cand_set``,
    ``width_in``, ``height_in``; or ``{"error": <message>}`` if the ontology
    file or the candidate mapping isn't available, so callers can size their
    figure correctly *before* drawing.
    """
    if ontology_path is None:
        from .io import get_default_cell_ontology_path
        ontology_path = get_default_cell_ontology_path()

    try:
        G = _parse_obo_graph(str(ontology_path))
    except Exception as exc:  # pragma: no cover - depends on packaged file
        return {"error": f"Ontology not available:\n{exc}"}

    # Resolve candidate CL IDs: prefer the exact CL_celltype -> CL_ID mapping
    # from the packaged marker database, then fall back to matching cell-type
    # names against the ontology term names.
    db_map = _marker_db_name_to_id(condition)
    obo_map = _ontology_name_to_id(G)

    ids = []
    if candidate_ids:
        ids = [i for i in candidate_ids if i in G]
    if not ids:
        for nm in candidate_names:
            key = str(nm).strip().lower()
            cid = db_map.get(key) or obo_map.get(key)
            if cid is not None and cid in G:
                ids.append(cid)
    ids = list(dict.fromkeys(ids))  # de-duplicate, keep order

    if len(ids) == 0:
        return {"error": "No candidate cell types\nmapped to the ontology."}

    # Minimal connecting subtree: keep the candidates and only the ancestors
    # needed to connect them, rooted at their lowest common ancestor(s) --
    # not the full path up to "cell".
    H = _minimal_ontology_subgraph(G, ids)
    cand_set = set(ids)
    pos, half_width = _ontology_layout(H, cand_set=cand_set)
    width_in, height_in = _ontology_layout_extent(pos, half_width)
    return {
        "H": H, "pos": pos, "half_width": half_width, "cand_set": cand_set,
        "width_in": width_in, "height_in": height_in,
    }


def _render_ontology_tree(ax, layout, highlight_color="#8B1A1A", other_color="0.5"):
    """Draw a layout previously built by ``_build_ontology_layout`` onto ``ax``."""
    if "error" in layout:
        ax.axis("off")
        ax.text(0.5, 0.5, layout["error"], ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
        return ax

    H, pos = layout["H"], layout["pos"]
    half_width, cand_set = layout["half_width"], layout["cand_set"]

    # Plain coloured text labels (no boxes) and thin arrows that point
    # downward, from the general parent to the more specific child.
    # H edges go child -> parent, so the arrow head sits at the child (u).
    for u, v in H.edges():
        if u in pos and v in pos:
            (xc, yc), (xp, yp) = pos[u], pos[v]
            ann = ax.annotate(
                "", xy=(xc, yc), xytext=(xp, yp),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1,
                                shrinkA=18, shrinkB=18),
                annotation_clip=False, zorder=1,
            )
            # arrows are FancyArrowPatch artists that clip to the axes by
            # default; disable so edges to extreme-depth nodes are never cut off.
            if getattr(ann, "arrow_patch", None) is not None:
                ann.arrow_patch.set_clip_on(False)

    for n, (x, y) in pos.items():
        is_cand = n in cand_set
        col = highlight_color if is_cand else other_color
        name = H.nodes[n].get("name", n)
        ax.text(x, y + 0.16, str(name), ha="center", va="center",
                fontsize=9, color=col, fontweight="bold" if is_cand else "normal",
                zorder=2)
        ax.text(x, y - 0.16, str(n), ha="center", va="center",
                fontsize=8, color=col, zorder=2)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        max_hw = max(half_width.values()) if half_width else 0.6
        ax.set_xlim(min(xs) - max_hw - 0.3, max(xs) + max_hw + 0.3)
        ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.axis("off")
    return ax


def plot_celltype_ontology_tree(
        adata,
        cluster,
        annotation_name: str = "accordion",
        table_key: str = "top_celltypes",
        celltype_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        cl_id_col: Optional[str] = None,
        ontology_path: Optional[str] = None,
        condition: str = "health",
        title: Optional[str] = None,
        highlight_color: str = "#8B1A1A",
        other_color: str = "0.5",
        figsize: Optional[Tuple[float, float]] = None,
        ax=None,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
):
    """Standalone Cell Ontology subtree for the candidate cell types of a cluster.

    The same ontology panel drawn next to the cell-type lollipop in
    ``plot_celltypes_lollipop(with_ontology=True)``, usable standalone.
    Candidate cell types are highlighted in dark red and their ontology
    ancestors in grey.

    title
        Plot title. Defaults to ``f"Cell Ontology — cluster {cluster}"``.
    highlight_color, other_color
        Colors for candidate cell types and their ontology ancestors.
        Defaults ``"#8B1A1A"`` (dark red) / ``"0.5"`` (grey).
    figsize
        Figure size in inches. When ``None`` (default), it is computed
        automatically from the actual rendered width/height of the tree's
        labels, so wide or deep trees (many candidates, long cell-type names)
        get a bigger canvas instead of squeezing their labels together.

    Returns
    -------
    fig, ax
    """
    detailed = _get_detailed_info(adata, annotation_name)
    if table_key not in detailed:
        available = ", ".join(sorted(detailed.keys())) if detailed else "none"
        raise KeyError(
            f"Could not find {table_key!r} in adata.uns[{annotation_name!r}]"
            f"['detailed_annotation_info']. Available keys: {available}."
        )
    df = detailed[table_key].copy()
    if df.empty:
        raise ValueError(f"Table {table_key!r} is empty.")

    if celltype_col is None:
        celltype_col = _first_existing_column(df, [
            f"{annotation_name}_per_cluster", annotation_name,
            "accordion_per_cluster", "CL_celltype", "celltype",
        ])
    if cluster_col is None:
        cluster_col = _first_existing_column(df, ["cluster", "leiden", "Cluster", "clusters"])
    if celltype_col is None or cluster_col is None:
        raise KeyError(
            "Could not identify cell-type/cluster columns. Pass them explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    sub = df[df[cluster_col].astype(str) == str(cluster)].copy()
    if sub.empty:
        clusters = sorted(df[cluster_col].astype(str).unique())
        raise ValueError(f"No rows for cluster={cluster!r}. Available: {clusters}")

    names = sub[celltype_col].astype(str).tolist()
    if cl_id_col is None:
        cl_id_col = _first_existing_column(sub, ["CL_ID", "CL_id", "cl_id"])
    cand_ids = None
    if cl_id_col is not None:
        cand_ids = [str(v) for v in sub[cl_id_col] if isinstance(v, str) and v.strip()]

    layout = _build_ontology_layout(candidate_names=names, candidate_ids=cand_ids,
                                    ontology_path=ontology_path, condition=condition)

    sns.set_theme(style="white", context="talk")
    if ax is None:
        if figsize is None:
            w_in = layout.get("width_in", 6.0)
            h_in = layout.get("height_in", 7.5)
            figsize = (max(5.0, min(w_in, 16.0)), max(3.5, min(h_in + 0.8, 16.0)))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _render_ontology_tree(ax, layout, highlight_color=highlight_color, other_color=other_color)
    ax.set_title(title if title is not None else f"Cell Ontology — cluster {cluster}",
                 fontsize=14, weight="bold", pad=10)

    fig.tight_layout()
    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Spatial scatter-pie (per-spot composition after accordion_spatial_deconvolution)
# ---------------------------------------------------------------------------

def plot_spatial_scatterpie(
        adata,
        annotation_name: str = "accordion",
        proportions_key: Optional[str] = None,
        celltypes: Optional[Sequence[str]] = None,
        palette: Optional[Mapping[str, str]] = None,
        color_key: Optional[str] = None,
        radius: Optional[float] = None,
        radius_scale: float = 0.42,
        min_slice_percentage: float = 0.0,
        show_empty_spots: bool = False,
        background_color: str = "#D9D9D9",
        edgecolor: str = "white",
        linewidth: float = 0.3,
        show_legend: bool = True,
        legend_fontsize: int = 9,
        legend_ncol: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        ax=None,
        title: str = "Spatial cell-type composition",
        xlabel: str = "Spatial X",
        ylabel: str = "Spatial Y",
        legend_title: str = "Cell type",
        invert_yaxis: bool = True,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
):
    """Scatter-pie plot of per-spot cell-type composition.

    Draws one pie chart per spot, positioned at its spatial coordinates
    (``adata.obsm["spatial"]``), with slices sized by the estimated cell-type
    composition from :func:`pycellmarkeraccordion.accordion_spatial_deconvolution`
    (``adata.obsm[f"{annotation_name}_proportions"]``). Spots whose composition
    sums to zero -- either every candidate was dropped by
    ``deconvolution_threshold``, or no candidate ever scored positively for
    that spot in the first place (see
    :func:`pycellmarkeraccordion.accordion_spatial_deconvolution`) -- are, by
    default, simply not drawn at all (``show_empty_spots=False``): no pie, no
    placeholder circle, just blank space at that spot's coordinates. Pass
    ``show_empty_spots=True`` to instead draw them as plain
    ``background_color`` circles with their own "No dominant cell type"
    legend entry (showing the count), if you'd rather see where they are.

    Parameters
    ----------
    adata
        Spatial AnnData annotated by
        :func:`pycellmarkeraccordion.accordion_spatial_deconvolution`, with
        ``adata.obsm["spatial"]`` coordinates.
    annotation_name
        Prefix used when running
        :func:`pycellmarkeraccordion.accordion_spatial_deconvolution`. Default
        ``"accordion"``.
    proportions_key
        ``adata.obsm`` key holding the per-spot composition table. Defaults to
        ``f"{annotation_name}_proportions"``.
    celltypes
        Restrict the pies (and legend) to these cell types; the remaining
        columns of the composition table are ignored. ``None`` uses every
        column.
    palette
        Optional explicit mapping ``{cell_type: color}``. Missing entries fall
        back to the auto-generated palette below.
    color_key
        ``adata.obs`` column whose Scanpy/UMAP colors should be reused for the
        cell-type palette (e.g. ``"leiden"``). Defaults to
        ``f"{annotation_name}_per_cell"``. Falls back to a generated palette
        (see ``_fallback_palette``) for any cell type without a matching
        color -- distinct colors even with dozens of cell types.
    radius
        Pie radius, in the same units as ``adata.obsm["spatial"]``. If
        ``None``, set automatically to ``radius_scale`` times the median
        nearest-neighbour distance between spots.
    radius_scale
        Fraction of the median spot spacing used for the automatic radius.
        Default ``0.42`` (pies just short of touching their neighbours).
    min_slice_percentage
        Drop slices below this percentage (0-100) of a spot's composition, for
        a visually cleaner pie; the remaining slices are renormalised to sum
        to 1. Default ``0`` (keep every slice).
    show_empty_spots
        Whether to draw spots with an empty (all-zero) composition at all.
        Default ``False`` -- they're left blank, matching the surrounding
        empty space, rather than drawn as a placeholder circle. Set to
        ``True`` to draw them in ``background_color`` instead, with a
        dedicated legend entry.
    background_color
        Fill color for spots with an empty composition, when
        ``show_empty_spots=True``. Default ``"#D9D9D9"``.
    edgecolor, linewidth
        Outline color / width for every pie slice. Default ``"white"`` / ``0.3``.
    show_legend, legend_fontsize, legend_ncol, legend_title
        Cell-type color legend controls. The legend is drawn as a wide,
        multi-column grid *below* the plot (not a single tall column to its
        side), since real spatial deconvolution runs routinely surface
        dozens of candidate cell types. ``legend_ncol=None`` (default) picks
        the column count automatically from the figure width and the actual
        rendered width of the longest cell-type name; pass an int to force a
        specific column count. ``legend_title`` defaults to ``"Cell type"``.
    figsize
        Matplotlib figure size; chosen automatically when ``None`` (a square
        base plus enough extra height for the legend grid below it).
    ax
        Existing axes to draw on; a new figure is created when ``None``.
    title, xlabel, ylabel
        Text overrides for the plot title and axis labels (defaults
        ``"Spatial X"`` / ``"Spatial Y"``).
    invert_yaxis
        ``adata.obsm["spatial"]`` uses image coordinates (y increases
        downward); invert the y-axis so spots are shown in the same
        orientation as ``sc.pl.spatial``. Default ``True``.
    show, save, dpi
        Standard display / export options.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Examples
    --------
    >>> adata_sp = accordion_spatial_deconvolution(adata_sp, n_top_celltypes=5)
    >>> fig, ax = plot_spatial_scatterpie(adata_sp)
    """
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from scipy.spatial import cKDTree

    if "spatial" not in adata.obsm:
        raise KeyError(
            'adata.obsm["spatial"] not found. plot_spatial_scatterpie requires '
            "spatial coordinates (as produced by squidpy/scanpy spatial readers)."
        )

    proportions_key = proportions_key or f"{annotation_name}_proportions"
    if proportions_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{proportions_key!r}] not found. Run "
            "accordion_spatial_deconvolution first -- it stores the per-spot "
            "composition there."
        )

    comp = adata.obsm[proportions_key]
    if not isinstance(comp, pd.DataFrame):
        comp = pd.DataFrame(comp, index=adata.obs_names)
    comp = comp.reindex(adata.obs_names).fillna(0.0)
    if celltypes is not None:
        keep = [str(c) for c in celltypes if str(c) in comp.columns]
        if not keep:
            raise KeyError(
                f"None of {list(celltypes)} found in adata.obsm[{proportions_key!r}]. "
                f"Available: {list(comp.columns)}"
            )
        comp = comp[keep]
    celltype_names = [str(c) for c in comp.columns]

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.shape[0] != comp.shape[0]:
        raise ValueError(
            "adata.obsm['spatial'] and the composition table "
            f"(adata.obsm[{proportions_key!r}]) have a different number of spots."
        )

    # Palette: reuse Scanpy colors when possible, else the generated fallback
    # (distinct colors even for dozens of cell types, see _fallback_palette).
    resolved_palette = dict(palette) if palette else {}
    color_key = color_key or f"{annotation_name}_per_cell"
    mapping = _scanpy_color_mapping(adata, color_key, celltype_names)
    for ct in celltype_names:
        if ct not in resolved_palette and ct in mapping:
            resolved_palette[ct] = mapping[ct]
    missing = [ct for ct in celltype_names if ct not in resolved_palette]
    if missing:
        resolved_palette.update(_fallback_palette(missing))

    values = comp.to_numpy(dtype=float)
    if min_slice_percentage > 0:
        values = np.where(values * 100.0 < min_slice_percentage, 0.0, values)
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)

    # Automatic radius: a fraction of the median nearest-neighbour spacing.
    if radius is None:
        if coords.shape[0] > 1:
            tree = cKDTree(coords)
            nn_dist, _ = tree.query(coords, k=2)
            neighbour_dist = nn_dist[:, 1]
            neighbour_dist = neighbour_dist[neighbour_dist > 0]
            spacing = float(np.median(neighbour_dist)) if neighbour_dist.size else 1.0
        else:
            spacing = 1.0
        radius = radius_scale * spacing

    n_items = len(celltype_names)
    # Legend layout: a wide grid *below* the plot instead of one long single
    # column to its side -- real spatial deconvolution runs routinely surface
    # dozens of candidate cell types, and a single-column list of them ends
    # up far taller than the plot itself for no extra information over a
    # compact multi-column block.
    # Use a high percentile (not the max) of label width to size columns: real
    # ontology names are wildly uneven in length (e.g. "B cell" vs. "CD34-
    # positive, CD56-positive, CD117-positive common innate lymphoid
    # precursor, human"), and sizing every column off the single longest
    # label would collapse the whole grid back down to one column. A handful
    # of unusually long rows just end up a bit wider than their neighbours,
    # which is a fine trade for keeping everything else compact.
    _widths = [_text_width_in(str(ct), legend_fontsize) for ct in celltype_names] or [1.0]
    item_w_in = min(float(np.percentile(_widths, 75)) + 0.45, 3.2)
    row_h_in = (legend_fontsize + 4) / 72.0 * 1.6

    auto_figsize = figsize is None
    base_w, base_h = figsize if figsize is not None else (8.0, 8.0)
    legend_h_in = 0.0
    ncol = legend_ncol or 1
    if show_legend and n_items:
        if legend_ncol is not None:
            ncol = legend_ncol
        elif auto_figsize:
            # Pick columns so the legend grid is roughly as tall as the plot
            # itself (not a fixed 8in-wide guess): with dozens of long cell
            # -type names that means growing the figure sideways instead of
            # letting the legend tower over the map, capped so it never gets
            # absurdly wide. Only applies when we're also choosing figsize --
            # an explicit user figsize is left alone, so ncol is capped to
            # actually fit inside it.
            max_rows = max(6, int(base_h / row_h_in))
            ncol = max(1, -(-n_items // max_rows))  # ceil(n_items / max_rows)
            fig_w_for_legend = min(20.0, ncol * item_w_in)
            base_w = max(base_w, fig_w_for_legend)
            ncol = max(1, min(n_items, int(base_w / item_w_in)))
        else:
            ncol = max(1, min(n_items, int(base_w / item_w_in)))
        nrow = -(-n_items // ncol)  # ceil division
        legend_h_in = nrow * row_h_in + 0.5

    sns.set_theme(style="white", context="talk")
    if figsize is None:
        figsize = (base_w, base_h + legend_h_in)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    wedges = []
    colors = []
    n_empty_spots = 0
    for (x, y), row in zip(coords, values):
        total = row.sum()
        if total <= 0:
            n_empty_spots += 1
            if show_empty_spots:
                wedges.append(Wedge((x, y), radius, 0, 360))
                colors.append(background_color)
            continue
        theta0 = 0.0
        for ct, frac in zip(celltype_names, row):
            if frac <= 0:
                continue
            theta1 = theta0 + frac * 360.0
            wedges.append(Wedge((x, y), radius, theta0, theta1))
            colors.append(resolved_palette.get(ct, "#999999"))
            theta0 = theta1

    pc = PatchCollection(wedges, facecolor=colors, edgecolor=edgecolor, linewidths=linewidth)
    ax.add_collection(pc)

    pad = radius * 2.0
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.set_aspect("equal")
    # Keep the axes box + tick labels (like plot_spatial_entropy / sc.pl.spatial),
    # instead of stripping the frame -- consistent framing across the spatial plots.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if invert_yaxis and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title(title, fontsize=16, weight="bold", pad=12)

    if show_legend and n_items:
        handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=resolved_palette.get(ct, "#999999"),
                   markeredgecolor="none", markersize=10, label=ct)
            for ct in celltype_names
        ]
        if show_empty_spots and n_empty_spots > 0:
            # Explain the gray/background_color spots directly in the legend
            # (rather than leaving the reader to guess). They're spots whose
            # whole composition summed to <= 0 -- either every candidate cell
            # type was dropped by accordion_spatial_deconvolution's
            # deconvolution_threshold, or (even at deconvolution_threshold=0)
            # no candidate ever scored positively for that spot in the first
            # place (e.g. very low-count/background spots with no marker
            # signal). Either way, they're spots with no confident call, not
            # missing/dropped spots. Only added when show_empty_spots=True --
            # otherwise these spots aren't drawn at all, so there's nothing
            # in the legend to explain.
            handles.append(
                Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=background_color, markeredgecolor="none",
                       markersize=10, label=f"No dominant cell type (n={n_empty_spots})")
            )
        # Figure-level legend (not ax.legend) so the multi-column grid spans
        # the full figure width below the plot rather than being squeezed
        # into the scatter-pie axes. Its height was only *estimated* above
        # (to size the figure/columns); draw it, then measure its *actual*
        # rendered height and reserve exactly that below (see the next
        # comment for what happens when the estimate undershot).
        #
        # Center it on the axes' actual horizontal center, not a hardcoded
        # 0.5 of the whole figure: matplotlib's default axes rect isn't
        # perfectly centered in the figure (e.g. left=0.125/right=0.9 gives
        # an axes center of 0.5125, not 0.5), so anchoring the legend at a
        # flat 0.5 left its (now-centered) title visibly off from "Spatial
        # X", which is centered on the axes itself.
        fig.canvas.draw()
        ax_pos = ax.get_position()
        ax_center_x = 0.5 * (ax_pos.x0 + ax_pos.x1)
        legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(ax_center_x, 0.0),
                            ncol=ncol, fontsize=legend_fontsize, frameon=False, title=legend_title,
                            borderaxespad=0.1, alignment="center")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_h_actual_in = legend.get_window_extent(renderer).height / fig.dpi

        # The legend is drawn at its natural size regardless of what we
        # reserve via subplots_adjust below -- it doesn't shrink to fit. Our
        # figure height above was sized from `legend_h_in`, a row-count
        # *estimate*; with dozens of cell types (and therefore dozens of
        # legend rows packed into relatively few columns) the real rendered
        # legend can end up taller than that estimate. Previously the
        # reserved bottom margin was capped at a fixed 0.85 fraction to avoid
        # `bottom >= top`, but capping the *reserved space* without the
        # legend actually being smaller just made it spill upward past the
        # cap and collide with the axes/xlabel above it. Instead, grow the
        # figure to fit the *measured* legend height whenever it exceeds the
        # estimate, then size the margin off that same real measurement --
        # no collision regardless of how far off the row-count guess was.
        if auto_figsize and legend_h_actual_in > legend_h_in:
            fig.set_size_inches(fig.get_figwidth(), base_h + legend_h_actual_in + 0.3, forward=True)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

        bbox_fig = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        fig.subplots_adjust(bottom=min(0.9, bbox_fig.height + 0.01), top=0.94)
    else:
        fig.tight_layout()

    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
