from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io import read_marker_database, normalize_condition


def _as_list_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_bulk_palette(labels, manual_palette: Optional[Mapping[str, str]]) -> dict:
    """Generated fallback palette, with any ``manual_palette`` entries overriding it."""
    from .plots import _fallback_palette
    fallback = _fallback_palette(labels)
    if not manual_palette:
        return fallback
    manual = {str(k): v for k, v in manual_palette.items()}
    return {lab: manual.get(lab, fallback[lab]) for lab in labels}


def _read_bulk_expression(bulk, genes_on_rows: bool = True) -> pd.DataFrame:
    """Read or normalize bulk expression input to genes × samples DataFrame.

    Accepted inputs
    ---------------
    - pandas DataFrame
    - pandas Series
    - path to .csv/.tsv/.txt

    The returned DataFrame always has genes on rows and samples on columns.
    """
    if isinstance(bulk, pd.Series):
        df = bulk.to_frame(name=bulk.name or "bulk_sample")
    elif isinstance(bulk, pd.DataFrame):
        df = bulk.copy()
    elif isinstance(bulk, (str, Path)):
        path = Path(bulk)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, index_col=0)
        elif suffix in {".tsv", ".txt"}:
            df = pd.read_csv(path, sep="\t", index_col=0)
        else:
            raise ValueError("bulk path must be .csv, .tsv, or .txt")
    else:
        raise TypeError("bulk must be a pandas DataFrame, pandas Series, or path to .csv/.tsv/.txt")

    if not genes_on_rows:
        df = df.T

    # Ensure numeric expression values.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("Bulk expression matrix is empty after numeric conversion.")

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _prepare_marker_database(
        database=None,
        condition: str = "health",
        species: str | Iterable[str] = "Human",
        tissue=None,
        disease=None,
        celltypes=None,
        NCIT_celltypes=None,
        log2FC_threshold=None,
        evidence_consistency_score_threshold=None,
        specificity_score_threshold=None,
        min_n_marker=5,
        max_n_marker=30,
        combined_score_quantile_threshold=None) -> pd.DataFrame:
    """Load and filter marker database for bulk deconvolution."""
    condition = normalize_condition(condition)
    marker = read_marker_database(database=database, condition=condition).copy()

    if condition == "disease":
        if NCIT_celltypes is not None and celltypes is None:
            celltypes = NCIT_celltypes
        if disease is not None and "DO_diseasetype" in marker.columns:
            vals = {str(x).upper() for x in _as_list_or_none(disease)}
            filt = marker["DO_diseasetype"].astype(str).str.upper().isin(vals)
            if filt.any():
                marker = marker[filt].copy()
            else:
                print("WARNING: disease not found. The entire disease Accordion database is considered")

    # Species filter
    species_list = _as_list_or_none(species)
    if species_list is not None and "species" in marker.columns:
        marker = marker[marker["species"].isin(species_list)].copy()

    # Tissue filter
    tissue_list = _as_list_or_none(tissue)
    if tissue_list is not None and "Uberon_tissue" in marker.columns:
        tissue_lower = {str(t).lower() for t in tissue_list}
        filt = marker["Uberon_tissue"].astype(str).str.lower().isin(tissue_lower)
        if filt.any():
            marker = marker[filt].copy()
        else:
            print("WARNING: tissue not found. The entire marker database is considered")

    # Cell type filter
    celltypes_list = _as_list_or_none(celltypes)
    if celltypes_list is not None:
        vals = {str(x) for x in celltypes_list}
        marker = marker[marker["CL_celltype"].astype(str).isin(vals)].copy()

    # log2FC filter
    if log2FC_threshold is not None and "log2FC" in marker.columns:
        marker = marker[marker["log2FC"].isna() | (marker["log2FC"] >= log2FC_threshold)].copy()

    # Build ECs if needed
    if "ECs" not in marker.columns:
        if "ECs_global" in marker.columns:
            marker["ECs"] = marker["ECs_global"]
        elif "ECs_NCIT_global" in marker.columns:
            marker["ECs"] = marker["ECs_NCIT_global"]
        else:
            marker["ECs"] = 1

    marker["ECs"] = pd.to_numeric(marker["ECs"], errors="coerce").fillna(1)

    if evidence_consistency_score_threshold is not None:
        marker = marker[marker["ECs"] >= evidence_consistency_score_threshold].copy()

    # Specificity score: lower marker reuse is better.
    if "SPs" not in marker.columns:
        sp = (
            marker.groupby(["marker", "marker_type"], observed=True)
            .size()
            .reset_index(name="SPs_raw")
        )
        marker = marker.merge(sp, on=["marker", "marker_type"], how="left")
        marker["SPs"] = 1.0 / marker["SPs_raw"].clip(lower=1)
    else:
        marker["SPs"] = pd.to_numeric(marker["SPs"], errors="coerce").fillna(1)

    if specificity_score_threshold is not None:
        marker = marker[marker["SPs"] <= specificity_score_threshold].copy()

    # Regularized scores similar in spirit to Accordion.
    marker["ECs_reg"] = np.log10(marker["ECs"].clip(lower=1)) + 1
    marker["SPs_reg"] = np.log10(pd.to_numeric(marker["SPs"], errors="coerce").fillna(1).clip(lower=1e-12)) + 1
    # Avoid negative/zero weights if SPs are fractions.
    marker["SPs_reg"] = marker["SPs_reg"].abs().clip(lower=1e-6)
    marker["combined_score"] = marker["ECs_reg"] * marker["SPs_reg"]

    if combined_score_quantile_threshold is not None:
        q = marker.groupby("CL_celltype", observed=True)["combined_score"].transform(
            lambda s: s.quantile(combined_score_quantile_threshold)
        )
        marker = marker[marker["combined_score"] > q].copy()

    # Minimum/maximum markers per cell type
    marker["n_marker_celltype"] = marker.groupby("CL_celltype", observed=True)["marker"].transform("count")
    if min_n_marker is not None:
        marker = marker[marker["n_marker_celltype"] >= min_n_marker].copy()

    if max_n_marker is not None:
        marker = (
            marker.sort_values("combined_score", ascending=False)
            .groupby("CL_celltype", observed=True)
            .head(max_n_marker)
            .copy()
        )

    marker = marker.drop_duplicates(subset=["CL_celltype", "marker", "marker_type"])
    if marker.empty:
        raise ValueError("Marker database is empty after filtering.")
    return marker


def _scale_bulk_expression(expr: pd.DataFrame, log1p: bool = False, scale: str = "sample_zscore") -> pd.DataFrame:
    """Scale bulk expression for marker scoring."""
    X = expr.copy()

    if log1p:
        X = np.log1p(X.clip(lower=0))

    if scale == "none":
        return X

    if scale == "sample_zscore":
        # For each sample, z-score genes. This works for single-sample bulk too.
        mean = X.mean(axis=0)
        std = X.std(axis=0).replace(0, np.nan)
        return ((X - mean) / std).fillna(0)

    if scale == "gene_zscore":
        # Across samples for each gene. Useful for cohorts, not single-sample.
        mean = X.mean(axis=1)
        std = X.std(axis=1).replace(0, np.nan)
        return X.sub(mean, axis=0).div(std, axis=0).fillna(0)

    if scale == "minmax_sample":
        minv = X.min(axis=0)
        maxv = X.max(axis=0)
        return X.sub(minv, axis=1).div((maxv - minv).replace(0, np.nan), axis=1).fillna(0)

    raise ValueError("scale must be one of: 'sample_zscore', 'gene_zscore', 'minmax_sample', 'none'")


def accordion_bulk_deconvolution(
        bulk,
        genes_on_rows: bool = True,
        database=None,
        condition: str = "health",
        disease=None,
        celltypes=None,
        NCIT_celltypes=None,
        species: str | Iterable[str] = "Human",
        tissue=None,
        log1p: bool = False,
        scale: str = "sample_zscore",
        min_n_marker: int | None = 5,
        max_n_marker: int | None = 30,
        evidence_consistency_score_threshold=None,
        specificity_score_threshold=None,
        log2FC_threshold=None,
        combined_score_quantile_threshold=None,
        top_n_celltypes: int | None = None,
        score_transform: str = "positive_normalize",
        temperature: float = 1.0,
        return_scores: bool = False):
    """Infer cell-type composition from bulk expression using Accordion markers.

    Parameters
    ----------
    bulk
        Bulk expression matrix. Genes × samples by default.
    genes_on_rows
        If False, input is treated as samples × genes and transposed.
    score_transform
        How to convert raw scores to proportions:
        - "positive_normalize": set negative scores to zero, normalize to sum 1
        - "softmax": softmax over raw scores
    return_scores
        If True, returns a dict with composition, raw scores, marker table, and scaled expression.

    Returns
    -------
    pandas.DataFrame or dict
        Composition DataFrame with cell types as rows and samples as columns.
    """
    expr = _read_bulk_expression(bulk, genes_on_rows=genes_on_rows)
    marker = _prepare_marker_database(
        database=database,
        condition=condition,
        species=species,
        tissue=tissue,
        disease=disease,
        celltypes=celltypes,
        NCIT_celltypes=NCIT_celltypes,
        log2FC_threshold=log2FC_threshold,
        evidence_consistency_score_threshold=evidence_consistency_score_threshold,
        specificity_score_threshold=specificity_score_threshold,
        min_n_marker=min_n_marker,
        max_n_marker=max_n_marker,
        combined_score_quantile_threshold=combined_score_quantile_threshold,
    )

    common_markers = marker["marker"].astype(str).isin(expr.index.astype(str))
    marker = marker[common_markers].copy()
    if marker.empty:
        raise ValueError("No marker genes were detected in the bulk expression data.")

    expr_scaled = _scale_bulk_expression(expr, log1p=log1p, scale=scale)

    scores = {}
    for celltype, ct in marker.groupby("CL_celltype", observed=True):
        pos = ct[ct["marker_type"] == "positive"]
        neg = ct[ct["marker_type"] == "negative"]

        score = pd.Series(0.0, index=expr_scaled.columns)

        if not pos.empty:
            pos_markers = [m for m in pos["marker"].astype(str) if m in expr_scaled.index]
            if pos_markers:
                weights = pos.drop_duplicates("marker").set_index("marker")["combined_score"].reindex(pos_markers)
                norm = np.sqrt(np.sum(pos.drop_duplicates("marker").set_index("marker").reindex(pos_markers)["ECs_reg"] *
                                      pos.drop_duplicates("marker").set_index("marker").reindex(pos_markers)["SPs_reg"]))
                norm = norm if norm and norm > 0 else 1.0
                score = score + expr_scaled.loc[pos_markers].mul(weights, axis=0).sum(axis=0) / norm

        if not neg.empty:
            neg_markers = [m for m in neg["marker"].astype(str) if m in expr_scaled.index]
            if neg_markers:
                weights = neg.drop_duplicates("marker").set_index("marker")["combined_score"].reindex(neg_markers)
                norm = np.sqrt(np.sum(neg.drop_duplicates("marker").set_index("marker").reindex(neg_markers)["ECs_reg"] *
                                      neg.drop_duplicates("marker").set_index("marker").reindex(neg_markers)["SPs_reg"]))
                norm = norm if norm and norm > 0 else 1.0
                score = score - expr_scaled.loc[neg_markers].mul(weights, axis=0).sum(axis=0) / norm

        scores[str(celltype)] = score

    raw_scores = pd.DataFrame(scores).T
    raw_scores.index.name = "celltype"

    if top_n_celltypes is not None:
        keep = raw_scores.mean(axis=1).sort_values(ascending=False).head(top_n_celltypes).index
        raw_scores = raw_scores.loc[keep]

    if score_transform == "positive_normalize":
        comp = raw_scores.clip(lower=0)
        sums = comp.sum(axis=0).replace(0, np.nan)
        comp = comp.div(sums, axis=1).fillna(0)
    elif score_transform == "softmax":
        temp = max(float(temperature), 1e-9)
        Z = raw_scores / temp
        Z = Z.sub(Z.max(axis=0), axis=1)
        expZ = np.exp(Z)
        comp = expZ.div(expZ.sum(axis=0), axis=1)
    else:
        raise ValueError("score_transform must be 'positive_normalize' or 'softmax'")

    comp.index.name = "celltype"

    if return_scores:
        return {
            "composition": comp,
            "raw_scores": raw_scores,
            "marker_table": marker,
            "scaled_expression": expr_scaled,
        }

    return comp


def plot_bulk_composition(
        composition: pd.DataFrame,
        kind: str = "stacked_bar",
        top_n: int | None = 15,
        figsize=None,
        title: str | None = "Accordion bulk deconvolution",
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend_title: str = "Cell type",
        manual_palette: Optional[Mapping[str, str]] = None,
        cmap: Optional[str] = None,
        ax=None,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300):
    """Plot inferred bulk cell-type composition.

    Parameters
    ----------
    composition
        Cell types × samples composition DataFrame.
    kind
        "stacked_bar", "bar", or "heatmap".
    top_n
        Keep the top N mean-abundance cell types and aggregate the rest as "Other".
    xlabel, ylabel
        Axis label overrides. ``None`` (default) keeps each ``kind``'s natural
        labels ("Sample" / "Estimated fraction" for stacked_bar; "Cell type" /
        "Estimated fraction" for bar; "Sample" / "Cell type" for heatmap).
    legend_title
        Legend title for the ``kind="stacked_bar"`` cell-type legend. Default
        ``"Cell type"``. (``kind="heatmap"`` uses a colorbar instead of a
        legend; its "Estimated fraction" label describes the color scale, not
        a set of categories, so it isn't controlled by this parameter.)
    manual_palette
        Optional explicit ``{cell_type: color}`` mapping for ``kind="stacked_bar"``
        / ``"bar"``. Values here override the generated fallback palette; cell
        types not listed keep their generated color. Not used by ``"heatmap"``
        (see ``cmap`` instead).
    cmap
        Matplotlib colormap name for ``kind="heatmap"``'s continuous color
        scale. ``None`` keeps matplotlib's default.
    ax
        Existing axes to draw on; a new figure is created when ``None``.
    show, save, dpi
        Standard display / export options, consistent with the other
        ``plot_*`` functions: ``save="path.png"`` writes the figure to disk,
        ``show=False`` skips ``plt.show()`` (handy when only saving), and
        ``dpi`` controls the resolution used when saving.
    """
    comp = composition.copy()

    if top_n is not None and len(comp) > top_n:
        top = comp.mean(axis=1).sort_values(ascending=False).head(top_n).index
        other = comp.drop(index=top).sum(axis=0)
        comp = comp.loc[top]
        comp.loc["Other"] = other

    if figsize is None:
        if kind == "heatmap":
            # A fixed formula based only on comp.shape[1] (as used for the bar
            # kinds) starves this layout: long CL_celltype names on the
            # y-axis (e.g. "CD4-negative, CD8-negative, alpha-beta
            # intraepithelial T cell") eat most of a small total width once
            # tight_layout() makes room for them, leaving the actual heatmap
            # squeezed into a sliver a few pixels wide -- which is exactly
            # what turns the rotated sample xtick labels into illegible
            # squiggles. Measure the real label width instead and budget
            # separate space for labels / heatmap / colorbar.
            from .plots import _text_width_in
            label_w_in = max(
                (_text_width_in(str(lbl), 10) for lbl in comp.index), default=1.0
            )
            label_w_in = min(label_w_in, 4.5) + 0.35  # cap + tick padding
            heatmap_w_in = max(3.0, comp.shape[1] * 0.6)
            colorbar_w_in = 1.3  # colorbar + its "Estimated fraction" label
            figsize = (
                label_w_in + heatmap_w_in + colorbar_w_in,
                max(4.0, min(14.0, comp.shape[0] * 0.35 + 1.6)),
            )
        else:
            figsize = (max(8, comp.shape[1] * 0.8), max(5, min(12, comp.shape[0] * 0.35)))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if kind == "stacked_bar":
        # pandas' own stacked-bar coloring falls back to matplotlib's 10-color
        # default cycle, which silently repeats (and visually merges) cell
        # types once top_n exceeds 10. Use the same generated palette as the
        # rest of the package instead, so every cell type stays distinct;
        # manual_palette entries (if any) take priority over it.
        palette = _resolve_bulk_palette(comp.index.astype(str).tolist(), manual_palette)
        colors = [palette[str(ct)] for ct in comp.index]
        comp.T.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_ylabel(ylabel if ylabel is not None else "Estimated fraction")
        ax.set_xlabel(xlabel if xlabel is not None else "Sample")
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    elif kind == "bar":
        if comp.shape[1] != 1:
            raise ValueError("kind='bar' is intended for a single sample. Use kind='stacked_bar' for multiple samples.")
        ordered = comp.iloc[:, 0].sort_values(ascending=False)
        palette = _resolve_bulk_palette(ordered.index.astype(str).tolist(), manual_palette)
        colors = [palette[str(ct)] for ct in ordered.index]
        ordered.plot(kind="bar", ax=ax, color=colors)
        ax.set_ylabel(ylabel if ylabel is not None else "Estimated fraction")
        ax.set_xlabel(xlabel if xlabel is not None else "Cell type")
    elif kind == "heatmap":
        im = ax.imshow(comp.values, aspect="auto", cmap=cmap)
        ax.set_yticks(range(comp.shape[0]))
        ax.set_yticklabels(comp.index)
        ax.set_xticks(range(comp.shape[1]))
        ax.set_xticklabels(comp.columns, rotation=90)
        ax.set_xlabel(xlabel if xlabel is not None else "Sample")
        ax.set_ylabel(ylabel if ylabel is not None else "Cell type")
        fig.colorbar(im, ax=ax, label="Estimated fraction")
        # Cell (row/column) separators driven by explicit minor-tick gridlines
        # placed at the actual square boundaries (i.e. offset by 0.5 from the
        # major ticks, which sit at each cell's center for the labels). This
        # is independent of the ambient seaborn theme, so the look doesn't
        # depend on whichever style another plotting call happened to leave
        # active most recently, and the horizontal lines divide one cell
        # type's square from the next rather than cutting through its middle.
        ax.grid(False)
        ax.set_xticks(np.arange(comp.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(comp.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
    else:
        raise ValueError("kind must be 'stacked_bar', 'bar', or 'heatmap'")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
