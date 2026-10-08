"""Custom marker-set annotation and pathway/signature scoring.

Instead of the built-in Accordion database, the user supplies their own marker table mapping *categories*
(cell types, signatures or pathways) to marker genes, with optional marker type
(positive/negative) and per-marker weight.

The scoring machinery is identical to :func:`pycellmarkeraccordion.accordion`: the custom
table is converted into the internal marker schema with ``ECs_global = weight``
(so that the evidence term ``ECs_reg = log10(weight) + 1`` matches 
``weight_scaled``), and the full annotation pipeline is reused.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import pandas as pd
import polars as pl

from .accordion import accordion


def _normalize_custom_marker_table(
        marker_table,
        category_column: str,
        marker_column: str,
        marker_type_column: str,
        weight_column: str,
        species: str,
) -> pd.DataFrame:
    """Convert a user marker table into the internal Accordion marker schema."""
    df = marker_table
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.copy()

    if category_column not in df.columns or marker_column not in df.columns:
        raise ValueError(
            f"marker_table must contain at least the columns {category_column!r} "
            f"(categories / cell types / pathways) and {marker_column!r} (genes). "
            f"Found columns: {list(df.columns)}"
        )

    category = df[category_column].astype(str)
    marker = df[marker_column].astype(str)

    if marker_type_column in df.columns:
        mtype = df[marker_type_column].astype(str).str.lower()
        mtype = mtype.where(mtype.isin(["positive", "negative"]), "positive")
    else:
        mtype = pd.Series(["positive"] * len(df), index=df.index)

    if weight_column in df.columns:
        weight = pd.to_numeric(df[weight_column], errors="coerce").fillna(1.0)
    else:
        weight = pd.Series([1.0] * len(df), index=df.index)

    custom = pd.DataFrame({
        "species": species if isinstance(species, str) else "Human",
        "Uberon_tissue": "ALL",
        "Uberon_ID": "ALL",
        "CL_celltype": category.values,
        "CL_ID": "",
        "marker": marker.values,
        "marker_type": mtype.values,
        "ECs_global": weight.astype(float).values,
        "resource": "custom",
    })
    # Drop fully-identical rows (same category, marker, weight, etc.) so an
    # accidentally repeated input row doesn't inflate weights. Note this is an
    # exact-row dedup: two rows with the same (category, marker) but a
    # different weight are NOT deduped and will both be kept.
    custom = custom.drop_duplicates()
    return custom


def accordion_custom(
        data,
        marker_table,
        category_column: str = "cell_type",
        marker_column: str = "marker",
        marker_type_column: str = "marker_type",
        weight_column: str = "weight",
        species: str = "Human",
        cluster_info: str = "leiden",
        min_n_marker: Optional[int] = 5,
        max_n_marker: Optional[int] = None,
        combined_score_quantile_threshold: Optional[float] = None,
        annotation_resolution: str = "cluster",
        cluster_score_quantile_threshold: float = 0.75,
        allow_unknown: bool = True,
        annotation_name: str = "accordion_custom",
        include_detailed_annotation_info: bool = True,
        condition_group_info: Optional[str] = None,
        celltype_group_info: Optional[str] = None,
        group_markers_by: str = "celltype_cluster",
        top_cell_score_quantile_threshold: float = 0.90,
        n_top_celltypes: int = 5,
        n_top_markers: int = 5,
        top_marker_score_quantile_threshold: float = 0.75,
        is_normalized: bool = False,
        **kwargs,
):
    """Annotate cells/clusters or score signatures/pathways from a custom marker set.

    Supply a table mapping each
    *category* (a cell type, gene signature or pathway) to its marker genes; the
    function scores every cell against each category using the same Accordion
    scoring as :func:`pycellmarkeraccordion.accordion`, and assigns each cluster/cell to its
    top-scoring category.

    Parameters
    ----------
    data : AnnData or str
        An :class:`~anndata.AnnData` object, or a path to a raw/normalized count
        file (``.h5ad``, ``.tsv``, ``.csv``, ``.mtx``, ``.npz``) — same input as
        :func:`pycellmarkeraccordion.accordion`. For cluster resolution, the object must
        contain the ``cluster_info`` column (default ``"leiden"``).
    marker_table
        ``pandas.DataFrame`` (or anything convertible to one) with at least the
        category and marker columns. Optional marker-type and weight columns are
        used when present.
    category_column, marker_column
        Column names for the category (cell type / signature / pathway) and the
        marker gene. Defaults ``"cell_type"`` and ``"marker"``.
    marker_type_column
        Optional column with ``"positive"`` / ``"negative"`` per marker. Missing
        or unrecognised values default to ``"positive"``. Default
        ``"marker_type"``.
    weight_column
        Optional per-marker weight column (analogous to the evidence-consistency
        score). Missing weights default to ``1``. Default ``"weight"``.
    species
        Species label; markers are matched as provided (no case conversion when a
        single species string is given). Default ``"Human"``.
    cluster_info
        Cluster column (cluster resolution). Default ``"leiden"``.
    min_n_marker, max_n_marker
        Min/max markers kept per category (markers ranked by combined score). For
        small signatures/pathways you may want ``min_n_marker=1``. Defaults ``5``
        / ``None``.
    combined_score_quantile_threshold
        Keep only markers above this within-category combined-score quantile.
        Default ``None``.
    annotation_resolution
        ``"cluster"`` and/or ``"cell"``. Default ``"cluster"``.
    cluster_score_quantile_threshold
        Per-cluster score quantile used to assign a cluster to a category.
        Default ``0.75``.
    allow_unknown
        Allow cells/clusters with non-positive scores to be labelled
        ``"unknown"``. Default ``True``.
    annotation_name
        Prefix for the output columns / ``adata.uns`` entry. Results are stored in
        ``{annotation_name}_per_cluster`` / ``{annotation_name}_per_cell`` (and
        their ``_score`` columns). Default ``"accordion_custom"``.
    include_detailed_annotation_info
        Store the detailed top-categories / top-markers tables. Default ``True``.
    condition_group_info, celltype_group_info
        ``adata.obs`` columns with per-cell condition / cell-type labels. When
        given, the cell-resolution top-marker table is split by condition (and,
        when supported, by cell type). Defaults ``None``.
    group_markers_by
        How to group the reported top markers: ``"celltype_cluster"``,
        ``"celltype_cell"``, ``"cluster"``, ``"cell"`` or ``"score_cell"``.
        Default ``"celltype_cluster"``.
    top_cell_score_quantile_threshold
        Cell-score quantile used by ``group_markers_by="score_cell"``. Default
        ``0.90``.
    n_top_celltypes, n_top_markers
        Number of top categories / markers to report. Default ``5``.
    top_marker_score_quantile_threshold
        Marker-score quantile used to pick the top markers. Default ``0.75``.
    is_normalized
        Whether the expression matrix is already log-normalized. Default
        ``False``.
    **kwargs
        Forwarded to :func:`pycellmarkeraccordion.accordion`, including the file-loading
        options (``features``, ``barcodes``, ``has_genes_on_rows``,
        ``cell_column``, ``cluster_column``, ``layer``) and ``n_jobs``.

    Returns
    -------
    AnnData
        Annotated object with the custom assignment in ``adata.obs`` and, when
        ``include_detailed_annotation_info=True``, detailed tables in
        ``adata.uns[annotation_name]`` (compatible with
        :func:`pycellmarkeraccordion.plot_top_celltypes_per_cluster`,
        :func:`pycellmarkeraccordion.plot_markers_lollipop`, etc.).

    Notes
    -----
    Pathway / signature scoring is just custom annotation where each *category*
    is a pathway: every cluster/cell is labelled with its highest-scoring pathway.
    """
    custom = _normalize_custom_marker_table(
        marker_table,
        category_column=category_column,
        marker_column=marker_column,
        marker_type_column=marker_type_column,
        weight_column=weight_column,
        species=species,
    )

    # Hand the custom table to accordion as an in-database marker file. A temp
    # parquet keeps dtypes (weight stays numeric) and reuses the normal loader.
    tmp = tempfile.NamedTemporaryFile(
        prefix="pycellmarkeraccordion_custom_", suffix=".parquet", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        pl.from_pandas(custom).write_parquet(tmp_path)
        result = accordion(
            data,
            database=tmp_path,
            condition="health",
            species=species,
            tissue=None,
            cluster_info=cluster_info,
            min_n_marker=min_n_marker,
            max_n_marker=max_n_marker,
            combined_score_quantile_threshold=combined_score_quantile_threshold,
            annotation_resolution=annotation_resolution,
            cluster_score_quantile_threshold=cluster_score_quantile_threshold,
            allow_unknown=allow_unknown,
            annotation_name=annotation_name,
            include_detailed_annotation_info=include_detailed_annotation_info,
            condition_group_info=condition_group_info,
            celltype_group_info=celltype_group_info,
            group_markers_by=group_markers_by,
            top_cell_score_quantile_threshold=top_cell_score_quantile_threshold,
            n_top_celltypes=n_top_celltypes,
            n_top_markers=n_top_markers,
            top_marker_score_quantile_threshold=top_marker_score_quantile_threshold,
            is_normalized=is_normalized,
            **kwargs,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return result
