"""Cell-cycle state annotation.

``accordion_cellcycle`` scores every cell against the packaged cell-cycle
marker collection (phases G0, G1, G2M, S) using the same Accordion scoring as
:func:`pycellmarkeraccordion.accordion_custom` -- this is simply
``accordion_custom`` run on the built-in ``cell_cycle_markers`` table.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from .io import get_package_data_path
from .custom import accordion_custom


def _first_col(df, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_cell_cycle_markers(species):
    """Load the packaged cell-cycle marker table, filtered by species."""
    df = pl.read_parquet(get_package_data_path("cell_cycle_markers.parquet")).to_pandas()
    sp = [species] if isinstance(species, str) else list(species)
    if "species" in df.columns and sp:
        sub = df[df["species"].astype(str).isin([str(s) for s in sp])]
        if not sub.empty:
            df = sub
    return df


def accordion_cellcycle(
        data,
        species: str = "Human",
        annotation_resolution: str = "cell",
        annotation_name: str = "accordion_cell_cycle",
        cluster_info: str = "leiden",
        allow_unknown: bool = False,
        min_n_marker: Optional[int] = 1,
        is_normalized: bool = False,
        rescale_specificity_to_evidence: bool = False,
        category_column: str = "cell_type",
        marker_column: str = "marker",
        marker_type_column: str = "marker_type",
        **kwargs,
):
    """Assign a cell-cycle phase (G0/G1/G2M/S) to each cell (or cluster).

    Uses the packaged ``cell_cycle_markers`` table and the
    :func:`pycellmarkeraccordion.accordion_custom` scoring machinery.

    Parameters
    ----------
    data : AnnData or str
        An :class:`~anndata.AnnData` object, or a path to a raw/normalized count
        file (``.h5ad``, ``.tsv``, ``.csv``, ``.mtx``, ``.npz``) — same input as
        :func:`pycellmarkeraccordion.accordion`. For cluster resolution, the object must
        contain the ``cluster_info`` column.
    species
        ``"Human"`` or ``"Mouse"`` (markers are filtered accordingly).
        Default ``"Human"``.
    annotation_resolution
        ``"cell"`` (default) and/or ``"cluster"``.
    annotation_name
        Output prefix; results land in ``{annotation_name}_per_cell`` /
        ``{annotation_name}_per_cluster`` (+ ``_score``). Default
        ``"accordion_cell_cycle"``.
    allow_unknown
        Allow cells/clusters with non-positive scores to be labelled
        ``"unknown"``. Default ``False``: every cell/cluster gets a phase
        assigned outright, since every cell is always in some cell-cycle phase.
    min_n_marker
        Minimum markers per phase. Default ``1`` (keep all phases).
    is_normalized
        Whether the expression matrix is already log-normalized. Default
        ``False``.
    rescale_specificity_to_evidence
        If ``False`` (default), the specificity term ``SPs_reg`` is kept as-is
        instead of being rescaled onto the evidence range. With uniform weights
        that rescale would collapse the specificity weighting, so keeping it
        ``False`` preserves the down-weighting of markers shared across phases.
        Set ``True`` to mirror the ``accordion_custom`` behaviour instead.
    **kwargs
        Forwarded to :func:`pycellmarkeraccordion.accordion_custom` /
        :func:`pycellmarkeraccordion.accordion`, including the file-loading options
        (``features``, ``barcodes``, ``has_genes_on_rows``, ``layer`` ...).

    Returns
    -------
    AnnData
        With the cell-cycle assignment in ``adata.obs``.
    """
    markers = _load_cell_cycle_markers(species)

    category_column = (category_column if category_column in markers.columns
                       else _first_col(markers, ["cell_type", "phase",
                                                 "cell_cycle_phase", "category"]))
    marker_column = (marker_column if marker_column in markers.columns
                     else _first_col(markers, ["marker", "gene", "gene_symbol"]))
    if category_column is None or marker_column is None:
        raise KeyError(
            "Could not identify the phase/marker columns in cell_cycle_markers. "
            f"Available columns: {list(markers.columns)}"
        )

    species_one = species if isinstance(species, str) else (list(species)[0] if species else "Human")

    return accordion_custom(
        data,
        markers,
        category_column=category_column,
        marker_column=marker_column,
        marker_type_column=marker_type_column,
        species=species_one,
        annotation_name=annotation_name,
        cluster_info=cluster_info,
        annotation_resolution=annotation_resolution,
        min_n_marker=min_n_marker,
        allow_unknown=allow_unknown,
        is_normalized=is_normalized,
        # Keep specificity-based SPs_reg as-is (skip the rescale onto the
        # evidence range that accordion_custom uses by default).
        rescale_specificity_to_evidence=rescale_specificity_to_evidence,
        **kwargs,
    )
