"""Spatial transcriptomics wrappers.

Thin wrappers around :func:`pycellmarkeraccordion.accordion` for spatial data, where each
observation is a spot to be deconvolved into candidate cell types, plus a helper
to plot the resulting spot-level annotation entropy.
"""

from __future__ import annotations

from .accordion import accordion
from .spatial_entropy import create_spatial_entropy_plot


def accordion_spatial_deconvolution(
        adata,
        entropy="Overlap-Rao",
        deconvolution_threshold=0.05,
        min_percentage_celltype_entropy=0.05,
        annotation_name="accordion",
        **kwargs):
    """Run Accordion spatial deconvolution and optional spot-level entropy.

    This wrapper is intended for spatial transcriptomics data where observations
    are spots. Each spot is treated as the unit to deconvolve into candidate
    cell types.

    Internally this calls::

        accordion(
            adata,
            annotation_resolution="cell",
            do_deconvolution=True,
            entropy=entropy,
            ...
        )

    Parameters
    ----------
    adata
        Spatial AnnData object or .h5ad path.
    entropy
        Entropy metric or list of metrics. Examples:
        ``"Shannon"``, ``"Gini"``, ``"Overlap-Rao"``, ``"Jaccard-Rao"``,
        ``"Ontology-sp-Rao"``, ``"Ontology-Lin-Rao"``, or ``"all"``.
        Set to ``None`` or ``False`` to skip entropy.
    deconvolution_threshold
        Minimum cell-type contribution threshold used in deconvolution.
    min_percentage_celltype_entropy
        Minimum percentage/contribution threshold used for entropy calculation.
    annotation_name
        Prefix used for output columns and ``adata.uns`` storage.
    **kwargs
        Additional arguments passed to ``accordion``. Examples:
        ``condition``, ``tissue``, ``species``, ``n_jobs``, ``cell_chunk_size``,
        ``database``, ``n_top_celltypes``, marker thresholds, etc.
        Notably ``deconvolution_empty_fallback`` (default ``False``): when
        ``True``, spots whose proportions are fully zeroed by
        ``deconvolution_threshold`` are repopulated with their top
        ``n_top_celltypes`` pre-threshold types instead of staying empty.

    Returns
    -------
    AnnData
        Spatial AnnData with deconvolution and entropy outputs stored in
        ``adata.obs`` and ``adata.uns[annotation_name]``.

    Expected outputs
    ----------------
    ``adata.obs[f"{annotation_name}_per_cell"]``
        Main assigned cell type for each spot.
    ``adata.obs[f"{annotation_name}_per_cell_score"]``
        Main assignment score.
    ``adata.obsm[f"{annotation_name}_proportions"]``
        Per-spot cell-type **composition**: a (spots x cell types) DataFrame of
        estimated fractions (each row sums to 1), i.e. how much each candidate
        cell type contributes to the spot. Only the top ``n_top_celltypes``
        (default 5) types per spot are kept -- a spot is a mixture of a few cell
        types, not all candidates -- and contributions below
        ``deconvolution_threshold`` are dropped; the row is then renormalised.
    ``adata.obs[f"{annotation_name}_per_cell_percentage"]``
        Percentage (0--100) of the *assigned* (top) cell type in each spot.
    ``adata.obs[f"{annotation_name}_Overlap-Rao_entropy"]``
        Example entropy output when ``entropy="Overlap-Rao"``.
    ``adata.uns[annotation_name]["cell_entropy_input"]``
        Table used for spot-level entropy.
    ``adata.uns[annotation_name]["cell_entropy"]``
        Entropy result tables.
    """
    return accordion(
        adata,
        annotation_resolution="cell",
        do_deconvolution=True,
        entropy=entropy,
        deconvolution_threshold=deconvolution_threshold,
        min_percentage_celltype_entropy=min_percentage_celltype_entropy,
        annotation_name=annotation_name,
        **kwargs,
    )


def plot_spatial_entropy(
        adata,
        entropy_type="Overlap-Rao",
        annotation_name="accordion",
        **kwargs):
    """Plot spatial entropy after ``accordion_spatial_deconvolution``.

    Parameters
    ----------
    adata
        AnnData returned by ``accordion_spatial_deconvolution``.
    entropy_type
        Entropy type to plot, e.g. ``"Shannon"``, ``"Gini"`` or ``"Overlap-Rao"``.
    annotation_name
        Prefix used when running Accordion.
    **kwargs
        Additional arguments passed to ``create_spatial_entropy_plot``:
        ``figsize``, ``colormap``, ``spot_size``, ``title``/``xlabel``/
        ``ylabel``/``colorbar_label``, ``hist_title``/``hist_xlabel``/
        ``hist_ylabel``, and ``show``/``save``/``dpi`` (standard display /
        export options, same as every other ``plot_*`` function).

    Returns
    -------
    tuple
        ``(fig, plot_data)``
    """
    if annotation_name not in adata.uns:
        raise KeyError(f"adata.uns[{annotation_name!r}] not found. Run accordion_spatial_deconvolution first.")

    info = adata.uns[annotation_name]

    # The spatial fast-path stores the entropy result tables under
    # info["cell_entropy"] but skips the heavy "cell_entropy_input" table to save
    # memory. Rebuild a minimal version from the per-spot obs outputs so the plot
    # works out of the box.
    if "cell_entropy_input" not in info:
        import pandas as pd

        cell_col = f"{annotation_name}_per_cell"
        score_col = f"{annotation_name}_per_cell_score"
        if cell_col in adata.obs:
            diff_score = (
                adata.obs[score_col].to_numpy()
                if score_col in adata.obs
                else 0.0
            )
            info["cell_entropy_input"] = pd.DataFrame({
                "cell": adata.obs_names.astype(str),
                "diff_score": diff_score,
                "annotation_per_cell": adata.obs[cell_col].astype(str).to_numpy(),
            })
        else:
            raise KeyError(
                f"adata.uns[{annotation_name!r}]['cell_entropy_input'] not found "
                f"and adata.obs[{cell_col!r}] is missing, so it cannot be rebuilt. "
                "Run accordion_spatial_deconvolution with entropy enabled."
            )

    if "cell_entropy" not in info:
        raise KeyError(
            f"adata.uns[{annotation_name!r}]['cell_entropy'] not found. "
            "Run with entropy enabled."
        )

    return create_spatial_entropy_plot(
        adata,
        info["cell_entropy_input"],
        info["cell_entropy"],
        entropy_type=entropy_type,
        **kwargs,
    )
