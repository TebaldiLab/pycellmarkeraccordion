"""Pseudobulk fast-track annotation.

Condenses each cluster into a single pseudo-cell, annotates those pseudo-cells
with the standard Accordion scoring, and propagates the resulting label back to
every cell in the cluster. Much faster than per-cell scoring on large datasets.
"""

from __future__ import annotations

from .accordion import accordion


def accordion_pseudobulk(
        data,
        cluster_info="leiden",
        aggregation="mean",
        layer="X",
        **kwargs):
    """Run Accordion pseudobulk fast-track annotation.

    This function condenses each cluster into one pseudo-cell, annotates those
    pseudo-cells with Accordion cell-level scoring, and propagates the resulting
    annotation back to all original cells as cluster-level annotation.

    Parameters
    ----------
    data : AnnData or str
        An :class:`~anndata.AnnData` object or a path to an ``.h5ad`` file.
        Unlike :func:`pycellmarkeraccordion.accordion`, the pseudobulk fast-track only
        supports these two input types (it needs random access to the clusters),
        so ``.mtx`` / ``.tsv`` / ``.csv`` inputs must be loaded into AnnData
        first.
    cluster_info : str
        Column in ``adata.obs`` containing cluster identities. Required, since
        pseudo-cells are formed per cluster.
    aggregation : {"mean", "sum"}
        How to aggregate expression within each cluster.
    layer : str
        Layer to aggregate. Use ``"X"`` for ``adata.X``.
    **kwargs
        Additional parameters forwarded to :func:`pycellmarkeraccordion.accordion`, e.g.
        ``condition``, ``disease``, ``tissue``, ``species``, ``is_normalized``,
        ``n_jobs``, ``annotation_name``, marker thresholds, etc.

    Returns
    -------
    AnnData
        Original AnnData with propagated cluster-level annotation.

    Examples
    --------
    >>> adata_pb = accordion_pseudobulk(
    ...     adata,
    ...     cluster_info="leiden",
    ...     aggregation="mean",
    ... )
    >>> adata_pb.obs["accordion_per_cluster"]
    """
    return accordion(
        data,
        cluster_info=cluster_info,
        pseudobulk_annotation=True,
        pseudobulk_aggregation=aggregation,
        pseudobulk_layer=layer,
        **kwargs,
    )
