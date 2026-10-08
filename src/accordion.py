"""Core Accordion annotation engine.

This module implements :func:`accordion`, the main entry point for marker-based
cell-type annotation. It loads and filters the marker database (by species,
tissue and cell type), derives the per-marker evidence (``ECs``) and specificity
(``SPs``) scores, scales the expression matrix, and scores every cell against
each candidate cell type. Cells and/or clusters are then assigned to their
top-scoring type, optional annotation-entropy metrics are computed, and detailed
per-cluster/per-cell tables are stored in ``adata.uns``.

The higher-level wrappers (``accordion_custom``, ``accordion_pseudobulk``,
``accordion_spatial_deconvolution``, ``accordion_cellcycle``) all build on this
function.
"""

import scanpy as sc
import scipy.sparse as sp
import pandas as pd
import numpy as np
import anndata
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl

from .annotation_helpers import (
    include_detailed_annotation_info_helper,
    _build_cell_entropy_table_from_final_dt,
    _build_marker_table_for_cell_entropy,
)
from .spatial_entropy import calculate_spot_entropies, calculate_spot_entropies_parallel, calculate_entropies_from_score_matrix
from .io import read_marker_database, get_default_cell_ontology_path, normalize_condition, normalize_marker_database_schema




from .utils import (                             # noqa: F401
    DEFAULT_N_JOBS,
    resolve_n_jobs as _resolve_n_jobs,
    as_text_series as _as_text_series,
)


#: Upper bound applied to the z-scored marker expression, matching the
#: ``scale.max`` default of Seurat's ScaleData so that this package and the R
#: cellmarkeraccordion score the same matrix. R is the reference for the
#: annotation metrics, so this value tracks Seurat rather than being tuned here.
SCALE_MAX = 10.0


def _isa_only(graph):
    """Keep only the ``is_a`` edges of an OBO graph.

    ``cl-basic.obo`` also carries 383 ``RO:0002202`` (*develops_from*) edges, and
    the ontology metrics walk the graph with ``nx.shortest_path_length`` and
    ``nx.ancestors``, which follow every edge type. Two reasons to drop them:

    * Lin's information content counts how many terms are *a kind of* a term.
      ``develops_from`` is not subsumption -- a haematopoietic stem cell is not the
      supertype of everything that develops from it -- and counting those edges
      collapses the IC of the progenitors, making the most specific terms look the
      most general. Measured on CL: 4 subclasses becomes 616 for haematopoietic
      multipotent progenitor cell, IC 6.32 -> 1.50.
    * The R package's ``cell_onto`` holds ``is_a`` only (the same 3845 edges, as a
      set), so restricting here is also what makes the two comparable.

    Impact on 149 blood cell types: 31% of pairs change their shortest-path
    distance (up to 6 edges), and only 13% of the Lin distances stay the same.
    """
    import networkx as nx

    if graph is None:
        return None
    kept = [(u, v, k, d) for u, v, k, d in graph.edges(keys=True, data=True)
            if k == "is_a"]
    if len(kept) == graph.number_of_edges():
        return graph
    out = nx.MultiDiGraph()
    out.graph.update(graph.graph)
    out.add_nodes_from(graph.nodes(data=True))
    out.add_edges_from(kept)
    return out


def _rescale(x, to, frm=None):
    """``scales::rescale`` from R: map ``frm`` onto ``to`` linearly.

    ``(x - lo) / (hi - lo) * (b - a) + a``, with ``frm`` defaulting to the
    observed range of ``x``.

    Written out rather than delegated to sklearn's ``MinMaxScaler`` on purpose.
    MinMaxScaler computes ``scale_ = (b - a) / (hi - lo)`` and then
    ``x * scale_ + min_``: the same function, a different order of operations,
    and a result that differs from R's in the last bit. Measured on the marker
    scores, that 1e-16 was enough to flip which marker sat at the ``max_n_marker``
    cutoff for 63 of 514 cell types. This form is bit-identical to R.
    """
    a, b = float(to[0]), float(to[1])
    x = np.asarray(x, dtype=float)
    if frm is None:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        lo, hi = float(frm[0]), float(frm[1])
    if hi == lo:
        # scales::rescale's zero_range branch: the midpoint of the target range.
        return np.full(x.shape, (a + b) / 2.0)
    return (x - lo) / (hi - lo) * (b - a) + a


def _chunked(values, chunk_size=500):
    """Yield chunks from a list-like object."""
    values = list(values)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def _parallel_map(items, func, n_jobs=None, chunk_size=500):
    """Parallel map utility. Uses chunks to keep overhead low."""
    n_jobs = _resolve_n_jobs(n_jobs)
    chunks = list(_chunked(items, chunk_size))
    if not chunks:
        return []

    def _run_chunk(chunk):
        return [func(item) for item in chunk]

    if n_jobs == 1 or len(chunks) == 1:
        nested = [_run_chunk(chunk) for chunk in chunks]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            nested = list(executor.map(_run_chunk, chunks))

    out = []
    for part in nested:
        out.extend(part)
    return out



def _sanitize_for_h5ad(obj):
    """Make an ``adata.uns`` value safe to write to ``.h5ad`` (anndata < 0.11).

    Recurses through dicts/lists and, for every contained DataFrame/Series,
    converts pandas nullable ``"string"`` columns (``pd.arrays.StringArray``) to
    plain ``object`` strings -- anndata < 0.11 refuses to write StringArrays.
    Non-tabular values (and nested AnnData objects) are left untouched.
    """
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _sanitize_for_h5ad(obj[k])
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_for_h5ad(x) for x in obj)
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if isinstance(obj[col].dtype, pd.StringDtype):
                obj[col] = obj[col].fillna("").astype(object)
        return obj
    if isinstance(obj, pd.Series):
        if isinstance(obj.dtype, pd.StringDtype):
            return obj.fillna("").astype(object)
        return obj
    return obj


def _make_pseudobulk_anndata(adata, cluster_key, layer=None, aggregation="mean"):
    """Condense cells in each cluster into one pseudo-cell.

    Parameters
    ----------
    adata
        AnnData with cells/spots as observations.
    cluster_key
        Column in adata.obs containing cluster ids.
    layer
        Optional layer to aggregate. If None or "X", uses adata.X.
    aggregation
        "mean" or "sum".
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"pseudobulk_annotation=True requires cluster_info='{cluster_key}' in adata.obs")

    if layer is None or layer == "X":
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Invalid pseudobulk_layer={layer!r}; layer not found in adata.layers")
        X = adata.layers[layer]

    clusters = adata.obs[cluster_key].astype(str)
    unique_clusters = pd.Index(clusters.unique()).astype(str)

    rows = []
    for cl in unique_clusters:
        mask = (clusters == cl).values
        X_sub = X[mask]

        if sp.issparse(X_sub):
            if aggregation == "sum":
                vec = np.asarray(X_sub.sum(axis=0)).ravel()
            elif aggregation == "mean":
                vec = np.asarray(X_sub.mean(axis=0)).ravel()
            else:
                raise ValueError("pseudobulk_aggregation must be either 'mean' or 'sum'")
        else:
            X_sub = np.asarray(X_sub)
            if aggregation == "sum":
                vec = X_sub.sum(axis=0)
            elif aggregation == "mean":
                vec = X_sub.mean(axis=0)
            else:
                raise ValueError("pseudobulk_aggregation must be either 'mean' or 'sum'")

        rows.append(vec)

    X_pb = np.vstack(rows)
    pb = anndata.AnnData(
        X=X_pb,
        obs=pd.DataFrame(index=unique_clusters),
        var=adata.var.copy(),
    )
    pb.obs[cluster_key] = unique_clusters
    pb.obs["pseudobulk_cluster"] = unique_clusters
    pb.uns["pseudobulk"] = {
        "source_n_obs": int(adata.n_obs),
        "source_n_vars": int(adata.n_vars),
        "cluster_key": cluster_key,
        "aggregation": aggregation,
    }
    return pb


def _propagate_pseudobulk_annotation(
        original_adata,
        pseudobulk_adata,
        cluster_key,
        annotation_name,
        score_suffix="_score"):
    """Propagate pseudobulk per-cell annotation to all cells of each cluster."""
    cluster_values = original_adata.obs[cluster_key].astype(str)

    pb_label_col = f"{annotation_name}_per_cell"
    pb_score_col = f"{annotation_name}_per_cell_score"
    out_label_col = f"{annotation_name}_per_cluster"
    out_score_col = f"{annotation_name}_per_cluster_score"

    if pb_label_col not in pseudobulk_adata.obs.columns:
        raise ValueError(
            f"Pseudobulk annotation did not create {pb_label_col!r}. "
            "Check annotation_resolution='cell' on pseudobulk run."
        )

    label_map = pseudobulk_adata.obs[pb_label_col].to_dict()
    original_adata.obs[out_label_col] = cluster_values.map(label_map)

    if pb_score_col in pseudobulk_adata.obs.columns:
        score_map = pseudobulk_adata.obs[pb_score_col].to_dict()
        original_adata.obs[out_score_col] = cluster_values.map(score_map)

    # Pseudobulk fast-track is cluster-level only: remove any per-cell
    # annotation columns produced internally by the pseudo-cell run.
    for _col in [f"{annotation_name}_per_cell", f"{annotation_name}_per_cell_score"]:
        if _col in original_adata.obs.columns:
            original_adata.obs.drop(columns=[_col], inplace=True)

    original_adata.uns.setdefault(annotation_name, {})
    original_adata.uns[annotation_name]["pseudobulk"] = {
        "enabled": True,
        "cluster_key": cluster_key,
        "pseudobulk_obs": pseudobulk_adata.obs.copy(),
        "pseudobulk_uns": pseudobulk_adata.uns.get(annotation_name, {}),
    }
    return original_adata





def _accordion_progress(step, total_steps, label, start_time=None):
    """Lightweight Accordion progress printer with ETA."""
    prefix = f"[Accordion] Step {step}/{total_steps}"
    if start_time is None:
        print(f"{prefix} - {label}")
        return

    elapsed = time.time() - start_time
    avg = elapsed / max(step, 1)
    remaining = max(total_steps - step, 0) * avg

    if remaining < 60:
        eta = f"{remaining:.1f}s"
    else:
        eta = f"{remaining/60:.1f}min"

    print(f"{prefix} - {label} | elapsed={elapsed:.1f}s | ETA~{eta}")



def _get_matrix_for_layer(adata, layer="X"):
    """Return adata.X or a layer without materializing to dense."""
    if layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"Invalid layer {layer} provided")
    return adata.layers[layer]


def _looks_like_integer_counts_matrix(X, max_values=100000):
    """Memory-safe heuristic to check whether a matrix looks like raw counts.

    Avoids AnnData.to_df(), which densifies the entire matrix and can allocate
    tens of GB for large sparse datasets.
    """
    if sp.issparse(X):
        data = X.data
        if data.size == 0:
            return True
        sample = data[:max_values]
    else:
        arr = np.asarray(X)
        if arr.size == 0:
            return True
        sample = arr.ravel()[:max_values]

    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return True

    return np.all(np.equal(sample, np.floor(sample)))


def _copy_anndata_light(adata, copy_input=False):
    """Return a copy only if requested.

    For large AnnData, deep copying can duplicate X, layers, and large entries
    in .uns, causing avoidable memory errors. By default this returns the same
    object and modifies it in-place, which is safer for large datasets.
    """
    return adata.copy() if copy_input else adata



def accordion(
        data,  # A file: AnnData object or raw counts (.tsv, .csv, .mtx, .npz);
        features=None,  ## path to .tsv for features (for .mtx)
        barcodes=None,  ## path to .tsv for barcodes (for .mtx)
        has_genes_on_rows=False,  ## specify if the raw count matrix has genes on rows or not
        cluster_info="leiden",
        # if AnnData -> string containing clusters in .obs  # if raw -> .csv or .tsv "cell" "cluster"
        cell_column="cell",  ## string specifying the column name containing the cells (for raw cluster_info)
        cluster_column="cluster",  ## string specifying the column name containing the clusters (for raw cluster_info)
        layer="X",  # adata layer containing the counts
        database=None,  # specify whether to use the default Accordion database or a custom one (file path)
        condition="health",  # "health" or "disease": selects the packaged marker database
        is_normalized=False,  ## user input to specify whether data is normalized; True/False
        celltypes=None,  # string or vector of strings specifiying the cells to annotate; list_celltypes()
        species="Human",  # string or vector defining the species to filter by
        tissue=None,  # string or vector specifying the tissues to filter by; list_tissues()
        include_descendants=False,  # include tissue descendants
        evidence_consistency_score_threshold=None,  # minimum EC for each marker
        specificity_score_threshold=None,  # minimum SP specificity score
        log2FC_threshold=None,  # log2FC minimum
        min_n_marker=5,  # minimum number of markers for each cell type
        max_n_marker=30,  # maximum number of markers for each cell type
        combined_score_quantile_threshold=None,  # combined score threshold
        annotation_resolution="cluster",  # either "cluster" and/or "cell"; either annotate by cluster and/or by cell
        cluster_score_quantile_threshold=0.75,  # cluster score quantile threshold
        allow_unknown=True,  # allow unknown cells or clusters
        annotation_name="accordion",  # name of the column in either AnnData or cluster .csv/.tsv to save annotation
        include_detailed_annotation_info=True,  # include detailed information annotation
        disease=None,
        NCIT_celltypes=None,
        malignant_quantile_threshold=0.95,
        ECs_threshold=None,
        SPs_threshold=None,
        condition_group_info=None,
        celltype_group_info=None,
        group_markers_by="celltype_cluster",
        top_cell_score_quantile_threshold=0.90,
        n_top_celltypes=5,
        n_top_markers=5,
        top_marker_score_quantile_threshold=0.75,
        save_scaled_data=True,  # store scaled matrix in adata.uns
        save_entropy_to_obs=True,  # store requested cell-level entropy metrics in adata.obs
        verbose=True,  # print progress updates during computation
        progress_every=10,  # report every N completed cell types/clusters where applicable
        entropy="Overlap-Rao",
        ontology_path=None,
        min_percentage_celltype_entropy=0.05,
        do_deconvolution=False,
        deconvolution_threshold=0.05,
        deconvolution_empty_fallback=False,
        plot=True,
        color_by="cell_type",
        n_jobs=None,
        cell_chunk_size=500,
        pseudobulk_annotation=False,
        pseudobulk_layer=None,
        pseudobulk_aggregation="mean",
        rescale_specificity_to_evidence=True,
        copy_input=False):
    """Annotate cell types with the built-in Cell Marker Accordion database.

    Scores every cell against the curated marker database and assigns each
    cluster and/or cell to its top-scoring cell type. Optionally computes
    annotation-entropy metrics and stores detailed per-cluster / per-cell tables
    in ``adata.uns[annotation_name]``.

    Parameters
    ----------
    data : AnnData or str
        An :class:`~anndata.AnnData` object, or a path to a raw/normalized count
        file (``.h5ad``, ``.tsv``, ``.csv``, ``.mtx``, ``.npz``).
    features, barcodes : str, optional
        Paths to the features / barcodes ``.tsv`` files when ``data`` is an
        ``.mtx`` matrix.
    has_genes_on_rows : bool, default False
        Whether a raw count matrix has genes on rows (and cells on columns).
    cluster_info : str, default "leiden"
        Column in ``adata.obs`` holding cluster ids (used for cluster
        resolution). For a raw-matrix input, a path to a table with the
        ``cell_column`` and ``cluster_column`` columns.
    cell_column, cluster_column : str
        Column names holding cell ids / cluster ids in a raw ``cluster_info``
        table. Defaults ``"cell"`` and ``"cluster"``.
    layer : str, default "X"
        ``adata`` layer to score. Use ``"X"`` for ``adata.X``.
    database : str or pandas.DataFrame, optional
        Custom marker database (file path or in-memory table, e.g. the output of
        :func:`marker_database_integration`). If ``None``, the packaged database
        selected by ``condition`` is used.
    condition : {"health", "disease"}, default "health"
        Which packaged marker database to use.
    is_normalized : bool, default False
        Whether ``adata.X`` (or ``layer``) is already log-normalized.
    celltypes : str or list, optional
        Restrict annotation to these cell types (see :func:`list_celltypes`).
    species : str or list, default "Human"
        Species to filter markers by (``"Human"`` and/or ``"Mouse"``).
    tissue : str or list, optional
        Tissue(s) to filter markers by (see :func:`list_tissues`). ``None`` uses
        all tissues.
    include_descendants : bool, default False
        Also include Uberon-ontology descendant tissues of ``tissue``.
    evidence_consistency_score_threshold : int, optional
        Keep only markers with evidence-consistency score (``ECs``) ≥ this value.
    specificity_score_threshold : float, optional
        Keep only markers with specificity score (``SPs``) within (0, 1].
    log2FC_threshold : float, optional
        Keep markers with ``log2FC`` ≥ this value (or no ``log2FC``).
    min_n_marker : int, default 5
        Drop cell types with fewer than this many markers.
    max_n_marker : int, default 30
        Keep at most this many top markers per cell type (by combined score).
    combined_score_quantile_threshold : float, optional
        Keep only markers above this within-cell-type combined-score quantile.
    annotation_resolution : str or list, default "cluster"
        ``"cluster"`` and/or ``"cell"``.
    cluster_score_quantile_threshold : float, default 0.75
        Per-cluster score quantile used to assign a cluster to a cell type.
    allow_unknown : bool, default True
        Allow cells/clusters with non-positive scores to be labelled
        ``"unknown"``.
    annotation_name : str, default "accordion"
        Output prefix. Results land in ``{annotation_name}_per_cluster`` /
        ``{annotation_name}_per_cell`` (and ``_score`` columns); detailed tables
        in ``adata.uns[annotation_name]``.
    include_detailed_annotation_info : bool, default True
        Store the detailed top-cell-types / top-markers tables (needed by the
        ``plot_*`` functions).
    disease, NCIT_celltypes, malignant_quantile_threshold : optional
        Disease-database options (used when ``condition="disease"``).
    ECs_threshold, SPs_threshold : optional
        Aliases for ``evidence_consistency_score_threshold`` / ``specificity_score_threshold``.
    condition_group_info, celltype_group_info : str, optional
        ``adata.obs`` columns with per-cell condition / cell-type labels, used to
        report top markers grouped by condition and/or cell type.
    group_markers_by : str, default "celltype_cluster"
        How to group the reported top markers: ``"celltype_cluster"``,
        ``"celltype_cell"``, ``"cluster"``, ``"cell"`` or ``"score_cell"``.
    top_cell_score_quantile_threshold : float, default 0.90
        Cell-score quantile used by ``group_markers_by="score_cell"``.
    n_top_celltypes, n_top_markers : int, default 5
        Number of top cell types / markers to report.
    top_marker_score_quantile_threshold : float, default 0.75
        Marker-score quantile used to pick the top markers.
    save_scaled_data : bool, default True
        Store the scaled marker matrix in ``adata.uns["scaled"]``.
    save_entropy_to_obs : bool, default True
        Store requested cell-level entropy metrics in ``adata.obs``.
    verbose : bool, default True
        Print progress updates.
    progress_every : int, default 10
        Report progress every N completed cell types/clusters.
    entropy : str, list or False, default "Overlap-Rao"
        Entropy metric(s): ``"Shannon"``, ``"Gini"``, ``"Overlap-Rao"``,
        ``"Jaccard-Rao"``, ``"Ontology-sp-Rao"``, ``"Ontology-Lin-Rao"``,
        ``"all"``, or ``False`` to skip. ``"Gini"`` is the Gini impurity
        ``1 - sum(p^2)``, i.e. the Tsallis entropy of order ``q = 2``. Note the
        two conventions differ on purpose: ``Shannon_entropy`` is normalized by
        ``log2(n)`` to fit ``[0, 1]``, while ``Gini_entropy`` is reported raw --
        it is bounded by 1 for any ``n`` anyway, and staying raw preserves the
        richness (how many cell types contribute) that normalization would
        divide away.
    ontology_path : str, optional
        Path to a Cell Ontology ``.obo`` for the ontology-based entropies.
    min_percentage_celltype_entropy : float, default 0.05
        Minimum cell-type contribution included in the entropy computation.
    do_deconvolution : bool, default False
        Enable spot/cell deconvolution (used by the spatial wrapper).
    deconvolution_threshold : float, default 0.05
        Minimum contribution kept during deconvolution.
    deconvolution_empty_fallback : bool, default False
        How to treat spots whose proportions are entirely zeroed by
        ``deconvolution_threshold`` (the signal is spread over many cell types,
        none individually above the threshold). If ``False`` (default), the
        threshold decides and such spots keep an all-zero, empty composition.
        If ``True``, those spots are repopulated with their top
        ``n_top_celltypes`` cell types taken from the pre-threshold
        proportions, so they still carry a composition instead of being empty.
        Note that this restores types that were individually below the
        threshold, so use it knowingly (it can present diffuse spots as more
        confident than the raw evidence).
    plot : bool, default True
        Reserved flag for storing detailed plots (Python builds plots via the
        ``plot_*`` functions instead).
    color_by : str, default "cell_type"
        Whether per-cluster dot plots are coloured by cell type or cluster id.
    n_jobs : int, optional
        Parallel workers. ``None`` runs serially, which is the faster choice for
        most datasets: the work is numpy/pandas-bound and the parallel paths use
        threads, so it runs under the GIL and extra workers mostly add
        contention. Parallelism is opt-in: ``-1`` uses every core, ``-2`` all but
        one, or pass a positive count.
    cell_chunk_size : int, default 500
        Chunk size for the parallel cell-level work.
    pseudobulk_annotation : bool, default False
        Use the pseudobulk fast-track (see :func:`accordion_pseudobulk`).
    pseudobulk_layer, pseudobulk_aggregation : optional
        Layer and aggregation (``"mean"``/``"sum"``) for pseudobulk.
    rescale_specificity_to_evidence : bool, default True
        Rescale the specificity term onto the evidence range. Set ``False`` to
        keep the raw specificity weighting (used by :func:`accordion_cellcycle`).
    copy_input : bool, default False
        Annotate a copy instead of modifying the input in place.

    Returns
    -------
    AnnData or dict
        The annotated ``AnnData`` (for an ``AnnData`` input) or a dict of result
        tables (for a raw-matrix input).
    """
    # has to return AnnData object or a dictionary; Anndata --> Annotated Anndata
    # Score/quantile computations below routinely divide/compare NaN or
    # zero-variance columns (e.g. empty clusters, all-zero marker scores);
    # without this they'd each raise a RuntimeWarning even though the
    # resulting NaN is handled downstream (dropped or filled), so silence it
    # rather than have it fire on every normal run.
    np.seterr(invalid="ignore")
    accordion_start_time = time.time()
    n_jobs = _resolve_n_jobs(n_jobs)
    cell_chunk_size = max(1, int(cell_chunk_size))
    progress_every = max(1, int(progress_every))

    def _log_progress(message):
        if verbose:
            elapsed = time.time() - accordion_start_time
            print(f"[Accordion] {message} | elapsed={elapsed:.1f}s", flush=True)

    def _entropy_progress_callback(entropy_type, done, total, unit="cells", phase="processing", label="entropy"):
        if not verbose:
            return
        pct = 100.0 * done / max(total, 1)
        _log_progress(f"{label}: {entropy_type} {phase} {done:,}/{total:,} {unit} ({pct:.1f}%)")

    def _entropy_is_requested(value):
        return value is not None and value is not False and value != "False"

    def _entropy_list(value):
        if not _entropy_is_requested(value):
            return []
        if isinstance(value, str):
            values = [value]
        else:
            try:
                values = list(value)
            except TypeError:
                values = [value]
        if "all" in values:
            return ["Shannon", "Gini", "Ontology-sp-Rao", "Ontology-Lin-Rao", "Jaccard-Rao", "Overlap-Rao"]
        return values

    _requested_entropy_types = _entropy_list(entropy)
    _cluster_entropy_requested = (
        save_entropy_to_obs
        and _entropy_is_requested(entropy)
        and "cluster" in annotation_resolution
    )

    condition = normalize_condition(condition)
    _log_progress(f"Starting annotation; n_jobs={n_jobs}, save_scaled_data={save_scaled_data}, save_entropy_to_obs={save_entropy_to_obs}")

    # Disease-mode defaults: rename the output prefix and accept the
    # disease-specific parameter aliases (NCIT_celltypes, ECs_threshold, SPs_threshold).
    if condition == "disease":
        if annotation_name == "accordion":
            annotation_name = "accordion_disease"
        if NCIT_celltypes is not None and celltypes is None:
            celltypes = NCIT_celltypes
        if ECs_threshold is not None and evidence_consistency_score_threshold is None:
            evidence_consistency_score_threshold = ECs_threshold
        if SPs_threshold is not None and specificity_score_threshold is None:
            specificity_score_threshold = SPs_threshold

    input_is_anndata_object = isinstance(data, anndata.AnnData)
    if input_is_anndata_object:
        ext = ".h5ad"
    elif isinstance(data, (str, os.PathLike)):
        data = os.fspath(data)
        ext = os.path.splitext(data)[-1].lower()  # get the "data" file extension
    else:
        raise TypeError("data must be a file path or an anndata.AnnData object")

    #### pre-checks
    if not isinstance(has_genes_on_rows, bool):
        if has_genes_on_rows in ("T", "true", "True", "TRUE"):
            has_genes_on_rows = True
        elif has_genes_on_rows in ("F", "false", "False", "FALSE"):
            has_genes_on_rows = False
        else:
            raise ValueError(f"Variable 'has_genes_on_rows' is not boolean; set it to True/False")
    if not isinstance(is_normalized, bool):
        if is_normalized in ("T", "true", "True", "TRUE"):
            is_normalized = True
        elif is_normalized in ("F", "false", "False", "FALSE"):
            is_normalized = False
        else:
            raise ValueError(f"Variable 'is_normalized' is not boolean; set it to True/False")
    if not isinstance(cell_column, str) or not isinstance(cluster_column, str):
        raise ValueError("'cell_column' and 'cluster_column' should be two strings.")

    # Pseudobulk fast-track: aggregate clusters into pseudo-cells, annotate
    # pseudo-cells at cell resolution, then propagate labels back to all cells.
    if pseudobulk_annotation:
        if isinstance(data, anndata.AnnData):
            _adata_original = _copy_anndata_light(data, copy_input=copy_input)
        elif isinstance(data, (str, os.PathLike)) and os.path.splitext(os.fspath(data))[-1].lower() == ".h5ad":
            _adata_original = sc.read_h5ad(os.fspath(data))
        else:
            raise TypeError(
                "pseudobulk_annotation=True currently supports AnnData objects or .h5ad paths. "
                "Load raw matrices into AnnData first, then call accordion()."
            )

        if not isinstance(cluster_info, str):
            raise ValueError("pseudobulk_annotation=True requires cluster_info to be an adata.obs column name.")

        _pb = _make_pseudobulk_anndata(
            _adata_original,
            cluster_key=cluster_info,
            layer=pseudobulk_layer,
            aggregation=pseudobulk_aggregation,
        )

        _pb_annotated = accordion(
            _pb,
            cluster_info=cluster_info,
            layer="X",
            database=database,
            condition=condition,
            disease=disease,
            NCIT_celltypes=NCIT_celltypes,
            malignant_quantile_threshold=malignant_quantile_threshold,
            ECs_threshold=ECs_threshold,
            SPs_threshold=SPs_threshold,
            is_normalized=is_normalized,
            celltypes=celltypes,
            species=species,
            tissue=tissue,
            include_descendants=include_descendants,
            evidence_consistency_score_threshold=evidence_consistency_score_threshold,
            specificity_score_threshold=specificity_score_threshold,
            log2FC_threshold=log2FC_threshold,
            min_n_marker=min_n_marker,
            max_n_marker=max_n_marker,
            combined_score_quantile_threshold=combined_score_quantile_threshold,
            annotation_resolution="cell",
            cluster_score_quantile_threshold=cluster_score_quantile_threshold,
            allow_unknown=allow_unknown,
            annotation_name=annotation_name,
            include_detailed_annotation_info=include_detailed_annotation_info,
            condition_group_info=condition_group_info,
            celltype_group_info=celltype_group_info,
            group_markers_by="celltype_cell" if group_markers_by in ("cluster", "celltype_cluster") else group_markers_by,
            top_cell_score_quantile_threshold=top_cell_score_quantile_threshold,
            n_top_celltypes=n_top_celltypes,
            n_top_markers=n_top_markers,
            top_marker_score_quantile_threshold=top_marker_score_quantile_threshold,
            entropy=entropy,
            ontology_path=ontology_path,
            min_percentage_celltype_entropy=min_percentage_celltype_entropy,
            do_deconvolution=False,
            deconvolution_threshold=deconvolution_threshold,
            deconvolution_empty_fallback=deconvolution_empty_fallback,
            plot=plot,
            color_by=color_by,
            n_jobs=n_jobs,
            cell_chunk_size=cell_chunk_size,
            pseudobulk_annotation=False,
        )

        return _propagate_pseudobulk_annotation(
            _adata_original,
            _pb_annotated,
            cluster_key=cluster_info,
            annotation_name=annotation_name,
        )

    # Validate and normalize the requested entropy metric(s).
    if isinstance(entropy, str) or entropy is False or entropy is None:
        if entropy not in ("Shannon", "Gini", "Ontology-sp-Rao", "Ontology-Lin-Rao", "Jaccard-Rao", "Overlap-Rao", "all", False, None, "False"):
            print(f"WARNING: invalid entropy value {entropy}; value set to default 'Overlap-Rao'")
            entropy = "Overlap-Rao"
        if isinstance(entropy, str):
            if "Ontology" in entropy or entropy == "all":
                if ontology_path is None:
                    ontology_path = get_default_cell_ontology_path()
    else:
        try:
            for e in entropy:
                if e not in ("Shannon", "Gini", "Ontology-sp-Rao", "Ontology-Lin-Rao", "Jaccard-Rao", "Overlap-Rao", "all", False, None, "False"):
                    print(f"WARNING: invalid entropy value {e}; value set to default 'Overlap-Rao'")
                    entropy = "Overlap-Rao"
                    break
                if "Ontology" in e or entropy == "all":
                    if ontology_path is None:
                        ontology_path = get_default_cell_ontology_path()
        except:
            print(f"WARNING: invalid entropy value {entropy}; value set to default 'Overlap-Rao'")
            entropy = "Overlap-Rao"

    if ontology_path is not None:
        try:
            import obonet
            import networkx as nx
        except:
            raise ModuleNotFoundError("Error: libraries 'obonet' and 'networkx' are required for Ontology-based approaches.")
        try:
            ontology_graph = obonet.read_obo(ontology_path)
        except:
            raise ValueError(f"Error: file in ontology_path:{ontology_path} not found")
        ontology_graph = _isa_only(ontology_graph)
    else:
        ontology_graph = None

    if min_percentage_celltype_entropy is not None:
        if not isinstance(min_percentage_celltype_entropy, (int, float)):
            print("WARNING: min_percentage_celltype_entropy should be a int or float between 0 and 1, defaulting to 0.05")
            min_percentage_celltype_entropy = 5
        else:
            min_percentage_celltype_entropy *= 100

    if deconvolution_threshold is not None:
        if not isinstance(deconvolution_threshold, (int, float)):
            print("WARNING: deconvolution_threshold should be a int or float between 0 and 1, defaulting to 0.05")
            deconvolution_threshold = 0.05

    do_cl_ann = True

    #### RAW COUNTS DATA
    ## CHECK THE TYPE OF INPUT
    if ext in (".tsv", ".csv", ".npz", ".mtx"):
        if ext == ".tsv":
            df_raw = pd.read_csv(data, sep="\t", index_col=0)
            data_type = "matrix"
        elif ext == ".csv":
            df_raw = pd.read_csv(data, index_col=0)
            data_type = "matrix"
        elif ext == ".npz":
            sparse_matrix = sp.load_npz(data)
            try:
                genes = pd.read_csv(features, sep="\t", header=None)[0].values
            except:
                raise ValueError(
                    "Error: uploaded a .npz file but the genes/features .tsv is missing; set the parameter 'features' to a file path.")
            try:
                cells = pd.read_csv(barcodes, sep="\t", header=None)[0].values
            except:
                raise ValueError(
                    "Error: uploaded a .npz file but the cells/barcodes .tsv is missing; set the parameter 'barcodes' to a file path.")
            data_type = "npz_matrix"
            adata = anndata.AnnData(X=sparse_matrix, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
        else:
            sparse_matrix = sc.read_mtx(data)
            try:
                genes = pd.read_csv(features, sep="\t", header=None)[0].values
            except:
                raise ValueError(
                    "Error: uploaded a .mtx file but the genes/features .tsv is missing; set the parameter 'features' to a file path.")
            try:
                cells = pd.read_csv(barcodes, sep="\t", header=None)[0].values
            except:
                raise ValueError(
                    "Error: uploaded a .mtx file but the cells/barcodes .tsv is missing; set the parameter 'barcodes' to a file path.")
            data_type = "mtx_matrix"
            adata = sparse_matrix
            adata.obs = pd.DataFrame(index=cells)
            adata.var = pd.DataFrame(index=genes)
        if data_type == "matrix":
            if df_raw.empty:  # raise ValueError("The count matrix is empty")
                raise ValueError("The count matrix is empty")
            if has_genes_on_rows:
                adata = anndata.AnnData(X=df_raw.T)  # genes on rows, cells on columns --> TRASPOSE to opposite
            else:
                adata = anndata.AnnData(X=df_raw)  # genes on columns, cells on rows --> DEFAULT Anndata
        else:
            if adata.X is None or adata.X.size == 0:
                raise ValueError("The count matrix is empty")
        # annotation resolution
        if "cluster" not in annotation_resolution and "cell" not in annotation_resolution:
            raise ValueError(f"Incorrect annotation resolution {annotation_resolution}; use 'cell', 'cluster' or both")
        elif "cluster" in annotation_resolution and "cell" in annotation_resolution:
            if cluster_info is None:
                raise ValueError(
                    "cluster_info not found. Please provide a .tsv table specifying cell clusters to perform per cluster annotation, or set 'cell' in the annotation_resolution parameter to perform annotation with per cell resolution.")
            else:
                try:
                    ext2 = os.path.splitext(cluster_info)[-1].lower()
                    if ext2 == ".tsv":
                        ci = pd.read_csv(cluster_info, sep="\t")
                    else:
                        print(
                            "WARNING: cluster_info has an incorrect format. It should be a .tsv file. Only cell annotation will be performed.")
                        do_cl_ann = False
                except:
                    print(
                        "WARNING: cluster_info has an incorrect format or file path. Only cell annotation will be performed.")
                try:
                    if cell_column not in ci.columns:
                        print(
                            f"WARNING: cell column not found in cluster_info. Please provide a .tsv with a column named {cell_column} or 'cell' containing cell ids. Only cell annotation will be performed.")
                        do_cl_ann = False
                    elif cluster_column not in ci.columns:
                        print(
                            f"WARNING: cluster column not found in cluster_info. Please provide a .tsv with a column named {cluster_column} or 'cluster' containing cluster ids. Only cell annotation will be performed.")
                        do_cl_ann = False
                    else:
                        cluster_table = ci[[cell_column, cluster_column]].copy()
                        cluster_table.columns = ["cell", "cluster"]
                except:
                    print("WARNING: An error has occurred, please check the cluster_info file")

        elif "cluster" in annotation_resolution and "cell" not in annotation_resolution:
            if cluster_info is None:
                raise ValueError(
                    "cluster_info not found. Please provide a .tsv table specifying cell clusters to perform per cluster annotation, or set 'cell' in the annotation_resolution parameter to perform annotation with per cell resolution.")
            else:
                try:
                    ext2 = os.path.splitext(cluster_info)[-1].lower()
                    if ext2 == ".tsv":
                        ci = pd.read_csv(cluster_info, sep="\t")
                    else:
                        raise ValueError("cluster_info has an incorrect format. It should be a .tsv file")
                except:
                    raise ValueError("cluster_info has an incorrect format or file path.")
                if cell_column not in ci.columns:
                    raise ValueError(
                        f"cell column not found in cluster_info. Please provide a .tsv with a column named {cell_column} or 'cell' containing cell ids.")
                elif cluster_column not in ci.columns:
                    raise ValueError(
                        f"cluster column not found in cluster_info. Please provide a .tsv with a column named {cluster_column} or 'cluster' containing cluster ids.")
                else:
                    cluster_table = ci[[cell_column, cluster_column]].copy()
                    cluster_table.columns = ["cell", "cluster"]

    ## ANNDATA INPUT
    elif ext == ".h5ad":
        data_type = "anndata"
        if input_is_anndata_object:
            adata = _copy_anndata_light(data, copy_input=copy_input)
        else:
            try:
                adata = sc.read_h5ad(data)
            except:
                raise ValueError("Invalid AnnData file in input")
        # invalid layer check
        if layer not in adata.layers and layer != "X":
            raise ValueError(f"Invalid layer {layer} provided")
        # empty object check
        if layer == "X":
            if np.sum(adata.shape) == 0:
                raise ValueError("Count matrix is empty")
        if "cluster" not in annotation_resolution and "cell" not in annotation_resolution:
            raise ValueError(f"Incorrect annotation resolution {annotation_resolution}; use 'cell', 'cluster' or both")
        elif "cluster" in annotation_resolution and "cell" in annotation_resolution:
            if not isinstance(cluster_info, str):
                print(
                    "WARNING: Invalid input type: cluster_info needs to be a string specifying the name of the column in adata.obs containing cluster ids. Only cell annotation will be performed.")
                do_cl_ann = False
            elif cluster_info not in adata.obs.columns:
                print(
                    f"WARNING: \"{cluster_info}\" column not found. Please provide a valid string specifying the name of the column in adata.obs containing cluster ids. Only cell annotation will be performed.")
                do_cl_ann = False
            else:
                # Create cluster table
                cluster_table = pd.DataFrame({
                    'cell': adata.obs_names,
                    'cluster': adata.obs[cluster_info]
                })
                if cluster_info != "leiden" and "leiden" in adata.obs.columns:
                    adata.obs['orig.leiden'] = adata.obs['leiden']
                    adata.obs['leiden'] = adata.obs[cluster_info]

        elif "cluster" in annotation_resolution and "cell" not in annotation_resolution:
            if not isinstance(cluster_info, str):
                raise ValueError(
                    "Invalid input type: cluster_info needs to be a string specifying the name of the column in adata.obs containing cluster ids.")
            elif cluster_info not in adata.obs.columns:
                raise ValueError(
                    f"\"{cluster_info}\" column not found. Please provide a valid string specifying the name of the column in adata.obs containing cluster ids.")
            else:
                # Create cluster table
                cluster_table = pd.DataFrame({
                    'cell': adata.obs_names,
                    'cluster': adata.obs[cluster_info]
                })
                if cluster_info != "leiden" and "leiden" in adata.obs.columns:
                    adata.obs['orig.leiden'] = adata.obs['leiden']
                    adata.obs['leiden'] = adata.obs[cluster_info]
    else:
        raise ValueError(f"Invalid file {data} in input; formats accepted: AnnData object, .h5ad, .tsv, .csv, .mtx, .npz")

    #### Group marker check
    if group_markers_by not in ("cluster", "celltype_cluster", "cell", "celltype_cell", "score_cell"):
        print(
            "WARNING: invalid group_by. Please select \"cluster\", \"celltype_cluster\", \"cell\", \"celltype_cell\" or \"score_cell\"")
        if "cluster" in annotation_resolution:
            group_markers_by = "celltype_cluster"
        elif "cell" in annotation_resolution:
            group_markers_by = "celltype_cell"
    if "cluster" in annotation_resolution and "cell" in annotation_resolution:
        if group_markers_by not in ("cluster", "celltype_cluster", "cell", "celltype_cell", "score_cell"):
            group_markers_by = ("celltype_cluster", "celltype_cell")
    elif "cluster" in annotation_resolution and "cell" not in annotation_resolution:
        if group_markers_by not in ["cluster", "celltype_cluster"]:
            group_markers_by = "celltype_cluster"
    elif "cell" in annotation_resolution and "cluster" not in annotation_resolution:
        if group_markers_by not in ["cell", "celltype_cell", "score_cell"]:
            group_markers_by = "celltype_cell"

    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    #### Normalize data if it is not
    if is_normalized is False:
        if data_type in ("matrix", "mtx_matrix", "npz_matrix"):  # raw data
            adata.layers["raw"] = adata.X.copy()
            print("Normalizing raw data counts")
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
            adata.uns['log1p'] = {'done': True}
        else:  # AnnData
            if layer == "X" and not adata.uns.get('log1p', {}).get('done', False):
                if _looks_like_integer_counts_matrix(_get_matrix_for_layer(adata, layer=layer)):
                    print("Normalizing AnnData counts")
                    adata.layers["raw"] = adata.X.copy()
                    sc.pp.normalize_total(adata, target_sum=10000)
                    sc.pp.log1p(adata)
                    adata.uns['log1p'] = {'done': True}
                else:
                    print("WARNING: Counts seem to be already normalized; normalization will be skipped")

    #### Loading Accordion Marker Database
    accordion_marker = read_marker_database(database=database, condition=condition)
    accordion_marker = normalize_marker_database_schema(accordion_marker, condition=condition)

    # Basic validation: required columns used below must exist.
    required_marker_columns = {"marker", "species", "Uberon_tissue", "Uberon_ID", "CL_celltype", "CL_ID", "marker_type", "resource"}
    missing_marker_columns = required_marker_columns.difference(accordion_marker.columns)
    if missing_marker_columns:
        missing = ", ".join(sorted(missing_marker_columns))
        raise ValueError(f"Invalid marker database: missing required columns: {missing}")


    # Disease-specific filters. Disease marker databases are normalized to
    # CL_celltype/CL_ID internally, while DO_* and NCIT_* columns are preserved.
    if condition == "disease":
        if disease is not None and "DO_diseasetype" in accordion_marker.columns:
            disease_values = [disease] if isinstance(disease, str) else list(disease)
            disease_values_upper = {str(d).upper() for d in disease_values}
            filtered = accordion_marker[
                accordion_marker["DO_diseasetype"].astype(str).str.upper().isin(disease_values_upper)
            ]
            if filtered.empty:
                print("WARNING: disease not found. The entire disease Accordion database is considered")
            else:
                accordion_marker = filtered
                if len(disease_values) > 1:
                    accordion_marker.loc[:, "DO_ID"] = "ALL"
                    accordion_marker.loc[:, "DO_diseasetype"] = "ALL"
        elif disease is None:
            print("WARNING: no specific disease selected. The entire disease Accordion database is considered")

    # filter table by markers present in Anndata object
    var_names = adata.var_names.tolist()
    accordion_marker = accordion_marker[accordion_marker['marker'].isin(var_names)]

    if accordion_marker.empty:
        raise ValueError(
            "The marker database is empty; no markers in common between the Anndata object and the marker database")

    # Filter the markers based on log2FC threshold
    if log2FC_threshold is not None:
        if not isinstance(log2FC_threshold, (int, float)):
            print(
                "WARNING: Invalid log2FC_threshold type. Parameter log2FC_threshold must be a numeric value. No filter is applied")
        else:  # keeps those that have the log2FC value specified + the ones having FC = NA
            accordion_marker = accordion_marker[
                accordion_marker['log2FC'].isna() | (accordion_marker['log2FC'] >= log2FC_threshold)]

    # Filter markers by species. A single species string filters directly; a
    # list/tuple merges markers across species (and uppercases symbols to the
    # human convention) and sums their evidence scores.
    if isinstance(species, str):
        if not (species in accordion_marker["species"].unique()):
            raise ValueError(f"Species '{species}' not found in the dataset")
        else:
            accordion_marker = accordion_marker[accordion_marker["species"] == species]
    else:
        accordion_marker['marker'] = accordion_marker['marker'].str.upper()
        accordion_species = accordion_marker["species"].unique()
        if isinstance(species, (list, tuple)):
            for s in species:
                if s not in accordion_species:
                    raise ValueError(f"Species '{s}' not found in the dataset")
            input_species = species
            ECs = accordion_marker[['CL_celltype', 'marker', 'marker_type', 'ECs_global']].drop_duplicates()
            ECs['ECs_sum'] = ECs.groupby(['CL_celltype', 'marker', 'marker_type'], observed=True)['ECs_global'].transform('sum')
            accordion_marker['species'] = ', '.join(input_species)
            accordion_marker = pd.merge(accordion_marker, ECs, on=['CL_celltype', 'marker', 'marker_type'], how='left')
            accordion_marker = accordion_marker[['species', 'Uberon_tissue', 'Uberon_ID', 'CL_celltype',
                                                 'CL_ID', 'marker', 'marker_type', 'ECs_sum',
                                                 'resource']].drop_duplicates()
            accordion_marker.columns = ['species', 'Uberon_tissue', 'Uberon_ID', 'CL_celltype',
                                        'CL_ID', 'marker', 'marker_type', 'ECs', 'resource']
        else:
            raise ValueError("The parameter 'species' should either be a string or a vector [,] (,)")

    # Filter markers by tissue (optionally including Uberon-ontology descendants
    # when include_descendants=True). Evidence scores are recomputed on the
    # tissue-filtered set below.
    if tissue is not None:
        tissue = [t.lower() for t in tissue] if isinstance(tissue, (list, tuple)) else [tissue.lower()]
        tissue_not_in_accordion = [t for t in tissue if t not in accordion_marker["Uberon_tissue"].unique()]
        accordion_tissue = accordion_marker[accordion_marker["Uberon_tissue"].isin(tissue)]
        if len(tissue_not_in_accordion) == 1:
            if len(accordion_tissue["Uberon_tissue"].unique()) == 0:
                ct_not_present = ", ".join(tissue_not_in_accordion)
                print(
                    f"WARNING: {ct_not_present} tissue is not present. Annotation performed considering all tissues in the database")
                tissue = accordion_marker["Uberon_tissue"].unique().tolist()
            else:
                ct_not_present = ", ".join(tissue_not_in_accordion)
                ct_present = ", ".join(accordion_tissue["Uberon_tissue"].unique())
                print(
                    f"WARNING: {ct_not_present} tissue is not present. Annotation performed considering only {ct_present} tissues")
                accordion_marker = accordion_marker[accordion_marker["Uberon_tissue"].isin([ct_present])]
                tissue = [ct_present]
        elif len(tissue_not_in_accordion) > 1:
            if len(accordion_tissue["Uberon_tissue"].unique()) == 0:
                ct_not_present = ", ".join(tissue_not_in_accordion)
                print(
                    f"WARNING: {ct_not_present} tissues are not present. Annotation performed considering all tissues in the database")
                tissue = accordion_marker["Uberon_tissue"].unique().tolist()
            else:
                ct_not_present = ", ".join(tissue_not_in_accordion)
                ct_present = ", ".join(accordion_tissue["Uberon_tissue"].unique())
                print(
                    f"WARNING: {ct_not_present} tissues are not present. Annotation performed considering only {ct_present} tissues")
                accordion_marker = accordion_marker[accordion_marker["Uberon_tissue"].isin([ct_present])]
                tissue = [ct_present]

        if include_descendants:
            try:
                import obonet
                import networkx as nx
            except Exception:
                raise ModuleNotFoundError(
                    "Error: libraries 'obonet' and 'networkx' are required for include_descendants=True."
                )
            from .io import get_default_uberon_ontology_path
            uberon_graph = obonet.read_obo(get_default_uberon_ontology_path())
            name_to_uberon_id = {}
            for _nid, _data in uberon_graph.nodes(data=True):
                _nm = _data.get("name")
                if _nm:
                    name_to_uberon_id.setdefault(str(_nm).strip().lower(), _nid)

            expanded_tissue = set(tissue)
            for t in tissue:
                t_id = name_to_uberon_id.get(t)
                if t_id is None or t_id not in uberon_graph:
                    continue
                # obonet's `is_a` edges point child -> parent, so nx.ancestors()
                # here returns every more-specific descendant term whose is_a
                # chain eventually reaches `t` -- i.e. exactly the Uberon
                # descendant tissues we want to also include.
                for desc_id in nx.ancestors(uberon_graph, t_id):
                    desc_name = uberon_graph.nodes[desc_id].get("name")
                    if desc_name:
                        expanded_tissue.add(str(desc_name).strip().lower())

            accordion_marker = accordion_marker[accordion_marker['Uberon_tissue'].isin(expanded_tissue)]
            tissue = accordion_marker['Uberon_tissue'].unique().tolist()
            if len(tissue) > 1:
                # Get unique Uberon_tissue and Uberon_ID values
                ub_tissue = ', '.join(accordion_marker['Uberon_tissue'].unique())
                ub_id = ', '.join(accordion_marker['Uberon_ID'].unique())
                # Update the columns using loc to modify all rows
                accordion_marker.loc[:, 'Uberon_tissue'] = ub_tissue
                accordion_marker.loc[:, 'Uberon_ID'] = ub_id
        else:
            accordion_marker = accordion_marker[accordion_marker['Uberon_tissue'].isin(tissue)]
            if len(tissue) > 1:
                # Get unique Uberon_tissue and Uberon_ID values
                ub_tissue = ', '.join(accordion_marker['Uberon_tissue'].unique())
                ub_id = ', '.join(accordion_marker['Uberon_ID'].unique())
                # Update the columns using loc to modify all rows
                accordion_marker.loc[:, 'Uberon_tissue'] = ub_tissue
                accordion_marker.loc[:, 'Uberon_ID'] = ub_id

        _base_cols = ["species", "Uberon_tissue", "Uberon_ID", "CL_celltype",
                      "CL_ID", "marker", "marker_type", "resource"]
        _disease_cols = [c for c in ["DO_diseasetype", "DO_ID", "NCIT_celltype", "NCIT_ID"] if c in accordion_marker.columns]
        accordion_marker = accordion_marker[_disease_cols + _base_cols].drop_duplicates()
        # Avoid categorical groupby Cartesian expansion after tissue filtering.
        for _col in _disease_cols + ["species", "Uberon_tissue", "Uberon_ID", "CL_celltype", "CL_ID", "marker", "marker_type"]:
            if _col in accordion_marker.columns:
                accordion_marker[_col] = accordion_marker[_col].astype("string")
        # Group by and count rows to calculate ECs
        _group_cols = _disease_cols + ["species", "Uberon_tissue", "Uberon_ID", "CL_celltype",
                       "CL_ID", "marker", "marker_type"]
        ECs = accordion_marker.groupby(_group_cols, observed=True).size().reset_index(name="ECs")
        # Merge to add ECs column
        accordion_marker = pd.merge(accordion_marker, ECs, on=_group_cols, how="left")
    else:
        accordion_marker["Uberon_ID"] = "ALL"
        accordion_marker["Uberon_tissue"] = "ALL"
        # Select and rename columns, preserving disease-specific DO/NCIT metadata when present.
        _base_cols = ["species", "Uberon_tissue", "Uberon_ID", "CL_celltype",
                      "CL_ID", "marker", "marker_type", "ECs_global", "resource"]
        _disease_cols = [c for c in ["DO_diseasetype", "DO_ID", "NCIT_celltype", "NCIT_ID"] if c in accordion_marker.columns]
        accordion_marker = accordion_marker[_disease_cols + _base_cols]
        accordion_marker = accordion_marker.rename(columns={"ECs_global": "ECs"})

    # Restrict the marker set to the requested cell type(s), if any.
    if celltypes is not None:
        if not isinstance(celltypes, (str, list, tuple)):
            raise ValueError("'celltypes' must be a string or a vector [,] (,)")

        input_CL_celltype = celltypes
        if isinstance(celltypes, str):
            accordion_marker = accordion_marker[accordion_marker['CL_celltype'] == input_CL_celltype]
        else:
            accordion_marker = accordion_marker[accordion_marker['CL_celltype'].isin(input_CL_celltype)]

        input_CL_celltype_not_in_accordion = [ct for ct in input_CL_celltype if
                                              ct not in accordion_marker['CL_celltype'].unique()]
        if len(input_CL_celltype_not_in_accordion) == 1:
            if accordion_marker['CL_celltype'].nunique() == 0:
                ct_not_present = ', '.join(input_CL_celltype_not_in_accordion)
                print(
                    f"WARNING: {ct_not_present} cell type is not present. Annotation performed considering all cell types in the database.")
                input_CL_celltype = accordion_marker['CL_celltype'].unique()
            else:
                ct_not_present = ', '.join(input_CL_celltype_not_in_accordion)
                ct_present = ', '.join(accordion_marker['CL_celltype'].unique())
                print(
                    f"WARNING: {ct_not_present} cell type is not present. Annotation performed considering only {ct_present} cell types.")
                CL_celltype = ct_present
        elif len(input_CL_celltype_not_in_accordion) > 1:
            if accordion_marker['CL_celltype'].nunique() == 0:
                ct_not_present = ', '.join(input_CL_celltype_not_in_accordion)
                print(
                    f"WARNING: {ct_not_present} cell types are not present. Annotation performed considering all cell types in the database.")
                input_CL_celltype = accordion_marker['CL_celltype'].unique()
            else:
                ct_not_present = ', '.join(input_CL_celltype_not_in_accordion)
                ct_present = ', '.join(accordion_marker['CL_celltype'].unique())
                print(
                    f"WARNING: {ct_not_present} cell types are not present. Annotation performed considering only {ct_present} cell types.")
                CL_celltype = ct_present

        elif len(input_CL_celltype_not_in_accordion) == 0:
            accordion_marker = accordion_marker[accordion_marker['CL_celltype'].isin(input_CL_celltype)]

    # Columns that identify one row of the marker table from here on.
    #
    # The disease database records the same (cell type, marker) under several
    # diseases, and R keeps DO_diseasetype/DO_ID in the de-duplicated table, so
    # those stay separate rows. SPs is a *row count* per (marker, marker_type),
    # so dropping the two columns -- as this code used to -- collapsed them and
    # made SPs systematically smaller: CD3D went from 262 to 95. That propagates
    # through SPs_reg into combined_score and decides which markers survive the
    # max_n_marker cut, which is why four of six cluster winners disagreed with R
    # on PBMC3k. R is the reference for the annotation metrics, so the columns
    # are kept. On the healthy database they are absent and nothing changes.
    _id_cols = [c for c in ("DO_diseasetype", "DO_ID") if c in accordion_marker.columns]
    _core_cols = _id_cols + ["species", "Uberon_tissue", "Uberon_ID",
                             "CL_celltype", "CL_ID", "marker", "marker_type"]

    # Penalize Discordant markers
    accordion_marker["group"] = _as_text_series(accordion_marker["CL_celltype"]) + "_" + _as_text_series(accordion_marker["marker"])
    dup = accordion_marker.drop(columns=["resource"]).drop_duplicates()
    discordant_marker = dup[dup.duplicated(subset='group')]
    if len(discordant_marker) > 0:
        accordion_marker.loc[(accordion_marker["group"].isin(discordant_marker["group"])) & (
                accordion_marker["marker_type"] == "negative"), "ECs"] *= -1
        ECs = accordion_marker[accordion_marker["group"].isin(discordant_marker["group"])].groupby("group", observed=True)[
            "ECs"].sum().reset_index()
        ECs.columns = ["group", "ECs_new"]
        # Add new marker_type column based on sign of ECs_new
        ECs["marker_type_new"] = ECs["ECs_new"].apply(lambda x: "positive" if x > 0 else "negative")
        # Merge back to original dataframe
        accordion_marker = pd.merge(accordion_marker, ECs, on="group", how="left")
        # Take absolute value of ECs_new
        accordion_marker["ECs_new"] = accordion_marker["ECs_new"].abs()
        # Update ECs and marker_type for discordant groups
        accordion_marker.loc[accordion_marker["group"].isin(discordant_marker["group"]), "ECs"] = \
            accordion_marker.loc[accordion_marker["group"].isin(discordant_marker["group"]), "ECs_new"]
        accordion_marker.loc[accordion_marker["group"].isin(discordant_marker["group"]), "marker_type"] = \
            accordion_marker.loc[accordion_marker["group"].isin(discordant_marker["group"]), "marker_type_new"]
        # Keep only necessary columns and unique rows
        accordion_marker = accordion_marker[_core_cols + ["ECs"]]

    # Remove Markers with EC = 0
    accordion_marker = accordion_marker[accordion_marker["ECs"] > 0]
    accordion_marker = accordion_marker[_core_cols + ["ECs"]].drop_duplicates()

    # keep only markers with an EC > EC_threshold
    if evidence_consistency_score_threshold is not None:
        if not isinstance(evidence_consistency_score_threshold, int):
            print(
                "WARNING: Invalid evidence_consistency_score_threshold type. Parameter evidence_consistency_score_threshold must be an integer value. No filter is applied")
        else:  # keeps those that have the log2FC value specified + the ones having FC = NA
            accordion_marker = accordion_marker[accordion_marker['ECs'] >= evidence_consistency_score_threshold]

    # EC log transformed
    accordion_marker['ECs_reg'] = np.log10(accordion_marker['ECs']) + 1

    #### compute SP specificity for +/- markers
    # Calculate marker specificity (SPs)
    mark_spec = accordion_marker.groupby(["marker", "marker_type"], observed=True).size().reset_index(name="SPs")
    accordion_marker = pd.merge(accordion_marker, mark_spec, on=["marker", "marker_type"], how="left")

    # Count unique cell types for positive and negative markers
    length_ct_pos = len(accordion_marker[accordion_marker["marker_type"] == "positive"]["CL_celltype"].unique())
    length_ct_neg = len(accordion_marker[accordion_marker["marker_type"] == "negative"]["CL_celltype"].unique())

    # Scale and transform SPs. ``from`` is given explicitly and reversed, so this
    # first rescale is an inversion: a marker shared by many cell types scores low.
    # Algebraically it is exactly ``length_ct + 1 - SPs``.
    if length_ct_pos > 1:
        accordion_marker.loc[accordion_marker["marker_type"] == "positive", "SPs_reg"] = _rescale(
            accordion_marker.loc[accordion_marker["marker_type"] == "positive", "SPs"].to_numpy(),
            to=(1, length_ct_pos), frm=(length_ct_pos, 1))
    elif length_ct_pos == 1:
        accordion_marker.loc[accordion_marker["marker_type"] == "positive", "SPs_reg"] = 1

    if length_ct_neg > 1:
        accordion_marker.loc[accordion_marker["marker_type"] == "negative", "SPs_reg"] = _rescale(
            accordion_marker.loc[accordion_marker["marker_type"] == "negative", "SPs"].to_numpy(),
            to=(1, length_ct_neg), frm=(length_ct_neg, 1))
    elif length_ct_neg == 1:
        accordion_marker.loc[accordion_marker["marker_type"] == "negative", "SPs_reg"] = 1

    # Keep only necessary columns
    accordion_marker = accordion_marker[_core_cols + ["SPs", "SPs_reg", "ECs_reg", "ECs"]]

    # Further scaling of SPs_reg onto the evidence (ECs) range. 
    # accordion()/accordion_custom() functions do this; accordion_cellcycle()
    # deliberately skips it, keeping the specificity-based SPs_reg. Toggle with
    # rescale_specificity_to_evidence so the cell-cycle path stays faithful.
    if rescale_specificity_to_evidence:
        # Further scaling of SPs_reg for positive markers
        if length_ct_pos > 0:
            min_pos_ECs = accordion_marker[accordion_marker["marker_type"] == "positive"]["ECs"].min()
            max_pos_ECs = accordion_marker[accordion_marker["marker_type"] == "positive"]["ECs"].max()
            if min_pos_ECs == max_pos_ECs:
                min_pos_ECs = max_pos_ECs - 0.00001
            accordion_marker.loc[accordion_marker["marker_type"] == "positive", "SPs_reg"] = _rescale(
                accordion_marker.loc[accordion_marker["marker_type"] == "positive", "SPs_reg"].to_numpy(),
                to=(min_pos_ECs, max_pos_ECs))

        # Further scaling of SPs_reg for negative markers
        if length_ct_neg > 0:
            min_neg_ECs = accordion_marker[accordion_marker["marker_type"] == "negative"]["ECs"].min()
            max_neg_ECs = accordion_marker[accordion_marker["marker_type"] == "negative"]["ECs"].max()
            if min_neg_ECs == max_neg_ECs:
                min_neg_ECs = max_neg_ECs - 0.00001
            accordion_marker.loc[accordion_marker["marker_type"] == "negative", "SPs_reg"] = _rescale(
                accordion_marker.loc[accordion_marker["marker_type"] == "negative", "SPs_reg"].to_numpy(),
                to=(min_neg_ECs, max_neg_ECs))

    # Keep only necessary columns
    accordion_marker = accordion_marker[_core_cols + ["SPs", "SPs_reg", "ECs_reg", "ECs"]]

    # Transform SPs and SPs_reg
    # Two decimals, as R does in accordion(), accordion_disease() and
    # accordion_custom() alike (``round(1/SPs, 2)``). This column is reported in
    # the detailed tables, so the extra digit showed up as a difference there.
    accordion_marker["SPs"] = accordion_marker["SPs"].apply(lambda x: round(1 / float(x), 2))
    accordion_marker["SPs_reg"] = accordion_marker["SPs_reg"].apply(lambda x: np.log10(x) + 1)

    # keep only markers with an SP > SP_threshold
    if specificity_score_threshold is not None:
        if not isinstance(specificity_score_threshold, (int, float)):
            print(
                "WARNING: Invalid specificity_score_threshold type. Parameter specificity_score_threshold must be an integer value between (0,1]. No filter is applied")
        elif specificity_score_threshold <= 0 or specificity_score_threshold > 1:
            print(
                "WARNING: Invalid specificity_score_threshold value. Parameter specificity_score_threshold must be an integer value between (0,1]. No filter is applied")
        else:
            accordion_marker = accordion_marker[accordion_marker['SPs'] >= specificity_score_threshold]
    # Sort the dataframe by marker and CL_celltype, mirroring data.table's
    # setkey. Two things matter here, and both only show up at the
    # max_n_marker cut, where this order is what breaks ties:
    #
    # * the columns are cast out of Categorical first. setkey sorts the strings;
    #   pandas sorts a Categorical by its category codes, and the packaged
    #   database's categories are in insertion order, not alphabetical (its first
    #   two are LHX3 and LY6H). Only the tissue-filtering branch above cast these
    #   columns, so the sort was alphabetical with tissue="blood" and in database
    #   order with tissue=None -- on the healthy database as much as the disease
    #   one.
    # * the sort is stable, again like setkey: with the disease database a
    #   (marker, cell type) pair has one row per disease, and the default
    #   quicksort would order those arbitrarily.
    for _col in ("marker", "CL_celltype"):
        if str(accordion_marker[_col].dtype) == "category":
            accordion_marker[_col] = accordion_marker[_col].astype("string")
    accordion_marker = accordion_marker.sort_values(by=["marker", "CL_celltype"], kind="stable")

    # combined scores (EC, SP)
    accordion_marker["combined_score"] = accordion_marker["SPs_reg"] * accordion_marker["ECs_reg"]

    # filter by combined score quantile threshold
    if combined_score_quantile_threshold is not None:
        if not isinstance(combined_score_quantile_threshold, (int, float)):
            print(
                "WARNING: Invalid combined_score_quantile_threshold type. Parameter combined_score_quantile_threshold must be an integer value between (0,1]. No filter is applied")
        elif combined_score_quantile_threshold <= 0 or combined_score_quantile_threshold > 1:
            print(
                "WARNING: Invalid combined_score_quantile_threshold value. Parameter combined_score_quantile_threshold must be an integer value between (0,1]. No filter is applied")
        else:
            quantile_df = accordion_marker.groupby("CL_celltype", observed=True)["combined_score"].quantile(
                combined_score_quantile_threshold).reset_index()
            quantile_df.columns = ["CL_celltype", "quantile_combined_score"]
            accordion_marker = pd.merge(accordion_marker, quantile_df, on="CL_celltype", how="left")
            accordion_marker = accordion_marker[
                accordion_marker["combined_score"] > accordion_marker["quantile_combined_score"]]

    ## keep only the min_n_marker and max_n_marker genes for each cell type
    accordion_marker["length"] = accordion_marker.groupby("CL_celltype", observed=True)["marker"].transform("count")
    if min_n_marker is not None:
        if not isinstance(min_n_marker, int) or min_n_marker != int(min_n_marker):
            if min_n_marker != 1:
                print(
                    "WARNING: Invalid min_n_marker type. Parameter min_n_marker must be an integer value. No filter is applied")
        else:
            accordion_marker = accordion_marker[accordion_marker["length"] >= min_n_marker]

    if max_n_marker is not None:
        if not isinstance(max_n_marker, int) or max_n_marker != int(max_n_marker):
            print(
                "WARNING: Invalid max_n_marker type. Parameter max_n_marker must be an integer value. No filter is applied")
        else:
            # Sort by combined_score in descending order and keep top max_n_marker for each cell type.
            # kind="stable" is required, not a preference: combined_score is
            # SPs_reg * ECs_reg, both derived from resource counts, so ties at the
            # cutoff are common and pandas' default quicksort breaks them
            # arbitrarily. R uses order(), which is stable, and the marker table
            # is in the same row order in both packages, so a stable sort here
            # selects the same markers R does.
            accordion_marker = accordion_marker.sort_values(
                "combined_score", ascending=False, kind="stable")
            accordion_marker = accordion_marker.groupby("CL_celltype", observed=True).head(max_n_marker)

    # Validate quantile thresholds, falling back to their defaults with a warning.
    if not isinstance(cluster_score_quantile_threshold, (int, float)) or \
            cluster_score_quantile_threshold > 1 or cluster_score_quantile_threshold <= 0:
        print(
            "WARNING: Invalid cluster_score_quantile_threshold type. Parameter cluster_score_quantile_threshold must be a numeric value in (0,1]. Default 0.75 will be used.")
        cluster_score_quantile_threshold = 0.75
    if not isinstance(top_marker_score_quantile_threshold, (int, float)) or \
            top_marker_score_quantile_threshold > 1 or top_marker_score_quantile_threshold <= 0:
        print(
            "WARNING: Invalid top_marker_score_quantile_threshold type. Parameter top_marker_score_quantile_threshold must be a numeric value in (0,1]. Default 0.75 will be used.")
        top_marker_score_quantile_threshold = 0.75

    # store original scale.data slot if present
    if "scaled" in adata.uns and adata.uns["scaled"] is not None:
        scaled = adata.uns["scaled"]
        if hasattr(scaled, "shape") and sum(scaled.shape) != 0:
            adata.uns["original_scaling"] = scaled.copy()

    # check if the accordion table is empty
    if len(accordion_marker) == 0:
        raise ValueError(
            "0 markers found in the expression data table. Please try again by setting different parameters.")

    #### SCALE DATA based on markers used for the annotation
    # Get unique markers and find which ones exist in adata
    unique_markers = accordion_marker["marker"].unique()
    markers_in_adata = np.isin(adata.var_names, unique_markers)
    adata_subset = adata[:, markers_in_adata].copy()
    # Convert to dense format if needed
    if not isinstance(adata_subset.X, np.ndarray):
        adata_subset.X = adata_subset.X.toarray()
    # Z-score each marker gene, following Seurat's ScaleData so that this and the
    # R package score the same matrix. Two details matter and neither is what
    # sklearn's StandardScaler does:
    #
    #   * the standard deviation is the sample one (n-1), as in R's sd()
    #   * the upper tail is clipped at SCALE_MAX, Seurat's scale.max default.
    #     Only the upper tail: Seurat leaves negative values alone, so the
    #     minimum can sit below -SCALE_MAX.
    #
    # Without the clip a marker expressed in a handful of cells reaches a z of
    # 40+ and dominates the cluster score, which is how a 12-cell cluster scored
    # 73.8 here against R's 53.8. Computed in float64 because R works in double.
    _X = np.asarray(adata_subset.X, dtype=np.float64)
    _sd = _X.std(axis=0, ddof=1)
    Zscaled_data = np.minimum(
        (_X - _X.mean(axis=0)) / np.where(_sd > 0, _sd, 1.0), SCALE_MAX,
    ).astype(np.float32, copy=False)
    del _X
    if save_scaled_data:
        adata.uns["scaled"] = adata_subset
        adata.uns["scaled"].X = Zscaled_data
    marker_index = pd.Index(adata_subset.var_names)
    cell_index = pd.Index(adata_subset.obs_names)
    marker_to_col = {m: i for i, m in enumerate(marker_index)}
    n_cells = len(cell_index)
    n_markers_used = len(marker_index)
    n_candidate_celltypes = int(accordion_marker["CL_celltype"].nunique())
    # Score matrices are memory-heavy, so do not build them unless needed.
    # They are required for detailed marker outputs and for cluster Jaccard/Overlap Rao
    # entropy, because those metrics use per-cluster marker-impact tables created in
    # include_detailed_annotation_info_helper.
    _cluster_entropy_needs_marker_tables = _cluster_entropy_requested and any(
        e in _requested_entropy_types for e in ("Jaccard-Rao", "Overlap-Rao")
    )
    need_score_matrices = bool(include_detailed_annotation_info or _cluster_entropy_needs_marker_tables)
    _log_progress(
        f"Scaled marker matrix ready: {n_cells:,} cells x {n_markers_used:,} markers; "
        f"{n_candidate_celltypes:,} candidate cell types"
    )

    # score each candidate cell type without materializing a transposed
    # pandas DataFrame. This keeps final_dt unchanged while reducing temporary RAM.
    def _score_one_cell_type(cell_type):
        ct_markers = accordion_marker[accordion_marker["CL_celltype"] == cell_type]
        if len(ct_markers) == 0:
            return None

        pos_markers_df = ct_markers[ct_markers["marker_type"] == "positive"]
        neg_markers_df = ct_markers[ct_markers["marker_type"] == "negative"]
        pos_markers = [m for m in pos_markers_df["marker"].tolist() if m in marker_to_col]
        neg_markers = [m for m in neg_markers_df["marker"].tolist() if m in marker_to_col]
        if len(pos_markers) == 0 and len(neg_markers) == 0:
            return None

        pos_scores = np.zeros(n_cells, dtype=np.float32)
        neg_scores = np.zeros(n_cells, dtype=np.float32)

        all_markers = pos_markers + neg_markers
        score_matrix = None
        if need_score_matrices:
            score_matrix = pd.DataFrame(0.0, index=cell_index, columns=all_markers, dtype=np.float32)

        if len(pos_markers) > 0:
            pos_norm_data = pos_markers_df[pos_markers_df["marker"].isin(pos_markers)]
            pos_norm = np.sqrt(np.sum(pos_norm_data["ECs_reg"] * pos_norm_data["SPs_reg"])) if len(pos_norm_data) > 0 else 1
        else:
            pos_norm = 1

        if len(neg_markers) > 0:
            neg_norm_data = neg_markers_df[neg_markers_df["marker"].isin(neg_markers)]
            neg_norm = np.sqrt(np.sum(neg_norm_data["ECs_reg"] * neg_norm_data["SPs_reg"])) if len(neg_norm_data) > 0 else 1
        else:
            neg_norm = 1

        for marker in pos_markers:
            marker_data = pos_markers_df[pos_markers_df["marker"] == marker]
            if len(marker_data) == 0:
                continue
            combined_score = np.float32(marker_data["combined_score"].values[0])
            expr_values = Zscaled_data[:, marker_to_col[marker]]
            marker_scores = expr_values * combined_score
            if score_matrix is not None:
                score_matrix[marker] = marker_scores
            pos_scores += marker_scores

        for marker in neg_markers:
            marker_data = neg_markers_df[neg_markers_df["marker"] == marker]
            if len(marker_data) == 0:
                continue
            combined_score = np.float32(marker_data["combined_score"].values[0])
            expr_values = Zscaled_data[:, marker_to_col[marker]]
            marker_scores = expr_values * combined_score
            if score_matrix is not None:
                score_matrix[marker] = marker_scores
            neg_scores += marker_scores

        pos_scores = pos_scores / pos_norm if pos_norm > 0 else pos_scores
        neg_scores = neg_scores / neg_norm if neg_norm > 0 else neg_scores
        diff_scores = (pos_scores - neg_scores).astype(np.float32, copy=False)
        return cell_type, score_matrix, diff_scores

    _cell_types = list(accordion_marker["CL_celltype"].unique())
    total_celltypes = len(_cell_types)
    _log_progress(f"Scoring cell types: 0/{total_celltypes:,} completed")

    celltype_score_matrices = {}
    score_vectors = {}  # celltype -> dense diff_score vector; used for fast annotation/entropy
    result_frames = []
    completed = 0

    def _consume_score_output(output):
        if output is None:
            return
        _cell_type, _score_matrix, _diff_scores = output
        if _score_matrix is not None:
            celltype_score_matrices[_cell_type] = _score_matrix
        score_vectors[_cell_type] = _diff_scores
        result_frames.append(pd.DataFrame({
            "CL_celltype": _cell_type,
            "cell": cell_index,
            "diff_score": _diff_scores,
        }))

    if n_jobs == 1 or total_celltypes <= 1:
        for ct in _cell_types:
            _consume_score_output(_score_one_cell_type(ct))
            completed += 1
            if verbose and (completed == total_celltypes or completed % progress_every == 0):
                pct = 100.0 * completed / max(total_celltypes, 1)
                _log_progress(f"Scoring cell types: {completed:,}/{total_celltypes:,} completed ({pct:.1f}%)")
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_ct = {executor.submit(_score_one_cell_type, ct): ct for ct in _cell_types}
            for future in as_completed(future_to_ct):
                _consume_score_output(future.result())
                completed += 1
                if verbose and (completed == total_celltypes or completed % progress_every == 0):
                    pct = 100.0 * completed / max(total_celltypes, 1)
                    _log_progress(f"Scoring cell types: {completed:,}/{total_celltypes:,} completed ({pct:.1f}%)")

    # Build compact score matrix once. final_dt is still preserved below for
    # transparency, but annotation/entropy can now use this n_cells x n_celltypes
    # float32 matrix instead of repeatedly filtering an 80M+ row pandas table.
    score_celltypes = [ct for ct in _cell_types if ct in score_vectors]
    if score_celltypes:
        score_matrix_fast = np.column_stack([score_vectors[ct] for ct in score_celltypes]).astype(np.float32, copy=False)
    else:
        score_matrix_fast = np.empty((n_cells, 0), dtype=np.float32)
    _log_progress(f"Fast score matrix ready: {score_matrix_fast.shape[0]:,} cells x {score_matrix_fast.shape[1]:,} cell types")

    # Create final dataframe. This table is intentionally preserved because it
    # exposes the per-cell/per-celltype scores used by Accordion.
    if result_frames:
        final_dt = pd.concat(result_frames, ignore_index=True)
        final_dt["diff_score"] = final_dt["diff_score"].astype(np.float32, copy=False)
        final_dt["CL_celltype"] = final_dt["CL_celltype"].astype("category")
        final_dt["cell"] = final_dt["cell"].astype("category")
    else:
        final_dt = pd.DataFrame(columns=["CL_celltype", "cell", "diff_score"])
    _log_progress(f"final_dt built with {len(final_dt):,} rows")

    accordion_output = {}

    # Store full per-cell, per-celltype scores for downstream cell-level entropy.
    if isinstance(adata, anndata.AnnData):
        adata.uns.setdefault(annotation_name, {})
        adata.uns[annotation_name]["celltype_scores"] = final_dt.copy()
        adata.uns[annotation_name]["parallel"] = {
            "n_jobs": n_jobs,
            "cell_chunk_size": cell_chunk_size,
        }
        if condition == "disease":
            adata.uns[annotation_name]["condition"] = "disease"
            adata.uns[annotation_name]["disease_filter"] = disease
            adata.uns[annotation_name]["NCIT_celltypes_filter"] = NCIT_celltypes

    ###  _accordion_progress(5, 8, 'Running cluster annotation', accordion_start_time)

 # annotation per cluster
    if "cluster" in annotation_resolution and do_cl_ann:
        n_clusters = cluster_table["cluster"].nunique() if "cluster_table" in locals() and "cluster" in cluster_table.columns else "unknown"
        _log_progress(f"Running cluster annotation on {n_clusters} clusters")
        # Merge final_dt and cluster_table
        final_dt_cluster = pd.merge(final_dt, cluster_table, on='cell')
        # Calculate quantile score per cluster and cell type
        final_dt_cluster['quantile_score_cluster'] = (
            final_dt_cluster.groupby(['cluster', 'CL_celltype'], observed=True)['diff_score'].transform(
                lambda x: np.quantile(x, cluster_score_quantile_threshold)))
        # Annotation assignment
        if allow_unknown:
            final_dt_cluster['annotation_per_cell'] = np.where(
                final_dt_cluster['quantile_score_cluster'] < 0, 'unknown',
                final_dt_cluster['CL_celltype'])
        else:
            final_dt_cluster['annotation_per_cell'] = final_dt_cluster['CL_celltype']

        anno_dt_cell = final_dt_cluster.sort_values(by="diff_score", ascending=False).groupby("cell", observed=True).head(1)

        anno_dt_cell["ncell_celltype_cluster"] = \
            anno_dt_cell.groupby(["cluster", "annotation_per_cell"], observed=True)["cell"].transform("count")
        anno_dt_cell["ncell_tot_cluster"] = anno_dt_cell.groupby("cluster", observed=True)["cell"].transform("count")
        anno_dt_cell["perc_celltype_cluster"] = round(
            (anno_dt_cell["ncell_celltype_cluster"] / anno_dt_cell["ncell_tot_cluster"]) * 100, 3)

        anno_dt_cl = (
            anno_dt_cell.sort_values(by="quantile_score_cluster", ascending=False)
                .groupby("cluster", observed=True)
                .head(1)
                .drop(columns=["cell", "CL_celltype", "diff_score", "ncell_tot_cluster", "ncell_celltype_cluster"]))
        anno_dt_cl = anno_dt_cl[["cluster", "annotation_per_cell", "quantile_score_cluster", "perc_celltype_cluster"]]
        anno_dt_cl.columns = ["cluster", "annotation_per_cluster", "quantile_score_cluster", "percentage"]

        # Mark clusters with low percentage as unknown
        if allow_unknown:
            anno_dt_cl.loc[anno_dt_cl['percentage'] < 10, 'annotation_per_cluster'] = 'unknown'
        # Prepare output names
        name = f'{annotation_name}_per_cluster'
        name_score = f'{annotation_name}_per_cluster_score'
        if data_type == "anndata":
            adata.obs[name] = ""
            # Use a float NaN (not "") so the score column stays numeric even for
            # clusters that are not (re)written below; otherwise it becomes an
            # object column with mixed str/float values that .h5ad cannot save.
            adata.obs[name_score] = np.nan
            _clusters_to_write = list(anno_dt_cl["cluster"].unique())
            for _i, cl in enumerate(_clusters_to_write, start=1):
                if verbose and (_i == len(_clusters_to_write) or _i % progress_every == 0):
                    _log_progress(f"Writing cluster annotations: {_i:,}/{len(_clusters_to_write):,} clusters")
                mask = adata.obs[cluster_info] == cl
                adata.obs.loc[mask, name] = \
                    anno_dt_cl.loc[anno_dt_cl["cluster"] == cl, "annotation_per_cluster"].values[0]
                adata.obs.loc[mask, name_score] = \
                    anno_dt_cl.loc[anno_dt_cl["cluster"] == cl, "quantile_score_cluster"].values[0]
            adata.obs[name_score] = pd.to_numeric(adata.obs[name_score], errors="coerce")
        else:
            cluster_table = cluster_table.merge(anno_dt_cl[["cluster", "annotation_per_cluster"]], on="cluster",
                                                how="left")
            cluster_table = cluster_table[["cell", "cluster", "annotation_per_cluster"]]
            cluster_table.columns = ["cell", "cluster", name]
            scaled_matrix = adata.uns["scaled"]
            accordion_output = {"scaled_matrix": scaled_matrix, "cluster_annotation": cluster_table}

        # NOTE: cluster-level entropy is deliberately NOT computed here.
        # The correct cluster entropy definition depends on the detailed
        # top-celltypes table and (for Jaccard/Overlap Rao) the per-cluster
        # marker-impact table built in include_detailed_annotation_info_helper.
        # Computing it from anno_dt_cl here collapses/changes the distribution
        # and gives wrong values. The helper computes and saves the correct
        # cluster entropy to adata.obs.

    ###  _accordion_progress(6, 8, 'Running cell annotation', accordion_start_time)

  # annotation per cell
    if "cell" in annotation_resolution:
        _log_progress(f"Running cell annotation on {adata.n_obs:,} cells")
        name = f"{annotation_name}_per_cell"
        name_score = f"{annotation_name}_per_cell_score"
        adata.obs[name] = ""
        adata.obs[name_score] = ""
        # Fast cell annotation from compact matrix. This is equivalent to taking
        # the max diff_score per cell from final_dt, but avoids sorting/grouping
        # the full long table.
        if score_matrix_fast.shape[1] > 0:
            best_idx = np.argmax(score_matrix_fast, axis=1)
            best_scores = score_matrix_fast[np.arange(score_matrix_fast.shape[0]), best_idx].astype(np.float32, copy=False)
            best_celltypes = np.asarray(score_celltypes, dtype=object)[best_idx]
            if allow_unknown:
                best_ann = np.where(best_scores < 0, "unknown", best_celltypes)
            else:
                best_ann = best_celltypes
            anno_dt_cell = pd.DataFrame({
                "cell": cell_index.astype(str),
                "CL_celltype": best_celltypes,
                "diff_score": best_scores,
                "annotation_per_cell": best_ann,
            })
            anno_dt_cell = anno_dt_cell.set_index("cell").reindex(adata.obs.index.astype(str)).reset_index()
        else:
            anno_dt_cell = final_dt.sort_values(by="diff_score", ascending=False).groupby("cell", observed=True).head(1)
            if allow_unknown:
                anno_dt_cell["annotation_per_cell"] = np.where(anno_dt_cell["diff_score"] < 0, "unknown", anno_dt_cell["CL_celltype"])
            else:
                anno_dt_cell["annotation_per_cell"] = anno_dt_cell["CL_celltype"]
            anno_dt_cell = anno_dt_cell.set_index("cell").reindex(adata.obs.index).reset_index()
        if data_type == "anndata":
            adata.obs[name] = anno_dt_cell["annotation_per_cell"].values
            adata.obs[name_score] = anno_dt_cell["diff_score"].values

            # Per-spot cell-type composition (deconvolution proportions). When
            # do_deconvolution is on (e.g. accordion_spatial_deconvolution), store
            # for every spot/cell the fraction attributable to each candidate cell
            # type in adata.obsm[f"{annotation_name}_proportions"] (rows sum to 1),
            # plus the assigned type's percentage in
            # adata.obs[f"{annotation_name}_per_cell_percentage"].
            if do_deconvolution and score_matrix_fast.shape[1] > 0:
                _pos = np.clip(np.asarray(score_matrix_fast, dtype=float), 0.0, None)
                _rs = _pos.sum(axis=1, keepdims=True)
                _prop = np.divide(_pos, _rs, out=np.zeros_like(_pos), where=_rs > 0)
                # Keep the pre-threshold normalised proportions so we can fall back
                # to them for spots that the threshold would otherwise empty out.
                _prop_norm = _prop.copy()
                _n_cap = int(n_top_celltypes) if n_top_celltypes else 0
                # Drop contributions below the deconvolution threshold.
                if deconvolution_threshold:
                    _prop[_prop < float(deconvolution_threshold)] = 0.0
                # Keep at most n_top_celltypes per spot: a spot is a mixture of a
                # few cell types, not all candidates. Zero out the rest.
                if _n_cap and _prop.shape[1] > _n_cap:
                    _keep = np.argsort(-_prop, axis=1)[:, :_n_cap]
                    _mask = np.zeros_like(_prop, dtype=bool)
                    np.put_along_axis(_mask, _keep, True, axis=1)
                    _prop = np.where(_mask, _prop, 0.0)
                # Fallback: a spot whose signal is spread over many cell types (none
                # individually above deconvolution_threshold) would get its whole row
                # zeroed and end up with an empty, uninformative composition even
                # though it has positive signal. For those spots, restore the top
                # types from the pre-threshold proportions so they still carry a
                # composition instead of becoming all-zero.
                _emptied = (_prop.sum(axis=1) == 0) & (_rs.squeeze(axis=1) > 0)
                if deconvolution_empty_fallback and _emptied.any():
                    _n_fb = _n_cap if _n_cap else _prop_norm.shape[1]
                    _n_fb = min(_n_fb, _prop_norm.shape[1])
                    _sub = _prop_norm[_emptied]
                    _keep_fb = np.argsort(-_sub, axis=1)[:, :_n_fb]
                    _mask_fb = np.zeros_like(_sub, dtype=bool)
                    np.put_along_axis(_mask_fb, _keep_fb, True, axis=1)
                    _prop[_emptied] = np.where(_mask_fb, _sub, 0.0)
                # Renormalise so each spot's kept fractions sum to 1.
                _rs2 = _prop.sum(axis=1, keepdims=True)
                _prop = np.divide(_prop, _rs2, out=np.zeros_like(_prop), where=_rs2 > 0)
                _comp = pd.DataFrame(
                    _prop, index=pd.Index(cell_index, dtype=str),
                    columns=[str(c) for c in score_celltypes])
                _comp = _comp.reindex(pd.Index(adata.obs_names, dtype=str)).fillna(0.0)
                _comp.index = adata.obs_names
                adata.obsm[f"{annotation_name}_proportions"] = _comp
                adata.obs[f"{annotation_name}_per_cell_percentage"] = \
                    (_comp.max(axis=1) * 100.0).astype(np.float32).to_numpy()

            # Cell-level entropy is intentionally saved to adata.obs, independently
            # from include_detailed_annotation_info. This preserves the previous
            # user-facing output while allowing heavy detailed matrices to stay off.
            if save_entropy_to_obs and entropy is not None and entropy is not False and entropy != "False":
                _log_progress("Computing cell-level entropy for adata.obs")
                try:
                    entropy_types = [entropy] if isinstance(entropy, str) else list(entropy)
                    if "all" in entropy_types:
                        entropy_types = [
                            "Shannon",
                            "Gini",
                            "Ontology-sp-Rao",
                            "Ontology-Lin-Rao",
                            "Jaccard-Rao",
                            "Overlap-Rao",
                        ]

                    dt_all_marker_for_entropy = None
                    if any(_e in entropy_types for _e in ("Jaccard-Rao", "Overlap-Rao")):
                        dt_all_marker_for_entropy = _build_marker_table_for_cell_entropy(
                            accordion_marker,
                            annotation_name=annotation_name,
                        )
                        if dt_all_marker_for_entropy is None or getattr(dt_all_marker_for_entropy, "empty", True):
                            print(
                                "WARNING: Jaccard/Overlap entropy requested but marker table is empty; "
                                "those entropy columns will not be produced.",
                                flush=True,
                            )

                    _log_progress("Computing cell-level entropy using fast score-matrix path")
                    cell_entropy_results = calculate_entropies_from_score_matrix(
                        score_matrix_fast,
                        cells=cell_index.astype(str),
                        celltypes=score_celltypes,
                        entropy_types=entropy_types,
                        ontology_graph=ontology_graph,
                        dt_all_marker=dt_all_marker_for_entropy,
                        annot_name=name,
                        min_percentage_celltype_entropy=min_percentage_celltype_entropy,
                        top_cell_score_quantile_threshold=top_cell_score_quantile_threshold,
                        n_top_celltypes=n_top_celltypes,
                        allow_unknown=allow_unknown,
                        progress_callback=_entropy_progress_callback,
                        progress_every=progress_every,
                        progress_label="cell-level entropy",
                        n_jobs=n_jobs,
                        cell_chunk_size=max(1000, int(progress_every) * 1000),
                        unit_label="cells",
                    )
                    if not cell_entropy_results:
                        raise ValueError(
                            f"calculate_spot_entropies returned no results for entropy={entropy_types}. "
                            "Check ontology_graph for ontology entropies or marker table for Jaccard/Overlap."
                        )

                    adata.uns.setdefault(annotation_name, {})
                    # fast path avoids materializing the huge cell_entropy_input table
                    adata.uns[annotation_name]["cell_entropy_marker_table"] = dt_all_marker_for_entropy
                    adata.uns[annotation_name]["cell_entropy"] = cell_entropy_results

                    saved_entropy_cols = []
                    obs_index_as_str = pd.Index(adata.obs_names.astype(str))
                    for _entropy_type, _df in cell_entropy_results.items():
                        if not (hasattr(_df, "columns") and "cell" in _df.columns):
                            continue
                        adata.uns[annotation_name][f"cell_entropy_{_entropy_type}"] = _df
                        for _metric_col in [c for c in _df.columns if c != "cell"]:
                            obs_col = f"{annotation_name}_{_metric_col}"
                            entropy_map = dict(zip(_df["cell"].astype(str), _df[_metric_col]))
                            adata.obs[obs_col] = obs_index_as_str.map(entropy_map).astype("float32")
                            saved_entropy_cols.append(obs_col)

                    if saved_entropy_cols:
                        _log_progress("Cell-level entropy saved to adata.obs: " + ", ".join(saved_entropy_cols))
                    else:
                        raise ValueError("entropy results were computed, but no metric columns were saved to adata.obs")

                except Exception as exc:
                    print(f"WARNING: could not calculate/save cell-level entropy to adata.obs: {exc}", flush=True)
                    if verbose:
                        traceback.print_exc()
        else:
            cell_table = anno_dt_cell[["cell", "annotation_per_cell", "diff_score"]]
            cell_table.columns = ["cell", name, name_score]
            if accordion_output is not None and len(accordion_output) > 0:
                accordion_output["cell_annotation"] = cell_table
            else:
                accordion_output = {"cell_annotation": cell_table}
                if save_scaled_data and "scaled" in adata.uns:
                    accordion_output["scaled_matrix"] = adata.uns["scaled"]

    # Safe defaults for outputs that are only created for some annotation resolutions.
    if "final_dt_cluster" not in locals():
        final_dt_cluster = None
    if "celltype_score_matrices" not in locals():
        celltype_score_matrices = {}

    # detailed annotation info.  Cluster-level entropy is also computed here,
    # because the correct definition needs the top-N celltypes per cluster and,
    # for Jaccard/Overlap Rao, the marker-impact tables built by the helper.
    _run_detailed_helper = bool(include_detailed_annotation_info or _cluster_entropy_requested)
    if _run_detailed_helper:
        if include_detailed_annotation_info:
            _log_progress("Computing detailed annotation information")
        else:
            _log_progress("Computing cluster-level entropy via detailed helper")
        if data_type == "anndata":
            adata = include_detailed_annotation_info_helper(adata,
                                                            data_type,
                                                            annotation_resolution,
                                                            annotation_name,
                                                            accordion_marker,
                                                            group_markers_by,
                                                            cluster_info,
                                                            do_cl_ann,
                                                            final_dt_cluster,
                                                            allow_unknown,
                                                            celltype_score_matrices,
                                                            n_top_celltypes,
                                                            n_top_markers,
                                                            top_marker_score_quantile_threshold,
                                                            top_cell_score_quantile_threshold,
                                                            condition_group_info,
                                                            celltype_group_info,
                                                            entropy,
                                                            ontology_graph,
                                                            min_percentage_celltype_entropy,
                                                            do_deconvolution,
                                                            deconvolution_threshold,
                                                            n_jobs=n_jobs,
                                                            cell_chunk_size=cell_chunk_size,
                                                            verbose=verbose,
                                                            progress_every=progress_every,
                                                            progress_callback=_entropy_progress_callback)
            # Final safety pass: the helper stores cluster-level entropy tables in
            # adata.uns[annotation_name]["detailed_annotation_info"].  Map them to
            # adata.obs here as well so the output is present even if a helper-side
            # save path was skipped by a column-name mismatch.
            if (
                save_entropy_to_obs
                and isinstance(adata, anndata.AnnData)
                and "cluster" in annotation_resolution
                and cluster_info in adata.obs.columns
            ):
                try:
                    detailed_info = adata.uns.get(annotation_name, {}).get("detailed_annotation_info", {})
                    obs_cluster_values = adata.obs[cluster_info].astype(str)
                    saved_cluster_entropy_cols = []
                    for _key, _df in detailed_info.items():
                        if not (_key.startswith("cluster_entropy_") and hasattr(_df, "columns") and len(_df) > 0):
                            continue
                        _df = _df.copy()
                        _cluster_cols = [c for c in (cluster_info, "cluster") if c in _df.columns]
                        if not _cluster_cols:
                            print(
                                f"WARNING: cluster entropy table {_key} has no cluster column; columns={list(_df.columns)}",
                                flush=True,
                            )
                            continue
                        _cluster_col = _cluster_cols[0]
                        _metric_cols = [c for c in _df.columns if c != _cluster_col and c.endswith("_entropy")]
                        if not _metric_cols:
                            _metric_cols = [c for c in _df.columns if c != _cluster_col]
                        for _metric_col in _metric_cols:
                            _entropy_map = dict(zip(_df[_cluster_col].astype(str), _df[_metric_col]))
                            _obs_col = f"{annotation_name}_cluster_{_metric_col}"
                            adata.obs[_obs_col] = obs_cluster_values.map(_entropy_map).astype("float32")
                            saved_cluster_entropy_cols.append(_obs_col)
                    if saved_cluster_entropy_cols:
                        _log_progress("Cluster-level entropy confirmed in adata.obs: " + ", ".join(saved_cluster_entropy_cols))
                    elif entropy is not None and entropy is not False and entropy != "False" and "cluster" in annotation_resolution:
                        _log_progress("WARNING: cluster-level entropy requested but no cluster entropy table was available to save to adata.obs")
                except Exception as exc:
                    print(f"WARNING: could not finalize cluster-level entropy in adata.obs: {exc}", flush=True)
                    if verbose:
                        traceback.print_exc()
    # Safety net for .h5ad writing. An obs column left as object dtype with mixed
    # str/float values (e.g. a score initialized as "" and partially filled with
    # floats, or a label column with NaN for unannotated cells) cannot be saved
    # ("Can't implicitly convert non-string objects to strings"). Coerce the
    # score columns to float and the annotation-label columns to clean strings.
    if data_type == "anndata":
        _label_cols = {f"{annotation_name}_per_cluster", f"{annotation_name}_per_cell"}
        for _col in adata.obs.columns:
            if adata.obs[_col].dtype != object:
                continue
            if str(_col).endswith("_score"):
                adata.obs[_col] = pd.to_numeric(
                    adata.obs[_col].replace("", np.nan), errors="coerce")
            elif _col in _label_cols:
                adata.obs[_col] = adata.obs[_col].fillna("unknown").astype(str)

        # uns detailed tables can carry nullable "string" columns (StringArray)
        # that anndata < 0.11 cannot write; convert them to plain object strings.
        _sanitize_for_h5ad(adata.uns)

    _log_progress("Annotation completed")
    if data_type == "anndata":
        return adata
    else:
        return accordion_output
