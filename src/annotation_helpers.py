"""Helpers for building the detailed annotation tables and cell-level entropy.

These functions are called by :func:`pycellmarkeraccordion.accordion` after the per-cell
scoring is done. They assemble the ``detailed_annotation_info`` tables (top cell
types per cluster, top markers per cell type/cluster, etc.) stored in
``adata.uns[annotation_name]`` and compute the cell-level annotation-entropy
metrics. They are internal helpers and are not part of the stable public API.
"""

import networkx.exception
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
from .spatial_entropy import calculate_spot_entropies, calculate_spot_entropies_parallel
from . import entropy_core as _ec
from .entropy_core import normalized_shannon as _normalized_shannon, raw_gini as _raw_gini



from .utils import (                             # noqa: F401
    DEFAULT_N_JOBS,
    resolve_n_jobs as _resolve_n_jobs,
    as_text_series as _as_text_series,
)



def _ensure_cluster_aliases(df, cluster_info=None):
    """Ensure marker/detail tables have a usable generic cluster alias.

    Some paths rename the cluster column to the user-facing AnnData key
    (for example "leiden"), while other downstream code still expects
    "cluster". This helper keeps the existing columns and adds the alias
    only when it can do so safely.
    """
    if df is None or not hasattr(df, "columns"):
        return df
    if "cluster" in df.columns:
        return df
    candidates = []
    if cluster_info is not None:
        candidates.append(cluster_info)
    candidates.extend(["leiden", "louvain", "seurat_clusters", "clusters"])
    for col in candidates:
        if col in df.columns:
            out = df.copy()
            out["cluster"] = out[col]
            return out
    return df


def _build_cell_entropy_table_from_score_matrices(
        celltype_score_matrices,
        annotation_name,
        top_cell_score_quantile_threshold=0.90,
        n_top_celltypes=5):
    """Create a cell-level composition-like table from per-cell celltype score matrices.

    This is used when annotation_resolution='cell' and do_deconvolution=False.
    Each row is a candidate cell type for a cell. Scores are clipped at 0 and
    normalized within each cell to obtain significance_percentage.
    """
    if not celltype_score_matrices:
        return None

    rows = []
    for celltype, score_df in celltype_score_matrices.items():
        if score_df is None or getattr(score_df, "empty", True):
            continue

        # Per-cell aggregate score for this cell type.
        try:
            cell_scores = score_df.sum(axis=1, numeric_only=True)
        except TypeError:
            cell_scores = score_df.select_dtypes(include="number").sum(axis=1)

        if cell_scores is None or len(cell_scores) == 0:
            continue

        for cell, score in cell_scores.items():
            rows.append({
                "cell": cell,
                f"{annotation_name}_per_cell": celltype,
                "annotation_per_cell": celltype,
                "diff_score": float(score) if pd.notna(score) else 0.0,
                "celltype_impact_score": float(score) if pd.notna(score) else 0.0,
            })

    if not rows:
        return None

    entropy_df = pd.DataFrame(rows)
    entropy_df["celltype_impact_score"] = entropy_df["celltype_impact_score"].clip(lower=0)

    # Keep only candidate cell types above a per-cell quantile, plus at least top N.
    kept = []
    for cell, sub in entropy_df.groupby("cell", observed=True):
        sub = sub.sort_values("celltype_impact_score", ascending=False).copy()
        if len(sub) > n_top_celltypes:
            q = sub["celltype_impact_score"].quantile(top_cell_score_quantile_threshold)
            sub_q = sub[sub["celltype_impact_score"] >= q]
            sub_top = sub.head(n_top_celltypes)
            sub = pd.concat([sub_q, sub_top], ignore_index=True).drop_duplicates(
                subset=["cell", f"{annotation_name}_per_cell"]
            )
        kept.append(sub)

    entropy_df = pd.concat(kept, ignore_index=True)

    # Convert scores to within-cell percentages.
    entropy_df["_positive_score"] = entropy_df["celltype_impact_score"].clip(lower=0)
    sums = entropy_df.groupby("cell", observed=True)["_positive_score"].transform("sum")

    # If all scores are zero for a cell, assign equal weights.
    n_per_cell = entropy_df.groupby("cell", observed=True)["cell"].transform("count")
    entropy_df["significance_percentage"] = np.where(
        sums > 0,
        entropy_df["_positive_score"] / sums,
        1.0 / n_per_cell,
    )
    entropy_df["percentage"] = entropy_df["significance_percentage"] * 100.0
    entropy_df = entropy_df.drop(columns=["_positive_score"])

    return entropy_df



def _build_cell_entropy_table_from_final_dt(
        final_dt,
        annotation_name,
        top_cell_score_quantile_threshold=0.90,
        n_top_celltypes=5,
        allow_unknown=True):
    """Create a cell-level composition-like table from full per-celltype diff scores."""
    if final_dt is None or getattr(final_dt, "empty", True):
        return None

    required = {"CL_celltype", "cell", "diff_score"}
    if not required.issubset(set(final_dt.columns)):
        return None

    df = final_dt[["CL_celltype", "cell", "diff_score"]].copy()
    df = df.dropna(subset=["cell", "CL_celltype", "diff_score"])

    if df.empty:
        return None

    kept = []
    for cell, sub in df.groupby("cell", observed=True):
        sub = sub.sort_values("diff_score", ascending=False).copy()

        if allow_unknown:
            sub = sub[sub["diff_score"] > 0]

        if sub.empty:
            continue

        if len(sub) > n_top_celltypes:
            q = sub["diff_score"].quantile(top_cell_score_quantile_threshold)
            sub_q = sub[sub["diff_score"] >= q]
            sub_top = sub.head(n_top_celltypes)
            sub = pd.concat([sub_q, sub_top], ignore_index=True).drop_duplicates(
                subset=["cell", "CL_celltype"]
            )

        kept.append(sub)

    if not kept:
        return None

    entropy_df = pd.concat(kept, ignore_index=True)
    annot_col = f"{annotation_name}_per_cell"
    entropy_df[annot_col] = entropy_df["CL_celltype"].astype(str)
    entropy_df["annotation_per_cell"] = entropy_df[annot_col]
    entropy_df["celltype_impact_score"] = entropy_df["diff_score"].clip(lower=0)

    sums = entropy_df.groupby("cell", observed=True)["celltype_impact_score"].transform("sum")
    n_per_cell = entropy_df.groupby("cell", observed=True)["cell"].transform("count")

    entropy_df["significance_percentage"] = np.where(
        sums > 0,
        entropy_df["celltype_impact_score"] / sums,
        1.0 / n_per_cell,
    )
    entropy_df["percentage"] = entropy_df["significance_percentage"] * 100.0

    return entropy_df[[
        "cell",
        annot_col,
        "annotation_per_cell",
        "diff_score",
        "celltype_impact_score",
        "significance_percentage",
        "percentage",
    ]]


def _build_marker_table_for_cell_entropy(accordion_marker, annotation_name):
    """Build marker-set table compatible with calculate_spot_entropies for Rao entropies."""
    if accordion_marker is None or getattr(accordion_marker, "empty", True):
        return None

    required = {"CL_celltype", "marker", "marker_type"}
    if not required.issubset(set(accordion_marker.columns)):
        return None

    annot_col = f"{annotation_name}_per_cell"
    marker_df = accordion_marker[["CL_celltype", "marker", "marker_type"]].drop_duplicates().copy()
    marker_df[annot_col] = marker_df["CL_celltype"].astype(str)
    marker_df["annotation_per_cell"] = marker_df[annot_col]
    marker_df["gene_impact_score_per_cluster"] = 1.0

    return marker_df[[
        "marker",
        "marker_type",
        "gene_impact_score_per_cluster",
        annot_col,
        "annotation_per_cell",
    ]]




def _parallel_compute_top_markers_cluster(
        celltype_score_matrices,
        anno_dt_cell,
        top_marker_score_quantile_threshold,
        n_top_markers,
        n_jobs=None,
        cluster_annotation=None):
    """Compute top marker tables per (celltype, cluster), optionally in parallel.

    Returns ``(top_marker_dfs, extra_markers_dfs)``. The two have different scopes
    on purpose:

    * ``top_marker_dfs`` feeds the reported top-markers tables, and is limited to
      the ``(cluster, the cell type annotating it)`` pairs named by
      ``cluster_annotation``. Without that limit the quantile is also computed for
      every cell type that a stray cell in the cluster happens to carry: on PBMC3k
      a "dendritic cell" block came from the single dendritic cell sitting in
      cluster 3, and its quantile over one cell beat the real cluster-4 value and
      entered the list. R filters first
      (``dt_top_marker[CL_celltype == annotation_per_cluster]``), so this
      reproduces it.
    * ``extra_markers_dfs`` feeds ``dt_all_marker``, which the marker-set entropies
      (Jaccard-Rao, Overlap-Rao) need for **every** candidate cell type, not just
      the winners. So every pair is still computed for it.
    """
    n_jobs = _resolve_n_jobs(n_jobs)
    if cluster_annotation:
        cluster_annotation = {str(k): str(v) for k, v in cluster_annotation.items()}

    tasks = []
    for cl_type, score_df in celltype_score_matrices.items():
        sub_anno = anno_dt_cell[anno_dt_cell["annotation_per_cell"] == cl_type]
        if sub_anno.empty:
            continue
        for cluster_id in sub_anno["cluster"].unique():
            is_winner = (not cluster_annotation
                         or cluster_annotation.get(str(cluster_id)) == str(cl_type))
            tasks.append((cl_type, cluster_id, score_df, sub_anno, is_winner))

    def _work(task):
        cl_type, cluster_id, score_df, sub_anno, is_winner = task
        cluster_cells = sub_anno[sub_anno["cluster"] == cluster_id]["cell"]
        marker_scores = score_df.loc[score_df.index.intersection(cluster_cells)]
        if marker_scores.empty:
            return None

        quantile_scores = marker_scores.quantile(
            q=top_marker_score_quantile_threshold,
            axis=0,
            numeric_only=True,
        )

        top_markers = (
            quantile_scores
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={
                    "index": "marker",
                    top_marker_score_quantile_threshold: "quantile_score_marker",
                })
        )

        top_markers["annotation_per_cell"] = cl_type
        top_markers["cluster"] = cluster_id

        extra_markers = top_markers.copy()
        # Only the cluster's own cell type contributes to the reported table; every
        # pair still contributes to extra_markers, which dt_all_marker needs.
        top_markers = top_markers.head(n_top_markers) if is_winner else None

        return top_markers, extra_markers

    if not tasks:
        return [], []

    if n_jobs == 1 or len(tasks) == 1:
        outputs = [_work(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            outputs = list(executor.map(_work, tasks))

    outputs = [o for o in outputs if o is not None]
    top_marker_dfs = [o[0] for o in outputs if o[0] is not None]
    extra_markers_dfs = [o[1] for o in outputs]
    return top_marker_dfs, extra_markers_dfs


def _compute_top_markers_per_celltype_cell(
        celltype_score_matrices,
        assignment,
        accordion_marker,
        n_top_markers,
        top_marker_score_quantile_threshold,
        annotation_name,
        condition=None):
    """Top markers per *assigned cell type* at cell resolution.

    Analogous to :func:`_parallel_compute_top_markers_cluster`, but groups cells
    by the cell type assigned to them (per-cell annotation) rather than by
    cluster. ``assignment`` is a pandas Series indexed by cell name holding the
    assigned cell type.

    If ``condition`` (a per-cell Series of condition labels) is given, the top
    markers are computed *separately within each condition* and a ``condition``
    column is added, so the signature genes can be compared across conditions.

    Returns a DataFrame with one block of ``n_top_markers`` rows per cell type
    (and per condition, when given), with the marker metadata (``marker_type``,
    ``ECs``, ``SPs``) from the marker database.
    """
    name = f"{annotation_name}_per_cell"
    score_col = "gene_impact_score_per_celltype_cell"

    assignment = assignment.astype(str)
    assignment.index = assignment.index.astype(str)
    if condition is not None:
        condition = condition.astype(str)
        condition.index = condition.index.astype(str)

    blocks = []
    for celltype, score_df in celltype_score_matrices.items():
        cells_ct = set(assignment.index[assignment.values == str(celltype)])
        if not cells_ct:
            continue
        idx_str = score_df.index.astype(str)

        # One group of cells overall, or one group per condition value.
        if condition is None:
            groups = {None: cells_ct}
        else:
            groups = {}
            for cond_val in pd.unique(condition.reindex(list(cells_ct)).dropna()):
                cond_cells = cells_ct & set(condition.index[condition.values == cond_val])
                if cond_cells:
                    groups[cond_val] = cond_cells

        for cond_val, cells in groups.items():
            sub = score_df.loc[idx_str.isin(cells)]
            if sub.empty:
                continue
            quantile_scores = sub.quantile(
                q=top_marker_score_quantile_threshold, axis=0, numeric_only=True)
            top = (
                quantile_scores.sort_values(ascending=False).reset_index()
                .rename(columns={"index": "marker",
                                 top_marker_score_quantile_threshold: score_col})
            )
            top[name] = str(celltype)
            if cond_val is not None:
                top["condition"] = str(cond_val)
            blocks.append(top.head(n_top_markers))

    cols = ["marker", score_col, name, "marker_type", "ECs", "SPs"]
    if condition is not None:
        cols.insert(3, "condition")
    if not blocks:
        return pd.DataFrame(columns=cols)

    # Markers with a non-positive score are kept, so that every cell type gets
    # the full n_top_markers rows. The R package does the same -- it orders by
    # score and takes the head, with no sign filter -- and R is the reference for
    # the annotation tables. Dropping them is arguably more honest, since a
    # negative score is evidence *against* the cell type rather than weak
    # evidence for it, but that is a change to make on both sides at once.
    out = pd.concat(blocks, ignore_index=True)

    marker_meta = accordion_marker.rename(columns={"CL_celltype": name})
    meta_cols = [c for c in [name, "marker", "marker_type", "ECs", "SPs"] if c in marker_meta.columns]
    out = out.merge(marker_meta[meta_cols].drop_duplicates(), on=[name, "marker"], how="left")
    return out.reset_index(drop=True)



def include_detailed_annotation_info_helper(
        data,
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
        n_jobs=None,
        cell_chunk_size=500,
        verbose=False,
        progress_every=10,
        progress_callback=None):
    cell_res_detailed_annotation_info = {}

    # Per-resolution containers for the detailed annotation tables.
    cluster_res_detailed_annotation_info = {}
    # Cluster column used in detailed annotation tables.
    ci = cluster_info if data_type == "anndata" else "cluster"
    progress_every = max(1, int(progress_every))

    def _entropy_progress(entropy_type, done, total, unit="clusters", phase="processing", label="cluster-level entropy"):
        if progress_callback is not None:
            try:
                progress_callback(entropy_type=entropy_type, done=done, total=total, unit=unit, phase=phase, label=label)
            except Exception:
                pass
        elif verbose:
            pct = 100.0 * done / max(total, 1)
            print(f"[Accordion] {label}: {entropy_type} {phase} {done:,}/{total:,} {unit} ({pct:.1f}%)", flush=True)
    # Cell-only mode: no cluster-level final_dt_cluster is available.
    if "cluster" not in annotation_resolution and final_dt_cluster is None:
        if isinstance(data, anndata.AnnData):
            data.uns.setdefault(annotation_name, {})
            data.uns[annotation_name]["detailed_annotation_info"] = cell_res_detailed_annotation_info

            # Cell-resolution candidate table: for every cell, its top
            # n_top_celltypes cell types with their score, which is what makes the
            # per-cell annotation inspectable (why this label and what came
            # close). The R package returns the same table at cell resolution.
            # Built from the per-cell, per-celltype scores stored just before this
            # helper runs -- R's final_dt.
            try:
                scores = data.uns.get(annotation_name, {}).get("celltype_scores")
                if scores is not None and not scores.empty:
                    name_ct = f"{annotation_name}_per_cell"
                    ranked = scores.sort_values(
                        "diff_score", ascending=False, kind="stable")
                    top_ct = ranked.groupby(
                        "cell", observed=True, sort=False).head(n_top_celltypes)
                    # Group the rows by cell, cells ordered by their best score,
                    # matching how R lays the table out. The astype(object) is
                    # required: "cell" is categorical, and mapping a categorical
                    # Series returns a categorical whose own category order --
                    # here the original cell order -- is what sort_values would
                    # then sort by, silently ignoring the integers.
                    cell_order = {c: i for i, c in
                                  enumerate(ranked["cell"].drop_duplicates())}
                    top_ct = (top_ct
                              .assign(_o=top_ct["cell"].astype(object)
                                      .map(cell_order).astype("int64"))
                              .sort_values("_o", kind="stable")
                              .drop(columns="_o"))
                    top_ct = (top_ct[["cell", "CL_celltype", "diff_score"]]
                              .rename(columns={"CL_celltype": name_ct,
                                               "diff_score": f"{name_ct}_score"})
                              .reset_index(drop=True))
                    cell_res_detailed_annotation_info["top_celltypes"] = top_ct
            except Exception as exc:
                if verbose:
                    print(f"[Accordion] could not build cell-resolution top celltypes: {exc}", flush=True)

            # Cell-resolution detailed marker table: top markers per assigned
            # cell type. Mirrors the cluster-resolution table so the same plots
            # (e.g. plot_markers_lollipop) work at cell resolution. Wrapped in
            # try/except so a failure never breaks the annotation itself.
            try:
                cell_col = f"{annotation_name}_per_cell"
                if celltype_score_matrices and cell_col in data.obs:
                    # Per-cell condition labels, when requested, split the top
                    # markers by condition.
                    cond_series = None
                    if condition_group_info and condition_group_info in data.obs:
                        cond_series = data.obs[condition_group_info]
                    top_markers_cell = _compute_top_markers_per_celltype_cell(
                        celltype_score_matrices,
                        data.obs[cell_col],
                        accordion_marker,
                        n_top_markers,
                        top_marker_score_quantile_threshold,
                        annotation_name,
                        condition=cond_series,
                    )
                    if not top_markers_cell.empty:
                        cell_res_detailed_annotation_info["top_markers_per_celltype_cell"] = top_markers_cell
            except Exception as exc:
                if verbose:
                    print(f"[Accordion] could not build cell-resolution top markers: {exc}", flush=True)

        _cell_entropy_already_done = (
            isinstance(data, anndata.AnnData)
            and isinstance(data.uns.get(annotation_name, {}), dict)
            and bool(data.uns.get(annotation_name, {}).get("cell_entropy"))
        )
        if entropy is not None and entropy != "False" and entropy is not False and _cell_entropy_already_done:
            if verbose:
                print("[Accordion] cell-level entropy already computed by fast path; skipping helper recomputation", flush=True)
        if entropy is not None and entropy != "False" and entropy is not False and not _cell_entropy_already_done:
            try:
                entropy_types = [entropy] if isinstance(entropy, str) else list(entropy)
                if "all" in entropy_types:
                    entropy_types = ["Shannon", "Gini", "Ontology-sp-Rao", "Ontology-Lin-Rao", "Jaccard-Rao", "Overlap-Rao"]

                final_dt_scores = None
                if isinstance(data, anndata.AnnData):
                    final_dt_scores = data.uns.get(annotation_name, {}).get("celltype_scores")

                entropy_input = _build_cell_entropy_table_from_final_dt(
                    final_dt_scores,
                    annotation_name=annotation_name,
                    top_cell_score_quantile_threshold=top_cell_score_quantile_threshold,
                    n_top_celltypes=n_top_celltypes,
                    allow_unknown=allow_unknown,
                )

                if entropy_input is None:
                    entropy_input = _build_cell_entropy_table_from_score_matrices(
                        celltype_score_matrices,
                        annotation_name=annotation_name,
                        top_cell_score_quantile_threshold=top_cell_score_quantile_threshold,
                        n_top_celltypes=n_top_celltypes,
                    )

                if entropy_input is not None:
                    annot_col = f"{annotation_name}_per_cell"
                    entropy_name_col = annot_col if annot_col in entropy_input.columns else "annotation_per_cell"

                    dt_all_marker_for_entropy = None
                    if any(_e in entropy_types for _e in ("Jaccard-Rao", "Overlap-Rao")):
                        dt_all_marker_for_entropy = _build_marker_table_for_cell_entropy(
                            accordion_marker,
                            annotation_name=annotation_name,
                        )

                    cell_entropy_results = calculate_spot_entropies(
                        entropy_input,
                        entropy_types=entropy_types,
                        ontology_graph=ontology_graph,
                        dt_all_marker=dt_all_marker_for_entropy,
                        name=entropy_name_col,
                        annot_name=entropy_name_col,
                        min_percentage_celltype_entropy=min_percentage_celltype_entropy,
                        progress_callback=progress_callback,
                        progress_every=progress_every,
                        progress_label="cell-level entropy",
                    )

                    if isinstance(data, anndata.AnnData):
                        data.uns.setdefault(annotation_name, {})
                        data.uns[annotation_name]["cell_entropy_input"] = entropy_input
                        data.uns[annotation_name]["cell_entropy_marker_table"] = dt_all_marker_for_entropy
                        data.uns[annotation_name]["cell_entropy"] = cell_entropy_results

                        for _entropy_type, _df in cell_entropy_results.items():
                            if hasattr(_df, "columns") and "cell" in _df.columns:
                                data.uns[annotation_name][f"cell_entropy_{_entropy_type}"] = _df
                                for _metric_col in [c for c in _df.columns if c != "cell"]:
                                    data.obs[f"{annotation_name}_{_metric_col}"] = (
                                        data.obs_names.astype(str).map(dict(zip(_df["cell"].astype(str), _df[_metric_col])))
                                    )
                else:
                    print("WARNING: cell-level entropy requested, but no per-celltype score table was available.")
            except Exception as exc:
                print(f"WARNING: could not calculate cell-level entropy: {exc}")

        return data

    #### Detailed Annotation handler
    if "cluster" in annotation_resolution and do_cl_ann and final_dt_cluster is not None:
        anno_dt_cell = final_dt_cluster.sort_values(by="diff_score", ascending=False).groupby("cell", observed=True).head(1)
        anno_dt_cell["ncell_celltype_cluster"] = \
            anno_dt_cell.groupby(["cluster", "annotation_per_cell"], observed=True)["cell"].transform("count")
        anno_dt_cell["ncell_tot_cluster"] = anno_dt_cell.groupby("cluster", observed=True)["cell"].transform("count")
        anno_dt_cell["perc_celltype_cluster"] = round(
            (anno_dt_cell["ncell_celltype_cluster"] / anno_dt_cell["ncell_tot_cluster"]) * 100, 3)
        anno_dt_cl = (anno_dt_cell.drop(columns=["cell", "diff_score"]).drop_duplicates()
            .sort_values("quantile_score_cluster", ascending=False).groupby("cluster", observed=True)
            .head(n_top_celltypes)[
            ["cluster", "annotation_per_cell", "quantile_score_cluster", "ncell_tot_cluster", "perc_celltype_cluster"]])

        # Prepare output names
        name = f'{annotation_name}_per_cluster'
        name_score = f'{annotation_name}_per_cluster_score'

        anno_dt_cl.columns = ["cluster", name, "celltype_impact_score", "ncell_tot_cluster", "percentage"]

        # to filter with later (anno_dt_cl_top)
        if data_type == "anndata":
            anno_dt_cl = anno_dt_cl.rename(columns={"cluster": cluster_info})
            anno_dt_cl_top = anno_dt_cl.loc[
                anno_dt_cl.groupby(cluster_info, observed=True)["celltype_impact_score"].idxmax()].reset_index(
                drop=True)
        else:
            anno_dt_cl_top = anno_dt_cl.loc[
                anno_dt_cl.groupby("cluster", observed=True)["celltype_impact_score"].idxmax()].reset_index(drop=True)

        cluster_res_detailed_annotation_info["top_celltypes"] = anno_dt_cl.sort_values(by=ci)

        # Note: low-percentage clusters are already marked "unknown" in
        # accordion.py's own allow_unknown handling (anno_dt_cl['percentage']
        # < 10), so it isn't repeated here.

        valid_keys = set(anno_dt_cl[name].unique())
        celltype_score_matrices = {k: v for k, v in celltype_score_matrices.items() if k in valid_keys}

        # Each cluster's winning cell type. The marker quantiles are computed only
        # for these (cluster, cell type) pairs, as in R, so a cell type that a
        # handful of stray cells carry in some other cluster cannot contribute a
        # top marker for it.
        _cluster_annotation = dict(zip(
            anno_dt_cl_top[ci].astype(str), anno_dt_cl_top[name].astype(str)))

        if "cluster" in group_markers_by:
            top_marker_dfs, extra_markers_dfs = _parallel_compute_top_markers_cluster(
                celltype_score_matrices=celltype_score_matrices,
                anno_dt_cell=anno_dt_cell,
                top_marker_score_quantile_threshold=top_marker_score_quantile_threshold,
                n_top_markers=n_top_markers,
                n_jobs=n_jobs,
                cluster_annotation=_cluster_annotation,
            )

            # Combine all clusters into one dataframe
            if not top_marker_dfs:
                dt_top_marker_by_cl = pd.DataFrame(columns=["marker", "quantile_score_marker", "annotation_per_cell", "cluster"])
                dt_all_marker = dt_top_marker_by_cl.copy()
            else:
                dt_top_marker_by_cl = pd.concat(top_marker_dfs, ignore_index=True)
            accordion_marker_renamed = accordion_marker.rename(columns={"CL_celltype": "annotation_per_cell"})
            dt_top_marker_by_cl = dt_top_marker_by_cl.merge(
                accordion_marker_renamed[["annotation_per_cell", "marker", "marker_type", "ECs", "SPs"]],
                on=["annotation_per_cell", "marker"], how="left")
            dt_top_marker_by_cl.columns = ["marker", "gene_impact_score_per_cluster", name, "cluster", "marker_type",
                                           "ECs", "SPs"]
            if data_type == "anndata":
                dt_top_marker_by_cl = dt_top_marker_by_cl.rename(columns={"cluster": cluster_info})
            dt_top_marker_by_cl = _ensure_cluster_aliases(dt_top_marker_by_cl, cluster_info=cluster_info)

            # Create set of allowed (cluster, celltype) pairs.
            # Be robust about the cluster column name: at this point the
            # annotation table may use the user-facing name (e.g. "leiden"),
            # while marker tables can still use the generic "cluster" column on
            # some paths.  Do not assume they are already aligned.
            _anno_cluster_col = next(
                (c for c in (ci, cluster_info, "cluster") if c in anno_dt_cl_top.columns),
                None,
            )
            _score_cluster_col = next(
                (c for c in (ci, cluster_info, "cluster") if c in dt_top_marker_by_cl.columns),
                None,
            )
            if _anno_cluster_col is None or _score_cluster_col is None:
                raise KeyError(
                    "Could not identify cluster column while building top markers. "
                    f"anno columns={list(anno_dt_cl_top.columns)}, "
                    f"marker columns={list(dt_top_marker_by_cl.columns)}"
                )

            valid_pairs = set(
                tuple(row) for row in anno_dt_cl_top[[_anno_cluster_col, name]].to_numpy()
            )
            # Filter dt_top_marker_by_cl using those valid pairs
            filtered_dt = dt_top_marker_by_cl[
                dt_top_marker_by_cl.apply(
                    lambda row: (row[_score_cluster_col], row[name]) in valid_pairs, axis=1
                )
            ]
            filtered_dt = _ensure_cluster_aliases(filtered_dt, cluster_info=cluster_info)
            if _score_cluster_col not in filtered_dt.columns:
                _score_cluster_col = next((c for c in (ci, cluster_info, "cluster") if c in filtered_dt.columns), _score_cluster_col)
            top_markers_per_cluster = (
                filtered_dt
                    .sort_values([_score_cluster_col, "gene_impact_score_per_cluster"], ascending=[True, False])
                    .groupby(_score_cluster_col, as_index=False, observed=True)
                    .head(n_top_markers)
            )

            cluster_res_detailed_annotation_info["top_markers_per_cluster"] = top_markers_per_cluster
            #########################################################################################
            dt_all_marker = pd.concat(extra_markers_dfs, ignore_index=True)
            dt_all_marker = dt_all_marker.merge(
                accordion_marker_renamed[["annotation_per_cell", "marker", "marker_type", "ECs", "SPs"]],
                on=["annotation_per_cell", "marker"], how="left")
            dt_all_marker.columns = ["marker", "gene_impact_score_per_cluster", name, "cluster", "marker_type",
                                     "ECs", "SPs"]
            if data_type == "anndata":
                dt_all_marker = dt_all_marker.rename(columns={"cluster": cluster_info})
            dt_all_marker = _ensure_cluster_aliases(dt_all_marker, cluster_info=cluster_info)
            # Kept before the de-duplication below, which collapses each
            # (marker, cell type) to a single row and so to a single cluster's
            # quantile score -- whichever block came first in the concatenation.
            # The marker-set entropies need the score in the cluster they are
            # scoring, so they use this table instead. See the Jaccard/Overlap
            # block further down.
            dt_all_marker_by_cluster = dt_all_marker.copy()
            dt_all_marker = dt_all_marker.drop_duplicates(["marker", name])
            filtered_dt = dt_all_marker[
                dt_all_marker['marker'].isin(
                    set(cluster_res_detailed_annotation_info["top_markers_per_cluster"]['marker'])) &
                dt_all_marker[name].isin(
                    set(cluster_res_detailed_annotation_info["top_markers_per_cluster"][name]))
                ]

            shared_combos = set(zip(cluster_res_detailed_annotation_info["top_markers_per_cluster"][name],
                                    cluster_res_detailed_annotation_info["top_markers_per_cluster"]['marker']))
            filtered_dt = filtered_dt[
                ~filtered_dt.apply(lambda row: (row[name], row['marker']) in shared_combos, axis=1)]
            filtered_dt = filtered_dt[filtered_dt['gene_impact_score_per_cluster'] > 0]
            cluster_res_detailed_annotation_info["complete_markers_per_cluster"] = filtered_dt

        if "celltype_cluster" in group_markers_by:
            valid_celltypes = anno_dt_cl_top[name].unique()
            filtered_dt = dt_top_marker_by_cl[dt_top_marker_by_cl[name].isin(valid_celltypes)]
            filtered_dt = (
                filtered_dt
                    .sort_values("gene_impact_score_per_cluster", ascending=False)
                    .drop_duplicates(subset=["marker", name])
            )
            top_markers_per_celltype_cluster = (
                filtered_dt
                    .sort_values([name, "gene_impact_score_per_cluster"], ascending=[True, False])
                    .groupby(name, as_index=False, observed=True)
                    .head(n_top_markers)
            )

            cluster_res_detailed_annotation_info["top_markers_per_celltype_cluster"] = top_markers_per_celltype_cluster
            ###########################################################################################################
            filtered_dt = dt_all_marker[
                dt_all_marker['marker'].isin(
                    set(cluster_res_detailed_annotation_info["top_markers_per_celltype_cluster"]['marker'])) &
                dt_all_marker[name].isin(
                    set(cluster_res_detailed_annotation_info["top_markers_per_celltype_cluster"][name]))
                ]
            shared_combos = set(zip(cluster_res_detailed_annotation_info["top_markers_per_celltype_cluster"][name],
                                    cluster_res_detailed_annotation_info["top_markers_per_celltype_cluster"]['marker']))
            filtered_dt = filtered_dt[
                ~filtered_dt.apply(lambda row: (row[name], row['marker']) in shared_combos, axis=1)]
            filtered_dt = filtered_dt[filtered_dt['gene_impact_score_per_cluster'] > 0]
            cluster_res_detailed_annotation_info["complete_markers_per_celltype_cluster"] = filtered_dt


        if entropy is not None and entropy != "False" and entropy is not False:
            if isinstance(entropy, str):
                entropy = [entropy]
            if "all" in entropy:
                entropy = list(_ec.METRICS)
            entropy_top_celltypes = cluster_res_detailed_annotation_info["top_celltypes"].copy()
            entropy_top_celltypes = entropy_top_celltypes[
                (entropy_top_celltypes["percentage"] >= min_percentage_celltype_entropy) &
                (~entropy_top_celltypes[name].str.contains("unknown", case=False, na=False))
                ]
            _cluster_total_for_entropy = int(entropy_top_celltypes[ci].nunique()) if ci in entropy_top_celltypes.columns else 0

            def _cluster_probabilities(clust):
                """``(celltypes, p)`` for one cluster.

                ``p`` is ``percentage * celltype_impact_score`` renormalized to
                sum 1, in the table's own row order, so it lines up with the
                returned cell-type labels and hence with any distance submatrix
                indexed by them.
                """
                df_clust = entropy_top_celltypes[entropy_top_celltypes[ci] == clust].copy()
                df_clust["p_weighted"] = df_clust["percentage"] * df_clust["celltype_impact_score"]
                df_clust["p_weighted"] /= df_clust["p_weighted"].sum()
                cts = df_clust[name].unique()
                return cts, df_clust.set_index(name)["p_weighted"].reindex(cts).values

            def _run_metric(metric, value_of):
                """Loop the clusters, report progress, store the result table.

                ``value_of(celltypes, p, cluster)`` returns the scalar for one
                cluster.
                """
                clusters = entropy_top_celltypes[ci].unique()
                _entropy_progress(metric, 0, len(clusters), unit="clusters",
                                  phase="starting", label="cluster-level entropy")
                rows = []
                for _i, clust in enumerate(clusters, start=1):
                    cts, p_vec = _cluster_probabilities(clust)
                    rows.append({ci: clust, f"{metric}_entropy": value_of(cts, p_vec, clust)})
                    if _i == len(clusters) or _i % progress_every == 0:
                        _entropy_progress(metric, _i, len(clusters), unit="clusters",
                                          label="cluster-level entropy")
                cluster_res_detailed_annotation_info[f"cluster_entropy_{metric}"] = pd.DataFrame(rows)

            def _matrix_progress(metric):
                """Throttle the distance-matrix progress callback to progress_every."""
                def _cb(done, total):
                    if done == total or done % progress_every == 0:
                        _entropy_progress(metric, done, total, unit="celltypes",
                                          phase="building distance matrix",
                                          label="cluster-level entropy")
                return _cb

            if "Shannon" in entropy:
                _run_metric("Shannon", lambda cts, p, cl: _normalized_shannon(p))
            if "Gini" in entropy:
                _run_metric("Gini", lambda cts, p, cl: _raw_gini(p))

            _ont_requested = [m for m in _ec.ONTOLOGY_METRICS if m in entropy]
            if _ont_requested:
                _ont_celltypes = sorted(set(entropy_top_celltypes[name]))
                _unmapped = _ec.unmapped_celltypes(_ont_celltypes, ontology_graph)
                if _unmapped:
                    print(_ec.skip_message(_ont_requested, _unmapped,
                                           "have no Cell Ontology term"))
                    _ont_requested = []
                for _ont_metric in _ont_requested:
                    # One matrix for the whole run; each cluster then uses the
                    # submatrix for its own cell types.  For "Ontology-sp-Rao"
                    # entropy_core also divides by the matrix maximum, so the
                    # values are relative to the cell types present here.
                    _ont_dist = pd.DataFrame(
                        _ec.ontology_distance_matrix(
                            _ont_celltypes, ontology_graph, _ont_metric,
                            missing="nan", progress=_matrix_progress(_ont_metric)),
                        index=_ont_celltypes, columns=_ont_celltypes)
                    _run_metric(_ont_metric,
                                lambda cts, p, cl, _D=_ont_dist: _ec.rao(p, _D.loc[cts, cts].values))

            _mk_requested = [m for m in _ec.MARKER_METRICS if m in entropy]
            if _mk_requested:
                # Marker sets per cluster: for the cluster being scored, each
                # candidate's markers with a positive impact score *in that
                # cluster*. That is what Jaccard and Overlap are meant to compare
                # -- how distinguishable two candidates are by the evidence
                # present here -- and it is well defined.
                #
                # The earlier version took the sets from the de-duplicated
                # dt_all_marker, where drop_duplicates(["marker", name]) had
                # already collapsed each (marker, cell type) to one cluster's
                # score: whichever block came first in the concatenation, i.e. a
                # function of dict iteration order rather than of the data. For a
                # cell type that is a candidate in a single cluster the two agree;
                # otherwise the old sets came from an unrelated cluster.
                _mk_cl_col = next((c for c in (ci, cluster_info, "cluster")
                                   if c in dt_all_marker_by_cluster.columns), None)
                _mk_sets_by_cluster = {}
                if _mk_cl_col is not None:
                    for _cl, _sub in dt_all_marker_by_cluster.groupby(_mk_cl_col, observed=True):
                        _mk_sets_by_cluster[str(_cl)] = _ec.marker_sets(_sub, name)

                def _sets_for(clust):
                    return _mk_sets_by_cluster.get(str(clust), {})

                # Only a guard: a cell type reaches this table by being scored
                # from its own markers, so it normally has a marker set. Checked
                # per cluster, since the sets are now per cluster.
                _no_markers = sorted({
                    ct
                    for _cl in entropy_top_celltypes[ci].unique()
                    for ct in _ec.celltypes_without_markers(
                        entropy_top_celltypes.loc[entropy_top_celltypes[ci] == _cl, name],
                        _sets_for(_cl))
                })
                if _no_markers:
                    print(_ec.skip_message(_mk_requested, _no_markers,
                                           "have no markers left in the marker table"))
                    _mk_requested = []
                for _mk_metric in _mk_requested:
                    _run_metric(_mk_metric, lambda cts, p, cl, _k=_mk_metric: _ec.rao(
                        p, _ec.marker_distance_matrix(cts, _sets_for(cl), _k, missing="nan")))

        # Save detailed annotation outputs into AnnData.
        if isinstance(data, anndata.AnnData):
            data.uns.setdefault(annotation_name, {})
            data.uns[annotation_name]["detailed_annotation_info"] = cluster_res_detailed_annotation_info

            try:
                top_df = cluster_res_detailed_annotation_info.get("top_celltypes")
                if top_df is not None:
                    top_df = top_df.copy()
                    cluster_col = cluster_info if data_type == "anndata" else "cluster"
                    annot_col = f"{annotation_name}_per_cluster"

                    if cluster_col in top_df.columns and annot_col in top_df.columns:
                        idx = top_df.groupby(cluster_col, observed=True)["celltype_impact_score"].idxmax()
                        top_one = top_df.loc[idx, [cluster_col, annot_col, "celltype_impact_score"]].copy()

                        label_map = dict(zip(top_one[cluster_col].astype(str), top_one[annot_col]))
                        score_map = dict(zip(top_one[cluster_col].astype(str), top_one["celltype_impact_score"]))

                        data.obs[annot_col] = data.obs[cluster_info].astype(str).map(label_map)
                        data.obs[f"{annot_col}_score"] = data.obs[cluster_info].astype(str).map(score_map)
            except Exception as exc:
                print(f"WARNING: could not save cluster annotation to adata.obs: {exc}")

            # Save cluster-level entropy tables into `adata.uns` and map each
            # cluster-level value back to every cell in `adata.obs`.  Be deliberately
            # tolerant about the cluster column name: depending on the code path the
            # entropy table can contain either the user-facing `cluster_info` column
            # (e.g. "leiden") or a generic "cluster" column.
            try:
                obs_cluster_values = data.obs[cluster_info].astype(str)
                saved_cluster_entropy_cols = []

                for key, df in cluster_res_detailed_annotation_info.items():
                    if not (key.startswith("cluster_entropy_") and hasattr(df, "columns")):
                        continue
                    if df is None or len(df) == 0:
                        if verbose:
                            print(f"[Accordion] cluster-level entropy table {key} is empty; nothing to save", flush=True)
                        continue

                    df = df.copy()
                    data.uns[annotation_name][key] = df

                    possible_cluster_cols = [
                        c for c in (cluster_info, ci, "cluster")
                        if c in df.columns
                    ]
                    if not possible_cluster_cols:
                        print(
                            f"WARNING: could not save {key} to adata.obs: no cluster column found. "
                            f"Available columns: {list(df.columns)}",
                            flush=True,
                        )
                        continue

                    entropy_cluster_col = possible_cluster_cols[0]
                    metric_cols = [
                        c for c in df.columns
                        if c != entropy_cluster_col and c.endswith("_entropy")
                    ]
                    if not metric_cols:
                        metric_cols = [c for c in df.columns if c != entropy_cluster_col]

                    for metric_col in metric_cols:
                        # Main explicit cluster-level name.
                        obs_col = f"{annotation_name}_cluster_{metric_col}"
                        entropy_map = dict(zip(df[entropy_cluster_col].astype(str), df[metric_col]))
                        data.obs[obs_col] = obs_cluster_values.map(entropy_map).astype("float32")
                        saved_cluster_entropy_cols.append(obs_col)


                if verbose and saved_cluster_entropy_cols:
                    print(
                        "[Accordion] cluster-level entropy saved to adata.obs: "
                        + ", ".join(saved_cluster_entropy_cols),
                        flush=True,
                    )
                elif verbose:
                    print("[Accordion] no cluster-level entropy columns were produced to save to adata.obs", flush=True)
            except Exception as exc:
                print(f"WARNING: could not save entropy results to adata.obs: {exc}", flush=True)

    # Cell-resolution detailed tables are assembled elsewhere; nothing extra to
    # do here for the "cell" resolution.
    #
    # Note: do_deconvolution's actual per-cell deconvolution (score-matrix
    # normalization + deconvolution_threshold cutoff) is implemented in
    # accordion.py (the vectorized score_matrix_fast/_prop path used by
    # accordion_spatial_deconvolution); it isn't repeated here.

    return data
