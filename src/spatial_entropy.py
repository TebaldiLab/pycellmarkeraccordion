"""Annotation-entropy metrics and the spatial entropy plot.

This module computes the per-cell / per-spot annotation entropy from the
candidate cell-type score matrix using several diversity measures (Shannon,
Gini and the Rao quadratic-entropy variants: Jaccard-, Overlap-, Ontology-sp-
and Ontology-Lin-Rao). High entropy flags cells/spots with ambiguous or mixed
identity. It also provides ``create_spatial_entropy_plot`` to map the entropy
onto the tissue coordinates.

``Gini`` is the Gini impurity ``1 - sum(p_i^2)``, which is exactly the Tsallis
entropy of order ``q = 2``: ``S_q = (1 - sum(p_i^q)) / (q - 1)`` reduces to
``1 - sum(p_i^2)`` for ``q = 2``. Unlike Shannon it is not a logarithmic
(Boltzmann-Gibbs) entropy, but it belongs to the same generalized-entropy
family and behaves as a diversity index over the same probability vector, so it
is treated here like the other metrics.

IMPORTANT -- the two metrics use different conventions, on purpose:

* ``Shannon_entropy`` is NORMALIZED, i.e. divided by its maximum ``log2(n)``,
  so it lies in ``[0, 1]``. Raw Shannon spans ``[0, log2(n)]`` and would exceed
  1 as soon as more than two cell types contribute, which makes it hard to
  compare across cells with a different number of candidates.
* ``Gini_entropy`` is RAW, i.e. plain ``1 - sum(p_i^2)``, NOT divided by its
  maximum ``1 - 1/n``. It needs no normalization to stay comparable: it is
  bounded by 1 for any ``n`` and reads directly as the probability that two
  independent draws from the distribution fall on different cell types.

The consequence of that choice is deliberate. Normalizing keeps only the
evenness of a distribution and divides away its richness: a 50/50 split over
two cell types and a 20/20/20/20/20 split over five both give a normalized
score of 1.0. Raw Gini keeps the distinction (0.50 vs 0.80), so ``Gini_entropy``
answers "how mixed is this cell overall" while ``Shannon_entropy`` answers "how
evenly is it split among the candidates it has".

Each metric is emitted on one scale only, so a column name always means the same
thing.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from itertools import combinations
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Accent color used across the package's other plots (lollipops, scatter-pie,
# ontology tree) -- reused here so the entropy plot reads as part of the same
# family instead of falling back to plain matplotlib defaults.
_ACCENT_COLOR = "#8B1A1A"

# Per-entropy-type default colormaps for create_spatial_entropy_plot, matching
# the ones already used for the analogous cluster-level entropy rows in
# plot_top_celltypes_per_cluster (see plots.py's default_entropy_specs), so a
# given entropy metric gets a consistent color family everywhere it's plotted.
# Only used when the caller doesn't pass an explicit `colormap`.
_DEFAULT_ENTROPY_CMAPS = {
    "Shannon": "Greens",
    "Gini": "Oranges",
    "Ontology-sp-Rao": "Reds",
    "Ontology-Lin-Rao": "Purples",
    "Jaccard-Rao": "Blues",
    "Overlap-Rao": "YlOrBr",
}


from .utils import (                             # noqa: F401
    DEFAULT_N_JOBS,
    resolve_n_jobs as _resolve_n_jobs,
    as_text_series as _as_text_series,
)



from . import entropy_core as _ec
from .entropy_core import (                      # noqa: F401
    clean_probabilities as _clean_probabilities,
    normalized_shannon as _normalized_shannon,
    raw_gini as _raw_gini,
    rao as _rao,
    marker_sets as _marker_sets,
    distance_matrix as _distance_matrix,
)


def _chunked_values(values, chunk_size=500):
    values = list(values)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def calculate_spot_entropies_parallel(
        final_dt_cell_deconvoluted,
        entropy_types=["Shannon"],
        ontology_graph=None,
        dt_all_marker=None,
        name="annotation_per_cell",
        annot_name=None,
        min_percentage_celltype_entropy=0.05,
        n_jobs=None,
        cell_chunk_size=500,
        progress_callback=None,
        progress_every=1,
        progress_label="cell-level entropy",
        unit_label="cells"):
    """Parallel entropy wrapper for cell-level or cluster-level entropy.

    The input table is split by the generic `cell` entity column. For normal
    annotation this means cells; for cluster-level entropy callers pass cluster
    ids in the same column. Each worker receives only a slice of rows for one
    chunk and returns a small entropy dataframe. Results are concatenated per
    entropy type.
    """
    if final_dt_cell_deconvoluted is None or getattr(final_dt_cell_deconvoluted, "empty", True):
        return {}

    if isinstance(entropy_types, str):
        entropy_types = [entropy_types]
    entropy_types = list(entropy_types)

    def _progress(entropy_type, done, total, unit=unit_label, phase="processing"):
        if progress_callback is not None:
            try:
                progress_callback(entropy_type=entropy_type, done=done, total=total, unit=unit, phase=phase, label=progress_label)
            except TypeError:
                progress_callback(f"{progress_label}: {entropy_type} {phase} {done}/{total} {unit}")
            except Exception:
                pass

    if "cell" not in final_dt_cell_deconvoluted.columns:
        return calculate_spot_entropies(
            final_dt_cell_deconvoluted,
            entropy_types=entropy_types,
            ontology_graph=ontology_graph,
            dt_all_marker=dt_all_marker,
            name=name,
            annot_name=annot_name,
            min_percentage_celltype_entropy=min_percentage_celltype_entropy,
            progress_callback=progress_callback,
            progress_every=progress_every,
            progress_label=progress_label,
        )

    n_jobs = _resolve_n_jobs(n_jobs)
    progress_every = max(1, int(progress_every))
    entity_series = final_dt_cell_deconvoluted["cell"].astype(str)
    entities = list(pd.Index(entity_series.dropna().unique()))
    chunks = list(_chunked_values(entities, cell_chunk_size))

    if not chunks:
        return {}

    merged = {}

    for entropy_type in entropy_types:
        _progress(entropy_type, 0, len(entities), unit=unit_label, phase="starting")

        def _work(chunk):
            mask = entity_series.isin(chunk)
            sub = final_dt_cell_deconvoluted.loc[mask]
            return calculate_spot_entropies(
                sub,
                entropy_types=[entropy_type],
                ontology_graph=ontology_graph,
                dt_all_marker=dt_all_marker,
                name=name,
                annot_name=annot_name,
                min_percentage_celltype_entropy=min_percentage_celltype_entropy,
                progress_callback=None,  # progress is tracked per completed chunk here
                progress_every=progress_every,
                progress_label=progress_label,
            )

        parts = []
        done_entities = 0

        if n_jobs == 1 or len(chunks) == 1:
            for chunk in chunks:
                parts.append(_work(chunk))
                done_entities += len(chunk)
                if done_entities == len(entities) or done_entities % progress_every == 0 or len(chunk) >= progress_every:
                    _progress(entropy_type, min(done_entities, len(entities)), len(entities), unit=unit_label)
        else:
            # Keep only a small number of futures in flight. This gives steadier
            # progress updates and avoids the impression that all logs are emitted
            # at once at the end of the entropy step. Progress is printed only from
            # the main thread, never from workers, so it is flushed immediately.
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                chunk_iter = iter(chunks)
                future_to_size = {}
                submitted_chunks = 0
                completed_chunks = 0

                def _submit_next():
                    nonlocal submitted_chunks
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        return False
                    submitted_chunks += 1
                    future_to_size[executor.submit(_work, chunk)] = len(chunk)
                    return True

                for _ in range(min(n_jobs, len(chunks))):
                    _submit_next()

                while future_to_size:
                    for future in as_completed(list(future_to_size), timeout=None):
                        chunk_size_done = future_to_size.pop(future)
                        parts.append(future.result())
                        done_entities += chunk_size_done
                        completed_chunks += 1
                        _progress(
                            entropy_type,
                            min(done_entities, len(entities)),
                            len(entities),
                            unit=unit_label,
                            phase=f"completed chunk {completed_chunks}/{len(chunks)}; processed",
                        )
                        _submit_next()
                        break

        frames = [part[entropy_type] for part in parts if isinstance(part, dict) and entropy_type in part and part[entropy_type] is not None]
        frames = [df for df in frames if hasattr(df, "empty") and not df.empty]
        if frames:
            merged[entropy_type] = pd.concat(frames, ignore_index=True)
        _progress(entropy_type, len(entities), len(entities), unit=unit_label, phase="done")

    return merged


def calculate_spot_entropies(final_dt_cell_deconvoluted, entropy_types=["Shannon"],
                             ontology_graph=None, dt_all_marker=None,
                             name="annotation_per_cell", annot_name=None, min_percentage_celltype_entropy=0.05,
                             progress_callback=None, progress_every=1000, progress_label="cell-level entropy"):

    def _progress(entropy_type, done, total, unit="cells", phase="processing"):
        if progress_callback is not None:
            try:
                progress_callback(entropy_type=entropy_type, done=done, total=total, unit=unit, phase=phase, label=progress_label)
            except TypeError:
                progress_callback(f"{progress_label}: {entropy_type} {phase} {done}/{total} {unit}")
            except Exception:
                pass

    progress_every = max(1, int(progress_every))

    entropy_data = final_dt_cell_deconvoluted.copy()

    if "significance_percentage" not in entropy_data.columns:
        if "percentage" in entropy_data.columns:
            entropy_data["significance_percentage"] = entropy_data["percentage"]
        else:
            entropy_data["significance_percentage"] = 1.0

    if name not in entropy_data.columns:
        if "annotation_per_cell" in entropy_data.columns:
            name = "annotation_per_cell"
        elif "accordion_per_cell" in entropy_data.columns:
            name = "accordion_per_cell"

    entropy_data = entropy_data[
        ((entropy_data["significance_percentage"] >= min_percentage_celltype_entropy) | ((entropy_data["significance_percentage"] <= 1.0) & ((entropy_data["significance_percentage"] * 100.0) >= min_percentage_celltype_entropy))) &
        (~entropy_data[name].str.contains("unknown", case=False, na=False))
        ]

    # Add celltype_impact_score if not present (use diff_score as proxy)
    if 'celltype_impact_score' not in entropy_data.columns:
        entropy_data['celltype_impact_score'] = entropy_data['diff_score']

    entropy_results = {}

    _cells = entropy_data['cell'].unique()
    _celltypes_present = sorted(set(entropy_data[name]))

    def _cell_probabilities(cell):
        """``(celltypes, p)`` for one cell.

        ``p`` is ``significance_percentage * celltype_impact_score`` renormalized
        to sum 1, so Shannon, Gini and the four Rao metrics all see exactly the
        same distribution.
        """
        cell_data = entropy_data[entropy_data['cell'] == cell].copy()
        cell_data["p_weighted"] = cell_data["significance_percentage"] * cell_data["celltype_impact_score"]
        cell_data["p_weighted"] /= cell_data["p_weighted"].sum()
        cts = cell_data[name].unique()
        return cts, cell_data.set_index(name)["p_weighted"].reindex(cts).values

    def _run_metric(metric, value_of):
        """Loop the cells, report progress, store the result table."""
        _progress(metric, 0, len(_cells), unit="cells", phase="starting")
        rows = []
        for _i, cell in enumerate(_cells, start=1):
            cts, p_vec = _cell_probabilities(cell)
            rows.append({'cell': cell, f"{metric}_entropy": value_of(cts, p_vec)})
            if _i == len(_cells) or _i % progress_every == 0:
                _progress(metric, _i, len(_cells), unit="cells")
        entropy_results[metric] = pd.DataFrame(rows)

    def _matrix_progress(metric):
        """Throttle the distance-matrix progress callback to progress_every."""
        def _cb(done, total):
            if done == total or done % progress_every == 0:
                _progress(metric, done, total, unit="celltypes",
                          phase="building distance matrix")
        return _cb

    if "Shannon" in entropy_types:
        _run_metric("Shannon", lambda cts, p: _normalized_shannon(p))
    if "Gini" in entropy_types:
        _run_metric("Gini", lambda cts, p: _raw_gini(p))

    _ont_requested = [m for m in _ec.ONTOLOGY_METRICS if m in entropy_types]
    if _ont_requested and ontology_graph is not None:
        _unmapped = _ec.unmapped_celltypes(_celltypes_present, ontology_graph)
        if _unmapped:
            print(_ec.skip_message(_ont_requested, _unmapped,
                                   "have no Cell Ontology term"))
            _ont_requested = []
        for _ont_metric in _ont_requested:
            _ont_dist = pd.DataFrame(
                _ec.ontology_distance_matrix(
                    _celltypes_present, ontology_graph, _ont_metric,
                    missing="nan", progress=_matrix_progress(_ont_metric)),
                index=_celltypes_present, columns=_celltypes_present)
            _run_metric(_ont_metric,
                        lambda cts, p, _D=_ont_dist: _ec.rao(p, _D.loc[cts, cts].values))

    _mk_requested = [m for m in _ec.MARKER_METRICS if m in entropy_types]
    if _mk_requested and dt_all_marker is not None:
        _sets = _ec.marker_sets(dt_all_marker, annot_name)
        _no_markers = _ec.celltypes_without_markers(_celltypes_present, _sets)
        if _no_markers:
            print(_ec.skip_message(_mk_requested, _no_markers,
                                   "have no markers left in the marker table"))
            _mk_requested = []
        for _mk_metric in _mk_requested:
            _run_metric(_mk_metric, lambda cts, p, _k=_mk_metric: _ec.rao(
                p, _ec.marker_distance_matrix(cts, _sets, _k, missing="nan")))

    return entropy_results



# ---- Vectorized entropy helpers -------------------------------------------
# These operate on precomputed celltype x celltype distance/similarity
# matrices and feed calculate_entropies_from_score_matrix() below, which is
# used for cluster/spatial score-matrix entropy (see accordion.py). This is a
# separate code path from the per-spot calculate_spot_entropies() further up
# this file, which is used for cell-level entropy (see annotation_helpers.py)
# and operates directly on per-cell deconvolution results rather than a matrix.
def _prepare_marker_distance_matrix(celltypes, dt_all_marker, annot_name, kind="Overlap-Rao"):
    """Dense celltype x celltype distance matrix for the marker-set Rao entropies.

    Callers check ``celltypes_without_markers`` first and drop the metric when it
    is non-empty, so by the time this runs ``missing`` cannot fire.
    """
    sets = _marker_sets(dt_all_marker, annot_name)
    if not sets:
        return np.zeros((len(celltypes), len(celltypes)), dtype=np.float32)
    D = _distance_matrix(celltypes, kind, sets=sets, missing="max")
    return D.astype(np.float32)


def _prepare_ontology_distance_matrix(celltypes, ontology_graph, kind="Ontology-sp-Rao",
                                      progress_callback=None, progress_label="entropy"):
    """Dense celltype x celltype distance matrix for the ontology Rao entropies.

    Callers check ``unmapped_celltypes`` first, so ``missing`` can only still
    apply to two mapped cell types with no connecting path -- which does not
    happen on Cell Ontology, a single connected graph once undirected.
    """
    def _progress(done, total):
        if progress_callback is not None:
            try:
                progress_callback(entropy_type=kind, done=done, total=total,
                                  unit="celltypes", phase="building distance matrix",
                                  label=progress_label)
            except TypeError:
                progress_callback(f"{progress_label}: {kind} building distance matrix {done}/{total}")
            except Exception:
                pass

    D = _distance_matrix(celltypes, kind, ontology_graph=ontology_graph,
                         missing="zero", progress=_progress)
    return D.astype(np.float32)


def calculate_entropies_from_score_matrix(
        score_matrix,
        cells,
        celltypes,
        entropy_types=("Shannon",),
        ontology_graph=None,
        dt_all_marker=None,
        annot_name="annotation_per_cell",
        min_percentage_celltype_entropy=0.05,
        top_cell_score_quantile_threshold=0.90,
        n_top_celltypes=5,
        allow_unknown=True,
        n_jobs=None,
        cell_chunk_size=5000,
        progress_callback=None,
        progress_every=10,
        progress_label="cell-level entropy",
        unit_label="cells"):
    """Fast entropy computation directly from a dense score matrix.

    This version does the cell-level work per chunk with NumPy arrays instead of
    looping cell-by-cell in Python. Threads are still used across chunks, but the
    heavy work inside each chunk is vectorized, so NumPy/BLAS can actually do the
    expensive part efficiently.
    """
    if score_matrix is None:
        return {}

    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return {}

    cells = np.asarray(cells).astype(str)
    celltypes = np.asarray(celltypes).astype(str)
    if isinstance(entropy_types, str):
        entropy_types = [entropy_types]
    entropy_types = list(entropy_types)

    n_entities, n_ct = scores.shape
    n_jobs = _resolve_n_jobs(n_jobs)
    cell_chunk_size = max(1, int(cell_chunk_size))
    progress_every = max(1, int(progress_every))
    n_top_celltypes = max(1, int(n_top_celltypes))

    # Metrics whose distances cannot be defined are dropped, not guessed at --
    # see entropy_core.
    _ont_requested = [m for m in _ec.ONTOLOGY_METRICS if m in entropy_types]
    if _ont_requested:
        _unmapped = _ec.unmapped_celltypes(celltypes, ontology_graph)
        if _unmapped:
            print(_ec.skip_message(_ont_requested, _unmapped,
                                   "have no Cell Ontology term"))
            entropy_types = [m for m in entropy_types if m not in _ont_requested]

    _mk_requested = [m for m in _ec.MARKER_METRICS if m in entropy_types]
    if _mk_requested:
        _no_markers = _ec.celltypes_without_markers(
            celltypes, _marker_sets(dt_all_marker, annot_name))
        if _no_markers:
            print(_ec.skip_message(_mk_requested, _no_markers,
                                   "have no markers left in the marker table"))
            entropy_types = [m for m in entropy_types if m not in _mk_requested]

    # Precompute global distance matrices once per metric.
    distance_mats = {}
    if "Jaccard-Rao" in entropy_types:
        distance_mats["Jaccard-Rao"] = _prepare_marker_distance_matrix(
            celltypes, dt_all_marker, annot_name, "Jaccard-Rao"
        )
    if "Overlap-Rao" in entropy_types:
        distance_mats["Overlap-Rao"] = _prepare_marker_distance_matrix(
            celltypes, dt_all_marker, annot_name, "Overlap-Rao"
        )
    if "Ontology-sp-Rao" in entropy_types:
        distance_mats["Ontology-sp-Rao"] = _prepare_ontology_distance_matrix(
            celltypes, ontology_graph, "Ontology-sp-Rao"
        )
    if "Ontology-Lin-Rao" in entropy_types:
        distance_mats["Ontology-Lin-Rao"] = _prepare_ontology_distance_matrix(
            celltypes, ontology_graph, "Ontology-Lin-Rao"
        )

    def _progress(entropy_type, done, total, phase="processing"):
        if progress_callback is not None:
            try:
                progress_callback(
                    entropy_type=entropy_type,
                    done=done,
                    total=total,
                    unit=unit_label,
                    phase=phase,
                    label=progress_label,
                )
            except TypeError:
                progress_callback(f"{progress_label}: {entropy_type} {phase} {done}/{total} {unit_label}")
            except Exception:
                pass

    def _probability_chunk(start, stop):
        """Return P and selected counts for rows [start:stop].

        P has the same number of columns as score_matrix. Non-selected celltypes
        are zero. Selected probabilities follow the previous fast-path logic:
        keep positive/final finite scores, optionally reduce to quantile + top-N,
        then p is proportional to score^2.
        """
        chunk = scores[start:stop].astype(np.float32, copy=True)
        finite = np.isfinite(chunk)
        if allow_unknown:
            base_mask = finite & (chunk > 0)
        else:
            base_mask = finite
            chunk = np.clip(chunk, 0, None)

        counts = base_mask.sum(axis=1).astype(np.int32)
        selected = base_mask.copy()

        reduce_rows = counts > n_top_celltypes
        if np.any(reduce_rows):
            # Only the rows that actually need reducing. Restricting the work to
            # them is not just an optimization. A cell with no candidate at all --
            # every cell type scoring <= 0, which is how it ends up annotated
            # "unknown" -- has base_mask all False, so np.where(base_mask, ...)
            # makes its whole row NaN. np.nanquantile then warns "All-NaN slice
            # encountered" and returns NaN for it. The NaN was harmless, because
            # such a row has counts == 0 and is never in reduce_rows, so the
            # result was discarded a few lines later -- but the warning was real
            # and the work wasted. np.errstate did not suppress it: errstate
            # governs floating-point error flags, while that message comes through
            # warnings.warn as a RuntimeWarning.
            idx = np.flatnonzero(reduce_rows)
            sub_mask = base_mask[idx]
            sub_chunk = chunk[idx]

            # Quantile over valid values only, matching the old per-row helper.
            q = np.nanquantile(
                np.where(sub_mask, sub_chunk, np.nan),
                top_cell_score_quantile_threshold, axis=1)
            keep_q = sub_mask & (sub_chunk >= q[:, None])

            # Top-N mask. argpartition is vectorized across the rows.
            top_n = min(n_top_celltypes, n_ct)
            kth = n_ct - top_n
            top_idx = np.argpartition(
                np.where(sub_mask, sub_chunk, -np.inf), kth=kth, axis=1)[:, kth:]
            keep_top = np.zeros_like(sub_mask, dtype=bool)
            keep_top[np.arange(idx.size)[:, None], top_idx] = True
            keep_top &= sub_mask

            selected[idx] = keep_q | keep_top

        weights = np.where(selected, np.clip(chunk, 0, None), 0.0).astype(np.float32, copy=False)
        # Old fast-path semantics: significance = score/sum(score), then
        # p_weighted = significance * score, normalized again => proportional
        # to score^2.
        weights *= weights
        denom = weights.sum(axis=1, keepdims=True)
        valid = (denom[:, 0] > 0) & np.isfinite(denom[:, 0])
        P = np.zeros_like(weights, dtype=np.float32)
        if np.any(valid):
            P[valid] = weights[valid] / denom[valid]
        selected_counts = (P > 0).sum(axis=1).astype(np.int32)
        return P, selected_counts

    def _compute_shannon_chunk(start, stop):
        P, selected_counts = _probability_chunk(start, stop)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.sum(np.where(P > 0, P * np.log2(P), 0.0), axis=1)
        valid = selected_counts > 1
        norm = np.zeros_like(H, dtype=np.float32)
        norm[valid] = H[valid] / np.log2(selected_counts[valid]).astype(np.float32)
        norm = np.maximum(norm, 0.0).astype(np.float32)
        return pd.DataFrame({"cell": cells[start:stop], "Shannon_entropy": norm})

    def _compute_gini_chunk(start, stop):
        # Gini impurity 1 - sum(p^2) == Tsallis entropy with q = 2, normalized
        # by its maximum (1 - 1/n) exactly as Shannon is normalized by log2(n).
        P, selected_counts = _probability_chunk(start, stop)
        G = 1.0 - np.sum(P * P, axis=1)
        valid = selected_counts > 1
        # Raw Gini impurity: range [0, 1 - 1/n], always below 1 whatever n, so
        # it is emitted as-is rather than divided by its maximum. Cells with a
        # single contributing cell type are zero by definition.
        raw = np.where(valid, G, 0.0)
        raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        return pd.DataFrame({"cell": cells[start:stop], "Gini_entropy": raw})

    def _compute_rao_chunk(start, stop, entropy_type):
        P, _ = _probability_chunk(start, stop)
        dist = distance_mats.get(entropy_type)
        if dist is None:
            vals = np.zeros(stop - start, dtype=np.float32)
        else:
            D = np.asarray(dist, dtype=np.float32)
            # Rao: diag(P D P^T), vectorized as row-wise sum((P @ D) * P).
            vals = np.sum((P @ D) * P, axis=1).astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            vals = np.maximum(vals, 0.0).astype(np.float32)
        return pd.DataFrame({"cell": cells[start:stop], f"{entropy_type}_entropy": vals})

    ranges = [(i, min(i + cell_chunk_size, n_entities)) for i in range(0, n_entities, cell_chunk_size)]

    def _run_metric(entropy_type):
        _progress(entropy_type, 0, n_entities, phase="starting vectorized matrix path")
        parts = []
        done = 0
        if entropy_type == "Shannon":
            compute = lambda s, e: _compute_shannon_chunk(s, e)
        elif entropy_type == "Gini":
            compute = lambda s, e: _compute_gini_chunk(s, e)
        else:
            compute = lambda s, e: _compute_rao_chunk(s, e, entropy_type)

        if n_jobs == 1 or len(ranges) == 1:
            for k, (s, e) in enumerate(ranges, start=1):
                parts.append(compute(s, e))
                done += (e - s)
                _progress(entropy_type, min(done, n_entities), n_entities,
                          phase=f"completed chunk {k}/{len(ranges)}; processed")
        else:
            # Threading now helps more because each chunk mostly runs NumPy
            # operations that release the GIL. Avoid ProcessPool here because it
            # would copy score_matrix and distance matrices into worker processes.
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_range = {
                    executor.submit(compute, s, e): (s, e, k)
                    for k, (s, e) in enumerate(ranges, start=1)
                }
                completed_chunks = 0
                for future in as_completed(future_to_range):
                    s, e, _k = future_to_range[future]
                    parts.append(future.result())
                    completed_chunks += 1
                    done += (e - s)
                    _progress(entropy_type, min(done, n_entities), n_entities,
                              phase=f"completed chunk {completed_chunks}/{len(ranges)}; processed")

        _progress(entropy_type, n_entities, n_entities, phase="done")
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cell", f"{entropy_type}_entropy"])

    results = {}
    for entropy_type in entropy_types:
        results[entropy_type] = _run_metric(entropy_type)
    return results

def create_spatial_entropy_plot(adata, final_dt_cell_deconvoluted, entropy_results,
                                entropy_type="Shannon", figsize=(12, 5), colormap=None,
                                spot_size=30, title=None, xlabel="Spatial X", ylabel="Spatial Y",
                                colorbar_label=None, hist_title="Distribution",
                                hist_xlabel=None, hist_ylabel="Frequency",
                                show=True, save=None, dpi=300):
    """
    Create spatial plot with spots colored according to colormap based on entropy values.

    Parameters:
    -----------
    adata : AnnData object
        Spatial single cell data with obsm["spatial"] coordinates
    final_dt_cell_deconvoluted : pd.DataFrame
        Deconvolution results
    entropy_results : dict
        Dictionary with entropy DataFrames from calculate_spot_entropies
    entropy_type : str
        Which entropy type to use for coloring, e.g. ``"Shannon"``, ``"Gini"``
        or one of the Rao variants.
    figsize : tuple
        Figure size
    colormap : str, optional
        Matplotlib colormap name (e.g., "Blues", "Greens", "Purples", "viridis", etc.).
        ``None`` (default) picks the metric-specific default: "Greens" for
        "Shannon", "Oranges" for "Gini", "Reds" for "Ontology-sp-Rao",
        "Purples" for "Ontology-Lin-Rao", "Blues" for "Jaccard-Rao", "YlOrBr"
        for "Overlap-Rao" (falls back to "Blues" for any other/custom
        ``entropy_type``). Pass an explicit colormap name to override.
    spot_size : float
        Marker size (matplotlib ``scatter`` ``s``, points^2) for each spot. Increase
        for sparse/low-density tissues, decrease for dense ones with lots of
        overlap. Default ``30``.
    title, xlabel, ylabel, colorbar_label
        Text overrides for the spatial (left) panel: its title (default
        ``f"{entropy_type} entropy"``), axis labels (default ``"Spatial X"`` /
        ``"Spatial Y"``), and colorbar label (default same as ``title``).
    hist_title, hist_xlabel, hist_ylabel
        Text overrides for the distribution (right) panel: its title (default
        ``"Distribution"``), x-axis label (default same as ``title``) and
        y-axis label (default ``"Frequency"``).
    show, save, dpi
        Standard display / export options, consistent with the other
        ``plot_*`` functions: ``save="path.png"`` writes the figure to disk,
        ``show=False`` skips ``plt.show()`` (handy when only saving), and
        ``dpi`` controls the resolution used when saving.
    """
    colormap = colormap if colormap is not None else _DEFAULT_ENTROPY_CMAPS.get(entropy_type, "Blues")

    # Get spatial coordinates
    coords_df = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"])

    # Get top cell type for each spot (for potential future use/info)
    top_celltypes = final_dt_cell_deconvoluted.loc[
        final_dt_cell_deconvoluted.groupby('cell', observed=True)['diff_score'].idxmax()
    ][['cell', 'annotation_per_cell']].set_index('cell')

    # Get entropy values
    entropy_col = f"{entropy_type}_entropy"
    if entropy_type in entropy_results and entropy_col in getattr(entropy_results[entropy_type], "columns", []):
        entropy_df = entropy_results[entropy_type].set_index('cell')[entropy_col]
    else:
        _available = ", ".join(sorted(entropy_results)) if entropy_results else "none"
        print(f"Warning: {entropy_type} not found in entropy_results (available: {_available}). "
              f"Using default entropy=0.5")
        # The series must carry entropy_col as its name, otherwise the join
        # below fails with "Other Series must have a name".
        entropy_df = pd.Series(0.5, index=coords_df.index, name=entropy_col)

    # Merge data
    plot_data = coords_df.join(entropy_df, how='left')
    plot_data[entropy_col] = plot_data[entropy_col].fillna(0.0)  # Default low entropy for missing values

    # Set entropy range for normalization
    entropy_min = 0
    entropy_max = 1

    # Same seaborn theme (style + "talk" context) as every other plot in the
    # package, so typography/weight/sizing match instead of falling back to
    # plain matplotlib defaults, which is what made this plot feel visually
    # detached from the rest.
    sns.set_theme(style="white", context="talk")

    # Create the plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # Main spatial plot
    ax1 = fig.add_subplot(gs[0])

    # Plot spots with colors from the colormap based on entropy values
    scatter = ax1.scatter(plot_data['x'], plot_data['y'],
                          c=plot_data[entropy_col],
                          cmap=colormap,
                          vmin=entropy_min,
                          vmax=entropy_max,
                          s=spot_size,
                          edgecolors='none')

    resolved_title = title if title is not None else f'{entropy_type} entropy'
    resolved_cbar_label = colorbar_label if colorbar_label is not None else resolved_title
    resolved_hist_xlabel = hist_xlabel if hist_xlabel is not None else resolved_title

    ax1.set_xlabel(xlabel, fontsize=14, weight="bold")
    ax1.set_ylabel(ylabel, fontsize=14, weight="bold")
    ax1.set_title(resolved_title, fontsize=16, weight="bold", pad=12)
    ax1.set_aspect('equal')
    # obsm["spatial"] uses image coordinates (y increases downward); invert the
    # y-axis so the tissue is shown in the same orientation as scanpy's
    # sc.pl.spatial (otherwise the map is vertically flipped).
    if not ax1.yaxis_inverted():
        ax1.invert_yaxis()

    # Add colorbar for entropy values
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label(resolved_cbar_label, fontsize=12, weight="bold")

    # Entropy distribution plot
    ax2 = fig.add_subplot(gs[1])

    # Get a representative color from the colormap for the histogram
    cmap = plt.cm.get_cmap(colormap)
    hist_color = cmap(0.7)  # Use a color from the middle-high range of the colormap

    ax2.hist(plot_data[entropy_col], bins=30, alpha=0.85, edgecolor='white',
             linewidth=0.6, color=hist_color)
    ax2.set_xlabel(resolved_hist_xlabel, fontsize=13, weight="bold")
    ax2.set_ylabel(hist_ylabel, fontsize=13, weight="bold")
    ax2.set_title(hist_title, fontsize=14, weight="bold", pad=10)
    ax2.axvline(plot_data[entropy_col].mean(), color=_ACCENT_COLOR, linestyle='--', linewidth=1.6,
                label=f'Mean: {plot_data[entropy_col].mean():.3f}')
    ax2.legend(fontsize=10, frameon=False)
    ax2.grid(axis='y', alpha=0.25)
    for spine in ('top', 'right'):
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    if save is not None:
        fig.savefig(Path(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, plot_data
