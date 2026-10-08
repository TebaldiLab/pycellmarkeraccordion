"""Shared primitives for every annotation-entropy metric.

The three callers -- cluster-level entropy in ``annotation_helpers`` and the
per-cell and vectorized paths in ``spatial_entropy`` -- only prepare the
probability vector and ask for a distance matrix.

Unresolvable cell types: refuse the metric, do not guess
--------------------------------------------------------
A Rao metric needs a distance for every pair of candidate cell types, and a cell
type with no Cell Ontology term or no row in the marker table has none. The
policy is the one already applied to disease conditions, where no cell ontology
exists at all: **if any candidate cell type in the analysis cannot be placed, the
affected metric is not computed for that analysis.** ``unmapped_celltypes`` and
``celltypes_without_markers`` run the check up front, ``skip_message`` phrases
the refusal. The other metrics are unaffected: Shannon and Gini need no
distances, the marker metrics need no ontology.

``missing`` covers what is left after that check: two cell types that both exist
but have no connecting path (``Ontology-sp-Rao``). Cell Ontology is a single
connected graph once undirected, so in practice it does not fire. It stays a
parameter because the three paths pass different values and aligning them would
move numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "METRICS", "DISTANCE_METRICS", "ONTOLOGY_METRICS", "MARKER_METRICS",
    "clean_probabilities", "normalized_shannon", "raw_gini", "rao",
    "marker_sets", "marker_distance_matrix", "ontology_distance_matrix",
    "distance_matrix",
    "unmapped_celltypes", "celltypes_without_markers", "skip_message",
]

#: Every metric the package can compute, in display order.
METRICS = ("Shannon", "Gini", "Ontology-sp-Rao", "Ontology-Lin-Rao",
           "Jaccard-Rao", "Overlap-Rao")

#: The subset that needs a cell-type x cell-type distance matrix.
DISTANCE_METRICS = ("Ontology-sp-Rao", "Ontology-Lin-Rao",
                    "Jaccard-Rao", "Overlap-Rao")

#: Need every candidate cell type to have a Cell Ontology term.
ONTOLOGY_METRICS = ("Ontology-sp-Rao", "Ontology-Lin-Rao")

#: Need every candidate cell type to have at least one marker.
MARKER_METRICS = ("Jaccard-Rao", "Overlap-Rao")

_MISSING = {"nan": np.nan, "zero": 0.0, "max": 1.0}


# --- preconditions: can these metrics be computed at all? ---

def unmapped_celltypes(celltypes, ontology_graph):
    """Candidate cell types whose label has no term in the ontology.

    A non-empty result means the ontology metrics must be skipped for this
    analysis: there is no distance to place these labels at.
    """
    if ontology_graph is None:
        return sorted({str(c) for c in celltypes})
    labels = {data.get("name", "") for _node, data in ontology_graph.nodes(data=True)}
    return sorted({str(c) for c in celltypes if str(c) not in labels})


def celltypes_without_markers(celltypes, sets):
    """Candidate cell types with no marker set, so no marker distance either.

    In practice this is empty: a cell type reaches the entropy table by being
    scored from its markers. It is a guard against a degenerate marker table,
    not an expected condition.
    """
    return sorted({str(c) for c in celltypes if not sets.get(str(c))})


def skip_message(metrics, offenders, reason, limit=10):
    """One line explaining which metrics are skipped and why."""
    shown = ", ".join(offenders[:limit])
    if len(offenders) > limit:
        shown += f", ... (+{len(offenders) - limit} more)"
    return (f"WARNING: not computing {', '.join(metrics)} for this analysis -- "
            f"{len(offenders)} cell type(s) {reason}: {shown}. "
            "The remaining entropy metrics are unaffected.")


# --- probability vector ---

def clean_probabilities(probabilities):
    """Drop non-positive/non-finite entries and renormalize to sum 1.

    Returns ``(p, n)`` with ``n`` the number of contributing components, or
    ``(None, n)`` when the vector carries no usable signal.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    p = p[np.isfinite(p) & (p > 0)]
    n = p.size
    if n <= 1:
        return None, n
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return None, n
    return p / total, n


def normalized_shannon(probabilities):
    """``-sum(p log2 p)`` divided by its maximum ``log2(n)``, so it lies in
    ``[0, 1]``.

    Shannon is reported normalized because raw it ranges over ``[0, log2(n)]``
    and so grows with the number of candidate cell types: a cluster with 5
    equally likely candidates would score above one with 2, for reasons that
    have nothing to do with how ambiguous the annotation is. Dividing by
    ``log2(n)`` makes clusters with different numbers of candidates comparable.

    Zero probabilities are excluded rather than nudged by an epsilon, so the
    result is exact.
    """
    p, n = clean_probabilities(probabilities)
    if p is None:
        return 0.0
    return max(0.0, float(-np.sum(p * np.log2(p))) / np.log2(n))


def raw_gini(probabilities):
    """Gini impurity ``1 - sum(p^2)``, i.e. Tsallis entropy of order ``q = 2``.

    Gini is reported raw, the opposite convention to Shannon above, because its
    range ``[0, 1 - 1/n]`` is already bounded by 1 for any ``n``. Dividing by
    ``1 - 1/n`` would buy comparability that is already there and would inflate
    small clusters: with 2 candidates the maximum is 0.5, so normalizing doubles
    every value.
    """
    p, _n = clean_probabilities(probabilities)
    if p is None:
        return 0.0
    return max(0.0, 1.0 - float(np.sum(p ** 2)))


def rao(p, D):
    """Rao quadratic entropy ``Q = sum_i sum_j p_i p_j d_ij`` = ``p' D p``.

    With ``d_ii = 0`` and ``d_ij`` in ``[0, 1]`` this is always bounded above
    by the raw Gini impurity ``1 - sum(p^2)``, with equality only when every
    off-diagonal distance is 1 -- a useful invariant to assert in tests.
    """
    p = np.asarray(p, dtype=float).ravel()
    D = np.asarray(D, dtype=float)
    if p.size == 0 or D.size == 0 or D.shape[0] != p.size:
        return 0.0
    return float(p @ D @ p)


# --- distance matrices ---

def marker_sets(dt_all_marker, annot_name, require_positive_impact=True):
    """``{cell type: set of signed markers}`` from a marker table.

    Markers carry their sign (``GENE+`` / ``GENE-``) so that a gene marking one
    cell type positively and another negatively counts as a difference, not as
    shared evidence.
    """
    if dt_all_marker is None or getattr(dt_all_marker, "empty", True):
        return {}
    df = dt_all_marker
    if annot_name not in df.columns and "annotation_per_cell" in df.columns:
        annot_name = "annotation_per_cell"
    if annot_name not in df.columns:
        return {}
    if require_positive_impact and "gene_impact_score_per_cluster" in df.columns:
        df = df[df["gene_impact_score_per_cluster"] > 0]
    df = df.copy()
    sign = df["marker_type"].map({"positive": "+", "negative": "-"})
    df["_marker_sign"] = (df["marker"].astype("string").fillna("")
                          + sign.astype("string").fillna(""))
    return {str(k): v for k, v in
            df.groupby([annot_name], observed=True)["_marker_sign"].apply(set).to_dict().items()}


def marker_distance_matrix(celltypes, sets, kind="Overlap-Rao", missing="max"):
    """Jaccard or Overlap (Szymkiewicz-Simpson) distance between marker sets.

    ``Jaccard  = 1 - |A n B| / |A u B|``
    ``Overlap  = 1 - |A n B| / min(|A|, |B|)``

    Both denominators are guarded: an empty set on either side gives distance 1.
    """
    celltypes = [str(c) for c in celltypes]
    n = len(celltypes)
    D = np.zeros((n, n), dtype=float)
    fill = _MISSING[missing]
    for i in range(n):
        a = sets.get(celltypes[i])
        for j in range(i + 1, n):
            b = sets.get(celltypes[j])
            if a is None or b is None:
                d = fill
            else:
                inter = len(a & b)
                denom = len(a | b) if kind == "Jaccard-Rao" else min(len(a), len(b))
                d = 1.0 - (inter / denom) if denom > 0 else 1.0
            D[i, j] = D[j, i] = d
    return D


def ontology_distance_matrix(celltypes, ontology_graph, kind="Ontology-sp-Rao",
                             missing="max", progress=None):
    """Shortest-path or Lin distance between cell types on an OBO ontology.

    ``Ontology-sp-Rao`` counts edges on the undirected graph and divides the
    whole matrix by its own maximum, so the values are relative to the cell
    types present in THIS run and are not comparable across datasets.
    ``Ontology-Lin-Rao`` is ``1 - lin_similarity`` and is on an absolute scale.

    In an obonet graph the ``is_a`` edges run child -> parent, so
    ``nx.ancestors`` returns a term's SUBCLASSES and ``nx.descendants`` its
    SUPERCLASSES -- the opposite of the ontology wording. Information content
    counts subclasses so that it falls with generality.
    """
    import networkx as nx

    celltypes = [str(c) for c in celltypes]
    n = len(celltypes)
    D = np.zeros((n, n), dtype=float)
    if ontology_graph is None or n == 0:
        return D
    fill = _MISSING[missing]
    label_to_id = {data.get("name", ""): node
                   for node, data in ontology_graph.nodes(data=True)}
    ids = [label_to_id.get(c) for c in celltypes]

    if kind == "Ontology-sp-Rao":
        G = ontology_graph.to_undirected()
        finite_max = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if ids[i] is None or ids[j] is None:
                    d = fill
                else:
                    try:
                        d = float(nx.shortest_path_length(G, ids[i], ids[j]))
                    except Exception:
                        d = fill
                if np.isfinite(d):
                    finite_max = max(finite_max, d)
                D[i, j] = D[j, i] = d
            if progress is not None:
                progress(i + 1, n)
        if finite_max > 0:
            D = D / finite_max
        return D

    # ---- Lin ----
    total_nodes = len(ontology_graph.nodes)
    IC = {}
    for node in ontology_graph.nodes:
        subclasses = nx.ancestors(ontology_graph, node)
        IC[node] = -np.log((len(subclasses) + 1) / total_nodes)

    def _lin(a, b):
        sup_a = nx.descendants(ontology_graph, a) | {a}
        sup_b = nx.descendants(ontology_graph, b) | {b}
        common = sup_a & sup_b
        if not common:
            return 0.0
        mica = max(IC.get(x, 0.0) for x in common)
        denom = IC.get(a, 0.0) + IC.get(b, 0.0)
        return 0.0 if denom == 0 else (2.0 * mica) / denom

    for i in range(n):
        for j in range(i + 1, n):
            if ids[i] is None or ids[j] is None:
                d = fill
            else:
                try:
                    d = 1.0 - _lin(ids[i], ids[j])
                except Exception:
                    d = fill
            D[i, j] = D[j, i] = d
        if progress is not None:
            progress(i + 1, n)
    return D


def distance_matrix(celltypes, kind, ontology_graph=None, sets=None,
                    missing="max", progress=None):
    """Dispatch to the right distance for ``kind``; zeros for unknown kinds."""
    if kind in ("Ontology-sp-Rao", "Ontology-Lin-Rao"):
        return ontology_distance_matrix(celltypes, ontology_graph, kind,
                                        missing=missing, progress=progress)
    if kind in ("Jaccard-Rao", "Overlap-Rao"):
        return marker_distance_matrix(celltypes, sets or {}, kind, missing=missing)
    return np.zeros((len(celltypes), len(celltypes)), dtype=float)
