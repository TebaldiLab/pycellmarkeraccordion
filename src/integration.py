"""Integrate a custom marker set into the Accordion database.

``marker_database_integration`` appends a user marker table to the packaged
healthy or disease Accordion database and recomputes the evidence-consistency
(``ECs_*``) and specificity (``SPs_*``) scores on the combined set. The
returned table can be passed to ``accordion(database=...)`` (or saved to
Parquet first) to annotate with the augmented database.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from .io import (
    get_default_marker_database_path,
    get_default_cell_ontology_path,
    get_default_uberon_ontology_path,
    get_package_data_path,
)


# ---------------------------------------------------------------------------
# small loaders
# ---------------------------------------------------------------------------

def _read_parquet(path) -> pd.DataFrame:
    return pl.read_parquet(str(path)).to_pandas()


@lru_cache(maxsize=2)
def _ontology_name_to_id(path: str) -> dict:
    """name(lowercased) -> ontology id, parsed from a .obo file."""
    mapping: dict = {}
    id_cur = None
    name_cur = None
    in_term = False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("[Term]"):
                if id_cur and name_cur:
                    mapping.setdefault(name_cur.strip().lower(), id_cur)
                id_cur, name_cur, in_term = None, None, True
                continue
            if line.startswith("["):
                if id_cur and name_cur:
                    mapping.setdefault(name_cur.strip().lower(), id_cur)
                id_cur, name_cur, in_term = None, None, False
                continue
            if not in_term:
                continue
            if line.startswith("id:"):
                id_cur = line[3:].strip()
            elif line.startswith("name:"):
                name_cur = line[5:].strip()
        if id_cur and name_cur:
            mapping.setdefault(name_cur.strip().lower(), id_cur)
    return mapping


@lru_cache(maxsize=1)
def _gene_description_map() -> dict:
    try:
        gd = _read_parquet(get_package_data_path("gene_description.parquet"))
    except Exception:
        return {}
    cols = {c.lower(): c for c in gd.columns}
    gcol = cols.get("gene_symbol") or cols.get("marker") or cols.get("gene")
    dcol = cols.get("gene_description") or cols.get("description")
    if gcol is None or dcol is None:
        return {}
    return dict(zip(gd[gcol].astype(str), gd[dcol].astype(str)))


def _map_lower(series: pd.Series, mapping: dict) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(mapping)


# ---------------------------------------------------------------------------
# score recomputation
# ---------------------------------------------------------------------------

def _ecs(df: pd.DataFrame, group_cols, out_name: str) -> pd.DataFrame:
    """Evidence consistency = number of distinct resources per group.

    Uses groupby-transform (index-aligned, NaN-group safe) to avoid the
    NaN != NaN problem that a merge on keys with missing ids would hit.
    """
    cols = [c for c in group_cols if c in df.columns]
    if "resource" in df.columns:
        df[out_name] = df.groupby(cols, dropna=False)["resource"].transform("nunique")
    else:
        df[out_name] = df.groupby(cols, dropna=False)["marker"].transform("size")
    return df


def _sps(df: pd.DataFrame, celltype_key_cols, spec_cols, out_name: str) -> pd.DataFrame:
    """Specificity = round(1 / (#cell types sharing the marker in the group), 2)."""
    ck = [c for c in celltype_key_cols if c in df.columns]
    sc = [c for c in spec_cols if c in df.columns]
    ct_cols = [c for c in ck if c not in sc]
    if ct_cols:
        ctkey = df[ct_cols].astype(str).agg("|".join, axis=1)
    else:
        ctkey = pd.Series("", index=df.index)
    tmp = df.assign(_ctkey=ctkey)
    n = tmp.groupby(sc, dropna=False)["_ctkey"].transform("nunique")
    df[out_name] = (1.0 / n.replace(0, np.nan)).round(2)
    return df


# ---------------------------------------------------------------------------
# custom-table normalisation
# ---------------------------------------------------------------------------

def _normalize_custom(marker_table, *, database, celltype_column, marker_column,
                      species_column, tissue_column, disease_column,
                      marker_type_column, resource_column) -> pd.DataFrame:
    df = marker_table
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.copy()

    if celltype_column not in df.columns:
        raise ValueError(f"celltype_column {celltype_column!r} not found in marker_table.")
    if marker_column not in df.columns:
        raise ValueError(f"marker_column {marker_column!r} not found in marker_table.")

    celltype_out = "CL_celltype" if database == "healthy" else "NCIT_celltype"
    out = pd.DataFrame()
    out[celltype_out] = df[celltype_column].astype(str)
    out["marker"] = df[marker_column].astype(str)

    out["species"] = df[species_column].astype(str) if species_column in df.columns else "Human"

    if tissue_column in df.columns:
        out["Uberon_tissue"] = df[tissue_column].astype(str)
    else:
        out["Uberon_tissue"] = np.nan

    if marker_type_column in df.columns:
        mt = df[marker_type_column].astype(str).str.lower()
        out["marker_type"] = mt.where(mt.isin(["positive", "negative"]), "positive")
    else:
        out["marker_type"] = "positive"

    out["resource"] = df[resource_column].astype(str) if resource_column in df.columns else "custom_set"

    # ontology ids
    ub_map = _ontology_name_to_id(get_default_uberon_ontology_path())
    out["Uberon_ID"] = _map_lower(out["Uberon_tissue"], ub_map)

    if database == "healthy":
        cl_map = _ontology_name_to_id(get_default_cell_ontology_path())
        out["CL_ID"] = _map_lower(out[celltype_out], cl_map)
    else:
        out["NCIT_ID"] = np.nan
        if disease_column in df.columns:
            out["DO_diseasetype"] = df[disease_column].astype(str)
        else:
            out["DO_diseasetype"] = np.nan
        out["DO_ID"] = np.nan

    # gene description (optional / defensive)
    gd = _gene_description_map()
    if gd:
        out["gene_description"] = out["marker"].map(gd)

    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_HEALTHY_DROP = ["SPs_tissue_specific", "SPs_global", "ECs_tissue_specific", "ECs_global"]
_DISEASE_DROP = ["SPs_disease_tissue_specific", "SPs_disease_specific", "SPs_global",
                 "ECs_disease_tissue_specific", "ECs_disease_specific", "ECs_global"]


def marker_database_integration(
        marker_table,
        database: str = "healthy",
        species_column: str = "species",
        disease_column: str = "disease",
        tissue_column: str = "tissue",
        celltype_column: str = "cell_type",
        marker_column: str = "marker",
        marker_type_column: str = "marker_type",
        resource_column: str = "resource",
) -> pd.DataFrame:
    """Augment the Accordion database with a custom marker table.

    The custom markers are appended to the packaged ``database`` (``"healthy"``
    or ``"disease"``) and the evidence (``ECs_*``) and specificity (``SPs_*``)
    scores are recomputed on the combined set. Pass the result to
    ``accordion(database=...)`` (saving it to Parquet first, since ``accordion``
    reads a file path) to annotate with it.

    Parameters
    ----------
    marker_table
        Table with at least the cell-type and marker columns. Optional species,
        tissue, disease, marker-type and resource columns are used when present.
    database
        ``"healthy"`` (Cell Ontology cell types) or ``"disease"`` (NCI Thesaurus
        cell types). Default ``"healthy"``.
    *_column
        Column-name overrides for the input table.

    Returns
    -------
    pandas.DataFrame
        The integrated marker database with recomputed scores.
    """
    if database not in ("healthy", "disease"):
        raise ValueError('database must be "healthy" or "disease".')

    condition = "health" if database == "healthy" else "disease"
    base = _read_parquet(get_default_marker_database_path(condition))

    custom = _normalize_custom(
        marker_table,
        database=database,
        celltype_column=celltype_column,
        marker_column=marker_column,
        species_column=species_column,
        tissue_column=tissue_column,
        disease_column=disease_column,
        marker_type_column=marker_type_column,
        resource_column=resource_column,
    )

    # Drop the score columns from the built-in DB; they are recomputed below.
    drop = _HEALTHY_DROP if database == "healthy" else _DISEASE_DROP
    base = base.drop(columns=[c for c in drop if c in base.columns])

    # If the custom table has no tissue, ignore tissue in the base too, so
    # markers merge on cell type only.
    if custom["Uberon_tissue"].isna().all():
        for col in ("Uberon_tissue", "Uberon_ID"):
            if col in base.columns:
                base[col] = np.nan

    combined = pd.concat([base, custom], ignore_index=True, sort=False)
    combined = combined.drop_duplicates()

    if database == "healthy":
        combined = _ecs(
            combined,
            ["species", "Uberon_tissue", "Uberon_ID", "CL_celltype", "CL_ID", "marker", "marker_type"],
            "ECs_tissue_specific")
        combined = _ecs(
            combined,
            ["species", "CL_celltype", "CL_ID", "marker", "marker_type"],
            "ECs_global")
        combined = _sps(
            combined,
            ["species", "CL_celltype", "CL_ID", "marker", "marker_type"],
            ["species", "marker", "marker_type"],
            "SPs_global")
        combined = _sps(
            combined,
            ["species", "Uberon_tissue", "Uberon_ID", "CL_celltype", "CL_ID", "marker", "marker_type"],
            ["species", "Uberon_tissue", "Uberon_ID", "marker", "marker_type"],
            "SPs_tissue_specific")
    else:
        combined = _ecs(
            combined,
            ["species", "DO_diseasetype", "DO_ID", "Uberon_tissue", "Uberon_ID",
             "NCIT_celltype", "NCIT_ID", "marker", "marker_type"],
            "ECs_disease_tissue_specific")
        combined = _ecs(
            combined,
            ["species", "DO_diseasetype", "DO_ID", "NCIT_celltype", "NCIT_ID", "marker", "marker_type"],
            "ECs_disease_specific")
        combined = _ecs(
            combined,
            ["species", "NCIT_celltype", "NCIT_ID", "marker", "marker_type"],
            "ECs_global")
        combined = _sps(
            combined,
            ["species", "NCIT_celltype", "NCIT_ID", "marker", "marker_type"],
            ["species", "marker", "marker_type"],
            "SPs_global")
        combined = _sps(
            combined,
            ["species", "DO_diseasetype", "DO_ID", "NCIT_celltype", "NCIT_ID", "marker", "marker_type"],
            ["species", "DO_diseasetype", "DO_ID", "marker", "marker_type"],
            "SPs_disease_specific")
        combined = _sps(
            combined,
            ["species", "DO_diseasetype", "DO_ID", "Uberon_tissue", "Uberon_ID",
             "NCIT_celltype", "NCIT_ID", "marker", "marker_type"],
            ["species", "DO_diseasetype", "DO_ID", "Uberon_tissue", "Uberon_ID", "marker", "marker_type"],
            "SPs_disease_tissue_specific")

    return combined.reset_index(drop=True)
