"""Listing helpers for the packaged Cell Marker Accordion databases.

``list_celltypes`` / ``list_tissues`` / ``list_disease_tissues`` /
``list_aberrant_celltypes`` / ``list_diseases`` simply read the packaged
marker database and return the unique values of a column, optionally
filtered by species / tissue / cell type / disease.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import pandas as pd

from .io import read_marker_database

SpeciesArg = Union[str, Iterable[str]]


def _as_list(value) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _first_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _filter_in(df: pd.DataFrame, column: Optional[str], values) -> pd.DataFrame:
    values = _as_list(values)
    if column is not None and values is not None and column in df.columns:
        return df[df[column].astype(str).isin([str(v) for v in values])]
    return df


def _unique_sorted(series: pd.Series) -> list:
    return sorted(series.dropna().astype(str).unique().tolist())


def _require(col: Optional[str], what: str, df: pd.DataFrame) -> str:
    if col is None:
        raise KeyError(
            f"Could not find a {what} column in the marker database. "
            f"Available columns: {list(df.columns)}"
        )
    return col


# ---------------------------------------------------------------------------
# Healthy database
# ---------------------------------------------------------------------------

def list_celltypes(species: SpeciesArg = ("Human", "Mouse"),
                   tissue: Optional[SpeciesArg] = None) -> list:
    """Cell types available in the (healthy) Accordion database.

    Parameters
    ----------
    species
        Species to include (``"Human"`` and/or ``"Mouse"``). Default both.
    tissue
        Optional tissue filter. If ``None``, all tissues are considered.
    """
    df = read_marker_database(condition="health")
    df = _filter_in(df, "species", species)
    df = _filter_in(df, _first_col(df, ["Uberon_tissue"]), tissue)
    col = _require(_first_col(df, ["CL_celltype", "cell_type"]), "cell-type", df)
    return _unique_sorted(df[col])


def list_tissues(species: SpeciesArg = ("Human", "Mouse"),
                 celltype: Optional[SpeciesArg] = None) -> list:
    """Tissues available in the (healthy) Accordion database.

    Parameters
    ----------
    species
        Species to include. Default both.
    celltype
        Optional cell-type filter. If ``None``, all cell types are considered.
    """
    df = read_marker_database(condition="health")
    df = _filter_in(df, "species", species)
    df = _filter_in(df, _first_col(df, ["CL_celltype", "cell_type"]), celltype)
    col = _require(_first_col(df, ["Uberon_tissue"]), "tissue", df)
    return _unique_sorted(df[col])


# ---------------------------------------------------------------------------
# Disease database
# ---------------------------------------------------------------------------

def list_disease_tissues(species: SpeciesArg = ("Human", "Mouse"),
                         disease: Optional[SpeciesArg] = None,
                         aberrant_celltype: Optional[SpeciesArg] = None) -> list:
    """Tissues available in the disease Accordion database."""
    df = read_marker_database(condition="disease")
    df = _filter_in(df, "species", species)
    df = _filter_in(df, _first_col(df, ["NCIT_celltype", "CL_celltype"]), aberrant_celltype)
    df = _filter_in(df, _first_col(df, ["DO_diseasetype", "disease"]), disease)
    col = _require(_first_col(df, ["Uberon_tissue"]), "tissue", df)
    return _unique_sorted(df[col])


def list_aberrant_celltypes(species: SpeciesArg = ("Human", "Mouse"),
                            disease: Optional[SpeciesArg] = None,
                            tissue: Optional[SpeciesArg] = None) -> list:
    """Aberrant cell types available in the disease Accordion database."""
    df = read_marker_database(condition="disease")
    df = _filter_in(df, "species", species)
    df = _filter_in(df, _first_col(df, ["Uberon_tissue"]), tissue)
    df = _filter_in(df, _first_col(df, ["DO_diseasetype", "disease"]), disease)
    col = _require(_first_col(df, ["NCIT_celltype", "CL_celltype"]), "aberrant cell-type", df)
    return _unique_sorted(df[col])


def list_diseases(species: SpeciesArg = ("Human", "Mouse"),
                  tissue: Optional[SpeciesArg] = None,
                  aberrant_celltype: Optional[SpeciesArg] = None) -> list:
    """Diseases available in the disease Accordion database."""
    df = read_marker_database(condition="disease")
    df = _filter_in(df, "species", species)
    df = _filter_in(df, _first_col(df, ["Uberon_tissue"]), tissue)
    df = _filter_in(df, _first_col(df, ["NCIT_celltype", "CL_celltype"]), aberrant_celltype)
    col = _require(_first_col(df, ["DO_diseasetype", "disease"]), "disease", df)
    return _unique_sorted(df[col])
