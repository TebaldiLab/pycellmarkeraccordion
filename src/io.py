"""Input/output helpers: locating packaged data and loading marker databases.

Provides paths to the bundled marker databases and ontology files, reads a marker
database from a file path or an in-memory DataFrame, and normalizes its schema to
the columns the annotation engine expects (for both the healthy and disease
databases).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd
import polars as pl


def normalize_condition(condition: str | None) -> str:
    if condition is None:
        return "health"
    value = str(condition).strip().lower()
    if value in {"health", "healthy", "normal", "control"}:
        return "health"
    if value in {"disease", "diseased", "pathology", "pathological"}:
        return "disease"
    raise ValueError("condition must be 'health' or 'disease'")


def get_package_data_path(filename: str) -> str:
    return str(files("pycellmarkeraccordion").joinpath("data", filename))


def get_default_marker_database_path(condition: str | None = "health") -> str:
    condition = normalize_condition(condition)
    filename = "disease_accordion_marker.parquet" if condition == "disease" else "accordion_marker.parquet"
    return get_package_data_path(filename)


def get_default_cell_ontology_path() -> str:
    return get_package_data_path("cl-basic.obo")


def get_default_uberon_ontology_path() -> str:
    return get_package_data_path("uberon-basic.obo")


def read_marker_database(database=None, condition: str | None = "health"):
    """Load a marker database as a normalized ``pandas.DataFrame``.

    Shared loader used internally by :func:`~pycellmarkeraccordion.accordion`,
    :func:`~pycellmarkeraccordion.accordion_bulk_deconvolution`, the ``list_*``
    query helpers and the ontology-aware ``plot_*`` functions. It accepts
    either an in-memory marker table or a file path, and otherwise falls back
    to one of the two marker databases packaged with pycellmarkeraccordion:
    ``accordion_marker`` (healthy cell types, Cell Ontology) or
    ``disease_accordion_marker`` (disease/aberrant cell types, NCI Thesaurus).
    Call it directly to inspect the raw marker table -- e.g. to see which
    columns/scores are available -- or to check that a custom database
    normalizes as expected before passing it to ``accordion(database=...)``.

    Parameters
    ----------
    database : str, pandas.DataFrame, or None, default None
        Custom marker database: a file path (``.parquet``, ``.csv``, ``.tsv``)
        or an in-memory DataFrame (e.g. the output of
        :func:`~pycellmarkeraccordion.marker_database_integration`). If
        ``None``, the packaged database selected by ``condition`` is loaded
        instead.
    condition : {"health", "disease"} or None, default "health"
        Which packaged database to fall back to when ``database`` is
        ``None``: ``"health"`` loads ``accordion_marker`` (healthy cell
        types), ``"disease"`` loads ``disease_accordion_marker``
        (disease/aberrant cell types).

    Returns
    -------
    pandas.DataFrame
        The marker table, normalized to the schema the annotation engine
        expects (see :func:`normalize_marker_database_schema`): missing
        ``species``, ``Uberon_tissue``, ``Uberon_ID``, ``CL_ID`` and
        ``resource`` columns are filled with defaults, and for the disease
        database the ``NCIT_celltype``/``NCIT_ID`` columns are aligned with
        their ``CL_celltype``/``CL_ID`` counterparts.

    Examples
    --------
    >>> df = read_marker_database(condition="health")
    >>> df.columns
    """
    # Accept an in-memory marker DataFrame directly (e.g. the output of
    # marker_database_integration), so accordion(database=df) works without
    # writing a temporary file.
    if isinstance(database, pd.DataFrame):
        return normalize_marker_database_schema(database.copy(), condition=condition)

    path = Path(database) if database is not None else Path(get_default_marker_database_path(condition))
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return normalize_marker_database_schema(pl.read_parquet(path).to_pandas(), condition=condition)
    if suffix == ".tsv":
        return normalize_marker_database_schema(pl.read_csv(path, separator="	").to_pandas(), condition=condition)
    if suffix == ".csv":
        return normalize_marker_database_schema(pl.read_csv(path).to_pandas(), condition=condition)
    raise ValueError(f"Unsupported marker database format: {path}. Use .parquet, .tsv, or .csv.")


def normalize_marker_database_schema(marker_df, condition: str | None = "health"):
    """Normalize health/disease marker schemas to the internal Accordion schema."""
    condition_norm = normalize_condition(condition)
    marker_df = marker_df.copy()

    if condition_norm == "disease":
        rename_map = {}
        if "NCIT_celltype" in marker_df.columns and "CL_celltype" not in marker_df.columns:
            rename_map["NCIT_celltype"] = "CL_celltype"
        if "NCIT_ID" in marker_df.columns and "CL_ID" not in marker_df.columns:
            rename_map["NCIT_ID"] = "CL_ID"
        if "ECs_NCIT_global" in marker_df.columns and "ECs_global" not in marker_df.columns:
            rename_map["ECs_NCIT_global"] = "ECs_global"
        marker_df = marker_df.rename(columns=rename_map)

        if "NCIT_celltype" not in marker_df.columns and "CL_celltype" in marker_df.columns:
            marker_df["NCIT_celltype"] = marker_df["CL_celltype"]
        if "NCIT_ID" not in marker_df.columns and "CL_ID" in marker_df.columns:
            marker_df["NCIT_ID"] = marker_df["CL_ID"]

    defaults = {
        "species": "Human",
        "Uberon_tissue": "ALL",
        "Uberon_ID": "ALL",
        "CL_ID": "",
        "resource": "",
    }
    for col, value in defaults.items():
        if col not in marker_df.columns:
            marker_df[col] = value

    return marker_df
