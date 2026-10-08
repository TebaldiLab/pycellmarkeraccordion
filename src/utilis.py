"""Small helpers shared by accordion, annotation_helpers and spatial_entropy."""

from __future__ import annotations

import os

__all__ = ["DEFAULT_N_JOBS", "resolve_n_jobs", "as_text_series"]

#: Serial. The work is numpy/pandas-bound and the parallel paths use threads, so
#: it runs under the GIL and spreading it over cores has measured slower, not
#: just equal. Parallelism is opt-in.
DEFAULT_N_JOBS = 1


def resolve_n_jobs(n_jobs=None):
    """Resolve the worker count.

    ``None`` or ``"default"`` gives :data:`DEFAULT_N_JOBS` (serial). A negative
    value counts back from every core, as in scikit-learn: ``-1`` is all cores,
    ``-2`` all but one. A positive value is used as given. The result is never
    below 1.
    """
    if n_jobs is None:
        return DEFAULT_N_JOBS
    if isinstance(n_jobs, str) and n_jobs.lower() in {"none", "default"}:
        return DEFAULT_N_JOBS
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)


def as_text_series(series):
    """Convert a pandas column, categoricals included, to string safely.

    Used where columns are concatenated: ``astype(str)`` on a categorical with
    missing values yields the literal ``"nan"``, which would then become part of
    a marker name or a group key.
    """
    return series.astype("string").fillna("")
