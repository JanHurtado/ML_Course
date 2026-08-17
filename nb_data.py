"""Self-contained synthetic datasets for the Session 01 practical notebooks.

A sibling of the notebooks (like ``nb_utils.py``) so every notebook stays
self-contained and offline. The centerpiece is a small **apartments -> monthly
rent** table that matches Lecture 01's worked example. It comes in two forms:

* ``make_apartments(messy=False)`` -- a clean table, to show the anatomy of
  structured data (the design matrix ``X`` and the target ``y``);
* ``make_apartments(messy=True)`` -- the *same* data with realistic problems
  injected (missing values, duplicated rows, outliers/typos, inconsistent
  category spellings, a column stored with the wrong dtype) to practise
  exploratory analysis and cleaning.

``basic_clean(df)`` undoes the *structural* damage (dtypes, category spellings,
impossible values, duplicates) and is reused by the pipeline notebook. It
deliberately leaves genuine missing values in place so a downstream imputer has
something to do.

Usage inside a notebook (run from this directory)::

    from nb_data import make_apartments, basic_clean
    df = make_apartments(messy=True, seed=0)

Everything is generated from a transparent formula with a fixed seed, so the
numbers are reproducible and there is nothing to download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The five neighborhoods, most-expensive first, with a rent multiplier each.
NEIGHBORHOODS = ["Centro", "Norte", "Sur", "Este", "Oeste"]
_NB_MULT = {"Centro": 1.35, "Norte": 1.10, "Sur": 0.85, "Este": 0.95, "Oeste": 0.90}


def _rent(area, rooms, bathrooms, floor, age, nb, elevator, dist):
    """A transparent generating function for the monthly rent, in EUR."""
    base = (6.0 * area + 120 * rooms + 90 * bathrooms + 8 * floor
            - 3.5 * age + 300 * elevator - 40 * dist)
    return base * _NB_MULT[nb]


def make_apartments(n: int = 400, seed: int = 0, messy: bool = False) -> pd.DataFrame:
    """Return a synthetic apartments-vs-rent table.

    Columns (features + target):
        area_m2, rooms, bathrooms, floor, building_age, neighborhood,
        has_elevator, dist_center_km, rent_eur (the target).

    With ``messy=True`` the same table is returned with realistic data-quality
    problems injected, ready for cleaning.
    """
    rng = np.random.default_rng(seed)

    area = rng.normal(75, 25, n).clip(25, 200)
    rooms = np.clip((area / 28 + rng.normal(0, 0.6, n)).round(), 1, 6).astype(int)
    bathrooms = np.clip((rooms / 2 + rng.normal(0, 0.4, n)).round(), 1, 3).astype(int)
    floor = rng.integers(0, 12, n)
    age = rng.integers(0, 80, n)
    nb = rng.choice(NEIGHBORHOODS, n, p=[0.28, 0.20, 0.18, 0.17, 0.17])
    elevator = ((floor >= 3) | (rng.random(n) < 0.5)).astype(int)
    dist = np.abs(rng.normal(4, 2.5, n)).clip(0.2, 20)  # km to the city center

    noise = rng.normal(0, 120, n)
    rent = np.array([_rent(*t) for t in
                     zip(area, rooms, bathrooms, floor, age, nb, elevator, dist)])
    rent = (rent + noise).clip(250, None).round(0)

    df = pd.DataFrame({
        "area_m2": area.round(1),
        "rooms": rooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "building_age": age,
        "neighborhood": nb,
        "has_elevator": elevator.astype(bool),
        "dist_center_km": dist.round(2),
        "rent_eur": rent,
    })

    if not messy:
        return df

    # ---------------- inject realistic data-quality problems ----------------
    df = df.copy()

    # (1) Missing values, sprinkled at random into three columns.
    for col, frac in [("area_m2", 0.06), ("building_age", 0.08), ("dist_center_km", 0.05)]:
        idx = rng.choice(n, int(frac * n), replace=False)
        df.loc[idx, col] = np.nan

    # (2) Inconsistent category spellings: whitespace + case.
    variant = {"Centro": "centro ", "Norte": "NORTE", "Sur": "sur",
               "Este": " Este", "Oeste": "oeste"}
    mess_idx = rng.choice(n, int(0.15 * n), replace=False)
    df.loc[mess_idx, "neighborhood"] = df.loc[mess_idx, "neighborhood"].map(variant)

    # (3) Wrong dtype: store some room counts as strings, forcing an object column.
    df["rooms"] = df["rooms"].astype(object)
    str_idx = rng.choice(n, int(0.30 * n), replace=False)
    df.loc[str_idx, "rooms"] = df.loc[str_idx, "rooms"].map(str)

    # (4) Impossible values / data-entry typos.
    df.loc[int(rng.integers(0, n)), "building_age"] = 999.0   # sentinel typo
    df.loc[int(rng.integers(0, n)), "area_m2"] = 4000.0       # m2 typo
    df.loc[int(rng.integers(0, n)), "dist_center_km"] = -3.0  # impossible

    # (5) A handful of duplicated rows.
    dups = df.sample(5, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True)

    # Shuffle so the problems are not all at the end.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the *structural* damage from ``make_apartments(messy=True)``.

    Fixes dtypes, standardizes category spellings, turns impossible values into
    missing, and drops duplicates. Genuine missing values are left in place on
    purpose, so a downstream imputer (inside a Pipeline) still has work to do.
    """
    df = df.copy()

    # numeric columns that may have been stored as strings / objects
    for col in ["area_m2", "rooms", "bathrooms", "floor", "building_age", "dist_center_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # standardize the neighborhood spelling: trim whitespace, Title-case
    df["neighborhood"] = df["neighborhood"].astype(str).str.strip().str.title()

    # impossible values -> missing (to be imputed later)
    df.loc[df["building_age"] > 120, "building_age"] = np.nan
    df.loc[df["area_m2"] > 500, "area_m2"] = np.nan
    df.loc[df["dist_center_km"] < 0, "dist_center_km"] = np.nan

    return df.drop_duplicates().reset_index(drop=True)


# =====================================================================
# Regression datasets for the Session 02 (linear regression) notebooks.
# All are generated from a transparent formula with a fixed seed, so the
# "true" parameters are known and a from-scratch fit can be checked.
# =====================================================================
def make_regression_1d(n=40, seed=0, w0=3.0, w1=2.0, noise=1.5, xmax=5.0):
    """A noisy straight line y = w0 + w1*x + noise. Returns (x, y), the true
    (w0, w1) being the arguments so a fit can be checked against them."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, xmax, n))
    y = w0 + w1 * x + rng.normal(0, noise, n)
    return x, y


def make_wave(n=40, seed=0, noise=0.18):
    """A nonlinear target y = sin(2*pi*x) + noise on x in [0, 1] -- the running
    example for polynomial features and over/under-fitting."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, noise, n)
    return x, y


def make_regression_md(n=200, d=4, seed=0, noise=2.0):
    """A multivariate linear target y = X @ w_true + b_true + noise, with X ~
    N(0, 1). Returns (X, y, w_true, b_true) so a from-scratch solver can be
    checked against the coefficients that actually generated the data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d))
    w_true = rng.uniform(-3, 3, size=d)
    b_true = 1.5
    y = X @ w_true + b_true + rng.normal(0, noise, size=n)
    return X, y, w_true, b_true
