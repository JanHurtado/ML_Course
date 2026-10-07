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


# =====================================================================
# Classification datasets for the Session 04 (logistic regression) notebooks.
# Labels are drawn from a KNOWN logistic model y ~ Bernoulli(sigma(w.x + b)), so a
# from-scratch fit can be checked against the coefficients that generated the data
# (with enough samples, the maximum-likelihood fit is close to the truth).
# =====================================================================
def _sigmoid(z):
    """Numerically stable logistic sigmoid, elementwise."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def make_logistic_1d(n=60, seed=0, w=1.6, b=-0.4, xspan=(-5.0, 5.0)):
    """One feature, two classes: labels y ~ Bernoulli(sigma(w*x + b)). Returns
    (x, y, w, b); the decision boundary sits at the point x* = -b/w."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(xspan[0], xspan[1], n))
    y = (rng.random(n) < _sigmoid(w * x + b)).astype(int)
    return x, y, float(w), float(b)


def make_logistic_2d(n=200, seed=0, w=(1.6, -2.2), b=0.5, scale=1.6):
    """Two features, two classes: labels y ~ Bernoulli(sigma(X @ w + b)) with
    X ~ N(0, scale). Returns (X, y, w_true, b_true); the decision boundary is the
    line w[0]*x1 + w[1]*x2 + b = 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, scale, size=(n, 2))
    w_true = np.asarray(w, dtype=float)
    y = (rng.random(n) < _sigmoid(X @ w_true + b)).astype(int)
    return X, y, w_true, float(b)


# =====================================================================
# Generative datasets for the Session 05 notebooks (continuous features).
# The generator returns the TRUE parameters it used, so a from-scratch fit
# can be checked against the truth as well as against scikit-learn.
# =====================================================================
def make_gaussian_classes(n=400, seed=0, mus=((-1.4, 0.0), (1.6, 0.4)),
                          covs=None, priors=None):
    """K multivariate-Gaussian classes with KNOWN parameters.

    ``covs`` may be omitted (identity), a single (d, d) matrix (shared by every
    class -- the LDA assumption), or one (d, d) matrix per class (the QDA case).
    Returns ``(X, y, params)`` with ``params = {"mus", "covs", "priors"}``.
    """
    rng = np.random.default_rng(seed)
    mus = np.asarray(mus, dtype=float)
    K, d = mus.shape
    covs = np.eye(d) if covs is None else np.asarray(covs, dtype=float)
    if covs.ndim == 2:                          # one covariance shared by all classes
        covs = np.stack([covs] * K)
    priors = np.full(K, 1.0 / K) if priors is None else np.asarray(priors, dtype=float)

    y = rng.choice(K, size=n, p=priors)
    X = np.empty((n, d))
    for k in range(K):
        mask = y == k
        X[mask] = rng.multivariate_normal(mus[k], covs[k], size=int(mask.sum()))
    return X, y, {"mus": mus, "covs": covs, "priors": priors}


# =====================================================================
# Datasets for the Session 06/07 notebooks (SVMs and kernel methods).
# Labels are in {-1, +1}: the SVM convention, where y * f(x) > 0 means
# "classified correctly" for either class.
# =====================================================================
def make_two_blobs(n=200, sep=1.6, spread=1.0, seed=0):
    """Two round Gaussian clouds centered at (-sep, 0) and (+sep, 0), labels in {-1, +1}.

    A large ``sep`` gives separable data (a hard margin exists); a small one gives
    overlapping classes that need the soft margin.
    """
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.5, -1, 1)
    X = rng.normal(0.0, spread, size=(n, 2))
    X[:, 0] += sep * y
    return X, y


def make_rings(n=200, inner=(0.0, 1.0), outer=(1.7, 2.4), seed=0):
    """Two concentric rings, labels -1 (inner) and +1 (outer): no straight line separates them.

    Matches Lecture 06's quadratic-map figure (radii uniform in ``inner`` / ``outer``).
    The first half of the rows is the inner ring, the second half the outer one.
    """
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) < n // 2, -1, 1)
    r = np.where(y < 0, rng.uniform(*inner, n), rng.uniform(*outer, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)]), y


# =====================================================================
# The course's two RUNNING TOY EXAMPLES, used by Lectures 08, 09, 10 and 12.
#
# Both are small enough to work out with pencil and paper, and every number
# they produce is exact -- which is the point: a notebook can check scikit-learn
# against arithmetic the reader did themselves. They recur across four lectures,
# so the same ten points you split with a decision tree in Session 08 are the
# ten points you cluster in Session 10.
# =====================================================================
def make_ten_points():
    """Lecture 08's TEN example: 10 points, 2 integer features, 2 classes.

    Returns ``(X, y)`` with ``X`` of shape (10, 2) and ``y`` in {0, 1}, five of
    each. The exact facts this example was chosen for:

    * the root Gini impurity is exactly 1/2 and the root entropy exactly 1 bit;
    * the best root split is ``x1 <= 3.5``, with information gain exactly 3/14;
    * splitting on ``x2`` at 2.5 or at 8.5 gives an information gain of exactly 0
      -- two cuts that look reasonable and buy nothing;
    * a fully grown tree reaches 10/10 training accuracy with four leaves.

    Sessions 09 (boosting reuses the same three cuts) and 10 (DBSCAN on the same
    points) build on it.
    """
    class0 = [(1, 4), (2, 1), (3, 9), (6, 5), (8, 3)]
    class1 = [(4, 8), (5, 10), (7, 2), (9, 6), (10, 7)]
    X = np.array(class0 + class1, dtype=float)
    y = np.array([0] * 5 + [1] * 5)
    return X, y


def make_rent8():
    """Lecture 08's RENT8 example: 8 apartments, area -> rent, for regression.

    Returns ``(x, y)`` with ``x`` = 1..8 (area, in tens of m^2) and ``y`` the rent
    in hundreds of EUR. Exact by construction:

    * the best constant prediction is the mean 7.5, with SSE exactly 138;
    * the best single split is ``area <= 4.5``, leaving SSE exactly 10 -- so that
      one cut is worth a gain of exactly 128;
    * k-means with K=2 on the rents alone finds centers 3.5 and 11.5 with WCSS 10,
      the *same* 10 (Session 10's bridge back to Session 08).

    ``x`` is returned 1-D; reshape to ``x[:, None]`` for scikit-learn.
    """
    x = np.arange(1, 9, dtype=float)
    y = np.array([2, 3, 4, 5, 10, 11, 12, 13], dtype=float)
    return x, y


def make_xor():
    """The four corners of the XOR problem -- Lecture 12's wall.

    Returns ``(X, y)`` with ``X`` the corners (0,0), (0,1), (1,0), (1,1) and
    ``y = x1 XOR x2``. No single neuron (no affine function of the inputs) can
    separate these four points, and the one-line reason is an identity you can
    check by hand: for *any* affine f,

        f(0,0) + f(1,1) = f(0,1) + f(1,0),

    so the two diagonals always agree -- while XOR demands they disagree.
    """
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    return X, y


def make_cigars(n=300, spread=10.0, gap=1.6, thin=0.35, seed=11):
    """Two long, parallel class clouds -- Lecture 11's PCA-vs-LDA example.

    Returns ``(X, y)`` with ``n`` points *per class* (so 600 rows by default).
    The classes are stretched along ``x1`` (standard deviation ``spread``) and
    separated along ``x2`` by ``gap``, so the direction carrying almost all the
    *variance* is orthogonal to the direction carrying all the *class
    information*. Unsupervised PCA therefore keeps the useless axis and throws
    the useful one away, while LDA -- which sees the labels -- does the opposite.

    With the defaults (the lecture's own settings) PC1 holds **99.24%** of the
    variance and separates the classes at **0.5417** -- chance -- while LDA's
    single component reaches **0.9867**. Measure the accuracies with
    ``best_1d_threshold_accuracy`` rather than by fitting a classifier, so the
    number describes the *projection* and not whatever model comes after it.
    """
    rng = np.random.default_rng(seed)
    c0 = np.c_[rng.normal(0.0, spread, n), rng.normal(-gap / 2.0, thin, n)]
    c1 = np.c_[rng.normal(0.0, spread, n), rng.normal(+gap / 2.0, thin, n)]
    return np.vstack([c0, c1]), np.array([0] * n + [1] * n)


def make_ellipses3(seed=0):
    """Three long, differently ORIENTED Gaussian ellipses, 200 points each.

    The clustering dataset on which a round model and an elliptical one genuinely
    part company: k-means cannot express a tilted, stretched cluster and a
    full-covariance Gaussian mixture can. Matches Lecture 10's figure.

    Returns ``(X, y)`` with ``y`` the component that generated each point -- for
    *scoring* only, never for fitting.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(600, 2)) * np.array([3.0, 0.4])      # long and thin
    out, lab = [], []
    for k, (deg, cx, cy) in enumerate(((0, 0.0, 0.0), (60, 4.0, 0.0), (-60, 2.0, 3.5))):
        t = np.deg2rad(deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        out.append(base[k * 200:(k + 1) * 200] @ R.T + np.array([cx, cy]))
        lab.append(np.full(200, k))
    return np.vstack(out), np.concatenate(lab)


def make_uneven_clusters(n_per=200, d=20, radii=(1.0, 5.0, 25.0),
                         gaps=(60.0, 300.0), seed=0):
    """Isotropic Gaussian clusters strung along the first axis, with KNOWN sizes and gaps.

    ``radii[k]`` is the per-coordinate standard deviation of cluster ``k``, and
    ``gaps[k]`` is the distance between the centers of clusters ``k`` and ``k+1``.
    Both the SIZE ratios and the GAP ratios are therefore known before any
    embedding is computed -- which is what makes them worth measuring again in a
    t-SNE or UMAP map, where they come back as 1.

    Returns ``(X, y)`` with ``X`` of shape ``(n_per * len(radii), d)``. The
    defaults are Lecture 11's experiment: radii 1 : 5 : 25 and gaps 1 : 5, which
    the data reproduces as a measured 24.8x size ratio and 5.0x gap ratio.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((len(radii), d))
    centers[1:, 0] = np.cumsum(np.asarray(gaps, dtype=float))
    X = np.vstack([rng.normal(0.0, r, (n_per, d)) + c
                   for r, c in zip(radii, centers)])
    return X, np.repeat(np.arange(len(radii)), n_per)


def best_1d_threshold_accuracy(z, y):
    """Best accuracy ANY single threshold on the 1-D score ``z`` can reach.

    Used instead of fitting a classifier so the result measures the projection
    itself, not the model trained on top of it. Runs in O(n log n).
    """
    z = np.asarray(z, dtype=float).ravel()
    y = np.asarray(y).ravel()
    order = np.argsort(z)
    ys = y[order]
    n = len(ys)
    # the two constant predictors are always available
    best = max((ys == 0).mean(), (ys == 1).mean())
    left1 = np.cumsum(ys == 1)          # 1s at or left of each cut
    total1 = int(left1[-1])
    for i in range(n - 1):
        nl = i + 1
        # predict 0 left of the cut and 1 right of it, then the mirror image
        correct = (nl - left1[i]) + (total1 - left1[i])
        best = max(best, correct / n, (n - correct) / n)
    return float(best)
