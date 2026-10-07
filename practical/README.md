# Practical notebooks &mdash; real data, real decisions

Case studies on **real datasets**, each built around a practical question rather than around one
lecture. The session notebooks one folder up teach one method at a time, mostly on small or
synthetic data. These put several methods side by side on a real table and walk through the
decisions you face in practice: cleaning, encoding, honest evaluation, choosing a metric and a
threshold, and deciding what to deploy.

Each notebook is self-contained. It imports only `../nb_utils.py` (the course's plotting style) and
downloads its dataset **once** from the UCI Machine Learning Repository into `data/raw/`, which is
git-ignored and checked against SHA-256 fingerprints. Offline? Each notebook says which file to
download by hand and where to put it. Outputs are committed, so every table and figure can be read
without running anything.

| # | Notebook | Data | Covers |
| --- | --- | --- | --- |
| **01** | [`01-comparing-classifiers.ipynb`](01-comparing-classifiers.ipynb) | UCI *Adult*: the 1994 US census income table, 48,842 people (Becker &amp; Kohavi, 1996; CC BY 4.0) | **Logistic regression, naive Bayes, SVMs (linear and RBF) and a decision tree on one real table.** The data's quirks (test labels that end with a period, a sampling-weight column, a duplicated column, missing values that carry information); preprocessing and cross-validation **written by hand** &mdash; no `Pipeline`, no `GridSearchCV` &mdash; so that every fitted statistic visibly comes from training rows only; why naive Bayes needs the right likelihood for each column (AUC 0.816 &rarr; 0.895); an RBF kernel that buys nothing on this table; one look at the test file, with two kinds of uncertainty (other training samples, and a paired bootstrap of the test rows); thresholds chosen on out-of-fold scores; calibration and Platt scaling by hand; naive Bayes's overconfidence traced to double counting; learning curves from 100 to 32,561 rows that reverse the ranking at both ends; and a recommendation for a concrete use case. No ensembles: those are Session&nbsp;09. |

## Running them

```sh
cd courses/machine-learning/notebooks/practical
jupyter lab            # or: jupyter notebook
```

Requirements: `numpy`, `pandas`, `matplotlib`, `scipy` and `scikit-learn` &ge; 1.7 (written and run
with 1.8). The first run needs internet access once, for a 0.6 MB download. Notebook 01 takes two
to five minutes on a laptop, most of it in the kernel SVM and the learning curves. Cells marked
`# <-- CHANGE ME` are meant to be edited and re-run.
