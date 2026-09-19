# Machine Learning &mdash; practical session notebooks

Hands-on Jupyter companions to the lectures. The course is **practice-heavy**, so most sessions
have one or more notebooks here. Every notebook is **self-contained and offline**: it imports
two small sibling helpers and generates or uses only bundled data, so it runs top-to-bottom
with no downloads (a few clearly-marked *optional* cells fetch real data over the network).

## Running them

```sh
cd courses/machine-learning/notebooks
jupyter lab            # or: jupyter notebook
```

Requirements: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy` (all standard). Outputs
are committed **cleared** &mdash; run the cells (Shift+Enter) to produce the figures and tables.
Cells marked `# <-- CHANGE ME` are meant to be edited and re-run.

## Helper files (shared by the notebooks)

| File | What it provides |
| --- | --- |
| `nb_utils.py` | `use_style("notes"\|"slides")` &mdash; a clean, colorblind-safe matplotlib look matching the lecture figures. |
| `nb_data.py` | `make_apartments(messy=…)` / `basic_clean(…)` &mdash; the synthetic *apartments &rarr; rent* table used across Session&nbsp;01; `make_regression_1d`, `make_wave`, and `make_regression_md` (linear/nonlinear/multivariate targets with **known** coefficients) for Session&nbsp;02; `make_logistic_1d` / `make_logistic_2d` (two-class labels from a **known** logistic model) for Session&nbsp;04; and `make_gaussian_classes` (Gaussian classes with **known** parameters) for Session&nbsp;05; and `make_two_blobs`, `make_rings` (labels in $\{-1,+1\}$) for Sessions&nbsp;06&ndash;07. |

## Session 01 &mdash; Introduction (practical series)

A five-notebook arc that takes you from raw data to a trustworthy, end-to-end model. It matches
Lecture&nbsp;01's worked examples (tabular apartments&rarr;rent regression, images as tensors,
train/val/test, overfitting, gradient descent, the ML workflow) and extends them into a full
practical workflow.

| # | Notebook | Covers |
| --- | --- | --- |
| **01a** | [`session-01a-structured-data.ipynb`](session-01a-structured-data.ipynb) | Tables as the design matrix $\mathbf{X}$ and target $\mathbf{y}$; feature types; one-hot encoding; scaling; DataFrame &rarr; NumPy. |
| **01b** | [`session-01b-unstructured-data.ipynb`](session-01b-unstructured-data.ipynb) | Images (grayscale matrix, RGB channels/tensor, flattening); sound (waveform, FFT, spectrogram); time series (trend/seasonality, windowing, time-ordered split); a peek at text (bag-of-words). |
| **01c** | [`session-01c-eda-and-cleaning.ipynb`](session-01c-eda-and-cleaning.ipynb) | Exploratory data analysis; finding & fixing missing values, duplicates, outliers, wrong dtypes, messy categories; the split-before-transform (no-leakage) rule. |
| **01d** | [`session-01d-train-val-test.ipynb`](session-01d-train-val-test.ipynb) | **Honest evaluation:** the three-way split; *why a validation set matters*; tuning-on-test optimism; the winner's curse (selection inflates scores &mdash; even on pure noise); cross-validation; `GridSearchCV` & nested CV; leakage pitfalls. |
| **01e** | [`session-01e-first-ml-pipeline.ipynb`](session-01e-first-ml-pipeline.ipynb) | A leak-free `ColumnTransformer` + `Pipeline`; baseline; cross-validated model selection; test-once evaluation; the overfitting curve; gradient descent from scratch; saving the model. |
| &mdash; | [`session-01-demos.ipynb`](session-01-demos.ipynb) | Short interactive demos referenced from the slides (supervised digits, k-means, overfitting sweep). |

**Suggested order:** 01a &rarr; 01b &rarr; 01c &rarr; 01d &rarr; 01e. Notebooks 01a/01c/01d/01e
share the same apartments dataset, so the story carries across them.

## Session 02 &mdash; Linear Regression (practical series)

A four-notebook arc that builds linear regression **entirely from scratch in NumPy** and checks
every step against scikit-learn &mdash; the model, both solvers (closed form and gradient descent),
and how to evaluate and stretch it.

| # | Notebook | Covers |
| --- | --- | --- |
| **02a** | [`session-02a-model-and-cost.ipynb`](session-02a-model-and-cost.ipynb) | The line $\hat y = w_0 + w_1 x$; residuals; the least-squares (MSE) cost; the convex **cost bowl** (contour + 3-D); the design-matrix form. |
| **02b** | [`session-02b-normal-equations.ipynb`](session-02b-normal-equations.ipynb) | The **normal equations** $\boldsymbol\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$ from scratch; `inv`/`solve`/`lstsq`; recovering known coefficients & matching sklearn; the **column-space projection** geometry; collinearity/conditioning. |
| **02c** | [`session-02c-gradient-descent.ipynb`](session-02c-gradient-descent.ipynb) | **Gradient descent** from scratch (batch/stochastic/mini-batch); the **learning-rate** knob (crawl vs diverge); how **feature scaling** reshapes the bowl (ravine &rarr; round); a tidy `LinearRegressionGD` estimator vs sklearn. |
| **02d** | [`session-02d-evaluation-and-complexity.ipynb`](session-02d-evaluation-and-complexity.ipynb) | Metrics from scratch (MSE/RMSE/MAE/$R^2$); held-out evaluation; **polynomial features**; the **overfitting** U-curve & exploding weights; a one-line **ridge** preview; a real multivariate dataset (diabetes). |

**Suggested order:** 02a &rarr; 02b &rarr; 02c &rarr; 02d. Everything is implemented in NumPy and
validated against scikit-learn (and, where possible, against the coefficients that generated the
data). Ridge/lasso and cross-validated model selection are the subject of Session&nbsp;03.

## Session 03 &mdash; Regularization & Model Selection (practical series)

A three-notebook arc that **measures** the bias&ndash;variance trade-off, builds **ridge and lasso
from scratch in NumPy** (checked against scikit-learn), and picks the penalty **honestly** with
cross-validation &mdash; matching Lecture&nbsp;03.

| # | Notebook | Covers |
| --- | --- | --- |
| **03a** | [`session-03a-bias-variance.ipynb`](session-03a-bias-variance.ipynb) | The **bias&ndash;variance** trade-off *simulated*: resample &amp; refit; a **Monte-Carlo** estimate of bias&sup2;/variance/noise verifying $\mathbb{E}[(y-\hat f)^2]=\text{bias}^2+\text{variance}+\sigma^2$; the **U-curve** vs complexity. |
| **03b** | [`session-03b-ridge-lasso.ipynb`](session-03b-ridge-lasso.ipynb) | Regularization **from scratch**: exploding weights; **ridge** closed form; **lasso** by **coordinate descent** + **soft-thresholding**; the shrinkage maps; **coefficient paths**. |
| **03c** | [`session-03c-model-selection-cv.ipynb`](session-03c-model-selection-cv.ipynb) | **Cross-validation**: why one split is noisy; **k-fold from scratch** (matched to scikit-learn); picking $\lambda$ at the CV minimum; the **leak-free** train/CV/test workflow; **ridge vs lasso** on the **diabetes** dataset. |

**Suggested order:** 03a &rarr; 03b &rarr; 03c. Everything is NumPy-first and validated against
scikit-learn, continuing the Session&nbsp;02 style.

## Session 04 &mdash; Logistic Regression & Classification (practical series)

A four-notebook arc that builds logistic regression **from scratch in NumPy** &mdash; the model, the
cross-entropy loss and its gradient, how to *evaluate* a classifier, and the **softmax**
generalization to many classes &mdash; each step checked against scikit-learn, matching Lecture&nbsp;04.

| # | Notebook | Covers |
| --- | --- | --- |
| **04a** | [`session-04a-logistic-model.ipynb`](session-04a-logistic-model.ipynb) | The **sigmoid** (stable, with its properties); **odds/log-odds** and the odds ratio; the **decision boundary** as a point (1-D) then a line (2-D); the model as one **neuron**. |
| **04b** | [`session-04b-training-from-scratch.ipynb`](session-04b-training-from-scratch.ipynb) | **Why not squared error** (vanishing gradient, non-convex); **cross-entropy** = negative log-likelihood; the clean gradient $\tfrac1n\mathbf{X}^\top(\mathbf{p}-\mathbf{y})$ (numerically checked); **gradient descent** matching sklearn; **convexity**; **$L_2$**. |
| **04c** | [`session-04c-evaluation.ipynb`](session-04c-evaluation.ipynb) | **Confusion matrix**; **precision/recall/F1** from scratch; the **threshold** as a knob; **ROC/AUC** from scratch; why **accuracy lies** under class imbalance &mdash; on the breast-cancer dataset. |
| **04d** | [`session-04d-softmax-multiclass.ipynb`](session-04d-softmax-multiclass.ipynb) | **Softmax** (stable) and its cross-entropy/gradient from scratch; **binary = 2-class softmax**; multiclass **decision regions**; a real run on the **digits** dataset. |

**Suggested order:** 04a &rarr; 04b &rarr; 04c &rarr; 04d. Everything is NumPy-first and validated
against scikit-learn (and, where possible, against the coefficients that generated the data),
continuing the Session&nbsp;02/03 style.

## Session 05 &mdash; Generative & Probabilistic Models (practical series)

A three-notebook arc on **generative classifiers for continuous features**, built from scratch in
NumPy and checked against scikit-learn, matching Lecture&nbsp;05. All three fit in **closed form**
&mdash; no optimizer, no learning rate: a prior, a mean, and some form of covariance.

| # | Notebook | Covers |
| --- | --- | --- |
| **05a** | [`session-05a-gaussian-naive-bayes.ipynb`](session-05a-gaussian-naive-bayes.ipynb) | The generative recipe (prior $\times$ likelihood); the **1-D Gaussian** per feature; the **naive** independence assumption and its upright ellipses; a from-scratch `GaussianNBScratch` matching sklearn's `GaussianNB` exactly (`var_smoothing` included); the curved decision boundary; a run on **wine**. |
| **05b** | [`session-05b-lda-qda.ipynb`](session-05b-lda-qda.ipynb) | The **full covariance** (the tilt naive Bayes cannot have); fitting $\mu_k,\Sigma_k$ and the **pooled** $\Sigma$; the discriminant $\delta_k$; **QDA** (own $\Sigma_k$ &rarr; curved) and **LDA** (shared $\Sigma$ &rarr; straight, with $\mathbf{w}=\Sigma^{-1}(\mu_1-\mu_0)$), both matched to sklearn; the LDA posterior shown to be exactly a **sigmoid**; what the extra covariances cost. |
| **05c** | [`session-05c-comparing-generative-models.ipynb`](session-05c-comparing-generative-models.ipynb) | The three boundaries side by side; how much **data** each needs in both regimes (different vs shared class shapes); **wine** and **breast-cancer** benchmarks with a logistic-regression reference; why QDA needs regularization at $d{=}30$; and a practical rule for choosing. |

**Suggested order:** 05a &rarr; 05b &rarr; 05c. Everything is NumPy-first and validated against
scikit-learn (and, where possible, against the parameters that generated the data), continuing the
Session&nbsp;02/03/04 style.

## Sessions 06&ndash;07 &mdash; Support Vector Machines & Kernel Methods (practical series)

A three-notebook arc **shared by two lectures**, because the idea that ends Lecture&nbsp;06 &mdash; the
dual touches the data only through inner products &mdash; is the idea that starts Lecture&nbsp;07.
scikit-learn fits the larger models; the lectures' key claims are checked by hand in NumPy, and
the SVM dual is solved from scratch with SciPy's general optimizer on small problems (no QP library
needed).

| # | Notebook | Covers |
| --- | --- | --- |
| **06a** | [`session-06a-margin-and-soft-margin.ipynb`](session-06a-margin-and-soft-margin.ipynb) | Lecture&nbsp;06's three-point example checked by hand; **functional vs geometric** margin and the scale freedom; **only support vectors matter** (far points leave `coef_` and `intercept_` identical bit for bit); the **soft margin**, its three slack cases and the **$C$** sweep; the **hinge loss** as loss&nbsp;+&nbsp;penalty, with sklearn's answer verified to minimize it; **scaling** on breast cancer. |
| **06b** | [`session-06b-dual-and-kernel-trick.ipynb`](session-06b-dual-and-kernel-trick.ipynb) | The **dual solved from scratch** (SciPy SLSQP), recovering the lecture's $\alpha=(\tfrac12,\tfrac12,1)$ exactly; strong duality and $\sum_i\alpha_i=\lVert\mathbf{w}\rVert^2$; the three **KKT** cases; **sparsity** (refit on the support vectors only); rebuilding `decision_function` from `dual_coef_` for linear/RBF/polynomial kernels; an explicit **feature map** vs the **polynomial kernel**; the same solver turned into an **RBF SVM** by changing one line. |
| **07** | [`session-07-kernels-in-practice.ipynb`](session-07-kernels-in-practice.ipynb) | **Valid kernels**: symmetric PSD Gram matrices tested by their eigenvalues, and two impostors that fail; the Gram matrix as a similarity map; the **kernel zoo** and the RBF's infinite feature map, truncated; **$\gamma$** as the complexity dial and a $(C,\gamma)$ cross-validation grid; **kernel ridge** in closed form $(K+\lambda I)^{-1}\mathbf{y}$, matched to sklearn and shown to equal ordinary ridge for a linear kernel; an honest **diabetes** comparison. |

**Suggested order:** 06a &rarr; 06b &rarr; 07. Notebook 06b closes Session&nbsp;06 and opens
Session&nbsp;07. Keep the from-scratch dual solver to $n\le200$ points &mdash; its cost grows roughly
like $n^3$.

## Sessions 08&ndash;14

One notebook per lecture, named `session-NN-topic.ipynb`, each a hands-on companion to that
lecture (trees, ensembles, clustering, dimensionality reduction, neural networks, learning
theory).
