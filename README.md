# Machine Learning &mdash; practical session notebooks

Hands-on Jupyter companions to the lectures. The course is **practice-heavy**, so most sessions
have one or more notebooks here. Every notebook is **self-contained and offline**: it imports
two small sibling helpers and generates or uses only bundled data, so it runs top-to-bottom
with no downloads (a few clearly-marked *optional* cells fetch real data over the network).
The exception is the [`practical/`](practical/README.md) subfolder: its case studies work on real
datasets, which they download once and cache (see the last section).

## Running them

```sh
cd courses/machine-learning/notebooks
jupyter lab            # or: jupyter notebook
```

Requirements: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy` (all standard). Outputs
are committed **with the notebooks**, so every figure and table is visible on GitHub without
running anything; re-run the cells (Shift+Enter) to reproduce them. Cells marked
`# <-- CHANGE ME` are meant to be edited and re-run.

## Helper files (shared by the notebooks)

| File | What it provides |
| --- | --- |
| `nb_utils.py` | `use_style("notes"\|"slides")` &mdash; a clean, colorblind-safe matplotlib look matching the lecture figures. |
| `nb_data.py` | `make_apartments(messy=…)` / `basic_clean(…)` &mdash; the synthetic *apartments &rarr; rent* table used across Session&nbsp;01; `make_regression_1d`, `make_wave`, and `make_regression_md` (linear/nonlinear/multivariate targets with **known** coefficients) for Session&nbsp;02; `make_logistic_1d` / `make_logistic_2d` (two-class labels from a **known** logistic model) for Session&nbsp;04; and `make_gaussian_classes` (Gaussian classes with **known** parameters) for Session&nbsp;05; and `make_two_blobs`, `make_rings` (labels in $\{-1,+1\}$) for Sessions&nbsp;06&ndash;07. For Sessions&nbsp;08&ndash;13 it also carries the course's **running toy examples**, small enough to check with pencil and paper: `make_ten_points` (10 integer points &mdash; root Gini exactly $1/2$, best split $x_1\le3.5$ with gain exactly $3/14$) and `make_rent8` (8 apartments &mdash; mean $7.5$, SSE $138$, best split SSE $10$), both reused across 08/09/10/12; `make_xor` (the four corners); `make_ellipses3` (three tilted Gaussians, where round and elliptical models part company); `make_cigars` with `best_1d_threshold_accuracy` (PC1 keeps $99.24\%$ of the variance and classifies at chance, $0.5417$, where LDA reaches $0.9867$); and `make_uneven_clusters` (known cluster sizes and gaps, $24.8\times$ and $5.0\times$, for the t-SNE/UMAP warnings). |

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

## Sessions 08&ndash;13 &mdash; trees to deep networks (practical series)

Ten notebooks across six lectures. These are **scikit-learn-first**: the estimator is always
sklearn's, and NumPy is used only where writing the thing out is what teaches it &mdash; an
impurity, one EM iteration, one AdaBoost weight update, one backprop pass &mdash; and every
from-scratch piece is then checked against sklearn in the same cell.

Two rules run through all ten, and they are worth knowing before you read:

* **A number appears in the prose only if a cell computes it.** Where these notebooks quote the
  lecture, they recompute it.
* **No general claim rests on one seed or one train/test split.** Anything that would be a coin
  flip measured once is measured over 10&ndash;40 repeats and reported with its spread &mdash; and
  where the effect does not resolve at that sample size, the notebook says so instead of
  claiming it. (Session&nbsp;13's weight-decay section is the clearest example: the honest answer
  is *we cannot detect it here*.)

| # | Notebook | Covers |
| --- | --- | --- |
| **08** | [`session-08-decision-trees.ipynb`](session-08-decision-trees.ipynb) | A tree as rules and as boxes; **impurity as the cost of the best constant**; Gini/entropy/information gain by hand, matched to sklearn's own `tree_.impurity`; the $n-1$ candidate thresholds and why trees need no scaling; **greedy is short-sighted** &mdash; the XOR failure where every root gain is exactly $0$; size as the complexity knob; regression trees; **cost-complexity pruning** (the ten-point path by hand, then `ccp_alpha` by CV); MDI and what it does not mean; and the **instability** that motivates Session&nbsp;09. |
| **09a** | [`session-09a-bagging-and-random-forests.ipynb`](session-09a-bagging-and-random-forests.ipynb) | Condorcet's two conditions, counted; the bootstrap and its $63.2\%/36.8\%$ split, derived *and* measured; $\operatorname{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$ with $\sigma^2$ and $\rho$ **measured on real ensemble members**, so the **correlation floor** is visible; the stable model bagging does nothing for; **out-of-bag** error as a free validation set; random forests &mdash; paying for diversity by making each member worse; and **how MDI lies**, with permutation importance as the partial fix. |
| **09b** | [`session-09b-boosting.ipynb`](session-09b-boosting.ipynb) | Fitting the next model to the last one's mistakes; gradient boosting as descent in function space (SSE $138\to42\to18\to9.9$, below what any single stump reaches) matched to sklearn exactly; **AdaBoost's weights printed round by round**, landing on Session&nbsp;08's own three cuts; where the weight goes when a label is simply wrong; **$\nu$ and $M$ are one knob with two hands on it**; early stopping; **`max_depth` is the interaction order** (a sum of stumps is additive to machine precision, so XOR sits at chance); and an honest forest-vs-boosting bake-off. |
| **10a** | [`session-10a-kmeans-and-mixtures.ipynb`](session-10a-kmeans-and-mixtures.ipynb) | How you score a clustering at all (ARI, and why it is not an accuracy); **Lloyd's algorithm by hand** on Session&nbsp;08's eight rents &mdash; landing on WCSS $10$, the *same* $10$ as that lecture's best split; the identity $\mathrm{TSS}=\mathrm{WCSS}+\mathrm{BCSS}$; local minima and what `n_init` is for; the three ways to break k-means' assumptions; Gaussian mixtures and what each `covariance_type` costs; one **EM iteration by hand**; and **k-means as EM hardened** &mdash; the zero-variance limit, agreeing to the last digit. |
| **10b** | [`session-10b-density-hierarchy-and-spectral.ipynb`](session-10b-density-hierarchy-and-spectral.ipynb) | **DBSCAN** &mdash; core/border/noise counted by hand on the ten points, then `eps` from a k-distance plot; the shapes k-means cannot reach; why "no $K$" is half true; **hierarchical clustering** and the dendrogram; **the linkage *is* the model** (single linkage recovers the moons; Ward fails exactly as k-means does); **spectral clustering**, whose affinity *is* Session&nbsp;07's RBF Gram matrix &mdash; the Laplacian, the Fiedler vector, and the honest finding that **$\gamma$ is a window** that fails on both sides, with sklearn's default $\gamma=1$ no better than k-means; and choosing $K$ by elbow/silhouette/BIC **including where each one is wrong**. |
| **11a** | [`session-11a-pca-lda-and-factorizations.ipynb`](session-11a-pca-lda-and-factorizations.ipynb) | The linear methods, on one thread: **variance is not always the question you are asking**. The curse of dimensionality; all of PCA on four points; the max-variance and min-reconstruction views shown to be one answer; choosing $k$; standardizing changes the answer. Then the three breaks &mdash; **labels** (PC1 keeps $99.24\%$ of the variance and classifies at chance, $0.5417$, where LDA reaches $0.9867$; and LDA's two walls); **per-feature noise** (rescale one feature and PCA's direction swings where factor analysis's does not &mdash; and what the rotation costs); and **negativity** (why non-negativity turns a basis into parts). |
| **11b** | [`session-11b-tsne-and-umap.ipynb`](session-11b-tsne-and-umap.ipynb) | When a flat projection is not enough; **kernel PCA** &mdash; the same Gram matrix, a different job; **t-SNE** in depth: one knob and $n$ bandwidths, what perplexity controls, why the map kernel is Student-$t$ (crowding); **the warning that actually misleads people** &mdash; cluster sizes *and* between-cluster gaps both come back as $1$, measured; run-to-run variability and no `transform` for new points; then **UMAP**, implemented here in NumPy because `umap-learn` is not installed (the fitted $a=1.5769$, $b=0.8951$ against the package's published $1.577$, $0.895$), its two knobs, and which one can change your conclusions; and **ICA**, one relative that is not about pictures. |
| **12a** | [`session-12a-from-neuron-to-hidden-layer.ipynb`](session-12a-from-neuron-to-hidden-layer.ipynb) | **You already own a neuron** &mdash; it is Session&nbsp;04's logistic regression, agreeing to within one ulp. Then XOR and a wall you cannot train through, via the one-line **diagonal identity** $f(0,0)+f(1,1)=f(0,1)+f(1,0)$ and a $274{,}625$-neuron grid search that finds nothing; **two exact ways through** (OR/NAND/AND with step units, and an all-integer ReLU net); **what the hidden layer actually did** &mdash; it moves the points, not the boundary; the layer rule, the shapes, and counting parameters against a fitted `MLPClassifier`; and why the bend is not optional ($105$ parameters collapsing to $3$). |
| **12b** | [`session-12b-what-a-network-learns.ipynb`](session-12b-what-a-network-learns.ipynb) | **Universal approximation as a construction, not an incantation**: two sigmoid units make a bump, $H$ bumps take $2H$ units, and the error *halves when the width doubles*; the theorem stated properly &mdash; and **what it does not promise** (existence is not attainment); **depth against width** by folding a tent, $2L$ units giving $2^L$ pieces; activations and **the number $0.25$**; softmax, cross-entropy and the $\hat y-y$ cancellation; the **seed lottery** on a solution we can write down (standardizing the inputs buys more than width); **backpropagation by hand in integers**, checked against finite differences; and one step, handed off to Session&nbsp;13. |
| **13** | [`session-13-neural-networks-training.ipynb`](session-13-neural-networks-training.ipynb) | **It is all the condition number.** The learning rate on a problem you can do in your head (the threshold $2/\text{curvature}$, derived); conditioning; **momentum and the floor nobody mentions** &mdash; the exact $\sqrt\beta$ rate, so $\beta=0.9$ beats plain descent only when $\kappa>37.97$; **Adam, and why the bias correction is not bookkeeping**; where you start &mdash; the variance argument, and zero vs constant init as two *different* failures; standardizing the inputs; then regularization measured honestly over many splits &mdash; weight decay (**which does not resolve at this sample size, and the notebook says so**), early stopping, and dropout's weight-scaling rule as an approximation, not an identity. |

**Suggested order:** 08 &rarr; 09a &rarr; 09b &rarr; 10a &rarr; 10b &rarr; 11a &rarr; 11b &rarr;
12a &rarr; 12b &rarr; 13. The chain is deliberate: 08 ends on the instability that 09a cures,
09b's stumps cannot represent XOR and 12a proves why, 10a's k-means lands on the *same* number as
08's best split, 10b's spectral affinity is 07's Gram matrix, and 12b hands its gradient step
straight to 13.

## Session 14

[`session-14-learning-theory.ipynb`](session-14-learning-theory.ipynb) is still an **outline**
&mdash; section headings and short cells, committed without outputs, unlike the 33 notebooks
above.

## Practical notebooks &mdash; real data, real decisions

Case studies that are not tied to one lecture live in [`practical/`](practical/README.md). Each one
puts several methods side by side on a **real** dataset and goes through the decisions of a real
project: cleaning, encoding, honest evaluation, choosing a metric and a threshold, and deciding what
to deploy. They import `nb_utils.py` from this folder and download their data once from the UCI
repository (cached in `practical/data/raw/`, git-ignored, checked against SHA-256 fingerprints).

| # | Notebook | Covers |
| --- | --- | --- |
| **01** | [`practical/01-comparing-classifiers.ipynb`](practical/01-comparing-classifiers.ipynb) | **Logistic regression, naive Bayes, SVMs and a decision tree** on the UCI *Adult* census table (48,842 people), with no ensembles and no high-level scikit-learn tools: cleaning a real file, preprocessing and cross-validation written by hand, the likelihood that makes or breaks naive Bayes, one look at the test file with its uncertainty, thresholds, calibration and Platt scaling, double counting, learning curves that reverse the ranking, and a recommendation for a concrete use case. |
