---
layout: docs
title: "AI: Core ML Algorithms"
permalink: /docs/technology/ai/core-ml-algorithms.html
toc: true
toc_sticky: true
hide_title: true
---

[AI & Machine Learning](./) › Core ML Algorithms

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Core ML Algorithms</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The classical workhorses — regression, trees, boosting, SVMs, k-NN, and clustering — that still win most tabular problems.</p>
</div>

Before deep learning dominated perception and language, a compact toolbox of classical algorithms solved—and still solves—the majority of real-world prediction problems. On structured/tabular data they remain the default: they train in seconds, are easy to interpret, need little tuning, and routinely beat neural networks on the kind of spreadsheet-shaped data most organizations actually have. This page is a practical reference to that toolbox: linear and logistic regression, decision trees, random forests, gradient boosting (XGBoost and LightGBM), support vector machines, k-nearest neighbors, clustering, and the ensembling ideas that tie them together. Every section pairs the intuition and the math with runnable scikit-learn code.

> **When to reach for these over deep learning.** If your data fits in a dataframe with named columns (financial records, sensor readings, user features), start here. Tree ensembles in particular are the strongest baseline on tabular data and should be beaten before you justify a neural network. Reserve deep learning for images, audio, text, and other high-dimensional, weakly-structured signals — see [Neural Network Architectures](architectures.html).

## The Supervised Learning Setup

Every supervised algorithm below learns a function $f$ that maps inputs $\mathbf{x} \in \mathbb{R}^d$ to outputs $y$ from a training set of $n$ labeled examples $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$. We choose $f$ from some hypothesis class to minimize an average loss plus, usually, a penalty on complexity:

$$\hat{f} = \arg\min_{f}\; \frac{1}{n}\sum_{i=1}^{n} L\bigl(y_i,\, f(\mathbf{x}_i)\bigr) + \lambda\, \Omega(f)$$

The two tasks are **regression** (continuous $y$, typically squared-error loss) and **classification** (discrete $y$, typically log-loss / cross-entropy). The regularizer $\Omega$ and its strength $\lambda$ control the bias–variance tradeoff: too little and the model overfits, too much and it underfits. The art of applied ML is choosing the hypothesis class and tuning $\lambda$ by cross-validation. The algorithms differ mainly in the shape of $f$ they can represent and how they search for it.

## Linear Regression

Linear regression is the simplest useful model and the conceptual root of most others. It assumes the target is an affine function of the features plus noise:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = b + \sum_{j=1}^{d} w_j x_j$$

We fit the weights $\mathbf{w}$ and intercept $b$ by minimizing the **mean squared error** (equivalently, maximizing the likelihood under Gaussian noise):

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - \mathbf{w}^\top \mathbf{x}_i - b\bigr)^2$$

### The Closed-Form Solution

Stacking the data into a design matrix $X \in \mathbb{R}^{n \times d}$ (with a column of ones for the intercept) and targets $\mathbf{y}$, the loss is convex and has a unique minimum given by the **normal equations**:

$$\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

This is the orthogonal projection of $\mathbf{y}$ onto the column space of $X$. In practice libraries solve it via QR or SVD rather than inverting $X^\top X$ directly (which is numerically unstable when features are collinear).

### Regularization: Ridge, Lasso, Elastic Net

When features are correlated or $d$ is large relative to $n$, plain least squares overfits and the weights blow up. Adding a penalty shrinks them:

- **Ridge ($\ell_2$)** penalizes squared weight magnitude, which keeps all features but shrinks them smoothly. It has a closed form $\hat{\mathbf{w}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$ and is the go-to fix for multicollinearity.

$$L_{\text{ridge}} = \lVert \mathbf{y} - X\mathbf{w}\rVert_2^2 + \lambda\lVert \mathbf{w}\rVert_2^2$$

- **Lasso ($\ell_1$)** penalizes absolute weight magnitude, which drives some weights exactly to zero, performing automatic feature selection.

$$L_{\text{lasso}} = \lVert \mathbf{y} - X\mathbf{w}\rVert_2^2 + \lambda\lVert \mathbf{w}\rVert_1$$

- **Elastic Net** combines both, getting Lasso's sparsity while handling correlated feature groups more gracefully.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X = np.random.randn(500, 10)
y = X @ np.array([3, 0, -2, 0, 0, 1.5, 0, 0, 0, 0.0]) + 0.5 * np.random.randn(500)

# Always scale features before regularized linear models so the penalty is fair.
ols   = make_pipeline(StandardScaler(), LinearRegression())
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))

for name, model in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
    r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    print(f"{name:6s} CV R^2 = {r2:.3f}")

# Inspect which features Lasso zeroed out (automatic feature selection)
lasso.fit(X, y)
print("Lasso coefficients:", np.round(lasso[-1].coef_, 2))
```

**Practical notes.** Linear regression assumes a linear relationship, homoscedastic errors, and little multicollinearity. Standardize features before regularizing. Diagnose with residual plots; if residuals fan out or curve, transform the target (e.g. log) or add polynomial/interaction features. Coefficients are directly interpretable as "change in $y$ per unit change in $x_j$, holding others fixed."

## Logistic Regression

Despite the name, logistic regression is a **classification** algorithm. It models the probability of the positive class by squashing a linear score through the **sigmoid** (logistic) function:

$$p(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

The decision boundary $\mathbf{w}^\top \mathbf{x} + b = 0$ is a hyperplane, but the output is a calibrated probability between 0 and 1. We fit by minimizing the **binary cross-entropy** (negative log-likelihood), which is convex:

$$L(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}\Bigl[\,y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\,\Bigr]$$

There is no closed form, so we optimize with gradient-based methods. The gradient has a clean form—the same as linear regression, with the prediction passed through the sigmoid:

$$\nabla_{\mathbf{w}} L = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)\,\mathbf{x}_i$$

### Multiclass and Regularization

For $K > 2$ classes, **softmax (multinomial) regression** generalizes the sigmoid:

$$p(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x}}}$$

As with linear regression, $\ell_1$/$\ell_2$ penalties guard against overfitting; scikit-learn's `LogisticRegression` is $\ell_2$-regularized by default, controlled by `C` (inverse of $\lambda$).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

clf = make_pipeline(StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=1000))
clf.fit(Xtr, ytr)

proba = clf.predict_proba(Xte)[:, 1]
print(classification_report(yte, clf.predict(Xte)))
print(f"ROC-AUC: {roc_auc_score(yte, proba):.3f}")
```

**Why it endures.** Logistic regression is fast, well-calibrated, interpretable (a coefficient's exponential is an **odds ratio**), and a hard baseline to beat on linearly separable problems. It is the default first model for binary classification and the output layer of most neural network classifiers.

## Decision Trees

A decision tree partitions the feature space into axis-aligned rectangles by recursively asking yes/no questions ("Is $x_3 < 4.7$?"). Each internal node splits on one feature and threshold; each leaf predicts a constant (the mean target for regression, the majority class for classification). Trees are non-linear, handle mixed feature types, need no scaling, and are highly interpretable—you can read the rules straight off the tree.

### How Splits Are Chosen

At each node the algorithm greedily picks the feature and threshold that most reduce **impurity**. For classification, the two standard impurity measures over class proportions $p_k$ at a node are:

$$\text{Gini} = 1 - \sum_{k=1}^{K} p_k^2, \qquad \text{Entropy} = -\sum_{k=1}^{K} p_k \log_2 p_k$$

A split is scored by the weighted impurity of its children versus the parent—the **information gain**:

$$\Delta = I(\text{parent}) - \frac{n_L}{n}I(\text{left}) - \frac{n_R}{n}I(\text{right})$$

For regression, impurity is the variance (mean squared error) of the targets in the node, and a split is chosen to minimize the total child variance. Splitting continues until a stopping rule fires (max depth, minimum samples per leaf, or no impurity reduction).

### Controlling Overfitting

A fully grown tree memorizes the training set—it has near-zero bias but enormous variance. The cures are **pre-pruning** (cap `max_depth`, raise `min_samples_leaf`/`min_samples_split`) and **post-pruning** (cost-complexity pruning via `ccp_alpha`, which trims branches whose impurity reduction does not justify their added complexity).

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,            # pre-pruning: shallow trees generalize better
    min_samples_leaf=5,
    ccp_alpha=0.0,
    random_state=0,
)
tree.fit(X, y)
print("Train accuracy:", tree.score(X, y))
print("Feature importances:", tree.feature_importances_)

# A single shallow tree is fully human-readable:
plt.figure(figsize=(12, 6))
plot_tree(tree, filled=True)
plt.show()
```

**Strengths and the catch.** Trees are interpretable, invariant to monotone feature transforms, and capture interactions automatically. Their weakness is instability: a small change in the data can produce a very different tree (high variance). That single weakness is exactly what the next two methods—random forests and boosting—are built to fix by combining many trees.

## Ensemble Methods: The Big Idea

An **ensemble** combines many base models into one stronger predictor. Two complementary strategies dominate, and almost every top tabular model is one of them:

- **Bagging (bootstrap aggregating)** trains many high-variance models *independently* on bootstrap resamples of the data and averages them. Averaging uncorrelated errors reduces **variance** without raising bias. Random forests are the canonical example.
- **Boosting** trains models *sequentially*, each one correcting the residual errors of the ensemble so far. This drives down **bias**, building a strong learner from many weak ones. Gradient boosting (XGBoost, LightGBM) is the canonical example.

The decomposition that motivates both: expected test error splits into bias, variance, and irreducible noise:

$$\mathbb{E}\bigl[(y - \hat{f}(\mathbf{x}))^2\bigr] = \underbrace{\bigl(\text{Bias}[\hat{f}]\bigr)^2}_{\text{boosting attacks this}} + \underbrace{\text{Var}[\hat{f}]}_{\text{bagging attacks this}} + \sigma^2$$

Bagging averages independent learners to shrink variance; boosting adds dependent learners to shrink bias. A third strategy, **stacking**, trains a meta-model on the out-of-fold predictions of diverse base models and is covered at the end.

## Random Forests

A random forest is bagging applied to decision trees, with one extra trick. It grows $B$ deep trees, each on a bootstrap sample of the rows, and—crucially—at every split considers only a **random subset of features** (typically $\sqrt{d}$ for classification, $d/3$ for regression). This feature subsampling *decorrelates* the trees so that averaging actually cancels their errors. The final prediction averages the trees (regression) or takes a majority vote (classification):

$$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})$$

### Why Decorrelation Matters

If the $B$ trees each have variance $\sigma^2$ and pairwise correlation $\rho$, the variance of their average is:

$$\rho\,\sigma^2 + \frac{1 - \rho}{B}\sigma^2$$

Adding trees ($B \to \infty$) drives the second term to zero, but the first term $\rho\sigma^2$ remains. Random feature selection lowers $\rho$, which is what makes the forest's variance reduction so effective. This is the whole reason random forests beat plain bagged trees.

### Out-of-Bag Estimation and Importances

Each bootstrap sample omits roughly 37% of the rows; those **out-of-bag (OOB)** samples give a free cross-validation-like estimate of generalization error with no held-out set. Forests also report **feature importances** (mean impurity decrease, or the more reliable **permutation importance**), useful for understanding which features drive predictions.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",    # the decorrelation knob
    oob_score=True,         # free generalization estimate
    n_jobs=-1,
    random_state=0,
)
rf.fit(Xtr, ytr)
print(f"OOB accuracy:  {rf.oob_score_:.3f}")
print(f"Test accuracy: {rf.score(Xte, yte):.3f}")

# Permutation importance is more trustworthy than impurity importance,
# especially when features differ in cardinality or scale.
perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0, n_jobs=-1)
top = perm.importances_mean.argsort()[::-1][:5]
for i in top:
    print(f"feature {i}: {perm.importances_mean[i]:.3f} +/- {perm.importances_std[i]:.3f}")
```

**When to use.** Random forests are the great all-rounder: robust to outliers and irrelevant features, almost impossible to overfit by adding trees, parallelizable, and good with default settings. They are the ideal *first* nontrivial model on any tabular dataset. The main knobs are `n_estimators` (more is better but slower), `max_features` (the decorrelation control), and `max_depth`/`min_samples_leaf` to limit individual tree complexity.

## Gradient Boosting

Gradient boosting builds an additive model one tree at a time, where each new tree is fit to the **negative gradient** of the loss with respect to the current predictions—i.e., it corrects the ensemble's mistakes. Starting from a constant $F_0$, at stage $m$:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\, h_m(\mathbf{x})$$

where $h_m$ is a shallow regression tree fit to the **pseudo-residuals** $r_{im} = -\,\partial L(y_i, F(\mathbf{x}_i)) / \partial F(\mathbf{x}_i)$ evaluated at $F_{m-1}$, and $\nu \in (0, 1]$ is the **learning rate** (shrinkage) that scales each tree's contribution. For squared-error loss the pseudo-residuals are just the ordinary residuals $y_i - F_{m-1}(\mathbf{x}_i)$, which is why boosting is described as "fitting the next tree to the leftover error."

Because each tree depends on the previous ones, boosting reduces **bias** and can model very subtle structure. The price is that it can overfit if run too long, so the key regularizers are a small learning rate combined with many trees, shallow trees (depth 3–8), subsampling of rows and columns, and $\ell_1$/$\ell_2$ leaf penalties.

### XGBoost

XGBoost ("eXtreme Gradient Boosting") is a highly optimized implementation that made boosting the dominant approach for tabular competitions. Its key refinements over textbook gradient boosting:

- A **regularized objective** using a second-order (Newton) approximation of the loss, so both the gradient $g_i$ and Hessian $h_i$ inform each split, plus an explicit complexity penalty on the trees:

$$\mathcal{L} = \sum_i L(y_i, \hat{y}_i) + \sum_m \Omega(h_m), \qquad \Omega(h) = \gamma\, T + \tfrac{1}{2}\lambda \lVert \mathbf{w}\rVert^2$$

where $T$ is the number of leaves and $\mathbf{w}$ the leaf weights.
- **Sparsity-aware split finding** with a default direction for missing values (it handles NaNs natively).
- Built-in **column and row subsampling**, parallelized split finding, and out-of-core training for large data.

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,      # small rate + many trees = the canonical recipe
    max_depth=4,
    subsample=0.8,           # row subsampling
    colsample_bytree=0.8,    # column subsampling
    reg_lambda=1.0,          # L2 on leaf weights
    eval_metric="logloss",
    early_stopping_rounds=30,
    n_jobs=-1,
    random_state=0,
)
model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
print("Best iteration:", model.best_iteration)
print("Test accuracy:", model.score(Xte, yte))
```

### LightGBM

LightGBM is Microsoft's gradient-boosting framework, engineered for speed and scale on large datasets. Its distinctive choices:

- **Leaf-wise (best-first) growth** instead of level-wise: it splits the leaf with the largest loss reduction, producing deeper, more accurate trees per node-count (control with `num_leaves` and `max_depth` to avoid overfitting).
- **Histogram-based binning** of continuous features into discrete buckets, which slashes the cost of split finding and memory use.
- **GOSS** (Gradient-based One-Side Sampling) keeps high-gradient examples and subsamples low-gradient ones, and **EFB** (Exclusive Feature Bundling) packs sparse, mutually-exclusive features together. Together these make it dramatically faster than XGBoost on wide or large data, often with comparable accuracy.

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,           # the main capacity knob for leaf-wise growth
    max_depth=-1,            # unlimited depth, capped by num_leaves
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=0,
)
model.fit(
    Xtr, ytr,
    eval_set=[(Xte, yte)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
)
print("Test accuracy:", model.score(Xte, yte))
```

**XGBoost vs. LightGBM vs. CatBoost.** All three are excellent; differences are mostly engineering. LightGBM is usually fastest and shines on large datasets but its leaf-wise growth can overfit small ones (cap `num_leaves`). XGBoost is the robust, battle-tested default. CatBoost handles categorical features natively with ordered boosting and is strong out of the box when you have many categoricals. For most tabular problems a tuned gradient-boosting model is the single best-performing classical method—it should be your target to beat.

## Support Vector Machines

A support vector machine finds the hyperplane that separates two classes with the **maximum margin**—the largest possible gap between the boundary and the nearest points of either class. Maximizing the margin is a form of regularization that yields good generalization. The classifier is $\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$, and the **hard-margin** problem is:

$$\min_{\mathbf{w}, b}\; \tfrac{1}{2}\lVert \mathbf{w}\rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \;\;\forall i$$

The margin width equals $2 / \lVert \mathbf{w}\rVert$, so minimizing $\lVert \mathbf{w}\rVert$ maximizes the margin. Only the points lying on the margin—the **support vectors**—determine the solution; all others are irrelevant, which is why SVMs are memory-efficient at prediction time.

### Soft Margin and the C Parameter

Real data is rarely separable, so the **soft-margin** SVM introduces slack variables $\xi_i \ge 0$ allowing some violations, penalized by $C$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}}\; \tfrac{1}{2}\lVert \mathbf{w}\rVert^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.}\quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0$$

Large $C$ tolerates few violations (low bias, high variance); small $C$ permits a wider, softer margin (more regularization). Equivalently, the SVM minimizes **hinge loss** plus an $\ell_2$ penalty: $\sum_i \max(0, 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) + \tfrac{1}{2C}\lVert\mathbf{w}\rVert^2$.

### The Kernel Trick

SVMs become powerful nonlinear classifiers through the **kernel trick**. The dual formulation depends on the data only through inner products $\mathbf{x}_i^\top \mathbf{x}_j$, which we replace with a kernel $k(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$ that computes inner products in a high-dimensional feature space *without ever forming $\phi$ explicitly*. The decision function becomes:

$$f(\mathbf{x}) = \text{sign}\!\left(\sum_{i \in \text{SV}} \alpha_i\, y_i\, k(\mathbf{x}_i, \mathbf{x}) + b\right)$$

Common kernels:

- **Linear**: $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$ — fast, for high-dimensional sparse data (e.g. text).
- **RBF (Gaussian)**: $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\lVert \mathbf{x} - \mathbf{x}'\rVert^2)$ — the flexible default; $\gamma$ sets the reach of each point.
- **Polynomial**: $k(\mathbf{x}, \mathbf{x}') = (\gamma\,\mathbf{x}^\top \mathbf{x}' + r)^p$ — captures fixed-degree feature interactions.

For the geometric intuition behind why mapping to higher dimensions makes data separable, see the kernel-trick discussion on the [Neural Network Architectures](architectures.html) page.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# SVMs are scale-sensitive: ALWAYS standardize features first.
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

grid = GridSearchCV(
    svm,
    param_grid={"svc__C": [0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.01, 0.1, 1.0]},
    cv=5, n_jobs=-1,
)
grid.fit(Xtr, ytr)
print("Best params:", grid.best_params_)
print("Test accuracy:", grid.best_estimator_.score(Xte, yte))
```

**When to use.** SVMs excel on small-to-medium datasets with clear margins and many features (text classification with a linear kernel is a classic win). Their weakness is scaling: kernel SVMs are roughly $O(n^2)$–$O(n^3)$ in training and impractical past ~100k samples, where boosting or linear models take over. Always standardize features and tune $C$ and $\gamma$ together.

## k-Nearest Neighbors

k-NN is the simplest learning algorithm: it does no training at all. To predict a new point, it finds the $k$ closest training points by some distance metric and returns their majority class (classification) or mean target (regression):

$$\hat{y}(\mathbf{x}) = \frac{1}{k}\sum_{i \in N_k(\mathbf{x})} y_i$$

where $N_k(\mathbf{x})$ is the set of the $k$ nearest neighbors, usually by Euclidean distance $\lVert \mathbf{x} - \mathbf{x}_i\rVert_2$. It is a **non-parametric, instance-based** method—the entire training set *is* the model.

### The Critical Choices

- **$k$** controls the bias–variance tradeoff: $k = 1$ fits every point (zero bias, high variance, jagged boundary); large $k$ smooths the boundary (higher bias, lower variance). Tune by cross-validation; odd $k$ avoids ties in binary problems.
- **Distance metric and scaling.** k-NN is acutely sensitive to feature scale—a feature with a large range dominates the distance. **Standardize features.** Distance-weighted voting (`weights="distance"`) lets nearer neighbors count more.
- **The curse of dimensionality.** In high dimensions all points become roughly equidistant and "nearest" loses meaning, so k-NN degrades badly past a few dozen features unless you reduce dimensionality first (e.g. PCA).

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

best_k, best_score = None, -1
for k in [1, 3, 5, 9, 15, 25]:
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k, weights="distance"))
    score = cross_val_score(knn, Xtr, ytr, cv=5).mean()
    print(f"k={k:2d}  CV accuracy={score:.3f}")
    if score > best_score:
        best_k, best_score = k, score

print("Best k:", best_k)
```

**When to use.** k-NN is a fine, interpretable baseline on low-dimensional data and underlies recommendation and anomaly-detection systems. Its costs are at prediction time—it must search all training points (mitigated by KD-trees or ball-trees for low dimensions, or approximate-nearest-neighbor indices like HNSW/FAISS at scale)—and its sensitivity to scaling and dimensionality.

## Clustering (Unsupervised)

Clustering groups unlabeled data into clusters of similar points. Unlike everything above, there is no target $y$—the goal is to discover structure. The three most useful algorithms span different assumptions about cluster shape.

### k-Means

k-Means partitions data into $K$ clusters by alternating two steps until convergence (Lloyd's algorithm): assign each point to its nearest centroid, then move each centroid to the mean of its assigned points. It minimizes the within-cluster sum of squares (inertia):

$$J = \sum_{j=1}^{K}\sum_{\mathbf{x} \in C_j} \lVert \mathbf{x} - \boldsymbol{\mu}_j\rVert^2$$

It is fast and scalable but assumes roughly spherical, equally-sized clusters, requires choosing $K$ in advance, and is sensitive to initialization (use **k-means++** seeding, the default). Choose $K$ with the **elbow method** (inertia vs. $K$) or the **silhouette score**.

### Hierarchical (Agglomerative) Clustering

Agglomerative clustering starts with each point as its own cluster and repeatedly merges the two closest clusters, producing a **dendrogram** you can cut at any level to get any number of clusters. The **linkage** criterion (single, complete, average, or Ward) defines "closest" and shapes the result. It needs no preset $K$ and reveals nested structure, but is $O(n^2)$ in memory, limiting it to smaller datasets.

### DBSCAN

DBSCAN (Density-Based Spatial Clustering) groups points that are densely packed (at least `min_samples` neighbors within radius `eps`) and labels sparse points as **noise/outliers**. Unlike k-Means it finds arbitrarily-shaped clusters, does not require choosing the number of clusters, and is robust to outliers—but it struggles when clusters have very different densities and is sensitive to `eps`.

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score

Xb, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=0)
Xb = StandardScaler().fit_transform(Xb)

# k-Means: pick K by silhouette
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(Xb)
    print(f"k={k}  inertia={km.inertia_:.0f}  silhouette={silhouette_score(Xb, km.labels_):.3f}")

# DBSCAN excels on non-convex clusters where k-Means fails
Xm, _ = make_moons(n_samples=600, noise=0.06, random_state=0)
db = DBSCAN(eps=0.2, min_samples=5).fit(Xm)
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"DBSCAN found {n_clusters} clusters and "
      f"{(db.labels_ == -1).sum()} noise points")

# Hierarchical clustering with Ward linkage
agg = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(Xb)
```

**Choosing a clusterer.** Use k-Means for speed and roughly spherical clusters when you know $K$; DBSCAN for arbitrary shapes, outlier rejection, and unknown cluster count; hierarchical clustering when you want a full dendrogram of nested groupings. Always scale features first, and validate clusters with the silhouette score or domain knowledge—clustering has no ground truth, so results need a sanity check.

## Stacking and Combining Models

Beyond bagging and boosting, **stacking** (stacked generalization) combines *different kinds* of models. You train several diverse base learners (say, a random forest, a gradient-boosting model, an SVM, and logistic regression), generate their **out-of-fold predictions**, and train a simple **meta-model** (often logistic/linear regression) on those predictions to learn how best to weigh each base learner. Using out-of-fold predictions is essential—training the meta-model on in-sample predictions leaks information and overfits.

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
        ("svm", make_pipeline(StandardScaler(), SVC(probability=True))),
        ("lr",  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ],
    final_estimator=LogisticRegression(),
    cv=5,            # out-of-fold predictions feed the meta-model
    n_jobs=-1,
)
stack.fit(Xtr, ytr)
print("Stacked ensemble test accuracy:", stack.score(Xte, yte))
```

A lighter alternative is a **voting ensemble** (hard voting by majority, or soft voting by averaging predicted probabilities), which needs no meta-model. Ensembles help most when the base models are individually strong *and* make uncorrelated errors—combining near-identical models gains little.

## Choosing an Algorithm

A practical decision guide for the supervised methods on this page:

| Algorithm | Best for | Strengths | Watch out for |
|-----------|----------|-----------|---------------|
| Linear / Logistic Regression | Linear relationships, baselines | Fast, interpretable, calibrated | Underfits nonlinear data |
| Decision Tree | Interpretable rules | No scaling, handles interactions | High variance alone — prone to overfit |
| Random Forest | Strong general-purpose baseline | Robust, low-tuning, parallel | Larger models, less interpretable |
| Gradient Boosting (XGBoost/LightGBM) | Top accuracy on tabular data | Usually the best classical performer | Needs tuning; can overfit if run too long |
| SVM | Small–medium data, clear margins | Effective in high dimensions | Scales poorly past ~100k rows; must standardize |
| k-NN | Low-dimensional, simple baselines | No training, intuitive | Slow prediction; curse of dimensionality |

**A sensible workflow.** Establish a baseline with logistic/linear regression, then a random forest. If you need more accuracy, tune a gradient-boosting model (XGBoost or LightGBM)—on tabular data this is usually the winner. Reach for SVMs on small high-dimensional problems and k-NN as a quick sanity baseline. For unlabeled data, start with k-Means and switch to DBSCAN for non-spherical clusters. Always validate with proper cross-validation, scale features for the distance- and margin-based methods (SVM, k-NN, clustering, regularized linear models), and only graduate to deep learning when the data is genuinely high-dimensional and unstructured.

---

## See Also

- [Neural Network Architectures](architectures.html) — the learning-theory foundations (bias–variance, kernels, optimization) and the deep models that extend these ideas
- [Generative Models](generative-models.html) — diffusion, GANs, VAEs, and LLM generation built on neural architectures
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling laws and interpretability of large models
- [AI Deep Dive (Lecture)](../ai-lecture-2023.html) — transformers and LLM internals in depth
- [AI Mathematics](../../advanced/ai-mathematics/) — formal proofs for statistical learning theory and kernel methods
