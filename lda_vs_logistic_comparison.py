"""Comparison of Linear Discriminant Analysis (LDA) and Binary Logistic Regression.
Works in Google Colab or locally:  python lda_vs_logistic_comparison.py
Uses the UCI German Credit data (credit default). If it cannot be downloaded,
it falls back to scikit-learn's built-in breast cancer data so the script still runs."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line in Colab or Jupyter to show plots inline
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% 1. Load data (positive class = 1 = default / bad)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
try:
    d = pd.read_csv(URL, sep=r"\s+", header=None)
    X, y = d.iloc[:, :-1].values, (d.iloc[:, -1].values == 2).astype(int)
    NAME = "UCI German Credit"
except Exception:
    from sklearn.datasets import load_breast_cancer
    b = load_breast_cancer()
    X, y, NAME = b.data, 1 - b.target, "Breast cancer (fallback)"
print(f"Dataset: {NAME} | rows = {len(y)} | features = {X.shape[1]} | positive rate = {y.mean():.2f}\n")

# %% 2. Models (same scaling for both so the comparison is fair)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd")),
}

# %% 3. 5-fold cross-validation
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for n, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "roc_auc", "f1", "recall"])
    rows.append({"Model": n, "Accuracy": r["test_accuracy"].mean(), "AUC": r["test_roc_auc"].mean(),
                 "F1": r["test_f1"].mean(), "Recall": r["test_recall"].mean(), "AUC std": r["test_roc_auc"].std()})
print("5-fold cross-validation\n", pd.DataFrame(rows).round(3).to_string(index=False), "\n")

# %% 4. Hold-out test (80/20)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
prob, out = {}, []
for n, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    fpr, tpr, _ = roc_curve(yte, p)
    prob[n] = p
    out.append({"Model": n, "Accuracy": accuracy_score(yte, pred), "AUC": roc_auc_score(yte, p),
                "KS": np.max(tpr - fpr), "Sensitivity": recall_score(yte, pred), "Specificity": tn / (tn + fp),
                "F1": f1_score(yte, pred), "Brier": brier_score_loss(yte, p)})
print("Hold-out test set\n", pd.DataFrame(out).round(3).to_string(index=False), "\n")

# %% 5. Do the two models agree?
lr_c = models["Logistic Regression"][-1].coef_[0]
lda_c = models["LDA"][-1].coef_[0]
agree = np.mean((prob["Logistic Regression"] >= .5) == (prob["LDA"] >= .5))
print(f"Coefficient correlation (LR vs LDA): {np.corrcoef(lr_c, lda_c)[0, 1]:.3f}")
print(f"Same decision on {agree * 100:.1f}% of test cases\n")

# %% 6. Check LDA assumptions
Z = StandardScaler().fit_transform(X)
shapiro_p = [min(stats.shapiro(Z[y == k, j])[1] for k in (0, 1)) for j in range(Z.shape[1])]
print(f"Normality: {np.mean(np.array(shapiro_p) > 0.05) * 100:.0f}% of features pass Shapiro-Wilk (p > 0.05) in both classes")
ld = [np.linalg.slogdet(np.cov(Z[y == k].T))[1] for k in (0, 1)]
print(f"Equal covariance: log-determinants of class covariances = {ld[0]:.1f} vs {ld[1]:.1f} "
      f"(a big gap means LDA's shared-covariance assumption is violated)\n")

# %% 7. Plots: ROC, calibration, coefficients
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for n, p in prob.items():
    f, t, _ = roc_curve(yte, p)
    ax[0].plot(f, t, label=f"{n} (AUC {roc_auc_score(yte, p):.3f})")
    fy, fx = calibration_curve(yte, p, n_bins=6, strategy="quantile")
    ax[1].plot(fx, fy, "o-", label=n)
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set(title="Calibration", xlabel="Predicted probability", ylabel="Observed rate")
ax[2].scatter(lr_c, lda_c); lim = max(abs(lr_c).max(), abs(lda_c).max()) * 1.1
ax[2].plot([-lim, lim], [-lim, lim], "k--"); ax[2].set(title="Standardised coefficients", xlabel="Logistic", ylabel="LDA")
ax[0].legend(); ax[1].legend(); plt.tight_layout(); plt.savefig("lda_vs_logistic.png", dpi=130)
print("Saved plot: lda_vs_logistic.png")

# %% 8. Summary
best = max(out, key=lambda r: r["AUC"])["Model"]
print(f"\nHigher test AUC: {best}. Logistic Regression needs no normality assumption and gives calibrated "
      "probabilities (compare the Brier scores); LDA is efficient when its assumptions hold.")
