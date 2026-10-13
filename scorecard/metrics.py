"""Discrimination, calibration and stability metrics used in credit risk."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y_true, score) -> float:
    """Gini = 2 * AUC - 1 (the usual way banks report rank-ordering power)."""
    return 2 * roc_auc_score(y_true, score) - 1


def ks(y_true, score) -> float:
    """Kolmogorov-Smirnov: max distance between the cumulative score distributions of bads and goods."""
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(score)}).sort_values("s")
    cum_bad = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    cum_good = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float((cum_bad - cum_good).abs().max())


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score distributions.
    Rule of thumb: < 0.10 stable, 0.10-0.25 monitor, > 0.25 investigate / recalibrate."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y_true, pd_hat, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by predicted-PD decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(pd_hat)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    t = df.groupby("decile", observed=True).agg(n=("y", "size"), predicted_pd=("p", "mean"), observed_dr=("y", "mean"))
    return t.round(4)
