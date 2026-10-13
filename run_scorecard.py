"""Build, validate and document the credit card default scorecard.

Usage: python run_scorecard.py   (expects data/raw/ from scripts/download_data.py)
Writes everything to reports/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from scorecard import binning, data, metrics, scaling

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "default of credit card clients.xls"
OUT = ROOT / "reports"
SEED = 42
IV_MIN = 0.02          # below this a feature carries almost no information
CORR_MAX = 0.70        # drop the weaker of two features whose WoE values are this correlated
GROUPS = {
    "PAY_0": [[-2], [-1], [0], [1], [2], [3, 4, 5, 6, 7, 8]],  # last month's repayment status
}
# Business assumptions for the cut-off (see README): approving a customer who repays earns 1 unit,
# approving one who defaults loses LOSS_RATIO units. Sensitivity is reported for other ratios.
LOSS_RATIO = 5
SENSITIVITY = [3, 5, 8]


def fit_bins(train: pd.DataFrame) -> dict:
    y = train[data.TARGET]
    out = {}
    for f in data.CANDIDATE_FEATURES:
        if f in GROUPS:
            out[f] = binning.fit_categorical(train[f], y, f, GROUPS[f])
        else:
            out[f] = binning.fit_numeric(train[f], y, f)
    return out


def select_features(bins: dict, woe_train: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    iv = pd.DataFrame({"feature": list(bins), "iv": [b.iv for b in bins.values()],
                       "n_bins": [len(b.table) for b in bins.values()]}).sort_values("iv", ascending=False)
    iv["decision"] = np.where(iv["iv"] < IV_MIN, "dropped: IV < 0.02", "kept")
    kept = []
    for f in iv.loc[iv["iv"] >= IV_MIN, "feature"]:
        clash = [k for k in kept if abs(woe_train[f].corr(woe_train[k])) > CORR_MAX]
        if clash:
            iv.loc[iv["feature"] == f, "decision"] = f"dropped: WoE corr > {CORR_MAX} with {clash[0]}"
        else:
            kept.append(f)
    return kept, iv.round(4)


def fit_logit(X: pd.DataFrame, y: pd.Series, features: list, iv: pd.DataFrame) -> tuple[LogisticRegression, list]:
    """With WoE = ln(good/bad), every coefficient for P(default) must be negative. A positive one
    means multicollinearity flipped a sign; drop that feature and refit so points stay intuitive."""
    feats = list(features)
    while True:
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X[feats], y)
        wrong = [f for f, b in zip(feats, model.coef_[0]) if b > 0]
        if not wrong:
            return model, feats
        f = wrong[0]
        iv.loc[iv["feature"] == f, "decision"] = "dropped: coefficient sign flipped"
        feats.remove(f)


def cutoff_analysis(score: np.ndarray, y: np.ndarray, loss_ratio: float) -> pd.DataFrame:
    rows = []
    for cut in np.arange(np.floor(score.min()), np.ceil(score.max()) + 1, 1):
        approved = score >= cut
        goods = int(((y == 0) & approved).sum())
        bads = int(((y == 1) & approved).sum())
        rows.append({"cutoff": int(cut), "approval_rate": approved.mean(),
                     "bad_rate_approved": bads / max(approved.sum(), 1), "profit": goods - loss_ratio * bads})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = data.clean(data.load_raw(RAW))
    train, test = train_test_split(df, test_size=0.3, stratify=df[data.TARGET], random_state=SEED)
    y_tr, y_te = train[data.TARGET], test[data.TARGET]

    bins = fit_bins(train)
    woe_tr = pd.DataFrame({f: b.woe(train[f]) for f, b in bins.items()})
    woe_te = pd.DataFrame({f: b.woe(test[f]) for f, b in bins.items()})
    features, iv = select_features(bins, woe_tr)
    model, features = fit_logit(woe_tr, y_tr, features, iv)

    pd_tr = model.predict_proba(woe_tr[features])[:, 1]
    pd_te = model.predict_proba(woe_te[features])[:, 1]
    logit_tr = model.decision_function(woe_tr[features])
    logit_te = model.decision_function(woe_te[features])
    score_tr, score_te = scaling.score_from_logit(logit_tr), scaling.score_from_logit(logit_te)

    cv_auc = cross_val_score(LogisticRegression(C=1.0, max_iter=1000), woe_tr[features], y_tr,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")

    # Benchmark: a gradient-boosted model on the same non-protected raw inputs
    raw_cols = [c for c in data.CANDIDATE_FEATURES]
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(train[raw_cols], y_tr)
    gbm_auc = roc_auc_score(y_te, gbm.predict_proba(test[raw_cols])[:, 1])

    # Cut-off by expected profit on the test set
    cuts = {r: cutoff_analysis(score_te, y_te.to_numpy(), r) for r in SENSITIVITY}
    best = {r: cuts[r].loc[cuts[r]["profit"].idxmax()] for r in SENSITIVITY}
    chosen = best[LOSS_RATIO]
    approve_all_profit = int((y_te == 0).sum() - LOSS_RATIO * (y_te == 1).sum())

    # Fairness check (SEX not used by the model): approval and default rates at the chosen cut-off
    approved = score_te >= chosen["cutoff"]
    fair = (pd.DataFrame({"sex": test["SEX"].map({1: "male", 2: "female"}), "approved": approved, "y": y_te})
            .groupby("sex").agg(applicants=("y", "size"), approval_rate=("approved", "mean"),
                                observed_default_rate=("y", "mean")))
    fair["default_rate_if_approved"] = [y_te[approved & (test["SEX"].map({1: "male", 2: "female"}) == s)].mean()
                                        for s in fair.index]
    di_ratio = fair["approval_rate"].min() / fair["approval_rate"].max()

    results = {
        "rows": {"train": len(train), "test": len(test)},
        "default_rate": {"train": round(y_tr.mean(), 4), "test": round(y_te.mean(), 4)},
        "features_in_model": features,
        "discrimination": {
            "auc_train": round(roc_auc_score(y_tr, pd_tr), 4), "auc_test": round(roc_auc_score(y_te, pd_te), 4),
            "gini_train": round(metrics.gini(y_tr, pd_tr), 4), "gini_test": round(metrics.gini(y_te, pd_te), 4),
            "ks_train": round(metrics.ks(y_tr, pd_tr), 4), "ks_test": round(metrics.ks(y_te, pd_te), 4),
            "cv_auc_mean": round(cv_auc.mean(), 4), "cv_auc_std": round(cv_auc.std(), 4),
            "benchmark_gbm_auc_test": round(gbm_auc, 4),
        },
        "calibration": {"brier_test": round(brier_score_loss(y_te, pd_te), 4),
                        "mean_pd_test": round(pd_te.mean(), 4), "observed_dr_test": round(y_te.mean(), 4)},
        "stability": {"psi_train_vs_test": round(metrics.psi(score_tr, score_te), 4)},
        "cutoff": {
            "loss_ratio": LOSS_RATIO, "cutoff_score": int(chosen["cutoff"]),
            "approval_rate": round(chosen["approval_rate"], 4),
            "bad_rate_approved": round(chosen["bad_rate_approved"], 4),
            "profit_units": int(chosen["profit"]), "approve_all_profit_units": approve_all_profit,
            "sensitivity": {str(r): {"cutoff": int(best[r]["cutoff"]), "approval_rate": round(best[r]["approval_rate"], 4),
                                     "bad_rate_approved": round(best[r]["bad_rate_approved"], 4)} for r in SENSITIVITY},
        },
        "fairness": {"disparate_impact_ratio": round(di_ratio, 4)},
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    iv.to_csv(OUT / "iv_table.csv", index=False)
    coefs = dict(zip(features, model.coef_[0]))
    scaling.points_table(bins, coefs, model.intercept_[0]).to_csv(OUT / "scorecard_points.csv", index=False)
    metrics.calibration_table(y_te, pd_te).to_csv(OUT / "calibration_test.csv")
    fair.round(4).to_csv(OUT / "fairness_by_sex.csv")
    cuts[LOSS_RATIO].round(4).to_csv(OUT / "cutoff_analysis.csv", index=False)

    # Charts
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lbl, yy, pp in [("train", y_tr, pd_tr), ("test", y_te, pd_te)]:
        fpr, tpr, _ = roc_curve(yy, pp)
        ax[0].plot(fpr, tpr, label=f"{lbl} AUC {roc_auc_score(yy, pp):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend()
    ax[1].hist(score_te[y_te == 0], bins=40, alpha=0.6, label="repaid", density=True)
    ax[1].hist(score_te[y_te == 1], bins=40, alpha=0.6, label="defaulted", density=True)
    ax[1].axvline(chosen["cutoff"], color="k", ls="--", lw=1, label=f"cut-off {int(chosen['cutoff'])}")
    ax[1].set(title="Score distribution (test)", xlabel="Score"); ax[1].legend()
    for r in SENSITIVITY:
        ax[2].plot(cuts[r]["cutoff"], cuts[r]["profit"] / abs(cuts[r]["profit"].max()), label=f"bad costs {r}x a good")
    ax[2].axvline(chosen["cutoff"], color="k", ls="--", lw=1)
    ax[2].set(title="Expected profit by cut-off (scaled)", xlabel="Cut-off score", ylim=(-1, 1.1)); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT / "model_charts.png", dpi=130)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
