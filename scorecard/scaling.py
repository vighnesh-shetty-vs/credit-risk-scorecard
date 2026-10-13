"""Turn a WoE logistic regression into an additive points scorecard.

Convention used by many banks: a score of BASE_SCORE means good:bad odds of BASE_ODDS,
and every PDO points ("points to double the odds") doubles those odds.
    factor = PDO / ln(2),  offset = BASE_SCORE - factor * ln(BASE_ODDS)
    score  = offset + factor * ln(odds_good)   with ln(odds_good) = -(intercept + sum(beta_j * woe_j))
"""
import numpy as np
import pandas as pd

BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20


def factor_offset(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    factor = pdo / np.log(2)
    return factor, base_score - factor * np.log(base_odds)


def points_table(bins: dict, coefs: dict, intercept: float) -> pd.DataFrame:
    """Points per bin. The intercept is spread evenly across features so every row is readable.
    Our model predicts P(default), so ln(odds_good) = -logit and points get a minus sign."""
    factor, offset = factor_offset()
    n = len(coefs)
    rows = []
    for name, beta in coefs.items():
        t = bins[name].table
        for b, row in t.iterrows():
            pts = -(beta * row["woe"] + intercept / n) * factor + offset / n
            rows.append({"feature": name, "bin": row.get("range", b), "n": int(row["n"]),
                         "bad_rate": round(row["bad_rate"], 4), "woe": round(row["woe"], 4), "points": round(pts)})
    return pd.DataFrame(rows)


def score_from_logit(logit):
    factor, offset = factor_offset()
    return offset - factor * np.asarray(logit)
