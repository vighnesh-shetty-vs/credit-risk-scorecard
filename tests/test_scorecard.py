import numpy as np
import pandas as pd
import pytest

from scorecard import binning, data, metrics, scaling


@pytest.fixture
def risk_data():
    """Synthetic feature where default probability falls as x rises."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(0, 100, 20_000))
    y = pd.Series((rng.uniform(0, 1, 20_000) < 0.5 - 0.004 * x).astype(int))
    return x, y


def test_woe_sign_and_iv_hand_calculation():
    # bin 0: 80 goods / 20 bads, bin 1: 20 goods / 80 bads
    idx = pd.Series([0] * 100 + [1] * 100)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    t = binning.woe_table(idx, y)
    assert t.loc[0, "woe"] > 0 > t.loc[1, "woe"]                 # safer bin -> positive WoE
    # with +0.5 smoothing per cell: share of goods in bin 0 = 80.5 / 101, share of bads = 20.5 / 101
    g, b = 80.5 / 101, 20.5 / 101
    expected_iv = 2 * (g - b) * np.log(g / b)
    assert t["iv"].sum() == pytest.approx(expected_iv)
    assert expected_iv == pytest.approx(2 * 0.6 * np.log(4), rel=0.03)   # close to the unsmoothed textbook value


def test_numeric_binning_is_monotonic_with_min_bin_size(risk_data):
    x, y = risk_data
    fb = binning.fit_numeric(x, y, "x", max_bins=10, min_share=0.05)
    woe = fb.table["woe"].to_numpy()
    assert np.all(np.diff(woe) >= 0) or np.all(np.diff(woe) <= 0)
    assert (fb.table["n"] / len(x) >= 0.05).all()
    assert fb.iv > 0.1


def test_categorical_unknown_values_go_to_last_group():
    fb = binning.FeatureBins("c", categorical=True, categories=[[1], [2], [3, 4]])
    assert fb.assign(pd.Series([1, 2, 4, 99])).tolist() == [0, 1, 2, 2]


def test_ks_and_gini_extremes():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert metrics.ks(y, perfect) == pytest.approx(1.0)
    assert metrics.gini(y, perfect) == pytest.approx(1.0)
    assert metrics.gini(y, -perfect) == pytest.approx(-1.0)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 50_000)
    assert metrics.psi(a, rng.normal(0, 1, 50_000)) < 0.01
    assert metrics.psi(a, rng.normal(1, 1, 50_000)) > 0.25


def test_points_to_double_the_odds():
    factor, offset = scaling.factor_offset()
    # logit of P(default) for good:bad odds of 50:1 and 100:1
    s50 = scaling.score_from_logit(-np.log(50))
    s100 = scaling.score_from_logit(-np.log(100))
    assert s50 == pytest.approx(scaling.BASE_SCORE)
    assert s100 - s50 == pytest.approx(scaling.PDO)


def test_clean_fixes_undocumented_codes():
    raw = pd.DataFrame({
        "ID": [1, 2], "LIMIT_BAL": [10_000, 20_000], "SEX": [1, 2], "EDUCATION": [0, 6], "MARRIAGE": [0, 1],
        "AGE": [30, 40], **{c: [0, 2] for c in data.PAY_STATUS},
        **{c: [1_000, -500] for c in data.BILLS}, **{c: [100, 0] for c in data.PAYMENTS}, data.TARGET: [0, 1],
    })
    out = data.clean(raw)
    assert out["EDUCATION"].tolist() == [4, 4]
    assert out["MARRIAGE"].tolist() == [3, 1]
    assert out["max_delay_6m"].tolist() == [0, 2]
    assert out.loc[1, "payment_ratio_avg_5m"] == 1.0   # owed nothing -> counts as fully paid
