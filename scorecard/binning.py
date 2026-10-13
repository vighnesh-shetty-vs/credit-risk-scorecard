"""Weight-of-Evidence binning with a monotonic WoE constraint.

WoE of a bin = ln( share of goods in bin / share of bads in bin ).
Information Value = sum over bins of (share good - share bad) * WoE.
Monotonic WoE keeps the scorecard explainable: "higher utilisation -> fewer points", never zig-zag.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing so empty cells do not produce infinite WoE


@dataclass
class FeatureBins:
    name: str
    categorical: bool
    edges: list = field(default_factory=list)      # numeric: inner cut points
    categories: list = field(default_factory=list)  # categorical: list of category groups
    table: pd.DataFrame = None                      # per-bin counts, WoE, IV

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def assign(self, values: pd.Series) -> pd.Series:
        """Bin label for each value (string labels, stable across train/test)."""
        if self.categorical:
            lookup = {v: i for i, grp in enumerate(self.categories) for v in grp}
            idx = values.map(lookup).fillna(len(self.categories) - 1).astype(int)
        else:
            idx = pd.Series(np.searchsorted(self.edges, values.to_numpy(), side="right"), index=values.index)
        return idx

    def woe(self, values: pd.Series) -> pd.Series:
        return self.assign(values).map(self.table["woe"].to_dict()).astype(float)


def woe_table(bin_idx: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Counts, bad rate, WoE and IV per bin index."""
    t = pd.DataFrame({"bin": bin_idx, "y": y}).groupby("bin")["y"].agg(n="count", bads="sum")
    t["goods"] = t["n"] - t["bads"]
    total_g, total_b = t["goods"].sum(), t["bads"].sum()
    dist_g = (t["goods"] + EPS) / (total_g + EPS * len(t))
    dist_b = (t["bads"] + EPS) / (total_b + EPS * len(t))
    t["bad_rate"] = t["bads"] / t["n"]
    t["woe"] = np.log(dist_g / dist_b)
    t["iv"] = (dist_g - dist_b) * t["woe"]
    return t


def _is_monotonic(s: pd.Series) -> bool:
    d = np.diff(s.to_numpy())
    return bool(np.all(d >= 0) or np.all(d <= 0))


def fit_numeric(x: pd.Series, y: pd.Series, name: str, max_bins: int = 10, min_share: float = 0.05) -> FeatureBins:
    """Start from quantile bins, then merge the adjacent pair with the closest bad rate
    until WoE is monotonic and every bin holds at least `min_share` of the population."""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    edges = list(qs)
    while True:
        idx = pd.Series(np.searchsorted(edges, x.to_numpy(), side="right"), index=x.index)
        t = woe_table(idx, y).reindex(range(len(edges) + 1)).fillna(0)
        small = (t["n"] / len(x)) < min_share
        if len(edges) == 0 or (_is_monotonic(t["woe"]) and not small.any()):
            break
        # merge the adjacent pair whose bad rates are most similar (or the smallest bin)
        if small.any():
            i = int(np.argmin(t["n"].to_numpy()))
            drop = min(i, len(edges) - 1) if i > 0 else 0
        else:
            gaps = np.abs(np.diff(t["bad_rate"].to_numpy()))
            drop = int(np.argmin(gaps))
        edges.pop(drop)
    fb = FeatureBins(name=name, categorical=False, edges=edges)
    fb.table = woe_table(fb.assign(x), y)
    fb.table["range"] = _ranges(edges)
    return fb


def fit_categorical(x: pd.Series, y: pd.Series, name: str, groups: list) -> FeatureBins:
    """Categorical feature with analyst-defined groups (e.g. delay buckets)."""
    fb = FeatureBins(name=name, categorical=True, categories=groups)
    fb.table = woe_table(fb.assign(x), y)
    fb.table["range"] = [str(g) for g in groups][: len(fb.table)]
    return fb


def _ranges(edges: list) -> list:
    bounds = [-np.inf] + list(edges) + [np.inf]
    return [f"[{_fmt(a)}, {_fmt(b)})" for a, b in zip(bounds[:-1], bounds[1:])]


def _fmt(v: float) -> str:
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}"
