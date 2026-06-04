"""
Module C -- portfolio.py

Turn factor scores into target portfolio weights at each monthly rebalance.

Construction (standard choices -- NOT tuned to maximize Sharpe):
  * Rank the cross-section by score at each rebalance date.
  * LONG the top quintile (top 20%), SHORT the bottom quintile (bottom 20%).
  * Equal weight within each leg.
  * Dollar-neutral: long leg weights sum to +1, short leg to -1 (gross 2.0,
    net 0.0). Portfolio return is then the long-short spread:
        r_p = mean(long returns) - mean(short returns).
  * Positions are held until the next rebalance (carried forward in backtest).

DISCIPLINE -- look-ahead avoidance:
  * Weights at rebalance date t are built from factor scores that use only
    data <= t (see factors.py). They are then HELD over (t, t+1] and earn next
    period's return in backtest.py -- the signal is lagged one period by
    construction, so no future information enters the present decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors import FACTOR_NAMES, factor_scores

QUANTILE = 0.20          # top/bottom 20% (quintile) -- standard, not tuned
BACKTEST_START = "2017-01-01"


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
def rebalance_dates(monthly: pd.DataFrame, start: str = BACKTEST_START,
                    min_history: int = 60) -> pd.DatetimeIndex:
    """
    Month-end rebalance dates >= `start` that also have at least `min_history`
    prior monthly observations (so the 60-12 value factor is computable). The
    last month-end is excluded as a rebalance date because it has no subsequent
    period to earn a return.
    """
    idx = monthly.index
    eligible = idx[(idx >= pd.Timestamp(start))]
    # need >= min_history months before the date, and a next month to hold into
    out = [d for d in eligible
           if idx.get_loc(d) >= min_history and idx.get_loc(d) < len(idx) - 1]
    return pd.DatetimeIndex(out)


# --------------------------------------------------------------------------- #
# Ranking -> weights
# --------------------------------------------------------------------------- #
def rank_to_weights(score: pd.Series, quantile: float = QUANTILE) -> pd.Series:
    """
    Map a cross-sectional score (higher = more attractive) to dollar-neutral
    long-short weights: top `quantile` long (sum +1), bottom `quantile` short
    (sum -1), equal-weight within each leg.
    """
    s = score.dropna()
    n = len(s)
    if n < 5:
        return pd.Series(dtype=float)

    k = max(1, int(round(quantile * n)))
    ranked = s.sort_values(ascending=False)
    longs = ranked.index[:k]
    shorts = ranked.index[-k:]

    w = pd.Series(0.0, index=s.index)
    w.loc[longs] = 1.0 / len(longs)
    w.loc[shorts] = -1.0 / len(shorts)
    return w


# --------------------------------------------------------------------------- #
# Weight panel over the whole backtest
# --------------------------------------------------------------------------- #
def build_weight_panel(daily: pd.DataFrame, monthly: pd.DataFrame,
                       factor: str = "composite", quantile: float = QUANTILE,
                       start: str = BACKTEST_START) -> pd.DataFrame:
    """
    Build target weights for every rebalance date for one signal.

    `factor` is 'composite' or one of FACTOR_NAMES (for standalone attribution).
    Returns a DataFrame indexed by rebalance date, columns = full universe,
    values = target weight (0 where not selected). Each row is the book set on
    that date and HELD until the next rebalance date.
    """
    if factor not in FACTOR_NAMES + ["composite"]:
        raise ValueError(f"factor must be 'composite' or one of {FACTOR_NAMES}")

    dates = rebalance_dates(monthly, start=start)
    rows = {}
    for d in dates:
        scores = factor_scores(daily, monthly, d)
        w = rank_to_weights(scores[factor], quantile=quantile)
        rows[d] = w

    panel = pd.DataFrame(rows).T            # dates x tickers (selected only)
    panel = panel.reindex(columns=monthly.columns).fillna(0.0)
    panel.index.name = "rebalance_date"
    return panel


if __name__ == "__main__":
    from data_loader import load_prices, to_month_end

    daily = load_prices()
    monthly = to_month_end(daily)

    dates = rebalance_dates(monthly)
    print(f"Rebalance dates: {len(dates)}  "
          f"({dates.min().date()} -> {dates.max().date()}, monthly)\n")

    panel = build_weight_panel(daily, monthly, factor="composite")
    print(f"Composite weight panel shape: {panel.shape} (dates x tickers)\n")

    # ---- exposure / neutrality checks ----
    longs = (panel > 0).sum(axis=1)
    shorts = (panel < 0).sum(axis=1)
    long_sum = panel.clip(lower=0).sum(axis=1)
    short_sum = panel.clip(upper=0).sum(axis=1)
    gross = panel.abs().sum(axis=1)
    net = panel.sum(axis=1)
    print("Per-rebalance book stats (averages over backtest):")
    print(f"  # long names      : {longs.mean():.1f}  (range {longs.min()}-{longs.max()})")
    print(f"  # short names     : {shorts.mean():.1f}  (range {shorts.min()}-{shorts.max()})")
    print(f"  long  weight sum  : {long_sum.mean():+.3f}  (target +1.000)")
    print(f"  short weight sum  : {short_sum.mean():+.3f}  (target -1.000)")
    print(f"  gross exposure    : {gross.mean():.3f}  (target 2.000)")
    print(f"  NET exposure      : {net.mean():+.6f}  (target 0 -> dollar-neutral)\n")

    sample = dates[0]
    s = panel.loc[sample]
    print(f"Sample book @ {sample.date()}:")
    print("  LONGS :", list(s[s > 0].index))
    print("  SHORTS:", list(s[s < 0].index))
