"""
Module B -- factors.py

Compute cross-sectional factor scores POINT-IN-TIME at a given rebalance date.

Three factors (each returns a z-score where HIGHER = more attractive = LONG leg):

  momentum  : 12-1 momentum. Return from t-12m to t-1m (skip the most recent
              month to avoid short-term reversal). High momentum -> long.

  low_vol   : trailing annualized realized volatility of daily returns over the
              past ~12 months. LOW vol -> long, so the score is z(-vol). The
              low-vol anomaly: low-risk stocks have historically earned better
              risk-adjusted returns than CAPM predicts.

  value     : long-term-reversal PROXY (true point-in-time fundamentals are not
              available from yfinance -- see README). Return from t-60m to t-12m;
              past long-term LOSERS are treated as 'cheap' -> long. Score is
              z(-lt_reversal). Disclosed as a proxy, not a real B/M or E/P factor.

DISCIPLINE -- look-ahead avoidance:
  * Every factor at rebalance date t uses ONLY prices with timestamp <= t.
  * Momentum/value use month-end prices up to and including t (the 12-1 / 60-12
    windows all END at or before t).
  * Low-vol uses daily returns up to and including t.
  * Next-period returns are NEVER referenced here -- they live only in backtest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Factor sign convention: each *_score column is oriented so HIGHER => LONG.
FACTOR_NAMES = ["momentum", "low_vol", "value"]


# --------------------------------------------------------------------------- #
# Cross-sectional helpers
# --------------------------------------------------------------------------- #
def zscore(s: pd.Series, winsor: float = 3.0) -> pd.Series:
    """
    Cross-sectional z-score with winsorization of extremes at +/- `winsor`
    standard deviations (reduces outlier impact, as the spec asks).
    """
    s = s.dropna()
    if len(s) < 2 or s.std(ddof=0) == 0:
        return pd.Series(dtype=float)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.clip(-winsor, winsor)


def _loc_index(monthly_index: pd.DatetimeIndex, asof_date) -> int:
    """Integer position of asof_date in the month-end index (must be exact)."""
    asof = pd.Timestamp(asof_date)
    if asof not in monthly_index:
        raise KeyError(f"{asof.date()} is not a month-end rebalance date")
    return monthly_index.get_loc(asof)


# --------------------------------------------------------------------------- #
# Raw factor values (pre-standardization)
# --------------------------------------------------------------------------- #
def momentum_raw(monthly: pd.DataFrame, asof_date,
                 lookback: int = 12, skip: int = 1) -> pd.Series:
    """12-1 momentum: total return from t-`lookback`m to t-`skip`m (uses data <= t)."""
    i = _loc_index(monthly.index, asof_date)
    if i < lookback:
        return pd.Series(dtype=float)
    p_recent = monthly.iloc[i - skip]      # price at t-1m  (<= t)
    p_old = monthly.iloc[i - lookback]     # price at t-12m (<= t)
    mom = p_recent / p_old - 1.0
    return mom.replace([np.inf, -np.inf], np.nan).dropna()


def trailing_vol_raw(daily: pd.DataFrame, asof_date,
                     window_days: int = 252, min_obs: int = 200) -> pd.Series:
    """Annualized realized vol of daily returns over the trailing window ending at t."""
    asof = pd.Timestamp(asof_date)
    hist = daily.loc[:asof]                 # strictly <= t (no future data)
    hist = hist.iloc[-(window_days + 1):]   # last ~12 months of prices
    rets = hist.pct_change()
    vol = rets.std(ddof=0) * np.sqrt(252.0)
    enough = rets.notna().sum() >= min_obs  # require sufficient history
    vol = vol.where(enough)
    return vol.replace([np.inf, -np.inf], np.nan).dropna()


def lt_reversal_raw(monthly: pd.DataFrame, asof_date,
                    long_lb: int = 60, short_lb: int = 12) -> pd.Series:
    """Long-term reversal: total return from t-60m to t-12m (value proxy, data <= t)."""
    i = _loc_index(monthly.index, asof_date)
    if i < long_lb:
        return pd.Series(dtype=float)
    p_recent = monthly.iloc[i - short_lb]   # price at t-12m
    p_old = monthly.iloc[i - long_lb]       # price at t-60m
    ltr = p_recent / p_old - 1.0
    return ltr.replace([np.inf, -np.inf], np.nan).dropna()


# --------------------------------------------------------------------------- #
# Oriented factor z-scores (HIGHER = LONG)
# --------------------------------------------------------------------------- #
def momentum_score(monthly: pd.DataFrame, asof_date, **kw) -> pd.Series:
    return zscore(momentum_raw(monthly, asof_date, **kw))


def low_vol_score(daily: pd.DataFrame, asof_date, **kw) -> pd.Series:
    # low vol is attractive -> negate so HIGHER score = LOWER vol = long
    return zscore(-trailing_vol_raw(daily, asof_date, **kw))


def value_score(monthly: pd.DataFrame, asof_date, **kw) -> pd.Series:
    # long-term losers are 'cheap' -> negate so HIGHER score = bigger past loser = long
    return zscore(-lt_reversal_raw(monthly, asof_date, **kw))


# --------------------------------------------------------------------------- #
# Combined factor panel at one date
# --------------------------------------------------------------------------- #
def factor_scores(daily: pd.DataFrame, monthly: pd.DataFrame, asof_date,
                  weights: dict | None = None) -> pd.DataFrame:
    """
    Return a DataFrame indexed by ticker with columns
    [momentum, low_vol, value, composite] of oriented z-scores at `asof_date`.

    composite = equal-weight mean of the available factor z-scores (a stock is
    scored on whatever factors it has enough history for; composite uses the
    mean of present factors so late-IPO names aren't auto-excluded).
    """
    cols = {
        "momentum": momentum_score(monthly, asof_date),
        "low_vol": low_vol_score(daily, asof_date),
        "value": value_score(monthly, asof_date),
    }
    df = pd.DataFrame(cols)

    if weights is None:
        # equal weight; mean over present factors (skipna) then re-z for balance
        df["composite"] = df[FACTOR_NAMES].mean(axis=1, skipna=True)
    else:
        w = pd.Series(weights)
        df["composite"] = (df[FACTOR_NAMES] * w).sum(axis=1, min_count=1) / w.sum()

    return df


if __name__ == "__main__":
    from data_loader import load_prices, to_month_end

    daily = load_prices()
    monthly = to_month_end(daily)

    # Pick a rebalance date with full history for all factors (needs >= 60 months).
    asof = monthly.index[72]   # ~mid sample
    print(f"As-of rebalance date: {asof.date()}  (monthly index pos 72)\n")

    # ---- discipline check: show the exact windows used are all <= asof ----
    i = monthly.index.get_loc(asof)
    print("LOOK-AHEAD CHECK (all window endpoints must be <= as-of date):")
    print(f"  momentum   uses month-end prices at {monthly.index[i-12].date()} "
          f"-> {monthly.index[i-1].date()}")
    print(f"  value(LTR) uses month-end prices at {monthly.index[i-60].date()} "
          f"-> {monthly.index[i-12].date()}")
    print(f"  low_vol    uses daily returns up to {daily.loc[:asof].index[-1].date()}")
    print(f"  as-of date is {asof.date()}  -> no endpoint exceeds it: OK\n")

    scores = factor_scores(daily, monthly, asof)
    print(f"Scored {scores['composite'].notna().sum()} stocks "
          f"(of {monthly.shape[1]} in universe).\n")
    print("Top 8 by composite (LONG candidates):")
    print(scores.sort_values("composite", ascending=False).head(8).round(2).to_string())
    print("\nBottom 8 by composite (SHORT candidates):")
    print(scores.sort_values("composite").head(8).round(2).to_string())

    print("\nInter-factor correlation of z-scores at this date:")
    print(scores[FACTOR_NAMES].corr().round(2).to_string())
