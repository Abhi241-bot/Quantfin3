"""
Module D -- backtest.py

Turn the target-weight panel into realized P&L, net of transaction costs, and
build gross/net strategy equity curves plus the benchmark buy-and-hold curve.

TIMING (the core discipline of this module):
  * Weights set at rebalance date t (built from data <= t) are HELD over the
    next month and earn the realized return measured at the NEXT month-end.
    => portfolio return at t+1 = sum_i w_i(t) * r_i(t -> t+1).
    The signal is therefore lagged one period; no future data enters the
    present allocation decision.

TRANSACTION COSTS:
  * Cost is charged on every weight change at each rebalance:
        traded_t   = sum_i |w_i(t) - w_i(t-1)|     (buys + sells, gross notional)
        cost_t     = traded_t * cost_per_side
        turnover_t = 0.5 * traded_t                (one-way turnover, reported)
  * cost_per_side = COST_BPS / 1e4. Default 10 bps/side (stated assumption).
  * The first rebalance trades in from cash (w(t-1)=0), so it pays a full
    gross-exposure setup cost -- included, not hidden.
  * Simplification: turnover is computed target-to-target (intra-month weight
    drift between rebalances is ignored). Documented in the README.

MISSING NEXT-PERIOD PRICE:
  * A held name with no next-month price earns 0% that month and is dropped at
    the next rebalance (it gets no score). Chosen for transparency; slightly
    optimistic for shorts. Documented in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import build_weight_panel, BACKTEST_START

COST_BPS = 10.0                       # transaction cost per side, basis points
RF_ANNUAL = 0.0                       # risk-free assumed 0 for Sharpe (stated)
PERIODS_PER_YEAR = 12                 # monthly rebalancing


# --------------------------------------------------------------------------- #
# Return building blocks
# --------------------------------------------------------------------------- #
def monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns; NaNs NOT padded (missing -> NaN, handled later)."""
    return monthly_prices.pct_change(fill_method=None)


def _next_month_end(monthly_index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    """The month-end immediately following rebalance date d (the holding end)."""
    pos = monthly_index.get_loc(d)
    return monthly_index[pos + 1]


# --------------------------------------------------------------------------- #
# Core backtest of a weight panel
# --------------------------------------------------------------------------- #
def backtest_weights(weight_panel: pd.DataFrame, monthly_prices: pd.DataFrame,
                     cost_bps: float = COST_BPS) -> pd.DataFrame:
    """
    Backtest a panel of target weights (rows = rebalance dates, cols = tickers).

    Returns a DataFrame indexed by HOLDING-END date with columns:
      gross_ret, cost, net_ret, turnover, gross_equity, net_equity.
    """
    cost_per_side = cost_bps / 1e4
    rets = monthly_returns(monthly_prices)
    mi = monthly_prices.index

    records = []
    prev_w = pd.Series(0.0, index=weight_panel.columns)   # start from cash

    for d in weight_panel.index:
        w = weight_panel.loc[d]

        # --- cost charged at this rebalance (trading from prev_w into w) ---
        traded = (w - prev_w).abs().sum()
        cost = traded * cost_per_side
        turnover = 0.5 * traded

        # --- realized gross return over the NEXT month (one-period lag) ---
        end = _next_month_end(mi, d)
        r = rets.loc[end].reindex(w.index)
        gross_ret = float((w * r.fillna(0.0)).sum())       # missing -> 0% (held)
        net_ret = gross_ret - cost

        records.append((end, gross_ret, cost, net_ret, turnover))
        prev_w = w

    out = pd.DataFrame(records,
                       columns=["date", "gross_ret", "cost", "net_ret", "turnover"]
                       ).set_index("date")
    out["gross_equity"] = (1.0 + out["gross_ret"]).cumprod()
    out["net_equity"] = (1.0 + out["net_ret"]).cumprod()
    return out


# --------------------------------------------------------------------------- #
# Benchmark buy-and-hold over the same dates
# --------------------------------------------------------------------------- #
def benchmark_returns(benchmark: pd.Series, on_dates: pd.DatetimeIndex) -> pd.Series:
    """Benchmark monthly returns aligned to the strategy's holding-end dates."""
    bm = benchmark.resample("ME").last()
    bret = bm.pct_change(fill_method=None)
    return bret.reindex(on_dates)


# --------------------------------------------------------------------------- #
# Convenience: build weights for a signal and backtest in one call
# --------------------------------------------------------------------------- #
def run_strategy(daily: pd.DataFrame, monthly: pd.DataFrame,
                 benchmark: pd.Series | None = None, factor: str = "composite",
                 cost_bps: float = COST_BPS, start: str = BACKTEST_START
                 ) -> pd.DataFrame:
    """Build the weight panel for `factor` and backtest it. Adds benchmark cols."""
    panel = build_weight_panel(daily, monthly, factor=factor, start=start)
    res = backtest_weights(panel, monthly, cost_bps=cost_bps)
    if benchmark is not None:
        res["bench_ret"] = benchmark_returns(benchmark, res.index)
        res["bench_equity"] = (1.0 + res["bench_ret"].fillna(0.0)).cumprod()
    return res


# --------------------------------------------------------------------------- #
# Small inline summary (full metrics live in metrics.py)
# --------------------------------------------------------------------------- #
def _quick_stats(ret: pd.Series, label: str) -> None:
    ret = ret.dropna()
    n = len(ret)
    total = (1 + ret).prod() - 1
    ann = (1 + total) ** (PERIODS_PER_YEAR / n) - 1
    vol = ret.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (ann - RF_ANNUAL) / vol if vol > 0 else np.nan
    print(f"  {label:<18} total {total:>7.1%} | ann {ann:>7.2%} | "
          f"vol {vol:>6.2%} | Sharpe {sharpe:>5.2f}")


if __name__ == "__main__":
    from data_loader import load_prices, to_month_end, load_benchmark

    daily = load_prices()
    monthly = to_month_end(daily)
    bench = load_benchmark()

    res = run_strategy(daily, monthly, bench, factor="composite", cost_bps=COST_BPS)

    print(f"Backtest window: {res.index.min().date()} -> {res.index.max().date()} "
          f"({len(res)} monthly periods)\n")

    print(f"Headline (cost = {COST_BPS:.0f} bps/side, rf = {RF_ANNUAL:.0%}):")
    _quick_stats(res["gross_ret"], "Combined GROSS")
    _quick_stats(res["net_ret"], "Combined NET")
    _quick_stats(res["bench_ret"], "Benchmark (N100)")

    avg_turn = res["turnover"].mean()
    total_cost = res["cost"].sum()
    print(f"\n  Avg one-way turnover : {avg_turn:.1%} per month "
          f"(~{avg_turn * 12:.0%} annualized)")
    print(f"  Total cost drag      : {total_cost:.1%} of capital over the backtest")
    print(f"  Final NET equity     : {res['net_equity'].iloc[-1]:.3f}x  "
          f"(gross {res['gross_equity'].iloc[-1]:.3f}x, "
          f"benchmark {res['bench_equity'].iloc[-1]:.3f}x)")

    # ---- discipline spot-check: first period uses a FUTURE return vs decision date
    d0 = build_weight_panel(daily, monthly, factor="composite").index[0]
    print(f"\nLOOK-AHEAD CHECK: first rebalance decided on {d0.date()}, "
          f"first P&L realized on {res.index[0].date()} "
          f"({'OK: P&L date is AFTER decision date' if res.index[0] > d0 else 'VIOLATION'})")
