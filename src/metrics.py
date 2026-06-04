"""
Module E -- metrics.py

Performance and attribution metrics, all reported NET of costs unless labelled
gross. Built on the backtest output from backtest.py.

Reported per strategy:
  * Annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar.
  * vs benchmark: alpha (annualized), beta, information ratio -- from an OLS
    regression of strategy monthly returns on benchmark monthly returns.
  * Cost drag: gross Sharpe vs net Sharpe, average turnover, total cost.

Per-factor attribution:
  * Each factor backtested standalone (momentum / low_vol / value) plus the
    combined composite and the benchmark, in one table.
  * Inter-factor correlation of the standalone NET return streams -> shows the
    diversification (or lack of it) that drives the combined Sharpe.

Assumptions: risk-free rate = 0 (stated); monthly data, 12 periods/year.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from portfolio import build_all_panels
from backtest import (backtest_weights, backtest_weights_hedged,
                      benchmark_returns, COST_BPS, RF_ANNUAL, PERIODS_PER_YEAR)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "results")
RF_PER_PERIOD = RF_ANNUAL / PERIODS_PER_YEAR


# --------------------------------------------------------------------------- #
# Standalone performance metrics
# --------------------------------------------------------------------------- #
def perf_metrics(ret: pd.Series) -> dict:
    """Annualized return/vol, Sharpe, Sortino, max drawdown, Calmar (rf=RF_ANNUAL)."""
    ret = ret.dropna()
    n = len(ret)
    if n == 0:
        return {k: np.nan for k in
                ["ann_return", "ann_vol", "sharpe", "sortino", "max_dd", "calmar"]}

    total = (1 + ret).prod() - 1
    ann_return = (1 + total) ** (PERIODS_PER_YEAR / n) - 1
    ann_vol = ret.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)

    excess = ret - RF_PER_PERIOD
    sharpe = (excess.mean() / ret.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR) \
        if ret.std(ddof=1) > 0 else np.nan

    downside = ret[ret < RF_PER_PERIOD] - RF_PER_PERIOD
    dd_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(PERIODS_PER_YEAR) \
        if len(downside) else np.nan
    sortino = (ann_return - RF_ANNUAL) / dd_dev if dd_dev and dd_dev > 0 else np.nan

    equity = (1 + ret).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, calmar=calmar)


# --------------------------------------------------------------------------- #
# Benchmark regression: alpha / beta / information ratio
# --------------------------------------------------------------------------- #
def regression_metrics(strat_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """
    OLS of strategy excess returns on benchmark excess returns:
        r_strat - rf = alpha + beta (r_bench - rf) + eps
    alpha is annualized (monthly alpha * 12); IR = appraisal ratio
    (annualized alpha / annualized residual vol).
    """
    df = pd.concat([strat_ret, bench_ret], axis=1, keys=["s", "b"]).dropna()
    if len(df) < 12:
        return dict(alpha_ann=np.nan, beta=np.nan, ir=np.nan, r2=np.nan)

    y = df["s"] - RF_PER_PERIOD
    X = sm.add_constant(df["b"] - RF_PER_PERIOD)
    model = sm.OLS(y, X).fit()

    alpha_m = model.params["const"]
    beta = model.params["b"]
    resid_vol_ann = model.resid.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    alpha_ann = alpha_m * PERIODS_PER_YEAR
    ir = alpha_ann / resid_vol_ann if resid_vol_ann > 0 else np.nan
    return dict(alpha_ann=alpha_ann, beta=beta, ir=ir, r2=model.rsquared)


# --------------------------------------------------------------------------- #
# Per-strategy summary row (combines everything)
# --------------------------------------------------------------------------- #
def summarize(name: str, res: pd.DataFrame, bench_ret: pd.Series) -> dict:
    """One summary row from a backtest result frame (needs net_ret/gross_ret/...)."""
    net = perf_metrics(res["net_ret"])
    gross = perf_metrics(res["gross_ret"])
    reg = regression_metrics(res["net_ret"], bench_ret)
    return {
        "strategy": name,
        "ann_return_net": net["ann_return"],
        "ann_vol_net": net["ann_vol"],
        "sharpe_net": net["sharpe"],
        "sharpe_gross": gross["sharpe"],
        "cost_drag_sharpe": gross["sharpe"] - net["sharpe"],
        "sortino_net": net["sortino"],
        "max_dd_net": net["max_dd"],
        "calmar_net": net["calmar"],
        "alpha_ann": reg["alpha_ann"],
        "beta": reg["beta"],
        "info_ratio": reg["ir"],
        "avg_turnover": res["turnover"].mean() if "turnover" in res else np.nan,
        "total_cost": res["cost"].sum() if "cost" in res else np.nan,
    }


# --------------------------------------------------------------------------- #
# Full attribution: every factor + composite + benchmark
# --------------------------------------------------------------------------- #
def summarize_hedged(res_h: pd.DataFrame, bench_ret: pd.Series) -> dict:
    """Summary row for the beta-hedged composite (uses net_ret_hedged)."""
    m = perf_metrics(res_h["net_ret_hedged"])
    reg = regression_metrics(res_h["net_ret_hedged"], bench_ret)
    total_cost = res_h["cost"].sum() + res_h["hedge_cost"].sum()
    return {
        "strategy": "composite_hedged",
        "ann_return_net": m["ann_return"], "ann_vol_net": m["ann_vol"],
        "sharpe_net": m["sharpe"], "sharpe_gross": np.nan,
        "cost_drag_sharpe": np.nan, "sortino_net": m["sortino"],
        "max_dd_net": m["max_dd"], "calmar_net": m["calmar"],
        "alpha_ann": reg["alpha_ann"], "beta": reg["beta"], "info_ratio": reg["ir"],
        "avg_turnover": res_h["turnover"].mean(), "total_cost": total_cost,
    }


def attribution(daily: pd.DataFrame, monthly: pd.DataFrame, benchmark: pd.Series,
                cost_bps: float = COST_BPS):
    """
    Backtest each factor standalone + the composite + a beta-hedged composite,
    regress vs benchmark, and return
    (summary_table, net_return_streams, factor_correlation, hedged_result).
    """
    panels = build_all_panels(daily, monthly)

    net_streams = {}
    results = {}
    bench_ret = None
    for name, panel in panels.items():
        res = backtest_weights(panel, monthly, cost_bps=cost_bps)
        if bench_ret is None:
            bench_ret = benchmark_returns(benchmark, res.index)
        results[name] = res
        net_streams[name] = res["net_ret"]

    # beta-neutral overlay on the composite (stretch goal)
    hedged = backtest_weights_hedged(panels["composite"], monthly, daily,
                                     benchmark, cost_bps=cost_bps)
    net_streams["composite_hedged"] = hedged["net_ret_hedged"]

    rows = [summarize(name, results[name], bench_ret)
            for name in ["momentum", "low_vol", "value", "composite"]]
    rows.append(summarize_hedged(hedged, bench_ret))

    # benchmark as its own row (long-only buy-and-hold; beta=1, alpha=0 by defn)
    b = perf_metrics(bench_ret)
    rows.append({
        "strategy": "benchmark_N100",
        "ann_return_net": b["ann_return"], "ann_vol_net": b["ann_vol"],
        "sharpe_net": b["sharpe"], "sharpe_gross": b["sharpe"],
        "cost_drag_sharpe": 0.0, "sortino_net": b["sortino"],
        "max_dd_net": b["max_dd"], "calmar_net": b["calmar"],
        "alpha_ann": 0.0, "beta": 1.0, "info_ratio": np.nan,
        "avg_turnover": np.nan, "total_cost": np.nan,
    })

    table = pd.DataFrame(rows).set_index("strategy")
    net_df = pd.DataFrame(net_streams)
    factor_corr = net_df[["momentum", "low_vol", "value"]].corr()
    return table, net_df, factor_corr, hedged


if __name__ == "__main__":
    from data_loader import load_prices, to_month_end, load_benchmark

    daily = load_prices()
    monthly = to_month_end(daily)
    bench = load_benchmark()

    table, net_df, factor_corr, _hedged = attribution(daily, monthly, bench)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"\nMETRICS SUMMARY (net of {COST_BPS:.0f} bps/side, rf={RF_ANNUAL:.0%})")
    print("=" * 120)
    show = table.copy()
    pct = ["ann_return_net", "ann_vol_net", "max_dd_net", "alpha_ann",
           "avg_turnover", "total_cost"]
    for c in pct:
        show[c] = (show[c] * 100).round(1)
    for c in ["sharpe_net", "sharpe_gross", "cost_drag_sharpe", "sortino_net",
              "calmar_net", "beta", "info_ratio"]:
        show[c] = show[c].round(2)
    print(show.to_string())

    print("\nINTER-FACTOR CORRELATION (standalone NET monthly returns):")
    print(factor_corr.round(2).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    table.to_csv(out_csv)
    print(f"\nSaved -> {out_csv}")
