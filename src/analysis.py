"""
Module F -- analysis.py

Produce the deliverable figures and the (stretch) Information Coefficient study.
All compute reuses the earlier modules; this module only visualizes/aggregates.

Outputs (results/):
  equity_curves.png      each factor + composite (net) + benchmark, one chart
  factor_attribution.png net Sharpe / annualized alpha / beta per factor
  rolling_sharpe.png     12-month rolling annualized Sharpe -> regime discussion
  ic_analysis.png        IC time series + IC decay by horizon (stretch goal)

Information Coefficient (IC):
  At each rebalance date, the cross-sectional Spearman rank correlation between a
  factor's score (data <= t) and the subsequent realized return. Mean IC > 0
  means the ranking has predictive power; the decay-by-horizon plot shows how
  fast that power fades. Computed point-in-time -- score at t vs return after t.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                       # headless: save PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_loader import load_prices, to_month_end, load_benchmark
from factors import FACTOR_NAMES, factor_scores
from portfolio import build_all_panels, rebalance_dates
from backtest import (backtest_weights, backtest_weights_hedged,
                      benchmark_returns, monthly_returns,
                      COST_BPS, PERIODS_PER_YEAR)
from metrics import summarize, RESULTS_DIR

SIGNALS = FACTOR_NAMES + ["composite"]
_COLORS = {"momentum": "tab:blue", "low_vol": "tab:orange",
           "value": "tab:green", "composite": "black",
           "composite_hedged": "tab:purple", "benchmark": "tab:red"}


# --------------------------------------------------------------------------- #
# Backtest every signal once and collect result frames
# --------------------------------------------------------------------------- #
def run_all(daily, monthly, benchmark, cost_bps=COST_BPS):
    panels = build_all_panels(daily, monthly)
    results, net_streams = {}, {}
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
    results["composite_hedged"] = (
        hedged[["net_ret_hedged", "net_equity_hedged"]]
        .rename(columns={"net_ret_hedged": "net_ret",
                         "net_equity_hedged": "net_equity"}))
    net_streams["composite_hedged"] = hedged["net_ret_hedged"]

    bench_equity = (1 + bench_ret.fillna(0.0)).cumprod()
    return results, pd.DataFrame(net_streams), bench_ret, bench_equity


# --------------------------------------------------------------------------- #
# Figure 1: equity curves
# --------------------------------------------------------------------------- #
def plot_equity_curves(results, bench_equity, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    curves = SIGNALS + (["composite_hedged"] if "composite_hedged" in results else [])
    for name in curves:
        eq = results[name]["net_equity"]
        label = f"{name} (net)" if name != "composite_hedged" \
            else "composite (beta-hedged)"
        ax.plot(eq.index, eq.values, label=label, color=_COLORS[name],
                lw=2 if name in ("composite", "composite_hedged") else 1.3,
                ls="-." if name == "composite_hedged" else "-")
    ax.plot(bench_equity.index, bench_equity.values, label="benchmark N100",
            color=_COLORS["benchmark"], lw=1.6, ls="--")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title("Equity curves — long-short factors (net of costs) vs Nifty 100 "
                 "buy-and-hold")
    ax.set_ylabel("Growth of 1.0")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: factor attribution bars
# --------------------------------------------------------------------------- #
def plot_attribution(table, path):
    facs = ["momentum", "low_vol", "value", "composite"]
    sharpe = [table.loc[f, "sharpe_net"] for f in facs]
    alpha = [table.loc[f, "alpha_ann"] * 100 for f in facs]
    beta = [table.loc[f, "beta"] for f in facs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, vals, title, ylab in [
        (axes[0], sharpe, "Net Sharpe", "Sharpe"),
        (axes[1], alpha, "Annualized alpha vs benchmark", "alpha (%)"),
        (axes[2], beta, "Beta to benchmark", "beta"),
    ]:
        colors = [_COLORS[f] for f in facs]
        ax.bar(facs, vals, color=colors, alpha=0.85)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-factor attribution (net of costs)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: rolling Sharpe (regime view)
# --------------------------------------------------------------------------- #
def plot_rolling_sharpe(net_df, bench_ret, path, window=12):
    def roll_sharpe(r):
        return (r.rolling(window).mean() / r.rolling(window).std()) \
            * np.sqrt(PERIODS_PER_YEAR)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in SIGNALS:
        ax.plot(net_df.index, roll_sharpe(net_df[name]), label=name,
                color=_COLORS[name], lw=2 if name == "composite" else 1.2)
    ax.plot(bench_ret.index, roll_sharpe(bench_ret), label="benchmark N100",
            color=_COLORS["benchmark"], lw=1.4, ls="--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{window}-month rolling annualized Sharpe — factor regimes")
    ax.set_ylabel("Rolling Sharpe")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Information Coefficient (stretch)
# --------------------------------------------------------------------------- #
def compute_ic(daily, monthly, max_horizon=12):
    """
    Returns (ic_timeseries, ic_decay):
      ic_timeseries: DataFrame [rebalance_date x factor] of 1-month-ahead Spearman IC.
      ic_decay:      DataFrame [horizon(1..H) x factor] of mean IC over horizon h.
    """
    rets = monthly_returns(monthly)
    mi = monthly.index
    dates = rebalance_dates(monthly)

    ts_rows, decay_acc = {}, {h: {f: [] for f in FACTOR_NAMES}
                              for h in range(1, max_horizon + 1)}
    for d in dates:
        scores = factor_scores(daily, monthly, d)
        pos = mi.get_loc(d)
        # 1-month-ahead IC for the time series
        end1 = mi[pos + 1]
        fwd1 = rets.loc[end1]
        ts_rows[d] = {f: _spearman(scores[f], fwd1) for f in FACTOR_NAMES}
        # decay: cumulative forward return over h months
        for h in range(1, max_horizon + 1):
            if pos + h >= len(mi):
                continue
            fwd_h = monthly.iloc[pos + h] / monthly.loc[d] - 1.0
            for f in FACTOR_NAMES:
                ic = _spearman(scores[f], fwd_h)
                if not np.isnan(ic):
                    decay_acc[h][f].append(ic)

    ic_ts = pd.DataFrame(ts_rows).T
    ic_ts.index.name = "rebalance_date"
    decay = pd.DataFrame({f: {h: np.mean(decay_acc[h][f]) if decay_acc[h][f] else np.nan
                              for h in range(1, max_horizon + 1)}
                          for f in FACTOR_NAMES})
    decay.index.name = "horizon_months"
    return ic_ts, decay


def _spearman(score: pd.Series, fwd: pd.Series) -> float:
    df = pd.concat([score, fwd], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    rho, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return rho


def plot_ic(ic_ts, decay, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: 12m rolling-mean IC time series
    for f in FACTOR_NAMES:
        axes[0].plot(ic_ts.index, ic_ts[f].rolling(12).mean(),
                     label=f, color=_COLORS[f], lw=1.4)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_title("12-month rolling mean IC (1-month-ahead)")
    axes[0].set_ylabel("Spearman IC")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    # right: IC decay by horizon
    for f in FACTOR_NAMES:
        axes[1].plot(decay.index, decay[f], marker="o", label=f,
                     color=_COLORS[f], lw=1.4)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("IC decay by forward horizon")
    axes[1].set_xlabel("Horizon (months ahead)")
    axes[1].set_ylabel("Mean Spearman IC")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    daily = load_prices()
    monthly = to_month_end(daily)
    bench = load_benchmark()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Backtesting all signals ...")
    results, net_df, bench_ret, bench_equity = run_all(daily, monthly, bench)
    table = pd.DataFrame([summarize(n, results[n], bench_ret)
                          for n in SIGNALS]).set_index("strategy")

    p1 = os.path.join(RESULTS_DIR, "equity_curves.png")
    p2 = os.path.join(RESULTS_DIR, "factor_attribution.png")
    p3 = os.path.join(RESULTS_DIR, "rolling_sharpe.png")
    p4 = os.path.join(RESULTS_DIR, "ic_analysis.png")

    plot_equity_curves(results, bench_equity, p1)
    plot_attribution(table, p2)
    plot_rolling_sharpe(net_df, bench_ret, p3)
    print(f"Saved: {os.path.basename(p1)}, {os.path.basename(p2)}, "
          f"{os.path.basename(p3)}")

    print("Computing Information Coefficient ...")
    ic_ts, decay = compute_ic(daily, monthly)
    plot_ic(ic_ts, decay, p4)
    print(f"Saved: {os.path.basename(p4)}")

    print("\nMean 1-month-ahead IC (full sample):")
    mean_ic = ic_ts.mean()
    ic_ir = ic_ts.mean() / ic_ts.std()
    pct_pos = (ic_ts > 0).mean()
    ic_summary = pd.DataFrame({"mean_IC": mean_ic, "IC_IR": ic_ir,
                               "pct_positive": pct_pos}).round(3)
    print(ic_summary.to_string())
    print("\nIC decay (mean IC by horizon, months):")
    print(decay.round(3).to_string())
