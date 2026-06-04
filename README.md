# Multi-Factor Long-Short Equity Strategy (Indian Equities)

A systematic, monthly-rebalanced **long-short** equity strategy built on a Nifty 100
universe. It ranks the cross-section of stocks by three factors — **momentum**,
**low volatility**, and a **value proxy** — forms a dollar-neutral long-short book, and
backtests it **net of transaction costs** against a Nifty 100 buy-and-hold benchmark, with
full per-factor attribution.

> The point of the project is not to "win" — it is to build a factor strategy *correctly*
> (point-in-time, costed, attributed, benchmarked) and to **report the result honestly**,
> including a regime in which the strategy lost money. That honest result is itself the finding.

---

## Summary

- **Universe:** Nifty 100 (current constituents), 98 tradable large/mid-cap names after data cleaning.
- **Factors:** Momentum (12-1), Low Volatility (trailing 12m), Value (60-12 long-term-reversal **proxy**).
- **Construction:** monthly rebalance, long top quintile / short bottom quintile, equal-weight, dollar-neutral.
- **Costs:** 10 bps per side charged on turnover; all headline numbers are **net**.
- **Backtest window:** 2017-02 → 2024-12 (95 monthly periods). Price history loaded from 2011 so every factor's trailing window is fully populated at the first 2017 rebalance.

**Headline (net of costs):** the combined long-short book returned **−8.1% annualized**
(**Sharpe −0.32**, max drawdown **−61%**), versus the **Nifty 100 benchmark at +13.8%
annualized (Sharpe 0.86)**. The strategy was a *market-neutral* book, so it is not designed to
capture the market's +178% — but it genuinely lost money over this sample. **Crucially, the
benchmark regression shows the composite earned a positive +3.1% annualized alpha with a
beta of −0.66**: the signals added value, but an un-hedged **short-beta tilt (driven by the
low-volatility leg) dominated and bled during a strong high-beta bull market.** See
[Results](#results) and [Limitations](#limitations--honest-discussion).

> **One honest sentence on data:** the backtest uses *today's* Nifty 100 list and a
> price-based value *proxy* (no point-in-time fundamentals), so results carry survivorship
> bias and a weaker value factor than a fundamentals-based implementation — both disclosed below.

---

## Factor definitions

All factors are computed **point-in-time**: at each rebalance date *t*, a factor uses only
data with timestamp **≤ t**. Each factor is winsorized at ±3σ and converted to a cross-sectional
**z-score oriented so that a higher score = the long leg**.

| Factor | Definition | Long leg | Economic rationale |
|---|---|---|---|
| **Momentum** | 12-1 return: total return from *t−12m* to *t−1m* (skips the most recent month) | high momentum | Under-reaction / persistence of trends; the most-recent month is skipped to avoid short-term reversal. |
| **Low Volatility** | Trailing annualized realized vol of daily returns over ~252 trading days | **low** vol | The **low-vol anomaly**: low-risk stocks have historically delivered *better risk-adjusted* returns than CAPM predicts (CAPM says higher beta → higher return; empirically the security market line is too flat). |
| **Value (proxy)** | 60-12 **long-term reversal**: total return from *t−60m* to *t−12m*; past long-term **losers** treated as "cheap" | past LT losers | True point-in-time P/E, P/B were unavailable from the data source. Long-term reversal (De Bondt–Thaler) is the documented price-based **proxy** for value — disclosed, not a fabricated fundamentals factor. |

**Composite** = equal-weight mean of the three available factor z-scores at each date (a late-IPO
name with insufficient history for a factor is scored on whichever factors it *does* have — no
backfilling, which would inject look-ahead).

### How look-ahead was avoided
- Momentum / value windows **end at or before** *t* (verified by an assertion in `factors.py`).
- Low-vol uses daily returns strictly **≤ t**; `pct_change(fill_method=None)` so NaNs are never padded into the window.
- Weights set at *t* are **held over (t, t+1]** and earn the *next* month's realized return — the signal is lagged one period by construction (verified by a hand-calc in the Module D check).

---

## Portfolio construction

- **Ranking:** sort the cross-section by composite (or single-factor) score each month.
- **Quantiles:** **long the top 20%**, **short the bottom 20%** (quintile). Standard choice — *not* tuned to maximize Sharpe.
- **Weighting:** equal-weight within each leg.
- **Dollar-neutrality:** long leg sums to **+1**, short leg to **−1** → gross exposure 2.0, **net ≈ 0**. Portfolio return is the long-short spread, `mean(long) − mean(short)`.
- **Rebalance:** monthly; ~19 long / ~19 short names per side; positions carried until the next rebalance.
- **Turnover:** averaged **~41%/month** for the composite (~487% annualized) — high, as expected for monthly factor rotation. Low-vol turns over least (~18%), momentum most (~44%).

---

## Backtest assumptions

- **Transaction cost:** 10 bps **per side**, charged on `Σ|Δweight|` at every rebalance (so a round-trip name costs ~20 bps). The initial trade-in from cash pays a full gross-exposure setup cost — included, not hidden.
- **Benchmark:** Nifty 100 (`^CNX100`) buy-and-hold, the apples-to-apples long-only comparison for this universe. (`^CRSLDX` = Nifty 500 is also available in the loader.)
- **Risk-free rate:** assumed 0% for Sharpe/Sortino (stated).
- **Missing next-period price:** a held name with no next-month price earns **0%** that month and exits at the next rebalance. Transparent; slightly optimistic for shorts.
- **Turnover drift:** turnover is measured target-to-target (intra-month weight drift between rebalances is ignored) — a minor simplification.

---

## Results

All figures in `results/`. Metrics table in `results/metrics_summary.csv`.

### Metrics (net of 10 bps/side, rf = 0)

| Strategy | Ann. return | Vol | **Sharpe (net)** | Sharpe (gross) | Sortino | Max DD | Calmar | **Alpha (ann)** | **Beta** | Info ratio | Avg turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Momentum | +2.4% | 15.6% | **0.23** | 0.30 | 0.15 | −41% | 0.06 | **+6.9%** | −0.23 | 0.45 | 44.5% |
| Low Volatility | −11.3% | 21.1% | **−0.46** | −0.44 | −0.51 | −69% | −0.16 | +0.6% | **−0.71** | 0.04 | 18.3% |
| Value (proxy) | −1.8% | 15.4% | **−0.04** | −0.00 | −0.12 | −51% | −0.04 | +1.1% | −0.12 | 0.07 | 25.1% |
| **Composite** | **−8.1%** | 20.0% | **−0.32** | −0.27 | −0.37 | −61% | −0.13 | **+3.1%** | **−0.66** | 0.18 | 40.6% |
| Benchmark (N100) | +13.8% | 16.7% | **0.86** | 0.86 | 0.80 | −29% | 0.48 | — | 1.00 | — | — |

**Cost drag:** costs cut the composite Sharpe from **−0.27 (gross) to −0.32 (net)**; total cost
drag over the backtest ≈ **7.7%** of capital. Momentum, with the highest turnover, suffered the
largest Sharpe haircut (0.30 → 0.23).

### Factor attribution & inter-factor correlation

The standalone backtests (`factor_attribution.png`) decompose the combined result:
- **Momentum was the only profitable leg** (Sharpe +0.23, alpha +6.9%) — momentum has historically worked in Indian equities.
- **Low-volatility was the dominant drag** (Sharpe −0.46) — *not* because of poor stock selection (alpha ≈ 0) but because the long-low-vol / short-high-vol book is **implicitly short beta (−0.71)**, which is a losing posture in a bull market.
- **Value (long-term reversal) was roughly flat** at the monthly horizon.

Inter-factor correlation of the standalone **net return** streams is low — momentum↔low-vol 0.25,
momentum↔value 0.04, low-vol↔value −0.13 — i.e. the factors are genuinely diversifying. The
combined book's high volatility (20%) is inherited mainly from the low-vol leg.

### Regime dependence (rolling Sharpe)

`rolling_sharpe.png` (12-month rolling annualized Sharpe) shows there is **no all-weather
factor**: each leg swings between strongly positive and strongly negative regimes. The composite
and low-vol legs dive deeply negative around the **2020–2021 post-COVID high-beta recovery**
(when the short-beta tilt hurt most), while momentum recovers through 2022–2024.

### Information Coefficient (stretch)

`ic_analysis.png` — cross-sectional Spearman rank correlation between factor score and forward return:

| Factor | Mean 1m IC | IC IR | % positive | Decay profile |
|---|---|---|---|---|
| Momentum | **+0.020** | 0.11 | 54% | Positive; **peaks at ~3–4 months**, decays by 12m (medium-horizon signal). |
| Low Volatility | −0.010 | −0.05 | 46% | Negative and *deepening* with horizon in this regime. |
| Value (reversal) | −0.006 | −0.03 | 50% | Slightly negative at 1m but **turns positive and rises to +0.05 by 12m** — a slow, long-horizon mean-reversion signal that monthly rebalancing under-harvests. |

---

## Limitations & honest discussion

- **Survivorship bias.** The universe is *today's* Nifty 100 list applied to the past, so stocks
  that were dropped from the index (often prior losers) are excluded. This **inflates** results
  upward — yet the strategy still lost money, so the *true* result is likely no better. Direction
  of bias: optimistic.
- **Value factor is a proxy.** No point-in-time P/E, P/B were available, so value is implemented
  as 60-12 long-term reversal. This is a documented but weaker stand-in for a fundamentals-based
  value factor, and it conflates with price dynamics. A real B/M or earnings-yield factor (with the
  reporting lag below) would be a stronger test.
- **Fundamentals lag (not applicable here, but noted).** Had fundamentals been used, financials are
  reported with a lag; stamping them to the period-end date would be look-ahead. They would need to
  be lagged ~1–3 months.
- **Data gaps.** `TATAMOTORS` and `PEL` were dropped due to a Yahoo Finance outage (both recently
  demerged, disrupting their listings); `MCDOWELL-N` was mapped to its current symbol `UNITDSPR`.
  Late-IPO names carry NaNs until they have enough history (handled per-date, never backfilled).
- **Universe liquidity.** Restricted to Nifty 100 large/mid caps — deliberately liquid and tradable,
  avoiding micro-cap names that dominate factor backtests but cannot actually be traded.
- **Not beta-neutral.** The book is dollar-neutral but **not** beta-neutral; the low-vol leg makes
  it structurally short beta. This single design choice explains most of the underperformance.
- **No sector neutralization.** Rankings are not sector-adjusted, so the book can carry accidental
  sector bets.

### What I would add next
- **Beta hedging / beta-neutral construction** (overlay an index hedge so the residual is pure alpha) — given the +3.1% composite alpha, this is the single highest-value fix.
- **Sector / industry neutralization** (rank within sectors).
- **Risk-based weighting** (inverse-vol or risk-parity) instead of equal weight.
- A **quality** factor (ROE, low leverage) for a 4-factor model.
- **Fama-MacBeth** regressions to test factor significance, and a comparison against published Fama-French / momentum factor returns.

---

## How to run

```bash
# 1. install
pip install -r requirements.txt

# 2. (re)download data -> data/prices.csv, data/benchmark.csv  (cached after first run)
python src/data_loader.py

# 3. run any module's self-test (each prints sanity checks / results)
python src/factors.py        # point-in-time factor z-scores + look-ahead check
python src/portfolio.py      # weight panel + dollar-neutrality checks
python src/backtest.py       # net/gross/benchmark headline + timing check
python src/metrics.py        # full metrics table -> results/metrics_summary.csv

# 4. generate all figures + IC study -> results/*.png
python src/analysis.py
```

Modules are designed to be run from the repo root; each `src/*.py` script self-tests when run directly.

---

## Repo structure

```
multifactor-long-short-equity/
├── README.md
├── requirements.txt
├── data/
│   ├── prices.csv          # daily adjusted close, Nifty 100, 2011–2024
│   └── benchmark.csv       # Nifty 100 index (^CNX100)
├── src/
│   ├── data_loader.py      # Module A: universe, download/cache, month-end, benchmark
│   ├── factors.py          # Module B: momentum / low-vol / value(proxy) point-in-time z-scores
│   ├── portfolio.py        # Module C: rank -> dollar-neutral long-short weights
│   ├── backtest.py         # Module D: lagged P&L, turnover, costs, equity curves
│   ├── metrics.py          # Module E: Sharpe/Sortino/DD/Calmar, alpha/beta/IR, attribution
│   └── analysis.py         # Module F: figures + Information Coefficient
├── notebooks/
│   └── strategy.ipynb      # end-to-end walkthrough
└── results/
    ├── equity_curves.png
    ├── factor_attribution.png
    ├── rolling_sharpe.png
    ├── ic_analysis.png
    └── metrics_summary.csv
```

## Interview talking points this project supports
- *Why momentum works and occasionally crashes* — see the 2020 recovery in the rolling-Sharpe chart.
- *The low-vol anomaly vs CAPM* — and why a long-short low-vol book is implicitly short beta.
- *Avoiding look-ahead* — point-in-time windows, one-period signal lag, no NaN padding.
- *Turnover and cost drag* — ~41%/month, Sharpe −0.27 → −0.32 net.
- *Making it market-/beta-neutral* — the +3.1% alpha vs −0.66 beta is the whole story.
- *Factor correlation and combined Sharpe* — low correlations diversify, but a bad leg still dominates risk.
- *Survivorship bias direction* — optimistic, and the strategy still lost.
