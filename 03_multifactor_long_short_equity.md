# Project 3 — Multi-Factor Long-Short Equity Strategy

> **Build target:** Construct momentum, low-volatility, and value factors on a Nifty 500
> universe, rank stocks into a long-short portfolio, and backtest with transaction costs.
> Demonstrates portfolio construction, factor investing, and Fama-French literacy.

---

## 0. Intent (read this first)

This project shows you understand **how systematic equity strategies are actually built** at
quant funds: not predicting individual prices, but ranking a cross-section of stocks by
characteristics ("factors") and holding a diversified long-short book. It signals literacy with
the Fama-French / factor-investing literature, which is foundational reading at most systematic shops.

The discipline points that make it credible:
- Factors must be computed using **only past data** at each rebalance date (point-in-time, no look-ahead).
- The portfolio must be **rebalanced periodically**, not just constructed once.
- Costs must be charged on **turnover**.
- Performance must be reported **net of costs** and compared to a **benchmark** (Nifty 500 buy-and-hold).
- You should **attribute** returns to each factor, not just the combined strategy.

---

## 1. Learning objectives (defend these)

- What an equity "factor" is and the economic rationale for momentum, low-vol, and value.
- The Fama-French framework (market, size, value; momentum as Carhart's 4th) at a conceptual level.
- Cross-sectional ranking vs time-series prediction.
- Long-short (market-neutral-ish) vs long-only portfolio construction.
- Turnover, transaction costs, and how they erode factor returns.
- Information Coefficient (IC) and factor decay.
- Why factors have "regimes" (momentum crashes, value's lost decade) — no factor works always.

---

## 2. Tech stack

```
python >= 3.10
numpy
pandas
matplotlib
yfinance         # primary data source (.NS tickers)
scipy / statsmodels   # for IC, regressions, factor attribution
```

---

## 3. Data acquisition

**Universe:** Nifty 500 constituents (or a tractable subset — **Nifty 100 or Nifty 200 is fine
and more reliable** for data availability; state your universe choice). Get the constituent list
from an NSE index factsheet / Wikipedia / a Kaggle dataset, then append `.NS` to tickers.

```python
import yfinance as yf
prices = yf.download(tickers, start="2017-01-01", end="2024-12-31")["Close"]
```

You need ~6–8 years of **monthly** (or weekly) adjusted close prices. Monthly rebalancing keeps
turnover and cost realistic and data manageable.

**Survivorship bias warning:** if you use *today's* Nifty 500 list and backtest into the past,
you've excluded stocks that were dropped (often losers) — this inflates returns. You probably
can't fully fix this as a student, but **you must acknowledge it explicitly in the README**. That
acknowledgment alone signals sophistication.

**For value factor you need fundamentals** (P/E, P/B, earnings yield). Options:
- Pull from `yfinance` `.info` (limited, current snapshot only — weak for historical value).
- Use a Kaggle fundamentals dataset, or
- **Simplify honestly:** if good historical fundamentals are unavailable, run momentum +
  low-vol + a **proxy value factor** (e.g. long-term reversal as a value proxy) and clearly state
  the limitation. Better an honest 2-factor model than a fabricated value factor.

**Fallback:** if NSE data is painful, the entire pipeline works identically on US data (S&P 500
constituents via yfinance), and US fundamentals are easier to get. Acceptable — just state it.

---

## 4. Factor definitions (compute point-in-time, at each rebalance date)

### Momentum
- 12-month return **skipping the most recent month** (the standard "12-1" momentum to avoid
  short-term reversal): `return from t-12m to t-1m`.
- Rank stocks; high momentum → long, low → short.

### Low Volatility
- Trailing realized volatility of daily/weekly returns over the past ~12 months (annualized).
- **Low** vol → long, **high** vol → short. (The low-vol anomaly: low-vol stocks have historically
  delivered better risk-adjusted returns — explain why this is an anomaly relative to CAPM.)

### Value
- Earnings yield (E/P) or book-to-market (B/P): high → cheap → long, low → expensive → short.
- Use point-in-time fundamentals if available; otherwise use the documented proxy and disclose it.

> Each factor produces a cross-sectional **rank or z-score** at each rebalance date, computed only
> from data available **on or before** that date.

---

## 5. Implementation plan (modules)

### Module A — `data_loader.py`
- Load price panel and fundamentals; align to a monthly rebalance calendar.
- Resample to month-end. Handle missing tickers per date (a stock may not exist for the whole period).

### Module B — `factors.py`
- `momentum(prices, asof_date)`, `low_vol(prices, asof_date)`, `value(fundamentals, asof_date)`.
- Each returns a cross-sectional **z-score** per stock (winsorize extremes to reduce outlier impact).
- Combine into a composite score (equal-weight the three z-scores, or keep separate to compare).

### Module C — `portfolio.py`
- At each rebalance date: rank by composite score.
- **Long** the top quantile (e.g. top 20% / top decile), **short** the bottom quantile.
- Weighting: start equal-weight within each leg. Long and short legs sized to be roughly dollar-neutral.
- Carry positions until the next rebalance.

### Module D — `backtest.py`
- Compute portfolio returns each period from held positions and realized stock returns.
- Compute **turnover** = fraction of the book that changes at each rebalance.
- Charge transaction cost on turnover (e.g. 10–20 bps per side; state assumption).
- Build a **net-of-cost** equity curve. Also build the **benchmark** (Nifty 500/100 buy-and-hold).

### Module E — `metrics.py`
Report (net of costs):
- Annualized return, volatility, **Sharpe**, **Sortino**, **max drawdown**, Calmar.
- **vs benchmark**: alpha, beta (regress strategy returns on benchmark returns), information ratio.
- Average turnover and total cost drag (gross vs net Sharpe — show how much cost ate).
- **Per-factor attribution**: backtest each factor standalone (momentum-only, low-vol-only,
  value-only) and show the combined vs individual. Discuss correlation/diversification between factors.

### Module F — `analysis.py` / notebook
- Equity curves (each factor + combined + benchmark on one chart).
- Rolling Sharpe / rolling returns to expose **regime dependence** (when did momentum crash? did
  low-vol cushion drawdowns?).
- (Stretch) **Information Coefficient**: rank correlation between factor scores and next-period
  returns over time, and its decay.

---

## 6. Pitfalls to avoid

- **Look-ahead / point-in-time violation**: using a stock's full-period volatility or a return
  that extends past the rebalance date. Every factor value at date *t* must use data ≤ *t*.
- **Survivorship bias**: using only current constituents (disclose if unfixable).
- **Ignoring turnover/costs**: factor strategies, especially momentum, have high turnover; gross
  returns are misleading.
- **Look-ahead in fundamentals**: financials are reported with a lag. If you stamp Q4 earnings to
  the quarter-end date you've used data the market didn't have yet — lag fundamentals by ~1–3 months.
- **Tiny universe / illiquid names**: micro-caps dominate factor backtests but can't be traded;
  restrict to liquid large/mid caps and say so.
- **Overfitting the quantile/rebalance choices**: don't tune top-X% and rebalance frequency to
  maximize Sharpe; use standard choices (deciles, monthly) and report.

---

## 7. Deliverables & repo structure

```
multifactor-long-short-equity/
├── README.md
├── requirements.txt
├── data/
│   ├── prices.csv
│   └── fundamentals.csv
├── src/
│   ├── data_loader.py
│   ├── factors.py
│   ├── portfolio.py
│   ├── backtest.py
│   ├── metrics.py
│   └── analysis.py
├── notebooks/
│   └── strategy.ipynb
└── results/
    ├── equity_curves.png
    ├── factor_attribution.png
    ├── rolling_sharpe.png
    └── metrics_summary.csv
```

---

## 8. README template

```
# Multi-Factor Long-Short Equity Strategy (Indian Equities)

## Summary
One paragraph: universe, factors used, rebalance frequency, headline NET Sharpe and max drawdown,
and how it compared to the benchmark. One honest sentence on survivorship bias / data limits.

## Factor definitions
- Momentum (12-1), Low Volatility, Value — exact formulas and economic rationale
- Point-in-time computation (how look-ahead was avoided)

## Portfolio construction
- Ranking, quantiles, long-short legs, weighting, dollar-neutrality
- Rebalance frequency, turnover

## Backtest assumptions
- Transaction cost (bps), how charged on turnover, benchmark definition

## Results
- Equity curves: each factor + combined + benchmark
- Metrics table: gross vs net (show cost drag), alpha/beta/IR vs benchmark
- Factor attribution and inter-factor correlation
- Rolling Sharpe → regime discussion

## Limitations & honest discussion
- Survivorship bias, fundamentals lag, universe liquidity, factor regimes
- What you'd add (sector neutralization, risk-model-based weighting, more factors)

## How to run
```

---

## 9. Stretch goals

- **Sector/industry neutralization**: rank within sectors so the strategy isn't an accidental
  sector bet — a standard professional refinement.
- **Risk-based weighting** (inverse-vol or simple risk-parity) instead of equal weight.
- Add a **quality** factor (ROE, low leverage) for a 4-factor model.
- **Fama-MacBeth** style regressions to test factor significance — strong academic signal.
- Compare your low-vol/value/momentum loadings against published Fama-French factor returns.

---

## 10. Interview talking points

- "Why does momentum work, and why does it occasionally crash?"
- "The low-volatility anomaly contradicts CAPM — explain."
- "How did you avoid look-ahead bias with fundamentals?"
- "What's your turnover, and how much did costs reduce your Sharpe?"
- "How would you make this market-neutral / sector-neutral?"
- "Your factors are correlated/uncorrelated — what does that mean for the combined Sharpe?"
- "Did you handle survivorship bias? If not, which direction does it bias your results?"
