# Session Summary — Multi-Factor Long-Short Equity Strategy

**Date:** 2026-06-04/05  
**Repo:** https://github.com/Abhi241-bot/Quantfin3 (branch `main`, all work pushed)

---

## 1. What was built (all modules complete)

| Module | File | Status |
|---|---|---|
| A | `src/data_loader.py` | ✅ Nifty 100 universe (98 names), daily adj-close download + cache, month-end resample, Nifty 100 benchmark (`^CNX100`). |
| B | `src/factors.py` | ✅ Point-in-time z-scores: momentum (12-1), low-vol (252d), value (60-12 long-term-reversal **proxy**). Winsorized ±3σ, oriented higher=long. |
| C | `src/portfolio.py` | ✅ Rank → long top 20% / short bottom 20%, equal-weight, dollar-neutral. `build_all_panels` (one-pass, all signals). |
| D | `src/backtest.py` | ✅ One-period-lag P&L, turnover (½·Σ\|Δw\|), 10 bps/side cost, gross/net/benchmark equity curves. |
| E | `src/metrics.py` | ✅ Sharpe/Sortino/MaxDD/Calmar, alpha/beta/IR (OLS regression), full per-factor attribution + correlation → `results/metrics_summary.csv`. |
| F | `src/analysis.py` | ✅ Equity curves, factor attribution, rolling Sharpe, Information Coefficient + decay → 4 PNGs in `results/`. |
| Docs | `README.md`, `notebooks/strategy.ipynb` | ✅ README with honesty section; notebook executed end-to-end (0 errors). |

**Discipline checks performed and passed:**
- No look-ahead: every factor window ends ≤ rebalance date (asserted in `factors.py`); `pct_change(fill_method=None)` prevents NaN padding into the vol window.
- Signal lagged one period: weights at *t* earn the *t+1* realized return — **verified by hand-calc** (manual gross = backtest gross to 6 dp; P&L date strictly after decision date).
- Costs charged on every position change (incl. initial trade-in from cash).
- No parameter tuning: standard quintiles + monthly rebalance kept as-is despite negative result.

---

## 2. Current key numbers (net of 10 bps/side, rf=0, 2017-02 → 2024-12, 95 months)

| Strategy | Ann. ret | Sharpe (net) | Sharpe (gross) | Alpha (ann) | Beta | IR | Max DD | Avg turnover |
|---|---|---|---|---|---|---|---|---|
| Momentum | +2.4% | **0.23** | 0.30 | +6.9% | −0.23 | 0.45 | −41% | 44.5% |
| Low Vol | −11.3% | **−0.46** | −0.44 | +0.6% | **−0.71** | 0.04 | −69% | 18.3% |
| Value (proxy) | −1.8% | **−0.04** | −0.00 | +1.1% | −0.12 | 0.07 | −51% | 25.1% |
| **Composite** | **−8.1%** | **−0.32** | −0.27 | **+3.1%** | **−0.66** | 0.18 | −61% | 40.6% |
| Benchmark N100 | +13.8% | **0.86** | 0.86 | — | 1.00 | — | −29% | — |

- **Cost drag:** composite Sharpe −0.27 (gross) → −0.32 (net); ~7.7% of capital total.
- **IC (1-month, full sample):** momentum +0.020 (peaks ~3–4m), low_vol −0.010 (deepens with horizon), value −0.006 at 1m but **+0.05 by 12m** (slow long-horizon reversal).
- **Headline narrative:** market-neutral book lost money, BUT composite has **+3.1% alpha vs −0.66 beta** — signals add value; an un-hedged short-beta tilt (low-vol leg) bled in a high-beta bull market. Honest, defensible, fully attributed.

---

## 3. Assumptions / deviations from spec (and why)

- **Universe = Nifty 100, not Nifty 500.** Spec explicitly permits Nifty 100/200 for data reliability. 98 names kept.
- **Value = 60-12 long-term-reversal proxy**, not real fundamentals. yfinance gives only a current snapshot; stamping it to past dates = look-ahead. Spec sanctions the documented proxy. Disclosed.
- **Price history loaded from 2011-06** (not 2017): needed so the 60-month value window is full at the first 2017 rebalance — a data-coverage fix, not tuning. Backtest still starts 2017.
- **`TATAMOTORS` and `PEL` dropped** (Yahoo data outage, both recently demerged); **`MCDOWELL-N` → `UNITDSPR`** (renamed). Documented as data limitations.
- **Missing next-period price → 0% that month, exit next rebalance** (user-chosen, simplest/transparent).
- **Cost = 10 bps/side** (toward the conservative-standard end of the spec's 10–20 bps range).
- **Survivorship bias is present** (today's index list) — unfixable as a student; disclosed in README (bias direction = optimistic, yet strategy still lost).
- **Jupyter:** registered a `quantfin314` kernel because the default kernel (Py 3.13) lacks statsmodels; the project's verified env is Python 3.14.

---

## 4. Where to pick up next session

**The project is complete and all deliverables are pushed.** If continuing, the highest-value
**stretch goals** (in priority order) are:

1. **Beta-neutral construction / index hedge** — the single best fix. Composite has +3.1% alpha but
   −0.66 beta; overlay a Nifty 100 hedge to isolate the alpha. Add to `portfolio.py` (scale legs to
   net-beta 0 using trailing betas) → re-run `metrics.py`/`analysis.py`. *Start here.*
2. **Sector neutralization** — rank within GICS/NSE sector buckets in `factors.py` so the book isn't
   an accidental sector bet. Needs a sector-map (hardcode for Nifty 100 or pull from yfinance `.info`).
3. **Risk-based weighting** — inverse-vol weights within legs instead of equal-weight (`rank_to_weights`).
4. **Quality factor** (ROE, low leverage) for a 4-factor model — requires fundamentals (data gap caveat).
5. **Fama-MacBeth regressions** — cross-sectional factor significance test (`metrics.py`).

**Nothing is broken; no half-finished module.** Each `src/*.py` self-tests via `python src/<file>.py`,
and `notebooks/strategy.ipynb` runs end-to-end on the `quantfin314` kernel.
