"""
Module A -- data_loader.py

Responsibilities:
  - Define the investable universe (Nifty 100 large/mid caps, '.NS' tickers).
  - Download daily adjusted-close prices from yfinance.
  - Cache to data/prices.csv so we don't re-hit the network every run.
  - Provide a month-end resampled panel for the rebalance calendar.
  - Print basic sanity checks (shape, date range, null counts, head).

Design notes / discipline:
  - We download DAILY prices because the low-volatility factor needs daily
    returns over a trailing window. Momentum and the rebalance calendar use the
    month-end resample derived from the same daily panel.
  - yfinance auto_adjust=True returns split/dividend-adjusted close in 'Close'.
  - Universe = TODAY's Nifty 100 list => survivorship bias (disclosed in README).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRICES_CSV = os.path.join(DATA_DIR, "prices.csv")
BENCHMARK_CSV = os.path.join(DATA_DIR, "benchmark.csv")

# Benchmark = Nifty 100 (matches our Nifty 100 universe for an apples-to-apples
# buy-and-hold comparison). yfinance symbol '^CNX100'. (^CRSLDX = Nifty 500.)
BENCHMARK_SYMBOL = "^CNX100"

# --------------------------------------------------------------------------- #
# Universe: Nifty 100 constituents (current list -> survivorship bias, see README)
# Liquid large/mid caps; '.NS' is the NSE suffix used by yfinance.
# --------------------------------------------------------------------------- #
NIFTY100 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE", "HCLTECH", "AXISBANK",
    "MARUTI", "ASIANPAINT", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
    "ONGC", "NTPC", "NESTLEIND", "POWERGRID", "M&M", "TATASTEEL",
    "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "GRASIM",
    "HINDALCO", "TECHM", "BRITANNIA", "CIPLA", "DRREDDY", "EICHERMOT",
    "HEROMOTOCO", "DIVISLAB", "BAJAJ-AUTO", "INDUSINDBK", "APOLLOHOSP",
    "TATACONSUM", "BPCL", "HDFCLIFE", "SBILIFE", "DABUR", "GAIL",
    "GODREJCP", "HAVELLS", "DLF", "VEDL", "AMBUJACEM", "SHREECEM", "PIDILITIND",
    "SIEMENS", "BERGEPAINT", "MARICO", "BANKBARODA", "PNB", "IOC", "ICICIPRULI",
    "ICICIGI", "DMART", "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM",
    "COLPAL", "UNITDSPR", "BOSCHLTD", "MOTHERSON", "BEL", "HAL", "INDIGO",
    "NAUKRI", "PAGEIND", "MUTHOOTFIN", "CHOLAFIN", "BANDHANBNK", "IDFCFIRSTB",
    "FEDERALBNK", "TVSMOTOR", "ASHOKLEY", "BALKRISIND", "MRF", "ABBOTINDIA",
    "ALKEM", "GLAND", "PETRONET", "TATAPOWER", "ADANIPOWER", "ZYDUSLIFE",
    "NMDC", "SAIL", "ACC", "JUBLFOOD",
]


def to_yf_tickers(symbols: list[str]) -> list[str]:
    """Append the NSE '.NS' suffix used by yfinance."""
    return [f"{s}.NS" for s in symbols]


# --------------------------------------------------------------------------- #
# Download / cache
# --------------------------------------------------------------------------- #
def download_prices(
    start: str = "2016-06-01",
    end: str = "2024-12-31",
    symbols: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download daily adjusted-close prices and cache to data/prices.csv.

    We start 2016-06 (not 2017-01) so the first real rebalance in 2017 already
    has a full 12-month trailing window for momentum / low-vol (no warm-up gap).

    Returns a wide DataFrame: index = date, columns = ticker, values = close.
    """
    if symbols is None:
        symbols = NIFTY100

    if (not force) and os.path.exists(PRICES_CSV):
        print(f"[data_loader] Loading cached prices from {PRICES_CSV}")
        return load_prices()

    import yfinance as yf

    tickers = to_yf_tickers(symbols)
    print(f"[data_loader] Downloading {len(tickers)} tickers {start} -> {end} ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns a column MultiIndex (field, ticker) for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:  # single ticker edge case
        close = raw[["Close"]].copy()

    # Strip the '.NS' suffix back to plain symbols for readability downstream.
    close.columns = [c.replace(".NS", "") for c in close.columns]
    close = close.sort_index()

    close = drop_empty_tickers(close)

    os.makedirs(DATA_DIR, exist_ok=True)
    close.to_csv(PRICES_CSV)
    print(f"[data_loader] Saved {close.shape} price panel to {PRICES_CSV}")
    return close


def drop_empty_tickers(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Drop tickers with zero usable data (Yahoo outages / bad symbols). Tickers
    that merely start late (IPOs) are KEPT -- their NaNs are handled per-date
    downstream so we don't accidentally introduce look-ahead by backfilling.
    """
    empty = prices.columns[prices.notna().sum() == 0].tolist()
    if empty:
        print(f"[data_loader] Dropping {len(empty)} empty tickers: {empty}")
        prices = prices.drop(columns=empty)
    return prices


def load_prices() -> pd.DataFrame:
    """Load the cached daily price panel from CSV."""
    df = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    df = df.sort_index()
    return drop_empty_tickers(df)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def download_benchmark(
    start: str = "2016-06-01",
    end: str = "2024-12-31",
    force: bool = False,
) -> pd.Series:
    """Download the benchmark index daily close and cache to data/benchmark.csv."""
    if (not force) and os.path.exists(BENCHMARK_CSV):
        print(f"[data_loader] Loading cached benchmark from {BENCHMARK_CSV}")
        return load_benchmark()

    import yfinance as yf

    print(f"[data_loader] Downloading benchmark {BENCHMARK_SYMBOL} {start} -> {end} ...")
    raw = yf.download(BENCHMARK_SYMBOL, start=start, end=end,
                      auto_adjust=True, progress=False)
    bench = raw["Close"].squeeze()
    bench.name = "benchmark"

    os.makedirs(DATA_DIR, exist_ok=True)
    bench.to_csv(BENCHMARK_CSV)
    print(f"[data_loader] Saved benchmark ({bench.notna().sum()} obs) to {BENCHMARK_CSV}")
    return bench


def load_benchmark() -> pd.Series:
    """Load cached benchmark series."""
    s = pd.read_csv(BENCHMARK_CSV, index_col=0, parse_dates=True).squeeze("columns")
    s.name = "benchmark"
    return s.sort_index()


# --------------------------------------------------------------------------- #
# Resample to month-end (the rebalance calendar)
# --------------------------------------------------------------------------- #
def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end using the last available observation in
    each month ('ME' = month-end). This is the price panel on which monthly
    returns and the rebalance calendar are built.
    """
    return prices.resample("ME").last()


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def sanity_report(prices: pd.DataFrame, label: str = "DAILY") -> None:
    """Print shape, date range, null counts, coverage and head()."""
    print("\n" + "=" * 70)
    print(f"SANITY REPORT [{label}]")
    print("=" * 70)
    print(f"Shape (rows x tickers): {prices.shape}")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Tickers: {prices.shape[1]}")

    # Per-ticker non-null counts -> spot tickers with little/no data.
    non_null = prices.notna().sum().sort_values()
    n_rows = len(prices)
    fully_missing = non_null[non_null == 0]
    sparse = non_null[(non_null > 0) & (non_null < 0.5 * n_rows)]

    print(f"\nTotal cells: {prices.size:,}  |  Null cells: {prices.isna().sum().sum():,} "
          f"({prices.isna().sum().sum() / prices.size:.1%})")
    print(f"Tickers with ZERO data: {len(fully_missing)} "
          f"{list(fully_missing.index) if len(fully_missing) else ''}")
    print(f"Tickers with <50% coverage: {len(sparse)} "
          f"{list(sparse.index) if len(sparse) else ''}")

    print("\nLeast-covered tickers (non-null row count):")
    print(non_null.head(8).to_string())

    print("\nhead():")
    with pd.option_context("display.max_columns", 6, "display.width", 120):
        print(prices.iloc[:5, :6])

    print("\ntail():")
    with pd.option_context("display.max_columns", 6, "display.width", 120):
        print(prices.iloc[-3:, :6])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    daily = download_prices()
    sanity_report(daily, "DAILY")

    monthly = to_month_end(daily)
    sanity_report(monthly, "MONTH-END")

    bench = download_benchmark()
    bench_me = bench.resample("ME").last()
    print(f"\n[benchmark] {BENCHMARK_SYMBOL}: {bench.notna().sum()} daily obs, "
          f"{bench.index.min().date()} -> {bench.index.max().date()}")
    print(f"[benchmark] month-end total return over sample: "
          f"{bench_me.dropna().iloc[-1] / bench_me.dropna().iloc[0] - 1:.1%}")
