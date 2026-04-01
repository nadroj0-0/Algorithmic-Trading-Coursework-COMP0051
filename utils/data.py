# =============================================================================
# utils/data.py — COMP0051 Algorithmic Trading Coursework
# Data download, cleaning, and preparation pipeline.
#
# Assets    : BTC/USDT, ETH/USDT, SOL/USDT
# Source    : Binance API via ccxt (no API key required for public OHLCV)
# Frequency : 1-hour bars
# Period    : 2024-01-01 → 2024-12-31
# Storage   : Parquet (raw) + Parquet (cleaned)
#
# Usage:
#   python data.py               — downloads and cleans all data
#   from utils.data import load_clean, load_returns, get_close_matrix
# =============================================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt


# =============================================================================
# CONFIG
# =============================================================================

# BTC/ETH: required by brief. SOL: top-5 L1 with the strongest trending
# behaviour in 2024 ($20→$200) — academically defensible, highest signal quality.
SYMBOLS   = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
SINCE     = "2024-01-01"
UNTIL     = "2024-12-31"
DATA_DIR  = Path("./data")

# Binance column order returned by ccxt
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# =============================================================================
# 1. DOWNLOAD
# =============================================================================

def _fetch_ohlcv(symbol: str, timeframe: str, since: str, until: str) -> pd.DataFrame:
    """
    Fetch full OHLCV history for a single symbol from Binance.
    Handles pagination automatically — Binance returns max 1000 bars per request.

    Args:
        symbol    : e.g. "BTC/USDT"
        timeframe : e.g. "1h"
        since     : start date string "YYYY-MM-DD"
        until     : end date string "YYYY-MM-DD"

    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex (UTC).
    """
    exchange  = ccxt.binance({"enableRateLimit": True})
    since_ms  = exchange.parse8601(f"{since}T00:00:00Z")
    until_ms  = exchange.parse8601(f"{until}T23:59:59Z")
    all_bars  = []

    print(f"[data] Fetching {symbol} {timeframe} {since} → {until} ...")

    while since_ms < until_ms:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        since_ms = bars[-1][0] + 1          # advance past last fetched bar
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_bars, columns=OHLCV_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Trim to requested window (Binance may overshoot)
    df = df[df.index <= pd.Timestamp(until, tz="UTC") + pd.Timedelta(hours=23)]

    print(f"[data] {symbol}: {len(df):,} bars downloaded")
    return df


def download_all(
    symbols:   list[str] = SYMBOLS,
    timeframe: str       = TIMEFRAME,
    since:     str       = SINCE,
    until:     str       = UNTIL,
    data_dir:  Path      = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Download raw OHLCV data for all symbols and save to parquet.
    Skips download if raw file already exists (re-run safe).

    Returns:
        dict mapping symbol → raw DataFrame
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = {}

    for symbol in symbols:
        fname = data_dir / f"{symbol.replace('/', '_')}_raw.parquet"

        if fname.exists():
            print(f"[data] Found cached raw data: {fname}")
            raw[symbol] = pd.read_parquet(fname)
        else:
            df = _fetch_ohlcv(symbol, timeframe, since, until)
            df.to_parquet(fname)
            print(f"[data] Saved raw → {fname}")
            raw[symbol] = df

    return raw


# =============================================================================
# 2. CLEAN
# =============================================================================

def _clean_single(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Clean a single asset's OHLCV DataFrame.

    Steps:
        1. Remove duplicate timestamps
        2. Reindex to complete hourly grid — makes any gaps explicit
        3. Forward-fill missing OHLCV values (exchange downtime / missing bars)
        4. Detect and repair price outliers:
             - Flag bars where |log return| > 5σ of rolling 168-bar window
             - Replace flagged close with linearly interpolated value
             - Align open/high/low to repaired close on flagged bars
        5. Clip volume to [0, 99.9th percentile] — removes data entry errors

    Args:
        df     : Raw OHLCV DataFrame with DatetimeIndex (UTC)
        symbol : Asset name for logging

    Returns:
        Cleaned DataFrame with same columns.
    """
    df = df.copy()

    # --- 1. Remove duplicates ---
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        print(f"[clean] {symbol}: removed {n_dupes} duplicate timestamps")
        df = df[~df.index.duplicated(keep="first")]

    # --- 2. Complete hourly grid + forward fill ---
    full_idx  = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    n_missing = len(full_idx) - len(df)
    if n_missing:
        print(f"[clean] {symbol}: forward-filling {n_missing} missing bars")
    df = df.reindex(full_idx).ffill()

    # --- 3. Outlier detection and repair on close price ---
    log_ret    = np.log(df["close"] / df["close"].shift(1))
    roll_std   = log_ret.rolling(168, min_periods=24).std()
    outlier_mask = log_ret.abs() > 5 * roll_std
    n_outliers   = int(outlier_mask.sum())
    if n_outliers:
        print(f"[clean] {symbol}: interpolating {n_outliers} price outliers (>5σ)")
        df.loc[outlier_mask, "close"] = np.nan
        df["close"] = df["close"].interpolate(method="linear")
        # Align open/high/low to repaired close where flagged
        df.loc[outlier_mask, "open"] = df.loc[outlier_mask, "close"]
        df.loc[outlier_mask, "high"] = df.loc[outlier_mask, "close"]
        df.loc[outlier_mask, "low"]  = df.loc[outlier_mask, "close"]

    # --- 4. Clip volume extremes ---
    vol_cap = df["volume"].quantile(0.999)
    n_vol   = int((df["volume"] > vol_cap).sum())
    if n_vol:
        print(f"[clean] {symbol}: clipping {n_vol} volume outliers to 99.9th pct")
    df["volume"] = df["volume"].clip(upper=vol_cap)

    print(f"[clean] {symbol}: {len(df):,} bars after cleaning")
    return df


def clean_all(
    raw:      dict[str, pd.DataFrame],
    data_dir: Path = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Clean all raw DataFrames and save cleaned versions to parquet.

    Returns:
        dict mapping symbol → cleaned DataFrame
    """
    cleaned = {}
    for symbol, df in raw.items():
        df_clean = _clean_single(df, symbol)
        fname    = data_dir / f"{symbol.replace('/', '_')}_clean.parquet"
        df_clean.to_parquet(fname)
        print(f"[data] Saved clean → {fname}\n")
        cleaned[symbol] = df_clean
    return cleaned

def fetch_risk_free_rate(
    index: pd.DatetimeIndex,
    annual_default: float = 0.053,
) -> pd.Series:
    """
    Fetch Fed Funds rate from FRED and align to index.
    Falls back to constant rate if download fails.
    """
    try:
        import pandas_datareader.data as web

        start = index.min().date()
        end   = index.max().date()

        rf = web.DataReader("FEDFUNDS", "fred", start, end)
        rf = rf.ffill() / 100.0  # convert % → decimal

        # convert annual → hourly
        rf_hourly = (1 + rf) ** (1 / 8760) - 1

        # align to our index
        rf_hourly = rf_hourly.reindex(index, method="ffill")

        print("[rf] Loaded Fed Funds rate from FRED")
        return rf_hourly.iloc[:, 0]

    except Exception as e:
        print(f"[rf] FRED fetch failed ({e}), using constant rf={annual_default}")
        rf_hourly = (1 + annual_default) ** (1 / 8760) - 1
        return pd.Series(rf_hourly, index=index)

# =============================================================================
# 3. RETURNS
# =============================================================================

def compute_returns(
    cleaned:   dict[str, pd.DataFrame],
    rf_annual: float = 0.053,
) -> dict[str, pd.DataFrame]:
    """
    Compute simple excess returns for each asset.

    Excess return at time t:
        r_excess_t = (close_t - close_{t-1}) / close_{t-1}  −  rf_{t-1}

    Risk-free rate (annualised) converted to per-hour:
        rf_hourly = (1 + rf_annual)^(1/8760) − 1

    At 1h frequency, rf_hourly ≈ 6e-6 per bar, negligible vs crypto vol
    (σ ≈ 0.003–0.005 per bar). Included for academic correctness.
    Default rf_annual=0.053 reflects the 2024 effective Fed Funds rate.

    Args:
        cleaned   : dict of cleaned OHLCV DataFrames
        rf_annual : annualised risk-free rate

    Returns:
        dict mapping symbol → DataFrame with additional columns:
            ret        : simple gross return
            ret_excess : simple excess return (net of rf)
            log_ret    : log return (for statistical tests / ACF)
    """
    # rf_hourly = (1 + rf_annual) ** (1 / 8_760) - 1
    # print(f"[returns] rf_annual={rf_annual:.3f} → rf_hourly={rf_hourly:.2e} (effectively zero at this frequency)")
    print(f"[returns] Using time-varying risk-free rate (FRED) with fallback rf_annual={rf_annual:.3f}")
    common_index = next(iter(cleaned.values())).index
    rf_series = fetch_risk_free_rate(common_index, rf_annual)
    result = {}
    for symbol, df in cleaned.items():
        df = df.copy()
        df["ret"]        = df["close"].pct_change()
        # df["ret_excess"] = df["ret"] - rf_hourly
        # rf_series = fetch_risk_free_rate(df.index, rf_annual)
        df["ret_excess"] = df["ret"] - rf_series
        df["log_ret"]    = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna(subset=["ret"])
        result[symbol]   = df
        print(f"[returns] {symbol}: {len(df):,} return observations")
        print(
            f"[data-check] {symbol}: "
            f"{df.index.min()} → {df.index.max()} | {len(df):,} rows"
        )
        print(f"[rf] {symbol}: mean rf ≈ {rf_series.mean():.2e}")

    return result


# =============================================================================
# 4. LOAD HELPERS (called by strategy scripts and run.py)
# =============================================================================

def load_clean(
    symbols:  list[str] = SYMBOLS,
    data_dir: Path      = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Load cleaned OHLCV data from parquet.
    Raises FileNotFoundError if file not found — run data.py first.

    Returns:
        dict mapping symbol → cleaned DataFrame
    """
    result = {}
    for symbol in symbols:
        fname = data_dir / f"{symbol.replace('/', '_')}_clean.parquet"
        if not fname.exists():
            raise FileNotFoundError(
                f"Clean data not found for {symbol} at {fname}. "
                f"Run `python data.py` first to download and clean."
            )
        result[symbol] = pd.read_parquet(fname)
    return result


def load_returns(
    timeframe: str,
    since:     str,
    until:     str,
    symbols:   list[str] = SYMBOLS,
    data_dir:  Path      = DATA_DIR,
    rf_annual: float     = 0.053,
) -> dict[str, pd.DataFrame]:
    """
    Load cleaned data and compute returns in one call.
    Convenience wrapper used by run.py upfront data loading.

    Returns:
        dict mapping symbol → DataFrame with ret, ret_excess, log_ret columns
    """
    # Step 1: download (if needed)
    raw = download_all(
    symbols   = symbols,
    timeframe = timeframe,
    since     = since,
    until     = until,
    data_dir  = data_dir,
)

    # Step 2: clean
    cleaned = clean_all(raw, data_dir)

    # Step 3: compute returns
    return compute_returns(cleaned, rf_annual=rf_annual)


def get_close_matrix(
    data: dict[str, pd.DataFrame],
    col:  str = "close",
) -> pd.DataFrame:
    """
    Stack close prices (or any column) from multiple assets into a single
    aligned DataFrame. Used by cointegration tests and MVO.

    Returns:
        DataFrame with one column per symbol, aligned on shared timestamps.
        Rows with any NaN dropped.
    """
    frames = {symbol: df[col] for symbol, df in data.items()}
    out    = pd.DataFrame(frames).dropna()
    print(f"[data] Close matrix: {out.shape[0]:,} rows × {out.shape[1]} assets")
    return out


def get_returns_matrix(
    data: dict[str, pd.DataFrame],
    col:  str = "ret_excess",
) -> pd.DataFrame:
    """
    Stack returns (excess by default) for all assets into an aligned DataFrame.
    Used by MVO covariance estimation and portfolio.py PnL engine.

    Returns:
        DataFrame with one column per symbol, first row dropped (NaN from pct_change).
    """
    frames = {symbol: df[col] for symbol, df in data.items()}
    out    = pd.DataFrame(frames).dropna()
    return out


# =============================================================================
# 5. SUMMARY STATS (sanity check — called by data.py __main__)
# =============================================================================

def print_summary(data: dict[str, pd.DataFrame]) -> None:
    """
    Print basic summary statistics for each asset.
    Run after compute_returns to verify data looks sensible.
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    for symbol, df in data.items():
        ret = df["ret"].dropna()
        print(f"\n{symbol}")
        print(f"  Period     : {df.index.min().date()} → {df.index.max().date()}")
        print(f"  Bars       : {len(df):,}")
        print(f"  Close range: ${df['close'].min():,.2f} → ${df['close'].max():,.2f}")
        print(f"  Mean ret   : {ret.mean():.6f}  ({ret.mean() * 8_760 * 100:.1f}% annualised)")
        print(f"  Std ret    : {ret.std():.6f}   ({ret.std() * np.sqrt(8_760) * 100:.1f}% ann. vol)")
        print(f"  Min ret    : {ret.min():.4f}")
        print(f"  Max ret    : {ret.max():.4f}")
        print(f"  Skew       : {ret.skew():.3f}")
        print(f"  Kurt       : {ret.kurt():.3f}")
    print()


# =============================================================================
# MAIN — run standalone to download and clean everything
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMP0051 — Data Download and Cleaning Pipeline")
    print(f"Assets    : {SYMBOLS}")
    print(f"Frequency : {TIMEFRAME}")
    print(f"Period    : {SINCE} → {UNTIL}")
    print("=" * 60 + "\n")

    raw     = download_all()
    cleaned = clean_all(raw)
    data    = compute_returns(cleaned)
    print_summary(data)

    print("=" * 60)
    print("Done. Cleaned data saved to ./data/")
    print("Use load_clean() or load_returns() in strategy scripts.")
    print("=" * 60)
