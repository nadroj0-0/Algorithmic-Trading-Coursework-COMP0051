# =============================================================================
# utils/metrics.py — COMP0051 Algorithmic Trading Coursework
# Performance metrics for strategy evaluation.
#
# All functions accept a pandas Series of period returns (not prices).
# Convention: returns are simple returns (not log) unless stated otherwise.
#
# Annualisation: HOURS_PER_YEAR = 8760 (365 × 24) for hourly bars.
#
# Brief mapping:
#   Section 3 → roll_spread(), roll_spread_pct()
#   Section 4 → sharpe(), sortino(), calmar(), transaction_costs(),
#               turnover(), avg_holding_horizon(), summary()
#
# Usage:
#   from utils.metrics import sharpe, sortino, calmar, max_drawdown, summary
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

HOURS_PER_YEAR = 8_760   # 365 × 24 — for annualising hourly returns


# =============================================================================
# 1. DRAWDOWN
# =============================================================================

def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series from a returns series.

    Drawdown at time t is the percentage decline from the running peak
    of the cumulative wealth index up to t.

    Args:
        returns : Series of simple period returns.

    Returns:
        Series of drawdown values (0 = at peak, -0.30 = 30% below peak).
    """
    wealth   = (1 + returns).cumprod()
    peak     = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown — largest peak-to-trough decline in cumulative wealth.

    Returns:
        float: Maximum drawdown as a negative decimal (e.g. -0.35 = -35%).
    """
    return float(drawdown_series(returns).min())


def avg_drawdown(returns: pd.Series) -> float:
    """
    Average drawdown — mean of all drawdown values below zero.

    Returns:
        float: Average drawdown as a negative decimal.
    """
    dd         = drawdown_series(returns)
    below_peak = dd[dd < 0]
    if below_peak.empty:
        return 0.0
    return float(below_peak.mean())


# =============================================================================
# 2. RISK-ADJUSTED RETURN RATIOS
# =============================================================================

def sharpe(
    returns:       pd.Series,
    periods:       int   = HOURS_PER_YEAR,
    rf_per_period: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Sharpe = (mean_excess_return / std_return) × √(periods_per_year)

    If returns already have the risk-free rate subtracted (excess returns),
    pass rf_per_period=0.0 (default). Otherwise pass the per-period rf.

    Args:
        returns        : Series of simple period returns.
        periods        : Number of periods per year. Default 8760 (hourly).
        rf_per_period  : Per-period risk-free rate. Default 0.0.

    Returns:
        float: Annualised Sharpe ratio. NaN if std is zero.
    """
    excess = returns - rf_per_period
    mean_e = excess.mean()
    std_e  = excess.std(ddof=1)
    if std_e == 0 or np.isnan(std_e):
        return np.nan
    return float((mean_e / std_e) * np.sqrt(periods))


def sortino(
    returns:       pd.Series,
    periods:       int   = HOURS_PER_YEAR,
    rf_per_period: float = 0.0,
    target:        float = 0.0,
) -> float:
    """
    Annualised Sortino ratio.

    Penalises only downside volatility (returns below target).
    Sortino = (mean_excess_return / downside_std) × √(periods_per_year)

    Args:
        returns        : Series of simple period returns.
        periods        : Number of periods per year.
        rf_per_period  : Per-period risk-free rate.
        target         : Minimum acceptable return per period. Default 0.0.

    Returns:
        float: Annualised Sortino ratio. NaN if downside std is zero.
    """
    excess       = returns - rf_per_period
    mean_e       = excess.mean()
    downside     = excess[excess < target] - target
    downside_var = (downside ** 2).mean()
    downside_std = np.sqrt(downside_var)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float((mean_e / downside_std) * np.sqrt(periods))


def calmar(
    returns: pd.Series,
    periods: int = HOURS_PER_YEAR,
) -> float:
    """
    Calmar ratio = annualised_return / |max_drawdown|

    Measures return per unit of maximum drawdown risk.
    Higher is better. NaN if max drawdown is zero.

    Args:
        returns : Series of simple period returns.
        periods : Number of periods per year.

    Returns:
        float: Calmar ratio.
    """
    ann_return = annualised_return(returns, periods)
    mdd        = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann_return / abs(mdd))


# =============================================================================
# 3. RETURN STATISTICS
# =============================================================================

def annualised_return(
    returns: pd.Series,
    periods: int = HOURS_PER_YEAR,
) -> float:
    """
    Compound annualised growth rate (CAGR).

    CAGR = (1 + total_return)^(periods_per_year / n_periods) − 1

    Args:
        returns : Series of simple period returns.
        periods : Number of periods per year.

    Returns:
        float: Annualised return as a decimal (0.15 = 15%).
    """
    n            = len(returns)
    total_return = (1 + returns).prod() - 1
    if n == 0:
        return np.nan
    return float((1 + total_return) ** (periods / n) - 1)


def annualised_volatility(
    returns: pd.Series,
    periods: int = HOURS_PER_YEAR,
) -> float:
    """
    Annualised volatility (standard deviation of returns × √periods).

    Returns:
        float: Annualised volatility as a decimal (0.80 = 80% annual vol).
    """
    return float(returns.std(ddof=1) * np.sqrt(periods))


def total_return(returns: pd.Series) -> float:
    """
    Total cumulative return over the full sample.

    Returns:
        float: Total return as a decimal (0.50 = 50% gain).
    """
    return float((1 + returns).prod() - 1)


def total_pnl(returns: pd.Series, initial_capital: float) -> float:
    """
    Total PnL in USDT given a starting capital.

    Args:
        returns         : Series of simple period returns.
        initial_capital : Starting capital in USDT.

    Returns:
        float: Total PnL in USDT.
    """
    return float(total_return(returns) * initial_capital)


# =============================================================================
# 4. TRADING STATISTICS
# =============================================================================

def turnover(
    positions: pd.Series,
    prices:    pd.Series,
) -> float:
    """
    Average per-period turnover as a fraction of position size.

    Turnover_t = |θ_t − θ_{t-1} × (1 + r_{t-1})| / |θ_t|

    The drift term θ_{t-1} × (1 + r_{t-1}) is the position value after
    market moves but before rebalancing. The trade is the gap between
    this drifted position and the new target position.

    Args:
        positions : Series of USDT position sizes (θ_t).
        prices    : Series of asset close prices.

    Returns:
        float: Mean turnover per period.
    """
    ret      = prices.pct_change().shift(1)
    drifted  = positions.shift(1) * (1 + ret.fillna(0))
    trade    = (positions - drifted).abs()
    port_val = positions.abs()
    turn     = trade / port_val.replace(0, np.nan)
    return float(turn.mean())


def avg_holding_horizon(positions: pd.Series) -> float:
    """
    Average holding horizon in periods.

    Computed as the mean number of consecutive bars that a position
    maintains the same sign (long or short) without reversal.

    Args:
        positions : Series of position sizes (positive = long, negative = short).

    Returns:
        float: Mean holding horizon in periods (hours for hourly data).
    """
    sign_changes = (np.sign(positions) != np.sign(positions.shift(1))).astype(int)
    run_id       = sign_changes.cumsum()
    run_lengths  = run_id.groupby(run_id).transform("count")
    return float(run_lengths[sign_changes == 1].mean())


def win_rate(returns: pd.Series) -> float:
    """
    Fraction of periods with positive return.

    Returns:
        float: Win rate between 0 and 1.
    """
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """
    Ratio of gross profits to gross losses. >1 means more made than lost.

    Returns:
        float: Profit factor. NaN if no losing periods.
    """
    gains  = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


# =============================================================================
# 5. TRANSACTION COSTS — Roll Model (Section 3 of brief)
# =============================================================================

def roll_spread(prices: pd.Series) -> float:
    """
    Estimate bid-ask half-spread using the Roll (1984) model.

    s = √(max(−Cov(Δp_t, Δp_{t-1}), 0))

    where Δp_t = p_t − p_{t-1} (price changes, not returns).

    The negative serial covariance of price changes is the empirical
    signature of bid-ask bounce: a trade at the ask is more likely to
    be followed by a trade at the bid. max(., 0) handles trending
    assets where the estimate would be negative — returns 0 in that case.

    Args:
        prices : Series of transaction prices (typically close prices).

    Returns:
        float: Estimated half-spread as an absolute price level.
    """
    dp   = prices.diff().dropna()
    dp1  = dp.shift(1).dropna()
    # Align indices before computing covariance
    aligned_dp, aligned_dp1 = dp.align(dp1, join="inner")
    cov  = aligned_dp.cov(aligned_dp1)
    s_sq = max(-cov, 0.0)
    return float(np.sqrt(s_sq))


def roll_spread_pct(prices: pd.Series) -> float:
    """
    Roll spread as a fraction of average price (round-trip cost).

    To convert to one-way cost per trade: roll_spread_pct(prices) / 2

    Returns:
        float: Estimated round-trip spread as a decimal (0.001 = 0.1%).
    """
    s = roll_spread(prices)
    return float(s / prices.mean())


def transaction_costs(
    positions: pd.Series,
    prices:    pd.Series,
    slippage:  float,
) -> pd.Series:
    """
    Compute per-period transaction costs using the brief's exact formula.

    Cost_t = slippage × |θ_t − θ_{t-1} × (1 + r_{t-1})|

    where θ_t is the USDT position at the start of period t and the
    second term accounts for the position drifting with market returns
    before rebalancing. This drift correction is REQUIRED by the brief
    and is commonly omitted in naive implementations.

    Args:
        positions : Series of USDT position sizes (θ_t).
        prices    : Series of asset close prices.
        slippage  : Per-trade slippage as a decimal (from roll_spread_pct).

    Returns:
        Series of transaction costs in USDT per period.
    """
    ret          = prices.pct_change().fillna(0)
    drifted      = positions.shift(1).fillna(0) * (1 + ret)
    trade_amount = (positions - drifted).abs()
    return slippage * trade_amount


# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================

def summary(
    returns:         pd.Series,
    positions:       pd.Series | None = None,
    prices:          pd.Series | None = None,
    initial_capital: float            = 10_000,
    periods:         int              = HOURS_PER_YEAR,
    label:           str              = "Strategy",
) -> pd.DataFrame:
    """
    Compute a full performance summary table (Section 4 of brief).

    Args:
        returns         : Series of simple period returns (net of costs).
        positions       : Optional position series for turnover calculation.
        prices          : Optional price series for turnover calculation.
        initial_capital : Starting capital in USDT. Default $10,000.
        periods         : Periods per year. Default 8760 (hourly).
        label           : Strategy name for the output table index.

    Returns:
        Single-row DataFrame with all performance metrics as columns.
    """
    metrics = {
        "Total Return (%)":    round(total_return(returns) * 100, 2),
        "Total PnL (USDT)":    round(total_pnl(returns, initial_capital), 2),
        "Ann. Return (%)":     round(annualised_return(returns, periods) * 100, 2),
        "Ann. Volatility (%)": round(annualised_volatility(returns, periods) * 100, 2),
        "Sharpe":              round(sharpe(returns, periods), 3),
        "Sortino":             round(sortino(returns, periods), 3),
        "Calmar":              round(calmar(returns, periods), 3),
        "Max Drawdown (%)":    round(max_drawdown(returns) * 100, 2),
        "Avg Drawdown (%)":    round(avg_drawdown(returns) * 100, 2),
        "Win Rate (%)":        round(win_rate(returns) * 100, 2),
        "Profit Factor":       round(profit_factor(returns), 3),
    }

    if positions is not None and prices is not None:
        metrics["Avg Turnover"]      = round(turnover(positions, prices), 6)
        metrics["Avg Holding (hrs)"] = round(avg_holding_horizon(positions), 1)

    return pd.DataFrame(metrics, index=[label])


def compare(
    strategies:      dict[str, pd.Series],
    initial_capital: float = 10_000,
    periods:         int   = HOURS_PER_YEAR,
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side in one table.

    Args:
        strategies      : dict mapping strategy name → returns Series.
        initial_capital : Starting capital in USDT.
        periods         : Periods per year.

    Returns:
        DataFrame with one row per strategy and all metrics as columns.
    """
    rows = []
    for name, rets in strategies.items():
        row = summary(rets, initial_capital=initial_capital,
                      periods=periods, label=name)
        rows.append(row)
    return pd.concat(rows)
