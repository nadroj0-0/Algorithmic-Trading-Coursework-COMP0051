# =============================================================================
# utils/portfolio.py — COMP0051 Algorithmic Trading Coursework
# PnL engine: implements the brief's exact ΔVt formula.
#
# Brief formula (Section 4):
#   ΔV_t = Σ_i (θ_i_t × r_i_t) − Cost_t
#   Cost_t = s × Σ_i |θ_i_t − θ_i_{t-1} × (1 + r_i_{t-1})|
#
# Critical correctness note on the drift term:
#   Cost_t uses θ_{t-1} × (1 + r_{t-1}), NOT just θ_{t-1}.
#   This is the position value AFTER market moves but BEFORE rebalancing.
#   Omitting this drift understates costs in trending markets.
#   This file implements the correct form.
# =============================================================================

import numpy as np
import pandas as pd

from utils.metrics import (
    sharpe,
    sortino,
    calmar,
    max_drawdown,
    turnover,
    avg_holding_horizon,
    HOURS_PER_YEAR,
)


# =============================================================================
# PNL ENGINE CLASS
# =============================================================================

class PnLEngine:
    """
    Computes gross and net PnL series from positions and returns.

    Implements the brief's ΔVt formula exactly:
        ΔV_t = Σ_i (θ_i_t × r_i_t) − Cost_t
        Cost_t = s × Σ_i |θ_i_t − θ_i_{t-1} × (1 + r_i_{t-1})|

    Also computes:
        - Portfolio value series V_t = V_0 + Σ_{s≤t} ΔV_s
        - Cash balance (capital not deployed in positions)
        - Total turnover
        - Average holding horizon per asset

    Cash management (fixed gross cap approach):
        Capital not allocated to positions (|Σθ_i| < gross_cap) sits as
        USDT cash earning 0%. We maintain a fixed gross exposure cap rather
        than reinvesting profits into larger positions. This is the safer
        and more defensible approach for a backtesting context — it prevents
        leverage creep and makes the Sharpe calculation clean.

    Args:
        initial_capital : Starting capital V_0 in USDT. Default $10,000.
        slippage        : Per-trade slippage s as a decimal (from Roll model).
        periods         : Periods per year for annualisation. Default 8760.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        slippage:        float = 0.001,
        periods:         int   = HOURS_PER_YEAR,
    ):
        self.initial_capital = initial_capital
        self.slippage        = slippage
        self.periods         = periods

    def run(
        self,
        positions: pd.DataFrame,
        returns:   pd.DataFrame,
        prices:    pd.DataFrame,
    ) -> dict:
        """
        Compute the full PnL breakdown for a strategy run.

        Args:
            positions : DataFrame of USDT positions θ_i_t,
                        rows = bars, cols = assets.
                        Must be aligned with returns and prices.
            returns   : DataFrame of excess returns r_i_t,
                        rows = bars, cols = assets.
            prices    : DataFrame of close prices,
                        rows = bars, cols = assets.

        Returns:
            dict with keys:
                pnl_gross     : pd.Series — ΔVt before costs, per bar
                pnl_net       : pd.Series — ΔVt after costs, per bar
                costs         : pd.Series — Cost_t per bar
                value_gross   : pd.Series — cumulative portfolio value (gross)
                value_net     : pd.Series — cumulative portfolio value (net)
                cash          : pd.Series — undeployed capital per bar
                ret_gross     : pd.Series — gross return per bar (for metrics)
                ret_net       : pd.Series — net return per bar (for metrics)
                turnover      : float
                holding_hrs   : dict {asset: avg_holding_hours}
                total_pnl_gross : float
                total_pnl_net   : float
                pct_return_gross : float
                pct_return_net  : float
        """
        # Align all inputs on shared index
        idx   = positions.index.intersection(returns.index).intersection(prices.index)
        pos   = positions.loc[idx]
        ret   = returns.loc[idx]
        pr    = prices.loc[idx]
        assets = list(pos.columns)

        n = len(idx)

        pnl_gross = np.zeros(n)
        costs_arr = np.zeros(n)

        # Previous positions (initialised to zero — no position at t=0)
        prev_pos = pd.Series(np.zeros(len(assets)), index=assets)

        for i in range(n):
            r_t       = ret.iloc[i][assets]    # excess returns this bar
            theta_t   = pos.iloc[i][assets]    # positions this bar (USDT)

            # --- Gross PnL: Σ θ_i_t × r_i_t ---
            pnl_gross[i] = float((theta_t * r_t).sum())

            # --- Transaction cost: s × Σ|θ_t − θ_{t-1} × (1 + r_{t-1})|  ---
            # For i=0: prev_pos = 0, so cost = s × Σ|θ_0|  (entry cost)
            if i == 0:
                r_prev = pd.Series(np.zeros(len(assets)), index=assets)
            else:
                r_prev = ret.iloc[i - 1][assets]

            drifted   = prev_pos * (1 + r_prev)          # position drifts with market
            trade_amt = (theta_t - drifted).abs().sum()   # rebalancing trade size
            costs_arr[i] = self.slippage * trade_amt

            prev_pos = theta_t.copy()

        pnl_gross_s = pd.Series(pnl_gross, index=idx)
        costs_s     = pd.Series(costs_arr, index=idx)
        pnl_net_s   = pnl_gross_s - costs_s

        # --- Cumulative portfolio value ---
        value_gross = self.initial_capital + pnl_gross_s.cumsum()
        value_net   = self.initial_capital + pnl_net_s.cumsum()

        # --- Cash: capital not deployed ---
        gross_exposure = pos[assets].abs().sum(axis=1)
        cash           = (self.initial_capital - gross_exposure).clip(lower=0)

        # --- Per-bar returns (for Sharpe/Sortino/Calmar) ---
        # We use the net PnL as a return on initial_capital for comparability.
        # Dividing by initial_capital (fixed denominator) keeps the return series
        # clean and comparable to buy-and-hold benchmarks.
        ret_gross = pnl_gross_s / self.initial_capital
        ret_net   = pnl_net_s   / self.initial_capital

        # --- Turnover (average over all assets) ---
        avg_turn = 0.0
        for asset in assets:
            if asset in pr.columns:
                avg_turn += turnover(pos[asset], pr[asset])
        avg_turn /= max(len(assets), 1)

        # --- Average holding horizon per asset ---
        holding = {}
        for asset in assets:
            holding[asset] = avg_holding_horizon(pos[asset])

        # --- Summary scalars ---
        total_pnl_gross   = float(pnl_gross_s.sum())
        total_pnl_net     = float(pnl_net_s.sum())
        pct_return_gross  = total_pnl_gross / self.initial_capital
        pct_return_net    = total_pnl_net   / self.initial_capital

        return {
            "pnl_gross":        pnl_gross_s,
            "pnl_net":          pnl_net_s,
            "costs":            costs_s,
            "value_gross":      value_gross,
            "value_net":        value_net,
            "cash":             cash,
            "ret_gross":        ret_gross,
            "ret_net":          ret_net,
            "turnover":         avg_turn,
            "holding_hrs":      holding,
            "total_pnl_gross":  total_pnl_gross,
            "total_pnl_net":    total_pnl_net,
            "pct_return_gross": pct_return_gross,
            "pct_return_net":   pct_return_net,
        }

    def __repr__(self) -> str:
        return (
            f"PnLEngine(initial_capital={self.initial_capital:,.0f}, "
            f"slippage={self.slippage:.4f})"
        )
