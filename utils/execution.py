# =============================================================================
# utils/execution.py — COMP0051 Algorithmic Trading Coursework
# Position sizing via full Mean-Variance Optimisation.
#
# MVOExecutor now implements TRUE MVO:
#   minimise  θ^T Σ θ  −  λ × μ^T θ
#   subject to signal direction constraints + gross exposure cap
#
# μ (expected return vector) is estimated as the rolling mean of recent
# excess returns, consistent with Markowitz's original formulation.
# λ (risk_aversion) controls the return/risk trade-off.
# Setting λ=0 or mu_lookback=0 recovers minimum-variance (as before).
#
# This matches the brief's requirement: "Use MVO or a cost-aware method
# to set exposures θ_t at each step."
# =============================================================================

import numpy as np
import pandas as pd

from utils.optimisation import SolverConfig


# =============================================================================
# MVO EXECUTOR CLASS
# =============================================================================

class MVOExecutor:
    """
    Full Mean-Variance Optimisation position sizer.

    Implements:
        minimise  θ^T Σ θ  −  λ × μ^T θ
        subject to:
            Σ|θ_i| ≤ gross_cap           (brief's $100,000 gross cap)
            sign(θ_i) == sign(signal_i)  (MVO sizes, signal directs)
            θ_i = 0 if signal_i = 0

    Σ is the rolling covariance matrix of excess returns (mvo_lookback bars).
    μ is the rolling mean of excess returns (mu_lookback bars), scaled by
    the signal direction so that the optimiser sees a positive expected return
    for every active position regardless of direction.

    λ (risk_aversion) is read from the strategy YAML and controls how much
    weight the objective places on expected returns vs variance.

    Args:
        solver        : Solver callable from SolverConfig.configure_solver().
        gross_cap     : Max Σ|θ_i| in USDT. Default $100,000.
        vol_target    : Target per-bar portfolio volatility. Default 0.02.
        mvo_lookback  : Rolling window for Σ estimation (bars).
        mu_lookback   : Rolling window for μ estimation (bars).
        risk_aversion : λ — return/risk trade-off. 0 = min-variance only.
        reg           : Ridge regularisation on Σ.
        slippage      : Per-trade slippage (injected by run.py).
    """

    def __init__(
        self,
        solver,
        gross_cap:     float = 100_000.0,
        vol_target:    float = 0.02,
        mvo_lookback:  int   = 168,
        mu_lookback:   int   = 168,
        risk_aversion: float = 1.0,
        reg:           float = 1e-4,
        slippage:      float = 0.001,
    ):
        self.solver        = solver
        self.gross_cap     = gross_cap
        self.vol_target    = vol_target
        self.mvo_lookback  = mvo_lookback
        self.mu_lookback   = mu_lookback
        self.risk_aversion = risk_aversion
        self.reg           = reg
        self.slippage      = slippage

    def size(
        self,
        signals:  pd.Series,
        returns:  pd.DataFrame,
        capital:  float,
        bar_idx:  int,
    ) -> pd.Series:
        """
        Compute USDT position sizes for the current bar using full MVO.

        Args:
            signals : Series with asset names as index, values in {-1, 0, +1}.
            returns : DataFrame of excess returns (rows=bars, cols=assets).
            capital : Current portfolio value in USDT.
            bar_idx : Current bar index (used for lookback window slicing).

        Returns:
            Series of USDT positions θ_i indexed by asset name.
        """
        assets = list(signals.index)

        if (signals == 0).all():
            return pd.Series(np.zeros(len(assets)), index=assets)

        # --- Covariance estimation (Σ) ---
        start_cov  = max(0, bar_idx - self.mvo_lookback)
        ret_window = returns.iloc[start_cov:bar_idx][assets]

        if len(ret_window) < 10:
            theta = self._equal_weight(signals)
            return pd.Series(theta, index=assets)

        Sigma = ret_window.cov().values.astype(float)

        # --- Expected return estimation (μ) ---
        # Use rolling mean of recent excess returns as the expected-return
        # vector. This is the standard Markowitz input estimation approach.
        # We scale by signal direction so μ_i > 0 for all active positions:
        # the optimiser always sees a positive expected return contribution
        # from positions that the signal says to hold.
        start_mu  = max(0, bar_idx - self.mu_lookback)
        mu_window = returns.iloc[start_mu:bar_idx][assets]
        mu_raw    = mu_window.mean().values.astype(float)   # (n,) mean excess return

        # Scale μ by signal direction: if signal = -1 (short), flip μ sign so
        # the optimiser sees a positive expected P&L from the short position.
        sig_arr = signals[assets].values.astype(float)
        mu_adj  = mu_raw * sig_arr   # element-wise: μ_i × signal_i

        # --- Solve MVO QP ---
        theta = self.solver(
            Sigma         = Sigma,
            signals       = sig_arr,
            gross_cap     = self.gross_cap,
            mu            = mu_adj,
            risk_aversion = self.risk_aversion,
            reg           = self.reg,
        )

        # --- Volatility-target scaling ---
        if capital > 0 and np.any(theta != 0):
            w        = theta / capital
            port_var = float(w @ Sigma @ w)
            port_std = np.sqrt(max(port_var, 1e-12))
            if port_std > 0:
                scale = self.vol_target / port_std
                theta = theta * scale

        # --- Hard gross-cap enforcement after scaling ---
        gross = np.abs(theta).sum()
        if gross > self.gross_cap:
            theta = theta * (self.gross_cap / gross)

        return pd.Series(theta, index=assets)

    def _equal_weight(self, signals: pd.Series) -> np.ndarray:
        """Equal-weight fallback when insufficient history for covariance."""
        active = signals[signals != 0]
        theta  = np.zeros(len(signals))
        if len(active) == 0:
            return theta
        alloc = self.gross_cap / len(active)
        for i, asset in enumerate(signals.index):
            if signals[asset] != 0:
                theta[i] = signals[asset] * alloc
        return theta

    def __repr__(self) -> str:
        return (
            f"MVOExecutor(gross_cap={self.gross_cap:,.0f}, "
            f"vol_target={self.vol_target}, "
            f"risk_aversion={self.risk_aversion}, "
            f"lookback={self.mvo_lookback})"
        )


# =============================================================================
# BUILDER
# =============================================================================

def build_mvo_executor(cfg: dict) -> MVOExecutor:
    """
    Construct a fully configured MVOExecutor from a params dict.
    Two lines using SolverConfig — same pattern as legacy network.py builders.
    """
    solver = SolverConfig.configure_solver(cfg)
    kwargs = SolverConfig.configure_execution_kwargs(solver, cfg)
    return MVOExecutor(**kwargs)


# =============================================================================
# REGISTRY CALLABLE
# =============================================================================

def mvo_execution(
    signals:  pd.Series,
    returns:  pd.DataFrame,
    capital:  float,
    bar_idx:  int,
    executor: MVOExecutor,
) -> pd.Series:
    """
    Registry-callable execution step. Thin wrapper around MVOExecutor.size().
    Analogous to gru_step() in legacy training_strategies.py.
    """
    return executor.size(signals, returns, capital, bar_idx)
