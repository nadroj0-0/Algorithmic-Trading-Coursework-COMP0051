# =============================================================================
# utils/execution.py — COMP0051 Algorithmic Trading Coursework
# Position sizing via Mean-Variance Optimisation.
#
# Structural role: analogous to training_strategies.py in the legacy system.
# One callable per execution type, registered in registry.yml and resolved
# by config_loader.py — same pattern as gru_step, prob_gru_step etc.
#
# MVOExecutor implements the QP from Notebook 14 (Markowitz) adapted for
# USDT position sizing under the brief's gross exposure constraint.
#
# Call pattern in execution.py builders (analogous to network.py builders):
#   solver = SolverConfig.configure_solver(cfg)
#   kwargs = SolverConfig.configure_execution_kwargs(solver, cfg)
#   executor = MVOExecutor(solver=solver, **kwargs)
# =============================================================================

import numpy as np
import pandas as pd

from utils.optimisation import SolverConfig


# =============================================================================
# MVO EXECUTOR CLASS
# =============================================================================

class MVOExecutor:
    """
    Mean-Variance Optimisation position sizer.

    Takes a directional signal per asset and computes USDT position sizes
    that minimise portfolio variance subject to:
        1. Gross exposure constraint: Σ|θ_i| ≤ gross_cap  (brief's $100k)
        2. Direction constraint: sign(θ_i) == sign(signal_i)
           Positions must match signal direction — MVO sizes, signal directs.
        3. Zero position for assets with signal = 0.

    Covariance matrix Σ is estimated on a rolling lookback window of
    excess returns, recomputed at each step. Positions are then scaled
    such that portfolio variance ≈ vol_target^2 per period.

    Modelled on Notebook 14 (Markowitz.ipynb) — same cvxopt library,
    same QP structure, adapted for USDT positions and the brief's constraint.

    Args:
        solver      : Solver callable from SolverConfig.configure_solver().
        gross_cap   : Maximum Σ|θ_i| in USDT. Default $100,000.
        vol_target  : Target per-bar portfolio volatility. Default 0.02.
        mvo_lookback: Number of bars for rolling covariance estimation.
        reg         : Ridge regularisation on Σ diagonal. Default 1e-4.
        slippage    : Per-trade slippage decimal (injected by run.py).
    """

    def __init__(
        self,
        solver,
        gross_cap:    float = 100_000.0,
        vol_target:   float = 0.02,
        mvo_lookback: int   = 168,
        reg:          float = 1e-4,
        slippage:     float = 0.001,
    ):
        self.solver       = solver
        self.gross_cap    = gross_cap
        self.vol_target   = vol_target
        self.mvo_lookback = mvo_lookback
        self.reg          = reg
        self.slippage     = slippage

    def size(
        self,
        signals:    pd.Series,
        returns:    pd.DataFrame,
        capital:    float,
        bar_idx:    int,
    ) -> pd.Series:
        """
        Compute USDT position sizes for the current bar.

        Args:
            signals  : Series with asset names as index, values in {-1, 0, +1}.
            returns  : DataFrame of excess returns, rows = bars, cols = assets.
                       Must be aligned with signals index.
            capital  : Current portfolio value in USDT (for vol targeting).
            bar_idx  : Current bar index in returns (used for lookback window).

        Returns:
            Series of USDT positions θ_i, indexed by asset name.
            Respects gross_cap and signal direction constraints.
        """
        assets = list(signals.index)

        # --- Flat check: if all signals are 0, return zeros ---
        if (signals == 0).all():
            return pd.Series(np.zeros(len(assets)), index=assets)

        # --- Covariance estimation (rolling lookback) ---
        start_idx = max(0, bar_idx - self.mvo_lookback)
        ret_window = returns.iloc[start_idx:bar_idx][assets]

        if len(ret_window) < 10:
            # Insufficient history: equal-weight fallback
            theta = self._equal_weight(signals)
            return pd.Series(theta, index=assets)

        Sigma = ret_window.cov().values.astype(float)

        # --- QP: solve for minimum-variance positions ---
        sig_arr = signals[assets].values.astype(float)
        theta   = self.solver(Sigma, sig_arr, self.gross_cap, self.reg)

        # --- Vol-target scaling ---
        # Scale positions so that expected portfolio vol ≈ vol_target × capital
        # Portfolio variance: θ^T Σ θ  (in return-space, positions are weights)
        # We convert positions to weights w = θ / capital, then scale.
        if capital > 0 and np.any(theta != 0):
            w        = theta / capital
            port_var = float(w @ Sigma @ w)
            port_std = np.sqrt(max(port_var, 1e-12))
            if port_std > 0:
                scale = self.vol_target / port_std
                theta = theta * scale

        # --- Hard cap: ensure gross exposure ≤ gross_cap after scaling ---
        gross = np.abs(theta).sum()
        if gross > self.gross_cap:
            theta = theta * (self.gross_cap / gross)

        return pd.Series(theta, index=assets)

    def _equal_weight(self, signals: pd.Series) -> np.ndarray:
        """
        Equal-weight fallback when insufficient history for covariance.
        Allocates gross_cap equally among active signals.
        """
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
            f"vol_target={self.vol_target}, lookback={self.mvo_lookback})"
        )


# =============================================================================
# BUILDER FUNCTION — creates a configured MVOExecutor from cfg dict
# =============================================================================

def build_mvo_executor(cfg: dict) -> MVOExecutor:
    """
    Construct a fully configured MVOExecutor from a params dict.
    Analogous to build_baseline_gru() — two lines using SolverConfig.

    Called by Experiment.run() and StrategySession to build the executor
    from whatever params are in the strategy YAML (or best_config after search).

    Args:
        cfg : Merged params dict (strategy YAML params + any search overrides).

    Returns:
        Configured MVOExecutor ready to call .size().
    """
    solver = SolverConfig.configure_solver(cfg)
    kwargs = SolverConfig.configure_execution_kwargs(solver, cfg)
    return MVOExecutor(**kwargs)


# =============================================================================
# REGISTRY CALLABLE — registered as execution_step in registry.yml
# =============================================================================

def mvo_execution(
    signals:  pd.Series,
    returns:  pd.DataFrame,
    capital:  float,
    bar_idx:  int,
    executor: MVOExecutor,
) -> pd.Series:
    """
    Registry-callable execution step.
    Thin wrapper around MVOExecutor.size() — matches the registry pattern
    where execution_step is a plain callable, not a bound method.

    Analogous to gru_step() in legacy training_strategies.py — called once
    per bar inside the strategy session's run loop.

    Args:
        signals  : Per-asset directional signals {-1, 0, +1}.
        returns  : Excess returns DataFrame for covariance estimation.
        capital  : Current portfolio value in USDT.
        bar_idx  : Current bar index.
        executor : Pre-built MVOExecutor instance.

    Returns:
        Series of USDT positions θ_i.
    """
    return executor.size(signals, returns, capital, bar_idx)
