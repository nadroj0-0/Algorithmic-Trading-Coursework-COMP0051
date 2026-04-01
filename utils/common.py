# =============================================================================
# utils/common.py — COMP0051 Algorithmic Trading Coursework
# Central glue module — imports, helpers, and run_strategy().
#
# Mirrors the role of common.py in the legacy system:
#   - Central import hub (imported with * by experiment.py)
#   - Core orchestration functions (run_strategy ≈ full_train)
#   - File I/O helpers (save_results ≈ save_history, save_json kept verbatim)
#   - Slippage computation (compute_slippage — called once by run.py)
#
# All PyTorch removed. No neural-net code anywhere in this file.
# =============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.early_stopping    import EarlyStopping
from utils.data              import load_returns, get_close_matrix, get_returns_matrix
from utils.strategies        import build_pairs_signal, build_trend_signal
from utils.execution         import mvo_execution, MVOExecutor, build_mvo_executor
from utils.portfolio         import PnLEngine
from utils.strategy_session  import StrategySession
from utils.metrics           import roll_spread_pct, HOURS_PER_YEAR


# =============================================================================
# FILE I/O HELPERS (kept from legacy — save_json verbatim)
# =============================================================================

def save_json(data: dict, path: Path) -> None:
    """
    Save dictionary as formatted JSON.
    Kept verbatim from legacy common.py — used by hyperparameter.py.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path) -> dict:
    """Load a JSON file saved by save_json or save_results."""
    with open(path) as f:
        return json.load(f)


def save_results(
    results:      dict,
    name:         str,
    stage:        str,
    strategy_dir: Path,
) -> Path:
    """
    Save strategy run results as JSON.
    Analogous to save_history() in legacy common.py — same payload structure.

    Payload:
        { strategy, stage, timestamp, params, metrics, pnl_summary }

    Args:
        results      : Output dict from PnLEngine.run().
        name         : Strategy name (e.g. "pairs_cointegration").
        stage        : "train" or "full" or "search".
        strategy_dir : Directory to save file (inside runs/{run_name}/strategies/).

    Returns:
        Path to the saved JSON file.
    """
    strategy_dir.mkdir(parents=True, exist_ok=True)
    results_path = strategy_dir / f"{name}_{stage}_results.json"

    # Convert Series and DataFrames for JSON serialisation
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, pd.Series):
            # Series: preserve index as ISO strings for faithful reconstruction
            serialisable[k] = {
                "index":  list(v.index.astype(str)),
                "values": list(v.values),
            }
        elif isinstance(v, pd.DataFrame):
            # DataFrame: preserve DatetimeIndex AND column data so evaluate.py
            # can reconstruct a proper DataFrame with .columns and .index intact.
            # Previously used df.to_dict(orient="list") which dropped the index,
            # causing a crash in evaluate.py when calling positions_df.columns.
            serialisable[k] = {
                "index":   list(v.index.astype(str)),
                "columns": list(v.columns),
                "data":    {col: list(v[col].values) for col in v.columns},
            }
        elif isinstance(v, np.floating):
            serialisable[k] = float(v)
        elif isinstance(v, np.integer):
            serialisable[k] = int(v)
        elif isinstance(v, dict):
            # Nested dict (e.g. holding_hrs) — flatten values to Python scalars
            serialisable[k] = {
                kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()
            }
        else:
            serialisable[k] = v

    payload = {
        "strategy":  name,
        "stage":     stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results":   serialisable,
    }

    save_json(payload, results_path)
    print(f"[common] Results saved → {results_path}")
    return results_path


def load_results(path: Path) -> dict:
    """
    Load a JSON results file saved by save_results().
    Analogous to load_history() in legacy common.py.

    Returns:
        Full payload dict including the nested results dict.
    """
    return load_json(path)


# =============================================================================
# SLIPPAGE COMPUTATION (called once by run.py, injected into all strategies)
# =============================================================================

def compute_slippage(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Compute Roll model half-spread slippage for each asset.

    Called ONCE by run.py upfront, before the strategy loop.
    Results are injected into strategy params so both strategies use
    consistent, data-driven slippage estimates.

    The Roll model (1984): s = √(−Cov(Δp_t, Δp_{t-1}))
    Implemented in metrics.roll_spread_pct().

    Args:
        data : dict mapping symbol → DataFrame with 'close' column.

    Returns:
        dict mapping symbol → slippage as a decimal (e.g. 0.001 = 0.1%).
    """
    slippages = {}
    print("\n[slippage] Roll model estimates (one-way cost = round-trip / 2):")
    for symbol, df in data.items():
        # roll_spread_pct() returns the round-trip cost (see its docstring).
        # PnLEngine's Cost_t formula applies slippage per trade leg (one-way),
        # so we divide by 2. Previously the full round-trip was used, which
        # charged twice the correct amount per trade.
        round_trip = roll_spread_pct(df["close"])
        one_way    = round_trip / 2.0
        slippages[symbol] = one_way
        print(f"  {symbol}: round-trip={round_trip:.6f}  one-way={one_way:.6f} ({one_way * 100:.4f}%)")

    # Use the maximum slippage across all assets for a conservative estimate
    # applied uniformly — this simplification is stated in the report.
    max_slip = max(slippages.values())
    print(f"  → Using max across assets: {max_slip:.6f} ({max_slip * 100:.4f}%)")
    print()

    return slippages


def get_portfolio_slippage(slippages: dict[str, float]) -> float:
    """
    Return the conservative (max) slippage across all assets.
    This single value is used in PnLEngine as the uniform slippage s.
    """
    return max(slippages.values())


# =============================================================================
# STRATEGY RUNNER — analogous to full_train() in legacy common.py
# =============================================================================

def run_strategy(
    session:   StrategySession,
    prices:    pd.DataFrame,
    returns:   pd.DataFrame,
    n_bars:    int | None = None,
    capital:   float      = 10_000.0,
) -> dict:
    """
    Run a StrategySession for n_bars and return its full results dict.
    Analogous to full_train() — same role, same signature pattern.

    Args:
        session  : StrategySession with strategy, executor, pnl_engine configured.
        prices   : Close price DataFrame (all bars).
        returns  : Excess returns DataFrame (all bars).
        n_bars   : Number of bars to run. Default: run all available bars.
        capital  : Portfolio capital in USDT.

    Returns:
        Cumulative results dict from session.results.
    """
    if n_bars is None:
        n_bars = len(prices)

    start_time = time.time()

    session.run(
        n_bars  = n_bars,
        prices  = prices,
        returns = returns,
        capital = capital,
    )

    elapsed = time.time() - start_time
    print(f"[common] run_strategy completed in {elapsed:.2f}s | "
          f"bars={session.bar_idx} | "
          f"best_sharpe={session.best_sharpe:.3f}")

    return session.results
