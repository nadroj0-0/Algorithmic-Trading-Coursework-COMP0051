# =============================================================================
# run.py — COMP0051 Algorithmic Trading Coursework
# Master runner — reads all config from YAML files.
#
# Mirrors train.py from the legacy system exactly:
#   - Same CLI argument pattern (--experiment override)
#   - Same SEARCH flag (only line you edit)
#   - Same run dir creation + config snapshotting before anything else
#   - Same "load data once upfront, pass to all experiments" pattern
#   - Same loop over strategies, creating one Experiment per strategy
#
# Usage:
#   python run.py                              — uses configs/experiment.yml
#   python run.py --experiment configs/experiment.yml  — explicit path
#
# Before running: ensure data is downloaded.
#   python data.py    — downloads and cleans Binance OHLCV data
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.experiment import Experiment
from utils.config_loader import (
    load_experiment,
    load_registry,
    load_strategy_config,
    load_strategy_params,
    resolve_registry_entry,
    create_run_dir,
    snapshot_configs,
    get_strategy_run_dir,
)
from utils.data    import load_returns, get_close_matrix, get_returns_matrix
from utils.common  import save_json, compute_slippage, get_portfolio_slippage

PROJECT_DIR       = Path(__file__).resolve().parent
EXPERIMENT_PATH   = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH     = PROJECT_DIR / "configs" / "registry.yml"
STRATEGIES_CFG_DIR = PROJECT_DIR / "configs" / "strategies"

# =============================================================================
# SEARCH — read from experiment.yml, not a hardcoded flag.
# Set search.enabled: true in configs/experiment.yml to run parameter search.
# The YAML is the single source of truth for this setting.
# =============================================================================
SEARCH = None  # resolved below from experiment.yml after loading


# =============================================================================
# CLI — same pattern as legacy train.py
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="COMP0051 Algorithmic Trading — Run")
    p.add_argument(
        "--experiment",
        type    = str,
        default = str(EXPERIMENT_PATH),
        help    = "Path to experiment.yml (default: configs/experiment.yml)",
    )
    return p.parse_args()


# =============================================================================
# DATA LOADING — load once upfront, pass to all strategies
# Mirrors legacy _load_train_data() — same single-load pattern
# =============================================================================

def _load_data(exp_cfg: dict) -> tuple[dict, dict]:
    """
    Load clean data and compute returns once before the strategy loop.
    Analogous to _load_train_data() in legacy train.py.

    Returns:
        data    : dict mapping symbol → DataFrame with ret_excess, close, etc.
        prepared: dict with 'prices' and 'returns' DataFrames for strategies.
    """
    data_cfg = exp_cfg.get("data", {})
    symbols  = data_cfg.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    data_dir = Path(data_cfg.get("data_dir", "./data"))
    rf_annual = float(data_cfg.get("rf_annual", 0.053))
    timeframe = data_cfg.get("timeframe", "1h")
    since = data_cfg.get("since", "2024-01-01")
    until = data_cfg.get("until", "2024-12-31")

    print("\n[run] Loading data once for all strategies...")
    print(f"[run] Symbols   : {symbols}")
    print(f"[run] rf_annual : {rf_annual}")

    data = load_returns(
        symbols=symbols,
        data_dir=data_dir,
        rf_annual=rf_annual,
        timeframe=timeframe,
        since=since,
        until=until,
    )

    prices  = get_close_matrix(data, col="close")
    returns = get_returns_matrix(data, col="ret_excess")

    # Align prices and returns on same index
    common_idx = prices.index.intersection(returns.index)
    prices      = prices.loc[common_idx]
    returns     = returns.loc[common_idx]

    print(f"[run] Data loaded: {len(prices):,} bars × {len(symbols)} assets")
    return data, {"prices": prices, "returns": returns}


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load experiment and registry
    # ------------------------------------------------------------------
    exp_cfg    = load_experiment(args.experiment)
    registry   = load_registry(REGISTRY_PATH)
    run_name   = exp_cfg.get("run_name")
    strategies = exp_cfg.get("strategies", [])

    # Resolve SEARCH from experiment.yml — YAML is the single source of truth.
    # experiment.yml: search: enabled: true/false
    SEARCH = bool(exp_cfg.get("search", {}).get("enabled", False))

    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")
    if not strategies:
        raise ValueError("experiment.yml 'strategies' list is empty.")

    print(f"\n{'=' * 60}")
    print(f"  COMP0051 ALGORITHMIC TRADING — {run_name}")
    print(f"  Strategies : {strategies}")
    print(f"  Search     : {SEARCH}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 2. Create run directory and snapshot all configs
    # Must happen before search so search writes into the run copies.
    # Identical pattern to legacy train.py step 2.
    # ------------------------------------------------------------------
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir            = run_dir,
        experiment_yml     = args.experiment,
        strategy_names     = strategies,
        strategies_cfg_dir = STRATEGIES_CFG_DIR,
    )

    # ------------------------------------------------------------------
    # 3. Hyperparameter search (if SEARCH = True)
    # search.py writes best params back into run yml files so that
    # step 5 reads the winning parameters automatically.
    # Identical pattern to legacy train.py step 3.
    # ------------------------------------------------------------------
    if SEARCH:
        from search import run_search
        run_search(exp_cfg, run_dir)
        print(f"\n[run] Search complete — proceeding to full run.\n")

    # ------------------------------------------------------------------
    # 4. Load data ONCE upfront
    # All strategies share the same data — same pattern as legacy.
    # ------------------------------------------------------------------
    raw_data, prepared = _load_data(exp_cfg)
    print("[run] Data ready. Computing slippage estimates...")

    # ------------------------------------------------------------------
    # 5. Compute Roll model slippage ONCE — inject into all strategies
    # ------------------------------------------------------------------
    slippages          = compute_slippage(raw_data)
    portfolio_slippage = get_portfolio_slippage(slippages)

    capital_cfg = exp_cfg.get("capital", {})
    initial_capital = float(capital_cfg.get("initial", 10_000))
    gross_cap       = float(capital_cfg.get("gross_cap", 100_000))

    # Save slippage estimates to run dir for the report
    save_json(
        {"slippages": slippages, "portfolio_slippage": portfolio_slippage},
        run_dir / "slippage_estimates.json",
    )

    print(f"[run] Initial capital : ${initial_capital:,.0f} USDT")
    print(f"[run] Gross cap       : ${gross_cap:,.0f} USDT")
    print(f"[run] Portfolio slippage: {portfolio_slippage:.6f} ({portfolio_slippage * 100:.4f}%)")
    print("[run] Starting strategy runs.\n")

    # ------------------------------------------------------------------
    # 6. Run each strategy
    # Mirrors legacy train.py's per-model loop exactly.
    # ------------------------------------------------------------------
    for strategy_name in strategies:
        print(f"\n{'=' * 60}")
        print(f"  RUNNING: {strategy_name}")
        print(f"{'=' * 60}")

        # Load params from run snapshot (if search ran, best_config is merged in)
        run_strategy_yml = run_dir / "configs" / "strategies" / f"{strategy_name}.yml"
        strategy_cfg     = load_strategy_config(run_strategy_yml)
        params           = strategy_cfg.get("params", {})

        # Inject experiment-level capital and slippage into params
        params["initial_capital"] = initial_capital
        params["gross_cap"]       = gross_cap
        params["slippage"]        = portfolio_slippage

        # Resolve signal_builder and execution_step from registry
        if strategy_name not in registry:
            raise KeyError(
                f"'{strategy_name}' not in registry.yml. "
                f"Available: {sorted(registry.keys())}"
            )
        resolved       = resolve_registry_entry(registry[strategy_name])
        signal_builder = resolved["signal_builder"]
        execution_step = resolved["execution_step"]

        # Artifact directory for this strategy inside the run
        strategy_dir = get_strategy_run_dir(run_dir, strategy_name)

        # Create Experiment, inject pre-loaded data (same preloaded pattern)
        exp = Experiment(strategy_name, params, strategy_dir=strategy_dir)
        exp.prepare_data(prepared)

        # Run the strategy — generates signals, sizes positions, computes PnL
        exp.run(signal_builder, execution_step)

        print(f"\n  [DONE] {strategy_name} — artifacts in {strategy_dir}")

    print(f"\n{'=' * 60}")
    print(f"  ALL STRATEGIES COMPLETE")
    print(f"  Run folder: {run_dir}")
    print(f"  Next: python evaluate.py --run_name {run_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
