# =============================================================================
# search.py — Parameter Search Runner
# COMP0051 Algorithmic Trading Coursework
#
# Runs successive-halving parameter search for all strategies listed in
# experiment.yml, then writes the best params back into each strategy's yml
# inside the run directory.
#
# Mirrors legacy search.py structure exactly:
#   - Same CLI argument pattern
#   - Same run_search() / search_strategy() split
#   - Data loaded ONCE upfront, passed to all strategies
#   - write_best_config() writes winner back into run yml
#
# Called by run.py when SEARCH = True, or run standalone:
#   python search.py
#   python search.py --experiment configs/experiment.yml
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.config_loader import (
    load_experiment,
    load_registry,
    load_strategy_config,
    load_search_space,
    resolve_registry_entry,
    create_run_dir,
    snapshot_configs,
    write_best_config,
    get_strategy_run_dir,
)
from utils.data    import load_returns, get_close_matrix, get_returns_matrix
from utils.common  import compute_slippage, get_portfolio_slippage
from utils.hyperparameter import staged_search_strategy

PROJECT_DIR        = Path(__file__).resolve().parent
EXPERIMENT_PATH    = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH      = PROJECT_DIR / "configs" / "registry.yml"
STRATEGIES_CFG_DIR = PROJECT_DIR / "configs" / "strategies"


# =============================================================================
# SINGLE STRATEGY SEARCH
# Analogous to legacy search_model()
# =============================================================================

def search_strategy(
    strategy_name: str,
    run_dir:       Path,
    exp_search:    dict,
    registry:      dict,
    prepared:      dict,
    slippage:      float,
    capital:       float,
    gross_cap:     float,
) -> dict:
    """
    Run parameter search for a single strategy.

    Data is passed in pre-loaded — loading happens once in run_search(),
    not once per strategy.

    Flow (mirrors legacy search_model()):
        1. Load strategy yml from run snapshot
        2. Resolve signal_builder + execution_step from registry
        3. Load search space from run yml
        4. Call staged_search_strategy() from hyperparameter.py
        5. Write best params back to run yml via write_best_config()
        6. Return best_params dict

    Args:
        strategy_name : Name matching a key in registry.yml.
        run_dir       : Root of the current run.
        exp_search    : Search config block from experiment.yml.
        registry      : Loaded registry dict.
        prepared      : dict with 'prices' and 'returns' DataFrames.
        slippage      : Roll model slippage decimal.
        capital       : Initial capital in USDT.
        gross_cap     : Gross exposure cap in USDT.

    Returns:
        dict of best params found.
    """
    run_strategy_yml = run_dir / "configs" / "strategies" / f"{strategy_name}.yml"
    strategy_dir     = get_strategy_run_dir(run_dir, strategy_name)

    print(f"\n{'=' * 60}")
    print(f"  SEARCH: {strategy_name}")
    print(f"{'=' * 60}")

    # --- Load strategy config from run snapshot ---
    strategy_cfg = load_strategy_config(run_strategy_yml)
    params       = strategy_cfg.get("params", {})

    # --- Inject capital and slippage so search candidates use correct values ---
    params["initial_capital"] = capital
    params["gross_cap"]       = gross_cap
    params["slippage"]        = slippage

    # --- Resolve signal_builder and execution_step from registry ---
    if strategy_name not in registry:
        raise KeyError(
            f"'{strategy_name}' not in registry.yml. "
            f"Available: {sorted(registry.keys())}"
        )
    resolved       = resolve_registry_entry(registry[strategy_name])
    signal_builder = resolved["signal_builder"]
    execution_step = resolved["execution_step"]

    # --- Load search space ---
    search_space = load_search_space(run_strategy_yml)
    if not search_space:
        print(f"  [SKIP] {strategy_name} — no search_space in yml, using defaults")
        return {}

    # --- Build search schedule from experiment search block ---
    init_models  = int(exp_search.get("init_models", 15))
    val_fraction = float(exp_search.get("val_fraction", 0.30))
    schedule     = exp_search.get("schedule", None)
    if schedule is not None:
        schedule = [{"bars": int(s["bars"]), "keep": int(s["keep"])} for s in schedule]

    prices  = prepared["prices"]
    returns = prepared["returns"]

    # --- Run staged search ---
    best_params = staged_search_strategy(
        search_space    = search_space,
        prices          = prices,
        returns         = returns,
        signal_builder  = signal_builder,
        execution_step  = execution_step,
        strategy_dir    = strategy_dir,
        base_params     = params,
        schedule        = schedule,
        initial_models  = init_models,
        val_fraction    = val_fraction,
        capital         = capital,
        search_name     = f"{strategy_name}_search",
    )

    # --- Write best params back into run yml ---
    write_best_config(run_strategy_yml, best_params)

    print(f"\n  Best params for {strategy_name}:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    return best_params


# =============================================================================
# MAIN SEARCH LOOP — called by run.py or standalone
# Analogous to legacy run_search()
# =============================================================================

def run_search(exp_cfg: dict, run_dir: Path) -> dict:
    """
    Run parameter search for all strategies in the experiment.

    Data loaded ONCE upfront then passed to each strategy — same pattern
    as run.py / legacy train.py.

    Called by run.py when SEARCH = True.
    Can also be called standalone via main() below.
    """
    registry     = load_registry(REGISTRY_PATH)
    exp_search   = exp_cfg.get("search", {})
    data_cfg     = exp_cfg.get("data",   {})
    capital_cfg  = exp_cfg.get("capital", {})
    strategies   = exp_cfg.get("strategies", [])

    if not strategies:
        print("[search] No strategies in experiment.yml — nothing to search.")
        return {}

    print(f"\n[search] Parameter search — {len(strategies)} strategy/strategies")
    print(f"[search] Init models  : {exp_search.get('init_models', 15)}")
    print(f"[search] Val fraction : {exp_search.get('val_fraction', 0.30)}")
    print(f"[search] Schedule     : {exp_search.get('schedule')}")

    # ------------------------------------------------------------------
    # Load data ONCE upfront — same as run.py pattern
    # ------------------------------------------------------------------
    symbols   = data_cfg.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    data_dir  = Path(data_cfg.get("data_dir", "./data"))
    rf_annual = float(data_cfg.get("rf_annual", 0.053))

    # Read timeframe/since/until from experiment config — required positional
    # args in data.py load_returns(). run.py passes these correctly; search.py
    # was previously omitting them and silently falling back to module defaults.
    timeframe = data_cfg.get("timeframe", "1h")
    since     = data_cfg.get("since",     "2024-01-01")
    until     = data_cfg.get("until",     "2024-12-31")

    print(f"\n[search] Loading data once for all strategies...")
    raw_data = load_returns(
        symbols   = symbols,
        data_dir  = data_dir,
        rf_annual = rf_annual,
        timeframe = timeframe,
        since     = since,
        until     = until,
    )
    prices   = get_close_matrix(raw_data, col="close")
    returns  = get_returns_matrix(raw_data, col="ret_excess")

    # Align on shared index
    common_idx = prices.index.intersection(returns.index)
    prices     = prices.loc[common_idx]
    returns    = returns.loc[common_idx]

    prepared = {"prices": prices, "returns": returns}
    print(f"[search] Data loaded: {len(prices):,} bars")

    # Compute slippage for the search candidates
    slippages          = compute_slippage(raw_data)
    portfolio_slippage = get_portfolio_slippage(slippages)

    capital   = float(capital_cfg.get("initial",   10_000))
    gross_cap = float(capital_cfg.get("gross_cap", 100_000))

    # ------------------------------------------------------------------
    # Search each strategy
    # ------------------------------------------------------------------
    all_best = {}
    for strategy_name in strategies:
        best = search_strategy(
            strategy_name = strategy_name,
            run_dir       = run_dir,
            exp_search    = exp_search,
            registry      = registry,
            prepared      = prepared,
            slippage      = portfolio_slippage,
            capital       = capital,
            gross_cap     = gross_cap,
        )
        all_best[strategy_name] = best

    print(f"\n[search] Done — {len(all_best)} strategy/strategies searched.")
    return all_best


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="COMP0051 Parameter Search")
    parser.add_argument(
        "--experiment",
        type    = str,
        default = str(EXPERIMENT_PATH),
        help    = "Path to experiment.yml (default: configs/experiment.yml)",
    )
    args = parser.parse_args()

    exp_cfg  = load_experiment(args.experiment)
    run_name = exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")

    # Create run dir and snapshot configs
    # Idempotent — safe if run.py already did this
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir            = run_dir,
        experiment_yml     = args.experiment,
        strategy_names     = exp_cfg.get("strategies", []),
        strategies_cfg_dir = STRATEGIES_CFG_DIR,
    )

    run_search(exp_cfg, run_dir)


if __name__ == "__main__":
    main()
