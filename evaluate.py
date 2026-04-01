# =============================================================================
# evaluate.py — COMP0051 Algorithmic Trading Coursework
# Evaluation runner — produces all Section 4 report outputs.
#
# Mirrors legacy test.py structure exactly:
#   - Same CLI argument pattern (--run_name, --experiment)
#   - Same "load from run snapshot" pattern (guarantees exact config used)
#   - Same per-strategy loop calling an evaluate_strategy() function
#   - Same combined metrics saved to all_metrics.json
#
# Outputs produced (all saved to runs/{run_name}/plots/):
#   - cumulative_pnl.png        : gross vs net PnL for both strategies
#   - drawdown.png              : drawdown series
#   - roll_sensitivity.png      : Sharpe vs slippage multiplier (Section 3)
#   - performance_table.png     : all metrics including benchmarks
#   - {strategy}_signal.png     : signal diagnostic plots
#   - return_distribution.png   : return histogram
#   - all_metrics.json          : combined metrics for all strategies
#
# Usage:
#   python evaluate.py
#   python evaluate.py --run_name btc_eth_sol_2024
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.config_loader import (
    load_experiment,
    load_registry,
    get_strategy_run_dir,
)
from utils.data    import load_returns, get_close_matrix, get_returns_matrix
from utils.metrics import summary, compare, HOURS_PER_YEAR
from utils.plotting import (
    plot_cumulative_pnl,
    plot_drawdown,
    plot_roll_sensitivity,
    plot_performance_table,
    plot_signal,
    plot_return_distribution,
)
from utils.common import load_json

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"

# =============================================================================
# THE ONLY LINE YOU EDIT — which run to evaluate
# Can be overridden via --run_name CLI argument
# =============================================================================
RUN_NAME = "btc_eth_sol_2024"


# =============================================================================
# CLI — same pattern as legacy test.py
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="COMP0051 Evaluate")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Override RUN_NAME")
    p.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml")
    return p.parse_args()


# =============================================================================
# RESULTS RECONSTRUCTION
# =============================================================================

def _load_strategy_results(strategy_dir: Path, strategy_name: str) -> dict | None:
    """
    Load saved results JSON for a strategy.
    Analogous to loading .pt model + history.json in legacy test.py.

    Returns None if results file not found (strategy hasn't been run yet).
    """
    results_path = strategy_dir / f"{strategy_name}_full_results.json"
    if not results_path.exists():
        print(f"  [SKIP] {strategy_name} — results not found at {results_path}")
        print(f"         Have you run: python run.py --run_name {strategy_name}?")
        return None

    payload = load_json(results_path)
    raw     = payload.get("results", {})

    # Reconstruct pandas objects from the serialised format used by save_results:
    #   Series  : {"index": [...], "values": [...]}
    #   DataFrame: {"index": [...], "columns": [...], "data": {col: [...]}}
    #   Scalar  : Python primitive — kept as-is
    reconstructed = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "index" in val and "values" in val:
            # pd.Series
            reconstructed[key] = pd.Series(
                val["values"],
                index=pd.to_datetime(val["index"]),
            )
        elif isinstance(val, dict) and "index" in val and "columns" in val and "data" in val:
            # pd.DataFrame with DatetimeIndex — positions and similar
            reconstructed[key] = pd.DataFrame(
                val["data"],
                index=pd.to_datetime(val["index"]),
            )[val["columns"]]   # ensure column order matches saved order
        else:
            reconstructed[key] = val

    return reconstructed


# =============================================================================
# SINGLE STRATEGY EVALUATION
# Analogous to legacy evaluate_model()
# =============================================================================

def evaluate_strategy(
    strategy_name: str,
    run_dir:       Path,
    prices:        pd.DataFrame,
    returns:       pd.DataFrame,
    initial_capital: float,
) -> dict | None:
    """
    Evaluate a single strategy from a run directory.
    Loads results, computes metrics, returns summary dict.

    Args:
        strategy_name    : e.g. "pairs_cointegration"
        run_dir          : Root of the run.
        prices           : Close prices DataFrame.
        returns          : Excess returns DataFrame.
        initial_capital  : Starting capital for metrics.

    Returns:
        dict with strategy metrics, or None if results not found.
    """
    strategy_dir = get_strategy_run_dir(run_dir, strategy_name)
    results      = _load_strategy_results(strategy_dir, strategy_name)

    if results is None:
        return None

    ret_gross = results.get("ret_gross")
    ret_net   = results.get("ret_net")

    if ret_net is None or len(ret_net) == 0:
        print(f"  [SKIP] {strategy_name} — empty returns in results.")
        return None

    # --- Compute performance metrics ---
    positions_df = results.get("positions")
    first_asset  = prices.columns[0] if len(prices.columns) > 0 else None

    metrics_gross = summary(
        ret_gross,
        initial_capital = initial_capital,
        periods         = HOURS_PER_YEAR,
        label           = f"{strategy_name}_gross",
    )
    metrics_net = summary(
        ret_net,
        initial_capital = initial_capital,
        periods         = HOURS_PER_YEAR,
        label           = f"{strategy_name}_net",
    )

    # Print to console
    print(f"\n  {strategy_name} — GROSS:")
    for col in metrics_gross.columns:
        print(f"    {col}: {metrics_gross.iloc[0][col]}")
    print(f"\n  {strategy_name} — NET (after slippage):")
    for col in metrics_net.columns:
        print(f"    {col}: {metrics_net.iloc[0][col]}")

    print(f"\n  Total PnL (gross): ${results.get('total_pnl_gross', 'N/A'):,.2f}")
    print(f"  Total PnL (net)  : ${results.get('total_pnl_net',   'N/A'):,.2f}")
    print(f"  Return on capital (net): {results.get('pct_return_net', 0) * 100:.2f}%")
    print(f"  Turnover (avg/bar): {results.get('turnover', 'N/A')}")

    return {
        "name":          strategy_name,
        "metrics_gross": metrics_gross,
        "metrics_net":   metrics_net,
        "results":       results,
        "ret_gross":     ret_gross,
        "ret_net":       ret_net,
        "positions":     positions_df,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    args     = parse_args()
    run_name = args.run_name or RUN_NAME
    run_dir  = PROJECT_DIR / "runs" / run_name

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Have you run: python run.py with run_name='{run_name}'?"
        )

    # Load from run snapshot — guarantees exact config used during training
    exp_cfg   = load_experiment(run_dir / "configs" / "experiment.yml")
    exp_eval  = exp_cfg.get("eval", {})
    strategies = exp_cfg.get("strategies", [])
    capital_cfg = exp_cfg.get("capital", {})

    if not strategies:
        raise ValueError(f"No strategies in {run_dir}/configs/experiment.yml")

    initial_capital = float(capital_cfg.get("initial", 10_000))

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION — {run_name}")
    print(f"  Strategies : {strategies}")
    print(f"{'=' * 60}")

    # --- Load data ---
    data_cfg = exp_cfg.get("data", {})
    symbols  = eval_symbols = data_cfg.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    data_dir = Path(data_cfg.get("data_dir", "./data"))
    rf_annual = float(data_cfg.get("rf_annual", 0.053))

    # Read timeframe/since/until from experiment config — required positional
    # args in data.py load_returns(). Omitting them silently used module-level
    # defaults and ignored whatever was in experiment.yml.
    timeframe = data_cfg.get("timeframe", "1h")
    since     = data_cfg.get("since",     "2024-01-01")
    until     = data_cfg.get("until",     "2024-12-31")

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
    common_idx = prices.index.intersection(returns.index)
    prices     = prices.loc[common_idx]
    returns    = returns.loc[common_idx]

    # --- Load slippage from run ---
    slippage_path = run_dir / "slippage_estimates.json"
    base_slippage = 0.001   # fallback
    if slippage_path.exists():
        slip_data     = load_json(slippage_path)
        base_slippage = float(slip_data.get("portfolio_slippage", 0.001))
    print(f"[eval] Base slippage (Roll model): {base_slippage:.6f} ({base_slippage * 100:.4f}%)")

    # --- Output directory for plots ---
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Per-strategy evaluation
    # ------------------------------------------------------------------
    all_results = {}

    for strategy_name in strategies:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {strategy_name}")
        print(f"{'=' * 60}")

        result = evaluate_strategy(
            strategy_name    = strategy_name,
            run_dir          = run_dir,
            prices           = prices,
            returns          = returns,
            initial_capital  = initial_capital,
        )

        if result is not None:
            all_results[strategy_name] = result

            # Signal diagnostic plot
            strategy_results = result["results"]
            positions_df     = result["positions"]
            strategy_dir     = get_strategy_run_dir(run_dir, strategy_name)

            if positions_df is not None and len(positions_df) > 0:
                asset_cols = [c for c in positions_df.columns if c in prices.columns]
                # Reconstruct signal df from results for diagnostic
                signal_df = pd.DataFrame(index=positions_df.index)
                for col in asset_cols:
                    signal_df[col] = np.sign(positions_df[col]).fillna(0)

                # Add zscore/spread if available (pairs strategy)
                spread_col = "position"
                if "zscore" in strategy_results:
                    zs = strategy_results.get("zscore")
                    if isinstance(zs, pd.Series):
                        signal_df["zscore"] = zs.reindex(signal_df.index)
                        spread_col = "zscore"

                plot_signal(
                    prices        = prices[asset_cols].loc[positions_df.index],
                    signal_df     = signal_df,
                    positions     = positions_df[asset_cols],
                    strategy_name = strategy_name,
                    save_path     = plots_dir / f"{strategy_name}_signal.png",
                    spread_col    = spread_col,
                )

    if not all_results:
        print("\n[eval] No strategy results found. Run python run.py first.")
        return

    # ------------------------------------------------------------------
    # Buy-and-hold benchmarks
    # ------------------------------------------------------------------
    print("\n[eval] Computing buy-and-hold benchmarks...")
    benchmark_returns = {}
    benchmark_values  = {}
    for symbol in symbols:
        if symbol in raw_data:
            bh_ret = raw_data[symbol]["ret_excess"].loc[common_idx]
            bh_ret = bh_ret.dropna()
            bh_val = initial_capital * (1 + bh_ret).cumprod()
            label  = symbol.replace("/USDT", "") + " buy-and-hold"
            benchmark_returns[label] = bh_ret
            benchmark_values[label]  = bh_val

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    # 1. Cumulative PnL
    strategies_for_plot = {
        name: res["results"]
        for name, res in all_results.items()
        if "value_gross" in res["results"]
    }
    if strategies_for_plot:
        plot_cumulative_pnl(
            strategies = strategies_for_plot,
            benchmarks = benchmark_values,
            save_path  = plots_dir / "cumulative_pnl.png",
        )

    # 2. Drawdown
    net_returns = {name: res["ret_net"] for name, res in all_results.items()
                   if res["ret_net"] is not None}
    if net_returns:
        plot_drawdown(net_returns, save_path=plots_dir / "drawdown.png")

    # 3. Roll model sensitivity
    if len(all_results) > 0:
        strategy_results_for_sens = {
            name: res["results"] for name, res in all_results.items()
        }
        prices_for_sens = {
            name: prices[[c for c in (res["positions"].columns if res["positions"] is not None else []) if c in prices.columns]]
            for name, res in all_results.items()
            if res["positions"] is not None
        }
        plot_roll_sensitivity(
            strategies    = strategy_results_for_sens,
            base_slippage = base_slippage,
            returns_dict  = net_returns,
            prices_dict   = prices_for_sens,
            save_path     = plots_dir / "roll_sensitivity.png",
        )

    # 4. Performance table
    metric_rows = []
    for name, res in all_results.items():
        metric_rows.append(res["metrics_gross"])
        metric_rows.append(res["metrics_net"])
    for bname, bret in benchmark_returns.items():
        row = summary(bret, initial_capital=initial_capital,
                      periods=HOURS_PER_YEAR, label=bname)
        metric_rows.append(row)
    if metric_rows:
        combined = pd.concat(metric_rows)
        plot_performance_table(combined, save_path=plots_dir / "performance_table.png")

    # 5. Return distribution
    all_net_returns = {**net_returns, **benchmark_returns}
    if all_net_returns:
        plot_return_distribution(all_net_returns,
                                 save_path=plots_dir / "return_distribution.png")

    # ------------------------------------------------------------------
    # COMBINED METRICS JSON — same as legacy all_test_metrics.json
    # ------------------------------------------------------------------
    all_metrics_list = []
    for name, res in all_results.items():
        row_gross = res["metrics_gross"].to_dict(orient="index")
        row_net   = res["metrics_net"].to_dict(orient="index")
        all_metrics_list.append({
            "strategy":        name,
            "gross":           row_gross,
            "net":             row_net,
            "total_pnl_gross": res["results"].get("total_pnl_gross"),
            "total_pnl_net":   res["results"].get("total_pnl_net"),
            "pct_return_net":  res["results"].get("pct_return_net"),
            "turnover":        res["results"].get("turnover"),
            "holding_hrs":     res["results"].get("holding_hrs"),
        })

    out_path = run_dir / "all_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics_list, f, indent=2, default=str)
    print(f"\n[eval] Combined metrics saved → {out_path}")

    # ------------------------------------------------------------------
    # CONSOLE SUMMARY TABLE
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY (net of slippage)")
    print(f"{'=' * 70}")
    header = f"{'Strategy':<30} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD%':>8} {'PnL$':>10}"
    print(header)
    print("-" * 70)
    for item in all_metrics_list:
        net_m = item["net"]
        key   = list(net_m.keys())[0]
        m     = net_m[key]
        print(
            f"{item['strategy']:<30} "
            f"{m.get('Sharpe', float('nan')):>8.3f} "
            f"{m.get('Sortino', float('nan')):>8.3f} "
            f"{m.get('Calmar', float('nan')):>8.3f} "
            f"{m.get('Max Drawdown (%)', float('nan')):>8.2f} "
            f"${item.get('total_pnl_net', 0):>9,.2f}"
        )
    print("=" * 70)
    print(f"\n  Plots saved to: {plots_dir}")
    print(f"  Done — evaluated {len(all_results)}/{len(strategies)} strategies")


if __name__ == "__main__":
    main()
