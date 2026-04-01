# =============================================================================
# baseline.py — COMP0051 Algorithmic Trading Coursework
# Buy-and-hold benchmark for BTC, ETH, and SOL.
#
# Mirrors the role of legacy baseline.py — a standalone script that
# establishes a naive benchmark so the active strategies can be compared.
#
# Computes the same performance metrics as evaluate.py (Sharpe, Sortino,
# Calmar, drawdown, total PnL) using the same initial capital ($10,000)
# and the same date range as the active strategies.
#
# Outputs:
#   - benchmark_metrics.json  : all metrics for all three buy-and-hold strategies
#   - Console summary table
#
# Usage:
#   python baseline.py
#   python baseline.py --run_name btc_eth_sol_2024
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.data    import load_returns, get_close_matrix
from utils.metrics import summary, compare, HOURS_PER_YEAR
from utils.common  import save_json

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"

INITIAL_CAPITAL = 10_000.0
SYMBOLS         = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DATA_DIR        = Path("./data")
RF_ANNUAL       = 0.053


def parse_args():
    p = argparse.ArgumentParser(description="COMP0051 Buy-and-Hold Baseline")
    p.add_argument("--run_name", type=str, default=None,
                   help="Run name for saving results (optional)")
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                   help="Path to data directory")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("  COMP0051 — Buy-and-Hold Baselines")
    print(f"  Assets          : {SYMBOLS}")
    print(f"  Initial capital : ${INITIAL_CAPITAL:,.0f} USDT")
    print(f"  rf_annual       : {RF_ANNUAL}")
    print("=" * 60 + "\n")

    # --- Load data ---
    # Read date/timeframe settings from experiment config if available,
    # falling back to defaults. This keeps baseline consistent with run.py.
    timeframe = "1h"
    since     = "2024-01-01"
    until     = "2024-12-31"

    # Try to read from experiment.yml for consistency
    try:
        from utils.config_loader import load_experiment
        exp_cfg   = load_experiment(EXPERIMENT_PATH)
        data_cfg  = exp_cfg.get("data", {})
        timeframe = data_cfg.get("timeframe", timeframe)
        since     = data_cfg.get("since",     since)
        until     = data_cfg.get("until",     until)
        SYMBOLS_use = data_cfg.get("symbols", SYMBOLS)
    except Exception:
        SYMBOLS_use = SYMBOLS

    print("[baseline] Loading data...")
    data = load_returns(
        symbols   = SYMBOLS_use,
        data_dir  = data_dir,
        rf_annual = RF_ANNUAL,
        timeframe = timeframe,
        since     = since,
        until     = until,
    )

    # --- Compute buy-and-hold returns and metrics ---
    rows    = []
    details = {}

    for symbol, df in data.items():
        ret_excess = df["ret_excess"].dropna()
        label      = symbol.replace("/USDT", "") + " Buy-and-Hold"

        row = summary(
            returns         = ret_excess,
            initial_capital = INITIAL_CAPITAL,
            periods         = HOURS_PER_YEAR,
            label           = label,
        )
        rows.append(row)

        # PnL computation
        total_ret = float((1 + ret_excess).prod() - 1)
        total_pnl = total_ret * INITIAL_CAPITAL

        details[symbol] = {
            "label":        label,
            "total_pnl":    round(total_pnl, 2),
            "pct_return":   round(total_ret * 100, 2),
            "n_bars":       len(ret_excess),
            "period_start": str(ret_excess.index.min().date()),
            "period_end":   str(ret_excess.index.max().date()),
        }

        print(f"  {label}")
        print(f"    Period    : {details[symbol]['period_start']} → {details[symbol]['period_end']}")
        print(f"    Bars      : {len(ret_excess):,}")
        print(f"    Total PnL : ${total_pnl:,.2f}")
        print(f"    Return    : {total_ret * 100:.2f}%")
        for col in row.columns:
            print(f"    {col}: {row.iloc[0][col]}")
        print()

    # --- Combined table ---
    combined = pd.concat(rows)
    print("=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(combined.to_string())

    # --- Save results ---
    output = {
        "description":  "Buy-and-hold benchmark for BTC, ETH, SOL",
        "initial_capital": INITIAL_CAPITAL,
        "rf_annual":    RF_ANNUAL,
        "strategies":   details,
        "metrics":      combined.to_dict(orient="index"),
    }

    # Save next to run folder if run_name provided
    if args.run_name:
        out_dir = PROJECT_DIR / "runs" / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchmark_metrics.json"
    else:
        out_path = PROJECT_DIR / "benchmark_metrics.json"

    save_json(output, out_path)
    print(f"\n[baseline] Results saved → {out_path}")
    print("\n  Use these benchmarks in your report to show whether the active")
    print("  strategies add value vs a passive hold of each asset.")


if __name__ == "__main__":
    main()
