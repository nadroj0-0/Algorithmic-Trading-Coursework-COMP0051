# =============================================================================
# utils/plotting.py — COMP0051 Algorithmic Trading Coursework
# Publication-quality report plots for the PDF submission.
#
# All plots required by the brief's Section 4:
#   - Cumulative PnL: gross vs net for both strategies
#   - Drawdown series
#   - Roll model slippage sensitivity
#   - Performance table (Sharpe, Sortino, Calmar, etc.)
#   - Signal diagnostic plot
#
# Called by evaluate.py. All functions save to disk at the given path.
# Uses matplotlib throughout — same library as all lecturer notebooks.
# =============================================================================

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from utils.metrics import drawdown_series, sharpe, HOURS_PER_YEAR


# =============================================================================
# STYLE DEFAULTS
# =============================================================================

FIGSIZE_WIDE   = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL   = (14, 10)
DPI            = 150

COLORS = {
    "pairs_gross":  "#2563EB",   # blue — gross pairs
    "pairs_net":    "#93C5FD",   # light blue — net pairs
    "trend_gross":  "#16A34A",   # green — gross trend
    "trend_net":    "#86EFAC",   # light green — net trend
    "btc_hold":     "#F59E0B",   # amber — buy-and-hold BTC
    "eth_hold":     "#8B5CF6",   # purple — buy-and-hold ETH
    "sol_hold":     "#EC4899",   # pink — buy-and-hold SOL
    "cost":         "#EF4444",   # red — costs
    "drawdown":     "#EF4444",   # red — drawdown fill
    "signal_pos":   "#16A34A",   # green — positive signal
    "signal_neg":   "#EF4444",   # red — negative signal
    "price":        "#1F2937",   # dark — price line
}

plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
})


# =============================================================================
# 1. CUMULATIVE PNL — gross vs net, both strategies
# =============================================================================

def plot_cumulative_pnl(
    strategies: dict[str, dict],
    save_path:  Path,
    benchmarks: Optional[dict[str, pd.Series]] = None,
    val_start:  Optional[int] = None,
) -> None:
    """
    Plot cumulative portfolio value for both strategies (gross and net).
    Required by Section 4: "calculate both gross performance and net of slippage."

    Args:
        strategies : dict mapping strategy_name → results dict from PnLEngine.
                     Each results dict must have 'value_gross' and 'value_net' Series.
        save_path  : Path to save the figure.
        benchmarks : Optional dict mapping name → cumulative value Series.
        val_start  : Optional bar index marking train/val split (vertical line).
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TALL, sharey=False)

    strategy_color_map = {
        "pairs_cointegration": ("pairs_gross", "pairs_net"),
        "trend_following":     ("trend_gross",  "trend_net"),
    }

    for ax, (name, results) in zip(axes, strategies.items()):
        gross_key, net_key = strategy_color_map.get(
            name, ("pairs_gross", "pairs_net")
        )

        value_gross = results.get("value_gross")
        value_net   = results.get("value_net")

        if value_gross is not None:
            ax.plot(value_gross.values, color=COLORS[gross_key],
                    linewidth=2.0, label="Gross PnL", alpha=0.9)
        if value_net is not None:
            ax.plot(value_net.values, color=COLORS[net_key],
                    linewidth=2.0, label="Net PnL (after costs)", alpha=0.9)

        if benchmarks:
            for bname, bseries in benchmarks.items():
                ax.plot(bseries.values, linestyle="--", linewidth=1.0,
                        label=bname, alpha=0.7)

        if val_start is not None:
            ax.axvline(val_start, color="grey", linestyle=":", linewidth=1.5,
                       label="Train/Val split")

        ax.axhline(10_000, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="Initial capital")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Bar index (hours)")
        ax.set_ylabel("Portfolio value (USDT)")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"
        ))

    fig.suptitle("Cumulative Portfolio Value — Gross vs Net of Slippage",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# =============================================================================
# 2. DRAWDOWN SERIES
# =============================================================================

def plot_drawdown(
    strategies: dict[str, pd.Series],
    save_path:  Path,
) -> None:
    """
    Plot drawdown series for all strategies on a single axes.

    Args:
        strategies : dict mapping strategy_name → net returns Series.
        save_path  : Path to save figure.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    color_map = {
        "pairs_cointegration": COLORS["pairs_gross"],
        "trend_following":     COLORS["trend_gross"],
    }

    for name, returns in strategies.items():
        dd    = drawdown_series(returns)
        color = color_map.get(name, "#6B7280")
        ax.fill_between(range(len(dd)), dd.values, 0,
                        alpha=0.3, color=color, label=f"{name} (fill)")
        ax.plot(dd.values, color=color, linewidth=1.5,
                label=name.replace("_", " ").title())

    ax.set_title("Drawdown Series — Net of Slippage", fontweight="bold")
    ax.set_xlabel("Bar index (hours)")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# =============================================================================
# 3. ROLL MODEL SLIPPAGE SENSITIVITY
# =============================================================================

def plot_roll_sensitivity(
    strategies:       dict[str, dict],
    base_slippage:    float,
    returns_dict:     dict[str, pd.Series],
    prices_dict:      dict[str, pd.DataFrame],
    save_path:        Path,
    multipliers:      list = None,
) -> None:
    """
    Sensitivity of strategy Sharpe to slippage multiplier.
    Required by Section 3: "discuss strategy sensitivity to slippage."

    Reruns PnL at s=0, 0.5s, 1s (baseline), 2s, 3s and plots net Sharpe.

    Args:
        strategies    : dict mapping strategy_name → results dict.
        base_slippage : Roll model estimate (decimal).
        returns_dict  : dict strategy_name → net returns Series for recomputation.
        prices_dict   : dict strategy_name → positions DataFrame.
        save_path     : Path to save figure.
        multipliers   : List of slippage multipliers. Default [0, 0.5, 1, 2, 3].
    """
    from utils.metrics import transaction_costs, HOURS_PER_YEAR, sharpe as sharpe_fn

    if multipliers is None:
        multipliers = [0.0, 0.5, 1.0, 2.0, 3.0]

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        "pairs_cointegration": COLORS["pairs_gross"],
        "trend_following":     COLORS["trend_gross"],
    }

    for name, results in strategies.items():
        sharpes = []
        pnl_gross = results.get("pnl_gross")
        positions = results.get("positions")

        if pnl_gross is None or positions is None:
            continue

        initial_capital = 10_000.0

        for mult in multipliers:
            s_adj = base_slippage * mult
            # Recompute costs at this slippage multiplier
            # Approximate: use first asset's prices for cost sensitivity
            if not positions.empty:
                # Sum costs across assets
                total_cost = pd.Series(0.0, index=positions.index)
                for asset in positions.columns:
                    if asset in returns_dict.get(name, {}):
                        pass  # use pre-computed positions
                    prices_col = prices_dict.get(name, pd.DataFrame())
                    if hasattr(prices_col, "columns") and asset in prices_col.columns:
                        cost_asset = transaction_costs(
                            positions[asset], prices_col[asset], s_adj
                        )
                        total_cost = total_cost.add(cost_asset, fill_value=0)

                pnl_net = pnl_gross - total_cost
                ret_net = pnl_net / initial_capital
                s_val   = sharpe_fn(ret_net, periods=HOURS_PER_YEAR)
            else:
                s_val = np.nan

            sharpes.append(s_val)

        color = color_map.get(name, "#6B7280")
        ax.plot(multipliers, sharpes, "o-", color=color, linewidth=2.0,
                markersize=6, label=name.replace("_", " ").title())

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(1, color="grey", linewidth=0.8, linestyle=":", alpha=0.8,
               label="Baseline slippage (Roll model)")
    ax.set_title(
        f"Sharpe Ratio vs Slippage Multiplier\n"
        f"(baseline s = {base_slippage:.5f} = {base_slippage * 100:.3f}%)",
        fontweight="bold"
    )
    ax.set_xlabel("Slippage multiplier (× Roll model estimate)")
    ax.set_ylabel("Annualised Sharpe ratio (net of costs)")
    ax.legend()
    ax.set_xticks(multipliers)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# =============================================================================
# 4. PERFORMANCE TABLE
# =============================================================================

def plot_performance_table(
    summary_df: pd.DataFrame,
    save_path:  Path,
) -> None:
    """
    Render a matplotlib table of all performance metrics.
    Rows: strategies + benchmarks. Columns: all metrics from metrics.summary().

    Args:
        summary_df : DataFrame from metrics.compare() — rows = strategies.
        save_path  : Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(16, max(3, len(summary_df) * 0.9 + 1.5)))
    ax.axis("off")

    cols   = list(summary_df.columns)
    rows   = list(summary_df.index)
    vals   = summary_df.values

    table = ax.table(
        cellText  = vals,
        rowLabels = rows,
        colLabels = cols,
        loc       = "center",
        cellLoc   = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Header styling
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#1F2937")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row label styling
    for i in range(len(rows)):
        table[(i + 1, -1)].set_facecolor("#F3F4F6")
        table[(i + 1, -1)].set_text_props(fontweight="bold")

    ax.set_title("Performance Summary — All Strategies and Benchmarks",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# =============================================================================
# 5. SIGNAL DIAGNOSTIC PLOT
# =============================================================================

def plot_signal(
    prices:        pd.DataFrame,
    signal_df:     pd.DataFrame,
    positions:     pd.DataFrame,
    strategy_name: str,
    save_path:     Path,
    spread_col:    str = "zscore",
    max_bars:      int = 1000,
) -> None:
    """
    Three-panel diagnostic plot for a strategy's signal and positions.
    Essential for the report's strategy justification section.

    Panel 1: Normalised asset prices (log scale).
    Panel 2: Signal series (z-score for pairs, EWM diff for trend).
    Panel 3: USDT positions over time per asset.

    Args:
        prices        : Close price DataFrame.
        signal_df     : Signal DataFrame from strategy.generate().
        positions     : Positions DataFrame from PnLEngine.
        strategy_name : Name for title.
        save_path     : Path to save figure.
        spread_col    : Column from signal_df to plot in panel 2.
        max_bars      : Limit bars shown to avoid illegible plots.
    """
    # Trim to max_bars for readability
    n = min(max_bars, len(prices))
    pr = prices.iloc[:n]
    sg = signal_df.iloc[:n] if len(signal_df) >= n else signal_df
    po = positions.iloc[:n] if len(positions) >= n else positions

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.4)

    # --- Panel 1: Prices (normalised) ---
    ax1 = fig.add_subplot(gs[0])
    for col in pr.columns:
        normed = pr[col] / pr[col].iloc[0]   # normalise to 1 at start
        ax1.plot(normed.values, label=col, linewidth=1.0)
    ax1.set_title(f"{strategy_name.replace('_', ' ').title()} — Signal Diagnostic",
                  fontweight="bold")
    ax1.set_ylabel("Normalised price (t=0 baseline = 1)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_yscale("log")

    # --- Panel 2: Signal indicator ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if spread_col in sg.columns:
        spread = sg[spread_col].values
        ax2.plot(spread, color=COLORS["price"], linewidth=1.0, label=spread_col)
        # Shade entry zones
        ax2.axhline(1.5,  color=COLORS["signal_neg"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(-1.5, color=COLORS["signal_pos"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(0,    color="grey", linewidth=0.5, linestyle="-")
    else:
        # Plot first asset's signal as bar chart
        asset = [c for c in sg.columns if c in prices.columns]
        if asset:
            sig = sg[asset[0]].values
            ax2.bar(range(len(sig)), sig, color=[
                COLORS["signal_pos"] if s > 0 else
                COLORS["signal_neg"] if s < 0 else "#D1D5DB"
                for s in sig
            ], alpha=0.7)
    ax2.set_ylabel(spread_col if spread_col in sg.columns else "Signal")
    ax2.legend(loc="upper right", fontsize=9)

    # --- Panel 3: Positions ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for col in po.columns:
        ax3.plot(po[col].values, linewidth=1.0, label=col, alpha=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Position (USDT)")
    ax3.set_xlabel("Bar index (hours)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# =============================================================================
# 6. RETURN DISTRIBUTION (bonus diagnostic for the report)
# =============================================================================

def plot_return_distribution(
    strategies: dict[str, pd.Series],
    save_path:  Path,
) -> None:
    """
    Histogram of per-bar returns for both strategies, overlaid.
    Useful for showing fat tails and discussing VaR in the report.

    Args:
        strategies : dict mapping name → net returns Series.
        save_path  : Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    color_map = {
        "pairs_cointegration": COLORS["pairs_gross"],
        "trend_following":     COLORS["trend_gross"],
    }

    for name, returns in strategies.items():
        color = color_map.get(name, "#6B7280")
        ax.hist(returns.values, bins=100, alpha=0.5, color=color,
                label=name.replace("_", " ").title(), density=True)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Per-bar Return Distribution (net of slippage)", fontweight="bold")
    ax.set_xlabel("Per-bar return")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")
