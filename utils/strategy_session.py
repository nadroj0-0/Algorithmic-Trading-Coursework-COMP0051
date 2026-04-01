# =============================================================================
# utils/strategy_session.py — COMP0051 Algorithmic Trading Coursework
# Strategy session container — adapted from TrainingSession (COMP0197).
#
# TrainingSession kept: (model, optimiser, criterion, history, epoch_counter)
# StrategySession keeps: (strategy, executor, pnl_engine, results, bar_idx)
#
# The architectural value is identical:
#   - Keeps all state for one candidate self-contained
#   - Allows staged search to run N bars, store results, then continue
#     from bar N in the next stage — same as epoch continuation in search
#   - Provides a clean container that evaluate.py can inspect later
#
# Used by hyperparameter.py's staged_search_strategy() in exactly the same
# way TrainingSession was used by staged_search().
# =============================================================================

import copy
import numpy as np
import pandas as pd

from utils.portfolio import PnLEngine
from utils.metrics   import sharpe, HOURS_PER_YEAR


# =============================================================================
# STRATEGY SESSION CLASS
# =============================================================================

class StrategySession:
    """
    Self-contained container for one strategy run or search candidate.

    Analogous to TrainingSession — holds all state needed to run a strategy
    and resume from a previous bar index. Designed for staged search:
    run for N bars, evaluate, prune, continue survivors.

    Args:
        strategy   : PairsStrategy or TrendStrategy instance.
        executor   : MVOExecutor instance (position sizer).
        pnl_engine : PnLEngine instance (PnL computation).
        params     : Strategy params dict (from YAML, merged with best_config).
        bar_idx    : Starting bar index. Default 0 (beginning of data).
        results    : Existing results dict for continuation. Default None.
    """

    def __init__(
        self,
        strategy,
        executor,
        pnl_engine: PnLEngine,
        params:     dict,
        bar_idx:    int  = 0,
        results:    dict = None,
    ):
        self.strategy   = strategy
        self.executor   = executor
        self.pnl_engine = pnl_engine
        self.params     = params
        self.bar_idx    = bar_idx

        # Results dict — analogous to TrainingSession.history
        # Accumulates across multiple run() calls for staged search
        self.results = results or {
            "bar_metrics":  [],    # per-bar metrics (analogous to epoch_metrics)
            "pnl_gross":    None,  # pd.Series (built up across run() calls)
            "pnl_net":      None,
            "costs":        None,
            "value_gross":  None,
            "value_net":    None,
            "positions":    None,  # all positions for evaluate.py
        }

        # Best Sharpe seen so far — analogous to EarlyStopping tracking
        self.best_sharpe  = -np.inf
        self.best_bar_idx = 0
        self.best_state   = None

        # Unique ID for leaderboard tracking (set by staged_search_strategy)
        self.id = None

    # -------------------------------------------------------------------------
    # CORE SEGMENT RUNNER — single implementation, called by run()
    # -------------------------------------------------------------------------

    def _run_segment(
        self,
        start:   int,
        end:     int,
        prices:  pd.DataFrame,
        returns: pd.DataFrame,
        capital: float,
    ) -> dict:
        """
        Execute the strategy pipeline for bars [start, end).

        This is the single implementation of signal generation →
        position sizing → PnL computation. run() delegates here so
        there is exactly one code path, eliminating the previous
        duplication between run() and _run_segment().

        Steps:
            1. Slice prices and returns to [start, end)
            2. Generate directional signals from self.strategy
            3. Size positions bar-by-bar via self.executor
            4. Shift positions by 1 bar to eliminate lookahead bias
            5. Compute PnL via self.pnl_engine
            6. Accumulate into self.results

        Args:
            start   : First bar index (inclusive, into full prices/returns).
            end     : Last bar index (exclusive).
            prices  : Full close price DataFrame.
            returns : Full excess returns DataFrame.
            capital : Portfolio capital in USDT.

        Returns:
            PnL dict from PnLEngine.run() for this segment.
        """
        prices_seg  = prices.iloc[start:end]
        returns_seg = returns.iloc[start:end]

        # --- Generate signals ---
        signal_df  = self.strategy.generate(prices_seg)
        # Keep only asset columns — drop diagnostic cols (spread, zscore etc.)
        asset_cols = [c for c in signal_df.columns if c in prices.columns]
        signal_df  = signal_df[asset_cols]

        # --- Size positions bar by bar ---
        positions_list = []
        for i in range(len(signal_df)):
            bar_abs_idx = start + i
            signals     = signal_df.iloc[i]
            theta       = self.executor.size(
                signals = signals,
                returns = returns,
                capital = capital,
                bar_idx = bar_abs_idx,
            )
            positions_list.append(theta)

        positions_df = pd.DataFrame(
            positions_list,
            index   = signal_df.index,
            columns = asset_cols,
        )

        # Shift by 1 bar — positions sized at bar t are traded at bar t+1.
        # This is the standard lookahead-bias prevention technique.
        positions_df = positions_df.shift(1).fillna(0)

        # --- Compute PnL ---
        ret_seg  = returns_seg[asset_cols]
        pnl_dict = self.pnl_engine.run(positions_df, ret_seg, prices_seg[asset_cols])

        # --- Accumulate into self.results ---
        self._accumulate(pnl_dict, positions_df)

        return pnl_dict

    # -------------------------------------------------------------------------
    # PUBLIC RUN METHOD — delegates to _run_segment, tracks best Sharpe
    # -------------------------------------------------------------------------

    def run(
        self,
        n_bars:   int,
        prices:   pd.DataFrame,
        returns:  pd.DataFrame,
        capital:  float,
    ) -> dict:
        """
        Run the strategy for n_bars starting from self.bar_idx.

        Delegates to _run_segment() — all signal/position/PnL logic lives
        in one place. Analogous to TrainingSession.train(epochs, ...).

        After the segment runs, records a bar_metric entry and tracks the
        best Sharpe seen so far over the accumulated PnL.

        Args:
            n_bars  : Number of bars to run.
            prices  : Full close price DataFrame (all bars, all assets).
            returns : Full excess returns DataFrame (all bars, all assets).
            capital : Portfolio capital in USDT.

        Returns:
            PnL dict from PnLEngine.run() for this segment.
        """
        start = self.bar_idx
        end   = min(start + n_bars, len(prices))

        if start >= end:
            return {}

        pnl_dict = self._run_segment(start, end, prices, returns, capital)

        # --- Track best Sharpe on accumulated PnL (entire history so far) ---
        net_ret    = self.results["pnl_net"] / self.pnl_engine.initial_capital
        cur_sharpe = float(sharpe(net_ret, periods=HOURS_PER_YEAR))
        if cur_sharpe > self.best_sharpe:
            self.best_sharpe  = cur_sharpe
            self.best_bar_idx = end
            self.best_state   = copy.deepcopy(self.results)

        # --- Advance bar index ---
        self.bar_idx = end

        # --- Record bar metric (analogous to epoch_metrics entry) ---
        bar_metric = {
            "bar_start":       start,
            "bar_end":         end,
            "sharpe_net":      cur_sharpe,
            "total_pnl_net":   float(pnl_dict.get("total_pnl_net",   0)),
            "total_pnl_gross": float(pnl_dict.get("total_pnl_gross", 0)),
            "turnover":        float(pnl_dict.get("turnover",         0)),
        }
        self.results["bar_metrics"].append(bar_metric)

        return pnl_dict

    # -------------------------------------------------------------------------
    # ACCUMULATION — appends segment results into running totals
    # -------------------------------------------------------------------------

    def _accumulate(
        self,
        pnl_dict:     dict,
        positions_df: pd.DataFrame,
    ) -> None:
        """
        Accumulate segment results into self.results.
        Analogous to TrainingSession's epoch-continuation logic:
        history["epoch_metrics"].extend(new_metrics).
        """
        for key in ("pnl_gross", "pnl_net", "costs", "value_gross", "value_net"):
            seg_series = pnl_dict[key]
            if self.results[key] is None:
                self.results[key] = seg_series.copy()
            else:
                self.results[key] = pd.concat([self.results[key], seg_series])

        if self.results["positions"] is None:
            self.results["positions"] = positions_df.copy()
        else:
            self.results["positions"] = pd.concat(
                [self.results["positions"], positions_df]
            )

    # -------------------------------------------------------------------------
    # VALIDATION SCORING — used by Leaderboard in hyperparameter.py
    # -------------------------------------------------------------------------

    def get_val_sharpe(self, val_start: int) -> float:
        """
        Compute Sharpe on the validation window (bars from val_start onward).

        Analogous to reading min(epoch_metrics["validation_loss"]) in the
        reference codebase — the score used to rank search candidates.

        val_start must be the bar index where the held-out window begins,
        matching train_end from staged_search_strategy(). Passing 0 would
        score on all data including training bars — defeating the purpose.

        Args:
            val_start : Bar index where validation window begins (= train_end).

        Returns:
            float: Annualised Sharpe ratio on validation window.
                   -inf if insufficient data.
        """
        if self.results["pnl_net"] is None:
            return -np.inf
        net_pnl = self.results["pnl_net"]
        val_pnl = net_pnl.iloc[val_start:] if len(net_pnl) > val_start else net_pnl
        ret     = val_pnl / self.pnl_engine.initial_capital
        if len(ret) < 10:
            return -np.inf
        return float(sharpe(ret, periods=HOURS_PER_YEAR))

    def __repr__(self) -> str:
        return (
            f"StrategySession(strategy={self.strategy}, "
            f"bar_idx={self.bar_idx}, "
            f"best_sharpe={self.best_sharpe:.3f})"
        )
