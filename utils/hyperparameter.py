# =============================================================================
# utils/hyperparameter.py — COMP0051 Algorithmic Trading Coursework
# Strategy parameter search — adapted from the legacy successive-halving
# hyperparameter search (COMP0197 Applied Deep Learning).
#
# Preserved verbatim (domain-agnostic):
#   Leaderboard class (with added mode parameter)
#   sample_uniform, sample_log_uniform, sample_parameter
#   set_nested, sample_config, prune, select_best
#
# Changed:
#   staged_search() → staged_search_strategy()
#   Objective: min(validation_loss) → max(validation_sharpe)
#   Sessions: TrainingSession → StrategySession
#   Schedule: epochs+keep → bars+keep (same structure)
#
# The search loop skeleton is identical. Only what gets evaluated changes.
# =============================================================================

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from utils.common          import save_json, run_strategy
from utils.strategy_session import StrategySession
from utils.execution        import build_mvo_executor
from utils.portfolio        import PnLEngine
from utils.metrics          import sharpe, HOURS_PER_YEAR


# =============================================================================
# LEADERBOARD — kept verbatim with mode parameter added
# =============================================================================

class Leaderboard:
    """
    Simple leaderboard tracking search candidate performance.

    Adapted from legacy Leaderboard — same structure, added mode parameter:
        mode='max_sharpe'  → higher Sharpe = better (default for trading)
        mode='min_loss'    → lower loss = better (legacy neural-net mode)

    The internal 'loss' field stores a comparable scalar for ranked():
        max_sharpe mode: loss = -sharpe (negated so sort ascending = best first)
        min_loss mode:   loss = validation_loss (same as legacy)
    """

    def __init__(self, sessions: list, mode: str = "max_sharpe", val_start: int = 0):
        self.mode      = mode
        self.val_start = val_start
        self.entries   = []
        for cfg, session in sessions:
            loss = self._score(session)
            self.entries.append({"config": cfg, "session": session, "loss": loss})

    def _score(self, session: StrategySession) -> float:
        """
        Compute the comparable score for a session.
        Lower is better (we always sort ascending, negate for max problems).
        """
        if self.mode == "max_sharpe":
            s = session.get_val_sharpe(self.val_start)
            return -s if not np.isnan(s) else np.inf
        else:
            # min_loss mode — for future extensibility
            metrics = session.results.get("bar_metrics", [])
            if not metrics:
                return np.inf
            return min(m.get("sharpe_net", np.inf) for m in metrics)

    def add(self, cfg: dict, session: StrategySession) -> None:
        """Add a new candidate to the leaderboard."""
        loss = self._score(session)
        self.entries.append({"config": cfg, "session": session, "loss": loss})

    def ranked(self) -> list:
        """Return all entries sorted by loss ascending (best first)."""
        return sorted(self.entries, key=lambda x: x["loss"])

    def top(self, k: int) -> list:
        """Return the top-k candidates."""
        return self.ranked()[:k]


# =============================================================================
# SAMPLING UTILITIES — kept verbatim from legacy
# =============================================================================

def sample_uniform(low: float, high: float) -> float:
    return random.uniform(low, high)


def sample_log_uniform(low: float, high: float) -> float:
    return 10 ** random.uniform(math.log10(low), math.log10(high))


def sample_parameter(low: float, high: float, mode: str) -> float:
    if mode == "uniform":
        return sample_uniform(low, high)
    if mode == "log":
        return sample_log_uniform(low, high)
    raise ValueError(f"Unknown sampling mode: {mode}")


def set_nested(cfg: dict, key: str, value) -> None:
    """
    Set a value in a (potentially nested) dict using dot-notation key.
    Kept verbatim from legacy.

    Example:
        set_nested(cfg, "optimiser_params.lr", 0.001)
        → cfg["optimiser_params"]["lr"] = 0.001
    """
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def sample_config(base_config: dict, search_space: dict) -> dict:
    """
    Sample a random config by perturbing base_config within search_space.
    Kept verbatim from legacy.

    Args:
        base_config  : Starting params dict (from strategy YAML).
        search_space : Dict of param → (low, high, mode) tuples.

    Returns:
        New params dict with sampled values merged in.
    """
    cfg = {} if base_config is None else copy.deepcopy(base_config)
    for param, (low, high, mode) in search_space.items():
        set_nested(cfg, param, sample_parameter(low, high, mode))
    return cfg


def prune(sessions: list, keep: int) -> list:
    """
    Prune sessions to the top-k using Leaderboard ranking.
    Kept verbatim from legacy — returns (cfg, session) pairs.
    """
    leaderboard = Leaderboard(sessions)
    best = leaderboard.top(keep)
    return [(e["config"], e["session"]) for e in best]


def select_best(sessions: list) -> tuple:
    """
    Select the single best session by validation Sharpe.
    Analogous to legacy select_best — same interface, Sharpe not val_loss.
    """
    best_score = -np.inf
    best       = None
    for cfg, session in sessions:
        score = session.best_sharpe
        if score > best_score:
            best_score = score
            best       = (cfg, session)
    return best


# =============================================================================
# STAGED SEARCH — main search function
# =============================================================================

def staged_search_strategy(
    search_space:   dict,
    prices:         "pd.DataFrame",
    returns:        "pd.DataFrame",
    signal_builder,
    execution_step,
    strategy_dir:   Path,
    base_params:    dict        = None,
    schedule:       list        = None,
    initial_models: int         = 15,
    val_fraction:   float       = 0.30,
    capital:        float       = 10_000.0,
    search_name:    str         = "param_search",
) -> dict:
    """
    Successive-halving hyperparameter search over strategy parameters.

    Analogous to legacy staged_search() — same skeleton:
        1. Sample initial_models random configs
        2. For each stage: run all survivors for bars_this_stage bars
        3. Prune to top-k using validation Sharpe
        4. Repeat until 1 survivor remains
        5. Save search summary JSON

    Objective: maximise Sharpe ratio on the validation window.
    Train window: bars 0 → train_end
    Validation window: bars train_end → end

    The 70/30 train/val split prevents p-hacking (Lecture 9 requirement).

    Args:
        search_space    : Dict of param → (low, high, mode) tuples.
        prices          : Full close price DataFrame.
        returns         : Full excess returns DataFrame.
        signal_builder  : Strategy signal builder callable (from registry).
        execution_step  : Execution step callable (from registry).
        strategy_dir    : Directory for saving search JSON summary.
        base_params     : Base params to perturb. None = start from zeros.
        schedule        : List of {"bars": int, "keep": int} dicts.
                          Default: 3-stage schedule.
        initial_models  : Number of random configs to initialise.
        val_fraction    : Fraction of data held out for validation (default 0.30).
        capital         : Portfolio capital in USDT.
        search_name     : Name for the saved JSON summary file.

    Returns:
        Dict of best params (merged from base_params + winning sample).
    """
    n_bars      = len(prices)
    train_end   = int(n_bars * (1 - val_fraction))
    val_start   = train_end

    if schedule is None:
        schedule = [
            {"bars": train_end // 3,     "keep": math.ceil(initial_models / 2)},
            {"bars": train_end // 3,     "keep": math.ceil(initial_models / 4)},
            {"bars": train_end - 2 * (train_end // 3), "keep": 1},
        ]

    print(f"\n[search] Staged search — {initial_models} initial configs")
    print(f"[search] Train bars: {train_end} | Val bars: {n_bars - train_end}")
    print(f"[search] Schedule: {schedule}")

    # --- Initialise search candidates ---
    sessions    = []
    run_records = {}

    for i in range(initial_models):
        cfg       = sample_config(base_params, search_space)
        strategy  = signal_builder(cfg)
        executor  = build_mvo_executor(cfg)
        pnl_engine = PnLEngine(
            initial_capital = capital,
            slippage        = float(cfg.get("slippage", 0.001)),
        )
        session         = StrategySession(strategy, executor, pnl_engine, cfg)
        session.id      = f"candidate_{i}"
        sessions.append((cfg, session))

    # --- Staged successive halving ---
    for stage_idx, stage in enumerate(schedule):
        bars_this_stage = stage["bars"]
        keep            = stage["keep"]

        print(f"\n[search] Stage {stage_idx + 1}: running {len(sessions)} "
              f"candidates for {bars_this_stage} bars each")

        for i, (cfg, session) in enumerate(sessions):
            # Use training window only (bar_idx continues from previous stage)
            n_remaining = train_end - session.bar_idx
            bars        = min(bars_this_stage, max(n_remaining, 0))

            if bars <= 0:
                continue

            session.run(
                n_bars  = bars,
                prices  = prices,
                returns = returns,
                capital = capital,
            )

            # Record metrics for JSON summary
            if session.id not in run_records:
                run_records[session.id] = {
                    "id":          session.id,
                    "config":      cfg.copy(),
                    "bar_metrics": [],
                }
            run_records[session.id]["bar_metrics"] = session.results["bar_metrics"]

        if keep is not None and keep < len(sessions):
            sessions = prune(sessions, keep=keep)
            print(f"[search] Pruned to {len(sessions)} candidates")

    # --- Select winner ---
    best_cfg, best_session = select_best(sessions)
    best_sharpe = best_session.get_val_sharpe(val_start)
    best_metrics = {
        "val_sharpe":  best_sharpe,
        "train_bars":  train_end,
        "val_bars":    n_bars - train_end,
    }

    # --- Save search summary ---
    search_summary = {
        "search_type":       "successive_halving_strategy",
        "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_models":    initial_models,
        "schedule":          schedule,
        "val_fraction":      val_fraction,
        "search_space":      {k: list(v) for k, v in search_space.items()},
        "best_config":       best_cfg,
        "best_val_metrics":  best_metrics,
        "runs":              list(run_records.values()),
    }

    strategy_dir.mkdir(parents=True, exist_ok=True)
    summary_path = strategy_dir / f"{search_name}.json"
    save_json(search_summary, summary_path)
    print(f"\n[search] Summary saved → {summary_path}")
    print(f"[search] Best val Sharpe: {best_sharpe:.3f}")
    print(f"[search] Best config: {best_cfg}")

    return best_cfg
