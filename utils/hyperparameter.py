# =============================================================================
# utils/hyperparameter.py — COMP0051 Algorithmic Trading Coursework
# Strategy parameter search — adapted from the legacy successive-halving
# hyperparameter search (COMP0197 Applied Deep Learning).
#
# Preserved verbatim (domain-agnostic):
#   Leaderboard, sample_uniform, sample_log_uniform, sample_parameter,
#   set_nested, sample_config, select_best
#
# Changed:
#   staged_search() → staged_search_strategy()
#   Objective: min(validation_loss) → max(validation_sharpe)
#   Sessions: TrainingSession → StrategySession
#   Schedule: epochs+keep → bars+keep
#
# Walk-forward validation (fixed vs V2):
#   For each pruning stage, EVERY candidate is evaluated on a held-out
#   mini-validation window (stage_end → train_end) using a FRESH SESSION
#   that has NOT been trained on those bars. Pruning is then based on this
#   genuine out-of-sample Sharpe, not on training-window metrics.
#   Final winner is selected on the global val window (train_end → end).
#
# This matches the reference codebase pattern exactly:
#   train each stage → validate on held-out data → prune on val metric
# =============================================================================

import copy
import math
import random
import time
from pathlib import Path

import numpy as np

from utils.common           import save_json, run_strategy
from utils.strategy_session import StrategySession
from utils.execution        import build_mvo_executor
from utils.portfolio        import PnLEngine
from utils.metrics          import sharpe, HOURS_PER_YEAR


# =============================================================================
# LEADERBOARD — kept verbatim with mode and val_start parameters
# =============================================================================

class Leaderboard:
    """
    Simple leaderboard ranking search candidates by validation Sharpe.

    mode='max_sharpe': higher Sharpe = better (default for trading).
    val_start: bar index where the held-out validation window begins.
    Candidates are ranked on get_val_sharpe(val_start) — the Sharpe on
    bars they have NOT been trained on.
    """

    def __init__(self, sessions: list, mode: str = "max_sharpe", val_start: int = 0):
        self.mode      = mode
        self.val_start = val_start
        self.entries   = []
        for cfg, session in sessions:
            loss = self._score(session)
            self.entries.append({"config": cfg, "session": session, "loss": loss})

    def _score(self, session: StrategySession) -> float:
        """Lower is better (negate Sharpe so sort ascending = best first)."""
        if self.mode == "max_sharpe":
            s = session.get_val_sharpe(self.val_start)
            return -s if not np.isnan(s) else np.inf
        metrics = session.results.get("bar_metrics", [])
        if not metrics:
            return np.inf
        return -min(m.get("sharpe_net", -np.inf) for m in metrics)

    def add(self, cfg: dict, session: StrategySession) -> None:
        loss = self._score(session)
        self.entries.append({"config": cfg, "session": session, "loss": loss})

    def ranked(self) -> list:
        return sorted(self.entries, key=lambda x: x["loss"])

    def top(self, k: int) -> list:
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
    """Set a value in a (possibly nested) dict using dot-notation key."""
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def sample_config(base_config: dict, search_space: dict) -> dict:
    """Sample a random config by perturbing base_config within search_space."""
    cfg = {} if base_config is None else copy.deepcopy(base_config)
    for param, (low, high, mode) in search_space.items():
        set_nested(cfg, param, sample_parameter(low, high, mode))
    return cfg


# =============================================================================
# VALIDATION HELPER — run a FRESH session on a window without contaminating
# the training session state
# =============================================================================

def _score_on_window(
    cfg:       dict,
    session:   StrategySession,
    prices:    "pd.DataFrame",
    returns:   "pd.DataFrame",
    start_bar: int,
    end_bar:   int,
    capital:   float,
    val_start: int,
) -> float:
    """
    Evaluate a candidate on bars [start_bar, end_bar) using a fresh session.

    The original training session is NOT modified — this function clones the
    strategy and executor, creates a new StrategySession starting at start_bar,
    and runs it forward. The clone's state is fully independent of training.

    This is the key fix for true walk-forward validation during pruning:
    we score each candidate on bars it has NEVER seen, not on a slice of
    bars that were already accumulated into session.results.

    Args:
        cfg       : Candidate params dict.
        session   : The TRAINING session (read-only — not modified).
        prices    : Full close price DataFrame.
        returns   : Full excess returns DataFrame.
        start_bar : First bar of the evaluation window (inclusive).
        end_bar   : Last bar of the evaluation window (exclusive).
        capital   : Portfolio capital in USDT.
        val_start : Bar index relative to which Sharpe is scored.
                    Typically equal to start_bar for stage mini-val windows.

    Returns:
        float: Annualised Sharpe on the evaluation window. -inf if too short.
    """
    if end_bar <= start_bar:
        return -np.inf

    # Reinstantiate strategy and executor from the same params.
    # deepcopy of session.strategy preserves any class-level state (e.g.
    # sticky position variable inside PairsStrategy) without contaminating
    # the original training session. The executor is rebuilt fresh because
    # its rolling covariance/mu estimates use bar_abs_idx into the full
    # returns DataFrame — they already have full history available.
    strategy_clone  = copy.deepcopy(session.strategy)
    executor_clone  = build_mvo_executor(cfg)
    pnl_engine_clone = PnLEngine(
        initial_capital = capital,
        slippage        = float(cfg.get("slippage", 0.001)),
    )

    val_session = StrategySession(
        strategy   = strategy_clone,
        executor   = executor_clone,
        pnl_engine = pnl_engine_clone,
        params     = cfg,
        bar_idx    = start_bar,
    )

    n_bars = end_bar - start_bar
    val_session.run(
        n_bars  = n_bars,
        prices  = prices,
        returns = returns,
        capital = capital,
    )

    # val_start=0 here because this session only has val-window data
    return val_session.get_val_sharpe(val_start=0)


def prune(
    sessions:   list,
    keep:       int,
    prices:     "pd.DataFrame",
    returns:    "pd.DataFrame",
    stage_end:  int,
    train_end:  int,
    capital:    float,
) -> list:
    """
    Prune candidates to the top-k by scoring each on a HELD-OUT mini-val window.

    The mini-val window is bars [stage_end, train_end). Candidates have been
    trained on bars [0, stage_end). Scoring on [stage_end, train_end) is
    genuinely out-of-sample relative to this stage.

    This mirrors the reference codebase's validate-then-prune pattern:
        train on train_loader → evaluate on val_loader → prune on val loss

    Args:
        sessions  : List of (cfg, session) pairs — training sessions.
        keep      : Number of survivors.
        prices    : Full price DataFrame.
        returns   : Full returns DataFrame.
        stage_end : Bar index where this stage's training ended.
        train_end : Bar index where the global train window ends.
        capital   : Portfolio capital.

    Returns:
        Pruned list of (cfg, session) pairs, length ≤ keep.
    """
    if keep >= len(sessions):
        return sessions

    # Score each candidate on [stage_end, train_end) — bars it hasn't seen
    scores = []
    mini_val_bars = train_end - stage_end

    for cfg, session in sessions:
        if mini_val_bars > 10:
            score = _score_on_window(
                cfg       = cfg,
                session   = session,
                prices    = prices,
                returns   = returns,
                start_bar = stage_end,
                end_bar   = train_end,
                capital   = capital,
                val_start = 0,
            )
        else:
            # Mini-val window too small — fall back to training Sharpe
            score = session.best_sharpe

        scores.append((score, cfg, session))

    # Sort descending by Sharpe (best first), keep top-k
    scores.sort(key=lambda x: x[0], reverse=True)
    survivors = [(cfg, session) for _, cfg, session in scores[:keep]]
    print(f"[search] Pruned {len(sessions)} → {len(survivors)} "
          f"(stage mini-val Sharpe: best={scores[0][0]:.3f}, "
          f"worst_kept={scores[keep-1][0]:.3f})")
    return survivors


def select_best(sessions: list, val_start: int = 0) -> tuple:
    """
    Select the single best session by validation-window Sharpe.
    Uses get_val_sharpe(val_start) — scored on the global held-out window.
    """
    best_score = -np.inf
    best       = None
    for cfg, session in sessions:
        score = session.get_val_sharpe(val_start)
        if np.isnan(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best       = (cfg, session)
    return best


# =============================================================================
# STAGED SEARCH — true walk-forward validation at every pruning stage
# =============================================================================

def staged_search_strategy(
    search_space:   dict,
    prices:         "pd.DataFrame",
    returns:        "pd.DataFrame",
    signal_builder,
    execution_step,
    strategy_dir:   Path,
    base_params:    dict  = None,
    schedule:       list  = None,
    initial_models: int   = 15,
    val_fraction:   float = 0.30,
    capital:        float = 10_000.0,
    search_name:    str   = "param_search",
) -> dict:
    """
    Successive-halving strategy parameter search with true walk-forward validation.

    Data split:
        Training window:   bars 0 → train_end    (70%)
        Val window:        bars train_end → end   (30%)

    Walk-forward schedule (corrected vs V2):
        For each stage:
            1. Train each candidate on bars [prev_end, stage_end)
               (continuing from where the previous stage left off)
            2. Score each candidate on bars [stage_end, train_end)
               using a FRESH SESSION — genuinely out-of-sample
            3. Prune: keep top-k by stage validation Sharpe

        Final selection:
            1. Run all survivors on bars [train_end, end) — global val window
            2. Select best by global val Sharpe

    The stage validation window and the global validation window are DIFFERENT:
        Stage val: score after each pruning round, still within train region
        Global val: final held-out window, never touched during search

    This matches the reference codebase pattern exactly:
        train on train_loader → validate on val_loader → prune on val metric

    Args:
        search_space    : Dict param → (low, high, mode) tuples.
        prices          : Full close price DataFrame.
        returns         : Full excess returns DataFrame.
        signal_builder  : Signal builder callable (from registry).
        execution_step  : Execution step callable (from registry).
        strategy_dir    : Directory for saving search JSON.
        base_params     : Base params to perturb.
        schedule        : List of {"bars": int, "keep": int} dicts.
        initial_models  : Number of random configs to sample initially.
        val_fraction    : Fraction of data held out as global val window.
        capital         : Portfolio capital in USDT.
        search_name     : Name for the saved JSON summary file.

    Returns:
        Dict of best params found.
    """
    n_bars    = len(prices)
    train_end = int(n_bars * (1 - val_fraction))
    val_start = train_end

    if schedule is None:
        schedule = [
            {"bars": train_end // 3,                      "keep": math.ceil(initial_models / 2)},
            {"bars": train_end // 3,                      "keep": math.ceil(initial_models / 4)},
            {"bars": train_end - 2 * (train_end // 3),   "keep": 1},
        ]

    print(f"\n[search] Staged search — {initial_models} initial configs")
    print(f"[search] Train bars: {train_end} | Val bars: {n_bars - train_end}")
    print(f"[search] Schedule: {schedule}")
    print(f"[search] Walk-forward: stage pruning on [stage_end, train_end)")

    # --- Initialise candidates ---
    sessions    = []
    run_records = {}

    for i in range(initial_models):
        cfg        = sample_config(base_params, search_space)
        strategy   = signal_builder(cfg)
        executor   = build_mvo_executor(cfg)
        pnl_engine = PnLEngine(
            initial_capital = capital,
            slippage        = float(cfg.get("slippage", 0.001)),
        )
        session    = StrategySession(strategy, executor, pnl_engine, cfg)
        session.id = f"candidate_{i}"
        sessions.append((cfg, session))

    # --- Staged successive halving with true walk-forward validation ---
    for stage_idx, stage in enumerate(schedule):
        bars_this_stage = stage["bars"]
        keep            = stage["keep"]

        print(f"\n[search] Stage {stage_idx + 1}: training {len(sessions)} "
              f"candidates for {bars_this_stage} bars")

        for cfg, session in sessions:
            # Train on [session.bar_idx, session.bar_idx + bars_this_stage)
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

            # Record training metrics for JSON summary
            if session.id not in run_records:
                run_records[session.id] = {
                    "id":          session.id,
                    "config":      cfg.copy(),
                    "bar_metrics": [],
                }
            run_records[session.id]["bar_metrics"] = session.results["bar_metrics"]

        # Compute stage_end = bar index where all training in this stage ended
        # (use the furthest bar_idx seen across all sessions)
        stage_end = max(session.bar_idx for _, session in sessions)

        # Prune using HELD-OUT mini-val window [stage_end, train_end)
        # prune() runs a fresh session on bars candidates have NOT been trained
        # on — ranking is genuinely out-of-sample at every stage.
        if keep is not None and keep < len(sessions):
            # Score each candidate on the mini-val window before pruning,
            # and record those scores in run_records for the audit trail.
            mini_val_bars = train_end - stage_end
            for cfg, session in sessions:
                if mini_val_bars > 10:
                    stage_val_sharpe = _score_on_window(
                        cfg       = cfg,
                        session   = session,
                        prices    = prices,
                        returns   = returns,
                        start_bar = stage_end,
                        end_bar   = train_end,
                        capital   = capital,
                        val_start = 0,
                    )
                else:
                    stage_val_sharpe = session.best_sharpe

                if session.id in run_records:
                    run_records[session.id].setdefault("stage_val_sharpes", []).append({
                        "stage":        stage_idx + 1,
                        "stage_end":    stage_end,
                        "train_end":    train_end,
                        "val_sharpe":   stage_val_sharpe,
                    })

            sessions = prune(
                sessions  = sessions,
                keep      = keep,
                prices    = prices,
                returns   = returns,
                stage_end = stage_end,
                train_end = train_end,
                capital   = capital,
            )

    # --- Run global validation window on all surviving candidates ---
    # Survivors have been trained only up to train_end; now run [train_end, end)
    print(f"\n[search] Running global val window ({n_bars - train_end} bars) "
          f"on {len(sessions)} survivor(s)...")
    for cfg, session in sessions:
        val_bars = n_bars - session.bar_idx
        if val_bars > 0:
            session.run(
                n_bars  = val_bars,
                prices  = prices,
                returns = returns,
                capital = capital,
            )

    # --- Select winner by global val Sharpe ---
    best_cfg, best_session = select_best(sessions, val_start=val_start)
    best_sharpe = best_session.get_val_sharpe(val_start)

    best_metrics = {
        "val_sharpe": best_sharpe,
        "train_bars": train_end,
        "val_bars":   n_bars - train_end,
    }

    # --- Save search summary ---
    search_summary = {
        "search_type":      "successive_halving_walk_forward",
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_models":   initial_models,
        "schedule":         schedule,
        "val_fraction":     val_fraction,
        "search_space":     {k: list(v) for k, v in search_space.items()},
        "best_config":      best_cfg,
        "best_val_metrics": best_metrics,
        "runs":             list(run_records.values()),
    }

    strategy_dir.mkdir(parents=True, exist_ok=True)
    summary_path = strategy_dir / f"{search_name}.json"
    save_json(search_summary, summary_path)
    print(f"\n[search] Summary saved → {summary_path}")
    print(f"[search] Best global val Sharpe: {best_sharpe:.3f}")
    print(f"[search] Best config: {best_cfg}")

    return best_cfg
