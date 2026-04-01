# =============================================================================
# utils/hyperparameter.py — COMP0051 Algorithmic Trading Coursework
# Strategy parameter search — adapted from the legacy successive-halving
# hyperparameter search (COMP0197 Applied Deep Learning).
#
# Preserved verbatim (domain-agnostic):
#   Leaderboard, sample_uniform, sample_log_uniform, sample_parameter,
#   set_nested, sample_config
#
# Walk-forward design (V6):
#   3-window cross-validation per config at each pruning stage.
#   For each config, 3 sessions are maintained in parallel, each trained
#   on a different prefix of the training data and validated on the
#   immediately following non-overlapping window.
#
#   _window_bounds(stage_end, w, W):
#       w=1: trains [0, stage_end-W),   val=[stage_end-W, stage_end)
#       w=2: trains [0, stage_end-2W),  val=[stage_end-2W, stage_end-W)
#       w=3: trains [0, stage_end-3W),  val=[stage_end-3W, stage_end-2W)
#   Average val Sharpe across all 3 windows is used for pruning.
#
#   Final selection uses the global held-out window [train_end, end)
#   — the 30% that was never touched during any stage.
# =============================================================================

import copy
import math
import random
import time
from pathlib import Path

import numpy as np

from utils.common           import save_json
from utils.strategy_session import StrategySession
from utils.execution        import build_mvo_executor
from utils.portfolio        import PnLEngine
from utils.metrics          import sharpe, HOURS_PER_YEAR


# =============================================================================
# LEADERBOARD
# =============================================================================

class Leaderboard:
    """
    Ranks search candidates by validation Sharpe (higher = better).
    Internally stores negative Sharpe so sort ascending = best first.
    val_start: bar index from which get_val_sharpe() is scored.
    """

    def __init__(self, sessions: list, mode: str = "max_sharpe", val_start: int = 0):
        self.mode      = mode
        self.val_start = val_start
        self.entries   = []
        for cfg, session in sessions:
            loss = self._score(session)
            self.entries.append({"config": cfg, "session": session, "loss": loss})

    def _score(self, session) -> float:
        if self.mode == "max_sharpe":
            s = session.get_val_sharpe(self.val_start)
            return -s if not np.isnan(s) else np.inf
        metrics = session.results.get("bar_metrics", []) if hasattr(session, "results") else []
        if not metrics:
            return np.inf
        return -min(m.get("sharpe_net", -np.inf) for m in metrics)

    def add(self, cfg: dict, session) -> None:
        self.entries.append({"config": cfg, "session": session, "loss": self._score(session)})

    def ranked(self) -> list:
        return sorted(self.entries, key=lambda x: x["loss"])

    def top(self, k: int) -> list:
        return self.ranked()[:k]


# =============================================================================
# GROUP SESSION — wraps 3 per-window sessions for one config
# =============================================================================

class GroupSession:
    """
    Wraps the three per-window training sessions for one config so that
    Leaderboard can rank it via get_val_sharpe().

    get_val_sharpe() calls _score_on_window() for each window session
    and returns the mean val Sharpe. This makes pruning based on the
    average held-out performance across 3 non-overlapping windows.
    """

    def __init__(
        self,
        sessions:    list,
        stage_end:   int,
        window_size: int,
        prices,
        returns,
        capital:     float,
    ):
        self.sessions    = sessions
        self.stage_end   = stage_end
        self.window_size = window_size
        self.prices      = prices
        self.returns     = returns
        self.capital     = capital
        # Stub so Leaderboard._score can call .results without crashing
        self.results     = {"bar_metrics": []}

    def get_val_sharpe(self, val_start: int = 0) -> float:
        scores = []
        for s in self.sessions:
            score = _score_on_window(
                session     = s,
                stage_end   = self.stage_end,
                window_size = self.window_size,
                prices      = self.prices,
                returns     = self.returns,
                capital     = self.capital,
            )
            if score is not None and not np.isnan(score):
                scores.append(score)
        return float(np.mean(scores)) if scores else np.nan


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
# WINDOW HELPERS
# =============================================================================

def _window_bounds(stage_end: int, window_id: int, window_size: int) -> tuple:
    """
    Compute [val_start, val_end) for window window_id.

    Window layout (stage_end = X, window_size = W):
        w=1: val = [X-W,   X)
        w=2: val = [X-2W, X-W)
        w=3: val = [X-3W, X-2W)

    Each val window is immediately after its session's training window,
    so it is genuinely out-of-sample for that session.
    """
    val_end   = stage_end - (window_id - 1) * window_size
    val_start = stage_end - window_id * window_size
    return val_start, val_end


def _session_target_end(stage_end: int, window_id: int, window_size: int) -> int:
    """
    Training stop bar for window window_id.
    Session trains [0, val_start) so it has never seen its val window.
    """
    val_start, _ = _window_bounds(stage_end, window_id, window_size)
    return val_start


# =============================================================================
# VALIDATION SCORING — FIXED: fresh session, no training results carried over
# =============================================================================

def _score_on_window(
    session:     StrategySession,
    stage_end:   int,
    window_size: int,
    prices,
    returns,
    capital:     float,
) -> float:
    """
    Score a training session on its held-out validation window.

    Creates a FRESH eval session (results=None) starting at val_start.
    The eval session has no knowledge of the training session's PnL —
    it starts from scratch at val_start. This ensures:
        - pnl_net in eval_session ONLY contains val-window bars
        - get_val_sharpe(0) correctly scores those val-window bars
        - No index mismatch between bar indices and series positions

    This fixes the critical bug in the submitted version where
    results=deepcopy(session.results) was passed, causing:
        concat([train_pnl, val_pnl]) → pnl.iloc[val_start] always empty.

    The training session is NOT modified.
    """
    if not hasattr(session, "window_id"):
        return np.nan

    val_start, val_end = _window_bounds(stage_end, session.window_id, window_size)

    if val_start < 0 or val_end <= val_start:
        return np.nan

    n_val_bars = val_end - val_start
    if n_val_bars < 10:
        return np.nan

    # Create a FRESH eval session — results=None means empty PnL history.
    # warmup_bars is read from session.params so the strategy's rolling windows
    # (cointegration test, z-score, EWM) receive sufficient price history.
    # Without warmup, the first coint_window bars of a val window return NaN
    # signals, distorting val Sharpe. The warmup prefix contributes no PnL.
    eval_session = StrategySession(
        strategy   = copy.deepcopy(session.strategy),
        executor   = copy.deepcopy(session.executor),
        pnl_engine = PnLEngine(
            initial_capital = session.pnl_engine.initial_capital,
            slippage        = session.pnl_engine.slippage,
        ),
        params     = copy.deepcopy(session.params),
        bar_idx    = val_start,
        results    = None,    # FRESH — no training PnL carried over
    )

    eval_session.run(
        n_bars  = n_val_bars,
        prices  = prices,
        returns = returns,
        capital = capital,
    )

    # Score on position 0 onward (eval_session only has val-window bars)
    return eval_session.get_val_sharpe(val_start=0)


# =============================================================================
# PRUNE — uses GroupSession for 3-window averaged val Sharpe
# =============================================================================

def prune(
    sessions:    list,
    keep:        int,
    prices,
    returns,
    stage_end:   int,
    train_end:   int,
    capital:     float,
    window_size: int = 500,
) -> list:
    """
    Prune config groups to top-k by 3-window averaged validation Sharpe.

    Input: [(cfg, [s_w1, s_w2, s_w3]), ...]
    Each GroupSession averages _score_on_window() across its 3 sessions.
    """
    if keep >= len(sessions):
        return sessions

    grouped = [
        (cfg, GroupSession(group, stage_end, window_size, prices, returns, capital))
        for cfg, group in sessions
    ]

    leaderboard = Leaderboard(grouped, mode="max_sharpe", val_start=0)
    best_entries = leaderboard.top(keep)

    # Match best configs back to their original groups
    survivors = []
    for entry in best_entries:
        target_cfg = entry["config"]
        for orig_cfg, orig_group in sessions:
            if orig_cfg is target_cfg:   # identity check, not equality
                survivors.append((orig_cfg, orig_group))
                break

    if not survivors:
        # Fallback: identity check failed (shouldn't happen), use equality
        for entry in best_entries:
            target_cfg = entry["config"]
            for orig_cfg, orig_group in sessions:
                if orig_cfg == target_cfg:
                    survivors.append((orig_cfg, orig_group))
                    break

    best_score  = -best_entries[0]["loss"]  if best_entries else np.nan
    worst_kept  = -best_entries[-1]["loss"] if best_entries else np.nan
    print(f"[search] Pruned {len(sessions)} → {len(survivors)} "
          f"(3-window avg Sharpe: best={best_score:.3f}, worst_kept={worst_kept:.3f})")
    return survivors


# =============================================================================
# SELECT BEST — uses global val window [train_end, end)
# =============================================================================

def select_best(
    sessions:    list,
    val_start:   int,
    prices,
    returns,
    capital:     float,
    n_bars:      int,
) -> tuple:
    """
    Select the best surviving config by running each survivor on the global
    held-out validation window [train_end, end) and scoring val Sharpe.

    This is the final held-out evaluation — these bars were never touched
    during any training or pruning stage.

    Input: [(cfg, [s_w1, s_w2, s_w3]), ...]
    Returns: (best_cfg, best_group_list)
    """
    best_score = -np.inf
    best_item  = None

    for cfg, group in sessions:
        # Use the w=1 session as the representative for the global val run
        # (it has trained the furthest, so it's the best representative)
        rep_session = group[0]

        # Run a fresh eval session on [val_start, n_bars)
        val_bars = n_bars - val_start
        if val_bars < 10:
            continue

        eval_session = StrategySession(
            strategy   = copy.deepcopy(rep_session.strategy),
            executor   = copy.deepcopy(rep_session.executor),
            pnl_engine = PnLEngine(
                initial_capital = rep_session.pnl_engine.initial_capital,
                slippage        = rep_session.pnl_engine.slippage,
            ),
            params     = copy.deepcopy(rep_session.params),
            bar_idx    = val_start,
            results    = None,
        )
        eval_session.run(
            n_bars  = val_bars,
            prices  = prices,
            returns = returns,
            capital = capital,
        )
        score = eval_session.get_val_sharpe(val_start=0)
        if np.isnan(score):
            score = -np.inf

        if score > best_score:
            best_score = score
            best_item  = (cfg, group)

    if best_item is None:
        # Fallback: return first survivor
        best_item = sessions[0]

    print(f"[search] Final winner global val Sharpe: {best_score:.3f}")
    return best_item


# =============================================================================
# STAGED SEARCH
# =============================================================================

def staged_search_strategy(
    search_space:   dict,
    prices,
    returns,
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
    Successive-halving with 3-window cross-validation at each pruning stage.

    Per-stage walk-forward (3 windows of size W, centred on stage_end):
        w=1: trains [0, stage_end-W),   val=[stage_end-W, stage_end)
        w=2: trains [0, stage_end-2W),  val=[stage_end-2W, stage_end-W)
        w=3: trains [0, stage_end-3W),  val=[stage_end-3W, stage_end-2W)
    Pruning uses mean val Sharpe across all 3 windows.

    Final selection uses global val window [train_end, end) — never touched
    during any training or pruning stage.
    """
    n_bars    = len(prices)
    train_end = int(n_bars * (1 - val_fraction))
    val_start = train_end

    n_windows   = 3
    # window_size must exceed the largest strategy lookback so that
    # strategy.generate() receives enough price history to form valid signals.
    # pairs: coint_window=720 + zscore_window=168 → need > 888.
    # trend: slow_span=48 + vol_window=24 → well within 1000.
    # 1000 bars = ~42 days of 1h data. With 6132 train bars, this is fine.
    window_size = 1000

    if schedule is None:
        schedule = [
            {"bars": train_end // 3,                    "keep": math.ceil(initial_models / 2)},
            {"bars": train_end // 3,                    "keep": math.ceil(initial_models / 4)},
            {"bars": train_end - 2 * (train_end // 3), "keep": 1},
        ]

    print(f"\n[search] Staged search — {initial_models} configs × {n_windows} windows")
    print(f"[search] Train bars: {train_end} | Global val bars: {n_bars - train_end}")
    print(f"[search] Window size: {window_size} | Schedule: {schedule}")

    # --- Initialise: 3 sessions per config ---
    sessions    = []
    run_records = {}

    for i in range(initial_models):
        cfg   = sample_config(base_params, search_space)
        group = []
        for w in range(1, n_windows + 1):
            strategy   = signal_builder(cfg)
            executor   = build_mvo_executor(cfg)
            pnl_engine = PnLEngine(
                initial_capital = capital,
                slippage        = float(cfg.get("slippage", 0.001)),
            )
            session           = StrategySession(strategy, executor, pnl_engine, cfg)
            session.id        = f"candidate_{i}_w{w}"
            session.window_id = w
            group.append(session)
        sessions.append((cfg, group))

    prev_stage_end = 0

    # --- Staged successive halving ---
    for stage_idx, stage in enumerate(schedule):
        bars_this_stage = stage["bars"]
        keep            = stage["keep"]
        stage_end       = min(prev_stage_end + bars_this_stage, train_end)

        print(f"\n[search] Stage {stage_idx+1}: {len(sessions)} groups → "
              f"stage_end={stage_end}")

        # Train each window session to its target endpoint
        for cfg, group in sessions:
            for session in group:
                target_end = _session_target_end(stage_end, session.window_id, window_size)
                target_end = max(0, min(target_end, train_end))
                bars       = max(0, target_end - session.bar_idx)

                if bars > 0:
                    session.run(
                        n_bars  = bars,
                        prices  = prices,
                        returns = returns,
                        capital = capital,
                    )

                # Record training metrics
                sid = session.id
                if sid not in run_records:
                    run_records[sid] = {
                        "id":        sid,
                        "config":    cfg.copy(),
                        "window_id": session.window_id,
                        "bar_metrics": [],
                        "stage_val_sharpes": [],
                    }
                run_records[sid]["bar_metrics"] = session.results["bar_metrics"]

        # Score each session on its val window and record for audit trail
        for cfg, group in sessions:
            for session in group:
                val_sharpe = _score_on_window(
                    session     = session,
                    stage_end   = stage_end,
                    window_size = window_size,
                    prices      = prices,
                    returns     = returns,
                    capital     = capital,
                )
                vs, ve = _window_bounds(stage_end, session.window_id, window_size)
                run_records[session.id]["stage_val_sharpes"].append({
                    "stage":      stage_idx + 1,
                    "stage_end":  stage_end,
                    "val_start":  vs,
                    "val_end":    ve,
                    "val_sharpe": val_sharpe,
                })

        # Prune
        if keep is not None and keep < len(sessions):
            sessions = prune(
                sessions    = sessions,
                keep        = keep,
                prices      = prices,
                returns     = returns,
                stage_end   = stage_end,
                train_end   = train_end,
                capital     = capital,
                window_size = window_size,
            )

        prev_stage_end = stage_end

    # --- Final selection on global val window [train_end, end) ---
    best_cfg, best_group = select_best(
        sessions  = sessions,
        val_start = val_start,
        prices    = prices,
        returns   = returns,
        capital   = capital,
        n_bars    = n_bars,
    )

    best_metrics = {
        "global_val_start":  val_start,
        "global_val_bars":   n_bars - val_start,
        "window_size":       window_size,
        "n_windows":         n_windows,
        "search_type":       "3_window_cross_val_walk_forward",
    }

    search_summary = {
        "search_type":      "successive_halving_3window_walk_forward",
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
    print(f"[search] Best config: {best_cfg}")

    return best_cfg
