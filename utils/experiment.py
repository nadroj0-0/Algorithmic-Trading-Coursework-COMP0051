# =============================================================================
# utils/experiment.py — COMP0051 Algorithmic Trading Coursework
# Experiment class — adapted from the legacy COMP0197 version.
#
# Preserved: class structure, __init__ signature, prepare_data(), search(),
# run() method names, preloaded flag pattern, model_dir → strategy_dir.
#
# Changed: all PyTorch removed, train() → run(), model → session,
# history → results, epochs → bars, builder → signal_builder + execution_step.
# =============================================================================

from pathlib import Path

import pandas as pd

from utils.common           import run_strategy, save_results
from utils.config_loader    import get_strategy_run_dir
from utils.strategy_session import StrategySession
from utils.execution        import build_mvo_executor
from utils.portfolio        import PnLEngine
from utils.hyperparameter   import staged_search_strategy


def get_strategy_dir(strategy_name: str, base_dir: Path = None) -> Path:
    """
    Get or create the artifact directory for a strategy.
    Analogous to legacy get_model_dir().
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    d = base_dir / "strategies" / strategy_name
    d.mkdir(parents=True, exist_ok=True)
    return d


class Experiment:
    """
    Encapsulates a single strategy experiment: data loading, optional
    parameter search, strategy execution, and results persistence.

    Mirrors the legacy Experiment class structure exactly:
        __init__      : same signature (name, base_cfg/base_params, model_dir)
        prepare_data  : same pattern (inject pre-loaded data, set preloaded=True)
        search        : same pattern (update self.params with winner)
        run           : same pattern as legacy train() — execute strategy
        preloaded     : same flag for pre-injected data

    Fields mapped from legacy:
        self.cfg    → self.params   (strategy params dict)
        self.model  → self.session  (StrategySession)
        self.history→ self.results  (PnL results dict)
        self.stats  → self.data     (pre-loaded returns dict)
    """

    def __init__(
        self,
        name:         str,
        base_params:  dict,
        strategy_dir: Path = None,
    ):
        self.name         = name
        self.params       = base_params.copy()
        self.strategy_dir = strategy_dir or get_strategy_dir(name)
        self.session      = None    # StrategySession (≈ self.model)
        self.results      = None    # results dict (≈ self.history)
        self.data         = None    # {prices, returns} (≈ self.stats)
        self.preloaded    = False   # same flag as legacy

    def prepare_data(self, data: dict) -> None:
        """
        Store pre-loaded data dict and set preloaded=True.
        Analogous to legacy prepare_data(data_fn, **kwargs).

        Args:
            data : dict with keys "prices" and "returns" (both DataFrames).
        """
        self.data      = data
        self.preloaded = True

    def search(
        self,
        search_space:   dict,
        signal_builder,
        execution_step,
        schedule:       list = None,
        initial_models: int  = 15,
        val_fraction:   float = 0.30,
    ) -> None:
        """
        Run staged parameter search and update self.params with winner.
        Analogous to legacy Experiment.search().

        Calls staged_search_strategy() from hyperparameter.py — same pattern
        as the legacy search() calling staged_search().

        Args:
            search_space    : Dict of param → (low, high, mode) tuples.
            signal_builder  : Signal builder callable from registry.
            execution_step  : Execution step callable from registry.
            schedule        : Successive halving schedule.
            initial_models  : Number of initial random configs.
            val_fraction    : Fraction held out for validation.
        """
        if not self.preloaded or self.data is None:
            raise RuntimeError(
                "Data not loaded. Call prepare_data() or set preloaded=True "
                "before calling search()."
            )

        prices  = self.data["prices"]
        returns = self.data["returns"]
        capital = float(self.params.get("initial_capital", 10_000))

        print(f"\n[experiment] Starting search: {self.name}")

        best_params = staged_search_strategy(
            search_space    = search_space,
            prices          = prices,
            returns         = returns,
            signal_builder  = signal_builder,
            execution_step  = execution_step,
            strategy_dir    = self.strategy_dir,
            base_params     = self.params,
            schedule        = schedule,
            initial_models  = initial_models,
            val_fraction    = val_fraction,
            capital         = capital,
            search_name     = f"{self.name}_search",
        )

        if best_params is not None:
            print(f"\n[experiment] Best params for {self.name}:")
            for k, v in best_params.items():
                if not isinstance(v, dict):
                    print(f"  {k}: {v}")
            self.params = best_params.copy()

    def run(
        self,
        signal_builder,
        execution_step,
    ) -> None:
        """
        Run the strategy with self.params. Stores session and results.
        Analogous to legacy Experiment.train(builder, training_step).

        Constructs a fresh StrategySession from self.params, runs it over
        the full data, saves results to disk.
        """
        if not self.preloaded or self.data is None:
            raise RuntimeError(
                "Data not loaded. Call prepare_data() before run()."
            )

        prices  = self.data["prices"]
        returns = self.data["returns"]
        capital = float(self.params.get("initial_capital", 10_000))

        print(f"\n[experiment] Running strategy: {self.name}")
        print(f"  Params: {self.params}")

        # Build strategy components from params
        strategy   = signal_builder(self.params)
        executor   = build_mvo_executor(self.params)
        pnl_engine = PnLEngine(
            initial_capital = capital,
            slippage        = float(self.params.get("slippage", 0.001)),
        )

        self.session = StrategySession(
            strategy   = strategy,
            executor   = executor,
            pnl_engine = pnl_engine,
            params     = self.params,
        )

        # Run the full strategy
        self.results = run_strategy(
            session = self.session,
            prices  = prices,
            returns = returns,
            capital = capital,
        )

        # Save results to disk
        save_results(
            results      = self.results,
            name         = self.name,
            stage        = "full",
            strategy_dir = self.strategy_dir,
        )

        # print(f"\n  [DONE] {self.name} — artifacts in {self.strategy_dir}")
        # if "pnl_net" in self.results and self.results["pnl_net"] is not None:
        #     total_net = self.results.get("total_pnl_net", 0)
        #     pct       = self.results.get("pct_return_net", 0)
        #     print(f"  Total PnL (net): ${total_net:,.2f} ({pct * 100:.2f}%)")

        print(f"\n  [DONE] {self.name} — artifacts in {self.strategy_dir}")

        # handle both structures safely
        inner = self.results.get("results") if "results" in self.results else self.results

        bar_metrics = inner.get("bar_metrics", [])
        if bar_metrics:
            total_net = float(bar_metrics[-1].get("total_pnl_net", 0.0))
            pct = total_net / capital
            print(f"  Total PnL (net): ${total_net:,.2f} ({pct * 100:.2f}%)")
        else:
            print("  [WARN] No bar_metrics found in results")
