# =============================================================================
# utils/config_loader.py — COMP0051 Algorithmic Trading Coursework
# YAML config management for the experiment pipeline.
#
# Lightly adapted from the legacy COMP0197 version:
#   - All logic, patterns, and function signatures preserved
#   - model_* renamed to strategy_* throughout
#   - Directory skeleton: runs/{run}/models/ → runs/{run}/strategies/
#   - _BUILDER_MODULE → _SIGNAL_MODULE (utils.strategies)
#   - _STEP_MODULE    → _EXECUTION_MODULE (utils.execution)
#   - resolve_registry_entry returns signal_builder + execution_step keys
#
# Responsibilities (unchanged from legacy):
#   - Load experiment.yml, registry.yml, and per-strategy yml files
#   - Resolve string builder/step names to actual Python callables
#   - Create and manage the runs/ directory structure
#   - Snapshot configs into run dirs at experiment start
#   - Read and write best_config back into strategy yml files after search
# =============================================================================

import importlib
import shutil
import yaml
from pathlib import Path


# =============================================================================
# LOADING
# =============================================================================

def load_experiment(path: str | Path) -> dict:
    """
    Load experiment.yml.
    Returns the full experiment config dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Experiment config is empty: {path}")
    return cfg


def load_strategy_config(path: str | Path) -> dict:
    """
    Load a single strategy yml file.
    Returns dict with keys: params, search_space, best_config, strategy metadata.
    Renamed from load_model_config — same logic.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Strategy config is empty: {path}")
    return cfg


# Keep legacy name as alias so any old imports still work
load_model_config = load_strategy_config


def load_registry(path: str | Path) -> dict:
    """
    Load registry.yml.
    Returns raw dict — strings not yet resolved to callables.
    Use resolve_registry_entry() to get actual Python objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with open(path) as f:
        reg = yaml.safe_load(f)
    if reg is None:
        raise ValueError(f"Registry is empty: {path}")
    return reg


# =============================================================================
# CALLABLE RESOLUTION
# =============================================================================

# Maps string names to the modules where they live.
# Add new signal builders to utils/strategies.py and
# new execution steps to utils/execution.py.
_SIGNAL_MODULE    = "utils.strategies"
_EXECUTION_MODULE = "utils.execution"

_SIGNAL_REGISTRY    = None   # lazy-loaded cache
_EXECUTION_REGISTRY = None   # lazy-loaded cache


def _get_signal_builders() -> dict:
    """Lazy-load all build_* functions from utils.strategies."""
    global _SIGNAL_REGISTRY
    if _SIGNAL_REGISTRY is None:
        mod = importlib.import_module(_SIGNAL_MODULE)
        _SIGNAL_REGISTRY = {
            name: getattr(mod, name)
            for name in dir(mod)
            if name.startswith("build_") and callable(getattr(mod, name))
        }
    return _SIGNAL_REGISTRY


def _get_execution_steps() -> dict:
    """Lazy-load all execution step callables from utils.execution."""
    global _EXECUTION_REGISTRY
    if _EXECUTION_REGISTRY is None:
        mod = importlib.import_module(_EXECUTION_MODULE)
        _EXECUTION_REGISTRY = {
            name: getattr(mod, name)
            for name in dir(mod)
            if callable(getattr(mod, name)) and not name.startswith("_")
        }
    return _EXECUTION_REGISTRY


def resolve_registry_entry(entry: dict) -> dict:
    """
    Convert a raw registry entry (strings) into resolved Python callables.

    Input entry (from registry.yml):
        signal_builder: "build_pairs_signal"
        execution_step: "mvo_execution"

    Returns:
        {
            "signal_builder": <function build_pairs_signal>,
            "execution_step": <function mvo_execution>,
        }

    Analogous to the legacy resolve_registry_entry — same validation pattern,
    same lazy-loading, just different module sources and return keys.
    """
    builders = _get_signal_builders()
    steps    = _get_execution_steps()

    builder_name = entry["signal_builder"]
    step_name    = entry["execution_step"]

    if builder_name not in builders:
        raise ValueError(
            f"Signal builder '{builder_name}' not found in {_SIGNAL_MODULE}. "
            f"Available: {sorted(builders.keys())}"
        )
    if step_name not in steps:
        raise ValueError(
            f"Execution step '{step_name}' not found in {_EXECUTION_MODULE}. "
            f"Available: {sorted(steps.keys())}"
        )

    return {
        "signal_builder": builders[builder_name],
        "execution_step": steps[step_name],
    }


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def get_run_dir(base_dir: str | Path, run_name: str) -> Path:
    """Return the path to a run directory without creating it."""
    return Path(base_dir) / "runs" / run_name


def create_run_dir(base_dir: str | Path, run_name: str) -> Path:
    """
    Create the full run directory structure:

        runs/{run_name}/
        ├── configs/
        │   └── strategies/
        └── strategies/

    Returns the run root Path.
    Renamed from legacy create_run_dir — same logic, strategies/ not models/.
    """
    run_dir = get_run_dir(base_dir, run_name)
    (run_dir / "configs" / "strategies").mkdir(parents=True, exist_ok=True)
    (run_dir / "strategies").mkdir(parents=True, exist_ok=True)
    print(f"[config] Run directory: {run_dir}")
    return run_dir


def get_strategy_run_dir(run_dir: str | Path, strategy_name: str) -> Path:
    """
    Return the artifact directory for a specific strategy within a run.
    Creates it if it doesn't exist.
    Renamed from get_model_run_dir — same logic.
    """
    d = Path(run_dir) / "strategies" / strategy_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# Keep legacy alias so any direct imports still work
get_model_run_dir = get_strategy_run_dir


# =============================================================================
# CONFIG SNAPSHOTTING
# =============================================================================

def snapshot_configs(
    run_dir:          str | Path,
    experiment_yml:   str | Path,
    strategy_names:   list[str],
    strategies_cfg_dir: str | Path = "configs/strategies",
) -> None:
    """
    At the start of a run, copy all relevant config files into the run dir.
    Creates an immutable record of exactly what config was used.

        runs/{run_name}/configs/experiment.yml
        runs/{run_name}/configs/strategies/{strategy_name}.yml

    Analogous to legacy snapshot_configs — same logic, model → strategy rename.

    Args:
        run_dir            : Root of this run.
        experiment_yml     : Path to the master experiment.yml.
        strategy_names     : List of strategy names in the experiment.
        strategies_cfg_dir : Directory containing the master strategy ymls.
    """
    run_dir            = Path(run_dir)
    experiment_yml     = Path(experiment_yml)
    strategies_cfg_dir = Path(strategies_cfg_dir)

    # Copy experiment.yml
    dst_exp = run_dir / "configs" / "experiment.yml"
    shutil.copy2(experiment_yml, dst_exp)
    print(f"[config] Snapshotted experiment config → {dst_exp}")

    # Copy each strategy yml
    for strategy_name in strategy_names:
        src = strategies_cfg_dir / f"{strategy_name}.yml"
        dst = run_dir / "configs" / "strategies" / f"{strategy_name}.yml"
        if not src.exists():
            raise FileNotFoundError(
                f"Strategy config not found for '{strategy_name}': {src}"
            )
        shutil.copy2(src, dst)
        print(f"[config] Snapshotted strategy config → {dst}")


# =============================================================================
# SEARCH RESULT PERSISTENCE
# =============================================================================

def write_best_config(
    run_strategy_yml_path: str | Path,
    best_config:           dict,
) -> None:
    """
    After hyperparameter search, write the winning config back into the
    run copy of the strategy yml file.

    Updates two sections:
      - best_config : raw search winner (for reference / audit trail)
      - params      : merged with best_config so run.py reads a single
                      source of truth

    Analogous to legacy write_best_config — same logic, train_config → params.

    Args:
        run_strategy_yml_path : Path to strategy yml inside the run dir.
        best_config           : Dict of winning params from staged_search_strategy.
    """
    path = Path(run_strategy_yml_path)
    if not path.exists():
        raise FileNotFoundError(f"Run strategy config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Store raw best config for inspection
    cfg["best_config"] = best_config

    # Merge into params so run.py reads one consistent source
    if "params" not in cfg or cfg["params"] is None:
        cfg["params"] = {}
    cfg["params"].update(best_config)

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[config] Best config written → {path}")


def load_best_config(run_strategy_yml_path: str | Path) -> dict | None:
    """
    Read the best_config section from a run strategy yml.
    Returns None if no best_config has been written yet.
    """
    path = Path(run_strategy_yml_path)
    if not path.exists():
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("best_config", None)


def load_strategy_params(run_strategy_yml_path: str | Path) -> dict:
    """
    Load the params section from a run strategy yml.
    If search has run, best_config values are already merged in.
    Analogous to load_train_config().
    """
    path = Path(run_strategy_yml_path)
    cfg  = load_strategy_config(path)
    params = cfg.get("params")
    if params is None:
        raise ValueError(f"No 'params' section in {path}")
    return params


# Keep legacy alias
load_train_config = load_strategy_params


# =============================================================================
# SEARCH SPACE PARSING
# =============================================================================

def load_search_space(strategy_yml_path: str | Path) -> dict | None:
    """
    Load the search_space section from a strategy yml.
    Converts list format [low, high, mode] to tuple format (low, high, mode).
    Returns None if no search_space section exists.

    YAML format:
        search_space:
          entry_z:       [1.0, 2.5, uniform]
          zscore_window: [72, 336, uniform]

    Returns:
        {
            "entry_z":       (1.0, 2.5, "uniform"),
            "zscore_window": (72.0, 336.0, "uniform"),
        }

    Kept verbatim from legacy — domain agnostic.
    """
    cfg = load_strategy_config(strategy_yml_path)
    raw = cfg.get("search_space")
    if raw is None:
        return None
    return {
        key: (float(vals[0]), float(vals[1]), str(vals[2]))
        for key, vals in raw.items()
    }
