# =============================================================================
# utils/optimisation.py — COMP0051 Algorithmic Trading Coursework
# Centralised MVO solver construction — adapted from the legacy
# OptimisationConfig pattern (COMP0197 Applied Deep Learning).
#
# Design goals (identical to legacy):
#   - Fully config-driven (no hidden defaults in builders)
#   - Safe (invalid params caught early via _filter_valid_kwargs)
#   - Extensible (add new solvers via the SOLVERS registry)
#   - Registry pattern: YAML string → callable, resolved at runtime
#
# The class SolverConfig is the direct analogue of OptimisationConfig:
#   configure_optimiser(model, cfg)    →  configure_solver(cfg)
#   configure_training_kwargs(opt,cfg) →  configure_execution_kwargs(solver,cfg)
#   OPTIMISERS registry                →  SOLVERS registry
#   SCHEDULERS registry                →  CONSTRAINTS registry
#
# Callers (execution.py builders) use exactly two lines:
#   solver = SolverConfig.configure_solver(cfg)
#   kwargs = SolverConfig.configure_execution_kwargs(solver, cfg)
# =============================================================================

import inspect
import numpy as np


# =============================================================================
# SOLVER IMPLEMENTATIONS
# =============================================================================

def solve_qp_cvxopt(
    Sigma:     np.ndarray,
    signals:   np.ndarray,
    gross_cap: float,
    reg:       float = 1e-4,
) -> np.ndarray:
    """
    Minimum-variance QP via cvxopt (exact same library as Notebook 14).

    Solves:
        minimise    θ^T Σ θ
        subject to  Σ u_i ≤ gross_cap          (gross exposure cap)
                    θ_i ≥ 0  if signal_i > 0   (long if signal positive)
                    θ_i ≤ 0  if signal_i < 0   (short if signal negative)
                    θ_i = 0  if signal_i = 0    (flat if no signal)

    The |θ_i| ≤ u_i, Σ u_i ≤ gross_cap reformulation linearises the
    absolute value constraint so cvxopt's QP can handle it.

    Args:
        Sigma     : (n, n) covariance matrix of excess returns.
        signals   : (n,) array of directional signals, values in {-1, 0, +1}.
        gross_cap : Maximum Σ|θ_i| in USDT (brief's $100,000 constraint).
        reg       : Ridge regularisation added to diagonal of Sigma for
                    numerical stability (default 1e-4).

    Returns:
        (n,) array of USDT positions θ_i. Zero for assets with signal=0.
    """
    try:
        import cvxopt
        from cvxopt import matrix, solvers
        solvers.options["show_progress"] = False
    except ImportError:
        raise ImportError(
            "cvxopt not installed. Run: pip install cvxopt\n"
            "Or switch to solver: scipy_slsqp in strategy YAML."
        )

    n       = len(signals)
    active  = np.where(signals != 0)[0]         # indices with non-zero signal
    n_act   = len(active)

    if n_act == 0:
        return np.zeros(n)

    # --- Regularise Sigma for active assets only ---
    S_act = Sigma[np.ix_(active, active)] + reg * np.eye(n_act)

    # --- QP: minimise θ^T S θ ---
    # Variables: θ_i for active assets (n_act variables)
    P = matrix(S_act.astype(float))
    q = matrix(np.zeros(n_act))

    # --- Inequality constraints: −signal_i × θ_i ≤ 0 (enforce direction) ---
    # and θ_i ≤ gross_cap (per-asset cap as upper bound safety net)
    G_dir  = -np.diag(signals[active].astype(float))   # direction constraints
    h_dir  = np.zeros(n_act)
    G_cap  = np.eye(n_act) * signals[active].astype(float)  # |θ_i| bound
    h_cap  = np.full(n_act, gross_cap)

    G = matrix(np.vstack([G_dir, G_cap]))
    h = matrix(np.concatenate([h_dir, h_cap]))

    # --- Equality: Σ|θ_i| ≤ gross_cap via auxiliary --- (handled via h_cap sum)
    # sum of all allocations ≤ gross_cap (linear approximation using |signal|=1)
    G_gross = matrix(signals[active].astype(float), (1, n_act))
    h_gross = matrix([gross_cap])

    G_all = matrix(np.vstack([
        np.array(G),
        signals[active].astype(float).reshape(1, -1),
    ]))
    h_all = matrix(np.append(np.array(h).flatten(), gross_cap))

    try:
        sol = cvxopt.solvers.qp(P, q, G_all, h_all)
        if sol["status"] != "optimal":
            # Fall back to equal-weight allocation scaled to gross_cap
            return _equal_weight_fallback(signals, gross_cap)
        theta_act = np.array(sol["x"]).flatten()
    except Exception:
        return _equal_weight_fallback(signals, gross_cap)

    # Map active positions back to full asset vector
    theta = np.zeros(n)
    theta[active] = theta_act
    return theta


def solve_qp_scipy(
    Sigma:     np.ndarray,
    signals:   np.ndarray,
    gross_cap: float,
    reg:       float = 1e-4,
) -> np.ndarray:
    """
    Minimum-variance QP via scipy.optimize.minimize (SLSQP).
    Fallback when cvxopt is unavailable.

    Args:
        Sigma     : (n, n) covariance matrix.
        signals   : (n,) directional signals in {-1, 0, +1}.
        gross_cap : Maximum Σ|θ_i| in USDT.
        reg       : Ridge regularisation on Sigma diagonal.

    Returns:
        (n,) array of USDT positions.
    """
    from scipy.optimize import minimize

    n      = len(signals)
    active = np.where(signals != 0)[0]
    n_act  = len(active)

    if n_act == 0:
        return np.zeros(n)

    S_act = Sigma[np.ix_(active, active)] + reg * np.eye(n_act)
    dirs  = signals[active].astype(float)

    def objective(x):
        return float(x @ S_act @ x)

    def jac(x):
        return 2 * S_act @ x

    # Bounds: direction enforces sign; cap is magnitude bound
    bounds = [(0, gross_cap) if d > 0 else (-gross_cap, 0) for d in dirs]

    # Gross exposure constraint: Σ|θ_i| ≤ gross_cap
    constraints = [{"type": "ineq", "fun": lambda x: gross_cap - np.abs(x).sum()}]

    x0  = np.array([gross_cap / n_act * d for d in dirs])
    res = minimize(objective, x0, jac=jac, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 500})

    theta = np.zeros(n)
    if res.success:
        theta[active] = res.x
    else:
        return _equal_weight_fallback(signals, gross_cap)

    return theta


def _equal_weight_fallback(signals: np.ndarray, gross_cap: float) -> np.ndarray:
    """
    Equal-weight fallback: allocate gross_cap equally among active signals.
    Called when QP solver fails or no active signals exist.

    Args:
        signals   : (n,) directional signals.
        gross_cap : Maximum total exposure.

    Returns:
        (n,) array of equal-weight positions respecting direction.
    """
    n      = len(signals)
    active = np.where(signals != 0)[0]
    n_act  = len(active)
    theta  = np.zeros(n)
    if n_act == 0:
        return theta
    alloc         = gross_cap / n_act
    theta[active] = signals[active] * alloc
    return theta


def solve_equal_weight(
    Sigma:     np.ndarray,
    signals:   np.ndarray,
    gross_cap: float,
    reg:       float = 1e-4,
) -> np.ndarray:
    """
    Equal-weight allocation (no QP). Useful for comparison baseline.

    Args:
        Sigma     : Not used (kept for consistent function signature).
        signals   : (n,) directional signals.
        gross_cap : Maximum total exposure in USDT.
        reg       : Not used.

    Returns:
        (n,) array of equal-weight USDT positions.
    """
    return _equal_weight_fallback(signals, gross_cap)


# =============================================================================
# SOLVER CONFIG CLASS — direct analogue of OptimisationConfig
# =============================================================================

class SolverConfig:
    """
    Centralised construction of MVO solver and execution kwargs.

    Direct analogue of OptimisationConfig from the legacy neural-net
    codebase — same architectural pattern, same static-method class,
    same _filter_valid_kwargs, same registry → callable resolution.

    Usage (in execution.py builders, analogous to network.py builders):
        solver = SolverConfig.configure_solver(cfg)
        kwargs = SolverConfig.configure_execution_kwargs(solver, cfg)
        return MVOExecutor(solver=solver, **kwargs)
    """

    # -----------------------------------------------------------------------
    # Solver registry — analogous to OPTIMISERS in OptimisationConfig
    # YAML key → solver callable (must match signature above)
    # -----------------------------------------------------------------------
    SOLVERS = {
        "cvxopt_qp":    solve_qp_cvxopt,
        "scipy_slsqp":  solve_qp_scipy,
        "equal_weight": solve_equal_weight,
    }

    # -----------------------------------------------------------------------
    # Utility: validate kwargs against callable signature (kept verbatim)
    # -----------------------------------------------------------------------
    @staticmethod
    def _filter_valid_kwargs(cls, params: dict) -> dict:
        """
        Validate that all keys in params are valid for the given callable.
        Raises ValueError with the invalid keys and valid alternatives.
        Kept verbatim from legacy OptimisationConfig — domain agnostic.
        """
        sig        = inspect.signature(cls)
        valid_keys = set(sig.parameters.keys())
        filtered   = {k: v for k, v in params.items() if k in valid_keys}
        invalid    = set(params.keys()) - valid_keys

        if invalid:
            raise ValueError(
                f"Invalid parameters for {getattr(cls, '__name__', str(cls))}: {invalid}\n"
                f"Valid parameters: {valid_keys}"
            )

        return filtered

    # -----------------------------------------------------------------------
    # Solver constructor — analogous to configure_optimiser()
    # -----------------------------------------------------------------------
    @staticmethod
    def configure_solver(cfg: dict):
        """
        Read solver name from cfg and return the solver callable.

        YAML field: solver: cvxopt_qp   (default)
        Available:  cvxopt_qp | scipy_slsqp | equal_weight

        Args:
            cfg : Strategy params dict (from strategy YAML params section).

        Returns:
            Solver callable with signature (Sigma, signals, gross_cap, reg) → theta.
        """
        name = cfg.get("solver", "cvxopt_qp").lower()

        if name not in SolverConfig.SOLVERS:
            raise ValueError(
                f"Unknown solver: '{name}'. "
                f"Available: {sorted(SolverConfig.SOLVERS.keys())}"
            )

        return SolverConfig.SOLVERS[name]

    # -----------------------------------------------------------------------
    # Execution kwargs — analogous to configure_training_kwargs()
    # -----------------------------------------------------------------------
    @staticmethod
    def configure_execution_kwargs(solver, cfg: dict) -> dict:
        """
        Bundle all execution parameters into one kwargs dict.
        Passed directly to MVOExecutor constructor and mvo_execution().

        Analogous to configure_training_kwargs() — bundles scheduler +
        clip_grad_norm into kwargs for the training loop.

        Args:
            solver : Resolved solver callable (from configure_solver).
            cfg    : Strategy params dict.

        Returns:
            dict with keys: gross_cap, vol_target, mvo_lookback, reg,
                            slippage (if present in cfg)
        """
        kwargs = {
            "solver":       solver,
            "gross_cap":    float(cfg.get("gross_cap",    100_000)),
            "vol_target":   float(cfg.get("vol_target",   0.02)),
            "mvo_lookback": int(cfg.get("mvo_lookback",   168)),
            "reg":          float(cfg.get("regularisation", 1e-4)),
        }

        # Backwards compatibility: slippage may be pre-computed and injected
        # by run.py, or it can be specified directly in the strategy YAML.
        if "slippage" in cfg:
            kwargs["slippage"] = float(cfg["slippage"])

        return kwargs
