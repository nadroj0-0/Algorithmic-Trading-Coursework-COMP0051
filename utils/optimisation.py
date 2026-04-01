# =============================================================================
# utils/optimisation.py — COMP0051 Algorithmic Trading Coursework
# Centralised MVO solver construction — adapted from legacy OptimisationConfig.
#
# SolverConfig is the direct analogue of OptimisationConfig:
#   configure_optimiser(model, cfg) →  configure_solver(cfg)
#   configure_training_kwargs(opt,cfg) → configure_execution_kwargs(solver,cfg)
#
# Now implements FULL Mean-Variance Optimisation:
#   minimise  θ^T Σ θ  −  λ × μ^T θ
#   subject to signal direction constraints + gross exposure cap
#
# μ is the expected-return vector derived from rolling mean excess returns,
# scaled to the signal direction. λ (risk_aversion) controls the return/risk
# trade-off. Setting λ=0 recovers the previous minimum-variance-only form.
# =============================================================================

import inspect
import numpy as np


# =============================================================================
# SOLVER IMPLEMENTATIONS
# =============================================================================

def solve_qp_cvxopt(
    Sigma:         np.ndarray,
    signals:       np.ndarray,
    gross_cap:     float,
    mu:            np.ndarray = None,
    risk_aversion: float      = 1.0,
    reg:           float      = 1e-4,
) -> np.ndarray:
    """
    Full Mean-Variance QP via cvxopt (same library as Notebook 14).

    Solves:
        minimise    θ^T Σ θ  −  λ × μ^T θ
        subject to  Σ_i |θ_i| ≤ gross_cap    (brief's gross exposure cap)
                    θ_i ≥ 0  if signal_i > 0  (long if signal positive)
                    θ_i ≤ 0  if signal_i < 0  (short if signal negative)
                    θ_i = 0  if signal_i = 0  (flat — no position)

    When mu=None or risk_aversion=0, reduces to minimum-variance (λ=0).

    The quadratic term Σ controls portfolio variance (risk).
    The linear term −λμ incorporates expected returns into the objective,
    making this true Markowitz MVO rather than min-variance only.

    Args:
        Sigma         : (n, n) covariance matrix of excess returns.
        signals       : (n,) directional signals in {-1, 0, +1}.
        gross_cap     : Max Σ|θ_i| in USDT (brief's $100,000 constraint).
        mu            : (n,) expected return vector. None → set to zero.
        risk_aversion : λ — scales the return term. Higher = more return-seeking.
        reg           : Ridge regularisation on Sigma diagonal.

    Returns:
        (n,) array of USDT positions θ_i.
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

    n      = len(signals)
    active = np.where(signals != 0)[0]
    n_act  = len(active)

    if n_act == 0:
        return np.zeros(n)

    # Regularised covariance for active assets
    S_act = Sigma[np.ix_(active, active)] + reg * np.eye(n_act)

    # Expected return vector for active assets
    # If mu not provided, default to zeros (recovers min-variance)
    if mu is not None:
        mu_act = mu[active]
    else:
        mu_act = np.zeros(n_act)

    # QP objective: minimise θ^T S θ - λ μ^T θ
    # cvxopt form: minimise (1/2) x^T P x + q^T x
    # → P = 2*S_act,  q = -λ*mu_act
    P = matrix(2.0 * S_act.astype(float))
    q = matrix((-risk_aversion * mu_act).astype(float))

    # Direction constraints: -signal_i * θ_i ≤ 0  (enforce sign)
    G_dir = -np.diag(signals[active].astype(float))
    h_dir = np.zeros(n_act)

    # Per-asset upper bound: signal_i * θ_i ≤ gross_cap
    G_cap = np.eye(n_act) * signals[active].astype(float)
    h_cap = np.full(n_act, gross_cap)

    # Gross exposure: sum of signed positions ≤ gross_cap
    G_gross = signals[active].astype(float).reshape(1, -1)
    h_gross = np.array([gross_cap])

    G_all = matrix(np.vstack([G_dir, G_cap, G_gross]))
    h_all = matrix(np.concatenate([h_dir, h_cap, h_gross]))

    try:
        sol = cvxopt.solvers.qp(P, q, G_all, h_all)
        if sol["status"] != "optimal":
            return _equal_weight_fallback(signals, gross_cap)
        theta_act = np.array(sol["x"]).flatten()
    except Exception:
        return _equal_weight_fallback(signals, gross_cap)

    theta = np.zeros(n)
    theta[active] = theta_act
    return theta


def solve_qp_scipy(
    Sigma:         np.ndarray,
    signals:       np.ndarray,
    gross_cap:     float,
    mu:            np.ndarray = None,
    risk_aversion: float      = 1.0,
    reg:           float      = 1e-4,
) -> np.ndarray:
    """
    Full MVO QP via scipy.optimize.minimize (SLSQP). Fallback solver.

    Same objective as solve_qp_cvxopt:
        minimise  θ^T Σ θ  −  λ × μ^T θ
    """
    from scipy.optimize import minimize

    n      = len(signals)
    active = np.where(signals != 0)[0]
    n_act  = len(active)

    if n_act == 0:
        return np.zeros(n)

    S_act  = Sigma[np.ix_(active, active)] + reg * np.eye(n_act)
    mu_act = mu[active] if mu is not None else np.zeros(n_act)
    dirs   = signals[active].astype(float)

    def objective(x):
        return float(x @ S_act @ x) - risk_aversion * float(mu_act @ x)

    def jac(x):
        return 2 * S_act @ x - risk_aversion * mu_act

    bounds = [(0, gross_cap) if d > 0 else (-gross_cap, 0) for d in dirs]
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
    """Equal-weight fallback when QP solver fails."""
    n      = len(signals)
    active = np.where(signals != 0)[0]
    theta  = np.zeros(n)
    if len(active) == 0:
        return theta
    alloc         = gross_cap / len(active)
    theta[active] = signals[active] * alloc
    return theta


def solve_equal_weight(
    Sigma:         np.ndarray,
    signals:       np.ndarray,
    gross_cap:     float,
    mu:            np.ndarray = None,
    risk_aversion: float      = 1.0,
    reg:           float      = 1e-4,
) -> np.ndarray:
    """Equal-weight allocation (no QP). Baseline comparison."""
    return _equal_weight_fallback(signals, gross_cap)


# =============================================================================
# SOLVER CONFIG — analogue of OptimisationConfig
# =============================================================================

class SolverConfig:
    """
    Centralised construction of MVO solver and execution kwargs.
    Direct analogue of OptimisationConfig — same static-method pattern,
    same _filter_valid_kwargs, same registry → callable resolution.

    Now supports full MVO via the mu and risk_aversion parameters.
    """

    SOLVERS = {
        "cvxopt_qp":    solve_qp_cvxopt,
        "scipy_slsqp":  solve_qp_scipy,
        "equal_weight": solve_equal_weight,
    }

    @staticmethod
    def _filter_valid_kwargs(cls, params: dict) -> dict:
        """
        Validate params against callable signature.
        Kept verbatim from legacy OptimisationConfig — domain-agnostic.
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

    @staticmethod
    def configure_solver(cfg: dict):
        """
        Read solver name from cfg and return the solver callable.

        YAML field: solver: cvxopt_qp   (default)
        Available:  cvxopt_qp | scipy_slsqp | equal_weight
        """
        name = cfg.get("solver", "cvxopt_qp").lower()
        if name not in SolverConfig.SOLVERS:
            raise ValueError(
                f"Unknown solver: '{name}'. "
                f"Available: {sorted(SolverConfig.SOLVERS.keys())}"
            )
        return SolverConfig.SOLVERS[name]

    @staticmethod
    def configure_execution_kwargs(solver, cfg: dict) -> dict:
        """
        Bundle all execution parameters into one kwargs dict.
        Now includes risk_aversion (λ) for full MVO.

        Analogous to configure_training_kwargs() in legacy OptimisationConfig.
        """
        kwargs = {
            "solver":        solver,
            "gross_cap":     float(cfg.get("gross_cap",     100_000)),
            "vol_target":    float(cfg.get("vol_target",    0.02)),
            "mvo_lookback":  int(cfg.get("mvo_lookback",    168)),
            "reg":           float(cfg.get("regularisation", 1e-4)),
            "risk_aversion": float(cfg.get("risk_aversion", 1.0)),
            "mu_lookback":   int(cfg.get("mu_lookback",     168)),
        }
        if "slippage" in cfg:
            kwargs["slippage"] = float(cfg["slippage"])
        return kwargs
