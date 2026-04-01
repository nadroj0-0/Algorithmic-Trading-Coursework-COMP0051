# =============================================================================
# utils/strategies.py — COMP0051 Algorithmic Trading Coursework
# Signal generation for both trading strategies.
#
# Strategy 1: PairsStrategy — Rolling cointegration pairs (BTC/ETH)
#   Implements the full Engle-Granger two-step procedure from Notebook 3.5:
#     Step 1: rolling Engle-Granger test (statsmodels.tsa.stattools.coint)
#     Step 2: OLS hedge ratio on windows that pass the cointegration test
#     Step 3: z-score of the cointegration residual (spread)
#     Step 4: signal rules with entry/exit/stop thresholds
#
# Strategy 2: TrendStrategy — EWM momentum (BTC, ETH, SOL)
#   Modelled on Notebook 3.3 (A Trend Following Strategy).
# =============================================================================

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


# =============================================================================
# STRATEGY 1 — Rolling Cointegration Pairs (BTC/ETH)
# =============================================================================

class PairsStrategy:
    """
    Rolling Engle-Granger cointegration pairs strategy on BTC/USDT and ETH/USDT.

    Implements the full two-step Engle-Granger procedure from Notebook 3.5,
    applied as a rolling test over coint_window bars:

        Step 1 — Cointegration test (Notebook 3.5 pattern):
            Uses statsmodels.tsa.stattools.coint(log_BTC, log_ETH) over a
            rolling window of coint_window bars.
            Signal is suppressed (flat) when p-value ≥ coint_pvalue_threshold.
            This is the correct implementation of Engle-Granger — we only trade
            the spread when statistical evidence of cointegration exists.

        Step 2 — OLS hedge ratio:
            β = OLS(log_BTC ~ log_ETH) over the same window.
            Spread = log_BTC − β × log_ETH (the cointegration residual).

        Step 3 — Z-score of spread:
            Z = (spread − rolling_mean) / rolling_std  over zscore_window bars.

        Step 4 — Signal rules (sticky state):
            |z| > entry_z  and cointegrated → open position
            |z| < exit_z               → close position
            |z| > stop_z               → stop loss, force close

    Args:
        params : Strategy params dict from YAML. Keys:
            coint_window          : Rolling window for E-G test (default 720 bars)
            coint_pvalue_threshold: Max p-value to trade (default 0.05)
            hedge_window          : OLS regression window for β (default 720)
            zscore_window         : Rolling z-score window (default 168)
            entry_z               : |z| threshold to open (default 1.5)
            exit_z                : |z| threshold to close (default 0.5)
            stop_z                : |z| threshold for stop loss (default 3.0)
    """

    def __init__(self, params: dict):
        self.coint_window           = int(params.get("coint_window",           720))
        self.coint_pvalue_threshold = float(params.get("coint_pvalue_threshold", 0.05))
        self.hedge_window           = int(params.get("hedge_window",           720))
        self.zscore_window          = int(params.get("zscore_window",          168))
        self.entry_z                = float(params.get("entry_z",              1.5))
        self.exit_z                 = float(params.get("exit_z",               0.5))
        self.stop_z                 = float(params.get("stop_z",               3.0))

        self.base_asset  = "BTC/USDT"
        self.hedge_asset = "ETH/USDT"

    def _rolling_coint_pvalue(
        self,
        log_base:  pd.Series,
        log_hedge: pd.Series,
    ) -> pd.Series:
        """
        Compute rolling Engle-Granger p-value using statsmodels.coint.

        For each bar t, runs coint(log_base[t-W:t], log_hedge[t-W:t]) where
        W = coint_window. Returns a Series of p-values, NaN for early bars.

        A p-value below coint_pvalue_threshold (default 0.05) means we reject
        the null of no cointegration — it is safe to trade the spread.

        This is the same test as Notebook 3.5:
            score, pvalue, _ = coint(A, B)
            if pvalue < confidence_level: ...
        """
        n       = len(log_base)
        W       = self.coint_window
        pvalues = np.full(n, np.nan)

        for t in range(W, n):
            seg_base  = log_base.iloc[t - W : t].values
            seg_hedge = log_hedge.iloc[t - W : t].values
            try:
                _, pval, _ = coint(seg_base, seg_hedge)
                pvalues[t] = pval
            except Exception:
                pvalues[t] = 1.0   # treat as non-cointegrated on error

        return pd.Series(pvalues, index=log_base.index)

    def _rolling_hedge_ratio(
        self,
        log_base:  pd.Series,
        log_hedge: pd.Series,
    ) -> pd.Series:
        """
        Rolling OLS hedge ratio β = Cov(log_base, log_hedge) / Var(log_hedge).

        This is the first-step estimator in Engle-Granger: regress log_BTC on
        log_ETH to estimate the long-run equilibrium relationship. The residual
        log_BTC - β*log_ETH is the cointegration residual (spread).

        Computed over hedge_window bars (same as coint_window in practice).
        """
        W        = self.hedge_window
        roll_cov = log_base.rolling(W).cov(log_hedge)
        roll_var = log_hedge.rolling(W).var()
        return roll_cov / roll_var.replace(0, np.nan)

    def _compute_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Normalise spread to z-score using rolling zscore_window statistics.

        Z = (spread - rolling_mean) / rolling_std
        NaN for first zscore_window bars (insufficient history).
        """
        roll_mean = spread.rolling(self.zscore_window).mean()
        roll_std  = spread.rolling(self.zscore_window).std()
        return (spread - roll_mean) / roll_std.replace(0, np.nan)

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling cointegration pairs signals for BTC/ETH.

        Steps:
            1. Compute rolling Engle-Granger p-value (coint_window)
            2. Compute rolling OLS hedge ratio β (hedge_window)
            3. Compute spread = log_BTC - β * log_ETH
            4. Compute z-score of spread (zscore_window)
            5. Apply signal rules, suppressing signal when not cointegrated

        Returns:
            DataFrame with columns [BTC/USDT, ETH/USDT, spread, zscore, pvalue],
            signal values in {-1, 0, +1}.
        """
        if self.base_asset not in prices.columns:
            raise KeyError(f"PairsStrategy requires '{self.base_asset}' in prices")
        if self.hedge_asset not in prices.columns:
            raise KeyError(f"PairsStrategy requires '{self.hedge_asset}' in prices")

        log_base  = np.log(prices[self.base_asset])
        log_hedge = np.log(prices[self.hedge_asset])

        # --- Step 1: Rolling Engle-Granger cointegration test ---
        # This is the key implementation that was missing in V1/V2.
        # We only trade when p-value < threshold (evidence of cointegration).
        pvalues = self._rolling_coint_pvalue(log_base, log_hedge)
        cointegrated = pvalues < self.coint_pvalue_threshold

        # --- Step 2: Rolling OLS hedge ratio ---
        beta   = self._rolling_hedge_ratio(log_base, log_hedge)

        # --- Step 3: Spread = log_BTC - β * log_ETH ---
        spread = log_base - beta * log_hedge

        # --- Step 4: Z-score of spread ---
        zscore = self._compute_zscore(spread)

        # --- Step 5: Signal generation with sticky state ---
        n        = len(zscore)
        sig_btc  = np.zeros(n)
        sig_eth  = np.zeros(n)
        position = 0   # +1 = long spread, -1 = short spread, 0 = flat

        for i in range(1, n):
            z      = zscore.iloc[i]
            is_coint = bool(cointegrated.iloc[i]) if not np.isnan(pvalues.iloc[i]) else False

            # NaN z-score or loss of cointegration → close immediately
            if np.isnan(z) or not is_coint:
                if position != 0:
                    position = 0   # close on cointegration loss
                sig_btc[i] = 0
                sig_eth[i] = 0
                continue

            # Stop loss — blow-up of spread, force close
            if abs(z) > self.stop_z and position != 0:
                position = 0

            # Exit condition — spread has sufficiently reverted
            elif abs(z) < self.exit_z and position != 0:
                position = 0

            # Entry — only when cointegrated (is_coint already checked above)
            elif position == 0:
                if z > self.entry_z:
                    position = -1   # short spread: short BTC, long ETH
                elif z < -self.entry_z:
                    position = +1   # long spread: long BTC, short ETH

            # Assign signals: long spread = long BTC, short ETH (and vice versa)
            sig_btc[i] =  position
            sig_eth[i] = -position

        result = pd.DataFrame({
            self.base_asset:  sig_btc,
            self.hedge_asset: sig_eth,
            "spread":         spread.values,
            "zscore":         zscore.values,
            "pvalue":         pvalues.values,
        }, index=prices.index)

        return result

    def __repr__(self) -> str:
        return (
            f"PairsStrategy(coint_window={self.coint_window}, "
            f"pvalue_threshold={self.coint_pvalue_threshold}, "
            f"entry_z={self.entry_z}, exit_z={self.exit_z}, stop_z={self.stop_z})"
        )


# =============================================================================
# STRATEGY 2 — EWM Momentum Trend Following (BTC, ETH, SOL)
# Modelled on: Notebook 3.3 - A Trend Following Strategy.ipynb
# =============================================================================

class TrendStrategy:
    """
    EWM momentum trend-following strategy across BTC, ETH, and SOL.

    Signal logic (modelled on Notebook 3.3):
        1. fast_ewm   = price.ewm(span=fast_span).mean()
        2. slow_ewm   = price.ewm(span=slow_span).mean()
        3. raw_signal = fast_ewm - slow_ewm           (positive = uptrend)
        4. vol        = ret.rolling(vol_window).std()  (realised volatility)
        5. norm_signal = raw_signal / (price × vol)    (dimensionless)
        6. direction  = sign(norm_signal)              (-1, 0, +1)

    Vol-normalisation is critical for multi-asset: BTC trades at ~$40k and
    SOL at ~$100, so raw EWM differences are not comparable. Dividing by
    (price × vol) makes the signal dimensionless and comparable across assets.

    Args:
        params : Strategy params dict from YAML. Keys:
            fast_span  : EWM fast window in bars (default 12)
            slow_span  : EWM slow window in bars (default 48)
            vol_window : rolling std window for normalisation (default 24)
            assets     : list of asset names to trade (default: all in prices)
    """

    def __init__(self, params: dict):
        self.fast_span  = int(params.get("fast_span",  12))
        self.slow_span  = int(params.get("slow_span",  48))
        self.vol_window = int(params.get("vol_window", 24))
        self.assets     = params.get("assets", None)   # None = use all columns

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate EWM momentum signals for all assets.

        Returns:
            DataFrame with columns matching assets, values in {-1, 0, +1}.
        """
        assets = self.assets if self.assets is not None else list(prices.columns)
        sigs   = {}

        for asset in assets:
            if asset not in prices.columns:
                raise KeyError(f"TrendStrategy: asset '{asset}' not in prices")

            price = prices[asset]
            ret   = price.pct_change()

            # EWM crossover — Notebook 3.3 pattern
            fast_ewm   = price.ewm(span=self.fast_span,  adjust=False).mean()
            slow_ewm   = price.ewm(span=self.slow_span,  adjust=False).mean()
            raw_signal = fast_ewm - slow_ewm

            # Vol-normalise to make signal dimensionless across assets
            vol        = ret.rolling(self.vol_window).std()
            price_vol  = price * vol
            norm_signal = raw_signal / price_vol.replace(0, np.nan)

            # Direction: NaN → 0 (flat)
            direction = np.sign(norm_signal).fillna(0).astype(int)
            sigs[asset] = direction

        return pd.DataFrame(sigs, index=prices.index)

    def __repr__(self) -> str:
        return (
            f"TrendStrategy(fast_span={self.fast_span}, "
            f"slow_span={self.slow_span}, vol_window={self.vol_window})"
        )


# =============================================================================
# BUILDER FUNCTIONS — registered in registry.yml
# =============================================================================

def build_pairs_signal(params: dict) -> PairsStrategy:
    """Registry callable for rolling cointegration pairs strategy."""
    return PairsStrategy(params)


def build_trend_signal(params: dict) -> TrendStrategy:
    """Registry callable for EWM trend following strategy."""
    return TrendStrategy(params)
