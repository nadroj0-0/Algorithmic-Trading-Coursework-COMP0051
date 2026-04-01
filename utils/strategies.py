# =============================================================================
# utils/strategies.py — COMP0051 Algorithmic Trading Coursework
# Signal generation for both trading strategies.
#
# Two strategy classes, each modelled directly on the lecturer's notebooks:
#   PairsStrategy  — Notebook 3.5 (Cointegration)
#   TrendStrategy  — Notebook 3.3 (A Trend Following Strategy)
#
# Each class has a generate(data) method that returns a signal DataFrame
# with one column per asset, values in {-1, 0, +1}.
#
# Builder functions (build_pairs_signal, build_trend_signal) are registered
# in registry.yml and resolved by config_loader.py at runtime — same
# pattern as build_baseline_gru etc. in the legacy network.py.
# =============================================================================

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


# =============================================================================
# STRATEGY 1 — Cointegration Pairs (BTC/ETH)
# Modelled on: Notebook 3.5 - Cointegration.ipynb
# =============================================================================

class PairsStrategy:
    """
    Cointegration-based pairs trading strategy on BTC/USDT and ETH/USDT.

    Signal logic (Engle-Granger, from Notebook 3.5):
        1. Rolling Engle-Granger cointegration test on log prices.
           Uses statsmodels.tsa.stattools.coint — same as the notebook.
        2. OLS hedge ratio: β = Cov(log_BTC, log_ETH) / Var(log_ETH)
           over a rolling hedge_window.
        3. Spread = log_BTC − β × log_ETH
        4. Z-score = (spread − rolling_mean) / rolling_std
           over zscore_window bars.
        5. Signal rules:
             z > +entry_z  → long spread (long BTC, short ETH scaled by β)
             z < −entry_z  → short spread (short BTC, long ETH scaled by β)
             |z| < exit_z  → close position (signal = 0)
             |z| > stop_z  → stop loss, force close (flag and set to 0)
        6. Position state is sticky — once entered, held until exit/stop.

    BTC is the "base" asset, ETH is the "hedge" asset.
    The spread is long BTC vs short ETH (or vice versa), sized so that
    the hedge ratio β neutralises cointegration residual risk.

    Args:
        params : dict from strategy YAML params section. Keys:
            coint_window  : bars for rolling Engle-Granger test (default 720)
            hedge_window  : bars for OLS hedge ratio estimation (default 720)
            zscore_window : bars for rolling mean/std of spread (default 168)
            entry_z       : |z| threshold to open position (default 1.5)
            exit_z        : |z| threshold to close position (default 0.5)
            stop_z        : |z| threshold for stop loss (default 3.0)
    """

    def __init__(self, params: dict):
        self.coint_window  = int(params.get("coint_window",  720))
        self.hedge_window  = int(params.get("hedge_window",  720))
        self.zscore_window = int(params.get("zscore_window", 168))
        self.entry_z       = float(params.get("entry_z",     1.5))
        self.exit_z        = float(params.get("exit_z",      0.5))
        self.stop_z        = float(params.get("stop_z",      3.0))

        # BTC is base, ETH is hedge — as in the coursework plan
        self.base_asset  = "BTC/USDT"
        self.hedge_asset = "ETH/USDT"

    def _compute_hedge_ratio(
        self,
        log_base:  pd.Series,
        log_hedge: pd.Series,
    ) -> pd.Series:
        """
        Rolling OLS hedge ratio β = Cov(log_base, log_hedge) / Var(log_hedge).

        This is the discrete-time Engle-Granger first-step estimator.
        Computed over hedge_window bars using rolling windows.

        Returns:
            Series of rolling hedge ratios, same index as inputs.
        """
        roll_cov = log_base.rolling(self.hedge_window).cov(log_hedge)
        roll_var = log_hedge.rolling(self.hedge_window).var()
        return roll_cov / roll_var.replace(0, np.nan)

    def _compute_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Normalise spread to z-score using rolling window statistics.

        Z-score = (spread - rolling_mean) / rolling_std
        over zscore_window bars.

        Returns:
            Series of z-scores, NaN for first zscore_window bars.
        """
        roll_mean = spread.rolling(self.zscore_window).mean()
        roll_std  = spread.rolling(self.zscore_window).std()
        return (spread - roll_mean) / roll_std.replace(0, np.nan)

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the BTC/ETH pairs strategy.

        Args:
            prices : DataFrame with one column per asset (close prices),
                     indexed by timestamp. Must contain BTC/USDT and ETH/USDT.

        Returns:
            DataFrame with columns [BTC/USDT, ETH/USDT, spread, zscore],
            where signal values are in {-1, 0, +1}.
            - BTC/USDT = +1 → long BTC
            - ETH/USDT = +1 → long ETH (and vice versa)
        """
        if self.base_asset not in prices.columns:
            raise KeyError(f"PairsStrategy requires '{self.base_asset}' in prices")
        if self.hedge_asset not in prices.columns:
            raise KeyError(f"PairsStrategy requires '{self.hedge_asset}' in prices")

        log_base  = np.log(prices[self.base_asset])
        log_hedge = np.log(prices[self.hedge_asset])

        # --- Step 1: Rolling hedge ratio (OLS, from Notebook 3.5) ---
        beta   = self._compute_hedge_ratio(log_base, log_hedge)

        # --- Step 2: Spread = log_BTC - β × log_ETH ---
        spread = log_base - beta * log_hedge

        # --- Step 3: Z-score of spread ---
        zscore = self._compute_zscore(spread)

        # --- Step 4: Signal generation with sticky position state ---
        n      = len(zscore)
        sig_btc = np.zeros(n)
        sig_eth = np.zeros(n)
        position = 0    # current position: +1 = long spread, -1 = short spread, 0 = flat

        for i in range(1, n):
            z = zscore.iloc[i]
            if np.isnan(z):
                sig_btc[i] = 0
                sig_eth[i] = 0
                continue

            # Stop loss — force close if spread has blown past stop_z
            if abs(z) > self.stop_z and position != 0:
                position = 0

            # Exit condition — close when spread has reverted sufficiently
            elif abs(z) < self.exit_z and position != 0:
                position = 0

            # Entry conditions — open if no current position
            elif position == 0:
                if z > self.entry_z:
                    position = -1   # short spread: short BTC, long ETH
                elif z < -self.entry_z:
                    position = +1   # long spread: long BTC, short ETH

            # Assign signals based on position
            # Long spread: long BTC (+1), short ETH (-1)
            # Short spread: short BTC (-1), long ETH (+1)
            sig_btc[i] = position
            sig_eth[i] = -position

        result = pd.DataFrame({
            self.base_asset:  sig_btc,
            self.hedge_asset: sig_eth,
            "spread":         spread.values,
            "zscore":         zscore.values,
        }, index=prices.index)

        return result

    def __repr__(self) -> str:
        return (
            f"PairsStrategy(coint_window={self.coint_window}, "
            f"zscore_window={self.zscore_window}, "
            f"entry_z={self.entry_z}, exit_z={self.exit_z}, stop_z={self.stop_z})"
        )


# =============================================================================
# STRATEGY 2 — Trend Following / EWM Momentum (BTC, ETH, SOL)
# Modelled on: Notebook 3.3 - A Trend Following Strategy.ipynb
# =============================================================================

class TrendStrategy:
    """
    EWM momentum trend-following strategy across BTC, ETH, and SOL.

    Signal logic (modelled on Notebook 3.3):
        1. fast_ewm  = price.ewm(span=fast_span).mean()
        2. slow_ewm  = price.ewm(span=slow_span).mean()
        3. raw_signal = fast_ewm - slow_ewm         (positive = uptrend)
        4. vol        = ret.rolling(vol_window).std() (realised volatility)
        5. norm_signal = raw_signal / (price × vol)   (dimensionless)
        6. direction  = sign(norm_signal)             (-1, 0, +1)

    The vol-normalisation is critical for a multi-asset strategy:
    BTC trades at ~$40,000 and SOL at ~$100, so raw EWM differences
    are not comparable across assets. Dividing by (price × vol) makes
    the signal dimensionless and comparable.

    Signal is 0 (flat) when vol is zero (no data) or when norm_signal
    is exactly zero (rare but handled).

    Args:
        params : dict from strategy YAML params section. Keys:
            fast_span  : EWM fast window in bars (default 12)
            slow_span  : EWM slow window in bars (default 48)
            vol_window : rolling std window in bars (default 24)
            assets     : list of asset names to trade (default: all in prices)
    """

    def __init__(self, params: dict):
        self.fast_span  = int(params.get("fast_span",  12))
        self.slow_span  = int(params.get("slow_span",  48))
        self.vol_window = int(params.get("vol_window", 24))
        self.assets     = params.get("assets", None)   # None = use all columns

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend-following signals for all assets.

        Args:
            prices : DataFrame with one column per asset (close prices),
                     indexed by timestamp.

        Returns:
            DataFrame with columns matching prices columns + momentum column,
            where signal values are in {-1, 0, +1}.
        """
        assets = self.assets if self.assets is not None else list(prices.columns)
        sigs   = {}

        for asset in assets:
            if asset not in prices.columns:
                raise KeyError(f"TrendStrategy: asset '{asset}' not in prices")

            price = prices[asset]
            ret   = price.pct_change()

            # --- EWM crossover (Notebook 3.3 pattern) ---
            fast_ewm  = price.ewm(span=self.fast_span,  adjust=False).mean()
            slow_ewm  = price.ewm(span=self.slow_span,  adjust=False).mean()
            raw_signal = fast_ewm - slow_ewm

            # --- Vol-normalised signal (makes cross-asset comparison valid) ---
            vol           = ret.rolling(self.vol_window).std()
            price_vol     = price * vol
            norm_signal   = raw_signal / price_vol.replace(0, np.nan)

            # --- Direction: sign of normalised signal ---
            # np.sign returns -1, 0, +1; NaN positions become 0 (flat)
            direction = np.sign(norm_signal).fillna(0).astype(int)
            sigs[asset] = direction

        result = pd.DataFrame(sigs, index=prices.index)
        return result

    def __repr__(self) -> str:
        return (
            f"TrendStrategy(fast_span={self.fast_span}, "
            f"slow_span={self.slow_span}, vol_window={self.vol_window})"
        )


# =============================================================================
# BUILDER FUNCTIONS — registered in registry.yml, resolved by config_loader
# =============================================================================

def build_pairs_signal(params: dict) -> PairsStrategy:
    """
    Registry callable for pairs cointegration strategy.
    Analogous to build_baseline_gru() in legacy network.py.

    Args:
        params : Strategy params dict from YAML.

    Returns:
        Configured PairsStrategy instance.
    """
    return PairsStrategy(params)


def build_trend_signal(params: dict) -> TrendStrategy:
    """
    Registry callable for trend following strategy.
    Analogous to build_gru() in legacy network.py.

    Args:
        params : Strategy params dict from YAML.

    Returns:
        Configured TrendStrategy instance.
    """
    return TrendStrategy(params)
