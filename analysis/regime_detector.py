"""Regime Detector — determines market state and selects appropriate strategy.

Regimes:
- strong_trend: ADX > 30, volume increasing → momentum strategy
- squeeze: ADX < 20, BB width narrow → breakout strategy (wait for breakout)
- range: ADX < 20 → mean reversion strategy
- fading_trend: ADX > 25, volume decreasing → prepare for reversal
- choppy: none of the above → DO NOT TRADE

Each regime maps to specific entry/exit rules and confidence adjustments.
"""

import numpy as np
import pandas as pd
from loguru import logger


class RegimeDetector:
    """Determines current market regime and selects strategy."""

    @staticmethod
    def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
        """Compute ADX (Average Directional Index) without external library."""
        try:
            h = high.values.astype(float)
            l = low.values.astype(float)
            c = close.values.astype(float)
            n = len(c)
            if n < window * 2:
                return 20.0

            tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
            plus_dm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0), 0)
            minus_dm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0), 0)

            # Wilder smoothing
            atr = np.zeros(len(tr))
            atr[window - 1] = np.mean(tr[:window])
            for i in range(window, len(tr)):
                atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

            plus_di_smooth = np.zeros(len(plus_dm))
            minus_di_smooth = np.zeros(len(minus_dm))
            plus_di_smooth[window - 1] = np.mean(plus_dm[:window])
            minus_di_smooth[window - 1] = np.mean(minus_dm[:window])
            for i in range(window, len(plus_dm)):
                plus_di_smooth[i] = (plus_di_smooth[i - 1] * (window - 1) + plus_dm[i]) / window
                minus_di_smooth[i] = (minus_di_smooth[i - 1] * (window - 1) + minus_dm[i]) / window

            atr_safe = np.where(atr == 0, 1e-10, atr)
            plus_di = 100 * plus_di_smooth / atr_safe
            minus_di = 100 * minus_di_smooth / atr_safe

            di_sum = plus_di + minus_di
            di_sum_safe = np.where(di_sum == 0, 1e-10, di_sum)
            dx = 100 * np.abs(plus_di - minus_di) / di_sum_safe

            # ADX = smoothed DX
            valid_dx = dx[window - 1:]
            if len(valid_dx) < window:
                return 20.0
            adx_val = np.mean(valid_dx[:window])
            for i in range(window, len(valid_dx)):
                adx_val = (adx_val * (window - 1) + valid_dx[i]) / window
            return float(adx_val)
        except Exception:
            return 20.0

    @staticmethod
    def _compute_bb_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> float:
        """Compute Bollinger Band width without external library."""
        try:
            sma = close.rolling(window).mean()
            std = close.rolling(window).std()
            upper = sma + num_std * std
            lower = sma - num_std * std
            width = (upper - lower) / sma
            return float(width.iloc[-1])
        except Exception:
            return 0.05

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Compute ATR (Average True Range) without external library."""
        h = high.values.astype(float)
        l = low.values.astype(float)
        c = close.values.astype(float)
        tr = np.zeros(len(c))
        tr[0] = h[0] - l[0]
        for i in range(1, len(c)):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atr = pd.Series(tr, index=close.index).rolling(window).mean()
        return atr

    def detect(self, df_5m: pd.DataFrame) -> dict:
        """Detect regime from 5m candles (needs ~50 candles minimum).

        Returns:
            dict with regime, strategy, and confidence multiplier.
        """
        if len(df_5m) < 30:
            return {"regime": "unknown", "strategy": "none", "confidence_mult": 0.5}

        close = df_5m["close"]
        high = df_5m["high"]
        low = df_5m["low"]
        volume = df_5m["volume"]

        # ADX (trend strength)
        adx = self._compute_adx(high, low, close, window=14)

        # Bollinger Band width (squeeze detection)
        bb_width = self._compute_bb_width(close, window=20, num_std=2.0)

        # Volume trend (is volume increasing or decreasing?)
        vol_increasing = self._is_volume_increasing(volume, 10)

        # ATR ratio (current vs historical)
        try:
            atr = self._compute_atr(high, low, close, window=14)
            atr_current = float(atr.iloc[-1])
            atr_avg = float(atr.iloc[-20:].mean())
            atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
        except Exception:
            atr_ratio = 1.0

        # Efficiency ratio (trending vs choppy)
        er = self._efficiency_ratio(close, 20)

        # Determine regime
        if adx > 30 and vol_increasing:
            regime = "strong_trend"
            strategy = "momentum"
            confidence_mult = 1.2
            hint = "Strong trend with volume — trade with trend, use trailing stop"
        elif adx < 20 and bb_width < 0.03:
            regime = "squeeze"
            strategy = "breakout"
            confidence_mult = 0.8
            hint = "Volatility squeeze — wait for breakout with volume confirmation"
        elif adx < 20 and er < 0.3:
            regime = "range"
            strategy = "mean_reversion"
            confidence_mult = 0.9
            hint = "Ranging market — trade from extremes, tight TP"
        elif adx > 25 and not vol_increasing:
            regime = "fading_trend"
            strategy = "reversal_watch"
            confidence_mult = 0.7
            hint = "Trend fading (volume drying up) — prepare for reversal"
        elif er < 0.2 or (adx < 15 and atr_ratio < 0.7):
            regime = "choppy"
            strategy = "no_trade"
            confidence_mult = 0.0
            hint = "Choppy/random market — DO NOT TRADE, any entry is a coin flip"
        else:
            regime = "normal"
            strategy = "standard"
            confidence_mult = 1.0
            hint = "Normal conditions — standard scalping"

        return {
            "regime": regime,
            "strategy": strategy,
            "confidence_mult": round(confidence_mult, 2),
            "hint": hint,
            "metrics": {
                "adx": round(adx, 1),
                "bb_width": round(bb_width, 4),
                "vol_increasing": vol_increasing,
                "atr_ratio": round(atr_ratio, 3),
                "efficiency_ratio": round(er, 3),
            },
        }

    @staticmethod
    def _is_volume_increasing(volume: pd.Series, window: int = 10) -> bool:
        """Check if volume is trending up over last N candles."""
        if len(volume) < window:
            return False
        recent = volume.tail(window).values
        first_half = np.mean(recent[:window // 2])
        second_half = np.mean(recent[window // 2:])
        return second_half > first_half * 1.1

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int = 20) -> float:
        """Kaufman's Efficiency Ratio: |net change| / sum(|individual changes|).

        ER → 1: trending, ER → 0: choppy.
        """
        values = close.dropna().values
        if len(values) < window + 1:
            return 0.5

        recent = values[-window - 1:]
        net_change = abs(recent[-1] - recent[0])
        noise = sum(abs(recent[i] - recent[i - 1]) for i in range(1, len(recent)))

        return float(net_change / noise) if noise > 0 else 0.0
