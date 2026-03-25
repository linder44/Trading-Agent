"""Delta Divergence — micro-level buy/sell volume analysis.

For each candle:
  delta = buy_volume - sell_volume (estimated from OHLCV)

Patterns detected:
1. Delta Flip: delta changes sign 3+ times → reversal
2. Stacked Delta: 5+ candles same sign → strong trend
3. Exhaustion Delta: price makes new high/low but delta is opposite → reversal
4. Trapped Traders: sharp move → delta flip → price doesn't return
"""

import numpy as np
import pandas as pd
from loguru import logger


class DeltaDivergence:
    """Micro-level delta analysis for scalping."""

    def analyze(self, df: pd.DataFrame) -> dict:
        """Run all delta divergence analyses."""
        if len(df) < 15:
            return {"signal": "insufficient_data"}

        deltas = self._compute_deltas(df)

        return {
            "current_delta": round(float(deltas[-1]), 2),
            "delta_flip": self._delta_flip(deltas),
            "stacked_delta": self._stacked_delta(deltas),
            "exhaustion": self._exhaustion_delta(df, deltas),
            "delta_verdict": self._verdict(df, deltas),
        }

    @staticmethod
    def _compute_deltas(df: pd.DataFrame) -> np.ndarray:
        """Estimate per-candle delta (buy_vol - sell_vol) from OHLCV.

        Approximation: buy_volume = volume * (close - low) / (high - low)
                        sell_volume = volume * (high - close) / (high - low)
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values

        total_range = high - low
        # Avoid division by zero
        total_range = np.where(total_range == 0, 1e-10, total_range)

        buy_vol = volume * (close - low) / total_range
        sell_vol = volume * (high - close) / total_range

        return buy_vol - sell_vol

    @staticmethod
    def _delta_flip(deltas: np.ndarray) -> dict:
        """Detect delta sign flips in recent candles.

        3+ flips in 10 candles = indecision, potential reversal.
        """
        recent = deltas[-10:]
        signs = np.sign(recent)
        flips = np.sum(np.abs(np.diff(signs)) > 0)

        if flips >= 5:
            return {"count": int(flips), "signal": "high_indecision"}
        elif flips >= 3:
            return {"count": int(flips), "signal": "moderate_indecision"}
        else:
            return {"count": int(flips), "signal": "stable"}

    @staticmethod
    def _stacked_delta(deltas: np.ndarray) -> dict:
        """Detect stacked (consecutive same-sign) deltas.

        5+ candles with positive delta = strong buying pressure.
        5+ candles with negative delta = strong selling pressure.
        """
        recent = deltas[-10:]
        positive_streak = 0
        negative_streak = 0

        # Count from end
        for i in range(len(recent) - 1, -1, -1):
            if recent[i] > 0:
                if negative_streak > 0:
                    break
                positive_streak += 1
            elif recent[i] < 0:
                if positive_streak > 0:
                    break
                negative_streak += 1
            else:
                break

        if positive_streak >= 5:
            return {"streak": positive_streak, "direction": "bullish", "signal": "strong_buying"}
        elif negative_streak >= 5:
            return {"streak": negative_streak, "direction": "bearish", "signal": "strong_selling"}
        elif positive_streak >= 3:
            return {"streak": positive_streak, "direction": "bullish", "signal": "moderate_buying"}
        elif negative_streak >= 3:
            return {"streak": negative_streak, "direction": "bearish", "signal": "moderate_selling"}
        else:
            return {"streak": max(positive_streak, negative_streak), "direction": "mixed", "signal": "neutral"}

    @staticmethod
    def _exhaustion_delta(df: pd.DataFrame, deltas: np.ndarray) -> dict:
        """Detect exhaustion divergence.

        Price makes new high but delta is negative → buyers exhausted.
        Price makes new low but delta is positive → sellers exhausted.
        """
        if len(df) < 10:
            return {"signal": "insufficient_data"}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Check if price made new 10-candle high with negative delta
        recent_high = np.max(high[-10:])
        recent_low = np.min(low[-10:])

        latest_high = high[-1]
        latest_low = low[-1]
        latest_delta = deltas[-1]

        if latest_high >= recent_high * 0.999 and latest_delta < 0:
            return {"signal": "bullish_exhaustion", "hint": "New high but negative delta — buyers exhausted, prepare short"}
        elif latest_low <= recent_low * 1.001 and latest_delta > 0:
            return {"signal": "bearish_exhaustion", "hint": "New low but positive delta — sellers exhausted, prepare long"}

        return {"signal": "none"}

    def _verdict(self, df: pd.DataFrame, deltas: np.ndarray) -> dict:
        """Combined delta verdict."""
        stacked = self._stacked_delta(deltas)
        exhaustion = self._exhaustion_delta(df, deltas)
        flip = self._delta_flip(deltas)

        # Exhaustion overrides stacked (counter-trend signal)
        if exhaustion["signal"] == "bullish_exhaustion":
            return {"signal": "bearish_reversal", "source": "exhaustion_delta"}
        elif exhaustion["signal"] == "bearish_exhaustion":
            return {"signal": "bullish_reversal", "source": "exhaustion_delta"}

        # Stacked delta = trend confirmation
        if stacked["signal"] == "strong_buying":
            return {"signal": "strong_bullish", "source": "stacked_delta"}
        elif stacked["signal"] == "strong_selling":
            return {"signal": "strong_bearish", "source": "stacked_delta"}

        # High indecision = caution
        if flip["signal"] == "high_indecision":
            return {"signal": "choppy", "source": "delta_flip"}

        return {"signal": "neutral", "source": "none"}
