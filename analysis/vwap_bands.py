"""VWAP Deviation Bands — mean reversion and breakout signals.

VWAP = Volume Weighted Average Price
Bands = VWAP +/- N standard deviations (weighted by volume)

Strategy:
- Price at upper_2 + weakening momentum → short (mean reversion to VWAP)
- Price at lower_2 + strengthening momentum → long (mean reversion to VWAP)
- Price breaks upper_2 with volume > 2x → momentum long (breakout)
- Price between lower_1 and upper_1 → fair value zone, don't enter
"""

import numpy as np
import pandas as pd
from loguru import logger


class VWAPBands:
    """VWAP with deviation bands for scalping signals."""

    def analyze(self, df: pd.DataFrame) -> dict:
        """Compute VWAP bands and generate signals."""
        if len(df) < 20:
            return {"signal": "insufficient_data"}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # VWAP calculation
        typical_price = (high + low + close) / 3
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical_price * volume)

        # Avoid division by zero
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical_price)

        # Volume-weighted standard deviation (rolling)
        window = min(50, len(df))
        recent_tp = typical_price[-window:]
        recent_vol = volume[-window:]
        recent_vwap = vwap[-1]

        if np.sum(recent_vol) > 0:
            weighted_var = np.sum(recent_vol * (recent_tp - recent_vwap) ** 2) / np.sum(recent_vol)
            stdev = np.sqrt(max(weighted_var, 0))
        else:
            stdev = 0

        current_price = float(close[-1])
        current_vwap = float(vwap[-1])

        if stdev <= 0:
            return {
                "vwap": round(current_vwap, 6),
                "stdev": 0,
                "signal": "no_deviation",
            }

        upper_1 = current_vwap + stdev
        upper_2 = current_vwap + 2 * stdev
        lower_1 = current_vwap - stdev
        lower_2 = current_vwap - 2 * stdev

        # Determine zone
        deviation = (current_price - current_vwap) / stdev if stdev > 0 else 0

        # Volume context — is current volume above average?
        avg_vol = np.mean(volume[-20:])
        current_vol = volume[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Generate signal
        if deviation > 2.0 and vol_ratio < 1.5:
            signal = "mean_revert_short"
            hint = "Price at +2σ VWAP without volume confirmation — short toward VWAP"
        elif deviation < -2.0 and vol_ratio < 1.5:
            signal = "mean_revert_long"
            hint = "Price at -2σ VWAP without volume confirmation — long toward VWAP"
        elif deviation > 2.0 and vol_ratio > 2.0:
            signal = "breakout_long"
            hint = "Price broke +2σ VWAP WITH strong volume — momentum long"
        elif deviation < -2.0 and vol_ratio > 2.0:
            signal = "breakout_short"
            hint = "Price broke -2σ VWAP WITH strong volume — momentum short"
        elif abs(deviation) < 1.0:
            signal = "fair_value"
            hint = "Price in fair value zone (within 1σ VWAP) — no edge"
        elif deviation > 1.0:
            signal = "elevated"
            hint = "Price above +1σ VWAP — watch for rejection"
        elif deviation < -1.0:
            signal = "depressed"
            hint = "Price below -1σ VWAP — watch for bounce"
        else:
            signal = "neutral"
            hint = ""

        return {
            "vwap": round(current_vwap, 6),
            "deviation_sigma": round(float(deviation), 3),
            "upper_1": round(float(upper_1), 6),
            "upper_2": round(float(upper_2), 6),
            "lower_1": round(float(lower_1), 6),
            "lower_2": round(float(lower_2), 6),
            "stdev": round(float(stdev), 6),
            "vol_ratio": round(float(vol_ratio), 2),
            "signal": signal,
            "hint": hint,
        }
