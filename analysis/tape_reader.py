"""Tape Reader — analysis of trade flow from OHLCV data.

Professional scalpers read the tape (time & sales) to detect:
1. Trade intensity — sudden bursts of activity
2. Aggressive vs passive flow — taker buy vs taker sell
3. Absorption — large volume but no price movement (someone holding a level)
4. Sweep detection — rapid same-side trades (institutional activity)
5. Exhaustion — decreasing trade sizes near extremes
"""

import numpy as np
import pandas as pd
from loguru import logger


class TapeReader:
    """Analyzes trade flow patterns from OHLCV data (proxy for real tape)."""

    def analyze(self, df: pd.DataFrame) -> dict:
        """Run all tape reading analyses on OHLCV data.

        Since we don't have tick-level data, we approximate from candle data.
        """
        if len(df) < 15:
            return {"signal": "insufficient_data"}

        return {
            "trade_intensity": self._trade_intensity(df),
            "aggressive_flow": self._aggressive_flow(df),
            "absorption": self._absorption_detection(df),
            "sweep": self._sweep_detection(df),
            "exhaustion": self._exhaustion_detection(df),
            "tape_verdict": self._verdict(df),
        }

    @staticmethod
    def _trade_intensity(df: pd.DataFrame) -> dict:
        """Detect sudden bursts of trading activity.

        Uses volume as proxy for trade count.
        Volume spike = someone is aggressively entering.
        """
        vol = df["volume"].values
        if len(vol) < 15:
            return {"ratio": 1.0, "signal": "normal"}

        recent_avg = np.mean(vol[-5:])
        baseline_avg = np.mean(vol[-20:-5])

        if baseline_avg <= 0:
            return {"ratio": 1.0, "signal": "normal"}

        ratio = recent_avg / baseline_avg

        if ratio > 3.0:
            signal = "extreme_burst"
        elif ratio > 2.0:
            signal = "high_activity"
        elif ratio > 1.5:
            signal = "elevated"
        elif ratio < 0.5:
            signal = "dead_market"
        else:
            signal = "normal"

        return {
            "ratio": round(float(ratio), 2),
            "signal": signal,
        }

    @staticmethod
    def _aggressive_flow(df: pd.DataFrame) -> dict:
        """Estimate taker buy vs taker sell volume.

        Approximation: if close > open, most volume is taker buy.
        Weighted by how close the close is to high (strong buy) or low (strong sell).
        """
        if len(df) < 10:
            return {"buy_pct": 50, "sell_pct": 50, "signal": "balanced"}

        recent = df.tail(10)
        taker_buy = 0.0
        taker_sell = 0.0

        for _, row in recent.iterrows():
            total_range = row["high"] - row["low"]
            if total_range <= 0:
                continue
            # Close-to-high ratio → taker buy pressure
            buy_ratio = (row["close"] - row["low"]) / total_range
            sell_ratio = (row["high"] - row["close"]) / total_range
            taker_buy += row["volume"] * buy_ratio
            taker_sell += row["volume"] * sell_ratio

        total = taker_buy + taker_sell
        if total <= 0:
            return {"buy_pct": 50, "sell_pct": 50, "signal": "balanced"}

        buy_pct = taker_buy / total * 100
        sell_pct = taker_sell / total * 100

        if buy_pct > 65:
            signal = "aggressive_buying"
        elif sell_pct > 65:
            signal = "aggressive_selling"
        elif buy_pct > 55:
            signal = "moderate_buying"
        elif sell_pct > 55:
            signal = "moderate_selling"
        else:
            signal = "balanced"

        return {
            "buy_pct": round(buy_pct, 1),
            "sell_pct": round(sell_pct, 1),
            "signal": signal,
        }

    @staticmethod
    def _absorption_detection(df: pd.DataFrame) -> dict:
        """Detect absorption — large volume but no price movement.

        This means a large player is absorbing selling/buying pressure,
        holding a price level. Very strong signal for support/resistance.
        """
        if len(df) < 10:
            return {"detected": False, "signal": "no_data"}

        recent = df.tail(10)
        vol = recent["volume"].values
        price_change = np.abs(recent["close"].values - recent["open"].values)
        avg_vol = np.mean(vol)

        # Look for candles with high volume but tiny body
        absorptions = []
        for i in range(len(recent)):
            if avg_vol > 0 and vol[i] > avg_vol * 1.5:
                body_pct = price_change[i] / recent["close"].values[i] * 100 if recent["close"].values[i] > 0 else 0
                if body_pct < 0.05:  # Very small body despite high volume
                    absorptions.append({
                        "vol_ratio": round(vol[i] / avg_vol, 2),
                        "body_pct": round(body_pct, 4),
                    })

        if absorptions:
            # Determine direction: is price being held at support or resistance?
            last_price = float(df["close"].iloc[-1])
            recent_low = float(recent["low"].min())
            recent_high = float(recent["high"].max())
            range_pos = (last_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

            if range_pos < 0.3:
                signal = "absorption_at_support"
            elif range_pos > 0.7:
                signal = "absorption_at_resistance"
            else:
                signal = "absorption_mid_range"

            return {
                "detected": True,
                "count": len(absorptions),
                "signal": signal,
            }

        return {"detected": False, "signal": "none"}

    @staticmethod
    def _sweep_detection(df: pd.DataFrame) -> dict:
        """Detect liquidity sweeps — rapid same-side candles with increasing volume.

        A sweep is when an institutional player aggressively takes out
        resting orders in one direction. 3+ candles same direction with
        increasing volume = sweep.
        """
        if len(df) < 5:
            return {"detected": False, "signal": "none"}

        recent = df.tail(5)
        closes = recent["close"].values
        opens = recent["open"].values
        volumes = recent["volume"].values

        # Count consecutive same-direction candles
        bullish_streak = 0
        bearish_streak = 0
        vol_increasing = True

        for i in range(len(recent) - 1, 0, -1):
            if closes[i] > opens[i]:
                bullish_streak += 1
                if i > 0 and volumes[i] < volumes[i - 1] * 0.8:
                    vol_increasing = False
            else:
                break

        for i in range(len(recent) - 1, 0, -1):
            if closes[i] < opens[i]:
                bearish_streak += 1
                if i > 0 and volumes[i] < volumes[i - 1] * 0.8:
                    vol_increasing = False
            else:
                break

        if bullish_streak >= 3 and vol_increasing:
            return {"detected": True, "direction": "bullish", "streak": bullish_streak, "signal": "bullish_sweep"}
        elif bearish_streak >= 3 and vol_increasing:
            return {"detected": True, "direction": "bearish", "streak": bearish_streak, "signal": "bearish_sweep"}

        return {"detected": False, "signal": "none"}

    @staticmethod
    def _exhaustion_detection(df: pd.DataFrame) -> dict:
        """Detect exhaustion — volume rising but candle bodies shrinking.

        This is the "last gasp" of a move. Volume is high because people
        are still trying, but smaller and smaller — sellers/buyers are drying up.
        """
        if len(df) < 8:
            return {"detected": False, "signal": "none"}

        recent = df.tail(8)
        bodies = np.abs(recent["close"].values - recent["open"].values)
        volumes = recent["volume"].values

        # Check last 5 candles
        last5_bodies = bodies[-5:]
        last5_vols = volumes[-5:]

        if len(last5_bodies) < 5:
            return {"detected": False, "signal": "none"}

        # Bodies shrinking (each smaller than previous)?
        body_shrinking = all(last5_bodies[i] <= last5_bodies[i - 1] * 1.1 for i in range(2, 5))
        # Volume still elevated?
        avg_vol = np.mean(volumes[:3]) if len(volumes) >= 3 else 1
        vol_elevated = np.mean(last5_vols) > avg_vol * 0.8 if avg_vol > 0 else False

        if body_shrinking and vol_elevated:
            # Determine direction of exhaustion
            trend = recent["close"].values[-1] - recent["close"].values[0]
            if trend > 0:
                return {"detected": True, "signal": "bullish_exhaustion"}
            elif trend < 0:
                return {"detected": True, "signal": "bearish_exhaustion"}

        return {"detected": False, "signal": "none"}

    def _verdict(self, df: pd.DataFrame) -> dict:
        """Combine all tape signals into a verdict."""
        intensity = self._trade_intensity(df)
        flow = self._aggressive_flow(df)
        absorption = self._absorption_detection(df)
        sweep = self._sweep_detection(df)
        exhaustion = self._exhaustion_detection(df)

        score = 0  # -3 to +3

        # Aggressive flow
        if flow["signal"] == "aggressive_buying":
            score += 2
        elif flow["signal"] == "moderate_buying":
            score += 1
        elif flow["signal"] == "aggressive_selling":
            score -= 2
        elif flow["signal"] == "moderate_selling":
            score -= 1

        # Sweep
        if sweep["signal"] == "bullish_sweep":
            score += 1
        elif sweep["signal"] == "bearish_sweep":
            score -= 1

        # Absorption (reversal signal)
        if absorption["signal"] == "absorption_at_support":
            score += 1
        elif absorption["signal"] == "absorption_at_resistance":
            score -= 1

        # Exhaustion (counter-trend signal)
        if exhaustion["signal"] == "bullish_exhaustion":
            score -= 1  # Bulls exhausted → bearish
        elif exhaustion["signal"] == "bearish_exhaustion":
            score += 1  # Bears exhausted → bullish

        # Dead market = no trade
        if intensity["signal"] == "dead_market":
            return {"score": 0, "signal": "dead_market", "action": "no_trade"}

        if score >= 2:
            signal = "strong_bullish"
        elif score >= 1:
            signal = "bullish"
        elif score <= -2:
            signal = "strong_bearish"
        elif score <= -1:
            signal = "bearish"
        else:
            signal = "neutral"

        return {"score": score, "signal": signal}
