"""Candlestick pattern recognition and Fibonacci levels."""

import numpy as np
import pandas as pd
from loguru import logger


class PatternRecognizer:
    """Detects candlestick patterns and computes Fibonacci levels."""

    def detect_patterns(self, df: pd.DataFrame) -> list[dict]:
        """Detect candlestick patterns in OHLCV data."""
        patterns = []
        if len(df) < 5:
            return patterns

        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        body = np.abs(c - o)
        full_range = h - l
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        # Avoid division by zero
        full_range_safe = np.where(full_range == 0, 1e-10, full_range)
        body_safe = np.where(body == 0, 1e-10, body)

        i = len(df) - 1  # Latest candle
        prev = i - 1

        # --- Single candle patterns ---

        # Doji: very small body relative to range
        if body[i] < full_range_safe[i] * 0.1 and full_range[i] > 0:
            patterns.append({
                "pattern": "doji",
                "signal": "reversal_warning",
                "description": "Indecision - body < 10% of range. Potential reversal.",
            })

        # Hammer (bullish): small body at top, long lower shadow
        if (lower_shadow[i] > body_safe[i] * 2 and
                upper_shadow[i] < body_safe[i] * 0.5 and
                c[i] > o[i]):
            patterns.append({
                "pattern": "hammer",
                "signal": "bullish_reversal",
                "description": "Long lower shadow, small body at top. Buyers stepped in.",
            })

        # Inverted Hammer / Shooting Star
        if (upper_shadow[i] > body_safe[i] * 2 and
                lower_shadow[i] < body_safe[i] * 0.5):
            if c[i] < o[i]:
                patterns.append({
                    "pattern": "shooting_star",
                    "signal": "bearish_reversal",
                    "description": "Long upper shadow, rejection from highs.",
                })
            else:
                patterns.append({
                    "pattern": "inverted_hammer",
                    "signal": "bullish_reversal",
                    "description": "Long upper shadow after downtrend, potential reversal up.",
                })

        # --- Two candle patterns ---

        # Bullish Engulfing
        if (c[prev] < o[prev] and  # Previous red
                c[i] > o[i] and  # Current green
                o[i] < c[prev] and  # Open below prev close
                c[i] > o[prev]):  # Close above prev open
            patterns.append({
                "pattern": "bullish_engulfing",
                "signal": "strong_bullish",
                "description": "Green candle completely engulfs previous red candle.",
            })

        # Bearish Engulfing
        if (c[prev] > o[prev] and  # Previous green
                c[i] < o[i] and  # Current red
                o[i] > c[prev] and  # Open above prev close
                c[i] < o[prev]):  # Close below prev open
            patterns.append({
                "pattern": "bearish_engulfing",
                "signal": "strong_bearish",
                "description": "Red candle completely engulfs previous green candle.",
            })

        # Morning Star (3-candle bullish reversal)
        if len(df) >= 3:
            pp = i - 2
            if (c[pp] < o[pp] and  # First: big red
                    body[prev] < body[pp] * 0.3 and  # Second: small body (star)
                    c[i] > o[i] and  # Third: big green
                    c[i] > (o[pp] + c[pp]) / 2):  # Third closes above midpoint of first
                patterns.append({
                    "pattern": "morning_star",
                    "signal": "strong_bullish",
                    "description": "Three-candle bottom reversal pattern.",
                })

        # Evening Star (3-candle bearish reversal)
        if len(df) >= 3:
            pp = i - 2
            if (c[pp] > o[pp] and  # First: big green
                    body[prev] < body[pp] * 0.3 and  # Second: small body (star)
                    c[i] < o[i] and  # Third: big red
                    c[i] < (o[pp] + c[pp]) / 2):  # Third closes below midpoint of first
                patterns.append({
                    "pattern": "evening_star",
                    "signal": "strong_bearish",
                    "description": "Three-candle top reversal pattern.",
                })

        # Three White Soldiers (bullish)
        if len(df) >= 3:
            pp = i - 2
            if (c[pp] > o[pp] and c[prev] > o[prev] and c[i] > o[i] and  # All green
                    c[prev] > c[pp] and c[i] > c[prev] and  # Each closes higher
                    body[pp] > full_range_safe[pp] * 0.5 and  # Big bodies
                    body[prev] > full_range_safe[prev] * 0.5 and
                    body[i] > full_range_safe[i] * 0.5):
                patterns.append({
                    "pattern": "three_white_soldiers",
                    "signal": "strong_bullish",
                    "description": "Three consecutive strong green candles. Strong buying pressure.",
                })

        # Three Black Crows (bearish)
        if len(df) >= 3:
            pp = i - 2
            if (c[pp] < o[pp] and c[prev] < o[prev] and c[i] < o[i] and  # All red
                    c[prev] < c[pp] and c[i] < c[prev] and  # Each closes lower
                    body[pp] > full_range_safe[pp] * 0.5 and
                    body[prev] > full_range_safe[prev] * 0.5 and
                    body[i] > full_range_safe[i] * 0.5):
                patterns.append({
                    "pattern": "three_black_crows",
                    "signal": "strong_bearish",
                    "description": "Three consecutive strong red candles. Strong selling pressure.",
                })

        return patterns

    def compute_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        """Compute Fibonacci retracement levels from recent swing high/low."""
        if len(df) < lookback:
            lookback = len(df)

        recent = df.tail(lookback)
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())
        diff = swing_high - swing_low
        current_price = float(df["close"].iloc[-1])

        levels = {
            "swing_high": round(swing_high, 6),
            "swing_low": round(swing_low, 6),
            "fib_0.236": round(swing_high - 0.236 * diff, 6),
            "fib_0.382": round(swing_high - 0.382 * diff, 6),
            "fib_0.5": round(swing_high - 0.5 * diff, 6),
            "fib_0.618": round(swing_high - 0.618 * diff, 6),
            "fib_0.786": round(swing_high - 0.786 * diff, 6),
        }

        # Determine where price is relative to Fibonacci levels
        if current_price > levels["fib_0.236"]:
            levels["price_zone"] = "above_0.236 (strong uptrend)"
        elif current_price > levels["fib_0.382"]:
            levels["price_zone"] = "0.236-0.382 (shallow pullback)"
        elif current_price > levels["fib_0.5"]:
            levels["price_zone"] = "0.382-0.5 (moderate pullback)"
        elif current_price > levels["fib_0.618"]:
            levels["price_zone"] = "0.5-0.618 (golden zone - best entry)"
        elif current_price > levels["fib_0.786"]:
            levels["price_zone"] = "0.618-0.786 (deep pullback)"
        else:
            levels["price_zone"] = "below_0.786 (trend likely broken)"

        return levels

    def detect_divergences(self, df: pd.DataFrame) -> list[dict]:
        """Detect RSI and MACD divergences (leading reversal signals)."""
        divergences = []
        if len(df) < 30 or "rsi" not in df.columns:
            return divergences

        close = df["close"].values
        rsi = df["rsi"].values

        # Look at last 20 candles for swing points
        window = min(20, len(df) - 1)

        # Find local lows in price and RSI
        price_lows = []
        rsi_lows = []
        for j in range(len(df) - window, len(df) - 2):
            if close[j] < close[j - 1] and close[j] < close[j + 1]:
                price_lows.append((j, close[j]))
            if rsi[j] < rsi[j - 1] and rsi[j] < rsi[j + 1]:
                rsi_lows.append((j, rsi[j]))

        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (price_lows[-1][1] < price_lows[-2][1] and
                    rsi_lows[-1][1] > rsi_lows[-2][1]):
                divergences.append({
                    "type": "bullish_divergence",
                    "indicator": "RSI",
                    "signal": "bullish_reversal",
                    "description": "Price lower low + RSI higher low = buying pressure building.",
                })

        # Find local highs
        price_highs = []
        rsi_highs = []
        for j in range(len(df) - window, len(df) - 2):
            if close[j] > close[j - 1] and close[j] > close[j + 1]:
                price_highs.append((j, close[j]))
            if rsi[j] > rsi[j - 1] and rsi[j] > rsi[j + 1]:
                rsi_highs.append((j, rsi[j]))

        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if (price_highs[-1][1] > price_highs[-2][1] and
                    rsi_highs[-1][1] < rsi_highs[-2][1]):
                divergences.append({
                    "type": "bearish_divergence",
                    "indicator": "RSI",
                    "signal": "bearish_reversal",
                    "description": "Price higher high + RSI lower high = selling pressure building.",
                })

        # MACD divergences
        if "macd" in df.columns:
            macd = df["macd"].values
            macd_lows = []
            macd_highs = []
            for j in range(len(df) - window, len(df) - 2):
                if macd[j] < macd[j - 1] and macd[j] < macd[j + 1]:
                    macd_lows.append((j, macd[j]))
                if macd[j] > macd[j - 1] and macd[j] > macd[j + 1]:
                    macd_highs.append((j, macd[j]))

            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                if (price_lows[-1][1] < price_lows[-2][1] and
                        macd_lows[-1][1] > macd_lows[-2][1]):
                    divergences.append({
                        "type": "bullish_divergence",
                        "indicator": "MACD",
                        "signal": "bullish_reversal",
                        "description": "Price lower low + MACD higher low.",
                    })

            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                if (price_highs[-1][1] > price_highs[-2][1] and
                        macd_highs[-1][1] < macd_highs[-2][1]):
                    divergences.append({
                        "type": "bearish_divergence",
                        "indicator": "MACD",
                        "signal": "bearish_reversal",
                        "description": "Price higher high + MACD lower high.",
                    })

        return divergences

    def get_full_pattern_analysis(self, df: pd.DataFrame) -> dict:
        """Run pattern analysis on a DataFrame.

        REMOVED: Fibonacci levels — subjective, only useful on daily+ timeframes.
        KEPT: candlestick patterns (simplified) + divergences (leading signals).
        """
        return {
            "candlestick_patterns": self.detect_patterns(df),
            "divergences": self.detect_divergences(df),
        }
