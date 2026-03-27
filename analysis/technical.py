"""Technical analysis module — MACD + EMA 200 + S/R strategy.

Only computes what the strategy needs:
- EMA 200 (trend filter)
- MACD(12, 26, 9) with previous values (crossover detection)
- ATR (stop loss sizing)
- Support/Resistance levels (entry confirmation)
"""

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalAnalyzer:
    """Computes technical indicators for MACD + EMA200 + SR strategy."""

    MIN_CANDLES = 50

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strategy indicators to OHLCV DataFrame."""
        df = df.copy()

        if len(df) < self.MIN_CANDLES:
            logger.warning(f"compute_indicators: only {len(df)} candles (min {self.MIN_CANDLES})")
            for col in ["ema_200", "macd", "macd_signal", "macd_histogram", "atr"]:
                df[col] = float("nan")
            return df

        # EMA 200 — trend filter
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        # MACD (12, 26, 9) — standard
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ATR (14) — for SL sizing
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()

        return df

    @staticmethod
    def compute_support_resistance(df: pd.DataFrame, lookbacks: tuple = (20, 50, 100)) -> dict:
        """Detect support/resistance from swing highs and lows."""
        if len(df) < max(lookbacks):
            return {"supports": [], "resistances": []}

        price = float(df["close"].iloc[-1])
        all_levels: list[dict] = []

        for lb in lookbacks:
            highs = df["high"].rolling(window=lb, center=True).max()
            lows = df["low"].rolling(window=lb, center=True).min()

            swing_high_mask = df["high"] == highs
            for idx in df.index[swing_high_mask]:
                level = float(df.loc[idx, "high"])
                all_levels.append({"price": level, "type": "resistance", "lookback": lb})

            swing_low_mask = df["low"] == lows
            for idx in df.index[swing_low_mask]:
                level = float(df.loc[idx, "low"])
                all_levels.append({"price": level, "type": "support", "lookback": lb})

        if not all_levels:
            return {"supports": [], "resistances": []}

        # Cluster nearby levels (within 0.3% of each other)
        all_levels.sort(key=lambda x: x["price"])
        clusters: list[dict] = []
        cluster_threshold = price * 0.003

        current_cluster = [all_levels[0]]
        for level in all_levels[1:]:
            if level["price"] - current_cluster[-1]["price"] < cluster_threshold:
                current_cluster.append(level)
            else:
                avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                clusters.append({
                    "price": round(avg_price, 6),
                    "strength": len(current_cluster),
                    "type": "support" if avg_price < price else "resistance",
                })
                current_cluster = [level]

        avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
        clusters.append({
            "price": round(avg_price, 6),
            "strength": len(current_cluster),
            "type": "support" if avg_price < price else "resistance",
        })

        supports = sorted([c for c in clusters if c["price"] < price], key=lambda x: -x["price"])[:5]
        resistances = sorted([c for c in clusters if c["price"] >= price], key=lambda x: x["price"])[:5]

        return {
            "supports": supports,
            "resistances": resistances,
        }

    def generate_summary(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate analysis summary for the engine."""
        if df.empty or len(df) < self.MIN_CANDLES:
            return {"symbol": symbol, "error": "Insufficient data"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # S/R levels
        sr = self.compute_support_resistance(df)

        summary = {
            "symbol": symbol,
            "price": round(float(latest["close"]), 6),

            # EMA 200
            "ema_200": round(float(latest["ema_200"]), 6) if not pd.isna(latest.get("ema_200", float("nan"))) else 0,

            # MACD with previous values for crossover detection
            "macd": {
                "macd_line": round(float(latest["macd"]), 8) if not pd.isna(latest["macd"]) else 0,
                "signal_line": round(float(latest["macd_signal"]), 8) if not pd.isna(latest["macd_signal"]) else 0,
                "histogram": round(float(latest["macd_histogram"]), 8) if not pd.isna(latest["macd_histogram"]) else 0,
                "prev_macd_line": round(float(prev["macd"]), 8) if not pd.isna(prev["macd"]) else 0,
                "prev_signal_line": round(float(prev["macd_signal"]), 8) if not pd.isna(prev["macd_signal"]) else 0,
            },

            # ATR
            "atr": round(float(latest["atr"]), 6) if not pd.isna(latest.get("atr", float("nan"))) else 0,

            # S/R levels
            "support_levels": sr["supports"],
            "resistance_levels": sr["resistances"],
        }

        logger.debug(f"{symbol}: price={summary['price']}, EMA200={summary['ema_200']}, "
                    f"MACD={summary['macd']['macd_line']:.6f}, Signal={summary['macd']['signal_line']:.6f}")
        return summary

    def multi_timeframe_analysis(self, ohlcv_dict: dict[str, pd.DataFrame], symbol: str) -> dict:
        """Analyze multiple timeframes and combine signals."""
        results = {}
        for tf, df in ohlcv_dict.items():
            df_with_indicators = self.compute_indicators(df)
            results[tf] = self.generate_summary(df_with_indicators, symbol)
        return results
