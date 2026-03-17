"""Technical analysis module using ta library."""

import pandas as pd
import ta
from loguru import logger


class TechnicalAnalyzer:
    """Computes technical indicators and generates signals."""

    # Minimum candles required for reliable indicator computation.
    # Largest window is EMA-200, but ta library crashes below ~14 for ATR/ADX.
    MIN_CANDLES = 50

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to OHLCV DataFrame.

        Requires at least MIN_CANDLES rows. With fewer rows some indicators
        will be NaN but the function won't crash.
        """
        df = df.copy()

        if len(df) < self.MIN_CANDLES:
            logger.warning(f"compute_indicators: только {len(df)} свечей (минимум {self.MIN_CANDLES}), индикаторы будут неполными")
            # Return empty indicator columns to avoid downstream crashes
            for col in [
                "ema_9", "ema_21", "ema_50", "ema_200", "sma_50", "sma_200",
                "macd", "macd_signal", "macd_histogram", "rsi", "rsi_6",
                "stoch_rsi_k", "stoch_rsi_d",
                "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
                "atr", "adx", "adx_pos", "adx_neg",
                "obv", "vwap", "volume_sma_20", "volume_ratio",
                "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conv",
                "pivot", "support_1", "resistance_1",
                "vpoc", "vah", "val",
            ]:
                df[col] = float("nan")
            return df

        # Trend indicators
        df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
        df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
        df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # RSI
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["rsi_6"] = ta.momentum.rsi(df["close"], window=6)

        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(df["close"])
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        # ATR (volatility)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

        # ADX (trend strength)
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # Volume indicators
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df["high"], df["low"])
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()

        # Support/Resistance levels
        df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
        df["support_1"] = 2 * df["pivot"] - df["high"]
        df["resistance_1"] = 2 * df["pivot"] - df["low"]

        # Volume Profile / VPOC
        vpoc_data = self._compute_volume_profile(df)
        df["vpoc"] = vpoc_data["vpoc"]
        df["vah"] = vpoc_data["vah"]
        df["val"] = vpoc_data["val"]

        return df

    @staticmethod
    def _compute_volume_profile(df: pd.DataFrame, num_bins: int = 30) -> dict:
        """Compute Volume Profile and VPOC (Volume Point of Control).

        VPOC = price level with the highest traded volume
        VAH = Value Area High (upper 70% of volume)
        VAL = Value Area Low (lower 70% of volume)

        These act as strong support/resistance levels.
        """
        if len(df) < 20:
            return {"vpoc": float("nan"), "vah": float("nan"), "val": float("nan")}

        price_min = df["low"].min()
        price_max = df["high"].max()

        if price_max == price_min:
            return {"vpoc": float(price_min), "vah": float(price_max), "val": float(price_min)}

        # Create price bins
        bin_edges = pd.np_compat = None  # avoid stale import
        import numpy as np
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Assign volume to bins based on where price traded
        vol_profile = np.zeros(num_bins)
        for _, row in df.iterrows():
            # Distribute each candle's volume across its range
            candle_low = row["low"]
            candle_high = row["high"]
            candle_vol = row["volume"]
            for i in range(num_bins):
                if bins[i + 1] >= candle_low and bins[i] <= candle_high:
                    vol_profile[i] += candle_vol

        # VPOC = bin with max volume
        vpoc_idx = np.argmax(vol_profile)
        vpoc = float(bin_centers[vpoc_idx])

        # Value Area: 70% of total volume around VPOC
        total_vol = vol_profile.sum()
        target_vol = total_vol * 0.70
        accumulated = vol_profile[vpoc_idx]
        low_idx = vpoc_idx
        high_idx = vpoc_idx

        while accumulated < target_vol and (low_idx > 0 or high_idx < num_bins - 1):
            expand_low = vol_profile[low_idx - 1] if low_idx > 0 else 0
            expand_high = vol_profile[high_idx + 1] if high_idx < num_bins - 1 else 0
            if expand_low >= expand_high and low_idx > 0:
                low_idx -= 1
                accumulated += expand_low
            elif high_idx < num_bins - 1:
                high_idx += 1
                accumulated += expand_high
            else:
                low_idx -= 1
                accumulated += expand_low

        val = float(bin_centers[low_idx])
        vah = float(bin_centers[high_idx])

        return {"vpoc": round(vpoc, 6), "vah": round(vah, 6), "val": round(val, 6)}

    def generate_summary(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate a human-readable analysis summary for Claude."""
        if df.empty or len(df) < 50:
            return {"symbol": symbol, "error": "Insufficient data"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        summary = {
            "symbol": symbol,
            "price": round(float(latest["close"]), 6),
            "change_1h": round(float((latest["close"] - prev["close"]) / prev["close"] * 100), 2),

            # Trend
            "ema_9": round(float(latest["ema_9"]), 6),
            "ema_21": round(float(latest["ema_21"]), 6),
            "ema_50": round(float(latest["ema_50"]), 6),
            "ema_200": round(float(latest.get("ema_200", 0)), 6),
            "trend_short": "bullish" if latest["ema_9"] > latest["ema_21"] else "bearish",
            "trend_medium": "bullish" if latest["ema_21"] > latest["ema_50"] else "bearish",
            "golden_cross": bool(latest["sma_50"] > latest["sma_200"]),

            # Momentum
            "rsi": round(float(latest["rsi"]), 2),
            "rsi_zone": "overbought" if latest["rsi"] > 70 else ("oversold" if latest["rsi"] < 30 else "neutral"),
            "macd": round(float(latest["macd"]), 6),
            "macd_signal": round(float(latest["macd_signal"]), 6),
            "macd_histogram": round(float(latest["macd_histogram"]), 6),
            "macd_crossover": "bullish" if latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"] else (
                "bearish" if latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"] else "none"
            ),
            "stoch_rsi_k": round(float(latest["stoch_rsi_k"]), 2),
            "stoch_rsi_d": round(float(latest["stoch_rsi_d"]), 2),

            # Volatility
            "bb_upper": round(float(latest["bb_upper"]), 6),
            "bb_lower": round(float(latest["bb_lower"]), 6),
            "bb_position": round(float(latest["bb_pct"]), 2),
            "atr": round(float(latest["atr"]), 6),
            "atr_pct": round(float(latest["atr"] / latest["close"] * 100), 2),

            # Trend strength
            "adx": round(float(latest["adx"]), 2),
            "trend_strength": "strong" if latest["adx"] > 25 else "weak",

            # Volume
            "volume": float(latest["volume"]),
            "volume_ratio": round(float(latest["volume_ratio"]), 2),
            "obv_rising": bool(latest["obv"] > prev["obv"]),

            # Support/Resistance
            "support_1": round(float(latest["support_1"]), 6),
            "resistance_1": round(float(latest["resistance_1"]), 6),
            "vwap": round(float(latest["vwap"]), 6),
            "price_vs_vwap": "above" if latest["close"] > latest["vwap"] else "below",

            # Ichimoku
            "above_cloud": bool(latest["close"] > max(latest["ichimoku_a"], latest["ichimoku_b"])),

            # Volume Profile
            "vpoc": round(float(latest["vpoc"]), 6) if not pd.isna(latest.get("vpoc", float("nan"))) else None,
            "value_area_high": round(float(latest["vah"]), 6) if not pd.isna(latest.get("vah", float("nan"))) else None,
            "value_area_low": round(float(latest["val"]), 6) if not pd.isna(latest.get("val", float("nan"))) else None,
            "price_vs_vpoc": (
                "above" if not pd.isna(latest.get("vpoc", float("nan"))) and latest["close"] > latest["vpoc"]
                else "below" if not pd.isna(latest.get("vpoc", float("nan"))) else None
            ),
        }

        logger.debug(f"Analysis summary for {symbol}: RSI={summary['rsi']}, trend={summary['trend_short']}")
        return summary

    def multi_timeframe_analysis(self, ohlcv_dict: dict[str, pd.DataFrame], symbol: str) -> dict:
        """Analyze multiple timeframes and combine signals."""
        results = {}
        for tf, df in ohlcv_dict.items():
            df_with_indicators = self.compute_indicators(df)
            results[tf] = self.generate_summary(df_with_indicators, symbol)
        return results
