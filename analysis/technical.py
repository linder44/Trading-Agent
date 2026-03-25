"""Technical analysis module using ta library."""

import numpy as np
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

        # Trend: fast EMAs for scalping (3/5/8/13/21) — REMOVED slow SMA 50/200
        df["ema_3"] = ta.trend.ema_indicator(df["close"], window=3)
        df["ema_5"] = ta.trend.ema_indicator(df["close"], window=5)
        df["ema_8"] = ta.trend.ema_indicator(df["close"], window=8)
        df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
        df["ema_13"] = ta.trend.ema_indicator(df["close"], window=13)
        df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
        df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)

        # Fast MACD for scalping (5, 13, 4) — REMOVED standard MACD (too slow)
        macd_fast = ta.trend.MACD(df["close"], window_slow=13, window_fast=5, window_sign=4)
        df["macd_fast"] = macd_fast.macd()
        df["macd_fast_signal"] = macd_fast.macd_signal()
        df["macd_fast_histogram"] = macd_fast.macd_diff()
        # Keep standard MACD for multi-timeframe (5m/15m) only
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # RSI: fast for scalping (3/6/14)
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["rsi_6"] = ta.momentum.rsi(df["close"], window=6)
        df["rsi_3"] = ta.momentum.rsi(df["close"], window=3)

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

        # REMOVED: Ichimoku Cloud — too lagging for 1m/5m scalping
        # REMOVED: SMA 50/200 — daily/weekly indicators, useless for scalping

        # Support/Resistance levels
        df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
        df["support_1"] = 2 * df["pivot"] - df["high"]
        df["resistance_1"] = 2 * df["pivot"] - df["low"]

        # Volume Profile / VPOC
        vpoc_data = self._compute_volume_profile(df)
        df["vpoc"] = vpoc_data["vpoc"]
        df["vah"] = vpoc_data["vah"]
        df["val"] = vpoc_data["val"]

        # RVOL — Relative Volume (current vs average same period)
        df["rvol"] = df["volume"] / df["volume_sma_20"]

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

    @staticmethod
    def compute_support_resistance(df: pd.DataFrame, lookbacks: tuple = (20, 50, 100)) -> dict:
        """Detect support/resistance from swing highs and lows.

        Uses multiple lookback windows to find levels at different scales.
        Returns nearest levels above and below current price, plus key zones.
        """
        if len(df) < max(lookbacks):
            return {"supports": [], "resistances": [], "key_levels": []}

        price = float(df["close"].iloc[-1])
        all_levels: list[dict] = []

        for lb in lookbacks:
            highs = df["high"].rolling(window=lb, center=True).max()
            lows = df["low"].rolling(window=lb, center=True).min()

            # Swing highs: where high equals rolling max
            swing_high_mask = df["high"] == highs
            for idx in df.index[swing_high_mask]:
                level = float(df.loc[idx, "high"])
                all_levels.append({"price": level, "type": "resistance", "lookback": lb})

            # Swing lows: where low equals rolling min
            swing_low_mask = df["low"] == lows
            for idx in df.index[swing_low_mask]:
                level = float(df.loc[idx, "low"])
                all_levels.append({"price": level, "type": "support", "lookback": lb})

        # Cluster nearby levels (within 0.3% of each other)
        if not all_levels:
            return {"supports": [], "resistances": [], "key_levels": []}

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
        # Last cluster
        avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
        clusters.append({
            "price": round(avg_price, 6),
            "strength": len(current_cluster),
            "type": "support" if avg_price < price else "resistance",
        })

        supports = sorted([c for c in clusters if c["price"] < price], key=lambda x: -x["price"])[:5]
        resistances = sorted([c for c in clusters if c["price"] >= price], key=lambda x: x["price"])[:5]
        key_levels = sorted(clusters, key=lambda x: -x["strength"])[:8]

        return {
            "supports": supports,
            "resistances": resistances,
            "key_levels": key_levels,
        }

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
        """Detect order blocks (institutional buy/sell zones).

        An order block is the last opposite candle before a strong impulsive move.
        - Bullish OB: last bearish candle before a strong bullish move (buy zone)
        - Bearish OB: last bullish candle before a strong bearish move (sell zone)
        """
        if len(df) < lookback + 3:
            return []

        blocks = []
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else None
        if not atr:
            atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])

        recent = df.iloc[-lookback:]
        threshold = atr * 2  # Strong move = 2x ATR

        for i in range(1, len(recent) - 2):
            curr = recent.iloc[i]
            nxt = recent.iloc[i + 1]
            move = nxt["close"] - curr["close"]

            # Bullish OB: bearish candle followed by strong bullish move
            if curr["close"] < curr["open"] and move > threshold:
                blocks.append({
                    "type": "bullish_ob",
                    "zone_high": round(float(curr["open"]), 6),
                    "zone_low": round(float(curr["close"]), 6),
                    "strength": round(float(move / atr), 1),
                })

            # Bearish OB: bullish candle followed by strong bearish move
            elif curr["close"] > curr["open"] and move < -threshold:
                blocks.append({
                    "type": "bearish_ob",
                    "zone_high": round(float(curr["close"]), 6),
                    "zone_low": round(float(curr["open"]), 6),
                    "strength": round(float(abs(move) / atr), 1),
                })

        # Keep only the most recent and strongest blocks
        bullish = sorted([b for b in blocks if b["type"] == "bullish_ob"], key=lambda x: -x["strength"])[:3]
        bearish = sorted([b for b in blocks if b["type"] == "bearish_ob"], key=lambda x: -x["strength"])[:3]
        return bullish + bearish

    @staticmethod
    def detect_liquidity_zones(df: pd.DataFrame, lookback: int = 100) -> list[dict]:
        """Detect liquidity clusters — areas where many stop losses likely sit.

        Liquidity pools form:
        - Below equal lows (stop loss hunt zone for longs)
        - Above equal highs (stop loss hunt zone for shorts)
        - Below/above consolidation ranges (breakout liquidity)
        """
        if len(df) < lookback:
            return []

        recent = df.iloc[-lookback:]
        price = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
        tolerance = atr * 0.3
        zones = []

        # Equal lows: liquidity sitting below
        lows = recent["low"].values
        for i in range(len(lows)):
            equal_count = 0
            for j in range(i + 1, len(lows)):
                if abs(lows[j] - lows[i]) < tolerance:
                    equal_count += 1
            if equal_count >= 2:
                level = float(lows[i])
                zones.append({
                    "type": "buy_liquidity",
                    "price": round(level, 6),
                    "touches": equal_count + 1,
                    "side": "below" if level < price else "above",
                })

        # Equal highs: liquidity sitting above
        highs = recent["high"].values
        for i in range(len(highs)):
            equal_count = 0
            for j in range(i + 1, len(highs)):
                if abs(highs[j] - highs[i]) < tolerance:
                    equal_count += 1
            if equal_count >= 2:
                level = float(highs[i])
                zones.append({
                    "type": "sell_liquidity",
                    "price": round(level, 6),
                    "touches": equal_count + 1,
                    "side": "above" if level > price else "below",
                })

        # Deduplicate nearby zones
        zones.sort(key=lambda x: x["price"])
        deduped = []
        for z in zones:
            if not deduped or abs(z["price"] - deduped[-1]["price"]) > tolerance:
                deduped.append(z)
            elif z["touches"] > deduped[-1]["touches"]:
                deduped[-1] = z

        # Return closest zones to current price
        return sorted(deduped, key=lambda x: abs(x["price"] - price))[:8]

    def generate_summary(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate a human-readable analysis summary for Claude."""
        if df.empty or len(df) < 50:
            return {"symbol": symbol, "error": "Insufficient data"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        summary = {
            "symbol": symbol,
            "price": round(float(latest["close"]), 6),
            "change_1_candle": round(float((latest["close"] - prev["close"]) / prev["close"] * 100), 4),

            # Scalping EMA alignment (3/8/21) — Tier 2
            "scalp_trend": "bullish" if latest.get("ema_3", 0) > latest.get("ema_8", 0) > latest.get("ema_21", 0) else (
                "bearish" if latest.get("ema_3", 0) < latest.get("ema_8", 0) < latest.get("ema_21", 0) else "mixed"
            ),
            "trend_short": "bullish" if latest["ema_9"] > latest["ema_21"] else "bearish",

            # Fast MACD crossover (5/13/4) — Tier 2
            "macd_fast_histogram": round(float(latest.get("macd_fast_histogram", 0)), 6),
            "macd_fast_crossover": "bullish" if latest.get("macd_fast", 0) > latest.get("macd_fast_signal", 0) and prev.get("macd_fast", 0) <= prev.get("macd_fast_signal", 0) else (
                "bearish" if latest.get("macd_fast", 0) < latest.get("macd_fast_signal", 0) and prev.get("macd_fast", 0) >= prev.get("macd_fast_signal", 0) else "none"
            ),

            # RSI-3 for extremes — Tier 2
            "rsi_3": round(float(latest.get("rsi_3", 50)), 2),
            "rsi": round(float(latest["rsi"]), 2),

            # Volatility — Tier 3 (for sizing, not direction)
            "bb_position": round(float(latest["bb_pct"]), 2),
            "bb_width": round(float(latest["bb_width"]), 4) if not pd.isna(latest.get("bb_width")) else 0,
            "atr": round(float(latest["atr"]), 6),
            "atr_pct": round(float(latest["atr"] / latest["close"] * 100), 2),

            # ADX trend strength — Tier 2
            "adx": round(float(latest["adx"]), 2),

            # Volume — Tier 2
            "volume_ratio": round(float(latest["volume_ratio"]), 2) if not pd.isna(latest.get("volume_ratio")) else 1.0,
            "rvol": round(float(latest.get("rvol", 1.0)), 2) if not pd.isna(latest.get("rvol")) else 1.0,

            # Key S/R levels
            "support_1": round(float(latest["support_1"]), 6),
            "resistance_1": round(float(latest["resistance_1"]), 6),
            "vwap": round(float(latest["vwap"]), 6),

            # VPOC
            "vpoc": round(float(latest["vpoc"]), 6) if not pd.isna(latest.get("vpoc", float("nan"))) else None,

            # REMOVED: Ichimoku, SMA 50/200, golden_cross — useless for scalping
            # REMOVED: detailed order blocks, liquidity zones — too much noise
        }

        logger.debug(f"Analysis summary for {symbol}: RSI={summary['rsi']}, trend={summary['trend_short']}")
        return summary

    @staticmethod
    def _count_consecutive_candles(df: pd.DataFrame) -> dict:
        """Count consecutive bullish/bearish candles from the most recent."""
        if len(df) < 2:
            return {"direction": "none", "count": 0}
        direction = "bullish" if df["close"].iloc[-1] > df["open"].iloc[-1] else "bearish"
        count = 0
        for i in range(len(df) - 1, -1, -1):
            is_bull = df["close"].iloc[i] > df["open"].iloc[i]
            if (direction == "bullish" and is_bull) or (direction == "bearish" and not is_bull):
                count += 1
            else:
                break
        return {"direction": direction, "count": count}

    def multi_timeframe_analysis(self, ohlcv_dict: dict[str, pd.DataFrame], symbol: str) -> dict:
        """Analyze multiple timeframes and combine signals."""
        results = {}
        for tf, df in ohlcv_dict.items():
            df_with_indicators = self.compute_indicators(df)
            results[tf] = self.generate_summary(df_with_indicators, symbol)
        return results
