"""Scalping / short-term microstructure analysis module.

Indicators specifically designed for 1m-15m timeframe trading:
- Order flow imbalance (bid/ask volume proxy)
- Price momentum micro-bursts
- Volatility micro-regimes
- Spread/liquidity estimation
- Tick intensity & volume spikes
"""

import numpy as np
import pandas as pd
from loguru import logger


class ScalpingAnalyzer:
    """Microstructure and short-term momentum analysis for scalping."""

    def full_scalping_analysis(self, df: pd.DataFrame) -> dict:
        """Run all scalping-specific analyses."""
        if len(df) < 20:
            return {"error": "insufficient_data"}

        return {
            "order_flow": self.order_flow_imbalance(df),
            "micro_momentum": self.micro_momentum(df),
            "volume_profile_short": self.short_volume_profile(df),
            "volatility_regime": self.volatility_micro_regime(df),
            "price_action": self.price_action_signals(df),
            "spread_estimate": self.spread_estimation(df),
            "cvd": self.cumulative_volume_delta(df),
            "breakout": self.breakout_quality(df),
            "scalp_signal": self._aggregate_scalp_signal(df),
        }

    @staticmethod
    def order_flow_imbalance(df: pd.DataFrame) -> dict:
        """Estimate order flow imbalance from OHLCV data.

        Uses the tick rule: if close > open, volume is "buy volume".
        Aggregates buy/sell volume over recent candles to detect
        aggressive buying or selling pressure.
        """
        if len(df) < 10:
            return {"imbalance": 0, "signal": "neutral"}

        recent = df.tail(10)
        buy_vol = 0.0
        sell_vol = 0.0

        for _, row in recent.iterrows():
            body = row["close"] - row["open"]
            total_range = row["high"] - row["low"]
            if total_range == 0:
                continue
            # Proportion of volume attributed to buying
            buy_ratio = (row["close"] - row["low"]) / total_range
            sell_ratio = (row["high"] - row["close"]) / total_range
            buy_vol += row["volume"] * buy_ratio
            sell_vol += row["volume"] * sell_ratio

        total = buy_vol + sell_vol
        if total == 0:
            return {"imbalance": 0, "signal": "neutral"}

        imbalance = (buy_vol - sell_vol) / total  # -1 to +1

        if imbalance > 0.3:
            signal = "strong_buy_pressure"
        elif imbalance > 0.15:
            signal = "moderate_buy_pressure"
        elif imbalance < -0.3:
            signal = "strong_sell_pressure"
        elif imbalance < -0.15:
            signal = "moderate_sell_pressure"
        else:
            signal = "balanced"

        return {
            "imbalance": round(imbalance, 3),
            "buy_volume_pct": round(buy_vol / total * 100, 1),
            "sell_volume_pct": round(sell_vol / total * 100, 1),
            "signal": signal,
        }

    @staticmethod
    def micro_momentum(df: pd.DataFrame) -> dict:
        """Detect micro-momentum bursts over very short windows.

        Looks for acceleration in price movement — not just trend,
        but increasing speed of movement (2nd derivative of price).
        """
        if len(df) < 15:
            return {"momentum": 0, "acceleration": 0, "signal": "neutral"}

        close = df["close"].values

        # Rate of change over different micro windows
        roc_3 = (close[-1] - close[-4]) / close[-4] * 100 if close[-4] != 0 else 0
        roc_5 = (close[-1] - close[-6]) / close[-6] * 100 if close[-6] != 0 else 0
        roc_10 = (close[-1] - close[-11]) / close[-11] * 100 if close[-11] != 0 else 0

        # Acceleration: is momentum increasing or decreasing?
        prev_roc_3 = (close[-2] - close[-5]) / close[-5] * 100 if close[-5] != 0 else 0
        acceleration = roc_3 - prev_roc_3

        # Volume-weighted momentum
        vol = df["volume"].values[-5:]
        avg_vol = np.mean(df["volume"].values[-20:])
        vol_factor = np.mean(vol) / avg_vol if avg_vol > 0 else 1.0

        if roc_3 > 0.15 and acceleration > 0 and vol_factor > 1.2:
            signal = "strong_bullish_burst"
        elif roc_3 > 0.08 and acceleration > 0:
            signal = "bullish_momentum"
        elif roc_3 < -0.15 and acceleration < 0 and vol_factor > 1.2:
            signal = "strong_bearish_burst"
        elif roc_3 < -0.08 and acceleration < 0:
            signal = "bearish_momentum"
        elif abs(roc_3) < 0.03 and abs(acceleration) < 0.01:
            signal = "consolidation"
        else:
            signal = "neutral"

        return {
            "roc_3": round(roc_3, 4),
            "roc_5": round(roc_5, 4),
            "roc_10": round(roc_10, 4),
            "acceleration": round(acceleration, 4),
            "volume_factor": round(vol_factor, 2),
            "signal": signal,
        }

    @staticmethod
    def short_volume_profile(df: pd.DataFrame, window: int = 30) -> dict:
        """Short-term volume profile — where is volume concentrated recently.

        For scalping, we care about the last 30 candles, not 200.
        High-volume price levels act as micro support/resistance.
        """
        recent = df.tail(window)
        if len(recent) < 10:
            return {"vpoc": 0, "signal": "insufficient_data"}

        price_min = recent["low"].min()
        price_max = recent["high"].max()
        if price_max == price_min:
            return {"vpoc": float(price_min), "signal": "flat"}

        num_bins = 15
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        vol_profile = np.zeros(num_bins)

        for _, row in recent.iterrows():
            for i in range(num_bins):
                if bins[i + 1] >= row["low"] and bins[i] <= row["high"]:
                    vol_profile[i] += row["volume"]

        vpoc_idx = np.argmax(vol_profile)
        vpoc = float(bin_centers[vpoc_idx])
        current_price = float(df["close"].iloc[-1])

        # High volume node (HVN) and low volume node (LVN)
        avg_vol = np.mean(vol_profile)
        hvn_levels = [float(bin_centers[i]) for i in range(num_bins) if vol_profile[i] > avg_vol * 1.5]
        lvn_levels = [float(bin_centers[i]) for i in range(num_bins) if 0 < vol_profile[i] < avg_vol * 0.5]

        distance_to_vpoc_pct = (current_price - vpoc) / vpoc * 100 if vpoc > 0 else 0

        return {
            "micro_vpoc": round(vpoc, 6),
            "distance_to_vpoc_pct": round(distance_to_vpoc_pct, 3),
            "high_volume_nodes": [round(x, 6) for x in hvn_levels[:3]],
            "low_volume_nodes": [round(x, 6) for x in lvn_levels[:3]],
            "signal": "at_vpoc" if abs(distance_to_vpoc_pct) < 0.1 else (
                "above_vpoc" if distance_to_vpoc_pct > 0 else "below_vpoc"
            ),
        }

    @staticmethod
    def volatility_micro_regime(df: pd.DataFrame) -> dict:
        """Detect micro-volatility regime for position sizing.

        Uses ATR ratio and Bollinger Band width over short windows
        to classify current volatility state.
        """
        if len(df) < 30:
            return {"regime": "unknown", "signal": "insufficient_data"}

        # Short ATR (5-period) vs medium ATR (20-period)
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ], axis=1).max(axis=1)

        atr_5 = tr.rolling(5).mean().iloc[-1]
        atr_20 = tr.rolling(20).mean().iloc[-1]

        if atr_20 == 0:
            return {"regime": "unknown", "signal": "no_data"}

        atr_ratio = atr_5 / atr_20

        # Bollinger Band width (short)
        bb_mean = df["close"].rolling(10).mean()
        bb_std = df["close"].rolling(10).std()
        bb_width = (2 * bb_std / bb_mean * 100).iloc[-1] if bb_mean.iloc[-1] > 0 else 0

        # Recent range as % of price
        recent_range_pct = (df["high"].tail(5).max() - df["low"].tail(5).min()) / df["close"].iloc[-1] * 100

        if atr_ratio > 1.5:
            regime = "expanding"
            hint = "volatility expanding rapidly — tighten stops, reduce size"
        elif atr_ratio < 0.6:
            regime = "contracting"
            hint = "volatility compressing — breakout imminent, prepare entries"
        elif atr_ratio > 1.1:
            regime = "elevated"
            hint = "above-average volatility — use caution"
        else:
            regime = "normal"
            hint = "stable volatility — normal position sizing"

        return {
            "regime": regime,
            "atr_ratio": round(float(atr_ratio), 3),
            "atr_5": round(float(atr_5), 6),
            "atr_20": round(float(atr_20), 6),
            "bb_width_pct": round(float(bb_width), 3),
            "recent_range_pct": round(float(recent_range_pct), 3),
            "hint": hint,
        }

    @staticmethod
    def price_action_signals(df: pd.DataFrame) -> dict:
        """Detect short-term price action setups.

        - Pin bars (rejection wicks)
        - Inside bars (consolidation before breakout)
        - Engulfing patterns
        - Three white soldiers / three black crows (micro)
        """
        if len(df) < 5:
            return {"patterns": [], "signal": "insufficient_data"}

        patterns = []
        c = df.iloc[-1]
        p = df.iloc[-2]
        pp = df.iloc[-3]

        body_c = abs(c["close"] - c["open"])
        range_c = c["high"] - c["low"]
        body_p = abs(p["close"] - p["open"])
        range_p = p["high"] - p["low"]

        # Pin bar (hammer / shooting star)
        if range_c > 0:
            upper_wick = c["high"] - max(c["close"], c["open"])
            lower_wick = min(c["close"], c["open"]) - c["low"]

            if lower_wick > body_c * 2 and upper_wick < body_c * 0.5:
                patterns.append({"type": "bullish_pin_bar", "strength": round(lower_wick / range_c, 2)})
            if upper_wick > body_c * 2 and lower_wick < body_c * 0.5:
                patterns.append({"type": "bearish_pin_bar", "strength": round(upper_wick / range_c, 2)})

        # Inside bar
        if c["high"] <= p["high"] and c["low"] >= p["low"]:
            patterns.append({"type": "inside_bar", "strength": 0.5})

        # Bullish engulfing
        if (p["close"] < p["open"] and c["close"] > c["open"] and
                c["close"] > p["open"] and c["open"] < p["close"]):
            patterns.append({"type": "bullish_engulfing", "strength": round(body_c / body_p if body_p > 0 else 1, 2)})

        # Bearish engulfing
        if (p["close"] > p["open"] and c["close"] < c["open"] and
                c["close"] < p["open"] and c["open"] > p["close"]):
            patterns.append({"type": "bearish_engulfing", "strength": round(body_c / body_p if body_p > 0 else 1, 2)})

        # Three white soldiers (micro)
        if len(df) >= 4:
            ppp = df.iloc[-4]
            if (ppp["close"] > ppp["open"] and pp["close"] > pp["open"] and
                    p["close"] > p["open"] and c["close"] > c["open"]):
                patterns.append({"type": "four_white_soldiers", "strength": 0.8})
            if (ppp["close"] < ppp["open"] and pp["close"] < pp["open"] and
                    p["close"] < p["open"] and c["close"] < c["open"]):
                patterns.append({"type": "four_black_crows", "strength": 0.8})

        signal = "neutral"
        if patterns:
            bullish = [p for p in patterns if "bullish" in p["type"] or "white" in p["type"]]
            bearish = [p for p in patterns if "bearish" in p["type"] or "black" in p["type"]]
            if bullish and not bearish:
                signal = "bullish_patterns"
            elif bearish and not bullish:
                signal = "bearish_patterns"
            elif bullish and bearish:
                signal = "mixed_patterns"

        return {"patterns": patterns, "signal": signal}

    @staticmethod
    def spread_estimation(df: pd.DataFrame) -> dict:
        """Estimate bid-ask spread from OHLCV data.

        Uses the Corwin-Schultz (2012) estimator: derives spread
        from high-low prices. Useful when order book data is unavailable.

        Higher spread = lower liquidity = worse fills for scalping.
        """
        if len(df) < 5:
            return {"spread_pct": 0, "signal": "insufficient_data"}

        spreads = []
        for i in range(1, min(len(df), 20)):
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            h_prev = df["high"].iloc[i - 1]
            l_prev = df["low"].iloc[i - 1]

            # Corwin-Schultz formula (simplified)
            beta = (np.log(h / l) ** 2 + np.log(h_prev / l_prev) ** 2)
            gamma = np.log(max(h, h_prev) / min(l, l_prev)) ** 2

            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
            alpha = max(alpha, 0)  # Can't be negative

            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            spreads.append(spread * 100)  # As percentage

        avg_spread = float(np.mean(spreads)) if spreads else 0

        if avg_spread > 0.2:
            signal = "wide_spread"
            hint = "high spread — poor fill quality, avoid scalping"
        elif avg_spread > 0.08:
            signal = "moderate_spread"
            hint = "acceptable spread for scalping"
        else:
            signal = "tight_spread"
            hint = "tight spread — good conditions for scalping"

        return {
            "spread_pct": round(avg_spread, 4),
            "signal": signal,
            "hint": hint,
        }

    @staticmethod
    def cumulative_volume_delta(df: pd.DataFrame) -> dict:
        """Cumulative Volume Delta (CVD) — accumulated buy minus sell volume.

        Detects divergences between price and volume:
        - Price rising + CVD falling = distribution (fake rally, expect reversal)
        - Price falling + CVD rising = accumulation (fake dump, expect bounce)
        - Price + CVD aligned = trend confirmed
        """
        if len(df) < 20:
            return {"cvd": 0, "divergence": "insufficient_data", "signal": "neutral"}

        # Estimate buy/sell volume from candle structure
        deltas = []
        for _, row in df.iterrows():
            total_range = row["high"] - row["low"]
            if total_range == 0:
                deltas.append(0)
                continue
            buy_ratio = (row["close"] - row["low"]) / total_range
            sell_ratio = (row["high"] - row["close"]) / total_range
            delta = row["volume"] * (buy_ratio - sell_ratio)
            deltas.append(delta)

        cvd_series = np.cumsum(deltas)

        # Recent CVD trend (last 10 vs previous 10)
        cvd_recent = float(np.mean(cvd_series[-5:]))
        cvd_prev = float(np.mean(cvd_series[-15:-5])) if len(cvd_series) >= 15 else float(np.mean(cvd_series[:5]))
        cvd_trend = cvd_recent - cvd_prev

        # Price trend
        price_recent = float(df["close"].iloc[-1])
        price_prev = float(df["close"].iloc[-10]) if len(df) >= 10 else float(df["close"].iloc[0])
        price_change_pct = (price_recent - price_prev) / price_prev * 100

        # Divergence detection
        if price_change_pct > 0.1 and cvd_trend < 0:
            divergence = "bearish_divergence"
            signal = "distribution"
            hint = "Price up but CVD down — sellers hiding, expect reversal"
        elif price_change_pct < -0.1 and cvd_trend > 0:
            divergence = "bullish_divergence"
            signal = "accumulation"
            hint = "Price down but CVD up — buyers accumulating, expect bounce"
        elif price_change_pct > 0.1 and cvd_trend > 0:
            divergence = "none"
            signal = "bullish_confirmed"
            hint = "Price and CVD aligned upward — strong move"
        elif price_change_pct < -0.1 and cvd_trend < 0:
            divergence = "none"
            signal = "bearish_confirmed"
            hint = "Price and CVD aligned downward — strong move"
        else:
            divergence = "none"
            signal = "neutral"
            hint = "No clear divergence"

        return {
            "cvd": round(float(cvd_series[-1]), 2),
            "cvd_trend": round(cvd_trend, 2),
            "price_change_pct": round(price_change_pct, 3),
            "divergence": divergence,
            "signal": signal,
            "hint": hint,
        }

    @staticmethod
    def multi_timeframe_trend_score(ohlcv_dict: dict[str, pd.DataFrame]) -> dict:
        """Aggregate trend score across multiple timeframes (-1.0 to +1.0).

        Checks EMA alignment, momentum direction, and volume on each timeframe.
        |score| > 0.7 = strong setup, < 0.3 = no clear direction.
        """
        scores = {}
        weights = {"1m": 0.2, "5m": 0.35, "15m": 0.45}  # Higher TF = more weight

        for tf, df in ohlcv_dict.items():
            if len(df) < 21:
                continue

            close = df["close"]
            score = 0.0

            # EMA alignment
            ema_8 = close.ewm(span=8).mean().iloc[-1]
            ema_21 = close.ewm(span=21).mean().iloc[-1]
            current = close.iloc[-1]

            if current > ema_8 > ema_21:
                score += 1.0  # Bullish alignment
            elif current < ema_8 < ema_21:
                score -= 1.0  # Bearish alignment
            elif current > ema_21:
                score += 0.3
            elif current < ema_21:
                score -= 0.3

            # Momentum (ROC-5)
            roc = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) >= 6 else 0
            if roc > 0.2:
                score += 0.5
            elif roc < -0.2:
                score -= 0.5

            # Volume trend (recent vs average)
            if len(df) >= 20:
                vol_recent = df["volume"].iloc[-5:].mean()
                vol_avg = df["volume"].iloc[-20:].mean()
                if vol_avg > 0 and vol_recent / vol_avg > 1.3:
                    # High volume confirms the direction
                    score *= 1.2

            # Clamp to [-1.5, 1.5] before weighting
            score = max(-1.5, min(1.5, score))
            scores[tf] = round(score, 3)

        if not scores:
            return {"score": 0, "signal": "insufficient_data", "timeframe_scores": {}}

        # Weighted aggregate
        total_weight = 0
        weighted_sum = 0
        for tf, sc in scores.items():
            w = weights.get(tf, 0.3)
            weighted_sum += sc * w
            total_weight += w

        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        final_score = max(-1.0, min(1.0, final_score))

        # Agreement check
        signs = [1 if s > 0.2 else (-1 if s < -0.2 else 0) for s in scores.values()]
        agreement = len(set(signs)) == 1 and signs[0] != 0

        if final_score > 0.7:
            signal = "strong_bullish"
        elif final_score > 0.3:
            signal = "moderate_bullish"
        elif final_score < -0.7:
            signal = "strong_bearish"
        elif final_score < -0.3:
            signal = "moderate_bearish"
        else:
            signal = "neutral"

        return {
            "score": round(final_score, 3),
            "signal": signal,
            "agreement": agreement,
            "timeframe_scores": scores,
        }

    @staticmethod
    def breakout_quality(df: pd.DataFrame, lookback: int = 20) -> dict:
        """Evaluate quality of a potential breakout.

        Filters fake breakouts by checking:
        1. Volume at breakout > 1.5x average
        2. Candle body > 60% of total range (strong close, not wicks)
        3. Price actually breaks recent high/low

        Returns quality score and whether breakout is confirmed.
        """
        if len(df) < lookback + 1:
            return {"breakout": "none", "quality": 0, "signal": "insufficient_data"}

        recent = df.iloc[-lookback - 1:-1]  # Previous candles (not current)
        current = df.iloc[-1]

        recent_high = recent["high"].max()
        recent_low = recent["low"].min()

        # Check if current candle breaks out
        breaks_high = current["close"] > recent_high
        breaks_low = current["close"] < recent_low

        if not breaks_high and not breaks_low:
            return {"breakout": "none", "quality": 0, "signal": "no_breakout"}

        direction = "bullish" if breaks_high else "bearish"

        # Quality checks
        quality_score = 0

        # 1. Volume confirmation
        avg_vol = recent["volume"].mean()
        vol_ratio = current["volume"] / avg_vol if avg_vol > 0 else 0
        if vol_ratio > 2.0:
            quality_score += 3
        elif vol_ratio > 1.5:
            quality_score += 2
        elif vol_ratio > 1.0:
            quality_score += 1

        # 2. Body ratio (strong close)
        body = abs(current["close"] - current["open"])
        total_range = current["high"] - current["low"]
        body_ratio = body / total_range if total_range > 0 else 0
        if body_ratio > 0.7:
            quality_score += 2
        elif body_ratio > 0.5:
            quality_score += 1

        # 3. Close position (close near high for bullish, near low for bearish)
        if total_range > 0:
            if direction == "bullish":
                close_position = (current["close"] - current["low"]) / total_range
            else:
                close_position = (current["high"] - current["close"]) / total_range
            if close_position > 0.7:
                quality_score += 1

        # Classify
        if quality_score >= 5:
            quality_label = "high"
            signal = f"confirmed_{direction}_breakout"
        elif quality_score >= 3:
            quality_label = "moderate"
            signal = f"probable_{direction}_breakout"
        else:
            quality_label = "low"
            signal = f"weak_{direction}_breakout"

        return {
            "breakout": direction,
            "quality": quality_score,
            "quality_label": quality_label,
            "volume_ratio": round(vol_ratio, 2),
            "body_ratio": round(body_ratio, 3),
            "signal": signal,
        }

    def _aggregate_scalp_signal(self, df: pd.DataFrame) -> dict:
        """Combine all microstructure signals into a single scalp verdict."""
        of = self.order_flow_imbalance(df)
        mm = self.micro_momentum(df)
        vr = self.volatility_micro_regime(df)
        pa = self.price_action_signals(df)
        sp = self.spread_estimation(df)

        score = 0  # -5 to +5 scale (negative = bearish, positive = bullish)

        # Order flow
        imb = of.get("imbalance", 0)
        if imb > 0.2:
            score += 1
        elif imb < -0.2:
            score -= 1

        # Momentum
        mm_sig = mm.get("signal", "")
        if "bullish" in mm_sig:
            score += 2 if "strong" in mm_sig else 1
        elif "bearish" in mm_sig:
            score -= 2 if "strong" in mm_sig else 1

        # Price action
        pa_sig = pa.get("signal", "")
        if pa_sig == "bullish_patterns":
            score += 1
        elif pa_sig == "bearish_patterns":
            score -= 1

        # Penalty for bad conditions
        if sp.get("signal") == "wide_spread":
            quality = "poor"
        elif vr.get("regime") == "expanding":
            quality = "risky"
        else:
            quality = "good"

        if score >= 3:
            verdict = "strong_buy"
        elif score >= 1:
            verdict = "buy"
        elif score <= -3:
            verdict = "strong_sell"
        elif score <= -1:
            verdict = "sell"
        else:
            verdict = "neutral"

        return {
            "score": score,
            "verdict": verdict,
            "quality": quality,
        }
