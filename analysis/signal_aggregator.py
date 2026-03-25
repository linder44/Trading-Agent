"""Signal Aggregator — weighted tier-based signal system.

Replaces the flat list of 18+ indicators with a hierarchical system:
- Tier 1 (weight 3x): Order Flow, CVD divergence, Spread — decisive signals
- Tier 2 (weight 2x): Fast EMA, Micro Momentum, Order Book, Whale bias
- Tier 3 (weight 1x): ATR regime, Funding, Liquidations, Session context

If Tier 1 signals conflict → HOLD (no entry).
Output: aggregated score + top reasons for/against, passed to Claude instead of raw data.
"""

from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Signal:
    """A single signal from one indicator."""
    source: str
    tier: int  # 1, 2, or 3
    direction: float  # -1.0 (short) to +1.0 (long), 0 = neutral
    strength: float  # 0.0 to 1.0
    reason: str = ""


TIER_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.0}


class SignalAggregator:
    """Aggregates signals from multiple sources into a weighted verdict."""

    def aggregate(
        self,
        technical_data: dict,
        scalping_data: dict,
        onchain_data: dict | None = None,
        liquidation_data: dict | None = None,
        time_context_data: dict | None = None,
        symbol: str = "",
    ) -> dict:
        """Collect all signals, compute weighted score, check Tier 1 conflicts.

        Returns dict with:
          - verdict: "long" / "short" / "hold"
          - confidence: 0.0-1.0
          - weighted_score: raw score
          - tier1_conflict: bool
          - top_reasons_for / top_reasons_against: list[str]
          - signals: list of signal dicts (for logging)
        """
        signals = self._collect_signals(technical_data, scalping_data, onchain_data,
                                         liquidation_data, time_context_data, symbol)

        # Check Tier 1 conflicts
        tier1 = [s for s in signals if s.tier == 1]
        tier1_dirs = [s.direction for s in tier1 if abs(s.direction) > 0.1]
        tier1_conflict = False
        if tier1_dirs:
            has_long = any(d > 0.1 for d in tier1_dirs)
            has_short = any(d < -0.1 for d in tier1_dirs)
            if has_long and has_short:
                tier1_conflict = True

        # Compute weighted score
        weighted_sum = 0.0
        weight_total = 0.0
        for s in signals:
            w = TIER_WEIGHTS.get(s.tier, 1.0)
            weighted_sum += s.direction * s.strength * w
            weight_total += s.strength * w

        if weight_total > 0:
            normalized_score = weighted_sum / weight_total  # -1 to +1
        else:
            normalized_score = 0.0

        # Determine verdict
        if tier1_conflict:
            verdict = "hold"
            confidence = 0.0
        elif normalized_score > 0.15:
            verdict = "long"
            confidence = min(abs(normalized_score), 1.0)
        elif normalized_score < -0.15:
            verdict = "short"
            confidence = min(abs(normalized_score), 1.0)
        else:
            verdict = "hold"
            confidence = 0.0

        # Top reasons
        reasons_for = sorted(
            [s for s in signals if (s.direction > 0 and verdict == "long") or (s.direction < 0 and verdict == "short")],
            key=lambda s: abs(s.direction * s.strength * TIER_WEIGHTS.get(s.tier, 1)),
            reverse=True,
        )[:3]
        reasons_against = sorted(
            [s for s in signals if (s.direction < 0 and verdict == "long") or (s.direction > 0 and verdict == "short")],
            key=lambda s: abs(s.direction * s.strength * TIER_WEIGHTS.get(s.tier, 1)),
            reverse=True,
        )[:3]

        return {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "weighted_score": round(normalized_score, 4),
            "tier1_conflict": tier1_conflict,
            "top_reasons_for": [s.reason for s in reasons_for],
            "top_reasons_against": [s.reason for s in reasons_against],
            "signal_count": len(signals),
            "signals": [
                {"source": s.source, "tier": s.tier, "direction": round(s.direction, 3),
                 "strength": round(s.strength, 3), "reason": s.reason}
                for s in signals
            ],
        }

    def _collect_signals(
        self,
        technical_data: dict,
        scalping_data: dict,
        onchain_data: dict | None,
        liquidation_data: dict | None,
        time_context_data: dict | None,
        symbol: str,
    ) -> list[Signal]:
        """Extract signals from all data sources."""
        signals: list[Signal] = []

        # ── TIER 1: Decisive ──────────────────────────────────
        # 1a. Order Flow Imbalance
        of = scalping_data.get("order_flow", {})
        imb = of.get("imbalance", 0)
        if abs(imb) > 0.05:
            signals.append(Signal(
                source="order_flow", tier=1,
                direction=min(max(imb * 2, -1), 1),  # scale to [-1, 1]
                strength=min(abs(imb) * 2, 1.0),
                reason=f"Order flow imbalance={imb:.3f} ({of.get('signal', '')})",
            ))

        # 1b. CVD / Divergence (from scalp_signal or micro_momentum)
        mm = scalping_data.get("micro_momentum", {})
        accel = mm.get("acceleration", 0)
        roc3 = mm.get("roc_3", 0)
        vol_factor = mm.get("volume_factor", 1.0)
        if abs(roc3) > 0.05:
            direction = 1.0 if roc3 > 0 else -1.0
            strength = min(abs(roc3) * 3, 1.0)
            # Volume confirmation boosts strength
            if vol_factor > 1.2:
                strength = min(strength * 1.3, 1.0)
            signals.append(Signal(
                source="cvd_momentum", tier=1,
                direction=direction * strength,
                strength=strength,
                reason=f"ROC3={roc3:.4f}, accel={accel:.4f}, vol_factor={vol_factor:.2f}",
            ))

        # 1c. Spread (wide = block entry)
        sp = scalping_data.get("spread_estimate", {})
        spread_pct = sp.get("spread_pct", 0)
        if spread_pct > 0.15:
            signals.append(Signal(
                source="spread", tier=1,
                direction=0.0,  # neutral but blocking
                strength=1.0,
                reason=f"Spread={spread_pct:.4f}% — too wide for scalping",
            ))

        # ── TIER 2: Confirming ────────────────────────────────
        # 2a. Fast EMA alignment (3/8/21)
        tf_1m = technical_data.get("1m", {})
        scalp_trend = tf_1m.get("scalp_trend", "mixed")
        if scalp_trend == "bullish":
            signals.append(Signal(
                source="ema_alignment", tier=2,
                direction=0.7, strength=0.7,
                reason=f"EMA 3>8>21 bullish alignment",
            ))
        elif scalp_trend == "bearish":
            signals.append(Signal(
                source="ema_alignment", tier=2,
                direction=-0.7, strength=0.7,
                reason=f"EMA 3<8<21 bearish alignment",
            ))

        # 2b. Micro Momentum signal
        mm_sig = mm.get("signal", "neutral")
        if "strong_bullish" in mm_sig:
            signals.append(Signal(
                source="micro_momentum", tier=2, direction=0.9, strength=0.85,
                reason=f"Strong bullish momentum burst",
            ))
        elif "bullish" in mm_sig:
            signals.append(Signal(
                source="micro_momentum", tier=2, direction=0.5, strength=0.6,
                reason=f"Bullish momentum",
            ))
        elif "strong_bearish" in mm_sig:
            signals.append(Signal(
                source="micro_momentum", tier=2, direction=-0.9, strength=0.85,
                reason=f"Strong bearish momentum burst",
            ))
        elif "bearish" in mm_sig:
            signals.append(Signal(
                source="micro_momentum", tier=2, direction=-0.5, strength=0.6,
                reason=f"Bearish momentum",
            ))

        # 2c. Price action patterns
        pa = scalping_data.get("price_action", {})
        pa_sig = pa.get("signal", "neutral")
        if pa_sig == "bullish_patterns":
            patterns = [p["type"] for p in pa.get("patterns", [])]
            signals.append(Signal(
                source="price_action", tier=2, direction=0.6, strength=0.5,
                reason=f"Bullish patterns: {', '.join(patterns)}",
            ))
        elif pa_sig == "bearish_patterns":
            patterns = [p["type"] for p in pa.get("patterns", [])]
            signals.append(Signal(
                source="price_action", tier=2, direction=-0.6, strength=0.5,
                reason=f"Bearish patterns: {', '.join(patterns)}",
            ))

        # 2d. RSI-3 extremes
        rsi3 = tf_1m.get("rsi_3", 50)
        if rsi3 < 15:
            signals.append(Signal(
                source="rsi3", tier=2, direction=0.8, strength=0.7,
                reason=f"RSI-3={rsi3:.1f} extreme oversold",
            ))
        elif rsi3 > 85:
            signals.append(Signal(
                source="rsi3", tier=2, direction=-0.8, strength=0.7,
                reason=f"RSI-3={rsi3:.1f} extreme overbought",
            ))
        elif rsi3 < 25:
            signals.append(Signal(
                source="rsi3", tier=2, direction=0.4, strength=0.4,
                reason=f"RSI-3={rsi3:.1f} oversold",
            ))
        elif rsi3 > 75:
            signals.append(Signal(
                source="rsi3", tier=2, direction=-0.4, strength=0.4,
                reason=f"RSI-3={rsi3:.1f} overbought",
            ))

        # 2e. Fast MACD crossover
        macd_cross = tf_1m.get("macd_fast_crossover", "none")
        if macd_cross == "bullish":
            signals.append(Signal(
                source="macd_fast", tier=2, direction=0.6, strength=0.5,
                reason="Fast MACD bullish crossover",
            ))
        elif macd_cross == "bearish":
            signals.append(Signal(
                source="macd_fast", tier=2, direction=-0.6, strength=0.5,
                reason="Fast MACD bearish crossover",
            ))

        # 2f. Multi-timeframe alignment
        tf_5m = technical_data.get("5m", {})
        tf_15m = technical_data.get("15m", {})
        trend_1m = tf_1m.get("trend_short", "")
        trend_5m = tf_5m.get("trend_short", "")
        trend_15m = tf_15m.get("trend_short", "")
        if trend_1m == trend_5m == trend_15m == "bullish":
            signals.append(Signal(
                source="mtf_alignment", tier=2, direction=0.8, strength=0.8,
                reason="All timeframes (1m/5m/15m) bullish aligned",
            ))
        elif trend_1m == trend_5m == trend_15m == "bearish":
            signals.append(Signal(
                source="mtf_alignment", tier=2, direction=-0.8, strength=0.8,
                reason="All timeframes (1m/5m/15m) bearish aligned",
            ))

        # 2g. ADX trend strength
        adx = tf_1m.get("adx", 20)
        if adx > 30:
            signals.append(Signal(
                source="adx", tier=2, direction=0.0, strength=0.5,
                reason=f"ADX={adx:.1f} strong trend",
            ))

        # 2h. Bollinger Band position
        bb_pct = tf_1m.get("bb_position", 0.5)
        if bb_pct > 1.0:
            signals.append(Signal(
                source="bb_position", tier=2, direction=-0.4, strength=0.4,
                reason=f"Price above upper BB (pct={bb_pct:.2f})",
            ))
        elif bb_pct < 0.0:
            signals.append(Signal(
                source="bb_position", tier=2, direction=0.4, strength=0.4,
                reason=f"Price below lower BB (pct={bb_pct:.2f})",
            ))

        # ── TIER 3: Context ──────────────────────────────────
        # 3a. Volatility regime
        vr = scalping_data.get("volatility_regime", {})
        regime = vr.get("regime", "normal")
        if regime == "expanding":
            signals.append(Signal(
                source="volatility", tier=3, direction=0.0, strength=0.3,
                reason="Volatility expanding — tighten stops",
            ))
        elif regime == "contracting":
            signals.append(Signal(
                source="volatility", tier=3, direction=0.0, strength=0.3,
                reason="Volatility contracting — breakout imminent",
            ))

        # 3b. Funding rate (contrarian at extremes)
        if onchain_data and symbol in onchain_data:
            funding = onchain_data[symbol].get("funding_rate", {})
            rate = funding.get("funding_rate", 0)
            if rate > 0.001:
                signals.append(Signal(
                    source="funding", tier=3, direction=-0.4, strength=0.5,
                    reason=f"Extreme positive funding={rate:.6f} — contrarian short signal",
                ))
            elif rate < -0.001:
                signals.append(Signal(
                    source="funding", tier=3, direction=0.4, strength=0.5,
                    reason=f"Extreme negative funding={rate:.6f} — contrarian long signal",
                ))

        # 3c. Liquidation stress
        if liquidation_data and symbol in liquidation_data:
            liq = liquidation_data[symbol]
            stress = liq.get("stress_level", "low")
            if stress in ("extreme", "high"):
                signals.append(Signal(
                    source="liquidations", tier=3, direction=0.0, strength=0.6,
                    reason=f"Liquidation stress={stress} — avoid entry",
                ))

        # 3d. Volume profile position
        vp = scalping_data.get("volume_profile_short", {})
        vp_signal = vp.get("signal", "")
        dist = vp.get("distance_to_vpoc_pct", 0)
        if abs(dist) > 0.3:
            direction = -0.3 if dist > 0 else 0.3  # mean reversion toward VPOC
            signals.append(Signal(
                source="volume_profile", tier=3,
                direction=direction, strength=0.3,
                reason=f"Price {dist:+.3f}% from micro VPOC",
            ))

        return signals

    def get_rule_based_decision(
        self,
        technical_data: dict,
        scalping_data: dict,
        onchain_data: dict | None = None,
        liquidation_data: dict | None = None,
        time_context_data: dict | None = None,
        symbol: str = "",
    ) -> dict:
        """Pure rule-based decision without Claude — used for backtesting and fallback.

        Returns same format as Claude decisions.
        """
        result = self.aggregate(technical_data, scalping_data, onchain_data,
                                liquidation_data, time_context_data, symbol)

        verdict = result["verdict"]
        confidence = result["confidence"]

        # Spread block
        sp = scalping_data.get("spread_estimate", {})
        if sp.get("signal") == "wide_spread":
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": 0.0,
                "reason": "Wide spread — skip",
                "params": {},
            }

        if verdict == "hold" or confidence < 0.3:
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": confidence,
                "reason": f"Score={result['weighted_score']:.4f}, conflict={result['tier1_conflict']}",
                "params": {},
            }

        action = "open_long" if verdict == "long" else "open_short"
        reasons = result["top_reasons_for"]
        reason_str = "; ".join(reasons[:3]) if reasons else "aggregated signals"

        return {
            "symbol": symbol,
            "action": action,
            "confidence": min(confidence, 1.0),
            "reason": reason_str,
            "params": {},
        }
