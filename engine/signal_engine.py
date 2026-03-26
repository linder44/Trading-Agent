"""
Deterministic Signal Engine — replaces Claude API.

Same inputs, same output format, zero cost, zero latency.
Cascade of filters: each can REJECT but not APPROVE alone.
Entry requires ALL gates passed + minimum score.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

from engine.config import ENGINE_CONFIG


# ═══════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════

class Action(Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    UPDATE_SL = "update_sl"
    HOLD = "hold"


class TradeType(Enum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"


@dataclass
class Decision:
    action: Action
    symbol: str
    confidence: float
    trade_type: Optional[TradeType] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    tp2: Optional[float] = None
    trailing_stop: Optional[float] = None
    position_size_pct: float = 5.0
    reason: str = ""
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to the same dict format brain.py returned."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "params": {
                "trigger_price": None,
                "new_stop_loss": self.stop_loss,
            },
        }


# ═══════════════════════════════════════════════════════════
# GateKeeper — fast reject filters
# ═══════════════════════════════════════════════════════════

class GateKeeper:
    """Mandatory pre-entry checks. Any failure = HOLD."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._symbol_losses: dict[str, list[float]] = {}  # symbol -> list of loss timestamps

    def check_all(self, symbol: str, scalping: dict, technical_1m: dict,
                  regime: dict, positions: dict, risk_level: str) -> tuple[bool, str]:
        checks = [
            self._check_spread(scalping),
            self._check_volume(technical_1m),
            self._check_regime(regime),
            self._check_max_positions(positions),
            self._check_symbol_cooldown(symbol),
        ]
        for passed, reason in checks:
            if not passed:
                return False, reason
        return True, "all gates passed"

    def record_loss(self, symbol: str):
        """Record a loss for symbol cooldown tracking."""
        self._symbol_losses.setdefault(symbol, [])
        self._symbol_losses[symbol].append(time.time())
        # Keep only last hour
        cutoff = time.time() - 3600
        self._symbol_losses[symbol] = [t for t in self._symbol_losses[symbol] if t > cutoff]

    def _check_spread(self, scalping: dict) -> tuple[bool, str]:
        spread = scalping.get("spread_estimate", {})
        spread_pct = spread.get("spread_pct", 0)
        if spread_pct > self.cfg["max_spread_pct"]:
            return False, f"spread too wide: {spread_pct:.3f}%"
        return True, "spread ok"

    def _check_volume(self, technical_1m: dict) -> tuple[bool, str]:
        rvol = technical_1m.get("rvol", technical_1m.get("volume_ratio", 1.0))
        if rvol < self.cfg["min_volume_ratio"]:
            return False, f"volume too low: rvol={rvol:.2f}"
        return True, "volume ok"

    def _check_regime(self, regime: dict) -> tuple[bool, str]:
        if regime.get("regime") == "choppy":
            return False, "choppy market regime"
        return True, f"regime ok: {regime.get('regime', 'unknown')}"

    def _check_max_positions(self, positions: dict) -> tuple[bool, str]:
        if len(positions) >= self.cfg["max_positions"]:
            return False, f"max positions reached: {len(positions)}"
        return True, "positions ok"

    def _check_symbol_cooldown(self, symbol: str) -> tuple[bool, str]:
        losses = self._symbol_losses.get(symbol, [])
        cooldown = self.cfg["symbol_cooldown_minutes"] * 60
        recent = [t for t in losses if time.time() - t < cooldown]
        if len(recent) >= 2:
            return False, f"symbol cooldown: {len(recent)} recent losses"
        return True, "no cooldown"


# ═══════════════════════════════════════════════════════════
# SignalScorer — numerical scoring replaces Claude "intuition"
# ═══════════════════════════════════════════════════════════

class SignalScorer:
    """
    Computes entry score from -10 to +10.
    Positive = long, negative = short.
    |score| < 3.0 = no trade.
    """

    def __init__(self, cfg: dict):
        self.T1 = cfg["tier1_weight"]
        self.T2 = cfg["tier2_weight"]
        self.T3 = cfg["tier3_weight"]

    def score(self, scalping: dict, technical_1m: dict, technical_5m: dict,
              tape: dict, vwap: dict, delta: dict, onchain: dict,
              regime: dict) -> dict:
        signals = {}
        total = 0.0

        # ══ TIER 1 (weight 3x) ══
        of = self._score_order_flow(scalping)
        signals["order_flow"] = of
        total += of["score"] * self.T1

        cvd = self._score_cvd(delta)
        signals["cvd"] = cvd
        total += cvd["score"] * self.T1

        mom = self._score_momentum(scalping)
        signals["momentum"] = mom
        total += mom["score"] * self.T1

        # Tier 1 conflict check
        dirs = [s["direction"] for s in [of, cvd, mom] if s["direction"] != "neutral"]
        if len(set(dirs)) > 1:
            return {
                "direction": "neutral", "score": 0, "confidence": 0,
                "trade_type": None, "signals": signals,
                "conflicts": ["tier1_conflict"],
            }

        # ══ TIER 2 (weight 2x) ══
        ema = self._score_ema(technical_1m)
        signals["ema"] = ema
        total += ema["score"] * self.T2

        tape_s = self._score_tape(tape)
        signals["tape"] = tape_s
        total += tape_s["score"] * self.T2

        pa = self._score_price_action(scalping)
        signals["price_action"] = pa
        total += pa["score"] * self.T2

        vwap_s = self._score_vwap(vwap)
        signals["vwap"] = vwap_s
        total += vwap_s["score"] * self.T2

        # ══ TIER 3 (weight 1x) ══
        funding = self._score_funding(onchain)
        signals["funding"] = funding
        total += funding["score"] * self.T3

        regime_s = self._score_regime_boost(regime)
        signals["regime"] = regime_s
        total += regime_s["score"] * self.T3

        # Scalp aggregate as tiebreaker
        scalp_agg = self._score_scalp_aggregate(scalping)
        signals["scalp_aggregate"] = scalp_agg
        total += scalp_agg["score"] * self.T3

        # ══ Result ══
        direction = "long" if total > 0 else "short" if total < 0 else "neutral"
        abs_score = abs(total)
        confidence = min(abs_score / 10.0, 1.0)

        trade_type = self._determine_trade_type(signals, regime)

        return {
            "direction": direction,
            "score": abs_score,
            "confidence": confidence,
            "trade_type": trade_type,
            "signals": signals,
            "conflicts": [],
        }

    # ── Tier 1 scorers ──

    def _score_order_flow(self, scalping: dict) -> dict:
        of = scalping.get("order_flow", {})
        value = of.get("imbalance", 0)
        if abs(value) < 0.3:
            return {"score": 0, "direction": "neutral", "detail": f"imbalance={value:.2f}"}
        score = value * min(abs(value) * 1.5, 1.0)
        return {"score": score, "direction": "long" if value > 0 else "short",
                "detail": f"imbalance={value:.2f}"}

    def _score_cvd(self, delta: dict) -> dict:
        verdict = delta.get("delta_verdict", {}).get("signal", "neutral")
        stacked = delta.get("stacked_delta", {})
        exhaustion = delta.get("exhaustion", {}).get("signal", "none")

        if exhaustion == "bullish_exhaustion":
            return {"score": -0.7, "direction": "short", "detail": "bullish exhaustion (reversal short)"}
        if exhaustion == "bearish_exhaustion":
            return {"score": 0.7, "direction": "long", "detail": "bearish exhaustion (reversal long)"}

        if verdict == "strong_bullish":
            return {"score": 0.6, "direction": "long", "detail": "strong bullish delta"}
        if verdict == "strong_bearish":
            return {"score": -0.6, "direction": "short", "detail": "strong bearish delta"}

        stacked_sig = stacked.get("signal", "neutral")
        if "buying" in stacked_sig:
            return {"score": 0.3, "direction": "long", "detail": stacked_sig}
        if "selling" in stacked_sig:
            return {"score": -0.3, "direction": "short", "detail": stacked_sig}

        return {"score": 0, "direction": "neutral", "detail": "delta neutral"}

    def _score_momentum(self, scalping: dict) -> dict:
        mm = scalping.get("micro_momentum", {})
        signal = mm.get("signal", "neutral")
        vol_factor = mm.get("volume_factor", 1.0)

        if signal == "strong_bullish_burst" and vol_factor > 1.2:
            return {"score": 0.9, "direction": "long", "detail": f"bull burst vol={vol_factor:.1f}x"}
        if signal == "strong_bearish_burst" and vol_factor > 1.2:
            return {"score": -0.9, "direction": "short", "detail": f"bear burst vol={vol_factor:.1f}x"}
        if signal in ("bullish_burst", "bullish_momentum"):
            return {"score": 0.4, "direction": "long", "detail": signal}
        if signal in ("bearish_burst", "bearish_momentum"):
            return {"score": -0.4, "direction": "short", "detail": signal}
        return {"score": 0, "direction": "neutral", "detail": "no momentum"}

    # ── Tier 2 scorers ──

    def _score_ema(self, tech: dict) -> dict:
        trend = tech.get("scalp_trend", "mixed")
        if trend == "bullish":
            return {"score": 0.6, "direction": "long", "detail": "EMA bullish"}
        if trend == "bearish":
            return {"score": -0.6, "direction": "short", "detail": "EMA bearish"}
        return {"score": 0, "direction": "neutral", "detail": "EMA mixed"}

    def _score_tape(self, tape: dict) -> dict:
        verdict = tape.get("tape_verdict", {})
        sig = verdict.get("signal", "neutral")
        score_val = verdict.get("score", 0)

        if sig == "dead_market":
            return {"score": 0, "direction": "neutral", "detail": "dead market"}
        if sig in ("strong_bullish", "bullish"):
            return {"score": min(score_val * 0.25, 0.7), "direction": "long", "detail": sig}
        if sig in ("strong_bearish", "bearish"):
            return {"score": max(score_val * 0.25, -0.7), "direction": "short", "detail": sig}
        return {"score": 0, "direction": "neutral", "detail": sig}

    def _score_price_action(self, scalping: dict) -> dict:
        pa = scalping.get("price_action", {})
        patterns = pa.get("patterns", [])
        score = 0.0
        details = []
        for p in patterns:
            ptype = p.get("type", "")
            strength = p.get("strength", 0.5)
            if "bullish" in ptype:
                score += strength * 0.5
                details.append(ptype)
            elif "bearish" in ptype:
                score -= strength * 0.5
                details.append(ptype)
        score = max(min(score, 1.0), -1.0)
        direction = "long" if score > 0 else "short" if score < 0 else "neutral"
        return {"score": score, "direction": direction, "detail": ", ".join(details[:3]) or "none"}

    def _score_vwap(self, vwap: dict) -> dict:
        sig = vwap.get("signal", "neutral")
        dev = vwap.get("deviation_sigma", 0)

        if sig == "breakout_long":
            return {"score": 0.6, "direction": "long", "detail": f"VWAP breakout long {dev:.1f}σ"}
        if sig == "breakout_short":
            return {"score": -0.6, "direction": "short", "detail": f"VWAP breakout short {dev:.1f}σ"}
        if sig == "mean_revert_long":
            return {"score": 0.4, "direction": "long", "detail": f"VWAP revert long {dev:.1f}σ"}
        if sig == "mean_revert_short":
            return {"score": -0.4, "direction": "short", "detail": f"VWAP revert short {dev:.1f}σ"}
        return {"score": 0, "direction": "neutral", "detail": sig}

    # ── Tier 3 scorers ──

    def _score_funding(self, onchain: dict) -> dict:
        funding = onchain.get("funding_rate", {})
        rate = funding.get("funding_rate", 0)
        if rate > 0.01:
            return {"score": -0.3, "direction": "short", "detail": f"high funding={rate:.4f}"}
        if rate < -0.01:
            return {"score": 0.3, "direction": "long", "detail": f"neg funding={rate:.4f}"}
        return {"score": 0, "direction": "neutral", "detail": f"funding={rate:.4f}"}

    def _score_regime_boost(self, regime: dict) -> dict:
        r = regime.get("regime", "normal")
        if r == "strong_trend":
            return {"score": 0.3, "direction": "neutral", "detail": "trend boost"}
        if r == "squeeze":
            return {"score": 0.2, "direction": "neutral", "detail": "squeeze boost"}
        if r == "fading_trend":
            return {"score": -0.2, "direction": "neutral", "detail": "fading penalty"}
        return {"score": 0, "direction": "neutral", "detail": r}

    def _score_scalp_aggregate(self, scalping: dict) -> dict:
        agg = scalping.get("scalp_signal", {})
        score_val = agg.get("score", 0)  # -5 to +5
        normalized = score_val / 5.0  # -1 to +1
        direction = "long" if normalized > 0 else "short" if normalized < 0 else "neutral"
        return {"score": normalized * 0.5, "direction": direction,
                "detail": f"scalp_agg={score_val}"}

    def _determine_trade_type(self, signals: dict, regime: dict) -> TradeType:
        r = regime.get("regime", "normal")
        if r == "squeeze":
            return TradeType.BREAKOUT

        cvd = signals.get("cvd", {})
        if "exhaustion" in cvd.get("detail", ""):
            return TradeType.REVERSAL

        return TradeType.MOMENTUM


# ═══════════════════════════════════════════════════════════
# ExitManager — SL/TP based on trade type + regime
# ═══════════════════════════════════════════════════════════

class ExitManager:
    """Calculates SL/TP based on trade type and regime."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def calculate(self, entry_price: float, direction: str,
                  trade_type: TradeType, regime: str, atr: float) -> dict | None:
        if atr is None or atr <= 0:
            atr = entry_price * 0.003  # fallback 0.3%

        if trade_type == TradeType.MOMENTUM:
            if regime == "strong_trend":
                sl_dist, tp1_dist, tp2_dist, trailing = 1.0 * atr, 1.5 * atr, 4.0 * atr, 0.4 * atr
            else:
                sl_dist, tp1_dist, tp2_dist, trailing = 1.2 * atr, 1.5 * atr, 3.0 * atr, 0.3 * atr
        elif trade_type == TradeType.REVERSAL:
            sl_dist, tp1_dist, tp2_dist, trailing = 1.5 * atr, 1.0 * atr, 2.0 * atr, None
        elif trade_type == TradeType.BREAKOUT:
            sl_dist, tp1_dist, tp2_dist, trailing = 1.5 * atr, 2.0 * atr, 5.0 * atr, 0.5 * atr
        else:
            sl_dist, tp1_dist, tp2_dist, trailing = 1.2 * atr, 1.5 * atr, 3.0 * atr, 0.3 * atr

        if direction == "long":
            sl = entry_price - sl_dist
            tp1 = entry_price + tp1_dist
            tp2 = entry_price + tp2_dist
        else:
            sl = entry_price + sl_dist
            tp1 = entry_price - tp1_dist
            tp2 = entry_price - tp2_dist

        risk = abs(entry_price - sl)
        reward = abs(tp1 - entry_price)
        if risk == 0 or reward / risk < 1.2:
            return None

        return {
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "trailing_stop": trailing,
            "risk_reward": round(reward / risk, 2),
        }


# ═══════════════════════════════════════════════════════════
# PositionManager — manage open positions each cycle
# ═══════════════════════════════════════════════════════════

class PositionManager:
    """Evaluates open positions for exit/update."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate(self, symbol: str, position: dict,
                 technical_1m: dict, regime: dict) -> Decision | None:
        entry_price = position.get("entry_price", 0)
        direction = position.get("side", position.get("direction", "long"))
        current_price = technical_1m.get("price", entry_price)
        age_minutes = position.get("age_minutes", 0)

        if entry_price <= 0:
            return None

        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # 1. Regime changed to choppy
        if regime.get("regime") == "choppy":
            return Decision(Action.CLOSE, symbol, 0.9, reason=f"regime choppy, pnl={pnl_pct:.2f}%")

        # 2. Max time
        if age_minutes > self.cfg["max_trade_minutes"]:
            return Decision(Action.CLOSE, symbol, 0.9, reason=f"max time {age_minutes:.0f}min")

        # 3. Dead trade
        if abs(pnl_pct) < 0.1 and age_minutes > self.cfg["dead_trade_minutes"]:
            return Decision(Action.CLOSE, symbol, 0.8, reason=f"dead trade {age_minutes:.0f}min")

        # 4. Losing + trend against
        if pnl_pct < -0.2 and age_minutes > self.cfg["losing_trade_minutes"]:
            trend = technical_1m.get("scalp_trend", "mixed")
            our_direction = (direction == "long" and trend == "bullish") or \
                           (direction == "short" and trend == "bearish")
            if not our_direction:
                return Decision(Action.CLOSE, symbol, 0.85,
                              reason=f"losing+trend against: pnl={pnl_pct:.2f}%")

        # 5. Move SL to breakeven
        if pnl_pct > self.cfg["breakeven_pnl_pct"] and not position.get("sl_at_breakeven"):
            commission = entry_price * 0.0006
            new_sl = entry_price + commission if direction == "long" else entry_price - commission
            return Decision(Action.UPDATE_SL, symbol, 0.9, stop_loss=new_sl,
                          reason=f"SL to breakeven, pnl={pnl_pct:.2f}%",
                          signals={"sl_at_breakeven": True})

        return None


# ═══════════════════════════════════════════════════════════
# RiskThrottle — graduated circuit breaker
# ═══════════════════════════════════════════════════════════

class RiskThrottle:
    """Graduated risk reduction on losses."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def check(self, daily_pnl: float, trade_history: list) -> str:
        if daily_pnl <= self.cfg["daily_loss_stop"]:
            return "stop"
        if daily_pnl <= self.cfg["daily_loss_critical"]:
            return "critical"
        if daily_pnl <= self.cfg["daily_loss_reduced"]:
            return "reduced"

        recent = trade_history[-5:] if len(trade_history) >= 5 else trade_history
        consecutive_losses = 0
        for trade in reversed(recent):
            if trade.get("pnl", trade.get("pnl_pct", 0)) < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 4:
            return "critical"
        if consecutive_losses >= self.cfg["max_consecutive_losses"]:
            return "reduced"

        return "normal"

    def adjust_params(self, risk_level: str, base_size: float,
                      min_confidence: float) -> tuple[float, float]:
        if risk_level == "reduced":
            return base_size * 0.5, max(min_confidence, 0.7)
        if risk_level == "critical":
            return base_size * 0.25, max(min_confidence, 0.85)
        return base_size, min_confidence


# ═══════════════════════════════════════════════════════════
# CorrelationFilter — prevent correlated positions
# ═══════════════════════════════════════════════════════════

class CorrelationFilter:
    """Prevents over-concentrated correlated positions."""

    HIGH_CORR_GROUPS = [
        {"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
         "AVAX/USDT:USDT", "DOT/USDT:USDT", "ADA/USDT:USDT", "NEAR/USDT:USDT"},
        {"ARB/USDT:USDT", "OP/USDT:USDT", "MATIC/USDT:USDT"},
        {"DOGE/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "SHIB/USDT:USDT", "FLOKI/USDT:USDT"},
        {"FET/USDT:USDT", "RENDER/USDT:USDT", "TAO/USDT:USDT"},
        {"UNI/USDT:USDT", "AAVE/USDT:USDT", "LINK/USDT:USDT"},
        {"SUI/USDT:USDT", "APT/USDT:USDT"},
    ]

    def filter(self, decisions: list[Decision], positions: dict) -> list[Decision]:
        filtered = []
        for d in decisions:
            if d.action not in (Action.OPEN_LONG, Action.OPEN_SHORT):
                filtered.append(d)
                continue

            direction = "long" if d.action == Action.OPEN_LONG else "short"
            group = self._find_group(d.symbol)
            if group is None:
                filtered.append(d)
                continue

            blocked = False
            # Check existing positions
            for sym in group:
                if sym in positions and sym != d.symbol:
                    if positions[sym].get("side", positions[sym].get("direction")) == direction:
                        blocked = True
                        break
            # Check already-approved decisions this cycle
            if not blocked:
                for approved in filtered:
                    if approved.action in (Action.OPEN_LONG, Action.OPEN_SHORT) and approved.symbol in group:
                        a_dir = "long" if approved.action == Action.OPEN_LONG else "short"
                        if a_dir == direction:
                            blocked = True
                            break

            if not blocked:
                filtered.append(d)

        return filtered

    def _find_group(self, symbol: str) -> set | None:
        for g in self.HIGH_CORR_GROUPS:
            if symbol in g:
                return g
        return None


# ═══════════════════════════════════════════════════════════
# DecisionLogger
# ═══════════════════════════════════════════════════════════

class DecisionLogger:
    """Logs every engine decision for post-analysis."""

    def __init__(self):
        Path("logs/decisions").mkdir(parents=True, exist_ok=True)
        self._file = Path("logs/decisions") / f"decisions_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    def log(self, decision: Decision, regime: str = "unknown"):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": decision.symbol,
            "action": decision.action.value,
            "confidence": decision.confidence,
            "trade_type": decision.trade_type.value if decision.trade_type else None,
            "reason": decision.reason,
            "signals": decision.signals,
            "regime": regime,
            "sl": decision.stop_loss,
            "tp": decision.take_profit,
        }
        with open(self._file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ═══════════════════════════════════════════════════════════
# SignalEngine — main entry point (replaces TradingBrain)
# ═══════════════════════════════════════════════════════════

class SignalEngine:
    """
    Deterministic replacement for Claude API.
    Same interface: receives market data, returns decisions dict.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or ENGINE_CONFIG
        self.gate_keeper = GateKeeper(self.cfg)
        self.signal_scorer = SignalScorer(self.cfg)
        self.exit_manager = ExitManager(self.cfg)
        self.position_manager = PositionManager(self.cfg)
        self.correlation_filter = CorrelationFilter()
        self.risk_throttle = RiskThrottle(self.cfg)
        self.logger = DecisionLogger()

    def decide(
        self,
        technical_data: dict,
        scalping_data: dict,
        tape_data: dict,
        vwap_data: dict,
        delta_data: dict,
        regime_data: dict,
        onchain_data: dict,
        positions: dict,
        trade_history: list,
        daily_pnl: float,
        symbols: list[str],
    ) -> dict:
        """
        Main method. Called every cycle.
        Returns dict in the same format as brain.analyze_and_decide().
        """
        decisions = []

        # 0. Risk throttle
        risk_level = self.risk_throttle.check(daily_pnl, trade_history)
        if risk_level == "stop":
            logger.warning("Risk throttle: STOP — daily loss limit reached")
            return {
                "decisions": [{"symbol": "*", "action": "hold", "confidence": 1.0,
                              "reason": "daily loss limit reached", "params": {}}],
                "market_outlook": "Стоп торговли: дневной лимит убытков",
                "risk_level": "high",
            }

        # 1. Manage existing positions
        for symbol, pos in positions.items():
            tech_1m = technical_data.get(symbol, {}).get("1m", {})
            regime = regime_data.get(symbol, {})
            pos_decision = self.position_manager.evaluate(symbol, pos, tech_1m, regime)
            if pos_decision:
                decisions.append(pos_decision)

        # 2. Evaluate new entries
        min_score = self.cfg["min_score"]
        min_conf = self.cfg["min_confidence"]
        adjusted_size, adjusted_conf = self.risk_throttle.adjust_params(
            risk_level, self.cfg["base_position_pct"], min_conf
        )

        entry_decisions = []
        for symbol in symbols:
            if symbol in positions:
                continue

            scalping = scalping_data.get(symbol, {})
            tech_1m = technical_data.get(symbol, {}).get("1m", {})
            tech_5m = technical_data.get(symbol, {}).get("5m", {})
            tape = tape_data.get(symbol, {})
            vwap = vwap_data.get(symbol, {})
            delta_d = delta_data.get(symbol, {})
            regime = regime_data.get(symbol, {})
            onchain = onchain_data.get(symbol, {})

            if not scalping or not tech_1m:
                continue

            # Gate check
            passed, reason = self.gate_keeper.check_all(
                symbol, scalping, tech_1m, regime, positions, risk_level
            )
            if not passed:
                continue

            # Score
            result = self.signal_scorer.score(
                scalping, tech_1m, tech_5m or {},
                tape, vwap, delta_d, onchain, regime
            )

            if result["conflicts"]:
                continue

            if result["score"] < min_score or result["confidence"] < adjusted_conf:
                continue

            # Calculate exits
            direction = result["direction"]
            trade_type = result["trade_type"] or TradeType.MOMENTUM
            price = tech_1m.get("price", 0)
            atr = tech_1m.get("atr", price * 0.003 if price > 0 else None)
            regime_name = regime.get("regime", "normal")

            exits = self.exit_manager.calculate(price, direction, trade_type, regime_name, atr)
            if exits is None:
                continue

            action = Action.OPEN_LONG if direction == "long" else Action.OPEN_SHORT

            # Build reason string
            top_signals = sorted(
                result["signals"].items(),
                key=lambda x: abs(x[1].get("score", 0)),
                reverse=True
            )[:3]
            reason_parts = [f"{k}:{v['detail']}" for k, v in top_signals if v.get("score", 0) != 0]

            d = Decision(
                action=action,
                symbol=symbol,
                confidence=result["confidence"],
                trade_type=trade_type,
                stop_loss=exits["stop_loss"],
                take_profit=exits["take_profit_1"],
                tp2=exits["take_profit_2"],
                trailing_stop=exits["trailing_stop"],
                position_size_pct=adjusted_size,
                reason=f"[{trade_type.value}] {'; '.join(reason_parts)}",
                signals=result["signals"],
            )
            entry_decisions.append(d)

        # 3. Sort by confidence, take best
        entry_decisions.sort(key=lambda x: x.confidence, reverse=True)

        # 4. Correlation filter
        entry_decisions = self.correlation_filter.filter(entry_decisions, positions)

        decisions.extend(entry_decisions)

        # 5. Log all
        for d in decisions:
            regime_name = regime_data.get(d.symbol, {}).get("regime", "unknown")
            self.logger.log(d, regime_name)

        # 6. Convert to brain.py format
        decision_dicts = [d.to_dict() for d in decisions]

        # Market outlook
        regimes = [r.get("regime", "?") for r in regime_data.values()]
        outlook = f"Rule-based engine | {len(decision_dicts)} decisions | " \
                  f"regimes: {', '.join(set(regimes))}"

        return {
            "decisions": decision_dicts,
            "market_outlook": outlook,
            "risk_level": risk_level if risk_level != "normal" else "low",
        }
