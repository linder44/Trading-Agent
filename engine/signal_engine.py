"""
MACD + 200 EMA + Support/Resistance Strategy Engine.

Simple, deterministic, 3-component strategy:
1. MACD(12,26,9) crossover for entry signal
2. 200 EMA as trend filter (longs above, shorts below)
3. S/R levels for confirmation

No AI, no complex scoring — just clean rules.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

from engine.config import STRATEGY_CONFIG


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
    MACD_TREND = "macd_trend"


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
        """Convert to the same dict format main.py expects."""
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
# DecisionLogger
# ═══════════════════════════════════════════════════════════

class DecisionLogger:
    """Logs every engine decision for post-analysis."""

    def __init__(self):
        Path("logs/decisions").mkdir(parents=True, exist_ok=True)
        self._file = Path("logs/decisions") / f"decisions_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    def log(self, decision: Decision):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": decision.symbol,
            "action": decision.action.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "signals": decision.signals,
            "sl": decision.stop_loss,
            "tp": decision.take_profit,
        }
        with open(self._file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ═══════════════════════════════════════════════════════════
# MACDStrategyEngine — main entry point
# ═══════════════════════════════════════════════════════════

class SignalEngine:
    """
    MACD + 200 EMA + S/R strategy engine.

    Entry rules:
    - LONG: MACD crosses above signal BELOW zero + price above EMA 200 + near support
    - SHORT: MACD crosses below signal ABOVE zero + price below EMA 200 + near resistance

    Exit rules:
    - SL behind EMA 200 with ATR buffer
    - TP = 1.5x SL distance
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or STRATEGY_CONFIG
        self.risk_throttle = RiskThrottle(self.cfg)
        self.decision_logger = DecisionLogger()
        self._symbol_losses: dict[str, list[float]] = {}

    def decide(
        self,
        technical_data: dict,
        positions: dict,
        trade_history: list,
        daily_pnl: float,
        symbols: list[str],
    ) -> dict:
        """
        Main method. Called every cycle.
        Returns dict compatible with main.py execution flow.
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
            tech_signal = technical_data.get(symbol, {}).get(self.cfg["signal_timeframe"], {})
            pos_decision = self._manage_position(symbol, pos, tech_signal)
            if pos_decision:
                decisions.append(pos_decision)

        # 2. Evaluate new entries
        min_conf = self.cfg["min_confidence"]
        adjusted_size, adjusted_conf = self.risk_throttle.adjust_params(
            risk_level, self.cfg["base_position_pct"], min_conf
        )

        logger.info(f"Engine: risk_level={risk_level}, min_conf={adjusted_conf:.2f}")

        entry_decisions = []
        for symbol in symbols:
            if symbol in positions:
                logger.debug(f"  {symbol}: SKIP (has open position)")
                continue

            # Cooldown check
            if self._is_on_cooldown(symbol):
                coin = symbol.replace("/USDT:USDT", "")
                logger.info(f"  {coin}: COOLDOWN — recent losses")
                continue

            # Max positions check
            total_open = len(positions) + len(entry_decisions)
            if total_open >= self.cfg["max_positions"]:
                logger.info("  Max positions reached, skipping remaining")
                break

            tech_signal = technical_data.get(symbol, {}).get(self.cfg["signal_timeframe"], {})
            tech_trend = technical_data.get(symbol, {}).get(self.cfg["trend_timeframe"], {})

            if not tech_signal or not tech_trend:
                coin = symbol.replace("/USDT:USDT", "")
                logger.warning(f"  {coin}: SKIP (no data)")
                continue

            entry = self._check_entry(symbol, tech_signal, tech_trend, adjusted_conf)
            if entry:
                entry.position_size_pct = adjusted_size
                entry_decisions.append(entry)

        decisions.extend(entry_decisions)

        # 3. Log all
        for d in decisions:
            self.decision_logger.log(d)

        # 4. Convert to main.py format
        decision_dicts = [d.to_dict() for d in decisions]

        outlook = f"MACD+EMA200+SR engine | {len(decision_dicts)} decisions"

        return {
            "decisions": decision_dicts,
            "market_outlook": outlook,
            "risk_level": risk_level if risk_level != "normal" else "low",
        }

    def _check_entry(self, symbol: str, tech_signal: dict, tech_trend: dict,
                     min_confidence: float) -> Decision | None:
        """Check MACD + EMA 200 + S/R entry conditions."""
        coin = symbol.replace("/USDT:USDT", "")
        price = tech_signal.get("price", 0)
        if price <= 0:
            return None

        # ── Component 1: MACD crossover ──
        macd = tech_signal.get("macd", {})
        macd_line = macd.get("macd_line", 0)
        signal_line = macd.get("signal_line", 0)
        prev_macd = macd.get("prev_macd_line", 0)
        prev_signal = macd.get("prev_signal_line", 0)
        histogram = macd.get("histogram", 0)

        # Detect crossover
        bullish_cross = prev_macd <= prev_signal and macd_line > signal_line
        bearish_cross = prev_macd >= prev_signal and macd_line < signal_line

        if not bullish_cross and not bearish_cross:
            logger.info(f"  {coin}: NO CROSSOVER — MACD={macd_line:.6f}, Signal={signal_line:.6f}")
            return None

        # MACD position relative to zero line
        # Bullish cross should happen BELOW zero (early momentum)
        # Bearish cross should happen ABOVE zero (early reversal)
        if bullish_cross and macd_line > 0:
            logger.info(f"  {coin}: BULLISH cross but ABOVE zero line — skip (late entry)")
            return None
        if bearish_cross and macd_line < 0:
            logger.info(f"  {coin}: BEARISH cross but BELOW zero line — skip (late entry)")
            return None

        direction = "long" if bullish_cross else "short"

        # ── Component 2: EMA 200 trend filter ──
        ema_200 = tech_trend.get("ema_200", 0)
        if ema_200 <= 0:
            logger.info(f"  {coin}: NO EMA 200 data — skip")
            return None

        if direction == "long" and price <= ema_200:
            logger.info(f"  {coin}: LONG rejected — price {price:.4f} BELOW EMA200 {ema_200:.4f}")
            return None
        if direction == "short" and price >= ema_200:
            logger.info(f"  {coin}: SHORT rejected — price {price:.4f} ABOVE EMA200 {ema_200:.4f}")
            return None

        # ── Component 3: S/R confirmation ──
        supports = tech_signal.get("support_levels", [])
        resistances = tech_signal.get("resistance_levels", [])
        sr_proximity = self.cfg["sr_proximity_pct"] / 100.0

        sr_confirmed = False
        nearest_sr = None

        if direction == "long":
            # Look for nearby support
            for s in supports:
                s_price = s.get("price", s) if isinstance(s, dict) else s
                if abs(price - s_price) / price <= sr_proximity:
                    sr_confirmed = True
                    nearest_sr = s_price
                    break
        else:
            # Look for nearby resistance
            for r in resistances:
                r_price = r.get("price", r) if isinstance(r, dict) else r
                if abs(price - r_price) / price <= sr_proximity:
                    sr_confirmed = True
                    nearest_sr = r_price
                    break

        # S/R is confirmation, not hard requirement — boost confidence if present
        base_confidence = 0.65
        if sr_confirmed:
            base_confidence = 0.80
            logger.info(f"  {coin}: S/R confirmed at {nearest_sr:.4f}")
        else:
            logger.info(f"  {coin}: No S/R nearby (supports={len(supports)}, resistances={len(resistances)})")

        if base_confidence < min_confidence:
            logger.info(f"  {coin}: confidence {base_confidence:.2f} < min {min_confidence:.2f}")
            return None

        # ── Calculate SL/TP ──
        atr = tech_signal.get("atr", price * 0.003)
        atr_buffer = atr * self.cfg["atr_sl_buffer"]

        if direction == "long":
            # SL behind EMA 200 with ATR buffer
            sl = ema_200 - atr_buffer
            sl_distance = price - sl
            tp = price + sl_distance * self.cfg["rr_ratio"]
        else:
            sl = ema_200 + atr_buffer
            sl_distance = sl - price
            tp = price - sl_distance * self.cfg["rr_ratio"]

        # Validate R/R
        if sl_distance <= 0:
            logger.info(f"  {coin}: Invalid SL distance — skip")
            return None

        risk_pct = (sl_distance / price) * 100
        if risk_pct > self.cfg["max_risk_pct"]:
            logger.info(f"  {coin}: Risk {risk_pct:.2f}% > max {self.cfg['max_risk_pct']}% — skip")
            return None

        rr = self.cfg["rr_ratio"]
        action = Action.OPEN_LONG if direction == "long" else Action.OPEN_SHORT

        reason = (f"MACD {'bullish' if bullish_cross else 'bearish'} cross "
                  f"{'below' if bullish_cross else 'above'} zero | "
                  f"Price {'above' if direction == 'long' else 'below'} EMA200={ema_200:.4f} | "
                  f"{'SR confirmed' if sr_confirmed else 'No SR'} | "
                  f"R/R={rr:.1f}")

        logger.info(f"  {coin}: >>> ENTRY {action.value} <<< conf={base_confidence:.0%} "
                    f"SL={sl:.4f} TP={tp:.4f} R/R={rr:.1f}")

        return Decision(
            action=action,
            symbol=symbol,
            confidence=base_confidence,
            trade_type=TradeType.MACD_TREND,
            stop_loss=sl,
            take_profit=tp,
            reason=reason,
            signals={
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "ema_200": ema_200,
                "sr_confirmed": sr_confirmed,
                "nearest_sr": nearest_sr,
                "atr": atr,
            },
        )

    def _manage_position(self, symbol: str, position: dict,
                         tech_signal: dict) -> Decision | None:
        """Manage open position — time exits + breakeven."""
        entry_price = position.get("entry_price", 0)
        direction = position.get("side", position.get("direction", "long"))
        current_price = tech_signal.get("price", entry_price)
        age_minutes = position.get("age_minutes", 0)

        if entry_price <= 0:
            return None

        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        coin = symbol.replace("/USDT:USDT", "")

        # 1. Max time
        if age_minutes > self.cfg["max_trade_minutes"]:
            logger.info(f"  {coin}: CLOSE — max time {age_minutes:.0f}min, pnl={pnl_pct:+.2f}%")
            return Decision(Action.CLOSE, symbol, 0.9,
                          reason=f"max time {age_minutes:.0f}min, pnl={pnl_pct:+.2f}%")

        # 2. Dead trade
        if abs(pnl_pct) < 0.1 and age_minutes > self.cfg["dead_trade_minutes"]:
            logger.info(f"  {coin}: CLOSE — flat {age_minutes:.0f}min")
            return Decision(Action.CLOSE, symbol, 0.8,
                          reason=f"dead trade {age_minutes:.0f}min")

        # 3. Move SL to breakeven
        if pnl_pct > self.cfg["breakeven_pnl_pct"] and not position.get("sl_at_breakeven"):
            commission = entry_price * 0.0006
            new_sl = entry_price + commission if direction == "long" else entry_price - commission
            logger.info(f"  {coin}: UPDATE SL to breakeven, pnl={pnl_pct:+.2f}%")
            return Decision(Action.UPDATE_SL, symbol, 0.9, stop_loss=new_sl,
                          reason=f"SL to breakeven, pnl={pnl_pct:+.2f}%",
                          signals={"sl_at_breakeven": True})

        return None

    def record_loss(self, symbol: str):
        """Record a loss for symbol cooldown tracking."""
        self._symbol_losses.setdefault(symbol, [])
        self._symbol_losses[symbol].append(time.time())
        cutoff = time.time() - 3600
        self._symbol_losses[symbol] = [t for t in self._symbol_losses[symbol] if t > cutoff]

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on cooldown after recent losses."""
        losses = self._symbol_losses.get(symbol, [])
        cooldown = self.cfg["symbol_cooldown_minutes"] * 60
        recent = [t for t in losses if time.time() - t < cooldown]
        return len(recent) >= 2
