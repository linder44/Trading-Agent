"""
Autonomous Trading Agent — MACD + 200 EMA + S/R Strategy v5.0

Simple, deterministic, 3-component strategy:
1. MACD(12,26,9) crossover for entry signal
2. 200 EMA as trend filter
3. Support/Resistance for confirmation

No AI, no complex indicators — just clean rules.

Usage:
    python main.py              # Demo mode (Bitget demo account)
    python main.py --live       # Live trading
    python main.py --once       # One cycle and exit
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config import config, TRADING_MODE
from exchange.client import ExchangeClient
from analysis.technical import TechnicalAnalyzer
from analysis.trade_history import TradeHistoryTracker
from risk.manager import RiskManager
from risk.drawdown_breaker import DrawdownBreaker
from orders.manager import OrderManager
from engine.signal_engine import SignalEngine
from engine.config import STRATEGY_CONFIG
from utils.notifications import Notifier


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")


class TradingAgent:
    """Main orchestrator — MACD + EMA200 + S/R strategy."""

    def __init__(self, mode: str = "demo"):
        self.mode = mode

        logger.info("=" * 60)
        logger.info("  TRADING AGENT v5.0 (MACD + EMA200 + S/R)")
        logger.info(f"  Mode: {self.mode.upper()}")
        if self.mode == "paper":
            logger.info(f"  Paper balance: {config.paper.initial_balance} USDT")
        elif self.mode == "demo":
            logger.info("  Bitget Demo Account")
        logger.info(f"  Symbols: {config.trading.symbols}")
        logger.info(f"  Signal TF: {STRATEGY_CONFIG['signal_timeframe']}, Trend TF: {STRATEGY_CONFIG['trend_timeframe']}")
        logger.info("=" * 60)

        # Core modules — only what we need
        self.exchange = ExchangeClient(config.bitget)
        self.analyzer = TechnicalAnalyzer()
        self.risk = RiskManager(config.trading)
        self.orders = OrderManager(self.exchange, self.risk)
        self.engine = SignalEngine()
        self.notifier = Notifier(config.notifications)
        self.drawdown = DrawdownBreaker()
        self.trade_history = TradeHistoryTracker()

        # Validate symbols
        self.symbols = self.exchange.validate_symbols(config.trading.symbols)
        if not self.symbols:
            logger.error("No valid symbols found!")
            sys.exit(1)
        logger.info(f"Active symbols: {self.symbols}")

        # Paper trading state
        self.paper_balance = config.paper.initial_balance
        self.paper_trades: list[dict] = []

        self._last_daily_reset = datetime.now(timezone.utc).date()
        self._cycle_count = 0
        self._decision_log: list[dict] = []

        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)

    def run_cycle(self):
        """Run one full analysis and trading cycle."""
        self._cycle_count += 1
        logger.info("-" * 40)
        logger.info(f"Cycle #{self._cycle_count}: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

        # Daily reset
        today = datetime.now(timezone.utc).date()
        if today > self._last_daily_reset:
            self.risk.reset_daily_stats()
            self.drawdown.reset_daily()
            self._last_daily_reset = today

        # Drawdown check
        can_trade, reason = self.drawdown.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            return

        # Sync positions
        if self.mode in ("demo", "live"):
            self.orders.sync_positions_from_exchange()

        # Auto-close expired positions
        max_age = STRATEGY_CONFIG["max_trade_minutes"]
        expired = self.risk.get_expired_positions(max_age)
        for symbol in expired:
            logger.warning(f"Position {symbol} expired (>{max_age} min), force closing")
            if self.mode == "paper":
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    price = ticker["last"]
                    pnl = self.risk.close_position(symbol, price)
                    self.paper_balance += pnl
                    self.drawdown.update(pnl, self.paper_balance)
                except Exception as e:
                    logger.error(f"Error closing expired {symbol}: {e}")
            else:
                result = self.orders.close_position(symbol)
                if result:
                    self.drawdown.update(result.get("pnl", 0), self.exchange.fetch_usdt_balance())
                    try:
                        self.notifier.send(
                            f"\u23F0 <b>AUTO-CLOSE (>{max_age} min)</b>\n"
                            f"{symbol} | PnL: {result.get('pnl', 0):+.2f} USDT"
                        )
                    except Exception:
                        pass

        # ── DATA COLLECTION ──────────────────────────────────
        # Only fetch what the strategy needs: signal TF + trend TF

        signal_tf = STRATEGY_CONFIG["signal_timeframe"]
        trend_tf = STRATEGY_CONFIG["trend_timeframe"]

        technical_data = {}
        for symbol in self.symbols:
            try:
                ohlcv_dict = {}
                for tf in [signal_tf, trend_tf]:
                    ohlcv_dict[tf] = self.exchange.fetch_ohlcv(symbol, tf, limit=250)
                technical_data[symbol] = self.analyzer.multi_timeframe_analysis(ohlcv_dict, symbol)
            except Exception as e:
                logger.error(f"Data fetch error {symbol}: {e}")

        if not technical_data:
            logger.warning("No technical data, skipping cycle")
            return

        # ── ENGINE DECISION ──────────────────────────────────

        if self.mode in ("demo", "live"):
            balance = self.exchange.fetch_usdt_balance()
        else:
            balance = self.paper_balance

        portfolio = self.risk.get_portfolio_summary()
        drawdown_status = self.drawdown.get_status()

        logger.info(f"Balance: {balance:.2f} USDT | Positions: {portfolio['num_positions']} | "
                     f"Daily PnL: {drawdown_status['daily_pnl_pct']:+.1f}%")

        # Build positions dict for engine
        engine_positions = {}
        for sym, pos in self.risk.positions.items():
            age = self.risk.get_position_age_minutes(sym) or 0
            engine_positions[sym] = {
                "side": pos.side,
                "entry_price": pos.entry_price,
                "amount": pos.amount,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "age_minutes": age,
                "sl_at_breakeven": getattr(pos, "sl_at_breakeven", False),
            }

        # Run engine
        logger.info("Running MACD+EMA200+SR engine...")
        decision = self.engine.decide(
            technical_data=technical_data,
            positions=engine_positions,
            trade_history=self.trade_history.get_recent_trades(20) if hasattr(self.trade_history, 'get_recent_trades') else [],
            daily_pnl=drawdown_status.get("daily_pnl_pct", 0),
            symbols=self.symbols,
        )

        logger.info(f"Engine: {decision.get('market_outlook', 'N/A')}")
        logger.info(f"Risk level: {decision.get('risk_level', 'N/A')}")

        # ── EXECUTE DECISIONS ────────────────────────────────

        min_confidence = self.drawdown.get_min_confidence()

        actions = decision.get("decisions", [])
        for action in actions:
            confidence = action.get("confidence", 0)
            if confidence < min_confidence and action.get("action") not in ("hold", "close", "update_sl"):
                logger.info(f"Skipping {action.get('symbol')} {action.get('action')}: "
                            f"confidence {confidence} < min {min_confidence}")
                continue

            self._execute_decision(action, balance)

            # Log decision
            self._log_structured_decision(action)

    def _execute_decision(self, decision: dict, balance: float):
        """Execute a single trading decision."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")

        min_conf = self.drawdown.get_min_confidence()
        if confidence < min_conf and action not in ("hold", "close", "update_sl"):
            logger.info(f"Skip {symbol} {action}: confidence {confidence} < {min_conf}")
            return

        logger.info(f"{'[PAPER] ' if self.mode == 'paper' else ''}Execute: {action} {symbol} (conf={confidence})")
        logger.info(f"  Reason: {reason}")

        if self.mode == "paper":
            self._execute_paper(decision, balance)
        else:
            self._execute_live(decision, balance)

    def _execute_paper(self, decision: dict, balance: float):
        """Paper trading execution."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")
        params = decision.get("params", {})

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
        except Exception:
            price = 0

        if action in ("open_long", "open_short"):
            side = "long" if action == "open_long" else "short"

            # Use SL/TP from engine params
            stop_loss = params.get("new_stop_loss")
            if not stop_loss:
                atr = self._get_atr(symbol)
                if atr:
                    stop_loss = price - atr * 1.5 if side == "long" else price + atr * 1.5
                else:
                    stop_loss = price * (0.995 if side == "long" else 1.005)

            sl_distance = abs(price - stop_loss)
            take_profit = price + sl_distance * STRATEGY_CONFIG["rr_ratio"] if side == "long" else price - sl_distance * STRATEGY_CONFIG["rr_ratio"]

            size_mult = self.drawdown.get_position_size_multiplier()
            amount = self.risk.calculate_position_size(balance, price, stop_loss) * size_mult

            can_open, msg = self.risk.can_open_position(symbol, balance)
            if not can_open:
                logger.warning(f"[PAPER] Cannot open {symbol}: {msg}")
                return

            self.risk.register_position(symbol, side, price, amount, stop_loss, take_profit)
            rr = STRATEGY_CONFIG["rr_ratio"]
            logger.info(f"[PAPER] {action} {symbol} @ {price} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | R/R: {rr}")

        elif action == "close":
            if symbol in self.risk.positions:
                pnl = self.risk.close_position(symbol, price)
                self.paper_balance += pnl
                self.drawdown.update(pnl, self.paper_balance)
                logger.info(f"[PAPER] Close {symbol} @ {price} | PnL: {pnl:+.2f} | Balance: {self.paper_balance:.2f}")

        elif action == "hold":
            logger.info(f"[PAPER] Hold {symbol}: {reason}")

        self._save_paper_log()
        msg = self._format_paper_message(action, symbol, price, confidence, reason)
        try:
            self.notifier.send(msg)
        except Exception:
            pass

    def _execute_live(self, decision: dict, balance: float):
        """Live/demo trading — place real orders."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")
        params = decision.get("params", {})

        result = None

        try:
            if action in ("open_long", "open_short"):
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker["last"]
                atr = self._get_atr(symbol)

                size_mult = self.drawdown.get_position_size_multiplier()
                if size_mult <= 0:
                    logger.info("Position sizing blocked by drawdown breaker")
                    return

                side = "long" if action == "open_long" else "short"

                if side == "long":
                    result = self.orders.open_long(symbol, balance * size_mult, price, atr)
                else:
                    result = self.orders.open_short(symbol, balance * size_mult, price, atr)

            elif action == "close":
                if symbol in self.risk.positions:
                    pos = self.risk.positions[symbol]
                    ticker = self.exchange.fetch_ticker(symbol)
                    exit_price = ticker["last"]
                    duration = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60
                    self.trade_history.record_trade(
                        symbol=symbol, side=pos.side,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        amount=pos.amount, reason_close=reason,
                        duration_minutes=duration,
                    )
                result = self.orders.close_position(symbol)
                if result:
                    self.drawdown.update(result.get("pnl", 0), self.exchange.fetch_usdt_balance())

            elif action == "update_sl":
                new_sl = params.get("new_stop_loss")
                if new_sl:
                    result = self.orders.update_stop_loss(symbol, new_sl)

            elif action == "hold":
                logger.info(f"Hold {symbol}: {reason}")

            if result:
                msg = self._format_trade_message(action, symbol, confidence, reason, result)
                try:
                    self.notifier.send(msg)
                except Exception:
                    pass
                logger.info(f"Executed: {result}")

        except Exception as e:
            logger.error(f"Execution error {action} {symbol}: {e}")
            try:
                self.notifier.send(f"\u274C <b>ERROR</b> {action} {symbol}: {e}")
            except Exception:
                pass

    def _log_structured_decision(self, decision: dict):
        """Log structured decision for post-analysis."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": decision.get("symbol", ""),
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "reason": decision.get("reason"),
            "result_pnl": None,
        }
        self._decision_log.append(entry)

        if len(self._decision_log) % 10 == 0:
            self._save_decision_log()

    def _save_decision_log(self):
        log_file = Path("data/decision_log.json")
        log_file.write_text(json.dumps(self._decision_log[-500:], indent=1, ensure_ascii=False))

    @staticmethod
    def _action_emoji(action: str) -> str:
        emojis = {
            "open_long": "\U0001F7E2", "open_short": "\U0001F534",
            "close": "\U0001F512", "close_position": "\U0001F512",
            "update_sl": "\U0001F6E1", "hold": "\u23F8",
        }
        return emojis.get(action, "\U0001F4CA")

    @staticmethod
    def _action_label(action: str) -> str:
        labels = {
            "open_long": "LONG", "open_short": "SHORT",
            "close": "CLOSE", "close_position": "CLOSE",
            "update_sl": "SL UPDATE", "hold": "HOLD",
        }
        return labels.get(action, action.upper())

    def _format_trade_message(self, action: str, symbol: str, confidence: float,
                              reason: str, result: dict) -> str:
        emoji = self._action_emoji(action)
        label = self._action_label(action)
        coin = symbol.replace("/USDT:USDT", "").replace("/USDT", "")
        lines = [f"{emoji} <b>{label}</b> {coin} | conf={confidence:.0%}"]
        if action in ("open_long", "open_short"):
            lines.append(f"Entry: {result.get('entry_price')} | SL: {result.get('stop_loss')} | TP: {result.get('take_profit')}")
        elif action in ("close", "close_position"):
            pnl = result.get("pnl")
            lines.append(f"Exit: {result.get('exit_price')} | PnL: {pnl:+.2f} USDT" if pnl else "")
        lines.append(f"{reason}")
        return "\n".join(lines)

    def _format_paper_message(self, action: str, symbol: str, price: float,
                              confidence: float, reason: str) -> str:
        emoji = self._action_emoji(action)
        label = self._action_label(action)
        coin = symbol.replace("/USDT:USDT", "").replace("/USDT", "")
        return f"\U0001F4DD [PAPER] {emoji} <b>{label}</b> {coin} @ {price} | conf={confidence:.0%}\n{reason}"

    def _get_atr(self, symbol: str) -> float | None:
        """Get latest ATR value for a symbol."""
        try:
            df = self.exchange.fetch_ohlcv(symbol, STRATEGY_CONFIG["signal_timeframe"], limit=50)
            df_analyzed = self.analyzer.compute_indicators(df)
            return float(df_analyzed["atr"].iloc[-1])
        except Exception:
            return None

    def _save_paper_log(self):
        log_file = Path("data/paper_trades.json")
        data = {
            "balance": self.paper_balance,
            "initial_balance": config.paper.initial_balance,
            "pnl_total": self.paper_balance - config.paper.initial_balance,
            "num_trades": len(self.paper_trades),
            "trades": self.paper_trades[-100:],
        }
        log_file.write_text(json.dumps(data, indent=2))

    def run(self, once: bool = False):
        """Main loop — continuous analysis."""
        try:
            self.notifier.send(
                f"\U0001F680 <b>Trading Agent v5.0 STARTED (MACD+EMA200+SR)</b>\n"
                f"Mode: <b>{self.mode.upper()}</b>\n"
                f"Symbols: {', '.join(self.symbols)}\n"
                f"Signal TF: {STRATEGY_CONFIG['signal_timeframe']}, Trend TF: {STRATEGY_CONFIG['trend_timeframe']}"
            )
        except Exception:
            pass

        if once:
            self.run_cycle()
            return

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self._save_decision_log()
                try:
                    self.notifier.send(f"\U0001F6D1 <b>Agent STOPPED</b>")
                except Exception:
                    pass
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                try:
                    self.notifier.send(f"\u274C <b>CYCLE ERROR</b>\n{e}")
                except Exception:
                    pass
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Trading Agent v5.0 — MACD + EMA200 + S/R")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--once", action="store_true", help="Run one cycle")
    args = parser.parse_args()

    mode = "live" if args.live else TRADING_MODE

    if mode == "live":
        logger.warning("!" * 60)
        logger.warning("  LIVE MODE — REAL MONEY AT RISK!")
        logger.warning("  Press Ctrl+C within 5 seconds to cancel...")
        logger.warning("!" * 60)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Cancelled.")
            return

    agent = TradingAgent(mode=mode)
    agent.run(once=args.once)


if __name__ == "__main__":
    main()
