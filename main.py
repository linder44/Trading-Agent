"""
Autonomous Trading Agent — Rule-Based Scalping v4.0

NO Claude API — fully deterministic SignalEngine.
Zero cost, zero latency, reproducible decisions.

Key features:
- SignalEngine: cascading filters + numerical scoring (replaces Claude)
- RegimeDetector: adapts strategy to market state (trend/range/squeeze/choppy)
- DynamicExitManager: context-aware SL/TP with partial take profits
- DrawdownBreaker: graduated loss protection
- CorrelationGuard: prevents over-concentrated positions
- TapeReader + VWAPBands + DeltaDivergence: scalping indicators
- Adaptive cycle: 60s/120s/300s by volatility
- Structured decision logging for post-analysis

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
from analysis.patterns import PatternRecognizer
from analysis.onchain import OnChainAnalyzer
from analysis.correlations import MarketCorrelations
from analysis.quant import QuantAnalyzer
from analysis.trade_history import TradeHistoryTracker
from analysis.liquidations import LiquidationAnalyzer
from analysis.cross_correlation import CrossCorrelationAnalyzer
from analysis.time_context import TimeContextAnalyzer
from analysis.scalping import ScalpingAnalyzer
from analysis.signal_aggregator import SignalAggregator
from analysis.regime_detector import RegimeDetector
from analysis.tape_reader import TapeReader
from analysis.vwap_bands import VWAPBands
from analysis.delta_divergence import DeltaDivergence
from news.fetcher import NewsFetcher
from news.social import SocialSentiment
from risk.manager import RiskManager
from risk.dynamic_exits import DynamicExitManager
from risk.drawdown_breaker import DrawdownBreaker
from risk.correlation_guard import CorrelationGuard
from orders.manager import OrderManager
from engine.signal_engine import SignalEngine
from utils.notifications import Notifier


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")


class TradingAgent:
    """Main orchestrator — optimized for scalping profitability."""

    def __init__(self, mode: str = "demo"):
        self.mode = mode

        logger.info("=" * 60)
        logger.info("  AUTONOMOUS TRADING AGENT v4.0 (RULE-BASED)")
        logger.info(f"  Mode: {self.mode.upper()}")
        if self.mode == "paper":
            logger.info(f"  Paper balance: {config.paper.initial_balance} USDT")
        elif self.mode == "demo":
            logger.info("  Bitget Demo Account")
        logger.info(f"  Symbols: {config.trading.symbols}")
        logger.info("=" * 60)

        # Core modules
        self.exchange = ExchangeClient(config.bitget)
        self.analyzer = TechnicalAnalyzer()
        self.risk = RiskManager(config.trading)
        self.orders = OrderManager(self.exchange, self.risk)
        self.engine = SignalEngine()
        self.notifier = Notifier(config.notifications)

        # New optimized modules
        self.signal_aggregator = SignalAggregator()
        self.regime_detector = RegimeDetector()
        self.exit_manager = DynamicExitManager()
        self.drawdown = DrawdownBreaker()
        self.corr_guard = CorrelationGuard()
        self.tape_reader = TapeReader()
        self.vwap_bands = VWAPBands()
        self.delta_div = DeltaDivergence()

        # Retained modules (slimmed)
        self.patterns = PatternRecognizer()
        self.onchain = OnChainAnalyzer()
        self.quant = QuantAnalyzer()
        self.trade_history = TradeHistoryTracker()
        self.liquidations = LiquidationAnalyzer()
        self.time_context = TimeContextAnalyzer()
        self.scalping = ScalpingAnalyzer()

        # Removed from main cycle: news, social, correlations, cross_correlation
        # These are either too slow or irrelevant for 2-min scalping cycles
        self.news = NewsFetcher(config.news)
        self.social = SocialSentiment()
        self.correlations = MarketCorrelations()
        self.cross_corr = CrossCorrelationAnalyzer()

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
        self._last_hourly_corr_update = 0.0
        self._cycle_count = 0

        # Structured decision log
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

        # Drawdown check — can we trade at all?
        can_trade, reason = self.drawdown.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            return

        # Sync positions (demo/live)
        if self.mode in ("demo", "live"):
            self.orders.sync_positions_from_exchange()

        # Auto-close expired positions (90 min instead of 120)
        max_age = 90
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
                    self.notifier.send(
                        f"\u23F0 <b>AUTO-CLOSE (>{max_age} min)</b>\n"
                        f"{symbol} | PnL: {result.get('pnl', 0):+.2f} USDT"
                    )

        # ── DATA COLLECTION ──────────────────────────────────

        # 1. OHLCV for all symbols (1m, 5m)
        technical_data = {}
        ohlcv_cache = {}
        for symbol in self.symbols:
            try:
                ohlcv_dict = {}
                for tf in ["1m", "5m"]:  # Skip 15m to save time
                    ohlcv_dict[tf] = self.exchange.fetch_ohlcv(symbol, tf, limit=200)
                technical_data[symbol] = self.analyzer.multi_timeframe_analysis(ohlcv_dict, symbol)
                ohlcv_cache[symbol] = ohlcv_dict
            except Exception as e:
                logger.error(f"Data fetch error {symbol}: {e}")

        if not technical_data:
            logger.warning("No technical data, skipping cycle")
            return

        # 2. Scalping microstructure (order flow, momentum, spread)
        scalping_data = {}
        for symbol, ohlcv_dict in ohlcv_cache.items():
            if "1m" in ohlcv_dict:
                scalping_data[symbol] = self.scalping.full_scalping_analysis(ohlcv_dict["1m"])

        # 3. New indicators: Tape Reading, VWAP Bands, Delta Divergence
        tape_data = {}
        vwap_data = {}
        delta_data = {}
        for symbol, ohlcv_dict in ohlcv_cache.items():
            if "1m" in ohlcv_dict:
                df_1m = ohlcv_dict["1m"]
                tape_data[symbol] = self.tape_reader.analyze(df_1m)
                vwap_data[symbol] = self.vwap_bands.analyze(df_1m)
                delta_data[symbol] = self.delta_div.analyze(df_1m)

        # 4. Regime detection (from 5m data)
        regime_data = {}
        for symbol, ohlcv_dict in ohlcv_cache.items():
            if "5m" in ohlcv_dict:
                regime_data[symbol] = self.regime_detector.detect(ohlcv_dict["5m"])

        # 5. Signal Aggregation (Tier 1-2-3 weighted)
        aggregated_signals = {}
        for symbol in self.symbols:
            if symbol in technical_data and symbol in scalping_data:
                aggregated_signals[symbol] = self.signal_aggregator.aggregate(
                    technical_data=technical_data[symbol],
                    scalping_data=scalping_data.get(symbol, {}),
                    onchain_data=None,  # Will add onchain in Tier 3
                    liquidation_data=None,
                    symbol=symbol,
                )

        # 6. On-chain data (Tier 3, not blocking)
        onchain_data = {}
        try:
            onchain_data = self.onchain.get_full_onchain_data(self.exchange, self.symbols)
        except Exception as e:
            logger.warning(f"On-chain data fetch failed: {e}")

        # 7. Liquidation data (Tier 3)
        liquidation_data = {}
        try:
            liquidation_data = self.liquidations.get_all_liquidations(self.symbols)
        except Exception as e:
            logger.warning(f"Liquidation data fetch failed: {e}")

        # 8. Time context
        time_context_data = self.time_context.get_time_context()

        # 9. Trade history
        trade_history_data = self.trade_history.get_summary_for_prompt()

        # 10. Trailing stops check
        if self.mode in ("demo", "live") and self.risk.positions:
            trailing_updates = self.risk.check_all_trailing_stops(
                lambda sym: self.exchange.fetch_ticker(sym)["last"]
            )
            for update in trailing_updates:
                logger.info(f"Trailing stop {update['symbol']}: SL {update['old_sl']} -> {update['new_sl']}")
                self.orders.update_stop_loss(update["symbol"], update["new_sl"])

        # Time-based exit checks
        for symbol, pos in list(self.risk.positions.items()):
            age = self.risk.get_position_age_minutes(symbol) or 0
            try:
                current_price = self.exchange.fetch_ticker(symbol)["last"]
                if pos.side == "long":
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

                action = self.exit_manager.time_based_exit_check(age, pnl_pct)
                if action == "close":
                    logger.info(f"Time-based close: {symbol} (age={age:.0f}m, pnl={pnl_pct:.2f}%)")
                    if self.mode == "paper":
                        pnl = self.risk.close_position(symbol, current_price)
                        self.paper_balance += pnl
                        self.drawdown.update(pnl, self.paper_balance)
                    else:
                        result = self.orders.close_position(symbol)
                        if result:
                            self.drawdown.update(result.get("pnl", 0), self.exchange.fetch_usdt_balance())
                elif action == "tighten" and self.exit_manager.should_move_to_breakeven(
                    pos.entry_price, current_price, pos.side,
                    self._get_atr(symbol) or pos.entry_price * 0.003
                ):
                    logger.info(f"Moving SL to breakeven: {symbol}")
                    if self.mode in ("demo", "live"):
                        self.orders.update_stop_loss(symbol, pos.entry_price)
                    else:
                        pos.stop_loss = pos.entry_price
            except Exception as e:
                logger.error(f"Time-based exit check failed for {symbol}: {e}")

        # ── ENGINE DECISION (rule-based, no API) ──────────────

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

        # Rule-based engine — instant, free
        logger.info("Running signal engine...")
        decision = self.engine.decide(
            technical_data=technical_data,
            scalping_data=scalping_data,
            tape_data=tape_data,
            vwap_data=vwap_data,
            delta_data=delta_data,
            regime_data=regime_data,
            onchain_data=onchain_data,
            positions=engine_positions,
            trade_history=self.trade_history.get_recent_trades(20) if hasattr(self.trade_history, 'get_recent_trades') else [],
            daily_pnl=drawdown_status.get("daily_pnl_pct", 0),
            symbols=self.symbols,
        )

        logger.info(f"Engine: {decision.get('market_outlook', 'N/A')}")
        logger.info(f"Risk level: {decision.get('risk_level', 'N/A')}")

        # ── EXECUTE DECISIONS ────────────────────────────────

        min_confidence = self.drawdown.get_min_confidence()
        size_mult = self.drawdown.get_position_size_multiplier()

        actions = decision.get("decisions", [])
        for action in actions:
            # Apply drawdown-adjusted confidence threshold
            confidence = action.get("confidence", 0)
            if confidence < min_confidence and action.get("action") not in ("hold", "close", "update_sl"):
                logger.info(f"Skipping {action.get('symbol')} {action.get('action')}: "
                            f"confidence {confidence} < min {min_confidence}")
                continue

            # Correlation guard check
            if action.get("action") in ("open_long", "open_short"):
                symbol = action.get("symbol", "")
                direction = "long" if action["action"] == "open_long" else "short"
                open_pos = {s: {"side": p.side} for s, p in self.risk.positions.items()}
                can_open, reason = self.corr_guard.can_open_position(symbol, direction, open_pos)
                if not can_open:
                    logger.info(f"Correlation guard blocked: {reason}")
                    continue

                # Regime check — skip if choppy
                regime = regime_data.get(symbol, {})
                if regime.get("regime") == "choppy":
                    logger.info(f"Skipping {symbol}: choppy regime")
                    continue

                # RVOL check
                tech = technical_data.get(symbol, {}).get("1m", {})
                rvol = tech.get("rvol", 1.0)
                if rvol < 0.5:
                    logger.info(f"Skipping {symbol}: RVOL={rvol:.2f} too low")
                    continue

            self._execute_decision(action, balance)

            # Log decision for analysis
            self._log_structured_decision(action, aggregated_signals, regime_data)

    def _execute_decision(self, decision: dict, balance: float):
        """Execute a single AI trading decision."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")
        params = decision.get("params", {})

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

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
        except Exception:
            price = 0

        if action in ("open_long", "open_short"):
            side = "long" if action == "open_long" else "short"
            atr = self._get_atr(symbol)

            # Dynamic exits based on regime
            regime = "normal"  # Would come from regime_detector in full cycle
            exit_plan = self.exit_manager.calculate_exits(price, side, atr, regime)
            if exit_plan is None:
                logger.warning(f"[PAPER] Bad R/R for {symbol}, skipping")
                return

            stop_loss = exit_plan.stop_loss
            take_profit = exit_plan.take_profit_1

            # Apply drawdown size multiplier
            size_mult = self.drawdown.get_position_size_multiplier()
            amount = self.risk.calculate_position_size(balance, price, stop_loss) * size_mult

            can_open, msg = self.risk.can_open_position(symbol, balance)
            if not can_open:
                logger.warning(f"[PAPER] Cannot open {symbol}: {msg}")
                return

            self.risk.register_position(symbol, side, price, amount, stop_loss, take_profit)
            logger.info(f"[PAPER] {action} {symbol} @ {price} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | R/R: {exit_plan.risk_reward}")

        elif action == "close":
            if symbol in self.risk.positions:
                pnl = self.risk.close_position(symbol, price)
                self.paper_balance += pnl
                self.drawdown.update(pnl, self.paper_balance)
                self.trade_history.record_trade(
                    symbol=symbol, side=self.risk.positions.get(symbol, type("", (), {"side": ""})()).side if hasattr(self.risk.positions.get(symbol), 'side') else "",
                    entry_price=price, exit_price=price, amount=0,
                    reason_close=reason,
                )
                logger.info(f"[PAPER] Close {symbol} @ {price} | PnL: {pnl:+.2f} | Balance: {self.paper_balance:.2f}")

        elif action == "hold":
            logger.info(f"[PAPER] Hold {symbol}: {reason}")

        self._save_paper_log()
        msg = self._format_paper_message(action, symbol, price, confidence, reason)
        self.notifier.send(msg)

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

                # Apply drawdown position size multiplier
                size_mult = self.drawdown.get_position_size_multiplier()
                if size_mult <= 0:
                    logger.info(f"Position sizing blocked by drawdown breaker")
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
                self.notifier.send(msg)
                logger.info(f"Executed: {result}")

        except Exception as e:
            logger.error(f"Execution error {action} {symbol}: {e}")
            self.notifier.send(f"\u274C <b>ERROR</b> {action} {symbol}: {e}")

    def _log_structured_decision(self, decision: dict, aggregated_signals: dict, regime_data: dict):
        """Log structured decision for post-analysis."""
        symbol = decision.get("symbol", "")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "reason": decision.get("reason"),
            "regime": regime_data.get(symbol, {}).get("regime", "unknown"),
            "signal_score": aggregated_signals.get(symbol, {}).get("weighted_score", 0),
            "tier1_conflict": aggregated_signals.get(symbol, {}).get("tier1_conflict", False),
            "result_pnl": None,  # Filled after close
        }
        self._decision_log.append(entry)

        # Persist every 10 decisions
        if len(self._decision_log) % 10 == 0:
            self._save_decision_log()

    def _save_decision_log(self):
        """Save structured decision log."""
        log_file = Path("data/decision_log.json")
        log_file.write_text(json.dumps(self._decision_log[-500:], indent=1, ensure_ascii=False))

    # ── Telegram message formatting ──────────────────────────

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
        """Get latest ATR value for a symbol (5m)."""
        try:
            df = self.exchange.fetch_ohlcv(symbol, "5m", limit=50)
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

    def _get_adaptive_interval(self) -> int:
        """Adaptive cycle interval based on market volatility.

        High volatility → faster cycles (60s)
        Low volatility → slower cycles (300s)
        Normal → 120s
        """
        try:
            df = self.exchange.fetch_ohlcv(self.symbols[0], "5m", limit=30)
            if len(df) < 20:
                return 120

            from analysis.scalping import ScalpingAnalyzer
            vr = ScalpingAnalyzer.volatility_micro_regime(df)
            atr_ratio = vr.get("atr_ratio", 1.0)

            if atr_ratio > 1.5:
                return 60   # High volatility — faster
            elif atr_ratio < 0.5:
                return 300  # Dead market — slower
            else:
                return 120  # Normal
        except Exception:
            return 120

    def run(self, once: bool = False):
        """Main loop with adaptive interval."""
        self.notifier.send(
            f"\U0001F680 <b>Trading Agent v4.0 STARTED (rule-based)</b>\n"
            f"Mode: <b>{self.mode.upper()}</b>\n"
            f"Symbols: {', '.join(self.symbols)}"
        )

        if once:
            self.run_cycle()
            return

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self._save_decision_log()
                self.notifier.send(f"\U0001F6D1 <b>Agent STOPPED</b>")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                self.notifier.send(f"\u274C <b>CYCLE ERROR</b>\n{e}")

            wait = self._get_adaptive_interval()
            logger.info(f"Next cycle in {wait}s...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Agent v3.0")
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
