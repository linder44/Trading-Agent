"""
Autonomous AI Trading Agent
Main entry point and orchestration loop.

Usage:
    python main.py              # Run in demo mode (default, Bitget demo account)
    python main.py --live       # Switch to live trading (real money!)
    python main.py --once       # Run analysis once and exit
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import config, TRADING_MODE
from exchange.client import ExchangeClient
from analysis.technical import TechnicalAnalyzer
from news.fetcher import NewsFetcher
from risk.manager import RiskManager
from orders.manager import OrderManager
from agent.brain import TradingBrain
from utils.notifications import Notifier


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")


class TradingAgent:
    """Main orchestrator that ties everything together."""

    def __init__(self, mode: str = "demo"):
        self.mode = mode  # "paper", "demo", or "live"

        logger.info("=" * 60)
        logger.info("  AUTONOMOUS AI TRADING AGENT")
        logger.info(f"  Mode: {self.mode.upper()}")
        if self.mode == "paper":
            logger.info(f"  Paper Balance: {config.paper.initial_balance} USDT")
        elif self.mode == "demo":
            logger.info("  Bitget DEMO account (virtual money)")
        logger.info(f"  Symbols: {config.trading.symbols}")
        logger.info(f"  Interval: {config.trading.analysis_interval_minutes} min")
        logger.info("=" * 60)

        self.exchange = ExchangeClient(config.bitget)
        self.analyzer = TechnicalAnalyzer()
        self.news = NewsFetcher(config.news)
        self.risk = RiskManager(config.trading)
        self.orders = OrderManager(self.exchange, self.risk)
        self.brain = TradingBrain(config.claude)
        self.notifier = Notifier(config.notifications)

        # Paper trading state
        self.paper_balance = config.paper.initial_balance
        self.paper_trades: list[dict] = []

        self._last_daily_reset = datetime.utcnow().date()

        # Create logs dir
        Path("logs").mkdir(exist_ok=True)

    def run_cycle(self):
        """Run one full analysis and trading cycle."""
        logger.info("-" * 40)
        logger.info(f"Starting analysis cycle at {datetime.utcnow().isoformat()}")

        # Reset daily stats if new day
        today = datetime.utcnow().date()
        if today > self._last_daily_reset:
            self.risk.reset_daily_stats()
            self._last_daily_reset = today

        # Sync positions from exchange (demo and live mode)
        if self.mode in ("demo", "live"):
            self.orders.sync_positions_from_exchange()

        # 1. Gather technical analysis for all symbols
        technical_data = {}
        for symbol in config.trading.symbols:
            try:
                ohlcv_dict = {}
                for tf in config.trading.timeframes:
                    ohlcv_dict[tf] = self.exchange.fetch_ohlcv(symbol, tf, limit=200)
                technical_data[symbol] = self.analyzer.multi_timeframe_analysis(ohlcv_dict, symbol)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")

        if not technical_data:
            logger.warning("No technical data available, skipping cycle")
            return

        # 2. Gather news and market context
        market_context = self.news.get_market_context()

        # 3. Get portfolio state
        if self.mode in ("demo", "live"):
            balance = self.exchange.fetch_usdt_balance()
        else:
            balance = self.paper_balance

        portfolio = self.risk.get_portfolio_summary()

        logger.info(f"Balance: {balance:.2f} USDT | Positions: {portfolio['num_positions']}")

        # 4. Ask Claude AI for decisions
        decision = self.brain.analyze_and_decide(
            technical_data=technical_data,
            market_context=market_context,
            portfolio=portfolio,
            balance=balance,
        )

        logger.info(f"AI Outlook: {decision.get('market_outlook', 'N/A')}")
        logger.info(f"Risk Level: {decision.get('risk_level', 'N/A')}")

        # 5. Execute decisions
        actions = decision.get("decisions", [])
        for action in actions:
            self._execute_decision(action, balance)

    def _execute_decision(self, decision: dict, balance: float):
        """Execute a single AI trading decision."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")
        params = decision.get("params", {})

        if confidence < 0.6:
            logger.info(f"Skipping {symbol} {action}: low confidence ({confidence})")
            return

        logger.info(f"{'[PAPER] ' if self.mode == 'paper' else ''}Executing: {action} {symbol} (confidence={confidence})")
        logger.info(f"  Reason: {reason}")

        if self.mode == "paper":
            self._execute_paper(decision, balance)
        else:
            # demo and live both send real orders (demo goes to Bitget demo exchange)
            self._execute_live(decision, balance)

    def _execute_paper(self, decision: dict, balance: float):
        """Paper trading - log decisions without real orders."""
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
            atr = self._get_atr(symbol)
            stop_loss = self.risk.compute_stop_loss(price, side, atr)
            take_profit = self.risk.compute_take_profit(price, side, atr)
            amount = self.risk.calculate_position_size(balance, price, stop_loss)

            can_open, msg = self.risk.can_open_position(symbol, balance)
            if not can_open:
                logger.warning(f"[PAPER] Cannot open {symbol}: {msg}")
                return

            self.risk.register_position(symbol, side, price, amount, stop_loss, take_profit)

            trade = {
                "time": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "action": action,
                "price": price,
                "amount": amount,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "reason": reason,
            }
            self.paper_trades.append(trade)
            logger.info(f"[PAPER] {action} {symbol} @ {price} | Amount: {amount:.6f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        elif action == "close":
            if symbol in self.risk.positions:
                pnl = self.risk.close_position(symbol, price)
                self.paper_balance += pnl
                trade = {
                    "time": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "action": "close",
                    "price": price,
                    "pnl": pnl,
                    "confidence": confidence,
                    "reason": reason,
                }
                self.paper_trades.append(trade)
                logger.info(f"[PAPER] Close {symbol} @ {price} | PnL: {pnl:+.2f} USDT | Balance: {self.paper_balance:.2f}")

        elif action == "hold":
            logger.info(f"[PAPER] Hold {symbol}: {reason}")

        # Save paper trades to file
        self._save_paper_log()

        # Send notification
        msg = f"[PAPER] *{action.upper()}* {symbol}\nPrice: {price}\nConfidence: {confidence}\nReason: {reason}"
        self.notifier.send(msg)

    def _execute_live(self, decision: dict, balance: float):
        """Live trading - place real orders."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")
        params = decision.get("params", {})

        result = None

        try:
            if action == "open_long":
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker["last"]
                atr = self._get_atr(symbol)
                result = self.orders.open_long(symbol, balance, price, atr)

            elif action == "open_short":
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker["last"]
                atr = self._get_atr(symbol)
                result = self.orders.open_short(symbol, balance, price, atr)

            elif action == "close":
                result = self.orders.close_position(symbol)

            elif action == "update_sl":
                new_sl = params.get("new_stop_loss")
                if new_sl:
                    result = self.orders.update_stop_loss(symbol, new_sl)

            elif action == "limit_buy":
                limit_price = params.get("limit_price")
                if limit_price:
                    amount = self.risk.calculate_position_size(balance, limit_price, limit_price * 0.97)
                    result = self.orders.place_limit_order(symbol, "buy", amount, limit_price)

            elif action == "limit_sell":
                limit_price = params.get("limit_price")
                if limit_price:
                    amount = self.risk.calculate_position_size(balance, limit_price, limit_price * 1.03)
                    result = self.orders.place_limit_order(symbol, "sell", amount, limit_price)

            elif action == "hold":
                logger.info(f"Holding {symbol}: {reason}")

            if result:
                msg = f"*{action.upper()}* {symbol}\nConfidence: {confidence}\nReason: {reason}\nDetails: {result}"
                self.notifier.send(msg)
                logger.info(f"Executed: {result}")

        except Exception as e:
            logger.error(f"Failed to execute {action} for {symbol}: {e}")
            self.notifier.send(f"ERROR executing {action} {symbol}: {e}")

    def _get_atr(self, symbol: str) -> float | None:
        """Get latest ATR value for a symbol."""
        try:
            df = self.exchange.fetch_ohlcv(symbol, "1h", limit=50)
            df_analyzed = self.analyzer.compute_indicators(df)
            return float(df_analyzed["atr"].iloc[-1])
        except Exception:
            return None

    def _save_paper_log(self):
        """Save paper trading log to file."""
        Path("data").mkdir(exist_ok=True)
        log_file = Path("data/paper_trades.json")
        data = {
            "balance": self.paper_balance,
            "initial_balance": config.paper.initial_balance,
            "pnl_total": self.paper_balance - config.paper.initial_balance,
            "num_trades": len(self.paper_trades),
            "trades": self.paper_trades[-100:],  # Last 100 trades
        }
        log_file.write_text(json.dumps(data, indent=2))

    def run(self, once: bool = False):
        """Main loop."""
        self.notifier.send(f"Trading Agent STARTED ({self.mode.upper()} mode)")

        if once:
            self.run_cycle()
            logger.info("Single cycle complete. Exiting.")
            return

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                self.notifier.send(f"Trading Agent STOPPED ({self.mode.upper()} mode)")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                self.notifier.send(f"Cycle ERROR: {e}")

            # Wait for next cycle
            wait_seconds = config.trading.analysis_interval_minutes * 60
            logger.info(f"Next cycle in {config.trading.analysis_interval_minutes} minutes...")
            time.sleep(wait_seconds)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    parser.add_argument("--live", action="store_true", help="Run in live mode (real money!)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    mode = "live" if args.live else TRADING_MODE

    if mode == "live":
        logger.warning("!" * 60)
        logger.warning("  LIVE MODE - REAL MONEY AT RISK!")
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
