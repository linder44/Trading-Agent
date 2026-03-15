"""
Autonomous AI Trading Agent
Main entry point and orchestration loop.

Usage:
    python main.py              # Run with default config
    python main.py --live       # Switch to live trading (override .env)
    python main.py --once       # Run analysis once and exit
"""

import argparse
import sys
import time
from datetime import datetime, timedelta

from loguru import logger

from config import config
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

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  AUTONOMOUS AI TRADING AGENT")
        logger.info(f"  Mode: {'TESTNET' if config.bitget.sandbox else 'LIVE'}")
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

        self._last_daily_reset = datetime.utcnow().date()

    def run_cycle(self):
        """Run one full analysis and trading cycle."""
        logger.info("-" * 40)
        logger.info(f"Starting analysis cycle at {datetime.utcnow().isoformat()}")

        # Reset daily stats if new day
        today = datetime.utcnow().date()
        if today > self._last_daily_reset:
            self.risk.reset_daily_stats()
            self._last_daily_reset = today

        # Sync positions from exchange
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
        balance = self.exchange.fetch_usdt_balance()
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

        logger.info(f"Executing: {action} {symbol} (confidence={confidence})")
        logger.info(f"  Reason: {reason}")

        result = None

        try:
            if action == "open_long":
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker["last"]
                # Get ATR from latest analysis for dynamic SL/TP
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
                msg = f"*{action.upper()}* {symbol}\n"
                msg += f"Confidence: {confidence}\n"
                msg += f"Reason: {reason}\n"
                msg += f"Details: {result}"
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

    def run(self, once: bool = False):
        """Main loop."""
        self.notifier.send("Trading Agent STARTED")

        if once:
            self.run_cycle()
            return

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                self.notifier.send("Trading Agent STOPPED (manual)")
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
    parser.add_argument("--live", action="store_true", help="Run in live mode (real money)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    if args.live:
        config.bitget.sandbox = False
        logger.warning("LIVE MODE ENABLED - REAL MONEY AT RISK!")

    agent = TradingAgent()
    agent.run(once=args.once)


if __name__ == "__main__":
    main()
