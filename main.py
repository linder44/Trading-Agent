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
from news.fetcher import NewsFetcher
from news.social import SocialSentiment
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
        logger.info("  АВТОНОМНЫЙ ИИ ТОРГОВЫЙ АГЕНТ v2.0")
        logger.info(f"  Режим: {self.mode.upper()}")
        if self.mode == "paper":
            logger.info(f"  Бумажный баланс: {config.paper.initial_balance} USDT")
        elif self.mode == "demo":
            logger.info("  Демо-счёт Bitget (виртуальные деньги)")
        logger.info(f"  Символы: {config.trading.symbols}")
        logger.info(f"  Интервал: {config.trading.analysis_interval_minutes} мин")
        logger.info("=" * 60)

        # Core modules
        self.exchange = ExchangeClient(config.bitget)
        self.analyzer = TechnicalAnalyzer()
        self.risk = RiskManager(config.trading)
        self.orders = OrderManager(self.exchange, self.risk)
        self.brain = TradingBrain(config.claude)
        self.notifier = Notifier(config.notifications)

        # Enhanced analysis modules
        self.patterns = PatternRecognizer()
        self.onchain = OnChainAnalyzer()
        self.correlations = MarketCorrelations()
        self.quant = QuantAnalyzer()
        self.news = NewsFetcher(config.news)
        self.social = SocialSentiment()

        # Валидация символов на бирже
        self.symbols = self.exchange.validate_symbols(config.trading.symbols)
        if not self.symbols:
            logger.error("Не найдено ни одного валидного символа! Проверь конфигурацию.")
            sys.exit(1)
        logger.info(f"Активные символы: {self.symbols}")

        # Paper trading state
        self.paper_balance = config.paper.initial_balance
        self.paper_trades: list[dict] = []

        self._last_daily_reset = datetime.now(timezone.utc).date()

        # Create dirs
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)

    def run_cycle(self):
        """Run one full analysis and trading cycle."""
        logger.info("-" * 40)
        logger.info(f"Начинаем цикл анализа: {datetime.now(timezone.utc).isoformat()}")

        # Reset daily stats if new day
        today = datetime.now(timezone.utc).date()
        if today > self._last_daily_reset:
            self.risk.reset_daily_stats()
            self._last_daily_reset = today

        # Sync positions from exchange (demo and live mode)
        if self.mode in ("demo", "live"):
            self.orders.sync_positions_from_exchange()

        # 1. Technical analysis (multi-timeframe)
        technical_data = {}
        ohlcv_cache = {}  # Cache for pattern analysis
        for symbol in self.symbols:
            try:
                ohlcv_dict = {}
                for tf in config.trading.timeframes:
                    ohlcv_dict[tf] = self.exchange.fetch_ohlcv(symbol, tf, limit=200)
                technical_data[symbol] = self.analyzer.multi_timeframe_analysis(ohlcv_dict, symbol)
                ohlcv_cache[symbol] = ohlcv_dict
            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")

        if not technical_data:
            logger.warning("Нет технических данных, пропускаем цикл")
            return

        # 2. Свечные паттерны, Фибоначчи, дивергенции
        logger.info("Анализируем паттерны, уровни Фибоначчи, дивергенции...")
        pattern_data = {}
        for symbol, ohlcv_dict in ohlcv_cache.items():
            pattern_data[symbol] = {}
            for tf, df in ohlcv_dict.items():
                df_with_indicators = self.analyzer.compute_indicators(df)
                pattern_data[symbol][tf] = self.patterns.get_full_pattern_analysis(df_with_indicators)

        # 2b. Количественный / научный анализ
        logger.info("Запускаем количественный анализ (Хёрст, Калман, FFT, VaR, энтропия)...")
        quant_data = {}
        for symbol, ohlcv_dict in ohlcv_cache.items():
            quant_data[symbol] = {}
            for tf, df in ohlcv_dict.items():
                quant_data[symbol][tf] = self.quant.full_quant_analysis(df)

        # 3. Ончейн и деривативы
        logger.info("Загружаем ончейн-данные (фандинг, OI, киты)...")
        onchain_data = self.onchain.get_full_onchain_data(self.exchange, self.symbols)

        # 4. Новости и фундаментальный контекст
        logger.info("Загружаем новости и рыночный контекст...")
        market_context = self.news.get_market_context()

        # 5. Социальные настроения
        logger.info("Загружаем социальные настроения (Reddit, CryptoPanic)...")
        social_data = self.social.get_full_social_data()

        # 6. Рыночные корреляции (DXY, S&P500, доминация BTC)
        logger.info("Загружаем рыночные корреляции (DXY, VIX, S&P500)...")
        correlation_data = self.correlations.get_full_correlation_data()

        # 7. Get portfolio state
        if self.mode in ("demo", "live"):
            balance = self.exchange.fetch_usdt_balance()
        else:
            balance = self.paper_balance

        portfolio = self.risk.get_portfolio_summary()

        logger.info(f"Баланс: {balance:.2f} USDT | Позиции: {portfolio['num_positions']}")

        # 8. Сводка качества данных
        self._log_data_quality(onchain_data, market_context, social_data, correlation_data)

        # 9. Отправляем ВСЁ в Claude AI для принятия решений
        logger.info("Отправляем данные в Claude AI для анализа...")
        decision = self.brain.analyze_and_decide(
            technical_data=technical_data,
            market_context=market_context,
            portfolio=portfolio,
            balance=balance,
            onchain_data=onchain_data,
            pattern_data=pattern_data,
            social_data=social_data,
            correlation_data=correlation_data,
            quant_data=quant_data,
        )

        logger.info(f"Прогноз ИИ: {decision.get('market_outlook', 'Н/Д')}")
        logger.info(f"Уровень риска: {decision.get('risk_level', 'Н/Д')}")

        # 9. Execute decisions
        actions = decision.get("decisions", [])
        for action in actions:
            self._execute_decision(action, balance)

    def _log_data_quality(self, onchain_data, market_context, social_data, correlation_data):
        """Log summary of what data was actually collected vs empty."""
        sources = {}

        # On-chain
        if onchain_data:
            market_wide = onchain_data.get("_market_wide", {})
            sources["whale_alerts"] = len(market_wide.get("whale_alerts", []))
            sources["exchange_netflow"] = market_wide.get("exchange_netflow", {}).get("signal", "unknown") != "unknown"
            # Check per-symbol data (sample first symbol)
            symbol_keys = [k for k in onchain_data if k != "_market_wide"]
            if symbol_keys:
                sample = onchain_data[symbol_keys[0]]
                sources["funding_rates"] = sample.get("funding_rate", {}).get("sentiment", "unknown") != "unknown"
                sources["open_interest"] = sample.get("open_interest", {}).get("open_interest_value_usd", 0) > 0
                sources["long_short_ratio"] = sample.get("long_short_ratio", {}).get("signal", "neutral") != "neutral" or sample.get("long_short_ratio", {}).get("ratio", 1.0) != 1.0
        else:
            sources["funding_rates"] = False
            sources["open_interest"] = False
            sources["long_short_ratio"] = False
            sources["whale_alerts"] = 0
            sources["exchange_netflow"] = False

        # News
        if market_context:
            sources["crypto_news"] = len(market_context.get("crypto_news", []))
            sources["geo_news"] = len(market_context.get("geopolitics_macro_news", []))
            sources["trending_coins"] = len(market_context.get("trending_coins", []))
            sources["fear_greed"] = market_context.get("fear_greed_index", {}).get("value", 50) != 50
        else:
            sources["crypto_news"] = 0
            sources["geo_news"] = 0
            sources["trending_coins"] = 0
            sources["fear_greed"] = False

        # Social
        if social_data:
            sources["cryptopanic"] = len(social_data.get("cryptopanic_hot", []))
            sources["reddit"] = len(social_data.get("reddit_sentiment", {}).get("top_discussions", []))
            sources["lunarcrush"] = len(social_data.get("social_trending", {}).get("trending_by_social", []))
        else:
            sources["cryptopanic"] = 0
            sources["reddit"] = 0
            sources["lunarcrush"] = 0

        # Correlations
        if correlation_data:
            sources["btc_dominance"] = correlation_data.get("btc_dominance", {}).get("btc_dominance", 0) > 0
            sources["stablecoin_market"] = bool(correlation_data.get("stablecoin_market"))
        else:
            sources["btc_dominance"] = False
            sources["stablecoin_market"] = False

        # Log summary
        loaded = []
        empty = []
        for name, val in sources.items():
            if isinstance(val, bool):
                (loaded if val else empty).append(name)
            elif isinstance(val, int):
                (loaded if val > 0 else empty).append(f"{name}({val})" if val > 0 else name)

        logger.info(f"Данные загружены: {', '.join(loaded) if loaded else 'нет'}")
        if empty:
            logger.warning(f"Данные ПУСТЫЕ/недоступны: {', '.join(empty)}")

    def _execute_decision(self, decision: dict, balance: float):
        """Execute a single AI trading decision."""
        symbol = decision.get("symbol")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")
        params = decision.get("params", {})

        if confidence < 0.6 and action != "hold":
            logger.info(f"Пропускаем {symbol} {action}: низкая уверенность ({confidence})")
            return

        logger.info(f"{'[БУМАГА] ' if self.mode == 'paper' else ''}Исполняем: {action} {symbol} (уверенность={confidence})")
        logger.info(f"  Причина: {reason}")

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
                logger.warning(f"[БУМАГА] Не могу открыть {symbol}: {msg}")
                return

            self.risk.register_position(symbol, side, price, amount, stop_loss, take_profit)

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
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
            logger.info(f"[БУМАГА] {action} {symbol} @ {price} | Объём: {amount:.6f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        elif action == "close":
            if symbol in self.risk.positions:
                pnl = self.risk.close_position(symbol, price)
                self.paper_balance += pnl
                trade = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "action": "close",
                    "price": price,
                    "pnl": pnl,
                    "confidence": confidence,
                    "reason": reason,
                }
                self.paper_trades.append(trade)
                logger.info(f"[БУМАГА] Закрытие {symbol} @ {price} | PnL: {pnl:+.2f} USDT | Баланс: {self.paper_balance:.2f}")

        elif action in ("trigger_long", "trigger_short"):
            trigger_price = decision.get("params", {}).get("trigger_price")
            if trigger_price:
                side = "long" if action == "trigger_long" else "short"
                logger.info(f"[БУМАГА] Триггерный ордер {side} {symbol} @ триггер {trigger_price} | Причина: {reason}")
                trade = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "action": action,
                    "trigger_price": trigger_price,
                    "confidence": confidence,
                    "reason": reason,
                }
                self.paper_trades.append(trade)

        elif action == "hold":
            logger.info(f"[БУМАГА] Удержание {symbol}: {reason}")

        # Save paper trades to file
        self._save_paper_log()

        msg = f"[БУМАГА] *{action.upper()}* {symbol}\nЦена: {price}\nУверенность: {confidence}\nПричина: {reason}"
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

            elif action == "trigger_long":
                trigger_price = params.get("trigger_price")
                if trigger_price:
                    result = self.orders.place_trigger_order(symbol, "long", balance, trigger_price)

            elif action == "trigger_short":
                trigger_price = params.get("trigger_price")
                if trigger_price:
                    result = self.orders.place_trigger_order(symbol, "short", balance, trigger_price)

            elif action == "hold":
                logger.info(f"Удержание {symbol}: {reason}")

            if result:
                msg = f"*{action.upper()}* {symbol}\nУверенность: {confidence}\nПричина: {reason}\nДетали: {result}"
                self.notifier.send(msg)
                logger.info(f"Исполнено: {result}")

        except Exception as e:
            logger.error(f"Ошибка исполнения {action} для {symbol}: {e}")
            self.notifier.send(f"ОШИБКА исполнения {action} {symbol}: {e}")

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
        """Main loop."""
        self.notifier.send(f"Торговый агент ЗАПУЩЕН (режим {self.mode.upper()})")

        if once:
            self.run_cycle()
            logger.info("Один цикл завершён. Выходим.")
            return

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Плавное завершение работы...")
                self.notifier.send(f"Торговый агент ОСТАНОВЛЕН (режим {self.mode.upper()})")
                break
            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")
                self.notifier.send(f"ОШИБКА цикла: {e}")

            wait_seconds = config.trading.analysis_interval_minutes * 60
            logger.info(f"Следующий цикл через {config.trading.analysis_interval_minutes} мин...")
            time.sleep(wait_seconds)


def main():
    parser = argparse.ArgumentParser(description="ИИ Торговый Агент")
    parser.add_argument("--live", action="store_true", help="Запуск в режиме live (реальные деньги!)")
    parser.add_argument("--once", action="store_true", help="Выполнить один цикл и выйти")
    args = parser.parse_args()

    mode = "live" if args.live else TRADING_MODE

    if mode == "live":
        logger.warning("!" * 60)
        logger.warning("  LIVE РЕЖИМ — РЕАЛЬНЫЕ ДЕНЬГИ ПОД УГРОЗОЙ!")
        logger.warning("  Нажми Ctrl+C в течение 5 секунд для отмены...")
        logger.warning("!" * 60)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Отменено.")
            return

    agent = TradingAgent(mode=mode)
    agent.run(once=args.once)


if __name__ == "__main__":
    main()
