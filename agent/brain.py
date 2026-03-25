"""AI Trading Brain - Claude-powered decision engine for SCALPING."""

import json
import math
import re
from datetime import datetime, timezone

import anthropic
from loguru import logger

from config import ClaudeConfig


def _compact_json(data, indent: int | None = None) -> str:
    """Serialize data to compact JSON, stripping noise."""
    cleaned = _strip_empty(data)
    return json.dumps(cleaned, indent=indent, ensure_ascii=False, default=str)


def _is_empty(v) -> bool:
    if v is None:
        return True
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.size == 0
    except ImportError:
        pass
    if isinstance(v, (list, dict, str)):
        return len(v) == 0
    if isinstance(v, (int, float)):
        try:
            if math.isnan(v) or math.isinf(v):
                return True
        except (TypeError, ValueError):
            pass
        return False
    return False


def _strip_empty(obj):
    """Recursively remove None, empty lists, empty dicts, NaN, and round floats."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if _is_empty(v):
                continue
            cleaned_v = _strip_empty(v)
            if not _is_empty(cleaned_v):
                cleaned[k] = cleaned_v
        return cleaned
    if isinstance(obj, list):
        return [_strip_empty(item) for item in obj if not _is_empty(item)]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        if abs(obj) >= 1:
            return round(obj, max(0, 4 - len(str(int(abs(obj))))))
        return round(obj, 6)
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return _strip_empty(float(obj))
        if isinstance(obj, np.ndarray):
            return _strip_empty(obj.tolist())
    except ImportError:
        pass
    return obj


def _repair_truncated_json(raw: str) -> dict | None:
    try:
        decisions = []
        pattern = r'\{\s*"symbol"\s*:.*?"params"\s*:\s*\{[^}]*\}\s*\}'
        for m in re.finditer(pattern, raw, re.DOTALL):
            try:
                obj = json.loads(m.group())
                decisions.append(obj)
            except json.JSONDecodeError:
                continue
        if not decisions:
            return None
        outlook_match = re.search(r'"market_outlook"\s*:\s*"([^"]*)"', raw)
        risk_match = re.search(r'"risk_level"\s*:\s*"([^"]*)"', raw)
        result = {
            "decisions": decisions,
            "market_outlook": outlook_match.group(1) if outlook_match else "Ответ был обрезан",
            "risk_level": risk_match.group(1) if risk_match else "medium",
        }
        logger.warning(f"JSON обрезан. Восстановлено {len(decisions)} решений.")
        return result
    except Exception:
        return None


SYSTEM_PROMPT = """Ты — скальпинг-трейдер криптовалют. Ты открываешь ТОЛЬКО короткие сделки длительностью до 2 часов. Цель — маленькие быстрые профиты 0.3-1.0%.

ВАЖНО: Все текстовые поля (reason, market_outlook) пиши ТОЛЬКО на русском языке.

## Режим работы: СКАЛЬПИНГ (1m / 5m / 15m)
- Таймфреймы: 1m (вход), 5m (подтверждение), 15m (контекст тренда)
- Время в сделке: 5-120 минут (МАКСИМУМ 2 часа)
- Цель прибыли: 0.3-1.0% за сделку
- Стоп-лосс: 0.3-0.5% (тайтовый)
- Risk/Reward: минимум 1.5:1

## Приоритет данных для скальпинга

### 1. Scalp Signal (СМОТРИ ПЕРВЫМ)
- scalp_signal.verdict: strong_buy/buy/neutral/sell/strong_sell
- scalp_signal.quality: good/risky/poor — если poor, НЕ ВХОДИ
- Это агрегация order flow + momentum + price action + spread

### 2. Order Flow Imbalance
- imbalance > 0.3 = сильное давление покупателей → лонг
- imbalance < -0.3 = сильное давление продавцов → шорт
- balanced = нет преимущества, жди

### 3. Micro Momentum
- strong_bullish_burst + volume_factor > 1.2 = немедленный лонг
- strong_bearish_burst + volume_factor > 1.2 = немедленный шорт
- consolidation = жди пробоя

### 4. Быстрые EMA (3/5/8/13/21)
- EMA3 > EMA8 > EMA21 = scalp_trend bullish → лонг
- EMA3 < EMA8 < EMA21 = scalp_trend bearish → шорт
- mixed = нет чёткого тренда, осторожно

### 5. Быстрый MACD (5, 13, 4)
- macd_fast_crossover = bullish → сигнал на лонг
- macd_fast_crossover = bearish → сигнал на шорт
- Гистограмма растёт = моментум усиливается

### 6. RSI-3 (ультра-быстрый)
- RSI-3 < 15 = экстремальная перепроданность → лонг
- RSI-3 > 85 = экстремальная перекупленность → шорт
- RSI-3 20-80 = нормальная зона

### 7. Volatility Regime
- expanding = высокая волатильность → тайтовые стопы, меньше размер
- contracting = сжатие → готовься к пробою
- normal = стандартные параметры

### 8. Spread Estimation
- wide_spread = плохая ликвидность → НЕ ВХОДИ (плохие заполнения)
- tight_spread = хорошие условия для скальпинга

### 9. Price Action (микро-паттерны)
- bullish_pin_bar на поддержке → лонг
- bearish_pin_bar на сопротивлении → шорт
- bullish_engulfing / bearish_engulfing → сильные сигналы
- inside_bar → готовься к пробою

### 10. Cumulative Volume Delta (CVD)
- bullish_confirmed = цена и CVD растут → сильный тренд, входи по направлению
- bearish_confirmed = цена и CVD падают → сильный даунтренд
- distribution (bearish_divergence) = цена растёт, CVD падает → фейковый рост, НЕ лонг
- accumulation (bullish_divergence) = цена падает, CVD растёт → фейковый дамп, ищи лонг

### 11. Whale Order Following
- whale_buying (bias > 0.4) = крупные покупки → ищи лонг на уровнях китов
- whale_selling (bias < -0.4) = крупные продажи → ищи шорт
- trigger_levels = цены входа китов → выставляй trigger_long/trigger_short на этих уровнях
- buy_walls = сильная поддержка, sell_walls = сильное сопротивление

### 12. Order Book Depth
- strong_bid_dominance → покупатели доминируют, поддержка цены → лонг
- strong_ask_dominance → продавцы доминируют, давление сверху → шорт
- weighted_imbalance учитывает расстояние от цены (ближние уровни важнее)
- balanced = нет преимущества

### 13. Breakout Quality Filter
- confirmed_*_breakout (quality >= 5) = настоящий пробой → входи
- weak_*_breakout = фейковый пробой → НЕ ВХОДИ
- volume_ratio > 1.5x + body_ratio > 0.6 = качественный пробой

### 14. Multi-Timeframe Trend Score
- score > 0.7 = сильный бычий сетап → лонг с высокой уверенностью
- score < -0.7 = сильный медвежий → шорт
- agreement = true → все TF совпадают, лучший сетап

### 15. Количественный анализ
- _regime_consensus: trending → торгуй по тренду, mean_reverting → от экстремумов
- Z-score |Z| > 2.0 → возврат к среднему
- Kalman: цена далеко от kalman_price → возможен откат
- Hurst > 0.6 → моментум работает, < 0.4 → mean reversion

### 16. Мульти-таймфрейм для скальпинга
- 15m определяет НАПРАВЛЕНИЕ (не торгуй против 15m тренда)
- 5m подтверждает сетап
- 1m даёт точку входа
- Минимум 2 из 3 таймфреймов должны совпадать

### 17. Ончейн (контекст)
- Экстремальный фандинг (>0.05%) → осторожно с лонгами
- Рост OI + рост цены → сильный тренд, скальпируй по нему
- Ликвидации extreme → НЕ ВХОДИ, жди стабилизации

### 18. Управление открытыми позициями
- Позиция старше 90 минут → рекомендуй ЗАКРЫТЬ (close)
- Позиция в прибыли > 0.5% → подтяни стоп (update_sl)
- Позиция в убытке и возраст > 60 мин → закрой, не жди

## Правила риска (НИКОГДА НЕ НАРУШАЙ)
- Макс 8% портфеля на сделку
- Макс 5 позиций одновременно
- Всегда стоп-лосс (1.5x ATR или 0.5% фиксированный)
- Risk/Reward минимум 1.5:1
- Дневной убыток > 5% → СТОП торговли
- RSI > 85 = НЕ лонг. RSI < 15 = НЕ шорт
- wide_spread → НЕ ВХОДИ
- Не гонись за движением > 1% за последние 5 минут
- При неясных сигналах → HOLD

## Формат вывода
Отвечай ТОЛЬКО валидным JSON:
{
  "decisions": [
    {
      "symbol": "BTC/USDT",
      "action": "open_long" | "open_short" | "close" | "update_sl" | "hold" | "trigger_long" | "trigger_short",
      "confidence": 0.0-1.0,
      "reason": "Краткое обоснование на русском с конкретными числами индикаторов",
      "params": {
        "trigger_price": null,
        "new_stop_loss": null
      }
    }
  ],
  "market_outlook": "Краткая оценка рынка на русском",
  "risk_level": "low" | "medium" | "high"
}

Типы ордеров:
- open_long/open_short — вход по рынку СЕЙЧАС
- trigger_long/trigger_short — вход когда цена дойдёт до trigger_price
- close — закрыть позицию
- update_sl — подтянуть стоп-лосс
- hold — ничего не делать

Рекомендуй действия (не hold) только с confidence >= 0.6.
Указывай конкретные числа: "RSI-3=12, EMA3>EMA8, order flow +0.35, spread tight".
"""


class TradingBrain:
    """Claude-powered trading decision engine."""

    def __init__(self, cfg: ClaudeConfig):
        self.client = anthropic.Anthropic(api_key=cfg.api_key)
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.trade_history: list[dict] = []

    def analyze_and_decide(
        self,
        technical_data: dict[str, dict],
        market_context: dict,
        portfolio: dict,
        balance: float,
        onchain_data: dict | None = None,
        pattern_data: dict | None = None,
        social_data: dict | None = None,
        correlation_data: dict | None = None,
        quant_data: dict | None = None,
        liquidation_data: dict | None = None,
        cross_corr_data: dict | None = None,
        time_context_data: dict | None = None,
        trade_history_data: dict | None = None,
        scalping_data: dict | None = None,
        whale_data: dict | None = None,
        orderbook_data: dict | None = None,
        mtf_scores: dict | None = None,
        performance_data: dict | None = None,
    ) -> dict:
        """Send all data to Claude and get trading decisions."""

        user_message = self._build_prompt(
            technical_data, market_context, portfolio, balance,
            onchain_data, pattern_data, social_data, correlation_data,
            quant_data, liquidation_data, cross_corr_data,
            time_context_data, trade_history_data, scalping_data,
            whale_data, orderbook_data, mtf_scores, performance_data,
        )

        prompt_chars = len(SYSTEM_PROMPT) + len(user_message)
        prompt_tokens_est = prompt_chars // 3
        logger.info(f"Размер промпта: ~{prompt_chars:,} символов (~{prompt_tokens_est:,} токенов)")

        logger.info("=" * 80)
        logger.info("ТЕКСТ ПРОМПТА, ОТПРАВЛЯЕМЫЙ В CLAUDE:")
        logger.info("=" * 80)
        logger.info(f"[SYSTEM PROMPT] ({len(SYSTEM_PROMPT)} символов)")
        logger.info(SYSTEM_PROMPT)
        logger.info("-" * 80)
        logger.info(f"[USER MESSAGE] ({len(user_message)} символов)")
        logger.info(user_message)
        logger.info("=" * 80)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.2,
            )

            usage = response.usage
            logger.info(f"Использовано токенов: вход={usage.input_tokens:,}, выход={usage.output_tokens:,}")

            raw_text = response.content[0].text.strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            decision = json.loads(raw_text)
            self._log_decision(decision)
            return decision

        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга JSON-ответа Claude: {e}")
            repaired = _repair_truncated_json(raw_text)
            if repaired:
                self._log_decision(repaired)
                return repaired
            logger.error(f"Не удалось восстановить JSON. Сырой ответ: {raw_text[:500]}...")
            return {"decisions": [], "market_outlook": "Ошибка парсинга ответа", "risk_level": "high"}

        except Exception as e:
            logger.error(f"Ошибка вызова Claude API: {e}")
            return {"decisions": [], "market_outlook": f"Ошибка API: {e}", "risk_level": "high"}

    def _build_prompt(
        self,
        technical_data: dict[str, dict],
        market_context: dict,
        portfolio: dict,
        balance: float,
        onchain_data: dict | None = None,
        pattern_data: dict | None = None,
        social_data: dict | None = None,
        correlation_data: dict | None = None,
        quant_data: dict | None = None,
        liquidation_data: dict | None = None,
        cross_corr_data: dict | None = None,
        time_context_data: dict | None = None,
        trade_history_data: dict | None = None,
        scalping_data: dict | None = None,
        whale_data: dict | None = None,
        orderbook_data: dict | None = None,
        mtf_scores: dict | None = None,
        performance_data: dict | None = None,
    ) -> str:
        """Build the analysis prompt with all market data."""

        prompt = f"""## Current Time
{datetime.now(timezone.utc).isoformat()} UTC

## Account Balance
USDT Available: {balance:.2f}

## Portfolio State
{_compact_json(portfolio, indent=1)}

## Technical Analysis (Multi-Timeframe: 1m, 5m, 15m)
"""
        for symbol, timeframes in technical_data.items():
            prompt += f"\n### {symbol}\n"
            for tf, data in timeframes.items():
                prompt += f"**{tf}:** {_compact_json(data)}\n"

        if scalping_data:
            prompt += f"\n## Scalping Microstructure (order flow, CVD, momentum, breakout, spread)\n{_compact_json(scalping_data)}\n"

        if whale_data:
            prompt += f"\n## Whale Order Following (large trades + trigger levels)\n{_compact_json(whale_data)}\n"

        if orderbook_data:
            prompt += f"\n## Order Book Depth (bid/ask imbalance)\n{_compact_json(orderbook_data)}\n"

        if mtf_scores:
            prompt += f"\n## Multi-Timeframe Trend Score\n{_compact_json(mtf_scores)}\n"

        if pattern_data:
            prompt += f"\n## Candlestick Patterns, Fibonacci & Divergences\n{_compact_json(pattern_data)}\n"

        if onchain_data:
            prompt += f"\n## On-Chain & Derivatives Data\n{_compact_json(onchain_data)}\n"

        prompt += f"\n## News & Fundamental Context\n{_compact_json(market_context, indent=1)}\n"

        if social_data:
            prompt += f"\n## Social Trends & Sector Rotation\n{_compact_json(social_data, indent=1)}\n"

        if correlation_data:
            prompt += f"\n## Market Correlations\n{_compact_json(correlation_data, indent=1)}\n"

        if quant_data:
            prompt += f"\n## Quantitative Analysis (Hurst, Kalman, FFT, VaR, Z-Score)\n{_compact_json(quant_data)}\n"

        if liquidation_data:
            prompt += f"\n## Liquidation Data\n{_compact_json(liquidation_data)}\n"

        if cross_corr_data:
            prompt += f"\n## Cross-Symbol Correlation\n{_compact_json(cross_corr_data, indent=1)}\n"

        if time_context_data:
            prompt += f"\n## Time & Session Context\n{_compact_json(time_context_data, indent=1)}\n"

        if trade_history_data and trade_history_data.get("total_trades", 0) > 0:
            prompt += f"\n## Trade History\n{_compact_json(trade_history_data, indent=1)}\n"

        if performance_data and performance_data.get("status") == "ok":
            prompt += f"\n## Trade Performance Analysis (best hours, durations, symbol edge)\n{_compact_json(performance_data, indent=1)}\n"

        missing = self._detect_missing_data(
            onchain_data, market_context, social_data, correlation_data, quant_data,
        )
        if missing:
            prompt += "\n## Недоступные источники данных\n"
            prompt += "Принимай решения БЕЗ этих данных:\n"
            for source in missing:
                prompt += f"- {source}\n"

        prompt += """
## Instructions (SCALPING MODE)
Analyze data for SHORT-TERM scalping (max 2 hours):
1. CHECK scalp_signal verdict FIRST
2. CHECK CVD for divergences (distribution/accumulation = avoid/counter-trade)
3. CHECK whale data — whale trigger_levels = use trigger_long/trigger_short at those prices
4. CHECK MTF trend score — |score| > 0.7 = high confidence setup
5. CHECK breakout quality — only enter confirmed breakouts (quality >= 5)
6. CHECK orderbook imbalance — confirms/denies direction
7. Verify timeframe alignment (1m + 5m + 15m)
8. Look for fast EMA crosses (3/8/21) and fast MACD
9. Use RSI-3 for extremes
10. Spread wide = DO NOT ENTER
11. Positions older than 90 min = recommend CLOSE
12. Use performance data to avoid worst symbols/hours
13. Cite specific indicator values in reason

Return your decisions as JSON.
"""
        return prompt

    @staticmethod
    def _detect_missing_data(
        onchain_data: dict | None,
        market_context: dict | None,
        social_data: dict | None,
        correlation_data: dict | None,
        quant_data: dict | None,
    ) -> list[str]:
        missing = []
        if not onchain_data:
            missing.append("On-chain данные")
        else:
            market_wide = onchain_data.get("_market_wide", {})
            if not market_wide.get("whale_alerts"):
                missing.append("Whale Alerts")
            if market_wide.get("exchange_netflow", {}).get("signal") == "unknown":
                missing.append("Exchange Netflow")
        if not market_context:
            missing.append("Новости")
        else:
            if not market_context.get("crypto_news"):
                missing.append("Крипто-новости")
            if not market_context.get("trending_coins"):
                missing.append("Трендовые монеты")
        if not social_data:
            missing.append("Социальные данные")
        if not correlation_data:
            missing.append("Рыночные корреляции")
        if not quant_data:
            missing.append("Количественный анализ")
        return missing

    def _log_decision(self, decision: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisions": decision.get("decisions", []),
            "outlook": decision.get("market_outlook", ""),
            "risk": decision.get("risk_level", ""),
        }
        self.trade_history.append(entry)
        logger.info(f"Решение ИИ: прогноз={entry['outlook']}, риск={entry['risk']}")
        for d in entry["decisions"]:
            logger.info(f"  -> {d['symbol']}: {d['action']} (уверенность={d.get('confidence', 0)})")

    def get_trade_history(self) -> list[dict]:
        return self.trade_history[-50:]
