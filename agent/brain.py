"""AI Trading Brain - Claude-powered decision engine."""

import json
import re
from datetime import datetime

import anthropic
from loguru import logger

from config import ClaudeConfig


def _repair_truncated_json(raw: str) -> dict | None:
    """Попытка восстановить обрезанный JSON-ответ от Claude.

    Если ответ обрезан по max_tokens, JSON будет неполным.
    Пытаемся извлечь уже полные решения из массива decisions.
    """
    try:
        # Ищем все полностью завершённые объекты в массиве decisions
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

        # Пытаемся извлечь market_outlook и risk_level
        outlook_match = re.search(r'"market_outlook"\s*:\s*"([^"]*)"', raw)
        risk_match = re.search(r'"risk_level"\s*:\s*"([^"]*)"', raw)

        result = {
            "decisions": decisions,
            "market_outlook": outlook_match.group(1) if outlook_match else "Ответ был обрезан, данные частичные",
            "risk_level": risk_match.group(1) if risk_match else "medium",
        }
        logger.warning(f"JSON был обрезан. Восстановлено {len(decisions)} решений из неполного ответа.")
        return result
    except Exception:
        return None


SYSTEM_PROMPT = """Ты — экспертный автономный агент для торговли криптовалютами. Ты анализируешь рыночные данные и принимаешь торговые решения.

ВАЖНО: Все текстовые поля в ответе (reason, market_outlook) пиши ТОЛЬКО на русском языке.

## Твоя роль
Ты получаешь комплексные данные: технические индикаторы (мульти-таймфрейм), свечные паттерны, уровни Фибоначчи, дивергенции, ончейн-метрики, ставки финансирования, открытый интерес, новости, социальные настроения, рыночные корреляции, КОЛИЧЕСТВЕННЫЙ/НАУЧНЫЙ анализ (экспонента Хёрста, фильтр Калмана, FFT-циклы, VaR, энтропия, z-score, автокорреляция) и состояние портфеля. На основании ВСЕХ этих данных ты решаешь, какие действия предпринять.

## Фреймворк принятия решений (по приоритету)

### 1. Определение рыночного режима
- **Трендовый**: ADX > 25 + выравнивание EMA → торгуй по тренду
- **Боковой**: ADX < 20 + сжатие BB → торгуй от границ диапазона или оставайся вне рынка
- **Волатильный/Новостной**: высокий VIX + крупные новости → сокращай позиции, сужай стопы

### 2. Мульти-таймфрейм анализ
- **Дневной (1d)**: Определяет общее направление тренда. НИКОГДА не торгуй против дневного тренда.
- **4h**: Подтверждение тренда и ключевые уровни
- **1h**: Точность входа и тайминг
- Правило: Входи только когда минимум 2 из 3 таймфреймов совпадают по направлению.

### 3. Техническая конфлюэнция (нужно 3+ подтверждающих)
- Выравнивание EMA (9/21/50/200)
- Уровни RSI и дивергенции
- Пересечения MACD и направление гистограммы
- Позиция и ширина Bollinger Bands
- Позиция облака Ichimoku
- Подтверждение объёмом (выше среднего)
- Свечные паттерны (поглощение, молот и т.д.)
- Поддержка/сопротивление по уровням Фибоначчи

### 4. Ончейн и деривативы
- **Ставка финансирования**: Экстремально положительная (>0.05%) = перенасыщенный лонг, возможная вершина. Экстремально отрицательная = перенасыщенный шорт, возможное дно.
- **Открытый интерес**: Растущий OI + рост цены = сильный тренд. Падающий OI = истощение тренда.
- **Соотношение Long/Short**: Экстремальные значения (>2 или <0.5) = контрарный сигнал
- **Движения китов**: Крупные депозиты на биржу = давление продаж. Вывод = накопление.
- **Нетфлоу бирж**: Чистый приток = медвежий. Чистый отток = бычий.

### 5. Фундаментальный анализ и настроения
- **Индекс страха и жадности**: Экстремальный страх (<20) = потенциальная покупка. Экстремальная жадность (>80) = потенциальная продажа.
- **Новостные настроения**: Крупный негатив (взломы, запреты, иски) → сокращай экспозицию. Позитив (одобрение ETF, принятие) → увеличивай.
- **Социальные настроения**: Reddit/CryptoPanic — экстремальный хайп = возможная вершина. Экстремальный FUD = возможное дно.
- **Геополитика**: Войны, санкции, тарифы = risk-off → медвежий. Мир, торговые сделки = risk-on → бычий.

### 6. Рыночные корреляции
- **DXY (Доллар)**: Сильный доллар = медвежий для крипты. Слабый доллар = бычий.
- **S&P 500**: Индикатор risk-on/off. Корреляция с BTC значительна.
- **VIX**: Высокий VIX (>25) = сокращай всю экспозицию. Низкий VIX = самоуспокоенность, тоже осторожно.
- **Доминация BTC**: Растёт = альтсезон заканчивается, держись BTC. Падает = альты обгоняют.
- **ETH/BTC**: Растёт = risk-on в крипте. Падает = бегство в BTC.

### 7. Входы по паттернам
- **Бычье поглощение / Утренняя звезда / Молот** на поддержке + RSI перепродан = сильный сигнал на покупку
- **Медвежье поглощение / Вечерняя звезда / Падающая звезда** на сопротивлении + RSI перекуплен = сильный сигнал на продажу
- **Дивергенция RSI/MACD** = высоковероятный сигнал разворота
- **Ретрейсмент Фибоначчи 0.618** (золотая зона) = лучшее соотношение риск/доход

### 8. Количественный / Научный анализ (ВЫСОКИЙ ПРИОРИТЕТ — математически строгие)

**Экспонента Хёрста** (фрактальный анализ, Мандельброт 1969):
- H > 0.6 → рынок ТРЕНДОВЫЙ → используй моментум/трендовые стратегии (EMA-кроссы, пробои)
- H < 0.4 → рынок ВОЗВРАЩАЕТСЯ К СРЕДНЕМУ → торгуй против экстремумов, покупай перепроданное, продавай перекупленное
- H ≈ 0.5 → СЛУЧАЙНОЕ БЛУЖДАНИЕ → статистического преимущества нет, СОКРАЩАЙ экспозицию или оставайся вне рынка
- Это САМЫЙ ВАЖНЫЙ индикатор режима. Он определяет, какой ТИП стратегии работает.

**Z-Score** (статистическое отклонение от среднего):
- |Z| > 2.0 → цена на 2+ стандартных отклонения от среднего → 95% вероятность возврата
- |Z| > 3.0 → экстремум (99.7%) → очень сильная возможность возврата к среднему
- Используй Z-score для тайминга входа при Hurst < 0.5 (режим возврата к среднему)

**Фильтр Калмана** (оптимальная фильтрация сигнала, Калман 1960):
- Калман-цена = математически оптимальная оценка «справедливой стоимости»
- Цена значительно выше Калмана → переоценена относительно справедливой стоимости
- Цена значительно ниже Калмана → недооценена относительно справедливой стоимости
- Скорость Калмана показывает направление тренда точнее скользящих средних

**FFT-циклы** (спектральный анализ Фурье):
- Выявляет скрытые периодичности в ценовых данных
- При наличии доминирующего цикла — входи вблизи минимумов цикла, выходи вблизи максимумов
- cycle_bottom_zone = потенциальная зона покупки, cycle_top_zone = потенциальная зона продажи

**Канал линейной регрессии** (МНК, Гаусс 1809):
- R² > 0.8 → сильный тренд, доверяй трендовым сигналам
- R² < 0.3 → тренда нет, используй стратегии бокового рынка
- Позиция > +2σ → цена выше верхнего канала → статистически перекуплена
- Позиция < -2σ → цена ниже нижнего канала → статистически перепродана

**Автокорреляция** (Бокс-Дженкинс 1976):
- Положительная ACF lag-1 → моментум существует → торгуй по тренду
- Отрицательная ACF lag-1 → возврат к среднему → торгуй против движения
- Незначимая → доходности случайны → преимущества нет

**Энтропия Шеннона** (теория информации, Шеннон 1948):
- Высокая энтропия → хаотичный/непредсказуемый рынок → сокращай размеры позиций
- Низкая энтропия → упорядоченный/структурированный → сигналы надёжнее, нормальный размер

**EWMA-волатильность** (RiskMetrics / GARCH, Боллерслев 1986):
- Всплеск волатильности → сокращай позиции, сужай стопы
- Сжатие волатильности → пробой близок, готовь входы
- Используй для динамического размера позиции: выше волатильность = меньше позиция

**Коэффициент эффективности** (Кауфман 1995):
- ER > 0.6 → цена движется эффективно (тренд) → торгуй по тренду
- ER < 0.3 → шумно/рвано → избегай трендовых входов

**Value at Risk (VaR)** (Базель III, Артцнер 1999):
- 95% VaR показывает макс. ожидаемый убыток в день с 95% уверенностью
- Используй max_recommended_position_pct для размера позиции
- CVaR (хвостовой риск) > 5% → высокий риск, сокращай всю экспозицию

**Консенсус режима** (поле _regime_consensus):
- Объединяет ВСЕ научные индикаторы в единое голосование по режиму
- ВСЕГДА проверяй это ПЕРВЫМ — говорит, торговать по тренду, против тренда или оставаться вне рынка
- Если режим = "high_risk" → перекрой все сигналы, сокращай экспозицию

## Правила риска (НИКОГДА НЕ НАРУШАЙ)
- Максимум 10% портфеля на сделку
- Всегда используй стоп-лоссы (предпочтительно на основе ATR)
- Минимальное соотношение риск/прибыль 2:1
- Максимум 5 открытых позиций одновременно
- Если дневной убыток превышает 5% → прекрати торговлю на сегодня
- При неопределённых/конфликтующих условиях → HOLD, не навязывай сделки
- RSI > 80 = НЕ открывай новые лонги. RSI < 20 = НЕ открывай новые шорты.
- Никогда не гонись за пампом/дампом, который уже сделал >5% за последний час
- Если ставка финансирования экстремальная И цена на сопротивлении → избегай лонгов
- Если VIX > 30 или крупная геополитическая эскалация → сокращай все позиции до 50%

## Правила дивергенций (высокоприоритетные сигналы)
- Бычья дивергенция RSI на поддержке Фибо 0.618 = СИЛЬНЫЙ сигнал на покупку
- Медвежья дивергенция RSI на сопротивлении Фибо 0.236 = СИЛЬНЫЙ сигнал на продажу
- Дивергенция MACD подтверждает дивергенцию RSI = ещё выше уверенность

## Правила объёма
- Никогда не покупай при падающем объёме (слабое движение)
- Пробои ДОЛЖНЫ быть на объёме выше среднего, чтобы быть валидными
- Volume ratio < 0.5 = мёртвый рынок, избегай входов

## Формат вывода
Отвечай ТОЛЬКО валидным JSON. Без markdown, без лишнего текста. Используй эту точную схему:
{
  "decisions": [
    {
      "symbol": "BTC/USDT",
      "action": "open_long" | "open_short" | "close" | "update_sl" | "hold" | "limit_buy" | "limit_sell" | "trigger_long" | "trigger_short",
      "confidence": 0.0-1.0,
      "reason": "Краткое обоснование на русском языке с конкретными индикаторами и данными",
      "params": {
        "limit_price": null,
        "trigger_price": null,
        "new_stop_loss": null
      }
    }
  ],
  "market_outlook": "Краткая общая оценка рынка на русском языке с ключевыми факторами",
  "risk_level": "low" | "medium" | "high"
}

### Типы ордеров:
- **open_long / open_short** — немедленный вход по рынку
- **trigger_long** — триггерный ордер на лонг: когда цена достигнет trigger_price, откроется лонг по рынку. Используй когда ждёшь откат к уровню (например, "жду откат до 72500") или пробой уровня вверх.
- **trigger_short** — триггерный ордер на шорт: когда цена достигнет trigger_price, откроется шорт по рынку. Используй для входа при пробое вниз или отскоке от сопротивления.
- **limit_buy / limit_sell** — лимитные ордера по указанной цене
- **close** — закрыть позицию
- **update_sl** — обновить стоп-лосс
- **hold** — ничего не делать

Если хороших сетапов нет, возвращай action "hold" для каждого символа.
Рекомендуй действия (не hold) только с confidence >= 0.6.
Указывай конкретные числа в обосновании (например, "RSI на 28 с бычьей дивергенцией на 4ч, фандинг -0.02%").
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
    ) -> dict:
        """Send all data to Claude and get trading decisions."""

        user_message = self._build_prompt(
            technical_data, market_context, portfolio, balance,
            onchain_data, pattern_data, social_data, correlation_data,
            quant_data,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.2,
            )

            raw_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
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
            # Попытка восстановить обрезанный JSON
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
    ) -> str:
        """Build the analysis prompt with all market data."""

        prompt = f"""## Current Time
{datetime.utcnow().isoformat()} UTC

## Account Balance
USDT Available: {balance:.2f}

## Portfolio State
{json.dumps(portfolio, indent=2)}

## Technical Analysis (Multi-Timeframe)
"""
        for symbol, timeframes in technical_data.items():
            prompt += f"\n### {symbol}\n"
            for tf, data in timeframes.items():
                prompt += f"\n**{tf} Timeframe:**\n{json.dumps(data, indent=2)}\n"

        if pattern_data:
            prompt += f"\n## Candlestick Patterns, Fibonacci & Divergences\n{json.dumps(pattern_data, indent=2)}\n"

        if onchain_data:
            prompt += f"\n## On-Chain & Derivatives Data\n{json.dumps(onchain_data, indent=2)}\n"

        prompt += f"\n## News & Fundamental Context\n{json.dumps(market_context, indent=2)}\n"

        if social_data:
            prompt += f"\n## Social Sentiment (Reddit, CryptoPanic)\n{json.dumps(social_data, indent=2)}\n"

        if correlation_data:
            prompt += f"\n## Market Correlations (DXY, S&P500, BTC Dominance)\n{json.dumps(correlation_data, indent=2)}\n"

        if quant_data:
            prompt += f"\n## Quantitative / Scientific Analysis (Hurst, Kalman, FFT, VaR, Entropy, Z-Score)\n{json.dumps(quant_data, indent=2)}\n"

        # Report missing/empty data sources so Claude knows what's unavailable
        missing = self._detect_missing_data(
            onchain_data, market_context, social_data, correlation_data, quant_data,
        )
        if missing:
            prompt += "\n## ⚠ Недоступные источники данных\n"
            prompt += "Следующие источники не вернули данные (API недоступен или пуст). "
            prompt += "Принимай решения БЕЗ этих данных, опирайся на доступные источники:\n"
            for source in missing:
                prompt += f"- {source}\n"

        prompt += """
## Instructions
Analyze ALL the data above systematically:
1. CHECK REGIME CONSENSUS FIRST (from quantitative analysis) — this determines your strategy type
2. Determine market regime (trending/ranging/volatile) using Hurst + ADX + Efficiency Ratio
3. Check multi-timeframe alignment
4. Look for technical confluence (3+ confirming signals)
5. Use Z-score and Kalman filter for entry precision
6. Factor in on-chain data, funding rates, whale activity
7. Consider news, social sentiment, geopolitics
8. Check market correlations (DXY, VIX, S&P500)
9. Use VaR for position sizing, Shannon entropy for confidence adjustment
10. For each symbol, decide the best action
11. Cite specific scientific indicators in your reasoning (Hurst value, Z-score, VaR, etc.)
12. In your reason field, EXPLICITLY reference data from on-chain, social, correlation sections — not just technical/quant

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
        """Detect which data sources returned empty or missing data."""
        missing = []

        if not onchain_data:
            missing.append("On-chain данные (фандинг, OI, киты, нетфлоу) — полностью недоступны")
        else:
            market_wide = onchain_data.get("_market_wide", {})
            if not market_wide.get("whale_alerts"):
                missing.append("Whale Alerts (крупные транзакции китов)")
            if market_wide.get("exchange_netflow", {}).get("signal") == "unknown":
                missing.append("Exchange Netflow (приток/отток с бирж)")

        if not market_context:
            missing.append("Новости и фундаментальный контекст — полностью недоступны")
        else:
            if not market_context.get("crypto_news"):
                missing.append("Крипто-новости (NewsAPI)")
            if not market_context.get("geopolitics_macro_news"):
                missing.append("Геополитические/макро новости")
            if not market_context.get("trending_coins"):
                missing.append("Трендовые монеты (CoinGecko)")

        if not social_data:
            missing.append("Социальные настроения — полностью недоступны")
        else:
            if not social_data.get("cryptopanic_hot"):
                missing.append("CryptoPanic (горячие новости)")
            if not social_data.get("reddit_sentiment", {}).get("top_discussions"):
                missing.append("Reddit (дискуссии крипто-сообщества)")
            if not social_data.get("social_trending", {}).get("trending_by_social"):
                missing.append("LunarCrush (социальный тренд)")

        if not correlation_data:
            missing.append("Рыночные корреляции — полностью недоступны")
        else:
            tradfi = correlation_data.get("traditional_markets", {})
            if "DXY" not in tradfi:
                missing.append("DXY (индекс доллара)")
            if "VIX" not in tradfi:
                missing.append("VIX (индекс страха)")
            if "SPY" not in tradfi:
                missing.append("S&P 500")
            if correlation_data.get("btc_dominance", {}).get("btc_dominance", 0) == 0:
                missing.append("BTC Dominance")

        if not quant_data:
            missing.append("Количественный анализ (Хёрст, Калман, FFT, VaR) — полностью недоступен")

        return missing

    def _log_decision(self, decision: dict):
        """Log decision for audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decisions": decision.get("decisions", []),
            "outlook": decision.get("market_outlook", ""),
            "risk": decision.get("risk_level", ""),
        }
        self.trade_history.append(entry)
        logger.info(f"Решение ИИ: прогноз={entry['outlook']}, риск={entry['risk']}")
        for d in entry["decisions"]:
            logger.info(f"  -> {d['symbol']}: {d['action']} (уверенность={d.get('confidence', 0)})")

    def get_trade_history(self) -> list[dict]:
        """Return recent trade history for context."""
        return self.trade_history[-50:]
