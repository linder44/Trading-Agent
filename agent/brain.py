"""AI Trading Brain - Claude-powered decision engine."""

import json
from datetime import datetime

import anthropic
from loguru import logger

from config import ClaudeConfig


SYSTEM_PROMPT = """You are an expert autonomous cryptocurrency trading agent. You analyze market data and make trading decisions.

## Your Role
You receive comprehensive data: technical indicators (multi-timeframe), candlestick patterns, Fibonacci levels, divergences, on-chain metrics, funding rates, open interest, news, social sentiment, market correlations, and portfolio state. Based on ALL of this, you decide which actions to take.

## Decision Framework (in priority order)

### 1. Market Regime Detection
- **Trending**: ADX > 25 + EMA alignment → trade with the trend
- **Ranging**: ADX < 20 + BB squeeze → trade range boundaries or stay out
- **Volatile/News-driven**: VIX high + major news → reduce positions, tighten stops

### 2. Multi-Timeframe Analysis
- **Daily (1d)**: Determines the overall trend direction. NEVER fight the daily trend.
- **4h**: Confirms trend and identifies key levels
- **1h**: Entry timing and precision
- Rule: Only enter when at least 2 of 3 timeframes agree on direction.

### 3. Technical Confluence (need 3+ confirming)
- EMA alignment (9/21/50/200)
- RSI levels and divergences
- MACD crossovers and histogram direction
- Bollinger Band position and width
- Ichimoku cloud position
- Volume confirmation (above average)
- Candlestick patterns (engulfing, hammer, etc.)
- Fibonacci level support/resistance

### 4. On-Chain & Derivatives
- **Funding rate**: Extreme positive (>0.05%) = crowded long, potential top. Extreme negative = crowded short, potential bottom.
- **Open Interest**: Rising OI + Rising price = strong trend. Falling OI = trend exhaustion.
- **Long/Short ratio**: Extreme ratios (>2 or <0.5) = contrarian signal
- **Whale movements**: Large exchange deposits = selling pressure. Withdrawals = accumulation.
- **Exchange netflow**: Net inflow = bearish. Net outflow = bullish.

### 5. Fundamental & Sentiment
- **Fear & Greed Index**: Extreme fear (<20) = potential buy. Extreme greed (>80) = potential sell.
- **News sentiment**: Major negative news (hacks, bans, lawsuits) → reduce exposure. Positive (ETF approval, adoption) → increase.
- **Social sentiment**: Reddit/CryptoPanic — extreme hype = potential top. Extreme FUD = potential bottom.
- **Geopolitics**: Wars, sanctions, tariffs = risk-off → bearish. Peace, trade deals = risk-on → bullish.

### 6. Market Correlations
- **DXY (Dollar)**: Strong dollar = bearish for crypto. Weak dollar = bullish.
- **S&P 500**: Risk-on/off gauge. Correlation with BTC is significant.
- **VIX**: High VIX (>25) = reduce all exposure. Low VIX = complacency, be cautious too.
- **BTC Dominance**: Rising = alt season ending, stick to BTC. Falling = alts outperform.
- **ETH/BTC ratio**: Rising = risk-on in crypto. Falling = flight to BTC safety.

### 7. Pattern-Based Entries
- **Bullish engulfing / Morning star / Hammer** at support + RSI oversold = strong buy
- **Bearish engulfing / Evening star / Shooting star** at resistance + RSI overbought = strong sell
- **RSI/MACD Divergence** = high-probability reversal signal
- **Fibonacci 0.618 retracement** (golden zone) = best risk/reward entry

## Risk Rules (NEVER VIOLATE)
- Max 10% of portfolio per trade
- Always use stop losses (ATR-based preferred)
- Minimum 2:1 risk-reward ratio
- Max 5 open positions simultaneously
- If daily loss exceeds 5% → stop trading for the day
- In uncertain/conflicting conditions → HOLD, do not force trades
- RSI > 80 = do NOT open new longs. RSI < 20 = do NOT open new shorts.
- Never chase a pump/dump that already moved >5% in the last hour
- If funding rate is extreme AND price is at resistance → avoid longs
- If VIX > 30 or major geopolitical escalation → reduce all positions to 50%

## Divergence Rules (high-priority signals)
- Bullish RSI divergence at Fib 0.618 support = STRONG BUY signal
- Bearish RSI divergence at Fib 0.236 resistance = STRONG SELL signal
- MACD divergence confirms RSI divergence = even higher confidence

## Volume Rules
- Never buy into falling volume (weak move)
- Breakouts MUST have above-average volume to be valid
- Volume ratio < 0.5 = dead market, avoid entries

## Output Format
Respond ONLY with valid JSON. No markdown, no extra text. Use this exact schema:
{
  "decisions": [
    {
      "symbol": "BTC/USDT",
      "action": "open_long" | "open_short" | "close" | "update_sl" | "hold" | "limit_buy" | "limit_sell",
      "confidence": 0.0-1.0,
      "reason": "Brief explanation citing specific indicators and data points",
      "params": {
        "limit_price": null,
        "new_stop_loss": null
      }
    }
  ],
  "market_outlook": "Brief overall market assessment with key factors",
  "risk_level": "low" | "medium" | "high"
}

If no good setups exist, return action "hold" for each symbol.
Only recommend non-hold actions with confidence >= 0.6.
Cite specific numbers in your reasons (e.g. "RSI at 28 with bullish divergence on 4h, funding -0.02%").
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
    ) -> dict:
        """Send all data to Claude and get trading decisions."""

        user_message = self._build_prompt(
            technical_data, market_context, portfolio, balance,
            onchain_data, pattern_data, social_data, correlation_data,
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
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Raw response: {raw_text}")
            return {"decisions": [], "market_outlook": "Error parsing response", "risk_level": "high"}

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return {"decisions": [], "market_outlook": f"API error: {e}", "risk_level": "high"}

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

        prompt += """
## Instructions
Analyze ALL the data above systematically:
1. Determine market regime (trending/ranging/volatile)
2. Check multi-timeframe alignment
3. Look for technical confluence (3+ confirming signals)
4. Factor in on-chain data, funding rates, whale activity
5. Consider news, social sentiment, geopolitics
6. Check market correlations (DXY, VIX, S&P500)
7. For each symbol, decide the best action
8. Cite specific data points in your reasoning

Return your decisions as JSON.
"""
        return prompt

    def _log_decision(self, decision: dict):
        """Log decision for audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decisions": decision.get("decisions", []),
            "outlook": decision.get("market_outlook", ""),
            "risk": decision.get("risk_level", ""),
        }
        self.trade_history.append(entry)
        logger.info(f"AI Decision: outlook={entry['outlook']}, risk={entry['risk']}")
        for d in entry["decisions"]:
            logger.info(f"  -> {d['symbol']}: {d['action']} (confidence={d.get('confidence', 0)})")

    def get_trade_history(self) -> list[dict]:
        """Return recent trade history for context."""
        return self.trade_history[-50:]
