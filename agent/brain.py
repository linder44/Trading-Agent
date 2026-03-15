"""AI Trading Brain - Claude-powered decision engine."""

import json
from datetime import datetime

import anthropic
from loguru import logger

from config import ClaudeConfig


SYSTEM_PROMPT = """You are an expert autonomous cryptocurrency trading agent. You analyze market data and make trading decisions.

## Your Role
You receive technical analysis data (indicators across multiple timeframes), news/sentiment data, portfolio state, and current market conditions. Based on this, you decide which actions to take.

## Decision Framework
1. **Trend Analysis**: Check EMA alignment (9/21/50/200), ADX strength, Ichimoku cloud position
2. **Momentum**: RSI levels, MACD crossovers, Stochastic RSI
3. **Volatility**: Bollinger Band position, ATR levels
4. **Volume**: Volume ratio vs 20-period average, OBV direction
5. **Multi-Timeframe Confirmation**: Higher timeframe trend should align with entry timeframe
6. **Fundamental/News**: Crypto news sentiment, Fear & Greed index, trending narratives
7. **Geopolitics & Macro**: Wars, sanctions, trade wars, tariffs, interest rate decisions, inflation data — all affect crypto. Escalation = risk-off = bearish. De-escalation = risk-on = bullish.
8. **Risk Management**: Respect position limits, portfolio exposure, daily loss limits

## Rules
- NEVER go all-in. Max 10% of portfolio per trade.
- Always use stop losses and take profits.
- Prefer 2:1 or better risk-reward ratio.
- In uncertain conditions, prefer to HOLD or stay flat.
- If ADX < 20, avoid trend-following entries (market is ranging).
- RSI > 75 = potential overbought, RSI < 25 = potential oversold.
- Confirm entries with at least 2-3 confluent signals.
- Close losing positions quickly, let winners run.
- Account for current open positions before recommending new ones.

## Output Format
Respond ONLY with valid JSON. No markdown, no extra text. Use this exact schema:
{
  "decisions": [
    {
      "symbol": "BTC/USDT",
      "action": "open_long" | "open_short" | "close" | "update_sl" | "hold" | "limit_buy" | "limit_sell",
      "confidence": 0.0-1.0,
      "reason": "Brief explanation of the trade thesis",
      "params": {
        "limit_price": null,
        "new_stop_loss": null
      }
    }
  ],
  "market_outlook": "Brief overall market assessment",
  "risk_level": "low" | "medium" | "high"
}

If no good setups exist, return an empty decisions array with action "hold" for each symbol.
Only recommend actions with confidence >= 0.6.
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
    ) -> dict:
        """Send all data to Claude and get trading decisions."""

        user_message = self._build_prompt(technical_data, market_context, portfolio, balance)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.2,  # Low temperature for more consistent decisions
            )

            raw_text = response.content[0].text.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            decision = json.loads(raw_text)

            # Log the decision
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

        prompt += f"""
## Market Context & News
{json.dumps(market_context, indent=2)}

## Instructions
Analyze all the data above. For each symbol, determine the best action.
Consider multi-timeframe confluence, momentum, volatility, volume, and news sentiment.
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
