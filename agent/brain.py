"""AI Trading Brain - Claude-powered decision engine for SCALPING.

Optimized: sends aggregated signal scores + top reasons instead of raw data.
Prompt cut from 18 sections to focused scalping rules.
Includes trade history feedback loop.
Falls back to rule-based SignalAggregator when Claude API is unavailable.
"""

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


# ─────────────────────────────────────────────────────────────
# OPTIMIZED SYSTEM PROMPT — cut from 18 sections to focused rules
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ты — скальпинг AI-трейдер. Твоя ЕДИНСТВЕННАЯ цель: положительный PnL.
Все текстовые поля (reason, market_outlook) — ТОЛЬКО на русском.

ПРАВИЛА ВХОДА (в порядке приоритета):
1. SPREAD: если spread > 0.15% → HOLD. Не обсуждается.
2. REGIME: если режим = choppy → HOLD. Не пытайся найти сделку.
3. TIER 1 КОНФЛИКТ: если order_flow и momentum дают разные направления → HOLD.
4. RVOL: если rvol < 0.5 → HOLD. Нет объёма = нет движения.
5. CONFIRMATION: для входа нужно минимум 2 из 3: order_flow + momentum + tape_signal.

ТИПЫ СДЕЛОК:
- MOMENTUM: trend strong, EMA aligned, delta stacked → вход по тренду, trailing stop
- REVERSAL: exhaustion delta + absorption + VWAP band extreme → контр-тренд, тайтовый TP
- BREAKOUT: squeeze regime + volume surge + EMA alignment → вход на пробое

УПРАВЛЕНИЕ ПОЗИЦИЯМИ:
- Позиция в плюсе > 0.3% → передвинуть SL в безубыток (update_sl)
- Позиция во флэте > 45 мин → закрыть (close)
- Позиция в минусе > 45 мин → закрыть
- Позиция старше 90 мин → закрыть
- Если regime сменился на противоположный → закрыть

ПРАВИЛА РИСКА:
- Макс 8% портфеля на сделку
- Макс 5 позиций
- Risk/Reward минимум 1.5:1
- Не гонись за движением > 1% за 5 мин
- При неясных сигналах → HOLD

Формат: ТОЛЬКО валидный JSON.
{
  "decisions": [{"symbol":"...","action":"open_long|open_short|close|update_sl|hold","confidence":0.0-1.0,"reason":"конкретные числа","params":{"trigger_price":null,"new_stop_loss":null}}],
  "market_outlook": "краткая оценка на русском",
  "risk_level": "low|medium|high"
}
Рекомендуй действия (не hold) только с confidence >= 0.6.
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
        # NEW: pre-aggregated signals from SignalAggregator
        aggregated_signals: dict | None = None,
        regime_data: dict | None = None,
        tape_data: dict | None = None,
        vwap_data: dict | None = None,
        delta_data: dict | None = None,
        drawdown_status: dict | None = None,
    ) -> dict:
        """Send aggregated data to Claude and get trading decisions."""

        user_message = self._build_prompt(
            technical_data, portfolio, balance,
            onchain_data, scalping_data,
            aggregated_signals, regime_data, tape_data,
            vwap_data, delta_data,
            trade_history_data, time_context_data,
            drawdown_status,
        )

        prompt_chars = len(SYSTEM_PROMPT) + len(user_message)
        prompt_tokens_est = prompt_chars // 3
        logger.info(f"Prompt size: ~{prompt_chars:,} chars (~{prompt_tokens_est:,} tokens)")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.2,
            )

            usage = response.usage
            logger.info(f"Tokens: in={usage.input_tokens:,}, out={usage.output_tokens:,}")

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
            logger.warning(f"JSON parse error: {e}")
            repaired = _repair_truncated_json(raw_text)
            if repaired:
                self._log_decision(repaired)
                return repaired
            logger.error(f"Cannot repair JSON. Raw: {raw_text[:500]}...")
            return {"decisions": [], "market_outlook": "Ошибка парсинга", "risk_level": "high"}

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"decisions": [], "market_outlook": f"API error: {e}", "risk_level": "high"}

    def _build_prompt(
        self,
        technical_data: dict,
        portfolio: dict,
        balance: float,
        onchain_data: dict | None,
        scalping_data: dict | None,
        aggregated_signals: dict | None,
        regime_data: dict | None,
        tape_data: dict | None,
        vwap_data: dict | None,
        delta_data: dict | None,
        trade_history_data: dict | None,
        time_context_data: dict | None,
        drawdown_status: dict | None,
    ) -> str:
        """Build optimized prompt — aggregated signals + context, NOT raw data."""

        prompt = f"""## Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}
## Balance: {balance:.2f} USDT
## Portfolio: {_compact_json(portfolio)}
"""

        # Aggregated signals (Tier 1-2 pre-computed) — PRIMARY DATA
        if aggregated_signals:
            prompt += "\n## Signal Aggregation (per symbol)\n"
            for symbol, sig in aggregated_signals.items():
                prompt += f"**{symbol}**: verdict={sig.get('verdict')}, confidence={sig.get('confidence')}, "
                prompt += f"score={sig.get('weighted_score')}, tier1_conflict={sig.get('tier1_conflict')}\n"
                for_reasons = sig.get('top_reasons_for', [])
                against_reasons = sig.get('top_reasons_against', [])
                if for_reasons:
                    prompt += f"  FOR: {'; '.join(for_reasons[:3])}\n"
                if against_reasons:
                    prompt += f"  AGAINST: {'; '.join(against_reasons[:3])}\n"

        # Regime per symbol
        if regime_data:
            prompt += "\n## Market Regime\n"
            for symbol, regime in regime_data.items():
                prompt += f"**{symbol}**: regime={regime.get('regime')}, strategy={regime.get('strategy')}, "
                prompt += f"ADX={regime.get('metrics', {}).get('adx', '?')}, ER={regime.get('metrics', {}).get('efficiency_ratio', '?')}\n"

        # Tier 1-2 raw data (only scalping microstructure)
        if scalping_data:
            prompt += f"\n## Scalping Microstructure\n{_compact_json(scalping_data)}\n"

        # Tape reading
        if tape_data:
            prompt += f"\n## Tape Reading\n"
            for symbol, tape in tape_data.items():
                verdict = tape.get("tape_verdict", {})
                prompt += f"**{symbol}**: tape={verdict.get('signal', 'n/a')}, "
                prompt += f"intensity={tape.get('trade_intensity', {}).get('signal', '?')}, "
                prompt += f"flow={tape.get('aggressive_flow', {}).get('signal', '?')}\n"

        # VWAP bands
        if vwap_data:
            prompt += f"\n## VWAP Bands\n"
            for symbol, vwap in vwap_data.items():
                prompt += f"**{symbol}**: deviation={vwap.get('deviation_sigma', 0):.2f}σ, signal={vwap.get('signal', '?')}\n"

        # Delta divergence
        if delta_data:
            prompt += f"\n## Delta Analysis\n"
            for symbol, delta in delta_data.items():
                verdict = delta.get("delta_verdict", {})
                prompt += f"**{symbol}**: delta={verdict.get('signal', 'n/a')}, stacked={delta.get('stacked_delta', {}).get('signal', '?')}\n"

        # Tier 3 context (one line each)
        if onchain_data:
            prompt += "\n## On-Chain Context (Tier 3)\n"
            for symbol, data in onchain_data.items():
                if symbol == "_market_wide":
                    continue
                funding = data.get("funding_rate", {})
                prompt += f"{symbol}: funding={funding.get('sentiment', '?')}, "
                prompt += f"OI={data.get('open_interest', {}).get('open_interest_value_usd', 0):.0f}\n"

        if time_context_data:
            session = time_context_data.get("session", {})
            prompt += f"\n## Session: {', '.join(session.get('active', []))} | volatility={session.get('volatility_expected', '?')}\n"

        # Drawdown status
        if drawdown_status:
            prompt += f"\n## Risk Status: daily_pnl={drawdown_status.get('daily_pnl_pct', 0):+.1f}%, "
            prompt += f"consecutive_losses={drawdown_status.get('consecutive_losses', 0)}, "
            prompt += f"min_confidence={drawdown_status.get('min_confidence', 0.6)}\n"

        # Trade history feedback (last 5 trades)
        if trade_history_data and trade_history_data.get("total_trades", 0) > 0:
            prompt += "\n## Recent Trades (learn from these)\n"
            recent = trade_history_data.get("recent_trades", [])[-5:]
            for t in recent:
                prompt += f"  {t.get('symbol')} {t.get('side')} {t.get('pnl_pct', 0):+.1f}% ({t.get('result')})\n"
            prompt += f"Win rate: {trade_history_data.get('win_rate_pct', 0):.0f}%, "
            prompt += f"PF: {trade_history_data.get('profit_factor', 0):.1f}\n"
            worst = trade_history_data.get("worst_symbols", [])
            if worst:
                prompt += f"Worst symbols: {', '.join(w['symbol'] for w in worst)} — avoid these\n"

        prompt += "\nReturn trading decisions as JSON. If Tier 1 signals conflict — HOLD.\n"
        return prompt

    def _log_decision(self, decision: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisions": decision.get("decisions", []),
            "outlook": decision.get("market_outlook", ""),
            "risk": decision.get("risk_level", ""),
        }
        self.trade_history.append(entry)
        logger.info(f"AI decision: outlook={entry['outlook']}, risk={entry['risk']}")
        for d in entry["decisions"]:
            logger.info(f"  -> {d['symbol']}: {d['action']} (confidence={d.get('confidence', 0)})")

    def get_trade_history(self) -> list[dict]:
        return self.trade_history[-50:]
