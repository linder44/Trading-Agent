"""On-chain and derivatives data module.

Fetches data that is not available from standard OHLCV:
- Funding rates (sentiment of futures traders)
- Open interest (how much money is in the market)
- Liquidations (forced closures, cascade potential)
- Whale alerts (large transactions)
- Exchange inflows/outflows (selling/buying pressure)
"""

import time
from datetime import datetime

import requests
from loguru import logger


class OnChainAnalyzer:
    """Fetches on-chain and derivatives market data."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 300  # 5 min

    def get_funding_rate(self, exchange_client, symbol: str) -> dict:
        """Get current funding rate from exchange.

        Positive = longs pay shorts (market is long-heavy, potential top)
        Negative = shorts pay longs (market is short-heavy, potential bottom)
        """
        try:
            funding = exchange_client.exchange.fetch_funding_rate(symbol)
            rate = float(funding.get("fundingRate", 0))
            return {
                "funding_rate": round(rate, 6),
                "funding_rate_pct": round(rate * 100, 4),
                "sentiment": "extreme_greed" if rate > 0.001 else (
                    "bullish" if rate > 0 else (
                        "extreme_fear" if rate < -0.001 else "bearish"
                    )
                ),
            }
        except Exception as e:
            logger.warning(f"Funding rate fetch failed for {symbol}: {e}")
            return {"funding_rate": 0, "funding_rate_pct": 0, "sentiment": "unknown"}

    def get_open_interest(self, exchange_client, symbol: str) -> dict:
        """Get open interest data.

        Rising OI + Rising price = strong trend (new money entering)
        Rising OI + Falling price = bearish pressure
        Falling OI + Rising price = short squeeze / weak rally
        Falling OI + Falling price = capitulation
        """
        try:
            oi = exchange_client.exchange.fetch_open_interest(symbol)
            oi_value = float(oi.get("openInterestValue", 0))
            oi_amount = float(oi.get("openInterestAmount", 0))
            return {
                "open_interest_value_usd": round(oi_value, 2),
                "open_interest_amount": round(oi_amount, 4),
            }
        except Exception as e:
            logger.warning(f"Open interest fetch failed for {symbol}: {e}")
            return {"open_interest_value_usd": 0, "open_interest_amount": 0}

    def get_long_short_ratio(self, exchange_client, symbol: str) -> dict:
        """Get long/short ratio from exchange.

        Ratio > 1 = more longs than shorts
        Extreme ratios often signal reversals (contrarian indicator)
        """
        try:
            # Try fetching via ccxt if supported
            ratio_data = exchange_client.exchange.fetch_long_short_ratio_history(symbol, limit=1)
            if ratio_data:
                latest = ratio_data[-1]
                long_pct = float(latest.get("longAccount", 0.5))
                short_pct = float(latest.get("shortAccount", 0.5))
                ratio = long_pct / short_pct if short_pct > 0 else 1.0
                return {
                    "long_pct": round(long_pct * 100, 1),
                    "short_pct": round(short_pct * 100, 1),
                    "ratio": round(ratio, 2),
                    "signal": "contrarian_bearish" if ratio > 2.0 else (
                        "contrarian_bullish" if ratio < 0.5 else "neutral"
                    ),
                }
        except Exception as e:
            logger.warning(f"Long/short ratio fetch failed for {symbol}: {e}")

        return {"long_pct": 50, "short_pct": 50, "ratio": 1.0, "signal": "neutral"}

    def get_whale_alerts(self) -> list[dict]:
        """Fetch recent large crypto transactions from Whale Alert (free tier)."""
        cache_key = "whale_alerts"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            # Whale Alert free API - last large transactions
            resp = requests.get(
                "https://api.whale-alert.io/v1/transactions",
                params={
                    "api_key": "free",  # Free tier
                    "min_value": 1000000,  # $1M+
                    "limit": 10,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                txs = resp.json().get("transactions", [])
                result = [
                    {
                        "symbol": tx.get("symbol", "").upper(),
                        "amount_usd": tx.get("amount_usd", 0),
                        "from": tx.get("from", {}).get("owner_type", "unknown"),
                        "to": tx.get("to", {}).get("owner_type", "unknown"),
                        "type": self._classify_whale_tx(tx),
                    }
                    for tx in txs
                ]
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Whale alert fetch failed: {e}")

        return []

    def get_exchange_netflow(self) -> dict:
        """Get exchange net flow data from CryptoQuant (public endpoints).

        Inflow to exchanges = selling pressure (bearish)
        Outflow from exchanges = accumulation (bullish)
        """
        cache_key = "exchange_netflow"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            # CoinGlass public API for exchange flow
            resp = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/exchange_netflow",
                params={"symbol": "BTC", "time_type": "h4"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    latest = data[-1] if isinstance(data, list) else data
                    netflow = float(latest.get("value", 0))
                    result = {
                        "btc_netflow": netflow,
                        "signal": "selling_pressure" if netflow > 0 else "accumulation",
                    }
                    self._set_cache(cache_key, result)
                    return result
        except Exception as e:
            logger.warning(f"Exchange netflow fetch failed: {e}")

        return {"btc_netflow": 0, "signal": "unknown"}

    def get_full_onchain_data(self, exchange_client, symbols: list[str]) -> dict:
        """Get all on-chain/derivatives data for all symbols."""
        result = {}
        for symbol in symbols:
            result[symbol] = {
                "funding_rate": self.get_funding_rate(exchange_client, symbol),
                "open_interest": self.get_open_interest(exchange_client, symbol),
                "long_short_ratio": self.get_long_short_ratio(exchange_client, symbol),
            }

        result["_market_wide"] = {
            "whale_alerts": self.get_whale_alerts(),
            "exchange_netflow": self.get_exchange_netflow(),
        }

        return result

    def _classify_whale_tx(self, tx: dict) -> str:
        """Classify whale transaction type."""
        from_type = tx.get("from", {}).get("owner_type", "")
        to_type = tx.get("to", {}).get("owner_type", "")

        if from_type == "exchange" and to_type == "unknown":
            return "exchange_withdrawal (bullish)"
        elif from_type == "unknown" and to_type == "exchange":
            return "exchange_deposit (bearish)"
        elif from_type == "exchange" and to_type == "exchange":
            return "exchange_transfer (neutral)"
        else:
            return "wallet_transfer (neutral)"

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
