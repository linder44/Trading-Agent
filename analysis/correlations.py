"""Market correlation data.

Tracks assets correlated with crypto to provide broader market context:
- BTC Dominance: % of total crypto market cap that is Bitcoin
- Stablecoin market cap: proxy for capital in crypto
- Total crypto market cap change: overall market direction
"""

import time
from datetime import datetime

import requests
from loguru import logger

from utils.http import request_with_retry


class MarketCorrelations:
    """Fetches broader market data that correlates with crypto."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 min

    def get_btc_dominance(self) -> dict:
        """Get BTC dominance and global market data from CoinGecko.

        Rising dominance = altcoins underperforming (alt season ending)
        Falling dominance = altcoins outperforming (alt season)
        """
        cache_key = "btc_dominance"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry("https://api.coingecko.com/api/v3/global")
            if not resp:
                return {"btc_dominance": 0, "signal": "unknown"}
            data = resp.json().get("data", {})
            market_cap_pct = data.get("market_cap_percentage", {})

            result = {
                "btc_dominance": round(market_cap_pct.get("btc", 0), 2),
                "eth_dominance": round(market_cap_pct.get("eth", 0), 2),
                "total_market_cap_usd": data.get("total_market_cap", {}).get("usd", 0),
                "total_volume_24h_usd": data.get("total_volume", {}).get("usd", 0),
                "market_cap_change_24h": round(data.get("market_cap_change_percentage_24h_usd", 0), 2),
                "active_cryptocurrencies": data.get("active_cryptocurrencies", 0),
                "signal": "alt_season" if market_cap_pct.get("btc", 50) < 40 else (
                    "btc_dominant" if market_cap_pct.get("btc", 50) > 60 else "balanced"
                ),
            }
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"BTC dominance fetch failed: {e}")
            return {"btc_dominance": 0, "signal": "unknown"}

    def get_stablecoin_market(self) -> dict:
        """Get stablecoin market data from CoinGecko.

        Growing stablecoin mcap = capital waiting on sidelines (bullish potential).
        Shrinking stablecoin mcap = capital leaving crypto (bearish).
        """
        cache_key = "stablecoin_market"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "tether,usd-coin",
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true",
                },
            )
            if resp:
                data = resp.json()
                usdt = data.get("tether", {})
                usdc = data.get("usd-coin", {})

                usdt_mcap = usdt.get("usd_market_cap", 0)
                usdc_mcap = usdc.get("usd_market_cap", 0)
                total_stable_mcap = usdt_mcap + usdc_mcap

                result = {
                    "usdt_market_cap": round(usdt_mcap / 1e9, 2),  # in billions
                    "usdc_market_cap": round(usdc_mcap / 1e9, 2),
                    "total_stablecoin_cap_b": round(total_stable_mcap / 1e9, 2),
                    "usdt_24h_vol": round(usdt.get("usd_24h_vol", 0) / 1e9, 2),
                }
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Stablecoin market fetch failed: {e}")

        return {}

    def get_full_correlation_data(self, exchange_client=None) -> dict:
        """Get all market correlation data."""
        data = {
            "btc_dominance": self.get_btc_dominance(),
            "stablecoin_market": self.get_stablecoin_market(),
            "fetched_at": datetime.utcnow().isoformat(),
        }
        return data

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
