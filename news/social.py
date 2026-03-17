"""Social sentiment and market momentum sources.

All sources are free and require no API keys:
- CoinGecko trending: social momentum proxy (most searched coins)
- CoinGecko categories: capital flow into DeFi, Meme, L1, L2 sectors
"""

import time
from datetime import datetime, timezone

import requests
from loguru import logger

from utils.http import request_with_retry


class SocialSentiment:
    """Fetches social momentum and sector rotation data."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 min cache

    def fetch_coingecko_trending(self) -> dict:
        """Fetch trending search coins from CoinGecko (free, no key).

        Acts as a social momentum proxy — coins trending on CoinGecko
        are getting attention from retail traders.
        """
        cache_key = "cg_trending"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry("https://api.coingecko.com/api/v3/search/trending")
            if resp:
                coins = resp.json().get("coins", [])
                result = {
                    "trending_by_social": [
                        {
                            "symbol": c["item"]["symbol"],
                            "name": c["item"]["name"],
                            "market_cap_rank": c["item"].get("market_cap_rank", 0),
                            "score": c["item"].get("score", 0),
                        }
                        for c in coins[:10]
                    ],
                }
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"CoinGecko trending fetch failed: {e}")

        return {"trending_by_social": []}

    def fetch_sector_performance(self) -> dict:
        """Fetch top crypto category performance from CoinGecko.

        Shows where capital is flowing: DeFi, Meme coins, L1, L2, etc.
        Rising category = sector rotation into that narrative.
        """
        cache_key = "sector_perf"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry("https://api.coingecko.com/api/v3/coins/categories")
            if resp:
                categories = resp.json()
                # Pick key sectors relevant to trading
                key_sectors = {
                    "layer-1": "Layer 1",
                    "layer-2": "Layer 2",
                    "decentralized-finance-defi": "DeFi",
                    "meme-token": "Meme Coins",
                    "artificial-intelligence": "AI Tokens",
                    "gaming": "Gaming/Metaverse",
                    "real-world-assets-rwa": "RWA",
                }
                sectors = []
                for cat in categories:
                    cat_id = cat.get("id", "")
                    if cat_id in key_sectors:
                        change_24h = cat.get("market_cap_change_24h", 0) or 0
                        sectors.append({
                            "sector": key_sectors[cat_id],
                            "market_cap_change_24h": round(change_24h, 2),
                            "volume_24h": cat.get("volume_24h", 0),
                            "top_coins_count": cat.get("top_3_coins_count", 0),
                        })

                # Sort by 24h change to show leaders/laggards
                sectors.sort(key=lambda x: x["market_cap_change_24h"], reverse=True)

                # Determine overall narrative
                if sectors:
                    best = sectors[0]
                    worst = sectors[-1]
                    narrative = f"{best['sector']} лидирует ({best['market_cap_change_24h']:+.1f}%), {worst['sector']} отстаёт ({worst['market_cap_change_24h']:+.1f}%)"
                else:
                    narrative = "unknown"

                result = {
                    "sectors": sectors,
                    "narrative": narrative,
                }
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"CoinGecko categories fetch failed: {e}")

        return {"sectors": [], "narrative": "unknown"}

    def get_full_social_data(self) -> dict:
        """Get all social/momentum data."""
        return {
            "social_trending": self.fetch_coingecko_trending(),
            "sector_performance": self.fetch_sector_performance(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
