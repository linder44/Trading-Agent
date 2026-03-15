"""News fetcher and sentiment analysis module."""

import time
from datetime import datetime, timedelta

import requests
from loguru import logger

from config import NewsConfig


class NewsFetcher:
    """Fetches crypto news from multiple sources."""

    COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"
    CRYPTOPANIC_API = "https://cryptopanic.com/api/free/v1/posts/"

    def __init__(self, cfg: NewsConfig):
        self.cfg = cfg
        self._cache: dict = {}
        self._cache_ttl = 300  # 5 min cache

    def fetch_newsapi(self, query: str = "cryptocurrency bitcoin") -> list[dict]:
        """Fetch news from NewsAPI.org."""
        if not self.cfg.api_key:
            return []

        cache_key = f"newsapi_{query}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "from": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
                    "language": "en",
                    "apiKey": self.cfg.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            result = [
                {
                    "title": a["title"],
                    "description": a.get("description", ""),
                    "source": a["source"]["name"],
                    "url": a["url"],
                    "published_at": a["publishedAt"],
                }
                for a in articles
                if a.get("title")
            ]
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    def fetch_coingecko_trending(self) -> list[dict]:
        """Fetch trending coins from CoinGecko."""
        cache_key = "coingecko_trending"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(self.COINGECKO_TRENDING, timeout=10)
            resp.raise_for_status()
            coins = resp.json().get("coins", [])
            result = [
                {
                    "name": c["item"]["name"],
                    "symbol": c["item"]["symbol"],
                    "market_cap_rank": c["item"].get("market_cap_rank"),
                    "score": c["item"]["score"],
                }
                for c in coins
            ]
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"CoinGecko trending fetch failed: {e}")
            return []

    def fetch_fear_greed_index(self) -> dict:
        """Fetch Crypto Fear & Greed Index."""
        cache_key = "fear_greed"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            resp.raise_for_status()
            data = resp.json()["data"][0]
            result = {
                "value": int(data["value"]),
                "classification": data["value_classification"],
                "timestamp": data["timestamp"],
            }
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return {"value": 50, "classification": "Neutral", "timestamp": ""}

    def get_market_context(self) -> dict:
        """Get full market context for AI decision making."""
        # Запрашиваем новости по двум группам: крипто + геополитика/макро
        crypto_keywords = [k for k in self.cfg.keywords if k in (
            "bitcoin", "ethereum", "crypto", "altcoin",
            "SEC", "Fed", "interest rate", "inflation", "CPI",
        )]
        geopolitics_keywords = [k for k in self.cfg.keywords if k not in crypto_keywords]

        crypto_news = self.fetch_newsapi(" OR ".join(crypto_keywords[:5]))
        geo_news = self.fetch_newsapi(" OR ".join(geopolitics_keywords[:5])) if geopolitics_keywords else []

        trending = self.fetch_coingecko_trending()
        fear_greed = self.fetch_fear_greed_index()

        return {
            "crypto_news": [
                {"title": n["title"], "source": n["source"]}
                for n in crypto_news[:10]
            ],
            "geopolitics_macro_news": [
                {"title": n["title"], "source": n["source"]}
                for n in geo_news[:10]
            ],
            "trending_coins": trending[:7],
            "fear_greed_index": fear_greed,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
