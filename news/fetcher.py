"""News fetcher and sentiment analysis module.

Sources (in priority order):
1. NewsData.io — free API key (200 req/day), global, crypto category
   Get free key: https://newsdata.io/register
2. CryptoPanic — free public API, no key needed
3. NewsAPI — paid key required (NEWS_API_KEY env var)
4. CoinGecko — trending coins
5. Alternative.me — Fear & Greed Index
"""

import time
from datetime import datetime, timedelta, timezone

import requests
from loguru import logger

from config import NewsConfig
from utils.http import HttpClientError, request_with_retry


class NewsFetcher:
    """Fetches crypto news from multiple sources."""

    COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"
    CRYPTOPANIC_API = "https://cryptopanic.com/api/free/v1/posts/"
    NEWSDATA_API = "https://newsdata.io/api/1/latest"

    def __init__(self, cfg: NewsConfig):
        self.cfg = cfg
        self._cache: dict = {}
        self._cache_ttl = 300  # 5 min cache

    # ─── NewsData.io (free key, 200 req/day) ──────────────────────

    def fetch_newsdata(self, query: str = "cryptocurrency OR bitcoin",
                       category: str | None = None) -> list[dict]:
        """Fetch news from NewsData.io (requires free NEWSDATA_API_KEY).

        Free tier: 200 requests/day, works globally without geo-restrictions.
        """
        if not self.cfg.newsdata_api_key:
            return []

        cache_key = f"newsdata_{query}_{category}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            params: dict = {
                "apikey": self.cfg.newsdata_api_key,
                "q": query,
                "language": "en",
            }
            if category:
                params["category"] = category

            resp = request_with_retry(
                self.NEWSDATA_API,
                params=params,
                timeout=10,
            )
            if not resp:
                return []
            data = resp.json()
            articles = data.get("results", [])
            result = [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source_name", a.get("source_id", "NewsData")),
                    "url": a.get("link", ""),
                    "published_at": a.get("pubDate", ""),
                }
                for a in articles
                if a.get("title")
            ]
            self._set_cache(cache_key, result)
            return result
        except HttpClientError as e:
            logger.warning(f"NewsData.io returned {e.status_code}: {e}")
            return []
        except Exception as e:
            logger.warning(f"NewsData.io fetch error: {e}")
            return []

    # ─── CryptoPanic (free, no key needed) ────────────────────────

    def fetch_cryptopanic(self, kind: str = "news") -> list[dict]:
        """Fetch posts from CryptoPanic free API."""
        cache_key = f"cryptopanic_{kind}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            params: dict = {"public": "true"}
            if kind == "news":
                params["kind"] = "news"
                params["filter"] = "hot"

            resp = request_with_retry(
                self.CRYPTOPANIC_API,
                params=params,
                timeout=10,
            )
            if not resp:
                return []
            posts = resp.json().get("results", [])
            result = [
                {
                    "title": p["title"],
                    "source": p.get("source", {}).get("title", "CryptoPanic"),
                    "url": p.get("url", ""),
                    "published_at": p.get("published_at", ""),
                }
                for p in posts
                if p.get("title")
            ]
            self._set_cache(cache_key, result)
            return result
        except HttpClientError as e:
            logger.debug(f"CryptoPanic {kind} returned {e.status_code}: {e}")
            return []
        except Exception as e:
            logger.warning(f"CryptoPanic fetch error: {e}")
            return []

    # ─── NewsAPI (requires paid key) ──────────────────────────────

    def fetch_newsapi(self, query: str = "cryptocurrency bitcoin") -> list[dict]:
        """Fetch news from NewsAPI.org (requires NEWS_API_KEY)."""
        if not self.cfg.api_key:
            return []

        cache_key = f"newsapi_{query}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "from": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
                    "language": "en",
                    "apiKey": self.cfg.api_key,
                },
            )
            if not resp:
                return []
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
            logger.warning(f"Ошибка загрузки NewsAPI: {e}")
            return []

    # ─── CoinGecko & Fear/Greed ───────────────────────────────────

    def fetch_coingecko_trending(self) -> list[dict]:
        """Fetch trending coins from CoinGecko."""
        cache_key = "coingecko_trending"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry(self.COINGECKO_TRENDING)
            if not resp:
                return []
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
            logger.warning(f"Ошибка загрузки трендов CoinGecko: {e}")
            return []

    def fetch_fear_greed_index(self) -> dict:
        """Fetch Crypto Fear & Greed Index."""
        cache_key = "fear_greed"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry("https://api.alternative.me/fng/?limit=1")
            if not resp:
                return {"value": 50, "classification": "Neutral", "timestamp": ""}
            data = resp.json()["data"][0]
            result = {
                "value": int(data["value"]),
                "classification": data["value_classification"],
                "timestamp": data["timestamp"],
            }
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Ошибка загрузки индекса страха и жадности: {e}")
            return {"value": 50, "classification": "Neutral", "timestamp": ""}

    # ─── Main entry point ─────────────────────────────────────────

    def get_market_context(self) -> dict:
        """Get full market context for AI decision making.

        News priority: NewsData.io (free key) → CryptoPanic (no key) → NewsAPI (paid).
        """
        # 1. Крипто-новости: NewsData.io → CryptoPanic → NewsAPI
        crypto_news = self.fetch_newsdata("cryptocurrency OR bitcoin OR ethereum OR crypto")
        if not crypto_news:
            crypto_news = self.fetch_cryptopanic("news")
        if not crypto_news and self.cfg.api_key:
            crypto_news = self.fetch_newsapi("bitcoin OR ethereum OR crypto")

        # 2. Геополитика/макро: NewsData.io → CryptoPanic media → NewsAPI
        geo_news = self.fetch_newsdata(
            "inflation OR interest rate OR sanctions OR tariffs OR recession",
            category="business",
        )
        if not geo_news:
            geo_news = self.fetch_cryptopanic("media")
        if not geo_news and self.cfg.api_key:
            geopolitics_keywords = [k for k in self.cfg.keywords if k not in (
                "bitcoin", "ethereum", "crypto", "altcoin",
                "SEC", "Fed", "interest rate", "inflation", "CPI",
            )]
            if geopolitics_keywords:
                geo_news = self.fetch_newsapi(" OR ".join(geopolitics_keywords[:5]))

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
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    # ─── Cache helpers ────────────────────────────────────────────

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
