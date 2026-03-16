"""Social sentiment and additional news sources.

Sources:
- CryptoPanic: aggregated crypto news with sentiment (free API)
- Reddit: crypto subreddit sentiment
- LunarCrush: social metrics for crypto (public endpoints)
"""

import time
from datetime import datetime

import requests
from loguru import logger


class SocialSentiment:
    """Fetches social media and alternative news sentiment."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 min cache

    def fetch_cryptopanic(self, filter_type: str = "hot") -> list[dict]:
        """Fetch news from CryptoPanic (no API key needed for public posts).

        filter_type: hot | rising | bullish | bearish | important
        """
        cache_key = f"cryptopanic_{filter_type}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(
                "https://cryptopanic.com/api/free/v1/posts/",
                params={
                    "auth_token": "free",
                    "filter": filter_type,
                    "public": "true",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                posts = resp.json().get("results", [])
                result = [
                    {
                        "title": p.get("title", ""),
                        "source": p.get("source", {}).get("title", ""),
                        "published_at": p.get("published_at", ""),
                        "kind": p.get("kind", ""),  # news, media, etc.
                        "currencies": [c["code"] for c in p.get("currencies", [])],
                        "votes": {
                            "positive": p.get("votes", {}).get("positive", 0),
                            "negative": p.get("votes", {}).get("negative", 0),
                        },
                    }
                    for p in posts[:15]
                ]
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.debug(f"CryptoPanic fetch failed: {e}")

        return []

    def fetch_reddit_sentiment(self) -> dict:
        """Fetch sentiment from crypto subreddits via Reddit JSON API."""
        cache_key = "reddit_sentiment"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        subreddits = ["cryptocurrency", "bitcoin", "ethtrader"]
        all_posts = []

        for sub in subreddits:
            try:
                resp = requests.get(
                    f"https://www.reddit.com/r/{sub}/hot.json",
                    params={"limit": 10},
                    headers={"User-Agent": "TradingAgent/1.0"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    posts = resp.json().get("data", {}).get("children", [])
                    for p in posts:
                        data = p.get("data", {})
                        all_posts.append({
                            "subreddit": sub,
                            "title": data.get("title", ""),
                            "score": data.get("score", 0),
                            "num_comments": data.get("num_comments", 0),
                            "upvote_ratio": data.get("upvote_ratio", 0.5),
                        })
            except Exception as e:
                logger.debug(f"Reddit r/{sub} fetch failed: {e}")

        # Calculate overall sentiment
        if all_posts:
            avg_ratio = sum(p["upvote_ratio"] for p in all_posts) / len(all_posts)
            top_posts = sorted(all_posts, key=lambda x: x["score"], reverse=True)[:10]
        else:
            avg_ratio = 0.5
            top_posts = []

        result = {
            "avg_upvote_ratio": round(avg_ratio, 2),
            "sentiment": "bullish" if avg_ratio > 0.7 else ("bearish" if avg_ratio < 0.4 else "neutral"),
            "top_discussions": [
                {"title": p["title"], "score": p["score"], "sub": p["subreddit"]}
                for p in top_posts[:7]
            ],
        }
        self._set_cache(cache_key, result)
        return result

    def fetch_lunarcrush_sentiment(self, symbol: str = "BTC") -> dict:
        """Fetch social metrics from LunarCrush public API."""
        cache_key = f"lunarcrush_{symbol}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(
                "https://lunarcrush.com/api4/public/coins/list/v2",
                params={"sort": "galaxy_score", "limit": 10},
                headers={"User-Agent": "TradingAgent/1.0"},
                timeout=10,
            )
            if resp.status_code == 200:
                coins = resp.json().get("data", [])
                result = {
                    "trending_by_social": [
                        {
                            "symbol": c.get("symbol", ""),
                            "name": c.get("name", ""),
                            "galaxy_score": c.get("galaxy_score", 0),
                            "alt_rank": c.get("alt_rank", 0),
                        }
                        for c in coins[:10]
                    ],
                }
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.debug(f"LunarCrush fetch failed: {e}")

        return {"trending_by_social": []}

    def get_full_social_data(self) -> dict:
        """Get all social sentiment data."""
        return {
            "cryptopanic_hot": self.fetch_cryptopanic("hot"),
            "cryptopanic_bearish": self.fetch_cryptopanic("bearish"),
            "reddit_sentiment": self.fetch_reddit_sentiment(),
            "social_trending": self.fetch_lunarcrush_sentiment(),
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
