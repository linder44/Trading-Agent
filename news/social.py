"""Social sentiment and additional news sources.

Sources:
- CryptoPanic: aggregated crypto news with sentiment (requires free API key from cryptopanic.com)
- Reddit: crypto subreddit sentiment via old.reddit.com JSON
- CoinGecko trending: social momentum proxy (free, no key)
"""

import os
import time
from datetime import datetime

import requests
from loguru import logger


class SocialSentiment:
    """Fetches social media and alternative news sentiment."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 min cache
        self._cryptopanic_token = os.getenv("CRYPTOPANIC_API_KEY", "")

    def fetch_cryptopanic(self, filter_type: str = "hot") -> list[dict]:
        """Fetch news from CryptoPanic.

        Requires a free API key from https://cryptopanic.com/developers/api/
        Set CRYPTOPANIC_API_KEY in .env to enable.
        filter_type: hot | rising | bullish | bearish | important
        """
        if not self._cryptopanic_token:
            return []

        cache_key = f"cryptopanic_{filter_type}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(
                "https://cryptopanic.com/api/free/v1/posts/",
                params={
                    "auth_token": self._cryptopanic_token,
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
                        "kind": p.get("kind", ""),
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
            else:
                logger.warning(f"CryptoPanic returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"CryptoPanic fetch failed: {e}")

        return []

    def fetch_reddit_sentiment(self) -> dict:
        """Fetch sentiment from crypto subreddits via Reddit JSON API.

        Uses old.reddit.com which is more reliable for JSON access.
        """
        cache_key = "reddit_sentiment"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        subreddits = ["cryptocurrency", "bitcoin"]
        all_posts = []

        for sub in subreddits:
            try:
                resp = requests.get(
                    f"https://old.reddit.com/r/{sub}/hot.json",
                    params={"limit": 10, "raw_json": 1},
                    headers={
                        "User-Agent": "TradingBot/2.0 (market research; +https://github.com)",
                        "Accept": "application/json",
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    posts = resp.json().get("data", {}).get("children", [])
                    for p in posts:
                        data = p.get("data", {})
                        if data.get("stickied"):
                            continue
                        all_posts.append({
                            "subreddit": sub,
                            "title": data.get("title", ""),
                            "score": data.get("score", 0),
                            "num_comments": data.get("num_comments", 0),
                            "upvote_ratio": data.get("upvote_ratio", 0.5),
                        })
                elif resp.status_code == 429:
                    logger.warning(f"Reddit rate limited on r/{sub}, skipping")
                    break
                else:
                    logger.warning(f"Reddit r/{sub} returned {resp.status_code}")
            except Exception as e:
                logger.warning(f"Reddit r/{sub} fetch failed: {e}")

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

    def fetch_coingecko_trending(self) -> dict:
        """Fetch trending search coins from CoinGecko (free, no key).

        Acts as a social momentum proxy — coins trending on CoinGecko
        are getting attention from retail traders.
        """
        cache_key = "cg_trending"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=10,
            )
            if resp.status_code == 200:
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

    def get_full_social_data(self) -> dict:
        """Get all social sentiment data."""
        return {
            "cryptopanic_hot": self.fetch_cryptopanic("hot"),
            "cryptopanic_bearish": self.fetch_cryptopanic("bearish"),
            "reddit_sentiment": self.fetch_reddit_sentiment(),
            "social_trending": self.fetch_coingecko_trending(),
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
