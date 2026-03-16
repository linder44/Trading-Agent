#!/usr/bin/env python3
"""Standalone test for all data sources.

Tests every external API call without placing orders or calling Claude API.
Run this to verify which data sources work in your environment.

Usage:
    python test_data_sources.py              # Test all sources
    python test_data_sources.py --source onchain  # Test specific group
"""

import argparse
import json
import sys
import time

import requests
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


def test_coingecko_global():
    """Test CoinGecko global data (BTC dominance, market cap)."""
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        pct = data.get("market_cap_percentage", {})
        logger.info(f"  BTC dominance: {pct.get('btc', 0):.1f}%")
        logger.info(f"  ETH dominance: {pct.get('eth', 0):.1f}%")
        logger.info(f"  Market cap change 24h: {data.get('market_cap_change_percentage_24h_usd', 0):.2f}%")
        return True
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_coingecko_stablecoins():
    """Test CoinGecko stablecoin data."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "tether,usd-coin",
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        usdt_mcap = data.get("tether", {}).get("usd_market_cap", 0)
        usdc_mcap = data.get("usd-coin", {}).get("usd_market_cap", 0)
        logger.info(f"  USDT market cap: ${usdt_mcap / 1e9:.1f}B")
        logger.info(f"  USDC market cap: ${usdc_mcap / 1e9:.1f}B")
        return True
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_coingecko_trending():
    """Test CoinGecko trending coins."""
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        resp.raise_for_status()
        coins = resp.json().get("coins", [])
        names = [c["item"]["symbol"] for c in coins[:5]]
        logger.info(f"  Trending: {', '.join(names)}")
        return True
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_fear_greed():
    """Test Fear & Greed Index."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        logger.info(f"  Fear & Greed: {data['value']} ({data['value_classification']})")
        return True
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_bitget_long_short():
    """Test Bitget public long/short ratio API."""
    try:
        resp = requests.get(
            "https://api.bitget.com/api/v2/mix/market/account-long-short",
            params={"symbol": "BTCUSDT", "period": "5m", "productType": "USDT-FUTURES"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "00000":
            logger.error(f"  Bitget API error: {data.get('msg', 'unknown')}")
            return False
        items = data.get("data", [])
        if items:
            latest = items[-1]
            long_r = float(latest.get("longAccountRatio", 0))
            short_r = float(latest.get("shortAccountRatio", 0))
            logger.info(f"  BTC Long/Short: {long_r*100:.1f}% / {short_r*100:.1f}%")
            return True
        logger.warning("  Empty data from Bitget")
        return False
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_bitget_funding():
    """Test Bitget funding rate via ccxt (if available)."""
    try:
        import ccxt
        exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        exchange.load_markets()
        funding = exchange.fetch_funding_rate("BTC/USDT:USDT")
        rate = float(funding.get("fundingRate", 0))
        logger.info(f"  BTC funding rate: {rate*100:.4f}%")
        return True
    except ImportError:
        logger.warning("  SKIP: ccxt not installed (pip install ccxt)")
        return None
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_bitget_open_interest():
    """Test Bitget open interest via ccxt."""
    try:
        import ccxt
        exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        exchange.load_markets()
        oi = exchange.fetch_open_interest("BTC/USDT:USDT")
        oi_val = float(oi.get("openInterestValue") or 0)
        logger.info(f"  BTC open interest: ${oi_val:,.0f}")
        return True
    except ImportError:
        logger.warning("  SKIP: ccxt not installed")
        return None
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_blockchain_whale():
    """Test Blockchain.com unconfirmed transactions (whale proxy)."""
    try:
        resp = requests.get(
            "https://blockchain.info/unconfirmed-transactions?format=json",
            timeout=8,
        )
        resp.raise_for_status()
        txs = resp.json().get("txs", [])
        large = 0
        for tx in txs:
            total_btc = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
            if total_btc >= 10:
                large += 1
        logger.info(f"  Found {large} transactions >= 10 BTC in mempool ({len(txs)} total)")
        return True
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_bitget_orderbook():
    """Test order book fetch for exchange netflow proxy."""
    try:
        import ccxt
        exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        exchange.load_markets()
        ob = exchange.fetch_order_book("BTC/USDT:USDT", limit=50)
        bid_vol = sum(b[1] for b in ob.get("bids", []))
        ask_vol = sum(a[1] for a in ob.get("asks", []))
        total = bid_vol + ask_vol
        ratio = bid_vol / total if total > 0 else 0.5
        logger.info(f"  Order book: bids={bid_vol:.2f} BTC, asks={ask_vol:.2f} BTC, bid_ratio={ratio:.3f}")
        return True
    except ImportError:
        logger.warning("  SKIP: ccxt not installed")
        return None
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_coingecko_categories():
    """Test CoinGecko categories (sector rotation)."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/categories",
            timeout=10,
        )
        resp.raise_for_status()
        categories = resp.json()
        key_sectors = {
            "layer-1": "Layer 1",
            "layer-2": "Layer 2",
            "decentralized-finance-defi": "DeFi",
            "meme-token": "Meme Coins",
        }
        found = []
        for cat in categories:
            cat_id = cat.get("id", "")
            if cat_id in key_sectors:
                change = cat.get("market_cap_change_24h", 0) or 0
                found.append(f"{key_sectors[cat_id]}: {change:+.1f}%")
        for f in found:
            logger.info(f"  {f}")
        return len(found) > 0
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def test_newsapi():
    """Test NewsAPI (requires NEWS_API_KEY in .env)."""
    import os
    key = os.getenv("NEWS_API_KEY", "")
    if not key:
        logger.warning("  SKIP: NEWS_API_KEY not set. Get free key at https://newsapi.org/register")
        return None
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "bitcoin", "pageSize": 3, "sortBy": "publishedAt", "language": "en", "apiKey": key},
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        logger.info(f"  Got {len(articles)} articles")
        if articles:
            logger.info(f"  Latest: {articles[0].get('title', '')[:80]}")
        return len(articles) > 0
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test data sources for Trading Agent")
    parser.add_argument("--source", choices=["onchain", "social", "news", "correlations", "all"], default="all")
    args = parser.parse_args()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    results = {}

    groups = {
        "correlations": [
            ("CoinGecko Global (BTC dominance)", test_coingecko_global),
            ("CoinGecko Stablecoins", test_coingecko_stablecoins),
            ("Fear & Greed Index", test_fear_greed),
        ],
        "onchain": [
            ("Bitget Long/Short Ratio (REST)", test_bitget_long_short),
            ("Bitget Funding Rate (ccxt)", test_bitget_funding),
            ("Bitget Open Interest (ccxt)", test_bitget_open_interest),
            ("Bitget Order Book (netflow proxy)", test_bitget_orderbook),
            ("Blockchain.com Whales", test_blockchain_whale),
        ],
        "social": [
            ("CoinGecko Trending", test_coingecko_trending),
            ("CoinGecko Categories (sectors)", test_coingecko_categories),
        ],
        "news": [
            ("NewsAPI", test_newsapi),
        ],
    }

    selected = list(groups.keys()) if args.source == "all" else [args.source]

    for group in selected:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {group.upper()}")
        logger.info(f"{'='*50}")

        for name, test_fn in groups[group]:
            logger.info(f"\n[{name}]")
            start = time.time()
            result = test_fn()
            elapsed = time.time() - start
            status = "OK" if result else ("SKIP" if result is None else "FAIL")
            results[name] = status
            logger.info(f"  -> {status} ({elapsed:.1f}s)")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")

    ok = sum(1 for v in results.values() if v == "OK")
    fail = sum(1 for v in results.values() if v == "FAIL")
    skip = sum(1 for v in results.values() if v == "SKIP")

    for name, status in results.items():
        icon = {"OK": "+", "FAIL": "X", "SKIP": "-"}[status]
        logger.info(f"  [{icon}] {name}: {status}")

    logger.info(f"\nTotal: {ok} OK, {fail} FAIL, {skip} SKIP")

    if fail > 0:
        logger.warning("\nFailed sources will return empty data. The bot will still work but with less context for AI.")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
