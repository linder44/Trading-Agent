"""Market correlation data.

Tracks assets correlated with crypto to provide broader market context:
- BTC Dominance: % of total crypto market cap that is Bitcoin
- DXY (Dollar Index): strong dollar = bearish for crypto
- S&P 500: risk-on/risk-off correlation
- Gold: safe haven correlation
- US Treasury yields: rising yields = bearish for risk assets
"""

import time
from datetime import datetime

import requests
from loguru import logger


class MarketCorrelations:
    """Fetches broader market data that correlates with crypto."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 min

    def get_btc_dominance(self) -> dict:
        """Get BTC dominance from CoinGecko.

        Rising dominance = altcoins underperforming (alt season ending)
        Falling dominance = altcoins outperforming (alt season)
        """
        cache_key = "btc_dominance"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            market_cap_pct = data.get("market_cap_percentage", {})

            result = {
                "btc_dominance": round(market_cap_pct.get("btc", 0), 2),
                "eth_dominance": round(market_cap_pct.get("eth", 0), 2),
                "total_market_cap_usd": data.get("total_market_cap", {}).get("usd", 0),
                "total_volume_24h_usd": data.get("total_volume", {}).get("usd", 0),
                "market_cap_change_24h": round(data.get("market_cap_change_percentage_24h_usd", 0), 2),
                "signal": "alt_season" if market_cap_pct.get("btc", 50) < 40 else (
                    "btc_dominant" if market_cap_pct.get("btc", 50) > 60 else "balanced"
                ),
            }
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.debug(f"BTC dominance fetch failed: {e}")
            return {"btc_dominance": 0, "signal": "unknown"}

    def get_dxy_and_tradfi(self) -> dict:
        """Get traditional finance data (DXY, S&P500, Gold, Treasury yields).

        Uses free Yahoo Finance alternative endpoints.
        """
        cache_key = "tradfi"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        result = {}

        # Fetch from multiple sources
        symbols = {
            "DXY": "Dollar Index",
            "SPY": "S&P 500 ETF",
            "GLD": "Gold ETF",
            "TLT": "20Y Treasury Bond ETF",
            "VIX": "Volatility Index (Fear gauge)",
        }

        for symbol, name in symbols.items():
            try:
                # Using Yahoo Finance v8 API (public)
                resp = requests.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"range": "5d", "interval": "1d"},
                    headers={"User-Agent": "TradingAgent/1.0"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    chart = resp.json().get("chart", {}).get("result", [{}])[0]
                    meta = chart.get("meta", {})
                    price = meta.get("regularMarketPrice", 0)
                    prev_close = meta.get("chartPreviousClose", price)
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                    result[symbol] = {
                        "name": name,
                        "price": round(price, 2),
                        "change_pct": round(change_pct, 2),
                    }
            except Exception as e:
                logger.debug(f"Failed to fetch {symbol}: {e}")

        # Add interpretations
        if "DXY" in result:
            dxy_change = result["DXY"]["change_pct"]
            result["dxy_impact"] = "bearish_for_crypto" if dxy_change > 0.3 else (
                "bullish_for_crypto" if dxy_change < -0.3 else "neutral"
            )

        if "VIX" in result:
            vix_price = result["VIX"]["price"]
            result["market_fear"] = "extreme_fear" if vix_price > 30 else (
                "elevated_fear" if vix_price > 20 else "low_fear_complacency"
            )

        if "SPY" in result:
            spy_change = result["SPY"]["change_pct"]
            result["risk_appetite"] = "risk_on" if spy_change > 0.5 else (
                "risk_off" if spy_change < -0.5 else "neutral"
            )

        self._set_cache(cache_key, result)
        return result

    def get_eth_btc_ratio(self, exchange_client) -> dict:
        """Get ETH/BTC ratio - indicates altcoin strength."""
        try:
            ticker = exchange_client.fetch_ticker("ETH/BTC")
            price = ticker["last"]
            return {
                "eth_btc_ratio": round(price, 6),
                "signal": "alts_strong" if price > 0.05 else (
                    "alts_weak" if price < 0.03 else "neutral"
                ),
            }
        except Exception as e:
            logger.debug(f"ETH/BTC ratio fetch failed: {e}")
            return {"eth_btc_ratio": 0, "signal": "unknown"}

    def get_full_correlation_data(self, exchange_client=None) -> dict:
        """Get all market correlation data."""
        data = {
            "btc_dominance": self.get_btc_dominance(),
            "traditional_markets": self.get_dxy_and_tradfi(),
        }

        if exchange_client:
            data["eth_btc_ratio"] = self.get_eth_btc_ratio(exchange_client)

        data["fetched_at"] = datetime.utcnow().isoformat()
        return data

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
