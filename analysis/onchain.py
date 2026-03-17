"""On-chain and derivatives data module.

Fetches data that is not available from standard OHLCV:
- Funding rates (Bitget REST API, ccxt fallback)
- Open interest (how much money is in the market)
- Long/short ratio (positioning of accounts)
- Whale detection (large trades on Bitget futures)
- Exchange inflows/outflows (order book imbalance)

All HTTP calls use request_with_retry (3 attempts, 5s timeout each).
"""

import time
from datetime import datetime

import requests
from loguru import logger

from utils.http import HttpClientError, request_with_retry


class OnChainAnalyzer:
    """Fetches on-chain and derivatives market data."""

    BITGET_BASE = "https://api.bitget.com/api/v2/mix/market"
    BYBIT_V5 = "https://api.bybit.com/v5/market"

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = 300  # 5 min

    def get_funding_rate(self, exchange_client, symbol: str) -> dict:
        """Get current funding rate using Bitget REST API directly."""
        base_coin = symbol.split("/")[0]
        cache_key = f"funding_{base_coin}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry(
                f"{self.BITGET_BASE}/current-fund-rate",
                params={"symbol": f"{base_coin}USDT", "productType": "USDT-FUTURES"},
                timeout=15,
            )
        except HttpClientError:
            resp = None
        if resp:
            api_data = resp.json().get("data", [])
            if api_data:
                item = api_data[0] if isinstance(api_data, list) else api_data
                rate = float(item.get("fundingRate", 0))
                result = self._format_funding(rate)
                self._set_cache(cache_key, result)
                return result

        # Fallback: ccxt
        try:
            funding = exchange_client.exchange.fetch_funding_rate(symbol)
            rate = float(funding.get("fundingRate", 0))
            result = self._format_funding(rate)
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Funding rate ccxt fallback failed for {symbol}: {e}")

        return {"funding_rate": 0, "funding_rate_pct": 0, "sentiment": "unknown"}

    def get_open_interest(self, exchange_client, symbol: str) -> dict:
        """Get open interest data via Bitget REST API."""
        base_coin = symbol.split("/")[0]
        cache_key = f"oi_{base_coin}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry(
                f"{self.BITGET_BASE}/open-interest",
                params={"symbol": f"{base_coin}USDT", "productType": "USDT-FUTURES"},
                timeout=15,
            )
        except HttpClientError:
            resp = None
        if resp:
            api_data = resp.json().get("data", {})
            oi_list = api_data.get("openInterestList", [])
            if oi_list:
                amount = float(oi_list[0].get("size", 0))
                result = {"open_interest_amount": round(amount, 4), "open_interest_value_usd": 0}
                try:
                    ticker = exchange_client.exchange.fetch_ticker(symbol)
                    price = float(ticker.get("last", 0))
                    if price > 0:
                        result["open_interest_value_usd"] = round(amount * price, 2)
                except Exception:
                    pass
                self._set_cache(cache_key, result)
                return result

        # Fallback: ccxt
        try:
            oi = exchange_client.exchange.fetch_open_interest(symbol)
            result = {
                "open_interest_value_usd": round(float(oi.get("openInterestValue") or 0), 2),
                "open_interest_amount": round(float(oi.get("openInterestAmount") or 0), 4),
            }
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Open interest ccxt fallback failed for {symbol}: {e}")

        return {"open_interest_value_usd": 0, "open_interest_amount": 0}

    _LS_DEFAULT = {"long_pct": 50, "short_pct": 50, "ratio": 1.0, "signal": "neutral", "_source": "default"}

    def get_long_short_ratio(self, exchange_client, symbol: str) -> dict:
        """Get long/short account ratio via Bybit v5 public API.

        Bybit работает глобально (без гео-блокировок как у Binance 451),
        поддерживает все основные фьючерсные пары, ключи не нужны.
        """
        base_coin = symbol.split("/")[0]
        cache_key = f"ls_ratio_{base_coin}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        # Bybit v5 — основной источник
        try:
            resp = request_with_retry(
                f"{self.BYBIT_V5}/account-ratio",
                params={
                    "category": "linear",
                    "symbol": f"{base_coin}USDT",
                    "period": "1h",
                    "limit": "1",
                },
                timeout=10,
            )
        except HttpClientError as e:
            logger.debug(f"Bybit long/short ratio 4xx for {base_coin}: {e}")
            resp = None
        if resp:
            body = resp.json()
            items = body.get("result", {}).get("list", [])
            if items:
                result = self._parse_bybit_ls(items[0])
                self._set_cache(cache_key, result)
                return result

        # Fallback: ccxt (через биржу пользователя)
        try:
            ls = exchange_client.exchange.fetch_long_short_ratio_history(symbol, limit=1)
            if ls:
                long_ratio = float(ls[0].get("longAccount", 0.5))
                short_ratio = float(ls[0].get("shortAccount", 0.5))
                ratio = long_ratio / short_ratio if short_ratio > 0 else 1.0
                result = {
                    "long_pct": round(long_ratio * 100, 1),
                    "short_pct": round(short_ratio * 100, 1),
                    "ratio": round(ratio, 2),
                    "signal": self._ls_signal(ratio),
                    "_source": "ccxt",
                }
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.debug(f"Long/short ratio ccxt fallback failed for {symbol}: {e}")

        return self._LS_DEFAULT.copy()

    @staticmethod
    def _parse_bybit_ls(item: dict) -> dict:
        """Parse Bybit v5 account-ratio response item.

        Bybit returns: {"symbol":"BTCUSDT","buyRatio":"0.5162",
                        "sellRatio":"0.4838","timestamp":"1234567890000"}
        """
        long_ratio = float(item.get("buyRatio", 0.5))
        short_ratio = float(item.get("sellRatio", 0.5))
        ratio = long_ratio / short_ratio if short_ratio > 0 else 1.0
        return {
            "long_pct": round(long_ratio * 100, 1),
            "short_pct": round(short_ratio * 100, 1),
            "ratio": round(ratio, 2),
            "signal": "contrarian_bearish" if ratio > 2.0 else (
                "contrarian_bullish" if ratio < 0.5 else "neutral"
            ),
            "_source": "bybit",
        }

    @staticmethod
    def _ls_signal(ratio: float) -> str:
        if ratio > 2.0:
            return "contrarian_bearish"
        if ratio < 0.5:
            return "contrarian_bullish"
        return "neutral"

    def get_whale_alerts(self, exchange_client=None) -> list[dict]:
        """Detect large trades on Bitget (whale activity proxy)."""
        cache_key = "whale_alerts"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        large_trades = []

        try:
            resp = request_with_retry(
                f"{self.BITGET_BASE}/fills",
                params={"symbol": "BTCUSDT", "productType": "USDT-FUTURES", "limit": "50"},
            )
        except HttpClientError:
            resp = None
        if resp:
            fills = resp.json().get("data", [])
            if fills:
                sizes = [float(f.get("size", 0)) for f in fills if float(f.get("size", 0)) > 0]
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    threshold = avg_size * 5
                    for f in fills:
                        size = float(f.get("size", 0))
                        if size >= threshold:
                            large_trades.append({
                                "symbol": "BTC",
                                "size_contracts": size,
                                "side": f.get("side", "unknown"),
                                "price": float(f.get("price", 0)),
                                "ratio_to_avg": round(size / avg_size, 1),
                                "type": "whale_trade",
                            })
                        if len(large_trades) >= 10:
                            break
                    self._set_cache(cache_key, large_trades)
                    return large_trades

        # Fallback: order book large orders
        if exchange_client:
            try:
                ob = exchange_client.exchange.fetch_order_book("BTC/USDT:USDT", limit=50)
                all_orders = [(b[0], b[1], "bid") for b in ob.get("bids", [])] + \
                             [(a[0], a[1], "ask") for a in ob.get("asks", [])]
                if all_orders:
                    avg_vol = sum(o[1] for o in all_orders) / len(all_orders)
                    for price, vol, side in all_orders:
                        if vol >= avg_vol * 5:
                            large_trades.append({
                                "symbol": "BTC", "size_btc": round(vol, 4),
                                "price": price, "side": side, "type": "whale_order",
                            })
                        if len(large_trades) >= 10:
                            break
            except Exception as e:
                logger.warning(f"Whale detection (order book) failed: {e}")

        self._set_cache(cache_key, large_trades)
        return large_trades

    def get_exchange_netflow(self, exchange_client) -> dict:
        """Estimate exchange flow direction from order book imbalance."""
        cache_key = "exchange_netflow"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            resp = request_with_retry(
                f"{self.BITGET_BASE}/merge-depth",
                params={"symbol": "BTCUSDT", "productType": "USDT-FUTURES", "limit": "50"},
            )
        except HttpClientError:
            resp = None
        if resp:
            ob = resp.json().get("data", {})
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if bids and asks:
                result = self._calc_netflow(
                    sum(float(b[1]) for b in bids),
                    sum(float(a[1]) for a in asks),
                )
                if result:
                    self._set_cache(cache_key, result)
                    return result

        # Fallback: ccxt
        try:
            ob = exchange_client.exchange.fetch_order_book("BTC/USDT:USDT", limit=50)
            result = self._calc_netflow(
                sum(b[1] for b in ob.get("bids", [])),
                sum(a[1] for a in ob.get("asks", [])),
            )
            if result:
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Exchange netflow ccxt fallback failed: {e}")

        return {"bid_volume_btc": 0, "ask_volume_btc": 0, "bid_ratio": 0.5, "signal": "unknown"}

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
            "whale_alerts": self.get_whale_alerts(exchange_client),
            "exchange_netflow": self.get_exchange_netflow(exchange_client),
        }

        return result

    # ─── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _format_funding(rate: float) -> dict:
        return {
            "funding_rate": round(rate, 6),
            "funding_rate_pct": round(rate * 100, 4),
            "sentiment": "extreme_greed" if rate > 0.001 else (
                "bullish" if rate > 0 else (
                    "extreme_fear" if rate < -0.001 else "bearish"
                )
            ),
        }

    @staticmethod
    def _calc_netflow(bid_volume: float, ask_volume: float) -> dict | None:
        total = bid_volume + ask_volume
        if total <= 0:
            return None
        bid_ratio = bid_volume / total
        return {
            "bid_volume_btc": round(bid_volume, 2),
            "ask_volume_btc": round(ask_volume, 2),
            "bid_ratio": round(bid_ratio, 3),
            "signal": "accumulation" if bid_ratio > 0.55 else (
                "selling_pressure" if bid_ratio < 0.45 else "balanced"
            ),
        }

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
