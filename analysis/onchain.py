"""On-chain and derivatives data module.

Fetches data that is not available from standard OHLCV:
- Funding rates (Bitget REST API, ccxt fallback)
- Open interest (how much money is in the market)
- Long/short ratio (positioning of accounts)
- Whale detection (large trades on Bitget futures)
- Exchange inflows/outflows (order book imbalance)
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
        """Get current funding rate using Bitget REST API directly.

        Positive = longs pay shorts (market is long-heavy, potential top)
        Negative = shorts pay longs (market is short-heavy, potential bottom)
        """
        base_coin = symbol.split("/")[0]
        cache_key = f"funding_{base_coin}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        # Bitget public REST API — не требует load_markets()
        try:
            resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/current-fund-rate",
                params={
                    "symbol": f"{base_coin}USDT",
                    "productType": "USDT-FUTURES",
                },
                timeout=8,
            )
            if resp.status_code == 200:
                api_data = resp.json().get("data", [])
                if api_data:
                    item = api_data[0] if isinstance(api_data, list) else api_data
                    rate = float(item.get("fundingRate", 0))
                    result = {
                        "funding_rate": round(rate, 6),
                        "funding_rate_pct": round(rate * 100, 4),
                        "sentiment": "extreme_greed" if rate > 0.001 else (
                            "bullish" if rate > 0 else (
                                "extreme_fear" if rate < -0.001 else "bearish"
                            )
                        ),
                    }
                    self._set_cache(cache_key, result)
                    return result
        except Exception as e:
            logger.warning(f"Funding rate REST failed for {base_coin}: {e}")

        # Fallback: ccxt (если REST недоступен)
        try:
            funding = exchange_client.exchange.fetch_funding_rate(symbol)
            rate = float(funding.get("fundingRate", 0))
            result = {
                "funding_rate": round(rate, 6),
                "funding_rate_pct": round(rate * 100, 4),
                "sentiment": "extreme_greed" if rate > 0.001 else (
                    "bullish" if rate > 0 else (
                        "extreme_fear" if rate < -0.001 else "bearish"
                    )
                ),
            }
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Funding rate ccxt fallback failed for {symbol}: {e}")

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
            oi_value = float(oi.get("openInterestValue") or 0)
            oi_amount = float(oi.get("openInterestAmount") or 0)
            return {
                "open_interest_value_usd": round(oi_value, 2),
                "open_interest_amount": round(oi_amount, 4),
            }
        except Exception as e:
            logger.warning(f"Open interest fetch failed for {symbol}: {e}")
            return {"open_interest_value_usd": 0, "open_interest_amount": 0}

    def get_long_short_ratio(self, exchange_client, symbol: str) -> dict:
        """Get long/short ratio using Bitget public API directly.

        Ratio > 1 = more longs than shorts
        Extreme ratios often signal reversals (contrarian indicator)
        """
        base_coin = symbol.split("/")[0]
        cache_key = f"ls_ratio_{base_coin}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        # Bitget public API — no auth required
        try:
            resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/account-long-short",
                params={
                    "symbol": f"{base_coin}USDT",
                    "period": "5m",
                    "productType": "USDT-FUTURES",
                },
                timeout=8,
            )
            if resp.status_code == 200:
                api_data = resp.json().get("data", [])
                if api_data:
                    latest = api_data[-1] if isinstance(api_data, list) else api_data
                    long_ratio = float(latest.get("longAccountRatio", 0.5))
                    short_ratio = float(latest.get("shortAccountRatio", 0.5))
                    ratio = long_ratio / short_ratio if short_ratio > 0 else 1.0
                    result = {
                        "long_pct": round(long_ratio * 100, 1),
                        "short_pct": round(short_ratio * 100, 1),
                        "ratio": round(ratio, 2),
                        "signal": "contrarian_bearish" if ratio > 2.0 else (
                            "contrarian_bullish" if ratio < 0.5 else "neutral"
                        ),
                    }
                    self._set_cache(cache_key, result)
                    return result
            else:
                logger.warning(f"Bitget long/short API returned {resp.status_code} for {base_coin}")
        except Exception as e:
            logger.warning(f"Long/short ratio fetch failed for {base_coin}: {e}")

        return {"long_pct": 50, "short_pct": 50, "ratio": 1.0, "signal": "neutral"}

    def get_whale_alerts(self, exchange_client=None) -> list[dict]:
        """Detect large trades on Bitget (whale activity proxy).

        Uses Bitget recent fills API to find abnormally large trades.
        More reliable than blockchain.info which often times out.
        """
        cache_key = "whale_alerts"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        large_trades = []

        # Метод 1: Bitget REST API — последние сделки на BTC фьючерсах
        try:
            resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/fills",
                params={
                    "symbol": "BTCUSDT",
                    "productType": "USDT-FUTURES",
                    "limit": "100",
                },
                timeout=8,
            )
            if resp.status_code == 200:
                fills = resp.json().get("data", [])
                if fills:
                    # Считаем средний объём, ищем аномально крупные
                    sizes = [float(f.get("size", 0)) for f in fills if float(f.get("size", 0)) > 0]
                    if sizes:
                        avg_size = sum(sizes) / len(sizes)
                        threshold = avg_size * 5  # 5x среднего = "кит"

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
        except Exception as e:
            logger.warning(f"Whale detection (Bitget fills) failed: {e}")

        # Метод 2: order book — крупные заявки как прокси
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
                                "symbol": "BTC",
                                "size_btc": round(vol, 4),
                                "price": price,
                                "side": side,
                                "type": "whale_order",
                            })
                        if len(large_trades) >= 10:
                            break
            except Exception as e:
                logger.warning(f"Whale detection (order book) failed: {e}")

        self._set_cache(cache_key, large_trades)
        return large_trades

    def get_exchange_netflow(self, exchange_client) -> dict:
        """Estimate exchange flow direction from order book imbalance.

        Uses bid/ask volume ratio as a proxy for flow direction:
        - More bids than asks = accumulation (buying pressure)
        - More asks than bids = selling pressure
        """
        cache_key = "exchange_netflow"
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            ob = exchange_client.exchange.fetch_order_book("BTC/USDT:USDT", limit=50)
            bid_volume = sum(b[1] for b in ob.get("bids", []))
            ask_volume = sum(a[1] for a in ob.get("asks", []))
            total = bid_volume + ask_volume
            if total > 0:
                bid_ratio = bid_volume / total
                result = {
                    "bid_volume_btc": round(bid_volume, 2),
                    "ask_volume_btc": round(ask_volume, 2),
                    "bid_ratio": round(bid_ratio, 3),
                    "signal": "accumulation" if bid_ratio > 0.55 else (
                        "selling_pressure" if bid_ratio < 0.45 else "balanced"
                    ),
                }
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Exchange netflow (order book) fetch failed: {e}")

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

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            if time.time() - self._cache[key]["ts"] < self._cache_ttl:
                return True
        return False

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
