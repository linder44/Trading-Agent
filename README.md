# AI Trading Agent

Autonomous cryptocurrency trading agent powered by Claude AI and Bitget exchange.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   main.py                        │
│              (Orchestrator Loop)                 │
├──────┬──────┬──────┬──────┬──────┬──────────────┤
│  AI  │ Tech │ News │ Risk │Orders│  Exchange    │
│Brain │Analys│Fetch │ Mgr  │ Mgr  │  Client      │
│      │      │      │      │      │  (Bitget)    │
│Claude│ EMA  │NewsAPI│Size │Market│  ccxt        │
│ API  │ RSI  │CoinG │ SL/TP│Limit │              │
│      │ MACD │Fear& │MaxPos│Stop  │              │
│      │ BB   │Greed │Daily │Trail │              │
│      │ ADX  │      │Loss  │      │              │
│      │Ichim │      │      │      │              │
└──────┴──────┴──────┴──────┴──────┴──────────────┘
```

## Quick Start

1. Copy `.env.example` to `.env` and fill in your API keys
2. Install dependencies: `pip install -r requirements.txt`
3. Run in testnet: `python main.py`
4. Run once (dry run): `python main.py --once`
5. Run live: `python main.py --live`

## Docker

```bash
docker compose up -d
```

## Features

- **Technical Analysis**: EMA, SMA, RSI, MACD, Bollinger Bands, ADX, ATR, Ichimoku, OBV, VWAP, Stochastic RSI
- **Multi-Timeframe**: Analyzes 1h, 4h, 1d simultaneously
- **News & Sentiment**: NewsAPI, CoinGecko trending, Fear & Greed Index
- **Risk Management**: Position sizing, max exposure, daily loss limits, ATR-based SL/TP
- **Order Types**: Market, Limit, Stop-Loss, Take-Profit, Trailing Stop
- **AI Decisions**: Claude analyzes all data and makes autonomous trading decisions
- **Notifications**: Telegram alerts for all trades
- **Testnet First**: Safe testing before going live
