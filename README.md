# HSI Sentiment Agent (`hsi-sentiment-agent`)

> **Production-grade Next-Bar ($n+1$) Asset Price Return Predictor using Social & News Sentiment across HKEX and KRX Markets.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-brightgreen.svg)]()
[![Orchestration](https://img.shields.io/badge/Pipeline-LangGraph-purple.svg)]()
[![Tests](https://img.shields.io/badge/Tests-20%20Passed-success.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 1. Overview & Goal

`hsi-sentiment-agent` is an end-to-end quantitative AI agent that predicts the next-bar ($n+1$) return and price of benchmark equities on the **Hong Kong Exchanges (HKEX)** and the **Korea Exchange (KRX)** by integrating multilingual retail sentiment, institutional news, and 1-hour OHLCV market dynamics.

### Supported Exchanges & Tickers
- **HKEX (Hong Kong Exchanges)**:
  - `0700.HK` — Tencent Holdings (騰訊控股)
  - `9988.HK` — Alibaba Group (阿里巴巴)
  - `3690.HK` — Meituan (美團)
  - `1810.HK` — Xiaomi Corporation (小米集團)
  - `9618.HK` — JD.com (京東集團)
- **KRX (Korea Exchange)**:
  - `005930.KS` — Samsung Electronics (삼성전자)
  - `000660.KS` — SK Hynix (SK하이닉스)
  - `035420.KS` — NAVER Corporation (네이버)
  - `035720.KS` — Kakao (카카오)
  - `005380.KS` — Hyundai Motor (현대자동차)

---

## 2. Technical Requirements & Provider Stack

| Capability | Provider | Purpose |
| :--- | :--- | :--- |
| **Data Scraping (Alternative Data)** | [**Apify**](https://apify.com/) | Scrapes retail forum discussions (LIHKG, AAStocks for HKEX; Naver Finance, Daum Finance for KRX) to extract crowd conviction, likes, views, and retail long/short position claims. |
| **Financial News & Sentiment** | [**EXA.com**](http://exa.com) | Semantic AI search engine for financial news retrieval. Pulls verified recent headlines, URLs, published timestamps, and full text snippets for each equity. |
| **Multilingual Sentiment LLM** | **Groq / OpenAI API** | Scores raw text across Cantonese (`yue`), Mandarin (`zh`), Korean (`ko`), and English (`en`) into normalized sentiment $[-1.0, +1.0]$, direction, and position claims with SHA-256 caching and 429 rate-limit backoff. |
| **Market Data** | **yfinance** | Ingests 1-hour OHLCV candles, strictly converted to UTC. |
| **Pipeline Orchestrator** | **LangGraph** | Multi-node deterministic Directed Acyclic Graph (`StateGraph`). |

---

## 3. Strict Clean Architecture

The system strictly follows **Clean Architecture** with inward-only dependency directions:

```
hsi-sentiment-agent/
├── domain/                    # Pure business entities & protocols (Zero external I/O)
│   ├── models.py              # Post, NewsArticle, OHLCVBar, SentimentScore, etc.
│   ├── enums.py               # Direction, PositionClaim, Exchange, LevelType
│   ├── constants.py           # Ticker definitions, timezones (HKT, KST, UTC)
│   └── interfaces.py          # SocialSource, NewsSource, MarketDataSource protocols
├── application/               # Application use cases & quantitative pipeline logic
│   ├── ingestion.py           # Ingestion orchestrator (Apify, Exa, yfinance)
│   ├── sentiment.py           # Sentiment analysis coordinator & weighting
│   ├── features.py            # Leakage-free feature alignment & lagging
│   ├── snapshot.py            # Ticker snapshot & crowd stance aggregation
│   └── model_engine.py        # Pooled walk-forward validation & honest baselines
├── infrastructure/            # Adapters, external APIs, and persistence
│   ├── sources/
│   │   ├── apify/             # Apify client & LIHKG, AAStocks, Naver, Daum scrapers
│   │   ├── exa_news_source.py # EXA.com financial news search client
│   │   └── yfinance_source.py # 1h OHLCV bar fetcher with UTC timezone conversion
│   ├── llm/
│   │   └── sentiment_scorer.py# Dynamic LLM scorer (Groq/OpenAI) with disk caching
│   ├── analysis/
│   │   └── support_resistance.py # SciPy swing extrema & classical daily floor pivots
│   └── cache/
│       └── parquet_repository.py # data/raw/*.parquet zero-cost cache
├── interfaces/                # User delivery & UI
│   ├── cli.py                 # Click CLI interface (UTF-8 safe)
│   └── reporting.py           # Plotly + Jinja2 self-contained interactive HTML reports
├── agents/                    # LangGraph orchestration
│   └── pipeline.py            # StateGraph: Ingest -> Sentiment -> Features -> Model -> Report
├── tests/                     # Comprehensive Pytest test suite (20 tests)
├── data/                      # Raw parquet cache, LLM cache, and HTML reports
├── config.py                  # Dynamic Pydantic-settings (.env loader)
├── .gitignore                 # Strict rules protecting API keys and sensitive data
├── Dockerfile                 # Containerized reproducible runtime
├── Makefile                   # Automation shortcuts
└── pyproject.toml             # Packaging, Ruff & Mypy settings
```

---

## 4. Key Methodologies & Honest Science

### A. Ingestion & Resilient Zero-Cost Caching
- **https://apify.com/** — Used for data scraping of alternative data features across retail discussion boards (LIHKG + AAStocks for HKEX; Naver Finance + Daum Finance for KRX).
- **[EXA.com](http://exa.com)** — Dedicated AI search engine for financial news retrieval and news sentiment. Performs semantic entity queries (`{English Name} {Local Name} {Code}`) over the lookback window.
- **Parquet Raw Cache**: Automatically persists raw posts, news, and market bars in `data/raw/*.parquet`. Subsequent reruns never re-fetch existing data unless `--refresh` is supplied.

### B. Multilingual LLM Sentiment Scoring & Engagement Weighting
- Accepts texts in **Cantonese, Mandarin Chinese, Korean, and English**.
- Returns validated JSON:
  ```json
  {
    "sentiment": 0.65,
    "direction": "bull",
    "confidence": 0.85,
    "mentions_position": "long",
    "language": "zh"
  }
  ```
- **Engagement Weighting Formula**:
  $$\text{weight} = \max\left(1.0, \ln(1 + \text{likes} + \text{reposts} + \frac{\text{views}}{100}) \times \ln(1 + \text{followers})\right)$$
- **Deduplication & Cache**: Cached by unique SHA-256 content hash in `data/cache/llm_sentiment_cache.json` so duplicate posts are never re-billed.

### C. Strict Forward-Looking Leakage Prevention
- **Rule**: Any post published at time $t$ may **only** influence bars whose start time is $\ge t$.
- Verified via unit test: [`tests/test_features_and_leakage.py`](file:///tests/test_features_and_leakage.py) injects a sentiment shock at 14:30 and proves that bar 14:00 has zero knowledge of the shock.

### D. Support & Resistance (S/R) Detection
- **Swing Highs & Lows**: Extracted using `scipy.signal.argrelextrema` on 1h bars with percentage clustering for multiple touches.
- **Classical Pivot Points**: Computes standard daily floor pivots ($PP, R1, S1, R2, S2$).

### E. Honest Walk-Forward Validation vs Baselines
- **No Random Splits**: Uses expanding window walk-forward validation with an **embargo gap** to eliminate temporal autocorrelation leakage.
- **Pooled Model with Ticker Effects**: Pooled Ridge and LightGBM models with ticker fixed effects.
- **3 Rigorous Baselines**:
  1. *No-Change Baseline*: Predicts zero price return.
  2. *Last-Direction Baseline*: Predicts the sign of the previous bar's return.
  3. *Price-Only Model*: Trained strictly without sentiment features.
- **Statistical Significance**: Computes Binomial test $p$-value for the hit rate vs a 50% random coin toss ($H_0: p=0.5$).
- **Honest Lift Evaluation**: If sentiment fails to beat the price-only model, the system states it plainly on the report badge and metrics table.

---

## 5. Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone <repo_url>
cd henseng-Senti

# Create virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Credentials (`.env`)
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Fill in your API credentials:
```ini
# https://apify.com/ - for data scraping (alternative data features)
APIFY_TOKEN=your_apify_token_here

# http://exa.com - for financial news search & news sentiment
EXA_API_KEY=your_exa_api_key_here

# Groq LLM (or OpenAI compatible)
GROQ_API_KEY=gsk_...
LLM_MODEL=qwen/qwen3.8-27b  # or llama-3.3-70b-versatile, openai/gpt-oss-120b

# Pipeline Storage & Cache
DATA_DIR=data
RAW_CACHE_ENABLED=true
LLM_CACHE_ENABLED=true
```

---

## 6. How to Run

### Run Full Pipeline & Generate Reports
```powershell
# Run both HKEX and KRX (10-day lookback):
python -m interfaces.cli run --all --days 10

# Run a single exchange:
python -m interfaces.cli run --exchange HKEX --days 10
python -m interfaces.cli run --exchange KRX --days 10

# Force-refresh cached bars, Apify scrapes, and EXA.com news:
python -m interfaces.cli run --all --days 10 --refresh
```

### Test Ingestion Only (No Model Training)
```powershell
python -m interfaces.cli test-ingest --exchange HKEX --days 5
python -m interfaces.cli test-ingest --exchange KRX --days 5
```

---

## 7. Interactive HTML Reports

The generated self-contained HTML reports are saved in:
- `data/reports/report_HKEX.html`
- `data/reports/report_KRX.html`

Open them directly in any browser:
```powershell
# Windows
start data\reports\report_HKEX.html
start data\reports\report_KRX.html

# macOS
open data/reports/report_HKEX.html

# Linux
xdg-open data/reports/report_HKEX.html
```

### Each Report Includes:
1. **Interactive Candlestick & Technical Level Charts:**
   - 1-hour OHLCV candles via Plotly
   - Clustered Support & Resistance levels
   - Classical floor pivot points ($PP, R1, S1, R2, S2$)
2. **Real-time Crowd & News Sentiment Breakdown:**
   - Mentions, weighted score $\in [-1.0, +1.0]$, crowd stance (Bullish / Bearish / Neutral)
   - Retail long vs short position claims
   - Headlines pulled via EXA.com
3. **Walk-Forward Validation & Honest Baseline Lift:**
   - Next-bar return prediction ($n+1$)
   - Benchmarked against No-change, Last-direction, and Price-only models
   - Exact Binomial test $p$-value for directional accuracy
4. **Transparent Limitations Box:**
   - Explains lookback sample sizes, API rate limits, and market regime considerations.

---

## 8. Test Suite & Code Quality

Run all 20 tests (100% pass):
```powershell
python -m pytest tests/ -v
```

Run code style & linter checks:
```powershell
python -m ruff check
```

---


