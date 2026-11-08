# HSI Sentiment Agent (`hsi-sentiment-agent`)

> **Production-grade Next-Bar ($n+1$) Asset Price Return Predictor using Social & News Sentiment across HKEX and KRX Markets.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-brightgreen.svg)]()
[![Orchestration](https://img.shields.io/badge/Pipeline-LangGraph-purple.svg)]()
[![Tests](https://img.shields.io/badge/Tests-20%20Passed-success.svg)]()

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

## 2. Clean Architecture

The system strictly follows **Clean Architecture** with inward-only dependency directions:

```
hsi-sentiment-agent/
├── domain/                    # Pure business entities & protocols (No I/O)
│   ├── models.py              # Post, NewsArticle, OHLCVBar, SentimentScore, etc.
│   ├── enums.py               # Direction, PositionClaim, Exchange, LevelType
│   ├── constants.py           # Ticker definitions, timezones (HKT, KST, UTC)
│   └── interfaces.py          # SocialSource, NewsSource, MarketDataSource protocols
├── application/               # Application use cases & business logic
│   ├── ingestion.py           # Ingestion orchestrator
│   ├── sentiment.py           # Sentiment analysis coordinator
│   ├── features.py            # Leakage-free feature alignment & lagging
│   ├── snapshot.py            # Ticker snapshot & crowd stance aggregation
│   └── model_engine.py        # Pooled walk-forward validation & baselines
├── infrastructure/            # Adapters, external APIs, and persistence
│   ├── sources/
│   │   ├── apify/             # LIHKG, AAStocks, Naver, Daum scrapers
│   │   ├── exa_news_source.py # Exa.com financial news client
│   │   └── yfinance_source.py # 1h OHLCV bar fetcher with UTC tz conversion
│   ├── llm/
│   │   └── sentiment_scorer.py# Multilingual LLM scorer with hash caching
│   ├── analysis/
│   │   └── support_resistance.py # SciPy swing extrema & classical pivots
│   └── cache/
│       └── parquet_repository.py # data/raw/*.parquet zero-cost cache
├── interfaces/                # User delivery & UI
│   ├── cli.py                 # Click CLI interface
│   └── reporting.py           # Plotly + Jinja2 self-contained HTML report
├── agents/                    # LangGraph orchestration
│   └── pipeline.py            # StateGraph: Ingest -> Sentiment -> Features -> Model -> Report
├── tests/                     # Comprehensive Pytest test suite (20 tests)
├── data/                      # Parquet cache, logs, and generated HTML reports
├── config.py                  # Pydantic-settings config (.env loader)
├── Dockerfile                 # Containerized runtime
├── Makefile                   # Automation shortcuts
└── pyproject.toml             # Packaging, Ruff & Mypy settings
```

---

## 3. Key Methodologies & Honest Science

### A. Ingestion & Resilient Zero-Cost Caching
- **Social Sources**: Scrapes 2 distinct sources per exchange (LIHKG + AAStocks for HKEX; Naver Finance + Daum Finance for KRX).
- **Apify Integration**: Supports Apify cloud actors or automatic fallback scrapers with rate-limit backoff.
- **Exa News**: Multilingual entity search (`{English Name} {Local Name} {Code}`) over the lookback window.
- **Parquet Raw Cache**: Automatically persists raw posts, news, and market bars in `data/raw/*.parquet`. Subsequent reruns never re-fetch existing data.

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
- **Swing Highs & Lows**: Extracted using `scipy.signal.argrelextrema` on 1h bars with clustering for multiple touches.
- **Classical Pivot Points**: Computes standard daily pivots ($PP, R1, S1, R2, S2$).

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

## 4. Quickstart & How to Run

### Installation
```bash
# Clone and enter workspace
git clone <repo_url>
cd hsi-sentiment-agent

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Environment Configuration (Optional)
Copy `.env.example` to `.env` and fill in API credentials if available:
```bash
cp .env.example .env
```
```ini
APIFY_TOKEN=your_apify_token_here
EXA_API_KEY=your_exa_api_key_here
LLM_API_KEY=your_openai_or_compatible_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```
*(Note: If API keys are omitted, the pipeline runs using rule-based multilingual parsing and cached parquet data without crashing).*

---

### Running the CLI

#### 1. Run Complete Pipeline (Ingest -> Sentiment -> Features -> Model -> Report)
```bash
# Run for HKEX (default 10 days lookback)
python -m interfaces.cli run --exchange HKEX --days 10

# Run for KRX
python -m interfaces.cli run --exchange KRX --days 10

# Run for both exchanges
python -m interfaces.cli run --all --days 10
```

#### 2. Test Ingestion Only (Bars, Social, News)
```bash
python -m interfaces.cli test-ingest --exchange HKEX --days 5
```

---

## 5. Verification & Test Suite

Run all 20 tests covering domain entities, UTC enforcement, leakage prevention, S/R logic, and walk-forward validation:

```bash
pytest tests/ -v
```

Check code style and types:
```bash
ruff check .
mypy domain config.py
```

---

## 6. Docker & Makefile

```bash
# Using Makefile
make test
make run-hkex
make run-krx
make run-all

# Using Docker
make docker-build
make docker-run
```

---

## 7. Sample Output

Generated interactive reports are saved to:
- `data/reports/report_HKEX.html`
- `data/reports/report_KRX.html`

Each self-contained report includes:
- **Ranked Table**: Mentions, weighted sentiment, crowd long/short claims, $n+1$ target return and price.
- **Interactive Plotly Candlesticks**: 1h bars with green Support, red Resistance, and amber Pivot lines.
- **Walk-Forward Validation Chart**: Out-of-sample predicted vs actual returns.
- **Baselines Table**: Directional accuracy, binomial $p$-value, MAE lift vs price-only baseline.
- **Limitations Box**: Retail sentiment noise, slippage, and sample constraints.
