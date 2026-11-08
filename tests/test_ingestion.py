from datetime import UTC, datetime, timedelta
from pathlib import Path

from domain.enums import Exchange
from domain.models import NewsArticle, OHLCVBar, Post, Ticker
from infrastructure.cache.parquet_repository import ParquetRawRepository
from infrastructure.sources.apify.lihkg import LIHKGSource
from infrastructure.sources.apify.naver import NaverFinanceSource


def test_parquet_repository_posts(tmp_path: Path) -> None:
    repo = ParquetRawRepository(tmp_path)
    now = datetime.now(UTC)

    posts = [
        Post(
            source="lihkg",
            ticker="0700.HK",
            text="Tencent cloud growing fast",
            created_at_utc=now,
            likes=10,
            reposts=5,
            views=200,
        ),
        Post(
            source="lihkg",
            ticker="0700.HK",
            text="Another post",
            created_at_utc=now - timedelta(hours=1),
            likes=0,
            reposts=0,
            views=10,
        ),
    ]

    repo.save_posts(posts, exchange="hkex", source="lihkg")

    loaded = repo.load_posts(exchange="hkex", source="lihkg")
    assert len(loaded) == 2
    assert loaded[0].ticker == "0700.HK"
    assert loaded[0].created_at_utc.tzinfo == UTC
    assert loaded[0].likes == 10

    # Test ticker filter
    filtered = repo.load_posts(exchange="hkex", source="lihkg", ticker="0700.HK")
    assert len(filtered) == 2
    empty = repo.load_posts(exchange="hkex", source="lihkg", ticker="9988.HK")
    assert len(empty) == 0


def test_parquet_repository_news(tmp_path: Path) -> None:
    repo = ParquetRawRepository(tmp_path)
    now = datetime.now(UTC)

    news = [
        NewsArticle(
            source="exa_news",
            ticker="005930.KS",
            title="Samsung announces new semiconductor roadmap",
            url="https://example.com/samsung-news",
            published_at_utc=now,
            text_snippet="Samsung Electronics revealed today its latest 2nm chip timeline...",
        )
    ]

    repo.save_news(news, exchange="krx")
    loaded = repo.load_news(exchange="krx")
    assert len(loaded) == 1
    assert loaded[0].ticker == "005930.KS"
    assert "2nm" in loaded[0].text_snippet
    assert loaded[0].published_at_utc.tzinfo == UTC


def test_parquet_repository_bars(tmp_path: Path) -> None:
    repo = ParquetRawRepository(tmp_path)
    now = datetime.now(UTC)

    bars = [
        OHLCVBar(
            ticker="0700.HK",
            timestamp_utc=now - timedelta(hours=1),
            open=350.0,
            high=352.0,
            low=349.0,
            close=351.5,
            volume=500000.0,
        ),
        OHLCVBar(
            ticker="0700.HK",
            timestamp_utc=now,
            open=351.5,
            high=354.0,
            low=351.0,
            close=353.0,
            volume=450000.0,
        ),
    ]

    repo.save_bars(bars, ticker="0700.HK")
    loaded = repo.load_bars(ticker="0700.HK")
    assert len(loaded) == 2
    assert loaded[0].close == 351.5
    assert loaded[1].close == 353.0
    assert loaded[0].timestamp_utc < loaded[1].timestamp_utc


def test_lihkg_custom_parser_logic() -> None:
    src = LIHKGSource()
    ticker = Ticker(
        symbol="0700.HK",
        exchange=Exchange.HKEX,
        name_en="Tencent Holdings",
        name_local="騰訊控股",
        code="0700",
    )
    now = datetime.now(UTC)
    mock_items = [
        {
            "title": "700升穿350啦！",
            "text": "大家睇好唔睇好？",
            "date": now.isoformat(),
            "likes": 25,
            "reposts": 8,
            "views": 350,
            "url": "https://lihkg.com/thread/123",
        }
    ]
    posts = src._parse_items(mock_items, ticker, now - timedelta(days=1), now + timedelta(days=1))
    assert len(posts) == 1
    assert posts[0].ticker == "0700.HK"
    assert posts[0].created_at_utc.tzinfo == UTC
    assert posts[0].likes == 25


def test_naver_custom_parser_logic() -> None:
    src = NaverFinanceSource()
    ticker = Ticker(
        symbol="005930.KS",
        exchange=Exchange.KRX,
        name_en="Samsung Electronics",
        name_local="삼성전자",
        code="005930",
    )
    now = datetime.now(UTC)
    mock_items = [
        {
            "title": "삼전 드디어 반등 시작인가요",
            "date": now.isoformat(),
            "likes": 12,
            "reposts": 3,
            "views": 150,
            "url": "https://finance.naver.com/item/board_read.naver?code=005930",
        }
    ]
    posts = src._parse_items(mock_items, ticker, now - timedelta(days=1), now + timedelta(days=1))
    assert len(posts) == 1
    assert posts[0].ticker == "005930.KS"
    assert posts[0].created_at_utc.tzinfo == UTC
    assert posts[0].likes == 12
