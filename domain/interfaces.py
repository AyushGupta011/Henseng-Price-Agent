from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from domain.models import NewsArticle, OHLCVBar, Post, SentimentScore, Ticker


@runtime_checkable
class SocialSource(Protocol):
    """Interface for social media scraping sources."""

    source_name: str

    def fetch(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        """Fetch posts for ticker strictly within [since, until] UTC."""
        ...


@runtime_checkable
class NewsSource(Protocol):
    """Interface for news search sources."""

    source_name: str

    def fetch_news(
        self, ticker: Ticker, since: datetime, until: datetime
    ) -> list[NewsArticle]:
        """Fetch news articles for ticker within [since, until] UTC."""
        ...


@runtime_checkable
class MarketDataSource(Protocol):
    """Interface for OHLCV market bars."""

    def fetch_bars(
        self, ticker: str, start: datetime, end: datetime, interval: str = "1h"
    ) -> list[OHLCVBar]:
        """Fetch 1h OHLCV bars for ticker."""
        ...


@runtime_checkable
class LLMScorer(Protocol):
    """Interface for multilingual sentiment scoring."""

    def score_batch(
        self, items: list[dict[str, Any]]
    ) -> list[SentimentScore]:
        """Score a batch of texts returning structured SentimentScore entities."""
        ...


@runtime_checkable
class RawDataRepository(Protocol):
    """Interface for caching raw ingestion to parquet."""

    def save_posts(self, posts: list[Post], exchange: str, source: str) -> None:
        ...

    def load_posts(
        self, exchange: str, source: str, ticker: str | None = None
    ) -> list[Post]:
        ...

    def save_news(self, news: list[NewsArticle], exchange: str) -> None:
        ...

    def load_news(self, exchange: str, ticker: str | None = None) -> list[NewsArticle]:
        ...

    def save_bars(self, bars: list[OHLCVBar], ticker: str) -> None:
        ...

    def load_bars(self, ticker: str) -> list[OHLCVBar]:
        ...
