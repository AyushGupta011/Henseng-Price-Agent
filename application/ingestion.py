import logging
from datetime import UTC, datetime, timedelta

from config import settings
from domain.constants import TICKERS_BY_EXCHANGE
from domain.enums import Exchange
from domain.models import NewsArticle, OHLCVBar, Post, Ticker
from infrastructure.cache.parquet_repository import ParquetRawRepository
from infrastructure.sources.apify import (
    AAStocksSource,
    DaumFinanceSource,
    LIHKGSource,
    NaverFinanceSource,
)
from infrastructure.sources.exa_news_source import ExaNewsSource
from infrastructure.sources.yfinance_source import YFinanceMarketSource

logger = logging.getLogger(__name__)


class IngestionResult:

    def __init__(
        self,
        exchange: Exchange,
        tickers: list[Ticker],
        bars_by_ticker: dict[str, list[OHLCVBar]],
        posts_by_source: dict[str, list[Post]],
        news: list[NewsArticle],
    ) -> None:
        self.exchange = exchange
        self.tickers = tickers
        self.bars_by_ticker = bars_by_ticker
        self.posts_by_source = posts_by_source
        self.news = news

    @property
    def all_posts(self) -> list[Post]:
        combined: list[Post] = []
        for posts in self.posts_by_source.values():
            combined.extend(posts)
        return combined


class DataIngestionUseCase:
    """Orchestrates social, news, and market data ingestion with parquet caching."""

    def __init__(
        self,
        cache_repo: ParquetRawRepository | None = None,
        market_source: YFinanceMarketSource | None = None,
        news_source: ExaNewsSource | None = None,
    ) -> None:
        self.cache = cache_repo or ParquetRawRepository(settings.RAW_CACHE_DIR)
        self.market_source = market_source or YFinanceMarketSource()
        self.news_source = news_source or ExaNewsSource()

    def get_social_sources_for_exchange(self, exchange: Exchange) -> list[object]:
        if exchange == Exchange.HKEX:
            return [LIHKGSource(), AAStocksSource()]
        else:
            return [NaverFinanceSource(), DaumFinanceSource()]

    def execute(
        self,
        exchange: Exchange,
        days: int = 10,
        force_refresh: bool = False,
    ) -> IngestionResult:
        settings.ensure_directories()
        ticker_dicts = TICKERS_BY_EXCHANGE.get(exchange, [])
        tickers = [Ticker(exchange=exchange, **td) for td in ticker_dicts]

        until = datetime.now(UTC)
        since = until - timedelta(days=days)

        logger.info(
            "Starting ingestion for %s across %d tickers (window: %s to %s)",
            exchange.value,
            len(tickers),
            since.strftime("%Y-%m-%d"),
            until.strftime("%Y-%m-%d"),
        )

        # 1. Market Bars (1h)
        bars_by_ticker: dict[str, list[OHLCVBar]] = {}
        for ticker in tickers:
            cached_bars = [] if force_refresh else self.cache.load_bars(ticker.symbol)
            if cached_bars and len(cached_bars) >= 10:
                logger.info("Loaded %d cached 1h bars for %s", len(cached_bars), ticker.symbol)
                bars_by_ticker[ticker.symbol] = cached_bars
            else:
                logger.info("Fetching live 1h bars via yfinance for %s", ticker.symbol)
                live_bars = self.market_source.fetch_bars(
                    ticker.symbol, start=since - timedelta(days=5), end=until
                )
                if live_bars:
                    self.cache.save_bars(live_bars, ticker.symbol)
                    bars_by_ticker[ticker.symbol] = live_bars
                else:
                    bars_by_ticker[ticker.symbol] = cached_bars

        # 2. Social Posts (2 sources)
        sources = self.get_social_sources_for_exchange(exchange)
        posts_by_source: dict[str, list[Post]] = {}

        for src in sources:
            source_name = getattr(src, "source_name", "unknown")
            all_source_posts: list[Post] = []
            cached_posts = (
                [] if force_refresh else self.cache.load_posts(exchange.value, source_name)
            )

            if cached_posts:
                logger.info(
                    "Loaded %d cached posts for %s (%s)",
                    len(cached_posts),
                    exchange.value,
                    source_name,
                )
                all_source_posts.extend(cached_posts)
            else:
                logger.info(
                    "Scraping alternative data via %s for %s tickers (Apify: %s)...",
                    source_name,
                    exchange.value,
                    bool(getattr(src, "client", None)),
                )
                for ticker in tickers:
                    try:
                        ticker_posts = getattr(src, "fetch")(ticker, since, until)
                        all_source_posts.extend(ticker_posts)
                    except Exception as e:
                        logger.warning("Error fetching %s for %s: %s", source_name, ticker.symbol, e)

                if all_source_posts:
                    self.cache.save_posts(all_source_posts, exchange.value, source_name)

            posts_by_source[source_name] = all_source_posts

        # 3. News Articles (EXA.com for news sentiment)
        cached_news = [] if force_refresh else self.cache.load_news(exchange.value)
        news_articles: list[NewsArticle] = []
        if cached_news:
            logger.info("Loaded %d cached news articles for %s", len(cached_news), exchange.value)
            news_articles.extend(cached_news)
        else:
            logger.info("Pulling EXA.com news sentiment for %s tickers...", exchange.value)
            for ticker in tickers:
                try:
                    articles = self.news_source.fetch_news(ticker, since, until)
                    news_articles.extend(articles)
                except Exception as e:
                    logger.warning("Error pulling Exa news for %s: %s", ticker.symbol, e)

            if news_articles:
                self.cache.save_news(news_articles, exchange.value)

        return IngestionResult(
            exchange=exchange,
            tickers=tickers,
            bars_by_ticker=bars_by_ticker,
            posts_by_source=posts_by_source,
            news=news_articles,
        )
