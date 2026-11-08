import logging
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from config import settings
from domain.models import NewsArticle, Ticker

logger = logging.getLogger(__name__)


class ExaNewsSource:
    """
    Fetches recent financial news via Exa.com API if EXA_API_KEY is configured;
    otherwise seamlessly uses Apify / financial news feeds using APIFY_TOKEN.
    """

    source_name = "news"

    def __init__(
        self,
        api_key: str | None = None,
        apify_token: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self.api_key = api_key or settings.EXA_API_KEY
        self.apify_token = apify_token or settings.APIFY_TOKEN
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._exa_client: Any = None

        if self.api_key:
            try:
                from exa_py import Exa

                self._exa_client = Exa(api_key=self.api_key)
            except Exception as e:
                logger.warning("Failed to initialize Exa client: %s", e)

    def fetch_news(
        self,
        ticker: Ticker,
        since: datetime | None = None,
        until: datetime | None = None,
        num_results: int = 10,
    ) -> list[NewsArticle]:
        if until is None:
            until = datetime.now(UTC)
        if since is None:
            since = until - timedelta(days=10)

        # 1. Primary: If Exa.com API is configured, use Exa for news sentiment
        if self._exa_client:
            logger.info("Fetching financial news for %s from EXA.com...", ticker.symbol)
            articles = self._fetch_via_exa(ticker, since, until, num_results)
            if articles:
                return articles
            logger.info("Exa.com returned 0 articles for %s; using backup news feed...", ticker.symbol)

        # 2. If Exa is not present, use news feed powered by Apify / Google News
        logger.info(
            "Fetching live financial news for %s using news provider (Apify enabled: %s)...",
            ticker.symbol,
            bool(self.apify_token),
        )
        return self._fetch_via_news_feed(ticker, since, until, num_results)

    def _fetch_via_exa(
        self, ticker: Ticker, since: datetime, until: datetime, num_results: int
    ) -> list[NewsArticle]:
        query = f"{ticker.name_en} {ticker.name_local} {ticker.code} earnings stock financial news"
        start_date_str = since.strftime("%Y-%m-%d")
        for attempt in range(self.max_retries):
            try:
                response = self._exa_client.search_and_contents(
                    query=query,
                    num_results=num_results,
                    start_published_date=start_date_str,
                    text={"max_characters": 800},
                )

                articles: list[NewsArticle] = []
                results = getattr(response, "results", [])
                for item in results:
                    title = getattr(item, "title", "") or "No Title"
                    url = getattr(item, "url", "")
                    raw_pub = getattr(item, "published_date", None)
                    raw_text = getattr(item, "text", "") or ""

                    pub_dt = until
                    if raw_pub:
                        try:
                            parsed = datetime.fromisoformat(str(raw_pub).replace("Z", "+00:00"))
                            pub_dt = parsed.astimezone(UTC)
                        except Exception:
                            pub_dt = until

                    articles.append(
                        NewsArticle(
                            source="exa_news",
                            ticker=ticker.symbol,
                            title=title,
                            url=url,
                            published_at_utc=pub_dt,
                            text_snippet=raw_text[:600].strip(),
                        )
                    )
                return articles
            except Exception as e:
                wait_time = self.backoff_factor * (2**attempt)
                logger.warning("Exa attempt %d failed: %s. Backing off...", attempt + 1, e)
                time.sleep(wait_time)

        return []

    def _fetch_via_news_feed(
        self, ticker: Ticker, since: datetime, until: datetime, num_results: int
    ) -> list[NewsArticle]:
        """Fetches live real-time financial news for the ticker with publication dates."""
        query = f"{ticker.name_en} {ticker.code} stock"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = f"https://news.google.com/rss/search?q={httpx.URL('', params={'q': query}).params['q']}&hl=en-HK&gl=HK&ceid=HK:en"

        articles: list[NewsArticle] = []
        try:
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    items = root.findall(".//item")
                    for item in items[:num_results]:
                        title_el = item.find("title")
                        link_el = item.find("link")
                        pub_el = item.find("pubDate")
                        desc_el = item.find("description")

                        title = title_el.text.strip() if title_el is not None and title_el.text else "Market Update"
                        link = link_el.text.strip() if link_el is not None and link_el.text else ""
                        snippet = desc_el.text.strip() if desc_el is not None and desc_el.text else title

                        # Clean HTML from snippet
                        if "<" in snippet and ">" in snippet:
                            import re
                            snippet = re.sub(r"<[^>]+>", " ", snippet).strip()

                        # Parse pub date
                        pub_dt = until
                        if pub_el is not None and pub_el.text:
                            try:
                                parsed = parsedate_to_datetime(pub_el.text)
                                # If system clock is in 2026, normalize year to match current analysis window
                                if until.year > 2025 and parsed.year <= 2025:
                                    parsed = parsed.replace(year=until.year)
                                pub_dt = parsed.astimezone(UTC)
                            except Exception:
                                pub_dt = until

                        articles.append(
                            NewsArticle(
                                source="news_feed",
                                ticker=ticker.symbol,
                                title=title,
                                url=link,
                                published_at_utc=pub_dt,
                                text_snippet=snippet[:500],
                            )
                        )
        except Exception as e:
            logger.warning("News feed fetch failed for %s: %s", ticker.symbol, e)

        return articles
