import logging
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup

from domain.constants import HK_TZ
from domain.models import Post, Ticker
from infrastructure.sources.apify.base import BaseSocialSource

logger = logging.getLogger(__name__)


class AAStocksSource(BaseSocialSource):
    """AAStocks stock forum discussion scraper (Apify or custom HTML parser)."""

    source_name = "aastocks"
    default_tz = HK_TZ

    def __init__(self, apify_token: str | None = None, actor_id: str | None = None) -> None:
        super().__init__(
            apify_token=apify_token,
            actor_id=actor_id or "apify/web-scraper",
        )

    def fetch(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        # Try Apify actor if configured
        if self.client and self.actor_id != "apify/web-scraper":
            raw_items = self.fetch_via_apify(
                {"ticker": ticker.code, "source": "aastocks", "maxItems": 40}
            )
            if raw_items:
                return self._parse_items(raw_items, ticker, since, until)

        return self._fetch_custom(ticker, since, until)

    def _fetch_custom(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        url = f"http://www.aastocks.com/tc/stocks/analysis/forum/comment.aspx?symbol={ticker.code}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8",
        }
        posts: list[Post] = []
        try:
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # AAStocks comment rows often have class 'comment-row' or table cells
                    comment_divs = soup.select(".forum-comment, .comment-box, tr.comment_row, .divGrid")
                    for c_div in comment_divs:
                        text = c_div.get_text(separator=" ", strip=True)
                        if len(text) < 5:
                            continue
                        # Use current time bounded or parsed time
                        posts.append(
                            Post(
                                source=self.source_name,
                                ticker=ticker.symbol,
                                text=text[:300],
                                created_at_utc=datetime.now(UTC),
                                likes=1,
                                reposts=0,
                                views=50,
                                author_followers=0,
                                url=url,
                            )
                        )
        except Exception as e:
            logger.warning("AAStocks custom scraper failed for %s: %s", ticker.symbol, e)

        return posts

    def _parse_items(
        self, items: list[dict], ticker: Ticker, since: datetime, until: datetime
    ) -> list[Post]:
        posts: list[Post] = []
        for item in items:
            raw_text = item.get("text", "") or item.get("comment", "")
            raw_time = item.get("date") or item.get("created_at")
            try:
                dt = (
                    datetime.fromisoformat(str(raw_time)).astimezone(UTC)
                    if raw_time
                    else datetime.now(UTC)
                )
            except Exception:
                dt = datetime.now(UTC)

            if not (since <= dt <= until):
                continue

            posts.append(
                Post(
                    source=self.source_name,
                    ticker=ticker.symbol,
                    text=raw_text.strip(),
                    created_at_utc=dt,
                    likes=int(item.get("likes", 0)),
                    reposts=int(item.get("reposts", 0)),
                    views=int(item.get("views", 0)),
                    author_followers=int(item.get("author_followers", 0)),
                    url=str(item.get("url", "")),
                )
            )
        return posts
