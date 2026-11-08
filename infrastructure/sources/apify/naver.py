import logging
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup

from domain.constants import KR_TZ
from domain.models import Post, Ticker
from infrastructure.sources.apify.base import BaseSocialSource

logger = logging.getLogger(__name__)


class NaverFinanceSource(BaseSocialSource):
    """Naver Finance Stock Discussion Board (종목토론실) scraper."""

    source_name = "naver_finance"
    default_tz = KR_TZ

    def __init__(self, apify_token: str | None = None, actor_id: str | None = None) -> None:
        super().__init__(
            apify_token=apify_token,
            actor_id=actor_id or "apify/web-scraper",
        )

    def fetch(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        # Try Apify if configured
        if self.client and self.actor_id != "apify/web-scraper":
            raw_items = self.fetch_via_apify(
                {"code": ticker.code, "source": "naver", "maxItems": 50}
            )
            if raw_items:
                return self._parse_items(raw_items, ticker, since, until)

        return self._fetch_custom(ticker, since, until)

    def _fetch_custom(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        url = f"https://finance.naver.com/item/board.naver?code={ticker.code}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        posts: list[Post] = []
        try:
            with httpx.Client(timeout=10.0, headers=headers) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    return []
                # Naver pages are typically encoded in EUC-KR / CP949
                content = resp.content.decode("euc-kr", errors="replace")
                soup = BeautifulSoup(content, "html.parser")

                rows = soup.select("table.type2 tbody tr")
                for row in rows:
                    title_elem = row.select_one("td.title a")
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get("href", "")
                    post_url = f"https://finance.naver.com{href}" if href.startswith("/") else href

                    date_elem = row.select_one("td span.tah.p10.gray03")
                    date_str = date_elem.get_text(strip=True) if date_elem else ""

                    # Parse date: YYYY.MM.DD HH:MM
                    post_dt = until
                    if date_str:
                        try:
                            # 2025.05.01 14:30
                            parsed = datetime.strptime(date_str, "%Y.%m.%d %H:%M")
                            # Localize to KST then convert to UTC
                            kst_dt = parsed.replace(tzinfo=KR_TZ)
                            post_dt = kst_dt.astimezone(UTC)
                        except Exception:
                            post_dt = until

                    if not (since <= post_dt <= until):
                        continue

                    # Views and Upvotes
                    tds = row.select("td")
                    views = 0
                    upvotes = 0
                    if len(tds) >= 6:
                        try:
                            views = int(tds[3].get_text(strip=True))
                            upvotes = int(tds[4].get_text(strip=True))
                        except Exception:
                            pass

                    posts.append(
                        Post(
                            source=self.source_name,
                            ticker=ticker.symbol,
                            text=title,
                            created_at_utc=post_dt,
                            likes=upvotes,
                            reposts=0,
                            views=views,
                            author_followers=0,
                            url=post_url,
                        )
                    )
        except Exception as e:
            logger.warning("Naver Finance scraper failed for %s: %s", ticker.symbol, e)

        # De-duplicate by content hash
        unique: dict[str, Post] = {}
        for p in posts:
            unique[p.content_hash] = p
        return list(unique.values())

    def _parse_items(
        self, items: list[dict], ticker: Ticker, since: datetime, until: datetime
    ) -> list[Post]:
        posts: list[Post] = []
        for item in items:
            raw_text = item.get("title", "") or item.get("text", "")
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
