import logging
import time
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from apify_client import ApifyClient

from config import settings
from domain.constants import HK_TZ
from domain.models import Post, Ticker

logger = logging.getLogger(__name__)


class BaseSocialSource:
    """Base class for Apify or custom fallback social media scrapers."""

    source_name: str = "base"
    default_tz: ZoneInfo = HK_TZ

    def __init__(
        self,
        apify_token: str | None = None,
        actor_id: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self.apify_token = apify_token or settings.APIFY_TOKEN
        self.actor_id = actor_id
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.client: ApifyClient | None = None
        if self.apify_token:
            try:
                self.client = ApifyClient(self.apify_token)
            except Exception as e:
                logger.warning("Could not initialize ApifyClient: %s", e)

    def _to_utc(self, dt: datetime, source_tz: ZoneInfo | None = None) -> datetime:
        """Convert naive or local timezone datetime to tz-aware UTC."""
        tz = source_tz or self.default_tz
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(UTC)

    def fetch_via_apify(
        self, run_input: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Run configured Apify actor with retry and backoff."""
        if not self.client or not self.actor_id:
            return []

        for attempt in range(self.max_retries):
            try:
                actor_call = self.client.actor(self.actor_id).call(run_input=run_input)
                dataset_id = actor_call.get("defaultDatasetId")
                if not dataset_id:
                    return []
                items = list(self.client.dataset(dataset_id).iterate_items())
                return items
            except Exception as e:
                wait_time = self.backoff_factor * (2**attempt)
                logger.warning(
                    "Apify actor call %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    self.actor_id,
                    attempt + 1,
                    self.max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)

        logger.error("Apify actor %s exhausted retries.", self.actor_id)
        return []

    def fetch(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        raise NotImplementedError
