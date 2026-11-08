import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from config import settings
from domain.enums import Direction, PositionClaim
from domain.models import SentimentScore, compute_content_hash

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior multilingual financial sentiment analyst specializing in Hong Kong (HKEX) and Korean (KRX) equity markets.
Analyze the given text (which may be in Cantonese, Mandarin Chinese, Korean, or English).
Extract financial sentiment, direction, confidence, explicit position claims, and detected language.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "sentiment": <float between -1.0 (extremely bearish) and 1.0 (extremely bullish)>,
  "direction": "<bull | bear | neutral>",
  "confidence": <float between 0.0 and 1.0>,
  "mentions_position": "<long | short | none>",
  "language": "<en | zh | yue | ko>"
}

Do not include any commentary or markdown outside the JSON object.
"""


class LLMSentimentScorer:
    """Multilingual LLM sentiment scorer with disk caching and resilient fallbacks."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        cache_dir: Path | str = "data/cache",
    ) -> None:
        self.api_key = api_key or settings.LLM_API_KEY
        raw_url = base_url or settings.LLM_BASE_URL or "https://api.openai.com/v1"
        self.base_url = raw_url.rstrip("/")
        self.model = model or settings.LLM_MODEL or "gpt-4o-mini"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "llm_sentiment_cache.json"
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load sentiment cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save sentiment cache: %s", e)

    def _rule_based_fallback(self, text: str, content_hash: str) -> SentimentScore:
        """Heuristic multilingual fallback when LLM is unavailable or unparseable."""
        t = text.lower()
        bull_words = [
            "buy", "bull", "surge", "gain", "breakout", "rally", "growth",
            "升", "睇好", "買", "好倉", "牛", "大升", "破頂", "暴漲", "看好", "买入",
            "상승", "매수", "호재", "폭등", "반등", "급등", "목표가 상향", "추천"
        ]
        bear_words = [
            "sell", "bear", "plunge", "drop", "dump", "crash", "fall", "loss",
            "跌", "睇淡", "沽", "淡倉", "熊", "大跌", "沽空", "看淡", "卖出", "做空",
            "하락", "매도", "악재", "폭락", "급락", "손절", "목표가 하향"
        ]

        bull_count = sum(1 for w in bull_words if w in t)
        bear_count = sum(1 for w in bear_words if w in t)

        pos_claim = PositionClaim.NONE
        if any(w in t for w in ["hold", "bought", "long", "買入", "持有", "好倉", "매수 완료", "보유"]):
            pos_claim = PositionClaim.LONG
        elif any(w in t for w in ["short", "sold", "沽空", "賣出", "淡倉", "매도 완료", "숏"]):
            pos_claim = PositionClaim.SHORT

        # Detect language hint
        lang = "en"
        if any("\uac00" <= c <= "\ud7a3" for c in text):
            lang = "ko"
        elif any("\u4e00" <= c <= "\u9fff" for c in text):
            lang = "yue" if any(c in text for c in ["睇", "唔", "嘅", "喺", "咗", "點"]) else "zh"

        if bull_count > bear_count:
            sentiment = min(1.0, 0.3 + 0.15 * (bull_count - bear_count))
            return SentimentScore(
                content_hash=content_hash,
                sentiment=round(sentiment, 2),
                direction=Direction.BULL,
                confidence=0.7,
                mentions_position=pos_claim,
                language=lang,
                raw_response="rule_based_fallback",
            )
        elif bear_count > bull_count:
            sentiment = max(-1.0, -0.3 - 0.15 * (bear_count - bull_count))
            return SentimentScore(
                content_hash=content_hash,
                sentiment=round(sentiment, 2),
                direction=Direction.BEAR,
                confidence=0.7,
                mentions_position=pos_claim,
                language=lang,
                raw_response="rule_based_fallback",
            )
        else:
            return SentimentScore(
                content_hash=content_hash,
                sentiment=0.0,
                direction=Direction.NEUTRAL,
                confidence=0.5,
                mentions_position=pos_claim,
                language=lang,
                raw_response="rule_based_fallback",
            )

    def score_single(self, text: str, content_hash: str | None = None) -> SentimentScore:
        c_hash = content_hash or compute_content_hash(text)

        # Check cache
        if c_hash in self._cache:
            data = self._cache[c_hash]
            try:
                return SentimentScore(
                    content_hash=c_hash,
                    sentiment=float(data["sentiment"]),
                    direction=Direction(data["direction"]),
                    confidence=float(data["confidence"]),
                    mentions_position=PositionClaim(data.get("mentions_position", "none")),
                    language=str(data.get("language", "en")),
                    raw_response="from_cache",
                )
            except Exception as e:
                logger.debug("Cache entry invalid for %s: %s", c_hash, e)

        # If no API key configured, use rule-based scorer
        if not self.api_key:
            score = self._rule_based_fallback(text, c_hash)
            self._cache[c_hash] = {
                "sentiment": score.sentiment,
                "direction": score.direction.value,
                "confidence": score.confidence,
                "mentions_position": score.mentions_position.value,
                "language": score.language,
            }
            return score

        # Query LLM endpoint (OpenAI chat completions compatible)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{text}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        for attempt in range(3):
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code == 200:
                        raw_content = resp.json()["choices"][0]["message"]["content"]
                        parsed = json.loads(raw_content)

                        # Validate bounds & schema
                        raw_sent = float(parsed.get("sentiment", 0.0))
                        sent = max(-1.0, min(1.0, raw_sent))
                        raw_dir = str(parsed.get("direction", "neutral")).lower()
                        direction = (
                            Direction(raw_dir)
                            if raw_dir in [Direction.BULL.value, Direction.BEAR.value, Direction.NEUTRAL.value]
                            else Direction.NEUTRAL
                        )
                        conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                        raw_pos = str(parsed.get("mentions_position", "none")).lower()
                        pos = (
                            PositionClaim(raw_pos)
                            if raw_pos in [PositionClaim.LONG.value, PositionClaim.SHORT.value, PositionClaim.NONE.value]
                            else PositionClaim.NONE
                        )
                        lang = str(parsed.get("language", "en"))

                        score = SentimentScore(
                            content_hash=c_hash,
                            sentiment=round(sent, 3),
                            direction=direction,
                            confidence=round(conf, 3),
                            mentions_position=pos,
                            language=lang,
                            raw_response=raw_content,
                        )

                        self._cache[c_hash] = {
                            "sentiment": score.sentiment,
                            "direction": score.direction.value,
                            "confidence": score.confidence,
                            "mentions_position": score.mentions_position.value,
                            "language": score.language,
                        }
                        return score

                    elif resp.status_code == 429:
                        # Rate limit: wait and retry
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    else:
                        logger.warning("LLM API returned %d: %s", resp.status_code, resp.text)
                        break
            except Exception as e:
                logger.warning("LLM scoring attempt %d failed: %s", attempt + 1, e)
                time.sleep(1.0)

        # Fallback to neutral/rule-based
        fallback = self._rule_based_fallback(text, c_hash)
        self._cache[c_hash] = {
            "sentiment": fallback.sentiment,
            "direction": fallback.direction.value,
            "confidence": fallback.confidence,
            "mentions_position": fallback.mentions_position.value,
            "language": fallback.language,
        }
        return fallback

    def score_batch(
        self, items: list[dict[str, Any]]
    ) -> list[SentimentScore]:
        """Score a list of {'text': str, 'content_hash': str} items with caching."""
        results: list[SentimentScore] = []
        for item in items:
            text = item.get("text", "")
            c_hash = item.get("content_hash") or compute_content_hash(text)
            weight = float(item.get("engagement_weight", 1.0))
            score = self.score_single(text, c_hash)
            # Gentle pacing to respect API rate limits
            if self.api_key and score.raw_response != "from_cache":
                time.sleep(0.15)
            # Re-bind engagement weight
            score_with_weight = SentimentScore(
                content_hash=score.content_hash,
                sentiment=score.sentiment,
                direction=score.direction,
                confidence=score.confidence,
                mentions_position=score.mentions_position,
                language=score.language,
                raw_response=score.raw_response,
                engagement_weight=weight,
            )
            results.append(score_with_weight)

        self._save_cache()
        return results
