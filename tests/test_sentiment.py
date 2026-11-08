from pathlib import Path

from domain.enums import Direction
from domain.models import compute_content_hash
from infrastructure.llm.sentiment_scorer import LLMSentimentScorer


def test_rule_based_fallback_multilingual(tmp_path: Path) -> None:
    scorer = LLMSentimentScorer(api_key=None, cache_dir=tmp_path)

    # Cantonese / Chinese Bullish
    c_hash = compute_content_hash("700大升穿400，睇好後市！", "lihkg")
    res_hk = scorer.score_single("700大升穿400，睇好後市！", c_hash)
    assert res_hk.direction == Direction.BULL
    assert res_hk.sentiment > 0.0
    assert res_hk.language in ["yue", "zh"]

    # Korean Bearish
    k_hash = compute_content_hash("삼전 폭락 악재 발생 손절합니다", "naver")
    res_kr = scorer.score_single("삼전 폭락 악재 발생 손절합니다", k_hash)
    assert res_kr.direction == Direction.BEAR
    assert res_kr.sentiment < 0.0
    assert res_kr.language == "ko"

    # English Neutral / None
    e_hash = compute_content_hash("Market summary today", "exa")
    res_en = scorer.score_single("Market summary today", e_hash)
    assert res_en.direction == Direction.NEUTRAL
    assert res_en.sentiment == 0.0


def test_sentiment_scorer_hash_caching(tmp_path: Path) -> None:
    scorer = LLMSentimentScorer(api_key=None, cache_dir=tmp_path)
    text = "Breakout rally expected tomorrow with strong earnings"
    c_hash = compute_content_hash(text)

    # First call stores in cache
    res1 = scorer.score_single(text, c_hash)
    assert res1.sentiment is not None
    assert c_hash in scorer._cache

    # Modify cached value directly to test retrieval from cache
    scorer._cache[c_hash]["sentiment"] = 0.99
    scorer._cache[c_hash]["direction"] = Direction.BULL.value

    # Second call must retrieve from cache
    res2 = scorer.score_single(text, c_hash)
    assert res2.sentiment == 0.99
    assert res2.raw_response == "from_cache"
