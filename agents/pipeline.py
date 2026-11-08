import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from application.features import compute_leakage_free_features
from application.ingestion import DataIngestionUseCase
from application.model_engine import PooledModelEngine
from application.sentiment import ScoredItem, SentimentAnalysisUseCase
from application.snapshot import SnapshotBuilderUseCase
from domain.enums import Exchange
from domain.models import (
    NewsArticle,
    OHLCVBar,
    Post,
    PredictionResult,
    Ticker,
    TickerSnapshot,
    ValidationMetrics,
)
from interfaces.reporting import ReportGenerator

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    exchange: Exchange
    days: int
    force_refresh: bool
    tickers: list[Ticker]
    bars_by_ticker: dict[str, list[OHLCVBar]]
    posts: list[Post]
    news: list[NewsArticle]
    scored_items: list[ScoredItem]
    feature_dfs: list[Any]
    snapshots: list[TickerSnapshot]
    predictions: list[PredictionResult]
    metrics: ValidationMetrics | None
    eval_rows: list[dict[str, Any]]
    report_path: str


def ingest_step(state: PipelineState) -> dict[str, Any]:
    exchange = state["exchange"]
    days = state.get("days", 10)
    force = state.get("force_refresh", False)

    logger.info("Executing Ingestion Node for %s...", exchange.value)
    use_case = DataIngestionUseCase()
    res = use_case.execute(exchange=exchange, days=days, force_refresh=force)

    return {
        "tickers": res.tickers,
        "bars_by_ticker": res.bars_by_ticker,
        "posts": res.all_posts,
        "news": res.news,
    }


def sentiment_step(state: PipelineState) -> dict[str, Any]:
    posts = state.get("posts", [])
    news = state.get("news", [])

    logger.info("Executing Sentiment Node on %d posts and %d articles...", len(posts), len(news))
    use_case = SentimentAnalysisUseCase()
    scored = use_case.execute(posts=posts, news=news)

    return {"scored_items": scored}


def features_step(state: PipelineState) -> dict[str, Any]:
    tickers = state.get("tickers", [])
    bars_by_ticker = state.get("bars_by_ticker", {})
    scored_items = state.get("scored_items", [])
    news = state.get("news", [])

    logger.info("Executing Features & Snapshot Node across %d tickers...", len(tickers))
    feature_dfs = []
    snapshots = []
    snap_builder = SnapshotBuilderUseCase()

    for ticker in tickers:
        bars = bars_by_ticker.get(ticker.symbol, [])
        # Build snapshot
        snapshot = snap_builder.build_snapshot(ticker, bars, scored_items, news)
        snapshots.append(snapshot)

        # Compute leakage-free aligned features
        df_feat = compute_leakage_free_features(bars, scored_items, ticker)
        if not df_feat.empty:
            feature_dfs.append(df_feat)

    return {
        "snapshots": snapshots,
        "feature_dfs": feature_dfs,
    }


def model_step(state: PipelineState) -> dict[str, Any]:
    exchange = state["exchange"]
    feature_dfs = state.get("feature_dfs", [])
    tickers = state.get("tickers", [])

    logger.info("Executing Model Node (Pooled Walk-Forward Validation)...")
    engine = PooledModelEngine(exchange=exchange, embargo_bars=1)
    pooled_df = engine.prepare_pooled_dataset(feature_dfs)

    predictions: list[PredictionResult] = []
    if not pooled_df.empty:
        metrics, eval_rows = engine.walk_forward_validate(pooled_df, model_type="ridge")
        for ticker in tickers:
            pred = engine.predict_next_bar(pooled_df, ticker.symbol, model_type="ridge")
            predictions.append(pred)
    else:
        # Fallback dummy metrics if data is sparse
        metrics = ValidationMetrics(
            model_name="ridge",
            exchange=exchange,
            sample_count=0,
            mae=0.0,
            rmse=0.0,
            directional_accuracy=50.0,
            binomial_p_value=1.0,
            baseline_no_change_mae=0.0,
            baseline_last_direction_acc=50.0,
            baseline_price_only_mae=0.0,
            baseline_price_only_acc=50.0,
            lift_mae=0.0,
            lift_directional_acc=0.0,
            sentiment_adds_lift=False,
            notes="No features generated.",
        )
        eval_rows = []

    return {
        "metrics": metrics,
        "eval_rows": eval_rows,
        "predictions": predictions,
    }


def report_step(state: PipelineState) -> dict[str, Any]:
    exchange = state["exchange"]
    snapshots = state.get("snapshots", [])
    predictions = state.get("predictions", [])
    metrics = state.get("metrics")
    eval_rows = state.get("eval_rows", [])
    bars = state.get("bars_by_ticker", {})
    days = state.get("days", 10)

    logger.info("Executing Report Generation Node for %s...", exchange.value)
    generator = ReportGenerator()
    report_path = generator.generate_html_report(
        exchange=exchange,
        snapshots=snapshots,
        predictions=predictions,
        metrics=metrics,  # type: ignore
        eval_rows=eval_rows,
        bars_by_ticker=bars,
        days=days,
    )

    return {"report_path": str(report_path)}


class SentimentAgentPipeline:
    """LangGraph-driven end-to-end pipeline: Ingest -> Sentiment -> Features -> Model -> Report."""

    def __init__(self) -> None:
        graph = StateGraph(PipelineState)
        graph.add_node("ingest", ingest_step)
        graph.add_node("sentiment", sentiment_step)
        graph.add_node("features", features_step)
        graph.add_node("model", model_step)
        graph.add_node("report", report_step)

        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "sentiment")
        graph.add_edge("sentiment", "features")
        graph.add_edge("features", "model")
        graph.add_edge("model", "report")
        graph.add_edge("report", END)

        self.compiled = graph.compile()

    def run(
        self,
        exchange: Exchange,
        days: int = 10,
        force_refresh: bool = False,
    ) -> str:
        initial_state: PipelineState = {
            "exchange": exchange,
            "days": days,
            "force_refresh": force_refresh,
        }
        final_state = self.compiled.invoke(initial_state)
        return final_state["report_path"]
