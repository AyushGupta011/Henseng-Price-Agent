import io
import logging
import sys

import click

from config import settings
from domain.enums import Exchange

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def main() -> None:
    """hsi-sentiment-agent: Next-bar return predictor using social + news sentiment."""
    pass


@main.command()
@click.option(
    "--exchange",
    type=click.Choice(["HKEX", "KRX"], case_sensitive=False),
    default="HKEX",
    help="Target stock exchange (HKEX or KRX).",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Run analysis for both HKEX and KRX.",
)
@click.option(
    "--days",
    type=int,
    default=10,
    help="Lookback window in days (default: 10).",
)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    help="Bypass cache and force re-scraping / re-downloading.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose debug logging.",
)
def run(exchange: str, run_all: bool, days: int, refresh: bool, verbose: bool) -> None:
    """Execute end-to-end pipeline (ingest -> sentiment -> features -> model -> report)."""
    configure_logging(verbose)
    settings.ensure_directories()

    exchanges = [Exchange.HKEX, Exchange.KRX] if run_all else [Exchange(exchange.upper())]
    click.echo(f"[*] Starting sentiment agent pipeline for: {[e.value for e in exchanges]}")

    for ex in exchanges:
        click.echo(f"\n==================== EXCHANGE: {ex.value} ====================")
        from agents.pipeline import SentimentAgentPipeline

        pipeline = SentimentAgentPipeline()
        report_path = pipeline.run(exchange=ex, days=days, force_refresh=refresh)
        click.secho(f"[+] Success! Report generated at: {report_path}", fg="green", bold=True)


@main.command()
@click.option(
    "--exchange",
    type=click.Choice(["HKEX", "KRX"], case_sensitive=False),
    default="HKEX",
    help="Target stock exchange.",
)
@click.option(
    "--days",
    type=int,
    default=10,
    help="Lookback days.",
)
def test_ingest(exchange: str, days: int) -> None:
    """Test data ingestion (market bars, social scrapers, news) with caching."""
    configure_logging(True)
    from application.ingestion import DataIngestionUseCase

    ex = Exchange(exchange.upper())
    ingest = DataIngestionUseCase()
    res = ingest.execute(exchange=ex, days=days)
    click.secho(f"\n--- Ingestion Summary for {ex.value} ---", fg="cyan", bold=True)
    for sym, bars in res.bars_by_ticker.items():
        click.echo(f"  {sym}: {len(bars)} 1h bars")
    click.echo(f"  Total social posts: {len(res.all_posts)}")
    for src, p_list in res.posts_by_source.items():
        click.echo(f"    - {src}: {len(p_list)} posts")
    click.echo(f"  Total news articles: {len(res.news)}")


if __name__ == "__main__":
    main()
