import logging
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from domain.enums import Exchange
from domain.models import PredictionResult, TickerSnapshot, ValidationMetrics

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HSI Sentiment Agent - {{ exchange }} Market Analysis & n+1 Predictions</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --bull: #10b981;
      --bear: #ef4444;
      --neutral: #64748b;
      --accent: #3b82f6;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }
    .container {
      max-width: 1300px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    h1 { margin: 0; font-size: 26px; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-bull { background: rgba(16, 185, 129, 0.2); color: var(--bull); }
    .badge-bear { background: rgba(239, 68, 68, 0.2); color: var(--bear); }
    .badge-neutral { background: rgba(100, 116, 139, 0.2); color: var(--neutral); }
    .badge-lift-yes { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid var(--bull); }
    .badge-lift-no { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid var(--bear); }

    .card {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }
    h2 { font-size: 18px; margin-top: 0; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background-color: rgba(255, 255, 255, 0.03);
      color: var(--text-muted);
      font-weight: 600;
    }
    tr:hover { background-color: rgba(255, 255, 255, 0.02); }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }
    .metric-card {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 16px;
    }
    .metric-label { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }
    .metric-value { font-size: 20px; font-weight: 700; }

    .limitations-box {
      border-left: 4px solid #f59e0b;
      background: rgba(245, 158, 11, 0.08);
      padding: 16px;
      border-radius: 0 8px 8px 0;
      margin-top: 24px;
    }
    .limitations-box h3 {
      color: #fbbf24;
      margin-top: 0;
      font-size: 16px;
    }
    .limitations-box ul {
      margin: 0;
      padding-left: 20px;
      color: #cbd5e1;
      font-size: 13px;
      line-height: 1.6;
    }
    .chart-container {
      margin-bottom: 24px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>HSI Sentiment Agent — {{ exchange }}</h1>
        <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">
          Generated: {{ generated_at }} UTC &bull; Lookback: {{ days }} Days &bull; Pooled ML Engine
        </div>
      </div>
      <div>
        {% if metrics.sentiment_adds_lift %}
          <span class="badge badge-lift-yes">Sentiment Alpha Lift Detected (+{{ metrics.lift_directional_acc }}%)</span>
        {% else %}
          <span class="badge badge-lift-no">No Lift Over Price Baseline ({{ metrics.lift_directional_acc }}%)</span>
        {% endif %}
      </div>
    </header>

    <!-- SECTION 1: RANKED TABLE -->
    <div class="card">
      <h2>1. Ticker Sentiment & n+1 Prediction Summary (Ranked by Sentiment & Activity)</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Ticker</th>
            <th>Asset Name</th>
            <th>Last Price</th>
            <th>Weighted Sent.</th>
            <th>Direction</th>
            <th>Mentions</th>
            <th>Crowd Stance</th>
            <th>Predicted Price (n+1)</th>
            <th>Target Return</th>
          </tr>
        </thead>
        <tbody>
          {% for r in ranked_snapshots %}
          <tr>
            <td><strong>#{{ loop.index }}</strong></td>
            <td><code>{{ r.snapshot.ticker }}</code></td>
            <td>{{ r.snapshot.asset_name }}</td>
            <td>{{ "%.2f"|format(r.snapshot.current_price) }}</td>
            <td>
              <span style="color: {% if r.snapshot.weighted_sentiment > 0.05 %}var(--bull){% elif r.snapshot.weighted_sentiment < -0.05 %}var(--bear){% else %}var(--neutral){% endif %}; font-weight: 600;">
                {{ "%+.3f"|format(r.snapshot.weighted_sentiment) }}
              </span>
            </td>
            <td>
              <span class="badge badge-{{ r.snapshot.direction.value }}">
                {{ r.snapshot.direction.value }}
              </span>
            </td>
            <td>{{ r.snapshot.mention_count }}</td>
            <td>
              <small>{{ r.snapshot.crowd_position.suggested_stance }}</small>
              <div style="font-size: 11px; color: var(--text-muted);">
                L: {{ r.snapshot.crowd_position.long_pct }}% | S: {{ r.snapshot.crowd_position.short_pct }}%
              </div>
            </td>
            <td>
              <strong>{{ "%.2f"|format(r.prediction.predicted_price) }}</strong>
            </td>
            <td>
              <span style="color: {% if r.prediction.predicted_log_return > 0 %}var(--bull){% elif r.prediction.predicted_log_return < 0 %}var(--bear){% else %}var(--neutral){% endif %};">
                {{ "%+.2f%%"|format(r.prediction.predicted_log_return * 100) }}
              </span>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <!-- SECTION 2: CHARTS -->
    <div class="card">
      <h2>2. Per-Ticker Market Dynamics & Support/Resistance Levels</h2>
      {% for c in ticker_charts %}
        <div class="chart-container">
          <div id="{{ c.chart_id }}"></div>
          <script>
            var chartData = {{ c.chart_json }};
            Plotly.newPlot('{{ c.chart_id }}', chartData.data, chartData.layout, {responsive: true});
          </script>
        </div>
      {% endfor %}
    </div>

    <!-- SECTION 3: MODEL VS BASELINE VALIDATION -->
    <div class="card">
      <h2>3. Honest Walk-Forward Validation vs Baselines</h2>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Model Directional Accuracy</div>
          <div class="metric-value" style="color: var(--accent);">{{ metrics.directional_accuracy }}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Price-Only Baseline Acc</div>
          <div class="metric-value">{{ metrics.baseline_price_only_acc }}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Last-Direction Baseline Acc</div>
          <div class="metric-value">{{ metrics.baseline_last_direction_acc }}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Binomial p-value (vs 50%)</div>
          <div class="metric-value" style="color: {% if metrics.binomial_p_value < 0.05 %}var(--bull){% else %}var(--text-muted){% endif %};">
            {{ "%.4f"|format(metrics.binomial_p_value) }}
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Model MAE (Log Return)</div>
          <div class="metric-value">{{ "%.5f"|format(metrics.mae) }}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">No-Change Baseline MAE</div>
          <div class="metric-value">{{ "%.5f"|format(metrics.baseline_no_change_mae) }}</div>
        </div>
      </div>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 0;">
        <strong>Validation Note:</strong> {{ metrics.notes }}
      </p>

      {% if eval_chart_json %}
      <div style="margin-top: 20px;">
        <div id="eval_chart"></div>
        <script>
          var evalData = {{ eval_chart_json }};
          Plotly.newPlot('eval_chart', evalData.data, evalData.layout, {responsive: true});
        </script>
      </div>
      {% endif %}
    </div>

    <!-- SECTION 4: LIMITATIONS BOX -->
    <div class="limitations-box">
      <h3>Limitations & Assumptions</h3>
      <ul>
        <li><strong>Leakage Prevention:</strong> Strict time boundaries are enforced. Posts published at time <em>t</em> only influence price bars starting at or after <em>t</em>.</li>
        <li><strong>Lookback Constraint:</strong> The current model is trained over the last 10 days of 1h bars. High-frequency regime shifts may not be fully represented in short horizons.</li>
        <li><strong>Retail vs Institutional Noise:</strong> Social media posts (LIHKG, Naver) frequently represent retail crowd momentum which can either lead breakouts or serve as contrarian liquidity indicators.</li>
        <li><strong>Execution Slippage & Trading Hours:</strong> Next-bar predictions assume execution at the immediate open of bar <em>n+1</em>. Overnight gaps and auction sessions introduce slippage not modeled here.</li>
      </ul>
    </div>
  </div>
</body>
</html>
"""


class ReportGenerator:
    """Generates self-contained interactive HTML reports with Plotly and Jinja2."""

    def __init__(self, output_dir: Path | str = "data/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        exchange: Exchange,
        snapshots: list[TickerSnapshot],
        predictions: list[PredictionResult],
        metrics: ValidationMetrics,
        eval_rows: list[dict[str, object]],
        bars_by_ticker: dict[str, list[object]],
        days: int = 10,
    ) -> Path:
        # 1. Rank snapshots by mention count and sentiment
        ranked_items = []
        for s in snapshots:
            pred = next((p for p in predictions if p.ticker == s.ticker), None)
            if not pred:
                pred = PredictionResult(
                    ticker=s.ticker,
                    target_bar_time=datetime.now(UTC),
                    current_price=s.current_price,
                    predicted_log_return=0.0,
                    predicted_price=s.current_price,
                )
            ranked_items.append({"snapshot": s, "prediction": pred})

        # Sort by mentions descending, then absolute sentiment
        ranked_items.sort(
            key=lambda x: (x["snapshot"].mention_count, abs(x["snapshot"].weighted_sentiment)),
            reverse=True,
        )

        # 2. Build Plotly charts for each ticker
        ticker_charts = []
        for item in ranked_items:
            s: TickerSnapshot = item["snapshot"]
            bars = bars_by_ticker.get(s.ticker, [])
            chart_json = self._build_ticker_chart(s, bars)
            ticker_charts.append(
                {
                    "chart_id": f"chart_{s.ticker.replace('.', '_')}",
                    "chart_json": chart_json,
                }
            )

        # 3. Build Evaluation Predicted vs Actual Chart
        eval_chart_json = self._build_eval_chart(eval_rows)

        # 4. Render Jinja2 template
        template = Template(HTML_TEMPLATE)
        rendered_html = template.render(
            exchange=exchange.value,
            generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            days=days,
            ranked_snapshots=ranked_items,
            metrics=metrics,
            ticker_charts=ticker_charts,
            eval_chart_json=eval_chart_json,
        )

        output_path = self.output_dir / f"report_{exchange.value}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        logger.info("Generated HTML report at %s", output_path)
        return output_path

    def _build_ticker_chart(self, snapshot: TickerSnapshot, bars: list[object]) -> str:
        if not bars:
            fig = go.Figure()
            fig.update_layout(title=f"{snapshot.asset_name} (No Bar Data)")
            return fig.to_json()

        timestamps = [getattr(b, "timestamp_utc") for b in bars]
        closes = [float(getattr(b, "close")) for b in bars]
        highs = [float(getattr(b, "high")) for b in bars]
        lows = [float(getattr(b, "low")) for b in bars]
        opens = [float(getattr(b, "open")) for b in bars]

        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"{snapshot.asset_name} — Price, S/R Levels & News Summary"],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name="OHLCV 1h",
            )
        )

        # Add Support & Resistance lines
        for lvl in snapshot.support_resistance_levels:
            color = (
                "rgba(16, 185, 129, 0.7)"
                if "support" in lvl.level_type.value
                else "rgba(239, 68, 68, 0.7)"
                if "resistance" in lvl.level_type.value
                else "rgba(245, 158, 11, 0.7)"
            )
            dash = "solid" if "pivot" not in lvl.level_type.value else "dash"
            fig.add_hline(
                y=lvl.price,
                line_dash=dash,
                line_color=color,
                line_width=1.5,
                annotation_text=f"{lvl.level_type.value.upper()}: {lvl.price}",
                annotation_position="right",
                annotation_font_size=10,
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            height=400,
            margin={"l": 40, "r": 80, "t": 40, "b": 40},
            xaxis_rangeslider_visible=False,
        )
        return fig.to_json()

    def _build_eval_chart(self, eval_rows: list[dict[str, object]]) -> str:
        if not eval_rows:
            return ""

        times = [r["timestamp_utc"] for r in eval_rows]
        actual_returns = [float(r["actual_return"]) * 100 for r in eval_rows]
        predicted_returns = [float(r["predicted_return"]) * 100 for r in eval_rows]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=actual_returns,
                mode="lines+markers",
                name="Actual Return (%)",
                line={"color": "#38bdf8", "width": 1.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=predicted_returns,
                mode="lines+markers",
                name="Predicted Return (%)",
                line={"color": "#f43f5e", "dash": "dot", "width": 1.5},
            )
        )

        fig.update_layout(
            title="Out-of-Sample Walk-Forward: Predicted vs Actual Returns (%)",
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            height=320,
            margin={"l": 40, "r": 40, "t": 40, "b": 30},
        )
        return fig.to_json()
