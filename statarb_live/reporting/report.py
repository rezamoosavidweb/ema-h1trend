"""Report generation — HTML (always) + CSV exports + optional PDF."""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import SystemConfig
from ..storage import create_storage
from ..monitoring.metrics import (
    attribution_by_pair, attribution_by_regime, attribution_by_sleeve, compute_metrics,
)
from ..monitoring.charts import write_charts

# Reference research numbers (provenance: NB37 walk-forward, NB38 phase-4). These are the
# benchmark the live paper run is validated against — NOT live results.
RESEARCH_BENCHMARKS = {
    "reversion_oos_sharpe": 0.29,          # NB38 cell 1
    "reversion_carry_sharpe": 0.49,        # NB38 cell 3 (50/50)
    "walk_forward_sharpe": 0.43,           # NB37 §I (single walk-forward)
    "cpcv_mean_sharpe": 0.10,              # NB38 cell 14
    "cpcv_p05_sharpe": -0.60,              # NB38 cell 14
    "deflated_sharpe": 0.65,               # NB38 cell 13
}


def _window_bounds(period: str, anchor: datetime) -> tuple[datetime | None, datetime | None]:
    if period == "daily":
        s = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        return s, s + timedelta(days=1)
    if period == "weekly":
        s = (anchor - timedelta(days=anchor.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return s, s + timedelta(days=7)
    if period == "monthly":
        s = anchor.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt = (s.replace(day=28) + timedelta(days=4)).replace(day=1)
        return s, nxt
    return None, None  # final / all


def _df_to_html(df: pd.DataFrame, *, floatfmt: str = "{:.2f}") -> str:
    if df is None or df.empty:
        return "<p><em>no data</em></p>"
    d = df.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    return d.to_html(border=0, classes="tbl", justify="center")


def generate_report(config: SystemConfig, *, period: str = "daily",
                    anchor: str | None = None) -> Path:
    storage = create_storage(config, init=False)
    try:
        anchor_dt = (pd.Timestamp(anchor, tz="UTC").to_pydatetime() if anchor
                     else datetime.now(timezone.utc))
        start, end = _window_bounds(period, anchor_dt)

        metrics = compute_metrics(storage, window=period, start=start, end=end,
                                  starting_equity=config.starting_equity)
        ap = attribution_by_pair(storage)
        ar = attribution_by_regime(storage)
        asl = attribution_by_sleeve(storage)

        stamp = anchor_dt.strftime("%Y%m%d_%H%M%S")
        out_dir = config.report_path() / f"{period}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        charts = write_charts(storage, out_dir)

        # CSV exports
        pd.DataFrame([metrics.as_row()]).to_csv(out_dir / "metrics.csv", index=False)
        if not ap.empty:
            ap.to_csv(out_dir / "attribution_by_pair.csv")
        if not ar.empty:
            ar.to_csv(out_dir / "attribution_by_regime.csv")
        if not asl.empty:
            asl.to_csv(out_dir / "attribution_by_sleeve.csv")

        html_doc = _render_html(period, anchor_dt, metrics, ap, ar, asl, charts,
                                include_benchmarks=(period in ("final", "monthly")))
        html_path = out_dir / "report.html"
        html_path.write_text(html_doc, encoding="utf-8")

        # optional PDF
        try:
            from weasyprint import HTML  # type: ignore
            HTML(string=html_doc, base_url=str(out_dir)).write_pdf(str(out_dir / "report.pdf"))
        except Exception:
            pass

        return html_path
    finally:
        storage.close()


def _render_html(period, anchor, m, ap, ar, asl, charts, *, include_benchmarks: bool) -> str:
    def img(name: str, title: str) -> str:
        if name in charts:
            return f'<h3>{title}</h3><img src="{charts[name].name}" style="width:100%;max-width:980px">'
        return ""

    ex = m.extra
    bench = ""
    if include_benchmarks:
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
            for k, v in RESEARCH_BENCHMARKS.items()
        )
        bench = f"""
        <h2>Backtest / walk-forward benchmark (validation targets)</h2>
        <p>Reference Sharpe figures from the frozen research (NB37/38). The live paper run is
        compared against these to quantify performance decay; live Sharpe below is over the
        paper window only.</p>
        <table class="tbl"><tr><th>metric</th><th>value</th></tr>{rows}</table>
        <p><strong>Live (paper) Sharpe this run:</strong> {m.sharpe:.2f} &nbsp;|&nbsp;
        decay vs walk-forward (0.43): {m.sharpe - RESEARCH_BENCHMARKS['walk_forward_sharpe']:+.2f}</p>
        """

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>statarb_live {period} report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a}}
 h1{{margin-bottom:0}} .sub{{color:#666;margin-top:2px}}
 .grid{{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0}}
 .card{{border:1px solid #e2e2e2;border-radius:8px;padding:12px 16px;min-width:150px}}
 .card .v{{font-size:20px;font-weight:600}} .card .k{{color:#666;font-size:12px}}
 table.tbl{{border-collapse:collapse;margin:8px 0;font-size:13px}}
 table.tbl td,table.tbl th{{border:1px solid #e2e2e2;padding:4px 10px;text-align:center}}
 .neg{{color:#c0392b}} .pos{{color:#1e8449}}
</style></head><body>
<h1>Statistical Arbitrage — Live Paper Trading</h1>
<div class="sub">{period.upper()} report · anchor {anchor:%Y-%m-%d %H:%M UTC} · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</div>

<div class="grid">
 <div class="card"><div class="k">PnL</div><div class="v">{m.pnl:,.2f}</div></div>
 <div class="card"><div class="k">Return</div><div class="v">{m.return_pct:+.2%}</div></div>
 <div class="card"><div class="k">Sharpe</div><div class="v">{m.sharpe:.2f}</div></div>
 <div class="card"><div class="k">Max DD</div><div class="v">{m.max_drawdown:.2%}</div></div>
 <div class="card"><div class="k">Win rate</div><div class="v">{m.win_rate:.1%}</div></div>
 <div class="card"><div class="k">Trades</div><div class="v">{m.n_trades}</div></div>
 <div class="card"><div class="k">Avg bars held</div><div class="v">{m.avg_bars_held:.1f}</div></div>
</div>

<h2>PnL attribution (reversion vs carry vs cost)</h2>
<div class="grid">
 <div class="card"><div class="k">Reversion PnL</div><div class="v">{ex.get('reversion_pnl',0):,.2f}</div></div>
 <div class="card"><div class="k">Carry PnL</div><div class="v">{ex.get('carry_pnl',0):,.2f}</div></div>
 <div class="card"><div class="k">Cost PnL</div><div class="v">{ex.get('cost_pnl',0):,.2f}</div></div>
 <div class="card"><div class="k">Avg regime mult</div><div class="v">{ex.get('avg_regime_mult',float('nan')):.2f}</div></div>
</div>

{img('equity','Equity curve')}
{img('drawdown','Drawdown')}
{img('exposure','Exposure & regime')}

<h2>By pair</h2>{_df_to_html(ap)}
<h2>By regime</h2>{_df_to_html(ar)}
<h2>By sleeve</h2>{_df_to_html(asl)}

{bench}
<hr><p style="color:#888;font-size:12px">statarb_live · frozen strategy (NB38) · research validation only — not investment advice.</p>
</body></html>"""
