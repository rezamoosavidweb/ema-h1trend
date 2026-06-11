"""RiskManager — evaluates hard limits each cycle and gates new entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..config import SystemConfig
from ..storage.base import Storage


@dataclass
class RiskDecision:
    allow_new_entries: bool
    breaches: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.breaches


class RiskManager:
    """Stateless w.r.t. its own data — reads realised/cumulative PnL from storage so a
    restart cannot reset a daily-loss lockout."""

    def __init__(self, config: SystemConfig, storage: Storage) -> None:
        self.cfg = config
        self.storage = storage

    def _pnl_since(self, start: datetime, now: datetime) -> float:
        trades = self.storage.trades_between(start, now + timedelta(seconds=1))
        return float(sum(t.get("realized_pnl") or 0.0 for t in trades))

    def evaluate(self, *, equity: float, gross_exposure: float,
                 now: datetime | None = None) -> RiskDecision:
        now = now or datetime.now(timezone.utc)
        eq = max(equity, 1e-9)

        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = day_start - timedelta(days=day_start.weekday())

        daily_pnl = self._pnl_since(day_start, now)
        weekly_pnl = self._pnl_since(week_start, now)

        breaches: list[str] = []
        if daily_pnl <= -self.cfg.max_daily_loss_pct * eq:
            breaches.append(
                f"max_daily_loss: {daily_pnl:.2f} <= -{self.cfg.max_daily_loss_pct:.1%} of equity"
            )
        if weekly_pnl <= -self.cfg.max_weekly_loss_pct * eq:
            breaches.append(
                f"max_weekly_loss: {weekly_pnl:.2f} <= -{self.cfg.max_weekly_loss_pct:.1%} of equity"
            )
        if gross_exposure > self.cfg.max_gross_exposure:
            breaches.append(
                f"max_gross_exposure: {gross_exposure:.2f}x > {self.cfg.max_gross_exposure:.2f}x"
            )

        return RiskDecision(
            allow_new_entries=not breaches,
            breaches=breaches,
            metrics={"daily_pnl": daily_pnl, "weekly_pnl": weekly_pnl,
                     "gross_exposure": gross_exposure, "equity": equity},
        )
