"""Broker factory — selects MT5 or sim from config, with safe fallback."""

from __future__ import annotations

from ..config import SystemConfig
from .base import BrokerAdapter
from .sim_adapter import SimBrokerAdapter


def create_broker(config: SystemConfig) -> BrokerAdapter:
    """Build the broker adapter named by ``config.broker``.

    'mt5' -> real demo account (Windows/VPS). If the MetaTrader5 module is not
    importable (e.g. Linux dev box), we fall back to the file-backed sim so the
    pipeline still runs — the caller can detect this via ``adapter.name``.
    """
    if config.broker == "mt5":
        try:
            import MetaTrader5  # noqa: F401
            from .mt5_adapter import MT5BrokerAdapter

            return MT5BrokerAdapter(
                login=config.mt5_login, password=config.mt5_password,
                server=config.mt5_server, terminal_path=config.mt5_terminal_path,
                broker_tz=config.broker_tz,
            )
        except Exception:
            # MT5 unavailable — degrade to sim (logged by the runner).
            pass

    return SimBrokerAdapter(
        data_dir=str(config.data_path()), timeframe=config.timeframe,
        starting_equity=config.starting_equity,
    )
