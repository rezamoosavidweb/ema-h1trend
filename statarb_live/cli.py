"""
Command-line entry point for statarb_live.

    python -m statarb_live run                 # live cycle loop (paper or live mode)
    python -m statarb_live once                # one cycle then exit
    python -m statarb_live replay --max 200    # deterministic historical replay (sim broker)
    python -m statarb_live reselect            # re-run frozen pair selection (explicit only)
    python -m statarb_live report --period daily
    python -m statarb_live info                # show resolved config + universe

Operational knobs come from env / .env (prefix SAL_). e.g.:
    SAL_MODE=paper SAL_BROKER=sim python -m statarb_live replay --max 500
"""

from __future__ import annotations

import argparse

from .config import STRATEGY, load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--broker", choices=["mt5", "sim"], default=None,
                   help="override SAL_BROKER")
    p.add_argument("--reselect", action="store_true",
                   help="re-run frozen pair selection on startup")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="statarb_live", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="live cycle loop")
    _add_common(p_run)
    p_once = sub.add_parser("once", help="single cycle then exit")
    _add_common(p_once)

    p_rep = sub.add_parser("replay", help="historical replay (sim broker)")
    _add_common(p_rep)
    p_rep.add_argument("--start", default=None)
    p_rep.add_argument("--end", default=None)
    p_rep.add_argument("--step", type=int, default=1, help="bars between cycles")
    p_rep.add_argument("--max", dest="max_cycles", type=int, default=None)

    p_sel = sub.add_parser("reselect", help="re-run frozen pair selection")
    _add_common(p_sel)

    p_report = sub.add_parser("report", help="generate a report")
    p_report.add_argument("--period", choices=["daily", "weekly", "monthly", "final"],
                          default="daily")
    p_report.add_argument("--date", default=None, help="anchor date YYYY-MM-DD")

    sub.add_parser("reconcile", help="compare paper (logged) fills vs real broker orders")

    sub.add_parser("health", help="liveness + strategy-correctness snapshot -> logs/.../health.txt")

    sub.add_parser("info", help="show resolved config + frozen universe")

    args = parser.parse_args(argv)
    cfg = load_config()
    if getattr(args, "broker", None):
        cfg.broker = args.broker

    if args.cmd == "info":
        return _cmd_info(cfg)
    if args.cmd == "report":
        from .reporting import generate_report
        path = generate_report(cfg, period=args.period, anchor=args.date)
        print(f"report written: {path}")
        return 0
    if args.cmd == "health":
        from .health import run_health
        text, path = run_health(cfg)
        print(text)
        if path:
            print(f"\nwritten to: {path}  (committed via git — pull it to read remotely)")
        return 0
    if args.cmd == "reconcile":
        from .reconcile import run_reconcile
        res = run_reconcile(cfg)
        print("=== paper-vs-real reconciliation ===")
        for k, v in res["summary"].items():
            print(f"  {k:26s} {v}")
        if res["csv"]:
            print(f"\nper-fill CSV: {res['csv']}")
        else:
            print("\n(no filled live orders yet — run with SAL_LIVE_ORDERS=true first)")
        return 0

    from .orchestrator import Orchestrator
    orch = Orchestrator(cfg, reselect=getattr(args, "reselect", False))

    if args.cmd == "once":
        rep = orch.run_once()
        print(rep)
        return 0
    if args.cmd == "reselect":
        orch = Orchestrator(cfg, reselect=True)
        print("universe reselected:", orch.uni.pair_keys)
        orch.storage.close()
        return 0
    if args.cmd == "replay":
        reps = orch.replay(start=args.start, end=args.end, step=args.step,
                           max_cycles=args.max_cycles)
        if reps:
            last = reps[-1]
            print(f"\nreplay complete: {len(reps)} cycles | final equity {last.equity:,.2f} "
                  f"| open {last.held}")
        return 0
    if args.cmd == "run":
        orch.run_forever()
        return 0
    return 1


def _cmd_info(cfg) -> int:
    from .signal_engine import load_or_select_universe
    print("=== statarb_live config ===")
    print(f"mode={cfg.mode} broker={cfg.broker} timeframe={cfg.timeframe}")
    print(f"db={cfg.resolved_db_url()}")
    print(f"data_dir={cfg.data_path()}")
    print(f"strategy={STRATEGY.version}")
    print(f"  z_entry={STRATEGY.z_entry} z_exit={STRATEGY.z_exit} z_stop={STRATEGY.z_stop} "
          f"z_window={STRATEGY.z_window}")
    print(f"  target_ann_vol={STRATEGY.target_ann_vol} vol_window={STRATEGY.vol_window} "
          f"max_lev={STRATEGY.max_leverage}")
    print(f"  carry_w_rev={STRATEGY.carry_w_rev} regime={STRATEGY.regime_method}/"
          f"{STRATEGY.regime_max_mult}")
    try:
        uni = load_or_select_universe(str(cfg.data_path()), cfg.storage_path(),
                                      timeframe=cfg.timeframe)
        print(f"universe pairs: {uni.pair_keys}")
        print(f"carry universe: {len(uni.carry_symbols)} symbols")
    except Exception as exc:
        print(f"universe: <not available: {exc}>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
