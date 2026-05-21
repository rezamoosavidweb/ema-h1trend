# Execution Layer

Institutional-grade execution pipeline for the MT5 Order Block bot.
Strategy/signal generation lives in `mt5/run_ob_xauusd.py` and is **unchanged**;
everything in this package handles validation, order placement, pending-order
lifecycle, fallback, retries, and observability.

## Module map

| Module              | Responsibility                                                |
|---------------------|---------------------------------------------------------------|
| `symbol_config`     | Resolve `XAUUSD` -> `XAUUSD.i` once; snapshot broker metadata |
| `structured_logger` | JSON-Lines event log + daily file rotation                    |
| `broker_validator`  | Pre-flight: terminal, AutoTrading, spread, stops, freeze      |
| `order_factory`     | Build `order_send` dicts; snap prices; pick filling mode      |
| `pending_manager`   | Track / cancel / dedupe pending orders client-side (GTC)      |
| `risk_adapter`      | Lot sizing from balance × `risk_per_trade`                    |
| `fallback_engine`   | LIMIT -> MARKET cascade by retcode + re-validation            |
| `mt5_watchdog`      | Health check + automatic reconnect; structured state logs     |
| `execution_engine`  | Public facade `engine.place_signal(signal)` + heartbeat       |

Strategy script wires them together via `ExecutionEngine`, the only object it
needs to construct.

---

## Migration notes (from previous monolithic `run_ob_xauusd.py`)

### Bugs fixed

| Symptom in old logs                       | Cause                                                | Fix                                              |
|-------------------------------------------|------------------------------------------------------|--------------------------------------------------|
| `retcode 10022 "Invalid expiration"`      | UTC Unix timestamp passed as broker-local expiration | `order_factory` uses `ORDER_TIME_GTC`; lifecycle in `pending_manager` |
| `retcode 10015 "Invalid price"`           | Limit price inside `stops_level`                     | `broker_validator.validate_pending_geometry`     |
| LIMIT fail -> trade lost                  | Old code logged and returned                         | `fallback_engine` advances to MARKET             |
| Telegram failures invisible               | `except Exception: pass`                             | `mt5_notifier` logs `telegram_sent`/`telegram_error` |
| `XAUUSD` vs `XAUUSD.i` drift              | Symbol resolved per-call                             | `symbol_config.resolve_symbol` cached            |

### Behavioural changes (intentional)

* **No `expiration` field on LIMIT orders.** They live as GTC and are
  cancelled client-side after `stale_after_seconds`. This kills the entire
  "broker reads UTC Unix timestamp as local time" class of bugs.
* **Spread filter.** Trades are skipped when `(ask - bid) / point > 200` by
  default. Tune via `ExecutionEngine(max_spread_points=...)`.
* **Cooldown.** After 5 broker failures within 5 minutes the engine enters a
  15-minute cooldown and refuses to send new orders. Lets you investigate
  before churning more rejections.
* **Per-OB retry cap.** Each `(ob_time, direction)` may be retried at most
  3 times.
* **Orphan cancel.** If a pending order's signal disappears from the new
  bar, it is cancelled (event: `pending_order_cancelled` with
  `reason=signal_invalidated`).

### Things that did NOT change

* Strategy detection (`add_candle_features`, `detect_displacements`,
  `find_order_blocks`, `find_signal`) -- byte-identical to the previous
  version. Backtest parity is preserved.
* RR=2 target, SL buffer 0.5, displacement ATR multiplier 1.5, OB expiry
  100 bars -- all unchanged.
* `logs/seen_obs_XAUUSD.json` cross-restart deduplication.

---

## Event catalogue

All events are JSON Lines written to `logs/<symbol>.json`. Envelope is
always `{ ts, event, symbol, ...fields }`. New event names introduced by
the refactor are marked **NEW**.

### Lifecycle

| Event        | Fields                                                |
|--------------|-------------------------------------------------------|
| `bot_start`  | `requested_symbol`, `magic`, `risk_per_trade`, ...    |
| `bot_stop`   | `tracked_pendings: [int]`                             |
| `cycle`      | `bars`, `displacements`, `order_blocks`, `signal`     |
| `signal`     | full signal dict (entry/sl/tp/ob_time/...)            |

### Pending order management *(NEW group)*

| Event                       | Fields                                              |
|-----------------------------|-----------------------------------------------------|
| `pending_order_created`     | `ticket`, `entry`, `sl`, `tp`, `stale_after_s`      |
| `pending_order_cancelled`   | `ticket`, `reason`, `ok`, `retcode`                 |
| `stale_pending_removed`     | `ticket`, `age_s`                                   |
| `pending_order_disappeared` | `ticket`, `ob_key`  (broker-side fill or cancel)    |
| `pending_order_adopted`     | `ticket`, `entry`, `sl`, `tp`  (after restart)      |

### Execution

| Event                         | Fields                                              |
|-------------------------------|-----------------------------------------------------|
| `fallback_market_execution`   | `reason`, `slippage_pts` *(NEW)*                    |
| `slippage_adjusted`           | `direction`, `ob_entry`, `market_price`, `tp_*`     |
| `market_order_placed`         | `ticket`, `volume`, `price`, `sl`, `tp`             |
| `market_order_failed`         | `retcode`, `comment`, `sl`, `tp`                    |
| `market_order_aborted`        | `retcode`, `comment` *(hard retcode -- 10018/27)*   |

### Validation / safety *(NEW)*

| Event                       | Fields                                                  |
|-----------------------------|---------------------------------------------------------|
| `broker_validation_failed`  | `stage`, `code` (e.g. `spread_too_high`), `detail`      |
| `cooldown_engaged`          | `failures`, `window_s`, `duration_s`                    |
| `execution_latency`         | `stage`, `ticket`, `latency_ms`                         |

### Telegram observability *(NEW)*

| Event             | Fields                                                |
|-------------------|-------------------------------------------------------|
| `telegram_sent`   | `category`, `attempt`, `status`, `latency_ms`         |
| `telegram_error`  | `category`, `attempt`, `status`, `body`, `error_type` |
| `telegram_disabled` | `reason`, `has_token`, `has_chat_id`                |

### Reliability / uptime *(NEW group)*

These events are pure observability -- they do not influence trade selection,
risk sizing, or candle-confirmation timing. They exist so you can answer "was
the bot running, and was MT5 healthy, at time X?" without guessing.

| Event                       | Fields                                                                                  |
|-----------------------------|-----------------------------------------------------------------------------------------|
| `heartbeat`                 | `uptime_s`, `mt5_connected`, `mt5_trade_allowed`, `tracked_pendings`, `open_positions`, `cooldown_active`, `cooldown_remaining_s`, `balance`, `equity` |
| `state_recovered`           | `adopted_pendings: [int]`, `adopted_positions: [int]`, `balance`, `equity` -- one-shot at startup |
| `mt5_connected`             | `state="healthy"`, `connected`, `trade_allowed`, `ping_ms`, optional `periodic=true`    |
| `mt5_disconnected`          | `state` (`terminal_unreachable` / `broker_disconnected` / `autotrading_disabled`), `consecutive_failures` |
| `mt5_reconnect_attempt`     | `attempt`, `total_reconnects_so_far`                                                    |
| `mt5_reconnect_success`     | `attempt`, `total_reconnects`, `connected`, `trade_allowed`                             |
| `mt5_reconnect_failed`      | `attempt`, `state`, `error_type`, `error_msg`                                           |
| `mt5_reconnect_giveup`      | `attempts`, `advice` -- operator intervention required                                  |
| `cycle_skipped`             | `reason="mt5_unhealthy"` -- the strategy cycle was skipped this M5 boundary             |
| `startup_pending_recovery_error`  | `error_type`, `error_msg`                                                         |
| `startup_position_recovery_error` | `error_type`, `error_msg`                                                         |

#### Heartbeat example

```json
{"ts":"2026-05-21T08:30:00.456+00:00","event":"heartbeat","symbol":"XAUUSD.i","uptime_s":3604.2,"mt5_connected":true,"mt5_trade_allowed":true,"tracked_pendings":0,"open_positions":1,"cooldown_active":false,"cooldown_remaining_s":0.0,"balance":10050.32,"equity":10047.18}
```

#### State recovery example (one event at bot startup)

```json
{"ts":"2026-05-21T07:00:01.123+00:00","event":"state_recovered","symbol":"XAUUSD.i","adopted_pendings":[38875210],"adopted_positions":[],"balance":10050.32,"equity":10050.32}
```

#### Reconnect cycle example

```json
{"ts":"2026-05-21T05:13:02.001+00:00","event":"mt5_disconnected","symbol":"XAUUSD.i","state":"broker_disconnected","connected":false,"trade_allowed":true,"consecutive_failures":0}
{"ts":"2026-05-21T05:13:02.013+00:00","event":"mt5_reconnect_attempt","symbol":"XAUUSD.i","attempt":1,"total_reconnects_so_far":0}
{"ts":"2026-05-21T05:13:04.502+00:00","event":"mt5_reconnect_success","symbol":"XAUUSD.i","attempt":1,"total_reconnects":1,"connected":true,"trade_allowed":true}
{"ts":"2026-05-21T05:13:04.605+00:00","event":"mt5_connected","symbol":"XAUUSD.i","state":"healthy","connected":true,"trade_allowed":true,"recovered_after_failures":0}
```

---

## Reliability features (operational, NOT strategy)

The following items were added without altering any trading behaviour. Every
strategy primitive (signal generation, OB detection, FVG logic, rejection
candle test, SL/TP geometry, RR ratio, `risk_per_trade`, position sizing,
candle-confirmation timing) is byte-identical to the pre-reliability commit.

### Daily log rotation

`StructuredLogger(rotate_daily=True)` (default) writes to
`logs/XAUUSD-YYYY-MM-DD.json`. The base `logs/XAUUSD.json` path is maintained
as a best-effort symlink to today's file (silently skipped on Windows accounts
without symlink permission). This makes accidental truncation during a deploy
non-destructive -- only today's file is at risk.

If you depend on a single ever-growing file, pass
`ExecutionEngine(rotate_daily_logs=False)`.

### Heartbeat (cycle-driven)

`engine.heartbeat_if_due()` runs as the last step of `begin_cycle()`. It emits
a `heartbeat` event every `heartbeat_interval_seconds` (default 600 s = 10 min).
It is NOT a background thread, so it cannot race with strategy code or shift
order timing. Skip it entirely with `heartbeat_interval_seconds=0` if you
prefer external uptime monitoring.

### Startup recovery

`engine.initialize_state_from_broker()` runs ONCE at startup, before the main
loop. It:

* adopts existing pending orders for this `MAGIC` into `PendingOrderManager`
* adopts existing open positions for this `MAGIC` so close-detection works
* logs a `state_recovered` event
* DOES NOT call `find_signal`, fetch bars, or place any order

Combined with the cross-restart `seen_obs_XAUUSD.json` file (unchanged), a
restart in the middle of a signal bar will:

1. Recover broker state cleanly.
2. Run the FIRST cycle immediately (no leading sleep).
3. Evaluate the most recent CLOSED candle once (`iloc[:-1]` always drops the
   forming bar -- candle confirmation behaviour is preserved).
4. Continue normal M5-aligned cycle timing.

### MT5 watchdog

`Mt5Watchdog` is attached to the engine via `engine.attach_watchdog(...)`. At
the start of every cycle, the bot calls `engine.is_mt5_healthy()` which
returns False when the terminal is offline, the broker connection is down,
or AutoTrading is off. A failing cycle:

* logs `cycle_skipped reason=mt5_unhealthy`
* attempts `mt5_reconnect_attempt` -> success/failure
* does NOT fetch bars, generate signals, or send orders
* does NOT advance any per-OB retry counter
* still respects the M5 cadence so the NEXT cycle gets a fresh chance

Because `PendingOrderManager.sync_from_broker()` runs every cycle, a reconnect
cannot leave the bot's in-memory state out of sync with the broker -- no
duplicate orders, no replayed signals.

### Examples

```json
{"ts":"2026-05-21T08:30:00.123+00:00","event":"signal","symbol":"XAUUSD.i","direction":"BUY","entry":4497.09,"sl":4493.48,"tp":4504.31,"ob_time":"2026-05-20 15:15:00+00:00"}
{"ts":"2026-05-21T08:30:00.456+00:00","event":"pending_order_created","symbol":"XAUUSD.i","ticket":38875210,"side":"buy","entry":4497.09,"sl":4493.48,"tp":4504.31,"volume":0.01,"stale_after_s":300}
{"ts":"2026-05-21T08:30:00.789+00:00","event":"telegram_sent","symbol":"XAUUSD.i","category":"order_placed","attempt":1,"status":200,"latency_ms":312.4}
{"ts":"2026-05-21T08:35:01.012+00:00","event":"stale_pending_removed","symbol":"XAUUSD.i","ticket":38875210,"age_s":300.4}
```

---

## Configuration cheat sheet

Defaults in `mt5/run_ob_xauusd.py`:

```python
RISK_PER_TRADE             = 0.01     # 1% of balance per trade
RISK_REWARD                = 2.0      # TP = entry +/- 2 * |entry - sl|
SLIPPAGE_LIMIT_THRESHOLD   = 4.0      # pts -- below: try LIMIT
SLIPPAGE_MAX_POINTS        = 6.0      # pts -- above: reject
LIMIT_ORDER_STALE_SECONDS  = 300      # cancel LIMIT after one M5 bar
MAX_SPREAD_POINTS          = 200      # skip when spread > this
MAGIC                      = 8088080  # this bot's orders only
```

Engine safety knobs (override via `ExecutionEngine(...)`):

```python
max_failures_before_cooldown = 5
cooldown_window_seconds      = 300
cooldown_duration_seconds    = 900
max_retries_per_ob           = 3
```

---

## Verifying the fix in production

After deploying:

```bash
# 1. No more 10022 errors -- this should return nothing
grep '"retcode": 10022' logs/XAUUSD.json

# 2. LIMIT orders are being created with GTC
grep '"event": "pending_order_created"' logs/XAUUSD.json | tail -5

# 3. Stale pendings are cleaned up after 5 minutes
grep '"event": "stale_pending_removed"' logs/XAUUSD.json | tail -5

# 4. Telegram is actually delivering
grep '"event": "telegram_sent"' logs/XAUUSD.json | tail -5

# 5. If anything is dropping silently, you will now see it
grep '"event": "telegram_error"' logs/XAUUSD.json
```

A healthy log should show, per traded signal, in order:

1. `cycle` (signal=true)
2. `signal`
3. `pending_order_created` *or* `fallback_market_execution` -> `market_order_placed`
4. `telegram_sent` for `signal` and `order_placed`

---

## TODO / future scaling

* **Per-symbol filling-mode override.** Some brokers expose
  `SYMBOL_FILLING_BOA` (book-or-cancel); add to `OrderFactory.pick_filling`.
* **Broker-time offset detection.** Capture `tick.time - now_utc` at startup
  and expose it; useful for any future feature that *needs* to schedule by
  broker time. Right now we sidestep this with GTC.
* **Trail SL.** Once price moves > N points in our favour, raise SL by M.
  Plug into `pending_manager.modify_pending` / a sibling `position_manager`.
* **Multi-symbol orchestration.** Make `ExecutionEngine` symbol-agnostic and
  pool multiple instances under a supervisor; needs proper locking around
  `_failures` / `_cooldown_until`.
* **Metrics exporter.** Tail `logs/XAUUSD.json` into Prometheus/Grafana so
  `cooldown_engaged`, `telegram_error`, and `execution_latency` can page
  someone instead of just sitting in a file.
* **Unit tests.** `OrderFactory.snap`, `RiskAdapter.normalize`,
  `BrokerValidator.validate_pending_geometry`, and the retcode -> stage
  mapping in `FallbackEngine` are now pure and trivial to test.
