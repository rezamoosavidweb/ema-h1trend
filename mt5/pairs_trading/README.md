# Pairs Trading — Live MT5 Runner

استراتژی **statistical arbitrage** روی جفت‌های FX کوینتگره. به‌طور کامل از سایر
استراتژی‌های موجود (`mt5/run_multi_scalper.py`، `mt5/run_ob_xauusd.py`، …)
تفکیک شده — مستقل magic number، مستقل log files، مستقل state store.

> ⚠️ **خواندن این فایل قبل از اولین اجرا اجباری است.**

---

## ۱) معماری

```
┌──────────────────────────────────────────────────────────────────────────┐
│  mt5/run_pairs_trading.py   ← CLI entry point                            │
│  ─ argparse                                                              │
│  ─ mt5.initialize + watchdog                                             │
│  ─ load portfolio CSV                                                    │
│  ─ instantiate PairsRunner.run_forever()                                 │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PairsRunner  (runner.py)                                                │
│  ─ wait for next H4 close                                                │
│  ─ for each spread in portfolio:                                         │
│        1. fetch bars (data_fetcher)                                      │
│        2. refit β/α on training window (signals)                         │
│        3. compute z-score                                                │
│        4. decide action vs current state                                 │
│        5. if actionable → PairsExecutionEngine                           │
│  ─ heartbeat + summary log                                               │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PairsExecutionEngine  (pairs_engine.py)                                 │
│  ─ open_pair(side):                                                      │
│        submit y-leg MARKET  → if rejected, abort                         │
│        submit x-leg MARKET  → if rejected, IMMEDIATELY close y           │
│        persist (pair_key → {y_ticket, x_ticket, side, opened_at})        │
│  ─ close_pair():                                                         │
│        close y + x; persist removal                                      │
│  ─ all events logged + telegram-notified                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## ۲) فایل‌ها

| فایل | مسئولیت |
|---|---|
| `config.py` | `PairsConfig`، `PortfolioSpread`، loader از CSV |
| `data_fetcher.py` | `fetch_close_bars()` — MT5 + Nicosia tz fix |
| `signals.py` | `refit_beta()`، `compute_z()`، `decide_action()` |
| `sizing.py` | `calculate_lot_sizes()` با broker `volume_min/step` |
| `state.py` | `PairsStateStore` — JSON-persistent state per pair |
| `pairs_engine.py` | `PairsExecutionEngine` — two-leg orders + partial-fill protection |
| `runner.py` | `PairsRunner` — cycle loop |

---

## ۳) magic number و comment_prefix

- **MAGIC_BASE = 28_000_000** — هر pair یه offset یکتا می‌گیره (`MAGIC_BASE + idx`).
- **comment_prefix = `"pairs_v1"`** — تو هر اوردر به این شکل: `pairs_v1:{pair_key}:{leg}`

این یعنی توی MT5 history همیشه قابل تشخیص هست که trade مال این strategy‌ـه.

---

## ۴) لاگ‌ها

- **events JSON:** `logs/pairs_trading/pairs.json` (rotate روزانه: `pairs-YYYY-MM-DD.json`)
- **state snapshot:** `logs/pairs_trading/state.json` (overwritten بعد هر تغییر)
- **seen signals (dedupe):** `logs/pairs_trading/seen_signals.json`

### Event types (همه با snake_case)

| Event | Trigger |
|---|---|
| `bot_start`، `bot_stop` | startup / shutdown |
| `mt5_connected`، `mt5_disconnected`، `mt5_reconnect_*` | watchdog |
| `heartbeat` | هر ۱۰ دقیقه |
| `cycle_start`، `cycle_end` | هر H4 close |
| `spread_evaluated` | برای هر pair در هر cycle (z_now, action) |
| `pair_open_attempt`، `pair_open_success`، `pair_open_failed` | باز کردن position |
| `pair_close_attempt`، `pair_close_success`، `pair_close_failed` | بستن position |
| `partial_fill_emergency` | اگه فقط یه leg fill شد و دومی rejected → leg اول فوراً بسته شد |
| `state_recovered` | بعد از restart، positions از broker بازیابی شد |
| `cycle_skipped` | dry-run یا mt5 unhealthy یا حالت غیر-trading |

---

## ۵) اجرا

### پیش‌نیازها
1. MT5 terminal open، logged in، AutoTrading فعال
2. Symbol‌های portfolio در Market Watch فعال
3. env vars (اختیاری ولی توصیه‌شده):
   ```
   MT5_LOGIN=...
   MT5_PASSWORD=...
   MT5_SERVER=...
   MT5_TERMINAL_PATH=...
   ```
4. Telegram (اختیاری):
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```

### دستورات

```bash
# Dry-run یک بار (هیچ اوردر فرستاده نمی‌شه؛ فقط report)
python mt5/run_pairs_trading.py --once --dry-run

# Dry-run loop (هر H4 close evaluate می‌کنه ولی بدون اوردر)
python mt5/run_pairs_trading.py --dry-run

# Live one-shot (بعد از تست demo)
python mt5/run_pairs_trading.py --once

# Live loop (production)
python mt5/run_pairs_trading.py

# انتخاب pairs خاص (override portfolio_selected_H4.csv)
python mt5/run_pairs_trading.py --pairs EURCHF~GBPJPY NZDCAD~EURUSD --dry-run
```

---

## ۶) Two-leg execution semantics

برای هر signal:

```
        ┌──────────────────┐
        │  OPEN_LONG       │  (z < -2σ)
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  submit y BUY    │
        └─────────┬────────┘
            ┌─────┴─────┐
        FILLED       REJECTED → abort (no exposure)
            │
            ▼
        ┌──────────────────┐
        │  submit x SELL   │
        └─────────┬────────┘
            ┌─────┴─────┐
        FILLED       REJECTED → 🚨 close y MARKET + ERROR_LOG
            │
            ▼
        ┌──────────────────┐
        │  persist state   │
        └──────────────────┘
```

این منطق **حیاتی**ست — هرگز یه leg بدون hedge باز نمی‌مونه.

---

## ۷) Position sizing

```
risk_usd          = equity × RISK_PER_LEG_PCT / 100
lots_y_raw        = risk_usd / (contract_size_y × price_y × ASSUMED_STOP_PCT)
lots_y            = clamp(round_step(lots_y_raw, volume_step_y), volume_min_y, volume_max_y)

notional_y        = lots_y × contract_size_y × price_y
target_notional_x = |β| × notional_y
lots_x_raw        = target_notional_x / (contract_size_x × price_x)
lots_x            = clamp(round_step(lots_x_raw, volume_step_x), volume_min_x, volume_max_x)
```

اگه `lots_y_raw < volume_min`، با warning `UNDER_RISK` فقط `volume_min` فرستاده می‌شه.
اگه equity خیلی کمه، operator باید `--dry-run` بزنه و قبل از live capital بفهمه.

---

## ۸) State recovery

روی startup:
1. `state.json` خوانده می‌شه — به‌عنوان "آنچه قبل از restart باز بود"
2. `mt5.positions_get()` با `magic == MAGIC_BASE+idx` فیلتر می‌شه
3. positions باز که در state هستن → adopted
4. positions که در broker هستن ولی توی state نیستن → adopted با warning (manual intervention)
5. state‌هایی که در broker نیستن → drop با warning (شاید TP/SL hit در زمان down)

---

## ۹) Cycle timing

- **TF = H4** → هر ۴ ساعت، یه cycle
- بعد از H4 close (به وقت Nicosia)، یه `EXTRA_GRACE_SECONDS` (پیش‌فرض ۳۰ ثانیه) صبر می‌کنه تا quote stabilize بشه
- بین cycles `sleep` می‌کنه (watchdog در فواصل بزرگ‌تر هم چک می‌شه)

---

## ۱۰) قبل از live capital

1. حداقل **۲ هفته** `--dry-run` روی demo
2. مقایسه‌ی `pairs.json` logs با backtest expected behavior
3. حداقل **۱ هفته** live روی demo با `RISK_PER_LEG_PCT = 0.1` (کمترین)
4. بعد graduation به live capital کم
5. هر افزایش risk باید بعد از ۲+ هفته stability مرحله‌ی قبل باشه

⚠️ این strategy از NB27 walk-forward `Sharpe=3.8, MaxDD=50bps` رو نشون داد — این **در گذشته** بود. live performance ممکنه متفاوت باشه.
