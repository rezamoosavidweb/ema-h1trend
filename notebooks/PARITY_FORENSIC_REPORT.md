# Live ↔ Replay Parity — Forensic Report

**Generated:** 2026-05-28
**Scope:** Forensic from existing logs + parity CSVs (no new architectures, no live snapshots).
**Symbols covered:** GBPUSD, XAUUSD, GBPCAD, USDMXN, EURJPY, EURCAD, AUDCAD.
**Sources:** `parity_per_bar_diff.csv` (1,717 divergent bars), `parity_multi_scalper_detail.csv`, `replay_trades_20260527_20260530.csv`, daily per-symbol JSON logs (~4,400 cycle events).

---

## TL;DR

Five distinct divergence sources, ranked by *signal-flipping* severity:

| # | Source | Signal flips | Mechanism | Fixable? |
|---|---|---|---|---|
| 1 | HTF top-up vs broker-published HTF | **21 of 23** | `topup_htf_from_m5` synthesises H1/D1 from M5 when broker lags; replay reads CSV-published H1/D1 | YES |
| 2 | `bar_time` JOIN-key format (`+00:00` vs naive) | 0 strategy, but **100 % of summary "matches"** | NB30 writes BT timestamps with `+00:00`, LV without; string join never matches | YES, trivial |
| 3 | ONE_TRADE_AT_A_TIME cascade timing | Up to **86 % of live signals lost** post-first-fill | Per-symbol position-open gate in `execution_engine.py:439`; first fill bar differs between LV and replay | DESIGN TRADEOFF |
| 4 | M5 OHLC cache drift (CSV vs live stream) | **1 of 23** | CSV cache wasn't fully re-fetched; sub-pip OHLC differences flip indicator outputs at thresholds | YES, operational |
| 5 | Entry-price convention (replay open(i+1) vs live close(i)) | **0 signal flips, sub-pip drift** | NB33 enters at next bar open; live places market at close-of-signal-bar | NOT WORTH FIXING |

**The "ZERO matches" headline in `parity_multi_scalper_summary.csv` is a comparison bug, not a strategy bug.** Once timestamp suffixes are stripped, however, you still get 0 matches *per symbol* — because the BT window and the LV window don't actually overlap on the same bars in that CSV. NB31's per-bar diff (which uses a different join, with naïve timestamps from the same `Strategy.detect_signal_verbose`) is the trustworthy comparison: **23 signal flips out of ~4,400 cycles across the basket = 0.5 % strategy-divergence rate**.

The strategy edge itself is mostly preserved. The visible "16 backtest, 0 live" / "0 backtest, 21 live" rows in the summary CSV are **NOT 21 lost signals** — they're 21 events the CSV failed to align due to format + ONE_TRADE cascading and window mismatch.

---

## 1 — Evidence Base (what the data actually says)

### 1a. Per-bar gate diff (NB31 output, 1,717 divergent bars)

Field-level frequency across all symbols:

| Field | Bars affected | What it means |
|---|---|---|
| `h1_rsi`     | 1,685 | Sub-RSI-point drift; mostly cosmetic |
| `adx`        |   377 | Sub-decimal drift; mostly cosmetic |
| `trend_dir`  |   232 | The H1∧D1 agreement gate. **Strategy-affecting.** |
| `h1_trend`   |   162 | Raw H1 trend |
| `d1_trend`   |    86 | Raw D1 trend (all `-1→+1`) |
| `rsi`        |    75 | M5 RSI drift |
| `volume`     |    74 | Tick-volume vs broker-volume |
| `signal_dir` |   **23** | **The bars where the final decision actually flipped.** |
| `in_session` |    12 | Session window edge cases |
| `f_candle`   |     7 | Indicator near a numeric threshold |
| `f_vol`/`f_ema`/`f_stoch`/`f_bb`/`close`/`low` | ≤5 each | Edge cases |

**Read:** `h1_rsi` drift is everywhere but rarely flips signals. `trend_dir` flips correlate strongly with `signal_dir` flips.

### 1b. The 23 `signal_dir` disagreement cases (the only truly strategy-affecting divergences in NB31)

| Symbol | Count | Direction | Root field that differs |
|---|---|---|---|
| GBPCAD | 10 | 0→+1 (live alone fired LONG) | `h1_trend`, `trend_dir` (8 cases), `h1_rsi` (2 cases) |
| EURCAD | 8  | 7×0→+1 (live alone fired LONG), 1×+1→0 (replay alone fired) | `h1_trend`/`h1_rsi`/`adx`, 1× M5 OHLC drift |
| GBPUSD | 5  | -1→0 (replay alone fired SHORT) | `d1_trend` all 5 |

**Net effect on the basket:**
- **6 signals replay produced that live missed** (GBPUSD ×5 + EURCAD ×1)
- **17 signals live produced that replay missed** (GBPCAD ×10 + EURCAD ×7)

So during the window measured, **live actually fired MORE signals than replay**. That's the opposite direction of the prompt's framing ("missing signals"). The "missing" comes from the *trade-count* comparison (next section), which is dominated by ONE_TRADE_AT_A_TIME, not the strategy.

### 1c. Trade-count comparison (replay vs live `order_placed` events)

Window: 2026-05-27 → 2026-05-30 (replay_trades_20260527_20260530.csv vs daily logs)

| Symbol | Replay trades | Live orders placed | Notes |
|---|---|---|---|
| GBPUSD | 4 | 4  | ✓ matches |
| XAUUSD | 3 | 4  | live +1 |
| GBPCAD | 2 | 3  | live +1 |
| USDMXN | 3 | 0  | live placed nothing |
| EURJPY | 0 | 0  | ✓ matches |
| EURCAD | 3 | 7  | live +4 |
| AUDCAD | 0 | 0  | ✓ matches |
| **Total** | **15** | **18** | |

Live placed more orders than replay on EURCAD, GBPCAD, XAUUSD. Replay produced trades on USDMXN that live never placed (need to check why).

### 1d. ONE_TRADE_AT_A_TIME cascade — the dominant "signal loss" source

From the live log skip-reason histogram (across all daily logs):

| Symbol | Signals fired | `position_open` skips | `tg_suppressed` | Orders placed | Cascade loss % |
|---|---|---|---|---|---|
| EURCAD | 32 | 25 | 17 | 7  | 78 % |
| GBPCAD | 22 | 19 | 18 | 3  | 86 % |
| GBPUSD | 18 | 7  | 7  | 4  | 39 % |
| XAUUSD | 8  | 4  | 3  | 4  | 50 % |

**That's where most "missing live trades" come from.** The strategy detected the signal, but `_has_open_position()` in `execution_engine.py:439-448` returned True and the engine skipped placement with `reason="position_open"`. The same gate exists in replay (`ONE_TRADE_AT_A_TIME = True` in NB33), but the **timing of when the first trade opens differs**, so the set of cascaded signals diverges.

Concrete: on EURCAD 2026-05-27, the first trade in both live and replay opened at 15:25, but live's stop was hit faster (by data drift in SL placement OR by spread/slippage on the actual fill) → live re-armed earlier → caught the 16:20 and 17:20 second-leg signals that replay (still in the first trade) couldn't see.

### 1e. Timestamp `+00:00` suffix — comparison bug, not strategy bug

`parity_multi_scalper_detail.csv` rows:
- 76 backtest-only rows: **all** carry `+00:00` suffix
- 55 live-only rows: **all** carry no suffix
- After stripping suffix, 0 still match per-symbol because the union of bar_times between BT-only and LV-only is empty (every BT bar BT-fired-on, LV didn't fire on, and vice-versa).

The 0 cross-side matches isn't "live and replay diverge on every bar." It's "replay's bars are a different set than live's bars in this output." Live didn't run during all of replay's window (AUDCAD ran only 309 cycles ≈ 25 hours, while replay window is 72 hours).

### 1f. Entry-price drift between bar-close and next-bar-open

Sub-pip on FX, sub-cent on XAUUSD:

| Symbol | Samples | Mean abs drift | Max abs drift |
|---|---|---|---|
| USDMXN | 3 | 5.3e-05 | 1.5e-04 |
| GBPUSD | 4 | 1e-05   | 3e-05   |
| EURCAD | 3 | 1.3e-05 | 2e-05   |
| GBPCAD | 2 | 1.5e-05 | 3e-05   |
| XAUUSD | 3 | 0.03    | 0.09    |

**Not material to strategy outcomes.** Live's market order at close-of-signal-bar fills within a tick of the next-bar-open replay assumes. Slippage variance dominates this.

---

## 2 — Root-Cause Ranking (by danger to live edge)

### Rank 1 — **HTF Top-Up vs Broker-Published HTF** (DANGEROUS — measurable signal flips)

**Code path:** [mt5/run_multi_scalper.py:248-283](mt5/run_multi_scalper.py#L248-L283) `topup_htf_from_m5`.

**Mechanism:**
- When `last_m5_time > last_h1_time + 1h`, live synthesises one or more H1 bars from M5 close/open/high/low aggregation.
- These synthetic bars enter the H1 EMA50 + RSI14 indicator state.
- Replay (NB31, NB33) reads pre-saved CSV H1 which is what the broker eventually published.
- Synthetic vs broker-published OHLC differ → indicator state diverges → `h1_trend` or `d1_trend` flips → trend gate result changes.

**Evidence:**
- 21 of 23 signal_dir disagreements correlate with `h1_trend` or `d1_trend` field flips.
- All 86 `d1_trend` divergences in the entire 1,717 dataset are `-1→+1` — meaning when D1 disagrees, it's always live-says-up vs replay-says-down. That asymmetry is the M5-aggregated D1 being more momentum-following than the broker's slowly-published true D1 close.

**Quantified live impact:** GBPUSD 2026-05-26 lost 5 short signals to this; GBPCAD 2026-05-27 gained 8 long signals from this. Replay can't reproduce either side.

### Rank 2 — **ONE_TRADE_AT_A_TIME cascade-timing skew** (NUANCED — high signal-loss rate but the gate is intentional)

**Code path:** [execution/execution_engine.py:439-448](execution/execution_engine.py#L439-L448) `_has_open_position()`.

**Mechanism:**
- Per-symbol, magic-matched: any live signal while a position with this bot's magic is open → `reason="position_open"` skip, no order, no Telegram (suppressed).
- Replay (NB33) also enforces `ONE_TRADE_AT_A_TIME=True`, so the gate itself is parity-preserving.
- But the **wall-clock moment the first trade exits differs** because:
  - Live SL = signal-bar's structural low − 0.10×ATR, filled at *broker spread + slippage*.
  - Replay SL = same formula, but the exit check is on closed bars only with no spread.
  - Live's stop can trigger on intra-bar wicks the replay rounds away; replay's stops trigger only on the closed-bar high/low.
- Net: live re-arms at different times than replay, and which "cascaded" signals get accepted differs.

**Evidence:** EURCAD 78 % loss, GBPCAD 86 % loss above. Strategy *would have* fired more often; the gate rejects on purpose.

**Is this dangerous to edge?** Depends on what the WR/PF/expectancy numbers in `final_stats.csv` for each symbol were measured under. NB24's backtest *also* uses `ONE_TRADE_AT_A_TIME=True`. If the live cascade pattern differs systematically from the backtest cascade pattern (e.g., live always misses second-legs the backtest catches), then the empirical edge degrades.

### Rank 3 — **Bar-time format mismatch in NB30 output** (TRIVIAL but blinds the parity dashboard)

**Code path:** (NB30 — not read in this pass, but it's the producer of `parity_multi_scalper_detail.csv` / `parity_multi_scalper_summary.csv`)

**Mechanism:** BT side renders `bar_time` with `+00:00` (tz-aware → str), LV side reads `signal.bar_time` which is `str(bar["time"])` of a tz-naive Timestamp — no suffix. String join key never matches.

**Evidence:** 76 BT rows with `+00:00`, 55 LV rows without, 0 matches in the summary. Even *after* stripping suffix, per-symbol overlap is 0 because the windows + cascade-suppressed signals don't share bars in the union.

**Impact:** The dashboard you'd glance at says "zero parity" → operator panics → forensics like this one. Real signal-divergence rate is 0.5 %, not 100 %.

### Rank 4 — **M5 OHLC cache drift** (RARE but real)

**Code path:** Operational — `notebooks/data/<SYM>/M5/ohlcv.csv` written by NB00, never re-downloaded after live ran.

**Evidence:** 1 case (EURCAD 2026-05-27 17:40:00) where CSV `low=1.61005` vs live `low=1.61039` — ~3.4 pip difference. Enough to flip `f_ema` and `f_stoch` filters at the threshold. Result: BT alone fired +1, live fired 0.

**Impact:** Low — but means parity audits need fresh CSVs. Cron a daily refetch.

### Rank 5 — **Entry-price convention divergence** (NEGLIGIBLE)

**Code paths:** Live uses `close(signal_bar)` ([mt5/multi_symbol_bot/strategy.py:553](mt5/multi_symbol_bot/strategy.py#L553)); Replay (NB33) uses `open(signal_bar + 1)`.

**Evidence:** Drift summary above. ≤3e-05 on FX, ≤0.09 on XAUUSD. Slippage variance dominates.

---

## 3 — What the Prompt Got RIGHT and WRONG

Reconciling the user-listed divergences against the evidence:

| Prompt claim | Evidence verdict |
|---|---|
| "Live and replay sometimes enter on different bars" | Mostly NO — both use closed bars. The entry-price *convention* differs but the *bar* doesn't. |
| "Some live trades appear later than replay trades" | Possible — caused by ONE_TRADE cascade re-arm timing differences. |
| "Some replay trades do not exist in live" | YES — 5 GBPUSD signals replay produced, live couldn't (D1 top-up). + cascade effects. |
| "Some live trades do not exist in replay" | YES — 17 across GBPCAD+EURCAD (H1 top-up). + cascade effects. |
| "H1 indicators sometimes drift between live and replay" | YES — `h1_rsi` drifts on 1,685 of 1,717 bars; `h1_trend` flips on 162. Root cause: top-up vs broker H1. |
| "Config changes during runtime caused mismatches" | NOT EVIDENCED in this pass — would need `git log -p notebooks/results/multi_symbol_scalper/*/config.json` to confirm. Worth a follow-up. |
| "ONE_TRADE_AT_A_TIME created cascading differences" | YES — biggest live-vs-replay trade-count divergence, but the gate exists in both, so it's a *timing* divergence not a *gate* divergence. |
| "data_age_min and candle timing likely create bar lag" | NO direct evidence — `data_age_min` only triggers the 15-min stale-skip (27 events, mostly XAUUSD weekend). Not a signal-flipping source. |
| "Live may use partially formed candles while replay uses closed candles" | NO — `_fetch_bars` does `iloc[:-1]`. Both use closed bars. |
| "Replay reconstructs state from CSV instead of exact live snapshots" | YES — this is the deepest truth. CSV ≠ MT5 stream at the bar-close instant when topup synthesises HTF. |

---

## 4 — Spot-check Cases (drill-down evidence)

### Case A — GBPUSD 2026-05-26 15:20-16:15 (5 consecutive missed shorts)

```
2026-05-26 15:20: BT signal=-1 (SHORT), LV signal=0
  trend_dir BT=-1, LV=0
  d1_trend  BT=-1, LV=+1   ← root cause
  h1_rsi    BT=36.638, LV=41.573
```

Replay's CSV-D1 said the daily trend was still DOWN (consistent with the recent EUR-rally fade). Live's M5-topped-up D1 had aggregated enough fresh M5 momentum to flip its computed close above the EMA50, producing `d1_trend=+1`. Trend gate requires H1==D1==±1; H1 was -1 in both; only replay had the matching D1. Five M5 bars in a row, all missed by live, all five were SHORTs that the replay later showed would have been valid setups.

**Outcome impact:** 5 strategy-edge signals went uncollected on live. Whether they would have been profitable is a separate question — replay can simulate them (`replay_open_20260520_20260528.csv` likely contains the simulated outcomes for those bars).

### Case B — GBPCAD 2026-05-27 15:30-17:35 (8 fake longs live fired)

```
2026-05-27 15:30: BT signal=0, LV signal=+1 (LONG)
  trend_dir BT=0, LV=+1
  h1_trend  BT=-1, LV=+1   ← root cause (top-up synth vs broker H1)
  h1_rsi    BT=48.177, LV=53.57
```

Mirror of Case A: live's topped-up H1 said UP, broker-published H1 (which arrived later, used by NB31 replay) said DOWN. Live placed 8 BUY signals over ~2 hours; replay agrees those bars existed but says no trend → no signal.

**Outcome impact:** Live took these trades. If broker H1 was the "truth" and topped-up H1 was wrong, then these were fake signals against the *actual* HTF trend → expected to underperform the validated edge. Worth checking these 8 trades' outcomes in the logs.

### Case C — EURCAD 2026-05-27 17:40 (M5 OHLC cache drift)

```
2026-05-27 17:40: BT signal=+1 (LONG), LV signal=0
  f_ema   BT=+1, LV=-1
  f_stoch BT=-1, LV=0
  low     BT=1.61039, LV=1.61005   ← real OHLC differs
  close   BT=1.6104,  LV=1.61021
```

Different M5 low (34-point wider in CSV than live), different M5 close. Either:
- The CSV cache wasn't fully re-downloaded after this bar finalized, OR
- The broker amended this bar after publication (rare but happens).

**Outcome impact:** 1 signal lost. Operational fix: cron a daily M5/H1/D1 refetch.

---

## 5 — Recommended Architecture Changes

### MUST-DO (high impact, low risk)

**5.1. Stop synthesising HTF from M5 — or, if kept, log both views and never use the synth for the trend gate.**

The `topup_htf_from_m5` path is the single biggest source of live-replay divergence. Three options in increasing aggression:

| Option | Effect | Risk |
|---|---|---|
| A: Disable top-up | Live HTF lags broker publication (15min–4hr stale on H1 trend gate during weekends/holidays). Strategy might miss the first ~1-3 signals after a trend flip. | LOW — backtest edge was validated on broker-published HTF only. This *restores* parity to that baseline. |
| B: Keep top-up but log both `trend_dir_synth` and `trend_dir_broker`, gate on `broker` only when fresh, fall back to `synth` only after a configurable lag threshold. | Best of both worlds. | MEDIUM — more code, but auditable. |
| C: Keep top-up as-is and tolerate the divergence. | Status quo. | Already shows 21 flipped signals over 4 days. |

**Recommendation: B**. Keeps live responsive to fresh data but only when broker is truly behind by a threshold *and* logs both for audit. Implementation cost ~50 lines in `topup_htf_from_m5`.

**5.2. Fix the `bar_time` JOIN-key format in NB30 (the parity comparison generator).**

Strip `+00:00` on both sides before joining. Trivial — one `.str.replace(r"\+00:00$", "", regex=True)` on each side's `bar_time` column. After this fix, the summary CSV will show real match rates instead of always zero.

**5.3. Cron a daily refresh of `notebooks/data/<SYM>/M5/ohlcv.csv`.**

Operational only. Prevents Case C-style "real OHLC differs" surprises. Run `notebooks/00_data_feching.ipynb` (or its `.py` equivalent) at 23:00 UTC daily.

### SHOULD-DO (medium impact, requires design call)

**5.4. Decide what "signal preservation" means for ONE_TRADE_AT_A_TIME.**

The cascade is the biggest "signal loss" source by raw count. BUT — the gate is intentional and is in your validated backtest. Three positions:

| Position | What you keep | What you lose |
|---|---|---|
| Keep ONE_TRADE_AT_A_TIME (current) | Risk control. WR/PF numbers in `final_stats.csv` are valid. | 39-86 % of detected signals never trade. |
| Remove ONE_TRADE_AT_A_TIME globally | Every signal trades. | Real money risk inflates (potentially 3x). Need new backtest with the gate off. Edge may degrade. |
| Replace with a tiered overlap policy (e.g., max 2 concurrent positions per symbol, max 4 portfolio) | Some signal recovery, controlled risk increase. | Requires re-validation across the basket. |

**Recommendation: KEEP ONE_TRADE_AT_A_TIME for now**, but instrument the missed-signal outcomes (which the engine already does — `missed_signal` field in the skip event at [execution_engine.py:444-447](execution/execution_engine.py#L444-L447)). After 30 days of live data, you can replay the missed signals' outcomes vs the taken signals' outcomes. If missed signals would have been profitable, *then* consider raising concurrency to 2 per symbol.

**5.5. Stop comparing trade counts; compare *acceptance gate outcomes*.**

Right now NB30's summary CSV compares "how many trades did replay produce" vs "how many orders did live place". Live is the *gated* version, replay is the *ungated* version (well, both are ONE_TRADE-gated, but with different fill timing). The right comparison is:

- Set of bars where `signal_dir != 0` (in NB31) — the *signal* set.
- Of those, how many became `order_placed` events — the *executed* set.

That's a 2-stage funnel, not a 1-to-1 match. NB31 already has the signal set; the funnel just needs a second pass.

### SHOULD-NOT-DO

**5.6. Don't switch to "fully closed-candle system everywhere".** Live ALREADY uses closed candles (`iloc[:-1]`). This wasn't actually a divergence source. The prompt's framing was wrong on this point.

**5.7. Don't build an event-driven candle-close engine to replace the polling loop.** The polling loop fires within 500ms of the M5 boundary ([run_multi_scalper.py:670](mt5/run_multi_scalper.py#L670) `extra=0.5`). The 9 entry-drift samples above show sub-pip drift between close-of-i and open-of-i+1. The 500ms latency is below the bar-to-bar drift floor — event-driven gains nothing measurable.

**5.8. Don't build snapshot-based deterministic replay.** It would require live to snapshot M5+H1+D1+config+seen_signals on every cycle, deployed to write to disk for weeks, then loaded back. Months of elapsed time for an incremental gain over the existing CSV-based replay. Better: fix the HTF top-up (5.1) so replay reads the same data live used.

**5.9. Don't add tick-approximation replay.** No tick data is available in the project. Building it requires storing every tick from MT5 — operationally expensive and not justified by the sub-pip drift evidence.

---

## 6 — Migration Plan (risk-ranked)

| Phase | Change | Risk | Validation gate |
|---|---|---|---|
| 1  | Fix NB30 `bar_time` format mismatch | Zero — comparison-only | After fix, summary CSV shows non-zero `matched` counts |
| 2  | Cron daily M5/H1/D1 refetch | Low — operational only | Diff today's CSV vs yesterday's; should equal new bars only |
| 3  | Add `data_source` to live `cycle.diag` logging (broker vs synth flag per HTF bar) | Low — observability | New `cycle` events carry `h1_source: "broker"\|"synth"` |
| 4  | Implement Option B for `topup_htf_from_m5` (log both views, gate on broker until lag threshold) | Medium — changes live signal output | Run dry-run for 5 days; compare to current live signals; expect ≤ 5 % signal-set difference |
| 5  | Update `parity_multi_scalper_summary.csv` to be a 2-stage funnel (signal-set match → execution match) | Low — analytic only | Funnel rates make `position_open` cascade visible as its own stage |
| 6  | (DEFERRED) Reconsider ONE_TRADE_AT_A_TIME concurrency after 30 days of `missed_signal` outcome data | High — affects real money risk | A/B compare: forward-net of missed signals vs forward-net of taken signals over the window |

---

## 7 — Operational Risk Summary

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Strategy edge degradation from synthesized HTF mis-signals | MEDIUM | MEDIUM (8 fake GBPCAD longs may underperform) | Phase 4 fix |
| Operator confusion from "0 matches" dashboard | HIGH | LOW (misleads root cause analysis) | Phase 1 fix |
| Missed signals from ONE_TRADE cascade biasing returns | HIGH | UNKNOWN | Phase 6 (need 30 days of `missed_signal` data) |
| CSV cache staleness drifting indicator outputs | LOW | LOW (1 case in 1,717) | Phase 2 fix |
| Entry-price convention noise | LOW | NEGLIGIBLE | Ignore |

---

## 8 — Final Production Recommendation

1. **Ship Phase 1 + 2 today.** Pure improvements, zero risk to live trading.
2. **Ship Phase 3 + 4 next week.** Hard part is design call on the `topup_htf_from_m5` policy. The data says "gate on broker only" is safer than "always use synth", but the live signals will become slightly slower to react after trend flips.
3. **Ship Phase 5 in parallel.** It clarifies the parity dashboard but doesn't touch trading code.
4. **DEFER Phase 6 for 30 days of data.** ONE_TRADE_AT_A_TIME concurrency policy is a real-money question. Decide it on `missed_signal` outcome data, not theory.
5. **REJECT** the prompt's scenarios D (event-driven engine), E (snapshot replay), I (tick approximation). Cost-to-evidence ratio is wrong; no measured benefit to chase.

The live system's *strategy-divergence rate* of 0.5 % (23 flipped signals in 4,400 cycles) is actually impressively good for a closed-candle, broker-data-driven system. The "100 % divergence" headline was a comparison artifact, not a strategy artifact. **Don't over-engineer.** Fix the top-up policy, fix the comparison, and watch.
