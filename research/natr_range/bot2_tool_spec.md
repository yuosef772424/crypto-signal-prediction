# bot2 tool spec: `get_range_forecast` (from R3 / E-natr-001; format of bot2 `ai_agent/docs/TOOLS.md`)

Status: SPEC ONLY, nothing was changed in bot2. Provisional: 1h passed the exploration rule (val 2025-01..2026-09), the H2 confirmation is prepared but not run;
it becomes final only after that confirmation. 1d did NOT pass (F-0082) and must be refused.

## ToolSpec
```python
SPEC = ToolSpec(
    name="get_range_forecast",
    description="Calibrated forecast of the NEXT bar's high-low range as percent of the last close: p10/p50/p90, plus the median up and "
                "down excursion. A width forecast with stated coverage (80% of next ranges fell inside p10-p90 on validation), "
                "NOT a direction signal.",
    params={"type": "object", "additionalProperties": False, "required": ["symbol", "interval"],
            "properties": {"symbol": {"type": "string"}, "interval": {"type": "string", "enum": ["1h"]}}},
    max_output_chars=700, cost_class="free", effect="read")
```
`run(args, ctx)` uses only closed bars up to `ctx.asof` (no look-ahead), `interval` other than `1h` raises `ToolError("get_range_forecast: only 1h is validated (1d failed calibration)")`.

## Output (key=value lines, percent units, as_of on every number)
```
symbol=BTCUSDT interval=1h as_of=2026-10-17T09:00Z bar_open=2026-10-17T08:00Z
natr14_pct=0.60 natr_decile=1 as_of=2026-10-17T09:00Z
next_range_p10_pct=0.30 next_range_p50_pct=0.51 next_range_p90_pct=1.04 as_of=2026-10-17T09:00Z target_bar_open=2026-10-17T09:00Z
next_up_p50_pct=0.22 next_down_p50_pct=0.23 as_of=2026-10-17T09:00Z
coverage_p10_p90_val_pct=80.0 coverage_p10_p90_val_decile_pct=79.5 model_version=natr-range-v1 calibration_asof=2024-12-31 validated_through=2026-09-30
```
- Units: percent of the last closed 1h close (`range = 100 (high - low) / close`), `natr14_pct` = NATR_14 in percent (same unit as bot2 `natr`, Wilder, 14 bars).
- `natr_decile` 0..9 from the frozen train edges (0.548 0.720 0.858 0.991 1.132 1.291 1.495 1.798 2.391 percent); `coverage_p10_p90_val_decile_pct` is the validation coverage of that decile (77.2 in decile 9).
- Model (`natr-range-v1`, frozen): `p_tau = exp(a_tau + b_tau ln(natr14_pct))`, tau 0.1/0.5/0.9, coefficients in `research/natr_range/frozen_rules.json` (range p50: a -0.162, b 0.986; p10: -0.700, 1.005; p90: 0.519, 0.936), sorted across tau.
- Fail closed: fewer than 15 closed 1h bars, a missing bar in the last 14, `natr <= 0` or a non-finite number gives `unavailable` for every forecast line, never 0 and never a stale value; refuse a symbol that has no 1h feed.

## How it differs from bot2's existing `range_pct` / `natr`
`range_pct(candles)` and `natr` are CURRENT measures of what already happened. This tool returns a forecast of the NEXT bar with a validated 80% interval, so the model may size a stop or a target against it
("stop 0.4% is inside the p10-p50 range"). It adds no direction: per bot2 ADR 0003 and the H003 correction, high NATR precedes a wider range in both directions (IC on next high +0.21, next low -0.20, close about 0).

## Known limits to show the model (put them in the tool description or the knowledge card)
- Coverage in the top NATR decile is 77.2% (slightly narrow), BTC/ETH in 2026 covered 75.1% at p10-p90 (other coins 80.6%): treat p90 as a soft cap for BTC/ETH in high-vol states.
- Calibrated on 10 majors (Binance spot) 2020-2024 and validated 2025-01..2026-09; the H2 check on other coins is pending.
- Pinball loss is 10% below trailing-90-bar quantiles, i.e. a modest, real gain; the p10-p90 interval of the trailing baseline covers only 75.4%.

## Registered use (not part of this spec)
The same forecast is a candidate model feature or sizing target for R6 (pipeline column with a card and a test); it needs no extra data (NATR_14 and OHLC only).
