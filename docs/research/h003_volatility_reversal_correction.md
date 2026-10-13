# H003: wording correction (added, the original file is unchanged)

Phase M of `docs/research/plan/ROADMAP.md`. This note corrects the **interpretation** of H003, not its numbers. The original
[`h003_volatility_reversal.md`](h003_volatility_reversal.md) stays as it is (add, don't modify).

## What the original says

- Line 1 (title): "H003 — انعكاس/انكماش التقلّب المدفوع بـ`NATR_14`" (volatility **reversal / contraction** driven by `NATR_14`).
- Line 36 (the single registered hypothesis): "تقلّب مرتفع حالياً (`NATR_14`) يسبق انكماشاً في نطاق الشمعة القادمة (`high` أدنى،
  `low` أعلى)": high current volatility precedes a **contraction** of the next candle's range (lower `high`, higher `low`),
  explained as volatility exhaustion after a sharp move.
- Lines 23 to 26 (table): the accepted candidates are `NATR_neg_vol` = `-NATR_14` with mean IC **-0.1970** on `high` and
  **+0.1644** on `low` (30/30 windows), and `RSI_vol_adjusted_reversion`, which the file itself says is `-NATR_14` in disguise
  (lines 3 and 34).

## What is wrong

The candidate in the table is the **negated** feature `-NATR_14`. A negative IC of `-NATR_14` with the next `high` is a
positive IC of `NATR_14` itself, and a positive IC of `-NATR_14` with the next `low` is a negative IC of `NATR_14`. So the
signs in the table were read backwards in line 36:

| Feature | Target | IC (rank) | Windows | Source (`normalization_audit_and_corrected_retest.md`) |
|---|---|---|---|---|
| `NATR_14` | next `high`, absolute | in-sample **+0.197** (93 %, 30/30) | 30 | line 80 |
| `NATR_14` | next `low`, absolute | in-sample **-0.164** (97 %, 30/30) | 30 | line 80 |
| `NATR_14` | next `high`, absolute | out-of-sample **+0.211** (95 %, 38/38) | 38 (2023-07-21 to 2026-09-02) | line 149 |
| `NATR_14` | next `low`, absolute | out-of-sample **-0.199** (95 %, 38/38) | 38 | line 150 |
| `NATR_14` | next `high`, market-neutral | in-sample +0.223, out-of-sample **+0.230** (38/38) | 30 / 38 | lines 91, 147 |
| `NATR_14` | next `low`, market-neutral | in-sample -0.317, out-of-sample **-0.328** (38/38) | 30 / 38 | lines 92, 148 |
| `NATR_14` | next `close` | +0.041 in-sample, +0.007 out-of-sample (22/38) | 38 | line 151 |

(The targets are the future high and low relative to the last close, as in the original file; the lines above are in
`docs/research/normalization_audit_and_corrected_retest.md`.)

High `NATR_14` therefore precedes a **higher** next `high` and a **lower** next `low`: a **wider** next range. That is
volatility clustering. It is neither a reversal nor a contraction, and the "exhaustion after a sharp move" explanation in
line 36 has no support. The direction of the next close is not predicted (IC about zero, outside H003's scope as the
audit already says in its line 151).

## What stays true

- Every number in the original tables, the multiple-testing review and the out-of-sample confirmation is unchanged; H003 is
  still accepted, and the out-of-sample result (38/38 windows, IC +0.211 / -0.199) is the evidence the correction relies on.
- The note at the top of the original file already states the corrected signs (+0.223 / -0.317; +0.211 / -0.199) but still
  calls the finding a reversal; this note replaces the wording, not the statistics.
- The "RSI mechanism" of `RSI_vol_adjusted_reversion` was already withdrawn there (line 3).

## Consequences for later phases

- Any plan or card that cites H003 describes it as: **`NATR_14` forecasts the width of the next range** (up and down
  extension), not a direction and not a reversal or contraction.
- R3 (NATR range tool) is built on that statement; its baseline and accept rule are in `docs/research/plan/ROADMAP.md`.

Evidence file: `docs/research/normalization_audit_and_corrected_retest.md` (lines 80, 91 to 92, 147 to 151).
