# LAB_DESIGN: the lightweight screening lane (`lab/`)

> **Status: PROPOSED (2026-10-21), not accepted.** Nothing here is built. It becomes binding only when the owner accepts it in
> `decisions/` (see "Decisions needed"). New docs are in English (`STATUS.md` scope); Arabic summary first.

**ملخص بالعربية.** المشروع كبير على اختبار فكرة واحدة (إضافة ميزة تلمس 8–10 ملفات). المقترح مسار خفيف جديد `lab/` بجانب البنية
الحالية لا بداخلها: جدول مسطّح (عملة × إغلاق شمعة) فيه *حالات إشارات لها صلاحية* (لا تتطابق في الشمعة نفسها) ونتائج لاحقة بعدة
آفاق. إضافة إشارة = ملف واحد مسجَّل باسمه، واختباراتها العامة تُولَّد من السجلّ تلقائياً. التجارب تمرّ ببوابة «فرق كبير» ثم تُسجَّل
في دفتر لا يُعدَّل، والنموذج مساعد لاحق يعمل على ما نجا فقط. البنية تتحسن مع الزمن لأن كل قاعدة تُفرض آلياً (CI) وبمؤشر قياس واحد:
**كم ملفاً يلمس إضافة إشارة جديدة** (الهدف ≤ 2، وأي ارتفاع إنذار).

## 1. Problem (measured on master, 2026-10-21)
| Fact | Number / source |
|---|---|
| Code in the code packages | ~41.5 k lines, 360 `.py` files, 784 test functions |
| Cost of one extension | the last three `feat` commits touched 8, 9 and 10 files |
| Implicit (shared-namespace) dependencies tolerated | 104 allowlist entries (`tools/implicit_deps_allowlist.txt`) |
| Experiments closed | 85 failure rows; ~450 indicator trials from one survey alone |
| Planned but never tested | Fibonacci, FVG, order blocks, Ichimoku, liquidity pools (ROADMAP R1: "not started", no code) |
| Multi-step targets | the pipeline has one `forecast_horizon` (1) |
| Effective sample on the model line | ~1.5 k (`external_research_2026-10-20.md`, after the #45 fix) |

Diagnosis: the heavy machinery (windows, NIG heads, trainer) is the *unit of work* for every question, including questions that need
only a table and a few statistics. Testing an idea costs a model; so ideas are tested rarely, or in thousands of near-identical
variants.

## 2. Rules this design is built on (quoted)
Quoted rules are the contract. "How applied" is the only place this document adds meaning.

| Quote | Source | How applied |
|---|---|---|
| "adding a feature means *adding* code, not *rewriting* it." | xy `playbooks/12-architecture-rules.md` (goal) | Every extension point is a registry; §5 extension test |
| "For **every** Should/Could feature, write one line: *which modules change, and how*." | xy 12, A3 | §5 table, re-run at every health check (§9) |
| "Design for the full list, build only the MVP." | xy 12, A2 | Fibonacci/Ichimoku/FVG/multi-step/assistant model all have a place (§4) but only L0–L3 are scheduled (§10) |
| "Don't add extension points anywhere else; speculative abstraction is also a cost." | xy 12, D1 | Four registries only (signals, outcomes, confluence operators, screen tests); nothing else is pluggable |
| "Rules that live only in a document will drift, especially with AI builders. **Enforce them in CI**" | xy 12, F | §8 maps each rule to a check; no rule without one |
| "Ratchet, don't block: ... **no new** violations get in, and shrink the baseline over time" | xy 11, adopting in an existing project | `lab/` starts with a zero-entry allowlist; legacy baselines only shrink |
| "A big rewrite is a sign that small corrections were skipped." | xy 12, E3 | Legacy packages are frozen, not rewritten (§3); survivors are ported one by one (§10, L5) |
| "Decisions are recorded with what would make them wrong" | xy 12, G | §11 decision table has a `revisit if` column |
| "Add, don't modify: a new study is `research/<study>/` with a README; recorded results are never edited" | this repo, `CLAUDE.md` | Ledger and schema are append-only (§8) |
| "One component, one interface ... a new method = a new registered component (unknown names raise)" | this repo, `CLAUDE.md` | Every signal/outcome/operator/test is registered; learned state (assistant model) has `fit`/`transform`/`score` |
| "No silent defaults: unknown config keys ... raise" | this repo, `CLAUDE.md` | Strict config objects in `lab/cli.py` |
| "Any script used more than once goes into `tools/` ... with a card and a test" | this repo, `CLAUDE.md` | `lab/` modules each carry a PURPOSE/TAGS/PITFALLS card (`build_map.py --check`) |
| "الفحوص تفرض، والوثائق تقترح" (rule 7) and "الإضافة بدل التعديل" (rule 2) | this repo, `PHILOSOPHY.md` | same as above |
| "Correlation is not cause ... Every variant tried counts toward multiple testing, including rejected ones." | this repo, `RESEARCH_RULES.md` §4 | The ledger counts every config; §7 gate uses that count |
| "'More data', 'a newer period' ... are **not** reopen reasons unless the row's `reopen_if` lists them" | this repo, `RESEARCH_RULES.md` §3 | `lab` runs call `tools/experiment_registry.py check` on a family card before running |

## 3. Scope: MoSCoW (xy 12, A1)
| Class | Items |
|---|---|
| **Must** | Flat panel table; signal registry with *states that have validity*; outcome registry with multi-horizon outcomes in R units; screen with nulls and the "big difference" gate; append-only ledger; CI rules that keep `lab/` clean |
| **Should** | Confluence operators (count within window, ordered, distance in ATR); instrument-validation cases (§7); health check report (§9) |
| **Could** | Assistant model (meta-labeling) on survivors; funding/OI data adapter; relative-target outcome variants; promotion to the pipeline (L5) |
| **Won't (now)** | Training NIG-TimeNet inside `lab/`; refactoring `data/ model/ trainer/ evaluation/ workflow/ discovery/`; new model-architecture experiments (E-rel-001 and E-norm-* are parked until `lab/` reports); live trading; order-book/on-chain data |

**Freeze (proposed):** `data/ model/ trainer/ evaluation/ signal_eval/ workflow/ discovery/` accept `fix` commits only while `lab/` is
being built. No deletions, no test removals. Registry, holdout guard and run contracts (ROADMAP "Common contracts") are reused.

## 4. Architecture
### 4.1 Zone and direction (modular monolith, xy 12 B3; zones, `PHILOSOPHY.md` §2)
`lab/` is a **normal importable package** (explicit arguments, no shared namespace, no notebook globals). It is a new code package in
`tools/check_deps.py` (`CODE_PACKAGES += ("lab",)`).

```
core/schema (tier 0)        <- imports nothing from the repo
   ^
lab/schema        constants: column names, dtypes, SCHEMA_VERSION, ATR/R conventions
   ^
lab/bars          the only door to market data (wraps tools/holdout_guard); UTC, closed bars only
   ^
lab/signals/      registry + one file per family   |  lab/outcomes/  registry + one file per family
   ^                                                   ^
lab/table         joins signal states and outcomes into the panel (owns the table schema)
   ^
lab/confluence/   operators over states            |  lab/screen/   statistics, nulls, gate, verdict
   ^                                                   ^
lab/ledger        append-only run log + multiplicity counter + family budget
   ^
lab/cli           strict config -> run -> manifest.  (`python -m lab screen --config cfg.json`)
   later: lab/assist/   assistant model on survivors (Could)
```
Rules: imports point down only; no cycles; another module is imported **only through its package root** (`from lab.signals import
get_signal`, never `lab.signals.fvg`); `lab` imports `core`, third-party libraries, and from `tools/` only an explicit list
(`holdout_guard`, `experiment_registry`); **legacy packages never import `lab`**. Studies may import `lab` (as for any code package).

### 4.2 Data ownership (xy 12, C3)
| Module | Owns | Others get it through |
|---|---|---|
| `bars` | raw OHLCV access, resampling, seal enforcement | `Bars` handle |
| `signals` | state definitions, versions | `get_signal(name)`; states are *computed*, never stored by others |
| `outcomes` | outcome definitions, cost model | `get_outcome(name)` |
| `table` | the panel schema and its parquet files | `build_panel(...)`, `read_panel(...)` |
| `ledger` | `research/lab/ledger.csv` | `record_run(...)`, `count_tests(family)` |

Raw data and panels stay **outside the repo** (`$LAB_DATA_DIR`); the repo gets manifests and small result CSVs only
(`PHILOSOPHY.md` §6: files < ~1 MB, no raw data).

### 4.3 The signal contract: states with validity (fixes "signals never coincide on one candle")
A signal is not a 0/1 on the bar where it fired. It is a **state** that is *valid from* a causal bar and *expires*:

| Column (per coin, per decision bar `t`) | Meaning |
|---|---|
| `side` int8 | +1 bullish / -1 bearish / 0 none, as known at the **close** of `t` |
| `age` int16 | bars since this state became valid (`valid_from` = close of the bar that confirms it) |
| `lo`, `hi` float32 (nullable) | zone edges in price, for zone-type signals (FVG, order block, Fibonacci 0.5–0.618, S/R) |
| `dist_atr` float32 (nullable) | signed distance of the close to the zone, in ATR(14) of the decision timeframe |

`NaN`, never `0`, for insufficient history (the gate-cli lesson, `ROADMAP` G2). A state ends when its definition says so (FVG filled,
Ichimoku cloud flips) or at `expire_bars`. **Confluence is then computed, not coincident:** e.g. "FVG valid AND `dist_atr` to the Fibonacci
zone < 0.5 AND cloud side bullish, all valid at the same decision bar", even though they formed on different candles.

```python
@register_signal("fvg", family="structure", version=1)
def fvg(bars: Bars, *, min_gap_atr: float = 0.25, expire_bars: int = 48) -> StateFrame: ...
```
Unknown signal names or parameters raise (`CLAUDE.md`: no silent defaults). A new parameter has a default that reproduces the previous
behavior, and a changed definition is a **new version**, never an edit.

### 4.4 The outcome contract: multi-step targets are columns
For each decision bar: entry at the **next bar's open**; for each horizon `k in {1,3,6,12,24}` decision bars: `ret_k`, `mfe_k`, `mae_k`
(in ATR units and in R), `first_touch_k` (+1R before -1R; stop-first on ambiguity, the honest lower bound used by
`tools/bracket_eval.py`), `range_k` (next-range width, the H003 target). Cost enters as a parameter in R (0.14 % round trip, also 2x;
ROADMAP contracts). Variants are registered, not coded in the screen: `xs_demean` (relative to the same-timestamp median),
`mkt_resid`. **This answers the absolute-vs-relative question of E-rel-001 cheaply here, before any model.**

### 4.5 Assistant model (Could, later; "model as helper")
Meta-labeling shape: a *rule or confluence* proposes the event; a small model answers "is this event better than the base rate?".
Samples are the survivors' events only (few, homogeneous), one `fit` on train -> state, `score` on val; capacity controls of
`RESEARCH_RULES.md` §2.1 are mandatory (effective sample size, k-sweep, learning curve, simple baseline). Starts with logistic/GBM; it
does not require NIG-TimeNet. (The name "meta-labeling" is from López de Prado, *AFML*, cited from memory: verify before citing in a card.)

## 5. Extension test (xy 12, A3), written before building
| Future item | Modules that change | Result |
|---|---|---|
| New signal (Fibonacci, FVG, order block, Ichimoku, liquidity pool, Williams fractal) | **add** one file in `signals/` (+ its golden fixture) | ✅ |
| New signal family needing a new data source (funding, OI) | **add** an adapter file in `bars/` + signal files | ✅ (1 module + additions) |
| New outcome horizon / new multi-step target | add an entry in `outcomes/` or a config value | ✅ |
| New outcome variant (relative, residual, rank) | add an entry in `outcomes/` | ✅ |
| New confluence operator | add a file in `confluence/` | ✅ |
| New null / new test statistic | add an entry in `screen/` | ✅ |
| Different timeframe or universe | config only | ✅ |
| New column in the panel | append to `lab/schema.py` (append-only, `SCHEMA_VERSION` bumps) | ✅ (1 module) |
| Assistant model | add `lab/assist/` consuming the panel; no existing module changes | ✅ |
| Promote a survivor into the pipeline (L5) | add a feature in `data/` + a parity test in `tests/`; `lab` unchanged | ⚠️ 1 legacy module, justified (direction stays one-way) |
| Rename or remove a panel column | schema + every consumer | ❌ forbidden by D2 (stable contracts); add a column instead |

## 6. Why adding a signal gets cheaper *and* safer over time
The registry feeds a generic contract test (`tests/test_lab_registry_contract.py`) that runs **for every registered signal and
outcome**, so a new one is covered without writing new test code:
1. **Causality:** replacing all bars after `t` with noise leaves the states at `<= t` unchanged (the pattern of `tests/test_no_lookahead.py`).
2. **NaN, not 0,** for insufficient history. 3. **Determinism** (same input, same bytes). 4. **Schema/dtype** match `lab/schema.py`.
5. **Registered metadata** (`family`, `version`) and **unknown parameters raise**. 6. A **golden fixture** exists (small synthetic series with hand-checked states).
The more signals, the more the same contract protects; the author only writes the signal and its fixture.

## 7. Screening protocol: "is the difference big enough to test?"
**Tiers** (reuse the project's existing evidence ladder; no card per trial):
| Tier | What | Paperwork |
|---|---|---|
| 0 Screen | cheap panel run on the 10-coin cloud data (or Drive coins on Colab) | none; one ledger row per config, automatic |
| 1 Candidate | passed the gate below | one experiment card per *family* (`tools/experiment_registry.py check`) |
| 2 Confirm | one frozen run on a sealed bucket via `holdout_guard.open_holdout` | per ROADMAP R0 rules |

**Gate Tier 0 -> 1 (reused from the pre-registered R1 bar; all must hold):** n >= 300 events on >= 8 coins; gross mean R above the null's
95th percentile by >= 0.10 R; net mean R > 0 at 1x cost; lift >= +0.05 R over the best member alone; same sign in both halves of the period;
survives BH-FDR q = 0.10 over **every config the ledger counts for that family**; plus per-coin consistency and day-clustered t (the
evidence bar of the ROADMAP).
**Detectability pre-check (run before the screen; skip the test if it fails):** approximate minimum detectable effect
`MDE ~ (z_a + z_b) * sigma / sqrt(n_eff)` with `n_eff` = independent day-clusters of events. Illustration, not a result: sigma ~ 1 R,
`n_eff` = 400 gives MDE ~ 0.14 R; a family whose literature-plausible effect is below its MDE is not run.
**"Big difference" rule for variants:** inside a family, parameter variants span a factor of >= 2 (e.g. lookback 10/20/40), at most 3 values per
parameter; a different *idea* is a different family. **Family budget:** `lab/ledger` refuses the 13th config of a family unless
`--override-reason` is given (recorded in the ledger). Cost: tests per family stay in the tens, not thousands.

**Instrument validation before use (L1):** the screen must reproduce two known results or it is not trusted: (a) a known negative: hourly
`-RSI_14` reversal shows gross information but net <= 0 after 0.14 % (F-0036); (b) a known positive: `NATR_14` predicts next-range width
(H003, confirmed OOS). If either fails, fix the lab, not the conclusion.

## 8. Enforcement (xy 12, F): every rule has a check
| Rule | Check | Where |
|---|---|---|
| `lab` imports only `core`, third party, and the two named `tools` modules; legacy never imports `lab` | new zone rules | `tools/check_deps.py` |
| imports point down the layer order; no cycles; only package-root imports across modules | `LAB_LAYERS` list + test | `tools/check_deps.py` |
| zero allowlist entries for `lab/` | entry count in CI output | `tools/check_deps.py` |
| function ~50 / file ~300-400 lines (signals to look closer, xy E1) | size report, **new code only** | CI (warning, then ratchet) |
| each module has a card (PURPOSE / TAGS / PITFALLS); maps up to date | existing | `tools/build_map.py --check` |
| every signal/outcome obeys the contract | generic registry test | `pytest` |
| panel schema is append-only | compare `lab/schema.py` with `lab/schema_baseline.json` | `pytest` |
| ledger is append-only | each row carries the sha256 of the previous row | `pytest` |
| strict config, unknown keys raise | config class tests | `pytest` |
| run contract: manifest with git SHA, config, input fingerprint, **number of configs evaluated**, output sha256 | `lab/cli` writes it | `pytest` + review |
| reads only through the seal | `lab/bars` wraps `tools/holdout_guard` | `pytest` |

## 9. How the structure improves instead of decaying
Health check every 3 phases, by a reviewer agent (xy 12, F "architecture health check"), recorded as one row in
`docs/research/plan/lab/HEALTH.md`:

| Indicator | Target | Alarm |
|---|---|---|
| **Median files touched to add one signal** | <= 2 (signal + fixture) | any increase vs the previous row |
| Allowlist entries in `lab/` | 0 | any |
| Largest file / function (lines) | within guidelines | growing for 2 checks in a row |
| Signals with golden fixture and passing contract | 100 % | < 100 % |
| Extension-test table (§5) | all ✅ | any ❌ or new ⚠️ without a written reason |
| Time from "new idea" to first screen result | falling | rising |
Drift found is fixed in small PRs immediately; "a big rewrite is a sign that small corrections were skipped."

## 10. Phases (each one a working increment, one branch, one PR; xy 12, G)
| Phase | Delivers | Definition of done (checkable) |
|---|---|---|
| **L0** thinnest slice | `lab/` skeleton, contracts, CI rules, one trivial signal, one outcome, end-to-end on 10 coins at 1h | `check_deps` passes with the new zone; contract test green; one panel built; manifest written |
| **L1** instrument | multi-horizon outcomes, cost in R, nulls, ledger, family budget, gate, MDE pre-check, §7 validation cases | both validation cases reproduce; ledger hash-chain test; gate unit tests |
| **L2** signal batch | Fibonacci, FVG, order block, Ichimoku, swing S/R, liquidity pool (the R1 detector list) | each has a golden fixture; contract test covers all |
| **L3** confluence + first real screen | operators + one screen per family under §7 | candidate list produced mechanically; registry rows for every closed family |
| L4 (Could) | assistant model on survivors | only if L3 has a candidate; capacity controls pass |
| L5 (Could) | promote one survivor into `data/` with a parity test | byte-equal states vs `lab` on a fixed series |

**Stop rule (proposed):** time-box L0–L3 to **6 weeks**. If no family passes the gate by then, close the families in the registry and
stop searching for direction; ship what already survived (range/volatility tool, H07 trend filter) as the product. This is the "knowing
when to stop" the project lacks today.

**Relation to ROADMAP:** R1/R1b (chart-structure confluence) have not started; this design proposes implementing R1 *on* `lab/`
(card `E-conf-001` with the same pre-registered bar), instead of a separate heavy build. R2 (funding) is independent and can proceed in
parallel; its data adapter lands in `lab/bars` when it exists.

## 11. Decision records (with what would make them wrong)
| ID | Decision | Revisit if |
|---|---|---|
| D-L01 | New package `lab/` beside the legacy packages, legacy frozen (strangler), not an in-place refactor | the freeze blocks a needed legacy fix more than twice |
| D-L02 | Flat panel (coin x close time) instead of windows | a survivor needs raw windows (then the model lane builds them from the survivor's spec) |
| D-L03 | Signals are states with validity, not single-candle events | state definitions prove too coarse for a family |
| D-L04 | Registries + generic contract tests, not per-signal test files | tests need per-signal logic that a contract cannot express |
| D-L05 | `lab` never imports legacy; promotion by re-implementation with a parity test | parity tests cost more than they protect |
| D-L06 | No model in Tier 0; the model is an assistant on survivors | Tier 0 finds nothing and the owner wants a learned screen |
| D-L07 | Panels and raw data outside the repo; manifests in the repo | manifests cannot reproduce a run |
| D-L08 | 6-week time-box and a 12-config family budget | the first screen shows the budget is too tight or too loose |

## Decisions needed from the owner (one message, with my recommendation)
1. **Accept the lane and the freeze?** Recommended: yes. Alternative: restructure in place (not recommended; the ratchet protects legacy as it is).
2. **Time-box 6 weeks and family budget 12?** Recommended: yes, as defaults; both are in the ledger config and change by a decision record.
3. **Implement R1 on `lab/`?** Recommended: yes (same pre-registered bar, much cheaper build).
4. **First data:** 10 coins at 1h/4h in the cloud (reproducible now, all coins already "touched", so exploration only) with the confirmation on a sealed
   Drive bucket on Colab later. Recommended: yes.
5. **Park E-rel-001 until L1** (the relative-target question is answered cheaply by §4.4 first). Recommended: yes.

*Not covered / not verified:* the numbers in §1 were measured from the repository on 2026-10-21; the MDE figure is an illustration; the
meta-labeling attribution is from memory; nothing in `lab/` exists yet.
