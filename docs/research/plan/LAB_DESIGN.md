# LAB_DESIGN: a screening kernel that absorbs legacy code (working name `lab/`)

> **Status: ACCEPTED IN PRINCIPLE, DORMANT (D-011, 2026-10-21).** Nothing here is built and **nothing in this document is active**: no freeze, budget, time-box,
> parking or phase applies, and nobody relies on it, until the owner draws a roadmap/plan for completing the project and records an activation decision
> (see D-011). At activation it is re-read against that plan and revised if needed. New docs are in English (`STATUS.md` scope); Arabic summary first.
> Rev. 2 replaces rev. 1: it is no longer a parallel lane, it is the *destination* of extraction, with a complexity budget, a retirement clause,
> and a one-in-one-out rule.

> **حالة الوثيقة: جامدة.** لا تُنفَّذ ولا يُحتجّ بها حتى يرسم المالك خارطة وخطة لإتمام المشروع ويسجّل قرار التفعيل.

**ملخص بالعربية.** `lab/` ليست شرطاً بحد ذاتها، واسمها مؤقت. الشرط أن يُكتب الكود الجديد **مستورَداً بمعاملات صريحة** لا داخل النطاق
المشترك، لأن كل الحزم الحالية المناسبة (`signal_eval/`, `discovery/`) تعمل بالنطاق المشترك ووضع الجديد فيها يكرّر الترقيع. لذلك تُنشأ
حزمة واحدة هي **وجهة استخراج**: كل مرحلة تمتصّ جزءاً قديماً (يصير طبقة رقيقة تستورد منها) فينقص القديم ولا يتضخم المشروع. ميزانية تعقيد،
وشرط تقاعد للمسارات القديمة، وحدّ أدنى في L0، ومؤشر نجاح: الزمن من الفكرة إلى أول نتيجة.

## 0. Why a new package at all? (and why not)
| Option | What it means | Verdict |
|---|---|---|
| A. Put the new code inside `signal_eval/` or `discovery/` | They run every module in ONE shared namespace (`_loader.py`: `exec` into a dict) and need the data pipeline's names (`data/windows`, `data/parallel`) to evaluate one candidate. New code there inherits the coupling | ❌ this *is* patching |
| B. A parallel lane beside everything (rev. 1) | Clean, but a second system: more code, and the old one never shrinks | ❌ expansion without improvement |
| **C. One importable package that the old code is moved into** | New code is written once with explicit arguments; legacy functions become thin shims importing from it, so the legacy side only shrinks | ✅ chosen |

Evidence that C is feasible now (checked on master, 2026-10-21): `compute_ic`, `decile_spread`, `permutation_baseline`
(`signal_eval/core.py`) reference no name outside `np`/`pd`, so they can be moved without behavior change; a t-statistic helper exists in
5 places (`tools/bracket_eval.py:day_clustered_t`, `research/edge_discovery/lib.py`, `research/indicator_strategy/screen.py`,
`research/strategy_discovery/grid.py`, `docs/research/scripts/entry_range_eval/discover_val.py`; the study copies stay as recorded);
`tools/bracket_eval.py:bracket_returns` is a pure function of a frame. The package name is not important (working name `lab/`);
the **absorption rule is** (§6).

## 1. Problem (measured on master, 2026-10-21)
| Fact | Number / source |
|---|---|
| Code in the code packages | ~41.5 k lines, 360 `.py`, 784 test functions; **30,168 lines run in shared namespaces** (`data model trainer evaluation signal_eval workflow discovery`) |
| Cost of one extension | the last three `feat` commits touched 8, 9, 10 files |
| Implicit dependencies tolerated | 104 allowlist entries (`tools/implicit_deps_allowlist.txt`) |
| Experiments closed | 85 failure rows; ~450 indicator trials from one survey |
| Planned, never tested | Fibonacci, FVG, order blocks, Ichimoku, liquidity pools (ROADMAP R1: no code) |
| Multi-step targets | one `forecast_horizon` (1) in the pipeline |
Diagnosis: the unit of work for every question is the windowed model dataset, even for questions a table and a few statistics can answer.

## 2. Rules this design is built on (quoted)
| Quote | Source | How applied |
|---|---|---|
| "adding a feature means *adding* code, not *rewriting* it." | xy `playbooks/12-architecture-rules.md` | Four registries; §5 extension test |
| "For **every** Should/Could feature, write one line: *which modules change, and how*." | xy 12, A3 | §5, re-run at each health check |
| "Don't add extension points anywhere else; speculative abstraction is also a cost." | xy 12, D1 | L0 builds the minimum; every deferred mechanism has an activation trigger (§8) |
| "Rules that live only in a document will drift ... **Enforce them in CI**" | xy 12, F | §8 |
| "Ratchet, don't block: ... **no new** violations get in, and shrink the baseline over time" | xy 11 | §6: legacy size and allowlist only decrease |
| "A big rewrite is a sign that small corrections were skipped." | xy 12, E3 | extraction is incremental, one function at a time |
| "Decisions are recorded with what would make them wrong" | xy 12, G | §11 |
| "أي مهمة تعدّل دالة ... تعديلًا غير تافه تبدأ باستخراج تلك الدالة ... في commit من نوع `move` لا يغيّر السلوك" and "شرط قبل أول استخراج: أرقام مرجعية (golden)" | this repo, `PHILOSOPHY.md` §2.1 | §6 extraction protocol |
| "نقل أو تعديل، لا الاثنان معًا" (rule 6); "الفحوص تفرض، والوثائق تقترح" (rule 7) | `PHILOSOPHY.md` §1 | commit types; checks over prose |
| "Add, don't modify ... recorded results are never edited" | `CLAUDE.md` | studies under `research/` are not rewritten; shims cover only code packages |
| "One component, one interface ... a new method = a new registered component (unknown names raise)" | `CLAUDE.md` | registries; the assistant model has `fit`/`score` |
| "No silent defaults: unknown config keys ... raise" | `CLAUDE.md` | strict configs |
| "Every variant tried counts toward multiple testing, including rejected ones." | `RESEARCH_RULES.md` §4 | the ledger counts every config; §7 gate uses the count |

## 3. Scope (MoSCoW)
| Class | Items |
|---|---|
| **Must** | `stats` kernel; panel table; signal registry (states with validity); outcome registry (multi-horizon, R units); screen with nulls and the "big difference" gate; ledger (the existing hypothesis registry, absorbed) |
| **Should** | confluence operators; instrument validation (§7); health check |
| **Could** | assistant model on survivors; funding/OI adapter; relative-target outcome variants; promotion of a survivor into the pipeline |
| **Won't (now)** | training NIG-TimeNet here; rewriting `data model trainer evaluation workflow`; new architecture experiments (E-rel-001, E-norm-* parked until L1); live trading |
Legacy packages accept `fix` commits and **extractions**; no feature work in the shared-namespace packages while this runs.

## 4. Structure
### 4.1 Layers (imports point down only)
```
core/schema (tier 0, stdlib only)
   ^
lab/stats        t-stats (day-clustered), IC, decile spread, permutation null          <- absorbs signal_eval/core, bracket_eval.day_clustered_t
lab/bars         the only door to market data (wraps tools/holdout_guard); UTC, closed bars
lab/signals/     registry + one file per family      lab/outcomes/  registry + one file per family
lab/table        joins states and outcomes into the panel (owns the table schema)
lab/screen       nulls, gate, verdict               lab/ledger   hypothesis log + test counter   <- absorbs signal_eval/registry
lab/cli          strict config -> run -> manifest     (L3: lab/confluence/ ; Could: lab/assist/)
```
`lab` imports `core`, third-party libraries, and from `tools/` only `holdout_guard` and `experiment_registry`. **`lab` never imports a
legacy package.** Legacy packages may import `lab` public names, only inside shim modules (§6). Studies may import `lab`.
Cross-module imports use the package root (`from lab.signals import get_signal`), never internals.

### 4.2 The signal contract: states with validity (fixes "signals never coincide on one candle")
A signal is a **state**, valid from a causal bar and expiring, not a 0/1 on the firing bar:
| Column (per coin, per decision bar `t`) | Meaning |
|---|---|
| `side` int8 | +1 / -1 / 0 as known at the **close** of `t` |
| `age` int16 | bars since `valid_from` (close of the confirming bar) |
| `lo`, `hi` float32 (nullable) | zone edges for zone signals (FVG, order block, Fibonacci 0.5-0.618, S/R) |
| `dist_atr` float32 (nullable) | signed distance of the close to the zone in ATR(14) |
`NaN`, never `0`, for insufficient history. Confluence is *computed*, not coincident: "FVG valid AND Fibonacci zone within 0.5 ATR AND cloud
bullish, all valid at the same decision bar", although they formed on different candles.
```python
@register_signal("fvg", family="structure", version=1)
def fvg(bars: Bars, *, min_gap_atr: float = 0.25, expire_bars: int = 48) -> StateFrame: ...
```
Unknown names/parameters raise. A new parameter's default reproduces old behavior; a changed definition is a **new version**.

### 4.3 The outcome contract: multi-step targets are columns
Entry at the next bar's open; horizons `k in {1,3,6,12,24}` decision bars: `ret_k`, `mfe_k`, `mae_k` (ATR and R), `first_touch_k` (stop-first on
ambiguity, as `bracket_returns`), `range_k` (the H003 target). Cost is a parameter in R (0.14 % round trip, also 2x). Variants are registered
(`xs_demean`, `mkt_resid`), so **E-rel-001's absolute-vs-relative question is answered here first, cheaply**.

### 4.4 Assistant model (Could)
Meta-labeling: a rule/confluence proposes the event; a small model scores "better than base rate?" on survivors' events only, with the capacity
controls of `RESEARCH_RULES.md` §2.1. Logistic/GBM first. (Attribution to López de Prado, *AFML*, from memory: verify before citing in a card.)

## 5. Extension test (before building)
| Future item | Modules that change | Result |
|---|---|---|
| New signal (Fibonacci, FVG, order block, Ichimoku, liquidity pool, fractal) | add one file in `signals/` + its fixture | ✅ |
| New data source (funding, OI) | add an adapter in `bars/` + signal files | ✅ |
| New horizon / multi-step target / outcome variant | add an entry in `outcomes/` or config | ✅ |
| New confluence operator, null, or statistic | add a file/entry in `confluence/` or `screen/` | ✅ |
| Different timeframe or universe | config | ✅ |
| New panel column | append to `lab/schema.py` | ✅ |
| Assistant model | add `lab/assist/` | ✅ |
| Promote a survivor into `data/` | add a feature + parity test; `lab` unchanged | ⚠️ 1 legacy module, justified |
| Rename/remove a panel column | schema + consumers | ❌ forbidden; add a column instead |

## 6. No patching: how every change improves the structure
**PR checklist (each answer written in the PR; "no" to any needs a one-line reason):**
1. Which structural problem does this remove, not just which symptom?
2. Does it add a registry entry / a file, instead of an `if` for one case or an edit in a shared function?
3. Does it copy a function that already exists? (If so: lift the original first.)
4. Does it cross a layer or reach into another module's internals?
5. Is there a test that would have caught the problem this fixes?

**One in, one out (the ratchet toward a smaller, clearer system):** each phase must absorb a named legacy piece, by the project's own protocol:
(1) golden numbers of the old function on fixed data, (2) `move` commit: same code into `lab/`, legacy file becomes a shim importing it, byte-identical
results, (3) only then `feat` commits that modify it. Legacy never loses a public name (notebooks and tests keep working).
Enforced: shared-namespace lines (30,168 today) and allowlist entries (104 today) may not increase at any phase (`check_deps.py`,
`implicit_deps.py --prune-allowlist`).
| Phase | Absorbs (candidate; purity verified before moving) |
|---|---|
| L0 | `signal_eval/core.py` (`compute_ic`, `decile_spread`, `permutation_baseline`) and `tools/bracket_eval.py:day_clustered_t` into `lab/stats` |
| L1 | `bracket_eval.bracket_returns` / `breakeven_hit_rate` into `lab/outcomes`; `signal_eval/registry.py` into `lab/ledger` (the test counter extends the existing registry, no second log) |
| L2-L3 | `signal_eval/windows.py:evaluate_windows` and the scan loop of `discovery/` where they can run on the panel; the rest is listed as retired or frozen at L3 (below) |

## 7. Screening protocol: "is the difference big enough to test?"
| Tier | What | Paperwork |
|---|---|---|
| 0 Screen | panel run on the 10-coin cloud data (or Drive coins on Colab) | one ledger row per config, automatic |
| 1 Candidate | passed the gate | one card per *family* (`tools/experiment_registry.py check`) |
| 2 Confirm | one frozen run on a sealed bucket via `holdout_guard.open_holdout` | ROADMAP R0 rules |
**Gate 0 -> 1 (the pre-registered R1 bar; all must hold):** n >= 300 events on >= 8 coins; gross mean R above the null's 95th percentile by >= 0.10 R;
net mean R > 0 at 1x cost; lift >= +0.05 R over the best member alone; same sign in both halves; survives BH-FDR q = 0.10 over **every config the
ledger counts for the family**; per-coin consistency and day-clustered t.
**Detectability pre-check:** `MDE ~ (z_a + z_b) * sigma / sqrt(n_eff)`, `n_eff` = independent day-clusters of events. Illustration: sigma ~ 1 R,
`n_eff` = 400 gives ~0.14 R; a family whose plausible effect is below its MDE is not run.
**Variants:** within a family, parameter values span a factor >= 2 (lookback 10/20/40), at most 3 per parameter; a different idea is a different family.
**Family budget:** the 13th config of a family is refused unless `--override-reason` is given (recorded). **Instrument validation (L1):** the screen must reproduce
a known negative (hourly `-RSI_14`: information but net <= 0 after 0.14 %, F-0036) and a known positive (`NATR_14` -> next-range width, H003) or it is not trusted.

## 8. Enforcement: minimal now, more only when triggered
| Rule | Check | When active |
|---|---|---|
| `lab` imports only `core`, third party, two named `tools` modules; never legacy | `tools/check_deps.py` new zone | L0 |
| legacy shared-namespace lines and allowlist never increase | `check_deps.py` / `implicit_deps.py` | L0 |
| each module has a card; maps current | `build_map.py --check` | L0 |
| every signal/outcome obeys the contract (causality by perturbation, NaN not 0, determinism, schema, unknown params raise, golden fixture) | generic registry test | L0 |
| strict config; reads only through the seal | tests | L0 |
| run manifest (git SHA, config, input fingerprint, **configs evaluated**, output sha256) | `lab/cli` | L1 |
| imports point down a declared layer order (`LAB_LAYERS`) | `check_deps.py` | **when `lab/` has > 8 modules** |
| panel schema append-only (`schema_baseline.json`) | `pytest` | **when a second consumer of the panel exists** |
| ledger tamper-evidence (hash chain) | `pytest` | **when more than one person writes the ledger** |
| size ratchet (function ~50, file ~300-400, new code only) | CI warning | **at L2** |
| `HEALTH.md` indicator table | reviewer agent | **at L2, then every 3 phases** |

## 9. Budget, retirement, success
**Complexity budget (proposed):** `lab/` <= **4,000 lines** and <= **16 modules** (tests and fixtures excluded). Crossing either stops the phase: the PR
must justify it in one paragraph, or something is absorbed/retired to get back under. (Rev. 1 estimated 3-4 k lines; this is a ceiling, not a target.)
**Retirement clause:** at the end of L3, every legacy screening path (`discovery/scanner`, `batch_runner`, `phase3_tools`, `signal_eval/windows`, the
`hypothesis_*` notebooks' code) gets one of three labels in `decisions/`: **absorbed** (shim), **retired** (archived, its results stay recorded), or **frozen**
(kept, with a written reason and a revisit date). "Frozen forever" is not allowed without the reason.
**Success indicators (measured, not argued):**
| Indicator | Baseline | Target | Alarm |
|---|---|---|---|
| Time from "new idea" to first screen result | measure once on the old path (adding FVG there) before L0 | falling at each phase | rising |
| Files touched to add one signal | 8-10 (recent `feat` commits) | <= 2 (signal + fixture) | any increase |
| Shared-namespace lines / allowlist entries | 30,168 / 104 | non-increasing; both lower by L3 | any increase |
| `lab/` lines and modules | 0 | within budget | over budget |
| Signals with fixture and passing contract | n/a | 100 % | < 100 % |

## 10. Phases (each a working increment, one branch, one PR)
| Phase | Delivers | Definition of done |
|---|---|---|
| **L0** thinnest slice | `lab/stats` (absorbed, golden-equal), `bars`, one trivial signal, one outcome, `table`, `cli`; `check_deps` zone | golden equality of absorbed functions; contract test green; one panel built on 10 coins at 1h; legacy lines lower than before |
| **L1** instrument | multi-horizon outcomes, cost in R, nulls, `ledger` (absorbed registry), gate, MDE pre-check, family budget, §7 validation | both validation cases reproduce; gate tests; the old-path timing baseline recorded |
| **L2** signal batch | Fibonacci, FVG, order block, Ichimoku, swing S/R, liquidity pool (the R1 detector list) | each with fixture; contract covers all; size ratchet and `HEALTH.md` switched on |
| **L3** confluence + first screens | operators + one screen per family under §7; retirement labels (§9) | candidate list produced mechanically; registry rows for every closed family |
| L4 (Could) | assistant on survivors | only if L3 has a candidate |
| L5 (Could) | promote one survivor into `data/` with a parity test | byte-equal states vs `lab` |
**Stop rule (proposed):** time-box L0-L3 to **6 weeks**. If no family passes the gate, close the families in the registry and stop searching for direction;
ship what survived (range tool, H07 trend filter). **Relation to ROADMAP:** R1/R1b (not started) are implemented on this kernel (card `E-conf-001`, same bar); R2
(funding) is independent and lands as a `bars` adapter.

## 11. Decision records (with what would make them wrong)
| ID | Decision | Revisit if |
|---|---|---|
| D-L01 | One importable package that **absorbs** legacy pieces (option C); not in-place edits, not a parallel lane | an absorbed function cannot keep byte-identical results |
| D-L02 | Flat panel (coin x close time), not windows | a survivor needs raw windows (the model lane builds them from the survivor's spec) |
| D-L03 | Signals are states with validity | a family needs richer state |
| D-L04 | Registries + generic contract tests | tests need per-signal logic a contract cannot express |
| D-L05 | `lab` never imports legacy; legacy imports `lab` only through shims | shims multiply faster than they are retired |
| D-L06 | No model in Tier 0 | Tier 0 finds nothing and the owner wants a learned screen |
| D-L07 | Raw data and panels outside the repo; manifests in the repo | a manifest cannot reproduce a run |
| D-L08 | 6-week box, 12-config family budget, 4,000-line / 16-module ceiling | the first screens show the numbers are too tight or too loose |
| D-L09 | Heavy enforcement is deferred behind triggers (§8) | a drift happens that a deferred check would have caught |

## Decisions (recorded in D-011: accepted in principle, dormant)
1. Absorbing kernel (option C) and the freeze of feature work in shared-namespace packages: **accepted, applies only after activation.**
2. Budget and box (4,000 lines / 16 modules / 6 weeks / 12 configs per family): **accepted as defaults, apply only after activation.**
3. Implement R1 on the kernel: **accepted, after activation.**
4. First data: 10 coins at 1h/4h in the cloud, confirmation later on a sealed Drive bucket: **accepted, after activation.**
5. Park E-rel-001 until L1: **accepted, after activation.** Until then E-rel-001 is simply not run (its own decision is still open).

*Not verified:* numbers in §0-§1 were measured from the repository on 2026-10-21; shim feasibility is verified only for the listed pure functions (others must be
checked before moving); the MDE figure is an illustration; the meta-labeling attribution is from memory; nothing in `lab/` exists yet.
