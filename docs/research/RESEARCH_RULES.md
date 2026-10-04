# Research Rules — how every experiment in this project is run, closed, and never repeated by accident

> **بالعربية باختصار:** كل تجربة فاشلة تُغلَق بسجلّ يحدّد **مكان الفشل** و**سببه المُتحقَّق منه** و**الثوابت** التي تضمن
> تكراره، و**شروط إعادة الفتح** الصريحة (R1، R2...). أي تجربة جديدة يغطّيها فشل مُغلق تُرفض آلياً ما لم تُسمِّ شرط إعادة
> الفتح الذي تحقّق فعلاً، أو تثبت أن نطاقها مختلف. تغيير البيانات أو إضافة «شيء مختلف» لا يكفي وحده لإعادة تجربة: يجب أن
> يعالج التغيير **السبب المُتحقَّق** للفشل. والتحسين يأتي من دورة ثابتة: تحديد مكان الفشل ← فرضيات السبب ← اختبار السبب
> ← الإصلاح ← التحقق من أن الإصلاح حرّك ما تنبّأنا به. ليس مطلوباً اختبار كل الطبقات أو كل البيانات — المطلوب تحديد الموضع.

Scope: every experiment — features, targets, trading rules, data sources and models. Model-specific diagnostics live in
`MODEL_DEV_PROTOCOL.md`; evaluation integrity (val / test / sealed holdout, auditor independence) in `audit/PROTOCOL.md`.
Enforcement: `tools/experiment_registry.py` (run by `tests/test_experiment_registry.py` in CI).

## 1. The improvement cycle (the only accepted way to improve)

1. **Localize** the failure: name the level (data → representation → optimization → head → generalization → economics,
   or implementation) and the component, with the artifact that shows it. You do not need every layer or all the data —
   you need the *first* level that fails (`MODEL_DEV_PROTOCOL.md` §3).
2. **Hypothesize causes** for that location (at least two competing ones when possible), each with a cheap test that
   would distinguish them.
3. **Test the cause** before fixing anything. A cause is *verified* only when its test came out as predicted against a
   null (shuffled labels, random init, seed spread, a control subset).
4. **Fix** the verified cause — one change, on an experiment card.
5. **Validate the fix**: the diagnostic at the failure location must move as predicted **and** the outcome must improve
   beyond seed/sample noise. Otherwise the fix is rejected, even if the output looks better.

## 2. Closing a failure (mandatory, same commit as the result)

Every rejected or abandoned experiment adds one row to `docs/research/failure_registry.csv`:

| Column | Meaning |
|---|---|
| `id` | `F-0001`, append-only |
| `source` | study / hypothesis id and file (e.g. `research/edge_discovery H19`) |
| `claim` | what was tried, one line |
| `mechanism`, `timeframe`, `target`, `model_class` | **scope tags** (`;`-separated, `any` = all). They decide coverage — reuse existing spellings (`experiment_registry.py search`) |
| `universe`, `period`, `cost_model` | rest of the scope, for humans |
| `failure_level`, `failure_location` | where it failed (§1.1) |
| `evidence` | metric vs null, numbers, artifact link |
| `verified_cause`, `cause_test` | the cause and the test that verified it; if none was verified, status is `closed-unverified-cause` |
| `invariants` | the facts that make it fail regardless of incidental details (e.g. "gross edge < round-trip cost at every horizon ≤ 8h"; "direction AUC ≤ 0.53 for any representation of 1h OHLCV windows") |
| `reopen_if` | the **only** conditions that justify running it again, as `R1: … \| R2: …` — each must attack the verified cause or an invariant (e.g. `R1: maker round-trip ≤ 2 bp`, `R2: a new information source not derivable from OHLCV`) |
| `status` | `closed`, `closed-unverified-cause`, or `reopened` (a later row with `supersedes` replaces it) |

Rows are never edited. A reopened experiment that fails again adds a new row that `supersedes` the old one.

## 3. No repeat without a reason (enforced)

- Before writing a card: `python tools/experiment_registry.py search <words>`.
- Every experiment card (`docs/research/cards/*.md` or `research/<study>/cards/*.md`, template
  `templates/EXPERIMENT_CARD.md`) starts with front matter carrying the same scope tags. For **every** closed failure
  that covers it, the card must declare either
  - `reopens: F-0003:R1` — a `reopen_if` condition of that row that now holds (and the card shows it holds), or
  - `not_covered_by: F-0003=<which scope tag or invariant differs, concretely>`.
- `python tools/experiment_registry.py check <card>` must pass before the run; CI validates the registry and every card.
- "More data", "a newer period", "a bigger model" or "a different architecture" are **not** reopen reasons unless the
  row's `reopen_if` lists them — i.e. unless the verified cause says that is what was missing.

## 4. What counts as evidence

- Effect size **and** a null for every claim; seed spread for every model claim (≥ 3 seeds).
- Every variant tried counts toward multiple testing, including rejected ones.
- val selects, test compares, the sealed holdout is opened once at release.
- Correlation is not cause: "component X causes the failure" requires an intervention on X (ablate, freeze, replace)
  with the predicted result.

## 5. Reuse

Scripts used more than once go into `tools/` (or a model notebook section) with a card and a test; later sessions call
them instead of rebuilding (`CLAUDE.md`).
