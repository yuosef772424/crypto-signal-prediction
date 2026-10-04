# Model Development Protocol — every model explains why it works or fails

> **بالعربية باختصار:** لا تجربة بلا فرضية مكتوبة مسبقاً تتنبّأ *أين* سيتغيّر النموذج (أي طبقة/أي إشارة تشخيص)، ولا حكم
> «هذه البنية أفضل» إلا بدليل مزدوج: تحسّن يتجاوز ضجيج البذور **و**تحرّك إشارة التشخيص المتوقَّعة. كل تدريب ينتج تقرير
> صحّة النموذج طبقةً طبقة (لا مقاييس المخرجات وحدها)، وكل فشل يُحدَّد موضعه بإجراء ثابت. القواعد أدناه بالإنجليزية لأن
> النماذج تتبعها بدقة أكبر؛ تُترجم عند الطلب.

Companion to `docs/research/audit/PROTOCOL.md` (auditor ↔ builder, evaluation integrity) and
`research/edge_discovery/00_PREREGISTRATION.md` (pre-registered trading tests). This file governs **model architecture
and training experiments**.

## 1. No experiment without a card (written before running)

Each experiment gets a card (template: `docs/research/templates/EXPERIMENT_CARD.md`) committed **before** the run:

1. **One change** against a pinned baseline (same dataset fingerprint, splits, holdout, seeds, budget). Two changes =
   two experiments.
2. **Mechanism**: why the change should help, stated as a prediction about a *diagnostic*, not only the output metric —
   e.g. "the probe AUC of block 3 rises above block 2", "head-gradient imbalance on the trunk falls below 3×",
   "hard-to-learn share falls", "val→test AUC drop shrinks".
3. **Expected effect size** and the **accept / reject rule** on val (test only compares accepted variants; the sealed
   holdout is opened once at release).
4. Seeds ≥ 3 for any architectural claim; compute budget.

## 2. Every training run produces the model health report

A run is not finished until these artifacts are saved next to the checkpoint (JSON/CSV, with the config and the dataset
fingerprint):

| Layer of evidence | What | Tool |
|---|---|---|
| Data | label balance, normalization audit (level features, near-constant, extremes, train→test shift) | `normalization_audit`, `label_balance_report` |
| Optimization | per-head loss and **per-head gradient norm on the shared trunk**, per-layer gradient norms, update/weight ratio | training recorder callback (`model_diagnostics.md`) |
| Representation | **linear-probe skill per layer** (where predictive information appears or is lost) | per-layer report |
| Errors | per layer: how activations of **correctly vs wrongly predicted** samples differ, vs a null | correct-vs-wrong report |
| Data difficulty | dataset cartography: easy / ambiguous / hard-to-learn samples; most helpful / harmful training batches | recorder (cartography, TracIn-style influence) |
| Outputs | AUC / IC per head, calibration, NIG floors hit, seed spread | existing evaluation |
| Generalization | train/val/test gap; permutation (shuffled-label) control | `generalization_gap_report`, permutation control |

## 3. Locating a failure (fixed order — stop at the first level that fails)

1. **Data has no signal?** A linear/logistic baseline and the probe on the *input* layer are at chance on val →
   architecture work is pointless; go back to features/targets. (The edge-discovery study rejected most direction
   hypotheses at exactly this level.)
2. **Representation loses it?** Probe skill peaks at layer *k* and drops after → the later layers or the readout destroy
   information; change those, not the input.
3. **Optimization?** One head dominates trunk gradients, vanishing/exploding layer norms, update ratio ≪ 1e-4 or ≫ 1e-2.
4. **Head / calibration?** Trunk probes are good but the head is not, or NIG floors are hit, or predicted class share ≠
   label share.
5. **Generalization?** Train ≫ val, the shuffled-label control also fits, val→test drop beyond seed noise →
   memorization; regularize or reduce capacity.
6. **Economics?** Skill exists but net of costs is ≤ 0 → trading rule / sizing, not the model.

Each verdict names the level, the layer, and the artifact that proves it.

## 4. Evidence rules

- Every diagnostic difference is reported with an **effect size and a null**: the same statistic on shuffled labels, on a
  randomly initialised copy of the model, or across seeds. Flag only what exceeds the null.
- Diagnostics are **correlational**. A claim that "layer X causes the failure" needs an intervention: ablate / freeze /
  replace that component and show the predicted change.
- Every variant tried counts toward multiple testing (rejected ones included).
- "Architecture A is better" is accepted only if (a) it beats the baseline beyond seed noise on val, **and** (b) the
  diagnostic predicted on its card moved. Better outputs with an unexplained mechanism are recorded as *unexplained*,
  not adopted.

## 5. Record

One row per experiment in the study's log (`hypothesis_log.csv` style): card link, predicted diagnostic movement,
observed movement, verdict (accepted / rejected / unexplained), failure level and layer, artifact links. Rejected and
unexplained results are kept; recorded results are never edited (add a corrected copy).

## 6. Reuse, don't rebuild

Any script used more than once goes into a code package (`tools/` or a model notebook section) with a module card, a
test, and a line in its doc page, so a later session calls it instead of rewriting it. Current reusable entry points:
`tools/nb_cells.py` (edit notebook cells without reformatting), `tools/fetch_crypto_dataset.py` (research OHLCV from
GitHub when Binance is blocked), `tools/bracket_eval.py` + pipeline `entry_feature_table` (symmetric stop/target
evaluation on all samples and on filtered subsets, with a random-direction null), `tools/experiment_registry.py`
(failure registry), the diagnostics toolkit (`docs/research/model_diagnostics.md`).
