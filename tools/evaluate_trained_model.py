"""
PURPOSE:  Evaluates a model trained in main.ipynb without Drive or training: builds a RunSettings from the command line and calls the
          workflow/run.py steps (no notebook text is patched); also runs the panel model (--panel).
TAGS:     evaluate trained model, eval CLI, RunSettings, workflow.run steps, weights, target-mode, split-dates, panel
          evaluation, gap report, market-neutral portfolio
PITFALLS: --target-mode must match the mode the weights were trained with (the training folder name carries it: ..._relative_am).
          The four runner notebooks (pipeline, model_v2, trainer, chicks) are executed like Colab's %run (Drive/%cd cells skipped), so a
          run pays their self-tests; main.ipynb itself is never read. Reports run in the current directory = --out.

تقييم نموذج مُدرَّب من main.ipynb بلا Drive وبلا تدريب — بنفس خطوات main (دوال workflow/run.py) لا بنسخة من نصّ خلاياه.

    python tools/evaluate_trained_model.py --data preprocessing_output_latest.pkl.gz \
        --weights best.weights.h5 --target-mode relative --out eval_out

ما يحدث: تُنفَّذ الدفاتر الأربعة (خط الأنابيب، model_v2، المدرّب، chicks) كما يفعل %run في Colab (تُتخطّى خلايا Drive/‏%cd)، وتُحمَّل
الحزمة workflow/، ثم تُبنى RunSettings من سطر الأوامر وتُستدعى الخطوات:
  1) --data يُقرأ محلياً (بدل Drive) ويُمرَّر إلى run_main؛ --split-dates ← data.split_dates؛ --model-tfs ← data.model_tfs.
  2) --target-mode / --entry-close-reg / --group-freq ← target.* (نفس الوضع الذي دُرِّب عليه — اسم مجلد التدريب يحمله: ..._relative_am).
  3) train.run_main_training = --train؛ وبدونه تُحمَّل --weights في model (نفس plan.model_overrides ⇒ نفس المعمارية).
ثم تُستدعى تقارير القسم ٧ (run_reports): فجوة التعميم، التحقق المتكامل، المحفظة المحايدة للسوق، ومقارنة شكل الشمعة، والإشارات، وchicks.
الإخراج: يُطبع كل شيء، ويُحفظ ما تحفظه التقارير نفسها في --out.

نموذج اللوحة عبر العملات (القسم ٧-ح، docs/research/panel_phase1.md) — نفس خطوة Colab بـ panel.enabled=True:
    python tools/evaluate_trained_model.py --data preprocessing_output_latest.pkl.gz --target-mode relative \
        --split-dates 2025-06-24,2025-11-21 --panel A_ic,B_ic --panel-baseline eval_timesplit_sig --out eval_panel
    (--panel-subset 60,40 --panel-epochs 2 لاختبار سريع؛ مع --weights/--train يُقارَن بالنموذج المحمَّل نفسه)
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import pickle
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)    # core/ (المصدر الوحيد لأسماء أوضاع الهدف) بجانب tools/ — السكربت يُشغَّل من tools/
from core.schema import ENTRY_CLOSE_REGS, NO_RELATIVE_BASES, TARGET_MODES  # noqa: E402

REPORTS = ("gap", "verification", "market_neutral", "candle", "chicks", "signals")   # = workflow.run.REPORTS (tests/test_run_tool.py)
#: The four runner notebooks main.ipynb %runs, in its order; the tool executes them into one namespace like Colab does.
RUNNER_NOTEBOOKS = ("crypto_data_pipeline_v6.ipynb", "model_v2 (1).ipynb", "trainer_framework_v2.ipynb",
                    "chicks_v4_5_input_output_patterns.ipynb")
# أوضاع الهدف المقبولة = TARGET_MODES في core/schema.py (+ "+relative" لكل وضع يقبله؛ NO_RELATIVE_BASES لا تقبله) —
# tests/test_entry_range.py يثبّت التطابق مع workflow/retarget.py.
_BASE_MODES = tuple(m for m in TARGET_MODES if m not in NO_RELATIVE_BASES)
TARGET_MODE_CHOICES = (_BASE_MODES + ("relative",) + tuple(f"{b}+relative" for b in _BASE_MODES)
                       + tuple(NO_RELATIVE_BASES))
# تعريف انحدار close في entry_range = ENTRY_CLOSE_REGS في core/schema.py
ENTRY_CLOSE_REG_CHOICES = ENTRY_CLOSE_REGS
INVOKE_CWD = os.getcwd()


def _cell_src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def load_notebook(path, ns, quiet=True):
    """ينفّذ خلايا الكود في ns؛ %run داخلها يحمّل الدفتر المشار إليه، وأسطر ‎%/! الأخرى تُتخطّى."""
    nb = json.load(open(path, encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = _cell_src(c)
        if re.search(r"^(drive\.mount\(|%cd)", src, re.M):   # مستوى أعلى فقط: mount_drive() تستدعيه داخل دالة
            continue
        run_cell(src, ns, f"{os.path.basename(path)}#cell{i}", quiet)


def run_cell(src, ns, name, quiet=False):
    lines = []
    for line in src.splitlines():
        m = re.match(r'^\s*%run\s+["\']?(.+?\.ipynb)["\']?\s*$', line)
        if m:
            load_notebook(os.path.join(REPO, m.group(1)), ns)
            continue
        if line.lstrip().startswith(("%", "!")):
            continue
        lines.append(line)
    code = compile("\n".join(lines), name, "exec")
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        exec(code, ns)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="preprocessing_output_*.pkl.gz (نفس ما دُرِّب عليه النموذج)")
    ap.add_argument("--weights", default=None, help="best.weights.h5 من مجلد التدريب (أو --train لتدريب جديد)")
    ap.add_argument("--train", action="store_true", help="درّب هنا بخطوات main نفسها بدل تحميل أوزان")
    ap.add_argument("--epochs", type=int, default=None, help="مع --train: سقف الحقب (افتراضياً قيمة main)")
    ap.add_argument("--batch-size", type=int, default=None, help="مع --train: حجم الدفعة (افتراضياً قيمة main)")
    ap.add_argument("--patience", type=int, default=None, help="مع --train: صبر الإيقاف المبكر")
    ap.add_argument("--run-dir", default="run", help="مع --train: مجلد نقاط الحفظ (نسبة لـ --out)")
    ap.add_argument("--split-dates", default=None,
                    help="train_end,val_end (مثلاً 2025-06-24,2025-11-21): تقسيم زمني صريح بدل نِسَب العيّنات")
    ap.add_argument("--target-mode", default=None, choices=TARGET_MODE_CHOICES, metavar="MODE",
                    help=f"target.target_mode في main، أحد {TARGET_MODE_CHOICES}. بلا الخيار = أهداف خط الأنابيب")
    ap.add_argument("--entry-close-reg", default=None, choices=ENTRY_CLOSE_REG_CHOICES, metavar="REG",
                    help=f"target.entry_close_reg في main (مع --target-mode entry_range)، أحد {ENTRY_CLOSE_REG_CHOICES}. "
                         "بلا الخيار = abs_return")
    ap.add_argument("--group-freq", default=None,
                    help="عرض مجموعة الطوابع المقطعية (مثلاً 32h) لـ +relative (group_freq في retarget_splits) ولنموذج "
                         "اللوحة (day_ns) معاً — لبيانات لا تتطابق طوابع عملاتها. None = الطابع الدقيق / يوم UTC")
    ap.add_argument("--model-tfs", default=None,
                    help="فريمات مُدخل النموذج مفصولة بفواصل (مثلاً 1h,4h) = data.model_tfs في main: فرع مُرمِّز لكل فريم، والبيانات "
                         "يجب أن تحويها (build_hourly_4h_dataset). بلا الخيار = فريم واحد كما كان")
    ap.add_argument("--anti-memorization", default="true", choices=("true", "false"))
    ap.add_argument("--reports", default=",".join(REPORTS), help=f"من {REPORTS}")
    ap.add_argument("--mn-quantiles", default="0.05,0.1,0.2,0.3,0.5",
                    help="أطراف المحفظة المحايدة (0.5 = الكون كله لأوزان الرتب)")
    ap.add_argument("--mn-universe", default=None, help="None = كل العملات | categories | قائمة رموز مفصولة بفواصل")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--panel", default=None, help="متغيّرات نموذج اللوحة مفصولة بفواصل (A,A_ic,B,B_ic,A_ic_k) — يشغّل "
                                                  "القسم ٧-ح؛ preset = متغيّرات --panel-preset")
    ap.add_argument("--panel-seeds", default=None, help="بذور مفصولة بفواصل (افتراضياً 0، أو بذور --panel-preset)")
    ap.add_argument("--panel-epochs", type=int, default=None, help="None = حقب main (أو --epochs)")
    ap.add_argument("--panel-batch-samples", type=int, default=None)
    ap.add_argument("--panel-subset", default=None, help="last_days,coins — اختبار سريع على مجموعة فرعية")
    ap.add_argument("--panel-baseline", default=None,
                    help="model | مسار best.weights.h5 | مجلد signals_{val,test}.csv.gz (الافتراضي: model مع --weights/--train)")
    ap.add_argument("--panel-first-touch", default=None, help="bracket_first_5.pkl لحسم اللمس المزدوج في القوس")
    ap.add_argument("--panel-overrides", default=None, help='JSON: {"train": {...}, "model": {...}}')
    ap.add_argument("--panel-preset", default=None, help="panel.preset في القسم ٧-ح (مثلاً 1h_s8)")
    ap.add_argument("--panel-k-eval", default=None, help="مثلاً 5,10,20,all — مقاييس test حين يرى النموذج k عملة فقط")
    ap.add_argument("--panel-patience", type=int, default=None)
    return ap


def build_settings(a, settings_cls, invoke_cwd=None):
    """RunSettings (``settings_cls`` = the workflow RunSettings class) of the parsed command line ``a``, as the old tool's text patches
    expressed them: returns ``(settings, panel_explicit)`` where ``panel_explicit`` are the panel fields given on the command line (they
    win over a panel preset, so the caller applies the preset first and these after). Paths are absolutised against the invoking cwd
    (``--out`` is entered afterwards)."""
    invoke_cwd = invoke_cwd or INVOKE_CWD
    upd = {"target": {"target_mode": a.target_mode, "group_freq": a.group_freq},
           "model": {"anti_memorization": a.anti_memorization == "true"},
           # chicks: the tool never calibrated confidence (no val_dict), unlike main's section 6 cell
           "evaluation": {"calibrate_confidence": False,
                          "mn_quantiles": tuple(float(x) for x in a.mn_quantiles.split(",")),
                          "mn_universe": (a.mn_universe if a.mn_universe in (None, "categories") else tuple(a.mn_universe.split(",")))},
           "data": {}, "train": {"run_main_training": bool(a.train)}}
    if a.entry_close_reg:
        upd["target"]["entry_close_reg"] = a.entry_close_reg
    if a.model_tfs:
        upd["data"]["model_tfs"] = tuple(t.strip() for t in a.model_tfs.split(",") if t.strip())
    if a.split_dates:
        tr_end, va_end = a.split_dates.split(",")
        upd["data"]["split_dates"] = {"train_end": tr_end, "val_end": va_end}
    if a.train:
        upd["train"].update(run_dir=os.path.abspath(a.run_dir), fit_verbose=2)
        for key, val in (("epochs", a.epochs), ("batch_size", a.batch_size), ("early_stopping_patience", a.patience)):
            if val:
                upd["train"][key] = val
    explicit = {}
    if a.panel:
        weights = a.weights
        base = a.panel_baseline or ("model" if (a.train or weights) else None)
        if base not in (None, "model"):
            base = os.path.abspath(os.path.join(invoke_cwd, base))
        panel = {"enabled": True, "run_root": os.path.abspath("panel_runs"), "baseline": base}
        if a.panel_batch_samples:
            panel["batch_samples"] = a.panel_batch_samples
        if a.panel_subset:
            d, c = a.panel_subset.split(",")
            panel["subset"] = {"last_days": int(d), "coins": int(c)}
        if a.panel_first_touch:
            panel["first_touch"] = os.path.abspath(os.path.join(invoke_cwd, a.panel_first_touch))
        if a.panel_overrides:
            panel["overrides"] = json.loads(a.panel_overrides)
        if a.panel_preset:
            panel["preset"] = a.panel_preset
        # ما مُرِّر صراحةً في سطر الأوامر يكتب فوق الإعداد الجاهز أيضاً (preset ثم هذه)
        if a.panel != "preset":
            explicit["variants"] = tuple(a.panel.split(","))
        if a.panel_seeds or not a.panel_preset:
            explicit["seeds"] = tuple(int(x) for x in (a.panel_seeds or "0").split(","))
        if a.panel_epochs:
            explicit["epochs"] = a.panel_epochs
        if a.panel_patience:
            explicit["patience"] = a.panel_patience
        if a.group_freq:           # المجموعة المقطعية للّوحة (day_ns): نفس --group-freq
            explicit["group"] = a.group_freq
        if a.panel_k_eval:
            explicit["k_eval"] = tuple(None if x == "all" else int(x) for x in a.panel_k_eval.split(","))
        if a.panel_baseline:
            explicit["baseline"] = base
        upd["panel"] = panel
    return settings_cls().updated(upd), explicit


def _stage(label, t0):
    print(f"   {label} ✓ ({time.time() - t0:.0f}s)", flush=True)


def main():
    a = build_parser().parse_args()
    if not a.train and not a.weights and not a.panel:
        build_parser().error("--weights أو --train أو --panel")
    if a.panel and not (a.train or a.weights) and a.reports == ",".join(REPORTS):
        a.reports = ""          # لا نموذج حالي مدرَّب ⇒ تقارير القسم ٧ بلا معنى
    sys.path.insert(0, REPO)    # حزمة cross_asset بجانب الدفاتر (os.chdir أدناه يُخرجنا من المستودع)
    global INVOKE_CWD
    INVOKE_CWD = os.getcwd()
    data = os.path.abspath(a.data)
    weights = os.path.abspath(a.weights) if a.weights else None
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.out)
    import matplotlib
    matplotlib.use("Agg")
    import builtins
    builtins.display = print
    ns = {"__name__": "__main__", "display": print,
          "__file__": os.path.join(REPO, "main.ipynb")}     # الدفاتر المُشغِّلة (data/، model/...) تجد جذر المستودع منه رغم os.chdir أعلاه
    t0 = time.time()
    for nb in RUNNER_NOTEBOOKS:                               # ما يفعله %run في main.ipynb (القسم ١)
        load_notebook(os.path.join(REPO, nb), ns)
    import workflow
    workflow.load_into(ns)
    _stage("الدفاتر الأربعة + workflow", t0)
    settings, panel_explicit = build_settings(a, ns["RunSettings"])
    kit = ns["Toolkit"].from_namespace(ns)

    with gzip.open(data, "rb") as f:                          # بدل load_data_from_drive
        dataset = pickle.load(f)
    t0 = time.time()
    res = ns["run_main"](settings, kit, dataset=dataset, summary=False)
    model = res.model
    _stage("بيانات ← تقسيم ← أهداف ← نموذج ← إعداد التدريب ← chicks", t0)
    if weights and not a.train:
        model.load_weights(weights)
        print(f"✅ أُحمّلت الأوزان {os.path.basename(weights)} في النموذج "
              f"({model.count_params():,} معاملاً) — TARGET_MODE={a.target_mode!r}", flush=True)
    if a.train:
        model.save_weights(os.path.abspath("trained.weights.h5"))
        print(f"✅ تدريب مكتمل — الأوزان (أفضل حقبة) ← {os.path.abspath('trained.weights.h5')}", flush=True)

    if a.panel:
        # preset ثم ما مُرِّر صراحةً في سطر الأوامر (الأخير يغلب)
        import dataclasses
        p = settings.panel.applied().updated(**panel_explicit)
        settings = dataclasses.replace(settings, panel=p)
        ns["run_panel"](settings, kit, dataset=dataset, train=res.train, val=res.val, test=res.test, info=res.info, plan=res.plan,
                        model=model, model_builder=res.model_builder, main_config=res.main_config, apply_preset=False)
        _stage("نموذج اللوحة", t0)

    ns["run_reports"](settings, [r for r in a.reports.split(",") if r], kit, model=model, train=res.train, val=res.val,
                      test=res.test, dataset=dataset, info=res.info, plan=res.plan, chicks=res.chicks)


if __name__ == "__main__":
    sys.exit(main())
