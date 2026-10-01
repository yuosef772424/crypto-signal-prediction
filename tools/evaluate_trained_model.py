"""تقييم نموذج مُدرَّب من main.ipynb بلا Drive وبلا تدريب — نفس دفتر main حرفياً، بخلايا مُرقَّعة فقط.

    python tools/evaluate_trained_model.py --data preprocessing_output_latest.pkl.gz \
        --weights best.weights.h5 --target-mode relative --out eval_out

ما يحدث: تُنفَّذ خلايا main.ipynb بالترتيب (كل %run يحمّل دفتره كما في Colab)، مع أربعة ترقيعات فقط:
  1) خلية Drive/‏%cd تُتخطّى، و load_data_from_drive يقرأ --data محلياً.
  2) TARGET_MODE = --target-mode (نفس الوضع الذي دُرِّب عليه — اسم مجلد التدريب يحمله: ..._relative_am).
  3) RUN_MAIN_TRAINING = False، ثم تُحمَّل --weights في model (نفس MODEL_OVERRIDES ⇒ نفس المعمارية).
  4) اختبار التوصيل الذاتي (القسم ٨) يُتخطّى (لا علاقة له بالنموذج الحقيقي).
ثم تُستدعى تقارير القسم ٧: فجوة التعميم، التحقق المتكامل، المحفظة المحايدة للسوق، ومقارنة شكل الشمعة.
الإخراج: يُطبع كل شيء، ويُحفظ ما تحفظه التقارير نفسها في --out.

نموذج اللوحة عبر العملات (القسم ٧-ح، docs/research/panel_phase1.md) — نفس خلية Colab بـ PANEL_MODE=True:
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

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = ("gap", "verification", "market_neutral", "candle", "chicks", "signals")
# أوضاع الهدف المقبولة = TARGET_MODES في main القسم ٣-ب (+ "+relative" لكل وضع يقبله؛ entry_range لا يقبله) —
# tests/test_entry_range.py يثبّت التطابق، فوضع جديد في الدفتر بلا إضافته هنا يُفشل الاختبار لا التقييم بصمت.
_BASE_MODES = ("return", "return_close", "scaled", "magnitude", "volnorm")
TARGET_MODE_CHOICES = (_BASE_MODES + ("relative",) + tuple(f"{b}+relative" for b in _BASE_MODES)
                       + ("entry_range",))
# تعريف انحدار close في entry_range = ENTRY_CLOSE_REGS في main القسم ٣-ب (يثبّته tests/test_entry_range.py)
ENTRY_CLOSE_REG_CHOICES = ("abs_return", "range_pos")
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


def _named(ns, df, split):
    """val مدمج بلا أسماء عملات (asset="all"): تُستعاد من dataset بنفس أقنعة split_data، كما في market_neutral_report."""
    if (df["asset"] == "all").all():
        names = ns["split_asset_names"](ns["dataset"], split)
        if names is not None and len(names) == len(df):
            df = df.assign(asset=np.asarray(names))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="preprocessing_output_*.pkl.gz (نفس ما دُرِّب عليه النموذج)")
    ap.add_argument("--weights", default=None, help="best.weights.h5 من مجلد التدريب (أو --train لتدريب جديد)")
    ap.add_argument("--train", action="store_true", help="درّب هنا بخلية main نفسها بدل تحميل أوزان")
    ap.add_argument("--epochs", type=int, default=None, help="مع --train: سقف الحقب (افتراضياً قيمة main)")
    ap.add_argument("--batch-size", type=int, default=None, help="مع --train: حجم الدفعة (افتراضياً قيمة main)")
    ap.add_argument("--patience", type=int, default=None, help="مع --train: صبر الإيقاف المبكر")
    ap.add_argument("--run-dir", default="run", help="مع --train: مجلد نقاط الحفظ (نسبة لـ --out)")
    ap.add_argument("--split-dates", default=None,
                    help="train_end,val_end (مثلاً 2025-06-24,2025-11-21): تقسيم زمني صريح بدل نِسَب العيّنات")
    ap.add_argument("--target-mode", default=None, choices=TARGET_MODE_CHOICES, metavar="MODE",
                    help=f"TARGET_MODE في main، أحد {TARGET_MODE_CHOICES}. بلا الخيار = أهداف خط الأنابيب")
    ap.add_argument("--entry-close-reg", default=None, choices=ENTRY_CLOSE_REG_CHOICES, metavar="REG",
                    help=f"ENTRY_CLOSE_REG في main (مع --target-mode entry_range)، أحد {ENTRY_CLOSE_REG_CHOICES}. "
                         "بلا الخيار = abs_return")
    ap.add_argument("--group-freq", default=None,
                    help="عرض مجموعة الطوابع المقطعية (مثلاً 32h) لـ +relative (group_freq في retarget_splits) ولنموذج "
                         "اللوحة (day_ns) معاً — لبيانات لا تتطابق طوابع عملاتها. None = الطابع الدقيق / يوم UTC")
    ap.add_argument("--model-tfs", default=None,
                    help="فريمات مُدخل النموذج مفصولة بفواصل (مثلاً 1h,4h) = MODEL_TFS في main: فرع مُرمِّز لكل فريم، والبيانات "
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
    ap.add_argument("--panel-preset", default=None, help="PANEL_PRESET في القسم ٧-ح (مثلاً 1h_s8)")
    ap.add_argument("--panel-k-eval", default=None, help="مثلاً 5,10,20,all — مقاييس test حين يرى النموذج k عملة فقط")
    ap.add_argument("--panel-patience", type=int, default=None)
    a = ap.parse_args()
    if not a.train and not a.weights and not a.panel:
        ap.error("--weights أو --train أو --panel")
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
    ns = {"__name__": "__main__", "display": print}
    if a.model_tfs:
        ns["MODEL_TFS"] = [t.strip() for t in a.model_tfs.split(",") if t.strip()]     # يقرؤه القسم ٣ من main
    nb = json.load(open(os.path.join(REPO, "main.ipynb"), encoding="utf-8"))
    t0 = time.time()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = _cell_src(c)
        if re.search(r"^drive\.mount\(", src, re.M) or "def run_wiring_selftest" in src:
            continue
        if "PANEL_MODE = False" in src:
            if not a.panel:
                continue
            # ov: إعدادات الخلية قبل PANEL_PRESET؛ explicit: ما مُرِّر صراحةً في سطر الأوامر فيكتب فوق الإعداد الجاهز أيضاً
            ov = {"PANEL_RUN_ROOT": os.path.abspath("panel_runs"),
                  "PANEL_BASELINE": a.panel_baseline or ("model" if (a.train or weights) else None)}
            if ov["PANEL_BASELINE"] not in (None, "model"):
                ov["PANEL_BASELINE"] = os.path.abspath(os.path.join(INVOKE_CWD, ov["PANEL_BASELINE"]))
            explicit = {}
            if a.panel != "preset":
                explicit["PANEL_VARIANTS"] = a.panel.split(",")
            if a.panel_seeds or not a.panel_preset:
                explicit["PANEL_SEEDS"] = [int(x) for x in (a.panel_seeds or "0").split(",")]
            if a.panel_epochs:
                explicit["PANEL_EPOCHS"] = a.panel_epochs
            if a.panel_patience:
                explicit["PANEL_PATIENCE"] = a.panel_patience
            if a.panel_batch_samples:
                ov["PANEL_BATCH_SAMPLES"] = a.panel_batch_samples
            if a.panel_subset:
                d, c = a.panel_subset.split(",")
                ov["PANEL_SUBSET"] = {"last_days": int(d), "coins": int(c)}
            if a.panel_first_touch:
                ov["PANEL_FIRST_TOUCH"] = os.path.abspath(os.path.join(INVOKE_CWD, a.panel_first_touch))
            if a.panel_overrides:
                ov["PANEL_OVERRIDES"] = json.loads(a.panel_overrides)
            if a.group_freq:           # الخلية تمرّر day_ns (PANEL_GROUP) وtargets=PRICE_TARGETS بنفسها
                explicit["PANEL_GROUP"] = a.group_freq
            if a.panel_k_eval:
                explicit["PANEL_K_EVAL"] = tuple(None if x == "all" else int(x) for x in a.panel_k_eval.split(","))
            if a.panel_preset:
                ov["PANEL_PRESET"] = a.panel_preset
            if a.panel_baseline:
                explicit["PANEL_BASELINE"] = ov["PANEL_BASELINE"]
            src = src.replace("PANEL_MODE = False", "PANEL_MODE = True").replace(
                "# ── نهاية الإعدادات ──", f"globals().update({ {**ov, **explicit}!r})").replace(
                "    globals().update(PANEL_PRESETS[PANEL_PRESET])",
                f"    globals().update(PANEL_PRESETS[PANEL_PRESET])\n    globals().update({explicit!r})")
        if "run_full_analysis(" in src and "full_results" in src:   # القسم ٦ (chicks): اختياري، يُستدعى لاحقاً
            continue
        src = src.replace("TARGET_MODE = None", f"TARGET_MODE = {a.target_mode!r}")
        if a.entry_close_reg:
            src = src.replace('ENTRY_CLOSE_REG = "abs_return"', f"ENTRY_CLOSE_REG = {a.entry_close_reg!r}")
        if a.group_freq:
            src = src.replace("retarget_splits(train, val, test, mode=TARGET_MODE)",
                              f"retarget_splits(train, val, test, mode=TARGET_MODE, group_freq={a.group_freq!r})")
        if not a.train:
            src = src.replace("RUN_MAIN_TRAINING = True", "RUN_MAIN_TRAINING = False")
        else:
            src = src.replace('"/content/drive/MyDrive/training_runs/crypto_model_v1"', repr(os.path.abspath(a.run_dir)))
            src = src.replace("callbacks=callbacks, verbose=1,", "callbacks=callbacks, verbose=2,")
            if a.epochs:
                src = re.sub(r'"epochs": \d+,', f'"epochs": {a.epochs},', src, count=1)
            if a.patience:
                src = src.replace("RUN_MAIN_TRAINING = True",
                                  f"main_config['callbacks']['early_stopping']['patience'] = {a.patience}\n"
                                  "RUN_MAIN_TRAINING = True")
            if a.batch_size:
                src = re.sub(r'"batch_size": \d+,', f'"batch_size": {a.batch_size},', src, count=1)
        if a.split_dates and re.match(r"\s*train, val, test = split_data\(", src):
            tr_end, va_end = a.split_dates.split(",")
            ns["update_config"]({"split_dates": {"train_end": tr_end, "val_end": va_end}})
            print(f"📅 تقسيم زمني صريح: train ≤ {tr_end} | val ≤ {va_end} | test بعده (مع فجوة العزل)", flush=True)
        src = src.replace("ANTI_MEMORIZATION = True", f"ANTI_MEMORIZATION = {a.anti_memorization == 'true'}")
        src = src.replace("model.summary()", "")
        quiet = "%run" in src
        run_cell(src, ns, f"main#cell{i}", quiet=quiet)
        if "load_data_from_drive" in ns and not ns.get("_LOCAL_DATA_PATCHED"):
            ns["load_data_from_drive"] = lambda **kw: pickle.load(gzip.open(data, "rb"))
            ns["_LOCAL_DATA_PATCHED"] = True
        if "RUN_MAIN_TRAINING" in src and weights and not a.train and "model" in ns:
            ns["model"].load_weights(weights)
            print(f"✅ أُحمّلت الأوزان {os.path.basename(weights)} في النموذج "
                  f"({ns['model'].count_params():,} معاملاً) — TARGET_MODE={a.target_mode!r}", flush=True)
        if a.train and "RUN_MAIN_TRAINING" in src and "trainer" in ns:
            ns["model"] = ns["trainer"].model
            ns["model"].save_weights(os.path.abspath("trained.weights.h5"))
            print(f"✅ تدريب مكتمل — الأوزان (أفضل حقبة) ← {os.path.abspath('trained.weights.h5')}", flush=True)
        print(f"   main cell {i} ✓ ({time.time() - t0:.0f}s)", flush=True)

    model, train, val, test = ns["model"], ns["train"], ns["val"], ns["test"]
    tfs = ns["_tfs_of"]()
    tf_ = ns["MODEL_TF"] if len(tfs) == 1 else tfs       # فريم واحد: الاسم كما كان؛ أكثر: قائمة الفريمات لمُدخل النموذج
    want = set(a.reports.split(","))
    runs = [
        ("gap", "٧-ز فجوة التعميم", lambda: ns["generalization_gap_report"](model, train, val, test, model_tf=tf_)),
        ("verification", "٧-د التحقق المتكامل",
         lambda: ns["run_full_verification"](model, train, val, test, model_tf=tf_, out_dir="verification")),
        ("market_neutral", "٧-و المحفظة المحايدة للسوق",
         lambda: ns["market_neutral_report"](
             model, train, val, test, model_tf=tf_, quantiles=tuple(float(x) for x in a.mn_quantiles.split(",")),
             universe=(a.mn_universe if a.mn_universe in (None, "categories") else a.mn_universe.split(",")))),
        ("candle", "٧-ج مقابل شكل الشمعة", lambda: ns["candle_baseline_report"](model, train, val, test, model_tf=tf_)),
        ("signals", "تصدير الإشارات (val/test) للتحليل خارج الدفتر",
         lambda: [_named(ns, ns["collect_signals"](model, sp, tf_), name).assign(split=name)
                  .to_csv(f"signals_{name}.csv.gz", index=False) for name, sp in (("val", val), ("test", test))]),
        ("chicks", "٦ chicks (tearsheet + الدلالة الإحصائية)",
         lambda: ns["run_full_analysis"](model=model, test_dict=ns["test_dict"], timeframes=tfs,
                                         target_specs=ns["EVAL_TARGET_SPECS"], out_dir="analysis_outputs",
                                         pnl_market_neutral=ns["CHICKS_MARKET_NEUTRAL"])
         if ns.get("test_dict") is not None else print("⏭️ test_dict غير مدعوم لهذا الوضع")),
    ]
    for key, title, fn in runs:
        if key not in want:
            continue
        print("\n" + "#" * 100 + f"\n# {title}\n" + "#" * 100, flush=True)
        t = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — تقرير فاشل لا يوقف الباقي
            import traceback
            traceback.print_exc()
            print(f"❌ {title}: {e}")
        print(f"   ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
