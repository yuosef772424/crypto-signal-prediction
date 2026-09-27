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
REPORTS = ("gap", "verification", "market_neutral", "candle", "chicks")


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
    ap.add_argument("--target-mode", default=None, help="TARGET_MODE في main (مثلاً relative). None = أهداف خط الأنابيب")
    ap.add_argument("--anti-memorization", default="true", choices=("true", "false"))
    ap.add_argument("--reports", default=",".join(REPORTS), help=f"من {REPORTS}")
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    if not a.train and not a.weights:
        ap.error("--weights أو --train")
    data = os.path.abspath(a.data)
    weights = os.path.abspath(a.weights) if a.weights else None
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.out)
    import matplotlib
    matplotlib.use("Agg")
    import builtins
    builtins.display = print
    ns = {"__name__": "__main__", "display": print}
    nb = json.load(open(os.path.join(REPO, "main.ipynb"), encoding="utf-8"))
    t0 = time.time()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = _cell_src(c)
        if re.search(r"^drive\.mount\(", src, re.M) or "def run_wiring_selftest" in src:
            continue
        if "run_full_analysis(" in src and "full_results" in src:   # القسم ٦ (chicks): اختياري، يُستدعى لاحقاً
            continue
        src = src.replace("TARGET_MODE = None", f"TARGET_MODE = {a.target_mode!r}")
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

    model, train, val, test, tf_ = ns["model"], ns["train"], ns["val"], ns["test"], ns["MODEL_TF"]
    want = set(a.reports.split(","))
    runs = [
        ("gap", "٧-ز فجوة التعميم", lambda: ns["generalization_gap_report"](model, train, val, test, model_tf=tf_)),
        ("verification", "٧-د التحقق المتكامل",
         lambda: ns["run_full_verification"](model, train, val, test, model_tf=tf_, out_dir="verification")),
        ("market_neutral", "٧-و المحفظة المحايدة للسوق",
         lambda: ns["market_neutral_report"](model, train, val, test, model_tf=tf_)),
        ("candle", "٧-ج مقابل شكل الشمعة", lambda: ns["candle_baseline_report"](model, train, val, test, model_tf=tf_)),
        ("chicks", "٦ chicks (tearsheet + الدلالة الإحصائية)",
         lambda: ns["run_full_analysis"](model=model, test_dict=ns["test_dict"], timeframes=[tf_],
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
