"""تجربة كاملة: بيانات اللوحة ← تدريب كل متغيّر/بذرة ← تنبؤ وتصدير إشارات ← مقارنة بالنموذج الحالي.

يستدعيها القسم ٧-ح في main.ipynb (PANEL_MODE=True) بمتغيّرات الدفتر نفسها: نفس train/val/test بعد retarget_splits،
نفس MODEL_TF، ونفس مُرمِّز build_model_fn(..., config=MODEL_OVERRIDES) — فالعيّنات والميزات والأهداف والتقسيم
الزمني متطابقة مع النموذج الحالي حرفياً.

المتغيّرات (VARIANTS):
  A     : انتباه عبر العملات، بلا حدّ IC            B     : نفس النموذج بلا انتباه (FFN لكل عملة فقط)، بلا حدّ IC
  A_ic  : انتباه + حدّ IC اليومي لـ close           B_ic  : بلا انتباه + حدّ IC
"""
import json
import os
import time

import numpy as np
import pandas as pd

from .data import format_checks, panel_split_from
from .report import compare, up_share_table

VARIANTS = {
    "A": {"model": {"cross_attention": True}, "train": {"lambda_ic": 0.0}},
    "A_ic": {"model": {"cross_attention": True}, "train": {"lambda_ic": 0.5}},
    "B": {"model": {"cross_attention": False}, "train": {"lambda_ic": 0.0}},
    "B_ic": {"model": {"cross_attention": False}, "train": {"lambda_ic": 0.5}},
}
DEFAULT_MODEL_CFG = dict(d_model=64, n_cross_layers=1, num_heads=4, dropout=0.25, attn_dropout=0.1, ff_mult=2,
                         class_head_hidden=32, head_hidden=64)


def train_cfg_from_main(main_config):
    """يأخذ من main_config (القسم ٥) ما يخصّ المُحسِّن والإيقاف المبكر — نفس قيم النموذج الحالي."""
    o = main_config.get("optimizer", {})
    es = main_config.get("callbacks", {}).get("early_stopping", {})
    out = {k: o[k] for k in ("lr_initial", "lr_min", "lr_warmup_epochs", "lr_schedule", "weight_decay") if k in o}
    if "clip_norm" in o:
        out["clip_norm"] = o["clip_norm"]
    if "use_ema" in o:
        out["ema"] = bool(o["use_ema"])
    if o.get("ema_window_epochs"):
        out["ema_window_epochs"] = o["ema_window_epochs"]
    if "patience" in es:
        out["patience"] = es["patience"]
    if main_config.get("run", {}).get("epochs"):
        out["epochs"] = main_config["run"]["epochs"]
    return out


def _count(vars_):
    return int(sum(np.prod(v.shape) for v in vars_))


def build_panel_splits(train, val, test, model_tf, train_assets=None, val_assets=None, subset=None, verbose=True):
    """PanelSplit لكل قسم + فحوص السلامة (ترفع خطأً إن فشل فحص جوهري)."""
    tr = panel_split_from(train, model_tf, train_assets, "train")
    va = panel_split_from(val, model_tf, val_assets, "val")
    te = panel_split_from(test, model_tf, None, "test")
    if subset:
        tr, va, te = (s.subset(**subset) for s in (tr, va, te))
    checks = [s.check() for s in (tr, va, te)]
    if verbose:
        print("🧩 بيانات اللوحة (يوم UTC = عيّنة):\n" + format_checks(checks), flush=True)
    bad = [c["split"] for c in checks if not c["ok"]]
    if bad:
        raise RuntimeError(f"فحص بيانات اللوحة فشل في {bad} — راجع الجدول أعلاه")
    return tr, va, te, checks


def run_panel_variant(tr, va, te, encoder_builder, seq_len, n_features, run_dir, variant="A_ic", seed=0,
                      model_cfg=None, train_cfg=None, verbose=True):
    """يدرّب (أو يستأنف) متغيّراً واحداً ويصدّر إشاراته. يُرجع (val_df, test_df, state, trainer)."""
    import tensorflow as tf
    from .model import build_panel_model, extract_encoder
    from .train import PanelTrainer, export_signals, robust_scales

    v = VARIANTS[variant]
    mcfg = {**DEFAULT_MODEL_CFG, **(model_cfg or {}), **v["model"]}
    tcfg = {**(train_cfg or {}), **v["train"], "seed": seed}
    tf.keras.utils.set_random_seed(seed)
    encoder = extract_encoder(encoder_builder())
    model = build_panel_model(encoder, seq_len, n_features, **mcfg)
    reg_scale = robust_scales(tr.yreg)
    n_enc = _count(encoder.trainable_variables)
    n_all = _count(model.trainable_variables)
    if verbose:
        print(f"\n🧠 متغيّر {variant} (بذرة {seed}) — معاملات: المُرمِّز {n_enc:,} | الانتباه/الرؤوس {n_all - n_enc:,} | "
              f"المجموع {n_all:,} — {mcfg} | مقياس الانحدار {np.round(reg_scale, 4).tolist()}", flush=True)
    fp = {"variant": variant, "model": mcfg, "n": [tr.n, va.n, te.n],
          "days": [str(tr.days[0]), str(tr.days[-1]), str(va.days[-1])]}
    trainer = PanelTrainer(model, tcfg, run_dir, reg_scale, seq_len, n_features, fingerprint=fp, verbose=verbose)
    t0 = time.time()
    state = trainer.fit(tr, va)
    trainer.load_best()
    out = {}
    for name, ps in (("val", va), ("test", te)):
        logit, mu = trainer.predict(ps)
        out[name] = export_signals(ps, logit, mu, name)
        out[name].to_csv(os.path.join(run_dir, f"signals_{name}.csv.gz"), index=False)
    meta = {"variant": variant, "seed": seed, "model_cfg": mcfg, "train_cfg": trainer.cfg,
            "reg_scale": reg_scale.tolist(), "best_epoch": state["best_epoch"], "best": state["best"],
            "epochs_run": state["epoch"], "params": {"encoder": n_enc, "total": n_all},
            "train_minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f"✅ {variant}/s{seed}: أفضل حقبة {state['best_epoch']} من {state['epoch']} — الإشارات ← {run_dir}", flush=True)
    return out["val"], out["test"], state, trainer


def _align_baseline(runs, base_names, verbose=True):
    """يقصر إشارات النموذج الحالي على صفوف (عملة، طابع) نفسها التي في إشارات اللوحة — المقارنة على نفس العيّنات
    دائماً (ضروري مع subset، وفحص سلامة بدونه: يجب أن تتطابق الصفوف كلها)."""
    panel = [n for n in runs if n not in base_names]
    if not panel or not base_names:
        return runs
    out = dict(runs)
    for b in base_names:
        parts = []
        for i, sp in enumerate(("val", "test")):
            ref = runs[panel[0]][i][["asset", "timestamp"]]
            df = runs[b][i]
            m = df.merge(ref.drop_duplicates(), on=["asset", "timestamp"], how="inner")
            if verbose:
                print(f"🔗 {b}/{sp}: {len(m):,} صفاً مطابقاً من {len(df):,} (اللوحة {len(ref):,})"
                      + ("" if len(m) == len(ref) else " ⚠️ صفوف اللوحة ليست كلها في إشارات المرجع"), flush=True)
            parts.append(m)
        out[b] = tuple(parts)
    return out


def run_panel_experiment(train, val, test, model_tf, encoder_builder, seq_len, n_features, run_root,
                         variants=("A_ic", "B_ic"), seeds=(0,), model_cfg=None, train_cfg=None,
                         train_assets=None, val_assets=None, subset=None, baseline=None, first_touch=None,
                         verbose=True):
    """baseline: {اسم: (val_df, test_df)} إشارات النموذج الحالي على نفس التقسيم (collect_signals) — اختياري.
    first_touch: جدول (asset, day, first) من bracket_fetch.py لحسم اللمس المزدوج في القوس — اختياري.
    يُرجع {"runs": {اسم: (val_df, test_df)}, "table": جدول المقارنة, "brackets": {...}, "checks": [...]}."""
    tr, va, te, checks = build_panel_splits(train, val, test, model_tf, train_assets, val_assets, subset, verbose)
    runs = dict(baseline or {})
    states = {}
    for variant in variants:
        for seed in seeds:
            run_dir = os.path.join(run_root, f"panel_{variant}_s{seed}")
            v, t, st, _ = run_panel_variant(tr, va, te, encoder_builder, seq_len, n_features, run_dir, variant, seed,
                                            model_cfg, train_cfg, verbose)
            runs[f"{variant}_s{seed}"] = (v, t)
            states[f"{variant}_s{seed}"] = {"best_epoch": st["best_epoch"], "epochs": st["epoch"]}
    runs = _align_baseline(runs, set(baseline or {}), verbose)
    table, brackets = compare(runs, first_touch=first_touch, verbose=False)
    # متوسط ± نصف المدى عبر البذور لكل متغيّر
    for variant in variants:
        cols = [f"{variant}_s{s}" for s in seeds]
        if len(cols) > 1:
            table[f"{variant} mean"] = table[cols].mean(axis=1)
            table[f"{variant} ±"] = (table[cols].max(axis=1) - table[cols].min(axis=1)) / 2
    ups = up_share_table(brackets)
    os.makedirs(run_root, exist_ok=True)
    table.to_csv(os.path.join(run_root, "panel_compare.csv"))
    ups.to_csv(os.path.join(run_root, "panel_up_share.csv"))
    if verbose:
        with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
            print("\n" + "═" * 100 + "\n📊 المقارنة على test (الاختيار على val فقط)\n" + "═" * 100)
            print(table.to_string())
            print("\nقوس ±5%: نسبة «+5% أولاً» من المحسومة لكل عُشر p_up_close (1 = الأدنى) — المطلوب تصاعد:")
            print(ups.to_string())
            print(f"\nالحقب: {states}\n💾 {os.path.join(run_root, 'panel_compare.csv')}")
    return {"runs": runs, "table": table, "brackets": brackets, "checks": checks, "states": states}
