"""اختبارات ذاتية سريعة لنموذج اللوحة على بيانات تركيبية (ثوانٍ، بلا Drive).

    python -m cross_asset.selftest          # أو: python -m unittest tests.test_cross_asset
    run_panel_selftest(encoder_builder)     # من دفتر main: نفس الاختبارات بالمُرمِّز الحقيقي

ما يُثبَت:
  ١) التجميع: كل عيّنة في يوم واحد مرّة واحدة؛ لا يختلط يومان؛ الطوابع غير المحاذاة لـ 00:00 UTC والتكرار يُكشفان.
  ٢) الدفعات: كل عيّنة مرّة واحدة في كل حقبة؛ خلط الأيام حتمي من (seed, epoch)؛ val/test بترتيب التاريخ.
  ٣) القناع: مخرجات يومٍ لا تتغيّر بتغيّر الأيام الأخرى في الدفعة أو بالحشو؛ مع الانتباه تتغيّر بتغيّر عملة أخرى من
     اليوم نفسه، وبدونه (المتغيّر B) لا تتغيّر أبداً.
  ٤) حدّ الارتباط اليومي = Pearson المحسوب بـ numpy لكل يوم، والأيام الصغيرة مستبعدة.
  ٥) تدريب قصير: خسارة منتهية تنخفض، استئناف من run_dir، تصدير بأعمدة collect_signals، وتقرير المقارنة يعمل.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from .data import DAY_NS, PanelSplit

BASELINE_COLS = ["asset", "timestamp", "entry", "last_high", "last_low", "fut_close", "fut_high", "fut_low",
                 "mu_high", "p_up_high", "mu_low", "p_up_low", "mu_close", "p_up_close", "pred_high", "pred_low",
                 "up", "split"]


def synthetic_split(n_assets=30, n_days=60, seq_len=8, n_features=5, start_day=18000, seed=0, signal=0.0,
                    name="syn", shuffle_rows=True, day_shift=0.0):
    """عملات بتواريخ بداية مختلفة (عدد عملات اليوم يتغيّر). signal>0: عائد الغد يعتمد على آخر قيمة للميزة 0.
    day_shift>0: تُزاح الميزة 0 لكل عملات اليوم بمقدار عشوائي مشترك **بعد** حساب العائد — فالمعلومة في انحراف العملة
    عن عملات يومها لا في قيمتها المطلقة: نموذج لكل عملة وحدها لا يعرف الإزاحة، والانتباه عبر العملات يعرفها."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        first = int(rng.integers(0, n_days // 2)) if a >= 3 else 0     # ثلاث عملات تغطي كل الأيام
        for d in range(first, n_days):
            rows.append((f"C{a:03d}USDT", start_day + d))
    rows = pd.DataFrame(rows, columns=["asset", "day"])
    if shuffle_rows:
        rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(rows)
    X = rng.normal(size=(n, seq_len, n_features)).astype("float32")
    mkt = rng.normal(scale=0.02, size=n_days + start_day + 1)[rows["day"].to_numpy()]
    r = mkt + signal * X[:, -1, 0] * 0.02 + rng.normal(scale=0.03, size=n)
    if day_shift:
        # على آخر خطوة فقط: لو أُزيحت النافذة كلها لقدّرت كل عملة الإزاحة من خطواتها السابقة وحدها
        X[:, -1, 0] += rng.normal(scale=day_shift, size=n_days + start_day + 1)[rows["day"].to_numpy()]
    close = 100 * np.exp(rng.normal(size=n))
    rh, rl = np.abs(rng.normal(scale=0.02, size=n)), np.abs(rng.normal(scale=0.02, size=n))
    lc = np.stack([close * (1 + rh), close * (1 - rl), close, rows["day"].to_numpy().astype("float64") * DAY_NS,
                   close * (1 + r), close * (1 + np.minimum(r, 0) - rl), close * (1 + np.maximum(r, 0) + rh)], 1)
    ts = lc[:, 3]
    y = {}
    for t, fut, last in (("close", 4, 2), ("low", 5, 1), ("high", 6, 0)):
        raw = lc[:, fut] / lc[:, last] - 1
        med = pd.Series(raw).groupby(ts).transform("median").to_numpy()
        y[f"y_{t}_reg"] = np.clip(raw - med, -1, 1).astype("float32")
        y[f"y_{t}_class"] = np.where(raw > med, 1.0, -1.0).astype("float32")
    return PanelSplit(X, y, lc, rows["asset"].to_numpy(), name)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_grouping():
    ps = synthetic_split()
    c = ps.check()
    _check(c["ok"] and c["each_sample_once"] and c["day_groups_pure"], f"فحوص التجميع: {c}")
    _check(c["misaligned_timestamps"] == 0 and c["duplicate_asset_day"] == 0, "لا شذوذ متوقَّع")
    _check(ps.sizes.min() >= 3 and ps.sizes.max() == 30, f"أحجام الأيام {ps.sizes.min()}..{ps.sizes.max()}")
    for d in range(ps.n_days):
        ii = ps.day_indices(d)
        _check(len(set(ps.assets[ii])) == len(ii), "عملة مكرّرة داخل يوم")
        _check((ps.ts[ii] == ps.days[d]).all(), "يوم مختلط")
    # yrank: متمركزة لكل يوم، ورتيبة مع عائد close النسبي
    for d in range(ps.n_days):
        ii = ps.day_indices(d)
        _check(abs(ps.yrank[ii].mean()) < 1e-5, "yrank غير متمركزة")
        _check(pd.Series(ps.yrank[ii]).corr(pd.Series(ps.yreg[ii, 2]), method="spearman") > 0.999, "yrank ليست رتبة")
    # طابع مُزاح ساعة + تكرار (عملة، يوم) يُكشفان
    lc = ps.lc.copy()
    lc[0, 3] += 3600 * 10**9
    assets = ps.assets.copy()
    j = next(i for i in range(1, ps.n) if ps.day_key[i] == ps.day_key[0] and assets[i] != assets[0])
    assets[j] = assets[0]
    y = {f"y_{t}_{k}": (ps.ycls[:, i] * 2 - 1 if k == "class" else ps.yreg[:, i])
         for i, t in enumerate(ps.targets) for k in ("class", "reg")}
    bad = PanelSplit(ps.X, y, lc, assets, "bad").check()
    _check(bad["misaligned_timestamps"] == 1 and bad["misaligned_assets"] == [ps.assets[0]], f"الإزاحة لم تُكشف {bad}")
    _check(bad["duplicate_asset_day"] >= 1 and not bad["ok"], "التكرار لم يُكشف")
    return True


def test_batching():
    ps = synthetic_split()
    for shuffle in (False, True):
        seen = np.concatenate([b["idx"] for b in ps.iter_batches(100, 4, shuffle=shuffle, seed=3, epoch=1)])
        _check(np.array_equal(np.sort(seen), np.arange(ps.n)), "عيّنة مفقودة أو مكرّرة في الحقبة")
    p1 = ps.batch_plan(100, 4, shuffle=True, seed=3, epoch=1)
    _check(p1 == ps.batch_plan(100, 4, shuffle=True, seed=3, epoch=1), "الخلط غير حتمي")
    _check(p1 != ps.batch_plan(100, 4, shuffle=True, seed=3, epoch=2), "نفس الترتيب في حقبتين")
    ordered = [d for b in ps.batch_plan(100, 4) for d in b]
    _check(ordered == list(range(ps.n_days)), "val/test ليست بترتيب التاريخ")
    for dl in p1:
        _check(len(dl) <= 4 and (len(dl) == 1 or ps.sizes[dl].sum() <= 100), "حدود الدفعة")
        b = ps.make_batch(dl)
        for k in range(len(dl)):
            m = b["day"] == k
            _check(sorted(b["pos"][m].tolist()) == list(range(m.sum())), "pos ليست 0..n-1 داخل اليوم")
            _check(len(np.unique(ps.day_key[b["idx"][m]])) == 1, "يومان في نفس رقم day")
    return True


def test_masking(encoder=None, seq_len=8, n_features=5):
    import tensorflow as tf
    from .model import build_panel_model, tiny_encoder
    ps = synthetic_split(seq_len=seq_len, n_features=n_features)
    for att in (True, False):
        tf.keras.utils.set_random_seed(1)
        enc = encoder(att) if encoder else tiny_encoder(seq_len, n_features)
        m = build_panel_model(enc, seq_len, n_features, d_model=16, num_heads=4, cross_attention=att, dropout=0.0)
        big = ps.make_batch([0, 5, 9, 20])            # أحجام أيام مختلفة ⇒ حشو مختلف
        alone = ps.make_batch([5])
        o_big = m({k: big[k] for k in ("x", "day", "pos")}, training=False)["logit"].numpy()
        o_alone = m({k: alone[k] for k in ("x", "day", "pos")}, training=False)["logit"].numpy()
        pos_in_big = {i: r for r, i in enumerate(big["idx"])}
        rows = [pos_in_big[i] for i in alone["idx"]]
        _check(np.allclose(o_big[rows], o_alone, atol=1e-5), f"attention={att}: مخرجات اليوم تتغيّر بالأيام الأخرى/الحشو")
        # تغيير عملة واحدة في اليوم
        x2 = alone["x"].copy()
        x2[0] += 3.0
        o2 = m({"x": x2, "day": alone["day"], "pos": alone["pos"]}, training=False)["logit"].numpy()
        changed_other = not np.allclose(o2[1:], o_alone[1:], atol=1e-6)
        _check(changed_other == att, f"attention={att}: تغيّر عملة أخرى من اليوم {'لم' if att else ''} يؤثّر")
        # تبديل ترتيب العملات داخل اليوم (pos) لا يغيّر مخرج كل عملة (لا ترميز موضعي عبر العملات)
        perm = np.random.default_rng(0).permutation(len(alone["pos"]))
        o3 = m({"x": alone["x"], "day": alone["day"], "pos": alone["pos"][perm]}, training=False)["logit"].numpy()
        _check(np.allclose(o3, o_alone, atol=1e-5), f"attention={att}: المخرج يعتمد على ترتيب العملات")
        _check(np.isfinite(o_big).all(), "NaN في المخرجات")
    return True


def test_ic_loss():
    import tensorflow as tf
    from .train import day_pearson
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(4), [12, 3, 15, 11]).astype("int32")
    pred, tgt = rng.normal(size=len(day)).astype("float32"), rng.normal(size=len(day)).astype("float32")
    tgt[day == 0] += pred[day == 0]
    val, n_valid = day_pearson(tf.constant(pred), tf.constant(tgt), tf.constant(day), 4, 10.0)
    ref = np.mean([np.corrcoef(pred[day == d], tgt[day == d])[0, 1] for d in (0, 2, 3)])
    _check(abs(float(val) - ref) < 1e-4 and int(n_valid) == 3, f"IC {float(val)} ≠ {ref}")
    return True


def test_training(encoder_builder=None, seq_len=8, n_features=5, epochs=None, strict=True):
    """strict=False: بلا عتبات التعلّم (انخفاض الخسارة وIC الإشارة التركيبية) — للمُرمِّز الحقيقي المُنظَّم بشدّة في
    اختبار دفتر main، حيث 4 حقب صغيرة قد لا تكفي؛ الفحوص الحتمية (الأعمدة، الاستئناف، التقرير) تبقى."""
    import tensorflow as tf
    from .experiment import build_panel_splits, run_panel_experiment
    from .model import tiny_encoder
    from .report import compare
    epochs = epochs or (4 if strict else 2)
    tr = synthetic_split(40, 120 if strict else 40, seq_len, n_features, seed=1, signal=2.0, name="train")
    va = synthetic_split(40, 40, seq_len, n_features, start_day=18200, seed=2, signal=2.0, name="val")
    te = synthetic_split(40, 40, seq_len, n_features, start_day=18300, seed=3, signal=2.0, name="test")

    def as_split(ps):
        y = {f"y_{t}_{k}": (ps.ycls[:, i] * 2 - 1 if k == "class" else ps.yreg[:, i])
             for i, t in enumerate(ps.targets) for k in ("class", "reg")}
        return {"X_1D": ps.X, "y": y, "last_candles": ps.lc}

    test_dict = {a: as_split(te.take(np.flatnonzero(te.assets == a))) for a in np.unique(te.assets)}
    enc_b = encoder_builder or (lambda: _TinyBase(seq_len, n_features))
    root = tempfile.mkdtemp(prefix="panel_selftest_")
    try:
        cfg = dict(epochs=epochs, patience=10, batch_samples=150, max_days=8, lr_warmup_epochs=0,
                   lr_schedule={"type": "constant"}, lr_initial=3e-3, ema_window_epochs=0.5)
        res = run_panel_experiment(as_split(tr), as_split(va), test_dict, "1D", enc_b, seq_len, n_features, root,
                                   variants=("A_ic", "B") if strict else ("A_ic",), seeds=(0,), train_cfg=cfg,
                                   train_assets=tr.assets, val_assets=va.assets,
                                   model_cfg=dict(d_model=16, num_heads=4), verbose=False)
        for name, (v, t) in res["runs"].items():
            _check(list(t.columns) == BASELINE_COLS and list(v.columns) == BASELINE_COLS, f"أعمدة {name}: {list(t.columns)}")
            _check(len(t) == te.n and len(v) == va.n and t[["mu_close", "p_up_close"]].notna().all().all(), "صفوف/NaN")
            _check((t["asset"] != "all").all() and (v["asset"] != "all").all(), "أسماء العملات مفقودة")
        import json
        st = json.load(open(os.path.join(root, "panel_A_ic_s0", "state.json")))
        h = [r["train_loss"] for r in st["history"]]
        _check(all(np.isfinite(h)), f"خسارة غير منتهية: {h}")
        _check(not strict or h[-1] < h[0], f"خسارة التدريب لم تنخفض: {h}")
        # الاستئناف: نفس run_dir بحقب أكثر يكمل من حيث توقّف (لا يبدأ من الصفر)
        tr_ps, va_ps, te_ps, _ = build_panel_splits(as_split(tr), as_split(va), test_dict, "1D", tr.assets, va.assets,
                                                    verbose=False)
        from .experiment import run_panel_variant
        _, _, st2, _ = run_panel_variant(tr_ps, va_ps, te_ps, enc_b, seq_len, n_features,
                                         os.path.join(root, "panel_A_ic_s0"), "A_ic", 0,
                                         dict(d_model=16, num_heads=4), {**cfg, "epochs": epochs + 1}, verbose=False)
        _check(st2["epoch"] == epochs + 1 and len(st2["history"]) == epochs + 1, "الاستئناف لم يُكمل")
        # الإشارة التركيبية (ميزة 0) يجب أن تُلتقط: IC موجب
        _check(not strict or res["table"].loc["IC يومي p_up_close", "A_ic_s0"] > 0.05,
               f"لم يتعلّم الإشارة التركيبية\n{res['table']}")
        compare({"x": res["runs"]["A_ic_s0"]}, verbose=False)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        tf.keras.backend.clear_session()
    return res["table"]


class _TinyBase:
    """يحاكي نموذج build_nig_timenet_v2: طبقة باسم trunk_drop يقتطعها extract_encoder."""

    def __new__(cls, seq_len, n_features):
        import tensorflow as tf
        L = tf.keras.layers
        inp = L.Input((seq_len, n_features))
        h = L.GRU(16)(L.BatchNormalization()(inp))
        h = L.Dropout(0.1, name="trunk_drop")(h)
        return tf.keras.Model(inp, {"y_close": L.Dense(1)(h)})


def run_panel_selftest(encoder_builder=None, seq_len=8, n_features=5, verbose=True, train=True, strict=True):
    """encoder_builder: دالة بلا وسائط تُرجع نموذج build_model_fn (من دفتر main)؛ None = مُرمِّز صغير."""
    from .model import extract_encoder
    enc = (lambda att: extract_encoder(encoder_builder())) if encoder_builder else None
    for fn, args in ((test_grouping, ()), (test_batching, ()), (test_ic_loss, ()),
                     (test_masking, (enc, seq_len, n_features))):
        fn(*args)
        if verbose:
            print(f"  ✅ {fn.__name__}", flush=True)
    if train:
        test_training(encoder_builder, seq_len, n_features, strict=strict)
        if verbose:
            print("  ✅ test_training (" + ("خسارة تنخفض وإشارة تُلتقط، " if strict else "خسارة منتهية، ")
                  + "استئناف، تصدير بأعمدة collect_signals، تقرير المقارنة)", flush=True)
    return True


if __name__ == "__main__":
    run_panel_selftest()
    print("✅ اختبارات اللوحة الذاتية نجحت")
