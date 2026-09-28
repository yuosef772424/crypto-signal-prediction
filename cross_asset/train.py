"""تدريب نموذج اللوحة: خسائر لكل يوم، حلقة تدريب قابلة للاستئناف، تنبؤ، وتصدير إشارات بصيغة النموذج الحالي.

الخسارة لكل دفعة (أيام كاملة؛ العيّنات مسطّحة فلا حشو يدخل الخسارة أصلاً):
    Σ_هدف [ w_cls · BCE(logit, تتفوّق على وسيط اليوم؛ تنعيم 0.1 كالنموذج الحالي)
          + w_reg · Huber(mu, العائد النسبي ÷ مقياسه في train) ]            (العائد مقصوص ±1 أصلاً في retarget_splits)
    + lambda_ic · (−متوسط الأيام لارتباط Pearson بين التنبؤ ورتبة عائد close النسبي داخل اليوم)
حدّ الارتباط (IC) اختياري: قابل للاشتقاق، يُحسب لكل يوم فيه ic_min_coins عملة على الأقل، على logit وmu لـ close.

الإعدادات الافتراضية = إعداد مقاومة الحفظ في دفتر main (AdamW lr 3e-4 وweight_decay 0.05، تسخين 3 حقب ثم cosine
بإعادة تشغيل، قصّ تدرّج 1.0، أوزان EMA بنافذة ~حقبة واحدة ويُحفظ «الأفضل» بها، تنعيم تسميات 0.1، إيقاف مبكر على val).
"""
import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras
warnings.filterwarnings("ignore", message="An input array is constant")

DEFAULT_TRAIN_CFG = dict(
    epochs=25, patience=6, batch_samples=1024, max_days=16, max_coins=None, min_coins=None, seed=0,
    lr_initial=3e-4, lr_min=5e-7, lr_warmup_epochs=3,
    lr_schedule={"type": "cosine_restarts", "cycle_length": 10, "cycle_mult": 1.5},
    weight_decay=0.05, clip_norm=1.0, ema=True, ema_window_epochs=1.0,
    label_smoothing=0.1, w_cls=1.0, w_reg=1.0, huber_delta=1.0,
    lambda_ic=0.0, ic_min_coins=10, ic_on=("logit", "mu"),
    monitor="val_loss",          # "val_loss" (أصغر أفضل) | أي مقياس val آخر من evaluate (أكبر أفضل): val_ic_close،
                                 # val_ic_low، val_ic_asym، val_auc_high ...
    max_steps_per_epoch=None,    # للاختبار السريع فقط
)
WD_EXCLUDE = ["bias", "gamma", "beta", "pos_emb", "rel_bias", "scale"]
# مفاتيح أُضيفت بعد تشغيلات محفوظة: قيمتها الافتراضية لا تدخل البصمة، فتبقى run_dir القديمة قابلة للاستئناف
_FP_OPTIONAL_KEYS = {"min_coins": None}


def lr_at_epoch(cfg, epoch):
    """نفس build_lr_schedule_fn في trainer_framework_v2 (تسخين خطّي ثم constant/cosine/cosine_restarts)."""
    lr0, lr_min, warm = cfg["lr_initial"], cfg["lr_min"], int(cfg.get("lr_warmup_epochs", 0))
    if warm and epoch < warm:
        return lr0 * (epoch + 1) / warm
    e, sched = epoch - warm, cfg.get("lr_schedule") or {"type": "constant"}
    if sched["type"] == "cosine_restarts":
        L = sched.get("cycle_length", 10)
        while e >= L:
            e -= L
            L = int(L * sched.get("cycle_mult", 1.5))
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * e / L))
    if sched["type"] == "cosine":
        p = min(e / max(sched.get("total_epochs", 100), 1), 1.0)
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * p))
    return lr0


def robust_scales(yreg):
    """مقياس كل هدف انحدار من train: 1.4826·MAD (مقاوم للقفزات)، وإلا الانحراف المعياري."""
    med = np.median(yreg, axis=0)
    s = 1.4826 * np.median(np.abs(yreg - med), axis=0)
    std = yreg.std(axis=0)
    return np.where(s > 1e-8, s, np.where(std > 1e-8, std, 1.0)).astype("float32")


# ── الخسائر (دوال tf محضة — يختبرها selftest مقابل numpy) ──
def day_pearson(pred, target, day, n_days, min_n):
    """متوسط ارتباط Pearson بين pred وtarget داخل كل يوم (أيام بـ n ≥ min_n وتباين هدف > 0 فقط)."""
    ones = tf.ones_like(pred)
    n = tf.math.unsorted_segment_sum(ones, day, n_days)
    safe_n = tf.maximum(n, 1.0)
    pc = pred - tf.gather(tf.math.unsorted_segment_sum(pred, day, n_days) / safe_n, day)
    tc = target - tf.gather(tf.math.unsorted_segment_sum(target, day, n_days) / safe_n, day)
    cov = tf.math.unsorted_segment_sum(pc * tc, day, n_days)
    vp = tf.math.unsorted_segment_sum(pc * pc, day, n_days)
    vt = tf.math.unsorted_segment_sum(tc * tc, day, n_days)
    corr = cov / (tf.sqrt(vp + 1e-8) * tf.sqrt(vt + 1e-12))
    valid = tf.cast((n >= min_n) & (vt > 1e-12), pred.dtype)
    return tf.reduce_sum(corr * valid) / tf.maximum(tf.reduce_sum(valid), 1.0), tf.reduce_sum(valid)


def _ic_index(targets):
    """هدف حدّ IC: close إن وُجد، وإلا آخر الأهداف (مثلاً low حين close معلّق)."""
    return targets.index("close") if "close" in targets else len(targets) - 1


def panel_loss(out, batch, cfg, reg_scale, targets=("high", "low", "close")):
    """يُرجع (الخسارة الكلية، قاموس مكوّنات). targets بترتيب أعمدة logit/mu (افتراضياً الثلاثة)."""
    ls = cfg["label_smoothing"]
    ycls = batch["ycls"] * (1.0 - ls) + 0.5 * ls
    bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ycls, logits=out["logit"]), axis=0)   # (K,)
    err = out["mu"] - batch["yreg"] / reg_scale
    a, d = tf.abs(err), cfg["huber_delta"]
    huber = tf.reduce_mean(tf.where(a <= d, 0.5 * err * err, d * (a - 0.5 * d)), axis=0)                      # (K,)
    total = cfg["w_cls"] * tf.reduce_sum(bce) + cfg["w_reg"] * tf.reduce_sum(huber)
    comps = {**{f"bce_{t}": bce[i] for i, t in enumerate(targets)},
             **{f"huber_{t}": huber[i] for i, t in enumerate(targets)}}
    ci = _ic_index(list(targets))
    n_days = tf.reduce_max(batch["day"]) + 1
    ics = []
    for k in cfg["ic_on"]:
        ic, n_valid = day_pearson(out[k][:, ci], batch["yrank"], batch["day"], n_days, float(cfg["ic_min_coins"]))
        comps[f"ic_{k}_{targets[ci]}"] = ic
        ics.append(ic)
    if ics:
        comps["ic_days"] = n_valid
        if cfg["lambda_ic"]:
            total = total - cfg["lambda_ic"] * tf.add_n(ics) / len(ics)
    comps["loss"] = total
    return total, comps


_SIG = {"x": None, "day": tf.TensorSpec([None], tf.int32), "pos": tf.TensorSpec([None], tf.int32),
        "ycls": tf.TensorSpec([None, None], tf.float32), "yreg": tf.TensorSpec([None, None], tf.float32),
        "yrank": tf.TensorSpec([None], tf.float32)}


class PanelTrainer:
    """حلقة تدريب بسيطة وقابلة للاستئناف (run_dir على Drive يكفي لاستئناف بعد انقطاع Colab).

    run_dir/state.json            : الحقبة، الأفضل، الصبر، السجلّ، وبصمة الإعداد (لا يُستأنف تدريب بإعداد مختلف)
                                    وبصمة محتوى البيانات data_fp (لا يُستأنف على بيانات أخرى بنفس الشكل والتواريخ)
    run_dir/last.weights.h5 + last_ema.npz + last_opt.npz : حالة آخر حقبة مكتملة
    run_dir/best.weights.h5       : أفضل حقبة على val (بأوزان EMA إن فُعّلت)
    """

    def __init__(self, model, cfg, run_dir, reg_scale, seq_len, n_features, fingerprint=None, verbose=True,
                 data_fingerprint=None):
        self.model, self.run_dir, self.verbose = model, run_dir, verbose
        # منفصلة عن fingerprint عمداً: إدخالها في _fp يُبطل استئناف كل التشغيلات السابقة لها. حالة بلا data_fp (قبلها)
        # تُستأنف مع تحذير صريح؛ حالة بـ data_fp مختلف ترفض.
        self.data_fp = data_fingerprint
        self.targets = tuple(getattr(model, "targets", ("high", "low", "close")))   # ترتيب أعمدة logit/mu
        self.cfg = {**DEFAULT_TRAIN_CFG, **(cfg or {})}
        self.cfg["ic_on"] = tuple(self.cfg["ic_on"])
        self.reg_scale = tf.constant(np.asarray(reg_scale, dtype="float32"))
        self.fingerprint = fingerprint or {}
        os.makedirs(run_dir, exist_ok=True)
        self.opt = keras.optimizers.AdamW(learning_rate=self.cfg["lr_initial"], weight_decay=self.cfg["weight_decay"],
                                          global_clipnorm=self.cfg["clip_norm"])
        self.opt.exclude_from_weight_decay(var_names=WD_EXCLUDE)
        self.opt.build(model.trainable_variables)
        self.ema_vars = [tf.Variable(v, trainable=False) for v in model.trainable_variables] if self.cfg["ema"] else []
        self.ema_step = tf.Variable(0.0, trainable=False)
        self.ema_target = tf.Variable(0.999, trainable=False)
        sig = dict(_SIG, x=tf.TensorSpec([None, seq_len, n_features], tf.float32))
        self._train_step = tf.function(self._train_step_py, input_signature=[sig])
        self._eval_step = tf.function(self._eval_step_py, input_signature=[sig])
        self._predict_step = tf.function(
            lambda b: self.model(b, training=False),
            input_signature=[{k: sig[k] for k in ("x", "day", "pos")}])

    # ── الخطوات ──
    def _train_step_py(self, batch):
        with tf.GradientTape() as tape:
            out = self.model(batch, training=True)
            loss, comps = panel_loss(out, batch, self.cfg, self.reg_scale, self.targets)
        tv = self.model.trainable_variables
        grads = tape.gradient(loss, tv)
        self.opt.apply_gradients(zip(grads, tv))
        if self.ema_vars:
            self.ema_step.assign_add(1.0)
            decay = tf.minimum(self.ema_target, (1.0 + self.ema_step) / (10.0 + self.ema_step))
            for e, v in zip(self.ema_vars, tv):
                e.assign(decay * e + (1.0 - decay) * v)
        return comps

    def _eval_step_py(self, batch):
        out = self.model(batch, training=False)
        _, comps = panel_loss(out, batch, self.cfg, self.reg_scale, self.targets)
        return comps, out

    @staticmethod
    def _tf_batch(b, keys=("x", "day", "pos", "ycls", "yreg", "yrank")):
        return {k: tf.convert_to_tensor(b[k]) for k in keys}

    # ── أوزان EMA ──
    def _swap_in_ema(self):
        if not self.ema_vars:
            return None
        backup = [v.numpy() for v in self.model.trainable_variables]
        for v, e in zip(self.model.trainable_variables, self.ema_vars):
            v.assign(e)
        return backup

    def _restore(self, backup):
        if backup is not None:
            for v, b in zip(self.model.trainable_variables, backup):
                v.assign(b)

    # ── التقييم ──
    def evaluate(self, ps, use_ema=True):
        """خسائر val المتوسّطة (موزونة بعدد العيّنات) + AUC لكل هدف + IC يومي (Spearman) لـ logit close."""
        from sklearn.metrics import roc_auc_score
        backup = self._swap_in_ema() if use_ema else None
        try:
            sums, n_tot = {}, 0
            logits = np.zeros((ps.n, len(ps.targets)), "float32")
            mus = np.zeros((ps.n, len(ps.targets)), "float32")
            for b in ps.iter_batches(self.cfg["batch_samples"], self.cfg["max_days"]):
                comps, out = self._eval_step(self._tf_batch(b))
                m = len(b["idx"])
                for k, v in comps.items():
                    if k != "ic_days":
                        sums[k] = sums.get(k, 0.0) + float(v) * m
                n_tot += m
                logits[b["idx"]] = out["logit"].numpy()
                mus[b["idx"]] = out["mu"].numpy()
        finally:
            self._restore(backup)
        res = {f"val_{k}": v / max(n_tot, 1) for k, v in sums.items()}
        asym = asym_score(ps, mus * self.reg_scale.numpy())
        if asym is not None:
            res["val_ic_asym"] = group_ic(ps, asym, realized_return(ps))
        for i, t in enumerate(ps.targets):
            y = ps.ycls[:, i]
            if 0 < y.mean() < 1:
                res[f"val_auc_{t}"] = float(roc_auc_score(y, logits[:, i]))
        ci = _ic_index(list(ps.targets))
        df = pd.DataFrame({"d": ps.day_key, "s": logits[:, ci], "r": ps.yreg[:, ci]})
        ic = df.groupby("d").filter(lambda g: len(g) >= 10).groupby("d").apply(
            lambda g: g["s"].corr(g["r"], method="spearman"))
        res[f"val_ic_{ps.targets[ci]}"] = float(ic.mean()) if len(ic) else float("nan")
        return res

    def predict(self, ps, use_ema=False, groups=None):
        """(logit, mu بوحدات العائد) لكل عيّنة بترتيب X الأصلي. use_ema=False: الأوزان الحالية (بعد load_best هي الأفضل).
        groups: مجموعات انتباه بديلة (members, scored) من ps.chunk_groups(k) — النموذج نفسه يرى k عملة فقط لكل تنبؤ."""
        backup = self._swap_in_ema() if use_ema else None
        batches = (ps.iter_batches(self.cfg["batch_samples"], self.cfg["max_days"]) if groups is None
                   else ps.iter_group_batches(groups[0], self.cfg["batch_samples"], self.cfg["max_days"], groups[1]))
        try:
            logit = np.zeros((ps.n, len(ps.targets)), "float32")
            mu = np.zeros((ps.n, len(ps.targets)), "float32")
            for b in batches:
                out = self._predict_step(self._tf_batch(b, ("x", "day", "pos")))
                keep = b.get("score", slice(None))
                logit[b["idx"][keep]] = out["logit"].numpy()[keep]
                mu[b["idx"][keep]] = out["mu"].numpy()[keep] * self.reg_scale.numpy()
        finally:
            self._restore(backup)
        return logit, mu

    # ── الحفظ والاستئناف ──
    def _fp(self):
        cfg = {k: v for k, v in self.cfg.items() if k not in ("epochs", "patience")
               and not (k in _FP_OPTIONAL_KEYS and v == _FP_OPTIONAL_KEYS[k])}
        blob = json.dumps({"cfg": cfg,
                           **self.fingerprint}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def _save_state(self, state):
        self.model.save_weights(os.path.join(self.run_dir, "last.weights.h5"))
        if self.ema_vars:
            np.savez(os.path.join(self.run_dir, "last_ema.npz"), *[e.numpy() for e in self.ema_vars],
                     step=self.ema_step.numpy())
        np.savez(os.path.join(self.run_dir, "last_opt.npz"), *[v.numpy() for v in self.opt.variables])
        tmp = os.path.join(self.run_dir, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1, default=float)
        os.replace(tmp, os.path.join(self.run_dir, "state.json"))

    def _load_state(self):
        p = os.path.join(self.run_dir, "state.json")
        if not os.path.exists(p):
            return None
        state = json.load(open(p))
        if state.get("fingerprint") != self._fp():
            raise RuntimeError(f"run_dir {self.run_dir} فيه تدريب بإعداد مختلف (بصمة {state.get('fingerprint')} ≠ "
                               f"{self._fp()}) — غيّر run_dir أو احذفه")
        if self.data_fp is not None:
            if state.get("data_fp") is None:
                warnings.warn(f"⚠️ run_dir {self.run_dir} بلا بصمة بيانات (أقدم منها): يُستأنف دون تحقّق أن البيانات هي "
                              "نفسها التي دُرِّب عليها — احذفه إن أُعيد بناء البيانات", RuntimeWarning, stacklevel=2)
                state["data_fp"] = self.data_fp
            elif state["data_fp"] != self.data_fp:
                raise RuntimeError(f"run_dir {self.run_dir} دُرِّب على بيانات مختلفة (بصمة البيانات {state['data_fp']} ≠ "
                                   f"{self.data_fp}) — غيّر run_dir أو احذفه")
        self.model.load_weights(os.path.join(self.run_dir, "last.weights.h5"))
        if self.ema_vars:
            z = np.load(os.path.join(self.run_dir, "last_ema.npz"))
            for i, e in enumerate(self.ema_vars):
                e.assign(z[f"arr_{i}"])
            self.ema_step.assign(float(z["step"]))
        z = np.load(os.path.join(self.run_dir, "last_opt.npz"))
        for i, v in enumerate(self.opt.variables):
            v.assign(z[f"arr_{i}"])
        return state

    def load_best(self):
        self.model.load_weights(os.path.join(self.run_dir, "best.weights.h5"))

    # ── الحلقة ──
    def fit(self, train_ps, val_ps):
        c = self.cfg
        state = self._load_state() or {"epoch": 0, "best": None, "best_epoch": None, "wait": 0, "history": [],
                                       "fingerprint": self._fp(), "data_fp": self.data_fp}
        if state["epoch"] and self.verbose:
            print(f"↩️ استئناف من الحقبة {state['epoch']} (الأفضل {state['best']} @ {state['best_epoch']})", flush=True)
        steps = len(train_ps.batch_plan(c["batch_samples"], c["max_days"], shuffle=True, seed=c["seed"], epoch=0))
        if c["max_steps_per_epoch"]:
            steps = min(steps, c["max_steps_per_epoch"])
        if c["ema_window_epochs"]:
            self.ema_target.assign(float(np.clip(1.0 - 1.0 / max(c["ema_window_epochs"] * steps, 1), 0.9, 0.9999)))
        sign = -1.0 if c["monitor"] == "val_loss" else 1.0          # نحوّل كل شيء إلى «أكبر أفضل»
        for epoch in range(state["epoch"], c["epochs"]):
            if state["wait"] >= c["patience"]:
                break
            t0 = time.time()
            self.opt.learning_rate.assign(lr_at_epoch(c, epoch))
            agg, n_b = {}, 0
            for i, b in enumerate(train_ps.iter_batches(c["batch_samples"], c["max_days"], shuffle=True, seed=c["seed"],
                                                        epoch=epoch, max_coins=c["max_coins"],
                                                        min_coins=c["min_coins"])):
                if c["max_steps_per_epoch"] and i >= c["max_steps_per_epoch"]:
                    break
                comps = self._train_step(self._tf_batch(b))
                for k, v in comps.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                n_b += 1
            tr = {f"train_{k}": v / max(n_b, 1) for k, v in agg.items() if k != "ic_days"}
            if not np.isfinite(tr.get("train_loss", np.nan)):
                raise FloatingPointError(f"خسارة غير منتهية في الحقبة {epoch + 1}: {tr}")
            va = self.evaluate(val_ps)
            score = sign * va[c["monitor"]]
            improved = state["best"] is None or score > sign * state["best"]
            if improved:
                backup = self._swap_in_ema()
                self.model.save_weights(os.path.join(self.run_dir, "best.weights.h5"))
                self._restore(backup)
                state.update(best=va[c["monitor"]], best_epoch=epoch + 1, wait=0)
            else:
                state["wait"] += 1
            row = {"epoch": epoch + 1, "lr": float(self.opt.learning_rate.numpy()), "steps": n_b,
                   "sec": round(time.time() - t0, 1), **tr, **va}
            state["history"].append(row)
            state["epoch"] = epoch + 1
            self._save_state(state)
            if self.verbose:
                print(f"[{epoch + 1:>3}/{c['epochs']}] {row['sec']:.0f}s {n_b} خطوة | train loss {tr['train_loss']:.4f} | "
                      f"val loss {va['val_loss']:.4f} | AUC h/l/c {va.get('val_auc_high', np.nan):.3f}/"
                      f"{va.get('val_auc_low', np.nan):.3f}/{va.get('val_auc_close', np.nan):.3f} | "
                      f"IC {self.targets[_ic_index(list(self.targets))]} "
                      f"{va[f'val_ic_{self.targets[_ic_index(list(self.targets))]}']:+.4f}" + (" ⭐" if improved else f" (صبر {state['wait']}/{c['patience']})"),
                      flush=True)
        return state


def realized_return(ps):
    """عائد close المحقَّق لكل عيّنة (future_close ÷ last_close − 1) — من last_candles لا من y، فيبقى معرّفاً حين
    يُعلَّق هدف close."""
    return ps.lc[:, 4] / ps.lc[:, 2] - 1.0


def asym_score(ps, mu):
    """تباين المدى المتوقَّع log(up + 1e-3) − log(dn + 1e-3)، حيث up = القمة المتوقَّعة ÷ P − 1 وdn = 1 − القاع المتوقَّع ÷ P
    (P آخر إغلاق، mu بوحدات هدف التدريب — يُقسَم هنا على ps.target_scale ليصير عائداً؛ وكلاهما مقصوص عند 1e-5). نفس تعريف skew في
    docs/research/scripts/hourly_1h/exc_analysis.py، الذي حمل معلومة الاتجاه (IC ≈ 0.05 مع العائد، القسم ٤). None إن
    غاب high أو low."""
    if "high" not in ps.targets or "low" not in ps.targets:
        return None
    lc, t = ps.lc, list(ps.targets)
    mu = np.asarray(mu, dtype="float64") / getattr(ps, "target_scale", 1.0)
    up = np.maximum(lc[:, 0] * (1.0 + mu[:, t.index("high")]) / lc[:, 2] - 1.0, 1e-5)
    dn = np.maximum(1.0 - lc[:, 1] * (1.0 + mu[:, t.index("low")]) / lc[:, 2], 1e-5)
    return np.log(up + 1e-3) - np.log(dn + 1e-3)


def group_ic(ps, score, target, min_n=10):
    """متوسط Spearman داخل كل مجموعة مقطعية (طابع) فيها min_n عيّنة على الأقل."""
    df = pd.DataFrame({"g": ps.day_key, "s": np.asarray(score, "float64"), "r": np.asarray(target, "float64")})
    df = df[df.groupby("g")["g"].transform("size") >= min_n]
    if df.empty:
        return float("nan")
    rk = df.groupby("g")[["s", "r"]].rank()
    rk["g"] = df["g"]
    ic = rk.groupby("g").apply(lambda x: x["s"].corr(x["r"]))
    return float(ic.mean())


def export_signals(ps, logit, mu, split_name, targets=("high", "low", "close")):
    """إشارات بنفس أعمدة collect_signals في دفتر main (بلا wst/conf — لا رؤوس عدم يقين هنا).
    mu بوحدات هدف التدريب (عائد × ps.target_scale)؛ يُقسَم أولاً فتبقى أعمدة mu_* عائداً وpred_* أسعاراً حقيقية."""
    lc = ps.lc
    mu = np.asarray(mu, dtype="float64") / getattr(ps, "target_scale", 1.0)
    df = pd.DataFrame({
        "asset": ps.assets if ps.assets is not None else np.array(["all"] * ps.n, dtype=object),
        "timestamp": lc[:, 3], "entry": lc[:, 2], "last_high": lc[:, 0], "last_low": lc[:, 1],
        "fut_close": lc[:, 4], "fut_high": lc[:, 6], "fut_low": lc[:, 5],
    })
    p = 1.0 / (1.0 + np.exp(-logit.astype("float64")))
    for i, t in enumerate(targets):
        df[f"mu_{t}"] = mu[:, i].astype("float64")
        df[f"p_up_{t}"] = p[:, i]
    df["pred_high"] = df["last_high"] * (1.0 + df["mu_high"])
    df["pred_low"] = df["last_low"] * (1.0 + df["mu_low"])
    df["up"] = (df["fut_close"] > df["entry"]).astype(int)
    df["split"] = split_name
    # بعد أعمدة collect_signals (لا بينها): تسمية التدريب نفسها — report.summarize يقيس AUC عليها لا على تسمية يعيد اشتقاقها
    for i, t in enumerate(targets):
        df[f"y_{t}_class"] = ps.ycls[:, i].astype(int)
    return df
