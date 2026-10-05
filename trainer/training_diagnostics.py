"""
PURPOSE:  TrainingDiagnostics: optional diagnostic recorder (+ SAMPLE_IDX_KEY, with_sample_index).
TAGS:     TrainingDiagnostics, with_sample_index, SAMPLE_IDX_KEY, tap, per-sample loss, diagnostics callback
PITFALLS: Does nothing unless added to callbacks; main.ipynb wraps the train dataset with diag.tap. Executed into the
          one shared trainer namespace by trainer/_loader.py (never imported on its own): names from other modules
          resolve at call time.

### 6.5) `TrainingDiagnostics` — مسجّل تشخيصي اختياري (معطَّل ما لم يُضَف إلى callbacks)

يسجّل: خسارة كل دفعة وكل رأس؛ نظيم التدرّج الكلي وتدرّج كل رأس على **الجذع المشترك** (هيمنة/تضارب الرؤوس)؛ نظيم التدرّج ونسبة
التحديث/الوزن لكل مجموعة طبقات؛ **تأثير الدفعات** بتقريب TracIn (`lr·⟨∇L_دفعة, ∇L_probe⟩`)؛ و**خريطة البيانات** (easy/ambiguous/hard +
تسميات مشبوهة). لا يغيّر أوزان التدريب (قياسه بوضع الاستدلال وخارج `train_step`).

```python
diag = TrainingDiagnostics(probe=(x_val, y_val), cartography=(x_tr, with_sample_index(y_tr)))
train_ds = diag.tap(make_shuffled_dataset(x_tr, with_sample_index(y_tr), batch_size))   # tap لا يغيّر العناصر ولا ترتيبها
trainer.fit(train_ds, ..., callbacks=callbacks + [diag])
diag.summary(); diag.top_batches("harmful", target="high_class"); diag.hardest_samples("high_class")
```
في `main`: `diag, train_ds = make_training_diagnostics(train, val)` (القسم ٧-ط)، والتفسير في `docs/research/model_diagnostics.md`.
"""
# @title 6.5) TrainingDiagnostics — مسجّل تشخيصي اختياري (لا يعمل إلا إن أُضيف إلى callbacks)
from array import array

SAMPLE_IDX_KEY = "__sample_idx__"


def with_sample_index(y_dict, indices=None):
    """يُضيف إلى قاموس الأهداف مفتاحاً `__sample_idx__` (رقم صفّ كل عيّنة) ليعرف `TrainingDiagnostics` أيّ عيّنات دخلت كل دفعة.
    GenericTrainer لا يقرأ إلا مفاتيح `true_key` لأهدافه، فالمفتاح الإضافي لا يمسّ الخسارة ولا التدريب.
    float32 يمثّل الأعداد الصحيحة حتى 2^24 بدقّة، فيُرفض ما فوقها بدل أن يُفسَد الفهرس بصمت.
    indices: فهارس الصفوف الأصلية (الافتراضي arange = موضع العيّنة في مصفوفات التدريب)."""
    n = len(next(iter(y_dict.values())))
    if SAMPLE_IDX_KEY in y_dict:
        raise ValueError(f"y_dict يحوي {SAMPLE_IDX_KEY} أصلاً")
    idx = np.arange(n) if indices is None else np.asarray(indices)
    if idx.shape != (n,):
        raise ValueError(f"indices يجب أن يكون بطول {n}، لا {idx.shape}")
    if n and int(idx.max()) >= 2 ** 24:
        raise ValueError("فهارس ≥ 2^24 لا تُمثَّل بدقّة في float32 (مصفوفات y تُحوَّل إليه في make_shuffled_dataset)")
    return {**y_dict, SAMPLE_IDX_KEY: idx.astype("float32")}


class TrainingDiagnostics(tf.keras.callbacks.Callback):
    """مسجّل تشخيصي للتدريب — **معطَّل ما لم يُضَف إلى callbacks**، ولا يغيّر خطوة التحديث (أوزان النموذج بعد fit مطابقة
    لتشغيل بلا المسجّل بنفس البذرة: كل قياسه بوضع الاستدلال training=False فلا يستهلك عدّاد dropout، وتدرّجاته تُحسب
    خارج train_step بـ tf.GradientTape منفصلة).

    ما يسجّله:
      ١) كل دفعة (بلا تكلفة): الخسارة الكلية وخسارة كل رأس — تُستخلص من متوسطات Keras التراكمية (k·mₖ − (k−1)·mₖ₋₁).
      ٢) كل `grad_every` دفعة: نظيم التدرّج الكلي؛ **نظيم تدرّج كل رأس على الجذع المشترك** (متغيّرات تتلقّى تدرّجاً من ≥ ٢
         مهام) بعد وزن المهمة الفعلي (Kendall) — هيمنة رأس على الجذع؛ تضارب التدرّجات (cos بين الرؤوس)؛ نظيم التدرّج
         لكل مجموعة طبقات (تلاشي/انفجار)؛ ونسبة التحديث/الوزن الفعلية ‖Δw‖/‖w‖ لكل مجموعة (يشمل Adam والقصّ).
      ٣) كل `influence_every` دفعة (تأثير الدفعة، تقريب TracIn، Pruthi et al. 2020):
            influence ≈ lr · ⟨∇L_دفعة_التدريب , ∇L_probe⟩  على مجموعة probe ثابتة صغيرة من val.
         موجب = الدفعة تخفض خسارة val (مفيدة)، سالب = ترفعها (ضارّة/مشبوهة). `infl:<رأس>` = توافق **نفس المهمة** (تدرّج
         رأس الدفعة مع تدرّج رأسه على probe). المعاملات: `influence_scope` = "heads" (المتغيّرات الخاصة برأس واحد؛ الأرخص
         والافتراضي) | "trunk" (المشتركة) | "all". يُقاس على أوزان ما بعد الخطوة (تقريب من الدرجة الأولى؛ مع Adam تقدير
         اتجاهي لا كمّي).
      ٤) كل حقبة: «خريطة البيانات» (Swayamdipta et al. 2020) على مجموعة جزئية ثابتة `cartography=(x, y[, idx])`: ثقة كل عيّنة
         بالتسمية الصحيحة → متوسط الثقة، التباين عبر الحقب، نسبة الصواب → easy / ambiguous / hard (+ `noise_suspect`).
         ⚠️ تُقاس بنهاية كل حقبة بوضع الاستدلال (ليس أثناء التدريب كما في الورقة) — أنظف وأرخص.

      ٥) تقرير الطبقات أثناء التدريب/عند نهايته: `report_fn=lambda m: layer_probe_report(m, ...)` (أو أي دالة تأخذ النموذج الداخلي) تُستدعى
         كل `report_every` حقبة وعند نهاية التدريب (بعد استعادة «الأفضل») وتُحفظ في `diag.layer_reports = [(حقبة، نتيجة)]` — اتجاه
         مسبار كل طبقة عبر الحقب. و`diag.group_trend()` = اتجاه نظيم تدرّج كل طبقة (أخير ÷ أول) ونسبة التحديث/الوزن.

    الاستعمال (القياس على دفعات حقيقية يحتاج tap على dataset التدريب — لا يغيّر عناصره ولا ترتيبها):
        diag = TrainingDiagnostics(probe=(val_x, val_y), cartography=(tr_x, tr_y_with_idx), sample_meta=meta_df)
        train_ds = diag.tap(make_shuffled_dataset(tr_x, with_sample_index(tr_y), batch_size, seed=0))
        trainer.fit(train_ds, ..., callbacks=callbacks + [diag])
        diag.summary()      # {'batches', 'grad', 'groups', 'influence', 'top_harmful', 'top_helpful', 'cartography', ...}
    بلا tap: تُسجَّل الخسائر والخريطة فقط (تحذير صريح، لا تدرّجات). أسماء الخيارات الخاطئة ترفع TypeError، وقيمها الخاطئة ValueError."""

    INFLUENCE_SCOPES = ("heads", "trunk", "all")

    def __init__(self, probe=None, *, grad_every=25, influence_every=50, influence_scope="heads", probe_size=256,
                 top_k=10, cartography=None, cartography_size=2000, cartography_batch_size=512, sample_meta=None,
                 group_quantile=1.0 / 3.0, noise_correct_max=0.1, noise_conf_max=0.35, report_fn=None, report_every=0,
                 seed=0, verbose=1):
        super().__init__()
        for name, v in (("grad_every", grad_every), ("influence_every", influence_every)):
            if not isinstance(v, (int, np.integer)) or v < 0:
                raise ValueError(f"{name}: عدد صحيح ≥ 0 (0 = معطَّل)، لا {v!r}")
        if influence_scope not in self.INFLUENCE_SCOPES:
            raise ValueError(f"influence_scope: واحد من {self.INFLUENCE_SCOPES}، لا {influence_scope!r}")
        if int(top_k) < 1 or int(probe_size) < 1 or int(cartography_size) < 1 or int(cartography_batch_size) < 1:
            raise ValueError("top_k/probe_size/cartography_size/cartography_batch_size ≥ 1")
        if not 0.0 < group_quantile < 0.5:
            raise ValueError("group_quantile في (0، 0.5)")
        if probe is not None and (not isinstance(probe, (tuple, list)) or len(probe) != 2):
            raise ValueError("probe = (x, y) بصيغة ما يخرجه dataset التدريب (x مصفوفة أو قاموس {فريم: مصفوفة}، y قاموس أهداف المدرّب)")
        if cartography is not None and (not isinstance(cartography, (tuple, list)) or len(cartography) not in (2, 3)):
            raise ValueError("cartography = (x, y) أو (x, y, idx)")
        probe_missing = bool(influence_every) and probe is None
        self.grad_every, self.influence_every = int(grad_every), int(influence_every)
        self.influence_scope, self.top_k = influence_scope, int(top_k)
        self.verbose, self.group_quantile = int(verbose), float(group_quantile)
        self.noise_correct_max, self.noise_conf_max = float(noise_correct_max), float(noise_conf_max)
        if report_fn is not None and not callable(report_fn):
            raise ValueError("report_fn: دالة (النموذج الداخلي) ← أي نتيجة (مثلاً lambda m: layer_probe_report(m, ...))")
        if not isinstance(report_every, (int, np.integer)) or report_every < 0:
            raise ValueError("report_every: عدد صحيح ≥ 0 (0 = عند نهاية التدريب فقط)")
        self.report_fn, self.report_every, self.layer_reports = report_fn, int(report_every), []
        self.sample_meta = sample_meta
        self.cartography_batch_size = int(cartography_batch_size)
        self._rng = np.random.default_rng(seed)
        self._infl_on = bool(self.influence_every) and not probe_missing
        self._probe = None
        if probe is not None:
            x, y = probe
            n = len(next(iter(x.values())) if isinstance(x, dict) else x)
            sel = np.sort(self._rng.choice(n, min(int(probe_size), n), replace=False))
            self._probe = (self._take(x, sel, "float32"), {k: np.asarray(v)[sel].astype("float32") for k, v in y.items()
                                                           if k != SAMPLE_IDX_KEY})
        self._carto = None
        if cartography is not None:
            x, y = cartography[0], cartography[1]
            n = len(next(iter(x.values())) if isinstance(x, dict) else x)
            idx = np.arange(n) if len(cartography) == 2 else np.asarray(cartography[2])
            sel = np.sort(self._rng.choice(n, min(int(cartography_size), n), replace=False))
            self._carto = {"x": self._take(x, sel, "float32"), "idx": idx[sel],
                           "y": {k: np.asarray(v)[sel] for k, v in y.items() if k != SAMPLE_IDX_KEY}}
        self._tap_used = False
        self._reset_state()
        if probe_missing and self.verbose:
            print("ℹ️ [TrainingDiagnostics] influence_every>0 لكن بلا probe: تأثير الدفعات معطَّل (مرّر probe=(x, y)).")

    # ── مساعدات ────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _take(x, sel, dtype):
        if isinstance(x, dict):
            return {k: np.asarray(v)[sel].astype(dtype) for k, v in x.items()}
        return np.asarray(x)[sel].astype(dtype)

    def _reset_state(self):
        self._queue, self._tap_k, self._tap_struct = {}, 0, None
        self._k = self._global_step = self._epoch = 0
        self._b_step, self._b_epoch, self._b_batch, self._b_vals = array("i"), array("i"), array("i"), {}
        self._prev_cum, self._prev_kk = {}, 0
        self._grad_rows, self._group_rows, self._infl_rows, self._infl_idx, self._epoch_rows = [], [], [], [], []
        self._conn = self._shared = self._scope_mask = self._group_of = None
        self._w_before = None
        self._missed = 0
        self._carto_conf, self._carto_ok, self._carto_scale = {}, {}, {}
        self._vars = None

    def _wants(self, k):
        n = k + 1
        return bool((self.grad_every and n % self.grad_every == 0) or (self._infl_on and n % self.influence_every == 0))

    def tap(self, ds):
        """يُرجع dataset بنفس العناصر وبالترتيب نفسه، يُودِع نسخةً من الدفعات التي ستُقاس (كل grad_every/influence_every)
        في طابور داخلي موسوم برقم الخطوة. يجب أن يكون آخر عنصر في خط التدفّق (prefetch بعده مسموح)."""
        self._tap_used = True

        @tf.autograph.experimental.do_not_convert
        def _map(*elems):
            flat = tf.nest.flatten(elems)
            self._tap_struct = tf.nest.map_structure(lambda _: 0, elems)

            def _stash(*arrs):
                k = self._tap_k
                self._tap_k += 1
                if self._wants(k):
                    self._queue[k] = tuple(np.array(a) for a in arrs)
                return np.int32(0)
            tok = tf.numpy_function(_stash, flat, tf.int32, stateful=True)
            with tf.control_dependencies([tok]):
                flat = [tf.identity(f) for f in flat]
            return tf.nest.pack_sequence_as(elems, flat)
        return ds.map(_map)

    def _lr(self):
        lr = self._trainer.optimizer.learning_rate
        try:
            if callable(lr):
                lr = lr(self._trainer.optimizer.iterations)
            return float(np.asarray(tf.convert_to_tensor(lr)))
        except Exception:
            return float("nan")

    def _task_scales(self):
        """معامل كل مهمة في الخسارة الكلية (1 للمهام الثابتة؛ ½·e^(−log σ²) لمهام Kendall) = ما يراه المُحسِّن فعلاً."""
        tr = self._trainer
        s = {t: 1.0 for t in tr.target_names}
        ul = getattr(tr, "uncertainty_layer", None)
        if ul is not None:
            for t in tr.kendall_task_names:
                s[t] = float(0.5 * np.exp(-float(ul._log_vars[t].numpy())))
        return s

    @staticmethod
    def _np_grads(grads):
        return [None if g is None else np.asarray(tf.convert_to_tensor(g), dtype=np.float32) for g in grads]

    @tf.autograph.experimental.do_not_convert
    def _grads_graph(self, x, y, sw):
        with tf.GradientTape(persistent=True) as tape:
            total, _, per_t, _, _ = self._trainer._compute_losses(x, y, sw, training=False)
        gt = {t: tape.gradient(per_t[t], self._vars) for t in per_t}
        gtot = tape.gradient(total, self._vars)
        return total, per_t, gt, gtot

    def _grads(self, x, y, sw=None):
        tx = tf.nest.map_structure(tf.convert_to_tensor, x)
        ty = tf.nest.map_structure(tf.convert_to_tensor, y)
        tw = None if sw is None else tf.nest.map_structure(tf.convert_to_tensor, sw)
        total, per_t, gt, gtot = self._grad_fn(tx, ty, tw)
        return (float(total), {t: float(v) for t, v in per_t.items()},
                {t: self._np_grads(g) for t, g in gt.items()}, self._np_grads(gtot))

    def _vec(self, arrs, mask, scale=1.0):
        out = [arrs[i].astype(np.float64).ravel() * scale for i in np.flatnonzero(mask) if arrs[i] is not None]
        return np.concatenate(out) if out else np.zeros(0)

    def _init_structure(self, gt):
        cnt = np.zeros(len(self._vars), int)
        for t in gt:
            for i, g in enumerate(gt[t]):
                cnt[i] += g is not None
        self._conn = cnt
        self._shared = cnt >= 2 if (cnt >= 2).any() else cnt >= 1
        priv = cnt == 1
        self._scope_mask = {"heads": priv if priv.any() else cnt >= 1, "trunk": self._shared, "all": cnt >= 1}

    # ── Keras hooks ─────────────────────────────────────────────────────────────────────────────
    def on_train_begin(self, logs=None):
        tr = self.model
        if not hasattr(tr, "_compute_losses"):
            raise TypeError("TrainingDiagnostics يتطلّب GenericTrainer (له _compute_losses)؛ النموذج المُمرَّر لـ fit غيره")
        self._trainer = tr
        self._vars = list(tr.model.trainable_variables)
        self._grad_fn = tf.function(self._grads_graph, reduce_retracing=True)
        by_id = {}
        for layer in tr.model.layers:
            for w in layer.trainable_weights:
                by_id.setdefault(id(w), layer.name)
        self._group_of = [by_id.get(id(v), "other") for v in self._vars]
        self._queue, self._tap_k, self._k = {}, 0, 0
        self._w_before = None
        measuring = bool(self.grad_every or self._infl_on)
        if measuring and not self._tap_used:
            print("⚠️ [TrainingDiagnostics] لم يُستعمل diag.tap(train_ds): تُسجَّل الخسائر والخريطة فقط — لا تدرّجات ولا تأثير دفعات.")
        elif self.verbose:
            print(f"🔬 [TrainingDiagnostics] خسائر كل دفعة | تدرّجات/هيمنة الرؤوس كل {self.grad_every or '—'} | "
                  f"تأثير الدفعات كل {self.influence_every if self._infl_on else '—'} (نطاق {self.influence_scope}) | "
                  f"خريطة البيانات {'نعم' if self._carto else 'لا'}")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = int(epoch)
        self._prev_cum, self._prev_kk = {}, 0

    def on_train_batch_begin(self, batch, logs=None):
        if self._tap_used and self.grad_every and (self._k + 1) % self.grad_every == 0:
            self._w_before = [v.numpy() for v in self._vars]

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        kk = int(batch) + 1
        vals = {}
        span = max(kk - self._prev_kk, 1)
        for key, v in logs.items():
            try:
                m = float(v)
            except (TypeError, ValueError):
                continue
            vals[key] = (kk * m - self._prev_kk * self._prev_cum.get(key, 0.0)) / span
            self._prev_cum[key] = m
        self._prev_kk = kk
        if not self._b_vals:
            self._b_vals = {key: array("f") for key in vals}
        self._b_step.append(self._global_step)
        self._b_epoch.append(self._epoch)
        self._b_batch.append(int(batch))
        for key, arr in self._b_vals.items():
            arr.append(vals.get(key, float("nan")))

        k = self._k
        do_grad = bool(self.grad_every and (k + 1) % self.grad_every == 0)
        do_infl = bool(self._infl_on and (k + 1) % self.influence_every == 0)
        if self._tap_used and (do_grad or do_infl):
            item = self._queue.get(k)
            if item is None:
                self._missed += 1
            else:
                elems = tf.nest.pack_sequence_as(self._tap_struct, list(item))
                x, y = elems[0], elems[1]
                sw = elems[2] if len(elems) > 2 else None
                self._measure(x, y, sw, vals, int(batch), do_grad, do_infl)
        for kk_ in [q for q in self._queue if q <= k]:
            del self._queue[kk_]
        self._w_before = None
        self._k += 1
        self._global_step += 1

    def _measure(self, x, y, sw, batch_vals, batch, do_grad, do_infl):
        idx = y.get(SAMPLE_IDX_KEY) if isinstance(y, dict) else None
        idx = None if idx is None else np.asarray(idx).astype(np.int64).ravel()
        y = {kk: vv for kk, vv in y.items() if kk != SAMPLE_IDX_KEY}
        _, per_t, gt, gtot = self._grads(x, y, sw)
        if self._conn is None:
            self._init_structure(gt)
        scales = self._task_scales()
        lr = self._lr()
        step = self._global_step
        base = {"step": step, "epoch": self._epoch, "batch": batch, "lr": lr}
        if do_grad:
            row = dict(base)
            row["loss"] = batch_vals.get("loss", float("nan"))
            row["g_total"] = float(np.linalg.norm(self._vec(gtot, self._conn >= 1)))
            row["nonfinite_grad"] = bool(not np.isfinite(row["g_total"]))
            vecs = {t: self._vec(gt[t], self._shared, scales[t]) for t in gt}
            norms = {t: float(np.linalg.norm(v)) for t, v in vecs.items()}
            for t, nrm in norms.items():
                row[f"gn_trunk:{t}"] = nrm
                row[f"scale:{t}"] = scales[t]
            pos = {t: n for t, n in norms.items() if n > 0}
            row["dominance"] = (max(pos.values()) / min(pos.values())) if len(pos) > 1 else 1.0
            row["dominant"] = max(norms, key=norms.get) if norms else ""
            names, cos_min, cos_pair = list(vecs), 1.0, ""
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = vecs[names[i]], vecs[names[j]]
                    d = float(np.linalg.norm(a) * np.linalg.norm(b))
                    c = float(a @ b / d) if d > 0 else 0.0
                    row[f"cos:{names[i]}|{names[j]}"] = c
                    if c < cos_min:
                        cos_min, cos_pair = c, f"{names[i]}|{names[j]}"
            row["cos_min"], row["cos_min_pair"] = cos_min, cos_pair
            self._grad_rows.append(row)
            # مجموعات الطبقات: نظيم التدرّج الكلي + نسبة التحديث الفعلية (يتطلّب لقطة الأوزان قبل الخطوة)
            groups = {}
            for i, g in enumerate(self._group_of):
                d = groups.setdefault(g, [0.0, 0.0, 0.0])
                if gtot[i] is not None:
                    d[0] += float(np.sum(np.square(gtot[i], dtype=np.float64)))
                w = self._vars[i].numpy().astype(np.float64)
                d[1] += float(np.sum(w * w))
                if self._w_before is not None:
                    d[2] += float(np.sum(np.square(w - self._w_before[i].astype(np.float64))))
            for g, (g2, w2, dw2) in groups.items():
                self._group_rows.append({"step": step, "epoch": self._epoch, "group": g, "grad_norm": g2 ** 0.5,
                                         "weight_norm": w2 ** 0.5,
                                         "update_ratio": (dw2 ** 0.5) / (w2 ** 0.5 + 1e-12) if self._w_before is not None else float("nan")})
        if do_infl and self._probe is not None:
            px, py = self._probe
            _, _, pgt, pgtot = self._grads(px, py)
            mask = self._scope_mask[self.influence_scope]
            tr_tot, pr_tot = self._vec(gtot, mask), self._vec(pgtot, mask)
            dot = float(tr_tot @ pr_tot)
            denom = float(np.linalg.norm(tr_tot) * np.linalg.norm(pr_tot))
            row = dict(base)
            row.update(influence=lr * dot, cos=(dot / denom if denom > 0 else 0.0), n=int(len(next(iter(y.values())))))
            for t in gt:
                a, b = self._vec(gt[t], mask, scales[t]), self._vec(pgt[t], mask, scales[t])
                d = float(a @ b)
                nn = float(np.linalg.norm(a) * np.linalg.norm(b))
                row[f"infl:{t}"] = lr * d
                row[f"cos:{t}"] = d / nn if nn > 0 else 0.0
            self._infl_rows.append(row)
            self._infl_idx.append(idx)

    # ── خريطة البيانات ──────────────────────────────────────────────────────────────────────────
    def _carto_scores(self, tcfg, out, y, t):
        keys = tcfg["output_keys"]
        yt = np.asarray(y[tcfg["true_key"]]).astype(np.float64)
        if tcfg["task_type"] == "classification":
            p = np.asarray(out[keys["logits"]], dtype=np.float64)
            if tcfg.get("binary", True):
                p = p.reshape(len(yt))
                pos = yt.reshape(-1) > 0.5
                conf = np.where(pos, p, 1.0 - p)
                return conf, ((p >= 0.5) == pos).astype(np.float64)
            lab = yt.reshape(-1).astype(int)
            conf = p[np.arange(len(lab)), lab]
            return conf, (p.argmax(-1) == lab).astype(np.float64)
        if "mu" in keys or "pred" in keys:
            mu = np.asarray(out[keys.get("mu", keys.get("pred"))], dtype=np.float64).reshape(len(yt))
            err = np.abs(yt.reshape(-1) - mu)
            if t not in self._carto_scale:
                self._carto_scale[t] = 1.4826 * float(np.median(np.abs(yt - np.median(yt)))) + 1e-12
            s = self._carto_scale[t]
            return np.exp(-0.5 * (err / s) ** 2), (err <= s).astype(np.float64)
        return None, None

    def _carto_epoch(self):
        c, tr = self._carto, self._trainer
        n = len(c["idx"])
        outs = []
        for a in range(0, n, self.cartography_batch_size):
            sl = slice(a, a + self.cartography_batch_size)
            xb = {k: v[sl] for k, v in c["x"].items()} if isinstance(c["x"], dict) else c["x"][sl]
            o = tr.model(xb, training=False)
            outs.append({k: v.numpy() for k, v in o.items()})
        out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        for t, tcfg in tr.config["targets"].items():
            if tcfg["true_key"] not in c["y"]:
                continue
            conf, ok = self._carto_scores(tcfg, out, c["y"], t)
            if conf is None:
                continue
            self._carto_conf.setdefault(t, []).append(conf)
            self._carto_ok.setdefault(t, []).append(ok)

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_rows.append({"epoch": int(epoch), **{k: float(v) for k, v in (logs or {}).items()
                                                         if np.isscalar(v) or np.ndim(v) == 0}})
        if self._carto is not None:
            self._carto_epoch()
        if self.report_fn is not None and self.report_every and (int(epoch) + 1) % self.report_every == 0:
            self.layer_reports.append((int(epoch), self.report_fn(self._trainer.model)))

    def on_train_end(self, logs=None):
        if self.report_fn is not None and (not self.layer_reports or self.layer_reports[-1][0] != self._epoch):
            self.layer_reports.append((self._epoch, self.report_fn(self._trainer.model)))   # بعد استعادة «الأفضل» إن وُجد BestModelTracker
        if self._missed and self.verbose:
            print(f"⚠️ [TrainingDiagnostics] {self._missed} قياس تخطّاه المسجّل لأن الدفعة لم تصل عبر tap (ترتيب/استئناف غير متّسق).")
        if self.verbose:
            st = self.stats()
            bits = [f"{st['n_steps']} دفعة", f"{st['n_grad_measures']} قياس تدرّج", f"{st['n_influence_measures']} قياس تأثير"]
            if "cartography_hard_frac" in st:
                bits.append(f"hard={st['cartography_hard_frac']:.0%} noise?={st['noise_suspect_frac']:.0%}")
            print("🔬 [TrainingDiagnostics] " + " | ".join(bits) + " — diag.summary() / model_health_report")

    # ── الجداول ─────────────────────────────────────────────────────────────────────────────────
    def batch_losses(self):
        import pandas as pd
        d = {"step": np.asarray(self._b_step), "epoch": np.asarray(self._b_epoch), "batch": np.asarray(self._b_batch)}
        d.update({k: np.asarray(v) for k, v in self._b_vals.items()})
        return pd.DataFrame(d)

    def grad_table(self):
        import pandas as pd
        return pd.DataFrame(self._grad_rows)

    def group_table(self):
        import pandas as pd
        return pd.DataFrame(self._group_rows)

    def group_summary(self):
        """لكل مجموعة طبقات: وسيط نظيم التدرّج، وسيط نسبة التحديث/الوزن، وحصّتها من نظيم التدرّج الكلي."""
        import pandas as pd
        g = self.group_table()
        if g.empty:
            return pd.DataFrame(columns=["group", "grad_norm", "weight_norm", "update_ratio", "grad_share"])
        s = g.groupby("group").agg(grad_norm=("grad_norm", "median"), weight_norm=("weight_norm", "median"),
                                   update_ratio=("update_ratio", "median")).reset_index()
        s["grad_share"] = s["grad_norm"] / max(float(s["grad_norm"].sum()), 1e-30)
        return s.sort_values("grad_norm", ascending=False).reset_index(drop=True)

    def group_trend(self, parts=3):
        """اتجاه كل مجموعة طبقات عبر التدريب: وسيط نظيم التدرّج في الثلث الأول مقابل الأخير (trend = أخير ÷ أول؛ <1 يتلاشى، >1 ينفجر)
        ووسيط نسبة التحديث/الوزن في الأخير. هذا هو «اتجاه التدرّج لكل طبقة» في تقرير الطبقات."""
        import pandas as pd
        g = self.group_table()
        if g.empty:
            return pd.DataFrame(columns=["group", "grad_first", "grad_last", "grad_trend", "update_ratio_last"])
        steps = np.sort(g["step"].unique())
        cut = np.array_split(steps, parts)
        first, last = set(cut[0].tolist()), set(cut[-1].tolist())
        rows = []
        for name, sub in g.groupby("group", sort=False):
            a, b = sub[sub["step"].isin(first)], sub[sub["step"].isin(last)]
            gf, gl = float(a["grad_norm"].median()), float(b["grad_norm"].median())
            rows.append({"group": name, "grad_first": gf, "grad_last": gl, "grad_trend": gl / (gf + 1e-30),
                         "update_ratio_last": float(b["update_ratio"].median())})
        return pd.DataFrame(rows)

    def influence_table(self):
        import pandas as pd
        return pd.DataFrame(self._infl_rows)

    def _meta_summary(self, idx):
        if self.sample_meta is None or idx is None or len(idx) == 0:
            return ""
        import pandas as pd
        m = self.sample_meta.iloc[np.asarray(idx)]
        parts = []
        for col in m.columns:
            s = m[col]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                parts.append(f"{col}: {s.min()}..{s.max()}")
            else:
                vc = s.value_counts().head(3)
                parts.append(f"{col}: " + ", ".join(f"{k}×{v}" for k, v in vc.items()))
        return " | ".join(parts)

    def _infl_column(self, target, by):
        if by not in ("influence", "cos"):
            raise ValueError("by: 'influence' (lr·⟨g,g_probe⟩) | 'cos' (التوافق بلا مقياس)")
        if target is None:
            return "influence" if by == "influence" else "cos"
        return f"infl:{target}" if by == "influence" else f"cos:{target}"

    def top_batches(self, kind="harmful", k=None, target=None, by="influence"):
        """أكثر الدفعات ضرراً/نفعاً على probe (kind='harmful'|'helpful') مع فهارس عيّناتها وملخّص sample_meta.
        target=None: تأثير الخسارة الكلية (يهيمن عليه الرأس ذو التدرّج الأكبر، عادةً الانحدار)؛ target='high_class': توافق
        **نفس المهمة** — الكاشف الأوضح لدفعة تسمياتها خاطئة في رأس بعينه. by='cos' للمقارنة بين المهام بلا مقياس."""
        import pandas as pd
        if kind not in ("harmful", "helpful"):
            raise ValueError("kind: 'harmful' | 'helpful'")
        df = self.influence_table()
        if df.empty:
            return df
        col = self._infl_column(target, by)
        if col not in df:
            raise KeyError(f"عمود غير موجود: {col!r} — الأهداف: {[c[len('infl:'):] for c in df.columns if c.startswith('infl:')]}")
        order = np.argsort(df[col].to_numpy(), kind="stable")
        order = order if kind == "harmful" else order[::-1]
        rows = []
        for i in order[: (k or self.top_k)]:
            idx = self._infl_idx[i]
            r = df.iloc[i].to_dict()
            r["n_idx"] = 0 if idx is None else len(idx)
            r["samples"] = None if idx is None else idx.tolist()
            r["meta"] = self._meta_summary(idx)
            rows.append(r)
        return pd.DataFrame(rows)

    def sample_influence(self, target=None, by="influence", min_count=1):
        """تأثير مُحمَّل على العيّنات: لكل فهرس متوسّط (تأثير دفعاته ÷ حجمها) وعدد القياسات التي دخل فيها، مرتّباً من الأضرّ."""
        import pandas as pd
        df = self.influence_table()
        empty = pd.DataFrame(columns=["idx", "mean_influence", "count"])
        if df.empty or not any(i is not None for i in self._infl_idx):
            return empty
        col = self._infl_column(target, by)
        ids, vals = [], []
        for i, idx in enumerate(self._infl_idx):
            if idx is not None and len(idx) and np.isfinite(df[col].iloc[i]):
                ids.append(idx)
                vals.append(np.full(len(idx), df[col].iloc[i] / len(idx)))
        if not ids:
            return empty
        g = pd.DataFrame({"idx": np.concatenate(ids), "v": np.concatenate(vals)}).groupby("idx")["v"].agg(["sum", "count"]).reset_index()
        out = pd.DataFrame({"idx": g["idx"], "mean_influence": g["sum"] / g["count"], "count": g["count"]})
        out = out[out["count"] >= min_count].sort_values("mean_influence").reset_index(drop=True)
        if self.sample_meta is not None and len(out):
            out = pd.concat([out, self.sample_meta.iloc[out["idx"].to_numpy()].reset_index(drop=True)], axis=1)
        return out

    def cartography_table(self):
        """صفّ لكل عيّنة في المجموعة الثابتة وأعمدة لكل هدف: conf_mean / variability / correct / group / noise_suspect."""
        import pandas as pd
        if self._carto is None or not self._carto_conf:
            return pd.DataFrame()
        df = pd.DataFrame({"idx": self._carto["idx"]})
        q = self.group_quantile
        for t, conf_list in self._carto_conf.items():
            conf, ok = np.stack(conf_list), np.stack(self._carto_ok[t])
            cm, var, cr = conf.mean(0), conf.std(0), ok.mean(0)
            hi_c, lo_c, hi_v = np.quantile(cm, 1 - q), np.quantile(cm, q), np.quantile(var, 1 - q)
            amb = (var >= hi_v) & (hi_v > 1e-9) if len(conf) >= 2 else np.zeros(len(cm), bool)
            grp = np.where(amb, "ambiguous", np.where(cm >= hi_c, "easy", np.where(cm <= lo_c, "hard", "middle")))
            df[f"conf_mean:{t}"], df[f"variability:{t}"], df[f"correct:{t}"], df[f"group:{t}"] = cm, var, cr, grp
            df[f"noise_suspect:{t}"] = (cr <= self.noise_correct_max) & (cm <= self.noise_conf_max) & (len(conf) >= 3)
        if self.sample_meta is not None:
            df = pd.concat([df, self.sample_meta.iloc[df["idx"].to_numpy()].reset_index(drop=True)], axis=1)
        return df

    def hardest_samples(self, target, k=20):
        """أصعب عيّنات هدف (hard-to-learn): أدنى متوسط ثقة بالتسمية الصحيحة؛ عمود noise_suspect يُعلِّم الأشبه بتسمية خاطئة."""
        df = self.cartography_table()
        if df.empty:
            return df
        if f"conf_mean:{target}" not in df:
            raise KeyError(f"هدف غير مسجَّل في الخريطة: {target!r} — المتاح: {list(self._carto_conf)}")
        return df.sort_values(f"conf_mean:{target}", kind="stable").head(k).reset_index(drop=True)

    def cartography_summary(self):
        import pandas as pd
        df = self.cartography_table()
        rows = []
        for t in self._carto_conf:
            vc = df[f"group:{t}"].value_counts(normalize=True)
            rows.append({"target": t, **{g: float(vc.get(g, 0.0)) for g in ("easy", "ambiguous", "hard", "middle")},
                         "noise_suspect": float(df[f"noise_suspect:{t}"].mean()),
                         "conf_mean": float(df[f"conf_mean:{t}"].mean()), "n": len(df), "epochs": len(self._carto_conf[t])})
        return pd.DataFrame(rows)

    def epoch_table(self):
        import pandas as pd
        return pd.DataFrame(self._epoch_rows)

    def summary(self):
        """كل الجداول في قاموس واحد (DataFrames) + stats()."""
        return {"batches": self.batch_losses(), "epochs": self.epoch_table(), "grad": self.grad_table(),
                "groups": self.group_summary(), "group_trend": self.group_trend(), "group_steps": self.group_table(), "influence": self.influence_table(),
                "top_harmful": self.top_batches("harmful"), "top_helpful": self.top_batches("helpful"),
                "sample_influence": self.sample_influence(), "cartography": self.cartography_table(),
                "cartography_summary": self.cartography_summary(), "stats": self.stats()}

    def stats(self):
        """مقاييس رقمية مختصرة يستهلكها `model_health_verdicts`/`model_health_report` (لا شيء هنا يحتاج DataFrame خارجياً)."""
        st = {"n_steps": int(len(self._b_step)), "n_grad_measures": len(self._grad_rows),
              "n_influence_measures": len(self._infl_rows), "missed_measures": self._missed}
        loss = np.asarray(self._b_vals.get("loss", []), dtype=np.float64)
        if loss.size:
            st["nonfinite_loss_batches"] = int((~np.isfinite(loss)).sum())
            fin = loss[np.isfinite(loss)]
            if fin.size:
                med, mad = float(np.median(fin)), float(np.median(np.abs(fin - np.median(fin))) * 1.4826)
                st["loss_spike_z"] = float((fin.max() - med) / (mad + 1e-12))
        g = self.grad_table()
        if not g.empty:
            st["nonfinite_grad"] = int(g["nonfinite_grad"].sum())
            gt_ = g["g_total"].to_numpy()
            st["grad_total_median"] = float(np.nanmedian(gt_))
            st["grad_total_max_over_median"] = float(np.nanmax(gt_) / (np.nanmedian(gt_) + 1e-30))
            st["head_dominance_median"] = float(g["dominance"].median())
            st["head_dominance_max"] = float(g["dominance"].max())
            st["dominant_head"] = str(g["dominant"].mode().iloc[0]) if g["dominant"].notna().any() else ""
            st["head_trunk_norm_median"] = {c[len("gn_trunk:"):]: float(g[c].median()) for c in g.columns if c.startswith("gn_trunk:")}
            cos_cols = [c for c in g.columns if c.startswith("cos:")]
            if cos_cols:
                cs = g[cos_cols].to_numpy()
                st["conflict_frac"] = float(np.nanmean(cs < 0))
                st["cos_min_median"] = float(g["cos_min"].median())
                st["conflict_pair"] = str(g["cos_min_pair"].mode().iloc[0]) if g["cos_min_pair"].astype(bool).any() else ""
        gs = self.group_summary()
        if not gs.empty:
            pos = gs[gs["grad_norm"] > 0]
            if len(pos):
                st["group_grad_ratio"] = float(pos["grad_norm"].min() / pos["grad_norm"].max())
                st["weakest_group"], st["strongest_group"] = str(pos.iloc[-1]["group"]), str(pos.iloc[0]["group"])
            ur = gs.dropna(subset=["update_ratio"])
            if len(ur):
                st["update_ratio_median"] = float(ur["update_ratio"].median())
                st["update_ratio_min"], st["update_ratio_max"] = float(ur["update_ratio"].min()), float(ur["update_ratio"].max())
                st["update_ratio_min_group"], st["update_ratio_max_group"] = (str(ur.loc[ur["update_ratio"].idxmin(), "group"]),
                                                                              str(ur.loc[ur["update_ratio"].idxmax(), "group"]))
        inf = self.influence_table()
        if not inf.empty:
            v = inf["influence"].to_numpy()
            st["influence_harmful_frac"] = float(np.mean(v < 0))
            st["influence_sum"] = float(np.nansum(v))
            order = np.argsort(-np.abs(v))[: self.top_k]
            st["influence_topk_abs_share"] = float(np.abs(v)[order].sum() / (np.abs(v).sum() + 1e-30))
            st["influence_cos_median"] = float(inf["cos"].median())
        cs = self.cartography_summary()
        if not cs.empty:
            st["cartography_hard_frac"] = float(cs["hard"].mean())
            st["cartography_ambiguous_frac"] = float(cs["ambiguous"].mean())
            st["noise_suspect_frac"] = float(cs["noise_suspect"].mean())
        return st
