"""
PURPOSE:  BestModelTracker: best-copy tracking plus early stopping in one resumable callback.
TAGS:     BestModelTracker, early stopping, best weights, restore best, patience, resume
PITFALLS: State is saved with the checkpoint (get_state/set_state); changing it breaks resuming old runs. Executed
          into the one shared trainer namespace by trainer/_loader.py (never imported on its own): names from other
          modules resolve at call time.
"""
# @title 6.2) BestModelTracker — «أفضل نسخة» + التوقف المبكر في مكان واحد (قابل للاستئناف الكامل)
class BestModelTracker(tf.keras.callbacks.Callback):
    """يمتلك وحده تعريف «الأفضل» — بديل SmoothedEarlyStopping + BestWeightsSaver اللذين كانا يحسبانه بطريقتين
    مختلفتين (الأول بمتوسط نافذة، الثاني بالقيمة الخام) وكلاهما يفقد حالته عند الاستئناف.

    smoothing: كيف يُنعَّم المقياس قبل المقارنة —
        "none"   : القيمة الخام في كل حقبة
        "window" : متوسط آخر `window` حقب (السلوك القديم)
        "ema"    : متوسط أسّي متحرك  s_t = beta*s_(t-1) + (1-beta)*x_t   (s_1 = x_1)
    الحفظ: أفضل أوزان النموذج الأساسي تُكتب ذرّيًا إلى best_path مع best_meta.json (قيمة الأفضل وحقبته).
    الاستئناف: قيمة الأفضل تُقرأ من best_meta.json — فلا تُكتب فوق الملف حقبةٌ أسوأ بعد انقطاع Colab.
    الاسترجاع: يتم في on_train_end (بعد أن يبدّل Keras أوزان EMA الخاصة بالـ optimizer داخل fit)، من الملف على القرص
    فيعمل حتى لو لم يتحسّن شيء في الجلسة الحالية."""

    def __init__(self, base_model: tf.keras.Model, best_path: Optional[str] = None,
                 monitor: str = "val_loss", mode: str = "min", patience: int = 15, min_delta: float = 1e-4,
                 smoothing: str = "window", window: int = 3, ema_beta: float = 0.7,
                 restore_best_weights: bool = True, verbose: int = 1, initial_state: Optional[dict] = None,
                 trainer: Optional["GenericTrainer"] = None, weights_snapshot: str = "raw"):
        super().__init__()
        if smoothing not in ("none", "window", "ema"):
            raise ValueError("smoothing يجب أن يكون none|window|ema")
        if mode not in ("min", "max"):
            raise ValueError("mode يجب أن يكون min|max")
        if weights_snapshot not in ("raw", "ema_weights"):
            raise ValueError("weights_snapshot يجب أن يكون raw|ema_weights")
        if weights_snapshot == "ema_weights" and trainer is None:
            raise ValueError("weights_snapshot='ema_weights' يتطلب تمرير trainer=")
        self.trainer = trainer
        self.weights_snapshot = weights_snapshot
        self.base_model = base_model
        self.best_path = best_path
        self.best_meta_path = os.path.join(os.path.dirname(best_path), "best_meta.json") if best_path else None
        self.monitor, self.mode = monitor, mode
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.smoothing, self.window, self.ema_beta = smoothing, max(int(window), 1), float(ema_beta)
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.history: List[float] = []
        self.ema: Optional[float] = None
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.wait = 0
        self._mem_weights = None
        if best_path:
            os.makedirs(os.path.dirname(best_path), exist_ok=True)

        if initial_state:
            self.load_state(initial_state)
        self._load_best_meta()

    # ── الحالة (تُحفظ في meta.json عبر EpochCheckpointCallback) ─────────────────
    def get_state(self) -> dict:
        return {"monitor": self.monitor, "mode": self.mode, "smoothing": self.smoothing,
                "history": self.history[-200:], "ema": self.ema,
                "best": self.best, "best_epoch": self.best_epoch, "wait": int(self.wait)}

    def load_state(self, state: dict):
        if state.get("monitor") not in (None, self.monitor) or state.get("mode") not in (None, self.mode):
            self._metric_definition_changed(f"{state.get('monitor')}/{state.get('mode')}")
            return
        self.history = list(state.get("history", []))
        self.ema = state.get("ema")
        b = state.get("best")
        self.best = float(b) if b is not None and np.isfinite(b) else None   # الحالة القديمة كانت تحفظ inf
        self.best_epoch = state.get("best_epoch")
        self.wait = int(state.get("wait", 0))
        print(f"🔄 [BestModelTracker] استُرجعت الحالة: best={self.best} (حقبة {self.best_epoch}) wait={self.wait}")

    def _load_best_meta(self):
        """best_meta.json يُكتب مع ملف الأوزان نفسه، فهو المصدر الأصدق لقيمة الأفضل."""
        if not (self.best_meta_path and os.path.exists(self.best_meta_path) and os.path.exists(self.best_path)):
            return
        try:
            with open(self.best_meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"⚠️ [BestModelTracker] تعذّر قراءة best_meta.json ({e}) — سيُعاد تقييم الأفضل")
            return
        if meta.get("monitor") != self.monitor or meta.get("mode") != self.mode:
            self._metric_definition_changed(f"{meta.get('monitor')}/{meta.get('mode')}")
            return
        self.best, self.best_epoch = float(meta["best"]), meta.get("best_epoch")

    def _metric_definition_changed(self, old: str):
        """قيم مقياس مختلف لا تُقارن بقيم مقياس آخر: نبدأ التتبّع من جديد ونحتفظ بنسخة من الملف القديم."""
        print(f"⚠️ [BestModelTracker] تغيّر مقياس الاختيار ({old} → {self.monitor}/{self.mode}) — يبدأ التتبّع من جديد")
        if self.best_path and os.path.exists(self.best_path):
            backup = self.best_path.replace(".weights.h5", ".prev.weights.h5")
            os.replace(self.best_path, backup)
            print(f"   نسخة الأفضل القديمة محفوظة في: {backup}")
        for p in (self.best_meta_path,):
            if p and os.path.exists(p):
                os.remove(p)
        self.history, self.ema, self.best, self.best_epoch, self.wait = [], None, None, None, 0

    # ── المنطق ─────────────────────────────────────────────────────────────────
    def _smooth(self, current: float) -> float:
        self.history.append(current)
        if self.smoothing == "ema":
            self.ema = current if self.ema is None else self.ema_beta * self.ema + (1.0 - self.ema_beta) * current
            return float(self.ema)
        if self.smoothing == "window":
            return float(np.mean(self.history[-self.window:]))
        return float(current)

    def _is_better(self, s: float) -> bool:
        if self.best is None:
            return True
        return s < self.best - self.min_delta if self.mode == "min" else s > self.best + self.min_delta

    def _snapshot_weights(self) -> Optional[list]:
        """يُرجع أوزان EMA (المتوسط المتحرك للـ optimizer) إن طُلب ذلك وكان متاحًا، وإلا None (يعني: استخدم الأوزان الخام كما هي)."""
        if self.weights_snapshot != "ema_weights":
            return None
        opt = self.trainer.optimizer
        if not getattr(opt, "use_ema", False):
            print("⚠️ [BestModel] weights_snapshot='ema_weights' لكن optimizer.use_ema=False — استُخدمت الأوزان الخام بدلًا")
            return None
        shadow = getattr(opt, "_model_variables_moving_average", None)
        if not shadow:
            return None
        # مطابقة بالهوية (id) بين trainer.trainable_variables (ما بُني عليه EMA) وأوزان base_model —
        # نستخدم id() لا الاسم لأن Keras قد يكرر أسماء الطبقات؛ للأوزان غير القابلة للتدريب (BatchNorm) نُبقي القيمة الخام.
        mapping = {id(v): av for v, av in zip(self.trainer.trainable_variables, shadow)}
        return [(mapping[id(w)].numpy() if id(w) in mapping else w.numpy()) for w in self.base_model.weights]

    def _save_best(self):
        ema_values = self._snapshot_weights()
        if ema_values is not None:
            raw_values = self.base_model.get_weights()
            self.base_model.set_weights(ema_values)
        try:
            if not self.best_path:
                self._mem_weights = self.base_model.get_weights()
                return
            tmp = os.path.join(os.path.dirname(self.best_path), "best.tmp.weights.h5")
            self.base_model.save_weights(tmp)
            os.replace(tmp, self.best_path)
            atomic_write_json(self.best_meta_path, {
                "best": self.best, "best_epoch": self.best_epoch, "monitor": self.monitor, "mode": self.mode,
                "smoothing": self.smoothing, "weights_snapshot": self.weights_snapshot,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        finally:
            if ema_values is not None:
                self.base_model.set_weights(raw_values)   # لا نترك تدريب الحقبة القادمة يبدأ من أوزان EMA بالخطأ

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None or not np.isfinite(current):
            return
        smoothed = self._smooth(float(current))

        if self._is_better(smoothed):
            self.best, self.best_epoch, self.wait = smoothed, epoch + 1, 0
            self._save_best()
            if self.verbose:
                print(f"   ⭐ [BestModel] أفضل جديد: {self.monitor}={current:.5f} (smoothed={smoothed:.5f}) — حقبة {epoch + 1}")
        else:
            self.wait += 1

        if self.verbose:
            print(f"   [BestModel] {self.monitor}={current:.5f} (smoothed={smoothed:.5f}) "
                  f"best={self.best:.5f}@{self.best_epoch} wait={self.wait}/{self.patience}")

        if self.wait >= self.patience:
            self.model.stop_training = True
            print(f"⏹️ Early stopping عند الحقبة {epoch + 1}")

    def restore_best(self) -> bool:
        """يعيد أفضل أوزان إلى النموذج الأساسي (من القرص إن وُجد، وإلا من الذاكرة). يُرجع True عند النجاح."""
        if self.best_path and os.path.exists(self.best_path):
            self.base_model.load_weights(self.best_path)
            return True
        if self._mem_weights is not None:
            self.base_model.set_weights(self._mem_weights)
            return True
        return False

    def on_train_end(self, logs=None):
        # يعمل عند أي نهاية تدريب (توقف مبكر أو اكتمال الحقب)، وبعد تبديل EMA الذي يجريه Keras داخل fit()
        if self.restore_best_weights and self.best is not None:
            if self.restore_best():
                print(f"✅ [BestModel] استُرجعت أفضل أوزان (حقبة {self.best_epoch}, {self.monitor} المنعَّم={self.best:.5f})")
            else:
                print("⚠️ [BestModel] لا توجد نسخة أوزان مخزَّنة للأفضل — بقيت الأوزان الحالية")
